import torch
import torch.nn as nn
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from layers.AdaMamba_backbone import *
from layers.AdaMamba_adaptive_blocks import AdaptiveNormalizationBlock
from layers.AdaMamba_experts_blocks import ContextEncoder
from layers.RevIN import RevIN 
from utils.metrics import quantile_loss

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.is_training = configs.is_training
        self.lambda_h_loss = configs.lambda_h_loss
        self.lambda_q_loss = configs.lambda_q_loss
        
        # configs.norm_type: 'AdaNorm' (default), 'RevIN', 'None'
        self.norm_type = getattr(configs, 'norm_type', 'AdaNorm')

        if self.norm_type == 'AdaNorm':
            self.adaptive_norm_block = AdaptiveNormalizationBlock(configs)
        elif self.norm_type == 'RevIN':
            self.adaptive_norm_block = RevIN(configs.enc_in, affine=False)
        else:
            # No Norm
            self.adaptive_norm_block = nn.Identity()

        self.encoder = ContextEncoder(configs)
        self.mean_head = PredictionHead(configs)
        self.trend_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x_enc, batch_y):
        B, L, M = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1).reshape(B * M, L, 1)
        y_true = batch_y[:, -self.pred_len:, :].permute(0, 2, 1).reshape(B * M, self.pred_len, 1)

        if self.norm_type == 'AdaNorm':
            normalized_x, means, stdev, trend = self.adaptive_norm_block.normalize(x_enc)
            
        elif self.norm_type == 'RevIN':
            normalized_x = self.adaptive_norm_block(x_enc, 'norm')
            means = torch.zeros(B * M, 1, 1).to(x_enc.device) # RevIN 내부에서 처리하므로 외부 로직용으론 0
            stdev = torch.ones(B * M, 1, 1).to(x_enc.device)
            trend = torch.zeros(B * M, L + self.pred_len, 1).to(x_enc.device)
            
        else: # 'None' (No Norm)
            normalized_x = x_enc
            means = torch.zeros(B * M, 1, 1).to(x_enc.device)
            stdev = torch.ones(B * M, 1, 1).to(x_enc.device)
            trend = torch.zeros(B * M, L + self.pred_len, 1).to(x_enc.device)

        summary_context = self.encoder(normalized_x)
        mean_pred_norm = self.mean_head(summary_context)

        # Global Trend Fusion
        # ============================================================
        trend_len = trend.shape[1] 
        trend_reshaped = trend.reshape(B, M, trend_len, 1)
        
        naive_mean = torch.mean(trend_reshaped, dim=1, keepdim=True)
        deviation = torch.mean(torch.abs(trend_reshaped - naive_mean), dim=2, keepdim=True)
        weights = torch.softmax(-deviation / 0.5, dim=1)
        global_trend = torch.sum(trend_reshaped * weights, dim=1, keepdim=True)
        
        fused_trend = trend_reshaped + self.trend_gate * global_trend
        trend = fused_trend.reshape(B * M, trend_len, 1)
        # ============================================================

        y_true_detrended = y_true - trend[:, -self.pred_len:, :]
        
        if self.norm_type == 'RevIN':
            batch_mean = self.adaptive_norm_block.mean
            batch_stdev = self.adaptive_norm_block.stdev
            normalized_y_true = (y_true_detrended - batch_mean) / batch_stdev
            
        elif self.norm_type == 'AdaNorm':
            normalized_y_true = (y_true_detrended - means) / stdev
        else:
            normalized_y_true = y_true_detrended # No Norm

        # 5. Losses
        huber = F.smooth_l1_loss(mean_pred_norm, normalized_y_true)
        q10 = quantile_loss(mean_pred_norm, normalized_y_true, 0.1)
        q90 = quantile_loss(mean_pred_norm, normalized_y_true, 0.9)
        mean_loss = self.lambda_h_loss * huber + self.lambda_q_loss * (q10 + q90)

        return mean_loss

    def sample(self, x_enc):
        self.eval()
        with torch.no_grad():
            B, L, M = x_enc.shape
            x_enc = x_enc.permute(0, 2, 1).reshape(B * M, L, 1)

            if self.norm_type == 'AdaNorm':
                normalized_x, means, stdev, trend = self.adaptive_norm_block.normalize(x_enc)
            elif self.norm_type == 'RevIN':
                normalized_x = self.adaptive_norm_block(x_enc, 'norm')
                means = torch.zeros(B * M, 1, 1).to(x_enc.device)
                stdev = torch.ones(B * M, 1, 1).to(x_enc.device)
                trend = torch.zeros(B * M, L + self.pred_len, 1).to(x_enc.device)
            else:
                normalized_x = x_enc
                means = torch.zeros(B * M, 1, 1).to(x_enc.device)
                stdev = torch.ones(B * M, 1, 1).to(x_enc.device)
                trend = torch.zeros(B * M, L + self.pred_len, 1).to(x_enc.device)

            summary_context = self.encoder(normalized_x) 

            trend_len = trend.shape[1]
            trend_reshaped = trend.reshape(B, M, trend_len, 1)
            naive_mean = torch.mean(trend_reshaped, dim=1, keepdim=True)
            deviation = torch.mean(torch.abs(trend_reshaped - naive_mean), dim=2, keepdim=True)
            weights = torch.softmax(-deviation / 0.5, dim=1)
            global_trend = torch.sum(trend_reshaped * weights, dim=1, keepdim=True)
            fused_trend = trend_reshaped + self.trend_gate * global_trend
            trend = fused_trend.reshape(B * M, trend_len, 1)
            
            mean_pred_norm = self.mean_head(summary_context)
            
            trend_for_forecast = trend[:, -self.pred_len:, :]
            
            if self.norm_type == 'AdaNorm':
                final_forecast = self.adaptive_norm_block.denormalize(mean_pred_norm, means, stdev, trend_for_forecast)
            elif self.norm_type == 'RevIN':
                final_forecast = self.adaptive_norm_block(mean_pred_norm, 'denorm')
            else:
                final_forecast = mean_pred_norm

            final_forecast = final_forecast.reshape(B, M, self.pred_len).permute(0, 2, 1)
            
            return final_forecast