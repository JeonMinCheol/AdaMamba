import torch
import torch.nn as nn
import numpy as np
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from layers.AdaMamba_backbone import *
from layers.AdaMamba_adaptive_blocks import AdaptiveNormalizationBlock
from layers.AdaMamba_experts_blocks import ContextEncoder
from utils.metrics import quantile_loss

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.c_in = configs.enc_in
        self.d_model = configs.d_model
        self.is_training = configs.is_training
        self.lambda_h_loss = configs.lambda_h_loss
        self.lambda_q_loss = configs.lambda_q_loss
        
        self.adaptive_norm_block = AdaptiveNormalizationBlock(configs)
        self.encoder = ContextEncoder(configs)
        self.mean_head = PredictionHead(configs)
        self.trend_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x_enc, batch_y):
        B, L, M = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1).reshape(B * M, L, 1)
        y_true = batch_y[:, -self.pred_len:, :].permute(0, 2, 1).reshape(B * M, self.pred_len, 1)

        normalized_x, means, stdev, trend = self.adaptive_norm_block.normalize(x_enc)
        summary_context = self.encoder(normalized_x)
        mean_pred_norm = self.mean_head(summary_context)

        # ============================================================
        # 🔥 [업그레이드] Robust Global Trend Fusion (RGT)
        # ============================================================
        
        # (1) 형태 변환: (B*M, L, 1) -> (B, M, L, 1)
        trend_reshaped = trend.reshape(B, M, L, 1)
        
        # (2) 1차 단순 평균 (Naive Mean) 계산
        # 일단 다 섞어서 대략적인 '중심'을 잡습니다.
        naive_mean = torch.mean(trend_reshaped, dim=1, keepdim=True) # (B, 1, L, 1)
        
        # (3) "이상치 점수" 계산 (Distance from Mean)
        # 내 트렌드가 전체 평균과 얼마나 다른가? (L1 Distance)
        # 차이가 클수록 이상치(Outlier)일 확률이 높음
        deviation = torch.mean(torch.abs(trend_reshaped - naive_mean), dim=2, keepdim=True) # (B, M, 1, 1)
        
        # (4) 거리 역수 가중치 (Softmax)
        # 전체 흐름과 비슷한(거리가 가까운) 채널일수록 가중치를 높게 줌
        # temperature(tau)를 0.5 정도로 주어 변별력 강화
        weights = torch.softmax(-deviation / 0.5, dim=1) # (B, M, 1, 1)
        
        # (5) Robust Global Trend 생성 (가중 평균)
        global_trend = torch.sum(trend_reshaped * weights, dim=1, keepdim=True)
        
        # (6) 주입 (기존과 동일)
        fused_trend = trend_reshaped + self.trend_gate * global_trend
        trend = fused_trend.reshape(B * M, L, 1)
        
        # ============================================================

        y_true_detrended = y_true - trend[:, -y_true.size(1):, :]
        normalized_y_true = (y_true_detrended - means) / stdev

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

            # 1. 정규화 및 최종 컨텍스트 생성 (기존과 동일)
            normalized_x, means, stdev, trend = self.adaptive_norm_block.normalize(x_enc)
            summary_context = self.encoder(normalized_x) 

            # ============================================================
            # 🔥 [업그레이드] Robust Global Trend Fusion (RGT)
            # ============================================================
            
            # (1) 형태 변환: (B*M, L, 1) -> (B, M, L, 1)
            trend_reshaped = trend.reshape(B, M, L, 1)
            
            # (2) 1차 단순 평균 (Naive Mean) 계산
            # 일단 다 섞어서 대략적인 '중심'을 잡습니다.
            naive_mean = torch.mean(trend_reshaped, dim=1, keepdim=True) # (B, 1, L, 1)
            
            # (3) "이상치 점수" 계산 (Distance from Mean)
            # 내 트렌드가 전체 평균과 얼마나 다른가? (L1 Distance)
            # 차이가 클수록 이상치(Outlier)일 확률이 높음
            deviation = torch.mean(torch.abs(trend_reshaped - naive_mean), dim=2, keepdim=True) # (B, M, 1, 1)
            
            # (4) 거리 역수 가중치 (Softmax)
            # 전체 흐름과 비슷한(거리가 가까운) 채널일수록 가중치를 높게 줌
            # temperature(tau)를 0.5 정도로 주어 변별력 강화
            weights = torch.softmax(-deviation / 0.5, dim=1) # (B, M, 1, 1)
            
            # (5) Robust Global Trend 생성 (가중 평균)
            global_trend = torch.sum(trend_reshaped * weights, dim=1, keepdim=True)
            
            # (6) 주입 (기존과 동일)
            fused_trend = trend_reshaped + self.trend_gate * global_trend
            trend = fused_trend.reshape(B * M, L, 1)
            
            # ============================================================
            
            # 2. 평균 예측 (mean_head)
            mean_pred_norm = self.mean_head(summary_context)
            
            # 5. 전체 스케일 복원 (De-normalization)
            trend_for_forecast = trend[:, -self.pred_len:, :]
            final_forecast = self.adaptive_norm_block.denormalize(mean_pred_norm, means, stdev, trend_for_forecast).reshape(B, M, self.pred_len).permute(0, 2, 1)
            
            return final_forecast
