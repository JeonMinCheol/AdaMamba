import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers.AdaMamba_backbone import *
from layers.AdaMamba_adaptive_blocks import AdaptiveNormalizationBlock
from layers.AdaMamba_experts_blocks import ContextEncoder
from layers.RevIN import RevIN
from utils.metrics import quantile_loss


class Model(nn.Module):
    def __init__(self, configs, kernels):
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
            self.adaptive_norm_block = AdaptiveNormalizationBlock(configs, kernels)
        elif self.norm_type == 'RevIN':
            self.adaptive_norm_block = RevIN(1, affine=False)
        else:
            self.adaptive_norm_block = nn.Identity()  # No Norm

        self.encoder = ContextEncoder(configs)
        self.mean_head = PredictionHead(configs)

        # Global Trend Fusion gate
        self.trend_gate = nn.Parameter(torch.zeros(1))

        if self.pred_len > self.seq_len:
            self.trend_proj = nn.Linear(self.seq_len, self.pred_len)
            self.trend_proj.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
            )

    def forward(self, x_enc, batch_y):
        """
        x_enc:   [B, L, M]
        batch_y: [B, L+pred_len, M] or [B, ?, M] (너 코드 기준 그대로)
        """
        B, L, M = x_enc.shape

        # encoder/head는 기존대로 "변수별 시계열을 batch로 펼쳐서" 처리
        x_series = x_enc.permute(0, 2, 1).reshape(B * M, L, 1)  # [B*M, L, 1]
        y_true_series = batch_y[:, -self.pred_len:, :].permute(0, 2, 1).reshape(B * M, self.pred_len, 1)

        # ============================================================
        # 1) Normalize + Trend extraction
        # ============================================================
        if self.norm_type == 'AdaNorm':
            # ✅ reshape 전에 multivariate에서 normalize
            # returns: x_norm [B,L,M], means [B,1,M], stdev [B,1,M], trend [B,L,M], alpha_gate [1,1,M] or [B,1,M]
            norm_mv, means_mv, stdev_mv, trend_mv, alpha_gate = self.adaptive_norm_block.normalize(x_enc)

            # encoder input: [B*M,L,1]
            normalized_x = norm_mv.permute(0, 2, 1).reshape(B * M, L, 1)

            # loss용 통계 (series 형태)
            means = means_mv.permute(0, 2, 1).reshape(B * M, 1, 1)
            stdev = stdev_mv.permute(0, 2, 1).reshape(B * M, 1, 1)

        elif self.norm_type == 'RevIN':
            normalized_x = self.adaptive_norm_block(x_series, 'norm')
            means = torch.zeros(B * M, 1, 1, device=x_enc.device)
            stdev = torch.ones(B * M, 1, 1, device=x_enc.device)
            trend_mv = torch.zeros(B, L, M, device=x_enc.device)
            alpha_gate = None

        else:  # 'None'
            normalized_x = x_series
            means = torch.zeros(B * M, 1, 1, device=x_enc.device)
            stdev = torch.ones(B * M, 1, 1, device=x_enc.device)
            trend_mv = torch.zeros(B, L, M, device=x_enc.device)
            alpha_gate = None

        # ============================================================
        # 2) Global Trend Fusion (multivariate trend: [B,L,M])
        # ============================================================
        # [B,L,M] -> [B,M,L,1]
        t = trend_mv.permute(0, 2, 1).unsqueeze(-1)

        naive_mean = torch.mean(t, dim=1, keepdim=True)  # [B,1,L,1]
        deviation = torch.mean(torch.abs(t - naive_mean), dim=2, keepdim=True)  # [B,M,1,1]
        weights = torch.softmax(-deviation / 0.5, dim=1)  # [B,M,1,1]
        global_trend = torch.sum(t * weights, dim=1, keepdim=True)  # [B,1,L,1]

        fused = t + self.trend_gate * global_trend  # [B,M,L,1]
        trend_fused_mv = fused.squeeze(-1).permute(0, 2, 1)  # [B,L,M]

        # pred_len > seq_len이면 trend를 pred_len 길이로 사상
        if self.pred_len > self.seq_len:
            trend_for_forecast_mv = self.trend_proj(trend_fused_mv.permute(0, 2, 1)).permute(0, 2, 1)  # [B,pred_len,M]
        else:
            trend_for_forecast_mv = trend_fused_mv[:, -self.pred_len:, :]  # [B,pred_len,M]

        # ✅ α 게이트를 y detrend에도 동일 적용 (학습 안정/일관성)
        if (self.norm_type == 'AdaNorm') and (alpha_gate is not None):
            # alpha_gate expected: [1,1,M] or [B,1,M]
            trend_for_forecast_mv = trend_for_forecast_mv * alpha_gate  # broadcast

        # series 형태로 변환: [B*M,pred_len,1]
        trend_for_forecast = trend_for_forecast_mv.permute(0, 2, 1).reshape(B * M, self.pred_len, 1)

        # ============================================================
        # 3) Encoder + Head
        # ============================================================
        summary_context = self.encoder(normalized_x)
        mean_pred_norm = self.mean_head(summary_context)  # [B*M,pred_len,1] 가정

        # ============================================================
        # 4) y_true detrend + normalize (loss용)
        # ============================================================
        y_true_detrended = y_true_series - trend_for_forecast

        if self.norm_type == 'RevIN':
            batch_mean = self.adaptive_norm_block.mean
            batch_stdev = self.adaptive_norm_block.stdev
            normalized_y_true = (y_true_detrended - batch_mean) / batch_stdev

        elif self.norm_type == 'AdaNorm':
            normalized_y_true = (y_true_detrended - means) / stdev

        else:
            normalized_y_true = y_true_detrended

        # ============================================================
        # 5) Loss
        # ============================================================
        huber = F.smooth_l1_loss(mean_pred_norm, normalized_y_true)
        q10 = quantile_loss(mean_pred_norm, normalized_y_true, 0.1)
        q90 = quantile_loss(mean_pred_norm, normalized_y_true, 0.9)
        mean_loss = self.lambda_h_loss * huber + self.lambda_q_loss * (q10 + q90)

        return mean_loss

    def sample(self, x_enc):
        self.eval()
        with torch.no_grad():
            B, L, M = x_enc.shape

            x_series = x_enc.permute(0, 2, 1).reshape(B * M, L, 1)

            # ============================================================
            # 1) Normalize + Trend extraction
            # ============================================================
            if self.norm_type == 'AdaNorm':
                norm_mv, means_mv, stdev_mv, trend_mv, alpha_gate = self.adaptive_norm_block.normalize(x_enc)

                normalized_x = norm_mv.permute(0, 2, 1).reshape(B * M, L, 1)
                means = means_mv.permute(0, 2, 1).reshape(B * M, 1, 1)
                stdev = stdev_mv.permute(0, 2, 1).reshape(B * M, 1, 1)

            elif self.norm_type == 'RevIN':
                normalized_x = self.adaptive_norm_block(x_series, 'norm')
                means = torch.zeros(B * M, 1, 1, device=x_enc.device)
                stdev = torch.ones(B * M, 1, 1, device=x_enc.device)
                trend_mv = torch.zeros(B, L, M, device=x_enc.device)
                alpha_gate = None

            else:
                normalized_x = x_series
                means = torch.zeros(B * M, 1, 1, device=x_enc.device)
                stdev = torch.ones(B * M, 1, 1, device=x_enc.device)
                trend_mv = torch.zeros(B, L, M, device=x_enc.device)
                alpha_gate = None

            # ============================================================
            # 2) Global Trend Fusion
            # ============================================================
            t = trend_mv.permute(0, 2, 1).unsqueeze(-1)  # [B,M,L,1]
            naive_mean = torch.mean(t, dim=1, keepdim=True)
            deviation = torch.mean(torch.abs(t - naive_mean), dim=2, keepdim=True)
            weights = torch.softmax(-deviation / 0.5, dim=1)
            global_trend = torch.sum(t * weights, dim=1, keepdim=True)

            fused = t + self.trend_gate * global_trend
            trend_fused_mv = fused.squeeze(-1).permute(0, 2, 1)  # [B,L,M]

            if self.pred_len > self.seq_len:
                trend_for_forecast_mv = self.trend_proj(trend_fused_mv.permute(0, 2, 1)).permute(0, 2, 1)  # [B,pred_len,M]
            else:
                trend_for_forecast_mv = trend_fused_mv[:, -self.pred_len:, :]  # [B,pred_len,M]

            # ✅ α 게이트 적용
            if (self.norm_type == 'AdaNorm') and (alpha_gate is not None):
                trend_for_forecast_mv = trend_for_forecast_mv * alpha_gate

            trend_for_forecast = trend_for_forecast_mv.permute(0, 2, 1).reshape(B * M, self.pred_len, 1)

            # ============================================================
            # 3) Encoder + Head
            # ============================================================
            summary_context = self.encoder(normalized_x)
            mean_pred_norm = self.mean_head(summary_context)

            # ============================================================
            # 4) De-normalize
            # ============================================================
            if self.norm_type == 'AdaNorm':
                final_forecast = self.adaptive_norm_block.denormalize(
                    mean_pred_norm, means, stdev, trend_for_forecast
                )
            elif self.norm_type == 'RevIN':
                final_forecast = self.adaptive_norm_block(mean_pred_norm, 'denorm')
            else:
                final_forecast = mean_pred_norm

            final_forecast = final_forecast.reshape(B, M, self.pred_len).permute(0, 2, 1)  # [B,pred_len,M]

            # trend_output: 반환용 (forecast 구간)
            trend_output = trend_for_forecast_mv  # [B,pred_len,M]

            return final_forecast, trend_output
