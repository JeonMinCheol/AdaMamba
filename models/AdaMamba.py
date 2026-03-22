import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers.AdaMamba_adaptive_blocks import AdaptiveNormalizationBlock
from layers.AdaMamba_encoder_blocks import ContextEncoder
from layers.RevIN import RevIN
from utils.metrics import quantile_loss

class PredictionHead(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        d_model = configs.d_ff
        d_inner = configs.d_head
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_inner),
            nn.GELU(),
            nn.Dropout(configs.head_dropout),
            nn.Linear(d_inner, self.pred_len)
        )

    def forward(self, summary_context):
        output = self.mlp(summary_context)
        return output.view(-1, self.pred_len, 1)

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
        self.lambda_trend_loss = getattr(configs, "lambda_trend_loss", 0.0)
        self.norm_type = configs.norm_type
        self.use_trend_forecast = configs.use_trend_forecast
        self.use_encoder_input_mask = getattr(configs, "use_encoder_input_mask", False)
        self.encoder_mask_ratio = getattr(configs, "encoder_mask_ratio", 0.0)
        self.encoder_mask_block = max(
            1, int(getattr(configs, "encoder_mask_block", getattr(configs, "patch_len", 1)))
        )
        self.encoder_mask_keep_last = max(
            0, int(getattr(configs, "encoder_mask_keep_last", getattr(configs, "patch_len", 0)))
        )
        if self.norm_type == 'AdaNorm':
            self.adaptive_norm_block = AdaptiveNormalizationBlock(configs)
        elif self.norm_type == 'RevIN':
            self.adaptive_norm_block = RevIN(configs.enc_in, affine=True)
        elif self.norm_type == 'None':
            self.adaptive_norm_block = nn.Identity()  # No Norm
        else:
            raise ValueError(
                f"Unsupported norm_type '{self.norm_type}'. "
                "Use one of [AdaNorm, RevIN, None]."
            )
        
        self.encoder = ContextEncoder(configs)
        self.mean_head = PredictionHead(configs)
        self.trend_gate_feature_dim = 4
        self.trend_subtract_gate = nn.Linear(self.trend_gate_feature_dim, 1)
        self.trend_proj = nn.Linear(self.seq_len, self.pred_len)
        self._init_trend_subtract_gate()
        self._init_trend_projection()

    def _init_trend_subtract_gate(self):
        with torch.no_grad():
            self.trend_subtract_gate.weight.zero_()
            self.trend_subtract_gate.bias.fill_(-2.0)

    def _init_trend_projection(self):
        with torch.no_grad():
            self.trend_proj.weight.zero_()
            self.trend_proj.bias.zero_()

            if self.pred_len <= self.seq_len:
                start = self.seq_len - self.pred_len
                indices = torch.arange(self.pred_len)
                self.trend_proj.weight[indices, start + indices] = 1.0
            else:
                self.trend_proj.weight.fill_(1.0 / self.seq_len)

    def flatten_series(self, x):
        B, L, M = x.shape  # x: [B, L, M]
        return x.permute(0, 2, 1).contiguous().reshape(B * M, L, 1)  # [B*M, L, 1]

    def flatten_stats(self, stats):
        B, _, M = stats.shape  # stats: [B, 1, M]
        return stats.permute(0, 2, 1).contiguous().reshape(B * M, 1, 1)  # [B*M, 1, 1]

    def summarize_trend_for_gate(self, trend):
        trend_tokens = trend.detach().permute(0, 2, 1).contiguous()  # [B, M, L]
        trend_mean = trend_tokens.mean(dim=-1, keepdim=True)
        trend_std = trend_tokens.std(dim=-1, unbiased=False, keepdim=True)
        trend_last = trend_tokens[..., -1:]
        trend_delta = trend_tokens[..., -1:] - trend_tokens[..., :1]
        trend_abs_mean = trend_tokens.abs().mean(dim=-1, keepdim=True)
        trend_min = trend_tokens.min(dim=-1, keepdim=True).values
        trend_max = trend_tokens.max(dim=-1, keepdim=True).values
        trend_range = trend_max - trend_min

        # 8 * [B, M, 1]
        available_features = [
            trend_mean,
            trend_std,
            trend_last,
            trend_delta,
            trend_abs_mean,
            trend_min,
            trend_max,
            trend_range,
        ]
        if self.trend_gate_feature_dim > len(available_features):
            raise ValueError(
                f"trend_gate_feature_dim={self.trend_gate_feature_dim} exceeds available trend summaries={len(available_features)}"
            )
        return torch.cat(available_features[:self.trend_gate_feature_dim], dim=-1) # [B, M, trend_gate_feature_dim]

    def normalize_inputs(self, x_enc):
        B, L, M = x_enc.shape

        if self.norm_type == 'AdaNorm':
            trend = self.adaptive_norm_block.extract_trend(x_enc)  # [B, L, M]
            detrend_scale = self.get_trend_subtract_scale(trend)  # [B, 1, M]
            detrended_x = x_enc - trend * detrend_scale  # [B, L, M]
            means, stdev = self.adaptive_norm_block.compute_stats(detrended_x)  # [B, 1, M], [B, 1, M]
            x_norm = self.adaptive_norm_block.apply_normalization(
                detrended_x, means, stdev
            )  # [B, L, M]
        elif self.norm_type == 'RevIN':
            trend = torch.zeros_like(x_enc)  # [B, L, M]
            x_norm = self.adaptive_norm_block(x_enc, 'norm')  # [B, L, M]
            means = self.adaptive_norm_block.mean  # [B, 1, M]
            stdev = self.adaptive_norm_block.stdev  # [B, 1, M]
            detrended_x = x_enc  # [B, L, M]
        else:
            trend = torch.zeros_like(x_enc)  # [B, L, M]
            x_norm = x_enc  # [B, L, M]
            detrended_x = x_enc  # [B, L, M]
            means = torch.zeros(B, 1, M, device=x_enc.device, dtype=x_enc.dtype)  # [B, 1, M]
            stdev = torch.ones(B, 1, M, device=x_enc.device, dtype=x_enc.dtype)  # [B, 1, M]

        normalized_x = self.flatten_series(x_norm)  # [B*M, L, 1]
        means_series = self.flatten_stats(means)  # [B*M, 1, 1]
        stdev_series = self.flatten_stats(stdev)  # [B*M, 1, 1]
        return normalized_x, means_series, stdev_series, means, stdev, trend

    def get_trend_subtract_scale(self, trend=None):
        # Keep trend subtraction close to baseline (beta ~= 1.0) and only allow
        # limited relief to prevent over-detrending.
        if self.norm_type != 'AdaNorm':
            return 1.0
        if trend is None:
            raise ValueError("trend must be provided when norm_type is 'AdaNorm'.")
        trend_features = self.summarize_trend_for_gate(trend)  # [B, M, trend_gate_feature_dim]
        gate_logits = self.trend_subtract_gate(trend_features).permute(0, 2, 1)  # [B, 1, M]
        relief_ratio = torch.sigmoid(gate_logits)  # [B, 1, M]
        return 1.0 - relief_ratio  # [B, 1, M]

    def get_forecast_trend(self, trend):
        if not self.use_trend_forecast:
            bsz, _, n_vars = trend.shape
            return torch.zeros(
                bsz, self.pred_len, n_vars, device=trend.device, dtype=trend.dtype
            )  # [B, pred_len, M]
        
        trend_tokens = trend.permute(0, 2, 1)  # [B, M, L]
        projected_trend = self.trend_proj(trend_tokens)  # [B, M, pred_len]
        return projected_trend.permute(0, 2, 1).contiguous()  # [B, pred_len, M]

    def forward(self, x_enc, y):
        """
        x_enc:   [B, L, M]
        y_true:  [B, pred_len, M]
        """
        B, L, M = x_enc.shape

        # ============================================================
        # 1) Normalize + Trend extraction
        # ============================================================
        normalized_x, means_series, stdev_series, means, stdev, trend = self.normalize_inputs(x_enc)
        trend_for_forecast = self.get_forecast_trend(trend)  # [B, pred_len, M]

        # ============================================================
        # 3) Encoder + Head
        # ============================================================
        summary_context = self.encoder(normalized_x)  # [B*M, d_ff]
        mean_pred_norm = self.mean_head(summary_context)  # [B*M, pred_len, 1]

        # ============================================================
        # 4) y_true detrend + normalize (loss용)
        # ============================================================
        y_true_detrended = y - trend_for_forecast  # [B, pred_len, M]
        y_true_detrended_series = self.flatten_series(y_true_detrended)  # [B*M, pred_len, 1]

        if self.norm_type == 'None':
            normalized_y_true = y_true_detrended_series
        elif self.norm_type == 'AdaNorm':
            normalized_y_true = self.adaptive_norm_block.apply_normalization(
                y_true_detrended, means, stdev
            )  # [B, pred_len, M]
            normalized_y_true = self.flatten_series(normalized_y_true)  # [B*M, pred_len, 1]
        else:
            normalized_y_true = (y_true_detrended_series - means_series) / stdev_series  # [B*M, pred_len, 1]

        # ============================================================
        # 5) Loss
        # ============================================================
        huber = F.smooth_l1_loss(mean_pred_norm, normalized_y_true)
        q10 = quantile_loss(mean_pred_norm, normalized_y_true, 0.1)
        q90 = quantile_loss(mean_pred_norm, normalized_y_true, 0.9)
        mean_loss = self.lambda_h_loss * huber + self.lambda_q_loss * (q10 + q90)
        trend_loss = 0.0

        if self.norm_type == 'AdaNorm' and self.use_trend_forecast and self.lambda_trend_loss > 0:
            with torch.no_grad():
                future_trend_target = self.adaptive_norm_block.extract_trend(y).detach()  # [B, pred_len, M]
            trend_loss = F.smooth_l1_loss(trend_for_forecast, future_trend_target)

        return mean_loss + self.lambda_trend_loss * trend_loss

    def sample(self, x_enc):
        self.eval()
        with torch.no_grad():
            B, L, M = x_enc.shape

            # ============================================================
            # 1) Normalize + Trend extraction
            # ============================================================
            normalized_x, means_series, stdev_series, means, stdev, trend = self.normalize_inputs(x_enc)
            trend_for_forecast = self.get_forecast_trend(trend)  # [B, pred_len, M]

            # ============================================================
            # 3) Encoder + Head
            # ============================================================
            summary_context = self.encoder(normalized_x)  # [B*M, d_ff]
            mean_pred_norm = self.mean_head(summary_context)  # [B*M, pred_len, 1]
            mean_pred_norm = mean_pred_norm.reshape(B, M, self.pred_len).permute(0, 2, 1)  # [B, pred_len, M]

            # ============================================================
            # 4) De-normalize
            # ============================================================
            final_forecast = mean_pred_norm * stdev + means + trend_for_forecast  # [B, pred_len, M]
            
            # trend_output: 반환용 (forecast 구간)
            trend_output = trend_for_forecast  # [B, pred_len, M]

            return final_forecast, trend_output
