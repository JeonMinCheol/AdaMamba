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
        d_model = configs.d_model
        d_inner = configs.d_head
        self.summary_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_inner),
            nn.GELU(),
            nn.Dropout(configs.head_dropout),
            nn.Linear(d_inner, self.pred_len)
        )
        self.num_patches = max(1, (configs.seq_len - configs.patch_len) // configs.stride + 1)
        self.token_mixer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_inner),
            nn.GELU(),
            nn.Dropout(configs.head_dropout),
            nn.Linear(d_inner, d_model),
            nn.GELU(),
        )
        self.patch_to_horizon = nn.Linear(self.num_patches, self.pred_len)
        self.horizon_norm = nn.LayerNorm(d_model)
        self.summary_to_token = nn.Linear(d_model, d_model)
        self.token_out = nn.Linear(d_model, 1)
        self._init_token_residual_branch()

    def _init_token_residual_branch(self):
        with torch.no_grad():
            self.token_out.weight.zero_()
            self.token_out.bias.zero_()

    def forward(self, summary_context, encoded_tokens):
        output = self.summary_head(summary_context)

        token_features = self.token_mixer(encoded_tokens)
        horizon_features = self.patch_to_horizon(
            token_features.transpose(1, 2)
        ).transpose(1, 2)
        summary_bias = self.summary_to_token(summary_context).unsqueeze(1)
        token_residual = self.token_out(
            self.horizon_norm(horizon_features + summary_bias)
        ).squeeze(-1)
        output = output + token_residual
        return output.view(-1, self.pred_len, 1)


class FutureTrendHead(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        hidden_dim = min(max(self.seq_len, self.pred_len), 512)

        self.base_proj = nn.Linear(self.seq_len, self.pred_len)
        self.residual_proj = nn.Sequential(
            nn.LayerNorm(self.seq_len),
            nn.Linear(self.seq_len, hidden_dim),
            nn.GELU(),
            nn.Dropout(configs.head_dropout),
            nn.Linear(hidden_dim, self.pred_len),
        )
        self._init_base_projection()
        self._init_residual_projection()

    def _init_base_projection(self):
        with torch.no_grad():
            self.base_proj.weight.zero_()
            self.base_proj.bias.zero_()

            if self.pred_len <= self.seq_len:
                start = self.seq_len - self.pred_len
                indices = torch.arange(self.pred_len)
                self.base_proj.weight[indices, start + indices] = 1.0
            else:
                self.base_proj.weight.fill_(1.0 / self.seq_len)

    def _init_residual_projection(self):
        with torch.no_grad():
            last_linear = self.residual_proj[-1]
            last_linear.weight.zero_()
            last_linear.bias.zero_()

    def forward(self, trend):
        trend_tokens = trend.permute(0, 2, 1)  # [B, M, L]
        base = self.base_proj(trend_tokens)
        residual = self.residual_proj(trend_tokens)
        forecast_trend = base + residual
        return forecast_trend.permute(0, 2, 1).contiguous()  # [B, pred_len, M]

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
        self.trend_loss_mode = getattr(configs, "trend_loss_mode", "smooth_l1")
        self.trend_target_mode = getattr(configs, "trend_target_mode", "extractor")
        self.residual_target_mode = getattr(configs, "residual_target_mode", "forecast")
        self.norm_type = configs.norm_type
        self.use_trend_forecast = configs.use_trend_forecast
        
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
        self.trend_proj = FutureTrendHead(configs)
        self._init_trend_subtract_gate()

    def _init_trend_subtract_gate(self):
        with torch.no_grad():
            self.trend_subtract_gate.weight.zero_()
            self.trend_subtract_gate.bias.fill_(-2.0)

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

        return self.trend_proj(trend)  # [B, pred_len, M]

    def get_trend_targets(self, x_enc, y):
        input_trend_target = self.adaptive_norm_block.moving_average_trend(x_enc).detach()
        full_seq = torch.cat([x_enc, y], dim=1)
        full_trend_target = self.adaptive_norm_block.moving_average_trend(full_seq).detach()
        future_trend_target = full_trend_target[:, -self.pred_len:, :]
        return input_trend_target, future_trend_target

    def needs_teacher_trend_targets(self):
        return self.norm_type == 'AdaNorm' and (
            self.residual_target_mode == 'teacher'
            or (self.use_trend_forecast and self.lambda_trend_loss > 0)
        )

    def get_residual_target_trend(self, y, trend_for_forecast, future_trend_target=None):
        if not self.use_trend_forecast:
            return torch.zeros_like(y)

        if future_trend_target is not None:
            return future_trend_target
        return trend_for_forecast.detach()

    def compute_trend_alignment_loss(self, pred_trend, target_trend):
        if self.trend_loss_mode == 'smooth_l1':
            return F.smooth_l1_loss(pred_trend, target_trend)

        pred_centered = pred_trend - pred_trend.mean(dim=1, keepdim=True)
        target_centered = target_trend - target_trend.mean(dim=1, keepdim=True)
        scale = torch.sqrt(
            torch.var(target_centered, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        level_loss = F.smooth_l1_loss(pred_centered / scale, target_centered / scale)

        if self.trend_loss_mode == 'normalized_smooth_l1':
            return level_loss

        if pred_trend.size(1) < 2 or target_trend.size(1) < 2:
            return level_loss

        pred_diff = pred_trend[:, 1:, :] - pred_trend[:, :-1, :]
        target_diff = target_trend[:, 1:, :] - target_trend[:, :-1, :]
        diff_scale = torch.sqrt(
            torch.var(target_diff, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        diff_loss = F.smooth_l1_loss(pred_diff / diff_scale, target_diff / diff_scale)

        if self.trend_loss_mode == 'diff_smooth_l1':
            return diff_loss
        if self.trend_loss_mode == 'hybrid':
            return 0.5 * (level_loss + diff_loss)

        raise ValueError(f"Unsupported trend_loss_mode: {self.trend_loss_mode}")

    def compute_loss(self, x_enc, y):
        """
        x_enc:   [B, L, M]
        y_true:  [B, pred_len, M]
        """
        B, _, M = x_enc.shape

        # ============================================================
        # 1) Normalize + Trend extraction
        # ============================================================
        normalized_x, means_series, stdev_series, means, stdev, trend = self.normalize_inputs(x_enc)
        trend_for_forecast = self.get_forecast_trend(trend)  # [B, pred_len, M]

        # ============================================================
        # 3) Encoder + Head
        # ============================================================
        summary_context, encoded_tokens = self.encoder(normalized_x)  # [B*M, d_ff], [B*M, N, d_ff]
        mean_pred_norm = self.mean_head(summary_context, encoded_tokens)  # [B*M, pred_len, 1]
        mean_pred = mean_pred_norm.reshape(B, M, self.pred_len).permute(0, 2, 1)  # [B, pred_len, M]

        # ============================================================
        # 4) De-normalize residual forecast and compose final forecast
        # ============================================================
        residual_forecast = mean_pred * stdev + means  # [B, pred_len, M]
        final_forecast = residual_forecast + trend_for_forecast  # [B, pred_len, M]

        # ============================================================
        # 5) Loss
        # ============================================================
        huber = F.smooth_l1_loss(final_forecast, y)
        q10 = quantile_loss(final_forecast, y, 0.1)
        q90 = quantile_loss(final_forecast, y, 0.9)
        quantile = q10 + q90
        trend_loss = torch.zeros((), device=huber.device, dtype=huber.dtype)

        if self.norm_type == 'AdaNorm' and self.use_trend_forecast and self.lambda_trend_loss > 0:
            with torch.no_grad():
                future_trend_target = self.adaptive_norm_block.extract_trend(y).detach()
            trend_loss = self.compute_trend_alignment_loss(trend_for_forecast, future_trend_target)

        huber_component = self.lambda_h_loss * huber
        quantile_component = self.lambda_q_loss * quantile
        trend_component = self.lambda_trend_loss * trend_loss
        total_loss = huber_component + quantile_component + trend_component

        loss_components = {
            'huber': huber_component.detach(),
            'quantile': quantile_component.detach(),
            'trend': trend_component.detach(),
            'total': total_loss.detach(),
        }
        return total_loss, loss_components

    def forward(self, x_enc, y, return_components=False):
        total_loss, loss_components = self.compute_loss(x_enc, y)
        if return_components:
            return total_loss, loss_components
        return total_loss

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
            summary_context, encoded_tokens = self.encoder(normalized_x)  # [B*M, d_ff], [B*M, N, d_ff]
            mean_pred_norm = self.mean_head(summary_context, encoded_tokens)  # [B*M, pred_len, 1]
            mean_pred_norm = mean_pred_norm.reshape(B, M, self.pred_len).permute(0, 2, 1)  # [B, pred_len, M]

            # ============================================================
            # 4) De-normalize
            # ============================================================
            final_forecast = mean_pred_norm * stdev + means + trend_for_forecast  # [B, pred_len, M]
            
            # trend_output: 반환용 (forecast 구간)
            trend_output = trend_for_forecast  # [B, pred_len, M]

            return final_forecast, trend_output
