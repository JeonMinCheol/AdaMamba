import torch
import torch.nn as nn
import torch.nn.functional as F

class FusedSharedMultiKernelSameConv1d(nn.Module):
    """
    Compute all shared-kernel SAME convolutions in a single conv1d call.
    Input:  [B, M, L]
    Output: [B, M, K, L]
    """
    def __init__(self, kernel_sizes: list[int], bias: bool = True):
        super().__init__()
        self.kernels = [int(kernel_size) for kernel_size in kernel_sizes]
        self.K = len(self.kernels)
        self.max_k = max(self.kernels)
        self.max_pad_left = (self.max_k - 1) // 2
        self.max_pad_right = (self.max_k - 1) - self.max_pad_left

        weight_bank = torch.zeros(self.K, 1, self.max_k)
        kernel_mask = torch.zeros(self.K, 1, self.max_k)
        for idx, kernel_size in enumerate(self.kernels):
            kernel_pad_left = (kernel_size - 1) // 2
            start = self.max_pad_left - kernel_pad_left
            end = start + kernel_size

            weight_bank[idx, 0, start:end] = 1.0 / kernel_size
            kernel_mask[idx, 0, start:end] = 1.0

        self.weight_bank = nn.Parameter(weight_bank)
        self.register_buffer("kernel_mask", kernel_mask)
        self.bias = nn.Parameter(torch.zeros(self.K)) if bias else None

    def forward(self, x):  # x: [B, M, L]
        bsz, n_vars, seq_len = x.shape
        x_flat = x.reshape(bsz * n_vars, 1, seq_len)
        x_padded = F.pad(x_flat, (self.max_pad_left, self.max_pad_right))

        fused_weight = self.weight_bank * self.kernel_mask
        y = F.conv1d(x_padded, fused_weight, bias=self.bias, stride=1, padding=0)
        return y.view(bsz, n_vars, self.K, seq_len)


class SoftmaxKernelWeighting(nn.Module):
    """
    Learn kernel weights with a global prior plus optional sample/channel-specific logits.
    Input:
      stacked [B, M, K, L]
    Output:
      alpha   [B, M, K] with sum(alpha, dim=K) = 1
    """
    def __init__(self, num_kernels, hidden_dim=64, use_dynamic=True):
        super().__init__()
        self.K = num_kernels
        self.use_dynamic = use_dynamic
        self.kernel_prior = nn.Parameter(torch.zeros(num_kernels))
        self.last_alpha = None

        if self.use_dynamic:
            self.mlp = nn.Sequential(
                nn.Linear(num_kernels, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_kernels),
            )
            self._init_dynamic_logits()

    def _init_dynamic_logits(self):
        # Keep the dynamic branch trainable, but make the initial kernel mix
        # uniform so normalization is not sample-random at step 0.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, stacked: torch.Tensor):
        # stacked: [B, M, K, L]
        B, M, K, _ = stacked.shape
        logits = self.kernel_prior.view(1, 1, K).expand(B, M, K)

        if self.use_dynamic:
            pooled = stacked.mean(dim=-1)
            dyn_logits = self.mlp(pooled.view(B * M, K)).view(B, M, K)
            logits = logits + dyn_logits

        alpha = torch.softmax(logits, dim=2)
        self.last_alpha = alpha.detach()
        return alpha

class MultiKernelTrendExtractor(nn.Module):
    """
    Multi-kernel trend extractor:
      - per-channel independent conv (groups=M) BUT shared kernel weights across channels
      - softmax kernel attention to reweight kernels
      - weighted sum over kernel axis: [B, M, K, L] -> [B, M, L]

    Accepts x as [B,L,M]
    Returns:
      detrended_x [B,L,M]
      trend       [B,L,M]
    """
    def __init__(
        self,
        configs,
        hidden_dim=64,
        bias: bool = True,
    ):
        super().__init__()
        self.configs = configs
        self.features = int(configs.enc_in)
        self.kernels = list(configs.kernels)
        self.K = len(self.kernels)

        self.fused_conv = FusedSharedMultiKernelSameConv1d(self.kernels, bias=bias)
        self.kernel_attention = SoftmaxKernelWeighting(
            num_kernels=self.K,
            hidden_dim=hidden_dim,
            use_dynamic=configs.use_dynamic,
        )

    def extract_trend(self, x: torch.Tensor):
        # x: [B,L,M]
        x_tokens = x.permute(0, 2, 1).contiguous()  # [B,M,L]

        stacked = self.fused_conv(x_tokens)                    # [B, M, K, L]
        alpha = self.kernel_attention(stacked)                 # [B, M, K]
        trend = (stacked * alpha.unsqueeze(-1)).sum(dim=2)     # [B, M, L]
        return trend.permute(0, 2, 1).contiguous()             # [B, L, M]

    def detrend(self, x: torch.Tensor, trend: torch.Tensor = None):
        if trend is None:
            trend = self.extract_trend(x)
        detrended_x = x - trend
        return detrended_x, trend

    def forward(self, x: torch.Tensor):
        return self.detrend(x)


class AdaptiveNormalizationBlock(nn.Module):
    """
    AdaNorm block (multivariate):
      normalize(x:[B,L,M]) -> x_norm, means, stdev, trend
      denormalize(y_norm:[B,H,M], means, stdev, trend_for_forecast:[B,H,M]) -> y:[B,H,M]
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        hidden_dim=configs.n_hdim
        self.detrender = MultiKernelTrendExtractor(
            configs,
            hidden_dim=hidden_dim,
            bias=True,
        )
        self.eps = 1e-5

    def extract_trend(self, x: torch.Tensor):
        return self.detrender.extract_trend(x)

    def detrend(self, x: torch.Tensor, trend: torch.Tensor = None):
        return self.detrender.detrend(x, trend=trend)

    def compute_stats(self, detrended_x: torch.Tensor):
        means = detrended_x.mean(dim=1, keepdim=True).detach()  # [B, 1, M]
        stdev = torch.sqrt(
            torch.var(detrended_x, dim=1, keepdim=True, unbiased=False) + self.eps
        ).detach()  # [B, 1, M]
        return means, stdev

    def apply_normalization(
        self,
        detrended_x: torch.Tensor,
        means: torch.Tensor,
        stdev: torch.Tensor,
    ):
        return (detrended_x - means) / stdev

    def normalize_detrended(self, detrended_x: torch.Tensor):
        means, stdev = self.compute_stats(detrended_x)
        x_norm = self.apply_normalization(detrended_x, means, stdev)
        return x_norm, means, stdev

    def decompose(self, x: torch.Tensor, trend: torch.Tensor = None):
        detrended_x, trend = self.detrend(x, trend=trend)
        means, stdev = self.compute_stats(detrended_x)
        return {
            'trend': trend,
            'detrended_x': detrended_x,
            'means': means,
            'stdev': stdev,
        }

    def normalize(self, x: torch.Tensor):
        # x: [B,L,M]
        detrended_x, trend = self.detrend(x)
        x_norm, means, stdev = self.normalize_detrended(detrended_x)
        return x_norm, means, stdev, trend

    def denormalize(self, y_norm: torch.Tensor, means: torch.Tensor, stdev: torch.Tensor, trend: torch.Tensor):
        # y_norm: [B,H,C], means/stdev: [B,1,C]
        y = y_norm * stdev + means  # [B,H,C]

        if trend.size(1) == y.size(1):
            return y + trend
        elif trend.size(1) > y.size(1):
            return y + trend[:, -y.size(1):, :]
        else:
            raise ValueError(f"trend length {trend.size(1)} < y length {y.size(1)}")
