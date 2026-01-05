# layers/AdaMamba_adaptive_blocks.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedDepthwiseSameConv1d(nn.Module):
    """
    Depthwise conv (groups=C) BUT all channels share the SAME kernel weights.
    Works with even kernels via manual SAME padding.
    Input/Output: [B, C, L]
    """
    def __init__(self, kernel_size: int, dilation: int = 1, bias: bool = True):
        super().__init__()
        self.k = int(kernel_size)
        self.d = int(dilation)

        w = torch.empty(1, 1, self.k)
        nn.init.kaiming_uniform_(w, a=5**0.5)
        self.weight = nn.Parameter(w)

        self.bias = nn.Parameter(torch.zeros(1)) if bias else None

    def forward(self, x):  # x: [B, C, L]
        if x.dim() != 3:
            raise ValueError(f"Expected [B,C,L], got {x.shape}")

        B, C, L = x.shape

        total_pad = self.d * (self.k - 1)
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left
        x = F.pad(x, (pad_left, pad_right))

        weight = self.weight.repeat(C, 1, 1)  # [C,1,k]
        bias = self.bias.repeat(C) if self.bias is not None else None  # [C]
        y = F.conv1d(x, weight, bias=bias, stride=1, padding=0, dilation=self.d, groups=C)
        return y  # [B,C,L]


class KernelSE(nn.Module):
    """
    Lightweight kernel-wise SE:
      - cat_trends: [B, C*K, L]
      - gate per (B,C,K), MLP shared across channels
    """
    def __init__(self, num_kernels: int, reduction_ratio: int = 4):
        super().__init__()
        K = int(num_kernels)
        hidden = max(1, K // int(reduction_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(K, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, K, bias=True),
            nn.Sigmoid()
        )

    def forward(self, cat_trends: torch.Tensor, C: int, K: int):
        # cat_trends: [B, C*K, L]
        B, CK, L = cat_trends.shape
        if CK != C * K:
            raise ValueError(f"KernelSE shape mismatch: got CK={CK}, expected C*K={C*K} (C={C}, K={K}).")

        # GAP over time -> [B, C*K] -> [B, C, K]
        s = cat_trends.mean(dim=2).view(B, C, K)

        # MLP shared across channels: flatten (B*C, K)
        g = self.mlp(s.view(B * C, K)).view(B, C, K)  # (0,1)
        g = g.view(B, C * K, 1)

        return cat_trends * g


class MultiKernelSharedDWSEAlpha(nn.Module):
    """
    Multi-kernel trend extractor:
      - per-channel independent conv (groups=C) BUT shared kernel weights across channels
      - KernelSE to reweight kernels
      - per-channel kernel mixing (groups=C, 1x1 conv): [C*K] -> [C]
      - alpha gate per channel: detrended = x - sigmoid(alpha)*trend

    Accepts x as [B,L,C] or [B,C,L].
    Returns:
      detrended_x [B,L,C]
      trend       [B,L,C]
      alpha_gate  [1,1,C]  (broadcast-friendly)
    """
    def __init__(
        self,
        channels: int,
        kernel_sizes=(3, 5, 8, 13, 21, 34, 55, 89),
        dropout: float = 0.0,
        kernel_se_reduction: int = 4,
        alpha_init: float = 0.0,         # sigmoid(0)=0.5 (stable start)
        use_dilation: bool = False,
        dilation_base_kernel: int = 5,
        bias: bool = True,
    ):
        super().__init__()
        self.channels = int(channels)
        self.kernel_sizes = list(kernel_sizes)
        self.K = len(self.kernel_sizes)

        self.use_dilation = bool(use_dilation)
        self.k0 = int(dilation_base_kernel)

        def rf_to_dilation(target_rf: int, k0: int) -> int:
            # eff_rf = 1 + (k0-1)*d  ~= target_rf
            return max(1, int(round((target_rf - 1) / (k0 - 1))))

        branches = []
        for rf in self.kernel_sizes:
            rf = int(rf)
            if self.use_dilation and rf > self.k0:
                d = rf_to_dilation(rf, self.k0)
                branches.append(SharedDepthwiseSameConv1d(kernel_size=self.k0, dilation=d, bias=bias))
            else:
                branches.append(SharedDepthwiseSameConv1d(kernel_size=rf, dilation=1, bias=bias))
        self.branches = nn.ModuleList(branches)

        self.kernel_se = KernelSE(num_kernels=self.K, reduction_ratio=kernel_se_reduction)

        # per-channel kernel mixing: [B, C*K, L] -> [B, C, L], channel-wise independent mixing
        self.mix = nn.Conv1d(
            in_channels=self.channels * self.K,
            out_channels=self.channels,
            kernel_size=1,
            groups=self.channels,
            bias=True
        )

        self.drop = nn.Dropout(dropout)

        # alpha gate per channel
        self.alpha = nn.Parameter(torch.full((1, self.channels, 1), float(alpha_init)))

    def forward(self, x: torch.Tensor):
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {x.shape}")

        # robust layout handling -> x_c: [B,C,L]
        if x.shape[-1] == self.channels:
            x_c = x.permute(0, 2, 1)
        elif x.shape[1] == self.channels:
            x_c = x
        else:
            raise ValueError(
                f"Input shape {x.shape} doesn't match channels={self.channels}. "
                f"Expected last dim or dim=1 to be {self.channels}."
            )

        outs = [b(x_c) for b in self.branches]          # each [B,C,L]
        cat = torch.cat(outs, dim=1)                    # [B,C*K,L]

        C = x_c.size(1)
        cat = self.kernel_se(cat, C=C, K=self.K)        # [B,C*K,L]
        trend = self.drop(self.mix(cat))                # [B,C,L]

        a_ch = torch.sigmoid(self.alpha)                # [1,C,1]
        detrended = x_c - a_ch * trend                  # [B,C,L]

        # return alpha_gate as [1,1,C] for easy broadcast with [B,*,C]
        alpha_gate = a_ch.permute(0, 2, 1)              # [1,1,C]

        return detrended.permute(0, 2, 1), trend.permute(0, 2, 1), alpha_gate


class AdaptiveNormalizationBlock(nn.Module):
    """
    AdaNorm block (multivariate):
      normalize(x:[B,L,M]) -> x_norm, means, stdev, trend, alpha_gate
      denormalize(y_norm:[B,H,M], means, stdev, trend_for_forecast:[B,H,M]) -> y:[B,H,M]
    """
    def __init__(self, configs, kernel_sizes):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        channels = configs.enc_in
        dropout = configs.dropout

        # optional knobs
        kernel_se_reduction = configs.reduction_ratio
        alpha_init = float(getattr(configs, "alpha_init", 0.0))          # 추천: 0.0 or 1.0
        use_dilation = bool(getattr(configs, "use_dilation", False))
        dilation_base_kernel = int(getattr(configs, "dilation_base_kernel", 5))

        self.detrender = MultiKernelSharedDWSEAlpha(
            channels=channels,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
            kernel_se_reduction=kernel_se_reduction,
            alpha_init=alpha_init,
            use_dilation=use_dilation,
            dilation_base_kernel=dilation_base_kernel,
            bias=True,
        )

    def normalize(self, x: torch.Tensor):
        # x: [B,L,C]
        detrended_x, trend, alpha_gate = self.detrender(x)  # [B,L,C], [B,L,C], [1,1,C]
        means = detrended_x.mean(dim=1, keepdim=True).detach()  # [B,1,C]
        stdev = torch.sqrt(torch.var(detrended_x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # [B,1,C]
        x_norm = (detrended_x - means) / stdev
        return x_norm, means, stdev, trend, alpha_gate

    def denormalize(self, y_norm: torch.Tensor, means: torch.Tensor, stdev: torch.Tensor, trend: torch.Tensor):
        # y_norm: [B,H,C], means/stdev: [B,1,C]
        y = y_norm * stdev + means  # [B,H,C]

        if trend.size(1) == y.size(1):
            return y + trend
        elif trend.size(1) > y.size(1):
            return y + trend[:, -y.size(1):, :]
        else:
            raise ValueError(f"trend length {trend.size(1)} < y length {y.size(1)}")
