import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, int(in_channels // reduction_ratio), bias=True),
            nn.GELU(),
            nn.Linear(int(in_channels // reduction_ratio), in_channels, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [Batch, Channels, Seq_Len]
        B, C, _ = x.shape
        y = self.squeeze(x).view(B, C) # Squeeze: [B, C]
        y = self.excitation(y).view(B, C, 1) # Excitation: [B, C, 1]
        return x * y.expand_as(x) # Scale (Re-calibrate)

class MultiScaleTrendSE(nn.Module): # kernel size = pibonacci [3,5,8,13,21,34,55,89]
    def __init__(self, enc_in, seq_len, pred_len, dropout, kernel_sizes=[3,5,8,13,21,34,55,89], reduction_ratio=4):
        super().__init__()
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.kernel_sizes = kernel_sizes
        self.trend_convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.enc_in, out_channels=self.enc_in, kernel_size=k, padding='same')
            for k in kernel_sizes
        ])
        num_combined_channels = self.enc_in * len(kernel_sizes)
        self.se_block = SEBlock(num_combined_channels, reduction_ratio=reduction_ratio)
        self.projection = nn.Linear(num_combined_channels, self.enc_in)
        self.projection_dropout = nn.Dropout(dropout) 

    def forward(self, x):
        x_transposed = x.permute(0, 2, 1)
        trend_outputs = [conv(x_transposed) for conv in self.trend_convs]
        concatenated_trends = torch.cat(trend_outputs, dim=1)
        
        recalibrated_trends = self.se_block(concatenated_trends)
        recalibrated_trends = recalibrated_trends.permute(0, 2, 1)
        projected_output = self.projection_dropout(self.projection(recalibrated_trends))

        final_trend = projected_output
        detrended_x = x - final_trend

        return detrended_x, final_trend

class AdaptiveNormalizationBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.detrender_context_generator = MultiScaleTrendSE(configs.enc_in, configs.seq_len, configs.pred_len, configs.dropout, reduction_ratio=configs.reduction_ratio)
        
    def normalize(self, x):
        # x shape: [B, L, C]
        # context shape: [B, d_model]
        detrended_x, trend = self.detrender_context_generator(x)
        means = detrended_x.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(detrended_x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        normalized_x = (detrended_x - means) / stdev
        return normalized_x, means, stdev, trend

    def denormalize(self, y_norm, means, stdev, trend):
        y_detrended = y_norm * stdev + means
        final_y = y_detrended + trend[:, -y_detrended.size(1):, :]
        return final_y