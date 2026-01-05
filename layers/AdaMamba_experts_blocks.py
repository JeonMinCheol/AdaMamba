import torch
import torch.nn as nn
from .Embed import PatchingEmbedding, PositionalEmbedding
from .SelfAttention_Family import AttentionPool
from mambapy.mamba import Mamba, MambaConfig

import torch
import torch.nn as nn

class ContextualTSA(nn.Module):
    def __init__(self, d_model, n_layers=2, d_state=16, expand_factor=2, dropout=0.1):
        super().__init__()
        assert d_model % 2 == 0
        self.d_model = d_model
        self.d_channel = d_model // 2

        config = MambaConfig(
            d_model=self.d_channel,
            n_layers=n_layers,
            d_state=d_state,
            expand_factor=expand_factor
        )
        self.mamba = Mamba(config)

        # Output fusion
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        src: [B, L, d_model]
        return: output [B, L, d_model]
        """
        src = self.norm(src)

        x1, x2 = src.chunk(2, dim=-1)  # [B,L,D/2] each
        context = self.mamba(x1)       # [B,L,D/2] — temporal context with gating

        y2 = x2 + context
        out = torch.cat([x1, y2], dim=-1)
        return src + self.dropout(self.out_proj(out))

class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class MoETSAEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.tsa = ContextualTSA(d_model, dropout=dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src):
        norm_src = self.norm1(src) 
        src2 = self.tsa(norm_src)
        src = norm_src + src2

        norm_src = self.norm2(src)
        src2 = self.ffn(norm_src)
        src = norm_src + src2 

        return src

class ContextEncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.patching_embedding = PatchingEmbedding(configs.patch_len, configs.stride, 1, configs.d_model)
        self.position_embedding = PositionalEmbedding(configs.d_model)
        self.encoder_layer =  MoETSAEncoderLayer(configs.d_model, configs.d_ff, configs.dropout)
        self.attention_pool = AttentionPool(d_model=configs.d_model, n_heads=configs.n_heads)

    def forward(self, x):
        x_patched = self.patching_embedding(x)  # [B, num_patches, d_model]
        x_patched = x_patched + self.position_embedding(x_patched)
        x_patched = self.encoder_layer(x_patched)
        
        summary_context = self.attention_pool(x_patched)

        return summary_context