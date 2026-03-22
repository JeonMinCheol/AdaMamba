import torch.nn as nn
from .Embed import PatchingEmbedding, PositionalEmbedding
from .SelfAttention_Family import AttentionPool

class ContextEncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.encoder_dim = configs.d_ff
        self.num_heads = self._resolve_num_heads(self.encoder_dim, configs.n_heads)
        self.patching_embedding = PatchingEmbedding(
            configs.patch_len, configs.stride, 1, self.encoder_dim
        )
        self.position_embedding = PositionalEmbedding(self.encoder_dim)
        self.ln = nn.LayerNorm(self.encoder_dim)
        self.attention_pool = AttentionPool(d_model=self.encoder_dim, n_heads=self.num_heads)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.encoder_dim,
            nhead=self.num_heads,
            dim_feedforward=self.encoder_dim * 4,
            dropout=configs.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.sequence_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=max(1, getattr(configs, "e_layers", 1)),
        )

    def _resolve_num_heads(self, d_model, requested_heads):
        requested_heads = max(1, int(requested_heads))
        for num_heads in range(min(requested_heads, d_model), 0, -1):
            if d_model % num_heads == 0:
                return num_heads
        return 1

    def forward(self, x):
        x_patched = self.patching_embedding(x)  # [B, num_patches, d_ff]
        x_patched = x_patched + self.position_embedding(x_patched)
        x_patched = self.sequence_encoder(x_patched) # [B, num_patches, d_ff]
        x_patched = self.ln(x_patched)
        
        summary_context = self.attention_pool(x_patched)

        return summary_context
