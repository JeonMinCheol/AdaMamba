import torch
import torch.nn as nn
import torch.nn.functional as F
from .Embed import PatchingEmbedding, PositionalEmbedding
from .SelfAttention_Family import AttentionPool
from mambapy.mamba import Mamba, MambaConfig

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextualTSA(nn.Module):
    def __init__(self, d_model, n_layers=2, d_state=16, expand_factor=2, dropout=0.1):
        super().__init__()
        assert d_model % 2 == 0
        self.d_model = d_model
        self.d_channel = d_model // 2

        # Mamba for contextual feature extraction (replaces QKV+saliency)
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

class MoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, moe_output_dropout, moe_expert_dropout, tau=2.2):
        super().__init__()
        self.num_experts = num_experts
        self.tau = tau
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),               # 1. 활성화 함수 추가 (일반적인 구조)
                nn.Dropout(moe_expert_dropout),         # 2. 🔥 Hidden Layer Dropout 추가 (필수)
                nn.Linear(d_ff, d_model),
                nn.Dropout(moe_expert_dropout)          # 3. Output Dropout 추가 (선택사항)
            ) for _ in range(num_experts)
        ])
        self.gating_network = nn.Linear(d_model, num_experts)
        self.dropout = nn.Dropout(moe_output_dropout)

    def forward(self, x):
        # x shape: [B, L, C]
        # 1️⃣ gating logits → softmax with temperature
        logits = self.gating_network(x)               # [B, L, num_experts]
        gates = F.softmax(logits / self.tau, dim=-1)  # temperature-scaled soft routing

        # 2️⃣ 각 expert를 병렬로 계산
        expert_outputs = []
        for expert in self.experts:
            y_i = expert(x)                           # [B, L, D]
            expert_outputs.append(y_i.unsqueeze(-2))  # [B, L, 1, D]
        expert_outputs = torch.cat(expert_outputs, dim=-2)  # [B, L, num_experts, D]

        # 3️⃣ 가중합
        gates = gates.unsqueeze(-1)                   # [B, L, num_experts, 1]
        output = (expert_outputs * gates).sum(dim=-2) # [B, L, D]
        return self.dropout(output)

class MoETSAEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, moe_output_dropout, moe_expert_dropout, tau, dropout=0.1):
        super().__init__()
        self.tsa = ContextualTSA(d_model, dropout=dropout)
        self.moe_ffn = MoE(d_model, d_ff, num_experts, moe_output_dropout, moe_expert_dropout, tau)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src):
        norm_src = self.norm1(src) 
        src2 = self.tsa(norm_src)
        src = norm_src + src2

        norm_src = self.norm2(src)
        src2 = self.moe_ffn(norm_src)
        src = norm_src + src2 

        return src

class ContextEncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.patching_embedding = PatchingEmbedding(configs.patch_len, configs.stride, configs.enc_in, configs.d_model)
        self.position_embedding = PositionalEmbedding(configs.d_model)
        self.encoder_layer =  MoETSAEncoderLayer(configs.d_model, configs.d_ff, configs.num_experts, configs.moe_output_dropout, configs.moe_expert_dropout, configs.tau, configs.dropout)
        self.attention_pool = AttentionPool(d_model=configs.d_model, n_heads=configs.n_heads)

    def forward(self, x):
        x_patched = self.patching_embedding(x)  # [B, num_patches, d_model]
        x_patched = x_patched + self.position_embedding(x_patched)
        x_patched = self.encoder_layer(x_patched)
        
        summary_context = self.attention_pool(x_patched)

        return summary_context