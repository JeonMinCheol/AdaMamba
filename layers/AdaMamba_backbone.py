import torch.nn as nn
import torch.nn.functional as F

class PredictionHead(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        d_model = configs.d_model
        d_inner = configs.d_head
        self.resi_conn_dropout = nn.Dropout(configs.head_dropout)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_inner),
            nn.GELU(),
            nn.Dropout(configs.head_dropout),
            nn.Linear(d_inner, self.pred_len)
        )
        self.resi_conn = nn.Linear(d_model, self.pred_len)

    def forward(self, summary_context):
        output = self.mlp(summary_context) +  self.shortcut_dropout(self.resi_conn(summary_context))
        return output.view(-1, self.pred_len, 1)