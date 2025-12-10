import torch
import torch.nn as nn
import numpy as np
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from layers.AdaMamba_backbone import *
from layers.AdaMamba_adaptive_blocks import AdaptiveNormalizationBlock
from layers.AdaMamba_experts_blocks import ContextEncoder
from utils.metrics import quantile_loss, directional_loss

torch.manual_seed(42)
np.random.seed(42)

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.c_in = configs.enc_in
        self.d_model = configs.d_model
        self.is_training = configs.is_training
        self.lambda_h_loss = configs.lambda_h_loss
        self.lambda_q_loss = configs.lambda_q_loss
        self.lambda_d_loss = configs.lambda_d_loss
        
        self.adaptive_norm_block = AdaptiveNormalizationBlock(configs)
        self.encoder = ContextEncoder(configs)
        self.mean_head = MeanPredictionHead(configs)

    def forward(self, x_enc, batch_y):
        y_true = batch_y[:, -self.pred_len:, :]

        # 1. Adaptive normalization
        normalized_x, means, stdev, trend = self.adaptive_norm_block.normalize(x_enc)

        y_true_detrended = y_true - trend[:, -y_true.size(1):, :]
        normalized_y_true = (y_true_detrended - means) / stdev

        # 2. Context encoder
        summary_context = self.encoder(normalized_x)

        # 3. Deterministic mean prediction
        mean_pred_norm = self.mean_head(summary_context)

        # 5. Losses
        huber = F.smooth_l1_loss(mean_pred_norm, normalized_y_true)
        q10 = quantile_loss(mean_pred_norm, normalized_y_true, 0.1)
        q90 = quantile_loss(mean_pred_norm, normalized_y_true, 0.9)
        d_loss = directional_loss(mean_pred_norm, normalized_y_true)
        mean_loss = self.lambda_h_loss * huber + self.lambda_q_loss * (q10 + q90) + self.lambda_d_loss * d_loss

        return mean_loss

    def sample(self, x_enc):
        self.eval()
        with torch.no_grad():
            # 1. 정규화 및 최종 컨텍스트 생성 (기존과 동일)
            normalized_x, means, stdev, trend = self.adaptive_norm_block.normalize(x_enc)
            summary_context = self.encoder(normalized_x) 
            
            # 2. 평균 예측 (mean_head)
            mean_pred_norm = self.mean_head(summary_context)
            
            # 5. 전체 스케일 복원 (De-normalization)
            trend_for_forecast = trend[:, -self.pred_len:, :]
            final_forecast = self.adaptive_norm_block.denormalize(mean_pred_norm, means, stdev, trend_for_forecast)
            
            return final_forecast
