import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from .Normalizing_Flows import create_conditional_nsf_flow
    
class PredictionHead(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        d_model = configs.d_model
        d_inner = configs.d_head
        self.shortcut_dropout = nn.Dropout(configs.head_dropout)
        
        # 1. 메인 경로 (비선형성을 학습)
        # 안정적인 LayerNorm과 GELU 사용
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_inner),
            nn.GELU(),
            nn.Dropout(configs.head_dropout),
            nn.Linear(d_inner, self.pred_len * self.enc_in)
        )

        # 2. 지름길 경로 (Shortcut)
        # 입력에서 출력으로 바로 가는 고속도로를 뚫어줌
        # 이것이 학습 안정성을 보장하고 성능 저하를 막음
        self.shortcut = nn.Linear(d_model, self.pred_len * self.enc_in)

    def forward(self, summary_context):
        # 메인 경로의 결과와 지름길 경로의 결과를 더함
        # ResNet의 원리와 유사하여 그라디언트 소실을 방지
        output = self.mlp(summary_context) +  self.shortcut_dropout(self.shortcut(summary_context))
        
        return output.view(-1, self.pred_len, self.enc_in)

# class ProbabilisticResidualModel(nn.Module):
#     def __init__(self, configs):
#         super().__init__()
#         self.c_in = configs.enc_in
#         self.pred_len = configs.pred_len
#         self.d_model = configs.d_model
#         self.window_size = configs.patch_len
#         self.stride = configs.stride
#         self.flow_head = create_conditional_nsf_flow(configs)

#     def forward(self, summary_context, y):
#         # y shape: [B, pred_len, c_in]
#         B = y.size(0)
        
#         # --- 💡 [핵심] for 루프를 unfold 연산으로 대체 ---
#         # 1. y를 [B, C, L] 형태로 변환
#         y_transposed = y.permute(0, 2, 1)
        
#         # 2. unfold를 사용하여 모든 윈도우를 한 번에 추출
#         # 결과 shape: [B, C, num_windows, window_size]
#         y_windows = y_transposed.unfold(dimension=2, size=self.window_size, step=self.stride)
        
#         # 3. Flow에 입력하기 위해 shape 재정렬 및 flatten
#         # [B, C, num_windows, window_size] -> [B, num_windows, C, window_size]
#         y_windows = y_windows.permute(0, 2, 1, 3)
#         num_windows = y_windows.shape[1]
        
#         # [B, num_windows, C, window_size] -> [B * num_windows, C * window_size]
#         y_windows_flat = y_windows.reshape(B * num_windows, -1)
        
#         # 4. summary_context를 각 윈도우에 맞게 확장
#         # [B, d_model] -> [B, num_windows, d_model] -> [B * num_windows, d_model]
#         expanded_context = summary_context.unsqueeze(1).expand(-1, num_windows, -1)
#         expanded_context = expanded_context.reshape(B * num_windows, -1)
        
#         log_prob_windows = self.flow_head.log_prob(y_windows_flat, context=expanded_context)
        
#         nll_loss = -log_prob_windows.mean()
        
#         return nll_loss

#     def sample(self, summary_context, num_samples):
#         self.eval()
#         with torch.no_grad():
#             device = summary_context.device
#             B = summary_context.size(0)
#             S = num_samples
#             W, C, L, stride = self.window_size, self.c_in, self.pred_len, self.stride
            
#             num_windows = (L - W) // stride + 1

#             # 1) 컨텍스트 확장
#             expanded_context = summary_context.unsqueeze(1).expand(-1, num_windows, -1)
#             expanded_context = expanded_context.reshape(B * num_windows, -1)  # [B*NW, d]

#             # 2) 모든 윈도우 샘플 생성 + 로그 우도 동시 계산 (개선점 1)
#             # sample_windows_flat: [S, B*NW, W*C]
#             # logp_flat: [S, B*NW]
#             sample_windows_flat, logp_flat = self.flow_head.sample_and_log_prob(
#                 S, context=expanded_context
#             )

#             # 3) 윈도우 텐서로 복원: [S, B, NW, W, C]
#             sample_windows = sample_windows_flat.view(S, B, num_windows, W, C)

#             # 4) 우도 가중치 계산 (로그 우도 텐서 복원)
#             logp = logp_flat.view(S, B, num_windows)
#             logp_centered = logp - logp.amax(dim=2, keepdim=True)
#             w_win = torch.softmax(logp_centered, dim=2)  # [S, B, NW]

#             # 5) 가중 OLA 벡터화 (개선점 2: F.fold 사용)
            
#             # 5a) 가중치 준비
#             w_hann = torch.hann_window(W, periodic=False, device=device) # [W]
            
#             # 브로드캐스팅을 위한 텐서 shape
#             w_time = w_hann.view(1, 1, 1, W, 1)  # [1, 1, 1, W, 1] (Hann)
#             w_prob = w_win.view(S, B, num_windows, 1, 1) # [S, B, NW, 1, 1] (Likelihood)

#             # 5b) 두 가중치 적용
#             # [S, B, NW, W, C] * [1, 1, 1, W, 1] * [S, B, NW, 1, 1]
#             weighted_samples = sample_windows * w_time * w_prob
            
#             # 정규화용 가중치 합 (C 채널로 expand)
#             # [1, 1, 1, W, 1] * [S, B, NW, 1, 1] -> [S, B, NW, W, 1]
#             combined_weights = (w_time * w_prob).expand(-1, -1, -1, -1, C)

#             # 5c) F.fold를 위한 텐서 변형
#             # fold 입력 형식: [N, C*K, L_out] = [S*B, (C*W), NW]
#             N = S * B
            
#             # [S, B, NW, W, C] -> [S, B, NW, C, W] -> [S*B, NW, C*W] -> [N, C*W, NW]
#             fold_input = weighted_samples.permute(0, 1, 2, 4, 3).reshape(N, num_windows, C * W).permute(0, 2, 1)
#             fold_weights = combined_weights.permute(0, 1, 2, 4, 3).reshape(N, num_windows, C * W).permute(0, 2, 1)

#             # 5d) F.fold 실행 (1D 시계열을 2D 이미지처럼 처리)
#             output_size = (L, 1)  # (pred_len, 1)
#             kernel_size = (W, 1)  # (window_size, 1)
#             stride_1d = (stride, 1) # (stride, 1)

#             final_samples_flat = F.fold(
#                 fold_input, output_size=output_size, kernel_size=kernel_size, stride=stride_1d
#             )
#             weight_sum_flat = F.fold(
#                 fold_weights, output_size=output_size, kernel_size=kernel_size, stride=stride_1d
#             )
            
#             # 5e) fold 출력 텐서 복원
#             # fold 출력: [N, C, H_out, W_out] = [S*B, C, L, 1]
#             # [S*B, C, L, 1] -> [S, B, C, L] -> [S, B, L, C] (최종 shape)
#             final_samples = final_samples_flat.squeeze(-1).view(S, B, C, L).permute(0, 1, 3, 2)
#             weight_sum = weight_sum_flat.squeeze(-1).view(S, B, C, L).permute(0, 1, 3, 2)

#             # 6) 최종 정규화
#             final_samples = final_samples / weight_sum.clamp_min(1e-6)
            
#             return final_samples
        