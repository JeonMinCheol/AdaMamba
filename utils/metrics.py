import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def CRPS(samples, y_true):
    """
    샘플 기반 CRPS Loss 계산
    Args:
        samples (Tensor): 모델 예측 샘플들 (num_samples, B, pred_len)
        y_true (Tensor): 실제 값 (B, pred_len)
    """
    # 예측 샘플들과 실제 값 사이의 평균 절대 오차
    term1 = torch.abs(samples - y_true.unsqueeze(0)).mean(dim=0)
    # 예측 샘플들끼리의 평균 절대 오차
    term2 = torch.abs(samples.unsqueeze(1) - samples.unsqueeze(0)).mean(dim=[0, 1])
    
    loss = (term1 - 0.5 * term2).mean()
    return loss


def quantile_loss(y_pred, y_true, q):
    e = y_true - y_pred
    return torch.max((q-1)*e, q*e).mean()


def energy_score(y_true: torch.Tensor, y_samples: torch.Tensor) -> torch.Tensor:
    """
    Differentiable Energy Score (for training with NF samples).

    Args:
        y_true: (B, L, C) ground truth
        y_samples: (S, B, L, C) samples from NF (differentiable via reparam)

    Returns:
        scalar loss
    """
    S, B, L, C = y_samples.shape
    y_true = y_true.unsqueeze(0).expand(S, -1, -1, -1)  # (S, B, L, C)

    # Term1: E ||X - y||
    term1 = torch.linalg.norm(y_samples - y_true, dim=-1).mean()

    # Term2: 0.5 * E ||X - X'||
    # reshape to (S*B, L*C) for efficiency
    flat_samples = y_samples.reshape(S * B, -1)
    pw = torch.cdist(flat_samples, flat_samples, p=2)  # (S*B, S*B)
    term2 = 0.5 * pw.mean()

    return term1 - term2


def directional_loss(y_pred, y_true):
    # y_pred, y_true shape: [B, L, C]

    # 연속된 시점 간의 차이를 계산 (부호가 방향을 의미)
    pred_diff = y_pred[:, 1:, :] - y_pred[:, :-1, :]
    true_diff = y_true[:, 1:, :] - y_true[:, :-1, :]

    # 두 방향 벡터의 코사인 유사도를 손실로 사용 (유사하면 1, 반대면 -1)
    # 1에서 빼서 손실로 만듦 (반대 방향일수록 손실이 커짐)
    loss = 1 - F.cosine_similarity(pred_diff, true_diff, dim=1)

    return loss.mean()


def spectral_loss(y_pred, y_true):
    # y_pred, y_true shape: [B, L, C]

    # FFT를 위해 채널 차원을 뒤로 보냄
    pred_transposed = y_pred.permute(0, 2, 1)
    true_transposed = y_true.permute(0, 2, 1)

    # FFT 적용
    pred_fft = torch.fft.rfft(pred_transposed, dim=-1)
    true_fft = torch.fft.rfft(true_transposed, dim=-1)

    # 스펙트럼의 크기(magnitude)에 대한 L1 손실 계산
    magnitude_loss = F.l1_loss(torch.abs(pred_fft), torch.abs(true_fft))

    return magnitude_loss


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr
