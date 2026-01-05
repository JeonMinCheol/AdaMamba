import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import csv, os
import seaborn as sns
import torch.distributed as dist

import io
from PIL import Image

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd # pandas가 없다면 추가

plt.switch_backend('agg')

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        # Root Mean Square 계산
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        return x / (rms + self.eps) * self.scale

def dict_to_string(dict: dict):
    ret = ""
    for k, v in dict.items():
        ret += k + ": " + str(v.item()) +", "
    return ret

def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        if printout:
            print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.delta = delta

    def __call__(self, val_loss, test_loss, model, path):
        if not dist.is_initialized():
            # Single GPU fallback
            return self._update(val_loss, model, path)

        # 1) 모든 프로세스의 val_loss 모으기
        device = next(model.parameters()).device
        tensor_loss = torch.tensor([val_loss], dtype=torch.float32, device=device)
        all_losses = [torch.zeros_like(tensor_loss) for _ in range(dist.get_world_size())]
        dist.all_gather(all_losses, tensor_loss)
        all_losses = [t.item() for t in all_losses]

        # 2) 현재 스텝에서 가장 낮은 val_loss 가진 rank 찾기
        best_rank = min(range(len(all_losses)), key=lambda r: all_losses[r])
        best_val_loss = all_losses[best_rank]

        # 3) 기존 best_loss보다 더 좋아야만 저장
        if best_val_loss + self.delta < self.val_loss_min:
            if dist.get_rank() == best_rank:
                self.save_checkpoint(best_val_loss, test_loss, model, path, best_rank)
            self.val_loss_min = best_val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        # 4) early_stop 상태 동기화
        flag = torch.tensor(1 if self.early_stop else 0, device=device)
        dist.broadcast(flag, src=best_rank)
        self.early_stop = flag.item() == 1

    def save_checkpoint(self, val_loss, test_loss, model, path, rank):
        if self.verbose:
            print(f"[Rank {rank}] Validation loss decreased "
                  f"({self.val_loss_min:.6f} → {val_loss:.6f}). Saving model ...\nTest loss: {test_loss:6f}")
        os.makedirs(path, exist_ok=True)
        state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save(state_dict, os.path.join(path, f"checkpoint.pth"))

class SimpleLogger:
    def __init__(self, log_dir="logs", filename="train_log.csv"):
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, filename)
        # 헤더 작성
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["stage","epoch","step",
                                 "det_mse","res_mse","nll",
                                 "grad_norm","lr","time"])

    def log(self, **kwargs):
        # 콘솔 출력
        msg = " | ".join([f"{k}:{v:.5f}" if isinstance(v,(float,int)) else f"{k}:{v}"
                          for k,v in kwargs.items()])
        print(msg)

        # CSV 기록
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([kwargs.get("stage"),
                             kwargs.get("epoch"),
                             kwargs.get("step"),
                             kwargs.get("det_mse", ""),
                             kwargs.get("res_mse", ""),
                             kwargs.get("nll", ""),
                             kwargs.get("grad_norm", ""),
                             kwargs.get("lr", ""),
                             kwargs.get("time","")])

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def visual(true, preds=None, name='./pic/test.pdf',
           title="Forecast vs Ground Truth",
           xlabel="Time", ylabel="Value"):
    
    plt.figure(figsize=(10, 4))                       
    plt.plot(true, label='Ground Truth',
             color="#1e7cc0", linewidth=1.4, alpha=0.9)
    if preds is not None:
        plt.plot(preds, label='Prediction',
                 color="#ff0000", linewidth=1.0, alpha=0.7)

    # plt.title(title, fontsize=14, weight='bold', pad=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    # 눈금/격자 가독성
    plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.legend(fontsize=11, loc='best', frameon=True)
    plt.tight_layout()
    plt.savefig(name, bbox_inches='tight', dpi=300)
    plt.close()

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    
    # from ptflops import get_model_complexity_info    
    # with torch.cuda.device(0):
    #     macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
    #     print('Flops:' + flops)
    #     print('Params:' + params)
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # get_model_complexity_info:  네트워크의 연산량과 패러미터 수를 출력
        # model: nn.Modules 클래스로 만들어진 객체. 연산량과 패러미터 수를 측정할 네트워크입니다.
        # input_res: 입력되는 텐서의 모양을 나타내는 tuple. 이 때, batch size에 해당하는 차원은 제외합니다.
        # print_per_layer_stat: True일 시, Layer 단위로 연산량과 패러미터 수를 출력합니다.
        # as_strings: True일 시, 연산량 및 패러미터 수를 string으로 변환하여 출력합니다.
        # verbose: True일 시, zero-op에 대한 warning을 출력합니다.

def plot_attention_heatmap(attention_weights, layer_num, head_num, title, sample_num=0):
    """
    주어진 어텐션 가중치로 히트맵을 그리는 함수.

    Args:
        attention_weights (torch.Tensor): 어텐션 가중치 텐서. Shape: [B, n_heads, L, L]
        layer_num (int): 시각화할 레이어 번호.
        head_num (int): 시각화할 헤드 번호.
        sample_num (int): 시각화할 배치의 샘플 번호.
    """
    # 시각화할 특정 샘플, 특정 헤드의 어텐션 맵 선택
    # CPU로 데이터 이동 및 numpy 배열로 변환
    attn_map = attention_weights[sample_num, head_num].detach().cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_map, cmap='viridis') # 'viridis', 'plasma' 등 다양한 색상 맵 사용 가능
    
    plt.title(f'Attention Heatmap (Layer {layer_num+1}, Head {head_num+1})')
    plt.xlabel('Key (Attended Patches)')
    plt.ylabel('Query (Current Patches)')
    plt.savefig(title, bbox_inches='tight', dpi=300)

def get_heatmap_image_tensor(attention_weights, sample_num=0, head_num=0):
    """
    어텐션 가중치로 히트맵을 그려 PyTorch 이미지 텐서로 반환하는 함수.
    
    Args:
        attention_weights (torch.Tensor): 어텐션 가중치. Shape: [B, n_heads, L, L]
        sample_num (int): 시각화할 배치의 샘플 번호.
        head_num (int): 시각화할 헤드 번호.

    Returns:
        torch.Tensor: 이미지 텐서. Shape: [C, H, W]
    """
    # 특정 어텐션 맵 선택
    attn_map = attention_weights[sample_num, head_num].detach().cpu().numpy()

    # Matplotlib Figure 생성
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(attn_map, cmap='viridis', ax=ax, cbar=False) # cbar는 생략 가능
    ax.set_title(f'Attention Heatmap (Head {head_num+1})')
    ax.set_xlabel('Key Patches')
    ax.set_ylabel('Query Patches')
    fig.tight_layout()

    # 💡 Figure를 메모리 버퍼에 PNG 이미지로 저장
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)

    # 💡 버퍼의 이미지를 PIL로 열고 Numpy 배열로 변환
    img = Image.open(buf)
    img_array = np.array(img.convert('RGB'))

    # 💡 Numpy 배열을 PyTorch 텐서로 변환하고, 채널 순서 변경 [H, W, C] -> [C, H, W]
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)

    # 메모리 누수 방지를 위해 figure와 buffer를 닫음
    plt.close(fig)
    buf.close()

    return img_tensor

def log_layer_stats(writer, model, step, optimizer, prefix="stats"):
    for name, p in model.named_parameters():
        if not p.requires_grad: 
            continue
        w = p.data
        g = p.grad
        writer.add_scalar(f"weight_{prefix}/{name}_w_mean", w.mean().item(), step)
        writer.add_scalar(f"weight_{prefix}/{name}_w_std", w.std().item(), step)
        if g is None:
            writer.add_scalar(f"grad_{prefix}/{name}_norm", 0.0, step)
        else:
            writer.add_scalar(f"grad_{prefix}/{name}_norm", g.norm().item(), step)

        if p.grad is not None:
            update = -optimizer.param_groups[0]['lr'] * p.grad
            rel_update = (update.abs().mean() / (p.data.abs().mean() + 1e-8)).item()
            writer.add_scalar(f"update_ratio/{name}", rel_update, step)

def visual_trend(true_trend, pred_trend, raw_data=None, name='./trend_test.pdf'):
    """
    true_trend: (L, ) or (L, 1) - 실제 트렌드 (Moving Average 등)
    pred_trend: (L, ) or (L, 1) - 모델이 추출한 트렌드
    raw_data: (L, ) or (L, 1) - 원본 시계열 데이터 (선택 사항)
    """
    plt.figure(figsize=(10, 4))
    if raw_data is not None:
        plt.plot(raw_data, label='Raw Data', color='gray', linewidth=0.8, alpha=0.3)
    
    plt.plot(true_trend, label='Actual Trend (MA)', color='blue', linewidth=1.5, alpha=0.8)
    plt.plot(pred_trend, label='Model Extracted Trend', color='red', linewidth=1.5, linestyle='--', alpha=0.8)
    
    plt.title("Trend Comparison", fontsize=12)
    plt.xlabel("Time", fontsize=10)
    plt.ylabel("Value", fontsize=10)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(name, bbox_inches='tight', dpi=300)
    plt.close()

# [Added] 트렌드 유사도 계산 함수
def calc_trend_similarity(true_trend, pred_trend):
    """
    MSE와 Cosine Similarity를 계산하여 반환
    """
    # MSE
    mse = np.mean((true_trend - pred_trend)**2)
    
    # Cosine Similarity
    t = true_trend.flatten()
    p = pred_trend.flatten()
    
    norm_t = np.linalg.norm(t)
    norm_p = np.linalg.norm(p)
    
    if norm_t == 0 or norm_p == 0:
        cos_sim = 0.0
    else:
        cos_sim = np.dot(t, p) / (norm_t * norm_p)
        
    return mse, cos_sim

# [Added] Moving Average 계산 함수 (Ground Truth Trend 생성용)
def moving_average(data, kernel_size=25):
    """
    data: (B, L, D) Tensor
    kernel_size: int
    return: (B, L, D) Tensor (Same length with padding)
    """
    # (B, D, L) 형태로 변경하여 AvgPool1d 적용
    x = data.permute(0, 2, 1)
    
    # Same padding을 위해 앞뒤로 패딩
    pad_l = kernel_size // 2
    pad_r = kernel_size - 1 - pad_l
    
    # ReplicationPad1d 등을 사용할 수도 있지만, 여기선 간단히 zero padding 혹은 replication 사용
    # 트렌드이므로 경계 부분 처리가 중요하지만 간단한 비교를 위해 ReplicationPad 사용
    x_padded = torch.nn.functional.pad(x, (pad_l, pad_r), mode='replicate')
    
    avg = torch.nn.functional.avg_pool1d(x_padded, kernel_size=kernel_size, stride=1)
    
    # 원래 shape (B, L, D)로 복구
    return avg.permute(0, 2, 1)