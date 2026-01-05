import torch
import numpy as np
import math

# 상삼각행렬 마스킹
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device) 

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask

def random_segment_mask(x, mask_ratio=0.1):
    """
    Random Segment Masking
    x: [B, L, C]
    mask_ratio: 전체 길이 L 중 몇 %를 랜덤하게 가릴지
    """
    B, L, C = x.shape
    mask_len = max(1, int(L * mask_ratio))
    start = torch.randint(0, L - mask_len + 1, (1,)).item()
    
    x_masked = x.clone()
    x_masked[:, start:start+mask_len, :] = 0.0
    return x_masked

class LocalMask():
    def __init__(self, B, L,S,device="cpu"):
        mask_shape = [B, 1, L, S]
        with torch.no_grad():
            self.len = math.ceil(np.log2(L))
            self._mask1 = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
            self._mask2 = ~torch.triu(torch.ones(mask_shape,dtype=torch.bool),diagonal=-self.len).to(device)
            self._mask = self._mask1+self._mask2
    @property
    def mask(self):
        return self._mask
    
def random_shuffle(x, ids_shuffle=None, return_ids_shuffle=False, mask_ratio=0):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    if ids_shuffle is None:
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove

    # keep the first subset
    ids_shuffle_ = ids_shuffle[:, :len_keep]

    if ids_shuffle_.shape[0] != x.shape[0]:
        ids_shuffle_ = ids_shuffle_[[0]].repeat(x.shape[0], 1)

    ids_restore = torch.argsort(ids_shuffle_, dim=1)

    x_shuffled = torch.gather(x, dim=1, index=ids_shuffle_.unsqueeze(-1).repeat(1, 1, D))

    if return_ids_shuffle:
        return x_shuffled, ids_shuffle, ids_restore
    else:
        return x_shuffled, ids_restore


def unshuffle(x, ids_restore):
    x_restore = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
    return x_restore