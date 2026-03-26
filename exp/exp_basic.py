import os
import torch

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.rank = 0
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.pred_len = args.pred_len
    
    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            gpu_index = int(getattr(self.args, 'gpu', 0))
            device = torch.device('cuda:{}'.format(gpu_index))
            print('Use GPU: cuda:{}'.format(gpu_index))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
