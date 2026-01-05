from ast import mod
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import AdaMamba, Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, iTransformer, ModernTCN, FEDformer, TimeMixer, MambaTS
from utils.tools import *
from utils.metrics import metric
from tqdm import tqdm
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 
import torch.distributed as dist

import os
import time

import matplotlib.pyplot as plt
from scipy.fftpack import fft

import warnings

torch.backends.cudnn.deterministic=True
warnings.filterwarnings('ignore')

import tempfile
tempfile.tempdir = "/dev/shm"

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        # --- DDP 초기화 ---
        self.kernels = [int(x) for x in self.args.kernels.split(',') if x.strip() != ""]

        print(f"Using kernel sizes: {self.kernels}")

        if self.args.use_multi_gpu and self.args.use_gpu:
            dist.init_process_group(
                backend="nccl",      # GPU라면 NCCL이 가장 빠름
                init_method="env://",
                timeout=timedelta(seconds=300)
            )
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            self.device = torch.device("cuda", local_rank)
            self.rank = dist.get_rank()
        else:
            # 단일 GPU/CPU
            self.device = torch.device("cuda" if self.args.use_gpu else "cpu")
            self.rank = 0

        model_dict = {
            'FEDformer': FEDformer,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'iTransformer': iTransformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'ModernTCN': ModernTCN,
            'TimeMixer': TimeMixer,
            'MambaTS': MambaTS,
            'AdaMamba': AdaMamba
        }

        if self.args.model == 'AdaMamba':
            model = model_dict[self.args.model].Model(self.args, self.kernels).float()
        else:
            model = model_dict[self.args.model].Model(self.args).float()
            
        # --- DDP 래핑 ---
        if self.args.use_multi_gpu and self.args.use_gpu:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = model.to(self.device)
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.device.index], 
                output_device=self.device.index,
                find_unused_parameters=False
            )
            if self.rank == 0:
                print(f"[DDP] rank {self.rank}, local_rank {local_rank} -> device {self.device}")
            
            model = model.module
        else:
            model = model.to(self.device)

        return model

    def _get_data(self, flag):
        data_set, data_loader, sampler = data_provider(self.args, flag)
        return data_set, data_loader, sampler

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_loader, criterion, epoch, data = "val"):
        total_loss = []
        preds, trues = [], []
        predictions_on_cpu = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(vali_loader, desc="Validation")):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.model == "AdaMamba":
                    if self.args.use_amp:
                        with torch.amp.autocast():
                            outputs, batch_trend = self.model.sample(batch_x)
                            predictions_on_cpu.append(outputs.detach().cpu())
                            stacked_predictions = torch.stack(predictions_on_cpu, dim=0)
                            outputs = torch.mean(stacked_predictions, dim=0)
                    else:
                        outputs, batch_trend = self.model.sample(batch_x)
                        predictions_on_cpu.append(outputs.detach().cpu())
                        stacked_predictions = torch.stack(predictions_on_cpu, dim=0)
                        outputs = torch.mean(stacked_predictions, dim=0)

                else:
                    if self.args.use_amp:
                        with torch.amp.autocast():
                            if 'Linear' in self.args.model or 'TST' in self.args.model or 'MambaTS' in self.args.model:
                                outputs = self.model(batch_x)
                            elif 'TCN' in self.args.model:
                                outputs = self.model(batch_x, batch_x_mark)
                            else:
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if 'Linear' in self.args.model or 'TST' in self.args.model or 'MambaTS' in self.args.model:
                            outputs = self.model(batch_x)
                        elif 'TCN' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                preds.append(pred); trues.append(true)

                loss = criterion(pred, true)
                total_loss.append(loss)
                
        preds = np.array(preds)
        trues = np.array(trues)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        total_loss = np.average(total_loss)
        
        self.model.train()
        return total_loss, mae, mse, rmse, mape, mspe, rse, corr

    def train(self, setting):
        self.model.train()
        train_data, train_loader, sampler = self._get_data(flag='train')
        vali_data, vali_loader, _ = self._get_data(flag='val')
        test_data, test_loader, _ = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        
        os.makedirs(path, exist_ok=True)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.amp.GradScaler()
        
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                        steps_per_epoch = train_steps,
                                        pct_start = self.args.pct_start,
                                        epochs = self.args.train_epochs,
                                        max_lr = self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            train_loss = []
            iter_count = 0

            sampler.set_epoch(epoch) if sampler is not None else None
            epoch_start_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch}")):
                iter_count += 1 
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.model != "AdaMamba":
                    if self.args.use_amp:
                        with torch.amp.autocast():
                            if 'Linear' in self.args.model or 'TST' in self.args.model or 'MambaTS' in self.args.model:
                                outputs = self.model(batch_x, batch_y)
                            elif 'TCN' in self.args.model:
                                outputs = self.model(batch_x, batch_x_mark)
                            else:
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device)
                            loss = criterion(outputs, batch_y)
                            train_loss.append(loss.item())
                            
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                            
                    else:
                        if 'Linear' in self.args.model or 'TST' in self.args.model or 'MambaTS' in self.args.model:
                            outputs = self.model(batch_x)
                        elif 'TCN' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        model_optim.step()

                        train_loss.append(loss.item())

                else:
                    if self.args.use_amp:
                        with torch.amp.autocast():
                            # 모델 forward 한 번 호출
                            total_loss = self.model(batch_x, batch_y)
                            scaler.scale(total_loss).backward()
                            scaler.step(model_optim)
                            scaler.update()

                            train_loss.append(total_loss.item())
                    else:
                        total_loss = self.model(batch_x, batch_y)
                        total_loss.backward()
                        model_optim.step()
                        train_loss.append(total_loss.item())

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            train_loss = np.average(train_loss)
            epoch_end_time = time.time()
            
            vali_loss, vali_mae, vali_mse, vali_rmse, vali_mape, vali_mspe, vali_rse, _ = self.vali(vali_loader, criterion, epoch, "val")
            test_loss, test_mae, test_mse, test_rmse, test_mape, test_mspe, test_rse, _ = self.vali(test_loader, criterion, epoch, "test")

            if (self.args.use_multi_gpu and self.rank == 0) or not self.args.use_multi_gpu:
                print("[RANK {}] Epoch: {} cost time: {:.2f} | Train: {:.6f}, Val: {:.6f}, Test: {:.6f}".format(self.rank, epoch + 1, epoch_end_time - epoch_start_time, train_loss, vali_loss, test_loss))

            early_stopping(vali_loss, test_loss, self.model, path)
            if early_stopping.early_stop: 
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, False)
            elif (self.args.use_multi_gpu and self.rank == 0) or not self.args.use_multi_gpu:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
                
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        self.model.eval()
        test_data, test_loader, _ = self._get_data(flag='test')
        
        if test:
            checkpoint = torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace("module.", "")  # 제거
                new_state_dict[new_k] = v

            self.model.load_state_dict(new_state_dict)
            print('loading model')

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader, desc="Test")):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.model == "AdaMamba":
                    if self.args.use_amp:
                        with torch.amp.autocast():
                            outputs, batch_trend = self.model.sample(batch_x)
                    else:
                        outputs, batch_trend = self.model.sample(batch_x)

                else:
                    if self.args.use_amp:
                        with torch.amp.autocast():
                            if 'Linear' in self.args.model or 'TST' in self.args.model or 'MambaTS' in self.args.model:
                                outputs = self.model(batch_x)
                            elif 'TCN' in self.args.model:
                                outputs = self.model(batch_x, batch_x_mark)
                            else:
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if 'Linear' in self.args.model or 'TST' in self.args.model or 'MambaTS' in self.args.model:
                                outputs = self.model(batch_x)
                        elif 'TCN' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())

                if i % 20 == 0: # 숫자 변경
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

                    if self.args.model == "AdaMamba":
                        # Prepare Ground Truth Data (Input + Target)
                        # batch_x: (B, L, D), batch_y is already truncated to pred_len in loop but original is available?
                        # No, batch_y here is already sliced. We need the full y or construct it.
                        # Actually batch_y from loader usually has label_len + pred_len.
                        # But in the loop above: batch_y = batch_y[:, -self.pred_len:, f_dim:]
                        # So we can concat batch_x and batch_y (prediction part).
                        # batch_trend from model likely covers L+pred_len.
                        
                        # Reconstruct full GT sequence for the first sample in batch
                        input_tensor = batch_x.detach() # (B, L, D)
                        # Note: true variable is numpy array of prediction part only
                        true_tensor = torch.from_numpy(true).to(self.device) # (B, P, D)
                        
                        gt_seq = torch.cat([input_tensor, true_tensor], dim=1) # (B, L+P, D)
                        
                        # Calculate Actual Trend using Moving Average
                        # Use kernel size from args if available, else default 25
                        k_size = getattr(self.args, 'moving_avg', 25)
                        if isinstance(k_size, list): k_size = k_size[0] # handle list case
                        
                        gt_trend = moving_average(gt_seq, kernel_size=k_size)
                        
                        # Extract Data for Visualization (1st sample, last channel)
                        # batch_trend is (B, TrendLen, D)
                        
                        # Length check to avoid shape mismatch
                        t_len = min(gt_trend.shape[1], batch_trend.shape[1])
                        
                        gt_trend_np = gt_trend[0, :t_len, -1].detach().cpu().numpy()
                        pred_trend_np = batch_trend[0, :t_len, -1].detach().cpu().numpy()
                        raw_data_np = gt_seq[0, :t_len, -1].detach().cpu().numpy()
                        
                        # Visualize
                        visual_trend(gt_trend_np, pred_trend_np, raw_data_np, 
                                     name=os.path.join(folder_path, 'trend_comparison_' + str(i) + '.pdf'))
                        self.visualize_kernel_weights(self.model, test_loader, self.device, kernel_sizes=self.kernels,
                                                 save_path=os.path.join(folder_path, 'kernel_weights_' + str(i) + '.pdf'))
                        
        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1],batch_x.shape[2]))
            exit()

        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])


        # result save
        if (self.args.use_multi_gpu and self.rank == 0) or not self.args.use_multi_gpu:
            mae, mse, rmse, mape, mspe, rse, _ = metric(preds, trues)
            print('mae:{}, mse:{}, rmse:{}, mape:{}, mspe:{}, rse:{}'.format(mae, mse, rmse, mape, mspe, rse))
            f = open("result.txt", 'a')
            f.write(setting + "\n")
            f.write('mae:{}, mse:{}, rmse:{}, mape:{}, mspe:{}, rse:{}\n\n'.format(mae, mse, rmse, mape, mspe, rse))
            f.close()

        if (self.args.use_multi_gpu): 
            dist.destroy_process_group()

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(pred_loader, desc="Predict")):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.model == "AdaMamba":
                    if self.args.use_amp:
                        with torch.amp.autocast():
                            outputs, batch_trend = self.model.sample(batch_x)
                    else:
                        outputs, batch_trend = self.model.sample(batch_x)
                        
                else:
                    if self.args.use_amp:
                        with torch.amp.autocast():
                            if 'Linear' in self.args.model or 'TST' in self.args.model or 'MambaTS' in self.args.model:
                                outputs = self.model(batch_x)
                            elif 'TCN' in self.args.model:
                                outputs = self.model(batch_x, batch_x_mark)
                            else:
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if 'Linear' in self.args.model or 'TST' in self.args.model or 'MambaTS' in self.args.model:
                            outputs = self.model(batch_x)
                        elif 'TCN' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        if (self.args.use_multi_gpu and self.rank == 0) or not self.args.use_multi_gpu:
            np.save(folder_path + 'real_prediction.npy', preds)

        dist.destroy_process_group()
        return

    def analysis_batch(self, batch_x, setting):
        # DDP면 rank0만 저장 (중복 저장 방지)
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() != 0:
                return

        # 1) 내부 모델 언랩
        model_engine = self.model
        while hasattr(model_engine, 'module'):
            model_engine = model_engine.module

        print(f">>> [Analysis Debug] Model Type: {type(model_engine)}")

        if self.args.model != 'AdaMamba':
            print(">>> [Analysis] Skipping: Model is not AdaMamba")
            return

        if not hasattr(model_engine, 'adaptive_norm_block'):
            print(">>> [Analysis Error] Model missing 'adaptive_norm_block'")
            print(f"    Available attributes: {list(model_engine.__dict__.keys())}")
            return

        # 2) Trend 추출 (✅ [B,L,M] 그대로 normalize)
        try:
            with torch.no_grad():
                batch_x = batch_x.float().to(self.device)  # [B,L,M]
                if hasattr(model_engine.adaptive_norm_block, 'normalize'):
                    # new signature: x_norm, means, stdev, trend, alpha_gate
                    out = model_engine.adaptive_norm_block.normalize(batch_x)
                    if len(out) == 5:
                        _, _, _, trend, alpha_gate = out
                    else:
                        # 혹시 옛날 4개면 fallback
                        _, _, _, trend = out
                        alpha_gate = None
                else:
                    print(f">>> [Analysis Error] 'adaptive_norm_block' exists but has no 'normalize' method.")
                    print(f"    Current Norm Type: {getattr(model_engine, 'norm_type', 'Unknown')}")
                    return
        except Exception as e:
            print(f">>> [Analysis Error] Trend extraction failed: {e}")
            return

        # 3) numpy로 변환
        input_data = batch_x.detach().cpu().numpy()      # [B,L,M]
        trend_data = trend.detach().cpu().numpy()        # [B,L,M]

        sample_idx = 0
        channel_idx = -1

        series = input_data[sample_idx, :, channel_idx]
        extracted_trend = trend_data[sample_idx, :, channel_idx]
        residual = series - extracted_trend

        # 4) FFT
        def get_fft_values(y, T, N):
            f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
            fft_values_ = fft(y)
            fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
            return f_values, fft_values

        N = len(series)
        T = 1.0
        f_val, fft_raw = get_fft_values(series, T, N)
        _, fft_trend = get_fft_values(extracted_trend, T, N)
        _, fft_res = get_fft_values(residual, T, N)

        # 5) 저장/시각화
        save_path = f'./test_results/{setting}/spectral_analysis_fast.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.figure(figsize=(12, 12))

        plt.subplot(3, 1, 1)
        plt.plot(series, label='Original', color='gray', alpha=0.5)
        plt.plot(extracted_trend, label='Trend (AdaNorm)', color='blue', linewidth=2)
        plt.title(f'Time Domain (Sample 0, Channel {channel_idx})')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(f_val, fft_raw, label='Raw Data', color='gray')
        plt.plot(f_val, fft_trend, label='Extracted Trend', color='green', alpha=0.7)
        plt.title('Frequency Domain: Raw vs Trend')
        plt.yscale('log')
        plt.xlim(0, 0.1)
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(f_val, fft_res, label='Residual Spectrum', color='red')
        plt.title('Frequency Domain: Residual')
        plt.yscale('log')
        plt.xlim(0, 0.1)
        plt.legend()

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f">>> Fast Spectral Analysis Saved to: {save_path}")

        # (옵션) alpha_gate 로그
        if alpha_gate is not None:
            a = alpha_gate.detach().cpu().numpy().squeeze()  # [M] 또는 scalar
            print(f">>> [Analysis] alpha_gate (mean/min/max): {a.mean():.4f}/{a.min():.4f}/{a.max():.4f}")


    def visualize_kernel_weights(
        self,
        model,
        data_loader,
        device,
        kernel_sizes,
        save_path='./pic/kernel_weights.pdf',
        max_batches=20
    ):
        """
        AdaMamba의 KernelSE(커널 중요도 게이트 g)를 시각화합니다.
        - KernelSE는 [B, C*K, L] 입력을 받아 [B, C, K] 게이트를 만들므로,
        훅에서 동일 계산을 재현해 g를 캡처합니다.
        """

        # -----------------------------
        # 0) DDP면 rank0만 저장/시각화
        # -----------------------------
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() != 0:
                return None

        # -----------------------------
        # 1) 모델 언랩 (DDP/DataParallel 대응)
        # -----------------------------
        model_engine = model
        while hasattr(model_engine, 'module'):
            model_engine = model_engine.module

        model_engine.eval()

        # -----------------------------
        # 2) KernelSE 모듈 찾기
        # -----------------------------
        target_module = None
        if hasattr(model_engine, 'adaptive_norm_block'):
            norm_block = model_engine.adaptive_norm_block
            if hasattr(norm_block, 'detrender') and hasattr(norm_block.detrender, 'kernel_se'):
                target_module = norm_block.detrender.kernel_se

        if target_module is None:
            print("Error: KernelSE 모듈을 찾을 수 없습니다. (adaptive_norm_block.detrender.kernel_se)")
            return None

        captured_weights = []

        # -----------------------------
        # 3) hook: KernelSE 게이트 g를 재현해서 캡처
        # -----------------------------
        def hook_fn(module, inputs, output):
            # module: KernelSE
            # inputs[0]: cat_trends [B, C*K, L]
            with torch.no_grad():
                cat_trends = inputs[0]
                if cat_trends.dim() != 3:
                    return

                B, CK, L = cat_trends.shape
                K = len(kernel_sizes)
                if CK % K != 0:
                    # 예상치 못한 shape면 스킵
                    return
                C = CK // K

                # KernelSE 내부 로직 재현
                s = cat_trends.mean(dim=2).view(B, C, K)        # [B,C,K]
                g = module.mlp(s.view(B * C, K)).view(B, C, K)  # [B,C,K] (0,1)

                # 배치/채널 평균해서 [K]로 축약해 저장
                g_mean = g.mean(dim=(0, 1))  # [K]
                captured_weights.append(g_mean.cpu())

        hook = target_module.register_forward_hook(hook_fn)

        # -----------------------------
        # 4) 데이터 순회하면서 sample 호출로 hook 트리거
        # -----------------------------
        print("Collecting kernel weights (KernelSE gates)...")
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                # loader가 (batch_x, _, _, _) 형태라고 가정
                batch_x = batch[0].float().to(device)  # [B,L,M] 기대
                model_engine.sample(batch_x)           # forward 내부에서 kernel_se 호출되며 hook 트리거

                if i >= max_batches:
                    break

        hook.remove()

        if len(captured_weights) == 0:
            print("Warning: No kernel weights were captured. (KernelSE가 호출되지 않았거나 shape mismatch)")
            return None

        # -----------------------------
        # 5) 평균 게이트 계산 + 시각화
        # -----------------------------
        avg_weights = torch.stack(captured_weights, dim=0).mean(dim=0).numpy()  # [K]

        plt.figure(figsize=(10, 5))
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(kernel_sizes)))
        bars = plt.bar([str(k) for k in kernel_sizes], avg_weights,
                    color=colors, edgecolor='black', alpha=0.8)

        plt.title('Learned Kernel Gates (KernelSE Importance per Scale)', fontsize=14, fontweight='bold')
        plt.xlabel('Kernel Size (Fibonacci)', fontsize=12)
        plt.ylabel('Average Gate (0~1)', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01,
                    round(float(yval), 3), ha='center', va='bottom')

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Visualization saved to {save_path}")

        return avg_weights
