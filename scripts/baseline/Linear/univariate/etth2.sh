#!/usr/bin/bash
#SBATCH -J env_model_train
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_grad
#SBATCH -w ariel-v7
#SBATCH -t 4-0
#SBATCH -o logs/slurm-%A.out

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/univariate" ]; then
    mkdir ./logs/LongForecasting/univariate
fi
model_name=DLinear

# ETTh2, univariate results, pred_len= 24 48 96 192 336 720
torchrun --nproc_per_node=4 /data/a2019102224/PatchTST_supervised/run_longExp.py \
  --is_training 1 \
  --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_336_24 \
  --model $model_name \
  --data ETTh2 \
  --seq_len 336 \
  --pred_len 24 \
  --enc_in 1 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.005 --feature S >logs/LongForecasting/$model_name'_'fS_ETTh2_336_24.log

torchrun --nproc_per_node=4 /data/a2019102224/PatchTST_supervised/run_longExp.py \
  --is_training 1 \
  --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_336_48 \
  --model $model_name \
  --data ETTh2 \
  --seq_len 336 \
  --pred_len 48 \
  --enc_in 1 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.005 --feature S >logs/LongForecasting/$model_name'_'fS_ETTh2_336_48.log

torchrun --nproc_per_node=4 /data/a2019102224/PatchTST_supervised/run_longExp.py \
  --is_training 1 \
  --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_336_96 \
  --model $model_name \
  --data ETTh2 \
  --seq_len 336 \
  --pred_len 96 \
  --enc_in 1 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.005 --feature S >logs/LongForecasting/$model_name'_'fS_ETTh2_336_96.log


torchrun --nproc_per_node=4 /data/a2019102224/PatchTST_supervised/run_longExp.py \
  --is_training 1 \
  --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_336_192 \
  --model $model_name \
  --data ETTh2 \
  --seq_len 336 \
  --pred_len 192 \
  --enc_in 1 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.005 --feature S >logs/LongForecasting/$model_name'_'fS_ETTh2_336_192.log


torchrun --nproc_per_node=4 /data/a2019102224/PatchTST_supervised/run_longExp.py \
  --is_training 1 \
  --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_336_336 \
  --model $model_name \
  --data ETTh2 \
  --seq_len 336 \
  --pred_len 336 \
  --enc_in 1 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.005 --feature S >logs/LongForecasting/$model_name'_'fS_ETTh2_336_336.log
 

torchrun --nproc_per_node=4 /data/a2019102224/PatchTST_supervised/run_longExp.py \
  --is_training 1 \
  --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_336_720 \
  --model $model_name \
  --data ETTh2 \
  --seq_len 336 \
  --pred_len 720 \
  --enc_in 1 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.005 --feature S >logs/LongForecasting/$model_name'_'fS_ETTh2_336_720.log