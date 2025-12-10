#!/usr/bin/bash
#SBATCH -J env_model_train
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_grad
#SBATCH -w ariel-v7
#SBATCH -t 4-0
#SBATCH -o logs/slurm-%A.out

# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=DLinear

torchrun --nproc_per_node=4 /data/a2019102224/PatchTST_supervised/run_longExp.py \
  --is_training 1 \
  --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.001 >logs/LongForecasting/$model_name'_'ETTm2_$seq_len'_'96.log

torchrun --nproc_per_node=4 /data/a2019102224/PatchTST_supervised/run_longExp.py \
  --is_training 1 \
  --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'192 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.001 >logs/LongForecasting/$model_name'_'ETTm2_$seq_len'_'192.log

torchrun --nproc_per_node=4 /data/a2019102224/PatchTST_supervised/run_longExp.py \
  --is_training 1 \
  --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'336 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.01 >logs/LongForecasting/$model_name'_'ETTm2_$seq_len'_'336.log

torchrun --nproc_per_node=4 /data/a2019102224/PatchTST_supervised/run_longExp.py \
  --is_training 1 \
  --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'720 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.1 >logs/LongForecasting/$model_name'_'ETTm2_$seq_len'_'720.log
