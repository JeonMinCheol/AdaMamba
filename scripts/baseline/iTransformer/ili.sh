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
seq_len=104
model_name=DLinear

torchrun --nproc_per_node=4 /data/a2019102224/PatchTST_supervised/run_longExp.py \
  --is_training 1 \
  --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
  --data_path national_illness.csv \
  --model_id national_illness_$seq_len'_'24 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 18 \
  --pred_len 24 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.01 >logs/LongForecasting/$model_name'_'ili_$seq_len'_'24.log

torchrun --nproc_per_node=4 /data/a2019102224/PatchTST_supervised/run_longExp.py \
  --is_training 1 \
  --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
  --data_path national_illness.csv \
  --model_id national_illness_$seq_len'_'36 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 18 \
  --pred_len 36 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.01  >logs/LongForecasting/$model_name'_'ili_$seq_len'_'36.log

torchrun --nproc_per_node=4 /data/a2019102224/PatchTST_supervised/run_longExp.py \
  --is_training 1 \
  --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
  --data_path national_illness.csv \
  --model_id national_illness_$seq_len'_'48 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 18 \
  --pred_len 48 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.01  >logs/LongForecasting/$model_name'_'ili_$seq_len'_'48.log

torchrun --nproc_per_node=4 /data/a2019102224/PatchTST_supervised/run_longExp.py \
  --is_training 1 \
  --root_path /local_datasets/a2019102224/timeseries/all_six_datasets/ \
  --data_path national_illness.csv \
  --model_id national_illness_$seq_len'_'60 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 18 \
  --pred_len 60 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.01  >logs/LongForecasting/$model_name'_'ili_$seq_len'_'60.log
