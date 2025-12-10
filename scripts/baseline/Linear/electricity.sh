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

data_path_name=electricity.csv
model_id_name=electricity
data_name=custom
root_path_name=/local_datasets/a2019102224/timeseries/all_six_datasets/
d_model=256
d_ff=256
n_heads=8
features=M
e_layers=3
random_seed=2021
pred_len=336

torchrun --nproc_per_node=4 /data/a2019102224/PatchTST_supervised/run_longExp.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name_$seq_len'_'336 \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 321 \
  --des 'Exp' \
  --patience 25 \
  --itr 1 --batch_size 16  --learning_rate 0.001  >logs/LongForecasting/$model_name'_'electricity_$seq_len'_'336.log  