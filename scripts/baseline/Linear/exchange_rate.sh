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

data_path_name=exchange_rate.csv
model_id_name=Exchange
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
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features $features \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 8 \
      --e_layers $e_layers \
      --n_heads $n_heads \
      --d_ff $d_ff \
      --d_model $d_model \
      --num_bins 12 \
      --des 'Exp' \
      --momentum 0.99\
      --itr 1 --batch_size 8 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
