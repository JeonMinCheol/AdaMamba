#!/usr/bin/bash
#SBATCH -J env_model_train
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_grad
#SBATCH -w ariel-v11
#SBATCH -t 4-0
#SBATCH -o logs/slurm-%A.out

OMP_NUM_THREADS=16

# add --individual for TSF_TSM-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

OMP_NUM_THREADS=16

random_seed=2021
main_path=/data/a2019102224/AdaMamba/run_longExp.py
root_path_name=/local_datasets/a2019102224/timeseries_dataset/
model_name=AdaMamba

n_heads=4
features=M
seq_len=336
pred_len=336
nodes=4

data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1
torchrun --nproc_per_node=$nodes $main_path \
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
    --enc_in 1 \
    --n_heads 4 \
    --patch_len 48 \
    --stride 48 \
    --num_experts 8 \
    --dropout 0.4 \
    --d_model 128 \
    --d_ff 256 \
    --d_head 512 \
    --batch_size 256 \
    --learning_rate 0.0003 \
    --lambda_h_loss 1 \
    --lambda_q_loss 1 \
    --reduction_ratio 6 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 