#!/usr/bin/bash
#SBATCH -J ETTm2_TSF_TSM
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_grad
#SBATCH -w ariel-v1
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

root_path_name=/local_datasets/a2019102224/timeseries/all_six_datasets/
seq_len=336
model_name=TSF_TSM
d_model=256
d_ff=256
n_heads=16
features=M
e_layers=3
hidden_features=256
pred_len=336
random_seed=2021
flow_layers=3
num_bins=12

data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

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
      --enc_in 7 \
      --e_layers $e_layers \
      --n_heads $n_heads \
      --d_ff $d_ff \
      --d_model $d_model \
      --hidden_features $hidden_features \
      --flow_layers $flow_layers \
      --num_bins $num_bins \
      --des 'Exp' \
      --train_epochs 100\
      --alpha 1.0\
      --patch_len 24\
      --stride 12\
      --momentum 0.99\
      --num_experts 6\
      --dropout 0.3 \
      --batch_size 512 \
      --use_multi_gpu \
      --devices '0,1,2,3' \
      --itr 1 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log