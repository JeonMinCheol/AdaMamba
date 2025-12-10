#!/usr/bin/bash
#SBATCH -J traffic_TSF_TSM
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_grad
#SBATCH -w ariel-g5
#SBATCH -t 4-0
#SBATCH -o logs/slurm-%A.out

OMP_NUM_THREADS=16

root_path_name=/local_datasets/a2019102224/timeseries/all_six_datasets/
seq_len=336
model_name=TSF_TSM
d_model=256
d_ff=256
n_heads=8
features=M
e_layers=3
pred_len=336
random_seed=2021

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=TSF_TSM

root_path_name=/local_datasets/a2019102224/timeseries/all_six_datasets/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom

random_seed=2021

d_model=64
d_ff=64
hidden_features=64
flow_layers=5
num_bins=6
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
      --e_layers $e_layers \
      --n_heads $n_heads \
      --d_ff $d_ff \
      --d_model $d_model \
      --hidden_features $hidden_features \
      --enc_in 862 \
      --flow_layers $flow_layers \
      --num_bins $num_bins \
      --des 'Exp' \
      --train_epochs 10\
      --alpha 1.0\
      --patch_len 24\
      --stride 12\
      --dropout 0.3 \
      --momentum 0.9\
      --alpha 0.01\
      --num_experts 4\
      --use_multi_gpu \
      --devices '0,1,2,3' \
      --itr 1 --batch_size 16 --learning_rate 0.001 > logs/LongForecasting/$model_name'_'traffic_$seq_len'_'336.log  