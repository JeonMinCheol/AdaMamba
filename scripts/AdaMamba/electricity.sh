#!/usr/bin/bash
#SBATCH -J electricity_TSF_TSM
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_grad
#SBATCH -w ariel-v1
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

data_path_name=electricity.csv
model_id_name=electricity
data_name=custom

torchrun --nproc_per_node=4 --master-port 20070 /data/a2019102224/PatchTST_supervised/run_longExp.py \
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
      --enc_in 321 \
      --e_layers $e_layers \
      --n_heads $n_heads \
      --d_ff $d_ff \
      --d_model $d_model \
      --hidden_features 256 \
      --flow_layers 8 \
      --num_bins 12 \
      --des 'Exp' \
      --train_epochs 100\
      --alpha 1.0\
      --patch_len 24\
      --stride 12\
      --momentum 0.99\
      --num_experts 2 \
      --use_multi_gpu \
      --devices '0,1,2,3' \
      --itr 1 --batch_size 16 --learning_rate 0.001  >logs/LongForecasting/$model_name'_'electricity_$seq_len'_'336.log  