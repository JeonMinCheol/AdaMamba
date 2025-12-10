#!/usr/bin/bash
#SBATCH -J ETTh1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_grad
#SBATCH -w ariel-v4
#SBATCH -t 4-0
#SBATCH -o logs/slurm-%A.out

# add --individual for TSF_TSM-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=TSF_TSM

root_path_name=/local_datasets/a2019102224/timeseries/all_six_datasets/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
d_model=32
d_ff=32
n_heads=8
features=M
e_layers=2

random_seed=2021
for pred_len in 336
do
torchrun --nproc_per_node=3 /data/a2019102224/PatchTST_supervised/run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
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
      --hidden_features 64 \
      --flow_layers 4 \
      --num_bins 4 \
      --des 'Exp' \
      --train_epochs 50\
      --alpha 1.0\
      --patch_len 24\
      --stride 12\
      --momentum 0.99\
      --num_experts 4\
      --dropout 0.0\
      --itr 1 --batch_size 512 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done
