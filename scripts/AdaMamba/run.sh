#!/usr/bin/bash
#SBATCH -J TSF_TSM
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_grad
#SBATCH -w ariel-v10
#SBATCH -t 4-0
#SBATCH -o logs/slurm.out

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

OMP_NUM_THREADS=16

main_path=/data/a2019102224/AdaMamba/run_longExp.py
root_path_name=/local_datasets/a2019102224/timeseries_dataset/
model_name=AdaMamba

n_heads=4
features=M
seq_len=336
pred_len=336
nodes=4

for random_seed in 2021
do
# data_path_name=ETTh1.csv
# model_id_name=ETTh1
# data_name=ETTh1
# torchrun --nproc_per_node=$nodes $main_path \
#     --random_seed $random_seed \
#     --is_training 1 \
#     --root_path $root_path_name \
#     --data_path $data_path_name \
#     --model_id $model_id_name'_'$seq_len'_'$pred_len \
#     --model $model_name \
#     --data $data_name \
#     --features $features \
#     --seq_len $seq_len \
#     --pred_len $pred_len \
#     --n_heads 2 \
#     --enc_in 7 \
#     --c_out 7 \
#     --patch_len 48\
#     --stride 48\
#     --num_experts 8 \
#     --dropout 0.4 \
#     --d_model 64 \
#     --d_ff 8192 \
#     --d_head 2048 \
#     --head_dropout 0.0 \
#     --batch_size 512 \
#     --moe_output_dropout 0.0 \
#     --moe_expert_dropout 0.4 \
#     --learning_rate 0.0002 \
#     --lambda_h_loss 0.1 \
#     --lambda_q_loss 0.1 \
#     --lambda_d_loss 0.1 \
#     --tau 0.5 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

# data_path_name=ETTh2.csv
# model_id_name=ETTh2
# data_name=ETTh2
# torchrun --nproc_per_node=$nodes $main_path \
#     --random_seed $random_seed \
#     --is_training 1 \
#     --root_path $root_path_name \
#     --data_path $data_path_name \
#     --model_id $model_id_name'_'$seq_len'_'$pred_len \
#     --model $model_name \
#     --data $data_name \
#     --features $features \
#     --seq_len $seq_len \
#     --pred_len $pred_len \
#     --n_heads 2 \
#     --enc_in 7 \
#     --c_out 7 \
#     --d_model 128 \
#     --d_head 2048 \
#     --patch_len 48\
#     --stride 48\
#     --dropout 0.2 \
#     --num_experts 8 \
#     --batch_size 512 \
#     --moe_output_dropout 0.0 \
#     --moe_expert_dropout 0.0 \
#     --learning_rate 0.00011 \
#     --lambda_h_loss 0.0001 \
#     --lambda_q_loss 100.0 \
#     --lambda_d_loss 100.0 \
#     --tau 1.0 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

# data_path_name=ETTm1.csv
# model_id_name=ETTm1
# data_name=ETTm1
# torchrun --nproc_per_node=$nodes $main_path \
#     --random_seed $random_seed \
#     --is_training 1 \
#     --root_path $root_path_name \
#     --data_path $data_path_name \
#     --model_id $model_id_name'_'$seq_len'_'$pred_len \
#     --model $model_name \
#     --data $data_name \
#     --features $features \
#     --seq_len $seq_len \
#     --pred_len $pred_len \
#     --enc_in 7 \
#     --n_heads 4 \
#     --patch_len 48 \
#     --stride 48 \
#     --num_experts 8 \
#     --dropout 0.4 \
#     --d_model 512 \
#     --head_dropout 0.3 \
#     --d_ff 8192 \
#     --d_head 2048 \
#     --batch_size 256 \
#     --moe_output_dropout 0.1 \
#     --moe_expert_dropout 0.6 \
#     --learning_rate 0.000047 \
#     --lambda_h_loss 0.05 \
#     --lambda_q_loss 0.6 \
#     --lambda_d_loss 0.09 \
#     --reduction_ratio 6 \
#     --tau 200.0 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

# data_path_name=ETTm2.csv
# model_id_name=ETTm2
# data_name=ETTm2
# torchrun --nproc_per_node=$nodes $main_path \
#     --random_seed $random_seed \
#     --root_path $root_path_name \
#     --data_path $data_path_name \
#     --model_id $model_id_name'_'$seq_len'_'$pred_len \
#     --model $model_name \
#     --data $data_name \
#     --features $features \
#     --seq_len $seq_len \
#     --pred_len $pred_len \
#     --n_heads 8 \
#     --is_training 1 \
#     --enc_in 7 \
#     --patch_len 48 \
#     --stride 48 \
#     --num_experts 8 \
#     --moe_output_dropout 0.2 \
#     --moe_expert_dropout 0.5 \
#     --dropout 0.35 \
#     --head_dropout 0.0 \
#     --d_model 512 \
#     --d_ff 4096 \
#     --d_head 4096 \
#     --batch_size 512 \
#     --lambda_h_loss 0.8 \
#     --lambda_q_loss 20 \
#     --lambda_d_loss 0.0 \
#     --reduction_ratio 8 \
#     --learning_rate 0.0001 \
#     --tau 20 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=weather.csv
model_id_name=weather
data_name=custom
for d in 0.4
do
torchrun --nproc_per_node=$nodes $main_path \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --features M \
      --enc_in 21 \
      --n_heads 2 \
      --dropout $d \
      --patch_len 48 \
      --stride 48 \
      --num_experts 8 \
      --lambda_h_loss 1 \
      --lambda_q_loss 1 \
      --lambda_d_loss 1.0 \
      --d_ff 16384 \
      --d_head 8192 \
      --d_model 512 \
      --batch_size 128 \
      --moe_output_dropout 0.0 \
      --moe_expert_dropout 0.0 \
      --tau 10.0 \
      --learning_rate 0.0000025 \
      --reduction_ratio 6 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done
done

# data_path_name=exchange_rate.csv
# model_id_name=Exchange
# data_name=custom
# for lr in 0.000001 0.0000015 0.000002 0.0000025 0.000003 0.0000035 0.000004  
# do
# torchrun --nproc_per_node=$nodes $main_path \
#     --random_seed 2021 \
#     --is_training 1 \
#     --enc_in 8 \
#     --root_path $root_path_name \
#     --data_path $data_path_name \
#     --model_id $model_id_name'_'$seq_len'_'$pred_len \
#     --model $model_name \
#     --data $data_name \
#     --features $features \
#     --seq_len $seq_len \
#     --pred_len $pred_len \
#     --n_heads 4 \
#     --patch_len 48 \
#     --stride 48 \
#     --num_experts 8 \
#     --moe_output_dropout 0.0 \
#     --moe_expert_dropout 0.0 \
#     --reduction_ratio 3 \
#     --lambda_h_loss 0.1 \
#     --lambda_q_loss 0.1 \
#     --lambda_d_loss 0.1 \
#     --batch_size 16 \
#     --tau 1.2 \
#     --dropout 0.0 \
#     --head_dropout 0.0 \
#     --d_ff 16384 \
#     --d_head 8192 \
#     --d_model 256 \
#     --learning_rate $lr > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
# done
# data_path_name=electricity.csv
# model_id_name=electricity
# data_name=custom
# torchrun --nproc_per_node=$nodes $main_path \
#     --use_multi_gpu \
#     --random_seed 2021 \
#     --is_training 1 \
#     --root_path $root_path_name \
#     --data_path $data_path_name \
#     --model_id $model_id_name'_'$seq_len'_'$pred_len \
#     --model $model_name \
#     --data $data_name \
#     --features $features \
#     --seq_len $seq_len \
#     --pred_len $pred_len \
#     --n_heads 4 \
#     --enc_in 321 \
#     --dropout 0.15 \
#     --patch_len 48 \
#     --stride 48 \
#     --d_ff 4096 \
#     --d_model 1024 \
#     --batch_size 8 \
#     --num_experts 2 \
#     --tau 0.8 \
#     --d_head 6000 \
#     --moe_output_dropout 0.15 \
#     --moe_expert_dropout 0.4 \
#     --lambda_h_loss 1.0 \
#     --lambda_q_loss 0.00001 \
#     --lambda_d_loss 0.0005 \
#     --reduction_ratio 7 \
#     --learning_rate 0.000007 > logs/LongForecasting/$model_name'_'electricity_$seq_len'_'336.log  
