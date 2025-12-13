#!/usr/bin/bash
#SBATCH -J TSF_TSM
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_grad
#SBATCH -w ariel-g5
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
#     --enc_in 1 \
#     --patch_len 48 \
#     --stride 48 \
#     --dropout 0.35 \
#     --d_model 128 \
#     --d_ff 128 \
#     --d_head 1024 \
#     --learning_rate 0.0005 \
#     --lambda_h_loss 1 \
#     --lambda_q_loss 1 \
#     --reduction_ratio 4 \
#     --batch_size 512  >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

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
#     --enc_in 1 \
#     --d_model 128 \
#     --d_ff 128 \
#     --d_head 4096 \
#     --patch_len 48 \
#     --stride 48 \
#     --dropout 0.45 \
#     --batch_size 512 \
#     --reduction_ratio 4 \
#     --learning_rate 0.0003 \
#     --lambda_h_loss 1 \
#     --lambda_q_loss 0.1 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

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
#     --enc_in 1 \
#     --n_heads 4 \
#     --patch_len 48 \
#     --stride 48 \
#     --dropout 0.4 \
#     --d_model 128 \
#     --d_ff 256 \
#     --d_head 512 \
#     --batch_size 512 \
#     --learning_rate 0.0003 \
#     --lambda_h_loss 1 \
#     --lambda_q_loss 1 \
#     --reduction_ratio 6 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

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
#     --enc_in 1 \
#     --patch_len 48 \
#     --stride 48 \
#     --dropout 0.35 \
#     --head_dropout 0.0 \
#     --d_model 512 \
#     --d_ff 128 \
#     --d_head 128 \
#     --batch_size 512 \
#     --lambda_h_loss 1 \
#     --lambda_q_loss 1 \
#     --reduction_ratio 4 \
#     --learning_rate 0.0002  > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

# data_path_name=weather.csv
# model_id_name=weather
# data_name=custom
# torchrun --nproc_per_node=$nodes $main_path \
#     --random_seed $random_seed \
#     --is_training 1 \
#     --root_path $root_path_name \
#     --data_path $data_path_name \
#     --model_id $model_id_name'_'$seq_len'_'$pred_len \
#     --model $model_name \
#     --data $data_name \
#     --seq_len $seq_len \
#     --pred_len $pred_len \
#     --features M \
#     --enc_in 1 \
#     --n_heads 2 \
#     --dropout 0.3 \
#     --patch_len 48 \
#     --stride 48 \
#     --lambda_h_loss 1 \
#     --lambda_q_loss 1 \
#     --d_ff 256 \
#     --d_head 512 \
#     --d_model 128 \
#     --batch_size 512 \
#     --learning_rate 0.0002 \
#     --reduction_ratio 6 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

# data_path_name=exchange_rate.csv
# model_id_name=Exchange
# data_name=custom
# torchrun --nproc_per_node=$nodes $main_path \
#     --random_seed $random_seed \
#     --is_training 1 \
#     --enc_in 1 \
#     --root_path $root_path_name \
#     --data_path $data_path_name \
#     --model_id $model_id_name'_'$seq_len'_'$pred_len \
#     --model $model_name \
#     --data $data_name \
#     --seq_len $seq_len \
#     --pred_len $pred_len \
#     --features M \
#     --n_heads 4 \
#     --patch_len 48 \
#     --stride 48 \
#     --dropout 0.15 \
#     --head_dropout 0.45 \
#     --lambda_h_loss 0.0 \
#     --lambda_q_loss 1 \
#     --d_head 1024 \
#     --d_ff 512 \
#     --d_model 128 \
#     --batch_size 64 \
#     --reduction_ratio 6 \
#     --learning_rate 0.00014 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

# data_path_name=electricity.csv
# model_id_name=electricity
# data_name=custom
# torchrun --nproc_per_node=$nodes $main_path \
#     --use_multi_gpu \
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
#     --n_heads 4 \
#     --enc_in 1 \
#     --dropout 0.15 \
#     --patch_len 48 \
#     --stride 48 \
#     --d_ff 256 \
#     --d_model 256 \
#     --d_head 1024 \
#     --batch_size 8 \
#     --lambda_h_loss 1.0 \
#     --lambda_q_loss 0.1 \
#     --reduction_ratio 7 \
#     --learning_rate 0.0002 > logs/LongForecasting/$model_name'_'electricity_$seq_len'_'336.log  

done
done
done


