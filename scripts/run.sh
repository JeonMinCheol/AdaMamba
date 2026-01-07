#!/usr/bin/bash
#SBATCH -J all
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

main_path=/data/a2019102224/AdaMamba/run_longExp.py
root_path_name=/local_datasets/a2019102224/timeseries_dataset/

nodes=4
features=M
seq_len=336
pred_len=336
random_seed=42

model_name=AdaMamba
norm_type=AdaNorm

# =================================================================================================================================================================
# 6 kernel
# =================================================================================================================================================================

for kernels in "2,4,8,16,32,64,128,256" "3,9,27,81,243" "5,25,125" "7,49" "2,4,8,16,32,64" "3,9,27,54,81,243" "5,15,25,75,125,250" "7,14,21,35,49,98" "2,19,61,127,199,233" "3,6,12,24,48,96"; do
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --dec_in 7 \
    --n_heads 4 \
    --patch_len 48 \
    --stride 48 \
    --dropout 0.4 \
    --d_model 128 \
    --d_ff 256 \
    --d_head 512 \
    --batch_size 512 \
    --learning_rate 0.0003 \
    --lambda_h_loss 1 \
    --lambda_q_loss 1 \
    --kernels $kernels \
    --reduction_ratio 6 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

# =================================================================================================================================================================
# 7 kernel  
# =================================================================================================================================================================

for kernels in "2,4,8,16,32,64,128,256" "3,9,27,81,243" "5,25,125" "7,49"  "3,9,27,54,81,162,243" "5,15,25,50,75,125,250" "7,14,21,28,35,49,98" "2,19,61,89,167,199,233" "3,6,12,24,48,96,192"; do
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --dec_in 7 \
    --n_heads 2 \
    --d_model 128 \
    --d_ff 128 \
    --d_head 4096 \
    --patch_len 48 \
    --stride 48 \
    --dropout 0.45 \
    --batch_size 512 \
    --reduction_ratio 4 \
    --learning_rate 0.0003 \
    --lambda_h_loss 1 \
    --kernels $kernels \
    --lambda_q_loss 0.1 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --norm_type $norm_type \
    --enc_in 7 \
    --dec_in 7 \
    --n_heads 8 \
    --patch_len 48 \
    --stride 48 \
    --dropout 0.35 \
    --head_dropout 0.0 \
    --d_model 512 \
    --d_ff 128 \
    --d_head 128 \
    --batch_size 512 \
    --lambda_h_loss 1 \
    --lambda_q_loss 1 \
    --reduction_ratio 4 \
    --kernels 5,8,13,21,34,55,89 \
    --use_gt False \
    --learning_rate 0.0002  > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=weather.csv
model_id_name=weather
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --features $features \
    --enc_in 21 \
    --dec_in 21 \
    --n_heads 2 \
    --dropout 0.15 \
    --patch_len 48 \
    --stride 48 \
    --lambda_h_loss 1 \
    --lambda_q_loss 1 \
    --d_ff 256 \
    --d_head 512 \
    --d_model 128 \
    --batch_size 512 \
    --learning_rate 0.001 \
    --kernels $kernels \
    --reduction_ratio 6 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=exchange_rate.csv
model_id_name=Exchange
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --features $features \
    --enc_in 8 \
    --dec_in 8 \
    --n_heads 4 \
    --patch_len 48 \
    --stride 48 \
    --dropout 0.25 \
    --head_dropout 0.05 \
    --lambda_h_loss 0.1 \
    --lambda_q_loss 1.5 \
    --d_head 1024 \
    --d_ff 512 \
    --d_model 128 \
    --batch_size 8 \
    --reduction_ratio 6 \
    --kernels $kernels \
    --learning_rate 0.0001 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

# =================================================================================================================================================================
# 8 kernel
# =================================================================================================================================================================

for kernels in "2,4,8,16,32,64,128,256" "3,9,27,81,243" "5,25,125" "7,49" "2,4,8,16,32,64,128,256" "3,6,9,18,27,54,81,243" "5,15,25,50,75,100,125,250" "7,14,21,28,35,42,49,98" "2,19,61,89,127,167,199,233" "3,6,12,24,48,96,144,192"; do
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --norm_type $norm_type \
    --n_heads 2 \
    --patch_len 48 \
    --stride 48 \
    --dropout 0.35 \
    --d_model 128 \
    --d_ff 128 \
    --d_head 1024 \
    --enc_in 7 \
    --dec_in 7 \
    --learning_rate 0.0005 \
    --lambda_h_loss 1 \
    --lambda_q_loss 1 \
    --reduction_ratio 4 \
    --use_gt False \
    --kernels 3,5,8,13,21,34,55,89 \
    --batch_size 512  >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=electricity.csv
model_id_name=electricity
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --use_multi_gpu \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --norm_type $norm_type \
    --enc_in 321 \
    --dec_in 321 \
    --dropout 0.05 \
    --n_heads 4 \
    --patch_len 48 \
    --stride 48 \
    --d_ff 1024 \
    --d_model 256 \
    --d_head 2048 \
    --batch_size 8 \
    --lambda_h_loss 1.0 \
    --lambda_q_loss 0.2 \
    --reduction_ratio 4 \
    --kernels $kernels \
    --learning_rate 0.0002 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

# =================================================================================================================================================================
# use_gt False
# =================================================================================================================================================================

data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --norm_type $norm_type \
    --n_heads 2 \
    --patch_len 48 \
    --stride 48 \
    --dropout 0.35 \
    --d_model 128 \
    --d_ff 128 \
    --d_head 1024 \
    --enc_in 7 \
    --dec_in 7 \
    --learning_rate 0.0005 \
    --lambda_h_loss 1 \
    --lambda_q_loss 1 \
    --reduction_ratio 4 \
    --use_gt False \
    --kernels 3,5,8,13,21,34,55,89 \
    --batch_size 512  >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --norm_type $norm_type \
    --enc_in 7 \
    --dec_in 7 \
    --n_heads 2 \
    --d_model 128 \
    --d_ff 128 \
    --d_head 4096 \
    --patch_len 48 \
    --stride 48 \
    --dropout 0.45 \
    --batch_size 512 \
    --reduction_ratio 4 \
    --learning_rate 0.0003 \
    --lambda_h_loss 1 \
    --kernels 5,8,13,21,34,55,89 \
    --use_gt False \
    --lambda_q_loss 0.1 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --norm_type $norm_type \
    --enc_in 7 \
    --dec_in 7 \
    --n_heads 4 \
    --patch_len 48 \
    --stride 48 \
    --dropout 0.4 \
    --d_model 128 \
    --d_ff 256 \
    --d_head 512 \
    --batch_size 512 \
    --learning_rate 0.0003 \
    --lambda_h_loss 1 \
    --lambda_q_loss 1 \
    --kernels 5,8,13,21,34,55 \
    --use_gt False \
    --reduction_ratio 6 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --norm_type $norm_type \
    --enc_in 7 \
    --dec_in 7 \
    --n_heads 8 \
    --patch_len 48 \
    --stride 48 \
    --dropout 0.35 \
    --head_dropout 0.0 \
    --d_model 512 \
    --d_ff 128 \
    --d_head 128 \
    --batch_size 512 \
    --lambda_h_loss 1 \
    --lambda_q_loss 1 \
    --reduction_ratio 4 \
    --kernels 5,8,13,21,34,55,89 \
    --use_gt False \
    --learning_rate 0.0002  > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=weather.csv
model_id_name=weather
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --features $features \
    --norm_type $norm_type \
    --enc_in 21 \
    --dec_in 21 \
    --n_heads 2 \
    --dropout 0.15 \
    --patch_len 48 \
    --stride 48 \
    --lambda_h_loss 1 \
    --lambda_q_loss 1 \
    --d_ff 256 \
    --d_head 512 \
    --d_model 128 \
    --batch_size 512 \
    --learning_rate 0.001 \
    --kernels 5,8,13,21,34,55,89 \
    --use_gt False \
    --reduction_ratio 6 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=exchange_rate.csv
model_id_name=Exchange
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --features $features \
    --norm_type $norm_type \
    --enc_in 8 \
    --dec_in 8 \
    --n_heads 4 \
    --patch_len 48 \
    --stride 48 \
    --dropout 0.25 \
    --head_dropout 0.05 \
    --lambda_h_loss 0.1 \
    --lambda_q_loss 1.5 \
    --d_head 1024 \
    --d_ff 512 \
    --d_model 128 \
    --batch_size 8 \
    --reduction_ratio 6 \
    --kernels 5,8,13,21,34,55,89 \
    --use_gt False \
    --learning_rate 0.0001 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=electricity.csv
model_id_name=electricity
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --use_multi_gpu \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --norm_type $norm_type \
    --enc_in 321 \
    --dec_in 321 \
    --dropout 0.05 \
    --n_heads 4 \
    --patch_len 48 \
    --stride 48 \
    --d_ff 1024 \
    --d_model 256 \
    --d_head 2048 \
    --batch_size 8 \
    --lambda_h_loss 1.0 \
    --lambda_q_loss 0.2 \
    --reduction_ratio 4 \
    --kernels 3,5,8,13,21,34,55,89 \
    --use_gt False \
    --learning_rate 0.0002 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

# =================================================================================================================================================================
# use_se False
# =================================================================================================================================================================

data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --norm_type $norm_type \
    --n_heads 2 \
    --patch_len 48 \
    --stride 48 \
    --dropout 0.35 \
    --d_model 128 \
    --d_ff 128 \
    --d_head 1024 \
    --enc_in 7 \
    --dec_in 7 \
    --learning_rate 0.0005 \
    --lambda_h_loss 1 \
    --lambda_q_loss 1 \
    --reduction_ratio 4 \
    --use_se False \
    --kernels 3,5,8,13,21,34,55,89 \
    --batch_size 512  >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --norm_type $norm_type \
    --enc_in 7 \
    --dec_in 7 \
    --n_heads 2 \
    --d_model 128 \
    --d_ff 128 \
    --d_head 4096 \
    --patch_len 48 \
    --stride 48 \
    --dropout 0.45 \
    --batch_size 512 \
    --reduction_ratio 4 \
    --learning_rate 0.0003 \
    --lambda_h_loss 1 \
    --kernels 5,8,13,21,34,55,89 \
    --use_se False \
    --lambda_q_loss 0.1 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --norm_type $norm_type \
    --enc_in 7 \
    --dec_in 7 \
    --n_heads 4 \
    --patch_len 48 \
    --stride 48 \
    --dropout 0.4 \
    --d_model 128 \
    --d_ff 256 \
    --d_head 512 \
    --batch_size 512 \
    --learning_rate 0.0003 \
    --lambda_h_loss 1 \
    --lambda_q_loss 1 \
    --kernels 5,8,13,21,34,55 \
    --use_se False \
    --reduction_ratio 6 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --norm_type $norm_type \
    --enc_in 7 \
    --dec_in 7 \
    --n_heads 8 \
    --patch_len 48 \
    --stride 48 \
    --dropout 0.35 \
    --head_dropout 0.0 \
    --d_model 512 \
    --d_ff 128 \
    --d_head 128 \
    --batch_size 512 \
    --lambda_h_loss 1 \
    --lambda_q_loss 1 \
    --reduction_ratio 4 \
    --kernels 5,8,13,21,34,55,89 \
    --use_se False \
    --learning_rate 0.0002  > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=weather.csv
model_id_name=weather
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --features $features \
    --norm_type $norm_type \
    --enc_in 21 \
    --dec_in 21 \
    --n_heads 2 \
    --dropout 0.15 \
    --patch_len 48 \
    --stride 48 \
    --lambda_h_loss 1 \
    --lambda_q_loss 1 \
    --d_ff 256 \
    --d_head 512 \
    --d_model 128 \
    --batch_size 512 \
    --learning_rate 0.001 \
    --kernels 5,8,13,21,34,55,89 \
    --use_se False \
    --reduction_ratio 6 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=exchange_rate.csv
model_id_name=Exchange
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --features $features \
    --norm_type $norm_type \
    --enc_in 8 \
    --dec_in 8 \
    --n_heads 4 \
    --patch_len 48 \
    --stride 48 \
    --dropout 0.25 \
    --head_dropout 0.05 \
    --lambda_h_loss 0.1 \
    --lambda_q_loss 1.5 \
    --d_head 1024 \
    --d_ff 512 \
    --d_model 128 \
    --batch_size 8 \
    --reduction_ratio 6 \
    --kernels 5,8,13,21,34,55,89 \
    --use_se False \
    --learning_rate 0.0001 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=electricity.csv
model_id_name=electricity
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --use_multi_gpu \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --norm_type $norm_type \
    --enc_in 321 \
    --dec_in 321 \
    --dropout 0.05 \
    --n_heads 4 \
    --patch_len 48 \
    --stride 48 \
    --d_ff 1024 \
    --d_model 256 \
    --d_head 2048 \
    --batch_size 8 \
    --lambda_h_loss 1.0 \
    --lambda_q_loss 0.2 \
    --reduction_ratio 4 \
    --kernels 3,5,8,13,21,34,55,89 \
    --use_se False \
    --learning_rate 0.0002 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
    
# =================================================================================================================================================================
# norm_type RevIN
# =================================================================================================================================================================

norm_type=RevIN
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --norm_type $norm_type \
    --n_heads 2 \
    --patch_len 48 \
    --stride 48 \
    --dropout 0.35 \
    --d_model 128 \
    --d_ff 128 \
    --d_head 1024 \
    --enc_in 7 \
    --dec_in 7 \
    --learning_rate 0.0005 \
    --lambda_h_loss 1 \
    --lambda_q_loss 1 \
    --reduction_ratio 4 \
    --kernels 3,5,8,13,21,34,55,89 \
    --norm_type $norm_type \
    --batch_size 512  >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --norm_type $norm_type \
    --enc_in 7 \
    --dec_in 7 \
    --n_heads 2 \
    --d_model 128 \
    --d_ff 128 \
    --d_head 4096 \
    --patch_len 48 \
    --stride 48 \
    --dropout 0.45 \
    --batch_size 512 \
    --reduction_ratio 4 \
    --learning_rate 0.0003 \
    --lambda_h_loss 1 \
    --kernels 5,8,13,21,34,55,89 \
    --norm_type $norm_type \
    --lambda_q_loss 0.1 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --norm_type $norm_type \
    --enc_in 7 \
    --dec_in 7 \
    --n_heads 4 \
    --patch_len 48 \
    --stride 48 \
    --dropout 0.4 \
    --d_model 128 \
    --d_ff 256 \
    --d_head 512 \
    --batch_size 512 \
    --learning_rate 0.0003 \
    --lambda_h_loss 1 \
    --lambda_q_loss 1 \
    --kernels 5,8,13,21,34,55 \
    --norm_type $norm_type \
    --reduction_ratio 6 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --norm_type $norm_type \
    --enc_in 7 \
    --dec_in 7 \
    --n_heads 8 \
    --patch_len 48 \
    --stride 48 \
    --dropout 0.35 \
    --head_dropout 0.0 \
    --d_model 512 \
    --d_ff 128 \
    --d_head 128 \
    --batch_size 512 \
    --lambda_h_loss 1 \
    --lambda_q_loss 1 \
    --reduction_ratio 4 \
    --kernels 5,8,13,21,34,55,89 \
    --norm_type $norm_type \
    --learning_rate 0.0002  > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=weather.csv
model_id_name=weather
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --features $features \
    --norm_type $norm_type \
    --enc_in 21 \
    --dec_in 21 \
    --n_heads 2 \
    --dropout 0.15 \
    --patch_len 48 \
    --stride 48 \
    --lambda_h_loss 1 \
    --lambda_q_loss 1 \
    --d_ff 256 \
    --d_head 512 \
    --d_model 128 \
    --batch_size 512 \
    --learning_rate 0.001 \
    --kernels 5,8,13,21,34,55,89 \
    --norm_type $norm_type \
    --reduction_ratio 6 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=exchange_rate.csv
model_id_name=Exchange
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --features $features \
    --norm_type $norm_type \
    --enc_in 8 \
    --dec_in 8 \
    --n_heads 4 \
    --patch_len 48 \
    --stride 48 \
    --dropout 0.25 \
    --head_dropout 0.05 \
    --lambda_h_loss 0.1 \
    --lambda_q_loss 1.5 \
    --d_head 1024 \
    --d_ff 512 \
    --d_model 128 \
    --batch_size 8 \
    --reduction_ratio 6 \
    --kernels 5,8,13,21,34,55,89 \
    --norm_type $norm_type \
    --learning_rate 0.0001 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=electricity.csv
model_id_name=electricity
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --use_multi_gpu \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --norm_type $norm_type \
    --enc_in 321 \
    --dec_in 321 \
    --dropout 0.05 \
    --n_heads 4 \
    --patch_len 48 \
    --stride 48 \
    --d_ff 1024 \
    --d_model 256 \
    --d_head 2048 \
    --batch_size 8 \
    --lambda_h_loss 1.0 \
    --lambda_q_loss 0.2 \
    --reduction_ratio 4 \
    --kernels 3,5,8,13,21,34,55,89 \
    --norm_type $norm_type \
    --learning_rate 0.0002 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

# =================================================================================================================================================================
# norm_type None
# =================================================================================================================================================================

norm_type=None
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --norm_type $norm_type \
    --n_heads 2 \
    --patch_len 48 \
    --stride 48 \
    --dropout 0.35 \
    --d_model 128 \
    --d_ff 128 \
    --d_head 1024 \
    --enc_in 7 \
    --dec_in 7 \
    --learning_rate 0.0005 \
    --lambda_h_loss 1 \
    --lambda_q_loss 1 \
    --reduction_ratio 4 \
    --kernels 3,5,8,13,21,34,55,89 \
    --norm_type $norm_type \
    --batch_size 512  >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --norm_type $norm_type \
    --enc_in 7 \
    --dec_in 7 \
    --n_heads 2 \
    --d_model 128 \
    --d_ff 128 \
    --d_head 4096 \
    --patch_len 48 \
    --stride 48 \
    --dropout 0.45 \
    --batch_size 512 \
    --reduction_ratio 4 \
    --learning_rate 0.0003 \
    --lambda_h_loss 1 \
    --kernels 5,8,13,21,34,55,89 \
    --norm_type $norm_type \
    --lambda_q_loss 0.1 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --norm_type $norm_type \
    --enc_in 7 \
    --dec_in 7 \
    --n_heads 4 \
    --patch_len 48 \
    --stride 48 \
    --dropout 0.4 \
    --d_model 128 \
    --d_ff 256 \
    --d_head 512 \
    --batch_size 512 \
    --learning_rate 0.0003 \
    --lambda_h_loss 1 \
    --lambda_q_loss 1 \
    --kernels 5,8,13,21,34,55 \
    --norm_type $norm_type \
    --reduction_ratio 6 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --norm_type $norm_type \
    --enc_in 7 \
    --dec_in 7 \
    --n_heads 8 \
    --patch_len 48 \
    --stride 48 \
    --dropout 0.35 \
    --head_dropout 0.0 \
    --d_model 512 \
    --d_ff 128 \
    --d_head 128 \
    --batch_size 512 \
    --lambda_h_loss 1 \
    --lambda_q_loss 1 \
    --reduction_ratio 4 \
    --kernels 5,8,13,21,34,55,89 \
    --norm_type $norm_type \
    --learning_rate 0.0002  > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=weather.csv
model_id_name=weather
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --features $features \
    --norm_type $norm_type \
    --enc_in 21 \
    --dec_in 21 \
    --n_heads 2 \
    --dropout 0.15 \
    --patch_len 48 \
    --stride 48 \
    --lambda_h_loss 1 \
    --lambda_q_loss 1 \
    --d_ff 256 \
    --d_head 512 \
    --d_model 128 \
    --batch_size 512 \
    --learning_rate 0.001 \
    --kernels 5,8,13,21,34,55,89 \
    --norm_type $norm_type \
    --reduction_ratio 6 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=exchange_rate.csv
model_id_name=Exchange
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --features $features \
    --norm_type $norm_type \
    --enc_in 8 \
    --dec_in 8 \
    --n_heads 4 \
    --patch_len 48 \
    --stride 48 \
    --dropout 0.25 \
    --head_dropout 0.05 \
    --lambda_h_loss 0.1 \
    --lambda_q_loss 1.5 \
    --d_head 1024 \
    --d_ff 512 \
    --d_model 128 \
    --batch_size 8 \
    --reduction_ratio 6 \
    --kernels 5,8,13,21,34,55,89 \
    --norm_type $norm_type \
    --learning_rate 0.0001 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=electricity.csv
model_id_name=electricity
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --use_multi_gpu \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --norm_type $norm_type \
    --enc_in 321 \
    --dec_in 321 \
    --dropout 0.05 \
    --n_heads 4 \
    --patch_len 48 \
    --stride 48 \
    --d_ff 1024 \
    --d_model 256 \
    --d_head 2048 \
    --batch_size 8 \
    --lambda_h_loss 1.0 \
    --lambda_q_loss 0.2 \
    --reduction_ratio 4 \
    --kernels 3,5,8,13,21,34,55,89 \
    --norm_type $norm_type \
    --learning_rate 0.0002 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

# =================================================================================================================================================================
# default setting
# =================================================================================================================================================================

norm_type=AdaNorm
for pred_len in 96 192 336 720; do
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --norm_type $norm_type \
    --n_heads 2 \
    --patch_len 48 \
    --stride 48 \
    --dropout 0.35 \
    --d_model 128 \
    --d_ff 128 \
    --d_head 1024 \
    --enc_in 7 \
    --dec_in 7 \
    --learning_rate 0.0005 \
    --lambda_h_loss 1 \
    --lambda_q_loss 1 \
    --reduction_ratio 4 \
    --kernels 3,5,8,13,21,34,55,89 \
    --batch_size 512  >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --norm_type $norm_type \
    --enc_in 7 \
    --dec_in 7 \
    --n_heads 2 \
    --d_model 128 \
    --d_ff 128 \
    --d_head 4096 \
    --patch_len 48 \
    --stride 48 \
    --dropout 0.45 \
    --batch_size 512 \
    --reduction_ratio 4 \
    --learning_rate 0.0003 \
    --lambda_h_loss 1 \
    --kernels 5,8,13,21,34,55,89 \
    --lambda_q_loss 0.1 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --norm_type $norm_type \
    --enc_in 7 \
    --dec_in 7 \
    --n_heads 4 \
    --patch_len 48 \
    --stride 48 \
    --dropout 0.4 \
    --d_model 128 \
    --d_ff 256 \
    --d_head 512 \
    --batch_size 512 \
    --learning_rate 0.0003 \
    --lambda_h_loss 1 \
    --lambda_q_loss 1 \
    --kernels 5,8,13,21,34,55 \
    --reduction_ratio 6 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --norm_type $norm_type \
    --enc_in 7 \
    --dec_in 7 \
    --n_heads 8 \
    --patch_len 48 \
    --stride 48 \
    --dropout 0.35 \
    --head_dropout 0.0 \
    --d_model 512 \
    --d_ff 128 \
    --d_head 128 \
    --batch_size 512 \
    --lambda_h_loss 1 \
    --lambda_q_loss 1 \
    --reduction_ratio 4 \
    --kernels 5,8,13,21,34,55,89 \
    --learning_rate 0.0002  > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=weather.csv
model_id_name=weather
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --features $features \
    --norm_type $norm_type \
    --enc_in 21 \
    --dec_in 21 \
    --n_heads 2 \
    --dropout 0.15 \
    --patch_len 48 \
    --stride 48 \
    --lambda_h_loss 1 \
    --lambda_q_loss 1 \
    --d_ff 256 \
    --d_head 512 \
    --d_model 128 \
    --batch_size 512 \
    --learning_rate 0.001 \
    --kernels 5,8,13,21,34,55,89 \
    --reduction_ratio 6 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=exchange_rate.csv
model_id_name=Exchange
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --features $features \
    --norm_type $norm_type \
    --enc_in 8 \
    --dec_in 8 \
    --n_heads 4 \
    --patch_len 48 \
    --stride 48 \
    --dropout 0.25 \
    --head_dropout 0.05 \
    --lambda_h_loss 0.1 \
    --lambda_q_loss 1.5 \
    --d_head 1024 \
    --d_ff 512 \
    --d_model 128 \
    --batch_size 8 \
    --reduction_ratio 6 \
    --kernels 5,8,13,21,34,55,89 \
    --learning_rate 0.0001 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=electricity.csv
model_id_name=electricity
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --is_training 1 \
    --use_multi_gpu \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --norm_type $norm_type \
    --enc_in 321 \
    --dec_in 321 \
    --dropout 0.05 \
    --n_heads 4 \
    --patch_len 48 \
    --stride 48 \
    --d_ff 1024 \
    --d_model 256 \
    --d_head 2048 \
    --batch_size 8 \
    --lambda_h_loss 1.0 \
    --lambda_q_loss 0.2 \
    --reduction_ratio 4 \
    --kernels 3,5,8,13,21,34,55,89 \
    --learning_rate 0.0002 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

# ================================================================================================================================================================
# Baselines
# ================================================================================================================================================================

model_name=PatchTST
for pred_len in 96 192 336 720 ; do
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
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
      --e_layers 3 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --batch_size 512 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
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
       --e_layers 3 \
       --n_heads 4 \
       --d_model 16 \
       --d_ff 128 \
       --dropout 0.3\
       --fc_dropout 0.3\
       --head_dropout 0\
       --patch_len 16\
       --stride 8\
       --batch_size 512 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
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
       --e_layers 3 \
       --n_heads 16 \
       --d_model 128 \
       --d_ff 256 \
       --dropout 0.2\
       --fc_dropout 0.2\
       --head_dropout 0\
       --patch_len 16\
       --stride 8\
       --batch_size 512 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
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
       --e_layers 3 \
       --n_heads 16 \
       --d_model 128 \
       --d_ff 256 \
       --dropout 0.2\
       --fc_dropout 0.2\
       --head_dropout 0\
       --patch_len 16\
       --stride 8\
       --batch_size 512 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=weather.csv
model_id_name=weather
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
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
      --enc_in 21 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --batch_size 512 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=exchange_rate.csv
model_id_name=Exchange
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
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
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --batch_size 8 --learning_rate 0.000011 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

 data_path_name=electricity.csv
 model_id_name=Electricity
 data_name=custom
 torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
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
       --e_layers 3 \
       --n_heads 16 \
       --d_model 128 \
       --d_ff 256 \
       --dropout 0.2\
       --fc_dropout 0.2\
       --head_dropout 0\
       --patch_len 16\
       --stride 8\
       --batch_size 8 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

# ================================================================================================================================================================ #

model_name=ModernTCN
for pred_len in 96 192 336 720 ;do
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
   --random_seed $random_seed \
   --is_training 1 \
   --model_id $model_id_name'_'$seq_len'_'$pred_len \
   --model $model_name \
   --root_path $root_path_name \
   --data_path $data_path_name \
   --data $data_name \
   --features $features \
   --seq_len $seq_len \
   --pred_len $pred_len \
   --ffn_ratio 8 \
   --patch_size 8 \
   --patch_stride 4 \
   --num_blocks 1 \
   --large_size 51 \
   --small_size 5 \
   --dims 64 64 64 64 \
   --head_dropout 0.0 \
   --enc_in 7 \
   --dropout 0.3 \
   --batch_size 512 \
   --learning_rate 0.0001 \
   --des Exp \
   --use_multi_scale False \
   --small_kernel_merged False \
   --use_multi_gpu > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
   --random_seed $random_seed \
   --is_training 1 \
   --model_id $model_id_name'_'$seq_len'_'$pred_len \
   --model $model_name \
   --root_path $root_path_name \
   --data_path $data_path_name \
   --data $data_name \
   --features $features \
   --seq_len $seq_len \
   --pred_len $pred_len \
   --ffn_ratio 8 \
   --patch_size 8 \
   --patch_stride 4 \
   --num_blocks 1 \
   --large_size 51 \
   --small_size 5 \
   --dims 64 64 64 64 \
   --head_dropout 0.0 \
   --enc_in 7 \
   --dropout 0.3 \
   --batch_size 512 \
   --learning_rate 0.0001 \
   --des Exp \
   --use_multi_scale False \
   --small_kernel_merged False \
   --use_multi_gpu > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
   --random_seed $random_seed \
   --is_training 1 \
   --model_id $model_id_name'_'$seq_len'_'$pred_len \
   --model $model_name \
   --root_path $root_path_name \
   --data_path $data_path_name \
   --data $data_name \
   --features $features \
   --seq_len $seq_len \
   --pred_len $pred_len \
   --ffn_ratio 8 \
   --patch_size 8 \
   --patch_stride 4 \
   --num_blocks 3 \
   --large_size 51 \
   --small_size 5 \
   --dims 64 64 64 64 \
   --head_dropout 0.1 \
   --enc_in 7 \
   --dropout 0.3 \
   --batch_size 512 \
   --learning_rate 0.0001 \
   --des Exp \
   --use_multi_scale False \
   --small_kernel_merged False \
   --use_multi_gpu > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
   --random_seed $random_seed \
   --is_training 1 \
   --model_id $model_id_name'_'$seq_len'_'$pred_len \
   --model $model_name \
   --root_path $root_path_name \
   --data_path $data_path_name \
   --data $data_name \
   --features $features \
   --seq_len $seq_len \
   --pred_len $pred_len \
   --ffn_ratio 8 \
   --patch_size 8 \
   --patch_stride 4 \
   --num_blocks 3 \
   --large_size 51 \
   --small_size 5 \
   --dims 64 64 64 64 \
   --head_dropout 0.2 \
   --enc_in 7 \
   --dropout 0.8 \
   --batch_size 512 \
   --learning_rate 0.0001 \
   --des Exp \
   --use_multi_scale False \
   --small_kernel_merged False \
   --use_multi_gpu > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=weather.csv
model_id_name=weather
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
   --random_seed $random_seed \
   --is_training 1 \
   --model_id $model_id_name'_'$seq_len'_'$pred_len \
   --model $model_name \
   --root_path $root_path_name \
   --data_path $data_path_name \
   --data $data_name \
   --features $features \
   --seq_len $seq_len \
   --pred_len $pred_len \
  --ffn_ratio 8 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 \
  --large_size 51 \
  --small_size 5 \
  --dims 64 64 64 64 \
  --head_dropout 0.0 \
  --enc_in 21 \
  --dropout 0.4 \
  --batch_size 512 \
  --learning_rate 0.00001 \
  --des Exp \
  --use_multi_scale False \
  --small_kernel_merged False \
  --use_multi_gpu > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=exchange_rate.csv
model_id_name=Exchange
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
   --random_seed $random_seed \
   --is_training 1 \
   --model_id $model_id_name'_'$seq_len'_'$pred_len \
   --model $model_name \
   --root_path $root_path_name \
   --data_path $data_path_name \
   --data $data_name \
   --features $features \
   --seq_len $seq_len \
   --pred_len $pred_len \
  --label_len 0 \
  --ffn_ratio 1 \
  --patch_size 1 \
  --patch_stride 1 \
  --num_blocks 1 \
  --large_size 51 \
  --small_size 5 \
  --dims 64 64 64 64 \
  --head_dropout 0.2 \
  --enc_in 8 \
  --dropout 0.2 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --use_multi_scale False \
  --small_kernel_merged False > logs/LongForecasting/'ModernTCN_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
   --random_seed $random_seed \
   --is_training 1 \
   --model_id $model_id_name'_'$seq_len'_'$pred_len \
   --model $model_name \
   --root_path $root_path_name \
   --data_path $data_path_name \
   --data $data_name \
   --features $features \
   --seq_len $seq_len \
   --pred_len $pred_len \
  --ffn_ratio 8 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 \
  --large_size 51 \
  --small_size 5 \
  --dims 64 64 64 64 \
  --head_dropout 0.0 \
  --enc_in 321 \
  --dropout 0.9 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --des Exp \
  --use_multi_scale False \
  --small_kernel_merged False \
  --use_multi_gpu > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

# ================================================================================================================================================================ #

model_name=DLinear

e_layers=3
n_heads=8

for pred_len in 96 192 336 720 ;do
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
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
    --batch_size 512 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
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
    --batch_size 512 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
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
    --batch_size 512 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
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
    --batch_size 512 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=weather.csv
model_id_name=weather
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
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
    --enc_in 21 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 16\
    --batch_size 512 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=exchange_rate.csv
model_id_name=Exchange
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
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
    --e_layers 3 \
    --n_heads 8 \
    --d_ff 256 \
    --d_model 256 \
    --batch_size 8 --learning_rate 0.0003 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=electricity.csv
model_id_name=electricity
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --data $data_name \
    --pred_len $pred_len \
    --enc_in 321 \
    --batch_size 8 --learning_rate 0.001  >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

# ================================================================================================================================================================ #

model_name=iTransformer

d_model=128
d_ff=128
e_layers=2

for pred_len in 96 192 336 720 ;
do
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
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
    --d_ff $d_ff \
    --d_model $d_model \
    --batch_size 512 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
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
    --d_ff $d_ff \
    --d_model $d_model \
    --batch_size 512 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
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
    --batch_size 512 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
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
    --d_ff $d_ff \
    --d_model $d_model \
    --batch_size 512 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=weather.csv
model_id_name=weather
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
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
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 512\
    --d_ff 512\
    --batch_size 512 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=exchange_rate.csv
model_id_name=exchange_rate
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
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
       --dec_in 8 \
       --c_out 8 \
       --e_layers $e_layers \
       --d_model 128 \
       --d_ff 128 \
       --batch_size 8 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

data_path_name=electricity.csv
model_id_name=electricity
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
   --is_training 0 \
   --random_seed $random_seed \
   --root_path $root_path_name \
   --data_path $data_path_name \
   --model_id $model_id_name'_'$seq_len'_'$pred_len \
   --model $model_name \
   --data $data_name \
   --features $features \
   --seq_len $seq_len \
   --pred_len $pred_len \
   --e_layers $e_layers \
   --d_model 512\
   --d_ff 512 \
   --enc_in 321 \
   --batch_size 8 --learning_rate 0.0001 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

# ================================================================================================================================================================ #

model_name=FEDformer

for pred_len in 96 192 336 720 ;
do
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --is_training 1 \
  --data_path $data_path_name\
  --model $model_name \
  --data $data_name \
  --features $features \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 512 \
  --train_epochs 15 \
  --learning_rate 0.0001 \
  --batch_size 256 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log

data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data_path $data_path_name \
  --model $model_name \
  --data $data_name \
  --features $features \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 512 \
  --batch_size 128 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log

data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data_path $data_path_name \
  --model $model_name \
  --data $data_name \
  --features $features \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 512 \
  --learning_rate 0.0001 \
  --batch_size 128 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log

data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data_path $data_path_name \
  --model $model_name \
  --data $data_name \
  --features $features \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 512 \
  --learning_rate 0.00012 \
  --batch_size 128 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log

data_path_name=exchange_rate.csv
model_id_name=exchange_rate
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data_path $data_path_name \
  --model $model_name \
  --data $data_name \
  --features $features \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --batch_size 8 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log

data_path_name=weather.csv
model_id_name=weather
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data_path $data_path_name \
  --model $model_name \
  --data $data_name \
  --features $features \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --learning_rate 0.0001 \
  --batch_size 128 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log

data_path_name=electricity.csv
model_id_name=electricity
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data_path $data_path_name \
  --model $model_name \
  --data $data_name \
  --features $features \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --batch_size 8 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

# ============================================================================================================

model_name=TimeMixer

e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=32
d_ff=32

for pred_len in 96 192 336 720 ;
do
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data_path $data_path_name \
  --model $model_name \
  --data $data_name \
  --features $features \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --enc_in 7 \
  --c_out 7 \
  --d_model $d_model \
  --d_ff 16 \
  --learning_rate $learning_rate \
  --batch_size 512 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log

data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data_path $data_path_name \
  --model $model_name \
  --data $data_name \
  --features $features \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --enc_in 7 \
  --c_out 7 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 512 \
  --learning_rate $learning_rate \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log

data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data_path $data_path_name \
  --model $model_name \
  --data $data_name \
  --features $features \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --enc_in 7 \
  --c_out 7 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 512 \
  --learning_rate $learning_rate \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log

data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data_path $data_path_name \
  --model $model_name \
  --data $data_name \
  --features $features \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --enc_in 7 \
  --c_out 7 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 512 \
  --learning_rate $learning_rate \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log

data_path_name=exchange_rate.csv
model_id_name=exchange_rate
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data_path $data_path_name \
  --model $model_name \
  --data $data_name \
  --features $features \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 8 \
  --learning_rate $learning_rate \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log

e_layers=3
data_path_name=weather.csv
model_id_name=weather
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data_path $data_path_name \
  --model $model_name \
  --data $data_name \
  --features $features \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 512 \
  --learning_rate $learning_rate \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log

data_path_name=electricity.csv
model_id_name=electricity
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
  --is_training 0 \
  --random_seed $random_seed \
  --root_path $root_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data_path $data_path_name \
  --model $model_name \
  --data $data_name \
  --features $features \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --d_model $d_model \
  --d_ff 16 \
  --batch_size 8 \
  --learning_rate $learning_rate \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

# ============================================================================================================

model_name=MambaTS

for pred_len in 96 192 336 720 ;
do
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data_path $data_path_name \
  --model $model_name \
  --data $data_name \
  --features $features \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 2 \
  --factor 1 \
  --enc_in 7 \
  --c_out 7 \
  --d_model 32  \
  --d_ff 16 \
  --dropout 0.3 \
  --n_heads 16 \
  --learning_rate 0.0005 \
  --batch_size 512 \
  --patch_len 48 \
  --stride 48 \
  --VPT_mode 1 \
  --ATSP_solver SA > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log

data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data_path $data_path_name \
  --model $model_name \
  --data $data_name \
  --features $features \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 2 \
  --factor 1 \
  --enc_in 7 \
  --c_out 7 \
  --d_model 32  \
  --d_ff 16 \
  --n_heads 16 \
  --dropout 0.3 \
  --learning_rate 0.0005 \
  --batch_size 512 \
  --patch_len 48 \
  --stride 48 \
  --VPT_mode 1 \
  --ATSP_solver SA > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log

data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data_path $data_path_name \
  --model $model_name \
  --data $data_name \
  --features $features \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 3 \
  --d_layers 2 \
  --factor 1 \
  --enc_in 7 \
  --c_out 7 \
  --d_model 32  \
  --d_ff 16 \
  --n_heads 16 \
  --dropout 0.3 \
  --learning_rate 0.001 \
  --batch_size 512 \
  --patch_len 48 \
  --stride 48 \
  --VPT_mode 1 \
  --ATSP_solver SA > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log

data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data_path $data_path_name \
  --model $model_name \
  --data $data_name \
  --features $features \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 3 \
  --d_layers 2 \
  --factor 1 \
  --enc_in 7 \
  --c_out 7 \
  --n_heads 16 \
  --d_model 32  \
  --d_ff 16 \
  --dropout 0.3 \
  --learning_rate 0.001 \
  --batch_size 512 \
  --patch_len 48 \
  --stride 48 \
  --VPT_mode 1 \
  --ATSP_solver SA > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log   

data_path_name=exchange_rate.csv
model_id_name=exchange_rate
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data_path $data_path_name \
  --model $model_name \
  --data $data_name \
  --features $features \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 3 \
  --d_layers 2 \
  --factor 1 \
  --enc_in 8 \
  --c_out 8 \
  --n_heads 16 \
  --d_model 32  \
  --d_ff 16 \
  --dropout 0.3 \
  --learning_rate 0.001 \
  --batch_size 8 \
  --patch_len 48 \
  --stride 48 \
  --VPT_mode 1 \
  --ATSP_solver SA > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log   

data_path_name=weather.csv
model_id_name=weather
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data_path $data_path_name \
  --model $model_name \
  --data $data_name \
  --features $features \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 3 \
  --d_layers 2 \
  --factor 1 \
  --enc_in 21 \
  --c_out 21 \
  --n_heads 16 \
  --d_model 32  \
  --d_ff 16 \
  --dropout 0.3 \
  --learning_rate 0.001 \
  --batch_size 512 \
  --patch_len 48 \
  --stride 48 \
  --VPT_mode 1 \
  --ATSP_solver SA > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log   

data_path_name=electricity.csv
model_id_name=electricity
data_name=custom
torchrun --master_port 12345 --nproc_per_node=$nodes $main_path \
  --random_seed $random_seed \
  --is_training 0 \
  --root_path $root_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data_path $data_path_name \
  --model $model_name \
  --data $data_name \
  --features $features \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 4 \
  --d_layers 2 \
  --factor 1 \
  --enc_in 321 \
  --c_out 321 \
  --n_heads 16 \
  --d_model 128  \
  --d_ff 16 \
  --dropout 0.2 \
  --learning_rate 0.0005 \
  --batch_size 8 \
  --patch_len 48 \
  --stride 48 \
  --VPT_mode 1 \
  --ATSP_solver SA > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log   
done
