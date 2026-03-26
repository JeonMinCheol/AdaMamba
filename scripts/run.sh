#!/usr/bin/bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

main_path=/home/ubuntu216/mincheol/AdaMamba/run_longExp.py
root_path_name=/home/ubuntu216/mincheol/AdaMamba/timeseries_dataset/

features=M
seq_len=336
pred_len=336
random_seed=42

# =================================================================================================================================================================
# default setting
# =================================================================================================================================================================
lambda_trend_loss=${LAMBDA_TREND_LOSS:-1.0}
grad_clip=${GRAD_CLIP:-0.0}
trend_loss_mode=${TREND_LOSS_MODE:-smooth_l1} # smooth_l1, normalized_smooth_l1, diff_smooth_l1, hybrid
trend_target_mode=${TREND_TARGET_MODE:-extractor} # extractor, moving_avg
residual_target_mode=${RESIDUAL_TARGET_MODE:-forecast} # forecast, forecast_detach, teacher
trend_extractor_mode=${TREND_EXTRACTOR_MODE:-sliding_conv} # sliding_conv patch_interp
model_name=AdaMamba
for pred_len in  96 192 336 720; do
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
python $main_path \
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
    --lambda_q_loss 0 \
    --grad_clip $grad_clip \
    --trend_loss_mode $trend_loss_mode \
    --trend_target_mode $trend_target_mode \
    --trend_extractor_mode $trend_extractor_mode \
    --lambda_trend_loss $lambda_trend_loss \
    --residual_target_mode $residual_target_mode \
    --kernels 64 \
    --use_dynamic False \
    --norm_type AdaNorm \
    --use_trend_forecast True \
    --batch_size 128  >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_tlm'$trend_loss_mode'_ttm'$trend_target_mode'_rtm'$residual_target_mode'_tem'$trend_extractor_mode'_gc'$grad_clip.log 

data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
python $main_path \
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
    --batch_size 128 \
    --learning_rate 0.0003 \
    --lambda_h_loss 1 \
    --trend_loss_mode $trend_loss_mode \
    --lambda_trend_loss 0.01 \
    --trend_target_mode $trend_target_mode \
    --residual_target_mode $residual_target_mode \
    --trend_extractor_mode $trend_extractor_mode \
    --lambda_trend_loss $lambda_trend_loss \
    --kernels 84 \
    --use_dynamic False \
    --norm_type AdaNorm \
    --use_trend_forecast True \
    --lambda_q_loss 0.1 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_tlm'$trend_loss_mode'_tl'$lambda_trend_loss'_ttm'$trend_target_mode'_rtm'$residual_target_mode'_tem'$trend_extractor_mode.log 

# data_path_name=ETTm1.csv
# model_id_name=ETTm1
# data_name=ETTm1
# python $main_path \
#     --is_training 1 \
#     --random_seed $random_seed \
#     --root_path $root_path_name \
#     --data_path $data_path_name \
#     --model_id $model_id_name'_'$seq_len'_'$pred_len \
#     --model $model_name \
#     --data $data_name \
#     --features $features \
#     --seq_len $seq_len \
#     --pred_len $pred_len \
#     --enc_in 7 \
#     --dec_in 7 \
#     --n_heads 4 \
#     --patch_len 48 \
#     --stride 48 \
#     --dropout 0.4 \
#     --d_model 128 \
#     --d_ff 256 \
#     --d_head 512 \
#     --batch_size 128 \
#     --learning_rate 0.0003 \
#     --lambda_h_loss 1 \
#     --lambda_q_loss 1 \
#     --trend_target_mode $trend_target_mode \
#     --trend_extractor_mode $trend_extractor_mode \
#     --lambda_trend_loss $lambda_trend_loss \
#     --kernels 24 \
#     --use_dynamic False \
#     --norm_type AdaNorm \
#     --use_trend_forecast True \
#     --batch_size 128  >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_ttm'$trend_target_mode'_tem'$trend_extractor_mode.log 

# data_path_name=ETTm2.csv
# model_id_name=ETTm2
# data_name=ETTm2
# python $main_path \
#     --is_training 1 \
#     --random_seed $random_seed \
#     --root_path $root_path_name \
#     --data_path $data_path_name \
#     --model_id $model_id_name'_'$seq_len'_'$pred_len \
#     --model $model_name \
#     --data $data_name \
#     --features $features \
#     --seq_len $seq_len \
#     --pred_len $pred_len \
#     --enc_in 7 \
#     --dec_in 7 \
#     --n_heads 8 \
#     --patch_len 48 \
#     --stride 48 \
#     --dropout 0.35 \
#     --head_dropout 0.0 \
#     --d_model 512 \
#     --d_ff 128 \
#     --d_head 128 \
#     --batch_size 128 \
#     --lambda_h_loss 1 \
#     --lambda_q_loss 1 \
#     --lambda_trend_loss 0.01 \
#     --trend_target_mode $trend_target_mode \
#     --trend_extractor_mode $trend_extractor_mode \
#     --lambda_trend_loss $lambda_trend_loss \
#     --kernels 24 \
#     --use_dynamic False \
#     --norm_type AdaNorm \
#     --use_trend_forecast True \
#     --learning_rate 0.0002  > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_ttm'$trend_target_mode'_tem'$trend_extractor_mode.log 

# data_path_name=weather.csv
# model_id_name=weather
# data_name=custom
# python $main_path \
#     --is_training 1 \
#     --random_seed $random_seed \
#     --root_path $root_path_name \
#     --data_path $data_path_name \
#     --model_id $model_id_name'_'$seq_len'_'$pred_len \
#     --model $model_name \
#     --data $data_name \
#     --seq_len $seq_len \
#     --pred_len $pred_len \
#     --features $features \
#     --enc_in 21 \
#     --dec_in 21 \
#     --n_heads 2 \
#     --dropout 0.15 \
#     --patch_len 48 \
#     --stride 48 \
#     --lambda_h_loss 1 \
#     --lambda_q_loss 1 \
#     --d_ff 256 \
#     --d_head 512 \
#     --d_model 128 \
#     --learning_rate 0.001 \
#     --lambda_trend_loss $lambda_trend_loss \
#     --trend_target_mode $trend_target_mode \
#     --trend_extractor_mode $trend_extractor_mode \
#     --kernels 24 \
#     --use_dynamic False \
#     --norm_type AdaNorm \
#     --use_trend_forecast True \
#     --batch_size 128  > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_tl'$lambda_trend_loss'_ttm'$trend_target_mode'_tem'$trend_extractor_mode.log 

# data_path_name=exchange_rate.csv
# model_id_name=Exchange
# data_name=custom
# python $main_path \
#     --is_training 1 \
#     --random_seed $random_seed \
#     --root_path $root_path_name \
#     --data_path $data_path_name \
#     --model_id $model_id_name'_'$seq_len'_'$pred_len \
#     --model $model_name \
#     --data $data_name \
#     --seq_len $seq_len \
#     --pred_len $pred_len \
#     --features $features \
#     --enc_in 8 \
#     --dec_in 8 \
#     --n_heads 4 \
#     --patch_len 48 \
#     --stride 48 \
#     --dropout 0.25 \
#     --head_dropout 0.05 \
#     --lambda_h_loss 0.1 \
#     --lambda_q_loss 1.5 \
#     --d_head 1024 \
#     --d_ff 512 \
#     --d_model 128 \
#     --batch_size 8 \
#     --kernels 24 \
#     --norm_type AdaNorm \
#     --use_trend_forecast True \
#     --use_dynamic False \
#     --lambda_trend_loss 0.01 \
#     --trend_target_mode $trend_target_mode \
#     --trend_extractor_mode $trend_extractor_mode \
#     --lambda_trend_loss $lambda_trend_loss \
#     --learning_rate 0.0001 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_ttm'$trend_target_mode'_tem'$trend_extractor_mode.log
done
