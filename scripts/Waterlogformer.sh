export CUDA_VISIBLE_DEVICES=0

model_name=Waterlogformer

python -u run.py \
  --batch_size 4 \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/Flood/ \
  --data_path rain7_flood.csv \
  --model_id rain_7flood_12_12_Project \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 12 \
  --label_len 12 \
  --pred_len 12 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --n_heads 4 \
  --train_epochs 15 \
  --lambda_contrast 0.5 \
  --sharpness 1e-5 \
  --without_static 0 \
  --without_rain 0 \
  --without_gate 0 \
  --without_propagation 0 \
  --without_cls 0 \
  --without_contrastive 0 \
  --without_future_rain 0\
  --use_real_rain 1
