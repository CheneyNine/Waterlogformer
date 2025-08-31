model_name=MultiPatchFormer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Flood/ \
  --data_path rain7_flood.csv \
  --model_id rain_7flood_24_24_MultiPatchFormer \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 24 \
  --label_len 24 \
  --pred_len 24 \
  --e_layers 1 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 256 \
  --d_ff 512 \
  --des 'Exp' \
  --n_heads 8 \
  --batch_size 4 \
  --itr 1