export CUDA_VISIBLE_DEVICES=0

model_name=Autoformer
python -u run.py \
  --batch_size 4 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Flood/ \
  --data_path rain7_flood.csv \
  --model_id rain_7flood_24_24_Autoformer \
  --model $model_name \
  --data custom\
  --features M \
  --seq_len 24 \
  --label_len 24 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --n_heads 4 \
  --use_future_rain True \
  --train_epochs 10
  


model_name=Crossformer
python -u run.py \
  --batch_size 4 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Flood/ \
  --data_path rain7_flood.csv \
  --model_id rain_7flood_24_24_Crossformer \
  --model $model_name \
  --data custom\
  --features M \
  --seq_len 24 \
  --label_len 24 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --n_heads 4 \
  --use_future_rain True \
  --train_epochs 10 


model_name=iTransformer
python -u run.py \
  --batch_size 4 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Flood/ \
  --data_path rain7_flood.csv \
  --model_id rain_7flood_24_24_iTransformer \
  --model $model_name \
  --data custom\
  --features M \
  --seq_len 24 \
  --label_len 24 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --n_heads 4 \
  --use_future_rain True \
  --train_epochs 10 





model_name=Nonstationary_Transformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Flood/ \
  --data_path rain7_flood.csv \
  --model_id rain_7flood_24_24_Nonstationary_Transformer \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 24 \
  --label_len 24 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 10 \
  --p_hidden_dims 256 256 \
  --p_hidden_layers 2

model_name=SegRNN

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Flood/ \
  --data_path rain7_flood.csv \
  --model_id rain_7flood_24_24_SegRNN \
  --model $model_name \
  --data custom  \
  --batch_size 4 \
  --features M \
  --seq_len 24 \
  --pred_len 24 \
  --label_len 24 \
  --seg_len 24 \
  --enc_in 1 \
  --d_model 512 \
  --dropout 0.5 \
  --learning_rate 0.0001 \
  --des 'Exp' \
  --itr 1


model_name=TimesNet

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Flood/ \
  --data_path rain7_flood.csv \
  --model_id rain_7flood_24_24_TimesNet \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 24 \
  --label_len 24 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 512 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1


model_name=Transformer

python -u run.py \
  --batch_size 4 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Flood/ \
  --data_path rain7_flood.csv \
  --model_id rain_7flood_24_24_Transformer \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 24 \
  --label_len 24 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --n_heads 4 \
  --use_future_rain True \
  --train_epochs 10


model_name=TSMixer
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Flood/ \
  --data_path rain7_flood.csv \
  --model_id rain_7flood_24_24_TSMixer \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 24 \
  --label_len 24 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 512 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' 


model_name=LSTM

python -u run.py \
  --batch_size 4 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Flood/ \
  --data_path rain7_flood.csv \
  --model_id rain_7flood_24_24_LSTM \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 24 \
  --label_len 24 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --n_heads 4 \
  --use_future_rain True \
  --train_epochs 10




model_name=RNN

python -u run.py \
  --batch_size 4 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Flood/ \
  --data_path rain7_flood.csv \
  --model_id rain_7flood_24_24_RNN \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 24 \
  --label_len 24 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --n_heads 4 \
  --use_future_rain True \
  --train_epochs 10





model_name=ARIMAModel

python -u run.py \
  --batch_size 4 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Flood/ \
  --data_path rain7_flood.csv \
  --model_id rain_7flood_24_24_ARIMAModel \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 24 \
  --label_len 24 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --n_heads 4 \
  --use_future_rain True \
  --train_epochs 10