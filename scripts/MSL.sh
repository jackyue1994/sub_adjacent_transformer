export CUDA_VISIBLE_DEVICES=7

train:
python main.py --anormly_ratio 0.5 --num_epochs 1  --batch_size 128  --mode monte_carlo_span --dataset MSL_pku  --data_path dataset/MSL_pku --input_c 55   --output_c 55 --monte_carlo 10 --win_size 100 --softmax_span_range 100 200 --softmax_span_step 10 --no_gauss_dynamic

test:
python main.py --anormly_ratio 0.5 --num_epochs 1  --batch_size 128  --mode find_best --dataset MSL_pku  --data_path dataset/MSL_pku --input_c 55   --output_c 55 --monte_carlo 10 --win_size 100 --softmax_span_range 120 120 --softmax_span_step 10 --temperature_range 30 60 --temperature_step 1 --anormly_ratio_span 0.42 0.42 --anormly_ratio_step 0.01
