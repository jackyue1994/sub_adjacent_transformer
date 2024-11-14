export CUDA_VISIBLE_DEVICES=0
python main.py --anormly_ratio 0.7 --num_epochs 1   --batch_size 256  \
--mode monte_carlo_search --dataset SWaT  --data_path dataset/Swat --input_c 51    --output_c 51 \
--span 20 25 --monte_carlo 1
