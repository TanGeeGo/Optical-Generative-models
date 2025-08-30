#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# activate you conda env if not
# conda activate your_env_name

# run the training
## single gpu start
python main.py \
    --task diffusion_digital \
    --num_gpu 1 \
    --data_path ./data/your_datapath \
    --output_dir ./logs/exp \
    --sample_size 32 \
    --in_channels 3 \
    --out_channels 3 \
    --num_epochs 300 \
    --train_batch_size 200 \
    --eval_batch_size 64 \
    --learning_rate_digital 1e-4 \
    --prediction_type epsilon \
    --ddpm_num_steps 1000 \
    --ddpm_beta_schedule linear \
    --mixed_precision no \
    --seed 96 \
    --save_image_epochs 10 \
    --save_model_epochs 50 \
    --lr_warmup_steps 100 \
    --gradient_accumulation_steps 1 \
    --num_classes 0 \
    --time_embedding_type_d positional \
    --down_block_types DownBlock2D AttnDownBlock2D AttnDownBlock2D AttnDownBlock2D \
    --up_block_types AttnUpBlock2D AttnUpBlock2D AttnUpBlock2D UpBlock2D \
    --block_out_channels 224 448 672 896 \
    --layers_per_block 2 \
    --prediction_type_d epsilon

## multi gpu start
# accelerate launch --multi_gpu --num_processes 4 main.py \
#     --task diffusion_digital \
#     --num_gpu 4 \
#     --data_path ./data/your_datapath \
#     --output_dir ./logs/exp \
#     --sample_size 32 \
#     --in_channels 1 \
#     --out_channels 1 \
#     --num_epochs 300 \
#     --train_batch_size 4 \
#     --eval_batch_size 4 \
#     --learning_rate_digital 1e-4 \
#     --prediction_type epsilon \
#     --ddpm_num_steps 1000 \
#     --ddpm_beta_schedule linear \
#     --mixed_precision no \
#     --seed 96 \
#     --save_image_epochs 10 \
#     --save_model_epochs 50 \
#     --lr_warmup_steps 100 \
#     --gradient_accumulation_steps 1 \
#     --num_classes 10 \
#     --time_embedding_type_d positional \
#     --down_block_types DownBlock2D AttnDownBlock2D AttnDownBlock2D AttnDownBlock2D \
#     --up_block_types AttnUpBlock2D AttnUpBlock2D AttnUpBlock2D UpBlock2D \
#     --block_out_channels 224 448 672 896 \
#     --layers_per_block 2 \
#     --prediction_type_d epsilon
