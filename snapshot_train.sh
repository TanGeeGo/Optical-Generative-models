#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# activate you conda env if not
# conda activate your_env_name

# run the training
## single gpu start
python main.py \
    --task snapshot_optical \
    --num_gpu 1 \
    --data_path ./data/your_data_path \
    --output_dir ./logs/exp \
    --sample_size 32 \
    --in_channels 1 \
    --out_channels 1 \
    --num_epochs 100 \
    --train_batch_size 100 \
    --eval_batch_size 64 \
    --learning_rate_digital 1e-4 \
    --learning_rate_optical 5e-3 \
    --ddpm_num_steps 1000 \
    --ddpm_beta_schedule linear \
    --mixed_precision no \
    --seed 96 \
    --save_image_epochs 10 \
    --save_model_epochs 50 \
    --lr_warmup_steps 100 \
    --gradient_accumulation_steps 1 \
    --num_classes 10 \
    --c 299792458.0 \
    --ridx_air 1.0 \
    --object_layer_dist 0.05 \
    --layer_layer_dist 0.01 \
    --layer_sensor_dist 0.05 \
    --num_layer 1 \
    --total_num 800 \
    --obj_num 320 \
    --layer_neuron_num 400 \
    --dxdy 8e-6 \
    --layer_init_method zero \
    --amp_modulation False \
    --teacher_ckpt_snst ./ckpt/your_teacher_path \
    --inference_acc_snst True \
    --acc_ratio_snst 20 \
    --wavelength_snst 5.2e-7 \
    --ridx_layer_snst 1.0 \
    --attenu_factor_snst 0.0 \
    --apply_scale_snst True \
    --scale_type_snst neural_pred \
    --noise_perturb_snst 1e-4 \
    --eval_kl_snst True \
    --kl_ratio_snst 1e-4

## multi gpu start
# accelerate launch --multi_gpu --num_processes 4 main.py \
#     --task snapshot_optical \
#     --num_gpu 4 \
#     --data_path ./data/your_data_path \
#     --output_dir ./logs/exp \
#     --sample_size 32 \
#     --in_channels 1 \
#     --out_channels 1 \
#     --num_epochs 100 \
#     --train_batch_size 400 \
#     --eval_batch_size 64 \
#     --learning_rate_digital 1e-4 \
#     --learning_rate_optical 5e-3 \
#     --ddpm_num_steps 1000 \
#     --ddpm_beta_schedule linear \
#     --mixed_precision no \
#     --seed 96 \
#     --save_image_epochs 10 \
#     --save_model_epochs 50 \
#     --lr_warmup_steps 100 \
#     --gradient_accumulation_steps 1 \
#     --num_classes 10 \
#     --c 299792458.0 \
#     --ridx_air 1.0 \
#     --object_layer_dist 0.05 \
#     --layer_layer_dist 0.01 \
#     --layer_sensor_dist 0.05 \
#     --num_layer 1 \
#     --total_num 800 \
#     --obj_num 320 \
#     --layer_neuron_num 400 \
#     --dxdy 8e-6 \
#     --layer_init_method zero \
#     --amp_modulation False \
#     --teacher_ckpt_snst ./ckpt/your_teacher_path \
#     --inference_acc_snst True \
#     --acc_ratio_snst 20 \
#     --wavelength_snst 5.2e-7 \
#     --ridx_layer_snst 1.0 \
#     --attenu_factor_snst 0.0 \
#     --apply_scale_snst True \
#     --scale_type_snst neural_pred \
#     --noise_perturb_snst 1e-4 \
#     --eval_kl_snst True \
#     --kl_ratio_snst 1e-4