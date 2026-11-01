#!/bin/bash

recon_path="./result/LaBraM/full_mask_0.3/wo_label_channel_0418"
lora_r=8
lora_alpha=32
lora_dropout=0.1
cuda_device="cuda:2"

source activate EEGPT
export PYTHONPATH=.:$PYTHONPATH

python3 ./train/run_rest_tune_with_dynamics.py \
    --model labram_base_patch200_200 \
    --pretrained_path ../checkpoints/labram-base.pth \
    --h5_path "../data/cogbci/rest.h5" \
    --dataset 'cogbci' \
    --output_root "$recon_path/test" \
    --epochs 300 \
    --batch_size 32 \
    --min_lr 1e-6 \
    --weight_decay 0.01 \
    --warmup_epochs 10 \
    --lr 1e-4 \
    --sgd_lr 0.1 \
    --patience 40 \
    --mse_weight 0.1 \
    --var_weight 1 \
    --device $cuda_device \
    --num_workers 4 \
    --train_size 0.85 \
    --test_size 0.05 \
    --val_size 0.1 \
    --use_amp \
    --abs_pos_emb \
    --seed 42 \
    --disable_qkv_bias \
    --disable_rel_pos_bias \
    --fitune_type 'full' \
    --lora_r "$lora_r" \
    --lora_alpha "$lora_alpha" \
    --lora_dropout "$lora_dropout" \
    --target_modules "$recon_target_modules" \
    --dataset_suffle \
    --mask_type 'channel' \
    --mask_ratio 0.3 \
    --hidden_dim 2 \
    --head_dropout 0.1 \
    --dropout 0.1 \
    --contrastive_task_temperature 0.1 \
    --contrastive_task_alpha 0.1 \
    --embedding_high_level

cls_fitune_types=( "full" "linear"  "lora")
for cls_fitune_type in "${cls_fitune_types[@]}";
do
    python3 ./train/run_task_sft_cls.py \
        --model labram_base_patch200_200 \
        --dataset "cogbci" \
        --abs_pos_emb \
        --seed 42 \
        --disable_qkv_bias \
        --disable_rel_pos_bias \
        --epochs 200 \
        --batch_size 8 \
        --lr 1e-4 \
        --train_size 0.1 \
        --test_size 0.8 \
        --val_size 0.1 \
        --device $cuda_device \
        --recon_results_dir "$recon_path" \
        --h5_path "../data/cogbci/matb.h5" \
        --num_classes 3 \
        --output_root "$recon_path/classification/MATB/train_0.1_b8" \
        --target_modules "$recon_target_modules" \
        --lora_r "$lora_r" \
        --lora_alpha "$lora_alpha" \
        --lora_dropout "$lora_dropout" \
        --fitune_type 'full' \
        --cls_fitune_type $cls_fitune_type \
        --cls_lora_alpha 32 \
        --cls_lora_r 8 \
        --cls_lora_dropout 0.1 \
        --dataset_suffle
done

cls_fitune_types=( "full" "linear"  "lora")
for cls_fitune_type in "${cls_fitune_types[@]}";
do
    python3 ./train/run_task_sft_cls.py \
        --model labram_base_patch200_200 \
        --dataset "cogbci" \
        --abs_pos_emb \
        --seed 42 \
        --disable_qkv_bias \
        --disable_rel_pos_bias \
        --epochs 200 \
        --batch_size 8 \
        --lr 1e-4 \
        --train_size 0.7 \
        --test_size 0.2 \
        --val_size 0.1 \
        --device $cuda_device \
        --recon_results_dir "$recon_path" \
        --h5_path "../data/cogbci/matb.h5" \
        --num_classes 3 \
        --output_root "$recon_path/classification/MATB/train_0.7_b8" \
        --target_modules "$recon_target_modules" \
        --lora_r "$lora_r" \
        --lora_alpha "$lora_alpha" \
        --lora_dropout "$lora_dropout" \
        --fitune_type 'full' \
        --cls_fitune_type $cls_fitune_type \
        --cls_lora_alpha 32 \
        --cls_lora_r 8 \
        --cls_lora_dropout 0.1 \
        --dataset_suffle
done

cls_fitune_types=( "full" "linear"  "lora")
for cls_fitune_type in "${cls_fitune_types[@]}";
do
    python3 ./train/run_task_sft_cls.py \
        --model labram_base_patch200_200 \
        --dataset "cogbci" \
        --abs_pos_emb \
        --seed 42 \
        --disable_qkv_bias \
        --disable_rel_pos_bias \
        --epochs 200 \
        --batch_size 8 \
        --lr 1e-4 \
        --train_size 0.1 \
        --test_size 0.8 \
        --val_size 0.1 \
        --device $cuda_device \
        --recon_results_dir "$recon_path" \
        --h5_path "../data/cogbci/nback.h5" \
        --num_classes 3 \
        --output_root "$recon_path/classification/nback/train_0.1_b8" \
        --target_modules "$recon_target_modules" \
        --lora_r "$lora_r" \
        --lora_alpha "$lora_alpha" \
        --lora_dropout "$lora_dropout" \
        --fitune_type 'full' \
        --cls_fitune_type $cls_fitune_type \
        --cls_lora_alpha 32 \
        --cls_lora_r 8 \
        --cls_lora_dropout 0.1 \
        --dataset_suffle
done

cls_fitune_types=( "full" "linear"  "lora")
for cls_fitune_type in "${cls_fitune_types[@]}";
do
    python3 ./train/run_task_sft_cls.py \
        --model labram_base_patch200_200 \
        --dataset "cogbci" \
        --abs_pos_emb \
        --seed 42 \
        --disable_qkv_bias \
        --disable_rel_pos_bias \
        --epochs 200 \
        --batch_size 8 \
        --lr 1e-4 \
        --train_size 0.7 \
        --test_size 0.2 \
        --val_size 0.1 \
        --device $cuda_device \
        --recon_results_dir "$recon_path" \
        --h5_path "../data/cogbci/nback.h5" \
        --num_classes 3 \
        --output_root "$recon_path/classification/nback/train_0.7_b8" \
        --target_modules "$recon_target_modules" \
        --lora_r "$lora_r" \
        --lora_alpha "$lora_alpha" \
        --lora_dropout "$lora_dropout" \
        --fitune_type 'full' \
        --cls_fitune_type $cls_fitune_type \
        --cls_lora_alpha 32 \
        --cls_lora_r 8 \
        --cls_lora_dropout 0.1 \
        --dataset_suffle
done
