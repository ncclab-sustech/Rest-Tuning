#!/bin/bash

recon_path="./result/LaBraM/lora_mask_0.7/wo_label_channel_0418"
lora_r=8
lora_alpha=32
lora_dropout=0.1
cuda_device="cuda:6"

source activate rest-tuning
export PYTHONPATH=.:$PYTHONPATH

cls_fitune_types=( "lora" "full" "linear"  )
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
        --fitune_type 'lora' \
        --cls_fitune_type $cls_fitune_type \
        --cls_lora_alpha 32 \
        --cls_lora_r 8 \
        --cls_lora_dropout 0.1 \
        --dataset_suffle
done

cls_fitune_types=( "lora" "full" "linear"  )
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
        --fitune_type 'lora' \
        --cls_fitune_type $cls_fitune_type \
        --cls_lora_alpha 32 \
        --cls_lora_r 8 \
        --cls_lora_dropout 0.1 \
        --dataset_suffle
done

export PYTHONPATH=.
cls_fitune_types=(  "lora" "full" "linear" )
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
        --fitune_type 'lora' \
        --cls_fitune_type $cls_fitune_type \
        --cls_lora_alpha 32 \
        --cls_lora_r 8 \
        --cls_lora_dropout 0.1 \
        --dataset_suffle
done

export PYTHONPATH=.
cls_fitune_types=("lora" "full" "linear"  )
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
        --fitune_type 'lora' \
        --cls_fitune_type $cls_fitune_type \
        --cls_lora_alpha 32 \
        --cls_lora_r 8 \
        --cls_lora_dropout 0.1 \
        --dataset_suffle
done
