#!/bin/bash

recon_path="./result/EEGPT/lora_mask_0.7/wo_label_channel"
lora_r=8
lora_alpha=32
lora_dropout=0.1
cuda_device="cuda:4"

source activate EEGPT
export PYTHONPATH=.:$PYTHONPATH

cls_fitune_types=( "full"  "lora"  "linear" )
for cls_fitune_type in "${cls_fitune_types[@]}";
do
    python3 ./train/run_multitask_stf_cog.py \
        --model EEGPT \
        --abs_pos_emb \
        --seed 42 \
        --disable_qkv_bias \
        --disable_rel_pos_bias \
        --epochs 300 \
        --batch_size 8 \
        --lr 1e-4 \
        --train_size 0.1 \
        --test_size 0.8 \
        --val_size 0.1 \
        --device $cuda_device \
        --recon_results_dir "$recon_path" \
        --h5_path "../data/cogbci/multitask.h5" \
        --num_classes 6 \
        --output_root "$recon_path/Multitask/train_0.1" \
        --target_modules "$recon_target_modules" \
        --lora_r "$lora_r" \
        --lora_alpha "$lora_alpha" \
        --lora_dropout "$lora_dropout" \
        --fitune_type 'lora' \
        --cls_fitune_type $cls_fitune_type \
        --cls_lora_alpha 32 \
        --cls_lora_r 8 \
        --cls_lora_dropout 0.1 \
        --multi_task \
        --multi_task_ids 1,2
done

cls_fitune_types=( "full"  "lora"  "linear" )
for cls_fitune_type in "${cls_fitune_types[@]}";
do
    python3 ./train/run_multitask_stf_cog.py \
        --model EEGPT \
        --abs_pos_emb \
        --seed 42 \
        --disable_qkv_bias \
        --disable_rel_pos_bias \
        --epochs 300 \
        --batch_size 8 \
        --lr 1e-4 \
        --train_size 0.7 \
        --test_size 0.2 \
        --val_size 0.1 \
        --device $cuda_device \
        --recon_results_dir "$recon_path" \
        --h5_path "../data/cogbci/multitask.h5" \
        --num_classes 6 \
        --output_root "$recon_path/Multitask/train_0.7" \
        --target_modules "$recon_target_modules" \
        --lora_r "$lora_r" \
        --lora_alpha "$lora_alpha" \
        --lora_dropout "$lora_dropout" \
        --fitune_type 'lora' \
        --cls_fitune_type $cls_fitune_type \
        --cls_lora_alpha 32 \
        --cls_lora_r 8 \
        --cls_lora_dropout 0.1 \
        --multi_task \
        --multi_task_ids 1,2
done
