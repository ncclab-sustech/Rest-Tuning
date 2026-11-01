#!/bin/bash

recon_path="./result/CBraMod/lora_mask_0.7/wo_label_channel"
lora_r=8
lora_alpha=32
lora_dropout=0.1
cuda_device="cuda:3"

source activate EEGPT
export PYTHONPATH=.:$PYTHONPATH

cls_fitune_types=( "linear"  "lora")
for cls_fitune_type in "${cls_fitune_types[@]}";
do
    python3 ./train/run_task_sft_cls.py \
        --model "CBraMod" \
        --dataset "m3cv" \
        --abs_pos_emb \
        --seed 68 \
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
        --h5_path "../data/m3cv/ts.h5" \
        --num_classes 3 \
        --output_root "$recon_path/classification/TS/train_0.7_0502" \
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
