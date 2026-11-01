#!/bin/bash

recon_path="./result/EEGPT/full_mask_0.3/wo_label_channel"
lora_r=8
lora_alpha=32
lora_dropout=0.1
cuda_device="cuda:3"

source activate EEGPT
export PYTHONPATH=.:$PYTHONPATH

cls_fitune_types=( "full" "linear"  "lora")
for cls_fitune_type in "${cls_fitune_types[@]}";
do
    python3 ./train/run_task_sft_cls.py \
        --model "EEGPT" \
        --dataset "m3cv" \
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
        --h5_path "../data/m3cv/motor.h5" \
        --num_classes 3 \
        --output_root "$recon_path/classification/Motor/train_0.1_b8" \
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
        --model "EEGPT" \
        --dataset "m3cv" \
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
        --h5_path "../data/m3cv/motor.h5" \
        --num_classes 3 \
        --output_root "$recon_path/classification/Motor/train_0.7_b8" \
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
        --model "EEGPT" \
        --dataset "m3cv" \
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
        --h5_path "../data/m3cv/p300.h5" \
        --num_classes 2 \
        --output_root "$recon_path/classification/P300/train_0.1_b8" \
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
        --model "EEGPT" \
        --dataset "m3cv" \
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
        --h5_path "../data/m3cv/p300.h5" \
        --num_classes 2 \
        --output_root "$recon_path/classification/P300/train_0.7_b8" \
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
        --model "EEGPT" \
        --dataset "m3cv" \
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
        --h5_path "../data/m3cv/sss.h5" \
        --num_classes 3 \
        --output_root "$recon_path/classification/SSS/train_0.1_b8" \
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
        --model "EEGPT" \
        --dataset "m3cv" \
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
        --h5_path "../data/m3cv/sss.h5" \
        --num_classes 3 \
        --output_root "$recon_path/classification/SSS/train_0.7_b8" \
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
        --model "EEGPT" \
        --dataset "m3cv" \
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
        --h5_path "../data/m3cv/ts.h5" \
        --num_classes 3 \
        --output_root "$recon_path/classification/TS/train_0.1_b8" \
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
        --model "EEGPT" \
        --dataset "m3cv" \
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
        --h5_path "../data/m3cv/ts.h5" \
        --num_classes 3 \
        --output_root "$recon_path/classification/TS/train_0.7_b8" \
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
