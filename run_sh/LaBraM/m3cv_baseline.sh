export PYTHONPATH=.:$PYTHONPATH

lora_r=8
lora_alpha=32
lora_dropout=0.1
cuda_device="cuda:0"
recon_path="./result/LaBraM/baseline"

source activate rest-tuning

for cls_type in  "lora" "linear" "full"
do
    python3 ./train/run_multitask_baseline.py \
        --model labram_base_patch200_200 \
        --dataset "m3cv" \
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
        --h5_path "../data/m3cv/multitask.h5" \
        --pretrained_path '../checkpoints/labram-base.pth' \
        --num_classes 11 \
        --output_root "$recon_path/Multitask/train_0.7" \
        --cls_fitune_type "$cls_type" \
        --cls_lora_r $lora_r \
        --cls_lora_alpha $lora_alpha \
        --cls_lora_dropout $lora_dropout \
        --multi_task \
        --multi_task_ids 1 2 3 4
done

for cls_type in "lora" "full" "linear"
do
    python3 ./train/run_task_sft_cls.py \
        --model labram_base_patch200_200 \
        --dataset "m3cv" \
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
        --h5_path "../data/m3cv/motor.h5" \
        --pretrained_path '../checkpoints/labram-base.pth' \
        --num_classes 3 \
        --output_root "$recon_path/classification/Motor/train_0.1" \
        --dataset_suffle \
        --cls_fitune_type "$cls_type" \
        --cls_lora_r $lora_r \
        --cls_lora_alpha $lora_alpha \
        --cls_lora_dropout $lora_dropout \
        --dataset_suffle
done

for cls_type in "lora" "linear" "full"
do
    python3 ./train/run_baseline_stf.py \
        --model labram_base_patch200_200 \
        --dataset "m3cv" \
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
        --h5_path "../data/m3cv/motor.h5" \
        --pretrained_path '../checkpoints/labram-base.pth' \
        --num_classes 3 \
        --output_root "$recon_path/classification/Motor/train_0.7" \
        --dataset_suffle \
        --cls_fitune_type "$cls_type" \
        --cls_lora_r $lora_r \
        --cls_lora_alpha $lora_alpha \
        --cls_lora_dropout $lora_dropout \
        --dataset_suffle
done

for cls_type in "lora" "full" "linear"
do
    python3 ./train/run_task_sft_cls.py \
        --model labram_base_patch200_200 \
        --dataset "m3cv" \
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
        --h5_path "../data/m3cv/p300.h5" \
        --pretrained_path '../checkpoints/labram-base.pth' \
        --num_classes 2 \
        --output_root "$recon_path/classification/P300/train_0.1" \
        --dataset_suffle \
        --cls_fitune_type "$cls_type" \
        --cls_lora_r $lora_r \
        --cls_lora_alpha $lora_alpha \
        --cls_lora_dropout $lora_dropout \
        --dataset_suffle
done

for cls_type in "lora" "linear" "full"
do
    python3 ./train/run_baseline_stf.py \
        --model labram_base_patch200_200 \
        --dataset "m3cv" \
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
        --h5_path "../data/m3cv/p300.h5" \
        --pretrained_path '../checkpoints/labram-base.pth' \
        --num_classes 2 \
        --output_root "$recon_path/classification/P300/train_0.7" \
        --dataset_suffle \
        --cls_fitune_type "$cls_type" \
        --cls_lora_r $lora_r \
        --cls_lora_alpha $lora_alpha \
        --cls_lora_dropout $lora_dropout \
        --dataset_suffle
done

for cls_type in "lora" "full" "linear"
do
    python3 ./train/run_task_sft_cls.py \
        --model labram_base_patch200_200 \
        --dataset "m3cv" \
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
        --h5_path "../data/m3cv/ts.h5" \
        --pretrained_path '../checkpoints/labram-base.pth' \
        --num_classes 3 \
        --output_root "$recon_path/classification/TS/train_0.1" \
        --dataset_suffle \
        --cls_fitune_type "$cls_type" \
        --cls_lora_r $lora_r \
        --cls_lora_alpha $lora_alpha \
        --cls_lora_dropout $lora_dropout \
        --dataset_suffle
done

for cls_type in "lora" "linear" "full"
do
    python3 ./train/run_baseline_stf.py \
        --model labram_base_patch200_200 \
        --dataset "m3cv" \
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
        --h5_path "../data/m3cv/ts.h5" \
        --pretrained_path '../checkpoints/labram-base.pth' \
        --num_classes 3 \
        --output_root "$recon_path/classification/TS/train_0.7" \
        --dataset_suffle \
        --cls_fitune_type "$cls_type" \
        --cls_lora_r $lora_r \
        --cls_lora_alpha $lora_alpha \
        --cls_lora_dropout $lora_dropout \
        --dataset_suffle
done
