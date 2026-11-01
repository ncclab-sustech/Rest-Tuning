export PYTHONPATH=.:$PYTHONPATH

lora_r=8
lora_alpha=32
lora_dropout=0.1
cuda_device="cuda:0"
subject_id=1
recon_path="./result/LaBraM/baseline_continue_pretrain_${subject_id}"

source activate EEGPT
for cls_type in  "full" "lora"
do
    python3 ./train/run_multitask_baseline.py \
        --model labram_base_patch200_200 \
        --dataset "cogbci" \
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
        --h5_path "../data/cogbci/multitask.h5" \
        --pretrained_path "../checkpoints/labram_base_resume_${subject_id}/checkpoint.pth" \
        --num_classes 6 \
        --output_root "$recon_path/Multitask/train_0.1" \
        --cls_fitune_type "$cls_type" \
        --cls_lora_r $lora_r \
        --cls_lora_alpha $lora_alpha \
        --cls_lora_dropout $lora_dropout \
        --multi_task \
        --multi_task_ids 1 2
done
