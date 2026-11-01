import argparse
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import json
import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from collections import OrderedDict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from peft import get_peft_model, LoraConfig
from src.model import *
from src.utils import *
from timm.models import create_model
from src.data_loader import *
from torch.cuda.amp import GradScaler, autocast
import src.modeling_finetune
from src.cbramod import CBraMod
from src.finetune_trainer import apply_lora_to_model


def load_pretrained_compatible(model, pretrained_path, map_location,
                               wrapper_prefixes=('module.',), skip_prefixes=(), tag=''):
    ckpt = torch.load(pretrained_path, map_location=map_location, weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt and isinstance(ckpt['model'], dict):
        state = ckpt['model']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
        state = ckpt['state_dict']
    else:
        state = ckpt

    model_state = model.state_dict()

    def _resolve_key(k):
        if k in model_state:
            return k
        for p1 in wrapper_prefixes:
            if k.startswith(p1):
                k1 = k[len(p1):]
                if k1 in model_state:
                    return k1
                for p2 in wrapper_prefixes:
                    if k1.startswith(p2) and k1[len(p2):] in model_state:
                        return k1[len(p2):]
        return None

    matched = {}
    n_skipped_branch = 0
    for k, v in state.items():
        if any(k.startswith(sp) for sp in skip_prefixes):
            n_skipped_branch += 1
            continue
        rk = _resolve_key(k)
        if rk is not None and rk not in matched and model_state[rk].shape == v.shape:
            matched[rk] = v
    missing = [k for k in model_state if k not in matched]
    model.load_state_dict(matched, strict=False)
    skipped = len(state) - len(matched) - n_skipped_branch
    print(f"{tag} pretrained: loaded {len(matched)}/{len(model_state)} keys, "
          f"skipped {skipped} unexpected + {n_skipped_branch} pretrain-only, "
          f"{len(missing)} missing (random init)")
    if missing:
        print(f"  Missing keys (first 5): {missing[:5]}")
    return model


def load_cbramod_pretrained(model, pretrained_path, map_location):
    return load_pretrained_compatible(
        model, pretrained_path, map_location,
        wrapper_prefixes=('module.', 'encoder.'), tag='CBraMod')


def load_biot_pretrained(model, pretrained_path, map_location):
    return load_pretrained_compatible(
        model, pretrained_path, map_location,
        wrapper_prefixes=('module.', 'biot.'), skip_prefixes=('prediction.',), tag='BIOT')


def load_pretrained_encoder(encoder, checkpoint_path, method, device):
    """
    Load pretrained encoder from reconstruction checkpoint

    Args:
        encoder: LaBraM encoder model
        checkpoint_path: Path to checkpoint
        method: 'full', 'sgd', 'lora', or 'resnet'
        device: Device to load on

    Returns:
        Loaded encoder
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if method in ['full', 'sgd','lora']:

        if 'encoder_state_dict' in checkpoint:
            state_dict = checkpoint['encoder_state_dict']

    elif method == 'resnet':

        if 'encoder_state_dict' in checkpoint:
            state_dict = checkpoint['encoder_state_dict']
        elif 'model_state_dict' in checkpoint:
            full_state = checkpoint['model_state_dict']
            state_dict = OrderedDict()
            for key, value in full_state.items():
                if key.startswith('encoder.'):
                    new_key = key.replace('encoder.', '')
                    state_dict[new_key] = value
        else:
            state_dict = checkpoint

    else:
        raise ValueError(f"Unknown method: {method}")

    missing_keys, unexpected_keys = encoder.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"  Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"  Unexpected keys: {len(unexpected_keys)}")

    return encoder


def train_single_subject_cls_new(args, subject_id, device, num_patches, dataset):
    """Train a subject classifier with the selected task adaptation method"""
    try:
        valid_fitune_types = ['lora', 'full']
        valid_cls_fitune_types = ['lora', 'full', 'linear']
        if args.fitune_type not in valid_fitune_types:
            raise ValueError(f"fitune_type must be one of {valid_fitune_types}")
        if args.cls_fitune_type not in valid_cls_fitune_types:
            raise ValueError(f"cls_fitune_type must be one of {valid_cls_fitune_types}")

        output_path = os.path.join(args.output_root, args.dataset,  args.cls_fitune_type)
        subject_dir = os.path.join(output_path, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)

        metrics_path = os.path.join(subject_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            print(f"  ✓ Subject {subject_id:02d} already trained, skipping...")
            with open(metrics_path, 'r') as f:
                return json.load(f)

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Loading Data (Cognitive Task)")
        print(f"{'='*80}")

        if args.fewshot:
            train_loader, val_loader, test_loader= create_single_subject_few_shot_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                n_shot = args.n_shot,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers,
                val_size=args.val_size,
                oversample=args.oversample
            )
        elif args.dataset_suffle:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
        else:

            train_loader, val_loader, test_loader = create_single_subject_dataloaders_order(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )

        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

        if train_size < args.min_samples_per_class:
            raise ValueError(f"Insufficient training samples: {train_size}")

        if args.dataset=='cogbci':
            in_chans, ch_names, chan_number = get_cogbci_dataset(args, subject_id)
            print('in_chans',in_chans)
        else:
            in_chans, ch_names, chan_number = get_dataset_name(args)
            print('in_chans', in_chans)

        checkpoint_path = os.path.join(args.recon_results_dir, args.dataset, f'subject_{subject_id:02d}','best_model.pth')
        print(f"  Checkpoint: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Pretrained encoder checkpoint not found: {checkpoint_path}")

        print(f"\n[Step 1] Creating Encoder ({args.model})...")
        if args.model =='labram_base_patch200_200':
            encoder = create_model(
                args.model,
                pretrained=False,
                use_mean_pooling=args.use_mean_pooling,
                init_scale=args.init_scale,
                use_rel_pos_bias=args.rel_pos_bias,
                use_abs_pos_emb=args.abs_pos_emb,
                init_values=args.layer_scale_init_value,
                qkv_bias=args.qkv_bias,
                num_patches_per_channel_input=num_patches).to(device)

            if args.fitune_type =='lora':
                encoder_with_lora = create_lora(encoder,args)
                encoder = load_pretrained_encoder(encoder_with_lora, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False

            elif args.fitune_type =='full':
                encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    encoder = create_lora_cls(encoder,args)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True

                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False

        elif args.model =='CBraMod':
            encoder = CBraMod().to(device)
            if args.fitune_type =='lora':
                encoder = apply_lora_to_model(encoder, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
                checkpoint = torch.load(checkpoint_path)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])

                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                elif args.cls_fitune_type =='full':

                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':

                    for name, param in encoder.named_parameters():
                        param.requires_grad = False

            elif args.fitune_type =='full':

                checkpoint = torch.load(checkpoint_path)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                if args.cls_fitune_type =='lora':
                    encoder = apply_lora_to_model(encoder, r=args.cls_lora_r, alpha=args.cls_lora_alpha, dropout=args.cls_lora_dropout)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                elif args.cls_fitune_type =='full':

                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':

                    for name, param in encoder.named_parameters():
                        param.requires_grad = False

        if args.model =='BIOT':
            eeg_projector = BIOT_DataProjector()
            from src.biot import BIOTEncoder
            encoder = BIOTEncoder(emb_size=256, heads=8, depth=4,n_channels=18).to(device)

            if args.fitune_type =='lora':
                encoder = create_lora(encoder,args)
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        if param.is_floating_point():
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False

            elif args.fitune_type =='full':
                encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    encoder = create_lora_cls(encoder,args)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True

                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        if param.is_floating_point():
                            param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False

        if args.model =='BIOT':
            model = SimpleClassifier_BIOT(
                encoder=encoder,
                args=args,
                ch_names=ch_names,
                num_classes=args.num_classes
            ).to(device)
        else:
            model = SimpleClassifier_multi_model(
                encoder=encoder,
                num_channels= chan_number,
                in_chans = in_chans,
                args = args,
                num_classes=args.num_classes,
                embed_dim=args.embed_dim
            ).to(device)

        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Trainable parameter: {name}")
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"\n  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if args.use_amp else None

        best_val_acc = float("-inf")
        best_val_f1 = float("-inf")
        patience_counter = 0
        total_time_start = time.time()
        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Training task adaptation")
        print(f"Method: {args.cls_fitune_type.upper()}")
        print(f"{'='*80}")

        train_start = time.time()
        for epoch in range(args.epochs):

            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            model.train()
            train_loss = 0

            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    eeg, labels = batch
                else:
                    eeg = batch['eeg']
                    labels = batch['label']

                eeg = eeg.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                if args.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(eeg)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(eeg)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            val_loss, val_acc, val_f1, val_cm = evaluate(
                model, val_loader, criterion, device, args.use_amp
            )

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
                      f"TrainLoss={avg_train_loss:.4f}, "
                      f"ValAcc={val_acc:.4f}, ValF1={val_f1:.4f}, "
                      f"LR={current_lr:.6f}")

            if val_f1 > best_val_f1:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
        train_time = time.time() - train_start

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Final Test Evaluation")
        print(f"{'='*80}")

        checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_start = time.time()
        test_loss, test_acc, test_f1, test_cm = evaluate(
            model, test_loader, criterion, device, args.use_amp
        )
        test_inference_time = time.time() - test_start
        total_time =time.time() - total_time_start

        print(f"  Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
        print(f"  Confusion Matrix:\n{test_cm}")

        metrics = {
            'subject_id': int(subject_id),
            'method': args.cls_fitune_type,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'best_val_acc': float(best_val_acc),
            'best_val_f1': float(best_val_f1),
            'test_acc': float(test_acc),
            'test_f1': float(test_f1),
            'test_cm': test_cm.tolist(),
            'best_epoch': int(checkpoint['epoch']) + 1,
            'transfer_method': 'linear_probing',
            'source_task': 'resting_reconstruction',
            'target_task': 'cognitive_classification',
            'train_time_sec': float(train_time),
            'test_time_sec': float(test_inference_time),
            'total_time_sec': float(total_time),
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"  ✓ Results saved to {subject_dir}")

        del model
        del optimizer
        del encoder
        torch.cuda.empty_cache()

        return metrics

    except Exception as e:
        print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
        import traceback
        traceback.print_exc()

        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

        return None

import torch.nn.utils as utils
import time


def train_single_subject_cls_multi_task(args, subject_id, device, num_patches, dataset):
    """Train a subject classifier with the selected task adaptation method"""
    label_mapping = {
        10: 0,
        11: 1,
        12: 2,
        20: 3,
        21: 4,
        30: 5,
        31: 6,
        32: 7,
        40: 8,
        41: 9,
        42: 10
    }
    try:
        valid_fitune_types = ['lora', 'full']
        valid_cls_fitune_types = ['lora', 'full', 'linear']
        if args.fitune_type not in valid_fitune_types:
            raise ValueError(f"fitune_type must be one of {valid_fitune_types}")
        if args.cls_fitune_type not in valid_cls_fitune_types:
            raise ValueError(f"cls_fitune_type must be one of {valid_cls_fitune_types}")

        output_path = os.path.join(args.output_root, args.dataset,  args.cls_fitune_type)
        subject_dir = os.path.join(output_path, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)

        metrics_path = os.path.join(subject_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            print(f"  ✓ Subject {subject_id:02d} already trained, skipping...")
            with open(metrics_path, 'r') as f:
                return json.load(f)

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Loading Data")
        print(f"{'='*80}")

        if args.multi_task:
            train_loader, val_loader, test_loader= create_multitask_single_subject_dataloaders(
                dataset= dataset,
                target_subject_id=subject_id,
                target_task_id=[1,2,3,4],
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
        elif args.multitask_fewshot:
            train_loader, val_loader, test_loader= create_multitask_single_subject_few_shot_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                target_task_id=[1,2,3,4],
                n_shot = args.n_shot,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers,
                val_size=args.val_size,
                oversample=args.oversample
            )
        elif args.fewshot:
            train_loader, val_loader, test_loader= create_single_subject_few_shot_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                n_shot = args.n_shot,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers,
                val_size=args.val_size,
                oversample=args.oversample
            )
        elif args.dataset_suffle:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )

        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

        if train_size < args.min_samples_per_class:
            raise ValueError(f"Insufficient training samples: {train_size}")

        if args.dataset=='cogbci':
            in_chans, ch_names, chan_number = get_cogbci_dataset(args, subject_id)
            print('in_chans',in_chans)
        else:
            in_chans, ch_names, chan_number = get_dataset_name(args)
            print('in_chans', in_chans)

        checkpoint_path = os.path.join(args.recon_results_dir, args.dataset, f'subject_{subject_id:02d}','best_model.pth')
        print(f"  Checkpoint: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Pretrained encoder checkpoint not found: {checkpoint_path}")

        print(f"\n[Step 1] Creating Encoder ({args.model})...")

        if args.model =='labram_base_patch200_200':
            encoder = create_model(
                args.model,
                pretrained=False,
                use_mean_pooling=args.use_mean_pooling,
                init_scale=args.init_scale,
                use_rel_pos_bias=args.rel_pos_bias,
                use_abs_pos_emb=args.abs_pos_emb,
                init_values=args.layer_scale_init_value,
                qkv_bias=args.qkv_bias,
                num_patches_per_channel_input=num_patches).to(device)

            if args.fitune_type =='lora':
                encoder_with_lora = create_lora(encoder,args)
                encoder = load_pretrained_encoder(encoder_with_lora, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False

            elif args.fitune_type =='full':
                encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    encoder = create_lora_cls(encoder,args)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True

                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False

        elif args.model =='CBraMod':
            encoder = CBraMod().to(device)
            if args.fitune_type =='lora':
                encoder = apply_lora_to_model(encoder, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
                checkpoint = torch.load(checkpoint_path)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])

                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                elif args.cls_fitune_type =='full':

                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':

                    for name, param in encoder.named_parameters():
                        param.requires_grad = False

            elif args.fitune_type =='full':

                checkpoint = torch.load(checkpoint_path)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                if args.cls_fitune_type =='lora':
                    encoder = apply_lora_to_model(encoder, r=args.cls_lora_r, alpha=args.cls_lora_alpha, dropout=args.cls_lora_dropout)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                elif args.cls_fitune_type =='full':

                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':

                    for name, param in encoder.named_parameters():
                        param.requires_grad = False

        if args.model =='BIOT':
            eeg_projector = BIOT_DataProjector()
            from src.biot import BIOTEncoder
            encoder = BIOTEncoder(emb_size=256, heads=8, depth=4,n_channels=18).to(device)

            if args.fitune_type =='lora':
                encoder = create_lora(encoder,args)
                checkpoint = torch.load(checkpoint_path, map_location=device)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        if param.is_floating_point():
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False

            elif args.fitune_type =='full':
                encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    encoder = create_lora_cls(encoder,args)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True

                elif args.cls_fitune_type =='full':
                    if args.model =='BIOT':
                        for name, param in encoder.named_parameters():
                            if param.is_floating_point():
                                param.requires_grad = True
                    else:
                        for name, param in encoder.named_parameters():
                            param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False

        if args.model =='BIOT':
            model = SimpleClassifier_BIOT(
                encoder=encoder,
                args=args,
                ch_names=ch_names,
                num_classes=args.num_classes,
            ).to(device)
        else:
            model = SimpleClassifier_multi_model(
                encoder=encoder,
                num_channels= chan_number,
                in_chans = in_chans,
                args = args,
                num_classes=args.num_classes,
                embed_dim=args.embed_dim
            ).to(device)

        trainable_params = [p for p in model.parameters() if p.requires_grad]

        params_m = sum(p.numel() for p in trainable_params)
        print(f"\n  Trainable parameters: {params_m:,} ({params_m/1e6:.4f} M)")
        try:
            from torch.utils.flop_counter import FlopCounterMode
        except ModuleNotFoundError:
            FlopCounterMode = None
            print("[Warning] torch.utils.flop_counter is not available in this PyTorch version. Skip FLOPs counting.")

        def get_eeg_sample_from_loader(loader, device):
            batch = next(iter(loader))

            if isinstance(batch, (tuple, list)):
                eeg = batch[0]
            else:
                eeg = batch["eeg"]

            return eeg[:1].to(device)

        model.eval()

        dummy_eeg = get_eeg_sample_from_loader(train_loader, device)
        input_res = tuple(dummy_eeg.shape[1:])

        if FlopCounterMode is not None:
            with FlopCounterMode(model) as fcm:

                output = model(dummy_eeg)
                loss = output.sum()

                loss.backward()

                total_flops = fcm.get_total_flops()
                flops_g=total_flops/1e9
                print(f"Total training FLOPs: {total_flops / 1e9:.4f} GFLOPs")
        else:
            fcm = None

        total_time_start = time.time()
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        from torch.cuda.amp import GradScaler
        scaler = torch.amp.GradScaler('cuda') if args.use_amp else None

        best_val_acc = float("-inf")
        best_val_f1 = float("-inf")
        patience_counter = 0

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Training task adaptation")
        print(f"Method: {args.cls_fitune_type.upper()}")
        print(f"{'='*80}")

        train_start = time.time()
        for epoch in range(args.epochs):

            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            model.train()
            train_loss = 0

            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    eeg, labels = batch
                else:
                    eeg = batch['eeg']
                    labels = batch['label']
                    mapped_labels = torch.tensor([label_mapping[label.item()] for label in labels])

                eeg = eeg.to(device)
                mapped_labels = mapped_labels.to(device)

                optimizer.zero_grad()

                if args.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(eeg)
                        loss = criterion(logits, mapped_labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(eeg)
                    loss = criterion(logits, mapped_labels)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            val_loss, val_acc, val_f1, val_cm = evaluate_multitask(
                model, val_loader, criterion, device, label_mapping, args.use_amp
            )

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
                      f"TrainLoss={avg_train_loss:.4f}, "
                      f"ValAcc={val_acc:.4f}, ValF1={val_f1:.4f}, "
                      f"LR={current_lr:.6f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
        train_time = time.time() - train_start

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Final Test Evaluation")
        print(f"{'='*80}")

        checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_start = time.time()
        test_loss, test_acc, test_f1, test_cm = evaluate_multitask(
            model, test_loader, criterion, device,label_mapping, args.use_amp
        )
        test_inference_time = time.time() - test_start
        total_time =time.time() - total_time_start
        print(f"  Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
        print(f"  Confusion Matrix:\n{test_cm}")

        metrics = {
            'subject_id': int(subject_id),
            'method': args.cls_fitune_type,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'best_val_acc': float(best_val_acc),
            'best_val_f1': float(best_val_f1),
            'test_acc': float(test_acc),
            'test_f1': float(test_f1),
            'test_cm': test_cm.tolist(),
            'best_epoch': int(checkpoint['epoch']) + 1,
            'transfer_method': 'linear_probing',
            'source_task': 'resting_reconstruction',
            'target_task': 'cognitive_classification',
            'train_time_sec': float(train_time),
            'test_time_sec': float(test_inference_time),
            'total_time_sec': float(total_time),
            'flops_g': float(flops_g),
            'params_m': float(params_m)
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"  ✓ Results saved to {subject_dir}")

        del model
        del optimizer
        del encoder
        torch.cuda.empty_cache()

        return metrics

    except Exception as e:
        print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
        import traceback
        traceback.print_exc()

        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

        return None


def train_single_subject_baseline_cls_new(args, subject_id, device, num_patches, dataset):
    """Train a subject classifier with the selected task adaptation method"""
    try:

        output_path = os.path.join(args.output_root, args.dataset, args.cls_fitune_type)
        subject_dir = os.path.join(output_path, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)

        metrics_path = os.path.join(subject_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            print(f"  ✓ Subject {subject_id:02d} already trained, skipping...")
            with open(metrics_path, 'r') as f:
                return json.load(f)

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Loading Data (Cognitive Task)")
        print(f"{'='*80}")

        if args.fewshot:
            train_loader, val_loader, test_loader= create_single_subject_few_shot_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                n_shot = args.n_shot,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers,
                val_size=args.val_size,
                oversample=args.oversample
            )
        elif args.dataset_suffle:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
        else:

            train_loader, val_loader, test_loader = create_single_subject_dataloaders_order(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )

        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

        if train_size < args.min_samples_per_class:
            raise ValueError(f"Insufficient training samples: {train_size}")

        if args.dataset=='cogbci':
            in_chans, ch_names, chan_number = get_cogbci_dataset(args, subject_id)
            print('in_chans',in_chans)
        else:
            in_chans, ch_names, chan_number = get_dataset_name(args)
            print('in_chans', in_chans)

        print(f"\n[Step 1] Creating Encoder ({args.model})...")

        if args.model == 'labram_base_patch200_200':

            encoder = create_model(
                args.model,
                pretrained=False,
                use_mean_pooling=args.use_mean_pooling,
                init_scale=args.init_scale,
                use_rel_pos_bias=args.rel_pos_bias,
                use_abs_pos_emb=args.abs_pos_emb,
                init_values=args.layer_scale_init_value,
                qkv_bias=args.qkv_bias,
                num_patches_per_channel_input=num_patches,
            )

            load_vanilla_labram_weights(
                encoder,
                args.pretrained_path,
                model_key='model',
                model_prefix=''
            )

        elif args.model == 'CBraMod':
            encoder=CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
            )
            map_location = torch.device(args.device)
            load_cbramod_pretrained(encoder, args.pretrained_path, map_location)

        if args.model =='BIOT':
            eeg_projector = BIOT_DataProjector()
            from src.biot import BIOTEncoder
            encoder = BIOTEncoder(emb_size=256, heads=8, depth=4,n_channels=18).to(device)
            load_biot_pretrained(encoder, args.pretrained_path, device)

        if  args.cls_fitune_type == 'lora':
            if args.model == 'CBraMod':
                encoder = apply_lora_to_model(encoder, r=args.cls_lora_r, alpha=args.cls_lora_alpha, dropout=args.cls_lora_dropout)
            else:
                encoder = create_lora_cls(encoder,args)
            for name, param in encoder.named_parameters():
                if 'lora' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        elif args.cls_fitune_type == 'full':
            if args.model =='BIOT':
                for name, param in encoder.named_parameters():
                    if param.is_floating_point():
                        param.requires_grad = True
            else:
                for name, param in encoder.named_parameters():
                    param.requires_grad = True

        elif args.cls_fitune_type == 'linear':
            for name, param in encoder.named_parameters():
                param.requires_grad = False

        if args.model=='BIOT':
            model = SimpleClassifier_BIOT(
                encoder=encoder,
                args=args,
                ch_names=ch_names,
                num_classes=args.num_classes,
            ).to(device)
        else:
            model = SimpleClassifier(
                encoder=encoder,
                num_channels=chan_number,
                in_chans=in_chans,
                args=args,
                num_classes=args.num_classes,
                embed_dim=args.embed_dim
            ).to(device)
        trainable = [p for p in encoder.parameters() if p.requires_grad]
        total = sum(p.numel() for p in encoder.parameters())
        trainable_count = sum(p.numel() for p in trainable)
        print(f"trainable: {trainable_count:,} / {total:,}")
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        total_time_start = time.time()

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if args.use_amp else None
        best_val_acc = float("-inf")
        best_val_f1 = float("-inf")
        patience_counter = 0

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Training task adaptation")
        print(f"Method: {args.cls_fitune_type.upper()}")
        print(f"{'='*80}")

        train_start = time.time()

        for epoch in range(args.epochs):

            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            model.train()
            train_loss = 0

            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    eeg, labels = batch
                else:
                    eeg = batch['eeg']
                    labels = batch['label']

                eeg = eeg.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                if args.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(eeg)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(eeg)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            train_loss, train_acc, train_f1,  train_cm = evaluate(
                model, train_loader, criterion, device, args.use_amp
            )

            val_loss, val_acc, val_f1, val_cm = evaluate(
                model, val_loader, criterion, device, args.use_amp
            )
            if epoch ==0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                }, os.path.join(subject_dir, 'original_model.pth'))

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
                      f"TrainLoss={avg_train_loss:.4f}, "
                      f"Train_acc={train_acc:.4f}",
                      f"ValAcc={val_acc:.4f}, ValF1={val_f1:.4f}, "
                      f"LR={current_lr:.6f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'classifier_state_dict': model.classifier.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
        train_time = time.time() - train_start

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Final Test Evaluation")
        print(f"{'='*80}")

        checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_start = time.time()
        test_loss, test_acc, test_f1, test_cm = evaluate(
            model, test_loader, criterion, device, args.use_amp
        )
        test_inference_time = time.time() - test_start
        total_time =time.time() - total_time_start
        print(f"  Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
        print(f"  Confusion Matrix:\n{test_cm}")

        metrics = {
            'subject_id': int(subject_id),
            'method': args.cls_fitune_type,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'best_val_acc': float(best_val_acc),
            'best_val_f1': float(best_val_f1),
            'test_acc': float(test_acc),
            'test_f1': float(test_f1),
            'test_cm': test_cm.tolist(),
            'best_epoch': int(checkpoint['epoch']) + 1,
            'transfer_method': 'linear_probing',
            'source_task': 'resting_reconstruction',
            'target_task': 'cognitive_classification',
            'train_time_sec': float(train_time),
            'test_time_sec': float(test_inference_time),
            'total_time_sec': float(total_time),
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"  ✓ Results saved to {subject_dir}")

        del model
        del optimizer
        del encoder
        torch.cuda.empty_cache()

        return metrics

    except Exception as e:
        print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
        import traceback
        traceback.print_exc()

        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

        return None


def train_for_subject_embedding_with_dynamics(args, subject_id, device, num_patches, dataset,encoder_group):
    """Calibrate a subject encoder against the fixed population teacher."""
    try:

        subject_dir = os.path.join(args.output_root, args.dataset, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Loading Data")
        print(f"{'='*80}")

        if args.multi_task:
            train_loader, val_loader, test_loader= create_multitask_single_subject_dataloaders(
                dataset= dataset,
                target_subject_id=subject_id,
                target_task_id=args.multi_task_ids,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
        elif args.train_count is not None:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders_control(
                dataset=dataset, target_subject_id=subject_id,
                train_size=args.train_size, test_size=args.test_size, val_size=args.val_size,
                batch_size=args.batch_size, random_state=args.seed,
                num_workers=args.num_workers, train_count=args.train_count)
        elif args.dataset_suffle:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
        else:

            train_loader, val_loader, test_loader = create_single_subject_dataloaders_order(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )

        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

        if train_size < args.min_samples_per_class:
            raise ValueError(f"Insufficient training samples: {train_size}")

        if test_size == 0:
            print("Warning: Test set is empty! Skipping test evaluation.")
            test_metrics = None
        else:
            test_metrics = True

        if args.dataset=='cogbci':
            in_chans, ch_names, chan_number = get_cogbci_dataset(args, subject_id)
            print('in_chans',in_chans)
        else:
            in_chans, ch_names, chan_number = get_dataset_name(args)
            print('in_chans', in_chans)

        for name, param in encoder_group.named_parameters():
            param.requires_grad = False

        if args.model =='labram_base_patch200_200':
            encoder_sub = create_model(
                args.model,
                pretrained=False,
                use_mean_pooling=args.use_mean_pooling,
                init_scale=args.init_scale,
                use_rel_pos_bias=args.rel_pos_bias,
                use_abs_pos_emb=args.abs_pos_emb,
                init_values=args.layer_scale_init_value,
                qkv_bias=args.qkv_bias,
                num_patches_per_channel_input=num_patches
            ).to(device)
            load_vanilla_labram_weights(encoder_sub, args.pretrained_path, args.model_key, args.model_prefix)
            if args.fitune_type == 'lora':
                encoder_sub = create_lora(encoder_sub, args)
                for name, param in encoder_sub.named_parameters():
                    if 'lora' not in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            else:
                for name, param in encoder_sub.named_parameters():
                    param.requires_grad = True

        elif args.model =='CBraMod':
            encoder_sub = CBraMod().to(device)
            load_cbramod_pretrained(encoder_sub, args.pretrained_path, device)

            if args.fitune_type == 'lora':
                encoder_sub = apply_lora_to_model(encoder_sub, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)

                for name, param in encoder_sub.named_parameters():
                    if 'lora' not in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            else:
                for name, param in encoder_sub.named_parameters():
                    param.requires_grad = True

        if args.model =='BIOT':
            eeg_projector = BIOT_DataProjector()
            from src.biot import BIOTEncoder
            encoder_sub = BIOTEncoder(emb_size=256, heads=8, depth=4,n_channels=18).to(device)

            print(f"Loading BIOT continue-pretrain checkpoint from: {args.pretrained_path}")
            checkpoint = torch.load(args.pretrained_path, map_location=device, weights_only=False)

            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
                print(f"  Extracted state_dict from checkpoint['model']")
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('biot.'):
                    new_state_dict[k[5:]] = v
                elif k.startswith('prediction.'):

                    continue
                else:
                    new_state_dict[k] = v

            missing, unexpected = encoder_sub.load_state_dict(new_state_dict, strict=False)
            if missing:
                print(f"  Missing keys (random init): {missing[:3]}{'...' if len(missing)>3 else ''}")
            if unexpected:
                print(f"  Unexpected keys (skipped): {unexpected[:3]}{'...' if len(unexpected)>3 else ''}")
            print(f"  BIOT continue-pretrain weights loaded successfully!")
            if args.fitune_type == 'lora':
                encoder_sub=create_lora(encoder_sub,args)
                for name, param in encoder_sub.named_parameters():
                    if 'lora' in name.lower():

                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            else:
                for name, param in encoder_sub.named_parameters():
                    if param.is_floating_point():
                        param.requires_grad = True
                    else:

                        param.requires_grad = False

        total_params, trainable_params, trainable_percentage, trainable_modules = calculate_parameter_percentage_with_names(encoder_sub)
        print(f"encoder_sub Total parameters: {total_params}")
        print(f"encoder_sub Trainable parameters: {trainable_params}")
        print(f"encoder_sub Trainable parameters percentage: {trainable_percentage:.2f}%")

        trainable_params = [p for p in encoder_sub.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Training ({args.fitune_type})")
        print(f"{'='*80}")

        masking_strategy = Embedding_MaskingStrategy(mask_ratio=args.mask_ratio, mask_type=args.mask_type)
        max_grad_norm = 1.0

        for epoch in range(args.epochs):
            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            encoder_sub.train()
            encoder_group.eval()
            train_loss = 0
            train_loss_intra =0
            train_loss_cross =0

            for batch in train_loader:

                optimizer.zero_grad()
                eeg = batch['eeg'].to(device)
                label = batch['label'].to(device)
                B, C, A, T = eeg.shape

                if args.embedding_high_level:
                    mask_eeg, mask = masking_strategy(eeg)
                    if args.model =='CBraMod':
                        embedding_mask = encoder_sub(mask_eeg)
                        embedding_truth = encoder_group(eeg)
                    elif args.model =='labram_base_patch200_200':
                        embedding_mask = encoder_sub.forward_features(mask_eeg, input_chans=in_chans,return_patch_tokens=True)
                        embedding_truth = encoder_group.forward_features(eeg, input_chans=in_chans,return_patch_tokens=True)

                    loss, loss_intra, loss_cross = contrastive_4d_channel_dynamics(embedding_mask, embedding_truth, label, args,
                                                                                                epoch=epoch, warmup_epoch=args.warmup_epochs,
                                                                                                temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha,
                                                                                                main_weight=args.main_weight, lambda_mse=args.lambda_mse,
                                                                                                batch_size= B, channel=C, patch=A, dimension=T)
                else:
                    if args.model =='BIOT':
                        mask_eeg, mask = masking_strategy(eeg)
                        eeg = eeg_projector.channel_project(eeg, ch_names)
                        eeg_mask = eeg_projector.channel_project(mask_eeg, ch_names)
                        embedding_mask = encoder_sub(eeg_mask)
                        embedding_truth = encoder_group(eeg)
                    elif args.model =='labram_base_patch200_200':
                        mask_eeg,mask = masking_strategy(eeg)
                        embedding_mask = encoder_sub.forward_features(mask_eeg, input_chans=in_chans)
                        embedding_truth = encoder_group.forward_features(eeg, input_chans=in_chans)

                    loss, loss_intra, loss_cross = contrastive_dynamics(embedding_mask, embedding_truth, label,
                                                                                    epoch=epoch, warmup_epoch=args.warmup_epochs, main_weight=args.main_weight, lambda_mse=args.lambda_mse,
                                                                           temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha)

                loss.backward()
                utils.clip_grad_norm_(encoder_sub.parameters(), max_grad_norm)
                optimizer.step()

                train_loss += loss.item()
                train_loss_intra += loss_intra.item()
                train_loss_cross += loss_cross.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_train_loss_intra = train_loss_intra / len(train_loader)
            avg_train_loss_cross = train_loss_cross / len(train_loader)

            train_losses.append(avg_train_loss)

            encoder_sub.eval()
            encoder_group.eval()
            val_loss = 0
            val_loss_intra = 0
            val_loss_cross = 0
            with torch.no_grad():
                for batch in val_loader:
                    eeg = batch['eeg'].to(device)
                    label = batch['label'].to(device)
                    B, C, A, T = eeg.shape
                    if args.embedding_high_level:
                        mask_eeg, mask = masking_strategy(eeg)
                        if args.model =='CBraMod':
                            val_embedding_sub = encoder_sub(mask_eeg)
                            val_embedding_group = encoder_group(eeg)
                        elif args.model =='labram_base_patch200_200':
                            val_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans,return_patch_tokens=True)
                            val_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans,return_patch_tokens=True)

                        val_loss_curr, val_loss_intra_curr, val_loss_cross_curr = contrastive_4d_channel_dynamics(val_embedding_sub, val_embedding_group, label, args,
                                                                                                                     epoch=epoch, warmup_epoch=args.warmup_epochs,
                                                                                                                     temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha,
                                                                                                                     batch_size= B, channel=C, patch=A, dimension=T)

                    else:
                        if args.model =='BIOT':
                            mask_eeg, mask = masking_strategy(eeg)
                            eeg = eeg_projector.channel_project(eeg, ch_names)
                            eeg_mask = eeg_projector.channel_project(mask_eeg, ch_names)
                            val_embedding_sub = encoder_sub(eeg_mask)
                            val_embedding_group = encoder_group(eeg)
                        elif args.model =='CBraMod':

                            mask_eeg, mask = masking_strategy(eeg)
                            val_embedding_sub = encoder_sub(mask_eeg).mean(dim=(1, 2))
                            val_embedding_group = encoder_group(eeg).mean(dim=(1, 2))
                        elif args.model =='labram_base_patch200_200':
                            mask_eeg, mask = masking_strategy(eeg)
                            val_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans)
                            val_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans)
                        val_loss_curr, val_loss_intra_curr, val_loss_cross_curr = contrastive_dynamics(val_embedding_sub, val_embedding_group, label,
                                                                                                                    epoch=epoch, warmup_epoch=args.warmup_epochs,
                                                                                                                    temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha, )

                    val_loss += val_loss_curr.item()
                    val_loss_intra += val_loss_intra_curr.item()
                    val_loss_cross += val_loss_cross_curr.item()

            avg_val_loss = val_loss / len(val_loader)
            avg_val_loss_intra = val_loss_intra / len(val_loader)
            avg_val_loss_cross = val_loss_cross / len(val_loader)

            val_losses.append(avg_val_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={avg_train_loss:.4f}, "
                    f"Loss MSE={avg_train_loss_intra:.4f}, Loss Cross={avg_train_loss_cross:.4f}, "
                    f"Val Loss={avg_val_loss:.4f}, LR={current_lr:.6f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': encoder_sub.state_dict(),
                    'val_loss': avg_val_loss,
                }, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        losses = {
        'train_losses': train_losses,
        'val_losses': val_losses}

        if test_metrics is not None:
            print(f"\n{'='*80}")
            print(f"Subject {subject_id:02d} - Final Test Evaluation")
            print(f"{'='*80}")

            checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'))
            encoder_sub.load_state_dict(checkpoint['encoder_state_dict'])

            test_loss = 0
            test_loss_intra = 0
            test_loss_cross = 0
            encoder_sub.eval()
            encoder_group.eval()
            with torch.no_grad():
                for batch in test_loader:
                    eeg = batch['eeg'].to(device)
                    label = batch['label'].to(device)
                    B, C, A, T = eeg.shape
                    if args.embedding_high_level:
                        mask_eeg, mask = masking_strategy(eeg)
                        if args.model =='CBraMod':
                            test_embedding_sub = encoder_sub(mask_eeg)
                            test_embedding_group = encoder_group(eeg)
                        elif args.model =='labram_base_patch200_200':
                            test_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans,return_patch_tokens=True)
                            test_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans,return_patch_tokens=True)

                        test_loss_curr, test_loss_intra_curr, test_loss_cross_curr = contrastive_4d_channel_dynamics(test_embedding_sub, test_embedding_group, label,args,
                                                                                                                                epoch=epoch, warmup_epoch=args.warmup_epochs,
                                                                                                                                temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha,
                                                                                                                                batch_size= B, channel=C, patch=A, dimension=T)
                    else:
                        if args.model =='BIOT':
                            mask_eeg, mask = masking_strategy(eeg)
                            eeg = eeg_projector.channel_project(eeg, ch_names)
                            eeg_mask = eeg_projector.channel_project(mask_eeg, ch_names)
                            test_embedding_sub = encoder_sub(eeg_mask)
                            test_embedding_group = encoder_group(eeg)
                        elif args.model =='CBraMod':

                            mask_eeg, mask = masking_strategy(eeg)
                            test_embedding_sub = encoder_sub(mask_eeg).mean(dim=(1, 2))
                            test_embedding_group = encoder_group(eeg).mean(dim=(1, 2))
                        elif args.model =='labram_base_patch200_200':
                            mask_eeg,mask = masking_strategy(eeg)
                            test_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans)
                            test_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans)
                        test_loss_curr, test_loss_intra_curr, test_loss_cross_curr = contrastive_dynamics(test_embedding_sub, test_embedding_group,label,
                                                                                                             epoch=epoch, warmup_epoch=args.warmup_epochs,
                                                                                                             temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha, )

                    test_loss += test_loss_curr.item()
                    test_loss_intra += test_loss_intra_curr.item()
                    test_loss_cross += test_loss_cross_curr.item()

            avg_test_loss = test_loss / len(test_loader)
            avg_test_loss_intra = test_loss_intra / len(test_loader)
            avg_test_loss_cross = test_loss_cross / len(test_loader)

            metrics = {
                'subject_id': int(subject_id),
                'train_size': train_size,
                'train_loss': avg_train_loss,
                'train_loss_intra': avg_train_loss_intra,
                'train_loss_cross': avg_train_loss_cross,
                'val_size': val_size,
                'val_loss': avg_val_loss,
                'val_loss_intra': avg_val_loss_intra,
                'val_loss_cross': avg_val_loss_cross,
                'test_size': test_size,
                'test_loss': avg_test_loss,
                'test_loss_intra': avg_test_loss_intra,
                'test_loss_cross': avg_test_loss_cross,
                'best_epoch': checkpoint['epoch'] + 1,
            }

            with open(os.path.join(subject_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"  ✓ Results saved to {subject_dir}")

        else:
            metrics = {
                'subject_id': int(subject_id),
                'train_size': train_size,
                'train_loss': avg_train_loss,
                'val_size': val_size,
                'val_loss': avg_val_loss,
                'best_epoch': checkpoint['epoch'] + 1,
            }

            with open(os.path.join(subject_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"  ✓ Results saved to {subject_dir}")

        torch.cuda.empty_cache()

        return metrics, losses

    except Exception as e:
        print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
        import traceback
        traceback.print_exc()

        if 'encoder_sub' in locals():
            del encoder_sub
        torch.cuda.empty_cache()

        return None, None


def train_single_subject_baseline_multitask(args, subject_id, device, num_patches, dataset):
    """Train a subject classifier with the selected task adaptation method"""
    if args.dataset=='m3cv':
        label_mapping = {
            10: 0,
            11: 1,
            12: 2,
            20: 3,
            21: 4,
            30: 5,
            31: 6,
            32: 7,
            40: 8,
            41: 9,
            42: 10
        }
        target_task_ids =[1,2,3,4]
    else:
        label_mapping = {
            10: 0,
            11: 1,
            12: 2,
            20: 3,
            21: 4,
            22: 5
        }
        target_task_ids =[1,2]

    try:

        output_path = os.path.join(args.output_root, args.dataset, args.cls_fitune_type)
        subject_dir = os.path.join(output_path, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)

        metrics_path = os.path.join(subject_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            print(f"  ✓ Subject {subject_id:02d} already trained, skipping...")
            with open(metrics_path, 'r') as f:
                return json.load(f)

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Loading Data")
        print(f"{'='*80}")

        if args.multi_task:
            train_loader, val_loader, test_loader= create_multitask_single_subject_dataloaders(
                dataset= dataset,
                target_subject_id=subject_id,
                target_task_id=target_task_ids,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )

        elif args.fewshot:
            train_loader, val_loader, test_loader= create_single_subject_few_shot_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                n_shot = args.n_shot,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers,
                val_size=args.val_size,
                oversample=args.oversample
            )
        elif args.dataset_suffle:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )

        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

        if train_size < args.min_samples_per_class:
            raise ValueError(f"Insufficient training samples: {train_size}")

        if args.dataset=='cogbci':
            in_chans, ch_names, chan_number = get_cogbci_dataset(args, subject_id)
            print('in_chans',in_chans)
        else:
            in_chans, ch_names, chan_number = get_dataset_name(args)
            print('in_chans', in_chans)

        if args.model == 'labram_base_patch200_200':

            encoder = create_model(
                args.model,
                pretrained=False,
                use_mean_pooling=args.use_mean_pooling,
                init_scale=args.init_scale,
                use_rel_pos_bias=args.rel_pos_bias,
                use_abs_pos_emb=args.abs_pos_emb,
                init_values=args.layer_scale_init_value,
                qkv_bias=args.qkv_bias,
                num_patches_per_channel_input=num_patches,
            )
            load_vanilla_labram_weights(
                encoder,
                args.pretrained_path,
                model_key='model',
                model_prefix=''
            )

        elif args.model == 'CBraMod':
            encoder=CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )
            map_location = torch.device(args.device)
            load_cbramod_pretrained(encoder, args.pretrained_path, map_location)

        if args.model =='BIOT':
            eeg_projector = BIOT_DataProjector()
            from src.biot import BIOTEncoder
            encoder = BIOTEncoder(emb_size=256, heads=8, depth=4,n_channels=18).to(device)
            map_location = torch.device(args.device)
            load_biot_pretrained(encoder, args.pretrained_path, map_location)

        if args.cls_fitune_type == 'lora':
            if args.model=='CBraMod':
                encoder = apply_lora_to_model(encoder, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
            else:
                encoder = create_lora_cls(encoder,args)
            for name, param in encoder.named_parameters():
                if 'lora' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        elif args.cls_fitune_type == 'full':
            if args.model=='BIOT':
                for name, param in encoder.named_parameters():
                    if param.is_floating_point():
                        param.requires_grad = True
            else:
                for name, param in encoder.named_parameters():
                    param.requires_grad = True

        elif args.cls_fitune_type == 'linear':
            for name, param in encoder.named_parameters():
                param.requires_grad = False

        if args.model =='BIOT':
            model = SimpleClassifier_BIOT(
                encoder=encoder,
                args=args,
                ch_names=ch_names,
                num_classes=args.num_classes,
            ).to(device)
        else:
            model = SimpleClassifier_multi_model(
                encoder=encoder,
                num_channels= chan_number,
                in_chans = in_chans,
                args = args,
                num_classes=args.num_classes,
                embed_dim=args.embed_dim
            ).to(device)

        trainable_params = [p for p in model.parameters() if p.requires_grad]

        params_m = sum(p.numel() for p in trainable_params)
        print(f"\n  Trainable parameters: {params_m:,} ({params_m/1e6:.4f} M)")
        try:
            from torch.utils.flop_counter import FlopCounterMode
        except ModuleNotFoundError:
            FlopCounterMode = None
            print("[Warning] torch.utils.flop_counter is not available in this PyTorch version. Skip FLOPs counting.")

        def get_eeg_sample_from_loader(loader, device):
            batch = next(iter(loader))

            if isinstance(batch, (tuple, list)):
                eeg = batch[0]
            else:
                eeg = batch["eeg"]

            return eeg[:1].to(device)

        model.eval()

        dummy_eeg = get_eeg_sample_from_loader(train_loader, device)
        input_res = tuple(dummy_eeg.shape[1:])

        if FlopCounterMode is not None:
            with FlopCounterMode(model) as fcm:

                output = model(dummy_eeg)
                loss = output.sum()

                loss.backward()

                total_flops = fcm.get_total_flops()
                flops_g=total_flops/1e9
                print(f"Total training FLOPs: {total_flops / 1e9:.4f} GFLOPs")
        else:
            fcm = None

        total_time_start = time.time()
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if args.use_amp else None
        best_val_acc = float("-inf")
        best_val_f1 = float("-inf")
        patience_counter = 0

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Training task adaptation")
        print(f"Method: {args.cls_fitune_type.upper()}")
        print(f"{'='*80}")

        train_start = time.time()
        for epoch in range(args.epochs):

            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            model.train()
            train_loss = 0

            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    eeg, labels = batch
                else:
                    eeg = batch['eeg']
                    labels = batch['label']

                    labels = torch.tensor([label_mapping[label.item()] for label in labels])

                eeg = eeg.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                if args.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(eeg)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(eeg)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            train_loss, train_acc, train_f1, val_cm = evaluate_multitask(
                model, train_loader, criterion, device, label_mapping, args.use_amp
            )

            val_loss, val_acc, val_f1, val_cm = evaluate_multitask(
                model, val_loader, criterion, device, label_mapping, args.use_amp
            )

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
                      f"TrainLoss={avg_train_loss:.4f}, "
                      f"TrainAcc={train_acc:.4f}, trainF1={train_f1:.4f}, "
                      f"ValAcc={val_acc:.4f}, ValF1={val_f1:.4f}, "
                      f"LR={current_lr:.6f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'classifier_state_dict': model.classifier.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        train_time = time.time() - train_start

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Final Test Evaluation")
        print(f"{'='*80}")

        checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_start = time.time()
        test_loss, test_acc, test_f1, test_cm = evaluate_multitask(
            model, test_loader, criterion, device, label_mapping, args.use_amp
        )
        test_inference_time = time.time() - test_start
        total_time =time.time() - total_time_start
        print(f"  Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
        print(f"  Confusion Matrix:\n{test_cm}")

        metrics = {
            'subject_id': int(subject_id),
            'method': args.cls_fitune_type,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'best_val_acc': float(best_val_acc),
            'best_val_f1': float(best_val_f1),
            'test_acc': float(test_acc),
            'test_f1': float(test_f1),
            'test_cm': test_cm.tolist(),
            'best_epoch': int(checkpoint['epoch']) + 1,
            'transfer_method': 'linear_probing',
            'source_task': 'resting_reconstruction',
            'target_task': 'cognitive_classification',
            'train_time_sec': float(train_time),
            'test_time_sec': float(test_inference_time),
            'total_time_sec': float(total_time),

            'params_m': float(params_m)
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"  ✓ Results saved to {subject_dir}")

        del model
        del optimizer
        del encoder
        torch.cuda.empty_cache()

        return metrics

    except Exception as e:
        print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
        import traceback
        traceback.print_exc()

        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

        return None

from itertools import cycle


def train_single_subject_cls_multi_task_cog(args, subject_id, device, num_patches, dataset):
    """Train a subject classifier with the selected task adaptation method"""
    label_mapping = {
        10: 0,
        11: 1,
        12: 2,
        20: 3,
        21: 4,
        22: 5,
    }
    try:
        valid_fitune_types = ['lora', 'full']
        valid_cls_fitune_types = ['lora', 'full', 'linear']
        if args.fitune_type not in valid_fitune_types:
            raise ValueError(f"fitune_type must be one of {valid_fitune_types}")
        if args.cls_fitune_type not in valid_cls_fitune_types:
            raise ValueError(f"cls_fitune_type must be one of {valid_cls_fitune_types}")

        output_path = os.path.join(args.output_root, args.dataset,  args.cls_fitune_type)
        subject_dir = os.path.join(output_path, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)

        metrics_path = os.path.join(subject_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            print(f"  ✓ Subject {subject_id:02d} already trained, skipping...")
            with open(metrics_path, 'r') as f:
                return json.load(f)

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Loading Data")
        print(f"{'='*80}")

        if args.multi_task:
            train_loader, val_loader, test_loader= create_multitask_single_subject_dataloaders(
                dataset= dataset,
                target_subject_id=subject_id,
                target_task_id=[1,2],
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
        elif args.multitask_fewshot:
            train_loader, val_loader, test_loader= create_multitask_single_subject_few_shot_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                target_task_id=[1,2],
                n_shot = args.n_shot,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers,
                val_size=args.val_size,
                oversample=args.oversample
            )
        elif args.fewshot:
            train_loader, val_loader, test_loader= create_single_subject_few_shot_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                n_shot = args.n_shot,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers,
                val_size=args.val_size,
                oversample=args.oversample
            )
        elif args.dataset_suffle:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )

        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

        if train_size < args.min_samples_per_class:
            raise ValueError(f"Insufficient training samples: {train_size}")

        if args.dataset=='cogbci':
            in_chans, ch_names, chan_number = get_cogbci_dataset(args, subject_id)
            print('in_chans',in_chans)
        else:
            in_chans, ch_names, chan_number = get_dataset_name(args)
            print('in_chans', in_chans)

        checkpoint_path = os.path.join(args.recon_results_dir, args.dataset, f'subject_{subject_id:02d}','best_model.pth')
        print(f"  Checkpoint: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Pretrained encoder checkpoint not found: {checkpoint_path}")

        print(f"\n[Step 1] Creating Encoder ({args.model})...")

        if args.model =='labram_base_patch200_200':
            encoder = create_model(
                args.model,
                pretrained=False,
                use_mean_pooling=args.use_mean_pooling,
                init_scale=args.init_scale,
                use_rel_pos_bias=args.rel_pos_bias,
                use_abs_pos_emb=args.abs_pos_emb,
                init_values=args.layer_scale_init_value,
                qkv_bias=args.qkv_bias,
                num_patches_per_channel_input=num_patches).to(device)

            if args.fitune_type =='lora':
                encoder_with_lora = create_lora(encoder,args)
                encoder = load_pretrained_encoder(encoder_with_lora, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False

            elif args.fitune_type =='full':
                encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    encoder = create_lora_cls(encoder,args)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True

                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False

        elif args.model =='CBraMod':
            encoder = CBraMod().to(device)
            if args.fitune_type =='lora':
                encoder = apply_lora_to_model(encoder, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
                checkpoint = torch.load(checkpoint_path)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])

                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                elif args.cls_fitune_type =='full':

                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':

                    for name, param in encoder.named_parameters():
                        param.requires_grad = False

            elif args.fitune_type =='full':

                checkpoint = torch.load(checkpoint_path)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                if args.cls_fitune_type =='lora':
                    encoder = apply_lora_to_model(encoder, r=args.cls_lora_r, alpha=args.cls_lora_alpha, dropout=args.cls_lora_dropout)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                elif args.cls_fitune_type =='full':

                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':

                    for name, param in encoder.named_parameters():
                        param.requires_grad = False

        if args.model =='BIOT':
            eeg_projector = BIOT_DataProjector()
            from src.biot import BIOTEncoder
            encoder = BIOTEncoder(emb_size=256, heads=8, depth=4,n_channels=18).to(device)

            if args.fitune_type =='lora':
                encoder = create_lora(encoder,args)
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        if param.is_floating_point():
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False

            elif args.fitune_type =='full':
                encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    encoder = create_lora_cls(encoder,args)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True

                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        if param.is_floating_point():
                            param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False

        if args.model =='BIOT':
            model = SimpleClassifier_BIOT(
                encoder=encoder,
                args=args,
                ch_names=ch_names,
                num_classes=args.num_classes,
            ).to(device)
        else:
            model = SimpleClassifier_multi_model(
                encoder=encoder,
                num_channels= chan_number,
                in_chans = in_chans,
                args = args,
                num_classes=args.num_classes,
                embed_dim=args.embed_dim
            ).to(device)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        params_m = sum(p.numel() for p in trainable_params)
        print(f"\n  Trainable parameters: {params_m:,} ({params_m/1e6:.4f} M)")
        try:
            from torch.utils.flop_counter import FlopCounterMode
        except ModuleNotFoundError:
            FlopCounterMode = None
            print("[Warning] torch.utils.flop_counter is not available in this PyTorch version. Skip FLOPs counting.")

        def get_eeg_sample_from_loader(loader, device):
            batch = next(iter(loader))

            if isinstance(batch, (tuple, list)):
                eeg = batch[0]
            else:
                eeg = batch["eeg"]

            return eeg[:1].to(device)

        model.eval()

        dummy_eeg = get_eeg_sample_from_loader(train_loader, device)
        input_res = tuple(dummy_eeg.shape[1:])

        if FlopCounterMode is not None:
            with FlopCounterMode(model) as fcm:

                output = model(dummy_eeg)
                loss = output.sum()

                loss.backward()

                total_flops = fcm.get_total_flops()
                flops_g=total_flops/1e9
                print(f"Total training FLOPs: {total_flops / 1e9:.4f} GFLOPs")
        else:
            fcm = None

        total_time_start = time.time()

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if args.use_amp else None

        best_val_acc = float("-inf")
        best_val_f1 = float("-inf")
        patience_counter = 0

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Training task adaptation")
        print(f"Method: {args.cls_fitune_type.upper()}")
        print(f"{'='*80}")
        train_start = time.time()

        for epoch in range(args.epochs):

            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            model.train()
            train_loss = 0

            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    eeg, labels = batch
                else:
                    eeg = batch['eeg']
                    labels = batch['label']
                    mapped_labels = torch.tensor([label_mapping[label.item()] for label in labels])

                eeg = eeg.to(device)
                mapped_labels = mapped_labels.to(device)

                optimizer.zero_grad()

                if args.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(eeg)
                        loss = criterion(logits, mapped_labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(eeg)
                    loss = criterion(logits, mapped_labels)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            val_loss, val_acc, val_f1, val_cm = evaluate_multitask_cog(
                model, val_loader, criterion, device, args.use_amp
            )

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
                      f"TrainLoss={avg_train_loss:.4f}, "
                      f"ValAcc={val_acc:.4f}, ValF1={val_f1:.4f}, "
                      f"LR={current_lr:.6f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        train_time = time.time() - train_start

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Final Test Evaluation")
        print(f"{'='*80}")

        checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_start = time.time()
        test_loss, test_acc, test_f1, test_cm = evaluate_multitask_cog(
            model, test_loader, criterion, device, args.use_amp
        )
        test_inference_time = time.time() - test_start
        total_time =time.time() - total_time_start
        print(f"  Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
        print(f"  Confusion Matrix:\n{test_cm}")

        metrics = {
            'subject_id': int(subject_id),
            'method': args.cls_fitune_type,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'best_val_acc': float(best_val_acc),
            'best_val_f1': float(best_val_f1),
            'test_acc': float(test_acc),
            'test_f1': float(test_f1),
            'test_cm': test_cm.tolist(),
            'best_epoch': int(checkpoint['epoch']) + 1,
            'transfer_method': 'linear_probing',
            'source_task': 'resting_reconstruction',
            'target_task': 'cognitive_classification',
            'train_time_sec': float(train_time),
            'test_time_sec': float(test_inference_time),
            'total_time_sec': float(total_time),

            'params_m': float(params_m)
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"  ✓ Results saved to {subject_dir}")

        del model
        del optimizer
        del encoder
        torch.cuda.empty_cache()

        return metrics

    except Exception as e:
        print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
        import traceback
        traceback.print_exc()

        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

        return None
