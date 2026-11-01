"""Run the original Rest-Tuning subject experiments."""
from peft import get_peft_model, LoraConfig
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
from src.model_eegpt import *
from src.utils import *
from timm.models import create_model
from src.data_loader import *
from src.train_eegpt import *
from src.cbramod import CBraMod


def target_modules_list(value):
    if isinstance(value, str):
        return [m.strip() for m in value.split(",")]
    return value


def get_args(argv=None):
    parser = argparse.ArgumentParser('Rest-Tuning calibration',
                                     add_help=True)

    parser.add_argument('--model', default='EEGPT', type=str)
    parser.add_argument('--pretrained_path',
                        default=None,
                        type=str,
                        help='Path to ORIGINAL LaBraM Base pretrained weights')
    parser.add_argument('--embed_dim', default=200, type=int)
    parser.add_argument('--patch_size', default=200, type=int)

    parser.add_argument('--hidden_dim', default=1024, type=int,
                        help='V2-Fixed: Increased from 512 to 1024 for better capacity')
    parser.add_argument('--dropout', default=0.0, type=float,
                        help='V2-Fixed: Dropout rate (0.0 recommended for reconstruction)')

    parser.add_argument('--use_mean_pooling', action='store_true', default=True)
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--disable_qkv_bias', action='store_false', dest='qkv_bias')
    parser.set_defaults(qkv_bias=True)
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true', default=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float)
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)

    parser.add_argument('--h5_path',
                        default=None,
                        type=str,
                        help='Path to HDF5 file containing all subjects')
    parser.add_argument('--test_size', default=0.2, type=float)
    parser.add_argument('--val_size', default=0.1, type=float)
    parser.add_argument('--train_size', default=0.7, type=float)
    parser.add_argument('--normalize', action='store_true', default=True)
    parser.add_argument('--normalize_per_channel', action='store_true', default=False,
                        help='V2-Fixed: Set to False for per-trial normalization')
    parser.add_argument('--dataset',default='m3cv',type=str)

    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--patience', default=30, type=int)
    parser.add_argument('--sgd_lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='SGD momentum')
    parser.add_argument('--min_sgd_lr', default=0.08, type=float)

    parser.add_argument('--mse_weight', default=1.0, type=float,
                        help='V2-Fixed: Weight for MSE loss')
    parser.add_argument('--var_weight', default=0.5, type=float,
                        help='V2-Fixed: Weight for variance loss (critical!)')
    parser.add_argument('--spectral_weight', default=0.3, type=float,
                        help='V2-Fixed: Weight for spectral loss')

    parser.add_argument('--output_root',
                        default='results',
                        type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--use_amp', action='store_true', default=True)

    parser.add_argument('--min_samples_per_class', default=4, type=int)

    parser.add_argument('--fitune_type',default='full',type=str)

    parser.add_argument('--residual_hidden_dim', default=512, type=int,
                        help='Hidden dimension for ResEEGNet')
    parser.add_argument('--residual_dropout', default=0.1, type=float,
                        help='V2-Fixed: Dropout rate for ResEEGNet (reduced from 0.2 to 0.1)')

    parser.add_argument('--head_hidden_dim', default=1024, type=int,
                        help='V2-Fixed: Hidden dimension for reconstruction head (increased from 512 to 1024)')
    parser.add_argument('--head_dropout', default=0.1, type=float,
                        help='V2-Fixed: Dropout rate for reconstruction head (0.0 recommended)')

    parser.add_argument('--mask_ratio', default=0.4, type=float)
    parser.add_argument('--mask_type', type=str, choices=['random', 'channel', 'hybrid','non_mask','patch'], help="The type of mask to apply")

    parser.add_argument('--lora_r', default=16, type=int)
    parser.add_argument('--lora_alpha', default=32, type=int)
    parser.add_argument('--lora_dropout', default=0.1, type=float)

    parser.add_argument('--target_modules',
                   default="blocks.4.attn.qkv, blocks.5.attn.qkv, blocks.6.attn.qkv",
                   type=target_modules_list)

    parser.add_argument('--recon_type', default='MAE', type=str)
    parser.add_argument('--dataset_suffle', action='store_true', help="shuffle")
    parser.add_argument('--keep_channel', action='store_true', help="keep_channel")
    parser.add_argument('--mask_EEG', action='store_true', help="mask_EEG")
    parser.add_argument('--contrastive_temperature', default=0.1, type=float)
    parser.add_argument('--embedding_high_level', action='store_true', help="Use high-level embeddings for contrastive loss")
    parser.add_argument('--contrastive_task_alpha',default=0.1, type=float)
    parser.add_argument('--contrastive_task_temperature', default=0.07, type=float)
    parser.add_argument('--multi_task',action='store_true')
    parser.add_argument('--multi_task_ids', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--train_count', type=int, default=None)

    parser.add_argument('--lambda_mse', default=0.1, type=float)
    parser.add_argument('--main_weight', default=1.0, type=float)

    parser.add_argument('--subjects', type=int, nargs='+', help='Only load and train these subject IDs')
    parser.add_argument('--no_amp', action='store_false', dest='use_amp')
    return parser.parse_args(argv)


def main(args):
    prepare_run(args)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    os.makedirs(args.output_root, exist_ok=True)
    print('=' * 80)
    print(f'Rest-Tuning ({args.fitune_type})')
    print('=' * 80)
    print(f'Pretrained: {args.pretrained_path}')
    print(f'Output: {args.output_root}')
    print(f'Hidden Dim: {args.hidden_dim}, Dropout: {args.dropout}')
    print(f'Loss Weights: MSE={args.mse_weight}, Var={args.var_weight}, Spec={args.spectral_weight}')
    print('=' * 80)
    print('\nScanning subjects in dataset...')
    unique_subject_ids = np.asarray(args.subjects)
    print(f'Found {len(unique_subject_ids)} unique subjects: {unique_subject_ids}')
    output_path = os.path.join(args.output_root, args.dataset)
    os.makedirs(output_path, exist_ok=True)
    args_filepath = os.path.join(output_path, 'experiment_args.json')
    save_args_to_json(args, args_filepath)
    temp_dataset = EEGDataset(h5_path=args.h5_path,
        subject_ids=args.subjects, patch_size=args.patch_size, normalize=args.normalize, normalize_per_channel=args.normalize_per_channel)
    sample = temp_dataset[0]
    (C, A, T) = sample['eeg'].shape
    print(f'Data shape: C={C}, A={A}, T={T}')
    all_results = []
    all_train_losses = []
    all_val_losses = []
    error_log = []
    error_log_path = os.path.join(output_path, 'error_log.txt')
    from datetime import datetime
    start_time = datetime.now()
    for (idx, subject_id) in enumerate(unique_subject_ids, 1):
        print(f"\n\n{'#' * 80}")
        print(f'# Processing Subject {subject_id:02d} ({idx}/{len(unique_subject_ids)})')
        print(f"{'#' * 80}")
        existing_metrics = check_and_load_existing(output_path, subject_id)
        if existing_metrics:
            print(f'  >>>> Subject {subject_id:02d} already processed. Loading existing results and skipping training.')
            all_results.append(existing_metrics)
            continue
        sample = temp_dataset[0]
        (C, A, T) = sample['eeg'].shape
        if args.dataset == 'cogbci':
            (in_chans, ch_names, chan_number) = get_cogbci_dataset(args, subject_id)
        else:
            (in_chans, ch_names, chan_number) = get_dataset_name(args)
        valid_indices = None
        from src.eegpt import CHANNEL_DICT
        valid_indices_list = []
        valid_ch_names = []
        for (i, name) in enumerate(ch_names):
            clean_name = name.upper().strip('.')
            if clean_name in CHANNEL_DICT:
                valid_indices_list.append(i)
                valid_ch_names.append(name)
            else:
                print(f'not in dict')
        ch_names = valid_ch_names
        valid_indices = torch.tensor(valid_indices_list).long().to(device)
        from src.eegpt import EEGTransformer
        from functools import partial
        current_num_channels = len(ch_names)
        encoder_group = EEGTransformer(img_size=[current_num_channels, 200], patch_size=64, in_chans=1, embed_dim=512, depth=8, num_heads=8, embed_num=4, mlp_ratio=4.0, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-06)).to(device)
        pretrain_ckpt = torch.load(args.pretrained_path, map_location='cpu', weights_only=False)
        target_encoder_stat = {}
        for (k, v) in pretrain_ckpt['state_dict'].items():
            if k.startswith('target_encoder.'):
                target_encoder_stat[k[15:]] = v
        encoder_group.load_state_dict(target_encoder_stat)
        (metrics, losses) = train_for_subject_embedding_with_dynamics(args, subject_id, device, A, temp_dataset, encoder_group)
        if losses:
            all_train_losses.extend(losses['train_losses'])
            all_val_losses.extend(losses['val_losses'])
        if metrics is not None:
            all_results.append(metrics)
        else:
            error_msg = f'Subject {subject_id:02d}: Training failed'
            error_log.append(error_msg)
            with open(error_log_path, 'a') as f:
                f.write(f'{datetime.now()}: {error_msg}\n')
    end_time = datetime.now()
    duration = end_time - start_time
    print('\n\n' + '=' * 80)
    print('Generating Summary Results')
    print('=' * 80)
    import matplotlib.pyplot as plt
    import pandas as pd
    plt.figure(figsize=(10, 6))
    plt.plot(all_train_losses, label='Train Loss', color='blue')
    plt.plot(all_val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_path, 'loss_curve.png'))
    plt.close()
    if len(all_results) > 0:
        overall_stats = {'total_subjects': len(all_results), 'failed_subjects': len(error_log), 'duration_seconds': duration.total_seconds()}
        with open(os.path.join(args.output_root, f'overall_statistics.json'), 'w') as f:
            json.dump(overall_stats, f, indent=2)
    else:
        print('✗ No successful results to summarize')
    if len(error_log) > 0:
        print('\n' + '=' * 80)
        print('Error Summary')
        print('=' * 80)
        for error in error_log:
            print(f'  {error}')
    print('\n' + '=' * 80)
    print(f'Total Duration: {duration}')
    print('✓ Automated Full Fine-tuning Reconstruction V2-Fixed Completed!')
    print('=' * 80)
    print('V2-Fixed Applied Fixes:')
    print('  1. ✅ Mixed Loss (MSE + Variance + Spectral)')
    print('  2. ✅ Enhanced decoder (1024 hidden, residual connections)')
    print('  3. ✅ No dropout (0.0)')
    print('  4. ✅ Monitoring metrics (Std Ratio, High-Freq Ratio)')
    print('=' * 80)

    if error_log:
        raise RuntimeError(f"Training failed for {len(error_log)} subjects; see {error_log_path}")

if __name__ == '__main__':
    args = get_args()
    main(args)

