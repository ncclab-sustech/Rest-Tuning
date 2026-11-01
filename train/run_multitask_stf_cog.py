"""Run the original Rest-Tuning subject experiments."""
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from peft import get_peft_model, LoraConfig
from timm.models import create_model
from einops import rearrange
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import src.modeling_finetune
from src.data_loader import EEGDataset, create_single_subject_dataloaders
from src.utils import *
from src.model import *
from src.train import *


def target_modules_list(value):
    if isinstance(value, str):
        return [m.strip() for m in value.split(",")]
    return value


def get_args(argv=None):
    parser = argparse.ArgumentParser('Subject-level task adaptation', add_help=True)

    parser.add_argument('--fitune_type', default='full', type=str, choices=['full', 'sgd', 'lora', 'resnet','none', 'scratch'],
                        help='Pretrained method to transfer from')

    parser.add_argument('--recon_type', default='MAE', type=str, choices=['MAE', 'MLP'],
                    help='Pretrained method to transfer from')

    parser.add_argument('--model', default='labram_base_patch200_200', type=str)
    parser.add_argument('--embed_dim', default=200, type=int)
    parser.add_argument('--num_classes', default=3, type=int)

    parser.add_argument('--use_mean_pooling', action='store_true', default=True)

    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--disable_qkv_bias', action='store_false', dest='qkv_bias')
    parser.set_defaults(qkv_bias=True)
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true', default=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float)
    parser.add_argument('--init_scale', default=0.001, type=float)

    parser.add_argument('--residual_hidden_dim', default=512, type=int,
                        help='Hidden dimension for ResEEGNet')
    parser.add_argument('--residual_dropout', default=0.1, type=float,
                        help='V2-Fixed: Dropout rate for ResEEGNet (reduced from 0.2 to 0.1)')

    parser.add_argument('--h5_path',
                        default=None,
                        type=str)
    parser.add_argument('--test_size', default=0.2, type=float)
    parser.add_argument('--val_size', default=0.1, type=float)
    parser.add_argument('--train_size', default=0.7, type=float)
    parser.add_argument('--normalize', action='store_true', default=False)
    parser.add_argument('--normalize_per_channel', action='store_true', default=True)
    parser.add_argument('--dataset',  default='m3cv',type=str)

    parser.add_argument('--encoder_free', action='store_true', help="encoder free")
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--warmup_epochs', default=3, type=int)
    parser.add_argument('--patience', default=30, type=int)
    parser.add_argument('--label_smoothing', default=0.1, type=float)

    parser.add_argument('--recon_results_dir',
                        default=None,
                        type=str,
                        help='Root directory containing reconstruction results')

    parser.add_argument('--output_root',
                        default='results',
                        type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--use_amp', action='store_true', default=True)

    parser.add_argument('--mask_ratio', default=0.4, type=float)
    parser.add_argument('--mask_type', type=str, choices=['random', 'channel', 'hybrid','non_mask','none','patch'], help="The type of mask to apply")

    parser.add_argument('--lora_r', default=8, type=int)
    parser.add_argument('--lora_alpha', default=32, type=int)
    parser.add_argument('--lora_dropout', default=0.2, type=float)

    parser.add_argument('--target_modules',
                   default="blocks.4.attn.qkv, blocks.5.attn.qkv, blocks.6.attn.qkv",
                   type=target_modules_list)

    parser.add_argument('--min_samples_per_class', default=1, type=int)

    parser.add_argument('--fewshot', action='store_true', help="Deactivate few-shot learning")
    parser.add_argument('--n_shot', default=10, type=int)

    parser.add_argument('--baseline', action='store_true', help="Deactivate few-shot learning")

    parser.add_argument('--dataset_suffle', action='store_true', help="shuffle")
    parser.add_argument('--oversample', action='store_true', help="oversample")
    parser.add_argument('--cls_fitune_type', type=str, choices=['lora', 'full','linear', 'scratch', 'scratch_lora'], default='lora')
    parser.add_argument('--cls_lora_r', default=32, type=int)
    parser.add_argument('--cls_lora_alpha', default=8, type=int)
    parser.add_argument('--cls_lora_dropout', default=0.1, type=float)
    parser.add_argument('--cls_target_modules', default="blocks.8.attn.qkv, blocks.9.attn.qkv ", type=target_modules_list)
    parser.add_argument('--multitask_fewshot', action='store_true', help="Deactivate multitask few-shot learning")
    parser.add_argument('--multi_task',action='store_true')
    parser.add_argument('--multi_task_ids',default=[0,1],type=int_list)

    parser.add_argument('--subjects', type=int, nargs='+', help='Only load and train these subject IDs')
    parser.add_argument('--no_amp', action='store_false', dest='use_amp')
    return parser.parse_args(argv)


def main(args):
    prepare_run(args)
    if args.model == 'EEGPT':
        from src.train_eegpt import train_single_subject_cls_multi_task_cog
    else:
        from src.train import train_single_subject_cls_multi_task_cog
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    os.makedirs(args.output_root, exist_ok=True)

    print("="*80)
    print(f"Automated Transfer Learning Classification ")
    print(f"Method: {args.cls_fitune_type.upper()}")
    print("="*80)
    print(f"Source: Rest-Tuning encoder (or base encoder for baseline)")
    print(f"Target Task: Cognitive Classification")
    print(f"Data: {args.h5_path}")
    print(f"Rest-Tuning Results: {args.recon_results_dir}")
    print(f"Output: {args.output_root}")
    print("="*80)

    print("\nScanning subjects in dataset...")
    unique_subject_ids = np.asarray(args.subjects)
    print(f"Found {len(unique_subject_ids)} unique subjects: {unique_subject_ids}")

    print("\nLoading Cognitive Classification Dataset...")
    shared_dataset = EEGDataset(
        h5_path=args.h5_path,
        subject_ids=args.subjects,
        patch_size=200,
        normalize=args.normalize,
        normalize_per_channel=args.normalize_per_channel
    )
    sample = shared_dataset[0]
    C, A, T = sample['eeg'].shape
    print(f"Data shape: C={C}, A={A}, T={T}")

    all_results = []
    error_log = []
    output_path = os.path.join(args.output_root, args.dataset, args.cls_fitune_type)
    os.makedirs(output_path, exist_ok=True)

    args_filepath = os.path.join(output_path, 'experiment_args.json')
    save_args_to_json(args, args_filepath)

    error_log_path = os.path.join(output_path, 'error_log.txt')

    from datetime import datetime
    start_time = datetime.now()

    for idx, subject_id in enumerate(unique_subject_ids, 1):
        print(f"\n\n{'#'*80}")
        print(f"# Processing Subject {subject_id:02d} ({idx}/{len(unique_subject_ids)})")
        print(f"# Method: {args.cls_fitune_type.upper()}")
        print(f"{'#'*80}")

        metrics = train_single_subject_cls_multi_task_cog(args, subject_id, device, A, shared_dataset)

        if metrics is not None:
            all_results.append(metrics)
        else:
            error_msg = f"Subject {subject_id:02d}: Training failed"
            error_log.append(error_msg)
            with open(error_log_path, 'a') as f:
                f.write(f"{datetime.now()}: {error_msg}\n")

    end_time = datetime.now()
    duration = end_time - start_time

    print("\n\n" + "="*80)
    print("Generating Summary Results")
    print("="*80)

    if len(all_results) > 0:
        summary_df = pd.DataFrame(all_results)

        if 'train_time_sec' in summary_df.columns:
            mean_train_time = summary_df['train_time_sec'].mean()
            std_train_time = summary_df['train_time_sec'].std()
            print(f"【Average Train Time】: {mean_train_time:.2f} ± {std_train_time:.2f} sec")

        if 'test_time_sec' in summary_df.columns:
            mean_test_time = summary_df['test_time_sec'].mean()
            std_test_time = summary_df['test_time_sec'].std()
            print(f"【Average Test Inference Time】: {mean_test_time:.3f} ± {std_test_time:.3f} sec")

        if 'total_time_sec' in summary_df.columns:
            mean_total_time = summary_df['total_time_sec'].mean()
            std_total_time = summary_df['total_time_sec'].std()
            print(f"【Average Total Time】: {mean_total_time:.2f} ± {std_total_time:.2f} sec")

        if 'flops_g' in summary_df.columns:
            mean_flops = summary_df['flops_g'].mean()
            print(f"【FLOPs】: {mean_flops:.3f} GFLOPs")

        if 'params_m' in summary_df.columns:
            mean_params = summary_df['params_m'].mean()
            print(f"【Params】: {mean_params:.3f} M")

        summary_csv_path = os.path.join(output_path,f'transfer_summary_{args.cls_fitune_type}.csv')

        summary_df.to_csv(summary_csv_path, index=False)
        print(f"✓ Summary saved to: {summary_csv_path}")

        mean_test_acc = summary_df['test_acc'].mean()
        std_test_acc = summary_df['test_acc'].std()
        mean_test_f1 = summary_df['test_f1'].mean()
        std_test_f1 = summary_df['test_f1'].std()

        print("\n" + "="*80)
        print(f"Overall Statistics (All Subjects) - Method: {args.cls_fitune_type.upper()}")
        print("="*80)
        print(f"Total subjects processed: {len(all_results)}/{len(unique_subject_ids)}")
        print(f"Failed subjects: {len(error_log)}")
        print(f"\n【Average Test Accuracy】: {mean_test_acc:.4f} ± {std_test_acc:.4f}")
        print(f"【Average Test F1-Score】: {mean_test_f1:.4f} ± {std_test_f1:.4f}")
        print("="*80)
        overall_stats = {
            'method': args.cls_fitune_type,
            'total_subjects': len(all_results),
            'failed_subjects': len(error_log),
            'mean_test_acc': float(mean_test_acc),
            'std_test_acc': float(std_test_acc),
            'mean_test_f1': float(mean_test_f1),
            'std_test_f1': float(std_test_f1),

            'mean_train_time_sec': float(mean_train_time) if 'train_time_sec' in summary_df.columns else None,
            'std_train_time_sec': float(std_train_time) if 'train_time_sec' in summary_df.columns else None,
            'mean_test_time_sec': float(mean_test_time) if 'test_time_sec' in summary_df.columns else None,
            'std_test_time_sec': float(std_test_time) if 'test_time_sec' in summary_df.columns else None,
            'mean_total_time_sec': float(mean_total_time) if 'total_time_sec' in summary_df.columns else None,
            'std_total_time_sec': float(std_total_time) if 'total_time_sec' in summary_df.columns else None,

            'flops_g': float(mean_flops) if 'flops_g' in summary_df.columns else None,
            'params_m': float(mean_params) if 'params_m' in summary_df.columns else None,

            'duration_seconds': duration.total_seconds(),
            'experiment_type': 'transfer_learning_linear_probing',
            'source_task': 'resting_reconstruction',
            'target_task': 'cognitive_classification'
        }

        with open(os.path.join(output_path, f'overall_statistics_{args.cls_fitune_type}.json'), 'w') as f:
            json.dump(overall_stats, f, indent=2)
    else:
        print("✗ No successful results to summarize")

    if len(error_log) > 0:
        print("\n" + "="*80)
        print("Error Summary")
        print("="*80)
        for error in error_log:
            print(f"  {error}")

    print("\n" + "="*80)
    print(f"Total Duration: {duration}")
    print(f"✓ Transfer Learning Classification ({args.cls_fitune_type.upper()}) Completed!")
    print("="*80)

    if error_log:
        raise RuntimeError(f"Training failed for {len(error_log)} subjects; see {error_log_path}")

if __name__ == '__main__':
    args = get_args()
    main(args)
