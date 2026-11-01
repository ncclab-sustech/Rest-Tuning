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
import io
import time
import glob
from collections import defaultdict, deque
import datetime
from timm.utils import get_state_dict
from pathlib import Path
import torch.distributed as dist
from torch import inf
import h5py
import pickle
from scipy.signal import resample
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


def int_list(value):
    """Parse the comma-separated task IDs used by the original shell scripts."""
    return [int(item.strip()) for item in value.split(',')]


def prepare_run(args):
    """Validate input paths and subject metadata before loading EEG."""
    models = ('labram_base_patch200_200', 'BIOT', 'CBraMod', 'EEGPT')
    if args.model not in models:
        raise ValueError(f"Supported models: {models}")
    if args.dataset not in ('m3cv', 'cogbci'):
        raise ValueError("Supported datasets: m3cv, cogbci")
    if not args.h5_path or not Path(args.h5_path).is_file():
        raise FileNotFoundError(f"Provide an existing --h5_path: {args.h5_path}")
    if min(args.train_size, args.val_size, args.test_size) <= 0 or sum((args.train_size, args.val_size, args.test_size)) > 1.000001:
        raise ValueError("Positive split fractions must sum to at most one")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if args.fitune_type not in ('full', 'lora', 'none'):
        raise ValueError("Supported Rest methods: full, lora; use a baseline entry for none")
    if hasattr(args, 'embedding_high_level'):
        if args.model == 'EEGPT' and not args.embedding_high_level:
            raise ValueError("EEGPT calibration requires --embedding_high_level")
        if args.model == 'BIOT' and args.embedding_high_level:
            raise ValueError("BIOT calibration uses pooled embeddings; omit --embedding_high_level")
        if args.train_count is not None and args.train_count < 1:
            raise ValueError("train_count must be positive")
    with h5py.File(args.h5_path, 'r') as f:
        data_key = next((k for k in ('data', 'eeg') if k in f), None)
        subject_key = next((k for k in ('subject_ids', 'subject', 'subject_id') if k in f), None)
        if data_key is None or subject_key is None or 'labels' not in f:
            raise ValueError("HDF5 requires data/eeg, labels, and subject_ids/subject/subject_id")
        shape = f[data_key].shape
        if len(shape) != 3 or shape[1] != (64 if args.dataset == 'm3cv' else 62):
            raise ValueError(f"Expected [N, {64 if args.dataset == 'm3cv' else 62}, T], got {shape}")
        subjects = f[subject_key][:]
        available = sorted(int(s) for s in np.unique(subjects))
        requested = args.subjects if args.subjects is not None else available
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"Subjects absent from HDF5: {missing}")
        args.subjects = sorted(set(requested))
        if any(f[k].shape != (shape[0],) for k in (subject_key, 'labels')):
            raise ValueError("Subject and label arrays must have shape [N]")
        if getattr(args, 'multi_task', False) and 'task' not in f:
            raise ValueError("Multitask data requires a task array")
        count = int(np.isin(subjects, args.subjects).sum())
        print(f"HDF5: {shape}, {f[data_key].dtype}; selected {count} samples for subjects {args.subjects}")
        print(f"EEG memory before patching: approximately {count * shape[1] * shape[2] * 4 / 2**20:.1f} MiB")
    if hasattr(args, 'pretrained_path'):
        if not args.pretrained_path or not Path(args.pretrained_path).is_file():
            raise FileNotFoundError(f"Provide an existing --pretrained_path: {args.pretrained_path}")
    else:
        if not args.recon_results_dir:
            raise ValueError("Provide --recon_results_dir from the Rest stage")
        for subject in args.subjects:
            checkpoint = Path(args.recon_results_dir) / args.dataset / f'subject_{subject:02d}' / 'best_model.pth'
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing Rest checkpoint: {checkpoint}")
    if args.device.startswith('cpu'):
        args.use_amp = False


def get_input_chans(ch_names):
    input_chans = [0]
    for ch_name in ch_names:
        input_chans.append(standard_1020.index(ch_name) + 1)
    return input_chans

standard_1020 = ['FP1', 'FPZ', 'FP2', 'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', \
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', \
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', \
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', \
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', \
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', \
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', \
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', \
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', \
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8', \
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', \
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h', \
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]


def get_cogbci_dataset(args, subject_id):
    v2_subjects = [17, 25, 26, 28, 29]

    if args.dataset == 'cogbci':
        if subject_id in v2_subjects:
            ch_names = [
                'Fp1', 'Fz', 'F3', 'F7', 'F9', 'FC5', 'FC1', 'C3', 'T7', 'CP1',
                'Pz', 'P3', 'P7', 'P9', 'O1', 'Oz', 'O2', 'P10', 'P8', 'P4',
                'CP2', 'CP6', 'T8', 'C4', 'FC2', 'FC6', 'F10', 'F8', 'F4', 'Fp2',
                'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7',
                'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2',
                'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4',
                'F2', 'FCz'
            ]
            print(f"Subject {subject_id} detected as V2")
        else:
            ch_names = [
                'Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'CP5',
                'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10',
                'CP6', 'CP2', 'FCz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8',
                'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5',
                'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6',
                'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8',
                'AF4', 'F2'
            ]
            print(f"  [Config] Subject {subject_id} using V1")

        ch_names_upper = [ch.upper() for ch in ch_names]
        in_chans = get_input_chans(ch_names_upper)
        chan_number = 62
        return in_chans, ch_names, chan_number

    else:
        return get_dataset_name(args)


def get_dataset_name(args):
    if args.dataset == 'swu':
        ch_names = ['Fp1', 'AF3', 'AF7', 'Fz', 'F1', 'F3', 'F5', 'F7', 'FC1', 'FC3',\
                    'FC5', 'FT7', 'Cz', 'C1', 'C3', 'C5', 'T7', 'CP1', 'CP3', 'CP5',\
                    'TP7', 'TP9', 'Pz', 'P1', 'P3', 'P5', 'P7', 'PO3', 'PO7', 'Oz',\
                    'O1', 'Fpz', 'Fp2', 'AF4', 'AF8', 'F2', 'F4', 'F6', 'F8', 'FC2',\
                    'FC4', 'FC6', 'FT8', 'C2', 'C4', 'C6', 'T8', 'CPz', 'CP2', 'CP4',\
                    'CP6', 'TP8', 'TP10', 'P2', 'P4', 'P6', 'P8', 'POz', 'PO4', 'PO8','O2']
        ch_names_upper = [ch_name.upper() for ch_name in ch_names]
        in_chans=get_input_chans(ch_names_upper)

        chan_number = 61

    elif args.dataset == 'm3cv':
        ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', \
                    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'FC1', \
                    'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'FT9', 'FT10', \
                    'TP9', 'TP10', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4', \
                    'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', \
                    'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', \
                    'Fpz', 'CPz', 'POz', 'Oz', 'FCz']
        ch_names_upper = [ch_name.upper() for ch_name in ch_names]
        in_chans=get_input_chans(ch_names_upper)
        chan_number = 64

    return in_chans, ch_names, chan_number


class BIOT_DataProjector:
    def __init__(self, target_chans=None):

        if target_chans is None:

            self.target_chans = [
                "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
                "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
                "C3-A2", "C4-A1"
            ]
        else:
            self.target_chans = target_chans

    def channel_project(self, data, use_channel_names):
        device = data.device

        if len(data.shape) == 4 and data.shape[2] == 1:

            data = data.squeeze(2)
        elif len(data.shape) == 4:

            batch_size, n_channel_source, patch_size, n_timepoints = data.shape
            data = data.reshape(batch_size, n_channel_source * patch_size, n_timepoints)

        if len(data.shape) == 3:
            batch_size, n_channel_source, n_timepoints = data.shape
            patch_size = 1
        else:
            batch_size, n_channel_source, patch_size, n_timepoints = data.shape

        target_chans_number = len(self.target_chans)

        source_ch_map = {name.upper(): idx for idx, name in enumerate(use_channel_names)}

        if patch_size == 1:

            result = torch.zeros(
                (batch_size, target_chans_number, n_timepoints),
                device=device,
                dtype=data.dtype
            )
        else:

            result = torch.zeros(
                (batch_size, target_chans_number, patch_size, n_timepoints),
                device=device,
                dtype=data.dtype
            )

        for target_idx, target_name in enumerate(self.target_chans):
            target_name_upper = target_name.upper()

            if '-' in target_name_upper:
                try:
                    ch1_name, ch2_name = target_name_upper.split('-')
                    if ch1_name in source_ch_map and ch2_name in source_ch_map:
                        idx1 = source_ch_map[ch1_name]
                        idx2 = source_ch_map[ch2_name]

                        if patch_size == 1:
                            result[:, target_idx, :] = data[:, idx1, :] - data[:, idx2, :]
                        else:
                            result[:, target_idx, :, :] = data[:, idx1, :, :] - data[:, idx2, :, :]
                        continue
                except ValueError:
                    pass

            if target_name_upper in source_ch_map:
                src_idx = source_ch_map[target_name_upper]
                if patch_size == 1:
                    result[:, target_idx, :] = data[:, src_idx, :]
                else:
                    result[:, target_idx, :, :] = data[:, src_idx, :, :]
                continue

        if len(result.shape) == 4 and result.shape[2] == 1:
            result = result.squeeze(2)

        return result


def load_vanilla_labram_weights(encoder, pretrained_path, model_key='model|module', model_prefix=''):
    """
    Load ORIGINAL LaBraM Base pretrained weights

    Handles:
    - model_key extraction (model|module)
    - student. prefix filtering
    - head mismatch
    - relative_position_index removal
    - time_embed truncation (16 -> 10)
    """
    print(f"  Loading LaBraM Base weights from: {pretrained_path}")

    checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)

    checkpoint_model = None
    for key in model_key.split('|'):
        if key in checkpoint:
            checkpoint_model = checkpoint[key]
            print(f"  ✓ Load state_dict by model_key = {key}")
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint

    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    for key in all_keys:
        if key.startswith('student.'):
            new_dict[key[8:]] = checkpoint_model[key]
    if len(new_dict) > 0:
        checkpoint_model = new_dict
        print(f"  ✓ Filtered 'student.' prefix from {len(new_dict)} keys")

    state_dict = encoder.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"  ⚠ Removing key {k} from checkpoint (shape mismatch)")
            del checkpoint_model[k]

    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            checkpoint_model.pop(key)

    if 'time_embed' in checkpoint_model:
        current_shape = encoder.time_embed.shape
        ckpt_shape = checkpoint_model['time_embed'].shape
        if ckpt_shape != current_shape:
            print(f"  ⚠ time_embed shape mismatch: checkpoint {ckpt_shape} vs current {current_shape}")
            print(f"  ✓ Truncating time_embed: {ckpt_shape} -> {current_shape}")
            checkpoint_model['time_embed'] = checkpoint_model['time_embed'][:, :current_shape[1], :]

    load_state_dict(encoder, checkpoint_model, prefix=model_prefix)
    print("  ✓ LaBraM Base weights loaded successfully!")


def check_and_load_existing(output_root, subject_id):
    subject_dir = os.path.join(output_root, f'subject_{subject_id:02d}')
    metrics_path = os.path.join(subject_dir, 'metrics.json')

    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"  ⚠ Found metrics.json for Subject {subject_id} but failed to load: {e}")
            return None
    return None


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


def save_args_to_json(args, filepath):
    args_dict = vars(args).copy()

    for key, value in args_dict.items():
        if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
            args_dict[key] = str(value)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(args_dict, f, indent=2, ensure_ascii=False)

    print(f"Parameters saved to: {filepath}")
    return filepath


def count_parameters_with_names(model):
    total_params = 0
    trainable_params = 0
    trainable_modules = []

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            trainable_modules.append(name)

    return total_params, trainable_params, trainable_modules


def calculate_parameter_percentage_with_names(model):
    total_params, trainable_params, trainable_modules = count_parameters_with_names(model)
    trainable_percentage = (trainable_params / total_params) * 100
    return total_params, trainable_params, trainable_percentage, trainable_modules


def create_lora_cls(model, args):
    lora_config = LoraConfig(
        r=args.cls_lora_r,
        lora_alpha=args.cls_lora_alpha,
        target_modules=find_all_linear_fullnames(model),
        lora_dropout=args.cls_lora_dropout
    )
    model = get_peft_model(model, lora_config)
    print("LoRA modules added to the model.")
    return model


def find_all_linear_fullnames(model):
    lora_module_names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if "lm_head" not in name:
                lora_module_names.append(name)
    print(lora_module_names)
    return list(lora_module_names)


def create_lora(model, args):
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=find_all_linear_fullnames(model),
        lora_dropout=args.lora_dropout,
        bias="all"
        )
    model = get_peft_model(model, lora_config)
    print("LoRA modules added to the model.")
    return model

