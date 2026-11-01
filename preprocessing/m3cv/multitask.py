"""Original M3CV preprocessing; paths are provided through the command line."""

import os
import re
import numpy as np
import pandas as pd
import h5py
import scipy.io as sio
from scipy.signal import resample_poly, butter, filtfilt, iirnotch
from tqdm import tqdm
from collections import defaultdict
import argparse

DATASET_ROOT = ""
OUTPUT_DIR = ""
OUTPUT_FILENAME = "multitask.h5"
TARGET_SR = 200

TASK_CONFIGS = {
    'motor': {
        'task_id': 1,
        'desc': 'Motor Execution (Runs 3,6,10,13)',
        'task_ids': [11, 12, 13],
        'label_mapping': {11: 0, 12: 1, 13: 2},
        'run_list': [3, 6, 10, 13],
        'run_start_trial': {3: 1, 6: 61, 10: 121, 13: 181},
        'single_segment_only': False,
    },
    'p300': {
        'task_id': 2,
        'desc': 'P300 Oddball (Run 7)',
        'task_ids': [9, 10],
        'label_mapping': {10: 0, 9: 1},
        'run_list': [7],
        'run_start_trial': {7: 1},
        'single_segment_only': True,
    },
    'ts': {
        'task_id': 3,
        'desc': 'Transient-State Sensory (Runs 4,11)',
        'task_ids': [3, 4, 5],
        'label_mapping': {3: 0, 4: 1, 5: 2},
        'run_list': [4, 11],
        'run_start_trial': {4: 1, 11: 91},
        'single_segment_only': True,
    },
    'sss': {
        'task_id': 4,
        'desc': 'Steady-State Sensory (Runs 5, 9, 12)',
        'task_ids': [6, 7, 8],
        'label_mapping': {6: 0, 7: 1, 8: 2},
        'run_list': [5, 9, 12],
        'run_start_trial': {5: 1, 9: 61, 12: 121},
        'single_segment_only': False,
    }
}

SOURCE_CODES = {'Enrollment': 1, 'Calibration': 2, 'Testing': 3}


def load_all_metadata_session1():
    print("=" * 80)
    print("Loading CSV Metadata for Session 1...")
    print("=" * 80)

    df_enroll = pd.read_csv(os.path.join(DATASET_ROOT, "Enrollment_Info.csv"))
    df_calib = pd.read_csv(os.path.join(DATASET_ROOT, "Calibration_Info.csv"))

    df_enroll['SourceFolder'] = 'Enrollment'
    df_enroll['SourceCode'] = SOURCE_CODES['Enrollment']
    df_enroll['SessionInt'] = 1

    df_calib['SourceFolder'] = 'Calibration'
    df_calib['SourceCode'] = SOURCE_CODES['Calibration']
    df_calib['SessionInt'] = 1

    df_merged = pd.concat([df_enroll, df_calib], ignore_index=True)

    df_merged['SubjectIDInt'] = pd.to_numeric(
        df_merged['SubjectID'].astype(str).str.replace('sub', '', regex=False),
        errors='coerce'
    )
    df_merged = df_merged.dropna(subset=['SubjectIDInt'])
    df_merged['SubjectIDInt'] = df_merged['SubjectIDInt'].astype(int)

    if 'Usage' not in df_merged.columns:
        df_merged['Usage'] = 0

    return df_merged[df_merged['SessionInt'] == 1].copy()


def parse_epoch_id(epoch_id):
    if pd.isna(epoch_id): return None, None
    epoch_id = str(epoch_id).strip()
    patterns = [r'sub\d+_run(\d+)_trial(\d+)', r'run(\d+)_trial(\d+)', r'run(\d+)_(\d+)']
    for pattern in patterns:
        match = re.search(pattern, epoch_id, re.IGNORECASE)
        if match: return int(match.group(1)), int(match.group(2))
    return None, None


def infer_run_and_trial_by_order(df_subject, run_list):
    df = df_subject.copy().sort_values('EpochID').reset_index(drop=True)
    n_samples = len(df)
    n_runs = len(run_list)

    base_count = n_samples // n_runs
    remainder = n_samples % n_runs
    run_ids, trial_in_runs = [], []

    for i, run_id in enumerate(run_list):
        run_sample_count = base_count + (1 if i < remainder else 0)
        for j in range(run_sample_count):
            run_ids.append(run_id)
            trial_in_runs.append(j + 1)

    df['inferred_run'] = run_ids[:n_samples]
    df['inferred_trial_in_run'] = trial_in_runs[:n_samples]
    return df


def process_epoch(file_path, single_segment_only, target_samples=200):
    try:
        mat_data = sio.loadmat(file_path)
        data_key = [k for k in mat_data.keys() if not k.startswith('__')][0]
        eeg_data = mat_data[data_key]

        if eeg_data.ndim == 2:
            if eeg_data.shape[0] > eeg_data.shape[1]: eeg_data = eeg_data.T
        elif eeg_data.ndim == 3:
            eeg_data = eeg_data[:, :, 0] if eeg_data.shape[0] == 64 else eeg_data[0, :, :]

        eeg_data = eeg_data[:64, :]

        time_points = eeg_data.shape[1]
        estimated_sr = round(time_points / 4.0)
        original_sr = min([1000, 500, 250, 200], key=lambda x: abs(x - estimated_sr))

        if original_sr != TARGET_SR:
            processed = resample_poly(eeg_data, up=TARGET_SR, down=original_sr, axis=-1)
        else:
            processed = eeg_data

        nyq = 0.5 * TARGET_SR
        b_notch, a_notch = iirnotch(w0=50.0, Q=30.0, fs=TARGET_SR)
        processed = filtfilt(b_notch, a_notch, processed, axis=-1)

        b_band, a_band = butter(N=4, Wn=[0.1 / nyq, 45.0 / nyq], btype='bandpass')
        processed = filtfilt(b_band, a_band, processed, axis=-1)

        global_mean = np.mean(processed)
        global_std = np.std(processed)
        if global_std < 1e-8:
            global_std = 1.0
        processed = (processed - global_mean) / global_std

        n_total = processed.shape[1]
        segments = []

        if single_segment_only:
            padded = np.zeros((64, target_samples), dtype=np.float32)
            available = min(n_total, target_samples)
            padded[:, :available] = processed[:, :available]
            segments.append(padded)
        else:
            n_segments = n_total // target_samples
            for i in range(n_segments):
                start = i * target_samples
                seg = processed[:, start:start+target_samples]
                if seg.shape[1] == target_samples:
                    segments.append(seg.astype(np.float32))

            if len(segments) == 0:
                padded = np.zeros((64, target_samples), dtype=np.float32)
                available = min(n_total, target_samples)
                padded[:, :available] = processed[:, :available]
                segments.append(padded)

        return segments, len(segments)

    except Exception as e:
        print(f"\n    Error loading {file_path}: {e}")
        return [], 0


def extract_task_records(task_name, df_all):
    config = TASK_CONFIGS[task_name]
    print(f"\n🚀 Extracting Task: [{task_name.upper()}] - {config['desc']}")

    df_task = df_all[df_all['Task'].isin(config['task_ids'])].copy()
    subjects = sorted(df_task['SubjectIDInt'].unique())
    print(f"Found {len(df_task)} samples across {len(subjects)} subjects.")

    temp_records = []

    for subject_id in tqdm(subjects, desc=f"{task_name.upper()}"):
        df_sub = df_task[df_task['SubjectIDInt'] == subject_id].copy()

        df_sub['parsed_run'] = None
        df_sub['parsed_trial'] = None
        has_parseable = False

        for idx, row in df_sub.iterrows():
            run_id, trial_id = parse_epoch_id(row['EpochID'])
            if run_id is not None:
                df_sub.at[idx, 'parsed_run'] = run_id
                df_sub.at[idx, 'parsed_trial'] = trial_id
                has_parseable = True

        if not has_parseable or df_sub['parsed_run'].isna().all():
            df_sub = infer_run_and_trial_by_order(df_sub, config['run_list'])
        else:
            df_sub['inferred_run'] = df_sub['parsed_run'].fillna(method='ffill')
            df_sub['inferred_trial_in_run'] = df_sub['parsed_trial'].fillna(method='ffill')

        df_sub = df_sub.sort_values('EpochID').reset_index(drop=True)

        for _, row in df_sub.iterrows():
            file_path = os.path.join(DATASET_ROOT, row['SourceFolder'], f"{row['EpochID']}.mat")
            if not os.path.exists(file_path): continue

            segments, n_seg = process_epoch(file_path, config['single_segment_only'])
            if n_seg == 0: continue

            run_id = int(row['inferred_run'])
            trial_in_run = int(row['inferred_trial_in_run'])
            global_trial_id = config['run_start_trial'].get(run_id, 1) + trial_in_run - 1

            local_label = config['label_mapping'].get(row['Task'], 0)
            global_label = config['task_id'] * 10 + local_label

            for seg_idx, seg_data in enumerate(segments):
                temp_records.append({
                    'data': seg_data,
                    'label': global_label,
                    'task': config['task_id'],
                    'subject_id': subject_id,
                    'trial_id': global_trial_id,
                    'run_id': run_id,
                    'segment_in_trial': seg_idx,
                    'session': row['SessionInt'],
                    'source': row['SourceCode'],
                    'usage': row.get('Usage', 0),
                })

    return temp_records


def merge_and_save_multitask(all_records, output_path):
    print("\n" + "=" * 80)
    print("⏳ Merging all tasks and computing GLOBAL Continuous Time...")
    print("=" * 80)

    df_records = pd.DataFrame(all_records)
    print(f"Total multi-task samples collected: {len(df_records)}")

    df_records = df_records.sort_values(
        ['subject_id', 'run_id', 'trial_id', 'segment_in_trial']
    ).reset_index(drop=True)

    relative_times = np.zeros(len(df_records), dtype=np.float32)

    for subject_id in df_records['subject_id'].unique():
        subject_mask = df_records['subject_id'] == subject_id
        indices = df_records[subject_mask].index.values
        n_samples = len(indices)

        time_array = np.linspace(0, 1, n_samples)
        for i, idx in enumerate(indices):
            relative_times[idx] = time_array[i]

    df_records['relative_time'] = relative_times

    print(f"📦 Saving to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        data_stack = np.stack(df_records['data'].values).astype(np.float32)
        f.create_dataset('data', data=data_stack, compression='gzip', compression_opts=4)

        f.create_dataset('labels', data=df_records['label'].values.astype(np.int64))
        f.create_dataset('task', data=df_records['task'].values.astype(np.int64))
        f.create_dataset('subject_ids', data=df_records['subject_id'].values.astype(np.int64))
        f.create_dataset('trial_ids', data=df_records['trial_id'].values.astype(np.int64))
        f.create_dataset('run_ids', data=df_records['run_id'].values.astype(np.int8))
        f.create_dataset('segment_in_trial', data=df_records['segment_in_trial'].values.astype(np.int8))
        f.create_dataset('relative_time', data=df_records['relative_time'].values.astype(np.float32))
        f.create_dataset('sessions', data=df_records['session'].values.astype(np.int64))
        f.create_dataset('sources', data=df_records['source'].values.astype(np.int64))
        f.create_dataset('usages', data=df_records['usage'].values.astype(np.int64))

        f.attrs['description'] = "M3CV Session 1 MULTI-TASK (Motor, P300, TS, SSS) | 0.1-45Hz, 50Hz Notch, GLOBAL Z-Scored"
        f.attrs['session'] = 1
        f.attrs['sr'] = TARGET_SR
        f.attrs['n_samples'] = len(data_stack)
        f.attrs['global_label_formula'] = "Task_ID * 10 + Local_Label"

        label_str = []
        for name, cfg in TASK_CONFIGS.items():
            tid = cfg['task_id']
            mappings = ", ".join([f"{tid*10 + v}={k}" for k, v in cfg['label_mapping'].items()])
            label_str.append(f"{name.upper()} (Task {tid}): {mappings}")
        f.attrs['label_map'] = "; ".join(label_str)

    print(f"\n✅ Multi-task Dataset Saved Successfully!")

    print("\n📊 Task & Label Distribution:")
    unique_tasks, task_counts = np.unique(df_records['task'].values, return_counts=True)
    unique_labels, label_counts = np.unique(df_records['label'].values, return_counts=True)

    for t, tc in zip(unique_tasks, task_counts):
        print(f"  Task ID {t}: {tc} samples")
    print("-" * 20)
    for l, lc in zip(unique_labels, label_counts):
        print(f"  Global Label {l}: {lc} samples")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare M3CV session 1 using the original preprocessing pipeline.")
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def main():
    global DATASET_ROOT, OUTPUT_DIR
    args = parse_args()
    DATASET_ROOT, OUTPUT_DIR = args.dataset_root, args.output_dir
    print("=" * 80)
    print("M3CV Multi-Task Session 1 Preprocessing (Motor, P300, TS, SSS)")
    print("Features: Global Z-Score, 0.1-45Hz Filter, Cross-Task Temporal Baseline")
    print("=" * 80)

    df_all_metadata = load_all_metadata_session1()
    all_global_records = []

    for task_name in ['motor', 'p300', 'ts', 'sss']:
        task_records = extract_task_records(task_name, df_all_metadata)
        if task_records:
            all_global_records.extend(task_records)

    if not all_global_records:
        print("❌ Failed to extract any valid data.")
        return

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    merge_and_save_multitask(all_global_records, output_path)

if __name__ == "__main__":
    main()
