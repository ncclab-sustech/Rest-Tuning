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

DATASET_ROOT = ""
OUTPUT_DIR = ""
TARGET_SR = 200

MOTOR_TASK_IDS = [11, 12, 13]
MOTOR_LABEL_MAPPING = {11: 0, 12: 1, 13: 2}

RUN_START_TRIAL = {3: 1, 6: 61, 10: 121, 13: 181}
RUN_LIST = [3, 6, 10, 13]

SOURCE_CODES = {'Enrollment': 1, 'Calibration': 2, 'Testing': 3}


def load_metadata_session1():
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

    df_motor = df_merged[df_merged['Task'].isin(MOTOR_TASK_IDS)].copy()
    df_motor = df_motor[df_motor['SessionInt'] == 1].copy()

    print(f"\n  Total Motor samples in Session 1: {len(df_motor)}")
    print(f"  Unique subjects: {df_motor['SubjectIDInt'].nunique()}")
    print(f"  Subject ID range: {df_motor['SubjectIDInt'].min()} - {df_motor['SubjectIDInt'].max()}")

    return df_motor


def parse_epoch_id(epoch_id):
    if pd.isna(epoch_id):
        return None, None
    epoch_id = str(epoch_id).strip()
    patterns = [
        r'sub\d+_run(\d+)_trial(\d+)',
        r'run(\d+)_trial(\d+)',
        r'run(\d+)_(\d+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, epoch_id, re.IGNORECASE)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None, None


def infer_run_and_trial_by_order(df_subject):
    df = df_subject.copy().sort_values('EpochID').reset_index(drop=True)
    n_samples = len(df)
    n_runs = len(RUN_LIST)

    base_count = n_samples // n_runs
    remainder = n_samples % n_runs

    run_ids = []
    trial_in_runs = []

    for i, run_id in enumerate(RUN_LIST):
        run_sample_count = base_count + (1 if i < remainder else 0)
        for j in range(run_sample_count):
            run_ids.append(run_id)
            trial_in_runs.append(j + 1)

    df['inferred_run'] = run_ids[:n_samples]
    df['inferred_trial_in_run'] = trial_in_runs[:n_samples]
    return df


def process_epoch(file_path, target_samples=200):
    try:
        mat_data = sio.loadmat(file_path)
        data_key = [k for k in mat_data.keys() if not k.startswith('__')][0]
        eeg_data = mat_data[data_key]

        if eeg_data.ndim == 2:
            if eeg_data.shape[0] > eeg_data.shape[1]:
                eeg_data = eeg_data.T
        elif eeg_data.ndim == 3:
            if eeg_data.shape[0] == 64:
                eeg_data = eeg_data[:, :, 0]
            else:
                eeg_data = eeg_data[0, :, :]

        eeg_data = eeg_data[:64, :]

        time_points = eeg_data.shape[1]
        estimated_sr = round(time_points / 4.0)
        common_srs = [1000, 500, 250, 200]
        original_sr = min(common_srs, key=lambda x: abs(x - estimated_sr))

        if original_sr != TARGET_SR:
            downsampled = resample_poly(eeg_data, up=TARGET_SR, down=original_sr, axis=-1)
        else:
            downsampled = eeg_data

        nyq = 0.5 * TARGET_SR

        b_notch, a_notch = iirnotch(w0=50.0, Q=30.0, fs=TARGET_SR)
        downsampled = filtfilt(b_notch, a_notch, downsampled, axis=-1)

        low = 0.1 / nyq
        high = 45.0 / nyq
        b_band, a_band = butter(N=4, Wn=[low, high], btype='bandpass')
        downsampled = filtfilt(b_band, a_band, downsampled, axis=-1)

        mean = np.mean(downsampled)
        std = np.std(downsampled)
        if std < 1e-8:
            std = 1.0
        downsampled = (downsampled - mean) / std

        n_total = downsampled.shape[1]
        n_segments = n_total // target_samples

        segments = []
        for i in range(n_segments):
            start = i * target_samples
            segment = downsampled[:, start:start+target_samples]
            if segment.shape[1] == target_samples:
                segments.append(segment.astype(np.float32))

        if len(segments) == 0:

            return [], 0

        return segments, len(segments)

    except Exception as e:
        print(f"\n    Error loading {file_path}: {e}")
        return [], 0


def process_motor_session1_continuous_time(df_metadata):
    print("\n" + "=" * 80)
    print("Processing Motor Session 1 with SAMPLE-LEVEL Continuous Time")
    print("=" * 80)

    subjects = sorted(df_metadata['SubjectIDInt'].unique())
    temp_records = []
    stats = defaultdict(lambda: {'total': 0, 'runs': {r:0 for r in RUN_LIST}})

    for subject_id in tqdm(subjects, desc="Processing subjects"):
        df_sub = df_metadata[df_metadata['SubjectIDInt'] == subject_id].copy()

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
            df_sub = infer_run_and_trial_by_order(df_sub)
        else:
            df_sub['inferred_run'] = df_sub['parsed_run'].fillna(method='ffill')
            df_sub['inferred_trial_in_run'] = df_sub['parsed_trial'].fillna(method='ffill')

        df_sub = df_sub.sort_values('EpochID').reset_index(drop=True)

        for _, row in df_sub.iterrows():
            file_path = os.path.join(DATASET_ROOT, row['SourceFolder'], f"{row['EpochID']}.mat")

            if not os.path.exists(file_path):
                continue

            segments, n_seg = process_epoch(file_path)
            if n_seg == 0:
                continue

            run_id = int(row['inferred_run'])
            trial_in_run = int(row['inferred_trial_in_run'])
            global_trial_id = RUN_START_TRIAL.get(run_id, 1) + trial_in_run - 1

            for seg_idx, seg_data in enumerate(segments):
                temp_records.append({
                    'data': seg_data,
                    'label': MOTOR_LABEL_MAPPING[row['Task']],
                    'subject_id': subject_id,
                    'trial_id': global_trial_id,
                    'run_id': run_id,
                    'segment_in_trial': seg_idx,
                    'session': row['SessionInt'],
                    'source': row['SourceCode'],
                    'usage': row.get('Usage', 0),
                })

                stats[subject_id]['total'] += 1
                stats[subject_id]['runs'][run_id] += 1

    df_records = pd.DataFrame(temp_records)
    print(f"\nTotal samples collected: {len(df_records)}")

    df_records = df_records.sort_values(
        ['subject_id', 'run_id', 'trial_id', 'segment_in_trial']
    ).reset_index(drop=True)

    relative_times = np.zeros(len(df_records), dtype=np.int32)

    for subject_id in df_records['subject_id'].unique():
        subject_mask = df_records['subject_id'] == subject_id
        indices = df_records[subject_mask].index.values
        n_samples = len(indices)

        time_array = np.arange(n_samples, dtype=np.int32)
        for i, idx in enumerate(indices):
            relative_times[idx] = time_array[i]

    results = {
        'data': np.stack(df_records['data'].values).astype(np.float32),
        'labels': df_records['label'].values.astype(np.int64),
        'subject_ids': df_records['subject_id'].values.astype(np.int64),
        'trial_ids': df_records['trial_id'].values.astype(np.int64),
        'run_ids': df_records['run_id'].values.astype(np.int8),
        'segment_in_trial': df_records['segment_in_trial'].values.astype(np.int8),
        'relative_time': df_records['relative_time'].values.astype(np.int32),
        'sessions': df_records['session'].values.astype(np.int64),
        'sources': df_records['source'].values.astype(np.int64),
        'usages': df_records['usage'].values.astype(np.int64),
        'stats': dict(stats)
    }
    return results


def save_and_validate(output_path, data_dict):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('data', data=data_dict['data'], dtype='float32', compression='gzip', compression_opts=4)
        f.create_dataset('labels', data=data_dict['labels'], dtype='int64')
        f.create_dataset('subject_ids', data=data_dict['subject_ids'], dtype='int64')
        f.create_dataset('trial_ids', data=data_dict['trial_ids'], dtype='int64')
        f.create_dataset('run_ids', data=data_dict['run_ids'], dtype='int8')
        f.create_dataset('segment_in_trial', data=data_dict['segment_in_trial'], dtype='int8')
        f.create_dataset('relative_time', data=data_dict['relative_time'], dtype='int32')
        f.create_dataset('sessions', data=data_dict['sessions'], dtype='int64')
        f.create_dataset('sources', data=data_dict['sources'], dtype=np.int64)
        f.create_dataset('usages', data=data_dict['usages'], dtype=np.int64)

        f.attrs['description'] = "M3CV Session 1 Motor with SAMPLE-LEVEL Continuous Time, 0.1-45Hz Filtered, 50Hz Notch"
        f.attrs['session'] = 1
        f.attrs['sr'] = TARGET_SR
        f.attrs['n_samples'] = len(data_dict['data'])
        f.attrs['n_subjects'] = len(np.unique(data_dict['subject_ids']))
        f.attrs['label_mapping'] = "FT=0, RH=1, LH=2"


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
    df = load_metadata_session1()
    data_dict = process_motor_session1_continuous_time(df)
    output = os.path.join(OUTPUT_DIR, "motor.h5")
    save_and_validate(output, data_dict)

if __name__ == "__main__":
    main()
