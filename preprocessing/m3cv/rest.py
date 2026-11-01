"""Original M3CV preprocessing; paths are provided through the command line."""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import h5py
import scipy.io as sio
from scipy.signal import resample_poly, butter, filtfilt, iirnotch
from tqdm import tqdm
from collections import defaultdict
import argparse
import multiprocessing
from joblib import Parallel, delayed
import contextlib
import joblib

DATASET_ROOT = ""
OUTPUT_DIR = ""
OUTPUT_FILENAME = "rest.h5"
TARGET_SR = 200

TASK_CONFIGS = {
    'resting': {
        'desc': 'Resting State (Runs 1, 2)',
        'task_ids': [1, 2],

        'label_mapping': {1: 0, 2: 2},
        'run_list': [1, 2],
        'run_start_trial': {1: 1, 2: 1},
        'single_segment_only': False,
    }
}

SOURCE_CODES = {'Enrollment': 1, 'Calibration': 2, 'Testing': 3}


def load_all_metadata_session1():
    print("=" * 80)
    print("Loading CSV Metadata for Session 1 (Resting Only)...")
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


def process_epoch(file_path, target_samples=None):
    if target_samples is None:
        target_samples = TARGET_SR

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
        b_notch, a_notch = iirnotch(w0=50.0, Q=50.0/2.0, fs=TARGET_SR)
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
        n_segments = n_total // target_samples
        for i in range(n_segments):
            start = i * target_samples
            seg = processed[:, start:start+target_samples]
            if seg.shape[1] == target_samples:
                segments.append(seg.astype(np.float32))

        if len(segments) == 0:

            return [], 0

        return segments, len(segments)

    except Exception as e:
        print(f"\n    Error loading {file_path}: {e}")
        return [], 0


def process_file_wrapper(row_tuple):
    idx, row = row_tuple
    file_path = os.path.join(DATASET_ROOT, row['SourceFolder'], f"{row['EpochID']}.mat")
    if not os.path.exists(file_path):
        return []

    segments, n_seg = process_epoch(file_path)
    if n_seg == 0:
        return []

    task_id = row['Task']
    label = TASK_CONFIGS['resting']['label_mapping'].get(task_id, -1)

    if label == -1:
        return []

    try:
        start_ts = os.path.getmtime(file_path)
    except:
        start_ts = 0.0

    records = []
    for seg_idx, seg_data in enumerate(segments):
        records.append({
            'data': seg_data,
            'label': label,
            'task': 0,
            'subject_id': row['SubjectIDInt'],
            'trial_id': 1,
            'run_id': row['Task'],
            'segment_in_trial': seg_idx,
            'session': row['SessionInt'],
            'source': row['SourceCode'],
            'usage': row.get('Usage', 0),
            'start_ts': start_ts,
            'orig_idx': idx
        })
    return records

@contextlib.contextmanager


def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def merge_and_save(all_records, output_path):
    print("\n" + "=" * 80)
    print("⏳ Merging and computing GLOBAL Continuous Time...")
    print("=" * 80)

    flat_records = [item for sublist in all_records for item in sublist]
    df_records = pd.DataFrame(flat_records)
    print(f"Total resting samples collected: {len(df_records)}")

    if df_records.empty:
        print("❌ No valid resting samples were collected; output file was not written.")
        return

    df_records = df_records.sort_values(
        ['subject_id', 'run_id', 'orig_idx', 'segment_in_trial']
    ).reset_index(drop=True)

    relative_times = np.zeros(len(df_records), dtype=np.int32)

    for subject_id in df_records['subject_id'].unique():
        subject_mask = df_records['subject_id'] == subject_id
        indices = df_records[subject_mask].index.values
        n_samples = len(indices)

        time_array = np.arange(n_samples, dtype=np.int32)
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
        f.create_dataset('relative_time', data=df_records['relative_time'].values.astype(np.int32))
        f.create_dataset('sessions', data=df_records['session'].values.astype(np.int64))
        f.create_dataset('sources', data=df_records['source'].values.astype(np.int64))
        f.create_dataset('usages', data=df_records['usage'].values.astype(np.int64))

        f.attrs['description'] = "M3CV Session 1 RESTING | 0.1-45Hz, 50Hz Notch (w=2.0), GLOBAL Z-Scored. Labels Aligned to CogBCI."
        f.attrs['session'] = 1
        f.attrs['sr'] = TARGET_SR
        f.attrs['n_samples'] = len(data_stack)
        f.attrs['label_map'] = "0: Beg_EC (Task 1), 2: Beg_EO (Task 2)"

    print(f"\n✅ M3CV Resting Dataset Saved Successfully!")

    unique_labels, label_counts = np.unique(df_records['label'].values, return_counts=True)
    print("\n📊 Label Distribution:")
    for l, lc in zip(unique_labels, label_counts):
        label_name = {0: "Beg_EC", 2: "Beg_EO"}.get(l, "Unknown")
        print(f"  Label {l} ({label_name}): {lc} samples")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess M3CV Session 1 resting-state epochs into a temporal HDF5 dataset."
    )
    parser.add_argument("--dataset-root", required=True,
                        help="Path to the M3CV dataset root containing Enrollment_Info.csv and Calibration_Info.csv.")
    parser.add_argument("--output-dir", required=True,
                        help="Directory where the HDF5 output will be written.")
    parser.add_argument("--output-filename", default=OUTPUT_FILENAME,
                        help="Name of the HDF5 output file.")
    parser.add_argument("--target-sr", type=int, default=TARGET_SR,
                        help="Target sampling rate and one-second segment length.")
    parser.add_argument("--n-jobs", type=int, default=4,
                        help="Number of parallel workers to use.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N metadata rows.")
    return parser.parse_args()


def main():
    global DATASET_ROOT, OUTPUT_DIR, OUTPUT_FILENAME, TARGET_SR

    args = parse_args()
    DATASET_ROOT = args.dataset_root
    OUTPUT_DIR = args.output_dir
    OUTPUT_FILENAME = args.output_filename
    TARGET_SR = args.target_sr

    print("=" * 80)
    print("M3CV Resting Session 1 Preprocessing")
    print("Features: Global Z-Score, 0.1-45Hz Filter, 50Hz Notch (w=2.0)")
    print("Label Output: 0 (Beg_EC) and 2 (Beg_EO)")
    print("=" * 80)
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Output: {os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)}")
    print(f"Target sampling rate: {TARGET_SR} Hz")
    print(f"Workers: {args.n_jobs}")

    df_all_metadata = load_all_metadata_session1()

    df_resting = df_all_metadata[df_all_metadata['Task'].isin(TASK_CONFIGS['resting']['task_ids'])].copy()
    if args.limit is not None:
        df_resting = df_resting.head(args.limit)
    print(f"Found {len(df_resting)} resting state files to process.")

    if len(df_resting) == 0:
        print("❌ Failed to find any resting data.")
        return


    n_jobs = max(1, args.n_jobs)
    print(f"🔥 Processing with {n_jobs} cores...")

    rows_to_process = list(df_resting.iterrows())

    with tqdm_joblib(tqdm(total=len(rows_to_process))) as pbar:
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_file_wrapper)(row) for row in rows_to_process
        )

    valid_records = [r for r in results if len(r) > 0]

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    merge_and_save(valid_records, output_path)

if __name__ == "__main__":
    main()
