"""Original CogBCI preprocessing with explicit paths and session selection."""
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import mne
import numpy as np
import pandas as pd
import scipy.io as sio
import h5py
import warnings
import gc
import joblib
import multiprocessing
import contextlib
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

RAW_DIR = Path(".")
OUTPUT_DIR = Path(".")
SESSION = 1
N_JOBS = 4

TARGET_SFREQ = 200.0
FILTER_NOTCH = 50.0
FILTER_LOW = 0.1
FILTER_HIGH = 40.0
EPOCH_DURATION = 1.0
TARGET_N_CHANNELS = 62

STANDARD_CHANNELS_V1 = [
    'Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'CP5',
    'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10',
    'CP6', 'CP2', 'FCz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8',
    'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5',
    'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6',
    'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8',
    'AF4', 'F2'
]

STANDARD_CHANNELS_V2 = [
    'Fp1', 'Fz', 'F3', 'F7', 'F9', 'FC5', 'FC1', 'C3', 'T7', 'CP1',
    'Pz', 'P3', 'P7', 'P9', 'O1', 'Oz', 'O2', 'P10', 'P8', 'P4',
    'CP2', 'CP6', 'T8', 'C4', 'FC2', 'FC6', 'F10', 'F8', 'F4', 'Fp2',
    'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7',
    'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2',
    'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4',
    'F2', 'FCz'
]

RESTING_MAP = {'RS_Beg_EC': 0, 'RS_End_Ec': 0, 'RS_Beg_EO': 1, 'RS_End_EO': 1}
NBACK_MAP = {'zeroBACK': 0, 'oneBACK': 1, 'twoBACK': 2}
MATB_MAP = {'MATBeasy': 0, 'MATBmed': 1, 'MATBdiff': 2}


def preprocess_raw_pipeline(file_path):
    """
    Load -> Check Montage (V1 or V2) -> Clean -> Filter -> Resample
    Returns: (preprocessed_raw, montage_id)
             montage_id: 0 for V1, 1 for V2.
    """
    try:

        raw = mne.io.read_raw_eeglab(str(file_path), preload=True, verbose=False)
        current_chs = set(raw.ch_names)

        target_list = None
        montage_id = -1

        missing_v1 = [ch for ch in STANDARD_CHANNELS_V1 if ch not in current_chs]
        if not missing_v1:
            target_list = STANDARD_CHANNELS_V1
            montage_id = 0
        else:

            missing_v2 = [ch for ch in STANDARD_CHANNELS_V2 if ch not in current_chs]
            if not missing_v2:
                target_list = STANDARD_CHANNELS_V2
                montage_id = 1

        if target_list is None:
            return None, None

        raw.pick_channels(target_list)
        raw.reorder_channels(target_list)

        try:
            raw.set_montage('standard_1020', on_missing='ignore')
        except:
            pass

        if FILTER_NOTCH > 0:
            raw.notch_filter(
                freqs=FILTER_NOTCH,
                notch_widths=4.0,
                method='fir',
                verbose=False
            )

        raw.filter(l_freq=FILTER_LOW, h_freq=FILTER_HIGH, method='fir', verbose=False)

        if raw.info['sfreq'] != TARGET_SFREQ:
            raw.resample(TARGET_SFREQ, npad='auto', verbose=False)

        return raw, montage_id

    except Exception:
        return None, None


def extract_epochs_continuous(raw, label_val):
    try:
        epochs = mne.make_fixed_length_epochs(
            raw, duration=EPOCH_DURATION, preload=True, overlap=0.0, verbose=False
        )
        data = epochs.get_data().astype(np.float32)
        n_samples = len(data)
        if n_samples == 0: return None, None

        global_mean = np.mean(data)
        global_std = np.std(data)

        if global_std < 1e-8:
            global_std = 1.0

        data = (data - global_mean) / global_std

        labels = np.full(n_samples, label_val, dtype=np.float32)

        segments = np.arange(n_samples, dtype=np.int32)

        return data, labels, segments
    except:
        return None, None


def extract_epochs_event_locked(raw, events, event_id_map):
    try:
        epochs = mne.Epochs(
            raw, events, event_id=event_id_map,
            tmin=0.0, tmax=EPOCH_DURATION - (1/TARGET_SFREQ),
            baseline=None, preload=True, verbose=False
        )
        data = epochs.get_data().astype(np.float32)

        if len(data) == 0: return None, None

        current_codes = epochs.events[:, 2]
        labels = np.zeros(len(data), dtype=np.float32)

        for lbl_str, code_int in event_id_map.items():
            mask = current_codes == code_int
            labels[mask] = float(lbl_str)

        return data, labels
    except:
        return None, None


def process_file_wrapper(file_idx, file_path, task_type):
    """
    Returns: (eeg_data, labels, sub_id, ses_id, montage_id)
    """
    warnings.filterwarnings("ignore")
    mne.set_log_level('ERROR')

    try:
        fname = file_path.stem
        parts = file_path.parts

        try:
            sub_part = [p for p in parts if 'sub-' in p][-1]
            sub_id = int(sub_part.replace('sub-', ''))
            ses_part = [p for p in parts if 'ses-' in p][0]
            ses_id = int(ses_part.replace('ses-S', '').replace('ses-', ''))
        except:
            return None

        raw, montage_id = None, -1

        is_target = False
        if task_type == 'Resting':
            for k in RESTING_MAP:
                if k in fname: is_target = True; break
        elif task_type == 'NBack':
            for k in NBACK_MAP:
                if k in fname: is_target = True; break
        elif task_type == 'MATB':
            for k in MATB_MAP:
                if k in fname: is_target = True; break

        if not is_target: return None

        if task_type == 'Resting':
            label = RESTING_MAP.get(next(k for k in RESTING_MAP if k in fname))

            raw, montage_id = preprocess_raw_pipeline(file_path)
            if raw is None: return None

            data, lbls, segments = extract_epochs_continuous(raw, label)

        elif task_type == 'NBack':
            label = NBACK_MAP.get(next(k for k in NBACK_MAP if k in fname))

            raw, montage_id = preprocess_raw_pipeline(file_path)
            if raw is None: return None

            data, lbls, segments = extract_epochs_continuous(raw, label)

        elif task_type == 'MATB':

            key = next(k for k in MATB_MAP if k in fname)
            label = MATB_MAP.get(key)

            raw, montage_id = preprocess_raw_pipeline(file_path)
            if raw is None: return None

            data, lbls, segments = extract_epochs_continuous(raw, label)

        if data is None or len(data) == 0: return None

        n_samples = len(data)
        return (data, lbls,
                np.full(n_samples, sub_id, dtype=np.int32),
                np.full(n_samples, ses_id, dtype=np.int32),
                np.full(n_samples, montage_id, dtype=np.int32),
                segments,
                np.full(n_samples, file_idx, dtype=np.int32))

    except Exception:
        return None

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


def run_task_pipeline(task_name, file_list):
    if not file_list:
        print(f"  [Skip] No files found for {task_name}")
        return

    print(f"\n🔥 Processing {task_name} ({len(file_list)} files)...")
    n_jobs = N_JOBS

    with tqdm_joblib(tqdm(total=len(file_list))) as pbar:

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_file_wrapper)(i, f, task_name) for i, f in enumerate(file_list)
        )

    results = [r for r in results if r is not None]
    if not results: return

    all_data = np.concatenate([r[0] for r in results], axis=0)

    df_records = pd.DataFrame({
        'label': np.concatenate([r[1] for r in results], axis=0),
        'subject': np.concatenate([r[2] for r in results], axis=0),
        'session': np.concatenate([r[3] for r in results], axis=0),
        'montage': np.concatenate([r[4] for r in results], axis=0),
        'segment_in_trial': np.concatenate([r[5] for r in results], axis=0),
        'file_idx': np.concatenate([r[6] for r in results], axis=0),
        'orig_index': np.arange(len(all_data))
    })

    df_records = df_records.sort_values(['subject', 'file_idx', 'segment_in_trial']).reset_index(drop=True)

    relative_times = np.zeros(len(df_records), dtype=np.float32)
    for subject_id in sorted(df_records['subject'].unique()):
        mask = df_records['subject'] == subject_id
        n_sub_samples = mask.sum()
        relative_times[mask] = np.linspace(0, 1, n_sub_samples)

    df_records['relative_time'] = relative_times

    sorted_indices = df_records['orig_index'].values
    all_data_sorted = all_data[sorted_indices]

    name = 'rest' if task_name == 'Resting' else task_name.lower()
    out_name = OUTPUT_DIR / f"{name}.h5"
    print(f"  📦 Saving {all_data_sorted.shape} samples to {out_name}...")

    with h5py.File(out_name, 'w') as f:
        f.create_dataset('eeg', data=all_data_sorted, compression='gzip', compression_opts=4)
        f.create_dataset('labels', data=df_records['label'].values)
        f.create_dataset('subject', data=df_records['subject'].values)
        f.create_dataset('session', data=df_records['session'].values)
        f.create_dataset('montage_type', data=df_records['montage'].values)
        f.create_dataset('segment_in_trial', data=df_records['segment_in_trial'].values)
        f.create_dataset('relative_time', data=df_records['relative_time'].values)
        f.create_dataset('file_idx', data=df_records['file_idx'].values)

        f.attrs['sfreq'] = TARGET_SFREQ
        f.attrs['n_channels'] = TARGET_N_CHANNELS
        f.attrs['ch_names_v1'] = np.array(STANDARD_CHANNELS_V1, dtype='S')
        f.attrs['ch_names_v2'] = np.array(STANDARD_CHANNELS_V2, dtype='S')
        f.attrs['description'] = "0=Standard V1 (sub-01), 1=Alternative V2 (sub-17)"

    del all_data, all_data_sorted, df_records, results
    gc.collect()


def main():
    global RAW_DIR, OUTPUT_DIR, SESSION, N_JOBS
    import argparse
    parser = argparse.ArgumentParser(description="Prepare CogBCI using the original preprocessing pipeline.")
    parser.add_argument('--raw-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--session', type=int, choices=[1, 2, 3], default=1)
    parser.add_argument('--n-jobs', type=int, default=4)
    args = parser.parse_args()
    RAW_DIR, OUTPUT_DIR = Path(args.raw_dir), Path(args.output_dir)
    SESSION, N_JOBS = args.session, max(1, args.n_jobs)
    print("="*60)
    print("CogBCI Reprocessing Pipeline V2 (Dual Montage)")
    print("MODE: selected session Only + MATB (3 classes)")
    print("="*60)
    print(f"Input:  {RAW_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Scanning {RAW_DIR} for .set files...")
    all_files_total = sorted(list(RAW_DIR.rglob("*.set")))

    all_files = [f for f in all_files_total if f'ses-S{SESSION}' in f.parts]

    print(f"   Found {len(all_files)} files belonging to selected session")

    groups = {
        'Resting': [f for f in all_files if any(k in f.name for k in RESTING_MAP)],
        'NBack':   [f for f in all_files if any(k in f.name for k in NBACK_MAP)],
        'MATB':    [f for f in all_files if any(k in f.name for k in MATB_MAP)]
    }

    for task, files in groups.items():
        run_task_pipeline(task, files)

    print("\n✅ All tasks completed successfully.")

if __name__ == "__main__":
    main()

