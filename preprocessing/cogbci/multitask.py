"""Original CogBCI preprocessing with explicit paths and session selection."""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import mne
import numpy as np
import pandas as pd
import h5py
import warnings
import contextlib
import joblib
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

RAW_DIR = Path(".")
OUTPUT_DIR = Path(".")
SESSION = 1
N_JOBS = 4

TARGET_SFREQ = 200.0
FILTER_NOTCH = 50.0
FILTER_LOW = 0.1
FILTER_HIGH = 45.0
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

TASK_CONFIGS = {
    "Resting": {
        "task_id": 0,
        "map": {
            'RS_Beg_EC': 0, 'RS_Beg_Ec': 0,
            'RS_End_EC': 1, 'RS_End_Ec': 1,
            'RS_Beg_EO': 2,
            'RS_End_EO': 3
        }
    },
    "N-Back": {
        "task_id": 1,
        "map": {'zeroBACK': 0, 'oneBACK': 1, 'twoBACK': 2}
    },
    "MATB": {
        "task_id": 2,
        "map": {'MATBeasy': 0, 'MATBmed': 1, 'MATBdiff': 2}
    }
}


def preprocess_raw_pipeline(file_path):
    try:
        raw = mne.io.read_raw_eeglab(str(file_path), preload=True, verbose=False)
        current_chs = set(raw.ch_names)

        target_list = None
        montage_id = -1

        if not [ch for ch in STANDARD_CHANNELS_V1 if ch not in current_chs]:
            target_list = STANDARD_CHANNELS_V1
            montage_id = 0
        elif not [ch for ch in STANDARD_CHANNELS_V2 if ch not in current_chs]:
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
            raw.notch_filter(freqs=FILTER_NOTCH, notch_widths=2.0, verbose=False)

        raw.filter(l_freq=FILTER_LOW, h_freq=FILTER_HIGH, method='fir', verbose=False)

        if raw.info['sfreq'] != TARGET_SFREQ:
            raw.resample(TARGET_SFREQ, npad='auto', verbose=False)

        return raw, montage_id
    except Exception as e:
        return None, None


def extract_epochs_continuous(raw):
    try:
        epochs = mne.make_fixed_length_epochs(
            raw, duration=EPOCH_DURATION, preload=True, overlap=0.0, verbose=False
        )
        data = epochs.get_data().astype(np.float32)

        global_mean = np.mean(data)
        global_std = np.std(data)
        if global_std < 1e-8:
            global_std = 1.0

        data = (data - global_mean) / global_std
        return data
    except:
        return None


def process_file_wrapper(file_path):
    file_path = Path(file_path)

    warnings.filterwarnings("ignore")
    mne.set_log_level('ERROR')

    fname = file_path.stem

    matched_task_name = None
    local_label = -1

    for task_name, config in TASK_CONFIGS.items():
        for keyword, label_val in config["map"].items():
            if keyword in fname:
                matched_task_name = task_name
                local_label = label_val
                break
        if matched_task_name:
            break

    if matched_task_name is None:
        return None

    try:
        parts = file_path.parts
        sub_id = int([p for p in parts if 'sub-' in p][-1].replace('sub-', ''))
        ses_part = [p for p in parts if 'ses-' in p][0]
        if f'ses-S{SESSION}' not in ses_part:
            return None
        ses_id = int(ses_part.replace('ses-S', '').replace('ses-', ''))
    except:
        return None

    raw, montage_id = preprocess_raw_pipeline(file_path)
    if raw is None:
        return None

    try:
        start_time = raw.info['meas_date']
        start_ts = start_time.timestamp() if start_time else os.path.getmtime(file_path)
    except:
        start_ts = os.path.getmtime(file_path)

    data = extract_epochs_continuous(raw)
    if data is None or len(data) == 0:
        return None

    n_samples = len(data)
    task_id = TASK_CONFIGS[matched_task_name]["task_id"]

    global_label = task_id * 10 + local_label

    return {
        'data': data,
        'labels': np.full(n_samples, global_label, dtype='int64'),
        'task': np.full(n_samples, task_id, dtype='int64'),
        'subject_ids': np.full(n_samples, sub_id, dtype='int64'),
        'sessions': np.full(n_samples, ses_id, dtype='int64'),
        'montage_type': np.full(n_samples, montage_id, dtype='int64'),
        'start_ts': np.full(n_samples, start_ts, dtype='float64'),
        'segment_idx': np.arange(n_samples, dtype='int64')
    }

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


def save_subset_h5(file_path, data_list, subset_name, description_str):
    if not data_list:
        print(f"❌ No data to save for {subset_name}.")
        return

    print(f"\n🔄 Merging {subset_name} ({len(data_list)} files)...")
    keys = data_list[0].keys()
    final_dict = {k: np.concatenate([r[k] for r in data_list], axis=0) for k in keys}

    df_sort = pd.DataFrame({
        'subject': final_dict['subject_ids'],
        'start_ts': final_dict['start_ts'],
        'segment_idx': final_dict['segment_idx'],
        'orig_idx': np.arange(len(final_dict['labels']))
    })
    df_sort = df_sort.sort_values(['subject', 'start_ts', 'segment_idx'])
    sort_idx = df_sort['orig_idx'].values

    for k in final_dict.keys():
        final_dict[k] = final_dict[k][sort_idx]

    relative_times = np.zeros(len(sort_idx), dtype=np.float32)
    unique_subs = np.unique(final_dict['subject_ids'])
    for sub in unique_subs:
        mask = final_dict['subject_ids'] == sub
        relative_times[mask] = np.linspace(0, 1, np.sum(mask)).astype(np.float32)

    final_dict['relative_time'] = relative_times
    del final_dict['start_ts']
    del final_dict['segment_idx']

    print(f"📦 Saving {subset_name} to {file_path.name}...")
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('eeg', data=final_dict['data'], compression='gzip', compression_opts=4)
        f.create_dataset('labels', data=final_dict['labels'])
        f.attrs['global_label_formula'] = "Task_ID * 10 + Local_Label"

        f.create_dataset('task', data=final_dict['task'])
        f.create_dataset('subject', data=final_dict['subject_ids'])
        f.create_dataset('session', data=final_dict['sessions'])
        f.create_dataset('montage_type', data=final_dict['montage_type'])
        f.create_dataset('relative_time', data=final_dict['relative_time'])

        f.attrs['sfreq'] = TARGET_SFREQ
        f.attrs['n_channels'] = TARGET_N_CHANNELS
        f.attrs['description'] = description_str

        label_str = []
        for name, cfg in TASK_CONFIGS.items():

            if (subset_name == "Resting" and cfg['task_id'] == 0) or \
               (subset_name == "Multitask" and cfg['task_id'] > 0):
                tid = cfg['task_id']
                mappings = ", ".join([f"{tid*10 + v}={k}" for k, v in cfg['map'].items()])
                label_str.append(f"{name} (ID {tid}): {mappings}")
        f.attrs['label_map'] = "; ".join(label_str)

    print(f"✅ {subset_name} Saved! Total Samples: {len(final_dict['data'])}")

    unique, counts = np.unique(final_dict['labels'], return_counts=True)
    for u, c in zip(unique, counts):
        print(f"   Label {u}: {c} samples")


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
    print("CogBCI Separation Pipeline (selected session)")
    print("Will create TWO separate HDF5 files:")
    print(" 1. Resting (Task 0) - WITH LABELS (0,1,2,3)")
    print(" 2. Multitask (Tasks 1 & 2) - WITH LABELS")
    print("="*60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Scanning {RAW_DIR}...")
    all_files = sorted(list(RAW_DIR.rglob("*.set")))

    target_files = []
    all_keywords = []

    for config in TASK_CONFIGS.values():
        if "keywords" in config:
            all_keywords.extend(config["keywords"])
        elif "map" in config:
            all_keywords.extend(config["map"].keys())

    for f in all_files:
        if f'ses-S{SESSION}' not in f.parts: continue
        if any(k in f.name for k in all_keywords):
            target_files.append(f)

    if not target_files:
        print("❌ No files found.")
        return

    n_jobs = N_JOBS
    print(f"🔥 Processing {len(target_files)} files with {n_jobs} cores (Memory Safe Mode)...")

    with tqdm_joblib(tqdm(total=len(target_files))) as pbar:
        results = Parallel(n_jobs=n_jobs)(

            delayed(process_file_wrapper)(str(f)) for f in target_files
        )

    valid_results = [r for r in results if r is not None]

    resting_data = [r for r in valid_results if r['task'][0] == 0]
    multitask_data = [r for r in valid_results if r['task'][0] > 0]
    suffix = '' if SESSION == 1 else f'_s{SESSION}'

    save_subset_h5(
        OUTPUT_DIR / f"rest_conditions{suffix}.h5",
        resting_data,
        "Resting",
        "CogBCI selected session Resting Only. GLOBAL Z-Scored, 0.1-45Hz, w/ Relative Time. (Labels 0,1,2,3)"
    )

    save_subset_h5(
        OUTPUT_DIR / f"multitask{suffix}.h5",
        multitask_data,
        "Multitask",
        "CogBCI selected session Multi-Task (N-Back, MATB). GLOBAL Z-Scored, 0.1-45Hz, w/ Relative Time."
    )

if __name__ == "__main__":
    main()
