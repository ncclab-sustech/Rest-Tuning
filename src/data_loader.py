import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import h5py
from scipy import signal
from sklearn.model_selection import train_test_split, GroupKFold
from typing import Tuple, List


class EEGDataset(Dataset):
    """
    Dataset for EEG data with automatic resampling, Z-score normalization and patch reshaping.

    Features:
    - Automatic resampling: 250Hz -> 200Hz for LaBraM compatibility
    - Z-score normalization (per-channel or global)
    - Patch reshaping for transformer models

    Args:
        h5_path: Path to H5 file containing EEG data
        patch_size: Size of each patch for reshaping (default: 200)
        normalize: Whether to apply Z-score normalization (default: True)
        normalize_per_channel: If True, normalize each channel independently;
                               if False, use global normalization (default: True)
        transform: Optional transform to apply to EEG data
    """

    def __init__(
        self,
        h5_path: str,
        patch_size: int = 200,
        normalize: bool = True,
        normalize_per_channel: bool = True,
        transform=None,
        subject_ids=None
    ):
        self.h5_path = h5_path
        self.patch_size = patch_size
        self.normalize = normalize
        self.normalize_per_channel = normalize_per_channel
        self.transform = transform
        self.selected_subject_ids = subject_ids

        self._load_data()

    def _load_data(self):
        """
        Load EEG data, labels, and subject IDs from H5 file.

        Processing steps:
        1. Load data from H5 file
        2. Resample if needed (250Hz -> 200Hz)
        3. Apply Z-score normalization
        4. Verify normalization
        """
        print(f"Loading EEG data from: {self.h5_path}")

        with h5py.File(self.h5_path, 'r') as f:

            print(f"  Available keys in H5 file: {list(f.keys())}")

            subject_key = next((k for k in ('subject_ids', 'subject', 'subject_id') if k in f), None)
            if subject_key is None:
                raise KeyError("Expected subject_ids, subject, or subject_id in HDF5")
            all_subject_ids = f[subject_key][:]
            rows = np.flatnonzero(np.isin(all_subject_ids, self.selected_subject_ids)) if self.selected_subject_ids is not None else np.arange(len(all_subject_ids))
            if len(rows) == 0:
                raise ValueError("No samples match the selected subjects")
            if self.normalize and len(rows) != len(all_subject_ids):
                raise ValueError("Subset loading requires pre-normalized data (normalize=False) to preserve the original full-dataset statistics")
            self.row_indices = rows
            selection = slice(None) if len(rows) == len(all_subject_ids) else rows
            data_key = 'data' if 'data' in f else 'eeg'
            if data_key not in f or 'labels' not in f:
                raise KeyError("Expected data/eeg and labels in HDF5")
            self.data = f[data_key][selection].astype(np.float32)
            self.labels = torch.as_tensor(f['labels'][selection], dtype=torch.long)
            self.subject_ids = all_subject_ids[rows]
            if 'task' in f:
                self.tasks = f['task'][selection]
            self.montage_types = f['montage_type'][selection] if 'montage_type' in f else None

            if hasattr(f, 'attrs'):
                print(f"  Metadata:")
                for key, value in f.attrs.items():
                    print(f"    {key}: {value}")

        print(f"\n  Data shapes:")
        print(f"    EEG data: {self.data.shape} (Trials, Channels, Time)")
        print(f"    Labels: {self.labels.shape}")
        print(f"    Subject IDs: {self.subject_ids.shape}")
        print(f"\n  Statistics (Before Resampling):")
        print(f"    Number of trials: {len(self.data)}")
        print(f"    Unique subjects: {len(np.unique(self.subject_ids))}")
        print(f"    Label distribution: {dict(zip(*np.unique(self.labels, return_counts=True)))}")
        print(f"    Data range: [{self.data.min():.4f}, {self.data.max():.4f}]")
        print(f"    Global mean: {self.data.mean():.4f}, Global std: {self.data.std():.4f}")

        n_trials, n_channels, n_timepoints = self.data.shape
        if n_timepoints == 250:
            print(f"\n  Resampling detected: Time dimension is 250")
            print(f"    Resampling from 250Hz to 200Hz using scipy.signal.resample...")

            resampled_data = np.zeros((n_trials, n_channels, 200), dtype=np.float32)

            for i in range(n_trials):
                for ch in range(n_channels):

                    resampled_data[i, ch, :] = signal.resample(self.data[i, ch, :], 200)

            self.data = resampled_data

            print(f"    ✓ Resampling completed!")
            print(f"    New shape: {self.data.shape} (Trials, Channels, Time)")
            print(f"    Data range after resampling: [{self.data.min():.4f}, {self.data.max():.4f}]")
        elif n_timepoints == 200:
            print(f"\n  No resampling needed: Time dimension is already 200")
        elif n_timepoints % self.patch_size != 0:
            print(f"\n  ⚠ Warning: Time dimension ({n_timepoints}) is not 250 or divisible by patch_size ({self.patch_size})")
            print(f"    Attempting to resample to nearest multiple of patch_size...")

            target_length = (n_timepoints // self.patch_size) * self.patch_size
            if target_length == 0:
                target_length = self.patch_size

            print(f"    Resampling from {n_timepoints} to {target_length} timepoints...")

            resampled_data = np.zeros((n_trials, n_channels, target_length), dtype=np.float32)
            for i in range(n_trials):
                for ch in range(n_channels):
                    resampled_data[i, ch, :] = signal.resample(self.data[i, ch, :], target_length)

            self.data = resampled_data
            print(f"    ✓ Resampling completed!")
            print(f"    New shape: {self.data.shape}")

        if self.normalize:
            print(f"\n  Applying Z-score normalization...")
            if self.normalize_per_channel:

                print(f"    Method: Per-channel normalization")

                mean = self.data.mean(axis=(0, 2), keepdims=True)
                std = self.data.std(axis=(0, 2), keepdims=True)

                std = np.where(std == 0, 1.0, std)

                self.data = (self.data - mean) / std

                print(f"    Per-channel mean shape: {mean.shape}")
                print(f"    Per-channel std shape: {std.shape}")
            else:

                print(f"    Method: Global normalization")
                mean = self.data.mean()
                std = self.data.std()

                if std == 0:
                    std = 1.0

                self.data = (self.data - mean) / std

                print(f"    Global mean: {mean:.4f}")
                print(f"    Global std: {std:.4f}")

            print(f"\n  Normalization Verification:")
            normalized_mean = self.data.mean()
            normalized_std = self.data.std()
            print(f"    Normalized data mean: {normalized_mean:.6f}")
            print(f"    Normalized data std: {normalized_std:.6f}")
            print(f"    Normalized data range: [{self.data.min():.4f}, {self.data.max():.4f}]")

            mean_threshold = 1e-5
            std_threshold = 0.05

            if abs(normalized_mean) < mean_threshold:
                print(f"    ✓ Mean check passed: |{normalized_mean:.6f}| < {mean_threshold}")
            else:
                print(f"    ⚠ Warning: Mean check failed: |{normalized_mean:.6f}| >= {mean_threshold}")

            if abs(normalized_std - 1.0) < std_threshold:
                print(f"    ✓ Std check passed: |{normalized_std:.6f} - 1.0| < {std_threshold}")
            else:
                print(f"    ⚠ Warning: Std check failed: |{normalized_std:.6f} - 1.0| >= {std_threshold}")

            if self.normalize_per_channel:
                print(f"\n    Per-channel statistics:")
                for ch_idx in range(min(5, self.data.shape[1])):
                    ch_mean = self.data[:, ch_idx, :].mean()
                    ch_std = self.data[:, ch_idx, :].std()
                    print(f"      Channel {ch_idx}: mean={ch_mean:.6f}, std={ch_std:.6f}")
                if self.data.shape[1] > 5:
                    print(f"      ... (showing first 5 of {self.data.shape[1]} channels)")

        n_channels, n_timepoints = self.data.shape[1], self.data.shape[2]
        if n_timepoints % self.patch_size != 0:
            raise ValueError(
                f"Time dimension ({n_timepoints}) must be divisible by patch_size ({self.patch_size})"
            )
        self.num_patches = n_timepoints // self.patch_size
        print(f"\n  Patch configuration:")
        print(f"    Channels: {n_channels}")
        print(f"    Num patches: {self.num_patches}")
        print(f"    Patch size: {self.patch_size}")
        print(f"    Output shape per sample: ({n_channels}, {self.num_patches}, {self.patch_size})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            A dictionary containing:
                eeg: Tensor of shape (Channels, Num_Patches, Patch_Size)
                label: Label (0 or 1)
                subject_id: Subject identifier
        """
        eeg = self.data[idx]
        label = self.labels[idx]
        subject_id = self.subject_ids[idx]

        eeg = np.clip(eeg, -6.0, 6.0)

        if self.transform:
            eeg = self.transform(eeg)

        n_channels = eeg.shape[0]
        eeg_reshaped = eeg.reshape(n_channels, self.num_patches, self.patch_size)

        return {
            'eeg': torch.from_numpy(eeg_reshaped),
            'label': label,
            'subject_id': subject_id,
            'montage_type': (self.montage_types[idx]
                             if self.montage_types is not None else -1),
        }

    def get_subject_indices(self):
        """Get trial indices for each subject."""
        subject_to_indices = {}
        for idx, subj in enumerate(self.subject_ids):
            if subj not in subject_to_indices:
                subject_to_indices[subj] = []
            subject_to_indices[subj].append(idx)
        return subject_to_indices


def create_multitask_single_subject_dataloaders(
    dataset: EEGDataset,
    target_subject_id: int,
    target_task_id: list,
    train_size: float = 0.7,
    test_size: float = 0.2,
    val_size: float = 0.1,
    batch_size: int = 32,
    random_state: int = 42,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    mask = (dataset.subject_ids == target_subject_id)

    if target_task_id is not None:

        mask &= np.isin(dataset.tasks, target_task_id)

    all_indices = np.where(mask)[0]

    train_idx_list, val_idx_list, test_idx_list = [], [], []

    unique_labels = np.unique(dataset.labels[all_indices])

    print(f"\n[Random Split] Processing Subject: {target_subject_id}")

    for label in unique_labels:

        label_indices = all_indices[dataset.labels[all_indices] == label]

        np.random.seed(random_state)
        np.random.shuffle(label_indices)

        n_total = len(label_indices)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size)
        n_train = int(n_total * train_size)

        t_idx = label_indices[:n_train]
        v_idx = label_indices[n_train : n_train + n_val]
        if n_test+n_val+n_train >= n_total:
            s_idx = label_indices[n_train + n_val:]
        else:
            s_idx = label_indices[n_train + n_val:n_test+n_val+n_train]

        train_idx_list.extend(t_idx)
        val_idx_list.extend(v_idx)
        test_idx_list.extend(s_idx)

        print(f"  Class {label}: Total={n_total}, Train={len(t_idx)}, Val={len(v_idx)}, Test={len(s_idx)}")

    train_idx = np.array(train_idx_list)
    val_idx = np.array(val_idx_list)
    test_idx = np.array(test_idx_list)

    def get_dist(indices):

        labels = dataset.labels[indices]
        return dict(zip(*np.unique(labels, return_counts=True)))

    print(f"\nFinal Split Summary for Subject {target_subject_id}:")
    print(f"  Train Set: {len(train_idx)} trials, Dist: {get_dist(train_idx)}")
    print(f"  Val Set:   {len(val_idx)} trials, Dist: {get_dist(val_idx)}")
    print(f"  Test Set:  {len(test_idx)} trials, Dist: {get_dist(test_idx)}")

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def create_single_subject_dataloaders_order(
    dataset: EEGDataset,
    target_subject_id: int,
    train_size: float = 0.7,
    test_size: float = 0.2,
    val_size: float = 0.1,
    batch_size: int = 32,
    random_state: int = 42,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    all_indices = np.where(dataset.subject_ids == target_subject_id)[0]
    if len(all_indices) == 0:
        raise ValueError(f"Subject ID {target_subject_id} not found in dataset.")

    train_idx_list, val_idx_list, test_idx_list = [], [], []

    unique_labels = np.unique(dataset.labels[all_indices])

    print(f"\n[Within-Class Chronological Split] Processing Subject: {target_subject_id}")

    for label in unique_labels:

        label_indices = all_indices[dataset.labels[all_indices] == label]
        label_indices = np.sort(label_indices)

        n_total = len(label_indices)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size)

        n_train = int(n_total * train_size)

        t_idx = label_indices[:n_train]
        v_idx = label_indices[n_train : n_train + n_val]
        if n_test+n_val+n_train >= n_total:
            s_idx = label_indices[n_train + n_val:]
        else:
            s_idx = label_indices[n_train + n_val:n_test+n_val+n_train]

        train_idx_list.extend(t_idx)
        val_idx_list.extend(v_idx)
        test_idx_list.extend(s_idx)

        print(f"  Class {label}: Total={n_total}, Train={len(t_idx)}, Val={len(v_idx)}, Test={len(s_idx)}")

    train_idx = np.array(train_idx_list)
    val_idx = np.array(val_idx_list)
    test_idx = np.array(test_idx_list)

    def get_dist(indices):
        labels = dataset.labels[indices]
        return dict(zip(*np.unique(labels, return_counts=True)))

    print(f"\nFinal Split Summary for Subject {target_subject_id}:")
    print(f"  Train Set: {len(train_idx)} trials, Dist: {get_dist(train_idx)}")
    print(f"  Val Set:   {len(val_idx)} trials, Dist: {get_dist(val_idx)}")
    print(f"  Test Set:  {len(test_idx)} trials, Dist: {get_dist(test_idx)}")

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader

from torch.utils.data import DataLoader


def create_single_subject_dataloaders(
    dataset: EEGDataset,
    target_subject_id: int,
    train_size: float = 0.7,
    test_size: float = 0.2,
    val_size: float = 0.1,
    batch_size: int = 32,
    random_state: int = 42,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    all_indices = np.where(dataset.subject_ids == target_subject_id)[0]
    if len(all_indices) == 0:
        raise ValueError(f"Subject ID {target_subject_id} not found in dataset.")

    train_idx_list, val_idx_list, test_idx_list = [], [], []

    unique_labels = np.unique(dataset.labels[all_indices])

    print(f"\n[Random Split] Processing Subject: {target_subject_id}")

    for label in unique_labels:

        label_indices = all_indices[dataset.labels[all_indices] == label]

        np.random.seed(random_state)
        np.random.shuffle(label_indices)

        n_total = len(label_indices)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size)
        n_train = int(n_total * train_size)

        t_idx = label_indices[:n_train]
        v_idx = label_indices[n_train : n_train + n_val]
        if n_test+n_val+n_train >= n_total:
            s_idx = label_indices[n_train + n_val:]
        else:
            s_idx = label_indices[n_train + n_val:n_test+n_val+n_train]

        train_idx_list.extend(t_idx)
        val_idx_list.extend(v_idx)
        test_idx_list.extend(s_idx)

        print(f"  Class {label}: Total={n_total}, Train={len(t_idx)}, Val={len(v_idx)}, Test={len(s_idx)}")

    train_idx = np.array(train_idx_list)
    val_idx = np.array(val_idx_list)
    test_idx = np.array(test_idx_list)

    def get_dist(indices):
        labels = dataset.labels[indices]
        return dict(zip(*np.unique(labels, return_counts=True)))

    print(f"\nFinal Split Summary for Subject {target_subject_id}:")
    print(f"  Train Set: {len(train_idx)} trials, Dist: {get_dist(train_idx)}")
    print(f"  Val Set:   {len(val_idx)} trials, Dist: {get_dist(val_idx)}")
    print(f"  Test Set:  {len(test_idx)} trials, Dist: {get_dist(test_idx)}")

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def create_single_subject_few_shot_dataloaders(
    dataset: EEGDataset,
    target_subject_id: int,
    n_shot: int = 5,
    batch_size: int = 32,
    random_state: int = 42,
    num_workers: int = 4,
    val_size: float = 0,
    oversample: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    all_indices = np.where(dataset.subject_ids == target_subject_id)[0]
    if len(all_indices) == 0:
        raise ValueError(f"Subject ID {target_subject_id} not found in dataset.")

    label_indices = {label: all_indices[dataset.labels[all_indices] == label] for label in np.unique(dataset.labels[all_indices])}

    if oversample:

        class_counts = {label: len(indices) for label, indices in label_indices.items()}
        max_count = max(class_counts.values())

        for label, indices in label_indices.items():
            current_count = len(indices)
            if current_count < max_count:

                num_samples_to_add = max_count - current_count
                sampled_indices = np.random.choice(indices, num_samples_to_add, replace=True)
                label_indices[label] = np.concatenate([indices, sampled_indices])

    train_idx_list, val_idx_list, query_idx_list = [], [], []

    print(f"\n[Random Split] Processing Subject: {target_subject_id}")

    for label, indices in label_indices.items():

        np.random.seed(random_state)
        np.random.shuffle(indices)

        n_shot_samples = indices[:n_shot]

        n_val_samples = int(len(indices) * val_size)
        val_samples = indices[n_shot:n_shot + n_val_samples] if n_val_samples > 0 else []

        remaining_samples = indices[n_shot + n_val_samples:]

        train_idx_list.extend(n_shot_samples)

        val_idx_list.extend(val_samples)

        query_idx_list.extend(remaining_samples)

        print(f"  Class {label}: Total={len(indices)}, Support Set={n_shot}, Val Set={len(val_samples)}, Query Set={len(remaining_samples)}")

    train_idx = np.array(train_idx_list)
    val_idx = np.array(val_idx_list) if val_size > 0 else None
    query_idx = np.array(query_idx_list)

    def get_dist(indices):
        labels = dataset.labels[indices]
        return dict(zip(*np.unique(labels, return_counts=True)))

    print(f"\nFinal Split Summary for Subject {target_subject_id}:")
    print(f"  Train Set: {len(train_idx)} trials, Dist: {get_dist(train_idx)}")
    print(f"  Val Set:   {len(val_idx)} trials, Dist: {get_dist(val_idx)}" if val_idx is not None else "  Val Set:   None")
    print(f"  Query Set: {len(query_idx)} trials, Dist: {get_dist(query_idx)}")

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    ) if val_size > 0 else None

    query_loader = DataLoader(
        Subset(dataset, query_idx),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, query_loader


def create_multitask_single_subject_few_shot_dataloaders(
    dataset: EEGDataset,
    target_subject_id: int,
    target_task_id: list,
    n_shot: int = 5,
    batch_size: int = 32,
    random_state: int = 42,
    num_workers: int = 4,
    val_size: float = 0,
    oversample: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    mask = (dataset.subject_ids == target_subject_id)

    if target_task_id is not None:
        mask &= np.isin(dataset.tasks, target_task_id)

    all_indices = np.where(mask)[0]

    if len(all_indices) == 0:
        raise ValueError(f"Subject ID {target_subject_id} not found in dataset.")

    label_indices = {label: all_indices[dataset.labels[all_indices] == label] for label in np.unique(dataset.labels[all_indices])}

    if oversample:

        class_counts = {label: len(indices) for label, indices in label_indices.items()}
        max_count = max(class_counts.values())

        for label, indices in label_indices.items():
            current_count = len(indices)
            if current_count < max_count:

                num_samples_to_add = max_count - current_count
                sampled_indices = np.random.choice(indices, num_samples_to_add, replace=True)
                label_indices[label] = np.concatenate([indices, sampled_indices])

    train_idx_list, val_idx_list, query_idx_list = [], [], []

    print(f"\n[Random Split] Processing Subject: {target_subject_id}")

    for label, indices in label_indices.items():

        np.random.seed(random_state)
        np.random.shuffle(indices)

        n_shot_samples = indices[:n_shot]

        n_val_samples = int(len(indices) * val_size)
        val_samples = indices[n_shot:n_shot + n_val_samples] if n_val_samples > 0 else []

        remaining_samples = indices[n_shot + n_val_samples:]

        train_idx_list.extend(n_shot_samples)

        val_idx_list.extend(val_samples)

        query_idx_list.extend(remaining_samples)

        print(f"  Class {label}: Total={len(indices)}, Support Set={n_shot}, Val Set={len(val_samples)}, Query Set={len(remaining_samples)}")

    train_idx = np.array(train_idx_list)
    val_idx = np.array(val_idx_list) if val_size > 0 else None
    query_idx = np.array(query_idx_list)

    def get_dist(indices):
        labels = dataset.labels[indices]
        return dict(zip(*np.unique(labels, return_counts=True)))

    print(f"\nFinal Split Summary for Subject {target_subject_id}:")
    print(f"  Train Set: {len(train_idx)} trials, Dist: {get_dist(train_idx)}")
    print(f"  Val Set:   {len(val_idx)} trials, Dist: {get_dist(val_idx)}" if val_idx is not None else "  Val Set:   None")
    print(f"  Query Set: {len(query_idx)} trials, Dist: {get_dist(query_idx)}")

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    ) if val_size > 0 else None

    query_loader = DataLoader(
        Subset(dataset, query_idx),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, query_loader


def create_single_subject_dataloaders_control(
    dataset: EEGDataset,
    target_subject_id: int,
    train_size: float = 0.7,
    test_size: float = 0.2,
    val_size: float = 0.1,
    batch_size: int = 32,
    random_state: int = 42,
    num_workers: int = 4,
    train_count: int = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    all_indices = np.where(dataset.subject_ids == target_subject_id)[0]
    if len(all_indices) == 0:
        raise ValueError(f"Subject ID {target_subject_id} not found in dataset.")

    train_idx_list, val_idx_list, test_idx_list = [], [], []

    unique_labels = np.unique(dataset.labels[all_indices])

    print(f"\n[Random Split] Processing Subject: {target_subject_id}")

    for label in unique_labels:

        label_indices = all_indices[dataset.labels[all_indices] == label]

        np.random.seed(random_state)
        np.random.shuffle(label_indices)

        n_total = len(label_indices)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size)

        if train_count is not None:
            n_train = train_count
        else:
            n_train = int(n_total * train_size)

        n_train = min(n_train, n_total - n_val - max(1, n_test))

        t_idx = label_indices[:n_train]
        v_idx = label_indices[n_train : n_train + n_val]

        s_idx = label_indices[n_train + n_val:]

        train_idx_list.extend(t_idx)
        val_idx_list.extend(v_idx)
        test_idx_list.extend(s_idx)

        print(f"  Class {label}: Total={n_total}, Train={len(t_idx)}, Val={len(v_idx)}, Test={len(s_idx)}")

    train_idx = np.array(train_idx_list)
    val_idx = np.array(val_idx_list)
    test_idx = np.array(test_idx_list)

    def get_dist(indices):
        labels = dataset.labels[indices]
        return dict(zip(*np.unique(labels, return_counts=True)))

    print(f"\nFinal Split Summary for Subject {target_subject_id}:")
    print(f"  Train Set: {len(train_idx)} trials, Dist: {get_dist(train_idx)}")
    print(f"  Val Set:   {len(val_idx)} trials, Dist: {get_dist(val_idx)}")
    print(f"  Test Set:  {len(test_idx)} trials, Dist: {get_dist(test_idx)}")

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader
