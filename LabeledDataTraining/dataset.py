"""
Dataset and DataLoader for the labeled LMC dataset.

On-disk layout (produced by generate_labeled_lmc_dataset.py):
  X.npy              [N, C, T]  float32  — multivariate time series
  Y.npy              [N, 7]     float32  — generation parameters
  dataset_meta.json             — metadata (optional, for reference)

Y column layout (matches heads.py and generate_labeled_lmc_dataset.py):
  [0] latent_num        — int stored as float → cast to long for CrossEntropyLoss
  [1] dirichlet         — float
  [2] weibull_shape     — float
  [3] weibull_scale     — float
  [4] dirichlet_min     — float
  [5] dirichlet_max     — float
  [6] ess_length_scale  — float

X is stored channels-first [C, T]; the backbone expects [B, T, C], so each
sample is transposed to [T, C] on retrieval.

Both arrays are opened as read-only memmaps so multi-worker DataLoaders can
share them without copying the full dataset into RAM.
"""

import json
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# Y column indices — kept in sync with generate_labeled_lmc_dataset.py.
_COL_LATENT_NUM   = 0
_COL_DIRICHLET    = 1
_COL_WEIB_SHAPE   = 2
_COL_WEIB_SCALE   = 3
_COL_D_MIN        = 4
_COL_D_MAX        = 5
_COL_ESS_LS       = 6


class LMCDataset(Dataset):
    """Labeled LMC synthetic dataset.

    Args:
        data_dir  : directory containing X.npy and Y.npy
        indices   : optional subset of integer indices (for train/val/test splits)
    """

    def __init__(self, data_dir: Union[str, Path], indices=None, seq_len: int = None):
        data_dir = Path(data_dir)

        # Open as memmaps — the OS pages in only the slices that are accessed,
        # so the full dataset never needs to fit in RAM.
        self.X = np.load(data_dir / "X.npy", mmap_mode="r")  # [N, C, T]
        self.Y = np.load(data_dir / "Y.npy", mmap_mode="r")  # [N, 7]

        # Load metadata if available (informational; not required for training).
        meta_path = data_dir / "dataset_meta.json"
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        # indices lets callers pass a pre-computed split without copying data.
        self.indices = np.asarray(indices) if indices is not None else None
        # If seq_len < T, truncate each sample to the first seq_len timesteps.
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.indices) if self.indices is not None else len(self.X)

    def __getitem__(self, idx: int):
        # Resolve the real row index when a subset is active.
        real_idx = int(self.indices[idx]) if self.indices is not None else idx

        # X: [C, T]  →  [T, C]  (backbone expects time-first)
        x = torch.from_numpy(self.X[real_idx].T.copy())   # [T, C]
        if self.seq_len is not None:
            x = x[:self.seq_len]                           # truncate to seq_len

        # Single from_numpy conversion for all 7 labels, then cheap tensor slicing.
        # 7 separate torch.tensor() calls per sample is ~7× the Python overhead.
        y_raw = torch.from_numpy(self.Y[real_idx].copy())  # [7] float32

        y = {
            "dirichlet":        y_raw[_COL_DIRICHLET    : _COL_DIRICHLET    + 1],
            "weibull_shape":    y_raw[_COL_WEIB_SHAPE   : _COL_WEIB_SHAPE   + 1],
            "weibull_scale":    y_raw[_COL_WEIB_SCALE   : _COL_WEIB_SCALE   + 1],
            "dirichlet_min":    y_raw[_COL_D_MIN        : _COL_D_MIN        + 1],
            "dirichlet_max":    y_raw[_COL_D_MAX        : _COL_D_MAX        + 1],
            "ess_length_scale": y_raw[_COL_ESS_LS       : _COL_ESS_LS       + 1],
        }

        return x, y


def make_loaders(
    data_dir: Union[str, Path],
    batch_size: int = 256,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    num_workers: int = 4,
    seed: int = 42,
    seq_len: int = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Split the dataset and return (train_loader, val_loader, test_loader).

    Splits are index-based — no data is copied.  The same seed always produces
    the same split, so checkpointing and resuming are consistent.

    Args:
        data_dir    : directory containing X.npy and Y.npy
        batch_size  : samples per batch
        val_frac    : fraction of data reserved for validation
        test_frac   : fraction of data reserved for testing
        num_workers : DataLoader worker processes
        seed        : RNG seed for reproducible splitting
    """
    # Build a full-dataset index array and shuffle it once with a fixed seed.
    full = LMCDataset(data_dir, seq_len=seq_len)
    N    = len(full)

    rng     = np.random.default_rng(seed)
    perm    = rng.permutation(N)

    n_val   = int(N * val_frac)
    n_test  = int(N * test_frac)
    n_train = N - n_val - n_test

    train_idx = perm[:n_train]
    val_idx   = perm[n_train : n_train + n_val]
    test_idx  = perm[n_train + n_val :]

    train_ds = LMCDataset(data_dir, indices=train_idx, seq_len=seq_len)
    val_ds   = LMCDataset(data_dir, indices=val_idx,   seq_len=seq_len)
    test_ds  = LMCDataset(data_dir, indices=test_idx,  seq_len=seq_len)

    loader_kwargs = dict(
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory  = True,
        # prefetch_factor speeds up IO-bound loading; only valid when num_workers > 0.
        prefetch_factor = 2 if num_workers > 0 else None,
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
