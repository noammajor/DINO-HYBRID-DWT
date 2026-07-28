"""
Generate 10,000 small synthetic time series with varied parameters.
All series have a fixed length of 512 steps (one window).
X = time series (shape [N, 512])
Y = parameters used to generate each series (shape [N, num_label_features])

Varied parameters:
  - num_kernels:   how many base kernels are combined (1 to max_kernels)
  - kernel_ids:    which kernels from KERNEL_BANK are selected (indices 0-32)
  - operators:     binary operators between consecutive kernels (0=add, 1=mul)

Y label layout (1 + 2*max_kernels - 1 columns):
  [0]               num_kernels     (int, 1–max_kernels)
  [1 .. K]          kernel_id_0..K-1 (int 0–32, padded with -1 if unused)
  [K+1 .. 2K-1]    operator_0..K-2  (int 0=+, 1=*, padded with -1 if unused)
where K = max_kernels.

Usage:
    python scripts/synthetic_data_generation/generate_labeled_dataset.py -N 10000 -O <out_dir> --seed 42
"""

import argparse
import functools
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    RationalQuadratic,
    WhiteKernel,
)
from tqdm.auto import tqdm

# ── kernel bank (same as original kernel-synth.py, length-independent entries) ──
# We define them without LENGTH baked in; periodicity is re-scaled per sample.
_KERNEL_TEMPLATES = [
    ("ExpSineSquared", {"periodicity": 24}),
    ("ExpSineSquared", {"periodicity": 48}),
    ("ExpSineSquared", {"periodicity": 96}),
    ("ExpSineSquared", {"periodicity": 24 * 7}),
    ("ExpSineSquared", {"periodicity": 48 * 7}),
    ("ExpSineSquared", {"periodicity": 96 * 7}),
    ("ExpSineSquared", {"periodicity": 7}),
    ("ExpSineSquared", {"periodicity": 14}),
    ("ExpSineSquared", {"periodicity": 30}),
    ("ExpSineSquared", {"periodicity": 60}),
    ("ExpSineSquared", {"periodicity": 365}),
    ("ExpSineSquared", {"periodicity": 365 * 2}),
    ("ExpSineSquared", {"periodicity": 4}),
    ("ExpSineSquared", {"periodicity": 26}),
    ("ExpSineSquared", {"periodicity": 52}),
    ("ExpSineSquared", {"periodicity": 4}),
    ("ExpSineSquared", {"periodicity": 6}),
    ("ExpSineSquared", {"periodicity": 12}),
    ("ExpSineSquared", {"periodicity": 4}),
    ("ExpSineSquared", {"periodicity": 40}),
    ("ExpSineSquared", {"periodicity": 10}),
    ("DotProduct",     {"sigma_0": 0.0}),
    ("DotProduct",     {"sigma_0": 1.0}),
    ("DotProduct",     {"sigma_0": 10.0}),
    ("RBF",            {"length_scale": 0.1}),
    ("RBF",            {"length_scale": 1.0}),
    ("RBF",            {"length_scale": 10.0}),
    ("RationalQuadratic", {"alpha": 0.1}),
    ("RationalQuadratic", {"alpha": 1.0}),
    ("RationalQuadratic", {"alpha": 10.0}),
    ("WhiteKernel",    {"noise_level": 0.1}),
    ("WhiteKernel",    {"noise_level": 1.0}),
    ("ConstantKernel", {}),
]

NUM_KERNELS_IN_BANK = len(_KERNEL_TEMPLATES)  # 33

FIXED_LENGTH = 512
MAX_KERNELS  = 5
OPERATOR_ADD = 0
OPERATOR_MUL = 1


def build_kernel(idx: int, length: int):
    """Instantiate a kernel from the template bank, scaling periodicity by length."""
    name, params = _KERNEL_TEMPLATES[idx]
    p = dict(params)
    if "periodicity" in p:
        p["periodicity"] = p["periodicity"] / length
    cls = {
        "ExpSineSquared":    ExpSineSquared,
        "DotProduct":        DotProduct,
        "RBF":               RBF,
        "RationalQuadratic": RationalQuadratic,
        "WhiteKernel":       WhiteKernel,
        "ConstantKernel":    ConstantKernel,
    }[name]
    return cls(**p)


def generate_one(seed: int, max_kernels: int = MAX_KERNELS):
    """
    Sample parameters, generate a time series, return (ts, params).
    Uses an integer seed for joblib-safe reproducibility.

    Returns
    -------
    ts     : np.ndarray, shape (FIXED_LENGTH,)
    params : dict with keys num_kernels, kernel_ids, operators
    """
    rng = np.random.default_rng(seed)
    length = FIXED_LENGTH
    num_kernels = int(rng.integers(1, max_kernels + 1))

    kernel_ids = rng.integers(0, NUM_KERNELS_IN_BANK, size=num_kernels).tolist()
    operators  = rng.integers(0, 2, size=max(num_kernels - 1, 0)).tolist()  # 0=+, 1=*

    X = np.linspace(0, 1, length)[:, None]

    kernels = [build_kernel(i, length) for i in kernel_ids]

    def combine(a, b, op):
        return a + b if op == OPERATOR_ADD else a * b

    if len(kernels) == 1:
        kernel = kernels[0]
    else:
        kernel = kernels[0]
        for k, op in zip(kernels[1:], operators):
            kernel = combine(kernel, k, op)

    for _ in range(10):  # retry on LinAlgError
        try:
            gpr = GaussianProcessRegressor(kernel=kernel)
            ts = gpr.sample_y(X, n_samples=1, random_state=int(rng.integers(0, 2**31))).squeeze()
            return ts, {
                "num_kernels": num_kernels,
                "kernel_ids":  kernel_ids,
                "operators":   operators,
            }
        except np.linalg.LinAlgError:
            continue

    # fallback: white noise
    ts = rng.standard_normal(length)
    return ts, {
        "num_kernels": 1,
        "kernel_ids":  [30],  # WhiteKernel index
        "operators":   [],
    }


def encode_labels(params: dict, max_kernels: int = MAX_KERNELS) -> np.ndarray:
    """
    Encode parameter dict into a fixed-length label vector.

    Layout (1 + 2*max_kernels - 1 values):
      [0]             num_kernels
      [1 .. K]        kernel_ids  (padded with -1)
      [K+1 .. 2K-1]  operators   (padded with -1)
    where K = max_kernels.
    """
    k_ids = params["kernel_ids"][:max_kernels]
    ops   = params["operators"][:max_kernels - 1]

    k_ids_padded = k_ids + [-1] * (max_kernels - len(k_ids))
    ops_padded   = ops   + [-1] * (max_kernels - 1 - len(ops))

    return np.array(
        [params["num_kernels"]] + k_ids_padded + ops_padded,
        dtype=np.int32,
    )


def label_columns(max_kernels: int = MAX_KERNELS) -> list:
    cols = ["num_kernels"]
    cols += [f"kernel_id_{i}" for i in range(max_kernels)]
    cols += [f"operator_{i}" for i in range(max_kernels - 1)]
    return cols


def main():
    parser = argparse.ArgumentParser(
        description="Generate labeled synthetic time-series dataset (X=ts, Y=params)"
    )
    parser.add_argument("-N", "--num-samples",  type=int, default=10_000)
    parser.add_argument("-K", "--max-kernels",  type=int, default=MAX_KERNELS)
    parser.add_argument("-J", "--jobs",         type=int, default=4)
    parser.add_argument("-O", "--output-dir",   type=str, default=str(Path(__file__).parent))
    parser.add_argument("--seed",               type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate per-sample seeds from the master seed for reproducibility
    master_rng = np.random.default_rng(args.seed)
    seeds = master_rng.integers(0, 2**31, size=args.num_samples).tolist()

    results = Parallel(n_jobs=args.jobs)(
        delayed(generate_one)(s, max_kernels=args.max_kernels)
        for s in tqdm(seeds, desc="Generating")
    )

    # Pre-allocate: all series are exactly FIXED_LENGTH = 512
    X = np.empty((args.num_samples, FIXED_LENGTH), dtype=np.float32)
    Y = np.empty(
        (args.num_samples, 1 + 2 * args.max_kernels - 1),
        dtype=np.int32,
    )

    for i, (ts, params) in enumerate(results):
        X[i] = ts.astype(np.float32)
        Y[i] = encode_labels(params, max_kernels=args.max_kernels)

    # Save arrays
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "Y.npy", Y)

    # Save label metadata
    meta = {
        "num_samples":      args.num_samples,
        "series_length":    FIXED_LENGTH,
        "max_kernels":      args.max_kernels,
        "num_kernel_types": NUM_KERNELS_IN_BANK,
        "label_columns":    label_columns(args.max_kernels),
        "label_encoding": {
            "num_kernels": f"number of base kernels combined (1–{args.max_kernels})",
            "kernel_ids":  f"index into kernel bank (0–{NUM_KERNELS_IN_BANK-1}), -1 = unused slot",
            "operators":   "0 = addition (+), 1 = multiplication (*), -1 = unused slot",
        },
        "kernel_bank": [
            {"index": i, "type": name, "params": kparams}
            for i, (name, kparams) in enumerate(_KERNEL_TEMPLATES)
        ],
    }
    with open(out_dir / "dataset_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved {args.num_samples} samples to {out_dir}")
    print(f"  X.npy  : {X.shape}  (float32, all series length {FIXED_LENGTH})")
    print(f"  Y.npy  : {Y.shape}  (int32)")
    print(f"  Columns: {label_columns(args.max_kernels)}")
    print(f"  dataset_meta.json : kernel bank + label encoding details")


if __name__ == "__main__":
    main()
