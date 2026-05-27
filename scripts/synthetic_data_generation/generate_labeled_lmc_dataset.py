"""
Generate N labeled synthetic multivariate time series using the LMC (Linear
Coregionalization Model) approach from LMC_Synth.py.

X = multivariate time series,  shape [N, num_channels, 512]  (float32)
Y = 7 per-sample generation parameters, shape [N, 7]          (float32)

Y column layout:
  [0]  latent_num        — number of latent GPs drawn (int as float)
  [1]  dirichlet         — Dirichlet concentration actually used
  [2]  weibull_shape     — Weibull shape param used to draw latent_num
  [3]  weibull_scale     — Weibull scale param used to draw latent_num
  [4]  dirichlet_min     — lower bound used when sampling dirichlet
  [5]  dirichlet_max     — upper bound used when sampling dirichlet
  [6]  ess_length_scale  — ExpSineSquared length_scale (smoothness)

All 7 params are sampled freshly per series; the CLI args control the
sampling ranges (not fixed values).

Memory: X for 1M samples is ~20 GB. Files are written via numpy memmap so
the full dataset never lives in RAM at once. Standard np.load() works.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    RationalQuadratic,
    WhiteKernel,
)
from tqdm.auto import tqdm

# ── kernel bank ──────────────────────────────────────────────────────────────
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
    ("DotProduct",        {"sigma_0": 0.0}),
    ("DotProduct",        {"sigma_0": 1.0}),
    ("DotProduct",        {"sigma_0": 10.0}),
    ("RBF",               {"length_scale": 0.1}),
    ("RBF",               {"length_scale": 1.0}),
    ("RBF",               {"length_scale": 10.0}),
    ("RationalQuadratic", {"alpha": 0.1}),
    ("RationalQuadratic", {"alpha": 1.0}),
    ("RationalQuadratic", {"alpha": 10.0}),
    ("WhiteKernel",       {"noise_level": 0.1}),
    ("WhiteKernel",       {"noise_level": 1.0}),
    ("ConstantKernel",    {}),
]
NUM_KERNEL_TYPES = len(_KERNEL_TEMPLATES)  # 33

FIXED_LENGTH = 512
OPERATOR_ADD = 0
OPERATOR_MUL = 1
Y_DIM        = 7


def build_kernel(idx: int, ess_length_scale: float):
    name, params = _KERNEL_TEMPLATES[idx]
    p = dict(params)
    if name == "ExpSineSquared":
        p["periodicity"]  = p["periodicity"] / FIXED_LENGTH
        p["length_scale"] = ess_length_scale
    cls = {
        "ExpSineSquared":    ExpSineSquared,
        "DotProduct":        DotProduct,
        "RBF":               RBF,
        "RationalQuadratic": RationalQuadratic,
        "WhiteKernel":       WhiteKernel,
        "ConstantKernel":    ConstantKernel,
    }[name]
    return cls(**p)


def generate_one(
    seed: int,
    num_channels: int,
    max_kernels: int,
    max_latent: int,
    weibull_shape_range: tuple,
    weibull_scale_range: tuple,
    dirichlet_min_range: tuple,
    dirichlet_max_range: tuple,
    ess_ls_range: tuple,
):
    """Return (ts [num_channels, 512], y_row [7]) for one sample."""
    rng = np.random.default_rng(seed)
    X   = np.linspace(0, 1, FIXED_LENGTH)

    # ── sample all 7 generation parameters ───────────────────────────────────
    weibull_shape    = float(rng.uniform(*weibull_shape_range))
    weibull_scale    = float(rng.uniform(*weibull_scale_range))
    d_min            = float(rng.uniform(*dirichlet_min_range))
    d_max_lo         = max(d_min + 0.05, dirichlet_max_range[0])
    d_max            = float(rng.uniform(d_max_lo, dirichlet_max_range[1]))
    ess_ls           = float(rng.uniform(*ess_ls_range))

    # ── latent_num from Weibull ───────────────────────────────────────────────
    raw        = rng.weibull(weibull_shape) * weibull_scale + 1
    latent_num = int(np.clip(np.rint(raw), max(2, num_channels // 20), max_latent))

    # ── per-latent kernel structure ───────────────────────────────────────────
    per_latent = []
    for _ in range(latent_num):
        n_k   = int(rng.integers(1, max_kernels + 1))
        k_ids = rng.integers(0, NUM_KERNEL_TYPES, size=n_k).tolist()
        ops   = rng.integers(0, 2, size=max(n_k - 1, 0)).tolist()
        per_latent.append((k_ids, ops))

    # ── Dirichlet concentration ───────────────────────────────────────────────
    dirichlet = float(rng.uniform(d_min, d_max))

    y_row = np.array(
        [latent_num, dirichlet, weibull_shape, weibull_scale, d_min, d_max, ess_ls],
        dtype=np.float32,
    )

    for _ in range(10):
        try:
            latent_fns = []
            for k_ids, ops in per_latent:
                kernels = [build_kernel(i, ess_ls) for i in k_ids]
                kernel  = kernels[0]
                for k, op in zip(kernels[1:], ops):
                    kernel = kernel + k if op == OPERATOR_ADD else kernel * k
                cov = kernel(X[:, None])
                lf  = rng.multivariate_normal(np.zeros(FIXED_LENGTH), cov, method="eigh")
                latent_fns.append(lf)

            latent_fns = np.array(latent_fns)                         # (L, 512)
            weights    = rng.dirichlet(dirichlet * np.ones(latent_num), size=num_channels)
            ts         = (weights @ latent_fns).astype(np.float32)    # (C, 512)
            return ts, y_row

        except np.linalg.LinAlgError:
            continue

    # fallback: white noise — y_row already recorded the intended params
    ts = rng.standard_normal((num_channels, FIXED_LENGTH)).astype(np.float32)
    return ts, y_row


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate labeled LMC synthetic multivariate time-series dataset"
    )
    parser.add_argument("-N", "--num-samples",  type=int, default=5_000_000)
    parser.add_argument("-C", "--num-channels", type=int, default=10)
    parser.add_argument("-K", "--max-kernels",  type=int, default=5)
    parser.add_argument("-L", "--max-latent",   type=int, default=None,
                        help="Max latent GPs (defaults to num_channels)")
    parser.add_argument("-J", "--jobs",         type=int, default=64)
    parser.add_argument("--chunk-size",         type=int, default=None,
                        help="Samples per parallel batch (default: 100 × jobs)")
    parser.add_argument("-O", "--output-dir",   type=str,
                        default=str(Path(__file__).parent / "labeled_lmc_dataset"))
    parser.add_argument("--seed",               type=int, default=42)
    # ── per-sample param sampling ranges ──────────────────────────────────────
    parser.add_argument("--weibull-shape-min",  type=float, default=0.5)
    parser.add_argument("--weibull-shape-max",  type=float, default=5.0)
    parser.add_argument("--weibull-scale-min",  type=float, default=0.5)
    parser.add_argument("--weibull-scale-max",  type=float, default=10.0)
    parser.add_argument("--dirichlet-min-lo",   type=float, default=0.01)
    parser.add_argument("--dirichlet-min-hi",   type=float, default=1.5)
    parser.add_argument("--dirichlet-max-hi",   type=float, default=5.0)
    parser.add_argument("--ess-ls-min",         type=float, default=0.1)
    parser.add_argument("--ess-ls-max",         type=float, default=10.0)
    args = parser.parse_args()

    max_latent  = args.max_latent or args.num_channels
    chunk_size  = args.chunk_size or args.jobs * 100
    out_dir     = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── pre-allocate output files on disk via memmap ──────────────────────────
    # np.lib.format.open_memmap writes a proper .npy header so np.load() works.
    X_path = out_dir / "X.npy"
    Y_path = out_dir / "Y.npy"
    X_mm = np.lib.format.open_memmap(
        X_path, mode="w+", dtype="float32",
        shape=(args.num_samples, args.num_channels, FIXED_LENGTH),
    )
    Y_mm = np.lib.format.open_memmap(
        Y_path, mode="w+", dtype="float32",
        shape=(args.num_samples, Y_DIM),
    )

    x_gb = args.num_samples * args.num_channels * FIXED_LENGTH * 4 / 1e9
    print(f"Pre-allocated  X.npy : {tuple(X_mm.shape)}  ({x_gb:.1f} GB on disk)")
    print(f"Pre-allocated  Y.npy : {tuple(Y_mm.shape)}")
    print(f"Generating {args.num_samples:,} samples  |  {args.jobs} jobs  |  chunk {chunk_size:,}")

    # ── generate seeds once, deterministically ────────────────────────────────
    master_rng = np.random.default_rng(args.seed)
    seeds      = master_rng.integers(0, 2**31, size=args.num_samples).tolist()

    gen_kwargs = dict(
        num_channels           = args.num_channels,
        max_kernels            = args.max_kernels,
        max_latent             = max_latent,
        weibull_shape_range    = (args.weibull_shape_min,  args.weibull_shape_max),
        weibull_scale_range    = (args.weibull_scale_min,  args.weibull_scale_max),
        dirichlet_min_range    = (args.dirichlet_min_lo,   args.dirichlet_min_hi),
        dirichlet_max_range    = (args.dirichlet_min_lo,   args.dirichlet_max_hi),
        ess_ls_range           = (args.ess_ls_min,         args.ess_ls_max),
    )

    # ── chunked parallel generation → write directly to memmap ───────────────
    with tqdm(total=args.num_samples, unit="sample") as pbar:
        for start in range(0, args.num_samples, chunk_size):
            end          = min(start + chunk_size, args.num_samples)
            chunk_seeds  = seeds[start:end]

            results = Parallel(n_jobs=args.jobs, prefer="processes")(
                delayed(generate_one)(s, **gen_kwargs) for s in chunk_seeds
            )

            for j, (ts, y_row) in enumerate(results):
                X_mm[start + j] = ts
                Y_mm[start + j] = y_row

            X_mm.flush()
            Y_mm.flush()
            pbar.update(end - start)

    # ── save metadata ─────────────────────────────────────────────────────────
    Y_COLUMNS = [
        "latent_num", "dirichlet",
        "weibull_shape", "weibull_scale",
        "dirichlet_min", "dirichlet_max",
        "ess_length_scale",
    ]
    meta = {
        "num_samples":      args.num_samples,
        "series_length":    FIXED_LENGTH,
        "num_channels":     args.num_channels,
        "max_latent":       max_latent,
        "max_kernels":      args.max_kernels,
        "num_kernel_types": NUM_KERNEL_TYPES,
        "Y_columns":        Y_COLUMNS,
        "sampling_ranges": {
            "weibull_shape":    [args.weibull_shape_min,  args.weibull_shape_max],
            "weibull_scale":    [args.weibull_scale_min,  args.weibull_scale_max],
            "dirichlet_min":    [args.dirichlet_min_lo,   args.dirichlet_min_hi],
            "dirichlet_max":    [args.dirichlet_min_lo,   args.dirichlet_max_hi],
            "ess_length_scale": [args.ess_ls_min,         args.ess_ls_max],
        },
        "label_encoding": {
            "latent_num":       "number of latent GPs used (int stored as float)",
            "dirichlet":        "Dirichlet concentration actually used",
            "weibull_shape":    "Weibull shape param used to draw latent_num",
            "weibull_scale":    "Weibull scale param used to draw latent_num",
            "dirichlet_min":    "lower bound used when drawing dirichlet",
            "dirichlet_max":    "upper bound used when drawing dirichlet",
            "ess_length_scale": "ExpSineSquared length_scale (smoothness)",
        },
        "kernel_bank": [
            {"index": i, "type": name, "params": kp}
            for i, (name, kp) in enumerate(_KERNEL_TEMPLATES)
        ],
    }
    with open(out_dir / "dataset_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Files in {out_dir}/")
    print(f"  X.npy : {tuple(X_mm.shape)}  ({x_gb:.1f} GB)")
    print(f"  Y.npy : {tuple(Y_mm.shape)}")
    print(f"  dataset_meta.json")
    print(f"\nLoad with:  X = np.load('X.npy', mmap_mode='r')")


if __name__ == "__main__":
    main()
