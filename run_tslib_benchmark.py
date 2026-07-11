#!/usr/bin/env python
"""
Unified Time-Series-Library benchmark runner.

Run ONE model, a comma-list, or ALL models from Time-Series-Library-main-2/models/
on any of three tasks (forecasting / classification / anomaly_detection) using
*our* data loaders (shared/data_loaders/data_puller.py + dataset_registry).

By default every model uses ITS OWN default hyper-parameters, parsed from the
matching Time-Series-Library shell script under scripts/<task>/... (e.g.
TimesNet forecasting on ETTh1 -> d_model=16, d_ff=32, top_k=5). Any value you
pass explicitly on the CLI overrides the script default. Pass --no_defaults to
ignore the scripts and use this file's built-in defaults instead.

The TSLib models all share the same forward signature dispatched on
`configs.task_name`. Our loaders return patched tensors; this script flattens
them back to the flat [B, T, C] layout the models expect — identical
splits/normalisation to the rest of the project.

Examples
--------
    # one model, its TSL default config
    python run_tslib_benchmark.py --model TimesNet --task forecast --dataset etth1

    # ALL non-foundation models in one run, each with its own defaults
    python run_tslib_benchmark.py --model all --task forecast --dataset etth1

    # a subset
    python run_tslib_benchmark.py --model TimesNet,DLinear,iTransformer \
        --task classify --dataset SpokenArabicDigits

    # override a default
    python run_tslib_benchmark.py --model TimesNet --task anomaly --dataset MSL \
        --d_model 32 --epochs 3
"""

import argparse
import importlib
import os
import re
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn


# ── path wiring ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
TSLIB_DIR    = PROJECT_ROOT / "Time-Series-Library-main-2"
SHARED_DIR   = PROJECT_ROOT / "shared"
SCRIPTS_DIR  = TSLIB_DIR / "scripts"

for p in (str(PROJECT_ROOT), str(SHARED_DIR), str(TSLIB_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# TSLib auto-scans a *relative* `models/` dir and writes relative checkpoint
# folders, so run from inside the library directory.
os.chdir(TSLIB_DIR)

from data_paths import DATA_PATHS                       # noqa: E402
from dataset_registry import get_dataset_info           # noqa: E402

TASK_ALIASES = {
    "forecast":   "long_term_forecast",
    "forecasting": "long_term_forecast",
    "long_term_forecast": "long_term_forecast",
    "classify":    "classification",
    "classification": "classification",
    "anomaly":     "anomaly_detection",
    "anomaly_detection": "anomaly_detection",
}

# Foundation / weight-loading models — excluded from `--model all` (need
# pretrained checkpoints or large external packages, not trained from scratch).
WEIGHT_MODELS = {
    "Chronos", "Chronos2", "Moirai", "Sundial", "TimeMoE", "TimesFM", "TiRex",
}

# time-feature dimension per freq code (TSLib DataEmbedding timeF marks)
_FREQ_MARK_DIM = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}


# ══════════════════════════════════════════════════════════════════════════════
#  Default-config parsing (read each model's hyper-params from the TSL scripts)
# ══════════════════════════════════════════════════════════════════════════════
# TSL run.py flag  ->  (our namespace attr, cast)
_CFG_MAP = {
    "d_model": ("d_model", int), "d_ff": ("d_ff", int),
    "e_layers": ("e_layers", int), "d_layers": ("d_layers", int),
    "n_heads": ("n_heads", int), "factor": ("factor", int),
    "top_k": ("top_k", int), "num_kernels": ("num_kernels", int),
    "moving_avg": ("moving_avg", int), "dropout": ("dropout", float),
    "seg_len": ("seg_len", int),
    "channel_independence": ("channel_independence", int),
    "down_sampling_layers": ("down_sampling_layers", int),
    "down_sampling_window": ("down_sampling_window", int),
    "patch_len": ("patch_len", int), "label_len": ("label_len", int),
    "seq_len": ("seq_len", int), "learning_rate": ("lr", float),
    "train_epochs": ("epochs", int), "batch_size": ("batch_size", int),
    "patience": ("patience", int), "anomaly_ratio": ("anomaly_ratio", float),
}

_BLOCK_CACHE: dict = {}   # task_name -> list[dict of raw string flags]


def _parse_sh_file(path: Path):
    """Extract every `run.py` invocation in a shell script as a flag dict."""
    txt = path.read_text(errors="ignore").replace("\\\n", " ")
    mn = re.search(r"model_name\s*=\s*(\S+)", txt)
    model_name = mn.group(1) if mn else None

    blocks = []
    for seg in re.split(r"python\s+-?u?\s*run\.py", txt)[1:]:
        seg = seg.split("python")[0]   # stop before any next command
        flags = {}
        for fm in re.finditer(r"--([A-Za-z_]+)\s+('[^']*'|\"[^\"]*\"|[^\s\\]+)", seg):
            key = fm.group(1)
            val = fm.group(2).strip("'\"")
            if val == "$model_name":
                val = model_name
            flags[key] = val
        if flags.get("model") == "$model_name" and model_name:
            flags["model"] = model_name
        if "model" in flags:
            blocks.append(flags)
    return blocks


def _all_blocks(task_name: str):
    if task_name not in _BLOCK_CACHE:
        task_dir = SCRIPTS_DIR / task_name
        blocks = []
        if task_dir.exists():
            for sh in sorted(task_dir.rglob("*.sh")):
                try:
                    blocks.extend(_parse_sh_file(sh))
                except Exception:
                    pass
        _BLOCK_CACHE[task_name] = blocks
    return _BLOCK_CACHE[task_name]


def _dataset_matches(task_name: str, block: dict, dataset: str) -> bool:
    d = dataset.lower()
    if task_name == "long_term_forecast":
        try:
            csv = get_dataset_info(dataset)["csv_filename"].lower()
        except Exception:
            csv = d + ".csv"
        if block.get("data_path", "").lower() == csv:
            return True
        return block.get("data", "").lower() == d
    # classification / anomaly: folder-name keyed
    if d in (block.get("model_id", "").lower(), block.get("data", "").lower()):
        return True
    return d in block.get("root_path", "").lower()


def load_default_config(task_name: str, model: str, dataset: str, pred_len=None):
    """Return {our_attr: typed_value} from the matching TSL script, or None."""
    candidates = [b for b in _all_blocks(task_name)
                  if b.get("model") == model and _dataset_matches(task_name, b, dataset)]
    if task_name == "long_term_forecast" and pred_len is not None:
        exact = [b for b in candidates if b.get("pred_len") == str(pred_len)]
        candidates = exact or candidates
    if not candidates:
        return None
    block = candidates[0]
    cfg = {}
    for tsl_key, (attr, cast) in _CFG_MAP.items():
        if tsl_key in block:
            try:
                cfg[attr] = cast(block[tsl_key])
            except (ValueError, TypeError):
                pass
    return cfg


def _explicit_cli_flags():
    """Names of args the user passed explicitly (these beat script defaults)."""
    out = set()
    for a in sys.argv[1:]:
        if a.startswith("--"):
            out.add(a[2:].split("=")[0])
    return out


_EXPLICIT = _explicit_cli_flags()


def effective_args(a, task_name: str, pred_len=None):
    """Clone `a`, overlay the model's TSL default config (CLI flags win)."""
    ns = SimpleNamespace(**vars(a))
    if not a.use_defaults:
        return ns, None
    cfg = load_default_config(task_name, a.model, a.dataset, pred_len)
    if cfg:
        for k, v in cfg.items():
            if k not in _EXPLICIT:
                setattr(ns, k, v)
    return ns, cfg


# ── device ────────────────────────────────────────────────────────────────────
def pick_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── build the (large) args namespace TSLib Model constructors expect ──────────
def build_model_args(a, task_name: str) -> SimpleNamespace:
    """Replicates Time-Series-Library run.py defaults; CLI/script flags override."""
    return SimpleNamespace(
        task_name=task_name,
        model=a.model,
        seq_len=a.seq_len,
        label_len=a.label_len,
        pred_len=(a.pred_len[0] if a.pred_len else 96),
        enc_in=1, dec_in=1, c_out=1, num_class=2,
        d_model=a.d_model, n_heads=a.n_heads, e_layers=a.e_layers, d_layers=a.d_layers,
        d_ff=a.d_ff, moving_avg=a.moving_avg, factor=a.factor, distil=True,
        dropout=a.dropout, embed=a.embed, freq=a.freq, activation="gelu",
        output_attention=False,
        channel_independence=a.channel_independence,
        decomp_method="moving_avg", use_norm=1,
        down_sampling_layers=a.down_sampling_layers,
        down_sampling_window=a.down_sampling_window,
        down_sampling_method="avg",
        seg_len=a.seg_len,
        top_k=a.top_k, num_kernels=a.num_kernels,
        patch_len=a.patch_len, stride=a.patch_len,
        modes=32, mode_select="random", version="fourier",
        p_hidden_dims=[128, 128], p_hidden_layers=2,
        individual=False,
        alpha=0.1, top_p=0.5, pos=1,
        node_dim=10, gcn_depth=2, gcn_dropout=0.3, propalpha=0.3,
        conv_channel=32, skip_channel=32,
        use_future_temporal_feature=0,
        features="M",
    )


def build_tslib_model(args: SimpleNamespace) -> nn.Module:
    module = importlib.import_module(f"models.{args.model}")
    return module.Model(args).float()


def discover_models():
    """All model files minus foundation/weight models, sorted."""
    models = []
    for f in sorted((TSLIB_DIR / "models").glob("*.py")):
        name = f.stem
        if name == "__init__" or name in WEIGHT_MODELS:
            continue
        models.append(name)
    return models


# ══════════════════════════════════════════════════════════════════════════════
#  FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
class _ForecastWindowAdapter(torch.utils.data.Dataset):
    """PatchTSTForcastingAdapter -> (seq_x, seq_y, x_mark, y_mark) flat tensors."""

    def __init__(self, patched_ds, freq="h"):
        self._ds = patched_ds
        self._md = _FREQ_MARK_DIM.get(freq, 4)

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        ctx, tgt = self._ds[idx]
        seq_x = ctx.reshape(-1, ctx.shape[-1])      # [seq_len, C]
        seq_y = tgt.reshape(-1, tgt.shape[-1])      # [label_len + pred_len, C]
        x_mark = torch.zeros(seq_x.shape[0], self._md)
        y_mark = torch.zeros(seq_y.shape[0], self._md)
        return seq_x, seq_y, x_mark, y_mark


def run_forecast(a, device):
    from data_loaders.data_puller import PatchTSTForcastingAdapter

    info  = get_dataset_info(a.dataset)
    csv   = info["csv_path"]
    c_in  = info["c_in"]
    pred_lens = a.pred_len or [96, 192, 336, 720]

    print(f"\n=== FORECAST | {a.model} | {a.dataset} (C={c_in}) ===")

    results = {}
    for pred_len in pred_lens:
        ea, cfg = effective_args(a, "long_term_forecast", pred_len)
        if cfg:
            print(f"  [defaults] pl={pred_len}: {cfg}")
        elif a.use_defaults:
            print(f"  [defaults] pl={pred_len}: none found — using built-in defaults")

        seq_len, patch_len, label_len = ea.seq_len, ea.patch_len, ea.label_len
        bs, freq = ea.batch_size, ea.freq

        def loader(split, _pl=pred_len):
            ds = _ForecastWindowAdapter(
                PatchTSTForcastingAdapter(csv, split, seq_len, _pl, patch_len,
                                          label_len=label_len),
                freq=freq)
            return torch.utils.data.DataLoader(
                ds, batch_size=bs, shuffle=(split == "train"),
                num_workers=ea.num_workers, drop_last=(split == "train"))

        tr, va, te = loader("train"), loader("val"), loader("test")

        margs = build_model_args(ea, "long_term_forecast")
        margs.pred_len = pred_len
        margs.enc_in = margs.dec_in = margs.c_out = c_in
        model = build_tslib_model(margs).to(device)

        opt  = torch.optim.Adam(model.parameters(), lr=ea.lr)
        crit = nn.MSELoss()

        def forward_batch(bx, by, bxm, bym):
            bx, by  = bx.float().to(device), by.float().to(device)
            bxm, bym = bxm.float().to(device), bym.float().to(device)
            dec_zeros = torch.zeros_like(by[:, -pred_len:, :])
            dec_inp = torch.cat([by[:, :label_len, :], dec_zeros], dim=1).float()
            out = model(bx, bxm, dec_inp, bym)
            return out[:, -pred_len:, :], by[:, -pred_len:, :]

        best_val, best_state, no_improve = float("inf"), None, 0
        for epoch in range(ea.epochs):
            model.train(); t0 = time.time(); losses = []
            for batch in tr:
                opt.zero_grad()
                out, tgt = forward_batch(*batch)
                loss = crit(out, tgt); loss.backward(); opt.step()
                losses.append(loss.item())
            model.eval(); vlosses = []
            with torch.no_grad():
                for batch in va:
                    out, tgt = forward_batch(*batch)
                    vlosses.append(crit(out, tgt).item())
            vl = float(np.mean(vlosses)) if vlosses else float("inf")
            print(f"  pl={pred_len} epoch {epoch+1}/{ea.epochs} "
                  f"train={np.mean(losses):.4f} val={vl:.4f} ({time.time()-t0:.1f}s)")
            if vl < best_val:
                best_val, no_improve = vl, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= ea.patience:
                    print("  early stopping"); break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval(); preds, trues = [], []
        with torch.no_grad():
            for batch in te:
                out, tgt = forward_batch(*batch)
                preds.append(out.cpu().numpy()); trues.append(tgt.cpu().numpy())
        P, T = np.concatenate(preds), np.concatenate(trues)
        mse, mae = float(np.mean((P - T) ** 2)), float(np.mean(np.abs(P - T)))
        print(f"  >> pred_len={pred_len}  MSE={mse:.4f}  MAE={mae:.4f}")
        results[pred_len] = (mse, mae)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
def run_classify(a, device):
    from data_loaders.data_puller import ClassificationDataPuller, make_uea_dataloaders

    a, cfg = effective_args(a, "classification")
    if cfg:
        print(f"  [defaults] {cfg}")
    elif a.use_defaults:
        print("  [defaults] none found — using built-in defaults")

    cls_dir = DATA_PATHS["classification_data_dir"]
    p_s, bs = a.patch_len, a.batch_size

    ds_path = Path(cls_dir) / a.dataset
    is_uea  = bool(list(ds_path.glob("*_TRAIN.ts"))) if ds_path.exists() else False

    if is_uea:
        raw_tr, _, raw_te, n_classes = make_uea_dataloaders(cls_dir, a.dataset, batch_size=bs)
        ds_tr, ds_te = raw_tr.dataset, raw_te.dataset
        n_vars = ds_tr._samples[0].shape[-1]
        max_T  = max(s.shape[0] for s in ds_tr._samples + ds_te._samples)
        seq_len = int(np.ceil(max_T / p_s)) * p_s

        def collate(batch):
            import torch.nn.functional as F
            xs, ys, olens = zip(*batch)
            olens = torch.stack(olens)
            mt = max(x.shape[0] for x in xs)
            xs = torch.stack([F.pad(x, (0, 0, 0, mt - x.shape[0])) for x in xs])
            T = xs.shape[1]
            if T < seq_len:
                xs = torch.cat([xs, torch.zeros(xs.shape[0], seq_len - T, xs.shape[2])], dim=1)
            elif T > seq_len:
                xs = xs[:, :seq_len, :]
            mask = (torch.arange(seq_len).unsqueeze(0) < olens.unsqueeze(1)).float()
            return xs, torch.stack(ys), mask

        train = torch.utils.data.DataLoader(ds_tr, batch_size=bs, shuffle=True,  collate_fn=collate)
        test  = torch.utils.data.DataLoader(ds_te, batch_size=bs, shuffle=False, collate_fn=collate)

        def to_xb(batch):
            bx, by, mask = batch
            return bx.float().to(device), mask.to(device), by.long().to(device)
    else:
        def mk(split):
            ds = ClassificationDataPuller(cls_dir, a.dataset, p_s, which=split)
            return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=(split == "train"))
        train, test = mk("train"), mk("test")
        n_classes = train.dataset.n_classes
        n_vars    = train.dataset.X.shape[2]
        seq_len   = train.dataset.X.shape[1]

        def to_xb(batch):
            patches, by, pmask = batch
            bx   = patches.reshape(patches.shape[0], -1, patches.shape[-1]).float().to(device)
            mask = pmask.float().repeat_interleave(p_s, dim=1).to(device)
            return bx, mask, by.long().to(device)

    print(f"\n=== CLASSIFY | {a.model} | {a.dataset} | "
          f"{n_classes} classes, {n_vars} vars, seq_len={seq_len} ===")

    margs = build_model_args(a, "classification")
    margs.seq_len, margs.pred_len = seq_len, 0
    margs.enc_in = margs.dec_in = margs.c_out = n_vars
    margs.num_class = n_classes
    margs.channel_independence = 0       # classification needs multivariate mixing
    margs.down_sampling_layers = 0
    model = build_tslib_model(margs).to(device)

    opt  = torch.optim.RAdam(model.parameters(), lr=a.lr)
    crit = nn.CrossEntropyLoss()
    best_acc, no_improve = 0.0, 0

    for epoch in range(a.epochs):
        model.train(); t0 = time.time()
        for batch in train:
            bx, mask, by = to_xb(batch)
            opt.zero_grad()
            loss = crit(model(bx, mask, None, None), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            opt.step()

        model.eval(); preds, trues = [], []
        with torch.no_grad():
            for batch in test:
                bx, mask, by = to_xb(batch)
                preds.append(model(bx, mask, None, None).argmax(-1).cpu()); trues.append(by.cpu())
        acc = (torch.cat(preds) == torch.cat(trues)).float().mean().item()
        print(f"  epoch {epoch+1}/{a.epochs}  test acc={acc:.4f}  ({time.time()-t0:.1f}s)")
        if acc > best_acc:
            best_acc, no_improve = acc, 0
        else:
            no_improve += 1
            if no_improve >= a.patience:
                print("  early stopping"); break

    print(f"  >> {a.model} accuracy={best_acc:.4f}")
    return best_acc


# ══════════════════════════════════════════════════════════════════════════════
#  ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def _adjustment(gt, pred):
    # Segment-level point adjustment (Xu et al.), matching the DINO anomaly eval
    # in tsdino_timemixer/TSMixerAnomaly.py: if any point inside a true anomaly
    # segment is flagged, the whole segment is counted as detected.
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                if pred[j] == 0:
                    pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                if pred[j] == 0:
                    pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def run_anomaly(a, device):
    from data_loaders.data_puller import AnomalyDataPuller
    from sklearn.metrics import precision_recall_fscore_support

    a, cfg = effective_args(a, "anomaly_detection")
    if cfg:
        print(f"  [defaults] {cfg}")
    elif a.use_defaults:
        print("  [defaults] none found — using built-in defaults")

    anom_dir, bs = DATA_PATHS["anomaly_data_dir"], a.batch_size

    ds_tr = AnomalyDataPuller(anom_dir, a.dataset, a.patch_len, which="train")
    ds_te = AnomalyDataPuller(anom_dir, a.dataset, a.patch_len, which="test")
    seq_len, n_vars = ds_tr.padded_T, ds_tr.n_vars

    tr = torch.utils.data.DataLoader(ds_tr, batch_size=bs, shuffle=False, drop_last=False)
    te = torch.utils.data.DataLoader(ds_te, batch_size=bs, shuffle=False, drop_last=False)

    print(f"\n=== ANOMALY | {a.model} | {a.dataset} | "
          f"{n_vars} vars, win/seq_len={seq_len} ===")

    margs = build_model_args(a, "anomaly_detection")
    margs.seq_len, margs.pred_len = seq_len, 0
    margs.enc_in = margs.dec_in = margs.c_out = n_vars
    margs.channel_independence = 0
    margs.down_sampling_layers = 0
    model = build_tslib_model(margs).to(device)

    opt  = torch.optim.Adam(model.parameters(), lr=a.lr)
    crit = nn.MSELoss()

    for epoch in range(a.epochs):
        model.train(); t0 = time.time(); losses = []
        for batch in tr:
            # train/val split returns a bare patches tensor (no labels)
            patches = batch.float().to(device)
            bx = patches.reshape(patches.shape[0], -1, patches.shape[-1])  # [B,T,C]
            recon = model(bx, None, None, None)
            loss = crit(recon, bx)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"  epoch {epoch+1}/{a.epochs}  loss={np.mean(losses):.4f}  ({time.time()-t0:.1f}s)")

    model.eval()
    # ── train energy (no labels) — for the combined threshold ──────────────────
    train_energy = []
    with torch.no_grad():
        for batch in tr:
            patches = batch.float().to(device)              # train split → bare tensor
            bx = patches.reshape(patches.shape[0], -1, patches.shape[-1])
            err = ((model(bx, None, None, None) - bx) ** 2).mean(dim=-1)   # [B,T]
            train_energy.append(err.cpu().numpy())
    train_energy = np.concatenate(train_energy).reshape(-1)

    # ── test energy + labels ───────────────────────────────────────────────────
    scores, labels = [], []
    with torch.no_grad():
        for batch in te:
            patches, lbl = batch[0].float().to(device), batch[1]
            bx = patches.reshape(patches.shape[0], -1, patches.shape[-1])
            err = ((model(bx, None, None, None) - bx) ** 2).mean(dim=-1)   # [B,T]
            scores.append(err.cpu().numpy()); labels.append(lbl.numpy())

    scores = np.concatenate(scores).reshape(-1)
    labels = np.concatenate(labels).reshape(-1).astype(int)
    # Threshold on COMBINED train+test energy, then segment point-adjustment —
    # matches tsdino_timemixer/TSMixerAnomaly.py so baselines are comparable.
    thresh = np.percentile(
        np.concatenate([train_energy, scores]), 100 - a.anomaly_ratio)
    preds  = (scores > thresh).astype(int)
    labels, preds = _adjustment(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0)
    print(f"  >> {a.model}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}  "
          f"(anomaly_ratio={a.anomaly_ratio}%, point-adjusted, combined-threshold)")
    return {"precision": prec, "recall": rec, "f1": f1}


# ══════════════════════════════════════════════════════════════════════════════
def _summary(task, dataset, rows):
    print(f"\n{'='*64}\n  SUMMARY | {task} | {dataset}\n{'='*64}")
    if task == "long_term_forecast":
        print(f"  {'model':<26}{'pred_len':>9}{'MSE':>10}{'MAE':>10}")
        for model, res in rows:
            if isinstance(res, dict):
                for pl, (mse, mae) in res.items():
                    print(f"  {model:<26}{pl:>9}{mse:>10.4f}{mae:>10.4f}")
            else:
                print(f"  {model:<26}{'—':>9}{'  ' + str(res)}")
    elif task == "classification":
        print(f"  {'model':<26}{'accuracy':>10}")
        for model, res in rows:
            val = f"{res:.4f}" if isinstance(res, float) else str(res)
            print(f"  {model:<26}{val:>10}")
    else:
        print(f"  {'model':<26}{'P':>8}{'R':>8}{'F1':>8}")
        for model, res in rows:
            if isinstance(res, dict):
                print(f"  {model:<26}{res['precision']:>8.4f}{res['recall']:>8.4f}{res['f1']:>8.4f}")
            else:
                print(f"  {model:<26}{'  ' + str(res)}")
    print("=" * 64)


def main():
    p = argparse.ArgumentParser(description="Unified TSLib model runner over our data loaders")
    p.add_argument("--model", required=True,
                   help="model name, comma-list, or 'all' (all non-foundation models)")
    p.add_argument("--task", required=True, choices=sorted(TASK_ALIASES))
    p.add_argument("--dataset", required=True)

    p.add_argument("--no_defaults", dest="use_defaults", action="store_false", default=True,
                   help="ignore TSL scripts; use this file's built-in defaults")

    # shapes
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--label_len", type=int, default=48)
    p.add_argument("--pred_len", type=int, nargs="+", default=None,
                   help="forecast horizon(s); default 96 192 336 720")
    p.add_argument("--patch_len", type=int, default=16)

    # model size
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--d_ff", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--e_layers", type=int, default=2)
    p.add_argument("--d_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--factor", type=int, default=3)
    p.add_argument("--moving_avg", type=int, default=25)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--num_kernels", type=int, default=6)
    p.add_argument("--seg_len", type=int, default=48)
    p.add_argument("--channel_independence", type=int, default=1)
    p.add_argument("--down_sampling_layers", type=int, default=0)
    p.add_argument("--down_sampling_window", type=int, default=1)
    p.add_argument("--embed", type=str, default="timeF")
    p.add_argument("--freq", type=str, default="h")

    # optimization
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--anomaly_ratio", type=float, default=1.0)

    p.add_argument("--device", type=str, default="auto", help="auto | cuda:0 | mps | cpu")
    p.add_argument("--seed", type=int, default=2021)

    a = p.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    import random; random.seed(a.seed)

    device = pick_device(a.device)
    print(f"device: {device}  |  use_defaults={a.use_defaults}")

    task = TASK_ALIASES[a.task]
    runner = {"long_term_forecast": run_forecast,
              "classification": run_classify,
              "anomaly_detection": run_anomaly}[task]

    if a.model == "all":
        models = discover_models()
    else:
        models = [m.strip() for m in a.model.split(",") if m.strip()]
    print(f"models ({len(models)}): {', '.join(models)}")

    rows = []
    for model in models:
        ma = SimpleNamespace(**vars(a)); ma.model = model
        try:
            rows.append((model, runner(ma, device)))
        except Exception as e:
            print(f"\n[SKIP] {model}: {type(e).__name__}: {e}")
            if os.environ.get("TSL_TRACE"):
                traceback.print_exc()
            rows.append((model, f"FAILED ({type(e).__name__})"))

    if len(models) > 1:
        _summary(task, a.dataset, rows)


if __name__ == "__main__":
    main()
