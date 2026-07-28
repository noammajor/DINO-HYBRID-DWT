#!/usr/bin/env python
"""
tslib_xdomain_zeroshot.py — supervised ZERO-SHOT cross-domain transfer for any
Time-Series-Library model (DLinear, iTransformer, TimeMixer, PatchTST, ...).

Protocol (architecture-agnostic, works for linear models too):
  Train the full supervised model on the SOURCE dataset (early stop on source
  val), then run inference on the TARGET test set with NO target adaptation.

Reuses run_tslib_benchmark.py for the data loaders, the TSLib Model classes, and
the per-dataset default hyperparameters (via effective_args), so numbers are
comparable to the in-domain supervised baselines.

Same-variable-count pairs (e.g. any ETT->ETT) transfer directly. For
cross-variable-count pairs (e.g. weather(21)->etth1(7)) only channel-independent
models are supported: the model is trained at the source's channel count, then
rebuilt at the target's count with the shape-matched (channel-independent)
weights copied over and the C-sized RevIN affine reinitialized.
The SOURCE dataset drives the architecture + seq_len defaults (used for both).

How to run
----------
Run from the repo root (imports run_tslib_benchmark.py, data_loaders, dataset_registry).

  python scripts/xdomain/tslib_xdomain_zeroshot.py --model DLinear      --source etth1 --target etth2 --device cuda:0
  python scripts/xdomain/tslib_xdomain_zeroshot.py --model iTransformer --source ettm1 --target ettm2 --device cuda:0
  python scripts/xdomain/tslib_xdomain_zeroshot.py --model TimeMixer    --source etth1 --target ettm1 --device cuda:0

  # cross-variable-count transfer (only channel-independent models), pinned GPU, captured
  CUDA_VISIBLE_DEVICES=2 python scripts/xdomain/tslib_xdomain_zeroshot.py --model iTransformer \
      --source weather --target etth1 --device cuda:0 \
      > logs/tslib_xdomain_zeroshot/weather_to_etth1.log 2>&1

Flags
-----
  --model       (required) any TSLib model name (DLinear | iTransformer | TimeMixer
                | PatchTST | ...). Cross-variable-count transfer is only supported
                for models in _CROSS_C_MODELS (PatchTST, DLinear, SparseTSF,
                TimeMixer, iTransformer); other models skip mismatched-C pairs.
  --source      (required) source dataset key; full model trained here.
  --target      (required) target dataset key; only its TEST split is used
                (zero-shot). If == --source, prints "nothing to do" and exits.
  --device      auto | cuda:0 | cpu   (default: auto)
  --pred_lens   one or more horizons  (default: 96 192 336 720); per-horizon result + mean.

Output
------
  STDOUT only (no files written): per-epoch losses, cross-C copy stats, per-horizon
  "MSE=.. MAE=.." lines, and a mean. Redirect to keep them.

Notes
-----
  * Cross-C: differing source/target c_in with an unsupported model is [skip]ped.
    For supported models it trains at the source C, rebuilds at the target C, copies
    shape-matched (channel-independent) weights, and reinitialises the RevIN affine.
  * TimeMixer: down_sampling_layers=3 / window=2 / method=avg are injected (the
    generic builder leaves 0, which raises an IndexError).
  * LIBRARY: this module is imported by sparsetsf_xdomain_zeroshot.py and
    timebase_xdomain_zeroshot.py — it exports run_pair(model, source, target,
    device, pred_lens) plus _base_args/_loader/_make_forward/_train/_evaluate and
    _CROSS_C_MODELS. Do not delete it.

Usage:
    python scripts/xdomain/tslib_xdomain_zeroshot.py --model DLinear --source etth1 --target etth2
    python scripts/xdomain/tslib_xdomain_zeroshot.py --model iTransformer --source weather --target electricity --device cuda:0
"""
import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import run_tslib_benchmark as B  # noqa: E402  (has __main__ guard; safe to import)

# Models whose parameters are NOT sized to the variable count, so they can
# transfer across datasets with different C:
#   - channel-independent (weights shared per channel): PatchTST, DLinear,
#     SparseTSF, TimeMixer.
#   - variables-as-tokens (weights shared across tokens): iTransformer.
# The rest (TimesNet, FEDformer, Autoformer, ...) bake C into a Conv1d input
# embedding / output projection and cannot transfer across C without reinit.
_CROSS_C_MODELS = {"PatchTST", "DLinear", "SparseTSF", "TimeMixer", "iTransformer"}


def _base_args(model):
    """Argparse defaults from run_tslib_benchmark.main(), as a namespace."""
    return SimpleNamespace(
        model=model, use_defaults=True,
        seq_len=96, label_len=48, pred_len=None, patch_len=16,
        d_model=64, d_ff=128, n_heads=8, e_layers=2, d_layers=1,
        dropout=0.1, factor=3, moving_avg=25, top_k=5, num_kernels=6, seg_len=48,
        channel_independence=1, down_sampling_layers=0, down_sampling_window=1,
        embed="timeF", freq="h", epochs=10, batch_size=32, lr=1e-4,
        patience=5, num_workers=4, dataset=None,
    )


def _loader(csv, split, ea, pred_len, shuffle):
    from data_loaders.data_puller import PatchTSTForcastingAdapter
    ds = B._ForecastWindowAdapter(
        PatchTSTForcastingAdapter(csv, split, ea.seq_len, pred_len, ea.patch_len,
                                  label_len=ea.label_len),
        freq=ea.freq)
    return torch.utils.data.DataLoader(
        ds, batch_size=ea.batch_size, shuffle=shuffle,
        num_workers=ea.num_workers, drop_last=(split == "train"))


def _make_forward(model, device, pred_len, label_len):
    def forward_batch(bx, by, bxm, bym):
        bx, by = bx.float().to(device), by.float().to(device)
        bxm, bym = bxm.float().to(device), bym.float().to(device)
        dec_zeros = torch.zeros_like(by[:, -pred_len:, :])
        dec_inp = torch.cat([by[:, :label_len, :], dec_zeros], dim=1).float()
        try:                                    # TSLib 4-arg forward
            out = model(bx, bxm, dec_inp, bym)
        except TypeError:                       # single-arg forward(x)
            out = model(bx)
        return out[:, -pred_len:, :], by[:, -pred_len:, :]
    return forward_batch


def _train(model, tr, va, fwd, crit, lr, epochs, patience, tag):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, best_state, no_improve = float("inf"), None, 0
    for epoch in range(epochs):
        model.train(); t0 = time.time(); losses = []
        for batch in tr:
            opt.zero_grad()
            out, tgt = fwd(*batch)
            loss = crit(out, tgt); loss.backward(); opt.step()
            losses.append(loss.item())
        model.eval(); vlosses = []
        with torch.no_grad():
            for batch in va:
                out, tgt = fwd(*batch)
                vlosses.append(crit(out, tgt).item())
        vl = float(np.mean(vlosses)) if vlosses else float("inf")
        print(f"    [{tag}] epoch {epoch+1}/{epochs} "
              f"train={np.mean(losses):.4f} val={vl:.4f} ({time.time()-t0:.1f}s)", flush=True)
        if vl < best_val:
            best_val, no_improve = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    [{tag}] early stopping", flush=True); break
    if best_state is not None:
        model.load_state_dict(best_state)


@torch.no_grad()
def _evaluate(model, te, fwd):
    model.eval(); preds, trues = [], []
    for batch in te:
        out, tgt = fwd(*batch)
        preds.append(out.cpu().numpy()); trues.append(tgt.cpu().numpy())
    P, T = np.concatenate(preds), np.concatenate(trues)
    return float(np.mean((P - T) ** 2)), float(np.mean(np.abs(P - T)))


def run_pair(model_name, source, target, device, pred_lens):
    from dataset_registry import get_dataset_info
    src_csv = get_dataset_info(source)["csv_path"]
    tgt_info = get_dataset_info(target)
    tgt_csv, c_in = tgt_info["csv_path"], tgt_info["c_in"]
    src_c_in = get_dataset_info(source)["c_in"]
    cross_c  = src_c_in != c_in
    if cross_c and model_name not in _CROSS_C_MODELS:
        print(f"  [skip] {model_name} bakes the variable count into its params; "
              f"cannot transfer {source}(C={src_c_in}) -> {target}(C={c_in}) "
              f"without reinit. Cross-C models: {sorted(_CROSS_C_MODELS)}.", flush=True)
        return {}

    a = _base_args(model_name)
    a.dataset = source            # SOURCE drives architecture + seq_len defaults
    crit = nn.MSELoss()
    results = {}

    print(f"\n=== {model_name}(sup) ZERO-SHOT  {source} -> {target}  (C={c_in}) ===", flush=True)
    for pred_len in pred_lens:
        ea, cfg = B.effective_args(a, "long_term_forecast", pred_len)
        if cfg:
            print(f"  [defaults from {source}] pl={pred_len}: {cfg}", flush=True)
        else:
            print(f"  [defaults] pl={pred_len}: none found — using built-in defaults", flush=True)

        src_tr = _loader(src_csv, "train", ea, pred_len, True)
        src_va = _loader(src_csv, "val",   ea, pred_len, False)
        tgt_te = _loader(tgt_csv, "test",  ea, pred_len, False)

        def _build(C):
            margs = B.build_model_args(ea, "long_term_forecast")
            margs.pred_len = pred_len
            margs.enc_in = margs.dec_in = margs.c_out = C
            if model_name == "TimeMixer":
                # TimeMixer is inherently multi-scale: it down-samples the input
                # into several resolutions and mixes them. The generic builder
                # leaves down_sampling_layers=0 → season_list has one element →
                # IndexError at season_list[1]. Inject the standard config.
                margs.down_sampling_layers = 3
                margs.down_sampling_window = 2
                margs.down_sampling_method = "avg"
            return B.build_tslib_model(margs).to(device)

        # Train on the SOURCE at the source's channel count.
        model = _build(src_c_in)
        _train(model, src_tr, src_va,
               _make_forward(model, device, pred_len, ea.label_len),
               crit, ea.lr, ea.epochs, ea.patience, tag=f"src:{source}")

        # Cross-variable-count transfer: rebuild at the target's channel count and
        # copy only the shape-matched (channel-independent) weights. The C-sized
        # RevIN affine is reinitialized (→ plain per-instance norm), mirroring the
        # WINO-TS cross-C loader.
        if cross_c:
            tgt_model = _build(c_in)
            src_sd, tgt_sd = model.state_dict(), tgt_model.state_dict()
            keep = {k: v for k, v in src_sd.items()
                    if k in tgt_sd and tgt_sd[k].shape == v.shape}
            reinit = [k for k in tgt_sd if k not in keep]
            tgt_model.load_state_dict(keep, strict=False)
            print(f"    [cross-C {src_c_in}->{c_in}] copied {len(keep)}/{len(tgt_sd)} "
                  f"weights; reinit {len(reinit)} C-sized", flush=True)
            model = tgt_model

        fwd = _make_forward(model, device, pred_len, ea.label_len)
        mse, mae = _evaluate(model, tgt_te, fwd)
        print(f"  >> {source}->{target}  pred_len={pred_len}  MSE={mse:.4f}  MAE={mae:.4f}", flush=True)
        results[pred_len] = (mse, mae)

    if results:
        mm = np.mean([v[0] for v in results.values()])
        aa = np.mean([v[1] for v in results.values()])
        print(f"  >> {source}->{target}  MEAN  MSE={mm:.4f}  MAE={aa:.4f}", flush=True)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="TSLib model name (DLinear | iTransformer | TimeMixer | PatchTST | ...)")
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--device", default="auto", help="auto | cuda:0 | cpu")
    p.add_argument("--pred_lens", type=int, nargs="+", default=[96, 192, 336, 720])
    args = p.parse_args()

    if args.source == args.target:
        print("source == target; nothing to do."); return
    device = B.pick_device(args.device)
    run_pair(args.model, args.source, args.target, device, args.pred_lens)


if __name__ == "__main__":
    main()
