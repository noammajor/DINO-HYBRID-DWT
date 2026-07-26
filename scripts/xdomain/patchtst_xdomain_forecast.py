#!/usr/bin/env python
"""
patchtst_xdomain_forecast.py — supervised PatchTST CROSS-DOMAIN transfer
(frozen source encoder + linear head retrained on target).

Mirrors the DINO frozen-backbone linear-probe transfer, but with a *supervised*
PatchTST encoder:

  Phase 1 (source):  train the full supervised PatchTST (patch_embedding +
                     encoder + head) on the SOURCE dataset (early stop on source val).
  Phase 2 (target):  freeze patch_embedding + encoder, RE-INITIALISE the linear
                     head, and train ONLY the head on the TARGET train split
                     (early stop on target val).
  Eval:              report MSE/MAE on the TARGET test split.

It reuses run_tslib_benchmark.py for the data loaders, the TSLib PatchTST model,
and the per-dataset default hyperparameters (seq_len / patch_len / d_model /
e_layers / lr / epochs / batch_size), so the numbers are directly comparable to
the in-domain supervised PatchTST baseline (logs/patchtst_sup/).

All ETT datasets are 7-variable, so enc_in matches across any source/target pair.
The SOURCE config drives the architecture and seq_len (used for both datasets).

Usage
-----
  python scripts/patchtst_xdomain_forecast.py --source etth1 --target etth2 --device cuda:0
  # a full matrix is driven by the shell loop in the message that shipped this file.
"""
import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

# Import the benchmark module — it wires sys.path, chdir's into the TSLib dir,
# and exposes effective_args / build_model_args / build_tslib_model / adapters.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import run_tslib_benchmark as B  # noqa: E402  (has __main__ guard; safe to import)


def _base_args(model="PatchTST"):
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


def _loader(csv, split, ea, pred_len, c_in, shuffle):
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
        out = model(bx, bxm, dec_inp, bym)
        return out[:, -pred_len:, :], by[:, -pred_len:, :]
    return forward_batch


def _train(model, params, tr, va, fwd, crit, lr, epochs, patience, tag):
    opt = torch.optim.Adam(params, lr=lr)
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


def run_pair(source, target, device, pred_lens, head_epochs):
    from dataset_registry import get_dataset_info
    src_csv = get_dataset_info(source)["csv_path"]
    tgt_info = get_dataset_info(target)
    tgt_csv, c_in = tgt_info["csv_path"], tgt_info["c_in"]

    a = _base_args("PatchTST")
    a.dataset = source            # SOURCE drives architecture + seq_len defaults
    crit = nn.MSELoss()
    results = {}

    print(f"\n=== PatchTST(sup) X-DOMAIN  {source} -> {target}  (C={c_in}) ===", flush=True)
    for pred_len in pred_lens:
        ea, cfg = B.effective_args(a, "long_term_forecast", pred_len)
        if cfg:
            print(f"  [defaults from {source}] pl={pred_len}: {cfg}", flush=True)

        # loaders
        src_tr = _loader(src_csv, "train", ea, pred_len, c_in, True)
        src_va = _loader(src_csv, "val",   ea, pred_len, c_in, False)
        tgt_tr = _loader(tgt_csv, "train", ea, pred_len, c_in, True)
        tgt_va = _loader(tgt_csv, "val",   ea, pred_len, c_in, False)
        tgt_te = _loader(tgt_csv, "test",  ea, pred_len, c_in, False)

        # model
        margs = B.build_model_args(ea, "long_term_forecast")
        margs.pred_len = pred_len
        margs.enc_in = margs.dec_in = margs.c_out = c_in
        model = B.build_tslib_model(margs).to(device)
        fwd = _make_forward(model, device, pred_len, ea.label_len)

        # Phase 1 — supervised full training on SOURCE
        _train(model, model.parameters(), src_tr, src_va, fwd, crit,
               ea.lr, ea.epochs, ea.patience, tag=f"src:{source}")

        # Freeze encoder; re-init the linear head for a clean probe on TARGET
        for p in model.parameters():
            p.requires_grad = False
        model.head.linear.reset_parameters()
        for p in model.head.parameters():
            p.requires_grad = True

        # Phase 2 — train ONLY the head on TARGET
        _train(model, model.head.parameters(), tgt_tr, tgt_va, fwd, crit,
               ea.lr, head_epochs, ea.patience, tag=f"probe:{target}")

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
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--device", default="auto", help="auto | cuda:0 | cpu")
    p.add_argument("--pred_lens", type=int, nargs="+", default=[96, 192, 336, 720])
    p.add_argument("--head_epochs", type=int, default=20,
                   help="epochs for the target-head probe phase")
    args = p.parse_args()

    if args.source == args.target:
        print("source == target; nothing to do."); return
    device = B.pick_device(args.device)
    run_pair(args.source, args.target, device, args.pred_lens, args.head_epochs)


if __name__ == "__main__":
    main()
