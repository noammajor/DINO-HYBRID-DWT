#!/usr/bin/env python
"""
tslib_xdomain_probe.py — supervised CROSS-DOMAIN transfer via
FROZEN source encoder + linear head retrained on target, for any TSLib model
that has a clean encoder/head split (PatchTST, iTransformer, TimeMixer).

Protocol (matches the DINO frozen-backbone linear-probe transfer):
  Phase 1 (source):  train the full supervised model on the SOURCE dataset
                     (early stop on source val).
  Phase 2 (target):  freeze the encoder, RE-INITIALISE the head module(s), train
                     ONLY the head on the TARGET train split (early stop on target val).
  Eval:              MSE/MAE on the TARGET test split.

"Head" is model-specific (HEAD_PREFIXES below); everything else is frozen.
Reuses run_tslib_benchmark.py for loaders, model classes, and per-dataset default
hyperparameters (via effective_args). All ETT datasets are 7-variable, so enc_in
matches across any source/target pair; the SOURCE config drives architecture+seq_len.

Usage
-----
  python scripts/tslib_xdomain_probe.py --model iTransformer --source etth1 --target etth2 --device cuda:0
  python scripts/tslib_xdomain_probe.py --model TimeMixer    --source ettm1 --target ettm2 --device cuda:0
  python scripts/tslib_xdomain_probe.py --model PatchTST     --source etth1 --target ettm1 --device cuda:0
"""
import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import run_tslib_benchmark as B  # noqa: E402  (has __main__ guard; safe to import)

# Top-level module names that constitute the trainable forecast HEAD per model.
# Everything else in the model is frozen after source training.
HEAD_PREFIXES = {
    "PatchTST":     ["head"],
    "iTransformer": ["projection"],
    "TimeMixer":    ["predict_layers", "projection_layer",
                     "out_res_layers", "regression_layers"],
}


def _base_args(model):
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
        try:
            out = model(bx, bxm, dec_inp, bym)
        except TypeError:
            out = model(bx)
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


def _freeze_to_head(model, prefixes):
    """Freeze all params, re-init and unfreeze only the head modules."""
    for p in model.parameters():
        p.requires_grad = False
    # re-init head submodules for a clean probe
    for name, module in model.named_modules():
        if name and name.split(".")[0] in prefixes and hasattr(module, "reset_parameters"):
            module.reset_parameters()
    # unfreeze head params
    n_head = 0
    for name, p in model.named_parameters():
        if name.split(".")[0] in prefixes:
            p.requires_grad = True
            n_head += p.numel()
    if n_head == 0:
        raise RuntimeError(f"No head params matched prefixes {prefixes}; "
                           f"top-level modules are {[n for n,_ in model.named_children()]}")
    n_total = sum(p.numel() for p in model.parameters())
    print(f"    [probe] head params: {n_head:,} / {n_total:,} trainable", flush=True)
    return (p for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def _evaluate(model, te, fwd):
    model.eval(); preds, trues = [], []
    for batch in te:
        out, tgt = fwd(*batch)
        preds.append(out.cpu().numpy()); trues.append(tgt.cpu().numpy())
    P, T = np.concatenate(preds), np.concatenate(trues)
    return float(np.mean((P - T) ** 2)), float(np.mean(np.abs(P - T)))


def run_pair(model_name, source, target, device, pred_lens, head_epochs):
    from dataset_registry import get_dataset_info
    prefixes = HEAD_PREFIXES[model_name]
    src_csv = get_dataset_info(source)["csv_path"]
    tgt_info = get_dataset_info(target)
    tgt_csv, c_in = tgt_info["csv_path"], tgt_info["c_in"]

    a = _base_args(model_name)
    a.dataset = source
    crit = nn.MSELoss()
    results = {}

    print(f"\n=== {model_name}(sup) FROZEN+HEAD  {source} -> {target}  (C={c_in}) ===", flush=True)
    for pred_len in pred_lens:
        ea, cfg = B.effective_args(a, "long_term_forecast", pred_len)
        if cfg:
            print(f"  [defaults from {source}] pl={pred_len}: {cfg}", flush=True)
        else:
            print(f"  [defaults] pl={pred_len}: none found — using built-in defaults", flush=True)

        src_tr = _loader(src_csv, "train", ea, pred_len, True)
        src_va = _loader(src_csv, "val",   ea, pred_len, False)
        tgt_tr = _loader(tgt_csv, "train", ea, pred_len, True)
        tgt_va = _loader(tgt_csv, "val",   ea, pred_len, False)
        tgt_te = _loader(tgt_csv, "test",  ea, pred_len, False)

        margs = B.build_model_args(ea, "long_term_forecast")
        margs.pred_len = pred_len
        margs.enc_in = margs.dec_in = margs.c_out = c_in
        model = B.build_tslib_model(margs).to(device)
        fwd = _make_forward(model, device, pred_len, ea.label_len)

        # Phase 1 — full supervised training on SOURCE
        _train(model, model.parameters(), src_tr, src_va, fwd, crit,
               ea.lr, ea.epochs, ea.patience, tag=f"src:{source}")

        # Freeze encoder; re-init + unfreeze head only
        head_params = _freeze_to_head(model, prefixes)

        # Phase 2 — train ONLY the head on TARGET
        _train(model, head_params, tgt_tr, tgt_va, fwd, crit,
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
    p.add_argument("--model", required=True, choices=sorted(HEAD_PREFIXES))
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--device", default="auto", help="auto | cuda:0 | cpu")
    p.add_argument("--pred_lens", type=int, nargs="+", default=[96, 192, 336, 720])
    p.add_argument("--head_epochs", type=int, default=20)
    args = p.parse_args()

    if args.source == args.target:
        print("source == target; nothing to do."); return
    device = B.pick_device(args.device)
    run_pair(args.model, args.source, args.target, device, args.pred_lens, args.head_epochs)


if __name__ == "__main__":
    main()
