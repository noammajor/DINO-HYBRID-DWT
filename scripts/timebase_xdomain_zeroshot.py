#!/usr/bin/env python
"""
timebase_xdomain_zeroshot.py — supervised ZERO-SHOT cross-domain transfer for
TimeBase (TimeBase-main/models/TimeBase.py), mirroring tslib_xdomain_zeroshot.py.

TimeBase is NOT a Time-Series-Library model, so it can't go through the TSLib
driver. This wrapper reuses run_tslib_benchmark's data loaders + the tslib
zero-shot script's generic _loader, but builds/trains TimeBase directly.

Protocol: train TimeBase on the SOURCE dataset (early stop on source val), then
run inference on the TARGET test set with NO target adaptation. Same output
format ("<src>-><tgt> pred_len=N MSE=.. MAE=.." + MEAN) so logs line up with the
TimeMixer/FEDformer/TimesNet cross-domain runs.

Usage
-----
  python scripts/timebase_xdomain_zeroshot.py --source etth1 --target etth2 --device cuda:0
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import run_tslib_benchmark as B          # noqa: E402  (sets up data_loaders path)
from tslib_xdomain_zeroshot import _loader  # noqa: E402  (generic csv/split loader)

# TimeBase.py is self-contained (defines cal_orthogonal_loss inline) — load by
# path to avoid clashing with other `models` packages on sys.path.
_TB_PATH = _ROOT / "TimeBase-main" / "models" / "TimeBase.py"
_spec = importlib.util.spec_from_file_location("timebase_model", _TB_PATH)
_tb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tb)
TimeBaseModel = _tb.Model

# Per-dataset TimeBase hyperparameters (from run_baselines.sh TIMEBASE config).
# The SOURCE dataset drives period_len/basis_num, mirroring the tslib wrapper.
TB_CFG = {
    "etth1": dict(period_len=24, basis_num=6),
    "etth2": dict(period_len=24, basis_num=6),
    "ettm1": dict(period_len=4,  basis_num=20),
    "ettm2": dict(period_len=4,  basis_num=20),
}

SEQ_LEN = 336          # TimeBase's native context (ours_336 baseline)
LABEL_LEN = 0
PATCH_LEN = 16
ORTHO_WEIGHT = 0.2
LR = 1e-2
EPOCHS = 10
PATIENCE = 5
BATCH_SIZE = 64


def _build(source, pred_len, c_in, device):
    cfg = TB_CFG[source]
    configs = SimpleNamespace(
        seq_len=SEQ_LEN, pred_len=pred_len, enc_in=c_in,
        period_len=cfg["period_len"], basis_num=cfg["basis_num"],
        use_period_norm=1, use_orthogonal=1, individual=0,
    )
    return TimeBaseModel(configs).to(device)


def _forward(model, batch, device, pred_len):
    """batch = (seq_x, seq_y, x_mark, y_mark) -> (pred, target, ortho_loss)."""
    bx, by = batch[0].float().to(device), batch[1].float().to(device)
    out = model(bx)                          # (pred, ortho) since use_orthogonal=1
    ortho = None
    if isinstance(out, tuple):
        out, ortho = out
    return out[:, -pred_len:, :], by[:, -pred_len:, :], ortho


def _train(model, tr, va, device, pred_len, tag):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()
    best_val, best_state, no_improve = float("inf"), None, 0
    for epoch in range(EPOCHS):
        model.train(); t0 = time.time(); losses = []
        for batch in tr:
            opt.zero_grad()
            out, tgt, ortho = _forward(model, batch, device, pred_len)
            loss = crit(out, tgt)
            if ortho is not None:
                loss = loss + ORTHO_WEIGHT * ortho
            loss.backward(); opt.step()
            losses.append(loss.item())
        model.eval(); vlosses = []
        with torch.no_grad():
            for batch in va:
                out, tgt, _ = _forward(model, batch, device, pred_len)
                vlosses.append(crit(out, tgt).item())
        vl = float(np.mean(vlosses)) if vlosses else float("inf")
        print(f"    [{tag}] epoch {epoch+1}/{EPOCHS} "
              f"train={np.mean(losses):.4f} val={vl:.4f} ({time.time()-t0:.1f}s)", flush=True)
        if vl < best_val:
            best_val, no_improve = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"    [{tag}] early stopping", flush=True); break
    if best_state is not None:
        model.load_state_dict(best_state)


@torch.no_grad()
def _evaluate(model, te, device, pred_len):
    model.eval(); preds, trues = [], []
    for batch in te:
        out, tgt, _ = _forward(model, batch, device, pred_len)
        preds.append(out.cpu().numpy()); trues.append(tgt.cpu().numpy())
    P, T = np.concatenate(preds), np.concatenate(trues)
    return float(np.mean((P - T) ** 2)), float(np.mean(np.abs(P - T)))


def run_pair(source, target, device, pred_lens):
    from dataset_registry import get_dataset_info
    src_csv = get_dataset_info(source)["csv_path"]
    tgt_info = get_dataset_info(target)
    tgt_csv, c_in = tgt_info["csv_path"], tgt_info["c_in"]

    ea = SimpleNamespace(seq_len=SEQ_LEN, label_len=LABEL_LEN, patch_len=PATCH_LEN,
                         freq="h", batch_size=BATCH_SIZE, num_workers=4)
    results = {}
    print(f"\n=== TimeBase(sup) ZERO-SHOT  {source} -> {target}  (C={c_in}) ===", flush=True)
    print(f"  [defaults from {source}] period_len={TB_CFG[source]['period_len']} "
          f"basis_num={TB_CFG[source]['basis_num']} seq_len={SEQ_LEN}", flush=True)

    for pred_len in pred_lens:
        src_tr = _loader(src_csv, "train", ea, pred_len, True)
        src_va = _loader(src_csv, "val",   ea, pred_len, False)
        tgt_te = _loader(tgt_csv, "test",  ea, pred_len, False)

        model = _build(source, pred_len, c_in, device)
        _train(model, src_tr, src_va, device, pred_len, tag=f"src:{source}")
        mse, mae = _evaluate(model, tgt_te, device, pred_len)
        print(f"  >> {source}->{target}  pred_len={pred_len}  MSE={mse:.4f}  MAE={mae:.4f}", flush=True)
        results[pred_len] = (mse, mae)

    if results:
        mm = np.mean([v[0] for v in results.values()])
        aa = np.mean([v[1] for v in results.values()])
        print(f"  >> {source}->{target}  MEAN  MSE={mm:.4f}  MAE={aa:.4f}", flush=True)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, choices=list(TB_CFG))
    p.add_argument("--target", required=True, choices=list(TB_CFG))
    p.add_argument("--device", default="auto", help="auto | cuda:0 | cpu")
    p.add_argument("--pred_lens", type=int, nargs="+", default=[96, 192, 336, 720])
    args = p.parse_args()
    if args.source == args.target:
        print("source == target; nothing to do."); return
    device = B.pick_device(args.device)
    run_pair(args.source, args.target, device, args.pred_lens)


if __name__ == "__main__":
    main()
