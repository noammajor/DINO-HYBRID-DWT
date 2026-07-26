"""Downstream in-domain forecasting eval for the NTP-pretrained TimeMixer encoder.

Loads a pretrained encoder (encoder.* from an NTP checkpoint), attaches a fresh
linear forecasting head, trains it on the in-domain train split (default: linear
probe — encoder frozen), keeps the best-val checkpoint, then reports test MSE/MAE.

The forecasting model is the NTP model itself (encoder + linear head); here we
reload only the pretrained encoder and (re)train the head.

Example (run after train.py):
  python forecast.py --init_ckpt ./checkpoints/etth1/checkpoint_best.pth \
      --data_path ../data/ETTh1.csv --c_in 7 --seq_len 336 --pred_len 96 \
      --mode linear_probe
"""
import os
import sys
import argparse

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import torch
import torch.nn as nn
from model import TimeMixerNTP as TimeMixerForecast
from data import build_forecast_loaders


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--init_ckpt', default=None,
                   help='pretrained checkpoint with encoder.* (omit = random-init baseline)')
    p.add_argument('--mode', choices=['linear_probe', 'finetune'], default='linear_probe',
                   help='linear_probe = freeze encoder, train head only (default)')
    p.add_argument('--data_path', required=True)
    p.add_argument('--c_in', type=int, default=7)
    p.add_argument('--seq_len', type=int, default=336)
    p.add_argument('--pred_len', type=int, default=96)
    p.add_argument('--num_workers', type=int, default=6)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--clip_grad', type=float, default=1.0)
    p.add_argument('--dropout', type=float, default=0.1)
    # backbone (must match the pretrained encoder)
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--e_layers', type=int, default=3)
    p.add_argument('--d_ff', type=int, default=256)
    p.add_argument('--down_sampling_layers', type=int, default=3)
    p.add_argument('--down_sampling_window', type=int, default=2)
    p.add_argument('--down_sampling_method', default='avg')
    p.add_argument('--decomp_method', default='moving_avg')
    p.add_argument('--moving_avg', type=int, default=25)
    p.add_argument('--top_k', type=int, default=5)
    p.add_argument('--use_norm', type=int, default=1)
    p.add_argument('--channel_independence', type=int, default=1)
    return p.parse_args()


def build_model(args, device):
    return TimeMixerForecast(
        c_in=args.c_in, seq_len=args.seq_len, pred_len=args.pred_len,
        d_model=args.d_model, e_layers=args.e_layers, d_ff=args.d_ff,
        dropout=args.dropout, use_norm=args.use_norm,
        down_sampling_layers=args.down_sampling_layers,
        down_sampling_window=args.down_sampling_window,
        down_sampling_method=args.down_sampling_method,
        decomp_method=args.decomp_method, moving_avg=args.moving_avg,
        top_k=args.top_k, channel_independence=args.channel_independence,
    ).to(device)


def load_encoder(model, init_ckpt, device):
    ckpt = torch.load(init_ckpt, map_location=device, weights_only=False)
    enc_sd = ckpt.get('encoder', ckpt)
    missing, unexpected = model.encoder.load_state_dict(enc_sd, strict=False)
    loaded = len(model.encoder.state_dict()) - len(missing)
    print(f"  loaded {loaded}/{len(model.encoder.state_dict())} encoder weights "
          f"from {init_ckpt}")
    if missing:
        print(f"  encoder missing (random-init): {missing}")


def evaluate(model, loader, device):
    """Mean MSE / MAE over a loader."""
    model.eval()
    se = ae = n = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x)
            bs = x.size(0)
            se += ((out - y) ** 2).mean().item() * bs
            ae += (out - y).abs().mean().item() * bs
            n += bs
    return se / max(n, 1), ae / max(n, 1)


def main():
    args = get_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = build_model(args, device)
    if args.init_ckpt:
        load_encoder(model, args.init_ckpt, device)
    else:
        print("  no --init_ckpt: random-init baseline")

    if args.mode == 'linear_probe':
        for prm in model.encoder.parameters():
            prm.requires_grad = False
        params = model.head.parameters()
        print("  mode: linear_probe — encoder FROZEN")
    else:
        params = model.parameters()
        print("  mode: finetune — encoder + head trained")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {trainable:,}")

    train_loader, val_loader, test_loader = build_forecast_loaders(args, torch)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=args.epochs * len(train_loader),
        pct_start=0.3, anneal_strategy='cos')

    best_val, best_state = float('inf'), None
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
            optimizer.step()
            scheduler.step()

        if val_loader is not None:
            v_mse, _ = evaluate(model, val_loader, device)
            print(f"  epoch {epoch}  val MSE {v_mse:.6f}  (best {min(best_val, v_mse):.6f})")
            if v_mse < best_val:
                best_val = v_mse
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  restored best-val checkpoint (val MSE {best_val:.6f})")

    test_mse, test_mae = evaluate(model, test_loader, device)
    print("=" * 50)
    print(f"TEST  MSE {test_mse:.6f}  MAE {test_mae:.6f}  "
          f"[{args.mode}, {args.seq_len}->{args.pred_len}]")
    print("=" * 50)
    return test_mse


if __name__ == '__main__':
    main()
