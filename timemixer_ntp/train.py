"""NTP pretraining of the TimeMixer backbone (standalone flow).

Next-step prediction: encode a window of --seq_len timesteps and predict the
next --pred_len points (MSE). Saves the backbone under the ``encoder.*`` keys so
it can be transferred to a downstream model.

Example:
  python train.py --data_path ../data/ETTh1.csv --c_in 7 \
      --seq_len 336 --pred_len 96 --output_dir ./checkpoints/etth1
"""
import os
import sys
import argparse
from pathlib import Path

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import torch
import torch.nn as nn
from model import TimeMixerNTP
from data import build_loaders


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_path', required=True)
    p.add_argument('--output_dir', default='./checkpoints/ntp')
    p.add_argument('--c_in', type=int, default=7)
    p.add_argument('--seq_len', type=int, default=336)
    p.add_argument('--pred_len', type=int, default=96, help='NTP horizon')
    p.add_argument('--num_workers', type=int, default=6)
    # optim
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--clip_grad', type=float, default=3.0)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--saveckp_freq', type=int, default=10)
    # backbone
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


def main():
    args = get_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model = TimeMixerNTP(
        c_in=args.c_in, seq_len=args.seq_len, pred_len=args.pred_len,
        d_model=args.d_model, e_layers=args.e_layers, d_ff=args.d_ff,
        dropout=args.dropout, use_norm=args.use_norm,
        down_sampling_layers=args.down_sampling_layers,
        down_sampling_window=args.down_sampling_window,
        down_sampling_method=args.down_sampling_method,
        decomp_method=args.decomp_method, moving_avg=args.moving_avg,
        top_k=args.top_k, channel_independence=args.channel_independence,
    ).to(device)

    train_loader, val_loader = build_loaders(args, torch)
    criterion = nn.MSELoss()
    print(f"[ntp] params={sum(p.numel() for p in model.parameters()):,}  "
          f"{args.seq_len}->{args.pred_len}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=args.epochs * len(train_loader),
        pct_start=0.3, anneal_strategy='cos')

    best = float('inf')
    for epoch in range(args.epochs):
        model.train()
        s, n = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            scheduler.step()
            s += loss.item(); n += 1
        train_loss = s / max(n, 1)

        val_loss = None
        if val_loader is not None:
            model.eval()
            vs, vn = 0.0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    vs += criterion(model(x), y).item(); vn += 1
            val_loss = vs / max(vn, 1)

        msg = f"  epoch {epoch}  train {train_loss:.6f}"
        if val_loss is not None:
            msg += f"  val {val_loss:.6f}"
        print(msg)

        ckpt = {'encoder': model.encoder.state_dict(), 'model': model.state_dict(),
                'epoch': epoch + 1, 'args': vars(args)}
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            torch.save(ckpt, os.path.join(args.output_dir, f'checkpoint{epoch}.pth'))
        target = val_loss if val_loss is not None else train_loss
        if target < best:
            best = target
            torch.save(ckpt, os.path.join(args.output_dir, 'checkpoint_best.pth'))
            print(f"    → new best ({best:.6f}) — saved checkpoint_best.pth")

    print(f"[ntp] done. best={best:.6f}  ckpt dir={args.output_dir}")


if __name__ == '__main__':
    main()
