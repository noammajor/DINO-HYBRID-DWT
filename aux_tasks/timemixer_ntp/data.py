"""(context, target) dataset for NTP pretraining.

Reuses PatchTST's ETT/custom splits + normalization. Each item is
(context [seq_len, n_vars], target [pred_len, n_vars]) — the next-step horizon.
"""
import os
import sys

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))


def _ds_cls(fname):
    patchtst = os.path.join(_REPO_ROOT, "PatchTST_self_supervised")
    if patchtst not in sys.path:
        sys.path.insert(0, patchtst)
    from src.data.pred_dataset import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
    low = fname.lower()
    if 'etth' in low:
        return Dataset_ETT_hour
    if 'ettm' in low:
        return Dataset_ETT_minute
    return Dataset_Custom


def _make(csv_path, split, seq_len, pred_len):
    root = os.path.dirname(os.path.abspath(csv_path))
    fname = os.path.basename(csv_path)
    size = [seq_len, 0, pred_len]
    ds = _ds_cls(fname)(root, split=split, size=size, features='M',
                        data_path=fname, scale=True)
    print(f"NTP {split} ({fname}): {len(ds)} windows")
    return ds


def build_loaders(args, torch):
    train_ds = _make(args.data_path, 'train', args.seq_len, args.pred_len)
    try:
        val_ds = _make(args.data_path, 'val', args.seq_len, args.pred_len)
    except Exception:
        val_ds = None

    def _loader(ds, shuffle):
        return torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=shuffle,
            num_workers=args.num_workers, pin_memory=True, drop_last=shuffle)

    return _loader(train_ds, True), (_loader(val_ds, False) if val_ds else None)


def build_forecast_loaders(args, torch):
    """train/val/test DataLoaders of (context, target) for the forecasting eval."""
    def _safe(split):
        try:
            return _make(args.data_path, split, args.seq_len, args.pred_len)
        except Exception:
            return None

    def _loader(ds, shuffle):
        if ds is None:
            return None
        return torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=shuffle,
            num_workers=args.num_workers, pin_memory=True, drop_last=False)

    return (_loader(_safe('train'), True),
            _loader(_safe('val'), False),
            _loader(_safe('test'), False))
