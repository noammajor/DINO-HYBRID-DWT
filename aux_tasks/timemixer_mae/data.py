"""Raw sliding-window dataset for MAE pretraining.

Reuses PatchTST's ETT/custom splits + normalization so the windows match the
forecasting benchmark exactly. Each item is a window [seq_len, n_vars]; MAE has
no targets (the masked input is its own target).
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


class MAEWindows:
    """Sliding windows [seq_len, n_vars] for a given split."""

    def __init__(self, csv_path, split, seq_len, block_len):
        root = os.path.dirname(os.path.abspath(csv_path))
        fname = os.path.basename(csv_path)
        # pred_len must be >0 for the underlying dataset; block_len is a safe minimum.
        size = [seq_len, 0, block_len]
        self._ds = _ds_cls(fname)(root, split=split, size=size, features='M',
                                  data_path=fname, scale=True)
        print(f"MAEWindows [{split}] ({fname}): {len(self._ds)} windows")

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        seq_x, _ = self._ds[idx]    # [seq_len, n_vars]
        return seq_x


def build_loaders(args, torch):
    train_ds = MAEWindows(args.data_path, 'train', args.seq_len, args.block_len)
    try:
        val_ds = MAEWindows(args.data_path, 'val', args.seq_len, args.block_len)
    except Exception:
        val_ds = None

    def _loader(ds, shuffle):
        return torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=shuffle,
            num_workers=args.num_workers, pin_memory=True, drop_last=shuffle)

    return _loader(train_ds, True), (_loader(val_ds, False) if val_ds else None)


def _forecast_ds(csv_path, split, seq_len, pred_len):
    """(context [seq_len, C], target [pred_len, C]) pairs for downstream forecasting."""
    root = os.path.dirname(os.path.abspath(csv_path))
    fname = os.path.basename(csv_path)
    ds = _ds_cls(fname)(root, split=split, size=[seq_len, 0, pred_len],
                        features='M', data_path=fname, scale=True)
    print(f"forecast {split} ({fname}): {len(ds)} windows")
    return ds


def build_forecast_loaders(args, torch):
    """train/val/test DataLoaders of (context, target) for the forecasting eval."""
    def _make(split):
        try:
            return _forecast_ds(args.data_path, split, args.seq_len, args.pred_len)
        except Exception:
            return None

    def _loader(ds, shuffle):
        if ds is None:
            return None
        return torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=shuffle,
            num_workers=args.num_workers, pin_memory=True, drop_last=False)

    return (_loader(_make('train'), True),
            _loader(_make('val'), False),
            _loader(_make('test'), False))
