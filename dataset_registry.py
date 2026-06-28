"""Central dataset registry.

Add a new entry to DATASETS to support a new CSV dataset.
The forecasting CSV root comes from data_paths.DATA_PATHS["forecasting_data_dir"]
— edit that file once when moving between machines.
"""

import os
from pathlib import Path

from data_paths import DATA_PATHS

_DATA_DIR = Path(DATA_PATHS["forecasting_data_dir"])


# ── Registry ─────────────────────────────────────────────────────────────────
# Keys must match the names passed to  --dset_pretrain / --dset_finetune  in
# the PatchTST scripts and to  run(dataset=...)  in Train_and_downstream.py.
#
# patchtst_cls  "ETT_minute" | "ETT_hour" | "Custom"
# columns       list of data-column names (everything except the timestamp).
#               Set to None to auto-detect from the CSV header at runtime.

DATASETS: dict = {
    "ettm1": {
        "csv_filename":    "ETTm1.csv",
        "patchtst_cls":    "ETT_minute",
        "timestamp_col":   "date",
        "columns":         ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
    },
    "etth1": {
        "csv_filename":    "ETTh1.csv",
        "patchtst_cls":    "ETT_hour",
        "timestamp_col":   "date",
        "columns":         ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
    },
    "etth2": {
        "csv_filename":    "ETTh2.csv",
        "patchtst_cls":    "ETT_hour",
        "timestamp_col":   "date",
        "columns":         ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
    },
    "ettm2": {
        "csv_filename":    "ETTm2.csv",
        "patchtst_cls":    "ETT_minute",
        "timestamp_col":   "date",
        "columns":         ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
    },
    "weather": {
        "csv_filename":    "weather.csv",
        "patchtst_cls":    "Custom",
        "timestamp_col":   "date",
        "columns":         None,  # auto-detected from CSV header
    },
    "electricity": {
        "csv_filename":  "electricity.csv",
        "patchtst_cls":  "Custom",
        "timestamp_col": "date",
        "columns": None,  # auto-detected from CSV header
    },
    "traffic": {
        "csv_filename":  "traffic.csv",
        "patchtst_cls":  "Custom",
        "timestamp_col": "date",
        "columns": None,  # auto-detected from CSV header
    },

    # ── Long-format datasets, converted to wide via tools/convert_long_to_wide.py ──
    "exchange": {
        "csv_filename":  "Exchange.csv",
        "patchtst_cls":  "Custom",
        "timestamp_col": "date",
        "columns": None,  # auto-detected from CSV header
    },
    "wind": {
        "csv_filename":  "Wind.csv",
        "patchtst_cls":  "Custom",
        "timestamp_col": "date",
        "columns": None,
    },
    "solar": {
        "csv_filename":  "Solar.csv",
        "patchtst_cls":  "Custom",
        "timestamp_col": "date",
        "columns": None,
    },
    "metr_la": {
        "csv_filename":  "METR-LA.csv",
        "patchtst_cls":  "Custom",
        "timestamp_col": "date",
        "columns": None,
    },
    "aqwan": {
        "csv_filename":  "AQWan.csv",
        "patchtst_cls":  "Custom",
        "timestamp_col": "date",
        "columns": None,
    },
    "aqshunyi": {
        "csv_filename":  "AQShunyi.csv",
        "patchtst_cls":  "Custom",
        "timestamp_col": "date",
        "columns": None,
    },
}


def get_dataset_info(name: str) -> dict:
    """Return a fully-resolved info dict for *name*.

    Extra keys added at call time:
      csv_path      – absolute path to the CSV file
      data_dir      – directory containing the CSV (with trailing separator)
      c_in          – number of data columns
    """
    if name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available: {list(DATASETS)}"
        )
    info = dict(DATASETS[name])
    info["name"]     = name
    info["csv_path"] = str(_DATA_DIR / info["csv_filename"])
    info["data_dir"] = str(_DATA_DIR) + os.sep

    # Auto-detect columns if not explicitly listed
    if info["columns"] is None:
        import pandas as pd
        df = pd.read_csv(info["csv_path"], nrows=0)
        info["columns"] = [c for c in df.columns if c != info["timestamp_col"]]

    info["c_in"]        = len(info["columns"])
    return info
