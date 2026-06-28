"""Convert long/melted forecasting CSVs to the wide format the loaders expect.

Input  (long) : date,data,cols[,name]   — one row per timestamp x channel
Output (wide) : date,<ch0>,<ch1>,...    — one row per timestamp, one col per channel

The wide layout matches the ETT/weather/electricity CSVs, so the converted file
works directly with DataPuller and PatchTST's Dataset_Custom.

Usage:
    python tools/convert_long_to_wide.py SRC_DIR OUT_DIR [file1.csv file2.csv ...]

If no filenames are given, every *.csv in SRC_DIR is converted.
"""
import sys
import os
import pandas as pd


def convert(src_path: str, out_path: str) -> None:
    print(f"[read]  {src_path}")
    df = pd.read_csv(src_path)

    required = {"date", "data", "cols"}
    if not required.issubset(df.columns):
        raise ValueError(f"{src_path}: expected columns {required}, got {list(df.columns)}")

    # 'name' (if present) identifies the series; keep it in the channel name only
    # when it actually varies, otherwise it is redundant and dropped.
    if "name" in df.columns and df["name"].nunique() > 1:
        df["cols"] = df["name"].astype(str) + "__" + df["cols"].astype(str)

    df["data"] = pd.to_numeric(df["data"], errors="coerce")
    # Parse to real datetimes so sorting is chronological, not lexicographic
    # (e.g. '1990/1/2' must come before '1990/1/10'). utc=True handles the
    # tz-aware files (Solar) uniformly.
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

    # long -> wide; aggfunc='mean' collapses any duplicate (date, channel) pairs.
    wide = df.pivot_table(index="date", columns="cols", values="data", aggfunc="mean")
    wide = wide.sort_index()                      # chronological order
    wide.columns = [str(c) for c in wide.columns] # ensure string headers
    wide = wide.reset_index()                     # 'date' back to a column
    # Write naive ISO timestamps (drop tz) — matches the ETT/weather CSV style.
    wide["date"] = wide["date"].dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S")

    n_missing = int(wide.drop(columns=["date"]).isna().sum().sum())
    if n_missing:
        print(f"[warn]  {n_missing} missing cells (gaps in some channels) — "
              f"forward/backward filling")
        wide = wide.ffill().bfill()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    wide.to_csv(out_path, index=False)
    print(f"[write] {out_path}  ->  {wide.shape[0]} rows x {wide.shape[1]-1} channels")


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    src_dir, out_dir = sys.argv[1], sys.argv[2]
    files = sys.argv[3:] or [f for f in os.listdir(src_dir) if f.lower().endswith(".csv")]
    for fname in files:
        convert(os.path.join(src_dir, fname), os.path.join(out_dir, fname))


if __name__ == "__main__":
    main()
