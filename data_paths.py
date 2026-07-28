"""
Centralised data paths shared across all models.

Set each "ADD HERE" placeholder below to the absolute path on YOUR machine
(one-time per machine) — see the "Data paths" section of the README for what
each key should point to. Per-model configs may override individual keys.
"""

DATA_PATHS = {
    # ── Pretraining ──────────────────────────────────────────────────────────
    "monash_data_dir":         "ADD HERE",   # Monash pretraining corpus
    "monash_min_len":          512,
    "synthetic_data_dir":      "ADD HERE",   # synthetic .arrow files
    "synthetic_mix_data_dir":  "ADD HERE",   # smaller curated synthetic mix
    # ── Forecasting CSVs (consumed by dataset_registry) ──────────────────────
    "forecasting_data_dir":    "ADD HERE",   # forecasting CSV directory
    # ── Downstream task datasets ─────────────────────────────────────────────
    "classification_data_dir": "ADD HERE",   # UEA classification datasets
    "anomaly_data_dir":        "ADD HERE",   # anomaly datasets (SMD/MSL/SMAP/PSM/SWaT)
}
