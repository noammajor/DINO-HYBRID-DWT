config = {
    # ── Architecture ───────────────────────────────────────────────────────────
    "model":                        "TimeMixer",
    "seq_len":                      336,
    "label_len":                    0,
    "pred_len":                     96,
    "e_layers":                     2,
    "d_model":                      16,
    "d_ff":                         32,
    "n_heads":                      4,
    "dropout":                      0.1,
    "moving_avg":                   25,
    "down_sampling_layers":         3,
    "down_sampling_window":         2,
    "down_sampling_method":         "avg",
    "channel_independence":         1,
    "decomp_method":                "moving_avg",
    "use_norm":                     1,
    "use_future_temporal_feature":  0,
    "embed":                        "timeF",
    "freq":                         "h",
    "factor":                       1,
    "top_k":                        5,
    "num_kernels":                  6,
    "features":                     "M",

    # patch_len only used as stride for PatchTSTForcastingAdapter
    "patch_len":                    16,

    # ── Forecasting training ───────────────────────────────────────────────────
    "learning_rate":                0.001,
    "lr_forecasting":               5e-4,
    "train_epochs":                 10,
    "epochs_forecasting":           10,
    "batch_size":                   16,
    "batch_size_forecast":          128,
    "patience":                     5,
    "num_workers":                  4,
    "lradj":                        "TST",
    "pct_start":                    0.2,
    "loss":                         "MSE",
    "drop_last":                    True,

    # ── Classification training ────────────────────────────────────────────────
    "batch_size_classification":    64,
    "lr_classification":            1e-3,
    "epochs_classification":        30,
    "patience_classification":      5,
}
