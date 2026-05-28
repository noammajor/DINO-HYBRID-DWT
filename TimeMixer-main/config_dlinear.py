config = {
    "model":          "DLinear",
    "seq_len":        336,
    "label_len":      0,    # encoder-only, no decoder input needed
    "e_layers":       1,    # unused but kept for consistency
    "d_layers":       1,
    "d_model":        1,    # unused
    "d_ff":           1,    # unused
    "n_heads":        1,    # unused
    "factor":         1,
    "moving_avg":     25,
    "dropout":        0.1,
    "activation":     "gelu",
    "embed":          "timeF",
    "freq":           "h",
    "features":       "M",
    "patch_len":      16,

    "learning_rate":  0.0001,
    "train_epochs":   20,
    "epochs_forecasting": 20,
    "batch_size":     32,
    "batch_size_forecast": 32,
    "patience":       3,
    "num_workers":    4,
    "lradj":          "TST",
    "pct_start":      0.2,
    "loss":           "MSE",
    "drop_last":      True,
}
