# Hyperparameters
PHI - Phi*(dino loss) + (1-phi)MLM/Ibot
## DINO (PatchTST backbone)

### Architecture
| Param | Value |
|-------|-------|
| encoder_layers | 4 |
| embed_dim (d_model) | 128 |
| n_heads | 16 |
| patch_len | 16 |
| num_patches | 21 → seq_len 336 |
| out_dim | 1024 |
| dropout | 0.1 |

### Training
| Param | Value |
|-------|-------|
| pretrain lr | 0.001 |
| pretrain epochs | 75 |
| forecasting | linear probe (backbone frozen) |
| datasets | ETTh1, ETTh2, ETTm1, ETTm2 |
| pretrain | in-domain (same dataset as forecast) |

### Variants
| Variant | aug_global | aug_local | mlm_mode | mlm_phi | mask_ratio |
|---------|-----------|-----------|----------|---------|------------|
| DINO | default | default | — | — | — |
| DINO SWT | swt_soft | swt_hard | — | — | — |
| DINO MODWT | modwt_soft | modwt_hard | — | — | — |
| DINO DWT | dwt_soft | dwt_hard | — | — | — |
| DINO MAE | default | default | mae | 0.6 | 0.4 |
| DINO MAE SWT | swt_soft | swt_hard | mae | 0.6 | 0.4 |
| DINO MAE MODWT | modwt_soft | modwt_hard | mae | 0.6 | 0.4 |
| DINO iBOT | default | default | ibot | 0.6 | 0.4 |
| DINO iBOT SWT | swt_soft | swt_hard | ibot | 0.6 | 0.4 |
| DINO iBOT MODWT | modwt_soft | modwt_hard | ibot | 0.6 | 0.4 |

---

## DINO (TSMixer backbone)

### Architecture
| Param | Value |
|-------|-------|
| encoder_layers | 4 |
| backbone_type | tsmixer |
| seq_len | 336 |
| down_sampling_layers | 3 |
| down_sampling_window | 2 |

### Training
| Param | Value |
|-------|-------|
| pretrain lr | 0.001 |
| pretrain epochs | 75 |
| forecasting | linear probe (backbone frozen) |
| datasets | ETTh1, ETTh2, ETTm1, ETTm2 |

### Variants
| Variant | aug_global | aug_local | mlm_mode | mlm_phi | mask_ratio |
|---------|-----------|-----------|----------|---------|------------|
| DINO TSMixer | default | default | — | — | — |
| DINO TSMixer SWT | swt_soft | swt_hard | — | — | — |
| DINO TSMixer MODWT | modwt_soft | modwt_hard | — | — | — |
| DINO TSMixer MAE | default | default | mae | 0.6 | 0.4 |
| DINO TSMixer iBOT DWT | dwt_soft | dwt_hard | ibot | 0.6 | 0.4 |

---

## DINO Random Init (baseline)

| Param | Value |
|-------|-------|
| encoder_layers | 4 |
| skip_pretrain | true |
| checkpoints | 0 (random weights, no loading) |
| forecasting | linear probe |
| datasets | ETTh1, ETTh2, ETTm1, ETTm2 |

---

## PatchTST (MAE pretrain)

| Param | Value |
|-------|-------|
| n_layers | 3 |
| n_heads | 16 |
| d_model | 128 |
| d_ff | 512 |
| patch_len | 16 |
| seq_len | 336 |
| mask_ratio | 0.4 |
| pretrain epochs | 75 |
| pretrain lr | 5e-5 |
| finetune lr | 4e-4 |
| datasets | ETTh1, ETTh2, ETTm1, ETTm2 |

---

## TimeMixer (supervised)

| Param | Value |
|-------|-------|
| e_layers | 2 |
| d_model | 128 |
| d_ff | 256 |
| n_heads | 4 |
| seq_len | 336 |
| down_sampling_layers | 3 |
| down_sampling_window | 2 |
| down_sampling_method | avg |
| channel_independence | 1 |
| decomp_method | moving_avg |
| moving_avg | 25 |
| dropout | 0.1 |
| lr | 0.01 |
| epochs | 20 |
| batch_size | 128 |
| patience | 10 |
| datasets | ETTh1, ETTh2, ETTm1, ETTm2 |

---

## Autoformer (supervised)

| Param | Value |
|-------|-------|
| e_layers | 2 |
| d_layers | 1 |
| d_model | 512 |
| d_ff | 2048 |
| n_heads | 8 |
| seq_len | 336 |
| label_len | 48 |
| moving_avg | 25 |
| factor | 3 |
| dropout | 0.05 |
| lr | 1e-4 |
| epochs | 10 |
| batch_size | 32 |
| datasets | ETTh1, ETTh2, ETTm1, ETTm2 |

---

## FEDformer (supervised)

| Param | Value |
|-------|-------|
| e_layers | 2 |
| d_layers | 1 |
| d_model | 512 |
| d_ff | 2048 |
| n_heads | 8 |
| seq_len | 336 |
| label_len | 48 |
| modes | 32 |
| mode_select | random |
| moving_avg | 25 |
| factor | 3 |
| dropout | 0.05 |
| lr | 1e-4 |
| epochs | 20 |
| batch_size | 32 |
| datasets | ETTh1, ETTh2, ETTm1, ETTm2 |

---

## DLinear (supervised)

| Param | Value |
|-------|-------|
| seq_len | 336 |
| moving_avg | 25 |
| dropout | 0.1 |
| lr | 1e-4 |
| epochs | 20 |
| batch_size | 32 |
| datasets | ETTh1, ETTh2, ETTm1, ETTm2 |

---

## Context Length Sweep (run_ctx_sweep.py)

### Architecture
| Param | Value |
|-------|-------|
| encoder_layers | 2 |
| patch_len | 16 |
| ctx_lens | 96, 192, 720 |

### Training
| Param | Value |
|-------|-------|
| pretrain epochs | 30 |
| lr | 0.001 |
| forecasting | linear probe (backbone frozen) |
| pretrain | in-domain (same ctx_len for pretrain and forecast) |
| datasets | ETTh1, ETTh2, ETTm1, ETTm2 |

### Variants
| Variant | aug_global | aug_local | mlm_mode | mlm_phi | mask_ratio |
|---------|-----------|-----------|----------|---------|------------|
| DINO DWT | dwt_soft | dwt_hard | — | — | — |
| DINO SWT | swt_soft | swt_hard | — | — | — |
| DINO MODWT | modwt_soft | modwt_hard | — | — | — |
| DINO iBOT DWT | dwt_soft | dwt_hard | ibot | 0.6 | 0.4 |
| DINO iBOT SWT | swt_soft | swt_hard | ibot | 0.6 | 0.4 |
| DINO MLM DWT | dwt_soft | dwt_hard | mae | 0.6 | 0.4 |
| DINO MLM SWT | swt_soft | swt_hard | mae | 0.6 | 0.4 |
