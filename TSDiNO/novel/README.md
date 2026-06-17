# `TSDiNO/novel` — Disentangled Trend-Seasonality Dual-Stream backbone

Experimental, self-contained package. **Not yet wired into the training loop** —
it's a clean building block you can inspect and iterate on first.

## Idea

A time series = slow **trend/macro** + fast **season/micro**. A single shared
embedding lets the high-energy trend drown out the subtle periodic structure. We
split the signal *inside the network* and give each component a specialised
encoder, then use a cross-stream DINO objective to relate them.

```
x [B,T,C]
  │  LearnedDecomp        (learned conv pre-encoder, NOT the DWT augmentation)
  ├── trend  ─► TrendStream   : coarse patches → MLP-Mixer     → z_macro [B,C,d]
  └── season ─► SeasonStream  : fine patches   → Transformer/CLS → z_micro [B,C,d]
                                              │
                                   two DINO heads → DualStreamDINOLoss
```

## Why this is architecture, not augmentation

The repo's DWT **augmentation** is a fixed pywt transform on the *input* that
defines *what invariances* DINO learns. `LearnedDecomp` is a **trained layer**
inside the backbone that defines *how the encoder represents* the signal — it
gives trend and season their own learned subspaces. They are complementary; the
decomposition sees the already-augmented input and must be robust to it.

## Components

| File | Class | Role |
|---|---|---|
| `decomposition.py` | `LearnedDecomp` | Fully-learned depthwise conv split `x → (trend, season)`, stride-1 / length-preserving. |
| `dual_stream_backbone.py` | `DualStreamBackbone` | Decompose → `TrendStream` (Mixer) + `SeasonStream` (attention) → `{"macro", "micro"}`. Channel-independent. |
| `dual_stream_loss.py` | `DualStreamDINOLoss` | Reuses a faithful copy of `main.py`'s `DINOLoss` (`_DINOLoss`) composed per mode. |
| `smoke_test.py` | — | CPU shape/loss/backward check for all three modes. |

## Loss modes (`mode=`)

- **`concat`** — baseline: glue streams, one head, vanilla DINO. Control.
- **`dual`** — DINO within each stream + a consistency term. Stable, disentangles.
- **`cross`** — the novel objective: student **season** ↔ teacher **trend** and
  student **trend** ↔ teacher **season** across crops. Forces "how do these local
  cycles sit on the global trend."

## Phase-2 wiring plan (into `main.py`)

The backbone already follows the repo's channel-independent `[B,C,d]` convention,
so integration mirrors the existing `tsmixer`/`patchtst` branches:

1. `config.py`: add `backbone_type="dualstream"` plus
   `dualstream_*` knobs (patch sizes, layers, `loss_mode`, `consistency_weight`).
2. `main.py` model build: when `backbone_type=="dualstream"`, instantiate
   `DualStreamBackbone` and **two** `DINOHead`s (macro, micro) — or one `2*d` head
   for `concat`.
3. `TSMultiCropWrapper`: route `{"macro","micro"}` through the two heads and
   return the per-mode dict the loss expects.
4. Swap `DINOLoss` → `DualStreamDINOLoss(mode=cfg['dualstream_loss_mode'], ...)`.
5. iBOT/MAE aux heads (optional): apply to the season stream's patch tokens.

Run the smoke test any time:

```bash
python TSDiNO/novel/smoke_test.py
```
