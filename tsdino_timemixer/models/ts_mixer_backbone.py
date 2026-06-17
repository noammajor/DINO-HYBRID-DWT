import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from torch.nn.init import trunc_normal_

# Make TimeMixer-main importable from wherever this module is loaded.
# We add both the root (so 'layers.*' resolves) and models/ directly (so we
# can import 'TimeMixer' without going through the 'models' package, which
# would collide with TSDiNO's own 'models' already cached in sys.modules).
_TIMEMIXER_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'TimeMixer-main')
)
_TIMEMIXER_MODELS = os.path.join(_TIMEMIXER_ROOT, 'models')
for _p in (_TIMEMIXER_ROOT, _TIMEMIXER_MODELS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from layers.Autoformer_EncDec import series_decomp        # noqa: E402
from layers.Embed import DataEmbedding_wo_pos              # noqa: E402
from layers.StandardNorm import Normalize                  # noqa: E402
from TimeMixer import PastDecomposableMixing               # noqa: E402


class PatchReconDecoder(nn.Module):
    """Maps token embeddings back to raw values for the MAE / reconstruction head.

    Used by the DINO reconstruction and MAE auxiliary losses. On the TimeMixer
    backbone each token is a single timestep, so out_dim is 1.

    Input:  [N, num_tokens, d_model]
    Output: [N, num_tokens, out_dim]
    (Relocated here from the removed models/patchTST.py — it has no PatchTST deps.)
    """
    def __init__(self, d_model: int, out_dim: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, x):
        return self.decoder(x)


class TSMixerForDINO(nn.Module):
    """TimeMixer encoder wrapped for DINO/iBOT pretraining.

    Encoder pipeline matches TimeMixer.Model.forecast() 1-to-1 (minus task heads):
      _multi_scale_process  →  pool all scales, then per-scale RevIN + CI reshape
      pre_enc               →  season decomp before embedding (CI=1: passthrough)
      enc_embedding         →  DataEmbedding_wo_pos per scale
      pdm_blocks            →  PastDecomposableMixing × e_layers

    DINO-specific additions on top:
      mask_token            —  learnable [1, 1, d_model] for iBOT masking
      forward / forward_ibot / forward_ibot_multiscale / forward_recon

    WARNING: PDM Linear layers are fixed to seq_len. All DINO crop views must share
    the same temporal length — set crop_ratio=1.0 in global_crops / local_crops.
    """

    def __init__(
        self,
        c_in: int,
        seq_len: int,
        d_model: int,
        e_layers: int,
        d_ff: int,
        dropout: float,
        patch_len: int = 16,
        down_sampling_layers: int = 3,
        down_sampling_window: int = 2,
        down_sampling_method: str = 'avg',
        decomp_method: str = 'moving_avg',
        moving_avg: int = 25,
        top_k: int = 5,
        use_norm: int = 1,              # 1=RevIN on, 0=off — matches TimeMixer's use_norm
        channel_independence: int = 1,  # 1=CI (default), 0=joint
    ):
        super().__init__()
        self.d_model = d_model
        self.c_in = c_in
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_window = down_sampling_window
        self.down_sampling_method = down_sampling_method
        self.e_layers = e_layers
        self.channel_independence = channel_independence

        configs = SimpleNamespace(
            seq_len=seq_len,
            pred_len=0,          # required by PastDecomposableMixing; unused in encoder-only mode
            d_model=d_model, d_ff=d_ff, dropout=dropout,
            down_sampling_layers=down_sampling_layers,
            down_sampling_window=down_sampling_window,
            channel_independence=channel_independence,
            decomp_method=decomp_method,
            moving_avg=moving_avg,
            top_k=top_k,
        )

        # Season decomposition applied in pre_enc before embedding — matches TimeMixer.preprocess
        self.preprocess = series_decomp(moving_avg)

        # CI=1: univariate (1 feature per token); CI=0: multivariate (c_in features)
        enc_in = 1 if channel_independence == 1 else c_in
        self.enc_embedding = DataEmbedding_wo_pos(enc_in, d_model, 'timeF', 'h', dropout)

        # PastDecomposableMixing blocks (season + trend mixing across scales)
        self.pdm_blocks = nn.ModuleList(
            [PastDecomposableMixing(configs) for _ in range(e_layers)]
        )

        # Per-scale RevIN — Normalize IS RevIN: per-instance mean/std with learnable affine.
        # non_norm=True disables it entirely when use_norm=0, matching TimeMixer's flag.
        self.normalize_layers = nn.ModuleList([
            Normalize(c_in, affine=True, non_norm=True if use_norm == 0 else False)
            for _ in range(down_sampling_layers + 1)
        ])

        # Learnable mask token for iBOT — applied in embedding space [1, 1, d_model]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        trunc_normal_(self.mask_token, std=0.02)

        # Cross-attention global pooling: a single learnable CLS query attends to all T
        # PDM-output tokens. Runs AFTER PDM on real timestep tokens — avoids the
        # moving-average boundary-padding problem that made CLS-in-PDM input-independent.
        self.cls_query = nn.Parameter(torch.zeros(1, 1, d_model))
        trunc_normal_(self.cls_query, std=0.02)
        self.global_attn = nn.MultiheadAttention(d_model, num_heads=4, dropout=0.0, batch_first=True)

        # Timestep-level tokens: each of the T positions in enc_out_list[0] is a token.
        self.num_tokens = seq_len

    # ── internal helpers ───────────────────────────────────────────────────────

    def _multi_scale_process(self, x: torch.Tensor, normalize: bool = True):
        """Pool all scales then apply per-scale RevIN + CI reshape.

        normalize=False skips RevIN (used by DINO forward to mirror PatchTST behaviour:
        PatchTST DINO mode does not apply RevIN so raw amplitude/offset are preserved).
        """
        # Step 1: pool all scales (raw) — mirrors __multi_scale_process_inputs
        raw_list = [x]
        x_d = x.permute(0, 2, 1)   # [B, C, T] for 1-D pooling
        for _ in range(self.down_sampling_layers):
            if self.down_sampling_method == 'max':
                x_d = F.max_pool1d(x_d, self.down_sampling_window)
            else:   # 'avg' (default)
                x_d = F.avg_pool1d(x_d, self.down_sampling_window)
            raw_list.append(x_d.permute(0, 2, 1))

        # Step 2: per-scale RevIN then CI reshape — mirrors normalize loop in forecast()
        x_list = []
        for i, xs in enumerate(raw_list):
            B, T, N = xs.size()
            if normalize:
                xs = self.normalize_layers[i](xs, 'norm')
            if self.channel_independence == 1:
                xs = xs.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(xs)
        return x_list

    def pre_enc(self, x_list):
        """Season decomposition before embedding — exact copy of TimeMixer.Model.pre_enc.

        CI=1: passthrough, returns (x_list, None). Decomp is handled inside PDM blocks.
        CI=0: decomposes each scale into (season, trend) via series_decomp; season is
              passed to enc_embedding, trend is used by PDM internally.
        """
        if self.channel_independence == 1:
            return (x_list, None)
        out1_list, out2_list = [], []
        for x in x_list:
            x_1, x_2 = self.preprocess(x)
            out1_list.append(x_1)
            out2_list.append(x_2)
        return (out1_list, out2_list)

    def _embed_and_mix(self, x_list, mask_patches=None):
        """pre_enc → embed each scale → PDM blocks. Matches TimeMixer.Model.forecast().

        mask_patches: bool tensor at finest-scale — True positions replaced with mask_token.
          CI=1: [B*C, T]   CI=0: [B, T]
        """
        x_list = self.pre_enc(x_list)

        enc_out_list = []
        for scale_idx, x in enumerate(x_list[0]):
            T = x.shape[1]
            emb = self.enc_embedding(x, None)       # [B*C, T, d_model] (CI=1)

            if mask_patches is not None and scale_idx == 0:
                mt = self.mask_token.expand(emb.shape[0], T, -1)
                emb = torch.where(mask_patches.unsqueeze(-1), mt, emb)

            enc_out_list.append(emb)

        for pdm in self.pdm_blocks:
            enc_out_list = pdm(enc_out_list)

        return enc_out_list

    # ── public forward methods ─────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Global DINO embedding via cross-attention pooling over PDM output tokens.

        Cross-attention runs AFTER PDM on T real timestep tokens. Unlike CLS-in-PDM,
        this avoids the moving-average boundary-padding issue that caused mode collapse
        (52% of CLS position came from the constant learnable parameter, not the input).

        x: [B, T, C]
        returns: [B, C, d_model]  — per-variable global representation.
                  TSMultiCropWrapper reshapes to [B*C, d_model] before DINOHead.
        """
        B = x.shape[0]
        enc    = self._embed_and_mix(self._multi_scale_process(x, normalize=False))
        finest = enc[0]                                                    # [B*C, T, d_model]
        # Cross-attention: learned CLS query attends to all T encoder tokens
        q = self.cls_query.expand(B * self.c_in, 1, self.d_model)         # [B*C, 1, d_model]
        global_rep, _ = self.global_attn(q, finest, finest)               # [B*C, 1, d_model]
        global_rep = global_rep[:, 0, :]                                   # [B*C, d_model]
        return global_rep.reshape(B, self.c_in, self.d_model)             # [B, C, d_model]

    def forward_ibot(self, z: torch.Tensor, mask=None) -> torch.Tensor:
        """iBOT timestep encoding — TSMixer-native: each timestep is a token.

        z:    [B, T, C]
        mask: [B, T] bool — True=masked (student). None=full pass (teacher).
        returns: [B, T, C, d_model]
        """
        B, T, C = z.shape

        mask_patches = None
        if mask is not None:
            # Expand timestep mask to [B*C, T]
            mask_patches = mask.unsqueeze(1).expand(-1, C, -1).reshape(B * C, T)

        enc = self._embed_and_mix(self._multi_scale_process(z, normalize=False), mask_patches=mask_patches)

        # enc[0]: [B*C, T, d_model] — each timestep is a native TSMixer token
        finest = enc[0]                                                           # [B*C, T, d_model]
        return finest.reshape(B, self.c_in, T, self.d_model).permute(0, 2, 1, 3) # [B, T, C, d_model]

    def forward_ibot_multiscale(self, z: torch.Tensor, mask=None):
        """Multi-scale iBOT encoding — concatenates tokens from every TSMixer scale.

        enc_out_list[k] has T_k = T / window^k timesteps. Masking is applied only at
        the finest scale (k=0) so coarser scales always carry full-resolution context,
        which is the key TSMixer design principle.

        z:    [B, T, C]
        mask: [B, T] bool — True=masked at finest scale. None=full pass (teacher).
        returns:
            tokens    [B, N_total, C, d_model]  N_total = T + T/W + T/W^2 + ...
            full_mask [B, N_total] bool          True only at masked fine-scale positions
        """
        B, T, C = z.shape

        mask_patches = None
        if mask is not None:
            mask_patches = mask.unsqueeze(1).expand(-1, C, -1).reshape(B * C, T)

        enc = self._embed_and_mix(self._multi_scale_process(z, normalize=False), mask_patches=mask_patches)

        scale_tokens = []
        scale_masks  = []
        for k, enc_k in enumerate(enc):                              # [B*C, T_k, d_model]
            T_k = enc_k.shape[1]
            tok = enc_k.reshape(B, self.c_in, T_k, self.d_model).permute(0, 2, 1, 3)  # [B, T_k, C, d_model]
            scale_tokens.append(tok)
            if mask is not None and k == 0:
                scale_masks.append(mask)                             # [B, T]
            else:
                scale_masks.append(torch.zeros(B, T_k, dtype=torch.bool, device=z.device))

        tokens    = torch.cat(scale_tokens, dim=1)                   # [B, N_total, C, d_model]
        full_mask = torch.cat(scale_masks,  dim=1) if mask is not None else None
        return tokens, full_mask

    def forward_recon(self, z: torch.Tensor) -> torch.Tensor:
        """Full (unmasked) timestep encoding for reconstruction loss.

        z: [B, T, C]
        returns: [B, T, C, d_model]
        """
        return self.forward_ibot(z, mask=None)


class TSMixerForecastModel(nn.Module):
    """Pretrained TSMixerForDINO backbone + linear forecasting head.

    Forward:
      1. _multi_scale_process(normalize=True)  — per-instance RevIN
      2. _embed_and_mix                        — PDM blocks, finest scale [B*C, T, d_model]
      3. Flatten + Linear head                 → [B*C, pred_len]
      4. Reshape                               → [B, pred_len, C]
      5. RevIN denorm
    """

    def __init__(self, backbone: TSMixerForDINO, pred_len: int, use_revin: bool = True):
        super().__init__()
        self.backbone  = backbone
        self.pred_len  = pred_len
        self.use_revin = use_revin
        self.head = nn.Linear(backbone.seq_len * backbone.d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C]  →  [B, pred_len, C]"""
        B, T, C = x.shape
        x_list = self.backbone._multi_scale_process(x, normalize=self.use_revin)
        enc    = self.backbone._embed_and_mix(x_list)
        finest = enc[0]                                      # [B*C, T, d_model]
        flat   = finest.reshape(B * C, -1)                   # [B*C, T*d_model]
        pred   = self.head(flat)                             # [B*C, pred_len]
        pred   = pred.reshape(B, C, -1).permute(0, 2, 1)    # [B, pred_len, C]
        if self.use_revin:
            pred = self.backbone.normalize_layers[0](pred, 'denorm')
        return pred
