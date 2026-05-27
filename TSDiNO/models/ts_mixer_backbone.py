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


class TSMixerForDINO(nn.Module):
    """TimeMixer encoder wrapped for DINO/iBOT pretraining.

    Public API mirrors PatchTST's DINO interface so TSMultiCropWrapper and
    train_one_epoch work without modification:

      forward(x)              [B, T, C] -> [B, C, d_model]      global per-variable embedding
      forward_ibot(z, mask)   [B, T, C] -> [B, NP, C, d_model]  per-patch tokens for iBOT
      forward_recon(z)        [B, T, C] -> [B, NP, C, d_model]  per-patch tokens for recon loss

    Always uses channel_independence=1: each variable is embedded independently as a
    univariate series, matching PatchTST's per-channel CLS-token structure.

    WARNING: if down_sampling_layers > 0 the internal Linear layers are built for the
    configured seq_len.  All DINO crop views must therefore share the same temporal
    length — set crop_ratio=1.0 in every global_crops / local_crops config entry.
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

        # Minimal configs object consumed by PastDecomposableMixing
        configs = SimpleNamespace(
            seq_len=seq_len,
            d_model=d_model, d_ff=d_ff, dropout=dropout,
            down_sampling_layers=down_sampling_layers,
            down_sampling_window=down_sampling_window,
            channel_independence=1,
            decomp_method=decomp_method,
            moving_avg=moving_avg,
            top_k=top_k,
        )

        # CI=1: each variable embedded as a univariate series
        self.enc_embedding = DataEmbedding_wo_pos(1, d_model, 'timeF', 'h', dropout)

        # PastDecomposableMixing blocks (season + trend mixing across scales)
        self.pdm_blocks = nn.ModuleList(
            [PastDecomposableMixing(configs) for _ in range(e_layers)]
        )

        # Per-scale instance norm (applied on [B, T, C] before CI reshape)
        self.normalize_layers = nn.ModuleList([
            Normalize(c_in, affine=True)
            for _ in range(down_sampling_layers + 1)
        ])

        # Learnable mask token for iBOT — applied in embedding space [1, 1, d_model]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        trunc_normal_(self.mask_token, std=0.02)

        # Timestep-level tokens: each of the T positions in enc_out_list[0] is a token.
        self.num_tokens = seq_len

    # ── internal helpers ───────────────────────────────────────────────────────

    def _multi_scale_process(self, x: torch.Tensor):
        """Normalise + downsample x [B, T, C] into a list of scale tensors.

        x_list[0] = finest (T = seq_len), x_list[k] = T / window^k.
        """
        x_list = [self.normalize_layers[0](x, 'norm')]
        x_d = x.permute(0, 2, 1)  # [B, C, T] for 1-D pooling
        for i in range(self.down_sampling_layers):
            if self.down_sampling_method == 'max':
                x_d = F.max_pool1d(x_d, self.down_sampling_window)
            else:  # 'avg' (default)
                x_d = F.avg_pool1d(x_d, self.down_sampling_window)
            x_list.append(
                self.normalize_layers[i + 1](x_d.permute(0, 2, 1), 'norm')
            )
        return x_list

    def _embed_and_mix(self, x_list, mask_patches=None):
        """Embed each scale (CI=1) then run all PDM blocks.

        mask_patches: [B*C, T_finest] bool — positions replaced with mask_token.
                      Applied only at finest scale (index 0), in embedding space,
                      so coarser scales still carry full-resolution signal as context.

        Returns enc_out_list: list of [B*C, T_scale, d_model].
        """
        enc_out_list = []
        for scale_idx, x in enumerate(x_list):
            B, T, _ = x.shape
            x_ci = x.permute(0, 2, 1).contiguous().reshape(B * self.c_in, T, 1)
            emb = self.enc_embedding(x_ci, None)    # [B*C, T, d_model]

            if mask_patches is not None and scale_idx == 0:
                mt = self.mask_token.expand(B * self.c_in, T, -1)
                emb = torch.where(mask_patches.unsqueeze(-1), mt, emb)

            enc_out_list.append(emb)

        for pdm in self.pdm_blocks:
            enc_out_list = pdm(enc_out_list)

        return enc_out_list

    # ── public forward methods ─────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Global DINO embedding.

        x: [B, T, C]
        returns: [B, C, d_model]  — mean-pooled over time per variable.
                  TSMultiCropWrapper reshapes this to [B*C, d_model] before DINOHead.
        """
        B = x.shape[0]
        enc = self._embed_and_mix(self._multi_scale_process(x))
        # enc[0]: [B*C, T, d_model]  → mean-pool time → [B*C, d_model] → [B, C, d_model]
        return enc[0].mean(dim=1).reshape(B, self.c_in, self.d_model)

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

        enc = self._embed_and_mix(self._multi_scale_process(z), mask_patches=mask_patches)

        # enc[0]: [B*C, T, d_model] — each timestep is a native TSMixer token
        finest = enc[0]                                                  # [B*C, T, d_model]
        return finest.reshape(B, C, T, self.d_model).permute(0, 2, 1, 3)  # [B, T, C, d_model]

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

        enc = self._embed_and_mix(self._multi_scale_process(z), mask_patches=mask_patches)

        scale_tokens = []
        scale_masks  = []
        for k, enc_k in enumerate(enc):                              # [B*C, T_k, d_model]
            T_k = enc_k.shape[1]
            tok = enc_k.reshape(B, C, T_k, self.d_model).permute(0, 2, 1, 3)  # [B, T_k, C, d_model]
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
