import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from torch.nn.init import trunc_normal_

# Make TimeMixer-main importable from wherever this module is loaded
_TIMEMIXER_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'TimeMixer-main')
)
if _TIMEMIXER_ROOT not in sys.path:
    sys.path.insert(0, _TIMEMIXER_ROOT)

from layers.Autoformer_EncDec import series_decomp        # noqa: E402
from layers.Embed import DataEmbedding_wo_pos              # noqa: E402
from layers.StandardNorm import Normalize                  # noqa: E402
from models.TimeMixer import PastDecomposableMixing        # noqa: E402


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
        """iBOT patch encoding.

        z:    [B, T, C]
        mask: [B, NP] bool — True=masked (student). None=full pass (teacher).
        returns: [B, NP, C, d_model]
        """
        B, T, C = z.shape
        NP = T // self.patch_len

        mask_patches = None
        if mask is not None:
            # Expand patch mask to timestep resolution: [B*C, T]
            mask_exp = mask.unsqueeze(1).expand(-1, C, -1).reshape(B * C, NP)
            mask_patches = mask_exp.repeat_interleave(self.patch_len, dim=1)   # [B*C, T]

        enc = self._embed_and_mix(self._multi_scale_process(z), mask_patches=mask_patches)

        # enc[0]: [B*C, T, d_model] — segment into NP patches and mean-pool each
        finest = enc[0]                                                  # [B*C, T, d_model]
        finest = finest.reshape(B * C, NP, self.patch_len, self.d_model).mean(dim=2)  # [B*C, NP, d_model]
        return finest.reshape(B, C, NP, self.d_model).permute(0, 2, 1, 3)  # [B, NP, C, d_model]

    def forward_recon(self, z: torch.Tensor) -> torch.Tensor:
        """Full (unmasked) patch encoding for reconstruction loss.

        z: [B, T, C]
        returns: [B, NP, C, d_model]
        """
        return self.forward_ibot(z, mask=None)
