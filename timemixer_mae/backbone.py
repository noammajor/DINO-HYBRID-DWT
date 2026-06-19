"""TimeMixer encoder for MAE pretraining (standalone — no tsdino dependency).

Replicates the TimeMixer.Model.forecast() encoder path (multi-scale pooling →
per-scale RevIN → season-decomp pre_enc → embedding → PastDecomposableMixing),
exposing per-timestep tokens plus a learnable mask_token for masked modelling.
Built directly on the layers in ../TimeMixer-main.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from torch.nn.init import trunc_normal_

_TIMEMIXER_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'TimeMixer-main'))
_TIMEMIXER_MODELS = os.path.join(_TIMEMIXER_ROOT, 'models')
for _p in (_TIMEMIXER_ROOT, _TIMEMIXER_MODELS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from layers.Autoformer_EncDec import series_decomp        # noqa: E402
from layers.Embed import DataEmbedding_wo_pos              # noqa: E402
from layers.StandardNorm import Normalize                  # noqa: E402
from TimeMixer import PastDecomposableMixing               # noqa: E402


class TimeMixerEncoder(nn.Module):
    """TimeMixer encoder producing one token per timestep, with iBOT-style masking.

    Each crop view must share the temporal length seq_len (PDM Linear layers are
    sized for it). channel_independence=1 is the TimeMixer default (univariate
    tokens). Masking replaces masked timesteps with a learnable mask_token in
    embedding space at the finest scale only.
    """

    def __init__(self, c_in, seq_len, d_model, e_layers, d_ff, dropout,
                 down_sampling_layers=3, down_sampling_window=2,
                 down_sampling_method='avg', decomp_method='moving_avg',
                 moving_avg=25, top_k=5, use_norm=1, channel_independence=1):
        super().__init__()
        self.c_in = c_in
        self.seq_len = seq_len
        self.d_model = d_model
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_window = down_sampling_window
        self.down_sampling_method = down_sampling_method
        self.channel_independence = channel_independence

        configs = SimpleNamespace(
            seq_len=seq_len, pred_len=0, d_model=d_model, d_ff=d_ff, dropout=dropout,
            down_sampling_layers=down_sampling_layers,
            down_sampling_window=down_sampling_window,
            channel_independence=channel_independence,
            decomp_method=decomp_method, moving_avg=moving_avg, top_k=top_k,
        )
        self.preprocess = series_decomp(moving_avg)
        enc_in = 1 if channel_independence == 1 else c_in
        self.enc_embedding = DataEmbedding_wo_pos(enc_in, d_model, 'timeF', 'h', dropout)
        self.pdm_blocks = nn.ModuleList(
            [PastDecomposableMixing(configs) for _ in range(e_layers)])
        self.normalize_layers = nn.ModuleList([
            Normalize(c_in, affine=True, non_norm=(use_norm == 0))
            for _ in range(down_sampling_layers + 1)])

        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        trunc_normal_(self.mask_token, std=0.02)

    def _multi_scale_process(self, x, normalize):
        raw = [x]
        x_d = x.permute(0, 2, 1)
        for _ in range(self.down_sampling_layers):
            pool = F.max_pool1d if self.down_sampling_method == 'max' else F.avg_pool1d
            x_d = pool(x_d, self.down_sampling_window)
            raw.append(x_d.permute(0, 2, 1))
        out = []
        for i, xs in enumerate(raw):
            B, T, N = xs.size()
            if normalize:
                xs = self.normalize_layers[i](xs, 'norm')
            if self.channel_independence == 1:
                xs = xs.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            out.append(xs)
        return out

    def _pre_enc(self, x_list):
        if self.channel_independence == 1:
            return x_list
        return [self.preprocess(x)[0] for x in x_list]

    def _embed_and_mix(self, x_list, mask_patches=None):
        x_list = self._pre_enc(x_list)
        enc = []
        for scale_idx, x in enumerate(x_list):
            T = x.shape[1]
            emb = self.enc_embedding(x, None)
            if mask_patches is not None and scale_idx == 0:
                mt = self.mask_token.expand(emb.shape[0], T, -1)
                emb = torch.where(mask_patches.unsqueeze(-1), mt, emb)
            enc.append(emb)
        for pdm in self.pdm_blocks:
            enc = pdm(enc)
        return enc

    def forward(self, x, mask=None):
        """x: [B, T, C]; mask: [B, T] bool (True = masked) or None.

        returns per-timestep tokens [B*C, T, d_model] (finest scale).
        """
        B, T, C = x.shape
        mask_patches = None
        if mask is not None:
            mask_patches = mask.unsqueeze(1).expand(-1, C, -1).reshape(B * C, T)
        enc = self._embed_and_mix(
            self._multi_scale_process(x, normalize=False), mask_patches=mask_patches)
        return enc[0]   # [B*C, T, d_model]

    def encode(self, x, normalize):
        """Unmasked encoding for downstream forecasting (RevIN when normalize=True).

        x: [B, T, C] → finest-scale tokens [B*C, T, d_model].
        """
        return self._embed_and_mix(self._multi_scale_process(x, normalize))[0]
