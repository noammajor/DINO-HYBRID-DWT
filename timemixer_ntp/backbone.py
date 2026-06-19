"""TimeMixer encoder for NTP pretraining (standalone — no tsdino dependency).

Replicates the TimeMixer.Model.forecast() encoder path (multi-scale pooling →
per-scale RevIN → season-decomp pre_enc → embedding → PastDecomposableMixing).
No masking — NTP predicts the next pred_len points from the encoded window.
Built directly on the layers in ../TimeMixer-main.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

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
    """TimeMixer encoder producing one token per timestep (finest scale).

    channel_independence=1 is the TimeMixer default. Per-scale RevIN (Normalize)
    is applied when normalize=True; the matching denorm is exposed via
    normalize_layers[0] for the forecasting head.
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

    def _embed_and_mix(self, x_list):
        x_list = self._pre_enc(x_list)
        enc = [self.enc_embedding(x, None) for x in x_list]
        for pdm in self.pdm_blocks:
            enc = pdm(enc)
        return enc

    def encode(self, x, normalize):
        """x: [B, T, C] → finest-scale tokens [B*C, T, d_model]."""
        return self._embed_and_mix(self._multi_scale_process(x, normalize))[0]
