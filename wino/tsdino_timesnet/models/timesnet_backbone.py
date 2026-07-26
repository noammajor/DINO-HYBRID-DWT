"""
TimesNet encoder wrapped for DINO pretraining — VARIABLE input length,
CHANNEL-INDEPENDENT.

Two departures from the original TSLib TimesNet, both required here:

1. Variable length.  The original ``TimesBlock`` hardcodes ``seq_len + pred_len``
   for padding/reshape/truncation and the classification head flattens
   ``d_model * seq_len`` — so it only works at one fixed length. ``VarTimesBlock``
   drives everything by the runtime length ``T = x.size(1)`` (periods are already
   derived from the input via FFT), and the DINO representation is produced by
   cross-attention CLS pooling over the ``T`` tokens. → accepts any length ≤ 5000.
   Lets us pretrain at 1152 but classify shorter windows with the same weights.

2. Channel independence.  The original embeds ``c_in → d_model`` (mixes channels),
   which (a) forces ``c_in=1`` for univariate synthetic pretraining and (b) can't
   then classify a dataset with a different channel count. We instead fold the
   channel axis into the batch (``[B,T,C] → [B·C,T,1]``) and embed ``1 → d_model``,
   exactly like the TSMixer DINO backbone (``channel_independence=1``). The
   embedding is therefore always univariate, so a univariate-pretrained backbone
   transfers to any channel count downstream. Trade-off: no cross-variate mixing
   (same as PatchTST). Per-variate reps are returned as ``[B, C, d_model]``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from torch.nn.init import trunc_normal_

# Self-contained: DataEmbedding + Inception_Block_V1 are vendored from
# Time-Series-Library (layers/Embed.py, layers/Conv_Blocks.py) into _tslib_layers,
# so tsdino_timesnet has NO dependency on the Time-Series-Library-main-2 repo.
from ._tslib_layers import DataEmbedding, Inception_Block_V1


class PatchReconDecoder(nn.Module):
    """Maps token embeddings back to raw values for MAE/recon (interface parity).

    Input:  [N, num_tokens, d_model]  →  Output: [N, num_tokens, out_dim]
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


def FFT_for_Period(x, k=2):
    """Identical to TSLib's, but callers cap k to the available spectrum."""
    # x: [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class VarTimesBlock(nn.Module):
    """Length-agnostic TimesBlock.

    Same computation as TSLib's TimesBlock, except every reference to the fixed
    ``seq_len + pred_len`` is replaced by the runtime length ``T``, and the number
    of periods ``k`` is capped to the usable spectrum so short crops don't crash
    ``torch.topk`` (rfft of length T yields T//2 + 1 bins, bin 0 is zeroed).
    """

    def __init__(self, d_model: int, d_ff: int, num_kernels: int, top_k: int):
        super().__init__()
        self.k = top_k
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels),
        )

    def forward(self, x):
        B, T, N = x.size()
        k = max(1, min(self.k, T // 2))
        period_list, period_weight = FFT_for_Period(x, k)

        res = []
        for i in range(k):
            period = max(int(period_list[i]), 1)
            if T % period != 0:
                length = ((T // period) + 1) * period
                padding = torch.zeros([B, length - T, N], device=x.device, dtype=x.dtype)
                out = torch.cat([x, padding], dim=1)
            else:
                length = T
                out = x
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :T, :])
        res = torch.stack(res, dim=-1)                                   # [B, T, N, k]
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        return res + x                                                   # residual


class TimesNetForDINO(nn.Module):
    """Channel-independent, variable-length TimesNet encoder for DINO.

    Pipeline:
        [B,T,C] -- per-instance norm + fold channels --> [B*C, T, 1]
        DataEmbedding(1 -> d_model, no marks)         --> [B*C, T, d_model]
        VarTimesBlock x e_layers (+ LayerNorm)        --> [B*C, T, d_model]
        cross-attention CLS pool over T               --> [B*C, d_model]
        reshape                                       --> [B, C, d_model]   (forward)

    TSMultiCropWrapper reshapes [B, C, d_model] -> [B*C, d_model] before the head.
    Token-level outputs are [B, T, C, d_model] (forward_ibot/forward_recon),
    matching the TSMixer convention so the shared iBOT/MAE loop is compatible.
    """

    def __init__(
        self,
        c_in: int,
        d_model: int,
        e_layers: int,
        d_ff: int,
        dropout: float = 0.1,
        top_k: int = 5,
        num_kernels: int = 6,
        embed: str = "timeF",
        freq: str = "h",
        seq_len: int = 1152,   # nominal pretrain length (NOT a hard cap; reported as num_tokens)
        use_norm: int = 1,
    ):
        super().__init__()
        self.c_in = c_in            # nominal only; CI handles any channel count
        self.d_model = d_model
        self.e_layers = e_layers
        self.seq_len = seq_len
        self.num_tokens = seq_len   # nominal; real token count is the runtime T
        self.use_norm = use_norm

        # Channel-independent: always embed a single variate (1 -> d_model).
        self.enc_embedding = DataEmbedding(1, d_model, embed, freq, dropout)
        self.model = nn.ModuleList(
            [VarTimesBlock(d_model, d_ff, num_kernels, top_k) for _ in range(e_layers)]
        )
        self.layer_norm = nn.LayerNorm(d_model)

        # DINO global pooling: a learnable CLS query attends over the T tokens.
        self.cls_query = nn.Parameter(torch.zeros(1, 1, d_model))
        trunc_normal_(self.cls_query, std=0.02)
        self.global_attn = nn.MultiheadAttention(d_model, num_heads=4, dropout=0.0, batch_first=True)

        # Learnable mask token (iBOT/MAE), applied in embedding space.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        trunc_normal_(self.mask_token, std=0.02)

    # ── internal ────────────────────────────────────────────────────────────
    def _encode(self, x: torch.Tensor, mask_ci: torch.Tensor = None):
        """x: [B, T, C] → tokens [B*C, T, d_model], plus (B, C).

        mask_ci: [B*C, T] bool — True positions replaced by mask_token (iBOT/MAE).
        """
        B, T, C = x.shape
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev
        x = x.permute(0, 2, 1).reshape(B * C, T, 1)        # channel-independent

        enc = self.enc_embedding(x, None)                  # [B*C, T, d_model]
        if mask_ci is not None:
            mt = self.mask_token.expand(enc.shape[0], T, -1)
            enc = torch.where(mask_ci.unsqueeze(-1), mt, enc)
        for blk in self.model:
            enc = self.layer_norm(blk(enc))
        return enc, B, C

    # ── public ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Global DINO rep. x: [B, T, C] → [B, C, d_model] (any T, any C)."""
        enc, B, C = self._encode(x)                        # [B*C, T, d_model]
        q = self.cls_query.expand(enc.shape[0], 1, self.d_model)
        rep, _ = self.global_attn(q, enc, enc)             # [B*C, 1, d_model]
        return rep[:, 0, :].reshape(B, C, self.d_model)    # [B, C, d_model]

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Per-(channel,timestep) tokens for classification. → [B, C, T, d_model]."""
        enc, B, C = self._encode(x)
        T = enc.shape[1]
        return enc.reshape(B, C, T, self.d_model)

    def forward_ibot(self, z: torch.Tensor, mask=None) -> torch.Tensor:
        """iBOT/recon token encoding. z:[B,T,C], mask:[B,T] → [B, T, C, d_model]."""
        mask_ci = None
        if mask is not None:
            B, T, C = z.shape
            mask_ci = mask.unsqueeze(1).expand(-1, C, -1).reshape(B * C, T)
        enc, B, C = self._encode(z, mask_ci=mask_ci)       # [B*C, T, d_model]
        T = enc.shape[1]
        return enc.reshape(B, C, T, self.d_model).permute(0, 2, 1, 3)   # [B, T, C, d_model]

    def forward_recon(self, z: torch.Tensor) -> torch.Tensor:
        return self.forward_ibot(z, mask=None)
