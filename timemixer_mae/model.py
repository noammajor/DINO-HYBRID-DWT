"""MAE model: TimeMixer encoder + PatchTST-style block reconstruction head."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import TimeMixerEncoder


class BlockMAEHead(nn.Module):
    """Per-block reconstruction head on TimeMixer tokens (PatchTST-style, lossless).

    TimeMixer emits one token per timestep. We group T timestep tokens into
    T // block_len blocks and reconstruct each block's block_len raw values from
    its block_len tokens jointly: flatten the block's tokens to block_len*d_model
    and map to block_len with a single Linear. Unlike mean-pooling, this keeps all
    within-block information (one prediction per block, no averaging).

    Input:  tokens [N, T, d_model]   (N = B * n_vars)
    Output: recon  [N, n_blocks * block_len]
    """

    def __init__(self, d_model, block_len, dropout=0.0):
        super().__init__()
        self.block_len = block_len
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(block_len * d_model, block_len)

    def forward(self, x):
        N, T, D = x.shape
        nb = T // self.block_len
        x = x[:, :nb * self.block_len, :].reshape(N, nb, self.block_len * D)  # flatten block
        x = self.linear(self.dropout(x))                 # [N, nb, block_len]
        return x.reshape(N, nb * self.block_len)         # [N, T']


class TimeMixerMAE(nn.Module):
    """Masked-autoencoder pretraining model over the TimeMixer backbone.

    The window is split into non-overlapping blocks of block_len timesteps;
    mask_ratio of them are masked (embeddings → mask_token), the encoder runs,
    and the head reconstructs every block. MSE is computed on masked blocks only.

    forward(x): x [B, T, C] → (loss, recon, target, mask)
        recon/target/mask: [B*C, n_blocks*block_len]; mask True = masked timestep.
    """

    def __init__(self, c_in, seq_len, block_len=8, mask_ratio=0.4,
                 d_model=128, e_layers=3, d_ff=256, dropout=0.1, head_dropout=0.1,
                 **backbone_kwargs):
        super().__init__()
        if seq_len % block_len != 0:
            raise ValueError(f"seq_len ({seq_len}) must be divisible by block_len ({block_len})")
        self.block_len = block_len
        self.mask_ratio = mask_ratio
        self.n_blocks = seq_len // block_len
        self.encoder = TimeMixerEncoder(
            c_in=c_in, seq_len=seq_len, d_model=d_model, e_layers=e_layers,
            d_ff=d_ff, dropout=dropout, **backbone_kwargs)
        self.head = BlockMAEHead(d_model, block_len, head_dropout)

    def forward(self, x):
        B, T, C = x.shape
        Tb = self.n_blocks * self.block_len

        block_mask = torch.rand(B, self.n_blocks, device=x.device) < self.mask_ratio  # [B, nb]
        ts_mask = block_mask.unsqueeze(-1).expand(B, self.n_blocks, self.block_len).reshape(B, Tb)
        if Tb < T:
            ts_mask = F.pad(ts_mask, (0, T - Tb), value=False)

        tokens = self.encoder(x, mask=ts_mask)                       # [B*C, T, d_model]
        recon = self.head(tokens)                                    # [B*C, Tb]

        target = x[:, :Tb, :].permute(0, 2, 1).reshape(B * C, Tb)    # [B*C, Tb]
        mask = (block_mask.unsqueeze(1).expand(B, C, self.n_blocks)
                .reshape(B * C, self.n_blocks)
                .unsqueeze(-1).expand(-1, -1, self.block_len).reshape(B * C, Tb))

        denom = mask.sum().clamp(min=1)
        loss = (((recon - target) ** 2) * mask).sum() / denom
        return loss, recon, target, mask


class TimeMixerForecast(nn.Module):
    """Downstream forecasting model: (pretrained) TimeMixer encoder + linear head.

    Same structure as the NTP model — encode a seq_len window and predict the next
    pred_len points — used here to evaluate the MAE-pretrained encoder in-domain.
    Load pretrained weights into ``self.encoder`` (strict=False; the MAE mask_token
    is simply unused).

    forward(x): x [B, seq_len, C] → pred [B, pred_len, C]
    """

    def __init__(self, c_in, seq_len, pred_len, d_model=128, e_layers=3, d_ff=256,
                 dropout=0.1, use_norm=1, **backbone_kwargs):
        super().__init__()
        self.pred_len = pred_len
        self.use_revin = (use_norm == 1)
        self.encoder = TimeMixerEncoder(
            c_in=c_in, seq_len=seq_len, d_model=d_model, e_layers=e_layers,
            d_ff=d_ff, dropout=dropout, use_norm=use_norm, **backbone_kwargs)
        self.head = nn.Linear(seq_len * d_model, pred_len)

    def forward(self, x):
        B, T, C = x.shape
        tokens = self.encoder.encode(x, normalize=self.use_revin)   # [B*C, T, d_model]
        flat = tokens.reshape(B * C, -1)
        pred = self.head(flat).reshape(B, C, -1).permute(0, 2, 1)   # [B, pred_len, C]
        if self.use_revin:
            pred = self.encoder.normalize_layers[0](pred, 'denorm')
        return pred
