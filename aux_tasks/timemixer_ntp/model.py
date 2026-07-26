"""NTP model: TimeMixer encoder + next-step (forecasting) head."""
import torch
import torch.nn as nn

from backbone import TimeMixerEncoder


class TimeMixerNTP(nn.Module):
    """Next-step prediction over the TimeMixer backbone.

    Encodes a window of seq_len timesteps and predicts the next pred_len points
    with a Linear head over the flattened per-timestep tokens (the TimeMixer
    forecasting head). RevIN-normalize the input, predict, then denormalize.

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
        flat = tokens.reshape(B * C, -1)                            # [B*C, T*d_model]
        pred = self.head(flat).reshape(B, C, -1).permute(0, 2, 1)   # [B, pred_len, C]
        if self.use_revin:
            pred = self.encoder.normalize_layers[0](pred, 'denorm')
        return pred
