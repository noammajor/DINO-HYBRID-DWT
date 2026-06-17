"""TS2Vec encoder wrapped as a DINO backbone.

Exposes the same interface the DINO pipeline already calls on the PatchTST backbone
(`forward`, `forward_recon`, `forward_ibot`, and the attrs `d_model`, `patch_len`,
`n_vars`, `mask_token`, `head`, `backbone`) so it is a drop-in third backbone.

TS2Vec is channel-MIXED and timestep-tokenised: the full multivariate series
[B, T, C] goes into `TSEncoder` (input_dims=C) and out come per-timestep
representations [B, T, d_model].  Tokens are timesteps (patch_len = 1) and there is
no per-variable token dimension, so wherever the shared MLM code expects
[B, NP, n_vars, d_model] we emit n_vars = 1 and carry the C raw channels in the
reconstruction targets.
"""
import torch
from torch import nn

from models.ts2vec_encoder import TSEncoder          # vendored from ts2vec-main
from models.layers.revin import RevIN


class PatchReconDecoder(nn.Module):
    """Maps token embeddings back to raw values (recon / MAE head).

    Input:  [..., d_model]   Output: [..., out_dim]
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


class TS2VecForDINO(nn.Module):
    """TS2Vec `TSEncoder` adapted to the DINO backbone contract.

    Accepts the same constructor kwargs as PatchTST (extra ones are ignored via
    **kwargs) so the existing call sites in main.py work unchanged; only `c_in`,
    `target_dim`, `d_model`, `head_type`, and the TS2Vec-specific `depth`/
    `hidden_dims` are actually used.
    """
    def __init__(self, c_in: int, target_dim: int, d_model: int = 320,
                 head_type: str = "Dino", depth: int = None, n_layers: int = 10,
                 hidden_dims: int = 64, head_dropout: float = 0.0, mlp_head: bool = False,
                 patch_len: int = 1, **kwargs):
        super().__init__()
        # TS2Vec encoder depth is driven by `depth` if given, else `n_layers` — so the
        # existing layer-sweep wiring (--encoder_layers → n_layers) controls it like the
        # other backbones.
        _depth = depth if depth is not None else n_layers
        self.n_vars     = c_in
        self.d_model    = d_model
        self.patch_len  = patch_len           # timestep tokens
        self.hidden_dims = hidden_dims
        self.head_type  = head_type
        self.pred_len   = target_dim

        # channel-mixed encoder: input_dims = C
        self.backbone = TSEncoder(input_dims=c_in, output_dims=d_model,
                                  hidden_dims=hidden_dims, depth=_depth)
        # expose d_model on the inner encoder — main.py reads student.backbone.d_model
        self.backbone.d_model = d_model

        # learnable mask token injected after input_fc (hidden dim) for iBOT / MAE
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dims))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # instance norm for the forecasting head (mirrors PatchTST)
        self.normalization = RevIN(c_in, affine=True)

        self.head = None
        if head_type == "prediction":
            self.head = nn.Sequential(nn.Dropout(head_dropout),
                                      nn.Linear(d_model, target_dim * c_in))
        elif head_type == "classification":
            self.head = nn.Sequential(nn.Dropout(head_dropout),
                                      nn.Linear(d_model, target_dim))

    @property
    def num_patch(self):
        # timestep tokens: count is dynamic (= seq_len); not fixed at build time
        return None

    @staticmethod
    def _to_series(z):
        # Accept raw [B, T, C] or pre-patched [B, P, PL, C] (flatten patches back to series).
        if z.dim() == 4:
            B, P, PL, C = z.shape
            z = z.reshape(B, P * PL, C)
        return z

    def _encode(self, z):
        # 'all_true' disables TS2Vec's internal (contrastive) masking — DINO supplies
        # its own crop augmentations, so the encoder pass should be deterministic.
        return self.backbone(self._to_series(z), mask='all_true')   # [B, T, d_model]

    def _pool(self, reps):
        return reps.mean(dim=1)                            # [B, d_model]

    def forward(self, z, padding_mask=None):
        """z: [B, T, C]. Returns per head_type."""
        if self.head_type == "prediction":
            z    = self._to_series(z)
            zn   = self.normalization(z, mode='norm')
            reps = self._encode(zn)
            pooled = reps[:, -1, :]                        # last-step rep
            out  = self.head(pooled)                       # [B, pred_len * C]
            out  = out.reshape(out.shape[0], self.pred_len, self.n_vars)
            out  = self.normalization(out, mode='denorm')
            return out
        reps = self._encode(z)                             # [B, T, d_model]
        if self.head_type == "classification":
            return self.head(self._pool(reps))             # [B, n_classes]
        # "Dino" (and any default): time-pooled embedding for the DINO head
        return self._pool(reps)                            # [B, d_model]

    def forward_recon(self, z):
        """Full pass token reps. z: [B, T, C] → [B, T, 1, d_model]."""
        reps = self._encode(z)
        return reps.unsqueeze(2)

    def forward_ibot(self, z, mask=None):
        """iBOT/MAE encoding with a learnable mask token at masked timesteps.

        z:    [B, T, C]
        mask: [B, T] bool — True=masked (student). None=full pass (teacher).
        returns: [B, T, 1, d_model]
        """
        enc = self.backbone
        z = self._to_series(z)
        nan_mask = ~z.isnan().any(axis=-1)
        z = z.clone()
        z[~nan_mask] = 0
        x = enc.input_fc(z)                                # [B, T, hidden_dims]
        if mask is not None:
            m = mask.to(x.device).bool().unsqueeze(-1)     # [B, T, 1]
            x = torch.where(m, self.mask_token, x)
        x = x.transpose(1, 2)                              # [B, hidden, T]
        x = enc.repr_dropout(enc.feature_extractor(x))     # [B, d_model, T]
        x = x.transpose(1, 2)                              # [B, T, d_model]
        return x.unsqueeze(2)                              # [B, T, 1, d_model]
