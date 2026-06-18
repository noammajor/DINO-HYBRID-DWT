"""TS2Vec encoder wrapped as a DINO backbone.

Exposes the same interface the DINO pipeline already calls on the PatchTST backbone
(`forward`, `forward_recon`, `forward_ibot`, and the attrs `d_model`, `patch_len`,
`n_vars`, `mask_token`, `cls_query`, `head`, `backbone`) so it is a drop-in third backbone.

The global DINO embedding comes from cross-attention pooling (mirroring the TimeMixer
backbone): a single learnable CLS query attends over all T conv-encoder tokens. Running
attention on top of the conv stack makes the global token see the whole window at any
depth — unlike a CLS baked into the conv, which is bounded by the receptive field. The
per-timestep tokens are returned unchanged for iBOT / MAE / reconstruction.

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

        # Cross-attention global pooling (mirrors the TimeMixer backbone): a single learnable
        # CLS query attends over all T encoder tokens to form the global DINO embedding. Running
        # attention on TOP of the conv encoder makes the global token see the whole window at any
        # depth — unlike a CLS baked into the conv stack, which is bounded by the receptive field.
        self.cls_query   = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_query, std=0.02)
        self.global_attn = nn.MultiheadAttention(d_model, num_heads=4, dropout=0.0, batch_first=True)

        # instance norm for the forecasting head (mirrors PatchTST)
        self.normalization = RevIN(c_in, affine=True)

        # Forecast head flattens the FULL token embedding (all T timestep reps), mirroring
        # PatchTST/TimeMixer — not just the last-step rep. seq_len = num_patch * patch_len.
        _num_patch = kwargs.get('num_patch')
        self.seq_len = (_num_patch * patch_len) if _num_patch is not None else None

        self.head = None
        if head_type == "prediction":
            if self.seq_len is None:
                raise ValueError("TS2Vec prediction head needs num_patch (→ seq_len) at construction")
            self.head = nn.Sequential(nn.Dropout(head_dropout),
                                      nn.Linear(self.seq_len * d_model, target_dim * c_in))
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

    def _encode_tokens(self, z, mask=None):
        """Run the TS2Vec encoder, returning per-timestep token reps.

        z:    [B, T, C]
        mask: [B, T] bool — True = masked timestep (replaced by mask_token before the
              conv stack). None = no masking.
        returns reps [B, T, d_model].

        Mirrors the manual encode path TS2Vec uses (input_fc → conv feature_extractor),
        bypassing TSEncoder.forward so its internal contrastive masking stays disabled —
        DINO supplies its own crop augmentations.
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
        return x

    def _global_pool(self, reps):
        """Cross-attention pooling: a learned CLS query attends over all T token reps.

        reps: [B, T, d_model] → global embedding [B, d_model]. The query connects to every
        token directly, so the global token is full-window regardless of conv depth.
        """
        q = self.cls_query.expand(reps.shape[0], -1, -1)   # [B, 1, d_model]
        global_rep, _ = self.global_attn(q, reps, reps)    # [B, 1, d_model]
        return global_rep[:, 0, :]                         # [B, d_model]

    def forward(self, z, padding_mask=None):
        """z: [B, T, C]. Returns per head_type."""
        if self.head_type == "prediction":
            z    = self._to_series(z)
            zn   = self.normalization(z, mode='norm')
            reps = self._encode_tokens(zn)                 # [B, T, d_model]
            flat = reps.reshape(reps.shape[0], -1)         # [B, T*d_model] — full token embedding
            out  = self.head(flat)                         # [B, pred_len * C]
            out  = out.reshape(out.shape[0], self.pred_len, self.n_vars)
            out  = self.normalization(out, mode='denorm')
            return out
        reps = self._encode_tokens(z)                      # [B, T, d_model]
        glob = self._global_pool(reps)                     # [B, d_model]
        if self.head_type == "classification":
            return self.head(glob)                         # [B, n_classes]
        # "Dino" (and any default): cross-attention global embedding for the DINO head
        return glob                                        # [B, d_model]

    def forward_recon(self, z):
        """Per-timestep token reps. z: [B, T, C] → [B, T, 1, d_model]."""
        reps = self._encode_tokens(z)                      # [B, T, d_model]
        return reps.unsqueeze(2)                           # [B, T, 1, d_model]

    def forward_ibot(self, z, mask=None):
        """iBOT/MAE encoding with a learnable mask token at masked timesteps.

        z:    [B, T, C]
        mask: [B, T] bool — True=masked (student). None=full pass (teacher).
        returns: [B, T, 1, d_model]
        """
        reps = self._encode_tokens(z, mask=mask)           # [B, T, d_model]
        return reps.unsqueeze(2)                           # [B, T, 1, d_model]
