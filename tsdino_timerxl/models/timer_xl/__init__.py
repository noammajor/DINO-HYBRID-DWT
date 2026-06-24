"""Vendored Timer-XL (thuml / OpenLTM) — only the files needed for the model.

Unmodified Timer-XL source: model.py (the Model), Transformer_EncDec.py,
SelfAttention_Family.py, Attn_Bias.py, Attn_Projection.py, masking.py.
Only the internal `from layers.* / from utils.*` imports were rewritten to be
relative so this is a self-contained package.
"""
from .model import Model

__all__ = ["Model"]
