"""
CPU smoke test for the dual-stream package — shapes + all three loss modes.

Run:  python TSDiNO/novel/smoke_test.py

It does NOT touch the training loop or real data. It fabricates random crops,
runs the backbone, attaches stand-in linear heads (the real run will use the
repo's DINOHead), and checks every loss mode returns a finite scalar.
"""

import torch
import torch.nn as nn

# allow running as a plain script from anywhere
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # TSDiNO/
from novel import DualStreamBackbone, DualStreamDINOLoss


def _project(z_bcd, head):
    """[B, C, d] -> [B*C, out_dim] (mirrors the multi-crop wrapper flatten)."""
    B, C, d = z_bcd.shape
    return head(z_bcd.reshape(B * C, d))


def main():
    torch.manual_seed(0)
    B, T, C, d, out_dim = 4, 336, 7, 64, 1024
    n_global, n_local = 2, 2
    ncrops = n_global + n_local

    backbone = DualStreamBackbone(c_in=C, seq_len=T, d_model=d,
                                  patch_macro=32, patch_micro=8)
    crops = [torch.randn(B, T, C) for _ in range(ncrops)]

    print("== backbone output shapes ==")
    out0 = backbone(crops[0])
    print("  macro:", tuple(out0["macro"].shape), " micro:", tuple(out0["micro"].shape))

    # stand-in heads (real run uses TSDiNO/models/layers/Dino_Head.DINOHead)
    head_macro  = nn.Linear(d, out_dim)
    head_micro  = nn.Linear(d, out_dim)
    head_concat = nn.Linear(2 * d, out_dim)

    def stream_outputs(which_crops):
        outs = [backbone(c) for c in which_crops]
        macro = torch.cat([_project(o["macro"], head_macro) for o in outs], dim=0)
        micro = torch.cat([_project(o["micro"], head_micro) for o in outs], dim=0)
        cat = torch.cat([
            head_concat(torch.cat([o["macro"], o["micro"]], dim=-1).reshape(B * C, 2 * d))
            for o in outs], dim=0)
        return {"macro": macro, "micro": micro, "concat": cat}

    student = stream_outputs(crops)                 # all crops
    teacher = stream_outputs(crops[:n_global])      # global crops only

    print("\n== loss modes ==")
    common = dict(out_dim=out_dim, ncrops=ncrops, n_global=n_global,
                  warmup_teacher_temp=0.04, teacher_temp=0.04,
                  warmup_teacher_temp_epochs=0, nepochs=80)
    for mode in ("concat", "dual", "cross"):
        loss_fn = DualStreamDINOLoss(mode=mode, **common)
        loss = loss_fn(student, teacher, epoch=0)
        assert torch.isfinite(loss), f"{mode} loss not finite"
        loss.backward(retain_graph=True)            # gradients flow
        print(f"  mode={mode:7s}  loss={loss.item():.4f}  (finite, backward ok)")

    print("\nAll good.")


if __name__ == "__main__":
    main()
