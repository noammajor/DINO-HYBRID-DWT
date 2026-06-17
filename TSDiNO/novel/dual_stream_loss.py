"""
Dual-Stream DINO Loss  (three selectable modes)
===============================================

This deliberately reuses the EXACT DINO machinery from `TSDiNO/main.py`'s
`DINOLoss` — student temperature, teacher centering + sharpening, the cross-crop
loop that skips same-view pairs, and the dist-aware `update_center`. We just
factor it into a small `_DINOLoss` (identical to main's, but parameterised by
`n_global` instead of reading the module-level `cfg['global_crops']`) and then
**compose** instances of it, one per teacher "target" so each keeps its own
running center.

Modes (selected by `mode`):

  "concat"  — baseline. Glue the two stream embeddings, push through ONE head,
              run vanilla DINO. One `_DINOLoss`. Control to prove the backbone
              trains; disentangles nothing on its own.

  "dual"    — two `_DINOLoss` (macro, micro). DINO *within* each stream
              (student macro ↔ teacher macro, student micro ↔ teacher micro),
              plus a small consistency term so the streams describe one sample.

  "cross"   — the novel objective. Two `_DINOLoss` keyed by the TEACHER target:
              student MICRO ↔ teacher MACRO  and  student MACRO ↔ teacher MICRO.
              Forces local cycles to predict the global trend and vice-versa.

INPUT CONVENTION (matches the multi-crop wrapper output):
  student/teacher are dicts of head-projected outputs [n_crops*(B*C), out_dim]:
      concat : {"concat": ...}
      dual/cross : {"macro": ..., "micro": ...}   (teacher carries n_global crops)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class _DINOLoss(nn.Module):
    """Faithful copy of TSDiNO/main.py `DINOLoss`, parameterised by `n_global`.

    Only change vs. main: the teacher output is chunked by the explicit
    `n_global` argument instead of `len(cfg['global_crops'])`, so the class has
    no global-config dependency and can be instantiated several times.
    """

    def __init__(self, out_dim, ncrops, n_global, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.n_global = n_global
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(max(0, nepochs - warmup_teacher_temp_epochs)) * teacher_temp,
        ))

    def forward(self, student_output, teacher_output, epoch):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        temp = self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule) - 1)]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.n_global)

        total_loss, n_loss_terms = 0, 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            world_size = dist.get_world_size()
        else:
            world_size = 1
        batch_center = batch_center / (len(teacher_output) * world_size)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DualStreamDINOLoss(nn.Module):
    """Composes `_DINOLoss` instances per the chosen disentanglement mode."""

    def __init__(self, out_dim, ncrops, n_global, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, mode="cross",
                 student_temp=0.1, center_momentum=0.9, consistency_weight=0.5):
        super().__init__()
        assert mode in ("concat", "dual", "cross"), mode
        self.mode = mode
        self.consistency_weight = consistency_weight
        self.student_temp = student_temp

        def _mk():
            return _DINOLoss(out_dim, ncrops, n_global, warmup_teacher_temp,
                             teacher_temp, warmup_teacher_temp_epochs, nepochs,
                             student_temp, center_momentum)

        if mode == "concat":
            self.dino = _mk()
        else:
            # one head/center per TEACHER target stream (macro, micro)
            self.dino_macro = _mk()   # center tracks teacher MACRO targets
            self.dino_micro = _mk()   # center tracks teacher MICRO targets

    def forward(self, student: dict, teacher: dict, epoch: int):
        if self.mode == "concat":
            return self.dino(student["concat"], teacher["concat"], epoch)

        if self.mode == "dual":
            l_macro = self.dino_macro(student["macro"], teacher["macro"], epoch)
            l_micro = self.dino_micro(student["micro"], teacher["micro"], epoch)
            # streams should agree on the sample (symmetric soft CE, stop-grad target)
            ps = F.log_softmax(student["macro"] / self.student_temp, dim=-1)
            pm = F.softmax(student["micro"] / self.student_temp, dim=-1).detach()
            consist = -(pm * ps).sum(-1).mean()
            return l_macro + l_micro + self.consistency_weight * consist

        # mode == "cross": teacher target picks the matching center
        l_mic_to_mac = self.dino_macro(student["micro"], teacher["macro"], epoch)
        l_mac_to_mic = self.dino_micro(student["macro"], teacher["micro"], epoch)
        return 0.5 * (l_mic_to_mac + l_mac_to_mic)
