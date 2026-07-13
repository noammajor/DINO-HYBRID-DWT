"""
Four-panel augmentation diagnostic for TSDiNO.

Runs over multiple datasets and multiple pretrained backbones.
One output figure per dataset: Visuals/aug_views_{dataset}.png

Panels (columns = variables, samples = overlaid lines):
  1. Overlay   — original, teacher-aug, student-aug on the same axis
  2. FFT       — power spectrum of each view
  3. Phase     — teacher with sym4 vs db4 wavelet overlaid (phase-shift check)
  4. Repr sim  — cosine similarity between backbone(original) and backbone(teacher/student)
                 one bar group per checkpoint/backbone, two bars each (teacher, student)

Usage:
    python Visuals/visualize_augmentations.py \
        --checkpoints ./ckpt_ibot/checkpoint_best.pth \
                      ./ckpt_mae/checkpoint_best.pth \
                      ./ckpt_base/checkpoint_best.pth \
        --ckpt_labels ibot mae base \
        --datasets etth1 etth2 ettm1 ettm2
"""

import sys, os, argparse, random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from types import SimpleNamespace

# ── path setup ────────────────────────────────────────────────────────────────
_root      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_tsdino    = os.path.join(_root, "tsdino_timemixer")
_tm_root   = os.path.join(_root, "TimeMixer-main")
_tm_models = os.path.join(_tm_root, "models")
for p in [_tm_models, _tm_root, _root, _tsdino]:  # _tsdino last → position 0, wins over TimeMixer's models/
    if p not in sys.path:
        sys.path.insert(0, p)

import data_agumentation as aug
from config import config as cfg

# dataset name → CSV filename (no registry dependency)
_DATASET_FILES = {
    'etth1': 'ETTh1.csv',
    'etth2': 'ETTh2.csv',
    'ettm1': 'ETTm1.csv',
    'ettm2': 'ETTm2.csv',
    'weather':     'weather.csv',
    'electricity': 'electricity.csv',
    'traffic':     'traffic.csv',
}


# ── data ──────────────────────────────────────────────────────────────────────

def load_windows(dataset_name, data_dir, n_samples, seq_len, pred_len, seed):
    """
    Load n_samples windows of size seq_len+pred_len from the dataset CSV.
    Normalises with StandardScaler fit on the training portion (first 60%).
    Falls back to synthetic signals if the file is not found.
    Returns x_list, y_list, n_vars, col_names.
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    rng      = np.random.default_rng(seed)
    total    = seq_len + pred_len
    csv_name = _DATASET_FILES.get(dataset_name, f'{dataset_name}.csv')
    csv_path = os.path.join(data_dir, csv_name)

    if os.path.isfile(csv_path):
        df   = pd.read_csv(csv_path)
        data = df.select_dtypes('number').values.astype(np.float32)
        col_names = list(df.select_dtypes('number').columns)
        T, C = data.shape
        train_end = int(T * 0.6)
        sc = StandardScaler()
        sc.fit(data[:train_end])
        data    = sc.transform(data)
        starts  = rng.integers(0, T - total, size=n_samples)
        x_list  = [torch.tensor(data[s:s+seq_len])       for s in starts]
        y_list  = [torch.tensor(data[s+seq_len:s+total]) for s in starts]
        print(f"  loaded {csv_path}  shape={data.shape}")
        return x_list, y_list, C, col_names
    else:
        print(f"  [warn] {csv_path} not found — using synthetic signals")
        C, col_names = 7, [f"var{i}" for i in range(7)]
        x_list, y_list = [], []
        for _ in range(n_samples):
            t   = np.linspace(0, 6*np.pi, total)
            sig = np.stack([
                np.sin(t*(1+0.25*rng.random())) + 0.2*rng.standard_normal(total)
                for _ in range(C)
            ], axis=1).astype(np.float32)
            x_list.append(torch.tensor(sig[:seq_len]))
            y_list.append(torch.tensor(sig[seq_len:]))
        return x_list, y_list, C, col_names


# ── augmentation ──────────────────────────────────────────────────────────────

_ALIASES = {'soft': 'soft_threshold', 'hard': 'high_perturb'}

def _make_aug(spec, wavelet_override=None):
    t = spec['type']
    if isinstance(t, list):
        t = t[0]
    w_pool = None if wavelet_override else spec.get('wavelet_pool', cfg.get('dwt_wavelet_pool'))
    w      = wavelet_override or spec.get('wavelet', cfg['dwt_wavelet'])
    kw = dict(
        wavelet                  = w,
        wavelet_pool             = w_pool,
        level                    = spec.get('level',               cfg['dwt_level']),
        soft_threshold_sigma     = spec.get('soft_threshold_sigma', cfg.get('dwt_soft_threshold_sigma', 0.3)),
        zero_out_ratio           = spec.get('zero_out_ratio',       cfg.get('dwt_zero_out_ratio', 0.3)),
        finest_levels            = spec.get('finest_levels',        cfg.get('dwt_finest_levels', 1)),
        high_perturb_noise_range = spec.get('high_perturb_noise_range', cfg.get('dwt_high_perturb_noise_range', (0.05, 0.12))),
        band_scale_approx_range  = cfg.get('dwt_band_scale_approx_range', (0.80, 1.20)),
        band_scale_detail_range  = cfg.get('dwt_band_scale_detail_range', (0.40, 1.60)),
    )
    if t.startswith('modwt_'):
        return aug.MODWTAugmentation(mode=_ALIASES.get(t[6:], t[6:]), **kw)
    if t.startswith('swt_'):
        return aug.SWTAugmentation(mode=_ALIASES.get(t[4:], t[4:]), **kw)
    if t.startswith('dwt_'):
        return aug.DWTAugmentation(mode=_ALIASES.get(t[4:], t[4:]), **kw)
    raise ValueError(t)


# ── backbone ──────────────────────────────────────────────────────────────────

def load_backbone(ckpt_path, c_in, seq_len, device):
    """
    Build TSMixerForDINO using architecture params stored in the checkpoint's
    'args' namespace, then load teacher weights.
    Returns (backbone, loaded_ok, backbone_seq_len).
    """
    from models.ts_mixer_backbone import TSMixerForDINO

    arch = {}
    backbone_seq_len = seq_len
    patch_len = cfg.get('patch_len', 16)
    if ckpt_path and os.path.isfile(ckpt_path):
        try:
            ckpt_meta = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            raw_sd    = ckpt_meta.get('teacher', ckpt_meta)
            a         = ckpt_meta.get('args', None)

            # ── infer seq_len + e_layers + d_model directly from weight shapes ──
            # First season-mixing Linear has in_features == seq_len.
            n_layers = 0
            for k, v in raw_sd.items():
                kk = k.replace('module.', '')
                if kk.startswith('backbone.'):
                    kk = kk[len('backbone.'):]
                if 'mixing_multi_scale_season.down_sampling_layers.0.0.weight' in kk:
                    backbone_seq_len = v.shape[1]      # in_features == seq_len
                if kk.startswith('pdm_blocks.'):
                    try:
                        n_layers = max(n_layers, int(kk.split('.')[1]) + 1)
                    except (IndexError, ValueError):
                        pass
                if kk == 'mask_token':
                    d_model_inferred = v.shape[-1]

            arch = dict(
                c_in                 = c_in,   # always match the DATA (backbone is channel-independent)
                seq_len              = backbone_seq_len,
                e_layers             = n_layers or (getattr(a, 'tsmixer_e_layers', 4) if a is not None else 4),
                d_model              = locals().get('d_model_inferred',
                                          getattr(a, 'tsmixer_d_model', 128) if a is not None else 128),
                d_ff                 = getattr(a, 'tsmixer_d_ff', 256) if a is not None else 256,
                down_sampling_layers = getattr(a, 'tsmixer_down_sampling_layers', 3) if a is not None else 3,
                down_sampling_window = getattr(a, 'tsmixer_down_sampling_window', 2) if a is not None else 2,
                down_sampling_method = getattr(a, 'tsmixer_down_sampling_method', 'avg') if a is not None else 'avg',
                moving_avg           = getattr(a, 'tsmixer_moving_avg', 25) if a is not None else 25,
            )
            print(f"    arch from ckpt: e_layers={arch['e_layers']} d_model={arch['d_model']} "
                  f"seq_len={arch['seq_len']} c_in={arch['c_in']}  (inferred from weights)")
        except Exception as e:
            print(f"    [warn] could not read arch from checkpoint: {e}")

    backbone = TSMixerForDINO(
        c_in                  = arch.get('c_in',                 c_in),
        seq_len               = arch.get('seq_len',              seq_len),
        d_model               = arch.get('d_model',              cfg.get('tsmixer_d_model',             128)),
        e_layers              = arch.get('e_layers',             cfg.get('tsmixer_e_layers',             4)),
        d_ff                  = arch.get('d_ff',                 cfg.get('tsmixer_d_ff',                 256)),
        dropout               = cfg.get('dropout',               0.1),
        patch_len             = patch_len if arch else cfg.get('patch_len', 16),
        down_sampling_layers  = arch.get('down_sampling_layers', cfg.get('tsmixer_down_sampling_layers', 3)),
        down_sampling_window  = arch.get('down_sampling_window', cfg.get('tsmixer_down_sampling_window', 2)),
        down_sampling_method  = arch.get('down_sampling_method', cfg.get('tsmixer_down_sampling_method', 'avg')),
        decomp_method         = cfg.get('tsmixer_decomp_method', 'moving_avg'),
        moving_avg            = arch.get('moving_avg',           cfg.get('tsmixer_moving_avg',           25)),
        top_k                 = cfg.get('tsmixer_top_k',         5),
        use_norm              = cfg.get('tsmixer_use_norm',      1),
        channel_independence  = cfg.get('tsmixer_channel_independence', 1),
    ).to(device)

    loaded_ok = False
    if ckpt_path and os.path.isfile(ckpt_path):
        try:
            ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
            raw        = ckpt.get('teacher', ckpt)
            own_state  = backbone.state_dict()
            new_state  = {}
            for k, v in raw.items():
                k = k.replace('module.', '')
                # TSMultiCropWrapper stores as backbone.* → strip prefix
                if k.startswith('backbone.'):
                    k = k[len('backbone.'):]
                if k in own_state and own_state[k].shape == v.shape:
                    new_state[k] = v
            backbone.load_state_dict(new_state, strict=False)
            loaded_ok = True
            pct = 100 * len(new_state) / max(len(own_state), 1)
            print(f"  loaded {len(new_state)}/{len(own_state)} ({pct:.0f}%) weights from {os.path.basename(ckpt_path)}")
        except Exception as e:
            print(f"  [warn] failed to load {ckpt_path}: {e}")
    else:
        print(f"  [warn] checkpoint not found: {ckpt_path}")

    backbone.eval()
    return backbone, loaded_ok, backbone_seq_len


@torch.no_grad()
def get_repr(backbone, x, device):
    """x: [seq_len, C]  →  repr: [C, d_model] (cross-attention pooled)."""
    out = backbone(x.unsqueeze(0).float().to(device))   # [1, C, d_model]
    return out[0]                                        # [C, d_model]


@torch.no_grad()
def get_tokens(backbone, x, device):
    """x: [seq_len, C]  →  per-timestep tokens [T, C, d_model] (pre-pooling).
    Uses forward_ibot so the high-frequency, per-timestep augmentation effect
    survives (the cross-attention pooling in forward() averages it away)."""
    out = backbone.forward_ibot(x.unsqueeze(0).float().to(device))  # [1, T, C, d_model]
    return out[0]                                                    # [T, C, d_model]


def token_rel_l2(a, b):
    """Per-token relative L2 distance, averaged over timesteps and channels.
    a, b: [T, C, d_model]."""
    num = (a - b).norm(dim=-1)                # [T, C]
    den = a.norm(dim=-1).clamp_min(1e-8)      # [T, C]
    return (num / den).mean().item()


def cosine_sim(a, b):
    """a, b: [C, d_model]  →  mean cosine similarity across channels."""
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1).mean().item()             # scalar


def rel_l2(a, b):
    """Relative L2 distance ‖a-b‖ / ‖a‖, mean across channels.
    Unlike cosine it does not normalise away magnitude, so it stays sensitive
    even when the two representations point in nearly the same direction."""
    num = (a - b).norm(dim=-1)
    den = a.norm(dim=-1).clamp_min(1e-8)
    return (num / den).mean().item()


# ── helpers ───────────────────────────────────────────────────────────────────

def _fft_power(sig):
    fft  = np.fft.rfft(sig)
    pwr  = (np.abs(fft)**2) / len(sig)
    freq = np.fft.rfftfreq(len(sig))
    return freq, pwr


_SAMPLE_HUES  = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
_CKPT_COLORS  = ['#e6550d', '#756bb1', '#31a354']   # ibot / mae / base (overridden by labels)

def _sample_colors(si):
    base = matplotlib.colors.to_rgb(_SAMPLE_HUES[si % len(_SAMPLE_HUES)])
    def adj(c, f): return min(c * f, 1.0)
    return {
        'orig':    matplotlib.colors.to_hex([adj(c, 1.0) for c in base]),
        'teacher': matplotlib.colors.to_hex([adj(c, 0.55) for c in base]),
        'student': matplotlib.colors.to_hex([adj(c, 1.55) if c < 0.65 else adj(c, 0.85) for c in base]),
    }


# ── main plotting ─────────────────────────────────────────────────────────────

def make_figure(dataset_name, samples, futures, var_indices, col_names,
                teacher_tf, student_tf, teacher_sym4_tf, teacher_db4_tf,
                backbones, ckpt_labels, device, out_path,
                backbone_samples=None):

    n_v    = len(var_indices)
    n_rows = 4
    fig, axes = plt.subplots(n_rows, n_v, figsize=(4.5*n_v, 3.2*n_rows), squeeze=False)

    row_labels = [
        "Overlay: orig / teacher / student",
        "FFT power spectrum",
        "Phase: teacher  sym4 (—) vs db4 (--)",
        "Per-token rel-L2 dist to original",
    ]

    for vi, var in enumerate(var_indices):
        var_name = col_names[var] if var < len(col_names) else f"var{var}"

        # ── collect per-sample augmented views ─────────────────────────────
        orig_sigs, t_sigs, s_sigs = [], [], []
        for x in samples:
            orig_sigs.append(x[:, var].numpy())
            t_sigs.append(teacher_tf(x)[:, var].numpy())
            s_sigs.append(student_tf(x)[:, var].numpy())

        # ── panel 1: overlay ───────────────────────────────────────────────
        ax = axes[0, vi]
        for si, (orig, tv, sv) in enumerate(zip(orig_sigs, t_sigs, s_sigs)):
            c   = _sample_colors(si)
            lw  = 0.85
            a   = 0.75
            sfx = f" s{si+1}" if len(samples) > 1 else ""
            ax.plot(orig, color=c['orig'],    lw=lw, alpha=a, label=f"orig{sfx}")
            ax.plot(tv,   color=c['teacher'], lw=lw, alpha=a, ls='--', label=f"teacher{sfx}")
            ax.plot(sv,   color=c['student'], lw=lw, alpha=a, ls=':',  label=f"student{sfx}")

        # ── panel 2: FFT ───────────────────────────────────────────────────
        ax = axes[1, vi]
        for si, (orig, tv, sv) in enumerate(zip(orig_sigs, t_sigs, s_sigs)):
            c = _sample_colors(si)
            for sig, col, ls in [(orig, c['orig'], '-'), (tv, c['teacher'], '--'), (sv, c['student'], ':')]:
                freq, pwr = _fft_power(sig)
                ax.semilogy(freq, pwr + 1e-12, color=col, lw=0.75, alpha=0.75, ls=ls)

        # ── panel 3: sym4 vs db4 phase ─────────────────────────────────────
        ax = axes[2, vi]
        for si, x in enumerate(samples):
            c     = _sample_colors(si)
            sfx   = f" s{si+1}" if len(samples) > 1 else ""
            sym_v = teacher_sym4_tf(x)[:, var].numpy()
            db_v  = teacher_db4_tf(x)[:, var].numpy()
            ax.plot(sym_v, color=c['orig'],    lw=0.85, alpha=0.8,  label=f"sym4{sfx}")
            ax.plot(db_v,  color=c['student'], lw=0.85, alpha=0.8, ls='--', label=f"db4{sfx}")

        # ── panel 4: per-token relative-L2 distance to original ────────────
        ax = axes[3, vi]
        if backbones:
            n_ckpt  = len(backbones)
            n_samp  = len(samples)
            x_base  = np.arange(n_samp)
            width   = 0.35
            offsets = np.linspace(-width*(n_ckpt-1)/2, width*(n_ckpt-1)/2, n_ckpt)

            repr_src = backbone_samples if backbone_samples is not None else samples
            for ci, (backbone, label) in enumerate(zip(backbones, ckpt_labels)):
                t_d, s_d = [], []
                for x in repr_src:
                    r_orig    = get_tokens(backbone, x,             device)
                    r_teacher = get_tokens(backbone, teacher_tf(x), device)
                    r_student = get_tokens(backbone, student_tf(x), device)
                    t_d.append(token_rel_l2(r_orig, r_teacher))
                    s_d.append(token_rel_l2(r_orig, r_student))

                col = _CKPT_COLORS[ci % len(_CKPT_COLORS)]
                ax.bar(x_base + offsets[ci] - width*0.25, t_d, width*0.45,
                       color=col, alpha=0.85, label=f"{label} teacher")
                ax.bar(x_base + offsets[ci] + width*0.25, s_d, width*0.45,
                       color=col, alpha=0.4,  hatch='//', label=f"{label} student")

            ax.set_ylim(bottom=0)
            ax.set_xticks(x_base)
            ax.set_xticklabels([f"s{i+1}" for i in range(n_samp)], fontsize=6)
            ax.set_ylabel("per-token rel. L2  ‖Δ‖/‖orig‖", fontsize=7)
        else:
            ax.text(0.5, 0.5, "no checkpoints provided",
                    ha='center', va='center', transform=ax.transAxes, fontsize=8, color='grey')

        # ── axis decoration ────────────────────────────────────────────────
        for row in range(n_rows):
            ax = axes[row, vi]
            ax.set_title(var_name, fontsize=8)
            ax.tick_params(labelsize=6)
            if vi == 0:
                ax.set_ylabel(row_labels[row], fontsize=7, labelpad=4)
            if row != 1 and row != 3:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labels, fontsize=5, loc='upper right', ncol=2)
            if row == 3 and backbones:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labels, fontsize=5, loc='lower right', ncol=2)

    teacher_type = cfg['global_crops'][0]['type']
    student_type = cfg['local_crops'][0]['type']
    pool = cfg.get('dwt_wavelet_pool') or cfg['dwt_wavelet']
    fig.suptitle(
        f"{dataset_name}  |  teacher: {teacher_type}  student: {student_type}"
        f"  |  pool: {pool}",
        fontsize=9, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"saved → {out_path}")
    plt.close(fig)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets',     nargs='+',
                        default=['etth1', 'etth2', 'ettm1', 'ettm2'])
    parser.add_argument('--transforms',   nargs='+', default=['dwt', 'swt', 'modwt'],
                        choices=['dwt', 'swt', 'modwt'],
                        help='wavelet transform families to compare (each produces its own figure)')
    parser.add_argument('--data_dir',     default='/home/shared/datasets/data - forecasting timeseries',
                        help='directory containing ETTh1.csv etc.')
    parser.add_argument('--checkpoints',  nargs='*', default=[],
                        help='paths to checkpoint_best.pth files (one per backbone)')
    parser.add_argument('--ckpt_labels',  nargs='*', default=[],
                        help='display labels for each checkpoint (same order)')
    parser.add_argument('--vars',         nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--n_samples',    type=int,  default=3)
    parser.add_argument('--seq_len',      type=int,  default=336)
    parser.add_argument('--pred_len',     type=int,  default=96)
    parser.add_argument('--seed',         type=int,  default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = os.path.dirname(os.path.abspath(__file__))

    ckpt_labels = args.ckpt_labels or [
        os.path.basename(os.path.dirname(p)) for p in args.checkpoints
    ]
    while len(ckpt_labels) < len(args.checkpoints):
        ckpt_labels.append(f"ckpt{len(ckpt_labels)}")

    # teacher/student mode for each transform family
    _TRANSFORM_MODES = {
        'dwt':   ('dwt_low_pass',   'dwt_hard'),
        'swt':   ('swt_low_pass',   'swt_hard'),
        'modwt': ('modwt_low_pass', 'modwt_hard'),
    }

    for dataset in args.datasets:
        print(f"\n{'='*50}\n  dataset: {dataset}\n{'='*50}")

        samples, futures, n_vars, col_names = load_windows(
            dataset, args.data_dir, args.n_samples, args.seq_len, args.pred_len, args.seed)
        var_indices = [v for v in args.vars if v < n_vars] or list(range(min(3, n_vars)))

        # load backbones once per dataset (shared across transforms)
        backbones = []
        backbone_seq_len = args.seq_len
        for ckpt_path in args.checkpoints:
            print(f"  backbone: {ckpt_path}")
            bb, _, bsl = load_backbone(ckpt_path, n_vars, args.seq_len, device)
            backbones.append(bb)
            backbone_seq_len = bsl  # all checkpoints should share the same seq_len

        # load longer windows for backbone repr if needed
        if backbone_seq_len != args.seq_len:
            print(f"  loading backbone windows: seq_len={backbone_seq_len}")
            bb_samples, _, _, _ = load_windows(
                dataset, args.data_dir, args.n_samples, backbone_seq_len, args.pred_len, args.seed)
        else:
            bb_samples = samples

        for tfm in args.transforms:
            teacher_type, student_type = _TRANSFORM_MODES[tfm]
            teacher_tf = _make_aug({'type': teacher_type, 'crop_ratio': 1.0})
            student_tf = _make_aug({'type': student_type, 'crop_ratio': 1.0})
            sym4_tf    = _make_aug({'type': teacher_type, 'crop_ratio': 1.0}, wavelet_override='sym4')
            db4_tf     = _make_aug({'type': teacher_type, 'crop_ratio': 1.0}, wavelet_override='db4')
            print(f"  transform: {tfm}  teacher={teacher_type}  student={student_type}")

            make_figure(
                dataset_name     = dataset,
                samples          = samples,
                futures          = futures,
                var_indices      = var_indices,
                col_names        = col_names,
                teacher_tf       = teacher_tf,
                student_tf       = student_tf,
                teacher_sym4_tf  = sym4_tf,
                teacher_db4_tf   = db4_tf,
                backbones        = backbones,
                ckpt_labels      = ckpt_labels,
                device           = device,
                out_path         = os.path.join(out_dir, f"aug_views_{dataset}_{tfm}.png"),
                backbone_samples = bb_samples,
            )


if __name__ == '__main__':
    main()
