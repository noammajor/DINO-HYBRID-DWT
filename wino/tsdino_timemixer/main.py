import copy
import numpy as np
from config import config as cfg
import os
import torch
from torch import nn
import random
import torch.nn.functional as F
from tsdino_common import util as utils
from models.ts_mixer_backbone import TSMixerForDINO, TSMixerForecastModel, PatchReconDecoder
import torch.distributed as dist
import time
import datetime
import math
import sys
from pathlib import Path
import json
import data_agumentation as aug
from tsdino_common.dino_head import DINOHead
from tsdino_common import dataPuller as dpuller
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset


class RandomSubsetSampler(torch.utils.data.Sampler):
    """Yield a fresh random `frac` fraction of indices (no replacement) each epoch.

    The DataLoader calls iter(sampler) once per epoch; train_one_epoch calls
    set_epoch(epoch) (see the set_epoch hook in the loop), which re-seeds the RNG
    so a different random subset is drawn every epoch. len() reflects the subset
    size, so len(data_loader) — and thus the LR/momentum/wd schedules — scale down
    automatically.
    """
    def __init__(self, n, frac, seed=0):
        self.n = int(n)
        self.k = max(1, int(frac * self.n))
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        yield from torch.randperm(self.n, generator=g)[:self.k].tolist()

    def __len__(self):
        return self.k


def koleo_loss(x, eps=1e-8):
    """Kozachenko-Leonenko entropic regularizer (DINOv2).

    Spreads features apart by penalizing the (log) nearest-neighbor distance
    within the batch — directly counters representation collapse.
    x: [N, D] feature batch.
    """
    x = F.normalize(x, p=2, dim=-1, eps=eps)
    dots = torch.mm(x, x.t())
    n = x.shape[0]
    dots.view(-1)[::(n + 1)].fill_(-1)           # exclude self-similarity on diagonal
    nn_idx  = dots.max(dim=1).indices            # nearest neighbour (max cosine sim)
    nn_dist = (x - x[nn_idx]).norm(dim=1)
    return -torch.log(nn_dist + eps).mean()


def vicreg_loss(x, std_coeff=1.0, cov_coeff=0.04, eps=1e-4):
    """VICReg variance + covariance regularization on a [N, D] feature batch.

    Variance term keeps each dim's std near 1 (anti-collapse); covariance term
    decorrelates dims. The invariance term is omitted — DINO already aligns the
    student/teacher views.
    """
    x = x - x.mean(dim=0, keepdim=True)
    std      = torch.sqrt(x.var(dim=0) + eps)
    std_term = torch.mean(F.relu(1.0 - std))
    n, d     = x.shape
    cov      = (x.t() @ x) / (n - 1)
    cov_term = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / d
    return std_coeff * std_term + cov_coeff * cov_term


def train_TS_DINO(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    #cudnn.benchmark = True

    #-------------DATA-----------------
    dataAugmentationDino = DataAugmentationDino(
        global_crops = cfg['global_crops'],
        local_crops  = cfg['local_crops'],
        dwt_cfg      = cfg,
    )

    # ── Select pretraining dataset ─────────────────────────────────────────────
    _pretrain_source = cfg.get('pretrain_source') or (
        'monash' if cfg.get('pretrain_on_monash', False) else None
    )
    _shared_kwargs = dict(
        split     = 'train',
        transform = dataAugmentationDino,
        batch_size= args.num_patches,
        patch_size= args.patch_len,
        step_size = args.step_size,
        min_len   = cfg.get('monash_min_len', 512),
    )
    _synth_kwargs = dict(_shared_kwargs, window_step=cfg.get('window_step', None))
    # Pretrain directly on a classification dataset's TRAIN series (labels ignored).
    # Takes priority over every other source; c_in is inferred from the data.
    _cls_pretrain_ds  = cfg.get('pretrain_classification_dataset')
    _anom_pretrain_ds = cfg.get('pretrain_anomaly_dataset')
    if _cls_pretrain_ds:
        print(f"Using classification dataset '{_cls_pretrain_ds}' (train series) for DINO pretraining")
        _shared_dir = str(Path(__file__).parent.parent / "shared")
        if _shared_dir not in sys.path:
            sys.path.insert(0, _shared_dir)
        from data_loaders.data_puller import ClassificationPretrainPuller
        _cls_pre_kwargs = dict(
            data_dir     = cfg['classification_data_dir'],
            dataset_name = _cls_pretrain_ds,
            seq_len      = args.num_patches * args.patch_len,
            patch_size   = args.patch_len,
            transform    = dataAugmentationDino,
            val_fraction = cfg.get('pretrain_val_fraction', 0.0),
            val_min      = cfg.get('pretrain_val_min', 32),
            split_seed   = args.seed,
        )
        combined_dataset = ClassificationPretrainPuller(which='train', **_cls_pre_kwargs)
        args.c_in = combined_dataset.n_vars   # backbone built with the dataset's var count
        print(f"Pretrain dataset: {len(combined_dataset)} series (c_in={args.c_in})")
    elif _anom_pretrain_ds:
        print(f"Using anomaly dataset '{_anom_pretrain_ds}' (train stream) for DINO pretraining")
        _shared_dir = str(Path(__file__).parent.parent / "shared")
        if _shared_dir not in sys.path:
            sys.path.insert(0, _shared_dir)
        from data_loaders.data_puller import AnomalyPretrainPuller
        _anom_pre_kwargs = dict(
            data_dir     = cfg['anomaly_data_dir'],
            dataset      = _anom_pretrain_ds,
            seq_len      = args.num_patches * args.patch_len,
            patch_size   = args.patch_len,
            transform    = dataAugmentationDino,
            # Dense sliding stride for SSL — NOT the coarse forecasting window_step
            # (e.g. 336), which would leave only a handful of 100-ts windows.
            step         = cfg.get('anomaly_pretrain_step', None),
            val_fraction = cfg.get('pretrain_val_fraction', 0.0),
            val_min      = cfg.get('pretrain_val_min', 32),
        )
        combined_dataset = AnomalyPretrainPuller(which='train', **_anom_pre_kwargs)
        args.c_in = combined_dataset.n_vars   # backbone built with the stream's var count
        print(f"Pretrain dataset: {len(combined_dataset)} windows (c_in={args.c_in})")
    elif _pretrain_source in ('monash', 'monash+synthetic'):
        print("Using Monash dataset for DINO pretraining")
        combined_dataset = dpuller.MonashDataPuller(
            data_dir = cfg['monash_data_dir'], **_shared_kwargs)
        if _pretrain_source == 'monash+synthetic':
            print("Using Monash + Synthetic (mix) datasets for DINO pretraining")
            _mix_dir = cfg.get('synthetic_mix_data_dir', cfg['synthetic_data_dir'])
            syn_dataset = dpuller.SyntheticArrowDataPuller(
                data_dir = _mix_dir, **_synth_kwargs)
            combined_dataset = ConcatDataset([combined_dataset, syn_dataset])
    elif _pretrain_source == 'synthetic':
        print("Using Synthetic dataset for DINO pretraining")
        combined_dataset = dpuller.SyntheticArrowDataPuller(
            data_dir = cfg['synthetic_data_dir'], **_synth_kwargs)
    if _cls_pretrain_ds or _anom_pretrain_ds:
        pass   # combined_dataset already built from the classification/anomaly series
    elif _pretrain_source is not None:
        print(f"Pretrain dataset: {len(combined_dataset)} windows")
    elif 'UCI HAR' in args.data_path:
        print("Using UCI HAR Dataset for DINO training")
        combined_dataset = dpuller.DataPullerUCIDINO(
            data_dir=args.data_path,
            split='train',
            transform=dataAugmentationDino,
            batch_size=args.num_patches,
            patch_size=args.patch_len,
            step_size=args.step_size,
            c_in=args.c_in
        )
    else:
        print("Using CSV datasets for DINO training")
        _shared_dir = str(Path(__file__).parent.parent / "shared")
        if _shared_dir not in sys.path:
            sys.path.insert(0, _shared_dir)
        from data_loaders.data_puller import PatchTSTPretrainAdapter
        _seq_len = args.num_patches * args.patch_len
        dataset1 = PatchTSTPretrainAdapter(
            csv_path=args.data_path,
            split='train',
            seq_len=_seq_len,
            patch_size=args.patch_len,
            transform=dataAugmentationDino,
        )
        if args.data_path_forecast_training != args.data_path:
            dataset2 = PatchTSTPretrainAdapter(
                csv_path=args.data_path_forecast_training,
                split='train',
                seq_len=_seq_len,
                patch_size=args.patch_len,
                transform=dataAugmentationDino,
            )
            combined_dataset = ConcatDataset([dataset1, dataset2])
        else:
            combined_dataset = dataset1

    # ── val dataset (same source, split='val') ────────────────────────────────
    _val_kwargs = dict(_shared_kwargs, split='val')
    if _cls_pretrain_ds:
        # Optional small held-out val carved from the train series (empty for
        # datasets too small to spare one → final epoch saved as checkpoint_best).
        val_dataset = ClassificationPretrainPuller(which='val', **_cls_pre_kwargs)
    elif _anom_pretrain_ds:
        # Held-out val carved from the TAIL of the anomaly train stream.
        val_dataset = AnomalyPretrainPuller(which='val', **_anom_pre_kwargs)
    elif _pretrain_source in ('monash', 'monash+synthetic'):
        val_dataset = dpuller.MonashDataPuller(data_dir=cfg['monash_data_dir'], **_val_kwargs)
        if _pretrain_source == 'monash+synthetic':
            _val_synth_kwargs = dict(_val_kwargs, window_step=cfg.get('window_step', None))
            syn_val = dpuller.SyntheticArrowDataPuller(data_dir=cfg['synthetic_data_dir'], **_val_synth_kwargs)
            val_dataset = ConcatDataset([val_dataset, syn_val])
    elif _pretrain_source == 'synthetic':
        _val_synth_kwargs = dict(_val_kwargs, window_step=cfg.get('window_step', None))
        val_dataset = dpuller.SyntheticArrowDataPuller(data_dir=cfg['synthetic_data_dir'], **_val_synth_kwargs)
    elif 'UCI HAR' in args.data_path:
        val_dataset = None  # UCI HAR — no val split
    else:
        # CSV in-domain: use val split so checkpoint_best.pth gets saved
        val_dataset = PatchTSTPretrainAdapter(
            csv_path  = args.data_path,
            split     = 'val',
            seq_len   = args.num_patches * args.patch_len,
            patch_size= args.patch_len,
            transform = dataAugmentationDino,
        )

    _is_distributed = utils.is_dist_avail_and_initialized()
    # Optional: train on a fresh random fraction of the windows each epoch.
    _subset_frac = cfg.get('pretrain_subset_frac', 1.0) or 1.0
    if _subset_frac < 1.0 and not _is_distributed:
        _train_sampler = RandomSubsetSampler(len(combined_dataset), _subset_frac,
                                             seed=args.seed)
        print(f"[subset] each epoch draws {_subset_frac:.2%} of "
              f"{len(combined_dataset)} windows "
              f"({len(_train_sampler)} per epoch, re-randomized each epoch)")
    elif _is_distributed:
        if _subset_frac < 1.0:
            print("[subset] pretrain_subset_frac ignored under distributed training")
        _train_sampler = torch.utils.data.distributed.DistributedSampler(combined_dataset, shuffle=True)
    else:
        _train_sampler = torch.utils.data.RandomSampler(combined_dataset)
    data_loader = torch.utils.data.DataLoader(
        combined_dataset,
        sampler=_train_sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    ) if val_dataset is not None and len(val_dataset) > 0 else None



    #------------- Student - Teacher network (TimeMixer backbone) -----------
    # TSMixer uses raw [B, T, C] — seq_len is the full window, no patching.
    _seq_len = args.num_patches * args.patch_len
    _tm_kwargs = dict(
        c_in=args.c_in,
        seq_len=_seq_len,
        d_model=cfg.get('tsmixer_d_model', 16),
        e_layers=cfg.get('tsmixer_e_layers', 2),
        d_ff=cfg.get('tsmixer_d_ff', 32),
        dropout=args.dropout,
        patch_len=args.patch_len,
        down_sampling_layers=cfg.get('tsmixer_down_sampling_layers', 3),
        down_sampling_window=cfg.get('tsmixer_down_sampling_window', 2),
        down_sampling_method=cfg.get('tsmixer_down_sampling_method', 'avg'),
        decomp_method=cfg.get('tsmixer_decomp_method', 'moving_avg'),
        moving_avg=cfg.get('tsmixer_moving_avg', 25),
        top_k=cfg.get('tsmixer_top_k', 5),
        use_norm=cfg.get('tsmixer_use_norm', 1),
        channel_independence=cfg.get('tsmixer_channel_independence', 1),
    )
    student = TSMixerForDINO(**_tm_kwargs)
    teacher = TSMixerForDINO(**_tm_kwargs)
    embed_dim = student.d_model
    student = utils.TSMultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.TSMultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head, norm_last_layer=args.norm_last_layer),
    )
    # move networks to gpu
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    student, teacher = student.to(device), teacher.to(device)
    # synchronize batch norms (if any)
    device_ids = [args.gpu] if torch.cuda.is_available() else None
    if utils.has_batchnorms(student):
        # Around line 190 in train_TS_DINO
        if torch.cuda.is_available():
            student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
            teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        else:
            print("Skipping SyncBatchNorm on Mac/CPU—using standard BatchNorm instead.")
        if getattr(args, 'distributed', False):
            teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=device_ids, find_unused_parameters=True)
            teacher_without_ddp = teacher.module
        else:
            teacher_without_ddp = teacher
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    if getattr(args, 'distributed', False):
        student = nn.parallel.DistributedDataParallel(student, device_ids=device_ids, find_unused_parameters=True)
        student_without_ddp = student.module
    else:
        student_without_ddp = student
    teacher_without_ddp.load_state_dict(student_without_ddp.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both TS networks.")

    # ── Reconstruction decoders (MAE-style auxiliary loss) ────────────────────
    use_reconstruction = cfg.get('use_reconstruction', False)
    student_recon = None
    teacher_recon_decoder = None
    if use_reconstruction:
        # TimeMixer: each token is a single timestep, so predict 1 value.
        _recon_out = 1
        student_recon = PatchReconDecoder(embed_dim, _recon_out).to(device)
        teacher_recon_decoder = copy.deepcopy(student_recon)
        for p in teacher_recon_decoder.parameters():
            p.requires_grad = False
        print("Reconstruction decoders created.")

    # ── iBOT heads (patch-level, teacher-guided cross-entropy) ─────────────────
    mlm_phi        = cfg.get("mlm_phi", 0.0)
    mlm_mask_ratio = cfg.get("mlm_mask_ratio", 0.4)
    mlm_mode       = cfg.get("mlm_mode", "ibot")  # "ibot" or "mae"
    ibot_out_dim   = cfg.get("ibot_out_dim", args.out_dim)
    use_mlm        = mlm_phi > 0.0
    student_ibot_head = None
    teacher_ibot_head = None
    ibot_center       = None
    student_mae_head  = None
    if use_mlm and mlm_mode == "ibot":
        student_ibot_head = DINOHead(embed_dim, ibot_out_dim,
                                     use_bn=args.use_bn_in_head,
                                     norm_last_layer=args.norm_last_layer).to(device)
        teacher_ibot_head = DINOHead(embed_dim, ibot_out_dim,
                                     use_bn=args.use_bn_in_head,
                                     norm_last_layer=False).to(device)
        teacher_ibot_head.load_state_dict(student_ibot_head.state_dict())
        for p in teacher_ibot_head.parameters():
            p.requires_grad = False
        ibot_center = torch.zeros(1, ibot_out_dim, device=device)
        print(f"[DINO+iBOT] phi={mlm_phi}  mask_ratio={mlm_mask_ratio}")
    elif use_mlm and mlm_mode == "mae":
        # TimeMixer: predict 1 raw value per timestep token.
        _mae_out = 1
        student_mae_head = PatchReconDecoder(embed_dim, _mae_out).to(device)
        print(f"[DINO+MAE]  phi={mlm_phi}  mask_ratio={mlm_mask_ratio}")

#-----------------Loss function --------------------
    dino_loss = DINOLoss(
        args.out_dim,
        len(cfg['global_crops']) + len(cfg['local_crops']),
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).to(device)
# ----------------Optimizer --------------------
    params_groups = utils.get_params_groups(student)
    if use_reconstruction:
        # Add reconstruction decoder params (no weight decay on bias/norm, but simpler: just add all)
        params_groups.append({'params': student_recon.parameters()})
    if use_mlm and student_ibot_head is not None:
        params_groups.append({'params': student_ibot_head.parameters()})
    if use_mlm and student_mae_head is not None:
        params_groups.append({'params': student_mae_head.parameters()})
    if os.environ.get("TS_PRETRAIN_OPT", "").lower() == "prodigy":
        args.optimizer = "prodigy"   # propagates to train_one_epoch (skips manual LR schedule)
    if args.optimizer == "prodigy":
        try:
            from prodigyopt import Prodigy
            _dcoef_pre = float(os.environ.get("TS_PRODIGY_DCOEF", "1.0"))
            optimizer = Prodigy(params_groups, lr=1.0, d_coef=_dcoef_pre, weight_decay=0,
                                safeguard_warmup=True, use_bias_correction=True,
                                decouple=True)
            print(f"[DINO pretrain] optimizer: Prodigy (lr-free, d_coef={_dcoef_pre})")
        except ImportError:
            print("[DINO pretrain] prodigyopt not installed (`pip install prodigyopt`) "
                  "— falling back to AdamW.")
            args.optimizer = "adamw"
            optimizer = torch.optim.AdamW(params_groups)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
       # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
#-----------------Scheduler --------------------
    lr_schedule = utils.cosine_scheduler(
        args.lr* (args.batch_size_per_gpu * utils.get_world_size()) / 256.,
        args.min_lr,
        args.epochs,
        len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        len(data_loader),
    )
    momentum_schedule = utils.cosine_scheduler(
        args.momentum_teacher,
        1,
        args.epochs,
        len(data_loader),
    )
#----------------Train Loop --------------------
    start_epoch = 0
    best_val_loss = float('inf')

    start_time = time.time()
    print("Starting TS - DINO training !")

    for epoch in range(start_epoch, args.epochs):
        print(f'Starting epoch {epoch}/{args.epochs}')
        if hasattr(data_loader.sampler, 'set_epoch'):
            data_loader.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            student,
            teacher,
            teacher_without_ddp,
            dino_loss,
            data_loader,
            optimizer,
            epoch,
            fp16_scaler,
            lr_schedule,
            wd_schedule,
            momentum_schedule,
            args,
            student_recon=student_recon,
            teacher_recon_decoder=teacher_recon_decoder,
            use_mlm=use_mlm,
            mlm_mode=mlm_mode,
            mlm_phi=mlm_phi,
            mlm_mask_ratio=mlm_mask_ratio,
            student_ibot_head=student_ibot_head,
            teacher_ibot_head=teacher_ibot_head,
            ibot_center=ibot_center,
            student_mae_head=student_mae_head,
        )
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if use_reconstruction and student_recon is not None:
            save_dict['student_recon'] = student_recon.state_dict()
            save_dict['teacher_recon_decoder'] = teacher_recon_decoder.state_dict()
        #utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch}.pth'))

        # ── validation + best model ───────────────────────────────────────────
        if val_loader is not None and utils.is_main_process():
            student.eval()
            val_loss_sum = 0.0
            val_n = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch = [s.to(device, non_blocking=True) for s in val_batch]
                    _n_global = len(cfg['global_crops'])
                    _n_dino   = _n_global + len(cfg['local_crops'])
                    dino_val = val_batch[:_n_dino]
                    _t_in = dino_val if cfg.get('view_pairing', 'default') == 'symmetric' \
                            else dino_val[:_n_global]
                    teacher_out = teacher(_t_in)
                    student_out = student(dino_val)
                    v_loss = dino_loss(student_out, teacher_out, epoch)
                    val_loss_sum += v_loss.item() * val_batch[0].size(0)
                    val_n += val_batch[0].size(0)
            val_loss = val_loss_sum / max(val_n, 1)
            print(f"Epoch {epoch} — val loss: {val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                print(f"  → New best val loss: {best_val_loss:.6f} — saved checkpoint_best.pth")
            student.train()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    # When pretraining without a val split (e.g. classification train-only), no
    # checkpoint_best.pth is ever written above. Persist the final epoch under that
    # name so downstream probe/fine-tune can load it.
    if val_loader is None and args.epochs > 0 and utils.is_main_process():
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint_best.pth'))
        print("  → No val set — saved final epoch as checkpoint_best.pth")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader, optimizer, epoch, fp16_scaler, lr_schedule, wd_schedule, momentum_schedule, args, student_recon=None, teacher_recon_decoder=None, use_mlm=False, mlm_mode="ibot", mlm_phi=0.0, mlm_mask_ratio=0.4, student_ibot_head=None, teacher_ibot_head=None, ibot_center=None, student_mae_head=None):
    student_without_ddp = student.module if hasattr(student, 'module') else student
    student.train()
    teacher.eval()   # teacher must be deterministic — dropout off (no BN in TSMixer)
    if student_recon is not None:
        student_recon.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('weight_decay', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    n_global = len(cfg['global_crops'])
    n_dino   = n_global + len(cfg['local_crops'])
    recon_loss_weight = cfg.get('recon_loss_weight', 0.5)
    for it, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        samples = batch
        samples = [s.to(device, non_blocking=True) for s in samples]
        # Separate DINO crops from optional reconstruction pair
        dino_samples   = samples[:n_dino]
        use_recon_this = (student_recon is not None) and (len(samples) == n_dino + 2)
        # update learning rate and weight decay according to their schedule
        it_global = it + epoch * len(data_loader)
        for i, param_group in enumerate(optimizer.param_groups):
            if getattr(args, "optimizer", "adamw") != "prodigy":   # Prodigy adapts its own LR
                param_group['lr'] = lr_schedule[it_global]
            if i == 0:  # only the first group is regularized
                param_group['weight_decay'] = wd_schedule[it_global]
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        metric_logger.update(weight_decay=optimizer.param_groups[0]['weight_decay'])
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # 'symmetric' view-pairing: teacher processes ALL views, not just easy ones.
            _teacher_in = dino_samples if cfg.get('view_pairing', 'default') == 'symmetric' \
                          else dino_samples[:n_global]
            teacher_output = teacher(_teacher_in)
            student_output = student(dino_samples)
            loss = dino_loss(student_output, teacher_output, epoch)
            _dino_loss_val = loss.item()

            # ── Reconstruction loss (MAE-style, original data only) ────────────
            if use_recon_this:
                full_orig   = samples[n_dino]      # [B, seq_len, n_vars]  → teacher input
                masked_orig = samples[n_dino + 1]  # [B, seq_len, n_vars]  → student input

                # Teacher encodes full original (no grad — teacher already frozen)
                with torch.no_grad():
                    t_tokens = teacher_without_ddp.backbone.forward_recon(full_orig)
                    # [B, num_patch, n_vars, d_model]

                # Student encodes masked original
                s_tokens = student_without_ddp.backbone.forward_recon(masked_orig)
                # [B, num_patch, n_vars, d_model]

                B, num_patch, n_vars, d_model = t_tokens.shape
                # Reshape → [B * n_vars, num_patch, d_model] for decoder
                t_flat = t_tokens.permute(0, 2, 1, 3).reshape(B * n_vars, num_patch, d_model)
                s_flat = s_tokens.permute(0, 2, 1, 3).reshape(B * n_vars, num_patch, d_model)

                t_recon = teacher_recon_decoder(t_flat)  # [B*n_vars, num_patch, patch_len]
                s_recon = student_recon(s_flat)           # [B*n_vars, num_patch, patch_len]

                recon_loss = F.mse_loss(s_recon, t_recon.detach())
                loss = loss + recon_loss_weight * recon_loss
                metric_logger.update(recon_loss=recon_loss.item())

            # ── MLM auxiliary loss (iBOT or MAE) ──────────────────────────────
            if use_mlm:
                _mlm_in    = dino_samples[0]   # [B, T, n_vars] first global crop
                _B, _T, _C = _mlm_in.shape
                # TimeMixer: tokens are timestep-level (num_tokens=T).
                _bbone = student_without_ddp.backbone
                _NP    = _bbone.num_tokens
                # Block masking: mask contiguous spans of `mlm_block_size` timesteps
                # (default 8) rather than independent steps, so the model can't
                # trivially interpolate a masked value from its visible neighbours.
                # mlm_block_size=1 recovers per-step masking.
                _blk      = max(1, int(cfg.get('mlm_block_size', 8)))
                _n_blocks = (_NP + _blk - 1) // _blk
                _pmask    = (torch.rand(_B, _n_blocks, device=device) < mlm_mask_ratio)
                _pmask    = _pmask.repeat_interleave(_blk, dim=1)[:, :_NP]   # [B, _NP]
                # Student always uses mask_token at masked positions
                _s_tok = student_without_ddp.backbone.forward_ibot(_mlm_in, _pmask)
                # [B, NP, n_vars, d_model]
                _B2, _NP2, _NV, _DM = _s_tok.shape
                # Row order must match permute(0,2,1,3) below (B, n_vars, NP) and the
                # masking the backbone applies internally via
                # mask.unsqueeze(1).expand(-1, n_vars, -1). Using unsqueeze(2) here
                # scrambles masked positions relative to the tokens.
                _mexp = _pmask.unsqueeze(1).expand(-1, _NV, -1).reshape(_B2*_NV, _NP2)
                _s_all = _s_tok.permute(0,2,1,3).reshape(_B2*_NV, _NP2, _DM)
                _s_masked = _s_all[_mexp]   # [N_masked, d_model]
                if _s_masked.shape[0] > 0:
                    if mlm_mode == "ibot":
                        with torch.no_grad():
                            _t_tok = teacher_without_ddp.backbone.forward_ibot(_mlm_in, None)
                        _t_all    = _t_tok.permute(0,2,1,3).reshape(_B2*_NV, _NP2, _DM)
                        _t_masked = _t_all[_mexp]
                        _s_ibot = student_ibot_head(_s_masked)
                        _t_ibot = teacher_ibot_head(_t_masked)
                        _ttemp  = dino_loss.teacher_temp_schedule[
                                      min(epoch, len(dino_loss.teacher_temp_schedule)-1)]
                        _t_soft = F.softmax((_t_ibot - ibot_center) / _ttemp, dim=-1).detach()
                        _s_log  = F.log_softmax(_s_ibot / 0.1, dim=-1)
                        _mlm_loss = -(_t_soft * _s_log).sum(dim=-1).mean()
                        with torch.no_grad():
                            ibot_center.mul_(0.9).add_(_t_ibot.mean(0, keepdim=True) * 0.1)
                        metric_logger.update(ibot_loss=_mlm_loss.item())
                    else:  # mae
                        # TimeMixer: each token predicts its own raw normalized value.
                        _gt_all    = _mlm_in.permute(0, 2, 1).reshape(_B2*_NV, _NP2)
                        _gt_masked = _gt_all[_mexp]          # [N_masked]
                        _pred      = student_mae_head(_s_masked).squeeze(-1)  # [N_masked]
                        _mlm_loss  = F.mse_loss(_pred, _gt_masked)
                        metric_logger.update(mae_loss=_mlm_loss.item())
                    loss = mlm_phi * loss + (1.0 - mlm_phi) * _mlm_loss
                    print(f'DINO: {_dino_loss_val:.4f}  MLM: {_mlm_loss.item():.4f}  combined: {loss.item():.4f}')
                else:
                    print(f'DINO: {_dino_loss_val:.4f}  MLM: 0.0000  combined: {loss.item():.4f}')
            else:
                print(f'DINO: {_dino_loss_val:.4f}')

            # ── Anti-collapse regularizers on the global student feature ──────
            # Optional (cfg toggles, default off → skipped). Re-encodes the
            # global crop(s) through the backbone to get the pre-head embedding,
            # then spreads it out. Works for dino / dino+mae / dino+ibot.
            if cfg.get('use_koleo', False) or cfg.get('use_vicreg', False):
                _bb = student_without_ddp.backbone
                _kl_sum = _vc_sum = 0.0
                for _gi in range(n_global):
                    _gf = _bb(dino_samples[_gi])                 # [B, C, d_model]
                    _gf = _gf.reshape(_gf.shape[0], -1)          # [B, C*d_model]
                    if cfg.get('use_koleo', False):
                        _kl_sum = _kl_sum + koleo_loss(_gf)
                    if cfg.get('use_vicreg', False):
                        _vc_sum = _vc_sum + vicreg_loss(
                            _gf, cfg.get('vicreg_std_coeff', 1.0),
                            cfg.get('vicreg_cov_coeff', 0.04))
                if cfg.get('use_koleo', False):
                    _kl = _kl_sum / n_global
                    _kw = cfg.get('koleo_weight', 0.1)
                    loss = loss + _kw * _kl
                    metric_logger.update(koleo_loss=_kl.item())
                    print(f'KoLeo: {_kl.item():.4f}  (w={_kw})  -> +{(_kw*_kl).item():.4f}')
                if cfg.get('use_vicreg', False):
                    _vc = _vc_sum / n_global
                    loss = loss + _vc
                    metric_logger.update(vicreg_loss=_vc.item())
                    print(f'VICReg: {_vc.item():.4f}  (std={cfg.get("vicreg_std_coeff",1.0)} cov={cfg.get("vicreg_cov_coeff",0.04)})')

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)
        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                            args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it_global]  # momentum parameter
            for param_q, param_k in zip(student_without_ddp.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            # EMA update for the reconstruction teacher decoder
            if use_recon_this:
                for param_q, param_k in zip(student_recon.parameters(), teacher_recon_decoder.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            if use_mlm and student_ibot_head is not None:
                for param_q, param_k in zip(student_ibot_head.parameters(), teacher_ibot_head.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        if device.type == 'cuda':
            torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        if use_mlm:
            metric_logger.update(dino_loss=_dino_loss_val)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)

        # View-pairing mode (ablation). Crop layout in student_out is
        # [easy_0 .. easy_{n_global-1}, hard_0 .. hard_{n_local-1}].
        #   default       — teacher=easy views, loss on all cross-view pairs (skip identical view).
        #   hard_student  — loss only on teacher-easy → student-HARD pairs (student-easy excluded).
        #   same_view     — default + the identical-view (easy↔easy) pairs (no skip).
        #   symmetric     — teacher processes ALL views; loss over all cross-view pairs.
        mode = cfg.get('view_pairing', 'default')
        n_global = len(cfg['global_crops'])
        n_teacher = self.ncrops if mode == 'symmetric' else n_global
        teacher_out = teacher_out.detach().chunk(n_teacher)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if mode == 'same_view':
                    pass                              # include every pair, even v == iq
                elif mode == 'hard_student':
                    if v < n_global:                  # only student HARD views contribute
                        continue
                else:                                 # default, symmetric
                    if v == iq:                       # skip identical view
                        continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            world_size = dist.get_world_size()
        else:
            world_size = 1
        batch_center = batch_center / (len(teacher_output) * world_size)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
#------Data Augmentation for Time Series DINO -------
class DataAugmentationDino:
    """
    Config-driven augmentation.  Reads global_crops / local_crops from config.py.
    Each crop spec dict maps directly to a DWTAugmentation instance;
    if 'type' is a list, one type is drawn at random per sample.
    """

    def __init__(self, global_crops, local_crops, dwt_cfg):
        self.global_crops       = global_crops
        self.local_crops        = local_crops
        self.use_reconstruction = dwt_cfg.get('use_reconstruction', False)
        self.recon_mask_ratio   = dwt_cfg.get('recon_mask_ratio',   0.4)
        self.patch_len          = dwt_cfg.get('patch_len',          16)
        # Wavelet-basis sampling mode (ablation):
        #   'independent' (default) — every view draws its own basis from the pool.
        #   'shared'      — one basis drawn per sample, shared by all views.
        #   'fixed'       — always use the single fixed `dwt_wavelet` (pool ignored).
        self.wavelet_sampling_mode = dwt_cfg.get('wavelet_sampling_mode', 'independent')
        self._wavelet_pool         = dwt_cfg.get('dwt_wavelet_pool', None)
        self._fixed_wavelet        = dwt_cfg.get('dwt_wavelet')
        # Pre-build one DWTAugmentation per (crop_index, aug_type) pair
        self._transforms = self._build_transforms(global_crops + local_crops, dwt_cfg)

    # Registry of non-DWT transform names → classes in data_agumentation.py
    _NON_DWT_REGISTRY = {
        'polar':            aug.polar_transformation,
        'galilien':         aug.galilien_transformation,
        'rotation':         aug.rotation_transformation,
        'boost':            aug.boost_transformation,
        'lorentz':          aug.lorentz_transformation,
        'hyperbolic_warp':  aug.hyperbolic_amplitude_warp,
        'hyperbolic_geom':  aug.HyperBolicGeometry,
        # vision-style (1D adaptations)
        'gaussian_blur':    aug.gaussian_blur,
        'gaussian_noise':   aug.gaussian_noise,
        'gaussiancrop':     aug.gaussiancrop,   # crop (via crop_ratio) + gaussian noise
        'jitter_contrast':  aug.jitter_contrast,
    }

    # 'soft' and 'hard' are convenience aliases for the most common teacher/student modes
    _MODE_ALIASES = {'soft': 'soft_threshold', 'hard': 'high_perturb'}

    def _build_transforms(self, all_specs, dwt_cfg):
        transforms = []
        for spec in all_specs:
            types = spec['type'] if isinstance(spec['type'], list) else [spec['type']]
            per_type = {}
            for t in types:
                _wavelet_pool = spec.get('wavelet_pool', dwt_cfg.get('dwt_wavelet_pool', None))
                _wavelet      = spec.get('wavelet',      dwt_cfg['dwt_wavelet'])
                _shared = dict(
                    wavelet                  = _wavelet,
                    wavelet_pool             = _wavelet_pool,
                    level                    = spec.get('level',                      dwt_cfg['dwt_level']),
                    soft_threshold_sigma     = spec.get('soft_threshold_sigma',       dwt_cfg.get('dwt_soft_threshold_sigma', 0.3)),
                    zero_out_ratio           = spec.get('zero_out_ratio',             dwt_cfg.get('dwt_zero_out_ratio', 0.3)),
                    finest_levels            = spec.get('finest_levels',              dwt_cfg.get('dwt_finest_levels', 1)),
                    high_perturb_noise_range = spec.get('high_perturb_noise_range',   dwt_cfg.get('dwt_high_perturb_noise_range', (0.03, 0.08))),
                    band_scale_approx_range  = dwt_cfg.get('dwt_band_scale_approx_range', (0.9, 1.1)),
                    band_scale_detail_range  = dwt_cfg.get('dwt_band_scale_detail_range', (0.6, 1.4)),
                )
                if t.startswith('dwt_'):
                    raw_mode = t[4:]   # strip leading 'dwt_'
                    mode = self._MODE_ALIASES.get(raw_mode, raw_mode)
                    per_type[t] = aug.DWTAugmentation(mode=mode, **_shared)
                elif t.startswith('swt_'):
                    raw_mode = t[4:]   # strip leading 'swt_'
                    mode = self._MODE_ALIASES.get(raw_mode, raw_mode)
                    per_type[t] = aug.SWTAugmentation(mode=mode, **_shared)
                elif t.startswith('modwt_'):
                    raw_mode = t[6:]   # strip leading 'modwt_'
                    mode = self._MODE_ALIASES.get(raw_mode, raw_mode)
                    per_type[t] = aug.MODWTAugmentation(mode=mode, **_shared)
                elif t in self._NON_DWT_REGISTRY:
                    cls = self._NON_DWT_REGISTRY[t]
                    # Each class reads its params from the spec first, then falls back to cfg defaults
                    kwargs = {}
                    if t == 'lorentz':
                        kwargs['v_range']        = spec.get('v_range',        dwt_cfg.get('lorentz_v_range',        (0.2, 0.6)))
                    elif t == 'polar':
                        kwargs['warp_range']     = spec.get('warp_range',     dwt_cfg.get('polar_warp_range',       (0.7, 1.3)))
                    elif t == 'galilien':
                        kwargs['a_range']        = spec.get('a_range',        dwt_cfg.get('galilien_a_range',       (0.8, 1.2)))
                    elif t == 'rotation':
                        kwargs['angle_range']    = spec.get('angle_range',    dwt_cfg.get('rotation_angle_range',   (0, 0.3927)))
                    elif t == 'boost':
                        kwargs['b_range']        = spec.get('b_range',        dwt_cfg.get('boost_b_range',          (0.01, 0.3)))
                    elif t == 'hyperbolic_warp':
                        kwargs['warp_range']     = spec.get('warp_range',     dwt_cfg.get('hyperbolic_warp_range',  (0.5, 1.5)))
                    elif t == 'hyperbolic_geom':
                        kwargs['shift_magnitude']= spec.get('shift_magnitude',dwt_cfg.get('hyperbolic_shift_magnitude', 0.3))
                    elif t == 'gaussian_blur':
                        kwargs['sigma_range']    = spec.get('sigma_range',    dwt_cfg.get('gaussian_blur_sigma_range', (0.1, 2.0)))
                    elif t in ('gaussian_noise', 'gaussiancrop'):
                        kwargs['std_range']      = spec.get('std_range',      dwt_cfg.get('gaussian_noise_std_range', (0.05, 0.2)))
                    elif t == 'jitter_contrast':
                        kwargs['jitter_range']     = spec.get('jitter_range',     dwt_cfg.get('jitter_range',     (0.0, 0.1)))
                        kwargs['contrast_range']   = spec.get('contrast_range',   dwt_cfg.get('contrast_range',   (0.7, 1.3)))
                        kwargs['brightness_range'] = spec.get('brightness_range', dwt_cfg.get('brightness_range', (-0.2, 0.2)))
                    per_type[t] = cls(**kwargs)
                else:
                    raise ValueError(f"Unknown augmentation type '{t}'. "
                                     f"DWT types must start with 'dwt_', SWT with 'swt_', MODWT with 'modwt_'. "
                                     f"Non-DWT types: {list(self._NON_DWT_REGISTRY)}")
            transforms.append(per_type)
        return transforms

    def _random_crop(self, x, crop_ratio):
        # RandomResizedCrop analog (image-DINO): take a sub-window then resize it
        # back to the original length via linear interpolation, so fixed-seq_len
        # backbones (e.g. TimeMixer) can process the cropped view.
        timesteps = x.shape[0]
        crop_len  = int(timesteps * crop_ratio)
        if crop_len >= timesteps or crop_len < 2:
            return x
        start   = np.random.randint(0, timesteps - crop_len + 1)
        cropped = x[start : start + crop_len, :]                 # [crop_len, n_vars]
        resized = F.interpolate(
            cropped.transpose(0, 1).unsqueeze(0),               # [1, n_vars, crop_len]
            size=timesteps, mode='linear', align_corners=False,
        )
        return resized.squeeze(0).transpose(0, 1)                # [timesteps, n_vars]

    def _mask_patches(self, x):
        """Randomly zero out recon_mask_ratio fraction of non-overlapping patches.
        x: [seq_len, n_vars] — the ORIGINAL (unaugmented) input.
        Returns masked copy of same shape.
        """
        n_patches = x.shape[0] // self.patch_len
        masked    = x.clone()
        for p in range(n_patches):
            if random.random() < self.recon_mask_ratio:
                masked[p * self.patch_len : (p + 1) * self.patch_len] = 0.0
        return masked

    def __call__(self, x):
        # ── DINO contrastive views (DWT augmented) ────────────────────────────
        crops     = []
        all_specs = self.global_crops + self.local_crops
        # Resolve the per-sample forced basis once, so 'shared' uses ONE draw for
        # all views and 'fixed' always uses the single configured wavelet.
        forced_wavelet = None
        if self.wavelet_sampling_mode == 'fixed':
            forced_wavelet = self._fixed_wavelet
        elif self.wavelet_sampling_mode == 'shared':
            forced_wavelet = (random.choice(self._wavelet_pool)
                              if self._wavelet_pool else self._fixed_wavelet)
        for i, spec in enumerate(all_specs):
            crop_ratio = spec.get('crop_ratio', 1.0)
            x_in       = self._random_crop(x, crop_ratio) if crop_ratio < 1.0 else x
            aug_type   = spec['type']
            if isinstance(aug_type, list):
                aug_type = random.choice(aug_type)
            transform = self._transforms[i][aug_type]
            # None → 'independent' (transform draws its own basis from the pool).
            setattr(transform, '_forced_wavelet', forced_wavelet)
            crops.append(transform(x_in))

        # ── Reconstruction pair (original data only, no DWT) ──────────────────
        if self.use_reconstruction:
            crops.append(x.clone())          # full original  → teacher recon
            crops.append(self._mask_patches(x))  # masked original → student recon

        return crops

def _lr_find_forecast(model, criterion, loader, device,
                      start_lr=1e-7, end_lr=1.0, num_iter=100, weight_decay=1e-4):
    """torch-lr-finder range test on the trainable (forecast) params; returns the
    suggested LR (steepest-descent point). Opt-in via env TS_FORECAST_LR_FIND=1.
    Uses a fresh throwaway optimizer so it won't clash with the real OneCycle
    scheduler; reset() restores model weights before returning."""
    import numpy as _np
    try:
        from torch_lr_finder import LRFinder
    except ImportError:
        print("  [DINO forecast][lr-finder] torch-lr-finder not installed "
              "(`pip install torch-lr-finder`) — skipping, using configured LR.")
        return None
    trainable = [p for p in model.parameters() if p.requires_grad]
    tmp_opt = torch.optim.Adam(trainable, lr=start_lr, weight_decay=weight_decay)
    finder = LRFinder(model, tmp_opt, criterion, device=device)
    finder.range_test(loader, start_lr=start_lr, end_lr=end_lr, num_iter=num_iter)
    lrs    = finder.history["lr"]
    losses = finder.history["loss"]
    finder.reset()                      # restore model weights/state
    # Robust pick for a (near-convex) linear probe: LR at the loss minimum backed
    # off by a divisor — lands in the steep-descent zone. The steepest-gradient
    # heuristic mis-fires here (picks a uselessly tiny LR → undertraining).
    losses  = _np.array(losses)
    div     = float(os.environ.get("TS_LR_FIND_DIV", 10.0))
    min_idx = int(_np.argmin(losses))
    lr_min  = float(lrs[min_idx])
    best    = lr_min / div
    print(f"  [DINO forecast][lr-finder] min-loss lr={lr_min:.2e} → "
          f"suggested lr ≈ {best:.3e} (÷{div:g}; scanned {len(lrs)} pts, "
          f"{lrs[0]:.1e}→{lrs[-1]:.1e})")
    return best


#----Test run -----
def test_run(args):
    utils.init_distributed_mode(args)

    # ── PatchTST-identical data loading ──────────────────────────────────────
    _SEQ_LEN = args.num_patches * args.patch_len   # respect context length from pretraining
    _patchtst_dir = str(Path(__file__).parent.parent.parent / "models" / "PatchTST_self_supervised")
    if _patchtst_dir not in sys.path:
        sys.path.insert(0, _patchtst_dir)
    from src.data.pred_dataset import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom

    _csv_path = args.data_path_forecast_training
    _root = os.path.dirname(os.path.abspath(_csv_path))
    _fname = os.path.basename(_csv_path)
    _size = [_SEQ_LEN, 0, args.pred_len]
    if 'etth' in _fname.lower():
        _DS_CLS = Dataset_ETT_hour
    elif 'ettm' in _fname.lower():
        _DS_CLS = Dataset_ETT_minute
    else:
        _DS_CLS = Dataset_Custom

    dataset_forecasting_train = _DS_CLS(_root, split='train', size=_size, features='M', data_path=_fname, scale=True)
    try:
        dataset_forecasting_val = _DS_CLS(_root, split='val', size=_size, features='M', data_path=_fname, scale=True)
    except Exception:
        dataset_forecasting_val = None
    # ── zero-shot cross-domain: train/val head on the source dataset, but TEST
    # on a different (target) CSV with no further training. Set via env var
    # TS_FORECAST_TEST_CSV=<abs path to target csv>. Default = in-domain test.
    _test_csv = os.environ.get("TS_FORECAST_TEST_CSV", _csv_path)
    if _test_csv != _csv_path:
        _test_root  = os.path.dirname(os.path.abspath(_test_csv))
        _test_fname = os.path.basename(_test_csv)
        if 'etth' in _test_fname.lower():
            _DS_CLS_TEST = Dataset_ETT_hour
        elif 'ettm' in _test_fname.lower():
            _DS_CLS_TEST = Dataset_ETT_minute
        else:
            _DS_CLS_TEST = Dataset_Custom
        print(f"  [DINO forecast] ZERO-SHOT cross-domain: head trained on {_fname}, "
              f"TESTED on {_test_fname} (no target training)")
    else:
        _test_root, _test_fname, _DS_CLS_TEST = _root, _fname, _DS_CLS
    dataset_forecasting_test  = _DS_CLS_TEST(_test_root, split='test',  size=_size, features='M', data_path=_test_fname, scale=True)

    _is_distributed = utils.is_dist_avail_and_initialized()
    # TS_FORECAST_BS overrides the forecast batch size (args.batch_size_forecast comes
    # from config and is NOT affected by the pretrain --batch_size flag). Needed for
    # very high-channel datasets (e.g. traffic, 862ch): the channel-independent
    # backbone processes batch*channels sequences, and TimeMixer's decomposition
    # avg_pool overflows int32 when batch*channels*d_model*seq exceeds ~2.1B.
    _fc_bs_env = os.environ.get("TS_FORECAST_BS")
    if _fc_bs_env:
        args.batch_size_forecast = int(_fc_bs_env)
        print(f"  [DINO forecast] batch_size_forecast override -> {args.batch_size_forecast}")
    # TS_FORECAST_DROP_LAST=1 drops the last incomplete batch on train/val/test,
    # matching plain TimeMixer's forecast loaders (drop_last=True) for a fair
    # head-to-head comparison. Default 0 = score every window.
    _fc_drop_last = os.environ.get("TS_FORECAST_DROP_LAST", "0") == "1"
    data_loader_forecasting_train = torch.utils.data.DataLoader(
        dataset_forecasting_train,
        sampler=torch.utils.data.distributed.DistributedSampler(dataset_forecasting_train, shuffle=False) if _is_distributed else torch.utils.data.SequentialSampler(dataset_forecasting_train),
        batch_size=args.batch_size_forecast,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=_fc_drop_last,
    )
    data_loader_forecasting_test = torch.utils.data.DataLoader(
        dataset_forecasting_test,
        sampler=torch.utils.data.distributed.DistributedSampler(dataset_forecasting_test, shuffle=False) if _is_distributed else torch.utils.data.SequentialSampler(dataset_forecasting_test),
        batch_size=args.batch_size_forecast,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=_fc_drop_last,
    )
    data_loader_forecasting_val = None
    if dataset_forecasting_val is not None:
        data_loader_forecasting_val = torch.utils.data.DataLoader(
            dataset_forecasting_val,
            sampler=torch.utils.data.distributed.DistributedSampler(dataset_forecasting_val, shuffle=False) if _is_distributed else torch.utils.data.SequentialSampler(dataset_forecasting_val),
            batch_size=args.batch_size_forecast,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=_fc_drop_last,
        )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    _tm_kwargs = dict(
        c_in=args.c_in,
        seq_len=_SEQ_LEN,
        d_model=getattr(args, 'tsmixer_d_model', 128),
        e_layers=getattr(args, 'tsmixer_e_layers', 3),
        d_ff=getattr(args, 'tsmixer_d_ff', 256),
        dropout=args.dropout,
        patch_len=args.patch_len,
        down_sampling_layers=getattr(args, 'tsmixer_down_sampling_layers', 3),
        down_sampling_window=getattr(args, 'tsmixer_down_sampling_window', 2),
        down_sampling_method=getattr(args, 'tsmixer_down_sampling_method', 'avg'),
        decomp_method=getattr(args, 'tsmixer_decomp_method', 'moving_avg'),
        moving_avg=getattr(args, 'tsmixer_moving_avg', 25),
        top_k=getattr(args, 'tsmixer_top_k', 5),
        use_norm=getattr(args, 'tsmixer_use_norm', 1),
        channel_independence=getattr(args, 'tsmixer_channel_independence', 1),
    )
    _head_drop = float(os.environ.get("TS_FORECAST_HEAD_DROPOUT", "0.0"))  # opt-in; 0 = off
    _multi_scale = os.environ.get("TS_FORECAST_MULTISCALE", "0") == "1"
    model = TSMixerForecastModel(
        backbone=TSMixerForDINO(**_tm_kwargs),
        pred_len=args.pred_len,
        use_revin=getattr(args, 'tsmixer_use_revin', os.environ.get('LMC_NO_REVIN') != '1'),
        head_dropout=_head_drop,
        multi_scale=_multi_scale,
    )
    print(f"  [DINO forecast] head_dropout={_head_drop}  multi_scale={_multi_scale}")
    _loss_name = os.environ.get("TS_FORECAST_LOSS", "mse").lower()
    if _loss_name == "l1":
        criterion = nn.L1Loss()
    elif _loss_name == "huber":
        criterion = nn.HuberLoss(delta=float(os.environ.get("TS_FORECAST_HUBER_DELTA", "1.0")))
    else:
        criterion = nn.MSELoss()
    print(f"  [DINO forecast] loss={_loss_name}")
    _lp_fore  = getattr(args, 'linear_probe', True)
    _head_lr_fore = float(args.lr_forecasting)
    # Backbone (encoder) LR for fine-tune: defaults to the head LR, scaled by
    # TS_FORECAST_ENC_LR_SCALE (e.g. 0.1 = backbone trains 10x slower than head).
    _enc_lr_scale = float(os.environ.get("TS_FORECAST_ENC_LR_SCALE", "1.0"))
    _enc_lr       = float(getattr(args, 'lr_forecasting_encoder', None) or _head_lr_fore) * _enc_lr_scale
    _opt_name = os.environ.get("TS_FORECAST_OPT", "adam").lower()
    _fc_wd = float(os.environ.get("TS_FORECAST_WD", "1e-4"))   # forecast weight decay
    print(f"  [DINO forecast] weight_decay={_fc_wd}")
    if _opt_name == "prodigy":
        try:
            from prodigyopt import Prodigy
            _pparams = (model.head.parameters() if _lp_fore else model.parameters())
            _dcoef = float(os.environ.get("TS_PRODIGY_DCOEF", "1.0"))
            optimizer = Prodigy(_pparams, lr=1.0, d_coef=_dcoef, weight_decay=_fc_wd,
                                safeguard_warmup=True, use_bias_correction=True,
                                decouple=True)
            print(f"  [DINO forecast] optimizer: Prodigy (lr-free, d_coef={_dcoef})")
        except ImportError:
            print("  [DINO forecast] prodigyopt not installed "
                  "(`pip install prodigyopt`) — falling back to Adam.")
            _opt_name = "adam"
    if _opt_name != "prodigy":
        _mom = float(os.environ.get("TS_FORECAST_MOMENTUM", "0.9"))
        if _lp_fore:
            _p = model.head.parameters()
            if _opt_name == "adamw":
                optimizer = torch.optim.AdamW(_p, lr=_head_lr_fore, weight_decay=_fc_wd)
            elif _opt_name == "sgd":
                optimizer = torch.optim.SGD(_p, lr=_head_lr_fore, momentum=_mom, weight_decay=_fc_wd)
            else:
                optimizer = torch.optim.Adam(_p, lr=_head_lr_fore, weight_decay=_fc_wd)
        else:
            _g = [
                {"params": model.head.parameters(),     "lr": _head_lr_fore},
                {"params": model.backbone.parameters(), "lr": _enc_lr},
            ]
            if _opt_name == "adamw":
                optimizer = torch.optim.AdamW(_g, weight_decay=_fc_wd)
            elif _opt_name == "sgd":
                optimizer = torch.optim.SGD(_g, momentum=_mom, weight_decay=_fc_wd)
            else:
                optimizer = torch.optim.Adam(_g, weight_decay=_fc_wd)
            print(f"  [DINO forecast] head_lr={_head_lr_fore}  encoder_lr={_enc_lr}")
        print(f"  [DINO forecast] optimizer: {_opt_name} (mom={_mom if _opt_name=='sgd' else 'n/a'})")
    model = model.to(device)
    if args.path_num != 0:
        if args.path_num == "best":
            path = os.path.join(args.output_dir, 'checkpoint_best.pth')
        else:
            path = os.path.join(args.output_dir, f'checkpoint{args.path_num}.pth')
        print(f"Loading checkpoint: {path}")
        if not os.path.exists(path):
            print(f"  WARNING: checkpoint not found at {path}, using random init.")
        else:
            checkpoint = torch.load(path, weights_only=False, map_location=device)

            # Teacher (EMA) by default; TS_FORECAST_USE_STUDENT=1 loads the student instead.
            _src_key = "student" if os.environ.get("TS_FORECAST_USE_STUDENT", "0") == "1" else "teacher"
            state_dict = checkpoint[_src_key]
            print(f"  [DINO forecast] loading {_src_key} backbone weights")
            new_state_dict = {}
            model_state = model.state_dict()

            for key, value in state_dict.items():
                # Remove 'module.' prefix (DistributedDataParallel)
                new_key = key.replace('module.', '')
                # TimeMixer: checkpoint is TSMultiCropWrapper(TSMixerForDINO), keys are
                #   backbone.pdm_blocks.* which already matches TSMixerForecastModel.backbone.*
                # Only load if key exists in forecasting model and shapes match
                if new_key in model_state and model_state[new_key].shape == value.shape:
                    new_state_dict[new_key] = value

            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            print(f"✓ Loaded DINO teacher checkpoint from epoch {args.path_num}")
            print(f"  Loaded {len(new_state_dict)} / {len(model_state)} weights")
            print(f"  Missing: {missing}")
            print(f"  Missing (new head): {len(missing)}  |  Unexpected (DINO head): {len(unexpected)}")

    _total_steps = args.epochs_forecasting * len(data_loader_forecasting_train)
    if _opt_name == "prodigy":
        # Prodigy estimates the LR itself; cosine-anneal its multiplier toward 0.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_total_steps)
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr_forecasting,
            total_steps=_total_steps,
            pct_start=0.3, anneal_strategy='cos',
        )
    if _lp_fore:
        for param in model.backbone.parameters():
            param.requires_grad = False
        _trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _total     = sum(p.numel() for p in model.parameters())
        print(f"  [DINO forecast] MODE: linear probe — backbone FROZEN")
        print(f"  Trainable: {_trainable:,} / {_total:,} params")
    else:
        _total = sum(p.numel() for p in model.parameters())
        print(f"  [DINO forecast] MODE: full fine-tuning — backbone UNFROZEN")
        print(f"  Trainable: {_total:,} / {_total:,} params")

    # ── optional LR range-test (opt-in: TS_FORECAST_LR_FIND=1; N/A for Prodigy) ──
    if os.environ.get("TS_FORECAST_LR_FIND") == "1" and _opt_name != "prodigy":
        _best_lr = _lr_find_forecast(
            model, criterion, data_loader_forecasting_train, device,
            end_lr=float(os.environ.get("TS_LR_FIND_END", 1.0)),
            num_iter=int(os.environ.get("TS_LR_FIND_ITERS", 100)),
        )
        if _best_lr is not None:
            args.lr_forecasting = _best_lr
            # rebuild optimizer + OneCycle with the found LR (mirror the branches above)
            if _lp_fore:
                optimizer = torch.optim.Adam(model.head.parameters(),
                                             lr=_best_lr, weight_decay=1e-4)
            else:
                optimizer = torch.optim.Adam([
                    {"params": model.head.parameters(),     "lr": _best_lr},
                    {"params": model.backbone.parameters(), "lr": _best_lr},
                ], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=_best_lr,
                total_steps=args.epochs_forecasting * len(data_loader_forecasting_train),
                pct_start=0.3, anneal_strategy='cos',
            )
            print(f"  [DINO forecast] using lr-finder LR = {_best_lr:.3e}")

    best_val_loss_fc = float('inf')
    best_state_fc    = None
    for epoch in range(args.epochs_forecasting):
        model.train()
        for it, batch in enumerate(data_loader_forecasting_train):
            samples, labels = batch
            samples = samples.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(samples)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        # ── val MSE + keep-best ──
        if data_loader_forecasting_val is not None:
            model.eval()
            v_sum = 0.0; v_n = 0
            with torch.no_grad():
                for v_batch in data_loader_forecasting_val:
                    v_x, v_y = v_batch
                    v_x = v_x.to(device, non_blocking=True)
                    v_y = v_y.to(device, non_blocking=True)
                    v_out = model(v_x)
                    v_sum += criterion(v_out, v_y).item() * v_x.size(0)
                    v_n   += v_x.size(0)
            val_loss = v_sum / max(v_n, 1)
            if val_loss < best_val_loss_fc:
                best_val_loss_fc = val_loss
                best_state_fc    = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"  [DINO forecast] epoch {epoch} — val MSE: {val_loss:.6f}  (best: {best_val_loss_fc:.6f})")

    if best_state_fc is not None:
        model.load_state_dict(best_state_fc)
        print(f"  [DINO forecast] Restored best checkpoint (best val MSE={best_val_loss_fc:.6f})")

    # ── Cross-domain cross-C: the head was trained on the source (c_in=args.c_in),
    # but a zero-shot target (TS_FORECAST_TEST_CSV) may have a different channel
    # count. The forecast head is channel-independent, so rebuild the model at the
    # target C and copy the shape-matched weights; the C-sized RevIN affine (and
    # the c_in-based reshapes) adapt to the target — affine reinitializes to plain
    # per-instance normalization. ────────────────────────────────────────────────
    _test_c_in = getattr(dataset_forecasting_test, 'data_x', None)
    _test_c_in = _test_c_in.shape[-1] if _test_c_in is not None else args.c_in
    if _test_c_in != args.c_in:
        _tm_kwargs_t = dict(_tm_kwargs); _tm_kwargs_t['c_in'] = _test_c_in
        _test_model = TSMixerForecastModel(
            backbone=TSMixerForDINO(**_tm_kwargs_t),
            pred_len=args.pred_len,
            use_revin=getattr(args, 'tsmixer_use_revin', os.environ.get('LMC_NO_REVIN') != '1'),
            head_dropout=_head_drop,
            multi_scale=_multi_scale,
        ).to(device)
        _src_sd, _tgt_sd = model.state_dict(), _test_model.state_dict()
        _keep = {k: v for k, v in _src_sd.items() if k in _tgt_sd and _tgt_sd[k].shape == v.shape}
        _test_model.load_state_dict(_keep, strict=False)
        print(f"  [DINO forecast] cross-C {args.c_in}->{_test_c_in}: "
              f"copied {len(_keep)}/{len(_tgt_sd)} weights for zero-shot target test")
        model = _test_model
        # NB: do NOT mutate args.c_in — the forecast is called once per pred_len
        # with the same args, so the next horizon must still build/train at the
        # SOURCE channel count. `_test_c_in` drives the metric accumulators below.

    # Testing
    model.eval()
    os.makedirs("test_results/tests", exist_ok=True)
    model.operation = 'test'

    # Initialize accumulators for metrics across ALL test batches (target C for
    # cross-domain zero-shot; == args.c_in in-domain).
    num_vars = _test_c_in
    accumulated_mse = torch.zeros(num_vars).to(device)
    accumulated_mae = torch.zeros(num_vars).to(device)
    num_samples = 0

    with torch.no_grad():
        txt_save_path = f"test_results/tests/metrics_results_{args.path_num}.txt"
        first_batch_saved = False

        # Loop through ALL test batches
        for it, batch in enumerate(data_loader_forecasting_test):
            samples, labels = batch
            samples = samples.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)
            outputs = model(samples)

            # Compute MSE and MAE for this batch
            batch_size = outputs.shape[0]
            squared_errors = (outputs - labels) ** 2
            absolute_errors = torch.abs(outputs - labels)

            # Average over batch and time dimension, keep variable dimension
            batch_mse = squared_errors.mean(dim=(0, 1))  # [num_vars]
            batch_mae = absolute_errors.mean(dim=(0, 1))  # [num_vars]

            # Accumulate (weighted by batch size)
            accumulated_mse += batch_mse * batch_size
            accumulated_mae += batch_mae * batch_size
            num_samples += batch_size

            # Save visualization for first batch only (skip for large multivariate datasets)
            if not first_batch_saved:
                n_vars = outputs.shape[-1]
                if n_vars > 20:
                    first_batch_saved = True  # skip plotting
                    continue
                fig, axes = plt.subplots(n_vars, 1, figsize=(12, 3 * n_vars), sharex=True)
                if n_vars == 1: axes = [axes]

                for v in range(n_vars):
                    var_name = args.parms_for_testing_forecasting[v] if hasattr(args, 'parms_for_testing_forecasting') else f"Var {v}"

                    truth_segment = labels[0, :, v].cpu().numpy()
                    pred_segment  = outputs[0, :, v].cpu().numpy()
                    x = np.arange(len(truth_segment))

                    axes[v].plot(x, truth_segment, label="Ground Truth", color="black", linewidth=1.5)
                    axes[v].plot(x, pred_segment,  label="Forecast",     color="red",   linestyle="--", alpha=0.8)

                    axes[v].set_title(f"Variable {v}: {var_name}", fontsize=14, loc='left')
                    axes[v].legend(loc="upper left")
                    axes[v].grid(True, alpha=0.2)
                    axes[v].set_ylabel("Value")

                plt.xlabel("Forecast Time Steps")
                plt.tight_layout()

                save_path = f"test_results/tests/full_multivariable_forecast_{args.path_num}.png"
                plt.savefig(save_path)
                print(f"✅ Figure saved to: {save_path}")
                plt.close()
                first_batch_saved = True

        # Compute final averages across ALL test samples
        final_mse = (accumulated_mse / num_samples).cpu().numpy()
        final_mae = (accumulated_mae / num_samples).cpu().numpy()
        final_rmse = np.sqrt(final_mse)

        # Save metrics to file
        with open(txt_save_path, "w") as f:
            f.write(f"TEST SET METRICS (Averaged over {num_samples} samples)\n")
            f.write(f"Path: {args.path_num} (0=random, >0=DINO checkpoint)\n")
            f.write("="*60 + "\n\n")
            f.write(f"{'Var_Idx':<10} | {'MSE':<12} | {'MAE':<12} | {'RMSE':<12}\n")
            f.write("-" * 60 + "\n")

            for v in range(num_vars):
                f.write(f"{v:<10} | {final_mse[v]:<12.6f} | {final_mae[v]:<12.6f} | {final_rmse[v]:<12.6f}\n")

            f.write("\n" + "="*60 + "\n")
            f.write(f"OVERALL AVERAGES:\n")
            f.write(f"  Mean MSE:  {final_mse.mean():.6f}\n")
            f.write(f"  Mean MAE:  {final_mae.mean():.6f}\n")
            f.write(f"  Mean RMSE: {final_rmse.mean():.6f}\n")

        print(f"\n{'='*60}")
        print(f"TEST RESULTS (path_num={args.path_num}):")
        print(f"{'='*60}")
        print(f"Mean MSE:  {final_mse.mean():.6f}")
        print(f"Mean MAE:  {final_mae.mean():.6f}")
        print(f"Mean RMSE: {final_rmse.mean():.6f}")
        print(f"✅ Detailed metrics saved to {txt_save_path}")
        print(f"{'='*60}\n")
        return float(final_mse.mean())
