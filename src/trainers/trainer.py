"""
Trainer — single phase, 20 epochs

Cách dùng:
    from src.trainers.trainer import Trainer, TrainConfig
    cfg     = TrainConfig(dataset='coco', total_epochs=20)
    trainer = Trainer(model, loss_fn, cfg)
    trainer.train()
"""

import os
import math
import time
import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from tqdm import tqdm

import torchvision.datasets as dsets

from src.trainers.logger     import Logger
from src.trainers.visualizer import Visualizer
from src.datasets.aug_dataset import AugmentedDataset
from src.utils.xla_utils import (
    xla_available, is_tpu, is_master, xla_print,
    wrap_dataloader, clip_and_step, mark_step, bf16_context,
    setup_tpu, shard_model,
)


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class TrainConfig:
    run_name:    str   = "run"
    exp_dir:     str   = "experiments"
    data_root:   str   = "data"
    dataset:     str   = "coco"       # "coco" | "cifar10" | "cifar100" | "stl10"
    img_size:    int   = 128
    q:           int   = 3
    k:           int   = 2
    batch_size:  int   = 16
    grad_accum:  int   = 4            # effective batch = batch_size * grad_accum
    num_workers: int   = 8
    prefetch_factor: int = 2
    device:      str   = "cuda"
    use_amp:     bool  = True
    compile_model: bool = False
    use_vae:     bool  = True   # False = bỏ decoder + KL, chỉ train contrastive losses

    total_epochs: int  = 20
    lr:           float = 3e-4
    weight_decay: float = 1e-4

    # Loss weights (single phase, active from epoch 1)
    beta:          float = 0.1
    lambda_union:  float = 1.0
    lambda_sparse: float = 0.5
    lambda_ortho:  float = 0.1
    lambda_neg:    float = 0.3

    # Visualization
    recon_every:  int = 5
    curves_every: int = 10
    tsne_every:   int = 999   # tắt mặc định, bật nếu cần
    log_iter_every: int = 50  # log + update iter_curves.png mỗi N iteration


# ===========================================================================
# Metrics
# ===========================================================================

@torch.no_grad()
def compute_metrics(union_feat: torch.Tensor,
                    sparse_feat: torch.Tensor,
                    mse: float,
                    q: int) -> dict:
    B = union_feat.shape[0] // q
    u = union_feat.view(B, q, -1)
    s = sparse_feat.view(B, q, -1)

    u_sims = [(u[:, i] * u[:, j]).sum(dim=1)
              for i in range(q) for j in range(i+1, q)]
    s_sims = [(s[:, i] * s[:, j]).sum(dim=1)
              for i in range(q) for j in range(i+1, q)]

    union_consistency = torch.stack(u_sims).mean().item()
    sparse_divergence = 1.0 - torch.stack(s_sims).mean().item()
    ortho_score       = (u.mean(dim=2) * s.mean(dim=2)).abs().mean().item()
    psnr              = 10 * math.log10(1.0 / (mse + 1e-8))

    return {
        'union_consistency': union_consistency,
        'sparse_divergence': sparse_divergence,
        'ortho_score':       ortho_score,
        'recon_psnr':        psnr,
    }


# ===========================================================================
# Trainer
# ===========================================================================

class Trainer:

    def __init__(self, model, loss_fn, cfg: TrainConfig):
        self.model   = model.to(cfg.device)
        self.loss_fn = loss_fn
        self.cfg     = cfg
        self.device  = torch.device(cfg.device)

        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir  = os.path.join(cfg.exp_dir, f"{cfg.run_name}_{ts}")
        self.ckpt_dir = os.path.join(self.run_dir, 'checkpoints')
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.logger = Logger(self.run_dir)
        self.viz    = Visualizer(self.run_dir)

        self._best_total = float('inf')
        self._best_union = -float('inf')
        self._global_iter = 0   # tổng số iteration từ đầu training

        # Speedup: channels_last memory layout — 10-30% faster trên NVIDIA GPU
        # TPU không hỗ trợ channels_last → skip
        if not is_tpu():
            self.model = self.model.to(memory_format=torch.channels_last)
            torch.backends.cudnn.benchmark = True
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        # Multi-GPU (T4 x2): DataParallel tu dong chia batch qua cac GPU
        if not is_tpu() and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.logger.info(f'DataParallel: {torch.cuda.device_count()} GPUs')
        # TPU v5e: dùng bfloat16 qua bf16_context() thay cho GradScaler/AMP
        # GPU: dùng float16 GradScaler như cũ
        use_amp_gpu = cfg.use_amp and not is_tpu()
        # torch.cuda.amp.GradScaler works on PyTorch >= 1.6 (including Kaggle 2.1.x)
        # torch.amp.GradScaler only available on PyTorch >= 2.3
        if use_amp_gpu:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        if cfg.compile_model:
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                self.logger.info('torch.compile: ON (reduce-overhead)')
            except Exception as e:
                self.logger.info(f'torch.compile unavailable: {e}')

        self._write_config()

    # ------------------------------------------------------------------
    def _write_config(self):
        cfg = self.cfg
        lines = [
            f"Run:      {self.run_dir}",
            f"Dataset:  {cfg.dataset}  img_size={cfg.img_size}",
            f"q={cfg.q}  k={cfg.k}  batch={cfg.batch_size}  grad_accum={cfg.grad_accum}",
            f"Epochs:   {cfg.total_epochs}   lr={cfg.lr}   wd={cfg.weight_decay}",
            f"AMP:      {cfg.use_amp}",
            f"β={cfg.beta}  λu={cfg.lambda_union}  λs={cfg.lambda_sparse}"
            f"  λo={cfg.lambda_ortho}  λn={cfg.lambda_neg}",
        ]
        with open(os.path.join(self.run_dir, 'config.txt'), 'w') as f:
            f.write('\n'.join(lines))
        self.logger.info('\n'.join(lines))

    # ------------------------------------------------------------------
    def _build_loader(self):
        cfg = self.cfg
        print(f"[Data] Loading dataset '{cfg.dataset}' from {cfg.data_root} ...")
        if cfg.dataset == 'coco':
            from src.datasets.aug_dataset import CocoImageDataset
            base = CocoImageDataset(root=cfg.data_root, split='train')
        elif cfg.dataset == 'cifar10':
            base = dsets.CIFAR10(root=cfg.data_root, train=True,
                                 download=False, transform=None)
        elif cfg.dataset == 'cifar100':
            base = dsets.CIFAR100(root=cfg.data_root, train=True,
                                  download=False, transform=None)
        elif cfg.dataset == 'stl10':
            base = dsets.STL10(root=cfg.data_root, split='unlabeled',
                               download=False, transform=None)
        elif cfg.dataset == 'imagenet':
            import torchvision.datasets as _dsets
            # ImageFolder scans 1.2M files — có thể mất 2-5 phút trên Kaggle NFS
            print("[Data] Scanning ImageNet directory (may take 2-5 min on Kaggle)...")
            base = _dsets.ImageFolder(root=cfg.data_root, transform=None)
        else:
            raise ValueError(f"Unknown dataset: {cfg.dataset}")

        print(f"[Data] Dataset ready: {len(base):,} samples")
        ds = AugmentedDataset(base, q=cfg.q, k=cfg.k, img_size=cfg.img_size)

        # TPU: workers > 0 co the deadlock voi MpDeviceLoader -> force 0
        nw = 0 if is_tpu() else cfg.num_workers
        pf = cfg.prefetch_factor if nw > 0 else None

        # CUDA + fork = deadlock khi CUDA đã init trước khi spawn workers.
        # Dùng 'spawn' để tránh — workers khởi động lại từ đầu (an toàn).
        # Nếu nw=0 thì không spawn gì cả.
        mp_ctx = None
        if nw > 0 and torch.cuda.is_initialized():
            mp_ctx = 'spawn'
            print(f"[Data] Using multiprocessing_context='spawn' (CUDA pre-initialized)")

        loader = DataLoader(
            ds,
            batch_size             = cfg.batch_size,
            shuffle                = True,
            num_workers            = nw,
            prefetch_factor        = pf,
            pin_memory             = not is_tpu(),
            drop_last              = True,
            persistent_workers     = nw > 0,
            multiprocessing_context= mp_ctx,
        )
        print(f"[Data] DataLoader ready | workers={nw} prefetch={pf} pin_memory={not is_tpu()}")
        # Wrap với XLA MpDeviceLoader trên TPU để pipeline data → device
        return wrap_dataloader(loader, self.device)

    # ------------------------------------------------------------------
    def _set_loss_weights(self):
        cfg = self.cfg
        self.loss_fn.beta          = cfg.beta
        self.loss_fn.lambda_union  = cfg.lambda_union
        self.loss_fn.lambda_sparse = cfg.lambda_sparse
        self.loss_fn.lambda_ortho  = cfg.lambda_ortho
        self.loss_fn.lambda_neg    = cfg.lambda_neg

    # ------------------------------------------------------------------
    def _train_epoch(self, loader, optimizer, epoch: int):
        self.model.train()
        cfg = self.cfg

        acc = {k: 0.0 for k in ['total','vae','mse','kl','union','sparse','ortho','neg','uniform']}
        n_batches    = len(loader)
        viz_inputs   = viz_recons = viz_union = viz_sparse = viz_imgfeat = None
        nan_skipped  = 0

        optimizer.zero_grad()

        pbar = tqdm(loader, desc=f"Ep {epoch:03d}/{cfg.total_epochs}",
                    leave=False, dynamic_ncols=True)

        for batch_idx, (core_imgs, neg_imgs) in enumerate(pbar):
            B, q, C, H, W = core_imgs.shape
            use_inbatch_neg = (neg_imgs.numel() == 0)  # k=0 → in-batch negatives

            _mf = torch.channels_last if not is_tpu() else torch.contiguous_format
            x_bq = core_imgs.view(B * q, C, H, W).to(
                cfg.device, non_blocking=not is_tpu(), memory_format=_mf)

            with bf16_context():  # bfloat16 trên TPU v5e, float16 trên GPU, no-op trên CPU
                recon, mu, logvar, image_feat, union_feat, sparse_feat = \
                    self.model(x_bq, skip_decoder=not cfg.use_vae)

                if use_inbatch_neg:
                    # In-batch negatives: shuffle các ảnh trong batch làm negative
                    # Không tốn disk IO, chuẩn SimCLR style
                    perm = torch.randperm(B, device=self.device)
                    union_neg = union_feat.view(B, q, -1)[perm].view(B * q, -1).detach()
                else:
                    k_ = neg_imgs.shape[1]
                    x_bk = neg_imgs.view(B * k_, C, H, W).to(
                        cfg.device, non_blocking=True, memory_format=torch.channels_last)
                    with torch.no_grad():
                        _, _, _, _, union_neg, _ = self.model(x_bk, skip_decoder=not cfg.use_vae)

            # Cast về float32 để loss ổn định
            mu_f         = mu.float()
            logvar_f     = logvar.float() if logvar is not None else None
            recon_f      = recon.float()  if recon  is not None else None
            x_bq_f       = x_bq.float()
            image_feat_f = image_feat.float()
            union_feat_f = union_feat.float()
            sparse_feat_f= sparse_feat.float()
            union_neg_f  = union_neg.float()

            total, details = self.loss_fn(
                recon_f, x_bq_f, mu_f, logvar_f,
                image_feat_f, union_feat_f, sparse_feat_f, union_neg_f,
            )

            if not torch.isfinite(total):
                nan_skipped += 1
                optimizer.zero_grad()
                continue  # không gọi scaler.update() — chưa có backward pass

            total_s = total / cfg.grad_accum
            if self.scaler:
                self.scaler.scale(total_s).backward()
            else:
                total_s.backward()

            if (batch_idx + 1) % cfg.grad_accum == 0 or (batch_idx + 1) == n_batches:
                clip_and_step(optimizer, self.model, self.scaler, max_norm=0.5)

                # Kiểm tra NaN trong weights sau step (GPU only — TPU không cần)
                if not is_tpu() and any(not p.isfinite().all()
                       for p in self.model.parameters() if p.requires_grad):
                    self.logger.info(
                        f'  [CRITICAL] NaN in weights at batch {batch_idx}, '
                        f'restoring from last good checkpoint')
                    _last_good = os.path.join(self.ckpt_dir, 'checkpoint_latest.pt')
                    if os.path.exists(_last_good):
                        ckpt = torch.load(_last_good, map_location=self.device)
                        self.model.load_state_dict(ckpt['model_state_dict'])
                    nan_skipped += 1
                    optimizer.zero_grad()
                    continue

                optimizer.zero_grad()

            for k_, v in details.items():
                acc[k_] += v / n_batches

            # Per-iteration logging + graph update
            self._global_iter += 1
            if self._global_iter % cfg.log_iter_every == 0:
                lr_cur = optimizer.param_groups[0]['lr']
                # Tính quick batch metrics từ union/sparse feat hiện tại
                with torch.no_grad():
                    uf = union_feat.float().view(B, q, -1)
                    sf = sparse_feat.float().view(B, q, -1)
                    u_sim = (uf[:, 0] * uf[:, -1]).sum(-1).mean().item()
                    s_sim = (sf[:, 0] * sf[:, -1]).sum(-1).mean().item()
                    batch_metrics = {
                        'union_consistency': u_sim,
                        'sparse_divergence': 1.0 - s_sim,
                        'ortho_score': (uf.mean(2) * sf.mean(2)).abs().mean().item(),
                    }
                self.logger.log_iter(
                    epoch, cfg.total_epochs,
                    batch_idx, n_batches,
                    self._global_iter,
                    details, batch_metrics, lr_cur,
                    use_vae=cfg.use_vae,
                )
                self.viz.record_iter(self._global_iter, epoch, details, batch_metrics)
                self.viz.save_iter_curves(self._global_iter)

            # details đã là Python float → không cần .item(), không gây GPU sync
            if cfg.use_vae:
                pbar.set_postfix(ordered_dict={
                    'loss'  : f"{details['total']:.3f}",
                    'mse'   : f"{details['mse']:.3f}",
                    'union' : f"{details['union']:.3f}",
                    'sparse': f"{details['sparse']:.3f}",
                }, refresh=False)
            else:
                pbar.set_postfix(ordered_dict={
                    'loss'  : f"{details['total']:.3f}",
                    'union' : f"{details['union']:.3f}",
                    'sparse': f"{details['sparse']:.3f}",
                    'ortho' : f"{details['ortho']:.3f}",
                }, refresh=False)

            # Lưu batch đầu cho viz
            if viz_inputs is None:
                with torch.no_grad():
                    viz_inputs  = x_bq[:q].float().cpu()
                    viz_recons  = recon[:q].float().cpu() if recon is not None else None
                    viz_union   = union_feat[:q].float()
                    viz_sparse  = sparse_feat[:q].float()
                    viz_imgfeat = image_feat[:q].float()

        if nan_skipped > 0:
            self.logger.info(f"  [WARN] Epoch {epoch}: {nan_skipped} NaN batches skipped")

        # Nếu toàn bộ batch bị NaN, trả về placeholder zeros để không crash metrics
        if viz_union is None:
            q_ = cfg.q
            # Lấy dim từ model nếu có, fallback về 1
            dim_u = getattr(self.model, 'dim_inter',  1)
            dim_s = getattr(self.model, 'dim_unique', 1)
            viz_union   = torch.zeros(q_, dim_u)
            viz_sparse  = torch.zeros(q_, dim_s)
            viz_inputs  = torch.zeros(q_, 3, cfg.img_size, cfg.img_size)
            viz_imgfeat = torch.zeros(q_, 1)
            self.logger.info(f"  [WARN] Epoch {epoch}: ALL batches NaN! Placeholder used.")

        return acc, viz_inputs, viz_recons, viz_union, viz_sparse, viz_imgfeat

    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch, optimizer, scheduler, metrics, tag):
        torch.save({
            'epoch':               epoch,
            'model_state_dict':    self.model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'scheduler_state_dict':scheduler.state_dict() if scheduler else None,
            'metrics':             metrics,
        }, os.path.join(self.ckpt_dir, f'checkpoint_{tag}.pt'))

    # ------------------------------------------------------------------
    def train(self):
        cfg      = self.cfg
        loader   = self._build_loader()
        self._set_loss_weights()

        optimizer = optim.AdamW(self.model.parameters(),
                                lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.total_epochs, eta_min=1e-6)

        self.logger.info(f"\n{'='*60}\nTraining — {cfg.total_epochs} epochs\n{'='*60}")
        self.logger.print_legend(use_vae=cfg.use_vae)

        for epoch in range(1, cfg.total_epochs + 1):
            t0 = time.time()
            details, viz_in, viz_rec, viz_union, viz_sparse, viz_imgfeat = \
                self._train_epoch(loader, optimizer, epoch)
            scheduler.step()
            elapsed = time.time() - t0

            metrics = compute_metrics(viz_union, viz_sparse, details['mse'], cfg.q)
            # Skip recon_psnr khi không có decoder (mse=0 → PSNR vô nghĩa)
            if not cfg.use_vae:
                metrics['recon_psnr'] = 0.0
            lr      = optimizer.param_groups[0]['lr']

            # Log 1 lần/epoch
            self.logger.log_epoch(epoch, cfg.total_epochs, 1,
                                  details, metrics, lr, elapsed,
                                  use_vae=cfg.use_vae)
            self.viz.record(epoch, 1, details, metrics)

            # Visualizations
            if epoch % cfg.recon_every == 0 or epoch == 1:
                if viz_rec is not None:
                    self.viz.save_recon(epoch, viz_in, viz_rec, metrics['recon_psnr'])
            if epoch % cfg.curves_every == 0:
                self.viz.save_loss_curves(epoch)
                self.viz.save_metric_curves(epoch)
                self.viz.save_similarity_heatmaps(epoch, viz_imgfeat, viz_union, viz_sparse)
            if epoch % cfg.tsne_every == 0:
                labels = [f'aug{i+1}' for i in range(cfg.q)] * (viz_union.shape[0] // cfg.q)
                self.viz.save_tsne(epoch, viz_union,  labels, 'Union',  f'union_{epoch:04d}.png')
                self.viz.save_tsne(epoch, viz_sparse, labels, 'Sparse', f'sparse_{epoch:04d}.png')

            # Checkpoints — lưu từng epoch + best
            self._save_checkpoint(epoch, optimizer, scheduler, metrics, f'epoch_{epoch:03d}')
            self._save_checkpoint(epoch, optimizer, scheduler, metrics, 'latest')
            if details['total'] < self._best_total:
                self._best_total = details['total']
                self._save_checkpoint(epoch, optimizer, scheduler, metrics, 'best_total')
            if metrics['union_consistency'] > self._best_union:
                self._best_union = metrics['union_consistency']
                self._save_checkpoint(epoch, optimizer, scheduler, metrics, 'best_union')

        self.viz.save_loss_curves(epoch)
        self.viz.save_metric_curves(epoch)
        self.logger.info(f"\nDone. Results: {self.run_dir}")
        self.logger.close()
