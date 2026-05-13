"""
trainer.py — TensorFlow training loop (GradientTape).
"""

import os
import math
import time
import datetime
import numpy as np
import tensorflow as tf
import keras
from dataclasses import dataclass, field
from tqdm import tqdm

from src.trainers.logger     import Logger
from src.trainers.visualizer import Visualizer


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class TrainConfig:
    run_name:    str   = "run"
    exp_dir:     str   = "experiments"
    data_root:   str   = "data"
    dataset:     str   = "coco"
    img_size:    int   = 128
    q:           int   = 3
    k:           int   = 0
    batch_size:  int   = 256
    grad_accum:  int   = 1
    num_workers: int   = 8      # parallel map calls for tf.data
    prefetch_factor: int = 4
    device:      str   = "auto"   # "auto" | "cpu" | "gpu"
    use_amp:     bool  = True
    compile_model: bool = False   # tf.function JIT (auto via @tf.function)
    use_vae:     bool  = False    # False = skip decoder, contrastive only

    total_epochs: int   = 30
    lr:           float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int  = 5

    beta:          float = 0.1
    lambda_union:  float = 1.0
    lambda_sparse: float = 0.5
    lambda_ortho:  float = 0.1
    lambda_neg:    float = 0.3

    # Linear probe evaluation
    eval_every:        int  = 10
    eval_dataset:      str  = 'cifar10'
    eval_data_root:    str  = 'data'
    eval_probe_epochs: int  = 20

    # Logging / visualization
    recon_every:    int = 5
    curves_every:   int = 10
    tsne_every:     int = 999
    log_iter_every: int = 50

    # Compatibility fields (used by tests / main.py)
    dim_inter:  int = 128
    dim_unique: int = 64
    latent_ch:  int = 192


# ===========================================================================
# Metrics
# ===========================================================================

def compute_metrics(union_feat: np.ndarray,
                    sparse_feat: np.ndarray,
                    mse: float,
                    q: int) -> dict:
    """Compute decomposition quality metrics from numpy arrays."""
    B = union_feat.shape[0] // q
    u = union_feat.reshape(B, q, -1)
    s = sparse_feat.reshape(B, q, -1)

    u_sims = [(u[:, i] * u[:, j]).sum(axis=1)
              for i in range(q) for j in range(i + 1, q)]
    s_sims = [(s[:, i] * s[:, j]).sum(axis=1)
              for i in range(q) for j in range(i + 1, q)]

    union_consistency = float(np.stack(u_sims).mean())
    sparse_divergence = float(1.0 - np.stack(s_sims).mean())
    ortho_score       = float(np.abs(u.mean(axis=2) * s.mean(axis=2)).mean())
    psnr              = float(10 * math.log10(1.0 / (mse + 1e-8)))

    return {
        'union_consistency': union_consistency,
        'sparse_divergence': sparse_divergence,
        'ortho_score':       ortho_score,
        'recon_psnr':        psnr,
    }


# ===========================================================================
# LR Schedule: warmup + cosine decay
# ===========================================================================

class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, warmup_steps, total_steps, min_lr=1e-6):
        self.peak_lr      = float(peak_lr)
        self.warmup_steps = float(warmup_steps)
        self.total_steps  = float(total_steps)
        self.min_lr       = float(min_lr)

    def __call__(self, step):
        step  = tf.cast(step, tf.float32)
        ws    = tf.constant(self.warmup_steps, tf.float32)
        ts    = tf.constant(self.total_steps,  tf.float32)
        pk    = tf.constant(self.peak_lr,      tf.float32)
        mn    = tf.constant(self.min_lr,       tf.float32)

        warmup_lr = mn + (pk - mn) * step / tf.maximum(ws, 1.0)
        t         = tf.clip_by_value((step - ws) / tf.maximum(ts - ws, 1.0), 0.0, 1.0)
        cosine_lr = mn + 0.5 * (pk - mn) * (1.0 + tf.cos(np.pi * t))
        return tf.where(step < ws, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            'peak_lr': self.peak_lr, 'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps, 'min_lr': self.min_lr,
        }


# ===========================================================================
# Trainer
# ===========================================================================

class Trainer:

    def __init__(self, model, loss_fn, cfg: TrainConfig):
        self.model   = model
        self.loss_fn = loss_fn
        self.cfg     = cfg

        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir  = os.path.join(cfg.exp_dir, f"{cfg.run_name}_{ts}")
        self.ckpt_dir = os.path.join(self.run_dir, 'checkpoints')
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.logger = Logger(self.run_dir)
        self.viz    = Visualizer(self.run_dir)

        self._best_total = float('inf')
        self._best_union = -float('inf')
        self._global_iter = 0

        self._probe_history:  list = []
        self._metric_history: list = []

        # Setup device / mixed precision
        from src.utils.xla_utils import setup_mixed_precision, device_info
        if cfg.use_amp:
            setup_mixed_precision(use_amp=True)
        self.logger.info(f"Device: {device_info()}")

        self._write_config()

    # ------------------------------------------------------------------
    def _write_config(self):
        cfg = self.cfg
        lines = [
            f"Run:     {self.run_dir}",
            f"Dataset: {cfg.dataset}  img_size={cfg.img_size}",
            f"q={cfg.q}  k={cfg.k}  batch={cfg.batch_size}  grad_accum={cfg.grad_accum}",
            f"Epochs:  {cfg.total_epochs}   lr={cfg.lr}   wd={cfg.weight_decay}",
            f"AMP:     {cfg.use_amp}",
            f"β={cfg.beta}  λu={cfg.lambda_union}  λs={cfg.lambda_sparse}"
            f"  λo={cfg.lambda_ortho}  λn={cfg.lambda_neg}",
        ]
        with open(os.path.join(self.run_dir, 'config.txt'), 'w') as f:
            f.write('\n'.join(lines))
        self.logger.info('\n'.join(lines))

    # ------------------------------------------------------------------
    def _build_loader(self):
        cfg = self.cfg
        from src.datasets.aug_dataset import (
            make_aug_dataset, load_cifar10, load_cifar100,
            load_images_from_dir,
        )
        print(f"[Data] Loading '{cfg.dataset}' from {cfg.data_root} ...")

        if cfg.dataset == 'cifar10':
            images, _ = load_cifar10(cfg.data_root, train=True)
        elif cfg.dataset == 'cifar100':
            images, _ = load_cifar100(cfg.data_root, train=True)
        elif cfg.dataset in ('coco', 'imagenet'):
            images = load_images_from_dir(cfg.data_root)
        else:
            raise ValueError(f"Unknown dataset: {cfg.dataset}")

        n = len(images)
        print(f"[Data] {n:,} samples  →  ~{n // cfg.batch_size:,} steps/epoch")
        dataset = make_aug_dataset(
            images, cfg.q, cfg.k, cfg.img_size,
            batch_size=cfg.batch_size,
            shuffle_buffer=min(50000, n),
            prefetch=cfg.prefetch_factor,
        )
        steps = n // cfg.batch_size   # drop_remainder=True
        return dataset, steps

    # ------------------------------------------------------------------
    def _setup_train_fn(self, optimizer):
        """
        Compile the inner training step with @tf.function.
        Forward + backward + optimizer update runs as one fused graph — no eager overhead.
        """
        model    = self.model
        loss_fn  = self.loss_fn
        q        = self.cfg.q
        img_size = self.cfg.img_size
        use_vae  = self.cfg.use_vae
        tvars    = model.trainable_variables

        @tf.function(reduce_retracing=True)
        def _step(core_imgs):
            B    = tf.shape(core_imgs)[0]
            x_bq = tf.reshape(core_imgs, (B * q, img_size, img_size, 3))

            with tf.GradientTape() as tape:
                recon, mu, logvar, image_feat, union_feat, sparse_feat = \
                    model(x_bq, skip_decoder=not use_vae, training=True)

                uf = tf.cast(union_feat,  tf.float32)
                sf = tf.cast(sparse_feat, tf.float32)
                ff = tf.cast(image_feat,  tf.float32)

                perm      = tf.random.shuffle(tf.range(B))
                union_neg = tf.reshape(
                    tf.gather(tf.reshape(uf, (B, q, -1)), perm), (B * q, -1))

                recon_f = tf.cast(recon,  tf.float32) if recon  is not None else None
                mu_f    = tf.cast(mu,     tf.float32) if mu     is not None else None
                lv_f    = tf.cast(logvar, tf.float32) if logvar is not None else None

                total, details = loss_fn(
                    recon_f, tf.cast(x_bq, tf.float32), mu_f, lv_f,
                    ff, uf, sf, union_neg, q=q,
                )

            grads = tape.gradient(total, tvars)
            grads = [tf.zeros_like(v) if g is None else g for g, v in zip(grads, tvars)]
            optimizer.apply_gradients(zip(grads, tvars))

            # Stack all scalars into one tensor → single GPU→CPU transfer per step
            keys = ['total', 'vae', 'mse', 'kl', 'union', 'sparse', 'ortho', 'neg', 'uniform', 'aux']
            scalar_vec = tf.stack([details[k] for k in keys])
            return scalar_vec, uf, sf, ff, x_bq, recon_f

        _KEYS = ['total', 'vae', 'mse', 'kl', 'union', 'sparse', 'ortho', 'neg', 'uniform', 'aux']

        def unpack(scalar_vec):
            vals = scalar_vec.numpy()
            return {k: float(v) for k, v in zip(_KEYS, vals)}

        return _step, unpack

    # ------------------------------------------------------------------
    def _build_optimizer(self, steps_per_epoch: int):
        cfg      = self.cfg
        total    = cfg.total_epochs * steps_per_epoch
        warmup   = cfg.warmup_epochs * steps_per_epoch
        schedule = WarmupCosineDecay(
            peak_lr=cfg.lr, warmup_steps=warmup,
            total_steps=total, min_lr=1e-6,
        )
        return keras.optimizers.AdamW(
            learning_rate=schedule,
            weight_decay=cfg.weight_decay,
            clipnorm=0.5,
        )

    # ------------------------------------------------------------------
    def _train_epoch(self, loader, optimizer, train_step, unpack, epoch: int):
        cfg      = self.cfg
        q        = cfg.q
        img_size = cfg.img_size
        acc      = {k: 0.0 for k in [
            'total', 'vae', 'mse', 'kl', 'union', 'sparse', 'ortho', 'neg', 'uniform', 'aux'
        ]}
        n_batches = 0
        nan_skipped = 0
        viz_union = viz_sparse = viz_inputs = viz_recon = viz_imgfeat = None

        pbar = tqdm(loader, desc=f"Ep {epoch:03d}/{cfg.total_epochs}",
                    leave=False, dynamic_ncols=True)

        # Update tqdm every N steps to avoid stalling GPU
        _POSTFIX_EVERY = 20
        _last_postfix  = {'loss': '?', 'union': '?', 'sparse': '?'}

        for step, (core_imgs, _neg_imgs) in enumerate(pbar):
            # ── compiled GPU step (forward + backward + optimizer) ─────────
            scalar_vec, uf, sf, ff, x_bq, recon_f = train_step(core_imgs)

            self._global_iter += 1
            n_batches += 1

            # ── CPU sync: one transfer for all scalars ─────────────────────
            do_log     = cfg.log_iter_every > 0 and self._global_iter % cfg.log_iter_every == 0
            do_postfix = step % _POSTFIX_EVERY == 0
            do_viz     = viz_union is None

            if do_log or do_postfix or do_viz:
                details = unpack(scalar_vec)          # single GPU→CPU transfer

                if not np.isfinite(details['total']):
                    nan_skipped += 1

                for k_, v in details.items():
                    acc[k_] += v

                if do_postfix:
                    _last_postfix = {
                        'loss':   f"{details['total']:.3f}",
                        'union':  f"{details['union']:.3f}",
                        'sparse': f"{details['sparse']:.3f}",
                    }
                    pbar.set_postfix(ordered_dict=_last_postfix, refresh=False)

                if do_log:
                    uf_np = uf.numpy()
                    sf_np = sf.numpy()
                    bm = {
                        'union_consistency': float(
                            (uf_np[:q] * uf_np[-q:]).sum(axis=1).mean()),
                        'sparse_divergence': float(
                            1.0 - (sf_np[:q] * sf_np[-q:]).sum(axis=1).mean()),
                        'ortho_score': float(np.abs(
                            uf_np.reshape(-1, q, uf_np.shape[-1]).mean(2) *
                            sf_np.reshape(-1, q, sf_np.shape[-1]).mean(2)
                        ).mean()),
                    }
                    lr_val = float(self._optimizer_lr(optimizer))
                    self.logger.log_iter(
                        epoch, cfg.total_epochs, step, -1, self._global_iter,
                        details, bm, lr_val, use_vae=cfg.use_vae,
                    )
                    self.viz.record_iter(self._global_iter, epoch, details, bm)
                    self.viz.save_iter_curves(self._global_iter)

                if do_viz:
                    viz_union   = uf[:q].numpy()
                    viz_sparse  = sf[:q].numpy()
                    viz_imgfeat = ff[:q].numpy()
                    viz_inputs  = x_bq[:q].numpy()
                    viz_recon   = recon_f[:q].numpy() if recon_f is not None else None
            else:
                # No sync needed — accumulate TF scalars directly
                scalar_np = scalar_vec.numpy()
                for i, k in enumerate(['total', 'vae', 'mse', 'kl', 'union',
                                        'sparse', 'ortho', 'neg', 'uniform', 'aux']):
                    acc[k] += float(scalar_np[i])

        if nan_skipped > 0:
            self.logger.info(f"  [WARN] Epoch {epoch}: {nan_skipped} NaN batches skipped")

        if viz_union is None:
            viz_union   = np.zeros((q, cfg.dim_inter))
            viz_sparse  = np.zeros((q, cfg.dim_unique))
            viz_imgfeat = np.zeros((q, 1))
            viz_inputs  = np.zeros((q, img_size, img_size, 3))

        if n_batches > 0:
            for k_ in acc:
                acc[k_] /= n_batches

        return acc, viz_inputs, viz_recon, viz_union, viz_sparse, viz_imgfeat

    # ------------------------------------------------------------------
    @staticmethod
    def _optimizer_lr(opt) -> float:
        """Get current LR from optimizer (handles LR schedule)."""
        try:
            lr = opt.learning_rate
            if callable(lr):
                return float(lr(opt.iterations))
            return float(lr)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch, optimizer, metrics, tag):
        weights_path = os.path.join(self.ckpt_dir, f'weights_{tag}.keras')
        self.model.save_weights(weights_path)
        # Save metadata
        import json
        meta = {'epoch': epoch, 'metrics': metrics}
        with open(os.path.join(self.ckpt_dir, f'meta_{tag}.json'), 'w') as f:
            json.dump(meta, f, indent=2)

    # ------------------------------------------------------------------
    def _get_dim_inter(self) -> int:
        return getattr(self.model, 'dim_inter', self.cfg.dim_inter)

    # ------------------------------------------------------------------
    def _run_eval(self, epoch: int):
        from src.eval.linear_probe import run_linear_probe
        cfg = self.cfg
        if cfg.eval_every <= 0:
            return None
        self.logger.info(f"\n  ── Linear Probe eval (epoch {epoch}) ──")
        try:
            result = run_linear_probe(
                model         = self.model,
                dataset       = cfg.eval_dataset,
                data_root     = cfg.eval_data_root,
                img_size      = cfg.img_size,
                dim_inter     = self._get_dim_inter(),
                probe_epochs  = cfg.eval_probe_epochs,
                verbose       = True,
            )
        except Exception as e:
            self.logger.info(f"  [LinearProbe] ERROR: {e}")
            return None
        result['epoch']   = epoch
        result['dataset'] = cfg.eval_dataset
        self._probe_history.append(result)
        self.logger.log_probe(epoch, result,
                              dataset=cfg.eval_dataset,
                              probe_epochs=cfg.eval_probe_epochs)
        return result

    # ------------------------------------------------------------------
    def _print_final_summary(self):
        cfg = self.cfg
        sep = '=' * 68
        bar = '-' * 68
        lines = [
            f"\n{sep}",
            f"  TRAINING COMPLETE — Paper-Ready Summary",
            f"{sep}",
            f"  Dataset : {cfg.dataset}  img={cfg.img_size}px  q={cfg.q}  k={cfg.k}",
            f"  Epochs  : {cfg.total_epochs}  lr={cfg.lr}  warmup={cfg.warmup_epochs}",
            f"  Results : {self.run_dir}",
            bar,
        ]
        if self._metric_history:
            best_m = max(self._metric_history,
                         key=lambda x: x.get('union_consistency', 0))
            ep  = best_m['epoch']
            uc  = best_m.get('union_consistency', 0)
            sd  = best_m.get('sparse_divergence', 0)
            os_ = best_m.get('ortho_score', 1)
            lines += [
                f"  Best Decomposition Quality (epoch {ep:03d}):",
                f"    union_consistency : {uc:.4f}  {'✓' if uc >= 0.85 else '✗ (target >0.85)'}",
                f"    sparse_divergence : {sd:.4f}  {'✓' if sd >= 0.70 else '✗ (target >0.70)'}",
                f"    ortho_score       : {os_:.4f}  {'✓' if os_ <= 0.10 else '✗ (target <0.10)'}",
                bar,
            ]
        if self._probe_history:
            best_p = max(self._probe_history, key=lambda x: x['top1'])
            ds = self._probe_history[0]['dataset']
            lines.append(f"  Linear Probe ({ds}, {cfg.eval_probe_epochs}-epoch head):")
            lines.append(f"  {'Epoch':>6}  {'Top-1':>7}  {'Top-5':>7}  {'Time':>6}")
            lines.append(f"  {'------':>6}  {'-------':>7}  {'-------':>7}  {'------':>6}")
            for r in self._probe_history:
                mark = ' ◀ best' if r['epoch'] == best_p['epoch'] else ''
                lines.append(
                    f"  {r['epoch']:>6d}  {r['top1']:>6.2f}%  "
                    f"{r['top5']:>6.2f}%  {r['time_s']:>5.0f}s{mark}"
                )
            lines += [
                bar,
                f"  Best top-1: {best_p['top1']:.2f}%  (epoch {best_p['epoch']:03d})",
            ]
        else:
            lines.append("  Linear Probe: không chạy (--eval-every 0)")
        lines += [bar, f"  Checkpoints: {self.ckpt_dir}", sep]
        self.logger.info('\n'.join(lines))

    # ------------------------------------------------------------------
    def train(self):
        cfg    = self.cfg
        loader, steps_per_epoch = self._build_loader()
        optimizer = self._build_optimizer(steps_per_epoch)
        train_step, unpack = self._setup_train_fn(optimizer)
        self._optimizer_ref = optimizer  # for LR logging

        self.logger.info(f"\n{'='*60}\nTraining — {cfg.total_epochs} epochs\n{'='*60}")
        self.logger.print_legend(use_vae=cfg.use_vae)

        for epoch in range(1, cfg.total_epochs + 1):
            t0 = time.time()
            details, viz_in, viz_rec, viz_union, viz_sparse, viz_imgfeat = \
                self._train_epoch(loader, optimizer, train_step, unpack, epoch)
            elapsed = time.time() - t0

            metrics = compute_metrics(viz_union, viz_sparse, details['mse'], cfg.q)
            if not cfg.use_vae:
                metrics['recon_psnr'] = 0.0
            lr_val = self._optimizer_lr(optimizer)

            self.logger.log_epoch(epoch, cfg.total_epochs, 1,
                                  details, metrics, lr_val, elapsed,
                                  use_vae=cfg.use_vae)
            self.viz.record(epoch, 1, details, metrics)
            self._metric_history.append({'epoch': epoch, **metrics})

            # Visualizations
            if epoch % cfg.recon_every == 0 or epoch == 1:
                if viz_rec is not None and viz_in is not None:
                    # Convert NHWC numpy to torch-compatible for visualizer
                    import torch
                    in_t  = torch.from_numpy(viz_in.transpose(0, 3, 1, 2))
                    rec_t = torch.from_numpy(viz_rec.transpose(0, 3, 1, 2))
                    self.viz.save_recon(epoch, in_t, rec_t, metrics['recon_psnr'])
            if epoch % cfg.curves_every == 0:
                self.viz.save_loss_curves(epoch)
                self.viz.save_metric_curves(epoch)
                import torch
                imgf_t = torch.from_numpy(viz_imgfeat)
                uf_t   = torch.from_numpy(viz_union)
                sf_t   = torch.from_numpy(viz_sparse)
                self.viz.save_similarity_heatmaps(epoch, imgf_t, uf_t, sf_t)

            # Linear probe
            if cfg.eval_every > 0 and epoch % cfg.eval_every == 0:
                self._run_eval(epoch)

            # Checkpoints
            self._save_checkpoint(epoch, optimizer, metrics, f'epoch_{epoch:03d}')
            self._save_checkpoint(epoch, optimizer, metrics, 'latest')
            if details['total'] < self._best_total:
                self._best_total = details['total']
                self._save_checkpoint(epoch, optimizer, metrics, 'best_total')
            if metrics['union_consistency'] > self._best_union:
                self._best_union = metrics['union_consistency']
                self._save_checkpoint(epoch, optimizer, metrics, 'best_union')

        self.viz.save_loss_curves(epoch)
        self.viz.save_metric_curves(epoch)

        if cfg.eval_every > 0 and epoch % cfg.eval_every != 0:
            self._run_eval(epoch)

        self._print_final_summary()
        self.logger.info(f"Results: {self.run_dir}")
        self.logger.close()
