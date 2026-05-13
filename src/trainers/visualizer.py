import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Visualizer:
    """Saves PNG visualizations to run_dir/viz/ (single flat directory, files overwritten)."""

    def __init__(self, run_dir: str):
        self.viz_dir = os.path.join(run_dir, 'viz')
        os.makedirs(self.viz_dir, exist_ok=True)

        self._losses:  list[dict] = []
        self._metrics: list[dict] = []
        self._phases:  list[tuple] = []   # (epoch, phase)
        self._iter_records: list[dict] = []

    # ------------------------------------------------------------------
    def record(self, epoch, phase, details, metrics):
        self._losses.append({'epoch': epoch, **details})
        self._metrics.append({'epoch': epoch, **metrics})
        if not self._phases or self._phases[-1][1] != phase:
            self._phases.append((epoch, phase))

    # ------------------------------------------------------------------
    def record_iter(self, global_iter: int, epoch: int,
                    details: dict, batch_metrics: dict):
        self._iter_records.append({
            'iter':  global_iter,
            'epoch': epoch,
            **{k: v for k, v in details.items()},
            'union_consistency': batch_metrics.get('union_consistency', 0.0),
            'sparse_divergence': batch_metrics.get('sparse_divergence', 0.0),
            'ortho_score':       batch_metrics.get('ortho_score', 0.0),
        })

    # ------------------------------------------------------------------
    def save_iter_curves(self, global_iter: int, window: int = 20):
        if len(self._iter_records) < 2:
            return

        def ema(vals, alpha=None):
            if alpha is None:
                alpha = 2.0 / (window + 1)
            out, s = [], vals[0]
            for v in vals:
                s = alpha * v + (1 - alpha) * s
                out.append(s)
            return out

        iters  = [r['iter']           for r in self._iter_records]
        total  = [r['total']          for r in self._iter_records]
        union  = [r.get('union',  0)  for r in self._iter_records]
        sparse = [r.get('sparse', 0)  for r in self._iter_records]
        ortho  = [r.get('ortho',  0)  for r in self._iter_records]
        neg    = [r.get('neg',    0)  for r in self._iter_records]
        uc     = [r['union_consistency'] for r in self._iter_records]
        sd     = [r['sparse_divergence'] for r in self._iter_records]
        ort    = [r['ortho_score']       for r in self._iter_records]

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle(f'Training Progress  (iter {global_iter:,})', fontsize=12, fontweight='bold')

        ax = axes[0, 0]
        ax.plot(iters, total,      color='lightblue', alpha=0.35, lw=0.8, label='raw')
        ax.plot(iters, ema(total), color='steelblue', lw=2.0,    label=f'EMA-{window}')
        ax.set_title('Total Loss  ↓', fontsize=10)
        ax.set_xlabel('Iteration'); ax.set_ylabel('Loss')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = axes[0, 1]
        for vals, col, lbl in [
            (union,  'green',  'union  ↓'),
            (sparse, 'orange', 'sparse ↓'),
            (ortho,  'purple', 'ortho  ↓'),
            (neg,    'red',    'neg    ↓'),
        ]:
            ax.plot(iters, vals,      color=col, alpha=0.25, lw=0.8)
            ax.plot(iters, ema(vals), color=col, lw=1.8, label=lbl)
        ax.set_title('Component Losses  ↓', fontsize=10)
        ax.set_xlabel('Iteration'); ax.set_ylabel('Loss')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = axes[1, 0]
        ax.plot(iters, uc,      color='lightgreen', alpha=0.35, lw=0.8)
        ax.plot(iters, ema(uc), color='green',      lw=2.0, label='union_consistency  ↑')
        ax.axhline(0.85, color='green',  ls='--', alpha=0.5, lw=1, label='target 0.85')
        ax.plot(iters, sd,      color='bisque', alpha=0.35, lw=0.8)
        ax.plot(iters, ema(sd), color='orange', lw=2.0, label='sparse_divergence  ↑')
        ax.axhline(0.70, color='orange', ls='--', alpha=0.5, lw=1, label='target 0.70')
        ax.set_ylim(-0.05, 1.05)
        ax.set_title('Consistency & Divergence  ↑', fontsize=10)
        ax.set_xlabel('Iteration'); ax.set_ylabel('Score [0, 1]')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = axes[1, 1]
        ax.plot(iters, ort,      color='thistle', alpha=0.35, lw=0.8)
        ax.plot(iters, ema(ort), color='purple',  lw=2.0, label='ortho_score  ↓')
        ax.axhline(0.10, color='purple', ls='--', alpha=0.5, lw=1, label='target < 0.10')
        ax.set_title('Ortho Score  ↓', fontsize=10)
        ax.set_xlabel('Iteration'); ax.set_ylabel('Score')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        plt.tight_layout()
        self._save('iter_curves.png', fig)

    # ------------------------------------------------------------------
    def save_recon(self, epoch: int, inputs: torch.Tensor, recons: torch.Tensor, psnr: float):
        q = inputs.shape[0]
        fig, axes = plt.subplots(2, q, figsize=(3 * q, 6))
        axes = axes.reshape(2, q)
        aug_names = ['Conservative', 'Moderate', 'Aggressive']
        for i in range(q):
            for row, t in enumerate([inputs, recons]):
                img = t[i].permute(1, 2, 0).cpu().float().numpy().clip(0, 1)
                axes[row, i].imshow(img)
                axes[row, i].axis('off')
            axes[0, i].set_title(aug_names[i] if i < 3 else f'Aug{i+1}', fontsize=9)
        axes[0, 0].set_ylabel('Input', fontsize=10)
        axes[1, 0].set_ylabel('Recon', fontsize=10)
        fig.suptitle(f'Epoch {epoch:04d} | PSNR {psnr:.2f} dB', fontsize=11)
        plt.tight_layout()
        self._save('recon.png', fig)

    # ------------------------------------------------------------------
    def save_loss_curves(self, epoch: int):
        if len(self._losses) < 2:
            return
        eps = [h['epoch'] for h in self._losses]
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 8))

        for key, col, lbl in [('total', 'k', 'Total'), ('vae', 'b', 'VAE'), ('mse', 'c', 'MSE')]:
            a1.plot(eps, [h[key] for h in self._losses], color=col, label=lbl, lw=1.5)
        a1.set_ylabel('Loss'); a1.legend(); a1.grid(alpha=.3); a1.set_title('Loss Curves')

        for key, col, lbl in [('union', 'g', 'Union'), ('sparse', 'orange', 'Sparse'),
                               ('ortho', 'purple', 'Ortho'), ('neg', 'r', 'Neg')]:
            a2.plot(eps, [h[key] for h in self._losses], color=col, label=lbl, lw=1.5)
        a2.set_ylabel('Loss'); a2.set_xlabel('Epoch'); a2.legend(); a2.grid(alpha=.3)

        for ax in [a1, a2]:
            for ep, ph in self._phases[1:]:
                ax.axvline(x=ep, color='gray', ls='--', alpha=.5, lw=1)
                ax.text(ep + 0.3, ax.get_ylim()[1] * .95, f'P{ph}', fontsize=8, color='gray')

        plt.tight_layout()
        self._save('losses.png', fig)

    # ------------------------------------------------------------------
    def save_metric_curves(self, epoch: int):
        if len(self._metrics) < 2:
            return
        eps = [h['epoch'] for h in self._metrics]
        fig, ax = plt.subplots(figsize=(10, 5))

        for key, col, tgt, lbl in [
            ('union_consistency', 'g',      0.85, 'Union Consistency'),
            ('sparse_divergence', 'orange', 0.70, 'Sparse Divergence'),
            ('ortho_score',       'purple', 0.10, 'Ortho Score'),
        ]:
            ax.plot(eps, [h[key] for h in self._metrics], color=col, label=lbl, lw=1.5)
            ax.axhline(y=tgt, color=col, ls='--', alpha=.4, lw=1)

        ax.set_ylabel('Score'); ax.set_xlabel('Epoch')
        ax.legend(); ax.grid(alpha=.3); ax.set_title('Evaluation Metrics')
        plt.tight_layout()
        self._save('metrics.png', fig)

    # ------------------------------------------------------------------
    def save_similarity_heatmaps(self, epoch: int,
                                  image_feat: torch.Tensor,
                                  union_feat: torch.Tensor,
                                  sparse_feat: torch.Tensor):
        with torch.no_grad():
            mats = [
                (image_feat @ image_feat.T,  'Aug Similarity (w_ij)', 'Blues'),
                (union_feat  @ union_feat.T,  'Union Feature Sim',     'Greens'),
                (sparse_feat @ sparse_feat.T, 'Sparse Feature Sim',    'Oranges'),
            ]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        q = image_feat.shape[0]
        ticks = [f'aug{i+1}' for i in range(q)]

        for ax, (mat, title, cmap) in zip(axes, mats):
            m = mat.cpu().numpy()
            im = ax.imshow(m, cmap=cmap, vmin=-1, vmax=1)
            ax.set_title(f'{title}\nEpoch {epoch}', fontsize=10)
            ax.set_xticks(range(q)); ax.set_xticklabels(ticks, fontsize=8)
            ax.set_yticks(range(q)); ax.set_yticklabels(ticks, fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            for i in range(q):
                for j in range(q):
                    ax.text(j, i, f'{m[i,j]:.2f}', ha='center', va='center',
                            fontsize=7, color='white' if abs(m[i,j]) > 0.5 else 'black')

        plt.tight_layout()
        self._save('similarity.png', fig)

    # ------------------------------------------------------------------
    def save_tsne(self, epoch: int, feats: torch.Tensor, labels: list, title: str, fname: str):
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            return

        X = feats.cpu().numpy()
        if X.shape[0] < 10:
            return

        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, X.shape[0] // 2))
        X2d  = tsne.fit_transform(X)

        fig, ax = plt.subplots(figsize=(8, 6))
        unique_labels = list(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        for lbl, col in zip(unique_labels, colors):
            mask = [l == lbl for l in labels]
            ax.scatter(X2d[mask, 0], X2d[mask, 1], c=[col], label=str(lbl), s=15, alpha=0.7)
        ax.legend(loc='best', fontsize=8, markerscale=2)
        ax.set_title(f'{title} — Epoch {epoch}', fontsize=11)
        ax.axis('off')
        plt.tight_layout()
        self._save(fname, fig)

    # ------------------------------------------------------------------
    def _save(self, filename: str, fig):
        path = os.path.join(self.viz_dir, filename)
        fig.savefig(path, dpi=100, bbox_inches='tight')
        plt.close(fig)
