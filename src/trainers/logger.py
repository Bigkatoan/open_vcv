import os
import csv
import time
import torch


class Logger:
    """Terminal + train.log + losses.csv + metrics.csv + iter_losses.csv"""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self._log_f = open(os.path.join(run_dir, 'train.log'), 'w', buffering=1)
        self._prev_total = None
        self._prev_iter_total = None

        self.losses_path      = os.path.join(run_dir, 'losses.csv')
        self.metrics_path     = os.path.join(run_dir, 'metrics.csv')
        self.iter_losses_path = os.path.join(run_dir, 'iter_losses.csv')

        with open(self.losses_path, 'w', newline='') as f:
            csv.writer(f).writerow(
                ['epoch','phase','total','vae','mse','kl',
                 'union','sparse','ortho','neg','uniform','lr','time_s'])

        with open(self.metrics_path, 'w', newline='') as f:
            csv.writer(f).writerow(
                ['epoch','phase','union_consistency',
                 'sparse_divergence','ortho_score','recon_psnr'])

        with open(self.iter_losses_path, 'w', newline='') as f:
            csv.writer(f).writerow(
                ['global_iter','epoch','batch',
                 'total','union','sparse','ortho','neg','uniform',
                 'union_consistency','sparse_divergence','ortho_score','lr'])

    # ------------------------------------------------------------------
    def _write(self, msg: str):
        print(msg)
        self._log_f.write(msg + '\n')

    # ------------------------------------------------------------------
    def print_legend(self, use_vae: bool = True):
        """In một lần giải thích tất cả số liệu lúc bắt đầu train."""
        sep = '=' * 68
        bar = '-' * 68
        vae_block = (
            f"  mse     (↓)  MSE reconstruction loss — ảnh reconstruct vs gốc\n"
            f"             → càng thấp càng tốt (decoder tái tạo tốt hơn)\n"
            f"  kl      (↓)  KL divergence — phân phối latent vs N(0,I)\n"
            f"             → càng thấp càng tốt, nhưng quá thấp → collapse\n"
        ) if use_vae else ''

        msg = (
            f"\n{sep}\n"
            f"  LEGEND — Ý nghĩa các số liệu logging\n"
            f"{bar}\n"
            f"  [LOSSES — tất cả đều tính TRUNG BÌNH trên batch]\n"
            f"{bar}\n"
            f"  total   (↓)  Tổng loss = union + sparse + ortho + neg [+ vae]\n"
            f"             → kỳ vọng giảm dần qua các iteration\n"
            + vae_block +
            f"{bar}\n"
            f"  union   (↓)  Union loss — phạt khi v_inter của cùng 1 ảnh (aug khác)\n"
            f"             khác nhau. Mục tiêu: v_inter bất biến qua augmentation.\n"
            f"             → giảm = model học được feature chung tốt hơn\n"
            f"\n"
            f"  sparse  (↓)  Sparse loss — phạt khi v_unique của cùng 1 ảnh (aug khác)\n"
            f"             quá giống nhau. Mục tiêu: v_unique mã hóa đặc trưng riêng.\n"
            f"             → giảm = model học phân biệt augmentation tốt hơn\n"
            f"\n"
            f"  ortho   (↓)  Ortho loss — phạt khi v_inter và v_unique tương quan.\n"
            f"             Mục tiêu: hai không gian latent độc lập (góc 90°).\n"
            f"             → giảm = phân tách tốt hơn\n"
            f"\n"
            f"  neg     (↓)  Neg loss — phạt khi v_inter của ảnh KHÁC nhau gần nhau.\n"
            f"             Mục tiêu: feature của các ảnh khác nhau phải cách xa.\n"
            f"             → giảm = model phân biệt ảnh tốt hơn\n"
            f"\n"
            f"  uniform (↓)  Uniformity loss — phạt khi features tập trung cụm.\n"
            f"             Mục tiêu: feature phân bố đều trên hypersphere.\n"
            f"             → giảm = tránh feature collapse\n"
            f"{bar}\n"
            f"  [METRICS — tính trên batch hiện tại, dùng để track tiến độ]\n"
            f"{bar}\n"
            f"  union_consistency  (↑ > 0.85)\n"
            f"    Cosine similarity trung bình của v_inter giữa các aug của cùng 1 ảnh.\n"
            f"    = 1.0 → v_inter hoàn toàn bất biến qua augmentation (tốt nhất)\n"
            f"    = 0.0 → v_inter ngẫu nhiên, không học được gì\n"
            f"\n"
            f"  sparse_divergence   (↑ > 0.70)\n"
            f"    1 - cosine similarity của v_unique giữa các aug của cùng 1 ảnh.\n"
            f"    = 1.0 → v_unique hoàn toàn khác nhau giữa các aug (tốt nhất)\n"
            f"    = 0.0 → v_unique giống nhau, không mã hóa được gì riêng\n"
            f"\n"
            f"  ortho_score         (↓ < 0.10)\n"
            f"    Dot product trung bình giữa v_inter và v_unique.\n"
            f"    = 0.0 → hai không gian vuông góc hoàn toàn (tốt nhất)\n"
            f"    > 0.2 → hai không gian overlap, cần regularize mạnh hơn\n"
            f"{sep}\n"
        )
        self._write(msg)

    # ------------------------------------------------------------------
    def log_iter(self,
                 epoch: int, total_epochs: int,
                 batch_idx: int, n_batches: int,
                 global_iter: int,
                 details: dict,
                 batch_metrics: dict,
                 lr: float,
                 use_vae: bool = True):
        """Compact per-iteration log line (every N iters)."""
        delta = ''
        cur = details['total']
        if self._prev_iter_total is not None:
            diff = self._prev_iter_total - cur
            arrow = '↓' if diff > 0 else '↑'
            delta = f" {arrow}{abs(diff):.4f}"
        self._prev_iter_total = cur

        uc  = batch_metrics.get('union_consistency', 0.0)
        sd  = batch_metrics.get('sparse_divergence', 0.0)
        ort = batch_metrics.get('ortho_score', 0.0)

        if use_vae:
            core = (
                f"loss {cur:.4f}{delta}  "
                f"mse {details['mse']:.4f}  "
                f"union {details['union']:.4f}  "
                f"sparse {details['sparse']:.4f}  "
                f"ortho {details['ortho']:.4f}"
            )
        else:
            core = (
                f"loss {cur:.4f}{delta}  "
                f"union↓ {details['union']:.4f}  "
                f"sparse↓ {details['sparse']:.4f}  "
                f"ortho↓ {details['ortho']:.4f}  "
                f"neg↓ {details['neg']:.4f}"
            )

        metrics_str = (
            f"  [metrics]  "
            f"consistency↑ {uc:.3f}  "
            f"divergence↑ {sd:.3f}  "
            f"ortho↓ {ort:.3f}"
        )

        msg = (
            f"[Ep {epoch:03d}/{total_epochs}  "
            f"it {batch_idx+1:5d}/{n_batches}  "
            f"glob {global_iter:7d}]  "
            f"lr={lr:.2e}  {core}\n{metrics_str}"
        )
        self._write(msg)

        with open(self.iter_losses_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                global_iter, epoch, batch_idx + 1,
                f"{details['total']:.6f}",
                f"{details['union']:.6f}",
                f"{details['sparse']:.6f}",
                f"{details['ortho']:.6f}",
                f"{details['neg']:.6f}",
                f"{details.get('uniform', 0.0):.6f}",
                f"{uc:.6f}", f"{sd:.6f}", f"{ort:.6f}",
                f"{lr:.7f}",
            ])

    # ------------------------------------------------------------------
    def log_epoch(self, epoch: int, total_epochs: int, phase: int,
                  details: dict, metrics: dict, lr: float, elapsed: float,
                  use_vae: bool = True):
        delta = ''
        if self._prev_total is not None:
            diff = self._prev_total - details['total']
            arrow = '↓' if diff > 0 else '↑'
            delta = f"({arrow} {abs(diff):.4f})"
        self._prev_total = details['total']

        sep = '=' * 62
        bar = '-' * 62

        if use_vae:
            body = (
                f"  total  : {details['total']:.4f}  {delta}\n"
                f"  vae    : {details['vae']:.4f}  "
                f"mse={details['mse']:.4f}  kl={details['kl']:.4f}\n"
                f"  union  : {details['union']:.4f}\n"
                f"  sparse : {details['sparse']:.4f}\n"
                f"  ortho  : {details['ortho']:.4f}\n"
                f"  neg    : {details['neg']:.4f}\n"
                f"{bar}\n"
                f"  union_consistency : {metrics['union_consistency']:.4f}  (> 0.85)\n"
                f"  sparse_divergence : {metrics['sparse_divergence']:.4f}  (> 0.70)\n"
                f"  ortho_score       : {metrics['ortho_score']:.4f}  (< 0.10)\n"
                f"  recon_psnr        : {metrics['recon_psnr']:.2f} dB  (> 25 dB)\n"
            )
        else:
            # skip-decoder mode: chỉ show contrastive losses + metrics
            body = (
                f"  total  : {details['total']:.4f}  {delta}\n"
                f"  union  : {details['union']:.4f}   "
                f"sparse : {details['sparse']:.4f}\n"
                f"  ortho  : {details['ortho']:.4f}   "
                f"neg    : {details['neg']:.4f}\n"
                f"  uniform: {details.get('uniform', 0.0):.4f}\n"
                f"{bar}\n"
                f"  union_consistency : {metrics['union_consistency']:.4f}  (> 0.85)\n"
                f"  sparse_divergence : {metrics['sparse_divergence']:.4f}  (> 0.70)\n"
                f"  ortho_score       : {metrics['ortho_score']:.4f}  (< 0.10)\n"
            )

        msg = (
            f"\n{sep}\n"
            f"Epoch {epoch:03d}/{total_epochs}  lr={lr:.2e}  "
            f"{'[VAE]' if use_vae else '[Contrastive]'}  Time={elapsed:.1f}s\n"
            f"{bar}\n"
            + body
            + sep
        )
        self._write(msg)

        with open(self.losses_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch, phase,
                f"{details['total']:.6f}", f"{details['vae']:.6f}",
                f"{details['mse']:.6f}",  f"{details['kl']:.6f}",
                f"{details['union']:.6f}",f"{details['sparse']:.6f}",
                f"{details['ortho']:.6f}",f"{details['neg']:.6f}",
                f"{details.get('uniform', 0.0):.6f}",
                f"{lr:.7f}", f"{elapsed:.1f}",
            ])

        with open(self.metrics_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch, phase,
                f"{metrics['union_consistency']:.6f}",
                f"{metrics['sparse_divergence']:.6f}",
                f"{metrics['ortho_score']:.6f}",
                f"{metrics['recon_psnr']:.4f}",
            ])

    def info(self, msg: str):
        self._write(msg)

    def close(self):
        self._log_f.close()
