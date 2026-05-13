"""
benchmark.py — Đánh giá chất lượng encoder trên bài toán Intersect-Union Decomposition.

Metrics chất lượng:
  - union_consistency  : v_inter bất biến qua augmentation? (↑, target > 0.85)
  - sparse_divergence  : v_unique khác biệt giữa các aug? (↑, target > 0.70)
  - ortho_score        : v_inter ⊥ v_unique (↓, target < 0.10)
  - neg_separation     : features của ảnh khác cách xa (↑)
  - uniformity         : features phân tán đều, không collapse (↑)
  - quality_score      : tổng hợp → xếp hạng cuối

Checkpoint loading:
  - gated_vae          : tự động quét experiments/ → load checkpoint tốt nhất
  - resnet18/50/...    : ImageNet pretrained (--pretrained) hoặc random init

Cách dùng:
    python benchmark.py                             # gated_vae: auto-scan best ckpt
    python benchmark.py --pretrained                # encoders khác: ImageNet pretrained
    python benchmark.py --exp-dir experiments/      # chỉ định thư mục scan
    python benchmark.py --encoder gated_vae resnet50 --pretrained
    python benchmark.py --n-batches 100             # đánh giá kỹ hơn
"""

import argparse
import gc
import sys
import time
import os
import glob

import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
from src.models.VAE import VAE
from src.models.encoders import build_encoder, ENCODER_REGISTRY
from src.datasets.aug_dataset import AugmentedDataset, CocoImageDataset
from torch.utils.data import DataLoader
import torchvision.datasets as dsets


# ===========================================================================
# Args
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--encoder',    nargs='+', default=list(ENCODER_REGISTRY.keys()))
    p.add_argument('--pretrained', action='store_true', default=False,
                   help='Dùng ImageNet pretrained cho resnet/efficientnet/mobilenet')
    p.add_argument('--exp-dir',    type=str, default='experiments',
                   help='Thư mục chứa các run của gated_vae để auto-scan')

    # Data
    p.add_argument('--dataset',    type=str, default='coco',
                   choices=['coco', 'cifar10', 'stl10'])
    p.add_argument('--coco-split', type=str, default='val',
                   choices=['train', 'val'],
                   help='COCO split để benchmark (val = 5K imgs, nhanh hơn)')
    p.add_argument('--data-root',  type=str, default=None)
    p.add_argument('--img-size',   type=int, default=128)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--n-batches',  type=int, default=30)
    p.add_argument('--q',          type=int, default=3)
    p.add_argument('--k',          type=int, default=2)
    p.add_argument('--workers',    type=int, default=4)

    # Model dims
    p.add_argument('--latent-ch',  type=int, default=2048)
    p.add_argument('--dim-inter',  type=int, default=1024)
    p.add_argument('--dim-unique', type=int, default=1024)

    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed',   type=int, default=42)
    return p.parse_args()


DATASET_DEFAULTS = {
    'coco':    'data/coco2017',
    'cifar10': 'data',
    'stl10':   'data',
}


# ===========================================================================
# Auto-scan best checkpoint
# ===========================================================================

def auto_find_best_checkpoint(exp_dir: str) -> dict | None:
    """
    Quét tất cả run trong exp_dir, tìm checkpoint có union_consistency cao nhất.
    Thứ tự ưu tiên mỗi run: best_union → best_total → latest → epoch_*.
    Trả về dict: {path, run, epoch, tag, metrics} hoặc None.
    """
    if not os.path.isdir(exp_dir):
        return None

    candidates = []
    for run_name in sorted(os.listdir(exp_dir)):
        ckpt_dir = os.path.join(exp_dir, run_name, 'checkpoints')
        if not os.path.isdir(ckpt_dir):
            continue

        # Ưu tiên: best_union > best_total > latest > epoch tốt nhất
        priority = ['best_union', 'best_total', 'latest']
        chosen = None
        for tag in priority:
            p = os.path.join(ckpt_dir, f'checkpoint_{tag}.pt')
            if os.path.exists(p):
                chosen = (p, tag)
                break

        if chosen is None:
            # Tìm epoch_*.pt mới nhất
            epoch_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, 'checkpoint_epoch_*.pt')))
            if epoch_ckpts:
                chosen = (epoch_ckpts[-1], 'epoch_latest')

        if chosen is None:
            continue

        path, tag = chosen
        try:
            ckpt    = torch.load(path, map_location='cpu', weights_only=True)
            metrics = ckpt.get('metrics', {})
            epoch   = ckpt.get('epoch', 0)
            candidates.append({
                'path':    path,
                'run':     run_name,
                'epoch':   epoch,
                'tag':     tag,
                'metrics': metrics,
                'union':   metrics.get('union_consistency', -1.0),
            })
        except Exception:
            continue

    if not candidates:
        return None

    # Sắp xếp theo union_consistency (metric chính)
    candidates.sort(key=lambda c: c['union'], reverse=True)
    return candidates[0]


def print_checkpoint_info(info: dict):
    print(f"    Run   : {info['run']}")
    print(f"    Tag   : {info['tag']}  (epoch {info['epoch']})")
    m = info['metrics']
    if m:
        print(f"    Saved metrics: "
              f"union={m.get('union_consistency', '?'):.3f}  "
              f"sparse={m.get('sparse_divergence', '?'):.3f}  "
              f"ortho={m.get('ortho_score', '?'):.3f}")



# ===========================================================================
# Build encoder
# ===========================================================================

def build_model(name, latent_ch, dim_inter, dim_unique, pretrained):
    if name == 'gated_vae':
        return VAE(
            s1_out=16, s1_heads=4,  s1_blocks=1,
            s2_out=16, s2_heads=8,  s2_blocks=2,
            s3_out=16, s3_heads=16, s3_blocks=2,
            latent_ch=latent_ch, dec_ch3=128, dec_ch2=64, dec_ch1=32,
            dim_inter=dim_inter, dim_unique=dim_unique,
            feat_dim=64, hidden_dim=256,
        )
    return build_encoder(name, latent_ch=latent_ch, dim_inter=dim_inter,
                         dim_unique=dim_unique, pretrained=pretrained)


# ===========================================================================
# Quality metrics
# ===========================================================================

@torch.no_grad()
def compute_batch_metrics(v_inter: torch.Tensor,
                          v_unique: torch.Tensor,
                          image_feat: torch.Tensor,
                          union_neg: torch.Tensor,
                          q: int) -> dict:
    """
    v_inter  : (B*q, dim_inter)   L2-norm
    v_unique : (B*q, dim_unique)  L2-norm
    image_feat:(B*q, feat_dim)    L2-norm
    union_neg: (B*k, dim_inter)   L2-norm — features của negative images
    """
    B = v_inter.shape[0] // q

    # ---- Union consistency: cùng ảnh, aug khác → v_inter phải giống nhau ----
    u = v_inter.view(B, q, -1)            # (B, q, dim_inter)
    u_sims = []
    for i in range(q):
        for j in range(i + 1, q):
            cos = (u[:, i] * u[:, j]).sum(dim=1)   # (B,)
            u_sims.append(cos)
    union_consistency = torch.stack(u_sims).mean().item()

    # ---- Sparse divergence: cùng ảnh, aug khác → v_unique phải khác nhau ----
    s = v_unique.view(B, q, -1)
    s_sims = []
    for i in range(q):
        for j in range(i + 1, q):
            cos = (s[:, i] * s[:, j]).sum(dim=1)
            s_sims.append(cos)
    sparse_divergence = 1.0 - torch.stack(s_sims).mean().item()

    # ---- Ortho score: v_inter ⊥ v_unique (per aug version) ----
    u_mean = u.mean(dim=2)   # (B, q) — scalar proxy
    s_mean = s.mean(dim=2)
    ortho_score = (u_mean * s_mean).abs().mean().item()

    # ---- Neg separation: v_inter của ảnh khác nhau phải cách xa ----
    # Lấy mean v_inter per image: (B, dim_inter)
    u_img = u.mean(dim=1)   # (B, dim_inter)
    if union_neg is not None and union_neg.shape[0] > 0:
        # mean cosine sim giữa positives và negatives — thấp là tốt
        pos_neg_sim = (u_img @ union_neg.mean(dim=0, keepdim=True).T).mean().item()
        neg_separation = 1.0 - abs(pos_neg_sim)
    else:
        neg_separation = 0.0

    # ---- Uniformity: features phân tán đều trên hypersphere ----
    sq_dist_u = 2.0 - 2.0 * (v_inter @ v_inter.T)
    uniformity = -sq_dist_u.clamp(min=0).mul(2.0).exp().mean().log().item()
    # Giá trị càng cao (ít âm hơn) = phân tán tốt hơn

    return {
        'union_consistency': union_consistency,
        'sparse_divergence': sparse_divergence,
        'ortho_score':       ortho_score,
        'neg_separation':    neg_separation,
        'uniformity':        uniformity,
    }


def compute_quality_score(metrics: dict) -> float:
    """
    Tổng hợp thành 1 số [0, 1] để so sánh.
    Heuristic đơn giản:
        score = 0.35*union + 0.30*sparse + 0.15*(1-ortho) + 0.10*neg_sep + 0.10*uniform_norm
    """
    u   = max(0.0, min(1.0, metrics['union_consistency']))
    s   = max(0.0, min(1.0, metrics['sparse_divergence']))
    o   = 1.0 - max(0.0, min(1.0, metrics['ortho_score']))     # đảo chiều
    n   = max(0.0, min(1.0, metrics['neg_separation']))
    uni = max(0.0, min(1.0, (metrics['uniformity'] + 5) / 5))  # shift từ [-5,0] → [0,1]

    return 0.35*u + 0.30*s + 0.15*o + 0.10*n + 0.10*uni


# ===========================================================================
# Evaluate 1 encoder
# ===========================================================================

def evaluate_encoder(name, model, loader, device, n_batches, q, k) -> dict:
    model.eval()

    all_metrics = {
        'union_consistency': [],
        'sparse_divergence': [],
        'ortho_score':       [],
        'neg_separation':    [],
        'uniformity':        [],
    }

    # Throughput
    total_imgs = 0
    total_time = 0.0

    for batch_idx, (core_imgs, neg_imgs) in enumerate(loader):
        if batch_idx >= n_batches:
            break

        B, q_, C, H, W = core_imgs.shape
        _, k_, *_       = neg_imgs.shape

        x_bq = core_imgs.view(B * q_, C, H, W).to(device)
        x_bk = neg_imgs.view(B * k_, C, H, W).to(device)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                _, _, _, img_f, v_inter, v_unique = model(x_bq, skip_decoder=True)
                _, _, _, _,     u_neg,  _         = model(x_bk, skip_decoder=True)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        total_time += time.perf_counter() - t0
        total_imgs += B * q_

        # Kiểm tra NaN
        if any(not torch.isfinite(t).all() for t in [v_inter, v_unique, img_f]):
            continue

        m = compute_batch_metrics(
            v_inter.float(), v_unique.float(),
            img_f.float(), u_neg.float(), q_,
        )
        for key in all_metrics:
            all_metrics[key].append(m[key])

    if not all_metrics['union_consistency']:
        return None   # tất cả NaN

    # Average
    avg = {k: sum(v) / len(v) for k, v in all_metrics.items()}
    avg['throughput']    = total_imgs / total_time
    avg['quality_score'] = compute_quality_score(avg)
    avg['params_M']      = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return avg


# ===========================================================================
# Main
# ===========================================================================

def main():
    args   = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Build dataloader
    data_root = args.data_root or DATASET_DEFAULTS[args.dataset]
    if args.dataset == 'coco':
        base = CocoImageDataset(root=data_root, split=args.coco_split)
    elif args.dataset == 'cifar10':
        base = dsets.CIFAR10(root=data_root, train=False, download=False)
    else:
        base = dsets.STL10(root=data_root, split='test', download=False)

    ds = AugmentedDataset(base, q=args.q, k=args.k, img_size=args.img_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=True)

    print(f"\n{'='*65}")
    print(f"  Quality Benchmark — {args.dataset}  ({len(base)} ảnh)")
    print(f"  Device: {device}   batch={args.batch_size}   {args.n_batches} batches")
    print(f"  q={args.q} augmentations per image,  k={args.k} negatives")
    print(f"{'='*65}\n")
    print(f"  Metrics:")
    print(f"    union_consistency  ↑  (target > 0.85) — aug-invariant union features")
    print(f"    sparse_divergence  ↑  (target > 0.70) — aug-variant unique features")
    print(f"    ortho_score        ↓  (target < 0.10) — v_inter ⊥ v_unique")
    print(f"    neg_separation     ↑  — tách biệt features của các ảnh khác nhau")
    print(f"    uniformity         ↑  — features phân tán đều (chống collapse)")
    print(f"    quality_score      ↑  — tổng hợp 5 metrics (0=tệ, 1=hoàn hảo)")
    print(f"\n{'='*65}\n")

    results = []

    for name in args.encoder:
        if name not in ENCODER_REGISTRY:
            print(f"  [SKIP] Unknown encoder: {name}")
            continue

        # ---- Xác định checkpoint / weights ----
        ckpt_info  = None
        ckpt_label = 'random init'

        if name == 'gated_vae':
            ckpt_info = auto_find_best_checkpoint(args.exp_dir)
            if ckpt_info:
                ckpt_label = f"trained ep{ckpt_info['epoch']} ({ckpt_info['tag']})"
                print(f"\n  Evaluating {name}  [{ckpt_label}]")
                print_checkpoint_info(ckpt_info)
            else:
                ckpt_label = 'random init'
                print(f"\n  Evaluating {name}  [WARN: no checkpoint found in '{args.exp_dir}']")
        else:
            ckpt_label = 'ImageNet pretrained' if args.pretrained else 'random init'
            print(f"\n  Evaluating {name}  [{ckpt_label}]")

        try:
            model = build_model(
                name       = name,
                latent_ch  = args.latent_ch,
                dim_inter  = args.dim_inter,
                dim_unique = args.dim_unique,
                pretrained = args.pretrained if name != 'gated_vae' else False,
            ).to(device)

            if ckpt_info is not None:
                raw = torch.load(ckpt_info['path'], map_location=device, weights_only=True)
                model.load_state_dict(raw['model_state_dict'])

            result = evaluate_encoder(
                name=name, model=model, loader=loader,
                device=device, n_batches=args.n_batches,
                q=args.q, k=args.k,
            )

            if result is None:
                print(f"  → ALL NaN — bỏ qua")
            else:
                result['name']       = name
                result['ckpt_label'] = ckpt_label
                results.append(result)
                print(f"  → quality={result['quality_score']:.3f}  "
                      f"union={result['union_consistency']:.3f}  "
                      f"sparse={result['sparse_divergence']:.3f}  "
                      f"ortho={result['ortho_score']:.3f}")

        except torch.cuda.OutOfMemoryError:
            print(f"  → OOM")
        except Exception as e:
            print(f"  → ERROR: {e}")
        finally:
            try: del model
            except: pass
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    if not results:
        print("\nKhông có kết quả nào!")
        return

    results.sort(key=lambda r: r['quality_score'], reverse=True)

    # ===========================================================================
    # In bảng kết quả
    # ===========================================================================
    cols   = ['Encoder', 'Weights', 'Quality↑', 'Union↑', 'Sparse↑', 'Ortho↓', 'NegSep↑', 'Uniform↑', 'imgs/s', 'Params']
    widths = [20, 24, 9, 8, 8, 8, 9, 9, 8, 8]

    sep = '=' * sum(widths)
    print(f"\n{sep}")
    print(f"  FINAL RESULTS (sorted by Quality Score)")
    print(sep)

    header = ''.join(f'{h:>{w}}' for h, w in zip(cols, widths))
    print(f"\n{header}")
    print('-' * sum(widths))

    for i, r in enumerate(results):
        medal = ['🥇', '🥈', '🥉'][i] if i < 3 else '  '
        row = [
            f"{medal}{r['name']}"[:19],
            r.get('ckpt_label', 'unknown')[:23],
            f"{r['quality_score']:.3f}",
            f"{r['union_consistency']:.3f}",
            f"{r['sparse_divergence']:.3f}",
            f"{r['ortho_score']:.3f}",
            f"{r['neg_separation']:.3f}",
            f"{r['uniformity']:.2f}",
            f"{r['throughput']:.0f}",
            f"{r['params_M']:.1f}M",
        ]
        print(''.join(f'{v:>{w}}' for v, w in zip(row, widths)))

    print('-' * sum(widths))

    # Targets
    print(f"\n  Targets:  union>0.85  sparse>0.70  ortho<0.10")

    # Efficiency: quality per param
    print(f"\n  Quality/Param efficiency:")
    for r in sorted(results, key=lambda r: r['quality_score']/r['params_M'], reverse=True):
        score = r['quality_score'] / r['params_M']
        print(f"    {r['name']:<20} {score:.4f} quality/M_params")

    print(f"\n{'='*65}")
    print(f"  WINNER: {results[0]['name']}  (quality={results[0]['quality_score']:.3f})")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()
