"""
gradcam_analysis.py — Evaluate perceptual alignment of Gated VAE latent partitions
via GradCAM++ against COCO instance segmentation masks.

Metrics computed:
    Seg-IoU      : IoU between thresholded CAM (>50th pct) and COCO binary mask
    Pointing Game: Is argmax(CAM) inside the COCO mask?
    CAM Stability: 1 - mean(||CAM_aug_i - CAM_aug_j||_F / (H*W))  across aug pairs

Usage:
    python scripts/gradcam_analysis.py \\
        --exp-dir experiments/gated_vae_imagenet_is224_XXXX \\
        --n-images 200 \\
        --output experiments/gated_vae_imagenet_is224_XXXX/viz/gradcam

    # With custom annotation / image dirs
    python scripts/gradcam_analysis.py \\
        --exp-dir experiments/<run> \\
        --ann-file data/coco2017/annotations/instances_val2017.json \\
        --img-dir  data/coco2017/val2017 \\
        --n-images 200
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.VAE import VAE
from src.utils.gradcam import GradCAMPlusPlus
from src.utils.coco_seg_utils import load_coco_masks


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--exp-dir',   type=str, required=True,
                   help='Path to experiment run dir (contains checkpoints/)')
    p.add_argument('--n-images',  type=int, default=200,
                   help='Number of COCO val images to evaluate')
    p.add_argument('--n-aug',     type=int, default=3,
                   help='Number of augmented views per image for stability metric')
    p.add_argument('--output',    type=str, default=None,
                   help='Output dir (default: <exp-dir>/viz/gradcam)')
    p.add_argument('--ann-file',  type=str,
                   default='data/coco2017/annotations/instances_val2017.json')
    p.add_argument('--img-dir',   type=str,
                   default='data/coco2017/val2017')
    p.add_argument('--img-size',  type=int, default=224,
                   help='Resize images to this size before inference')
    p.add_argument('--min-area',  type=float, default=1024.0,
                   help='Minimum COCO annotation area (px²) to include')
    p.add_argument('--seed',      type=int, default=42)
    p.add_argument('--device',    type=str, default='cuda')
    p.add_argument('--save-images', action='store_true',
                   help='Save per-image PNG grids (slow — for qualitative inspection)')
    return p.parse_args()


# ===========================================================================
# Model loading
# ===========================================================================

def load_model(exp_dir: str, device: torch.device) -> VAE:
    """Auto-load best checkpoint from exp_dir/checkpoints/."""
    import glob
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    assert os.path.isdir(ckpt_dir), f"checkpoints/ not found in {exp_dir}"

    for tag in ['checkpoint_best_union.pt', 'checkpoint_best_total.pt', 'checkpoint_latest.pt']:
        path = os.path.join(ckpt_dir, tag)
        if os.path.exists(path):
            break
    else:
        epoch_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, 'checkpoint_epoch_*.pt')))
        assert epoch_ckpts, f"No checkpoints found in {ckpt_dir}"
        path = epoch_ckpts[-1]

    print(f"[gradcam] Loading checkpoint: {os.path.basename(path)}", flush=True)
    ckpt = torch.load(path, map_location='cpu', weights_only=True)

    model = VAE(
        s1_out=16, s1_heads=4,  s1_blocks=1,
        s2_out=16, s2_heads=8,  s2_blocks=2,
        s3_out=16, s3_heads=16, s3_blocks=2,
        latent_ch=2048, dec_ch3=128, dec_ch2=64, dec_ch1=32,
        dim_inter=1024, dim_unique=1024,
        feat_dim=64, hidden_dim=256,
    )
    state = ckpt.get('model_state', ckpt)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    print(f"[gradcam] Model loaded (epoch {ckpt.get('epoch', '?')})")
    return model


# ===========================================================================
# Augmentation transforms for stability evaluation
# ===========================================================================

def build_aug_transforms(img_size: int, n_aug: int) -> list:
    """Return list of n_aug different augmentation transforms (all return tensor)."""
    to_tensor = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])

    augs = [
        to_tensor,
        T.Compose([T.Resize(img_size + 16), T.RandomCrop(img_size), T.ToTensor()]),
        T.Compose([T.RandomResizedCrop(img_size, scale=(0.7, 1.0)), T.ToTensor()]),
        T.Compose([T.Resize((img_size, img_size)), T.RandomHorizontalFlip(1.0), T.ToTensor()]),
        T.Compose([
            T.Resize((img_size, img_size)),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.ToTensor(),
        ]),
    ]
    return augs[:n_aug]


# ===========================================================================
# Metrics
# ===========================================================================

def seg_iou(cam: np.ndarray, mask: np.ndarray, threshold_pct: float = 50.0) -> float:
    """
    IoU between binary thresholded CAM and COCO binary mask.
    cam  : (H, W) float in [0, 1]
    mask : (H, W) bool
    """
    thresh = np.percentile(cam, threshold_pct)
    cam_bin = cam >= thresh
    intersection = (cam_bin & mask).sum()
    union        = (cam_bin | mask).sum()
    return float(intersection) / float(union + 1e-8)


def pointing_game(cam: np.ndarray, mask: np.ndarray) -> bool:
    """True if argmax of CAM falls inside the ground-truth mask."""
    idx = np.unravel_index(np.argmax(cam), cam.shape)
    return bool(mask[idx])


def cam_stability(cams: list[np.ndarray]) -> float:
    """
    1 - mean pairwise normalised Frobenius distance.
    cams: list of (H, W) float arrays in [0, 1]
    """
    if len(cams) < 2:
        return 1.0
    H, W = cams[0].shape
    dists = []
    for i in range(len(cams)):
        for j in range(i + 1, len(cams)):
            diff = np.abs(cams[i].astype(np.float32) - cams[j].astype(np.float32))
            dists.append(diff.mean())          # mean absolute difference
    return float(1.0 - np.mean(dists))


# ===========================================================================
# Visualisation helpers
# ===========================================================================

def _overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Overlay jet colormap heatmap on RGB image."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm

    img_arr = np.array(image.resize((cam.shape[1], cam.shape[0]))).astype(np.float32) / 255.0
    heat    = cm.jet(cam)[..., :3]             # (H, W, 3)
    blended = (1 - alpha) * img_arr + alpha * heat
    blended = (np.clip(blended, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(blended)


def save_grid(
    out_path: str,
    image: Image.Image,
    cam_inter:  np.ndarray,
    cam_unique: np.ndarray,
    mask:       np.ndarray,
):
    """Save 4-panel PNG: original | cam_inter | cam_unique | gt_mask."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[1].imshow(_overlay_heatmap(image, cam_inter))
    axes[1].set_title('CAM Inter (||v_inter||²)')
    axes[2].imshow(_overlay_heatmap(image, cam_unique))
    axes[2].set_title('CAM Unique (||v_unique||²)')
    axes[3].imshow(mask.astype(np.uint8) * 255, cmap='gray')
    axes[3].set_title('GT Mask (COCO)')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================

def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = args.output or os.path.join(args.exp_dir, 'viz', 'gradcam')
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    model  = load_model(args.exp_dir, device)
    cam_fn = GradCAMPlusPlus(model)

    # Load COCO masks
    samples = load_coco_masks(
        ann_file  = args.ann_file,
        img_dir   = args.img_dir,
        n         = args.n_images,
        min_area  = args.min_area,
        seed      = args.seed,
    )
    if not samples:
        print("[gradcam] No samples found — check annotation file and image directory.")
        return

    augs = build_aug_transforms(args.img_size, args.n_aug)
    to_tensor = T.Compose([T.Resize((args.img_size, args.img_size)), T.ToTensor()])

    # Collect per-image metrics
    rows_inter  = []
    rows_unique = []

    print(f"\n[gradcam] Evaluating {len(samples)} images on {device} …\n")

    for idx, sample in enumerate(samples):
        img_pil = Image.open(sample['image_path']).convert('RGB')
        H_orig, W_orig = sample['mask'].shape

        # Resize mask to img_size for IoU/pointing game (matching CAM resolution)
        from PIL import Image as PILImage
        mask_pil    = PILImage.fromarray(sample['mask'].astype(np.uint8) * 255)
        mask_resized = np.array(
            mask_pil.resize((args.img_size, args.img_size), PILImage.NEAREST)
        ) > 0

        # Compute CAMs for n_aug views
        cams_inter_list  = []
        cams_unique_list = []

        for aug in augs:
            x = aug(img_pil).unsqueeze(0).to(device)   # (1, 3, H, W)
            ci, cu = cam_fn.compare(x)                  # (H, W) on CPU
            cams_inter_list.append(ci.numpy())
            cams_unique_list.append(cu.numpy())

        # Use first (canonical) CAM for IoU / pointing game
        ci0 = cams_inter_list[0]
        cu0 = cams_unique_list[0]

        iou_inter   = seg_iou(ci0, mask_resized)
        iou_unique  = seg_iou(cu0, mask_resized)
        pg_inter    = pointing_game(ci0, mask_resized)
        pg_unique   = pointing_game(cu0, mask_resized)
        stab_inter  = cam_stability(cams_inter_list)
        stab_unique = cam_stability(cams_unique_list)

        rows_inter.append({
            'image_id':  sample['image_id'],
            'seg_iou':   iou_inter,
            'pointing':  int(pg_inter),
            'stability': stab_inter,
        })
        rows_unique.append({
            'image_id':  sample['image_id'],
            'seg_iou':   iou_unique,
            'pointing':  int(pg_unique),
            'stability': stab_unique,
        })

        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"  [{idx+1:3d}/{len(samples)}]  "
                  f"IoU inter={iou_inter:.3f}  unique={iou_unique:.3f}  "
                  f"PG inter={int(pg_inter)}  unique={int(pg_unique)}")

        # Optionally save PNG
        if args.save_images:
            fname = os.path.join(out_dir, f"cam_{sample['image_id']:012d}.png")
            save_grid(fname, img_pil, ci0, cu0, mask_resized)

    # Aggregate
    def agg(rows: list[dict]) -> dict:
        return {
            'seg_iou_mean':    float(np.mean([r['seg_iou']   for r in rows])),
            'seg_iou_std':     float(np.std( [r['seg_iou']   for r in rows])),
            'pointing_acc':    float(np.mean([r['pointing']  for r in rows])),
            'stability_mean':  float(np.mean([r['stability'] for r in rows])),
        }

    agg_inter  = agg(rows_inter)
    agg_unique = agg(rows_unique)

    print(f"\n{'='*60}")
    print(f"  GradCAM++ Results  ({len(samples)} images, {args.n_aug} aug views)")
    print(f"{'='*60}")
    print(f"  {'Metric':<22} {'v_inter':>10} {'v_unique':>10}")
    print(f"  {'-'*44}")
    print(f"  {'Seg-IoU (mean):':<22} {agg_inter['seg_iou_mean']:>10.4f} {agg_unique['seg_iou_mean']:>10.4f}")
    print(f"  {'Seg-IoU (std):':<22} {agg_inter['seg_iou_std']:>10.4f}  {agg_unique['seg_iou_std']:>10.4f}")
    print(f"  {'Pointing Game Acc:':<22} {agg_inter['pointing_acc']:>10.4f} {agg_unique['pointing_acc']:>10.4f}")
    print(f"  {'CAM Stability:':<22} {agg_inter['stability_mean']:>10.4f} {agg_unique['stability_mean']:>10.4f}")
    print(f"{'='*60}\n")

    # Save per-image CSV
    for name, rows in [('inter', rows_inter), ('unique', rows_unique)]:
        csv_path = os.path.join(out_dir, f'gradcam_metrics_{name}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['image_id', 'seg_iou', 'pointing', 'stability'])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved: {csv_path}")

    # Save aggregate summary
    summary_path = os.path.join(out_dir, 'gradcam_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'v_inter', 'v_unique'])
        for k in ['seg_iou_mean', 'seg_iou_std', 'pointing_acc', 'stability_mean']:
            writer.writerow([k, f"{agg_inter[k]:.4f}", f"{agg_unique[k]:.4f}"])
    print(f"  Summary: {summary_path}\n")


if __name__ == '__main__':
    main()
