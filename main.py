"""
main.py — Entry point training

Cách dùng:
    python main.py
    python main.py --dataset coco --epochs 20 --batch-size 16
    python main.py --dataset cifar10 --img-size 32 --batch-size 64
    python main.py --resume experiments/run_xxx/checkpoints/checkpoint_latest.pt
"""

import argparse
import torch

from src.models.VAE import VAE
from src.models.encoders import build_encoder, ENCODER_REGISTRY
from src.losses.losses import IntersectUnionLoss
from src.trainers.trainer import Trainer, TrainConfig
from src.utils.xla_utils import get_device, is_tpu, device_str, setup_tpu


def parse_args():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument('--dataset',   type=str, default='coco',
                   choices=['coco', 'cifar10', 'cifar100', 'stl10', 'imagenet'])
    p.add_argument('--data-root', type=str, default=None)
    p.add_argument('--img-size',  type=int, default=None)

    # Training
    p.add_argument('--epochs',     type=int,   default=20)
    p.add_argument('--batch-size', type=int,   default=64)   # peak ~6.5GB với grad_checkpoint
    p.add_argument('--grad-accum', type=int,   default=1)
    p.add_argument('--lr',         type=float, default=3e-4)
    p.add_argument('--workers',    type=int,   default=8)
    p.add_argument('--prefetch',   type=int,   default=2,
                   help='DataLoader prefetch_factor (dùng 4+ cho ImageNet)')
    p.add_argument('--q',          type=int,   default=3)
    p.add_argument('--k',          type=int,   default=2,
                   help='Số negative samples (0 = in-batch negatives, nhanh hơn cho ImageNet)')
    p.add_argument('--device',     type=str,   default=device_str(),
                   help='cuda | cpu | xla (TPU — tự động detect nếu torch_xla available)')
    p.add_argument('--no-amp',     action='store_true')
    p.add_argument('--compile',    action='store_true')
    p.add_argument('--skip-decoder',action='store_true',
                   help='Bỏ qua VAE decoder+KL, chỉ train contrastive losses (nhanh hơn, ít VRAM)')

    # Loss weights
    p.add_argument('--beta',          type=float, default=0.1)
    p.add_argument('--lambda-union',  type=float, default=1.0)
    p.add_argument('--lambda-sparse', type=float, default=0.5)
    p.add_argument('--lambda-ortho',  type=float, default=0.1)
    p.add_argument('--lambda-neg',    type=float, default=0.3)

    # Model
    p.add_argument('--encoder',    type=str, default='gated_vae',
                   choices=list(ENCODER_REGISTRY.keys()),
                   help='Backbone encoder để so sánh: gated_vae | resnet18 | resnet50 | efficientnet | mobilenet')
    p.add_argument('--pretrained', action='store_true', default=True,
                   help='Dùng ImageNet pretrained weights cho encoder (không áp dụng với gated_vae)')
    p.add_argument('--latent-ch',  type=int, default=2048)
    p.add_argument('--dim-inter',  type=int, default=1024)
    p.add_argument('--dim-unique', type=int, default=1024)

    # Misc
    p.add_argument('--run-name', type=str, default=None)
    p.add_argument('--exp-dir',  type=str, default='experiments')
    p.add_argument('--resume',   type=str, default=None)
    p.add_argument('--seed',     type=int, default=42)

    return p.parse_args()


DATASET_DEFAULTS = {
    'coco':     {'data_root': 'data/coco2017',  'img_size': 128},
    'cifar10':  {'data_root': 'data',           'img_size': 32},
    'cifar100': {'data_root': 'data',           'img_size': 32},
    'stl10':    {'data_root': 'data',           'img_size': 96},
    'imagenet': {'data_root': 'data/imagenet',  'img_size': 224},
    # ImageNet recommended: --batch-size 32 --grad-accum 8  (eff_batch=256)
}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # TPU v5e: khởi tạo SPMD + bfloat16 trước tất cả operations khác
    if is_tpu():
        setup_tpu(use_bf16=True, use_spmd=True)

    defaults  = DATASET_DEFAULTS[args.dataset]
    data_root = args.data_root or defaults['data_root']
    img_size  = args.img_size  or defaults['img_size']
    run_name  = args.run_name  or f"{args.encoder}_{args.dataset}_is{img_size}"
    # TPU: dùng get_device() từ xla_utils; GPU/CPU: dùng args.device
    device = get_device() if is_tpu() else torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*55}")
    print(f"Encoder : {args.encoder}")
    print(f"Dataset : {args.dataset}  ({data_root})")
    print(f"Img size: {img_size}x{img_size}   Device: {device}")
    print(f"Epochs  : {args.epochs}   LR: {args.lr}   Batch: {args.batch_size} x {args.grad_accum}")
    print(f"{'='*55}\n")

    # Build model
    if args.encoder == 'gated_vae':
        if img_size >= 128:
            model = VAE(
                s1_out=16, s1_heads=4,  s1_blocks=1,
                s2_out=16, s2_heads=8,  s2_blocks=2,
                s3_out=16, s3_heads=16, s3_blocks=2,
                latent_ch=args.latent_ch,
                dec_ch3=128, dec_ch2=64, dec_ch1=32,
                dim_inter=args.dim_inter, dim_unique=args.dim_unique,
                feat_dim=64, hidden_dim=256,
            )
        else:
            model = VAE(
                s1_out=8, s1_heads=4,  s1_blocks=1,
                s2_out=8, s2_heads=8,  s2_blocks=2,
                s3_out=8, s3_heads=16, s3_blocks=2,
                latent_ch=args.latent_ch // 2,
                dec_ch3=64, dec_ch2=32, dec_ch1=16,
                dim_inter=args.dim_inter // 2, dim_unique=args.dim_unique // 2,
                feat_dim=32, hidden_dim=128,
            )
    else:
        model = build_encoder(
            name        = args.encoder,
            latent_ch   = args.latent_ch,
            dim_inter   = args.dim_inter,
            dim_unique  = args.dim_unique,
            pretrained  = args.pretrained,
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}\n")

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Resumed from epoch {ckpt.get('epoch', '?')}: {args.resume}\n")

    loss_fn = IntersectUnionLoss(
        beta          = args.beta,
        lambda_union  = args.lambda_union,
        lambda_sparse = args.lambda_sparse,
        lambda_ortho  = args.lambda_ortho,
        lambda_neg    = args.lambda_neg,
    )

    cfg = TrainConfig(
        run_name      = run_name,
        exp_dir       = args.exp_dir,
        data_root     = data_root,
        dataset       = args.dataset,
        img_size      = img_size,
        q             = args.q,
        k             = args.k,
        batch_size    = args.batch_size,
        grad_accum    = args.grad_accum,
        num_workers   = args.workers,
        prefetch_factor = args.prefetch,
        device        = str(device),
        use_amp       = not args.no_amp,
        compile_model = args.compile,
        use_vae       = not args.skip_decoder,
        total_epochs  = args.epochs,
        lr            = args.lr,
        beta          = args.beta,
        lambda_union  = args.lambda_union,
        lambda_sparse = args.lambda_sparse,
        lambda_ortho  = args.lambda_ortho,
        lambda_neg    = args.lambda_neg,
    )

    Trainer(model, loss_fn, cfg).train()


if __name__ == '__main__':
    main()
