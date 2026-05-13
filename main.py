"""
main.py — Entry point training (TensorFlow)

Cách dùng:
    python main.py

Chỉnh cấu hình trực tiếp ở phần CONFIG bên dưới.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # ẩn TF info logs

import tensorflow as tf

from src.models.VAE import VAE
from src.models.encoders import build_encoder
from src.losses.losses import IntersectUnionLoss
from src.trainers.trainer import Trainer, TrainConfig


# ===========================================================================
# CONFIG — chỉnh trực tiếp ở đây
# ===========================================================================

ENCODER      = 'gated_vae'   # 'gated_vae' | 'resnet50' | 'resnet18' | 'efficientnet' | 'mobilenet'
DATASET      = 'imagenet'    # 'imagenet' | 'coco' | 'cifar10' | 'cifar100'
IMG_SIZE     = 128
DATA_ROOT    = None           # None = dùng default

EPOCHS       = 30
BATCH_SIZE   = 32
GRAD_ACCUM   = 1
LR           = 3e-4
WORKERS      = 8
PREFETCH     = 4

USE_AMP      = True
SKIP_DECODER = True           # False = bật VAE decoder+KL

Q = 3   # số augmentation views
K = 0   # negative samples (0 = in-batch)

# Model dims — DIM_INTER + DIM_UNIQUE phải == LATENT_CH
LATENT_CH    = 192   # 128 + 64
DIM_INTER    = 128
DIM_UNIQUE   = 64

# Loss weights
BETA          = 0.1
LAMBDA_UNION  = 1.0
LAMBDA_SPARSE = 0.5
LAMBDA_ORTHO  = 0.1
LAMBDA_NEG    = 0.3

WARMUP_EPOCHS = 5

# Linear probe evaluation
EVAL_EVERY        = 10
EVAL_DATASET      = 'coco'
EVAL_DATA_ROOT    = 'data/coco2017'   # root chứa train2017/, val2017/, annotations/
EVAL_PROBE_EPOCHS = 20

# Misc
SEED     = 42
EXP_DIR  = 'experiments'
RUN_NAME = None   # None = tự động


# ===========================================================================
# Dataset defaults
# ===========================================================================

DATASET_DEFAULTS = {
    'imagenet': {'data_root': 'data/imagenet',       'img_size': 128},
    'coco':     {'data_root': 'data/coco2017',        'img_size': 128},
    'cifar10':  {'data_root': 'data',                 'img_size': 32},
    'cifar100': {'data_root': 'data',                 'img_size': 32},
}


def main():
    tf.random.set_seed(SEED)

    defaults  = DATASET_DEFAULTS[DATASET]
    data_root = DATA_ROOT or defaults['data_root']
    img_size  = IMG_SIZE  or defaults['img_size']
    run_name  = RUN_NAME  or f"{ENCODER}_{DATASET}_is{img_size}"

    gpus = tf.config.list_physical_devices('GPU')
    device_str = f"GPU × {len(gpus)}" if gpus else "CPU"

    print(f"\n{'='*55}")
    print(f"Framework: TensorFlow {tf.__version__}")
    print(f"Encoder  : {ENCODER}")
    print(f"Dataset  : {DATASET}  ({data_root})")
    print(f"Img size : {img_size}x{img_size}   Device: {device_str}")
    print(f"Epochs   : {EPOCHS}   LR: {LR}   Batch: {BATCH_SIZE} x {GRAD_ACCUM}")
    print(f"{'='*55}\n")

    # Build model
    if ENCODER == 'gated_vae':
        if img_size >= 128:
            # Stage 1 nhỏ (128×128 spatial là bottleneck) → stage 3 to (32×32, rẻ)
            # Peak tensor: (B*q, 128, 128, 96) ≈ 150 MB  →  tổng ~6 GB trên 3060 12GB
            model = VAE(
                s1_out=8,  s1_heads=4,  s1_blocks=1,   # 32ch tại 128×128
                s2_out=8,  s2_heads=8,  s2_blocks=2,   # 64ch tại 64×64
                s3_out=16, s3_heads=16, s3_blocks=2,   # 256ch tại 32×32
                latent_ch=LATENT_CH,
                dec_ch3=128, dec_ch2=64, dec_ch1=32,
                dim_inter=DIM_INTER, dim_unique=DIM_UNIQUE,
                feat_dim=64, hidden_dim=256,
            )
        else:
            # ch progression: 16 → 32 → 128
            model = VAE(
                s1_out=4,  s1_heads=4,  s1_blocks=1,
                s2_out=4,  s2_heads=8,  s2_blocks=2,
                s3_out=8,  s3_heads=16, s3_blocks=2,
                latent_ch=LATENT_CH // 2,
                dec_ch3=64, dec_ch2=32, dec_ch1=16,
                dim_inter=DIM_INTER // 2, dim_unique=DIM_UNIQUE // 2,
                feat_dim=32, hidden_dim=128,
            )
    else:
        model = build_encoder(
            name       = ENCODER,
            latent_ch  = LATENT_CH,
            dim_inter  = DIM_INTER,
            dim_unique = DIM_UNIQUE,
            pretrained = True,
        )

    # Warm-up build with dummy input
    dummy = tf.zeros((2, img_size, img_size, 3))
    model(dummy, skip_decoder=True, training=False)
    n_params = model.count_parameters()
    print(f"Model params: {n_params:,}\n")

    loss_fn = IntersectUnionLoss(
        beta          = BETA,
        lambda_union  = LAMBDA_UNION,
        lambda_sparse = LAMBDA_SPARSE,
        lambda_ortho  = LAMBDA_ORTHO,
        lambda_neg    = LAMBDA_NEG,
    )

    cfg = TrainConfig(
        run_name      = run_name,
        exp_dir       = EXP_DIR,
        data_root     = data_root,
        dataset       = DATASET,
        img_size      = img_size,
        q             = Q,
        k             = K,
        batch_size    = BATCH_SIZE,
        grad_accum    = GRAD_ACCUM,
        num_workers   = WORKERS,
        prefetch_factor = PREFETCH,
        use_amp       = USE_AMP,
        use_vae       = not SKIP_DECODER,
        total_epochs  = EPOCHS,
        lr            = LR,
        warmup_epochs = WARMUP_EPOCHS,
        beta          = BETA,
        lambda_union  = LAMBDA_UNION,
        lambda_sparse = LAMBDA_SPARSE,
        lambda_ortho  = LAMBDA_ORTHO,
        lambda_neg    = LAMBDA_NEG,
        eval_every        = EVAL_EVERY,
        eval_dataset      = EVAL_DATASET,
        eval_data_root    = EVAL_DATA_ROOT,
        eval_probe_epochs = EVAL_PROBE_EPOCHS,
        dim_inter         = DIM_INTER,
        dim_unique        = DIM_UNIQUE,
        latent_ch         = LATENT_CH,
    )

    Trainer(model, loss_fn, cfg).train()


if __name__ == '__main__':
    main()
