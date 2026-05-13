# open_vcv

**Augmentation-Aware Intersect-Union Decomposition**  
Unsupervised Visual Representation Learning · TensorFlow 2 / Keras 3

---

## Ý tưởng

Phân tách embedding ảnh thành hai không gian độc lập:

- **`v_inter`** — đặc trưng **bất biến** (invariant): không đổi qua các augmentation
- **`v_unique`** — đặc trưng **equivariant**: mang thông tin riêng của từng mức augmentation

**Điểm mới**: trọng số loss tính **động** theo aug-similarity (`w_ij = cos(image_feat_i, image_feat_j)`) thay vì hằng số cố định.

Chi tiết phương pháp: [`IDEA.md`](./IDEA.md)

---

## Cài đặt

```bash
git clone https://github.com/Bigkatoan/open_vcv.git
cd open_vcv
python -m venv venv && source venv/bin/activate
pip install tensorflow[and-cuda] keras pillow tqdm numpy matplotlib
```

> Yêu cầu: Python 3.10+, TensorFlow 2.15+, GPU với ≥ 8 GB VRAM (test trên RTX 3060 12 GB)

---

## Cấu trúc dự án

```
open_vcv/
├── main.py                   # Entry point — chỉnh config ở đây rồi chạy
├── src/
│   ├── models/
│   │   ├── VAE.py            # GatedConvEncoder + Decoder + VAE
│   │   └── encoders.py       # ResNet50 / EfficientNet / MobileNet wrappers
│   ├── losses/
│   │   └── losses.py         # IntersectUnionLoss (union, sparse, ortho, neg, uniformity)
│   ├── datasets/
│   │   └── aug_dataset.py    # TF-native data pipeline (no GIL) + COCO/ImageNet loaders
│   ├── trainers/
│   │   ├── trainer.py        # Training loop (@tf.function compiled step)
│   │   ├── logger.py         # CSV + text logging
│   │   └── visualizer.py     # Loss curves, similarity heatmaps, reconstructions
│   └── eval/
│       └── linear_probe.py   # Frozen backbone → linear head → top-1 / top-5
├── tests/                    # pytest test suite (31 tests)
├── experiments/              # Output: checkpoints, logs, plots (gitignored)
├── data/                     # Datasets (gitignored)
└── IDEA.md                   # Phương pháp chi tiết
```

---

## Tải dữ liệu

### ImageNet (train)

```bash
# Cách 1 — Kaggle CLI
pip install kaggle
kaggle competitions download -c imagenet-object-localization-challenge
# Sau đó giải nén vào data/imagenet/

# Cách 2 — rsync từ academic mirror (nếu có tài khoản ImageNet)
# rsync -azP username@image-net.org::imagenet/ILSVRC/Data/CLS-LOC/ data/imagenet/
```

Cấu trúc thư mục cần có:

```
data/imagenet/
└── train/
    ├── n01440764/   ← synset folders
    │   ├── n01440764_10026.JPEG
    │   └── ...
    ├── n01443537/
    └── ...          (1,000 classes, ~1.28M ảnh)
```

> Script kiểm tra: `python -c "from src.datasets.aug_dataset import load_images_from_dir; print(len(load_images_from_dir('data/imagenet')), 'images')"`

---

### COCO 2017 (linear probe eval)

```bash
mkdir -p data/coco2017
cd data/coco2017

# Ảnh
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip train2017.zip && unzip val2017.zip

# Annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip

cd ../..
```

Cấu trúc sau khi giải nén:

```
data/coco2017/
├── train2017/          (~118K ảnh)
├── val2017/            (~5K ảnh)
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

> Mỗi ảnh được gán nhãn theo category có annotation diện tích lớn nhất → 80 class classification.

---

## Training

Chỉnh config trực tiếp trong `main.py`:

```python
# ===========================================================================
# CONFIG — chỉnh trực tiếp ở đây
# ===========================================================================
ENCODER      = 'gated_vae'   # 'gated_vae' | 'resnet50' | 'efficientnet' | 'mobilenet'
DATASET      = 'imagenet'    # train dataset
IMG_SIZE     = 128
BATCH_SIZE   = 32            # tăng lên nếu còn VRAM (64 cho 3060 12GB)

EPOCHS       = 30
LR           = 3e-4

EVAL_EVERY   = 10            # chạy linear probe mỗi N epoch
EVAL_DATASET = 'coco'        # eval dataset
```

Sau đó:

```bash
python main.py
```

**Output mẫu:**

```
=======================================================
Framework: TensorFlow 2.21.0
Encoder  : gated_vae  |  Params: 2,473,248
Dataset  : imagenet (data/imagenet)  |  1,281,167 samples  →  ~40,036 steps/epoch
Device   : GPU × 1
=======================================================

[Data] Using TF-native file-decode pipeline (no GIL)

Ep 001/030: loss=2.134  union=1.021  sparse=0.443 ...
...
── Linear Probe eval (epoch 10) ──
  dataset=coco   top-1: 18.4%   top-5: 42.1%   time: 312s
```

> Lần đầu chạy mất thêm ~30s để `@tf.function` compile graph. Từ epoch 2 trở đi sẽ nhanh hơn.

---

## Theo dõi kết quả

Mỗi run tạo thư mục trong `experiments/<run_name>_<timestamp>/`:

```
experiments/gated_vae_imagenet_is128_20260513_093129/
├── config.txt              # Cấu hình đầy đủ
├── train.log               # Log toàn bộ training
├── metrics.csv             # Epoch-level metrics
├── probe_results.csv       # Linear probe top-1/top-5 theo epoch
├── checkpoints/
│   ├── weights_latest.keras
│   ├── weights_best_total.keras
│   └── weights_epoch_030.keras
└── viz/
    ├── losses.png           # Loss curves
    ├── metrics.png          # Decomposition quality curves
    ├── similarity.png       # Aug-similarity heatmap
    └── iter_curves.png      # Per-iteration loss
```

---

## Metrics ý nghĩa

| Metric | Hướng | Ý nghĩa |
|--------|-------|---------|
| `total` | ↓ | Tổng loss |
| `union` | ↓ | `v_inter` của cùng 1 ảnh (aug khác nhau) phải gần nhau |
| `sparse` | ↓ | `v_unique` của cùng 1 ảnh phải xa nhau |
| `ortho` | ↓ | `v_inter` và `v_unique` phải vuông góc (độc lập) |
| `neg` | ↓ | `v_inter` của ảnh **khác nhau** phải xa nhau |
| `union_consistency` | ↑ | Cosine sim trung bình giữa các view của cùng 1 ảnh (target > 0.85) |
| `sparse_divergence` | ↑ | 1 − sim của sparse features (target > 0.70) |
| `ortho_score` | ↓ | Cross-covariance giữa v_inter và v_unique (target < 0.10) |
| `recon_psnr` | ↑ | PSNR reconstruction (chỉ có khi `SKIP_DECODER=False`) |

**Linear probe** (top-1 trên COCO 80-class):
- Random baseline: ~1.25%
- Good SSL: > 15% sau 10 epoch
- Target: > 25% sau 30 epoch

---

## Tests

```bash
python -m pytest tests/ -v
```

31 tests bao gồm: VAE forward/backward shapes, loss functions, trainer components, linear probe.

---

## Cấu hình nhanh theo GPU

| GPU | VRAM | `BATCH_SIZE` | `IMG_SIZE` | `USE_AMP` | Ghi chú |
|-----|------|-------------|-----------|-----------|---------|
| GTX 1080 Ti | 11 GB | 32 | 128 | `True` (fp16) | OK |
| RTX 3060 | 12 GB | 64 | 128 | `True` (fp16) | Recommended |
| RTX 3090 / 4090 | 24 GB | 128 | 224 | `True` (fp16) | Full ImageNet size |
| **H100 SXM5** | **80 GB** | **512** | **224** | `True` (**bf16**) | Xem phần dưới |
| H100 × 8 (node) | 640 GB | 4096 | 224 | `True` (bf16) | Multi-GPU |

> Tăng `BATCH_SIZE` giúp in-batch negatives đa dạng hơn → contrastive learning tốt hơn.

---

## H100 — Cấu hình tối ưu

H100 hỗ trợ **bfloat16** (ổn định số học hơn float16, không bị underflow gradient) và **NVLink** cho multi-GPU. Cần 3 thay đổi so với config mặc định:

### 1. Đổi precision sang bfloat16

Trong `main.py`, tìm `USE_AMP = True` và đổi policy trong `xla_utils.py`, hoặc thêm dòng này vào đầu `main()`:

```python
# main.py — thêm sau import tensorflow as tf
import keras
keras.mixed_precision.set_global_policy('mixed_bfloat16')   # H100: dùng bf16 thay fp16
```

Và tắt `USE_AMP` để tránh set lại fp16:

```python
USE_AMP = False   # đã set bf16 thủ công ở trên
```

### 2. Config cho single H100 (80 GB)

```python
BATCH_SIZE   = 512
IMG_SIZE     = 224
EPOCHS       = 100          # H100 nhanh, train lâu hơn để hội tụ
LR           = 1e-3         # scale LR theo batch: LR_base × (batch / 256)
WARMUP_EPOCHS = 10

# Model lớn hơn — H100 80GB chứa thoải mái
LATENT_CH    = 384          # 256 + 128
DIM_INTER    = 256
DIM_UNIQUE   = 128
```

Và trong `main()`, dùng model lớn hơn:

```python
model = VAE(
    s1_out=16, s1_heads=4,  s1_blocks=2,   # 64ch tại 224×224
    s2_out=16, s2_heads=8,  s2_blocks=3,   # 128ch tại 112×112
    s3_out=16, s3_heads=16, s3_blocks=3,   # 256ch tại 56×56  ← thêm 1 stage vì IMG=224
    latent_ch=LATENT_CH,
    dec_ch3=256, dec_ch2=128, dec_ch1=64,
    dim_inter=DIM_INTER, dim_unique=DIM_UNIQUE,
    feat_dim=128, hidden_dim=512,
)
```

### 3. Multi-GPU (H100 × N)

`MirroredStrategy` tự động chia batch qua các GPU, không cần sửa model:

```python
# main.py — thêm trước khi build model
import tensorflow as tf
strategy = tf.distribute.MirroredStrategy()   # tự detect tất cả GPU
print(f"Running on {strategy.num_replicas_in_sync} GPUs")

with strategy.scope():
    model   = VAE(...)
    loss_fn = IntersectUnionLoss(...)
    cfg     = TrainConfig(
        batch_size = 512 * strategy.num_replicas_in_sync,  # global batch
        ...
    )
    Trainer(model, loss_fn, cfg).train()
```

> Với 8× H100 (node): `batch_size = 4096`, LR ~ `4e-3` (linear scaling rule).

### Ước tính tốc độ

| Setup | steps/s | Thời gian / epoch (ImageNet) | 100 epochs |
|-------|---------|------------------------------|------------|
| RTX 3060 (batch 64, fp16) | ~8 | ~45 min | ~75 h |
| H100 SXM5 (batch 512, bf16) | ~180 | ~2 min | ~3.3 h |
| 8× H100 (batch 4096, bf16) | ~900 | ~25 s | ~42 min |
