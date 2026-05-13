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

| GPU VRAM | `BATCH_SIZE` | `IMG_SIZE` | Ghi chú |
|----------|-------------|-----------|---------|
| 4 GB | 16 | 128 | OK |
| 8 GB | 32 | 128 | OK |
| 12 GB (3060) | 64 | 128 | Recommended |
| 24 GB | 128 | 128 | Tốt nhất |

> Tăng `BATCH_SIZE` giúp in-batch negatives đa dạng hơn → contrastive learning tốt hơn.
