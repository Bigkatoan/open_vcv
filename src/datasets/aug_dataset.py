"""
aug_dataset.py — PIL-based augmentation + tf.data pipeline.
Không cần torchvision — dùng Pillow + numpy + tf.data.
"""

import os
import random
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps


# ===========================================================================
# PIL Augmentation helpers
# ===========================================================================

def _random_crop(img: Image.Image, img_size: int, pad: int) -> Image.Image:
    img = img.resize((img_size + pad, img_size + pad), Image.BILINEAR)
    x = random.randint(0, pad)
    y = random.randint(0, pad)
    return img.crop((x, y, x + img_size, y + img_size))


def _color_jitter(img: Image.Image,
                  b=0.4, c=0.4, s=0.4, h=0.1) -> Image.Image:
    if b > 0:
        img = ImageEnhance.Brightness(img).enhance(1 + random.uniform(-b, b))
    if c > 0:
        img = ImageEnhance.Contrast(img).enhance(1 + random.uniform(-c, c))
    if s > 0:
        img = ImageEnhance.Color(img).enhance(1 + random.uniform(-s, s))
    return img


def augment_pil(pil_img: Image.Image, level: int, img_size: int) -> np.ndarray:
    """
    level 0: Conservative — flip + small crop
    level 1: Moderate     — + color jitter + rotation
    level 2: Aggressive   — + random resized crop + grayscale + erasing
    Returns float32 numpy (H, W, 3) in [0, 1].
    """
    pad = max(4, img_size // 16)
    img = pil_img.convert('RGB')

    if level == 0:
        img = _random_crop(img, img_size, pad)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    elif level == 1:
        img = _random_crop(img, img_size, pad)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = _color_jitter(img, 0.4, 0.4, 0.4, 0.1)
        img = img.rotate(random.uniform(-15, 15), resample=Image.BILINEAR)

    elif level == 2:
        w, h  = img.size
        scale = random.uniform(0.2, 1.0)
        cw = max(1, int(w * scale ** 0.5))
        ch = max(1, int(h * scale ** 0.5))
        x = random.randint(0, max(0, w - cw))
        y = random.randint(0, max(0, h - ch))
        img = img.crop((x, y, x + cw, y + ch)).resize((img_size, img_size), Image.BILINEAR)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = _color_jitter(img, 0.8, 0.8, 0.8, 0.2)
        if random.random() < 0.2:
            img = ImageOps.grayscale(img).convert('RGB')
        img = img.rotate(random.uniform(-30, 30), resample=Image.BILINEAR)
        # Random erasing
        if random.random() < 0.5:
            arr = np.array(img)
            eh = random.randint(max(1, img_size // 50), max(2, img_size // 5))
            ew = random.randint(max(1, img_size // 50), max(2, img_size // 5))
            ey = random.randint(0, img_size - eh)
            ex = random.randint(0, img_size - ew)
            arr[ey:ey + eh, ex:ex + ew] = 0
            img = Image.fromarray(arr)

    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 2:                   # grayscale edge case
        arr = np.stack([arr] * 3, axis=-1)
    return arr


# ===========================================================================
# Image sources
# ===========================================================================

def load_images_from_dir(root: str) -> list:
    """Returns list of PIL Images from directory tree."""
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    paths = [p for p in Path(root).rglob('*') if p.suffix.lower() in exts]
    assert paths, f"No images found in {root}"
    print(f"[Dataset] Found {len(paths):,} images in {root}")
    return paths   # lazy — don't load into RAM


def load_cifar10(root='data', train=True) -> tuple:
    """Returns (images_uint8, labels) — images: (N, 32, 32, 3) uint8."""
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.cifar10.load_data()
    images = x_tr if train else x_te
    labels = y_tr.flatten() if train else y_te.flatten()
    return images, labels


def load_cifar100(root='data', train=True) -> tuple:
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.cifar100.load_data()
    images = x_tr if train else x_te
    labels = y_tr.flatten() if train else y_te.flatten()
    return images, labels


# ===========================================================================
# tf.data pipeline
# ===========================================================================

def _tf_augment_level(img: tf.Tensor, level: int, img_size: int) -> tf.Tensor:
    """
    Pure-TF augmentation for a single image (H, W, 3) float32 [0,1].
    level is a Python int — branches are resolved at graph-trace time.
    """
    pad = max(4, img_size // 16)

    if level == 0:
        img_up = tf.image.resize(img, [img_size + pad, img_size + pad])
        img    = tf.image.random_crop(img_up, [img_size, img_size, 3])
        img    = tf.image.random_flip_left_right(img)

    elif level == 1:
        img_up = tf.image.resize(img, [img_size + pad, img_size + pad])
        img    = tf.image.random_crop(img_up, [img_size, img_size, 3])
        img    = tf.image.random_flip_left_right(img)
        img    = tf.image.random_brightness(img, max_delta=0.4)
        img    = tf.image.random_contrast(img, lower=0.6, upper=1.4)
        img    = tf.image.random_saturation(img, lower=0.6, upper=1.4)
        img    = tf.clip_by_value(img, 0.0, 1.0)

    elif level == 2:
        # Random resized crop (scale ∈ [0.2, 1.0])
        scale  = tf.random.uniform((), 0.2, 1.0)
        side   = tf.maximum(
            tf.cast(tf.cast(img_size, tf.float32) * tf.sqrt(scale), tf.int32), 1
        )
        img    = tf.image.random_crop(img, tf.stack([side, side, 3]))
        img    = tf.image.resize(img, [img_size, img_size])
        img    = tf.image.random_flip_left_right(img)
        img    = tf.image.random_brightness(img, max_delta=0.8)
        img    = tf.image.random_contrast(img, lower=0.2, upper=1.8)
        img    = tf.image.random_saturation(img, lower=0.2, upper=1.8)
        img    = tf.clip_by_value(img, 0.0, 1.0)

    return img


def _make_path_aug_dataset(paths, q: int, img_size: int,
                            batch_size: int, shuffle_buffer: int,
                            prefetch: int) -> tf.data.Dataset:
    """
    TF-native pipeline for large on-disk datasets (ImageNet, COCO train).
    Uses tf.io + tf.image — no Python GIL, fully parallel GPU-friendly decode.
    """
    path_strs = [str(p) for p in paths]
    n         = len(path_strs)

    def _load_and_augment(path):
        raw  = tf.io.read_file(path)
        img  = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img  = tf.image.resize(img, [img_size, img_size])
        img  = tf.cast(img, tf.float32) / 255.0
        # q views with increasing augmentation strength
        views = tf.stack([_tf_augment_level(img, i, img_size) for i in range(q)], axis=0)
        neg   = tf.zeros((0,), dtype=tf.float32)
        views.set_shape((q, img_size, img_size, 3))
        neg.set_shape((0,))
        return views, neg

    dataset = tf.data.Dataset.from_tensor_slices(path_strs)
    dataset = dataset.shuffle(min(shuffle_buffer, n))
    dataset = dataset.map(_load_and_augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(prefetch)
    return dataset


def make_aug_dataset(images, q: int, k: int, img_size: int,
                     batch_size: int, shuffle_buffer: int = 10000,
                     prefetch: int = tf.data.AUTOTUNE) -> tf.data.Dataset:
    """
    images : numpy array (N, H, W, 3) uint8  →  PIL-based augmentation
             list of Path / str             →  TF-native file-decode pipeline
    Returns batched tf.data.Dataset yielding (core_imgs, neg_imgs):
        core_imgs: (B, q, H, W, 3)  float32 [0,1]
        neg_imgs:  (B, 0)           float32  — empty for k=0 (in-batch negatives)
    """
    n        = len(images)
    is_array = isinstance(images, np.ndarray)

    # Path-based datasets (ImageNet, COCO): use TF-native decode — no GIL
    if not is_array:
        print(f"[Data] Using TF-native file-decode pipeline (no GIL)")
        return _make_path_aug_dataset(images, q, img_size, batch_size,
                                      shuffle_buffer, prefetch)

    # In-memory array datasets (CIFAR): keep PIL-based augmentation
    def _generate(idx_np):
        idx  = int(idx_np)
        img  = Image.fromarray(images[idx])
        core = np.stack([augment_pil(img, i, img_size) for i in range(q)], axis=0)
        neg  = np.zeros((0,), dtype=np.float32)
        return core, neg

    def _tf_generate(idx):
        core, neg = tf.numpy_function(
            _generate, [idx], [tf.float32, tf.float32]
        )
        core.set_shape((q, img_size, img_size, 3))
        neg.set_shape((0,))
        return core, neg

    dataset = tf.data.Dataset.range(n)
    dataset = dataset.shuffle(min(shuffle_buffer, n))
    dataset = dataset.map(_tf_generate, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(prefetch)
    return dataset


# ===========================================================================
# Convenience: load labeled dataset for linear probe
# ===========================================================================

def load_coco_labeled(root: str, train: bool, img_size: int) -> tuple:
    """
    COCO 80-class classification for linear probe.
    Each image is assigned the category of its largest-area annotation.
    Returns (image_paths: list[str], labels: np.ndarray[int64]).
    Expects COCO layout:
        root/train2017/*.jpg
        root/val2017/*.jpg
        root/annotations/instances_train2017.json
        root/annotations/instances_val2017.json
    """
    import json
    split    = 'train2017' if train else 'val2017'
    ann_file = os.path.join(root, 'annotations', f'instances_{split}.json')
    img_dir  = os.path.join(root, split)

    with open(ann_file) as f:
        coco = json.load(f)

    # Stable category_id → class index (sorted by id for reproducibility)
    cats = {
        c['id']: i
        for i, c in enumerate(sorted(coco['categories'], key=lambda x: x['id']))
    }

    # Per-image: keep only the annotation with the largest area
    img_best: dict = {}  # image_id → (class_idx, area)
    for ann in coco['annotations']:
        img_id = ann['image_id']
        cls    = cats.get(ann['category_id'])
        if cls is None:
            continue
        area = ann.get('area', 0)
        if img_id not in img_best or area > img_best[img_id][1]:
            img_best[img_id] = (cls, area)

    # Build id → filename map
    img_map = {img['id']: img['file_name'] for img in coco['images']}

    paths, labels = [], []
    for img_id, (cls, _) in img_best.items():
        fname = img_map.get(img_id)
        if fname is None:
            continue
        full_path = os.path.join(img_dir, fname)
        paths.append(full_path)
        labels.append(cls)

    assert paths, f"[COCO] No images found at {img_dir}"
    print(f"[Dataset] COCO {split}: {len(paths):,} images, 80 classes")
    return paths, np.array(labels, dtype=np.int64)


def load_labeled_dataset(dataset: str, root: str, train: bool, img_size: int):
    """
    Returns (images_uint8_or_pil_list, labels_array).
    Used by linear_probe.py.
    """
    if dataset == 'cifar10':
        images, labels = load_cifar10(root, train)
        if img_size != 32:
            resized = np.stack([
                np.array(Image.fromarray(images[i]).resize((img_size, img_size), Image.BILINEAR))
                for i in range(len(images))
            ])
            return resized, labels
        return images, labels

    elif dataset == 'cifar100':
        images, labels = load_cifar100(root, train)
        return images, labels

    elif dataset == 'imagenet':
        split = 'train' if train else 'val'
        image_root = os.path.join(root, split)
        classes = sorted(os.listdir(image_root))
        paths, labs = [], []
        for i, cls in enumerate(classes):
            cls_dir = os.path.join(image_root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    paths.append(os.path.join(cls_dir, fname))
                    labs.append(i)
        return paths, np.array(labs)

    elif dataset == 'coco':
        return load_coco_labeled(root, train, img_size)

    else:
        raise ValueError(f"Unsupported eval dataset: {dataset}")
