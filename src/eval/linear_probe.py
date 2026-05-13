"""
linear_probe.py — Linear evaluation protocol cho SSL features (TensorFlow).

Protocol:
  1. Freeze backbone (inference mode)
  2. Extract v_inter features một lần
  3. Train nn.Dense(dim_inter, n_classes) với SGD + cosine LR
  4. Report top-1 / top-5 accuracy
"""

import time
import numpy as np
import tensorflow as tf
import keras
from keras import layers

from src.datasets.aug_dataset import load_labeled_dataset


N_CLASSES = {
    'cifar10':  10,
    'cifar100': 100,
    'stl10':    10,
    'imagenet': 1000,
    'coco':     80,
}


# ===========================================================================
# Feature extraction
# ===========================================================================

def _preprocess(images_uint8: np.ndarray, img_size: int) -> np.ndarray:
    """Resize + normalize to [0,1]. images: (N, H, W, 3) uint8."""
    from PIL import Image
    if images_uint8.shape[1] == img_size:
        return images_uint8.astype(np.float32) / 255.0
    resized = np.stack([
        np.array(Image.fromarray(images_uint8[i]).resize((img_size, img_size), Image.BILINEAR))
        for i in range(len(images_uint8))
    ]).astype(np.float32) / 255.0
    return resized


def _load_batch_images(images, start: int, end: int, img_size: int) -> np.ndarray:
    """Load a slice of images — handles both numpy arrays and path lists."""
    from PIL import Image as PILImage
    if isinstance(images, np.ndarray):
        batch = images[start:end].astype(np.float32)
        if batch.max() > 1.0:
            batch /= 255.0
        if batch.shape[1] != img_size:
            batch = np.stack([
                np.array(PILImage.fromarray(batch[i].astype(np.uint8))
                         .resize((img_size, img_size), PILImage.BILINEAR))
                for i in range(len(batch))
            ]).astype(np.float32) / 255.0
        return batch
    # path list
    out = []
    for p in images[start:end]:
        img = PILImage.open(p).convert('RGB').resize((img_size, img_size), PILImage.BILINEAR)
        out.append(np.array(img, dtype=np.float32) / 255.0)
    return np.stack(out, axis=0)


def _extract_features(model, images, labels, img_size: int,
                       batch_size: int = 512) -> tuple:
    """
    Extract v_inter features. model runs in inference mode.
    images: numpy (N,H,W,3) uint8/float32  OR  list of file paths
    Returns: (feats: (N, dim_inter), labels: (N,)) numpy
    """
    n = len(images)
    feats_list, labels_list = [], []

    for start in range(0, n, batch_size):
        end   = min(start + batch_size, n)
        batch = tf.constant(_load_batch_images(images, start, end, img_size))
        _, _, _, _, v_inter, _ = model(batch, skip_decoder=True, training=False)
        feats_list.append(v_inter.numpy())
        labels_list.append(labels[start:end])

    return np.concatenate(feats_list, axis=0), np.concatenate(labels_list, axis=0)


# ===========================================================================
# Linear probe
# ===========================================================================

def run_linear_probe(
    model,
    dataset:      str,
    data_root:    str,
    img_size:     int,
    dim_inter:    int,
    probe_epochs: int  = 20,
    batch_size:   int  = 512,
    verbose:      bool = True,
    **kwargs,
) -> dict:
    """
    Chạy full linear probe.

    Returns:
        {'top1', 'top5', 'n_train', 'n_test', 'time_s'}
    """
    n_classes = N_CLASSES[dataset]
    t0        = time.perf_counter()

    if verbose:
        print(f"  [LinearProbe] dataset={dataset}  dim={dim_inter}→{n_classes}"
              f"  probe_epochs={probe_epochs}")

    # ── Load data ─────────────────────────────────────────────────────────
    train_images, train_labels = load_labeled_dataset(dataset, data_root, True,  img_size)
    test_images,  test_labels  = load_labeled_dataset(dataset, data_root, False, img_size)

    # Normalize if uint8
    if isinstance(train_images, np.ndarray) and train_images.dtype == np.uint8:
        train_images = train_images.astype(np.float32) / 255.0
        test_images  = test_images.astype(np.float32) / 255.0

    # ── Extract features ──────────────────────────────────────────────────
    if verbose:
        print(f"  [LinearProbe] Extracting train features ({len(train_images):,}) ...")
    train_f, train_y = _extract_features(model, train_images, train_labels,
                                         img_size, batch_size)

    if verbose:
        print(f"  [LinearProbe] Extracting test features  ({len(test_images):,}) ...")
    test_f, test_y   = _extract_features(model, test_images,  test_labels,
                                          img_size, batch_size)

    # ── In-memory tf.data ─────────────────────────────────────────────────
    train_ds = (tf.data.Dataset
                .from_tensor_slices((train_f, train_y.astype(np.int32)))
                .shuffle(len(train_f))
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE))
    test_ds  = (tf.data.Dataset
                .from_tensor_slices((test_f, test_y.astype(np.int32)))
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE))

    # ── Linear head ───────────────────────────────────────────────────────
    head = layers.Dense(
        n_classes,
        kernel_initializer=keras.initializers.RandomNormal(0, 0.01),
        bias_initializer='zeros',
        dtype='float32',
    )
    head.build((None, dim_inter))

    total_steps = probe_epochs * len(train_f) // batch_size
    schedule    = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.1,
        decay_steps=max(total_steps, 1),
        alpha=1e-3 / 0.1,
    )
    optimizer = keras.optimizers.SGD(learning_rate=schedule, momentum=0.9)

    if verbose:
        print(f"  [LinearProbe] Training linear head ({probe_epochs} epochs) ...")

    for _ in range(probe_epochs):
        for x_batch, y_batch in train_ds:
            with tf.GradientTape() as tape:
                logits = head(x_batch, training=True)
                loss   = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(y_batch, logits)
                )
            grads = tape.gradient(loss, head.trainable_variables)
            optimizer.apply_gradients(zip(grads, head.trainable_variables))

    # ── Evaluate ──────────────────────────────────────────────────────────
    c1 = c5 = n = 0
    for x_batch, y_batch in test_ds:
        logits = head(x_batch, training=False)
        k      = min(5, n_classes)
        top5   = tf.math.top_k(logits, k=k).indices   # (B, k)
        y_exp  = tf.expand_dims(y_batch, 1)            # (B, 1)
        c1    += int(tf.reduce_sum(tf.cast(top5[:, :1] == y_exp, tf.int32)))
        c5    += int(tf.reduce_sum(tf.cast(
            tf.reduce_any(top5 == y_exp, axis=1), tf.int32)))
        n     += len(y_batch)

    elapsed = time.perf_counter() - t0
    return {
        'top1':    round(100.0 * c1 / n, 2),
        'top5':    round(100.0 * c5 / n, 2),
        'n_train': len(train_f),
        'n_test':  n,
        'time_s':  round(elapsed, 1),
    }
