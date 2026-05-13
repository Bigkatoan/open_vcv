"""
xla_utils.py — Device / distribution helpers (TensorFlow).

TF tự động detect GPU/TPU. Không cần torch_xla.
"""

import tensorflow as tf


def get_strategy():
    """Trả về tf.distribute.Strategy phù hợp."""
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 1:
        return tf.distribute.MirroredStrategy()
    return tf.distribute.get_strategy()


def is_tpu() -> bool:
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        return True
    except Exception:
        return False


def setup_mixed_precision(use_amp: bool = True):
    """Enable float16 mixed precision on GPU, bfloat16 on TPU."""
    if not use_amp:
        return
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        import keras
        keras.mixed_precision.set_global_policy('mixed_float16')
        print("[Device] Mixed precision: float16")
    else:
        print("[Device] Mixed precision: disabled (no GPU)")


def device_info() -> str:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        return f"GPU × {len(gpus)}"
    return "CPU"
