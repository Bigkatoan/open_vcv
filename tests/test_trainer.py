"""Tests for src/trainers/trainer.py (TensorFlow)"""
import numpy as np
import tensorflow as tf

from src.trainers.trainer import TrainConfig, compute_metrics, WarmupCosineDecay


def test_trainconfig_defaults():
    cfg = TrainConfig()
    assert cfg.q == 3
    assert cfg.eval_every == 10
    assert cfg.eval_dataset == 'cifar10'
    assert cfg.eval_probe_epochs == 20
    assert cfg.warmup_epochs == 5
    assert cfg.dim_inter + cfg.dim_unique == cfg.latent_ch


def test_compute_metrics_shapes():
    B, q, dim = 4, 3, 64
    uf = tf.math.l2_normalize(tf.random.normal((B*q, dim)), axis=-1).numpy()
    sf = tf.math.l2_normalize(tf.random.normal((B*q, dim)), axis=-1).numpy()
    m = compute_metrics(uf, sf, mse=0.05, q=q)
    for k in ('union_consistency', 'sparse_divergence', 'ortho_score', 'recon_psnr'):
        assert k in m
        assert isinstance(m[k], float)


def test_compute_metrics_perfect_invariance():
    B, q, dim = 4, 3, 64
    base = tf.math.l2_normalize(tf.random.normal((B, dim)), axis=-1).numpy()
    uf   = np.repeat(base, q, axis=0)   # identical across views
    sf   = tf.math.l2_normalize(tf.random.normal((B*q, dim)), axis=-1).numpy()
    m    = compute_metrics(uf, sf, mse=0.01, q=q)
    assert abs(m['union_consistency'] - 1.0) < 1e-4


def test_warmup_cosine_schedule():
    sched = WarmupCosineDecay(peak_lr=3e-4, warmup_steps=100, total_steps=1000)
    # At step 0: should be close to min_lr
    assert float(sched(0)) < 1e-5
    # At warmup end: should equal peak_lr
    assert abs(float(sched(100)) - 3e-4) < 1e-6
    # At end: should be close to min_lr
    assert float(sched(1000)) < 1e-4
    # During warmup, LR should increase
    assert float(sched(50)) < float(sched(100))
