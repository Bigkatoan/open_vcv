"""Tests for src/losses/losses.py (TensorFlow)"""
import pytest
import numpy as np
import tensorflow as tf

from src.losses.losses import (
    union_loss, sparse_loss, ortho_loss, neg_loss,
    uniformity_loss, vae_loss, IntersectUnionLoss,
)


def _norm(t):
    return tf.math.l2_normalize(t, axis=-1)


B, Q, DIM = 4, 3, 64


# ---------------------------------------------------------------------------
# unit: individual losses
# ---------------------------------------------------------------------------

def test_union_loss_finite():
    uf   = _norm(tf.random.normal((B * Q, DIM)))
    imgf = _norm(tf.random.normal((B * Q, DIM)))
    loss = union_loss(uf, imgf, q=Q)
    assert loss.shape == ()
    assert np.isfinite(float(loss))


def test_union_loss_perfect_invariance():
    """Identical v_inter across augs → loss ≈ 0."""
    base = _norm(tf.random.normal((B, DIM)))
    uf   = tf.repeat(base, Q, axis=0)           # (B*Q, DIM) same per image
    imgf = _norm(tf.random.normal((B * Q, DIM)))
    loss = union_loss(uf, imgf, q=Q)
    assert float(loss) < 1e-5, f"expected ~0, got {float(loss)}"


def test_sparse_loss_finite():
    sf   = _norm(tf.random.normal((B * Q, DIM)))
    imgf = _norm(tf.random.normal((B * Q, DIM)))
    loss = sparse_loss(sf, imgf, q=Q)
    assert np.isfinite(float(loss))


def test_ortho_loss_orthogonal():
    """Block-orthogonal features → ortho_loss ≈ 0."""
    half = DIM // 2
    uf = np.zeros((B * Q, DIM), dtype=np.float32)
    sf = np.zeros((B * Q, DIM), dtype=np.float32)
    uf[:, :half] = np.random.randn(B * Q, half)
    sf[:, half:] = np.random.randn(B * Q, half)
    uf = _norm(tf.constant(uf))
    sf = _norm(tf.constant(sf))
    loss = ortho_loss(uf, sf)
    assert float(loss) < 1e-4, f"expected ~0, got {float(loss)}"


def test_neg_loss_finite():
    uf   = _norm(tf.random.normal((B * Q, DIM)))
    uneg = _norm(tf.random.normal((B * Q, DIM)))
    loss = neg_loss(uf, uneg)
    assert np.isfinite(float(loss))


def test_neg_loss_none():
    uf   = _norm(tf.random.normal((B * Q, DIM)))
    loss = neg_loss(uf, None)
    assert float(loss) == 0.0


def test_uniformity_finite():
    feat = _norm(tf.random.normal((B * Q, DIM)))
    loss = uniformity_loss(feat)
    assert np.isfinite(float(loss))


def test_vae_loss_components():
    recon  = tf.random.normal((B, 32, 32, 3))
    img    = tf.random.normal((B, 32, 32, 3))
    mu     = tf.random.normal((B, 96))
    logvar = tf.zeros((B, 96))
    total, mse, kl = vae_loss(recon, img, mu, logvar, beta=0.1)
    assert np.isfinite(float(total))
    assert float(mse) >= 0
    assert float(kl)  >= 0


# ---------------------------------------------------------------------------
# IntersectUnionLoss
# ---------------------------------------------------------------------------

class TestIntersectUnionLoss:
    def setup_method(self):
        self.fn   = IntersectUnionLoss()
        self.uf   = _norm(tf.random.normal((B * Q, DIM)))
        self.sf   = _norm(tf.random.normal((B * Q, DIM)))
        self.imgf = _norm(tf.random.normal((B * Q, DIM)))
        self.uneg = _norm(tf.random.normal((B * Q, DIM)))

    def test_skip_decoder_mode(self):
        total, details = self.fn(
            None, None, None, None,
            self.imgf, self.uf, self.sf, self.uneg, q=Q,
        )
        assert np.isfinite(float(total))
        assert details['vae'] == 0.0
        assert details['mse'] == 0.0

    def test_vae_mode(self):
        recon  = tf.random.normal((B * Q, 32, 32, 3))
        img    = tf.random.normal((B * Q, 32, 32, 3))
        mu     = tf.random.normal((B * Q, 96))
        logvar = tf.zeros((B * Q, 96))
        total, details = self.fn(
            recon, img, mu, logvar,
            self.imgf, self.uf, self.sf, self.uneg, q=Q,
        )
        assert np.isfinite(float(total))
        assert details['vae'] > 0

    def test_required_keys(self):
        _, details = self.fn(
            None, None, None, None,
            self.imgf, self.uf, self.sf, self.uneg, q=Q,
        )
        for k in ('total', 'vae', 'mse', 'kl', 'union', 'sparse', 'ortho', 'neg', 'uniform'):
            assert k in details

    def test_total_matches_details(self):
        total, details = self.fn(
            None, None, None, None,
            self.imgf, self.uf, self.sf, self.uneg, q=Q,
        )
        assert abs(float(total) - details['total']) < 1e-5
