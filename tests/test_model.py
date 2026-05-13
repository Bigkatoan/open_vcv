"""Tests for src/models/VAE.py (TensorFlow)"""
import pytest
import numpy as np
import tensorflow as tf

from src.models.VAE import VAE


def _small_vae():
    return VAE(
        s1_out=8, s1_heads=4,  s1_blocks=1,
        s2_out=8, s2_heads=8,  s2_blocks=2,
        s3_out=8, s3_heads=16, s3_blocks=2,
        latent_ch=96, dec_ch3=64, dec_ch2=32, dec_ch1=16,
        dim_inter=64, dim_unique=32, feat_dim=32, hidden_dim=128,
    )


class TestVAEForward:
    def setup_method(self):
        self.model = _small_vae()
        self.x = tf.random.normal((2, 32, 32, 3))
        # Warm-up build
        self.model(self.x, skip_decoder=True, training=False)

    def test_skip_decoder_output_shapes(self):
        recon, mu, logvar, imgf, uf, sf = self.model(
            self.x, skip_decoder=True, training=False)
        B = 2
        assert recon  is None
        assert logvar is None
        assert mu.shape   == (B, 96)
        assert imgf.shape == (B, 32)   # feat_dim=32
        assert uf.shape   == (B, 64)   # dim_inter=64
        assert sf.shape   == (B, 32)   # dim_unique=32

    def test_vae_output_shapes(self):
        recon, mu, logvar, imgf, uf, sf = self.model(
            self.x, skip_decoder=False, training=False)
        B = 2
        assert recon.shape  == (B, 32, 32, 3)
        assert logvar is not None
        assert uf.shape     == (B, 64)
        assert sf.shape     == (B, 32)

    def test_features_l2_normalized(self):
        _, _, _, _, uf, sf = self.model(self.x, skip_decoder=True, training=False)
        norms_u = tf.norm(uf, axis=-1).numpy()
        norms_s = tf.norm(sf, axis=-1).numpy()
        np.testing.assert_allclose(norms_u, 1.0, atol=1e-5)
        np.testing.assert_allclose(norms_s, 1.0, atol=1e-5)

    def test_dim_constraint(self):
        assert self.model.dim_inter + self.model.dim_unique == self.model.latent_ch

    def test_no_nan_skip_decoder(self):
        outputs = self.model(self.x, skip_decoder=True, training=False)
        for o in outputs:
            if o is not None:
                assert np.all(np.isfinite(o.numpy())), "NaN/Inf in output"

    def test_no_nan_vae_mode(self):
        outputs = self.model(self.x, skip_decoder=False, training=False)
        for o in outputs:
            if o is not None:
                assert np.all(np.isfinite(o.numpy())), "NaN/Inf in output"

    def test_recon_pixel_range(self):
        """Decoder output should be in [0, 1] (sigmoid activation)."""
        recon, *_ = self.model(self.x, skip_decoder=False, training=False)
        arr = recon.numpy()
        assert arr.min() >= -1e-6
        assert arr.max() <= 1.0 + 1e-6


def test_batch_size_1():
    model = _small_vae()
    x = tf.random.normal((1, 32, 32, 3))
    model(x, skip_decoder=True, training=False)   # build
    recon, mu, *_ = model(x, skip_decoder=False, training=False)
    assert recon.shape == (1, 32, 32, 3)
