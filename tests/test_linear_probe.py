"""Tests for src/eval/linear_probe.py (TensorFlow)"""
import numpy as np
import tensorflow as tf

from src.eval.linear_probe import _extract_features, run_linear_probe
from src.models.VAE import VAE


def _small_model():
    model = VAE(
        s1_out=8, s1_heads=4,  s1_blocks=1,
        s2_out=8, s2_heads=8,  s2_blocks=2,
        s3_out=8, s3_heads=16, s3_blocks=2,
        latent_ch=96, dec_ch3=64, dec_ch2=32, dec_ch1=16,
        dim_inter=64, dim_unique=32, feat_dim=32, hidden_dim=128,
    )
    # build
    dummy = tf.zeros((1, 32, 32, 3))
    model(dummy, skip_decoder=True, training=False)
    return model


def test_extract_features_shapes():
    model = _small_model()
    N, B  = 24, 6
    images = np.random.randint(0, 255, (N, 32, 32, 3), dtype=np.uint8).astype(np.float32) / 255.
    labels = np.zeros(N, dtype=np.int32)
    feats, labs = _extract_features(model, images, labels, img_size=32, batch_size=B)
    assert feats.shape == (N, 64)     # dim_inter=64
    assert labs.shape  == (N,)
    assert feats.dtype == np.float32


def test_run_linear_probe(tmp_path, monkeypatch):
    """Patch dataset loading to skip disk I/O."""
    import src.eval.linear_probe as lp_mod
    DIM, N_TRAIN, N_TEST, N_CLS = 64, 200, 50, 10

    fake_train = (
        np.random.randint(0, 255, (N_TRAIN, 32, 32, 3), dtype=np.uint8),
        np.random.randint(0, N_CLS, N_TRAIN),
    )
    fake_test = (
        np.random.randint(0, 255, (N_TEST, 32, 32, 3), dtype=np.uint8),
        np.random.randint(0, N_CLS, N_TEST),
    )

    call_count = [0]
    def fake_load(dataset, root, train, img_size):
        call_count[0] += 1
        return fake_train if train else fake_test

    def fake_extract(model, images, labels, img_size, batch_size=512):
        n = len(images)
        return np.random.randn(n, DIM).astype(np.float32), labels

    monkeypatch.setattr(lp_mod, 'load_labeled_dataset', fake_load)
    monkeypatch.setattr(lp_mod, '_extract_features',    fake_extract)

    model  = _small_model()
    result = run_linear_probe(
        model=model, dataset='cifar10', data_root=str(tmp_path),
        img_size=32, dim_inter=DIM, probe_epochs=2, verbose=False,
    )
    for k in ('top1', 'top5', 'n_train', 'n_test', 'time_s'):
        assert k in result
    assert 0.0 <= result['top1'] <= 100.0
    assert result['n_test'] == N_TEST
