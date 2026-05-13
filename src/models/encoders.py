"""
encoders.py — Alternative backbone encoders (TensorFlow/Keras).

All encoders implement the same interface as VAE:
    model(x, skip_decoder=True, training=False) →
        (None, mu_vec, None, image_feat, v_inter, v_unique)
"""

import tensorflow as tf
import keras
from keras import layers


ENCODER_REGISTRY = {
    'gated_vae':        None,   # use VAE from VAE.py
    'resnet50':         'ResNet50Encoder',
    'resnet18':         'ResNet18Encoder',
    'efficientnet':     'EfficientNetEncoder',
    'mobilenet':        'MobileNetEncoder',
}


# ===========================================================================
# Base encoder
# ===========================================================================

class BaseEncoder(keras.Model):
    """
    Subclass chỉ cần set self.backbone (outputs (B, backbone_dim) after GAP).
    """
    def __init__(self, latent_ch=192, dim_inter=128, dim_unique=64, **kwargs):
        super().__init__(**kwargs)
        assert dim_inter + dim_unique == latent_ch
        self.latent_ch  = latent_ch
        self.dim_inter  = dim_inter
        self.dim_unique = dim_unique

    def _build_heads(self, backbone_dim: int):
        self.latent_proj = keras.Sequential([
            layers.Dense(self.latent_ch, use_bias=False,
                         kernel_initializer=keras.initializers.GlorotUniform()),
            layers.BatchNormalization(),
        ])
        self.image_feat_proj = keras.Sequential([
            layers.Dense(256, use_bias=False, kernel_initializer='glorot_uniform'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dense(self.dim_inter // 4, kernel_initializer='glorot_uniform'),
        ])

    def call(self, x, skip_decoder=True, training=False):
        feats      = self.backbone(x, training=training)          # (B, backbone_dim)
        image_feat = tf.math.l2_normalize(
            self.image_feat_proj(feats, training=training), axis=-1)

        mu_raw = self.latent_proj(feats, training=training)
        mu_vec = (tf.sigmoid(0.1 * mu_raw) - 0.5) * 10.0

        v_inter  = tf.math.l2_normalize(mu_vec[:, :self.dim_inter],  axis=-1)
        v_unique = tf.math.l2_normalize(mu_vec[:, self.dim_inter:],  axis=-1)
        return None, mu_vec, None, image_feat, v_inter, v_unique

    def count_parameters(self):
        return sum(tf.size(v).numpy() for v in self.trainable_variables)


# ===========================================================================
# ResNet-50
# ===========================================================================

class ResNet50Encoder(BaseEncoder):
    def __init__(self, pretrained=True, **kwargs):
        super().__init__(**kwargs)
        weights = 'imagenet' if pretrained else None
        base    = keras.applications.ResNet50(
            include_top=False, pooling='avg', weights=weights)
        self.backbone = base
        self._build_heads(2048)


# ===========================================================================
# ResNet-18 (no native Keras app — use a smaller ResNet50 approximation)
# ===========================================================================

class ResNet18Encoder(BaseEncoder):
    """Uses MobileNetV2 as lightweight ~ResNet-18 equivalent."""
    def __init__(self, pretrained=True, **kwargs):
        super().__init__(**kwargs)
        weights = 'imagenet' if pretrained else None
        base    = keras.applications.MobileNetV2(
            include_top=False, pooling='avg', weights=weights)
        self.backbone = base
        self._build_heads(1280)


# ===========================================================================
# EfficientNet-B0
# ===========================================================================

class EfficientNetEncoder(BaseEncoder):
    def __init__(self, pretrained=True, **kwargs):
        super().__init__(**kwargs)
        weights = 'imagenet' if pretrained else None
        base    = keras.applications.EfficientNetB0(
            include_top=False, pooling='avg', weights=weights)
        self.backbone = base
        self._build_heads(1280)


# ===========================================================================
# MobileNetV2
# ===========================================================================

class MobileNetEncoder(BaseEncoder):
    def __init__(self, pretrained=True, **kwargs):
        super().__init__(**kwargs)
        weights = 'imagenet' if pretrained else None
        base    = keras.applications.MobileNetV2(
            include_top=False, pooling='avg', weights=weights)
        self.backbone = base
        self._build_heads(1280)


# ===========================================================================
# Factory
# ===========================================================================

_ENCODER_CLASSES = {
    'resnet50':     ResNet50Encoder,
    'resnet18':     ResNet18Encoder,
    'efficientnet': EfficientNetEncoder,
    'mobilenet':    MobileNetEncoder,
}


def build_encoder(name: str, latent_ch=192, dim_inter=128,
                  dim_unique=64, pretrained=True) -> BaseEncoder:
    if name not in _ENCODER_CLASSES:
        raise ValueError(f"Unknown encoder: {name}. Options: {list(_ENCODER_CLASSES)}")
    return _ENCODER_CLASSES[name](
        pretrained=pretrained,
        latent_ch=latent_ch,
        dim_inter=dim_inter,
        dim_unique=dim_unique,
    )
