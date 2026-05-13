# VAE + Intersect-Union Decomposition — TensorFlow/Keras
#
# Flow:
#   image (NHWC) → Backbone → feature_map (B, H/8, W/8, bb_out)
#       ├─ image_feat_branch → image_feat  (B, feat_dim)
#       ├─ mu_conv / logvar_conv → z_map
#       │       └─ Decoder → recon
#       └─ GAP(z_map) = z_vec
#               ├─ dim_inter → v_inter   (L2-norm)
#               └─ dim_unique → v_unique (L2-norm)

import tensorflow as tf
import keras
from keras import layers


# ===========================================================================
# ENCODER
# ===========================================================================

class GatedConvLayer(keras.layers.Layer):
    """
    Gated conv (fused): out = LeakyReLU(feat * sigmoid(gate) + bias)
    Input:  (B, H, W, in_ch)
    Output: (B, H, W, out_ch*n)
    """
    def __init__(self, out_ch, n, **kwargs):
        super().__init__(**kwargs)
        self.out_ch = out_ch
        self.n      = n
        total = out_ch * n
        # 1 conv thay vì 3 — 1 kernel launch
        self.fused_conv = layers.Conv2D(
            total * 3, 3, padding='same', use_bias=True,
            kernel_initializer='he_normal',
        )

    def call(self, x, training=False):
        out             = self.fused_conv(x)                     # (B, H, W, 3C)
        feat, gate, bias = tf.split(out, 3, axis=-1)             # mỗi (B, H, W, C)
        return tf.nn.leaky_relu(feat * tf.sigmoid(gate) + bias, alpha=0.01)


class ResidualEncoderBlock(keras.layers.Layer):
    def __init__(self, out_ch, n, **kwargs):
        super().__init__(**kwargs)
        self.out_ch = out_ch * n
        self.layer  = GatedConvLayer(out_ch, n)
        self.bn     = layers.BatchNormalization()
        self._shortcut_conv = None   # built lazily in `build`

    def build(self, input_shape):
        in_ch = input_shape[-1]
        if in_ch != self.out_ch:
            self._shortcut_conv = keras.Sequential([
                layers.Conv2D(self.out_ch, 1, use_bias=False, kernel_initializer='he_normal'),
                layers.BatchNormalization(),
            ])
        super().build(input_shape)

    def call(self, x, training=False):
        h = self.bn(self.layer(x, training=training), training=training)
        if self._shortcut_conv is not None:
            sc = self._shortcut_conv(x, training=training)
        else:
            sc = x
        return tf.nn.leaky_relu(h + sc, alpha=0.01)


class BackboneEncoder(keras.layers.Layer):
    """3 stages + MaxPool. (B,H,W,3) → (B,H/8,W/8,out_dim)"""
    def __init__(self,
                 s1_out=16, s1_heads=4, s1_blocks=1,
                 s2_out=16, s2_heads=8, s2_blocks=2,
                 s3_out=16, s3_heads=16, s3_blocks=2,
                 **kwargs):
        super().__init__(**kwargs)

        def _make_stage(in_ch_hint, out_ch, heads, n_blocks):
            return [ResidualEncoderBlock(out_ch, heads) for _ in range(n_blocks)]

        self.stage1_blocks = _make_stage(3,               s1_out, s1_heads, s1_blocks)
        self.stage2_blocks = _make_stage(s1_out*s1_heads, s2_out, s2_heads, s2_blocks)
        self.stage3_blocks = _make_stage(s2_out*s2_heads, s3_out, s3_heads, s3_blocks)
        self.pool          = layers.MaxPooling2D(2)
        self.out_dim       = s3_out * s3_heads

    def call(self, x, training=False):
        for blk in self.stage1_blocks:
            x = blk(x, training=training)
        x = self.pool(x)
        for blk in self.stage2_blocks:
            x = blk(x, training=training)
        x = self.pool(x)
        for blk in self.stage3_blocks:
            x = blk(x, training=training)
        x = self.pool(x)
        return x


# ===========================================================================
# DECODER
# ===========================================================================

class DecoderBlock(keras.layers.Layer):
    """Upsample ×2 + residual Conv."""
    def __init__(self, out_ch, **kwargs):
        super().__init__(**kwargs)
        self.out_ch   = out_ch
        self.up       = layers.UpSampling2D(2, interpolation='bilinear')
        self.conv     = layers.Conv2D(out_ch, 3, padding='same', use_bias=False,
                                      kernel_initializer='he_normal')
        self.bn       = layers.BatchNormalization()
        self._sc_conv = None  # built lazily

    def build(self, input_shape):
        in_ch = input_shape[-1]
        if in_ch != self.out_ch:
            self._sc_conv = keras.Sequential([
                layers.Conv2D(self.out_ch, 1, use_bias=False, kernel_initializer='he_normal'),
                layers.BatchNormalization(),
            ])
        super().build(input_shape)

    def call(self, x, training=False):
        x_up = self.up(x)
        h    = self.bn(self.conv(x_up), training=training)
        if self._sc_conv is not None:
            sc = self._sc_conv(x_up, training=training)
        else:
            sc = x_up
        return tf.nn.leaky_relu(h + sc, alpha=0.01)


class Decoder(keras.layers.Layer):
    """z_map (B, H/8, W/8, latent_ch) → recon (B, H, W, 3)"""
    def __init__(self, ch3=128, ch2=64, ch1=32, **kwargs):
        super().__init__(**kwargs)
        self.stage3   = DecoderBlock(ch3)
        self.stage2   = DecoderBlock(ch2)
        self.stage1   = DecoderBlock(ch1)
        self.out_conv = layers.Conv2D(3, 1, activation='sigmoid',
                                      kernel_initializer='glorot_uniform')

    def call(self, z, training=False):
        x = self.stage3(z, training=training)
        x = self.stage2(x, training=training)
        x = self.stage1(x, training=training)
        return self.out_conv(x)


# ===========================================================================
# PROJECTION HEADS
# ===========================================================================

class ImageFeatBranch(keras.layers.Layer):
    """feature_map → aug similarity signal (B, feat_dim) L2-norm"""
    def __init__(self, feat_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.gap = layers.GlobalAveragePooling2D()
        self.d1  = layers.Dense(feat_dim, use_bias=False,
                                kernel_initializer='glorot_uniform')
        self.bn  = layers.BatchNormalization()
        self.d2  = layers.Dense(feat_dim, kernel_initializer='glorot_uniform')

    def call(self, x, training=False):
        h = self.gap(x)
        h = tf.nn.gelu(self.bn(self.d1(h), training=training))
        h = self.d2(h)
        return tf.math.l2_normalize(h, axis=-1)


# ===========================================================================
# VAE
# ===========================================================================

class VAE(keras.Model):
    """
    Mô hình thống nhất: VAE + Intersect-Union Decomposition.

    Forward returns:
        recon:      (B, H, W, 3) or None
        mu:         (B, latent_ch)
        logvar:     (B, H/8, W/8, latent_ch) or None
        image_feat: (B, feat_dim)
        v_inter:    (B, dim_inter)   L2-norm
        v_unique:   (B, dim_unique)  L2-norm
    """
    def __init__(self,
                 s1_out=16, s1_heads=4,  s1_blocks=1,
                 s2_out=16, s2_heads=8,  s2_blocks=2,
                 s3_out=16, s3_heads=16, s3_blocks=2,
                 latent_ch=192,
                 dec_ch3=128, dec_ch2=64, dec_ch1=32,
                 dim_inter=128, dim_unique=64,
                 feat_dim=64, hidden_dim=256,
                 **kwargs):
        super().__init__(**kwargs)

        assert dim_inter + dim_unique == latent_ch, (
            f"dim_inter({dim_inter}) + dim_unique({dim_unique}) != latent_ch({latent_ch})"
        )
        self.dim_inter  = dim_inter
        self.dim_unique = dim_unique
        self.latent_ch  = latent_ch

        self.backbone = BackboneEncoder(
            s1_out, s1_heads, s1_blocks,
            s2_out, s2_heads, s2_blocks,
            s3_out, s3_heads, s3_blocks,
        )
        bb_out = self.backbone.out_dim

        # VAE spatial latent (used when skip_decoder=False)
        self.feat_bn     = layers.BatchNormalization()
        self.mu_conv     = layers.Conv2D(latent_ch, 1,
                                         kernel_initializer=keras.initializers.RandomNormal(0, 0.01))
        self.logvar_conv = layers.Conv2D(latent_ch, 1,
                                         kernel_initializer='zeros', bias_initializer='zeros')

        # Fast path (skip_decoder=True): GAP → BN → Linear
        self.mu_linear = keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(latent_ch, use_bias=False,
                         kernel_initializer=keras.initializers.RandomNormal(0, 0.01)),
        ])

        # Decoder
        self.decoder = Decoder(dec_ch3, dec_ch2, dec_ch1)

        # image_feat branch
        self.image_feat_branch = ImageFeatBranch(feat_dim)

    def call(self, x, skip_decoder=False, training=False):
        """
        x: (B, H, W, 3)  pixel ∈ [0, 1]
        Returns: (recon, mu, logvar, image_feat, v_inter, v_unique)
        """
        feats      = self.backbone(x, training=training)          # (B, H/8, W/8, bb_out)
        image_feat = self.image_feat_branch(feats, training=training)  # (B, feat_dim)

        if skip_decoder:
            mu_raw = self.mu_linear(feats, training=training)     # (B, latent_ch)
            mu_vec = (tf.sigmoid(0.1 * mu_raw) - 0.5) * 10.0
            mu     = mu_vec
            logvar = None
            recon  = None
        else:
            feats_n = self.feat_bn(feats, training=training)
            mu_raw  = self.mu_conv(feats_n)                       # (B, H/8, W/8, latent_ch)
            mu_map  = (tf.sigmoid(0.1 * mu_raw) - 0.5) * 10.0
            logvar  = tf.clip_by_value(self.logvar_conv(feats_n), -10.0, 2.0)
            z_map   = self._reparameterize(mu_map, logvar, training)
            recon   = self.decoder(z_map, training=training)      # (B, H, W, 3)
            # GAP → z_vec
            mu_vec  = tf.reduce_mean(mu_map, axis=[1, 2])         # (B, latent_ch)
            mu      = mu_vec

        v_inter  = tf.math.l2_normalize(mu_vec[:, :self.dim_inter],  axis=-1)
        v_unique = tf.math.l2_normalize(mu_vec[:, self.dim_inter:],  axis=-1)

        return recon, mu, logvar, image_feat, v_inter, v_unique

    def _reparameterize(self, mu, logvar, training):
        if training:
            eps = tf.random.normal(tf.shape(mu))
            return mu + eps * tf.exp(0.5 * logvar)
        return mu

    def count_parameters(self):
        return sum(tf.size(v).numpy() for v in self.trainable_variables)


# ===========================================================================
# Quick test
# ===========================================================================

if __name__ == "__main__":
    model = VAE(
        s1_out=8, s1_heads=4,  s1_blocks=1,
        s2_out=8, s2_heads=8,  s2_blocks=2,
        s3_out=8, s3_heads=16, s3_blocks=2,
        latent_ch=96, dec_ch3=64, dec_ch2=32, dec_ch1=16,
        dim_inter=64, dim_unique=32, feat_dim=32, hidden_dim=128,
    )
    x = tf.random.normal((2, 32, 32, 3))
    recon, mu, logvar, imgf, vi, vu = model(x, skip_decoder=True, training=False)
    print(f"skip=True  mu:{mu.shape}  v_inter:{vi.shape}  v_unique:{vu.shape}")
    recon, mu, logvar, imgf, vi, vu = model(x, skip_decoder=False, training=False)
    print(f"skip=False recon:{recon.shape}  mu:{mu.shape}")
    print(f"Params: {model.count_parameters():,}")
