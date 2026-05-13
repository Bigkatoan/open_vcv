# Losses — Augmentation-Aware Intersect-Union Decomposition (TensorFlow)
#
# Inputs:
#   rec_img:     (B, H, W, 3)           — ảnh reconstruct (None nếu skip_decoder)
#   img:         (B, H, W, 3)           — ảnh gốc
#   mu, logvar:  (B, latent_ch)         — VAE latent params
#   image_feat:  (B*q, feat_dim)        — aug similarity signal (L2-norm)
#   union_feat:  (B*q, dim_inter)       — invariant features (L2-norm)
#   sparse_feat: (B*q, dim_unique)      — equivariant features (L2-norm)
#   union_neg:   (B*q, dim_inter)       — negative features (in-batch)

import tensorflow as tf
import keras


# ===========================================================================
# 1. VAE LOSS
# ===========================================================================

def vae_loss_mse(rec_img, img):
    return tf.reduce_mean(tf.square(rec_img - img))


def vae_loss_kl(mu, logvar):
    logvar_c = tf.clip_by_value(logvar, -10.0, 2.0)
    return -0.5 * tf.reduce_mean(
        1.0 + logvar_c - tf.minimum(tf.square(mu), 100.0) - tf.exp(logvar_c)
    )


def vae_loss(rec_img, img, mu, logvar, beta=1.0):
    mse = vae_loss_mse(rec_img, img)
    kl  = vae_loss_kl(mu, logvar)
    return mse + beta * kl, mse, kl


# ===========================================================================
# 2. UNIFORMITY LOSS
# ===========================================================================

def uniformity_loss(feat, t=2.0):
    """Uniformity on hypersphere (Wang & Isola 2020). feat: (N, D) L2-normalized."""
    sq_dist = tf.maximum(2.0 - 2.0 * tf.matmul(feat, tf.transpose(feat)), 0.0)
    return tf.math.log(tf.reduce_mean(tf.exp(-t * sq_dist)))


# ===========================================================================
# 3. UNION LOSS  (aug-aware weighting — novelty chính)
# ===========================================================================

def union_loss(union_feat, image_feat, q=3):
    """
    Kéo union features lại gần nhau, weight theo aug similarity.
    Per-image computation với (B, q, q) bmm — không mix cross-image.

    L_union = mean_B [ Σ_{i≠j} w_ij · (1 - cos(u_i, u_j)) / Σ_{i≠j} w_ij ]
    """
    B = tf.shape(union_feat)[0] // q
    u = tf.reshape(union_feat, (B, q, -1))   # (B, q, D)
    f = tf.reshape(image_feat, (B, q, -1))   # (B, q, F)

    W     = tf.matmul(f, tf.transpose(f, [0, 2, 1]))              # (B, q, q)
    W     = tf.maximum(W, 0.0)
    cos_s = tf.matmul(u, tf.transpose(u, [0, 2, 1]))              # (B, q, q)

    mask  = 1.0 - tf.eye(q, dtype=W.dtype)                        # (q, q), 0 diagonal
    w_off = W * mask
    d_off = (1.0 - cos_s) * mask

    loss  = tf.reduce_sum(w_off * d_off, axis=[1, 2])             # (B,)
    denom = tf.maximum(tf.reduce_sum(w_off, axis=[1, 2]), 1e-8)   # (B,)
    return tf.reduce_mean(loss / denom)


# ===========================================================================
# 4. SPARSE LOSS
# ===========================================================================

def sparse_loss(sparse_feat, image_feat, base_margin=0.5, q=3):
    """
    Đẩy sparse features ra xa nhau, weight theo aug dissimilarity.

    L_sparse = mean_B [ Σ_{i≠j} (1-w_ij)·max(0, m_ij - cos(s_i,s_j)) / Σ_{i≠j}(1-w_ij) ]
    """
    B = tf.shape(sparse_feat)[0] // q
    s = tf.reshape(sparse_feat, (B, q, -1))
    f = tf.reshape(image_feat,  (B, q, -1))

    W     = tf.maximum(tf.matmul(f, tf.transpose(f, [0, 2, 1])), 0.0)
    cos_s = tf.matmul(s, tf.transpose(s, [0, 2, 1]))

    dis_w = 1.0 - W
    m_ij  = base_margin * dis_w

    mask     = 1.0 - tf.eye(q, dtype=W.dtype)
    dis_w_off= dis_w * mask
    hinge    = tf.nn.relu(m_ij - cos_s) * mask

    loss  = tf.reduce_sum(dis_w_off * hinge, axis=[1, 2])
    denom = tf.maximum(tf.reduce_sum(dis_w_off, axis=[1, 2]), 1e-8)
    return tf.reduce_mean(loss / denom)


# ===========================================================================
# 5. ORTHOGONALITY LOSS
# ===========================================================================

def ortho_loss(union_feat, sparse_feat):
    """Cross-covariance (Barlow Twins style). Minimize ||u^T s / N||_F^2."""
    N = tf.cast(tf.shape(union_feat)[0], tf.float32)
    cross = tf.matmul(tf.transpose(union_feat), sparse_feat) / N  # (D_inter, D_unique)
    return tf.reduce_mean(tf.square(cross))


# ===========================================================================
# 6. NEGATIVE LOSS
# ===========================================================================

def neg_loss(union_feat, union_neg, margin=0.5):
    """Đẩy core union features ra xa negative images."""
    if union_neg is None:
        return tf.zeros((), dtype=tf.float32)
    return tf.cond(
        tf.equal(tf.size(union_neg), 0),
        lambda: tf.zeros((), dtype=tf.float32),
        lambda: tf.reduce_mean(tf.nn.relu(
            margin - tf.matmul(union_feat, tf.transpose(union_neg))
        )),
    )


# ===========================================================================
# 7. COMBINED LOSS
# ===========================================================================

class IntersectUnionLoss(keras.layers.Layer):
    """
    L_total = L_vae + λ_u·L_union + λ_s·L_sparse + λ_o·L_ortho
            + λ_n·L_neg + λ_uni·L_uniform + λ_aux·L_aux
    """
    def __init__(self,
                 beta=1.0, lambda_union=1.0, lambda_sparse=0.5,
                 lambda_ortho=0.2, lambda_neg=0.3,
                 lambda_uniform=0.5, lambda_aux=0.1,
                 base_margin=0.5, neg_margin=0.5, **kwargs):
        super().__init__(**kwargs)
        self.beta            = beta
        self.lambda_union    = lambda_union
        self.lambda_sparse   = lambda_sparse
        self.lambda_ortho    = lambda_ortho
        self.lambda_neg      = lambda_neg
        self.lambda_uniform  = lambda_uniform
        self.lambda_aux      = lambda_aux
        self.base_margin     = base_margin
        self.neg_margin      = neg_margin

    def call(self, rec_img, img, mu, logvar,
             image_feat, union_feat, sparse_feat,
             union_neg=None, q=3):
        """rec_img / logvar = None → skip VAE loss."""
        # Cast to float32 for numerical stability
        union_feat  = tf.cast(union_feat,  tf.float32)
        sparse_feat = tf.cast(sparse_feat, tf.float32)
        image_feat  = tf.cast(image_feat,  tf.float32)

        # VAE loss
        if rec_img is not None and logvar is not None:
            rec_img  = tf.cast(rec_img,  tf.float32)
            img      = tf.cast(img,      tf.float32)
            mu       = tf.cast(mu,       tf.float32)
            logvar   = tf.cast(logvar,   tf.float32)
            l_vae, l_mse, l_kl = vae_loss(rec_img, img, mu, logvar, self.beta)
        else:
            zero = tf.zeros((), dtype=tf.float32)
            l_vae, l_mse, l_kl = zero, zero, zero

        l_union   = union_loss(union_feat, image_feat, q)
        l_sparse  = sparse_loss(sparse_feat, image_feat, self.base_margin, q)
        l_ortho   = ortho_loss(union_feat, sparse_feat)
        l_neg     = neg_loss(union_feat, union_neg, self.neg_margin)
        l_uniform = uniformity_loss(union_feat) + uniformity_loss(sparse_feat)

        if union_neg is not None:
            union_neg_f = tf.cast(union_neg, tf.float32)
            l_aux = tf.cond(
                tf.equal(tf.size(union_neg_f), 0),
                lambda: tf.zeros((), dtype=tf.float32),
                lambda: uniformity_loss(union_neg_f),
            )
        else:
            l_aux = tf.zeros((), dtype=tf.float32)

        total = (l_vae
                 + self.lambda_union   * l_union
                 + self.lambda_sparse  * l_sparse
                 + self.lambda_ortho   * l_ortho
                 + self.lambda_neg     * l_neg
                 + self.lambda_uniform * l_uniform
                 + self.lambda_aux     * l_aux)

        # Return TF scalars — caller converts to float only when needed (avoid per-step sync)
        details = {
            'total':   total,
            'vae':     l_vae,
            'mse':     l_mse,
            'kl':      l_kl,
            'union':   l_union,
            'sparse':  l_sparse,
            'ortho':   l_ortho,
            'neg':     l_neg,
            'uniform': l_uniform,
            'aux':     l_aux,
        }
        return total, details
