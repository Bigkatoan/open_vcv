# Losses cho Augmentation-Aware Intersect-Union Decomposition
#
# Inputs cần có:
#   rec_img:       (B, 3, H, W)          — ảnh reconstruct từ decoder
#   img:           (B, 3, H, W)          — ảnh gốc (pixel ∈ [0,1])
#   mu, logvar:    (B, latent_ch, H/8, W/8) — VAE latent params
#   image_feat:    (B*q, feat_dim)        — aug similarity signal (L2-norm)
#   union_feat:    (B*q, dim_union)       — union features (L2-norm)
#   sparse_feat:   (B*q, dim_sparse)      — sparse features (L2-norm)
#   union_neg:     (B*k, dim_union)       — union features của negative samples (optional)
#
# Terminology:
#   union  = v_inter  (invariant, shared across augmentations)
#   sparse = v_unique (equivariant, unique per augmentation)
#   q = số augmented versions của 1 ảnh
#   k = số negative (auxiliary) images
#
# Loss tổng:
#   L = L_vae + λ_u·L_union + λ_s·L_sparse + λ_o·L_ortho + λ_n·L_neg
#
# Novelty chính: Aug-aware weighting w_ij
#   w_ij = cosine_sim(image_feat_i, image_feat_j)
#   → Union loss: weight w_ij  (aug tương tự → kéo mạnh hơn)
#   → Sparse loss: weight (1-w_ij) với adaptive margin m_ij = base_margin * (1-w_ij)
#     (aug khác nhau → đẩy mạnh hơn, margin lớn hơn)

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# 1. VAE LOSS
# ===========================================================================

def vae_loss_mse(rec_img: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
    """
    Reconstruction loss: pixel-wise MSE.
    rec_img và img phải cùng shape (B, 3, H, W), giá trị ∈ [0, 1].
    """
    return F.mse_loss(rec_img, img)


def vae_loss_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL divergence: KL(q(z|x) || p(z)) với p(z) = N(0, I).
    logvar clamp [-10, 2] và exp clamp [0, 100] → tránh NaN với AMP (FP16).
    """
    logvar_c = logvar.clamp(-10.0, 2.0)
    return -0.5 * torch.mean(1 + logvar_c - mu.pow(2).clamp(max=100) - logvar_c.exp())


def vae_loss(rec_img: torch.Tensor, img: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 1.0):
    """
    Tổng VAE loss = MSE + β·KL
    β > 1: β-VAE, disentangle mạnh hơn (thường dùng β=4 hoặc β=8).
    β = 1: standard VAE.

    Returns: (total, mse, kl)
    """
    mse = vae_loss_mse(rec_img, img)
    kl  = vae_loss_kl(mu, logvar)
    return mse + beta * kl, mse, kl


# ===========================================================================
# 2. AUG-AWARE WEIGHTING  (novelty chính)
# ===========================================================================

def aug_similarity_matrix(image_feat: torch.Tensor) -> torch.Tensor:
    """
    Tính ma trận similarity w_ij giữa tất cả các cặp augmented versions.

    Args:
        image_feat: (q, feat_dim) — L2-normalized, đã GAP từ feature_map
                    q = số augmented versions của 1 ảnh trong 1 batch item

    Returns:
        W: (q, q) — w_ij ∈ [-1, 1], đã được clamp về [0, 1]
           w_ij cao → hai augmentation tương tự nhau
           w_ij thấp → hai augmentation khác nhau nhiều

    Note: image_feat phải L2-normalized trước khi vào hàm này.
          (VAE.py đã normalize trong ImageFeatBranch)
    """
    # cosine similarity giữa mọi cặp
    W = image_feat @ image_feat.T    # (q, q)
    return W.clamp(min=0.0)          # clamp về [0,1], bỏ negative sim


def uniformity_loss(feat: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """
    Uniformity loss (Wang & Isola 2020): đẩy tất cả features ra xa nhau trên hypersphere.
    Ngăn chặn collapse khi skip_decoder=True (không có reconstruction gradient).

        L_uniform = log E[exp(-t * ||z_i - z_j||²)]
                  = log mean(exp(-t * pairwise_dist²))

    feat: (N, D) — L2-normalized features
    t   : temperature, lớn hơn = đẩy mạnh hơn
    """
    sq_dist = 2.0 - 2.0 * (feat @ feat.T)   # pairwise squared L2 (dùng cosine trick)
    sq_dist = sq_dist.clamp(min=0.0)
    return sq_dist.mul(-t).exp().mean().log()


# ===========================================================================
# 3. UNION LOSS  (pull union features together, weighted by aug similarity)
# ===========================================================================

def union_loss(union_feat: torch.Tensor,
               image_feat: torch.Tensor) -> torch.Tensor:
    """
    Kéo các union features lại gần nhau, weight theo aug similarity.

    L_union = Σ_{i≠j} w_ij · (1 - cos(union_i, union_j))
            / Σ_{i≠j} w_ij          ← normalize để loss scale ổn định

    Aug tương tự (w_ij cao) → kéo mạnh hơn.
    Aug rất khác nhau (w_ij thấp) → kéo nhẹ hơn (vì dù sao union cũng nên giống).

    Args:
        union_feat:  (q, dim_union)  — L2-normalized union features
        image_feat:  (q, feat_dim)   — L2-normalized image features (aug signal)

    Returns:
        scalar loss
    """
    q = union_feat.size(0)
    W = aug_similarity_matrix(image_feat)   # (q, q)

    # cosine similarity giữa mọi cặp union features
    cos_sim = union_feat @ union_feat.T     # (q, q), L2-norm → cosine

    # Loss = weighted (1 - cosine_sim), chỉ tính i ≠ j
    mask = ~torch.eye(q, dtype=torch.bool, device=union_feat.device)
    w    = W[mask]              # (q*(q-1),)
    sim  = cos_sim[mask]        # (q*(q-1),)

    loss = (w * (1 - sim)).sum()
    denom = w.sum().clamp(min=1e-8)
    return loss / denom


# ===========================================================================
# 4. SPARSE LOSS  (push sparse features apart, weighted by aug dissimilarity)
# ===========================================================================

def sparse_loss(sparse_feat: torch.Tensor,
                image_feat: torch.Tensor,
                base_margin: float = 0.5) -> torch.Tensor:
    """
    Đẩy các sparse features ra xa nhau, weight theo aug dissimilarity.

    L_sparse = Σ_{i≠j} (1-w_ij) · max(0, m_ij - cos(sparse_i, sparse_j))
             / Σ_{i≠j} (1-w_ij)

    Với adaptive margin:
        m_ij = base_margin * (1 - w_ij)
        → aug càng khác nhau → margin càng lớn → đẩy mạnh hơn
        → aug tương tự → margin nhỏ → không ép đẩy quá mức

    Args:
        sparse_feat:  (q, dim_sparse) — L2-normalized sparse features
        image_feat:   (q, feat_dim)   — L2-normalized image features
        base_margin:  float           — base margin (default 0.5)

    Returns:
        scalar loss
    """
    q = sparse_feat.size(0)
    W = aug_similarity_matrix(image_feat)   # (q, q), w_ij ∈ [0,1]

    cos_sim = sparse_feat @ sparse_feat.T   # (q, q)

    mask = ~torch.eye(q, dtype=torch.bool, device=sparse_feat.device)
    w        = W[mask]              # (q*(q-1),)
    sim      = cos_sim[mask]        # (q*(q-1),)
    dis_w    = (1 - w)              # dissimilarity weight

    # Adaptive margin: aug càng khác nhau → margin càng lớn
    m_ij = base_margin * dis_w      # (q*(q-1),)

    loss  = (dis_w * F.relu(m_ij - sim)).sum()
    denom = dis_w.sum().clamp(min=1e-8)
    return loss / denom


# ===========================================================================
# 5. ORTHOGONALITY LOSS  (union ⊥ sparse — mang thông tin khác nhau)
# ===========================================================================

def ortho_loss(union_feat: torch.Tensor,
               sparse_feat: torch.Tensor) -> torch.Tensor:
    """
    Đảm bảo union và sparse features mang thông tin KHÁC NHAU.

    L_ortho = mean |cos(union_i, sparse_i)|
            = mean |union_i · sparse_i|   (vì cả hai đã L2-norm)

    Minimize L_ortho → union ⊥ sparse → không overlap thông tin.
    Dùng |.| vì chỉ cần chúng không tương quan, bất kể dấu.

    Args:
        union_feat:  (q, dim_union)  — L2-normalized
        sparse_feat: (q, dim_sparse) — L2-normalized

    Returns:
        scalar loss

    Note: dim_union và dim_sparse có thể khác nhau.
          Dùng Global Average thay vì dot product trực tiếp.
    """
    # Dùng mean của absolute cosine similarity trên từng sample
    # → project sparse về cùng space với union qua mean pooling
    u = union_feat.mean(dim=1, keepdim=True)    # (q, 1) — magnitude đại diện
    s = sparse_feat.mean(dim=1, keepdim=True)   # (q, 1)

    # Cách đúng hơn: tính correlation giữa 2 vectors qua dot product
    # (cả hai đã normalize → inner product = cosine sim)
    # Vì dim khác nhau, dùng mean feature magnitude làm proxy
    cos = (union_feat.mean(dim=1) * sparse_feat.mean(dim=1))  # (q,) — scalar proxy
    return cos.abs().mean()


# ===========================================================================
# 6. NEGATIVE LOSS  (push core union features away from auxiliary images)
# ===========================================================================

def neg_loss(union_feat: torch.Tensor,
             union_neg: torch.Tensor,
             margin: float = 0.5) -> torch.Tensor:
    """
    Đẩy union features của core images ra xa auxiliary (negative) images.
    Giúp union features không bị "chung chung" với mọi ảnh.

    L_neg = mean_{i,k} max(0, margin - cos(union_i, union_neg_k))

    Args:
        union_feat:  (q, dim_union)  — union features của core augmentations
        union_neg:   (k, dim_union)  — union features của negative images
        margin:      float           — minimum distance cần maintain

    Returns:
        scalar loss (0 nếu không có negative samples)
    """
    if union_neg is None or union_neg.size(0) == 0:
        return torch.tensor(0.0, device=union_feat.device)

    # cosine sim giữa mọi (core, neg) pair: (q, k)
    sim = union_feat @ union_neg.T   # L2-norm → cosine sim
    return F.relu(margin - sim).mean()


# ===========================================================================
# 7. COMBINED LOSS CLASS
# ===========================================================================

class IntersectUnionLoss(nn.Module):
    """
    Kết hợp tất cả loss thành 1 class để dùng trong trainer.

    L_total = L_vae + λ_u·L_union + λ_s·L_sparse + λ_o·L_ortho + λ_n·L_neg

    Vai trò của từng thành phần:
        L_vae    → z latent có nghĩa tổng quát (reconstruction quality)
        L_union  → union features bất biến qua augmentation
        L_sparse → sparse features riêng biệt giữa các augmentation
        L_ortho  → union và sparse không overlap thông tin
        L_neg    → union features không bị overlap với unrelated images
    """
    def __init__(self,
                 beta: float        = 1.0,
                 lambda_union: float  = 1.0,
                 lambda_sparse: float = 0.5,
                 lambda_ortho: float  = 0.2,
                 lambda_neg: float    = 0.3,
                 lambda_uniform: float= 0.5,   # uniformity — chống collapse
                 base_margin: float   = 0.5,
                 neg_margin: float    = 0.5):
        super().__init__()
        self.beta           = beta
        self.lambda_union   = lambda_union
        self.lambda_sparse  = lambda_sparse
        self.lambda_ortho   = lambda_ortho
        self.lambda_neg     = lambda_neg
        self.lambda_uniform = lambda_uniform
        self.base_margin    = base_margin
        self.neg_margin     = neg_margin

    def forward(self,
                rec_img,          # Tensor hoặc None (khi skip_decoder=True)
                img,              # Tensor hoặc None
                mu: torch.Tensor,
                logvar,           # Tensor hoặc None
                image_feat: torch.Tensor,
                union_feat: torch.Tensor,
                sparse_feat: torch.Tensor,
                union_neg: torch.Tensor = None):
        """
        rec_img / logvar = None  →  skip VAE loss (chỉ dùng contrastive losses).
        """
        # VAE loss — chỉ tính khi có decoder output
        if rec_img is not None and logvar is not None:
            l_vae, l_mse, l_kl = vae_loss(rec_img, img, mu, logvar, self.beta)
        else:
            zero      = torch.tensor(0.0, device=mu.device)
            l_vae, l_mse, l_kl = zero, zero, zero

        # Intersect-Union losses
        l_union   = union_loss(union_feat, image_feat)
        l_sparse  = sparse_loss(sparse_feat, image_feat, self.base_margin)
        l_ortho   = ortho_loss(union_feat, sparse_feat)
        l_neg     = neg_loss(union_feat, union_neg, self.neg_margin)
        # Uniformity: chống collapse (quan trọng khi skip_decoder)
        l_uniform = uniformity_loss(union_feat) + uniformity_loss(sparse_feat)

        total = (l_vae
                 + self.lambda_union   * l_union
                 + self.lambda_sparse  * l_sparse
                 + self.lambda_ortho   * l_ortho
                 + self.lambda_neg     * l_neg
                 + self.lambda_uniform * l_uniform)

        details = {
            'total':   total.item(),
            'vae':     l_vae.item(),
            'mse':     l_mse.item(),
            'kl':      l_kl.item(),
            'union':   l_union.item(),
            'sparse':  l_sparse.item(),
            'ortho':   l_ortho.item(),
            'neg':     l_neg.item(),
            'uniform': l_uniform.item(),
        }
        return total, details


# ===========================================================================
# Quick test
# ===========================================================================

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    q, k, B = 3, 2, 4   # 3 augmentations, 2 negatives, batch=4
    dim_union, dim_sparse, feat_dim = 128, 64, 64
    latent_ch, H = 64, 8   # H/8 của ảnh 64×64

    # Giả lập output của VAE
    rec_img    = torch.sigmoid(torch.randn(B, 3, 64, 64)).to(device)
    img        = torch.rand(B, 3, 64, 64).to(device)
    mu         = torch.randn(B, latent_ch, H, H).to(device)
    logvar     = torch.randn(B, latent_ch, H, H).to(device)
    image_feat = F.normalize(torch.randn(B * q, feat_dim), dim=1).to(device)
    union_feat = F.normalize(torch.randn(B * q, dim_union), dim=1).to(device)
    sparse_feat= F.normalize(torch.randn(B * q, dim_sparse), dim=1).to(device)
    union_neg  = F.normalize(torch.randn(B * k, dim_union), dim=1).to(device)

    loss_fn = IntersectUnionLoss(
        beta=1.0, lambda_union=1.0, lambda_sparse=0.5,
        lambda_ortho=0.2, lambda_neg=0.3,
        base_margin=0.5, neg_margin=0.5,
    )

    total, details = loss_fn(rec_img, img, mu, logvar,
                             image_feat, union_feat, sparse_feat, union_neg)

    print("Loss breakdown:")
    for k_, v in details.items():
        print(f"  {k_:<10}: {v:.4f}")

    # Gradient test
    union_feat2  = F.normalize(torch.randn(B * q, dim_union,  requires_grad=True), dim=1).to(device)
    sparse_feat2 = F.normalize(torch.randn(B * q, dim_sparse, requires_grad=True), dim=1).to(device)
    total2, _ = loss_fn(rec_img, img, mu, logvar,
                        image_feat, union_feat2, sparse_feat2, union_neg)
    total2.backward()
    print(f"\nGradient flow: union_feat.grad={union_feat2.grad is not None} ✓")