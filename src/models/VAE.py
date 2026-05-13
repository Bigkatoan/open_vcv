# Kiến trúc thống nhất: VAE + Intersect-Union Decomposition
#
# Nguyên tắc:
#   - z (latent VAE) = không gian feature tổng quát, có nghĩa nhờ reconstruction loss
#   - v_inter, v_unique = TẬP CON của z (chiếu từ z_vec, không phải từ feature_map)
#   - Tất cả train CÙNG LÚC:
#       L = L_recon + β·L_KL + λ₁·L_inter + λ₂·L_unique + λ₃·L_neg + λ₄·L_aux
#
# Flow:
#   image → Backbone → feature_map (H/8)
#       ├─ image_feat_branch → image_feat       (aug similarity signal)
#       ├─ mu_conv / logvar_conv → z_map
#       │       └─ Decoder → recon              (VAE reconstruction)
#       └─ GAP(z_map) = z_vec
#               ├─ intersect_head → v_inter     (tập con của z)
#               └─ unique_head   → v_unique     (tập con của z)
#
# encoder_layer: out_i = SELU(feature_i(x) * weight_i(x) + bias_i(x))
#   → Vectorized: 3 Conv2d lớn thay vì n vòng lặp Python
# ResNet: Mỗi layer bọc trong ResidualBlock (shortcut 1×1 hoặc identity)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


# ===========================================================================
# ENCODER
# ===========================================================================

class encoder_layer(nn.Module):
    """
    Gated conv — fused: 1 Conv2d(3x channels) thay vì 3 Conv2d riêng lẻ.
    Giảm CUDA kernel launches từ 3 → 1: ~1.8-2x nhanh hơn với batch nhỏ.

        fused  = Conv2d(in_ch, out_ch*n*3, 3, padding=1)
        feat, gate, bias = fused.chunk(3, dim=1)
        out    = LeakyReLU(feat * sigmoid(gate) + bias)

    Input:  (B, in_ch, H, W)
    Output: (B, out_ch*n, H, W)
    """
    def __init__(self, in_ch: int, out_ch: int, n: int):
        super().__init__()
        self.n, self.out_ch = n, out_ch
        total = out_ch * n
        # 1 conv thay vì 3 — memory layout liên tục, 1 kernel launch
        self.fused_conv = nn.Conv2d(in_ch, total * 3, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out              = self.fused_conv(x)          # (B, 3*C, H, W)
        feat, gate, bias = out.chunk(3, dim=1)         # mỗi (B, C, H, W)
        return F.leaky_relu(feat * torch.sigmoid(gate) + bias, 0.01)

    @property
    def out_channels_total(self):
        return self.out_ch * self.n


class ResidualEncoderBlock(nn.Module):
    """encoder_layer + BN + residual shortcut"""
    def __init__(self, in_ch: int, out_ch: int, n: int):
        super().__init__()
        self.layer = encoder_layer(in_ch, out_ch, n)
        ch = self.layer.out_channels_total
        self.bn = nn.BatchNorm2d(ch)
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_ch, ch, 1, bias=False), nn.BatchNorm2d(ch))
            if in_ch != ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.bn(self.layer(x)) + self.shortcut(x), 0.01)

    @property
    def out_channels(self):
        return self.layer.out_channels_total


class BackboneEncoder(nn.Module):
    """
    3 stages + MaxPool.  Input: (B,3,H,W) → Output: (B, out_dim, H/8, W/8)

    Default: stage1→64ch, stage2→128ch, stage3→256ch
    """
    def __init__(self,
                 s1_out=16, s1_heads=4,  s1_blocks=1,
                 s2_out=16, s2_heads=8,  s2_blocks=2,
                 s3_out=16, s3_heads=16, s3_blocks=2,
                 use_checkpoint: bool = True):  # gradient checkpointing — giảm activation memory
        super().__init__()
        self.stage1 = self._make(3,                s1_out, s1_heads, s1_blocks)
        ch1 = s1_out * s1_heads
        self.stage2 = self._make(ch1,              s2_out, s2_heads, s2_blocks)
        ch2 = s2_out * s2_heads
        self.stage3 = self._make(ch2,              s3_out, s3_heads, s3_blocks)
        self.out_dim = s3_out * s3_heads
        self.use_checkpoint = use_checkpoint

    @staticmethod
    def _make(in_ch, out_ch, heads, n_blocks):
        blocks, cur = [], in_ch
        for _ in range(n_blocks):
            blocks.append(ResidualEncoderBlock(cur, out_ch, heads))
            cur = out_ch * heads
        return nn.Sequential(*blocks)

    def _fwd(self, stage, x):
        """Wrapper để dùng với grad_checkpoint."""
        if self.use_checkpoint and x.requires_grad:
            return grad_checkpoint(stage, x, use_reentrant=False)
        return stage(x)

    def forward(self, x):
        x = F.max_pool2d(self._fwd(self.stage1, x), 2)
        x = F.max_pool2d(self._fwd(self.stage2, x), 2)
        x = F.max_pool2d(self._fwd(self.stage3, x), 2)
        return x


# ===========================================================================
# DECODER
# ===========================================================================

class DecoderBlock(nn.Module):
    """
    Residual upsample ×2 (bilinear + Conv, tránh checkerboard):
        x_up = Upsample(x)
        out  = GELU(BN(Conv(x_up)) + shortcut(x_up))
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_up = self.up(x)
        return F.leaky_relu(self.bn(self.conv(x_up)) + self.shortcut(x_up), 0.01)


class Decoder(nn.Module):
    """
    z_map (B, latent_ch, H/8, W/8) → recon (B, 3, H, W)
    3 upsample stages đối xứng với BackboneEncoder.
    """
    def __init__(self, latent_ch=64, ch3=128, ch2=64, ch1=32):
        super().__init__()
        self.stage3   = DecoderBlock(latent_ch, ch3)
        self.stage2   = DecoderBlock(ch3, ch2)
        self.stage1   = DecoderBlock(ch2, ch1)
        self.out_conv = nn.Sequential(
            nn.Conv2d(ch1, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.out_conv(self.stage1(self.stage2(self.stage3(z))))


# ===========================================================================
# PROJECTION HEADS
# ===========================================================================

class SpatialProjectionHead(nn.Module):
    """feature_map (B, C, H', W') → GAP → MLP → L2-norm → (B, out_dim)"""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.mlp(self.gap(x).flatten(1)), p=2, dim=1)


class VectorProjectionHead(nn.Module):
    """z_vec (B, in_dim) → MLP → L2-norm → (B, out_dim)  [tập con của z]"""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z_vec):
        return F.normalize(self.mlp(z_vec), p=2, dim=1)


class ImageFeatBranch(nn.Module):
    """feature_map → aug similarity signal → (B, feat_dim)  L2-norm"""
    def __init__(self, in_dim: int, feat_dim: int = 64):
        super().__init__()
        self.branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(in_dim, feat_dim),
            nn.BatchNorm1d(feat_dim), nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, x):
        return F.normalize(self.branch(x), p=2, dim=1)


# ===========================================================================
# VAE — Mô hình thống nhất
# ===========================================================================

class VAE(nn.Module):
    """
    Mô hình thống nhất: VAE + Intersect-Union Decomposition.

    z là không gian latent tổng quát (được guide bởi reconstruction).
    v_inter và v_unique là TẬP CON của z — chiếu từ z_vec qua MLP.

    Forward trả về:
        recon:      (B, 3, H, W)               — VAE reconstruction
        mu, logvar: (B, latent_ch, H/8, W/8)   — spatial latent params
        image_feat: (B, feat_dim)              — aug similarity (từ feature_map)
        v_inter:    (B, dim_inter)             — tập con của z, L2-norm
        v_unique:   (B, dim_unique)            — tập con của z, L2-norm

    Loss kết hợp (train đồng thời):
        L = L_recon + β·L_KL + λ₁·L_inter + λ₂·L_unique + λ₃·L_neg + λ₄·L_aux
    """
    def __init__(self,
                 # Backbone
                 s1_out=16, s1_heads=4,  s1_blocks=1,
                 s2_out=16, s2_heads=8,  s2_blocks=2,
                 s3_out=16, s3_heads=16, s3_blocks=2,
                 # VAE latent
                 latent_ch: int = 64,
                 # Decoder
                 dec_ch3: int = 128, dec_ch2: int = 64, dec_ch1: int = 32,
                 # Projection (từ z)
                 dim_inter: int = 128,
                 dim_unique: int = 64,
                 # image_feat (từ feature_map)
                 feat_dim: int = 64,
                 hidden_dim: int = 256):
        super().__init__()

        self.backbone = BackboneEncoder(s1_out, s1_heads, s1_blocks,
                                        s2_out, s2_heads, s2_blocks,
                                        s3_out, s3_heads, s3_blocks)
        bb_out = self.backbone.out_dim   # 256

        # VAE: feature_map → z_map (spatial)
        self.feat_bn     = nn.BatchNorm2d(bb_out)              # normalize trước mu_conv
        self.mu_conv     = nn.Conv2d(bb_out, latent_ch, 1)
        self.logvar_conv = nn.Conv2d(bb_out, latent_ch, 1)

        # Decoder: z_map → recon
        self.decoder = Decoder(latent_ch, dec_ch3, dec_ch2, dec_ch1)

        # image_feat: từ feature_map (aug similarity, cần raw appearance)
        self.image_feat_branch = ImageFeatBranch(bb_out, feat_dim)

        # v_inter, v_unique: SLICE TRỰC TIẾP TỪ mu
        # dim_inter + dim_unique phải = latent_ch
        assert dim_inter + dim_unique == latent_ch, (
            f"dim_inter({dim_inter}) + dim_unique({dim_unique}) phải bằng latent_ch({latent_ch})"
        )
        self.dim_inter  = dim_inter
        self.dim_unique = dim_unique
        self.latent_ch  = latent_ch

        self._init_weights()   # đảm bảo mu/logvar gần 0 lúc đầu

    # ------------------------------------------------------------------
    def _init_weights(self):
        """Kaiming init cho Conv2d, Xavier cho Linear, zero bias cho logvar."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init logvar conv → logvar ≈ 0, std ≈ 1 lúc khởi đầu
        if hasattr(self, 'logvar_conv'):
            nn.init.zeros_(self.logvar_conv.weight)
            if self.logvar_conv.bias is not None:
                nn.init.zeros_(self.logvar_conv.bias)
        # Small-init mu_conv → mu ≈ 0 lúc đầu, tránh KL gradient explosion
        if hasattr(self, 'mu_conv'):
            nn.init.normal_(self.mu_conv.weight, 0.0, 0.01)
            if self.mu_conv.bias is not None:
                nn.init.zeros_(self.mu_conv.bias)

    # ------------------------------------------------------------------
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Spatial reparameterization trick.
        logvar clamp [-10, 2] → std ∈ [0.006, 2.7] — tránh KL explosion với AMP.
        """
        logvar = logvar.clamp(-10.0, 2.0)
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, skip_decoder: bool = False):
        """
        Args:
            x           : (B, 3, H, W) — pixel ∈ [0, 1]
            skip_decoder: nếu True, bỏ qua Decoder + KL — chỉ train contrastive losses
        """
        # 1. Backbone → feature_map
        feats = self.backbone(x)                               # (B, 256, H/8, W/8)

        # 2. Aug similarity (từ feature_map, không qua z)
        image_feat = self.image_feat_branch(feats)             # (B, feat_dim)

        # 3. VAE encoder: feature_map → mu
        feats_n = self.feat_bn(feats)                          # BN normalize
        mu_raw  = self.mu_conv(feats_n)
        mu      = (torch.sigmoid(0.1 * mu_raw) - 0.5) * 10.0  # mu ∈ (-5, 5)

        if skip_decoder:
            # Chỉ tính mu, bỏ qua logvar và decoder — tiết kiệm VRAM
            logvar = None
            recon  = None
        else:
            logvar = self.logvar_conv(feats_n).clamp(-10.0, 2.0)
            z_map  = self.reparameterize(mu, logvar)
            recon  = self.decoder(z_map)                       # (B, 3, H, W)

        # 4. v_inter, v_unique = slice trực tiếp từ mu_vec
        mu_vec   = mu.mean(dim=[2, 3])                         # (B, latent_ch)
        v_inter  = F.normalize(mu_vec[:, :self.dim_inter], dim=1)
        v_unique = F.normalize(mu_vec[:, self.dim_inter:], dim=1)

        return recon, mu, logvar, image_feat, v_inter, v_unique

    # ------------------------------------------------------------------
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# LOSS
# ===========================================================================

def vae_loss(recon, target, mu, logvar, beta=1.0):
    """
    L_vae = MSE(recon, target) + β · KL(mu, logvar)

    Là thành phần đảm bảo z có nghĩa tổng quát.
    Kết hợp với intersect-union loss ở trainer:
        L = vae_loss + λ₁·L_inter + λ₂·L_unique + λ₃·L_neg + λ₄·L_aux
    """
    recon_loss = F.mse_loss(recon, target)
    kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# ===========================================================================
# Quick test
# ===========================================================================

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    model = VAE(
        s1_out=16, s1_heads=4,  s1_blocks=1,
        s2_out=16, s2_heads=8,  s2_blocks=2,
        s3_out=16, s3_heads=16, s3_blocks=2,
        latent_ch=64, dec_ch3=128, dec_ch2=64, dec_ch1=32,
        dim_inter=128, dim_unique=64, feat_dim=64, hidden_dim=256,
    ).to(device)

    print(f"Params: {model.count_parameters():,}")
    print("Backbone:")
    for name, stage in [("stage1", model.backbone.stage1),
                        ("stage2", model.backbone.stage2),
                        ("stage3", model.backbone.stage3)]:
        print(f"  {name}: {len(stage)} block(s) → {[b.out_channels for b in stage]} ch")

    x = torch.rand(2, 3, 64, 64).to(device)

    with torch.no_grad():
        recon, mu, logvar, image_feat, v_inter, v_unique = model(x)

    print(f"\n[Forward]")
    print(f"  Input:      {x.shape}")
    print(f"  recon:      {recon.shape}      ← Decoder (VAE path)")
    print(f"  mu/logvar:  {mu.shape}  ← spatial latent")
    print(f"  z_vec dim:  {mu.shape[1]}              ← latent_ch (GAP của z_map)")
    print(f"  v_inter:    {v_inter.shape}    ← tập con của z")
    print(f"  v_unique:   {v_unique.shape}     ← tập con của z")
    print(f"  image_feat: {image_feat.shape}     ← từ feature_map (aug sim)")

    total, rl, kl = vae_loss(recon, x, mu, logvar, beta=1.0)
    print(f"\n[VAE Loss]  recon={rl:.4f}  kl={kl:.4f}  total={total:.4f}")

    print(f"\n[Norm check — phải ≈ 1.0]")
    print(f"  image_feat: {image_feat.norm(dim=1).mean():.4f}")
    print(f"  v_inter:    {v_inter.norm(dim=1).mean():.4f}")
    print(f"  v_unique:   {v_unique.norm(dim=1).mean():.4f}")

    # Gradient flow
    model.train()
    recon, mu, logvar, _, v_i, v_u = model(x)
    total, _, _ = vae_loss(recon, x, mu, logvar)
    (total + v_i.sum() + v_u.sum()).backward()
    n_grad  = sum(1 for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for p in model.parameters())
    print(f"\n[Gradient flow] {n_grad}/{n_total} tensors có grad ✓")