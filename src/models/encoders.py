"""
encoders.py — Alternative backbone encoders để so sánh với ResNet Gated VAE.

Tất cả đều implement cùng interface:
    forward(x, skip_decoder=True) →
        (None, mu_vec, None, image_feat, v_inter, v_unique)

Supported:
    - ResNet18Encoder       (~12M  params, lightweight baseline)
    - ResNet50Encoder       (~23M  params, standard baseline)
    - EfficientNetV2SEncoder(~21M  params, efficient modern)
    - ConvNeXtTEncoder      (~28M  params, modern ConvNet)
    - SwinTEncoder          (~28M  params, Swin Transformer)
    - ViTB16Encoder         (~86M  params, ViT-B/16)
    - DINOv2ViTSEncoder     (~21M  params, self-supervised ViT-S/14)

Cách dùng với main.py:
    python main.py --encoder resnet18
    python main.py --encoder resnet50
    python main.py --encoder efficientnet_v2_s
    python main.py --encoder convnext_t
    python main.py --encoder swin_t
    python main.py --encoder vit_b16
    python main.py --encoder dinov2_s
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


# ===========================================================================
# Base class — shared logic cho tất cả encoders
# ===========================================================================

class BaseEncoder(nn.Module):
    """
    Base encoder: backbone → latent projection → slice v_inter/v_unique.

    Subclass chỉ cần implement:
        self.backbone    : nn.Module   — output (B, backbone_dim)
        self.backbone_dim: int         — số channels của backbone output
    """

    def __init__(self,
                 latent_ch:  int = 2048,
                 dim_inter:  int = 1024,
                 dim_unique: int = 1024,
                 feat_dim:   int = 64):
        super().__init__()

        assert dim_inter + dim_unique == latent_ch, (
            f"dim_inter({dim_inter}) + dim_unique({dim_unique}) phải bằng latent_ch({latent_ch})"
        )
        self.latent_ch  = latent_ch
        self.dim_inter  = dim_inter
        self.dim_unique = dim_unique

        # Sẽ được set trong subclass trước khi gọi _build_heads()
        self.backbone_dim = None

    def _build_heads(self):
        """Gọi sau khi backbone_dim được set trong subclass."""
        d = self.backbone_dim

        # Latent projection: backbone → mu_vec (với BN + sigmoid clamping)
        self.latent_proj = nn.Sequential(
            nn.Linear(d, self.latent_ch, bias=False),
            nn.BatchNorm1d(self.latent_ch),
        )

        # image_feat branch: cho aug similarity weighting
        self.image_feat_proj = nn.Sequential(
            nn.Linear(d, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.dim_inter // 4),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone forward → flat feature vector (B, backbone_dim)."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor, skip_decoder: bool = True):
        """
        Args:
            x           : (B, 3, H, W)
            skip_decoder: không dùng, chỉ để tương thích interface với VAE
        Returns:
            (None, mu_vec, None, image_feat, v_inter, v_unique)
        """
        feats = self._encode(x)                            # (B, backbone_dim)

        # image_feat — cho aug similarity weighting
        image_feat = F.normalize(self.image_feat_proj(feats), dim=1)  # (B, feat_dim)

        # Latent projection + soft-clamp (sigmoid k=0.1)
        mu_raw  = self.latent_proj(feats)                  # (B, latent_ch)
        mu_vec  = (torch.sigmoid(0.1 * mu_raw) - 0.5) * 10.0  # mu ∈ (-5, 5)

        # Slice: v_inter = mu[:dim_inter], v_unique = mu[dim_inter:]
        v_inter  = F.normalize(mu_vec[:, :self.dim_inter],  dim=1)   # (B, dim_inter)
        v_unique = F.normalize(mu_vec[:, self.dim_inter:],  dim=1)   # (B, dim_unique)

        return None, mu_vec, None, image_feat, v_inter, v_unique

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# 1. ResNet-18 (~11M params)
# ===========================================================================

class ResNet18Encoder(BaseEncoder):
    """
    Backbone: ResNet-18 (không có FC layer).
    Output dim: 512.
    Rất nhanh, phù hợp làm baseline.
    """
    def __init__(self, pretrained: bool = True, **kwargs):
        super().__init__(**kwargs)

        backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        # Bỏ FC layer, giữ đến AvgPool
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.backbone_dim = 512
        self._build_heads()

    def _encode(self, x):
        return self.backbone(x).flatten(1)    # (B, 512)


# ===========================================================================
# 2. ResNet-50 (~23M params)
# ===========================================================================

class ResNet50Encoder(BaseEncoder):
    """
    Backbone: ResNet-50 (không có FC layer).
    Output dim: 2048.
    Standard choice cho contrastive learning (SimCLR, MoCo).
    """
    def __init__(self, pretrained: bool = True, **kwargs):
        super().__init__(**kwargs)

        backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT if pretrained else None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.backbone_dim = 2048
        self._build_heads()

    def _encode(self, x):
        return self.backbone(x).flatten(1)    # (B, 2048)


# ===========================================================================
# 3. EfficientNetV2-S (~21M params)
# ===========================================================================

class EfficientNetV2SEncoder(BaseEncoder):
    """
    Backbone: EfficientNetV2-S (ImageNet-1K pretrained).
    Output dim: 1280.
    Nhanh hơn EfficientNetV1 nhờ Fused-MBConv.
    """
    def __init__(self, pretrained: bool = True, **kwargs):
        super().__init__(**kwargs)

        backbone = tvm.efficientnet_v2_s(
            weights=tvm.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None)
        self.backbone     = backbone.features
        self.pool         = backbone.avgpool
        self.backbone_dim = 1280
        self._build_heads()

    def _encode(self, x):
        return self.pool(self.backbone(x)).flatten(1)   # (B, 1280)


# ===========================================================================
# 4. ConvNeXt-Tiny (~28M params)
# ===========================================================================

class ConvNeXtTEncoder(BaseEncoder):
    """
    Backbone: ConvNeXt-Tiny (ImageNet-1K pretrained).
    Output dim: 768.
    Modern ConvNet — tiệm cận ViT accuracy với FLOPs tương đương.
    """
    def __init__(self, pretrained: bool = True, **kwargs):
        super().__init__(**kwargs)

        backbone = tvm.convnext_tiny(
            weights=tvm.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        self.backbone     = backbone.features
        self.pool         = backbone.avgpool        # AdaptiveAvgPool2d(1)
        self.backbone_dim = 768
        self._build_heads()

    def _encode(self, x):
        return self.pool(self.backbone(x)).flatten(1)   # (B, 768)


# ===========================================================================
# 5. Swin Transformer-Tiny (~28M params)
# ===========================================================================

class SwinTEncoder(BaseEncoder):
    """
    Backbone: Swin-T (ImageNet-1K pretrained).
    Output dim: 768.
    Hierarchical window attention — tốt trên dense tasks.

    Note: Swin output là NHWC → cần permute trước pooling.
    """
    def __init__(self, pretrained: bool = True, **kwargs):
        super().__init__(**kwargs)

        backbone = tvm.swin_t(
            weights=tvm.Swin_T_Weights.DEFAULT if pretrained else None)
        # features[-2] = patch merging stage, features[-1] = LayerNorm
        self.features     = backbone.features
        self.norm         = backbone.norm
        self.pool         = nn.AdaptiveAvgPool2d(1)
        self.backbone_dim = 768
        self._build_heads()

    def _encode(self, x):
        feats = self.features(x)                  # (B, H', W', C) NHWC
        feats = self.norm(feats)
        feats = feats.permute(0, 3, 1, 2)         # → NCHW
        return self.pool(feats).flatten(1)         # (B, 768)


# ===========================================================================
# 6. ViT-B/16 (~86M params)
# ===========================================================================

class ViTB16Encoder(BaseEncoder):
    """
    Backbone: ViT-B/16 (ImageNet-21k → ImageNet-1K pretrained).
    Output dim: 768.
    Minimum input: 224×224 — tự động upsample nếu nhỏ hơn.
    """
    _MIN_SIZE = 224

    def __init__(self, pretrained: bool = True, **kwargs):
        super().__init__(**kwargs)

        backbone = tvm.vit_b_16(
            weights=tvm.ViT_B_16_Weights.DEFAULT if pretrained else None)
        backbone.heads = nn.Identity()            # bỏ classification head
        self.backbone     = backbone
        self.backbone_dim = 768
        self._build_heads()

    def _encode(self, x):
        if x.shape[-1] < self._MIN_SIZE or x.shape[-2] < self._MIN_SIZE:
            x = F.interpolate(x, size=(self._MIN_SIZE, self._MIN_SIZE),
                              mode='bilinear', align_corners=False)
        return self.backbone(x)                   # (B, 768)


# ===========================================================================
# 7. DINOv2 ViT-S/14 (~21M params)
# ===========================================================================

class DINOv2ViTSEncoder(BaseEncoder):
    """
    Backbone: DINOv2 ViT-S/14 (self-supervised, no labels).
    Output dim: 384.
    Minimum input: 224×224 (patch_size=14 → needs multiple-of-14).
    First use: ~85MB download from torch.hub.
    """
    _MIN_SIZE = 224

    def __init__(self, pretrained: bool = True, **kwargs):
        super().__init__(**kwargs)

        if pretrained:
            self.backbone = torch.hub.load(
                'facebookresearch/dinov2', 'dinov2_vits14',
                verbose=False)
        else:
            # Random init fallback (rare)
            self.backbone = torch.hub.load(
                'facebookresearch/dinov2', 'dinov2_vits14',
                pretrained=False, verbose=False)

        self.backbone_dim = 384
        self._build_heads()

    def _encode(self, x):
        if x.shape[-1] < self._MIN_SIZE or x.shape[-2] < self._MIN_SIZE:
            x = F.interpolate(x, size=(self._MIN_SIZE, self._MIN_SIZE),
                              mode='bilinear', align_corners=False)
        return self.backbone(x)                   # (B, 384) — [CLS] token


# ===========================================================================
# Factory
# ===========================================================================

ENCODER_REGISTRY = {
    'gated_vae':        None,                  # default — dùng VAE từ main.py
    'resnet18':         ResNet18Encoder,
    'resnet50':         ResNet50Encoder,
    'efficientnet_v2_s':EfficientNetV2SEncoder,
    'convnext_t':       ConvNeXtTEncoder,
    'swin_t':           SwinTEncoder,
    'vit_b16':          ViTB16Encoder,
    'dinov2_s':         DINOv2ViTSEncoder,
}


def build_encoder(name: str,
                  latent_ch:  int = 2048,
                  dim_inter:  int = 1024,
                  dim_unique: int = 1024,
                  pretrained: bool = True) -> BaseEncoder:
    """
    Factory function:
        enc = build_encoder('resnet18', latent_ch=2048, dim_inter=1024, dim_unique=1024)
        out = enc(images, skip_decoder=True)
    """
    if name not in ENCODER_REGISTRY or name == 'gated_vae':
        raise ValueError(f"Unknown encoder: {name}. Dùng: {list(ENCODER_REGISTRY.keys())}")

    cls = ENCODER_REGISTRY[name]
    return cls(
        pretrained  = pretrained,
        latent_ch   = latent_ch,
        dim_inter   = dim_inter,
        dim_unique  = dim_unique,
    )


# ===========================================================================
# Quick test
# ===========================================================================

if __name__ == '__main__':
    import time
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.rand(8, 3, 128, 128, device=device)

    print(f"Device: {device}  | Input: {x.shape}\n")
    print(f"{'Encoder':<20} {'Params':>10} {'Peak VRAM':>10} {'Time/batch':>12}")
    print("-" * 58)

    for name in ['resnet18', 'resnet50', 'efficientnet_v2_s', 'convnext_t', 'swin_t', 'vit_b16', 'dinov2_s']:
        enc = build_encoder(name, latent_ch=2048, dim_inter=1024, dim_unique=1024).to(device)
        enc.eval()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Warmup
        with torch.no_grad():
            enc(x)

        # Benchmark
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _, mu, _, img_f, v_i, v_u = enc(x)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000

        peak = torch.cuda.max_memory_allocated() / 1e6
        n_params = enc.count_parameters() / 1e6

        print(f"{name:<20} {n_params:>9.1f}M {peak:>8.1f}MB {elapsed:>10.1f}ms")
        print(f"  mu: {mu.shape}  v_inter: {v_i.shape}  v_unique: {v_u.shape}")
