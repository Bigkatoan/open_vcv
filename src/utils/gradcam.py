"""
gradcam.py — GradCAM++ for Gated VAE: per-pixel influence on v_inter / v_unique.

Usage:
    from src.utils.gradcam import GradCAMPlusPlus

    cam = GradCAMPlusPlus(model)
    cam_inter, cam_unique = cam.compare(x)   # x: (1, 3, H, W) tensor on same device as model

    # Or individually
    heat = cam(x, target='inter')            # (H, W) in [0, 1], upsampled to input size

Notes:
    - Disables gradient checkpointing on model.backbone automatically.
    - Target layer: model.backbone.stage3[-1]  (last ResidualEncoderBlock)
    - Objective: ||v_inter||² for 'inter', ||v_unique||² for 'unique'
    - Upsamples heatmap back to input (H, W) via bilinear interpolation.
"""

from __future__ import annotations
from typing import Literal
import torch
import torch.nn.functional as F


class GradCAMPlusPlus:
    """
    GradCAM++ hooked onto `model.backbone.stage3[-1]`.

    Parameters
    ----------
    model : VAE
        Instance of src.models.VAE.VAE — must have `backbone.stage3` attribute.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients:   torch.Tensor | None = None
        self._handles: list = []

        # Validate target layer
        try:
            self._target_layer = model.backbone.stage3[-1]
        except AttributeError as e:
            raise AttributeError(
                "model.backbone.stage3 not found. GradCAMPlusPlus only "
                "supports Gated VAE with BackboneEncoder architecture."
            ) from e

    # -----------------------------------------------------------------------
    # Hook management
    # -----------------------------------------------------------------------

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            self._activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self._handles = [
            self._target_layer.register_forward_hook(fwd_hook),
            self._target_layer.register_full_backward_hook(bwd_hook),
        ]

    def _remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    # -----------------------------------------------------------------------
    # Core
    # -----------------------------------------------------------------------

    def _compute_cam(
        self,
        x: torch.Tensor,
        target: Literal['inter', 'unique'],
    ) -> torch.Tensor:
        """
        Returns upsampled GradCAM++ heatmap of shape (H, W) in [0, 1].
        x: (1, 3, H, W)
        """
        H_in, W_in = x.shape[-2], x.shape[-1]

        # Temporarily disable gradient checkpointing so activations are kept
        orig_use_checkpoint = getattr(self.model.backbone, 'use_checkpoint', False)
        self.model.backbone.use_checkpoint = False

        self._register_hooks()
        self.model.zero_grad()

        try:
            self.model.eval()
            with torch.enable_grad():
                x_in = x.requires_grad_(False)
                _, _, _, _, v_inter, v_unique = self.model(x_in, skip_decoder=True)

                if target == 'inter':
                    score = (v_inter ** 2).sum()
                else:
                    score = (v_unique ** 2).sum()

                score.backward()
        finally:
            self._remove_hooks()
            self.model.backbone.use_checkpoint = orig_use_checkpoint

        A  = self._activations   # (1, C, h, w)
        G  = self._gradients     # (1, C, h, w)

        # GradCAM++ alpha weights
        G2 = G ** 2
        G3 = G ** 3
        denom = 2 * G2 + (G3 * A).sum(dim=(2, 3), keepdim=True)
        alpha = G2 / (denom + 1e-8)                        # (1, C, h, w)
        weights = (alpha * F.relu(G)).sum(dim=(2, 3))       # (1, C)

        # Weighted sum over channels
        cam = (weights[:, :, None, None] * A).sum(dim=1)   # (1, h, w)
        cam = F.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(
            cam.unsqueeze(0), size=(H_in, W_in),
            mode='bilinear', align_corners=False,
        ).squeeze()                                         # (H, W)

        # Normalize to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if (cam_max - cam_min) > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.cpu()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def __call__(
        self,
        x: torch.Tensor,
        target: Literal['inter', 'unique'] = 'inter',
    ) -> torch.Tensor:
        """
        Compute GradCAM++ heatmap for a single image.

        Parameters
        ----------
        x      : (1, 3, H, W) tensor on model's device
        target : 'inter' or 'unique'

        Returns
        -------
        heatmap: (H, W) float tensor in [0, 1] on CPU
        """
        assert x.shape[0] == 1, "GradCAMPlusPlus expects batch size 1"
        return self._compute_cam(x, target)

    def compare(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute both inter and unique heatmaps for a single image.

        Parameters
        ----------
        x : (1, 3, H, W) on model's device

        Returns
        -------
        cam_inter  : (H, W) in [0, 1] on CPU
        cam_unique : (H, W) in [0, 1] on CPU
        """
        cam_inter  = self._compute_cam(x, 'inter')
        cam_unique = self._compute_cam(x, 'unique')
        return cam_inter, cam_unique
