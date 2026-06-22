"""
Anatomical Field Generator and Heatmap Rendering (Sec. 2.3, Eqs. 4-6).

    H(u)   = sum_k  mu_k * exp( -||u - c_k||^2 / (2 sigma_k^2) )       (Eq. 4)
    L_smooth = sum_u || grad H(u) ||^2                                 (Eq. 5)
    I_VPM  = alpha * C(H) + (1 - alpha) * I_canonical                  (Eq. 6)

The whole module is *fully differentiable* w.r.t. the region intensities ``mu``,
so gradients from the downstream Vision Encoder propagate back through the
synthesised Visual Phenotype Map (VPM) into the Anatomical Region Mapper and the
clinical encoders.

Because a matplotlib colour map is non-differentiable, the in-graph colour map
``C(.)`` is a smooth "pseudo-jet" mapping (R/G/B as clipped triangular functions
of the normalised field).  A faithful matplotlib rendering for *human* inspection
lives in ``medtriage/viz.py`` and is not part of the forward pass.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..data.anatomy import build_silhouette, region_pixel_coords


def _pseudo_jet(h: torch.Tensor) -> torch.Tensor:
    """Differentiable jet-like colour map. ``h``: [...,H,W] in [0,1] ->
    RGB ``[...,3,H,W]``."""
    r = (1.5 - (4.0 * h - 3.0).abs()).clamp(0.0, 1.0)
    g = (1.5 - (4.0 * h - 2.0).abs()).clamp(0.0, 1.0)
    b = (1.5 - (4.0 * h - 1.0).abs()).clamp(0.0, 1.0)
    return torch.stack([r, g, b], dim=-3)


class GaussianFieldGenerator(nn.Module):
    """Synthesise the VPM image (and raw heat field) from region intensities mu."""

    def __init__(self, num_regions: int, height: int = 64, width: int = 48,
                 alpha: float = 0.6, out_channels: int = 3,
                 body_masked: bool = True, sigma_scale: float = 1.0):
        super().__init__()
        assert out_channels in (1, 3)
        self.num_regions = num_regions
        self.height = height
        self.width = width
        self.alpha = alpha
        self.out_channels = out_channels
        self.body_masked = body_masked

        # ---- precompute the K fixed Gaussian basis maps (Eq. 4 kernels) ----
        rows, cols, sig = region_pixel_coords(height, width)
        sig = sig * sigma_scale
        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        basis = np.zeros((num_regions, height, width), dtype=np.float32)
        for k in range(num_regions):
            d2 = (yy - rows[k]) ** 2 + (xx - cols[k]) ** 2
            basis[k] = np.exp(-d2 / (2.0 * sig[k] ** 2))
        self.register_buffer("basis", torch.from_numpy(basis))          # [K,H,W]

        sil = build_silhouette(height, width).astype(np.float32)         # [H,W]
        self.register_buffer("silhouette", torch.from_numpy(sil))

    # ---- core synthesis ---------------------------------------------------
    def synthesize_field(self, mu: torch.Tensor) -> torch.Tensor:
        """mu: [B,K] -> raw heat field H: [B,H,W] (Eq. 4)."""
        H = torch.einsum("bk,khw->bhw", mu, self.basis)
        if self.body_masked:
            H = H * self.silhouette.unsqueeze(0)
        return H

    @staticmethod
    def normalize_field(H: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Per-sample min-max normalisation to [0,1] (differentiable)."""
        B = H.shape[0]
        flat = H.view(B, -1)
        lo = flat.min(dim=1, keepdim=True).values
        hi = flat.max(dim=1, keepdim=True).values
        norm = (flat - lo) / (hi - lo + eps)
        return norm.view_as(H)

    def render(self, H_norm: torch.Tensor) -> torch.Tensor:
        """Render I_VPM from a normalised field (Eq. 6). Returns [B,C,H,W]."""
        sil = self.silhouette.unsqueeze(0)                       # [1,H,W]
        if self.out_channels == 1:
            # single-channel: heat blended onto the body silhouette
            vpm = self.alpha * H_norm + (1.0 - self.alpha) * sil
            return vpm.unsqueeze(1)                              # [B,1,H,W]
        color = _pseudo_jet(H_norm)                              # [B,3,H,W]
        sil_rgb = sil.unsqueeze(1).expand(-1, 3, -1, -1)        # [B,3,H,W] gray body
        vpm = self.alpha * color + (1.0 - self.alpha) * sil_rgb
        return vpm

    def forward(self, mu: torch.Tensor):
        """mu: [B,K] -> (vpm [B,C,H,W], H_field [B,H,W], H_norm [B,H,W])."""
        H = self.synthesize_field(mu)
        H_norm = self.normalize_field(H)
        vpm = self.render(H_norm)
        return vpm, H, H_norm

    # ---- smoothness regulariser (Eq. 5) ----------------------------------
    @staticmethod
    def smoothness_loss(H: torch.Tensor) -> torch.Tensor:
        """L_smooth = mean over batch of sum_u ||grad H(u)||^2 (finite diff)."""
        dy = H[:, 1:, :] - H[:, :-1, :]
        dx = H[:, :, 1:] - H[:, :, :-1]
        return (dy.pow(2).sum(dim=(1, 2)) + dx.pow(2).sum(dim=(1, 2))).mean()
