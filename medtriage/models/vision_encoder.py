"""
Vision encoder for the Visual Phenotype Map (Sec. 2.4).

Processes the rendered VPM ``I_VPM`` into a sequence of spatial visual tokens
``V in R^{Nv x dv}`` that serve as Keys/Values in the cross-modal alignment.

Default back-end is a compact from-scratch ViT (offline). A ``timm`` pretrained
ViT can be plugged in via ``build_vision_encoder(backend="timm", ...)`` when the
environment provides ``timm`` + internet, matching the paper's pretrained-ViT
initialisation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .ft_transformer import TransformerBlock


class PatchEmbed(nn.Module):
    """Conv patchifier: [B,C,H,W] -> [B, N, d]."""

    def __init__(self, in_ch: int, d_model: int, patch: int = 8):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                         # [B, d, H/p, W/p]
        B, d, h, w = x.shape
        return x.flatten(2).transpose(1, 2)      # [B, N, d]


class LiteViT(nn.Module):
    """Small Vision Transformer returning the full token sequence V."""

    def __init__(self, in_ch: int = 3, d_model: int = 96, patch: int = 8,
                 n_blocks: int = 2, n_heads: int = 4, img_h: int = 64,
                 img_w: int = 48, dropout: float = 0.1, prepend_cls: bool = True):
        super().__init__()
        self.d_model = d_model
        self.prepend_cls = prepend_cls
        self.patch_embed = PatchEmbed(in_ch, d_model, patch)
        n_patches = (img_h // patch) * (img_w // patch)
        n_tokens = n_patches + (1 if prepend_cls else 0)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model)) if prepend_cls else None
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.normal_(self.pos, std=0.02)
        if self.cls is not None:
            nn.init.normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout=dropout)
             for _ in range(n_blocks)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    @property
    def out_dim(self) -> int:
        return self.d_model

    def forward(self, vpm: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(vpm)                # [B, N, d]
        if self.cls is not None:
            cls = self.cls.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
        x = self.drop(x + self.pos[:, : x.shape[1]])
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)                      # V: [B, Nv, d]


class TimmViT(nn.Module):  # pragma: no cover - requires timm + internet
    """Pretrained ViT wrapper (token sequence) via timm."""

    def __init__(self, model_name: str = "vit_tiny_patch16_224",
                 out_dim: int = 96, img_size: int = 64, in_ch: int = 3):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            model_name, pretrained=True, num_classes=0,
            img_size=img_size, in_chans=in_ch,
        )
        feat = self.backbone.num_features
        self.proj = nn.Linear(feat, out_dim)
        self._out_dim = out_dim

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, vpm: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.forward_features(vpm)   # [B, N, feat]
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(1)
        return self.proj(tokens)


def build_vision_encoder(backend: str = "lite", **kw) -> nn.Module:
    if backend == "lite":
        return LiteViT(**kw)
    if backend == "timm":
        allowed = {"model_name", "out_dim", "img_size", "in_ch"}
        return TimmViT(**{k: v for k, v in kw.items() if k in allowed})
    raise ValueError(f"Unknown vision backend: {backend!r}")
