"""
Cross-Modal Vision-Language Alignment (Sec. 2.4, Eq. 7).

The clinical state ``z`` forms the Query; the visual tokens ``V`` form Keys and
Values.  Scaled dot-product attention lets the clinical state dynamically
localise relevant regions in the Visual Phenotype Map:

    Attention(Q, K, V) = Softmax( Q K^T / sqrt(d_k) ) V

The attended output then passes Add&Norm -> FFN -> Add&Norm to yield the
enriched cross-modal representation ``h_multi``.  The attention weights over the
spatial visual tokens are returned for interpretability/visualisation.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class CrossModalAlignment(nn.Module):
    """Single-query cross-attention + Transformer post-block -> h_multi."""

    def __init__(self, d_state: int, d_vis: int, d_model: int = 128,
                 n_heads: int = 4, ffn_mult: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.q_proj = nn.Linear(d_state, d_model)
        self.kv_in = nn.Linear(d_vis, d_model) if d_vis != d_model else nn.Identity()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * ffn_mult)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, d_model),
        )
        self.drop = nn.Dropout(dropout)
        self._out_dim = d_model

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, z: torch.Tensor, V: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """z: [B, d_state], V: [B, Nv, d_vis] ->
        (h_multi [B, d_model], attn_weights [B, Nv])."""
        q = self.q_proj(z).unsqueeze(1)                  # [B, 1, d]
        kv = self.kv_in(V)                               # [B, Nv, d]
        attn_out, attn_w = self.attn(q, kv, kv, need_weights=True,
                                     average_attn_weights=True)  # [B,1,d],[B,1,Nv]
        x = self.norm1(q + self.drop(attn_out))          # Add & Norm
        x = self.norm2(x + self.drop(self.ffn(x)))       # FFN + Add & Norm
        h_multi = x.squeeze(1)                           # [B, d]
        return h_multi, attn_w.squeeze(1)                # [B, Nv]
