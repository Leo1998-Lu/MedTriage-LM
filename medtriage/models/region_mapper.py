"""
Multimodal fusion (Eq. 1) and the Anatomical Region Mapper (Eq. 2).

    z   = sigma( W_f [h_s || h_t] + b_f )          (Eq. 1)
    mu  = Sigmoid( W_r z + b_r )                    (Eq. 2)

``z`` is the unified latent clinical-state embedding; ``mu in [0,1]^K`` is the
predicted per-region physiological distress intensity that drives the Gaussian
field synthesis (Eq. 4) and is supervised by the chief-complaint weak labels
through the BCE weak-supervision loss (Eq. 3).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FusionLayer(nn.Module):
    """Concatenate tabular & text embeddings and fuse into z (Eq. 1)."""

    def __init__(self, d_struct: int, d_text: int, d_state: int = 128,
                 dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.proj = nn.Linear(d_struct + d_text, d_state)
        self.norm = nn.LayerNorm(d_state)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self._out_dim = d_state

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, h_struct: torch.Tensor,
                h_text: torch.Tensor) -> torch.Tensor:
        h = torch.cat([h_struct, h_text], dim=-1)        # [h_s || h_t]
        z = self.act(self.proj(h))                       # sigma(W_f[.] + b_f)
        return self.drop(self.norm(z))                   # latent clinical state z


class AnatomicalRegionMapper(nn.Module):
    """MLP mapping z -> region intensity vector mu in [0,1]^K (Eq. 2)."""

    def __init__(self, d_state: int, num_regions: int, hidden: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_state, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_regions),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(z))                # mu in [0,1]^K
