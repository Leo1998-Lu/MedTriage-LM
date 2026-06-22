"""
Prediction heads (Sec. 2.4, Eq. 9).

    y_inst = Softmax( MLP_inst(h_multi) )       3-class triage instruction
    y_crit = Sigmoid( MLP_crit(h_multi) )       binary critical outcome (ICU)

Heads return *logits*; softmax/sigmoid are applied inside the loss / at
inference for numerical stability.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PredictionHeads(nn.Module):
    """Instruction (3-class) + critical-outcome (binary) heads."""

    def __init__(self, d_in: int, n_classes: int = 3, hidden: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.inst = MLPHead(d_in, n_classes, hidden, dropout)
        self.crit = MLPHead(d_in, 1, hidden, dropout)

    def forward(self, h_multi: torch.Tensor):
        inst_logits = self.inst(h_multi)                 # [B, n_classes]
        crit_logit = self.crit(h_multi).squeeze(-1)      # [B]
        return inst_logits, crit_logit
