"""Loss functions for MedTriage-LM.

Implements every term of the composite objective (Eq. 10):

    L_total = L_CE                       (primary triage instruction, Eq. 9)
            + lambda1 * L_BCE            (critical-outcome auxiliary task)
            + lambda2 * L_weak           (weak anatomical supervision, Eq. 3)
            + lambda3 * L_smooth         (field smoothness regulariser, Eq. 5)
            + lambda4 * L_gen            (rationale generation, Eq. 8)

Default weights follow the paper: lambda1=0.5, lambda2=1.0, lambda3=0.5,
lambda4=1.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models.field_generator import GaussianFieldGenerator
from .models.medtriage_lm import MedTriageOutput


# --------------------------------------------------------------------------- #
# Loss weights
# --------------------------------------------------------------------------- #
@dataclass
class LossWeights:
    """Weights for the composite objective (paper Section: Training)."""

    lambda1_critical: float = 0.5     # L_BCE   (critical-outcome head)
    lambda2_weak: float = 1.0         # L_weak  (weak anatomical supervision)
    lambda3_smooth: float = 0.5       # L_smooth(field smoothness)
    lambda4_gen: float = 1.0          # L_gen   (rationale generation)
    rationale_pad_id: int = 0         # ignore index for L_gen


# --------------------------------------------------------------------------- #
# Individual terms
# --------------------------------------------------------------------------- #
def instruction_loss(
    inst_logits: torch.Tensor,
    target: torch.Tensor,
    class_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """L_CE: cross-entropy for the 3-class triage instruction (Eq. 9)."""
    return F.cross_entropy(inst_logits, target, weight=class_weight)


def critical_loss(
    crit_logit: torch.Tensor,
    target: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """L_BCE: binary cross-entropy for the critical-outcome auxiliary task."""
    return F.binary_cross_entropy_with_logits(
        crit_logit, target.float(), pos_weight=pos_weight
    )


def weak_supervision_loss(mu: torch.Tensor, y_weak: torch.Tensor) -> torch.Tensor:
    """L_weak (Eq. 3): sum_k BCE(mu_k, y_weak_k), averaged over the batch.

    ``mu`` are intensities already in [0,1] (Sigmoid output of the region
    mapper), so we use the probability form of BCE and sum across the K
    regions before averaging over the batch.
    """
    bce = F.binary_cross_entropy(mu, y_weak, reduction="none")   # [B, K]
    return bce.sum(dim=1).mean()


def smoothness_loss(H: torch.Tensor) -> torch.Tensor:
    """L_smooth (Eq. 5): spatial-gradient energy of the heat field."""
    return GaussianFieldGenerator.smoothness_loss(H)


def rationale_loss(
    rationale_logits: torch.Tensor,
    rationale_tgt: torch.Tensor,
    pad_id: int = 0,
) -> torch.Tensor:
    """L_gen (Eq. 8): teacher-forced token cross-entropy.

    ``rationale_logits``: [B, T, V] aligned so position t predicts
    ``rationale_tgt[:, t]``. PAD positions are ignored.
    """
    B, T, V = rationale_logits.shape
    return F.cross_entropy(
        rationale_logits.reshape(B * T, V),
        rationale_tgt.reshape(B * T),
        ignore_index=pad_id,
    )


# --------------------------------------------------------------------------- #
# Composite objective
# --------------------------------------------------------------------------- #
class MedTriageLoss(nn.Module):
    """Computes L_total and returns a breakdown of every component."""

    def __init__(
        self,
        weights: Optional[LossWeights] = None,
        class_weight: Optional[torch.Tensor] = None,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.w = weights or LossWeights()
        # registered as buffers so they move with .to(device)
        self.register_buffer(
            "class_weight",
            class_weight if class_weight is not None else None,
            persistent=False,
        )
        self.register_buffer(
            "pos_weight",
            pos_weight if pos_weight is not None else None,
            persistent=False,
        )

    def forward(
        self,
        out: MedTriageOutput,
        batch: Dict[str, torch.Tensor],
        rationale_tgt: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        cw = self.class_weight if self.class_weight is not None else None
        pw = self.pos_weight if self.pos_weight is not None else None

        l_ce = instruction_loss(out.inst_logits, batch["instruction"], cw)
        l_bce = critical_loss(out.crit_logit, batch["critical"], pw)
        l_weak = weak_supervision_loss(out.mu, batch["weak"])
        l_smooth = smoothness_loss(out.H)

        total = (
            l_ce
            + self.w.lambda1_critical * l_bce
            + self.w.lambda2_weak * l_weak
            + self.w.lambda3_smooth * l_smooth
        )

        terms = {
            "loss": total,
            "L_CE": l_ce.detach(),
            "L_BCE": l_bce.detach(),
            "L_weak": l_weak.detach(),
            "L_smooth": l_smooth.detach(),
        }

        if out.rationale_logits is not None and rationale_tgt is not None:
            l_gen = rationale_loss(
                out.rationale_logits, rationale_tgt, self.w.rationale_pad_id
            )
            total = total + self.w.lambda4_gen * l_gen
            terms["loss"] = total
            terms["L_gen"] = l_gen.detach()

        return terms
