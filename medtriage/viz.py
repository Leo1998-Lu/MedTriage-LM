"""Visualisation utilities for MedTriage-LM.

Produces the human-facing figures the paper shows in Fig. 2:

  * the synthesised Visual Phenotype Map (VPM) rendered with a real perceptual
    colormap over the canonical body silhouette,
  * the per-region intensity vector mu,
  * the cross-modal attention over VPM patches,
  * the generated textual rationale.

Unlike the in-graph renderer used for training (a differentiable pseudo-jet),
this module uses matplotlib's colormaps purely for inspection, so there is no
constraint that the rendering be differentiable here.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")                       # headless / offline rendering
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from scipy.ndimage import gaussian_filter
    _HAS_SCIPY = True
except Exception:                            # pragma: no cover - scipy optional
    _HAS_SCIPY = False

from .data.anatomy import (
    REGION_NAMES, NUM_REGIONS, build_silhouette, region_pixel_coords,
)
from .data.preprocess import INSTRUCTION_CLASSES
from .trainer import FoldArtifacts


# --------------------------------------------------------------------------- #
# Thermographic VPM rendering (reference-quality, decoupled from the model)
# --------------------------------------------------------------------------- #
_JET = cm.get_cmap("jet")
_BG_NAVY = np.array([0.015, 0.02, 0.32], dtype=np.float32)   # dark background


def _blur(field: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with a small separable fallback if SciPy is absent."""
    if sigma <= 0:
        return field
    if _HAS_SCIPY:
        return gaussian_filter(field, sigma=sigma)
    # cheap separable box-ish fallback
    k = int(max(1, round(sigma)))
    ker = np.ones(2 * k + 1, dtype=np.float32)
    ker /= ker.sum()
    out = np.apply_along_axis(lambda m: np.convolve(m, ker, mode="same"), 0, field)
    out = np.apply_along_axis(lambda m: np.convolve(m, ker, mode="same"), 1, out)
    return out


def synth_field_hires(mu: np.ndarray, height: int, width: int,
                      sigma_scale: float = 1.0) -> np.ndarray:
    """Raw Gaussian heat field (Eq. 4) at an arbitrary resolution, in numpy."""
    rows, cols, sig = region_pixel_coords(height, width)
    sig = sig * sigma_scale
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    field = np.zeros((height, width), dtype=np.float32)
    for k in range(min(len(mu), NUM_REGIONS)):
        d2 = (yy - rows[k]) ** 2 + (xx - cols[k]) ** 2
        field += float(mu[k]) * np.exp(-d2 / (2.0 * sig[k] ** 2))
    return field


def render_thermal(mu: np.ndarray, height: int = 440, width: int = 330,
                   base_warm: float = 0.16, blur: float = 2.6,
                   supersample: int = 2, vignette: float = 0.22):
    """Render a reference-style thermographic Visual Phenotype Map.

    A smooth anatomical body filled with a ``jet`` colormap over a dark navy
    background: cold tissue reads blue, distress hotspots glow red. Returns an
    ``(H, W, 3)`` float RGB image and the per-body normalised heat field.
    """
    sil = build_silhouette(height, width, supersample=supersample)
    field = synth_field_hires(mu, height, width) * sil
    field = _blur(field, blur) * sil                      # smooth, then re-mask

    inb = sil > 0.05
    if inb.any():
        v = field[inb]
        lo, hi = float(v.min()), float(v.max())
        fn = (field - lo) / (hi - lo + 1e-6)
    else:
        fn = field
    heat = np.clip(base_warm + (1.0 - base_warm) * fn, 0.0, 1.0)

    rgb_body = _JET(heat)[..., :3].astype(np.float32)
    alpha = np.clip(sil, 0.0, 1.0)[..., None]

    bg = np.tile(_BG_NAVY, (height, width, 1))
    if vignette > 0:
        yy, xx = np.mgrid[0:height, 0:width]
        r = np.sqrt(((yy - height / 2) / (height / 2)) ** 2
                    + ((xx - width / 2) / (width / 2)) ** 2)
        bg = bg * np.clip(1.0 - vignette * r, 1.0 - vignette, 1.0)[..., None]

    img = alpha * rgb_body + (1.0 - alpha) * bg
    return np.clip(img, 0.0, 1.0), (fn * (sil > 0.05))


def render_template_showcase(out_path: str, mu: Optional[np.ndarray] = None,
                             height: int = 520, width: int = 380) -> str:
    """Save a standalone, reference-style figure: bare body + thermal VPM.

    ``mu`` defaults to a canonical thoracic/abdominal distress pattern so the
    showcase resembles a real bedside thermography image.
    """
    if mu is None:
        mu = np.zeros(NUM_REGIONS, dtype=np.float32)
        preset = {"chest": 0.98, "abdomen": 0.92, "pelvis": 0.45, "neck": 0.42,
                  "head": 0.40, "face": 0.30, "left_arm": 0.50, "right_arm": 0.50,
                  "left_leg": 0.58, "right_leg": 0.58, "systemic": 0.45,
                  "back": 0.30}
        for i, name in enumerate(REGION_NAMES):
            mu[i] = preset.get(name, 0.2)

    sil = build_silhouette(height, width, supersample=2)
    img, _ = render_thermal(mu, height, width)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 6))
    fig.patch.set_facecolor("white")
    axes[0].imshow(sil, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Canonical body template", fontsize=11)
    axes[0].axis("off")
    axes[1].imshow(img)
    axes[1].set_title("Synthesized Visual Phenotype Map", fontsize=11)
    axes[1].axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Low-level field rendering (model-coupled, low-res; used for the patch grid)
# --------------------------------------------------------------------------- #
def field_from_mu(art: FoldArtifacts, mu: np.ndarray) -> np.ndarray:
    """Render a normalised heat field [H,W] from a region-intensity vector."""
    gen = art.model.field_generator
    with torch.no_grad():
        m = torch.tensor(mu, dtype=torch.float32).unsqueeze(0)
        H = gen.synthesize_field(m)
        H_norm = gen.normalize_field(H)
    return H_norm.squeeze(0).cpu().numpy()


def _silhouette(art: FoldArtifacts) -> np.ndarray:
    return art.model.field_generator.silhouette.cpu().numpy()


# --------------------------------------------------------------------------- #
# Per-sample figure
# --------------------------------------------------------------------------- #
def visualize_sample(
    art: FoldArtifacts,
    batch_single: dict,
    out_path: str,
    title: Optional[str] = None,
) -> str:
    """Render a 3-panel VPM figure for a single sample dict (batched size 1).

    ``batch_single`` must contain tensors with a leading batch dim of 1 for
    keys: num, cat, input_ids, attn_mask. Returns the saved file path.
    """
    art.model.eval()
    device = next(art.model.parameters()).device
    num = batch_single["num"].to(device)
    cat = batch_single["cat"].to(device)
    ids = batch_single["input_ids"].to(device)
    mask = batch_single["attn_mask"].to(device)

    with torch.no_grad():
        out = art.model(num, cat, ids, mask)
        mu = out.mu.squeeze(0).cpu().numpy()
        attn = out.attn.squeeze(0).cpu().numpy()
        inst_pred = int(out.inst_logits.argmax(-1).item())
        crit_prob = float(torch.sigmoid(out.crit_logit).item())
        rationale = ""
        if art.model.rationale_decoder is not None and art.rvocab is not None:
            gen_ids = art.model.rationale_decoder.generate(out.h_multi)[0]
            rationale = art.rvocab.decode(gen_ids)

    field = field_from_mu(art, mu)
    sil = _silhouette(art)

    # high-resolution, reference-style thermal render (decoupled from model res)
    thermal_img, _ = render_thermal(mu)
    sil_hi = build_silhouette(thermal_img.shape[0], thermal_img.shape[1],
                              supersample=2)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5))

    # panel 1: canonical body template (high-res silhouette)
    axes[0].imshow(sil_hi, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Canonical body template")
    axes[0].axis("off")

    # panel 2: VPM = thermographic heat over the anatomical body
    axes[1].imshow(thermal_img)
    axes[1].set_title("Visual Phenotype Map (VPM)")
    axes[1].axis("off")
    sm = cm.ScalarMappable(cmap="jet",
                           norm=matplotlib.colors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)

    # panel 3: region intensities
    order = np.argsort(-mu)
    axes[2].barh([REGION_NAMES[i] for i in order][::-1],
                 [mu[i] for i in order][::-1], color="#c0392b")
    axes[2].set_xlim(0, 1)
    axes[2].set_title("Region intensities  mu")
    axes[2].set_xlabel("activation")

    head = title or "MedTriage-LM sample"
    pred_line = (f"pred: {INSTRUCTION_CLASSES[inst_pred]}  |  "
                 f"P(critical)={crit_prob:.2f}")
    sup = head + "\n" + pred_line
    if rationale:
        sup += "\nrationale: " + _wrap(rationale, 110)
    fig.suptitle(sup, fontsize=10, y=1.02)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _wrap(text: str, width: int) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Attention overlay
# --------------------------------------------------------------------------- #
def visualize_attention(
    art: FoldArtifacts,
    batch_single: dict,
    out_path: str,
) -> str:
    """Overlay cross-modal attention weights on the VPM patch grid."""
    art.model.eval()
    device = next(art.model.parameters()).device
    with torch.no_grad():
        out = art.model(batch_single["num"].to(device),
                        batch_single["cat"].to(device),
                        batch_single["input_ids"].to(device),
                        batch_single["attn_mask"].to(device))
        attn = out.attn.squeeze(0).cpu().numpy()
        mu = out.mu.squeeze(0).cpu().numpy()

    cfg = art.model.cfg
    n_h = cfg.vpm_height // cfg.vit_patch
    n_w = cfg.vpm_width // cfg.vit_patch
    n_patch = n_h * n_w
    # drop the CLS token's weight if present
    patch_attn = attn[-n_patch:] if attn.shape[0] >= n_patch else attn
    grid = patch_attn[:n_patch].reshape(n_h, n_w)

    field = field_from_mu(art, mu)
    sil = _silhouette(art)
    thermal_img, _ = render_thermal(mu)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.5))
    axes[0].imshow(thermal_img)
    axes[0].set_title("VPM")
    axes[0].axis("off")

    im = axes[1].imshow(grid, cmap="viridis")
    axes[1].set_title("Cross-modal attention over patches")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Batch helper
# --------------------------------------------------------------------------- #
def visualize_dataset_samples(
    art: FoldArtifacts,
    dataset,
    indices: Sequence[int],
    out_dir: str,
) -> List[str]:
    """Render VPM + attention figures for selected dataset rows."""
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []
    for idx in indices:
        item = dataset[idx]
        single = {k: (v.unsqueeze(0) if torch.is_tensor(v) else v)
                  for k, v in item.items()}
        stay = int(item["stay_id"])
        true_inst = INSTRUCTION_CLASSES[int(item["instruction"])]
        title = f"stay {stay}  |  true: {true_inst}  |  cc: {item['text'][:60]}"
        p1 = visualize_sample(art, single,
                              os.path.join(out_dir, f"vpm_stay{stay}.png"), title)
        p2 = visualize_attention(art, single,
                                 os.path.join(out_dir, f"attn_stay{stay}.png"))
        paths.extend([p1, p2])
    return paths
