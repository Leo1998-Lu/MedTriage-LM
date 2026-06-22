"""End-to-end MedTriage-LM model.

This module wires the four functional blocks described in the paper into a
single ``nn.Module``:

    1. Clinical State Encoder
         - FT-Transformer over the tabular triage variables  -> h_s   (Eq. tabular)
         - Text encoder over the chief complaint              -> h_t
         - Gated fusion                                       -> z      (Eq. 1)
    2. Anatomical Region Mapper
         - MLP z -> region intensities mu in [0,1]^K          (Eq. 2)
    3. Anatomical Field Generator + Heatmap Rendering
         - Gaussian field H(u)                                (Eq. 4)
         - differentiable render -> Visual Phenotype Map      (Eq. 6)
    4. Multimodal Integration
         - vision encoder over the VPM -> visual tokens V
         - cross-modal attention (z as query)                (Eq. 7)
         - prediction heads (instruction + critical)         (Eq. 9)
         - rationale decoder conditioned on h_multi           (Eq. 8)

The forward pass returns every intermediate tensor the loss function
(:mod:`medtriage.losses`) needs, so the whole objective ``L_total`` (Eq. 10)
is computed from a single call.

Heavy pretrained backbones (ClinicalBERT, timm ViT, Qwen rationale model) are
selected purely through the config; the lightweight from-scratch defaults make
the full pipeline trainable offline on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .ft_transformer import FTTransformer
from .text_encoder import build_text_encoder
from .region_mapper import FusionLayer, AnatomicalRegionMapper
from .field_generator import GaussianFieldGenerator
from .vision_encoder import build_vision_encoder
from .cross_attention import CrossModalAlignment
from .heads import PredictionHeads
from .rationale import LiteRationaleDecoder


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class MedTriageLMConfig:
    """All architectural hyper-parameters and backbone switches.

    The defaults are the small/offline configuration used for the sandbox
    reproduction. Switch ``text_backbone='clinicalbert'`` /
    ``vision_backbone='timm'`` (and supply weights + internet) to recover the
    paper's full-scale setting.
    """

    # --- task dimensions (filled from the data unless overridden) ----------
    num_regions: int = 12
    n_instruction_classes: int = 3

    # --- clinical state encoder -------------------------------------------
    ft_d_token: int = 64
    ft_n_blocks: int = 2
    ft_n_heads: int = 4
    ft_ffn_mult: float = 2.0

    text_backbone: str = "lite"            # {"lite","clinicalbert"}
    text_d_model: int = 64
    text_n_blocks: int = 2
    text_n_heads: int = 4
    text_max_len: int = 32
    text_hf_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    text_freeze: bool = True

    d_state: int = 128                     # latent clinical state z

    # --- region mapper -----------------------------------------------------
    mapper_hidden: int = 64

    # --- field generator / VPM --------------------------------------------
    vpm_height: int = 64
    vpm_width: int = 48
    vpm_alpha: float = 0.6
    vpm_channels: int = 3                  # 3 = RGB pseudo-jet, 1 = grayscale
    field_base_sigma_scale: float = 1.0
    body_masked: bool = True

    # --- vision encoder ----------------------------------------------------
    vision_backbone: str = "lite"          # {"lite","timm"}
    vit_d_model: int = 96
    vit_patch: int = 8
    vit_n_blocks: int = 2
    vit_n_heads: int = 4
    vit_use_cls: bool = True
    vit_timm_name: str = "vit_tiny_patch16_224"

    # --- cross-modal integration ------------------------------------------
    fusion_d_model: int = 128
    fusion_n_heads: int = 4
    fusion_ffn_mult: float = 2.0

    # --- prediction heads --------------------------------------------------
    head_hidden: int = 64

    # --- rationale decoder -------------------------------------------------
    use_rationale: bool = True
    rationale_backend: str = "lite"        # {"lite","qwen","none"}
    rationale_d_model: int = 96
    rationale_n_blocks: int = 2
    rationale_n_heads: int = 4
    rationale_max_len: int = 48

    # --- regularisation ----------------------------------------------------
    dropout: float = 0.1


@dataclass
class MedTriageOutput:
    """Container for every tensor produced by a forward pass."""

    inst_logits: torch.Tensor                       # [B, n_classes]
    crit_logit: torch.Tensor                        # [B]
    mu: torch.Tensor                                # [B, K]  region intensities
    H: torch.Tensor                                 # [B, h, w] raw field
    H_norm: torch.Tensor                            # [B, h, w] normalised field
    vpm: torch.Tensor                               # [B, C, h, w]
    attn: torch.Tensor                              # [B, Nv] cross-attn weights
    h_multi: torch.Tensor                           # [B, d] fused representation
    z: torch.Tensor                                 # [B, d_state] clinical state
    rationale_logits: Optional[torch.Tensor] = None  # [B, T, V]


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class MedTriageLM(nn.Module):
    """Anatomically grounded visual-phenotype triage model."""

    def __init__(
        self,
        cfg: MedTriageLMConfig,
        n_numeric: int,
        cat_cardinalities: List[int],
        text_vocab_size: int,
        rationale_vocab_size: Optional[int] = None,
    ):
        super().__init__()
        self.cfg = cfg

        # 1a. structured (tabular) encoder -- FT-Transformer ----------------
        self.struct_encoder = FTTransformer(
            n_numeric=n_numeric,
            cat_cardinalities=cat_cardinalities,
            d_token=cfg.ft_d_token,
            n_blocks=cfg.ft_n_blocks,
            n_heads=cfg.ft_n_heads,
            ffn_mult=cfg.ft_ffn_mult,
            dropout=cfg.dropout,
        )

        # 1b. text encoder --------------------------------------------------
        self.text_encoder = build_text_encoder(
            backend=cfg.text_backbone,
            vocab_size=text_vocab_size,
            d_model=cfg.text_d_model,
            max_len=cfg.text_max_len,
            n_blocks=cfg.text_n_blocks,
            n_heads=cfg.text_n_heads,
            hf_name=cfg.text_hf_name,
            freeze=cfg.text_freeze,
        ) if cfg.text_backbone == "lite" else build_text_encoder(
            backend=cfg.text_backbone,
            vocab_size=text_vocab_size,
            d_model=cfg.text_d_model,
            max_len=cfg.text_max_len,
            hf_name=cfg.text_hf_name,
            freeze=cfg.text_freeze,
        )

        # 1c. gated fusion --> latent clinical state z (Eq. 1) --------------
        self.fusion = FusionLayer(
            d_struct=self.struct_encoder.out_dim,
            d_text=self.text_encoder.out_dim,
            d_state=cfg.d_state,
            dropout=cfg.dropout,
        )

        # 2. anatomical region mapper (Eq. 2) -------------------------------
        self.region_mapper = AnatomicalRegionMapper(
            d_state=cfg.d_state,
            num_regions=cfg.num_regions,
            hidden=cfg.mapper_hidden,
            dropout=cfg.dropout,
        )

        # 3. anatomical field generator + render (Eq. 4 / Eq. 6) -----------
        self.field_generator = GaussianFieldGenerator(
            num_regions=cfg.num_regions,
            height=cfg.vpm_height,
            width=cfg.vpm_width,
            alpha=cfg.vpm_alpha,
            out_channels=cfg.vpm_channels,
            sigma_scale=cfg.field_base_sigma_scale,
            body_masked=cfg.body_masked,
        )

        # 4a. vision encoder over the VPM ----------------------------------
        if cfg.vision_backbone == "lite":
            self.vision_encoder = build_vision_encoder(
                backend="lite",
                in_ch=cfg.vpm_channels,
                d_model=cfg.vit_d_model,
                patch=cfg.vit_patch,
                n_blocks=cfg.vit_n_blocks,
                n_heads=cfg.vit_n_heads,
                img_h=cfg.vpm_height,
                img_w=cfg.vpm_width,
                prepend_cls=cfg.vit_use_cls,
                dropout=cfg.dropout,
            )
        else:
            self.vision_encoder = build_vision_encoder(
                backend="timm",
                model_name=cfg.vit_timm_name,
                out_dim=cfg.vit_d_model,
                in_ch=cfg.vpm_channels,
            )

        # 4b. cross-modal attention (Eq. 7) --------------------------------
        self.cross_attention = CrossModalAlignment(
            d_state=cfg.d_state,
            d_vis=self.vision_encoder.out_dim,
            d_model=cfg.fusion_d_model,
            n_heads=cfg.fusion_n_heads,
            ffn_mult=cfg.fusion_ffn_mult,
            dropout=cfg.dropout,
        )

        # 4c. prediction heads (Eq. 9) -------------------------------------
        self.heads = PredictionHeads(
            d_in=self.cross_attention.out_dim,
            n_classes=cfg.n_instruction_classes,
            hidden=cfg.head_hidden,
            dropout=cfg.dropout,
        )

        # 4d. rationale decoder (Eq. 8) ------------------------------------
        self.rationale_decoder: Optional[LiteRationaleDecoder] = None
        if cfg.use_rationale and cfg.rationale_backend == "lite":
            assert rationale_vocab_size is not None, \
                "rationale_vocab_size required for the lite rationale decoder"
            self.rationale_decoder = LiteRationaleDecoder(
                vocab_size=rationale_vocab_size,
                d_cond=self.cross_attention.out_dim,
                d_model=cfg.rationale_d_model,
                n_blocks=cfg.rationale_n_blocks,
                n_heads=cfg.rationale_n_heads,
                max_len=cfg.rationale_max_len,
                dropout=cfg.dropout,
            )

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def encode_state(
        self,
        num: torch.Tensor,
        cat: torch.Tensor,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run encoders + fusion to obtain the latent clinical state z."""
        h_s = self.struct_encoder(num, cat)                 # [B, d_struct]
        h_t = self.text_encoder(input_ids, attn_mask)       # [B, d_text]
        z = self.fusion(h_s, h_t)                           # [B, d_state]
        return z

    def forward(
        self,
        num: torch.Tensor,
        cat: torch.Tensor,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        rationale_tgt: Optional[torch.Tensor] = None,
    ) -> MedTriageOutput:
        # --- clinical state ------------------------------------------------
        z = self.encode_state(num, cat, input_ids, attn_mask)

        # --- region intensities + VPM -------------------------------------
        mu = self.region_mapper(z)                          # [B, K]
        vpm, H, H_norm = self.field_generator(mu)           # field + render

        # --- visual tokens + cross-modal fusion ---------------------------
        V = self.vision_encoder(vpm)                        # [B, Nv, d_vis]
        h_multi, attn = self.cross_attention(z, V)          # [B, d], [B, Nv]

        # --- predictions ---------------------------------------------------
        inst_logits, crit_logit = self.heads(h_multi)

        # --- rationale (teacher forced when targets provided) -------------
        rationale_logits = None
        if self.rationale_decoder is not None and rationale_tgt is not None:
            rationale_logits = self.rationale_decoder(h_multi, rationale_tgt)

        return MedTriageOutput(
            inst_logits=inst_logits,
            crit_logit=crit_logit,
            mu=mu,
            H=H,
            H_norm=H_norm,
            vpm=vpm,
            attn=attn,
            h_multi=h_multi,
            z=z,
            rationale_logits=rationale_logits,
        )

    @torch.no_grad()
    def generate_rationale(
        self,
        num: torch.Tensor,
        cat: torch.Tensor,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        bos: int = 1,
        eos: int = 2,
    ) -> List[List[int]]:
        """Greedy-decode rationale token ids for a batch (lite decoder)."""
        if self.rationale_decoder is None:
            raise RuntimeError("Model has no lite rationale decoder.")
        out = self.forward(num, cat, input_ids, attn_mask)
        return self.rationale_decoder.generate(out.h_multi, bos=bos, eos=eos)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_medtriage_lm(
    cfg: MedTriageLMConfig,
    n_numeric: int,
    cat_cardinalities: List[int],
    text_vocab_size: int,
    rationale_vocab_size: Optional[int] = None,
) -> MedTriageLM:
    """Convenience constructor mirroring the dataclass config."""
    return MedTriageLM(
        cfg=cfg,
        n_numeric=n_numeric,
        cat_cardinalities=cat_cardinalities,
        text_vocab_size=text_vocab_size,
        rationale_vocab_size=rationale_vocab_size,
    )


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
