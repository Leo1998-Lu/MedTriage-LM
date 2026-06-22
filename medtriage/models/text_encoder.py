"""
Clinical text encoder (Sec. 2.1).

Produces the holistic textual embedding ``h_t = f_BERT(x_t)[CLS]`` from the
chief complaint.  Two interchangeable back-ends:

  * ``"lite"``         a small from-scratch Transformer encoder over the lite
                       vocabulary (default; fully offline);
  * ``"clinicalbert"`` HuggingFace ``Bio_ClinicalBERT`` -- the encoder used in
                       the paper [Alsentzer 2019] (needs ``transformers`` +
                       internet; can be frozen or fine-tuned).

Both expose ``out_dim`` and a uniform ``forward(input_ids, attn_mask) -> [B, d]``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .ft_transformer import TransformerBlock


class LiteTextEncoder(nn.Module):
    """Compact Transformer text encoder returning the [CLS]-position embedding."""

    def __init__(self, vocab_size: int, d_model: int = 64, n_blocks: int = 2,
                 n_heads: int = 4, max_len: int = 32, dropout: float = 0.1,
                 pad_idx: int = 0):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.tok_emb.weight, std=d_model ** -0.5)
        nn.init.normal_(self.pos_emb, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout=dropout)
             for _ in range(n_blocks)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    @property
    def out_dim(self) -> int:
        return self.d_model

    def forward(self, input_ids: torch.Tensor,
                attn_mask: torch.Tensor) -> torch.Tensor:
        T = input_ids.shape[1]
        x = self.tok_emb(input_ids) + self.pos_emb[:, :T]
        x = self.drop(x)
        # MultiheadAttention in TransformerBlock does not consume a mask here;
        # padding tokens are zero-embedded (padding_idx) and contribute little.
        # We additionally zero-out padded positions before pooling.
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]                          # [CLS] position -> [B, d]


class ClinicalBERTEncoder(nn.Module):
    """Wrapper around HuggingFace Bio_ClinicalBERT (CLS pooling + projection)."""

    def __init__(self, hf_name: str = "emilyalsentzer/Bio_ClinicalBERT",
                 out_dim: int = 128, freeze: bool = True):
        super().__init__()
        try:
            from transformers import AutoModel
        except Exception as e:  # pragma: no cover
            raise ImportError("ClinicalBERTEncoder requires `transformers`.") from e
        self.bert = AutoModel.from_pretrained(hf_name)
        hidden = self.bert.config.hidden_size
        self.proj = nn.Linear(hidden, out_dim)
        self._out_dim = out_dim
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, input_ids: torch.Tensor,
                attn_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attn_mask)
        cls = out.last_hidden_state[:, 0]       # [CLS]
        return self.proj(cls)


def build_text_encoder(backend: str, vocab_size: int, d_model: int,
                       max_len: int, hf_name: str = "emilyalsentzer/Bio_ClinicalBERT",
                       freeze: bool = True, **kw) -> nn.Module:
    if backend == "clinicalbert":
        return ClinicalBERTEncoder(hf_name=hf_name, out_dim=d_model, freeze=freeze)
    if backend == "lite":
        return LiteTextEncoder(vocab_size=vocab_size, d_model=d_model,
                               max_len=max_len, **kw)
    raise ValueError(f"Unknown text backend: {backend!r}")
