"""
FT-Transformer tabular encoder (Sec. 2.1, Gorishniy et al., NeurIPS 2021).

Produces the structured latent representation ``h_s = f_FT(x_s)`` from the
triage-time numeric vitals and categorical demographics.

Pipeline:
    Feature Tokenizer
        numeric  x_j  ->  e_j = x_j * W_j + b_j          (W_j, b_j in R^d)
        categ.   c_j  ->  E[c_j]  (learned lookup)
        prepend a learnable [CLS] token
    L pre-norm Transformer blocks (MHSA + FFN, GELU)
    return the [CLS] embedding as h_s in R^d
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class NumericalFeatureTokenizer(nn.Module):
    """Per-feature affine embedding for numeric inputs: e_j = x_j * W_j + b_j."""

    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.normal_(self.weight, std=d_token ** -0.5)
        nn.init.normal_(self.bias, std=d_token ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, n_features]
        # [B, n_features, d_token]
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class CategoricalFeatureTokenizer(nn.Module):
    """One embedding table per categorical feature."""

    def __init__(self, cardinalities: List[int], d_token: int):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, d_token) for card in cardinalities]
        )
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, std=d_token ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, n_cat] (long)
        if len(self.embeddings) == 0:
            return x.new_zeros((x.shape[0], 0, 0), dtype=torch.float32)
        toks = [emb(x[:, j]) for j, emb in enumerate(self.embeddings)]
        return torch.stack(toks, dim=1)  # [B, n_cat, d_token]


class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block."""

    def __init__(self, d_token: int, n_heads: int, ffn_mult: float = 2.0,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(d_token, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(d_token)
        hidden = int(d_token * ffn_mult)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, d_token),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class FTTransformer(nn.Module):
    """FT-Transformer encoder returning the [CLS] representation h_s."""

    def __init__(self, n_numeric: int, cat_cardinalities: List[int],
                 d_token: int = 64, n_blocks: int = 2, n_heads: int = 4,
                 ffn_mult: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.d_token = d_token
        self.num_tok = NumericalFeatureTokenizer(n_numeric, d_token) \
            if n_numeric > 0 else None
        self.cat_tok = CategoricalFeatureTokenizer(cat_cardinalities, d_token) \
            if len(cat_cardinalities) > 0 else None
        self.cls = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.cls, std=d_token ** -0.5)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_token, n_heads, ffn_mult, dropout)
             for _ in range(n_blocks)]
        )
        self.norm = nn.LayerNorm(d_token)

    @property
    def out_dim(self) -> int:
        return self.d_token

    def forward(self, num: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        B = num.shape[0] if num is not None else cat.shape[0]
        tokens = [self.cls.expand(B, -1, -1)]
        if self.num_tok is not None and num is not None and num.shape[1] > 0:
            tokens.append(self.num_tok(num))
        if self.cat_tok is not None and cat is not None and cat.shape[1] > 0:
            tokens.append(self.cat_tok(cat))
        x = torch.cat(tokens, dim=1)            # [B, 1 + n_num + n_cat, d]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x[:, 0])               # h_s = [CLS]  -> [B, d]
