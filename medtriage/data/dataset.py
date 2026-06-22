"""
Dataset, feature processing, and tokenisation for MedTriage-LM.

* ``TabularProcessor`` -- fit on the *training* fold only (median imputation +
  z-score standardisation for numerics, level->index maps for categoricals),
  then applied to val/test to avoid information leakage.
* ``TextTokenizer``    -- two interchangeable back-ends:
      - ``"lite"``         a whitespace/word-piece-free vocabulary built from the
                           training complaints (offline, default);
      - ``"clinicalbert"`` the HuggingFace ``emilyalsentzer/Bio_ClinicalBERT``
                           tokenizer (requires ``transformers`` + internet).
* ``TriageDataset``    -- yields model-ready tensor dictionaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocess import Cohort


# --------------------------------------------------------------------------- #
# Tabular processing
# --------------------------------------------------------------------------- #


class TabularProcessor:
    """Median-impute + standardise numerics; index-encode categoricals."""

    def __init__(self, numeric_cols: List[str], categorical_cols: List[str],
                 cat_vocab: Dict[str, List[str]]):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.cat_vocab = {k: list(v) for k, v in cat_vocab.items()}
        self.cat_to_idx = {
            c: {lvl: i for i, lvl in enumerate(levels)}
            for c, levels in self.cat_vocab.items()
        }
        self.medians_: Dict[str, float] = {}
        self.means_: Dict[str, float] = {}
        self.stds_: Dict[str, float] = {}
        self._fitted = False

    @property
    def cat_cardinalities(self) -> List[int]:
        return [len(self.cat_vocab[c]) for c in self.categorical_cols]

    def fit(self, df: pd.DataFrame) -> "TabularProcessor":
        for c in self.numeric_cols:
            col = pd.to_numeric(df[c], errors="coerce")
            med = float(col.median()) if col.notna().any() else 0.0
            filled = col.fillna(med)
            self.medians_[c] = med
            self.means_[c] = float(filled.mean())
            self.stds_[c] = float(filled.std(ddof=0)) or 1.0
        self._fitted = True
        return self

    def transform_numeric(self, df: pd.DataFrame) -> np.ndarray:
        assert self._fitted, "TabularProcessor must be fit before transform."
        out = np.zeros((len(df), len(self.numeric_cols)), dtype=np.float32)
        for j, c in enumerate(self.numeric_cols):
            col = pd.to_numeric(df[c], errors="coerce").fillna(self.medians_[c])
            out[:, j] = ((col - self.means_[c]) / self.stds_[c]).to_numpy(np.float32)
        return out

    def transform_categorical(self, df: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(df), len(self.categorical_cols)), dtype=np.int64)
        for j, c in enumerate(self.categorical_cols):
            mapping = self.cat_to_idx[c]
            out[:, j] = df[c].map(lambda v: mapping.get(str(v).upper(), 0)).to_numpy(np.int64)
        return out


# --------------------------------------------------------------------------- #
# Text tokenisation
# --------------------------------------------------------------------------- #

_WORD_RE = re.compile(r"[a-z0-9]+|[/]")


class TextTokenizer:
    """Pluggable tokeniser. ``backend in {"lite", "clinicalbert"}``."""

    PAD, UNK, CLS = 0, 1, 2

    def __init__(self, backend: str = "lite", max_len: int = 32,
                 hf_name: str = "emilyalsentzer/Bio_ClinicalBERT",
                 min_freq: int = 1):
        self.backend = backend
        self.max_len = max_len
        self.hf_name = hf_name
        self.min_freq = min_freq
        self.vocab: Dict[str, int] = {}
        self._hf = None
        if backend == "clinicalbert":
            self._init_hf()

    # ---- lite back-end ----------------------------------------------------
    def _tokenize_lite(self, text: str) -> List[str]:
        return _WORD_RE.findall(str(text).lower())

    def fit(self, texts: List[str]) -> "TextTokenizer":
        if self.backend != "lite":
            return self
        from collections import Counter
        counter: Counter = Counter()
        for t in texts:
            counter.update(self._tokenize_lite(t))
        vocab = {"<pad>": self.PAD, "<unk>": self.UNK, "<cls>": self.CLS}
        for tok, freq in counter.most_common():
            if freq >= self.min_freq and tok not in vocab:
                vocab[tok] = len(vocab)
        self.vocab = vocab
        return self

    @property
    def vocab_size(self) -> int:
        if self.backend == "clinicalbert":
            return self._hf.vocab_size
        return max(len(self.vocab), 3)

    # ---- HF back-end ------------------------------------------------------
    def _init_hf(self):
        try:
            from transformers import AutoTokenizer
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "backend='clinicalbert' requires the `transformers` package."
            ) from e
        self._hf = AutoTokenizer.from_pretrained(self.hf_name)
        self.PAD = self._hf.pad_token_id or 0
        self.CLS = self._hf.cls_token_id or 101

    # ---- encoding ---------------------------------------------------------
    def encode(self, text: str):
        """Return (input_ids[max_len], attention_mask[max_len]) as int64 arrays."""
        if self.backend == "clinicalbert":
            enc = self._hf(str(text), truncation=True, padding="max_length",
                           max_length=self.max_len, return_tensors="np")
            return (enc["input_ids"][0].astype(np.int64),
                    enc["attention_mask"][0].astype(np.int64))
        # lite: [CLS] + tokens, padded/truncated
        toks = [self.CLS] + [self.vocab.get(t, self.UNK)
                             for t in self._tokenize_lite(text)]
        toks = toks[: self.max_len]
        mask = [1] * len(toks)
        pad = self.max_len - len(toks)
        toks += [self.PAD] * pad
        mask += [0] * pad
        return np.asarray(toks, np.int64), np.asarray(mask, np.int64)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


@dataclass
class TriageDataset(Dataset):
    """Tensor dataset for a single split.

    Each item is a dict with keys:
        num        [Dnum]        standardised numeric features
        cat        [Ncat]        categorical indices
        input_ids  [max_len]     text token ids
        attn_mask  [max_len]     text attention mask
        weak       [K]           weak anatomical labels (multi-hot)
        instruction  scalar      3-class target
        critical     scalar      binary critical-outcome target
        stay_id      scalar      identifier (for analysis)
        text         str         raw chief complaint (for rationale viz)
    """
    df: pd.DataFrame
    processor: TabularProcessor
    tokenizer: TextTokenizer
    weak_cols: List[str]
    _num: np.ndarray = field(init=False)
    _cat: np.ndarray = field(init=False)
    _weak: np.ndarray = field(init=False)
    _ids: np.ndarray = field(init=False)
    _crit: np.ndarray = field(init=False)
    _texts: List[str] = field(init=False)
    _stay: np.ndarray = field(init=False)

    def __post_init__(self):
        self.df = self.df.reset_index(drop=True)
        self._num = self.processor.transform_numeric(self.df)
        self._cat = self.processor.transform_categorical(self.df)
        self._weak = self.df[self.weak_cols].to_numpy(np.float32)
        self._ids = self.df["instruction"].to_numpy(np.int64)
        self._crit = self.df["critical"].to_numpy(np.float32)
        self._texts = self.df["text"].astype(str).tolist()
        self._stay = self.df["stay_id"].to_numpy(np.int64)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> Dict:
        input_ids, attn = self.tokenizer.encode(self._texts[i])
        return {
            "num": torch.from_numpy(self._num[i]),
            "cat": torch.from_numpy(self._cat[i]),
            "input_ids": torch.from_numpy(input_ids),
            "attn_mask": torch.from_numpy(attn),
            "weak": torch.from_numpy(self._weak[i]),
            "instruction": torch.tensor(self._ids[i], dtype=torch.long),
            "critical": torch.tensor(self._crit[i], dtype=torch.float32),
            "stay_id": torch.tensor(self._stay[i], dtype=torch.long),
            "text": self._texts[i],
        }


def collate(batch: List[Dict]) -> Dict:
    """Stack tensor fields; keep ``text`` as a list of strings."""
    out: Dict = {}
    keys = [k for k in batch[0] if k != "text"]
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["text"] = [b["text"] for b in batch]
    return out
