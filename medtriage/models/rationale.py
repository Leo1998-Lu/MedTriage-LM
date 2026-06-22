"""
Rationale generation (Sec. 2.4, Eq. 8).

The paper conditions a Qwen3-VL-8B backbone on ``h_multi`` (as a soft prompt)
and decodes an actionable textual rationale, optimised with a causal-LM loss
``L_gen``.  We provide three interchangeable mechanisms:

1. ``build_template_rationale`` -- a deterministic, structured rationale that
   mirrors the *exact format* of the rationales shown in Fig. 2 of the paper
   ("{Instruction} is assigned because {symptoms} are used to synthesize a VPM
   showing {distress}, consistent with {condition}").  Always available offline.
   It is used both for display and as the supervision target for L_gen.

2. ``LiteRationaleDecoder`` -- a compact autoregressive Transformer decoder
   conditioned on ``h_multi`` via a learned prefix.  Trained with teacher
   forcing against the template rationale, it makes ``L_gen`` a *real,
   optimisable* term in the offline pipeline and can ``generate`` text at
   inference.

3. ``QwenRationaleGenerator`` -- optional prompt-based wrapper around a
   HuggingFace causal LM (e.g. Qwen) for the user's GPU environment.  A
   documented extension point shows where true soft-prompt injection of
   ``h_multi`` would attach.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..data.anatomy import REGION_NAMES
from ..data.preprocess import INSTRUCTION_CLASSES
from .ft_transformer import TransformerBlock


# --------------------------------------------------------------------------- #
# 1. Deterministic template rationale  (matches Fig. 2 format)
# --------------------------------------------------------------------------- #

# region -> short anatomical descriptor used in the rationale text
_REGION_PHRASE = {
    "head": "cranial/neurological",
    "face": "facial",
    "neck": "cervical",
    "chest": "thoracic",
    "abdomen": "abdominal",
    "pelvis": "pelvic/genitourinary",
    "left_arm": "left-arm",
    "right_arm": "right-arm",
    "left_leg": "left-leg",
    "right_leg": "right-leg",
    "back": "spinal",
    "systemic": "multifocal systemic",
}

_INSTRUCTION_CLAUSE = {
    0: "immediate life-saving intervention is indicated",
    1: "a high-risk situation warrants urgent assessment",
    2: "anticipated resource needs should be estimated for further workup",
}


def _clean_symptoms(text: str) -> str:
    """Turn a raw chief complaint into a readable comma-listed symptom phrase."""
    if not text:
        return "the presenting symptoms"
    parts = re.split(r"[,/]", str(text))
    parts = [p.strip().lower() for p in parts if p.strip()]
    parts = [p for p in parts if p not in {"transfer", "s", "p", "uta", "ua"}]
    if not parts:
        return "the presenting symptoms"
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + (", and " if len(parts) > 2 else " and ") + parts[-1]


def build_template_rationale(instruction: int, mu: np.ndarray, text: str,
                             top_k: int = 3, thr: float = 0.3) -> str:
    """Construct the structured rationale string for one sample."""
    name = INSTRUCTION_CLASSES[int(instruction)]
    order = np.argsort(-mu)
    active = [i for i in order if mu[i] >= thr][:top_k]
    if not active:
        active = list(order[:1])
    distress = ", ".join(_REGION_PHRASE[REGION_NAMES[i]] for i in active) + " distress"
    symptoms = _clean_symptoms(text)
    clause = _INSTRUCTION_CLAUSE[int(instruction)]
    return (f"{name} is assigned because {symptoms} are used to synthesize a "
            f"VPM showing {distress}, consistent with the assessment that {clause}.")


# --------------------------------------------------------------------------- #
# 2. Lite autoregressive rationale decoder (gives a real L_gen)
# --------------------------------------------------------------------------- #

class RationaleVocab:
    """Word-level vocabulary for rationale decoding."""
    PAD, BOS, EOS, UNK = 0, 1, 2, 3

    def __init__(self):
        self.tok2id: Dict[str, int] = {
            "<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        self.id2tok: List[str] = ["<pad>", "<bos>", "<eos>", "<unk>"]

    @staticmethod
    def _split(s: str) -> List[str]:
        return re.findall(r"[a-z0-9]+|[.,/]", s.lower())

    def fit(self, texts: List[str]) -> "RationaleVocab":
        for t in texts:
            for w in self._split(t):
                if w not in self.tok2id:
                    self.tok2id[w] = len(self.id2tok)
                    self.id2tok.append(w)
        return self

    def __len__(self) -> int:
        return len(self.id2tok)

    @property
    def vocab_size(self) -> int:
        return len(self.id2tok)

    def encode(self, text: str, max_len: int) -> np.ndarray:
        ids = [self.BOS] + [self.tok2id.get(w, self.UNK)
                            for w in self._split(text)] + [self.EOS]
        ids = ids[:max_len]
        ids += [self.PAD] * (max_len - len(ids))
        return np.asarray(ids, dtype=np.int64)

    def decode(self, ids: List[int]) -> str:
        out = []
        for i in ids:
            if i in (self.PAD, self.BOS):
                continue
            if i == self.EOS:
                break
            out.append(self.id2tok[i] if i < len(self.id2tok) else "<unk>")
        # join with spaces but keep punctuation attached
        s = " ".join(out)
        s = re.sub(r"\s+([.,/])", r"\1", s)
        return s


class LiteRationaleDecoder(nn.Module):
    """Causal Transformer decoder conditioned on h_multi via a learned prefix."""

    def __init__(self, vocab_size: int, d_cond: int, d_model: int = 96,
                 n_blocks: int = 2, n_heads: int = 4, max_len: int = 48,
                 dropout: float = 0.1):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.cond_proj = nn.Linear(d_cond, d_model)      # h_multi -> prefix token
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout=dropout)
             for _ in range(n_blocks)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, T: int, device) -> torch.Tensor:
        return torch.triu(torch.full((T, T), float("-inf"), device=device),
                          diagonal=1)

    def _run_blocks(self, x: torch.Tensor) -> torch.Tensor:
        # apply causal masking inside each MultiheadAttention via attn_mask
        T = x.shape[1]
        mask = self._causal_mask(T, x.device)
        for blk in self.blocks:
            h = blk.norm1(x)
            a, _ = blk.attn(h, h, h, attn_mask=mask, need_weights=False)
            x = x + blk.drop(a)
            x = x + blk.drop(blk.ffn(blk.norm2(x)))
        return self.norm(x)

    def forward(self, h_multi: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        """Teacher-forced logits. h_multi:[B,d_cond], tgt_ids:[B,T] ->
        logits:[B,T,V] aligned so that logits[:, t] predicts tgt_ids[:, t]."""
        B, T = tgt_ids.shape
        cond = self.cond_proj(h_multi).unsqueeze(1)       # [B,1,d] prefix
        tok = self.tok_emb(tgt_ids)                       # [B,T,d]
        x = torch.cat([cond, tok[:, :-1]], dim=1)         # shift: prefix + y_<t
        x = x + self.pos_emb[:, : x.shape[1]]
        x = self._run_blocks(x)
        return self.head(x)                               # [B,T,V]

    @torch.no_grad()
    def generate(self, h_multi: torch.Tensor, bos: int = 1, eos: int = 2,
                 max_len: Optional[int] = None) -> List[List[int]]:
        max_len = max_len or self.max_len
        B = h_multi.shape[0]
        device = h_multi.device
        cond = self.cond_proj(h_multi).unsqueeze(1)       # [B,1,d] prefix
        # ys holds y_0..y_k; to predict y_{k+1} we feed [cond, emb(ys)] and read
        # the last position -- exactly mirroring the teacher-forced forward pass.
        ys = torch.full((B, 1), bos, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(max_len):
            tok = self.tok_emb(ys)                         # [B, k+1, d]
            x = torch.cat([cond, tok], dim=1)             # [B, k+2, d]
            x = x[:, : self.pos_emb.shape[1]]             # respect max length
            x = x + self.pos_emb[:, : x.shape[1]]
            x = self._run_blocks(x)
            logits = self.head(x[:, -1])                   # [B,V] -> y_{k+1}
            nxt = logits.argmax(-1)                        # greedy
            nxt = torch.where(finished, torch.full_like(nxt, 0), nxt)
            ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
            finished = finished | (nxt == eos)
            if bool(finished.all()):
                break
        return ys.tolist()


# --------------------------------------------------------------------------- #
# 3. Optional Qwen / HF causal-LM rationale generator (user GPU env)
# --------------------------------------------------------------------------- #

class QwenRationaleGenerator:  # pragma: no cover - requires transformers + weights
    """Prompt-based rationale generation with a HuggingFace causal LM.

    NOTE: the paper injects ``h_multi`` as a *soft prompt*. True soft-prompt
    conditioning requires model-specific embedding plumbing; the hook below
    (``_soft_prefix``) marks where projected ``h_multi`` embeddings would be
    concatenated to the input embeddings. By default we condition the LLM on a
    *textual* summary of the structured evidence, which is fully runnable.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct",
                 device: str = "cuda", max_new_tokens: int = 96):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map=device)
        self.max_new_tokens = max_new_tokens

    def _prompt(self, instruction: int, mu: np.ndarray, text: str) -> str:
        name = INSTRUCTION_CLASSES[int(instruction)]
        active = [REGION_NAMES[i] for i in np.argsort(-mu)[:3] if mu[i] >= 0.3]
        regions = ", ".join(active) if active else "systemic"
        return (
            "You are an emergency-triage assistant. Given the predicted triage "
            f"instruction '{name}', the chief complaint '{text}', and the most "
            f"active anatomical regions [{regions}] from the Visual Phenotype "
            "Map, write one concise, clinically actionable sentence explaining "
            "the decision in the form: '<Instruction> is assigned because "
            "<symptoms> are used to synthesize a VPM showing <distress>, "
            "consistent with <condition>.'"
        )

    def generate(self, instruction: int, mu: np.ndarray, text: str) -> str:
        msgs = [{"role": "user", "content": self._prompt(instruction, mu, text)}]
        inputs = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        out = self.model.generate(inputs, max_new_tokens=self.max_new_tokens,
                                  do_sample=False)
        return self.tok.decode(out[0, inputs.shape[1]:], skip_special_tokens=True).strip()

    # extension point for true soft-prompt injection of h_multi
    def _soft_prefix(self, h_multi_proj):  # pragma: no cover
        raise NotImplementedError(
            "Attach projected h_multi embeddings to inputs_embeds here for "
            "soft-prompt conditioning, matching Eq. 8 of the paper.")
