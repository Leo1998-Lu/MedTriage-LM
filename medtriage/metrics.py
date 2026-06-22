"""Evaluation metrics for MedTriage-LM (paper Table 1).

Primary triage instruction (3-class):   Accuracy, Macro-F1
Critical-outcome (binary):              AUROC, AUPRC
Rationale quality:                      BERTScore (proxy offline)

The classification metrics use scikit-learn. For rationale quality the paper
reports BERTScore; computing the real metric needs a pretrained contextual
encoder + internet, so the default offline implementation is a transparent
token-overlap F1 (precision/recall/F1 over word tokens, the same quantity
BERTScore approximates with contextual embeddings). Set
``use_real_bertscore=True`` to fall back to the ``bert_score`` package when it
and its weights are available.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)


# --------------------------------------------------------------------------- #
# Classification metrics
# --------------------------------------------------------------------------- #
def instruction_metrics(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> Dict[str, float]:
    """Accuracy + Macro-F1 for the 3-class triage instruction."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
    }


def critical_metrics(
    y_true: Sequence[int], y_score: Sequence[float]
) -> Dict[str, float]:
    """AUROC + AUPRC for the binary critical-outcome head.

    Falls back gracefully to NaN when a fold contains a single class (AUROC is
    undefined there).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    out: Dict[str, float] = {}
    if len(np.unique(y_true)) < 2:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
        return out
    out["auroc"] = float(roc_auc_score(y_true, y_score))
    out["auprc"] = float(average_precision_score(y_true, y_score))
    return out


# --------------------------------------------------------------------------- #
# Rationale quality
# --------------------------------------------------------------------------- #
def _tokenize(text: str) -> List[str]:
    return [t for t in text.lower().replace(".", " ").replace(",", " ").split() if t]


def token_overlap_f1(pred: str, ref: str) -> float:
    """Token-level F1 between two strings (BERTScore proxy)."""
    p_tok, r_tok = _tokenize(pred), _tokenize(ref)
    if not p_tok and not r_tok:
        return 1.0
    if not p_tok or not r_tok:
        return 0.0
    p_set, r_set = set(p_tok), set(r_tok)
    overlap = len(p_set & r_set)
    if overlap == 0:
        return 0.0
    precision = overlap / len(p_set)
    recall = overlap / len(r_set)
    return 2 * precision * recall / (precision + recall)


def rationale_score(
    preds: Sequence[str],
    refs: Sequence[str],
    use_real_bertscore: bool = False,
    lang: str = "en",
) -> Dict[str, float]:
    """Mean rationale similarity over a set of (pred, ref) pairs."""
    assert len(preds) == len(refs)
    if use_real_bertscore:  # pragma: no cover - requires bert_score + internet
        from bert_score import score as bertscore
        _, _, f1 = bertscore(list(preds), list(refs), lang=lang, verbose=False)
        return {"bertscore_f1": float(f1.mean().item())}
    scores = [token_overlap_f1(p, r) for p, r in zip(preds, refs)]
    return {"bertscore_proxy_f1": float(np.mean(scores)) if scores else 0.0}


# --------------------------------------------------------------------------- #
# Aggregation across cross-validation runs
# --------------------------------------------------------------------------- #
def aggregate_runs(run_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Aggregate a list of per-run metric dicts into mean +/- 95% CI.

    The 95% confidence interval uses the normal approximation
    ``1.96 * std / sqrt(n)`` (the paper reports mean +/- 95% CI over 5 runs).
    """
    if not run_metrics:
        return {}
    keys = run_metrics[0].keys()
    agg: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = np.array([m[k] for m in run_metrics if not np.isnan(m.get(k, np.nan))],
                        dtype=float)
        if vals.size == 0:
            agg[k] = {"mean": float("nan"), "ci95": float("nan"),
                      "std": float("nan"), "n": 0}
            continue
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        ci95 = 1.96 * std / np.sqrt(vals.size) if vals.size > 1 else 0.0
        agg[k] = {"mean": mean, "ci95": float(ci95), "std": std, "n": int(vals.size)}
    return agg


def format_aggregate(agg: Dict[str, Dict[str, float]]) -> str:
    """Pretty 'metric: mean +/- ci95' table for logging."""
    lines = ["metric              mean      95% CI     (n)"]
    for k, v in agg.items():
        lines.append(f"{k:18s} {v['mean']:.4f}  +/-{v['ci95']:.4f}   "
                     f"({v['n']})")
    return "\n".join(lines)
