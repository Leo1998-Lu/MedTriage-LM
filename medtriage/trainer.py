"""Training, evaluation and cross-validation for MedTriage-LM.

The trainer fits all data-dependent preprocessors on the *training fold only*
(numeric standardisation, categorical vocab, text vocab, rationale vocab) to
avoid leakage, builds the model, optimises the composite objective
(:class:`medtriage.losses.MedTriageLoss`), and evaluates the Table-1 metrics.

``cross_validate`` runs stratified k-fold CV and aggregates results as
mean +/- 95% CI, mirroring the paper's 5-run protocol.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from .data.dataset import TabularProcessor, TextTokenizer, TriageDataset, collate
from .data.preprocess import INSTRUCTION_CLASSES
from .models.medtriage_lm import (
    MedTriageLM,
    MedTriageLMConfig,
    build_medtriage_lm,
    count_parameters,
)
from .models.rationale import RationaleVocab, build_template_rationale
from .losses import LossWeights, MedTriageLoss
from .metrics import (
    aggregate_runs,
    critical_metrics,
    format_aggregate,
    instruction_metrics,
    rationale_score,
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 2e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "cpu"
    num_workers: int = 0
    n_folds: int = 5
    val_fraction: float = 0.2            # used by single train/val split helper
    text_max_len: int = 32
    rationale_max_len: int = 48
    use_class_weight: bool = True
    use_pos_weight: bool = True
    use_rationale: bool = True
    use_real_bertscore: bool = False
    log_every: int = 0                   # 0 = silent per-step, log per-epoch only
    num_threads: int = 0                 # 0 = leave torch default


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_reference_rationales(df: pd.DataFrame, weak_cols: List[str]) -> List[str]:
    """Deterministic reference rationale per row from (instruction, weak, text).

    Ground-truth weak labels act as the region-intensity vector ``mu`` so the
    reference describes the truly involved anatomy in the Fig. 2 template
    format. These strings are the supervision target for L_gen and the
    reference for the rationale quality metric.
    """
    refs: List[str] = []
    weak = df[weak_cols].to_numpy(np.float32)
    instr = df["instruction"].to_numpy(int)
    texts = df["text"].astype(str).tolist()
    for i in range(len(df)):
        refs.append(build_template_rationale(instr[i], weak[i], texts[i]))
    return refs


def _class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights, normalised to mean 1."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def _pos_weight(y: np.ndarray) -> torch.Tensor:
    """pos_weight = #neg / #pos for BCEWithLogits."""
    pos = float(y.sum())
    neg = float(len(y) - pos)
    val = neg / pos if pos > 0 else 1.0
    return torch.tensor(val, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Fold preparation
# --------------------------------------------------------------------------- #
@dataclass
class FoldArtifacts:
    model: MedTriageLM
    tab: TabularProcessor
    tok: TextTokenizer
    rvocab: Optional[RationaleVocab]
    train_loader: DataLoader
    val_loader: DataLoader
    rationale_tgt: Optional[np.ndarray]          # [N_all_rows_in_fold, T] by stay
    stay_to_idx: Dict[int, int]
    criterion: MedTriageLoss


def prepare_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    cat_vocab: Dict[str, List[str]],
    weak_cols: List[str],
    model_cfg: MedTriageLMConfig,
    train_cfg: TrainConfig,
) -> FoldArtifacts:
    # --- fit preprocessors on TRAIN ONLY ----------------------------------
    tab = TabularProcessor(numeric_cols, categorical_cols, cat_vocab).fit(train_df)
    tok = TextTokenizer(
        backend="lite" if model_cfg.text_backbone == "lite" else "clinicalbert",
        max_len=train_cfg.text_max_len,
        hf_name=model_cfg.text_hf_name,
    ).fit(train_df["text"].astype(str).tolist())

    # --- rationale targets -------------------------------------------------
    rvocab: Optional[RationaleVocab] = None
    rationale_tgt: Optional[np.ndarray] = None
    stay_to_idx: Dict[int, int] = {}
    if train_cfg.use_rationale and model_cfg.use_rationale:
        train_refs = build_reference_rationales(train_df, weak_cols)
        rvocab = RationaleVocab().fit(train_refs)
        # encode references for *all* rows in this fold (train + val), keyed by stay_id
        all_df = pd.concat([train_df, val_df], axis=0)
        all_refs = build_reference_rationales(all_df, weak_cols)
        stays = all_df["stay_id"].to_numpy(int)
        ids = np.stack(
            [rvocab.encode(r, train_cfg.rationale_max_len) for r in all_refs], axis=0
        )
        rationale_tgt = ids
        stay_to_idx = {int(s): i for i, s in enumerate(stays)}

    # --- datasets / loaders -----------------------------------------------
    train_ds = TriageDataset(train_df, tab, tok, weak_cols)
    val_ds = TriageDataset(val_df, tab, tok, weak_cols)
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size, shuffle=True,
        collate_fn=collate, num_workers=train_cfg.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg.batch_size, shuffle=False,
        collate_fn=collate, num_workers=train_cfg.num_workers,
    )

    # --- model -------------------------------------------------------------
    rvocab_size = rvocab.vocab_size if rvocab is not None else None
    model = build_medtriage_lm(
        cfg=model_cfg,
        n_numeric=len(numeric_cols),
        cat_cardinalities=tab.cat_cardinalities,
        text_vocab_size=tok.vocab_size,
        rationale_vocab_size=rvocab_size,
    ).to(train_cfg.device)

    # --- loss --------------------------------------------------------------
    cw = (_class_weights(train_df["instruction"].to_numpy(int),
                         model_cfg.n_instruction_classes).to(train_cfg.device)
          if train_cfg.use_class_weight else None)
    pw = (_pos_weight(train_df["critical"].to_numpy(int)).to(train_cfg.device)
          if train_cfg.use_pos_weight else None)
    criterion = MedTriageLoss(weights=LossWeights(), class_weight=cw,
                              pos_weight=pw).to(train_cfg.device)

    return FoldArtifacts(
        model=model, tab=tab, tok=tok, rvocab=rvocab,
        train_loader=train_loader, val_loader=val_loader,
        rationale_tgt=rationale_tgt, stay_to_idx=stay_to_idx,
        criterion=criterion,
    )


# --------------------------------------------------------------------------- #
# Train / eval steps
# --------------------------------------------------------------------------- #
def _gather_rationale_tgt(
    batch: Dict, art: FoldArtifacts, device: str
) -> Optional[torch.Tensor]:
    if art.rationale_tgt is None:
        return None
    stays = batch["stay_id"].tolist()
    rows = [art.stay_to_idx[int(s)] for s in stays]
    tgt = torch.from_numpy(art.rationale_tgt[rows]).long().to(device)
    return tgt


def train_one_epoch(art: FoldArtifacts, optim, cfg: TrainConfig) -> Dict[str, float]:
    art.model.train()
    running: Dict[str, float] = {}
    n = 0
    for batch in art.train_loader:
        for k in ("num", "cat", "input_ids", "attn_mask", "weak",
                  "instruction", "critical", "stay_id"):
            batch[k] = batch[k].to(cfg.device)
        rtgt = _gather_rationale_tgt(batch, art, cfg.device)

        out = art.model(batch["num"], batch["cat"], batch["input_ids"],
                        batch["attn_mask"], rationale_tgt=rtgt)
        terms = art.criterion(out, batch, rationale_tgt=rtgt)
        loss = terms["loss"]

        optim.zero_grad()
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(art.model.parameters(), cfg.grad_clip)
        optim.step()

        bs = batch["instruction"].size(0)
        n += bs
        for k, v in terms.items():
            running[k] = running.get(k, 0.0) + float(v.detach()) * bs
    return {k: v / max(n, 1) for k, v in running.items()}


@torch.no_grad()
def evaluate(art: FoldArtifacts, cfg: TrainConfig) -> Dict[str, float]:
    art.model.eval()
    inst_true, inst_pred = [], []
    crit_true, crit_score = [], []
    rat_pred, rat_ref = [], []

    for batch in art.val_loader:
        for k in ("num", "cat", "input_ids", "attn_mask", "weak",
                  "instruction", "critical", "stay_id"):
            batch[k] = batch[k].to(cfg.device)
        out = art.model(batch["num"], batch["cat"], batch["input_ids"],
                        batch["attn_mask"])

        inst_true.extend(batch["instruction"].tolist())
        inst_pred.extend(out.inst_logits.argmax(-1).tolist())
        crit_true.extend(batch["critical"].tolist())
        crit_score.extend(torch.sigmoid(out.crit_logit).tolist())

        # rationale generation (lite decoder only)
        if art.model.rationale_decoder is not None and art.rationale_tgt is not None:
            gen_ids = art.model.rationale_decoder.generate(out.h_multi)
            for j, ids in enumerate(gen_ids):
                rat_pred.append(art.rvocab.decode(ids))
                ridx = art.stay_to_idx[int(batch["stay_id"][j])]
                rat_ref.append(art.rvocab.decode(art.rationale_tgt[ridx].tolist()))

    metrics: Dict[str, float] = {}
    metrics.update(instruction_metrics(inst_true, inst_pred))
    metrics.update(critical_metrics(crit_true, crit_score))
    if rat_pred:
        metrics.update(rationale_score(rat_pred, rat_ref,
                                       use_real_bertscore=cfg.use_real_bertscore))
    return metrics


# --------------------------------------------------------------------------- #
# Single fold
# --------------------------------------------------------------------------- #
def train_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    cat_vocab: Dict[str, List[str]],
    weak_cols: List[str],
    model_cfg: MedTriageLMConfig,
    train_cfg: TrainConfig,
    verbose: bool = True,
) -> Tuple[Dict[str, float], FoldArtifacts]:
    art = prepare_fold(train_df, val_df, numeric_cols, categorical_cols,
                       cat_vocab, weak_cols, model_cfg, train_cfg)
    optim = torch.optim.AdamW(art.model.parameters(), lr=train_cfg.lr,
                              weight_decay=train_cfg.weight_decay)

    best, best_key = {}, -1.0
    for epoch in range(1, train_cfg.epochs + 1):
        tr = train_one_epoch(art, optim, train_cfg)
        ev = evaluate(art, train_cfg)
        # model selection on macro-F1 (primary task)
        sel = ev.get("macro_f1", 0.0)
        if sel >= best_key:
            best_key, best = sel, ev
        if verbose and (epoch == 1 or epoch % max(1, train_cfg.epochs // 6) == 0
                        or epoch == train_cfg.epochs):
            msg = (f"  epoch {epoch:3d}/{train_cfg.epochs}  "
                   f"loss={tr.get('loss', 0):.3f}  "
                   f"acc={ev.get('accuracy', 0):.3f}  "
                   f"f1={ev.get('macro_f1', 0):.3f}  "
                   f"auroc={ev.get('auroc', float('nan')):.3f}")
            if "bertscore_proxy_f1" in ev:
                msg += f"  rat={ev['bertscore_proxy_f1']:.3f}"
            print(msg)
    return best, art


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
def cross_validate(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    cat_vocab: Dict[str, List[str]],
    weak_cols: List[str],
    model_cfg: MedTriageLMConfig,
    train_cfg: TrainConfig,
    verbose: bool = True,
) -> Dict:
    if train_cfg.num_threads > 0:
        torch.set_num_threads(train_cfg.num_threads)

    y = df["instruction"].to_numpy(int)
    skf = StratifiedKFold(n_splits=train_cfg.n_folds, shuffle=True,
                          random_state=train_cfg.seed)
    run_metrics: List[Dict[str, float]] = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        set_seed(train_cfg.seed + fold)
        tr_df = df.iloc[tr_idx].reset_index(drop=True)
        va_df = df.iloc[va_idx].reset_index(drop=True)
        if verbose:
            print(f"\n=== Fold {fold}/{train_cfg.n_folds} "
                  f"(train={len(tr_df)}, val={len(va_df)}) ===")
        best, _ = train_fold(tr_df, va_df, numeric_cols, categorical_cols,
                             cat_vocab, weak_cols, model_cfg, train_cfg,
                             verbose=verbose)
        run_metrics.append(best)
        if verbose:
            print(f"  fold {fold} best: " +
                  "  ".join(f"{k}={v:.3f}" for k, v in best.items()))

    agg = aggregate_runs(run_metrics)
    result = {"per_fold": run_metrics, "aggregate": agg}
    if verbose:
        print("\n===== Cross-validation summary (mean +/- 95% CI) =====")
        print(format_aggregate(agg))
    return result


# --------------------------------------------------------------------------- #
# Train on all data (for deployment / inference artifacts)
# --------------------------------------------------------------------------- #
def train_full(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    cat_vocab: Dict[str, List[str]],
    weak_cols: List[str],
    model_cfg: MedTriageLMConfig,
    train_cfg: TrainConfig,
    verbose: bool = True,
) -> FoldArtifacts:
    """Fit on the entire cohort; a small held-out slice is used only for logs."""
    if train_cfg.num_threads > 0:
        torch.set_num_threads(train_cfg.num_threads)
    set_seed(train_cfg.seed)
    # tiny stratified holdout purely for progress logging
    skf = StratifiedKFold(n_splits=max(5, int(1 / train_cfg.val_fraction)),
                          shuffle=True, random_state=train_cfg.seed)
    tr_idx, va_idx = next(skf.split(np.zeros(len(df)),
                                    df["instruction"].to_numpy(int)))
    tr_df = df.iloc[tr_idx].reset_index(drop=True)
    va_df = df.iloc[va_idx].reset_index(drop=True)
    _, art = train_fold(tr_df, va_df, numeric_cols, categorical_cols,
                        cat_vocab, weak_cols, model_cfg, train_cfg,
                        verbose=verbose)
    if verbose:
        print(f"\nTrained MedTriage-LM with "
              f"{count_parameters(art.model):,} parameters.")
    return art
