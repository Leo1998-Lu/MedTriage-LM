"""Smoke tests for the MedTriage-LM pipeline.

Runs with pytest (``pytest -q``) or standalone (``python tests/test_pipeline.py``).
All tests are CPU-only and use the tiny demo cohort, so the whole suite
finishes in seconds.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medtriage.data.anatomy import (
    NUM_REGIONS,
    REGION_NAMES,
    build_silhouette,
    complaint_to_weak_labels,
)
from medtriage.data.preprocess import (
    INSTRUCTION_CLASSES,
    PreprocessConfig,
    build_cohort,
)
from medtriage.data.dataset import (
    TabularProcessor,
    TextTokenizer,
    TriageDataset,
    collate,
)
from medtriage.models.medtriage_lm import (
    MedTriageLMConfig,
    build_medtriage_lm,
    count_parameters,
)
from medtriage.models.rationale import RationaleVocab, build_template_rationale
from medtriage.losses import LossWeights, MedTriageLoss
from medtriage.metrics import aggregate_runs, critical_metrics, instruction_metrics
from torch.utils.data import DataLoader

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data")


# --------------------------------------------------------------------------- #
# Fixtures (plain helpers so the file also runs without pytest)
# --------------------------------------------------------------------------- #
def _cohort():
    return build_cohort(PreprocessConfig(data_dir=_DATA_DIR))


def _mini_setup(n=16):
    cohort = _cohort()
    df = cohort.df.iloc[:n].reset_index(drop=True)
    num, catc, wk = cohort.numeric_cols, cohort.categorical_cols, cohort.weak_label_cols
    tab = TabularProcessor(num, catc, cohort.cat_vocab).fit(df)
    tok = TextTokenizer("lite", 32).fit(df["text"].astype(str).tolist())
    refs = [build_template_rationale(int(df["instruction"].iloc[i]),
            df[wk].to_numpy(np.float32)[i], str(df["text"].iloc[i]))
            for i in range(len(df))]
    rv = RationaleVocab().fit(refs)
    mcfg = MedTriageLMConfig()
    model = build_medtriage_lm(mcfg, len(num), tab.cat_cardinalities,
                               tok.vocab_size, rv.vocab_size)
    ds = TriageDataset(df, tab, tok, wk)
    return df, num, wk, tab, tok, rv, model, ds, refs


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_anatomy_weak_labels():
    y = complaint_to_weak_labels("Chest pain, Jaw pain, L Arm pain")
    assert y.shape == (NUM_REGIONS,)
    assert set(np.unique(y)).issubset({0.0, 1.0})
    assert y[REGION_NAMES.index("chest")] == 1.0
    # empty complaint falls back to systemic
    y0 = complaint_to_weak_labels("")
    assert y0[REGION_NAMES.index("systemic")] == 1.0


def test_silhouette_shape():
    sil = build_silhouette(64, 48)
    assert sil.shape == (64, 48)
    assert sil.max() <= 1.0 and sil.min() >= 0.0
    assert sil.sum() > 0


def test_preprocess_labels():
    cohort = _cohort()
    df = cohort.df
    assert {"instruction", "critical", "text"}.issubset(df.columns)
    assert df["instruction"].isin(range(len(INSTRUCTION_CLASSES))).all()
    assert df["critical"].isin([0, 1]).all()
    assert len(cohort.weak_label_cols) == NUM_REGIONS
    assert len(df) > 100  # demo cohort kept after dropping unlabeled acuity


def test_field_is_differentiable():
    _, _, _, _, _, _, model, _, _ = _mini_setup(8)
    mu = torch.rand(4, NUM_REGIONS, requires_grad=True)
    vpm, H, H_norm = model.field_generator(mu)
    vpm.sum().backward()
    assert mu.grad is not None and torch.isfinite(mu.grad).all()
    assert mu.grad.abs().sum() > 0


def test_model_forward_shapes():
    df, num, wk, tab, tok, rv, model, ds, refs = _mini_setup(16)
    dl = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate)
    batch = next(iter(dl))
    rtgt = torch.from_numpy(np.stack([rv.encode(r, 48) for r in refs[:8]])).long()
    out = model(batch["num"], batch["cat"], batch["input_ids"],
                batch["attn_mask"], rationale_tgt=rtgt)
    assert out.inst_logits.shape == (8, 3)
    assert out.crit_logit.shape == (8,)
    assert out.mu.shape == (8, NUM_REGIONS)
    assert out.vpm.shape[0] == 8 and out.vpm.shape[1] == 3
    assert out.rationale_logits.shape[0] == 8
    assert count_parameters(model) > 0


def test_loss_backward_flows_to_mapper():
    df, num, wk, tab, tok, rv, model, ds, refs = _mini_setup(16)
    dl = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate)
    batch = next(iter(dl))
    rtgt = torch.from_numpy(np.stack([rv.encode(r, 48) for r in refs[:8]])).long()
    out = model(batch["num"], batch["cat"], batch["input_ids"],
                batch["attn_mask"], rationale_tgt=rtgt)
    crit = MedTriageLoss(LossWeights())
    terms = crit(out, batch, rationale_tgt=rtgt)
    for key in ("L_CE", "L_BCE", "L_weak", "L_smooth", "L_gen"):
        assert key in terms
    terms["loss"].backward()
    grad = model.region_mapper.net[0].weight.grad
    assert grad is not None and grad.abs().sum() > 0  # visual path is connected


def test_metrics_basic():
    m = instruction_metrics([0, 1, 2, 1], [0, 1, 2, 0])
    assert 0 <= m["accuracy"] <= 1 and 0 <= m["macro_f1"] <= 1
    c = critical_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert c["auroc"] == 1.0
    agg = aggregate_runs([{"accuracy": 0.8}, {"accuracy": 0.9}])
    assert abs(agg["accuracy"]["mean"] - 0.85) < 1e-9
    assert agg["accuracy"]["ci95"] >= 0


# --------------------------------------------------------------------------- #
# Standalone runner
# --------------------------------------------------------------------------- #
def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} tests passed.")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    _run_all()
