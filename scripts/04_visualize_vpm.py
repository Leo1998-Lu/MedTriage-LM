#!/usr/bin/env python3
"""Step 04 -- Visualise Visual Phenotype Maps from a trained checkpoint.

Loads the checkpoint saved by step 02, picks representative ED stays (by
default one per triage-instruction class), and renders the VPM + region
intensities + cross-modal attention + generated rationale to PNG files.

Usage:
    python scripts/04_visualize_vpm.py --ckpt artifacts/medtriage_lm.pt \
        --cohort artifacts/cohort.csv --out assets --per-class 1
"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medtriage.data.dataset import TriageDataset
from medtriage.data.preprocess import INSTRUCTION_CLASSES
from medtriage.models.medtriage_lm import build_medtriage_lm
from medtriage.viz import visualize_dataset_samples, render_template_showcase


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/medtriage_lm.pt")
    ap.add_argument("--cohort", default="artifacts/cohort.csv")
    ap.add_argument("--out", default="assets")
    ap.add_argument("--per-class", type=int, default=1,
                    help="how many example stays to render per instruction class")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, weights_only=False)
    model_cfg = ckpt["model_cfg"]
    tab, tok, rvocab = ckpt["tab"], ckpt["tok"], ckpt["rvocab"]
    numeric_cols = ckpt["numeric_cols"]
    weak_cols = ckpt["weak_cols"]

    model = build_medtriage_lm(
        cfg=model_cfg,
        n_numeric=len(numeric_cols),
        cat_cardinalities=tab.cat_cardinalities,
        text_vocab_size=tok.vocab_size,
        rationale_vocab_size=rvocab.vocab_size if rvocab is not None else None,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    art = SimpleNamespace(model=model, rvocab=rvocab)

    df = pd.read_csv(args.cohort)
    dataset = TriageDataset(df, tab, tok, weak_cols)

    # pick representative rows: first `per_class` of each instruction class
    indices = []
    for cls in range(len(INSTRUCTION_CLASSES)):
        rows = df.index[df["instruction"] == cls].tolist()[: args.per_class]
        indices.extend(rows)

    paths = visualize_dataset_samples(art, dataset, indices, args.out)

    # standalone reference-style template + thermal VPM showcase
    showcase = render_template_showcase(os.path.join(args.out, "template_showcase.png"))
    paths.append(showcase)

    print("Rendered figures:")
    for p in paths:
        print(" ", p)


if __name__ == "__main__":
    main()
