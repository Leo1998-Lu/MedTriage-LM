#!/usr/bin/env python3
"""Step 02 -- Cross-validate and train MedTriage-LM.

Runs stratified k-fold cross-validation (paper protocol: mean +/- 95% CI),
then trains a final model on the full cohort and saves a checkpoint with the
fitted preprocessors so it can be used for evaluation and visualisation.

Usage:
    python scripts/02_train.py --config configs/default.yaml \
        --cohort artifacts/cohort.csv --out artifacts
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medtriage.config import configs_from_file
from medtriage.data.preprocess import build_cohort, save_cohort
from medtriage.trainer import cross_validate, train_full


def _load_cohort(cohort_csv, pre_cfg, out_dir):
    """Load cohort.csv + meta; build it first if missing."""
    meta_path = os.path.join(os.path.dirname(cohort_csv), "cohort_meta.json")
    if not (os.path.exists(cohort_csv) and os.path.exists(meta_path)):
        print("[02_train] cohort not found, building it now...")
        cohort = build_cohort(pre_cfg)
        save_cohort(cohort, out_dir)
        cohort_csv = os.path.join(out_dir, "cohort.csv")
        meta_path = os.path.join(out_dir, "cohort_meta.json")
    df = pd.read_csv(cohort_csv)
    with open(meta_path) as f:
        meta = json.load(f)
    return df, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--cohort", default="artifacts/cohort.csv")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--skip-cv", action="store_true",
                    help="skip cross-validation, only train the final model")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pre_cfg, model_cfg, train_cfg = configs_from_file(args.config)
    df, meta = _load_cohort(args.cohort, pre_cfg, args.out)

    numeric_cols = meta["numeric_cols"]
    categorical_cols = meta["categorical_cols"]
    cat_vocab = meta["cat_vocab"]
    weak_cols = meta["weak_label_cols"]

    if not args.skip_cv:
        print("=" * 70)
        print("Cross-validation")
        print("=" * 70)
        cv = cross_validate(df, numeric_cols, categorical_cols, cat_vocab,
                            weak_cols, model_cfg, train_cfg, verbose=True)
        with open(os.path.join(args.out, "cv_results.json"), "w") as f:
            json.dump(cv, f, indent=2)
        print(f"\nSaved CV results -> {os.path.join(args.out, 'cv_results.json')}")

    print("\n" + "=" * 70)
    print("Training final model on the full cohort")
    print("=" * 70)
    art = train_full(df, numeric_cols, categorical_cols, cat_vocab,
                     weak_cols, model_cfg, train_cfg, verbose=True)

    ckpt = {
        "model_state": art.model.state_dict(),
        "model_cfg": model_cfg,
        "tab": art.tab,
        "tok": art.tok,
        "rvocab": art.rvocab,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "weak_cols": weak_cols,
    }
    ckpt_path = os.path.join(args.out, "medtriage_lm.pt")
    torch.save(ckpt, ckpt_path)
    print(f"\nSaved checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
