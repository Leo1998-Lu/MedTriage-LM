#!/usr/bin/env python3
"""Step 01 -- Preprocess the MIMIC-IV-ED demo CSVs into a model-ready cohort.

Usage:
    python scripts/01_preprocess.py --config configs/default.yaml \
        --out artifacts

Outputs:
    <out>/cohort.csv         one row per ED stay (features + derived labels)
    <out>/cohort_meta.json   columns, category vocab, label schema
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medtriage.config import configs_from_file
from medtriage.data.preprocess import build_cohort, cohort_summary, save_cohort


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()

    pre_cfg, _, _ = configs_from_file(args.config)
    cohort = build_cohort(pre_cfg)
    print(cohort_summary(cohort))
    path = save_cohort(cohort, args.out)
    print(f"\nSaved cohort -> {path}")
    print(f"Saved meta   -> {os.path.join(args.out, 'cohort_meta.json')}")


if __name__ == "__main__":
    main()
