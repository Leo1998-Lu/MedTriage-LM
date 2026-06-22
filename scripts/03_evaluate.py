#!/usr/bin/env python3
"""Step 03 -- Summarise cross-validation results (paper Table 1 style).

Reads the ``cv_results.json`` produced by step 02 and prints / writes a
mean +/- 95% CI table for every metric.

Usage:
    python scripts/03_evaluate.py --results artifacts/cv_results.json \
        --out artifacts/results_table.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medtriage.metrics import format_aggregate

_PRETTY = {
    "accuracy": "Accuracy (instruction, 3-class)",
    "macro_f1": "Macro-F1 (instruction, 3-class)",
    "auroc": "AUROC (critical outcome)",
    "auprc": "AUPRC (critical outcome)",
    "bertscore_proxy_f1": "Rationale BERTScore-proxy F1",
    "bertscore_f1": "Rationale BERTScore F1",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="artifacts/cv_results.json")
    ap.add_argument("--out", default="artifacts/results_table.md")
    args = ap.parse_args()

    with open(args.results) as f:
        cv = json.load(f)
    agg = cv["aggregate"]
    n_folds = len(cv.get("per_fold", []))

    print("\nCross-validation summary (mean +/- 95% CI):\n")
    print(format_aggregate(agg))

    lines = [
        "# MedTriage-LM -- reproduction results",
        "",
        f"Stratified {n_folds}-fold cross-validation on the MIMIC-IV-ED demo "
        "cohort. Values are mean +/- 95% CI.",
        "",
        "| Metric | Mean | 95% CI | n |",
        "|---|---|---|---|",
    ]
    for key, stats in agg.items():
        name = _PRETTY.get(key, key)
        lines.append(f"| {name} | {stats['mean']:.4f} | "
                     f"+/-{stats['ci95']:.4f} | {stats['n']} |")
    lines += [
        "",
        "_Note: metrics come from lightweight from-scratch backbones running "
        "offline on a tiny demo cohort, so absolute numbers are not comparable "
        "to the paper's full-scale results; they demonstrate that the full "
        "pipeline trains and evaluates end-to-end._",
        "",
    ]
    # Accept either a file path or a directory for --out.
    out_path = args.out
    if os.path.isdir(out_path) or out_path.endswith(os.sep):
        out_path = os.path.join(out_path, "results_table.md")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved table -> {out_path}")


if __name__ == "__main__":
    main()
