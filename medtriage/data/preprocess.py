"""
Data preprocessing for MedTriage-LM on MIMIC-IV-ED style tables.

The paper trains/evaluates on the MIMIC-IV-Ext Triage Instruction Corpus
(MIETIC), whose inputs are *strictly constrained to triage-time variables*
(structured haemodynamics + unstructured chief complaints) and whose targets
are the three Instruction Types {Life-saving, High-Risk Assessment,
Resource Estimation} plus a secondary critical-outcome (ICU-admission) flag.

The public demo cohort shipped with this repo does not contain the MIETIC
instruction labels, so we *derive* faithful surrogates directly from the
Emergency Severity Index (ESI) acuity, which is exactly what those three
instruction classes operationalise in the ESI algorithm:

    ESI 1  -> requires immediate life-saving intervention   -> "Life-saving"
    ESI 2  -> high-risk situation / cannot wait              -> "High-Risk Assessment"
    ESI 3+ -> stratified by anticipated resource needs       -> "Resource Estimation"

The critical-outcome surrogate (ICU-admission proxy) is configurable; the
default flags high-acuity admissions/transfers.

Leakage control
---------------
Only information available *at triage* is used as model input:
  * triage.csv vital signs + numeric pain                      (triage-time)
  * edstays.csv demographics (gender, arrival transport)       (presentation)
  * medrecon.csv home-medication reconciliation counts         (pre-existing)
Explicitly EXCLUDED from the feature set (post-triage / label leakage):
  * acuity            -> used only to build the label
  * disposition       -> used only to build the critical-outcome label
  * pyxis.csv         -> ED medications given *after* triage
  * vitalsign.csv     -> repeated charting throughout the stay
These excluded tables are still loaded and returned for analysis/EDA.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .anatomy import NUM_REGIONS, REGION_NAMES, complaint_to_weak_labels


# Triage instruction classes (primary 3-class target).
INSTRUCTION_CLASSES: Tuple[str, ...] = (
    "Life-saving",          # 0
    "High-Risk Assessment", # 1
    "Resource Estimation",  # 2
)

NUMERIC_FEATURES: Tuple[str, ...] = (
    "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain_num",
    "n_home_meds",
)
CATEGORICAL_FEATURES: Tuple[str, ...] = ("gender", "arrival_transport")

# Physiologically plausible clipping ranges to tame data-entry errors
# (e.g. the dbp = 879 outlier observed in the demo cohort).
_CLIP = {
    "temperature": (85.0, 110.0),   # Fahrenheit (as stored in MIMIC-IV-ED)
    "heartrate": (20.0, 250.0),
    "resprate": (3.0, 60.0),
    "o2sat": (50.0, 100.0),
    "sbp": (40.0, 300.0),
    "dbp": (20.0, 200.0),
}


@dataclass
class PreprocessConfig:
    data_dir: str = "data"
    critical_rule: str = "high_acuity_admit"  # see _critical_outcome
    drop_unlabeled_acuity: bool = True        # rows w/o acuity cannot get a class
    text_field: str = "chiefcomplaint"
    random_state: int = 42


@dataclass
class Cohort:
    """Container holding the fully assembled, model-ready cohort."""
    df: pd.DataFrame                              # one row per ED stay
    numeric_cols: List[str]
    categorical_cols: List[str]
    cat_vocab: Dict[str, List[str]]              # category -> ordered level list
    weak_label_cols: List[str]                   # K columns y_weak_<region>
    raw_tables: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.df)


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #


def _parse_pain(value) -> float:
    """Parse the free-text ``pain`` column into a 0-10 numeric score.

    Handles values seen in the demo cohort: integers, out-of-range ints
    (capped at 10), the literal ``"Critical"`` (-> 10), and non-numeric tokens
    such as ``"unable"/"UA"/"uta"/"ett"/"o"`` (-> NaN).
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).strip().lower()
    if s in {"critical"}:
        return 10.0
    m = re.match(r"^-?\d+(\.\d+)?$", s)
    if m:
        v = float(s)
        if v < 0:
            return np.nan
        return float(min(v, 10.0))
    return np.nan


def _critical_outcome(row, rule: str) -> int:
    """Binary critical-outcome surrogate for the ICU-admission auxiliary task.

    ``high_acuity_admit`` (default): a high-acuity (ESI<=2) encounter that was
    admitted or transferred -- a clinically sensible ICU-likely proxy when no
    explicit ICU flag is available.
    """
    disp = str(row.get("disposition", "")).upper()
    acuity = row.get("acuity", np.nan)
    admitted = disp in {"ADMITTED", "TRANSFER"}
    if rule == "high_acuity_admit":
        return int(admitted and (not np.isnan(acuity)) and acuity <= 2)
    if rule == "esi1":
        return int((not np.isnan(acuity)) and acuity == 1)
    if rule == "admitted":
        return int(admitted)
    raise ValueError(f"Unknown critical_rule: {rule!r}")


def _instruction_from_acuity(acuity: float) -> Optional[int]:
    if acuity is None or np.isnan(acuity):
        return None
    a = int(round(acuity))
    if a <= 1:
        return 0   # Life-saving
    if a == 2:
        return 1   # High-Risk Assessment
    return 2       # Resource Estimation


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def load_raw_tables(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Load the six MIMIC-IV-ED demo CSVs into a dict of DataFrames."""
    names = ["triage", "edstays", "diagnosis", "vitalsign", "pyxis", "medrecon"]
    tables: Dict[str, pd.DataFrame] = {}
    for n in names:
        path = os.path.join(data_dir, f"{n}.csv")
        if os.path.exists(path):
            tables[n] = pd.read_csv(path)
    if "triage" not in tables or "edstays" not in tables:
        raise FileNotFoundError(
            f"Required tables triage.csv/edstays.csv not found under {data_dir!r}. "
            f"Found: {sorted(tables)}"
        )
    return tables


def build_cohort(cfg: PreprocessConfig) -> Cohort:
    """Assemble the leakage-free, model-ready cohort described in the module docstring."""
    tables = load_raw_tables(cfg.data_dir)
    triage = tables["triage"].copy()
    edstays = tables["edstays"].copy()

    # --- merge stay-level demographics & disposition -----------------------
    keep_ed = ["subject_id", "stay_id", "gender", "race",
               "arrival_transport", "disposition"]
    df = triage.merge(edstays[keep_ed], on=["subject_id", "stay_id"], how="left")

    # --- numeric vitals: clip implausible values ---------------------------
    for col, (lo, hi) in _CLIP.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lo, hi)

    # --- pain -> numeric ---------------------------------------------------
    df["pain_num"] = df["pain"].apply(_parse_pain)

    # --- home-medication count (pre-existing comorbidity proxy) ------------
    if "medrecon" in tables:
        mr = tables["medrecon"]
        # de-duplicate (subject, stay, drug name) before counting
        n_meds = (mr.drop_duplicates(["stay_id", "name"])
                    .groupby("stay_id").size().rename("n_home_meds"))
        df = df.merge(n_meds, on="stay_id", how="left")
    df["n_home_meds"] = df.get("n_home_meds", pd.Series(0, index=df.index)).fillna(0.0)

    # --- categorical tidy-up ----------------------------------------------
    df["gender"] = df["gender"].fillna("UNKNOWN").astype(str).str.upper()
    df["arrival_transport"] = (df["arrival_transport"].fillna("UNKNOWN")
                               .astype(str).str.upper())

    # --- text field --------------------------------------------------------
    df["text"] = df[cfg.text_field].fillna("").astype(str)

    # --- labels ------------------------------------------------------------
    df["instruction"] = df["acuity"].apply(_instruction_from_acuity)
    df["critical"] = df.apply(lambda r: _critical_outcome(r, cfg.critical_rule), axis=1)

    if cfg.drop_unlabeled_acuity:
        before = len(df)
        df = df[df["instruction"].notna()].copy()
        dropped = before - len(df)
        if dropped:
            print(f"[preprocess] dropped {dropped} stays with missing acuity "
                  f"(no derivable instruction label).")
    df["instruction"] = df["instruction"].astype(int)

    # --- weak anatomical labels y_weak in {0,1}^K --------------------------
    weak = np.stack([complaint_to_weak_labels(t) for t in df["text"]], axis=0)
    weak_cols = [f"weak_{name}" for name in REGION_NAMES]
    for j, c in enumerate(weak_cols):
        df[c] = weak[:, j].astype(np.float32)

    # --- categorical vocabularies (UNKNOWN reserved at index 0) -----------
    cat_vocab: Dict[str, List[str]] = {}
    for c in CATEGORICAL_FEATURES:
        levels = sorted(x for x in df[c].unique() if x != "UNKNOWN")
        cat_vocab[c] = ["UNKNOWN"] + levels

    df = df.reset_index(drop=True)
    return Cohort(
        df=df,
        numeric_cols=list(NUMERIC_FEATURES),
        categorical_cols=list(CATEGORICAL_FEATURES),
        cat_vocab=cat_vocab,
        weak_label_cols=weak_cols,
        raw_tables=tables,
    )


def cohort_summary(cohort: Cohort) -> str:
    """Human-readable summary string (printed by scripts/01_preprocess.py)."""
    df = cohort.df
    lines = [f"Cohort: {len(df)} ED stays | {df['subject_id'].nunique()} unique patients"]
    inst = df["instruction"].value_counts().sort_index()
    lines.append("Instruction class distribution:")
    for k, name in enumerate(INSTRUCTION_CLASSES):
        lines.append(f"    [{k}] {name:22s}: {int(inst.get(k, 0)):4d}")
    pos = int(df["critical"].sum())
    lines.append(f"Critical-outcome positives: {pos}/{len(df)} "
                 f"({100*pos/max(len(df),1):.1f}%)")
    wk = df[cohort.weak_label_cols].mean().sort_values(ascending=False)
    lines.append("Weak anatomical-region prevalence (top 6):")
    for c, v in wk.head(6).items():
        lines.append(f"    {c.replace('weak_',''):10s}: {v:.3f}")
    miss = df[cohort.numeric_cols].isna().mean().mean()
    lines.append(f"Mean numeric-feature missingness: {100*miss:.1f}%")
    return "\n".join(lines)


def save_cohort(cohort: Cohort, out_dir: str) -> str:
    """Persist the assembled cohort + metadata to ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "cohort.csv")
    cohort.df.to_csv(csv_path, index=False)
    meta = {
        "numeric_cols": cohort.numeric_cols,
        "categorical_cols": cohort.categorical_cols,
        "cat_vocab": cohort.cat_vocab,
        "weak_label_cols": cohort.weak_label_cols,
        "instruction_classes": list(INSTRUCTION_CLASSES),
        "region_names": list(REGION_NAMES),
        "num_regions": NUM_REGIONS,
        "n_rows": int(len(cohort.df)),
    }
    with open(os.path.join(out_dir, "cohort_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return csv_path
