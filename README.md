# MedTriage-LM

Reproduction of the
*"MedTriage-LM: Anatomically Grounded Visual Phenotype Synthesis for
Interpretable ED Triage."*

The paper proposes an interpretable emergency-department (ED) triage model that
**synthesizes "Visual Phenotype Maps" (VPMs) from tabular + text clinical data**. It injects an anatomical visual prior *without* needing any real patient
photographs and fuses that synthetic visual modality with the clinical state
through cross-attention to predict a 3-class triage instruction, a binary
critical-outcome flag, and a free-text rationale.


> **Read this first — scope of the reproduction.**
> Due to strict credentialing and data privacy requirements, the complete MIMIC-IV-ED dataset must be acquired independently by users. This repository provides an end-to-end workflow using the [MIMIC-IV-ED demo](https://physionet.org/content/mimic-iv-ed/) as an illustrative example.
> We (a) **derive** the 3-class instruction label from the ESI
> acuity column and (b) **synthesize** the visual modality exactly as the paper describes.
> The architecture, equations, losses and training loop are faithful;
> the **absolute metrics are not comparable to the paper** because the cohort is
> ~3 orders of magnitude smaller and the default backbones are randomly initialised. 
> The value here is a correct, inspectable, end-to-end implementation you can scale up.


---

## 1. What maps to what (paper → code)

The paper's four modules and ten equations map one-to-one onto the code:

| Paper component | Equation | Code |
|---|---|---|
| Clinical State Encoder — tabular (FT-Transformer) | h_s = f_FT(x_s) | `models/ft_transformer.py` |
| Clinical State Encoder — text (ClinicalBERT) | h_t = f_BERT(x_t)[CLS] | `models/text_encoder.py` |
| Fusion layer → latent clinical state z | Eq. 1 | `models/region_mapper.py::FusionLayer` |
| Anatomical Region Mapper → region intensities μ | Eq. 2 | `models/region_mapper.py::AnatomicalRegionMapper` |
| Weak supervision via chief-complaint mapping | Eq. 3 | `data/anatomy.py` (rules) + `losses.py::weak_supervision_loss` |
| Gaussian Field Synthesis (raw heat field H) | Eq. 4 | `models/field_generator.py::synthesize_field` |
| Spatial smoothness regularisation | Eq. 5 | `models/field_generator.py::smoothness_loss` / `losses.py` |
| Heatmap rendering → VPM (colormap + alpha-blend) | Eq. 6 | `models/field_generator.py::render` |
| Vision encoder over the VPM | V = ViT(I_VPM) | `models/vision_encoder.py` |
| Cross-modal vision-language alignment | Eq. 7 | `models/cross_attention.py` |
| Rationale generation (causal LM) | Eq. 8 | `models/rationale.py` |
| Prediction heads (instruction + critical) | Eq. 9 | `models/heads.py` |
| Joint multi-task objective | Eq. 10 | `losses.py::MedTriageLoss` |
| Full assembled model | Fig. 1 | `models/medtriage_lm.py::MedTriageLM` |


### The differentiable-VPM detail
Matplotlib's colormap is not differentiable, so a real colormap would cut the
gradient path from the prediction heads back to the region mapper. To keep the
**whole** model trainable end-to-end, the in-graph render uses a smooth
"pseudo-jet" colormap (`field_generator.py::_pseudo_jet`); the true matplotlib
jet render is used only in `viz.py` for human-readable figures. A unit test
(`tests/test_pipeline.py`) asserts gradients actually reach the region mapper
through the visual branch, proving the synthetic visual modality is learned, not
bolted on.

### The anatomical body template
The canonical human silhouette `I_canonical` (Eq. 6) is generated procedurally
in `data/anatomy.py::build_silhouette`, no external image asset. It is composed
from smooth primitives (an ellipsoidal head, a curved torso whose half-width
follows a shoulder→waist→hip profile, and tapered "swept-circle" capsules for the
neck, arms, hands, legs and feet), combined by an anti-aliased soft union with
super-sampling. The same builder serves both the low-res (64×48) ViT input and a
high-resolution figure base. For human-facing figures, `viz.py::render_thermal`
renders a reference-style thermographic VPM: the anatomical body filled with a
`jet` colormap over a dark navy background, with a warm baseline and Gaussian
smoothing so cold tissue reads blue and distress hotspots glow red.
`viz.py::render_template_showcase` (also emitted by `scripts/04`) saves a
standalone body-template + VPM showcase (`assets/template_showcase.png`).

---

## 2. Data handling

### Label derivation (instruction, 3-class)

| ESI acuity | Instruction class |
|---|---|
| 1 | `Life-saving` (0) |
| 2 | `High-Risk Assessment` (1) |
| ≥ 3 | `Resource Estimation` (2) |

Rows with missing acuity have no derivable label and are dropped (15 of 222).
Resulting class counts: **18 / 97 / 92** over **207** stays (56 unique patients).

### Critical-outcome label (binary, secondary task)
A configurable proxy for the paper's ICU-admission target
(`preprocess.critical_rule`, default `high_acuity_admit`): positive when the stay
was admitted/transferred **and** acuity ≤ 2. 95/207 (≈46%) are positive.

### Leakage control
Triage-time inputs only. Features = `triage.csv` vitals (temperature, heart rate,
resp. rate, SpO₂, SBP, DBP) + numeric pain + gender + arrival transport +
`n_home_meds` (count from `medrecon.csv`). **Excluded** on purpose: `acuity`
(it is the label source), `disposition` (label source), `pyxis.csv` (medications
given *after* triage), and the repeated `vitalsign.csv` charting. All six tables
are still loaded for EDA/inspection.


### Weak anatomical labels (for Eq. 3)
Chief complaints are matched with regex rules in `data/anatomy.py` to **K = 12**
canonical regions (head, face, neck, chest, abdomen, pelvis, left/right arm,
left/right leg, back, systemic). E.g. "chest pain" → chest; "SOB"/"fever" →
systemic. These weak labels supervise the region intensities μ.

---

## 3. Repository layout

```
medtriage-lm/
├── configs/
│   ├── default.yaml        # offline/CPU profile (lite backbones) + paper overrides (commented)
│   └── smoke.yaml          # fast config for CI / quick checks (2 folds, few epochs)
├── data/                   # bundled MIMIC-IV-ED demo CSVs (6 tables)
├── medtriage/
│   ├── data/
│   │   ├── anatomy.py      # K=12 regions, complaint→region rules, body silhouette
│   │   ├── preprocess.py   # cohort build, label derivation, leakage control
│   │   └── dataset.py      # tabular processor, text tokenizer, torch Dataset/collate
│   ├── models/
│   │   ├── ft_transformer.py   # tabular encoder (Eq. h_s)
│   │   ├── text_encoder.py     # lite / ClinicalBERT text encoder (Eq. h_t)
│   │   ├── region_mapper.py    # fusion (Eq. 1) + region mapper (Eq. 2)
│   │   ├── field_generator.py  # Gaussian field (Eq. 4), smoothness (Eq. 5), render (Eq. 6)
│   │   ├── vision_encoder.py   # lite / timm ViT over the VPM
│   │   ├── cross_attention.py  # cross-modal alignment (Eq. 7)
│   │   ├── rationale.py        # template rationale + lite causal decoder / Qwen wrapper (Eq. 8)
│   │   ├── heads.py            # instruction + critical heads (Eq. 9)
│   │   └── medtriage_lm.py     # full model assembly (Fig. 1)
│   ├── losses.py           # all loss terms + joint objective (Eq. 3,5,10)
│   ├── metrics.py          # Acc, Macro-F1, AUROC, AUPRC, BERTScore(+proxy), mean±95%CI
│   ├── trainer.py          # stratified k-fold CV, per-fold fit/eval, full-cohort train
│   ├── config.py           # YAML → dataclass configs (with key validation)
│   └── viz.py              # VPM / cross-attention figure rendering
├── scripts/
│   ├── 01_preprocess.py    # build & save the cohort
│   ├── 02_train.py         # cross-validate + train on full cohort (saves checkpoint)
│   ├── 03_evaluate.py      # turn CV results into a paper-style table
│   └── 04_visualize_vpm.py # render VPM + attention figures from a checkpoint
├── tests/test_pipeline.py  # 7 demo tests (data, differentiable VPM, forward, loss, metrics)
├── requirements.txt
└── README.md
```

---

## 4. Quickstart

```bash
pip install -r requirements.txt

# 1) build the cohort (writes artifacts/cohort.csv + cohort_meta.json)
PYTHONPATH=. python3 scripts/01_preprocess.py --config configs/default.yaml --out artifacts

# 2) 5-fold cross-validation + train on the full cohort (writes a checkpoint)
PYTHONPATH=. python3 scripts/02_train.py --config configs/default.yaml \
    --cohort artifacts/cohort.csv --out artifacts

# 3) render a paper-style results table from the CV results
PYTHONPATH=. python3 scripts/03_evaluate.py --results artifacts/cv_results.json --out artifacts

# 4) render Visual Phenotype Map + cross-attention figures (one per class)
PYTHONPATH=. python3 scripts/04_visualize_vpm.py --ckpt artifacts/medtriage_lm.pt \
    --cohort artifacts/cohort.csv --out assets --per-class 1

# run the demo tests
PYTHONPATH=. python3 tests/test_pipeline.py        # or: pytest tests/
```


---

## 5. Switching on the paper's full-scale backbones

`configs/default.yaml` ends with a commented "Full-scale (paper) overrides"
block. To reproduce the paper's real setting (needs a GPU + internet to download
weights), set:

```yaml
model:
  text_backbone: clinicalbert        # emilyalsentzer/Bio_ClinicalBERT
  vision_backbone: timm              # e.g. vit_base_patch16_224
  rationale_backend: qwen            # Qwen3-VL-8B prompt-based rationales
train:
  device: cuda
  use_real_bertscore: true           # real BERTScore instead of token-overlap proxy
```


---



