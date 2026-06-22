"""
Anatomical priors for MedTriage-LM.

This module encodes the *deterministic* clinical knowledge used by the
"Anatomical Region Mapper" (Sec. 2.2) and the "Anatomical Field Generator"
(Sec. 2.3) of the paper:

  * ``REGIONS``                  -- K predefined clinical anatomical regions, each
                                    with a fixed 2-D coordinate ``c_k`` and base
                                    variance ``sigma_k`` on a canonical template
                                    (used in Eq. 4: the Gaussian field synthesis).
  * ``COMPLAINT_RULES``          -- the rule-based "Chief Complaint Mapping" that
                                    converts a free-text chief complaint into the
                                    weak anatomical labels ``y_weak in {0,1}^K``
                                    (used in Eq. 3: the weak-supervision loss).
  * ``build_silhouette``         -- programmatically renders the canonical human
                                    silhouette ``I_canonical`` used for alpha
                                    blending in Eq. 6 (no external image asset).

Everything here is intentionally transparent and editable -- it is the
interpretable, knowledge-injected part of the pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Canonical template geometry.
#
# Coordinates are expressed in a normalised [0, 1] x [0, 1] frame where
# (x=0.5, y=0.0) is the top of the head and (x=0.5, y=1.0) is between the feet.
# 'x' increases to the viewer's right; we use anatomical sidedness, so the
# patient's LEFT limbs sit on the viewer's right (x > 0.5).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Region:
    index: int
    name: str
    cx: float          # normalised x coordinate of c_k
    cy: float          # normalised y coordinate of c_k
    sigma: float       # normalised base std of the Gaussian blob


REGIONS: Tuple[Region, ...] = (
    Region(0,  "head",       0.500, 0.095, 0.055),
    Region(1,  "face",       0.500, 0.140, 0.040),
    Region(2,  "neck",       0.500, 0.185, 0.038),
    Region(3,  "chest",      0.500, 0.310, 0.080),
    Region(4,  "abdomen",    0.500, 0.450, 0.076),
    Region(5,  "pelvis",     0.500, 0.560, 0.058),
    Region(6,  "left_arm",   0.665, 0.400, 0.062),   # patient-left -> viewer right
    Region(7,  "right_arm",  0.335, 0.400, 0.062),
    Region(8,  "left_leg",   0.560, 0.790, 0.085),
    Region(9,  "right_leg",  0.440, 0.790, 0.085),
    Region(10, "back",       0.500, 0.370, 0.072),
    Region(11, "systemic",   0.500, 0.450, 0.210),   # diffuse / whole-body
)

NUM_REGIONS: int = len(REGIONS)
REGION_NAMES: Tuple[str, ...] = tuple(r.name for r in REGIONS)
REGION_INDEX: Dict[str, int] = {r.name: r.index for r in REGIONS}


# --------------------------------------------------------------------------- #
# Rule-based Chief Complaint Mapping  ->  weak anatomical labels y_weak.
#
# Each rule is (compiled_regex, [region_names]).  A complaint may fire several
# rules; the union of the matched regions becomes the multi-hot weak label.
# Patterns are matched against a lower-cased complaint string.
#
# The "systemic" region acts as the diffuse catch-all for metabolic / infectious
# / constitutional presentations and as the fallback when nothing else matches.
# --------------------------------------------------------------------------- #

_RAW_RULES: List[Tuple[str, List[str]]] = [
    # ---- Head / neuro -----------------------------------------------------
    (r"\bhead\b|head bleed|head injury|headache|\bich\b|intracerebral|"
     r"\bsah\b|subarachnoid|\bsdh\b|subdural|\bivh\b|cranial|"
     r"\bcva\b|stroke|tia\b|seizure|grand mal|status epilept|epileptic|"
     r"altered mental|\bams\b|confusion|letharg|encephalo|"
     r"dizziness|vertigo|presyncope|syncope|diplopia|hallucinat", ["head"]),

    # facial droop / numbness is BOTH a neuro sign (head) and facial (face)
    (r"facial droop|facial numb|face droop", ["head", "face"]),

    # ---- Face / eye / ear / jaw ------------------------------------------
    (r"\beye\b|eye pain|eye eval|eye swell|ocular|vision|periorbital", ["face"]),
    (r"\bear\b|ear pain|ear drain|otalgia", ["face"]),
    (r"\bjaw\b|jaw pain", ["face"]),
    (r"\bfacial\b|\bface\b|nasal|epistaxis", ["face"]),

    # ---- Neck / throat ----------------------------------------------------
    (r"\bneck\b|cervicalg|cervical|throat|laryngit|pharyng|"
     r"sob with talking|dysphagia", ["neck"]),

    # ---- Chest / cardiac / respiratory -----------------------------------
    (r"chest pain|\bchest\b|angina|\bnstemi\b|\bstemi\b|\bmi\b|"
     r"myocard|ischemia|coronary|palpitation|\bekg\b|\becg\b|"
     r"atrial fib|\bafib\b|\bchf\b|heart fail|cardiac|aortic dis|"
     r"\brib\b", ["chest"]),
    (r"dyspnea|\bsob\b|shortness of breath|short of breath|respiratory|"
     r"resp arrest|\bcough\b|hypox|\bpe\b|pulmonary|pneumon|wheez|"
     r"\bili\b|influenza", ["chest"]),

    # ---- Abdomen / GI -----------------------------------------------------
    (r"\babd\b|abdominal|epigastr|\bruq\b|\bluq\b|\brlq\b|\bllq\b|"
     r"stomach|gastro|\bgi\b|hematemes|vomiting blood|coffee ground|"
     r"\bbrbpr\b|melena|nausea|\bn/v\b|vomit|diarrhea|"
     r"distention|distension|ascites|abd tube|abd mass|hepat|"
     r"pancreat|biliary|cholecyst", ["abdomen"]),

    # ---- Pelvis / GU / rectal --------------------------------------------
    (r"urinary|\burine\b|hematuria|dysuria|suprapubic|foley|"
     r"retention|pelvic|\bgu\b|genital|bladder", ["pelvis"]),
    (r"rectal|\banal\b|perineal", ["pelvis"]),
    (r"lower abdominal", ["abdomen", "pelvis"]),

    # ---- Left upper limb --------------------------------------------------
    (r"\bl arm\b|left arm|\bl hand\b|left hand|\bl wrist\b|left wrist|"
     r"\bl shoulder\b|left shoulder|\bl elbow\b|left elbow|"
     r"\bl forearm\b", ["left_arm"]),

    # ---- Right upper limb -------------------------------------------------
    (r"\br arm\b|right arm|\br hand\b|right hand|\br wrist\b|right wrist|"
     r"\br shoulder\b|right shoulder|\br elbow\b|right elbow|"
     r"\br forearm\b", ["right_arm"]),

    # generic shoulder / arm / wrist / hand (no side) -> both arms
    (r"\bshoulder\b|\bwrist\b|\bforearm\b|\belbow\b", ["left_arm", "right_arm"]),

    # ---- Left lower limb --------------------------------------------------
    (r"\bl leg\b|left leg|\bl foot\b|left foot|\bl knee\b|left knee|"
     r"\bl hip\b|left hip|\bl ankle\b|\bl toe\b|left hip fx", ["left_leg"]),

    # ---- Right lower limb -------------------------------------------------
    (r"\br leg\b|right leg|\br foot\b|right foot|\br knee\b|right knee|"
     r"\br hip\b|right hip|\br ankle\b|\br toe\b|right foot", ["right_leg"]),

    # bilateral / unspecified lower limb -> both legs
    (r"\bb leg\b|bilateral leg|lower extremit|\bleg\b|\bfoot\b|"
     r"\bknee\b|\bankle\b|\btoe\b|\bhip\b|cellulit|\bdvt\b", ["left_leg", "right_leg"]),

    # ---- Back / spine -----------------------------------------------------
    (r"back pain|lower back|low back|lumbar|\bt-?spine\b|thoracic spine|"
     r"\bspine\b|spinal|sciatic", ["back"]),

    # ---- Systemic / constitutional / metabolic / psych --------------------
    (r"\bfever\b|febrile|sepsis|septic|infection|bacteremia|"
     r"fatigue|weakness|malaise|dehydrat|hypotension|hypotensive|"
     r"hypothermia|hyperthermia|"
     r"abnormal lab|abnormal ct|abnormal ekg|elevated inr|"
     r"hyperglycem|hypoglycem|\bdka\b|ketoacid|"
     r"\betoh\b|alcohol|intoxicat|overdose|withdrawal|"
     r"anemia|neutropenia|transfus|"
     r"allergic|anaphylax|\brash\b|"
     r"\bsi\b|suicid|psych|depression|anxiety|insomnia|"
     r"hemodialysis|\bpicc\b|catheter|med refill|"
     r"assault|\bmvc\b|trauma|\bfall\b|s/p fall|"
     r"\barrest\b|\bsw\b|stab|gunshot|\bgsw\b|"
     r"\bpe\b|wound|laceration", ["systemic"]),
]

COMPLAINT_RULES: List[Tuple[re.Pattern, List[int]]] = [
    (re.compile(pat, flags=re.IGNORECASE), [REGION_INDEX[n] for n in names])
    for pat, names in _RAW_RULES
]


def complaint_to_weak_labels(complaint: str) -> np.ndarray:
    """Map a free-text chief complaint to a multi-hot weak label y_weak in {0,1}^K.

    If no rule fires (e.g. ``"Transfer"`` or ``"Unknown-CC"``) the diffuse
    ``systemic`` region is set, so that the field synthesis always has at least
    one anchor and the weak-supervision loss remains well defined.
    """
    y = np.zeros(NUM_REGIONS, dtype=np.float32)
    if complaint is None:
        y[REGION_INDEX["systemic"]] = 1.0
        return y
    text = str(complaint).strip().lower()
    if not text:
        y[REGION_INDEX["systemic"]] = 1.0
        return y
    for pattern, region_ids in COMPLAINT_RULES:
        if pattern.search(text):
            for rid in region_ids:
                y[rid] = 1.0
    if y.sum() == 0:
        y[REGION_INDEX["systemic"]] = 1.0
    return y


# --------------------------------------------------------------------------- #
# Canonical human silhouette I_canonical (for Eq. 6 alpha blending + viz).
#
# The body is composed from smooth primitives -- an ellipsoidal head, a curved
# torso whose half-width follows a shoulder -> waist -> hip profile, and tapered
# "swept-circle" capsules for the neck, arms, hands, legs and feet -- combined by
# a soft (anti-aliased) union and rendered with super-sampling.  This yields an
# anatomically plausible, smooth anterior silhouette with no external image
# asset, suitable both as the low-res ViT input and as a high-res figure base.
# --------------------------------------------------------------------------- #


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _capsule_coverage(gx, gy, ax, ay, bx, by, r1, r2, aa):
    """Soft coverage of a *tapered* capsule (segment a->b with radius r1->r2).

    The radius is linearly interpolated along the segment's projection, giving
    a smooth cone-like limb. ``aa`` is the anti-aliasing width in pixels.
    """
    bax, bay = (bx - ax), (by - ay)
    l2 = bax * bax + bay * bay + 1e-9
    t = ((gx - ax) * bax + (gy - ay) * bay) / l2
    t = np.clip(t, 0.0, 1.0)
    px, py = ax + t * bax, ay + t * bay
    d = np.sqrt((gx - px) ** 2 + (gy - py) ** 2)
    r = r1 + t * (r2 - r1)
    return np.clip((r - d) / aa + 0.5, 0.0, 1.0)


def _ellipse_coverage(gx, gy, cx, cy, rx, ry, aa):
    """Soft coverage of a filled (optionally non-circular) ellipse."""
    nx, ny = (gx - cx) / rx, (gy - cy) / ry
    d = np.sqrt(nx * nx + ny * ny)                      # ~1 on the boundary
    scale = float(min(rx, ry))
    return np.clip((1.0 - d) * scale / aa + 0.5, 0.0, 1.0)


def _torso_coverage(gx, gy, Hs, Ws, aa):
    """Curved torso: per-row half-width following shoulders->waist->hips."""
    cx = 0.50 * Ws
    # control points: normalised y (top->bottom) and half-width
    ys = np.array([0.205, 0.255, 0.305, 0.360, 0.420, 0.475, 0.525, 0.585]) * Hs
    hw = np.array([0.140, 0.132, 0.122, 0.104, 0.092, 0.104, 0.130, 0.112]) * Ws
    hw_at = np.interp(gy.ravel(), ys, hw, left=hw[0], right=hw[-1]).reshape(gy.shape)
    horiz = np.clip((hw_at - np.abs(gx - cx)) / aa + 0.5, 0.0, 1.0)
    caps = (_smoothstep(0.205 * Hs - aa, 0.205 * Hs + 1.5 * aa, gy)
            * (1.0 - _smoothstep(0.585 * Hs - 1.5 * aa, 0.585 * Hs + aa, gy)))
    return horiz * caps


def build_silhouette(height: int = 64, width: int = 48,
                     supersample: int = 4) -> np.ndarray:
    """Render a canonical anterior human silhouette as a float mask in [0, 1].

    Returns an ``(height, width)`` array; ``1`` = body, ``0`` = background.
    The figure (head, neck, curved torso, tapered arms + hands, tapered legs +
    feet) is built from smooth primitives, super-sampled by ``supersample`` and
    area-averaged down for clean anti-aliased edges.
    """
    ss = max(1, int(supersample))
    Hs, Ws = height * ss, width * ss
    xs = np.arange(Ws, dtype=np.float32)
    ys = np.arange(Hs, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)                        # gx: x(cols), gy: y(rows)
    aa = max(1.0, 0.9 * ss)                             # AA width (super-px)

    cov = np.zeros((Hs, Ws), dtype=np.float32)

    def add(c):
        np.maximum(cov, c, out=cov)

    # ---- head + neck ----
    add(_ellipse_coverage(gx, gy, 0.500 * Ws, 0.095 * Hs,
                          0.056 * Ws, 0.072 * Hs, aa))
    add(_capsule_coverage(gx, gy, 0.500 * Ws, 0.150 * Hs,
                          0.500 * Ws, 0.210 * Hs, 0.036 * Ws, 0.050 * Ws, aa))

    # ---- torso (+ rounded shoulder cap) ----
    add(_torso_coverage(gx, gy, Hs, Ws, aa))
    add(_capsule_coverage(gx, gy, 0.372 * Ws, 0.230 * Hs,
                          0.628 * Ws, 0.230 * Hs, 0.058 * Ws, 0.058 * Ws, aa))

    # ---- arms (shoulder -> elbow -> wrist) + hands, both sides ----
    for s in (+1.0, -1.0):
        shx, shy = (0.500 + s * 0.128) * Ws, 0.238 * Hs   # shoulder
        elx, ely = (0.500 + s * 0.165) * Ws, 0.400 * Hs   # elbow
        wrx, wry = (0.500 + s * 0.188) * Ws, 0.555 * Hs   # wrist
        add(_capsule_coverage(gx, gy, shx, shy, elx, ely,
                              0.050 * Ws, 0.038 * Ws, aa))  # upper arm
        add(_capsule_coverage(gx, gy, elx, ely, wrx, wry,
                              0.038 * Ws, 0.028 * Ws, aa))  # forearm
        add(_ellipse_coverage(gx, gy, wrx, 0.578 * Hs,
                              0.034 * Ws, 0.046 * Hs, aa))  # hand

    # ---- legs (hip -> knee -> ankle) + feet, both sides ----
    for s in (+1.0, -1.0):
        hpx, hpy = (0.500 + s * 0.060) * Ws, 0.575 * Hs   # hip
        knx, kny = (0.500 + s * 0.072) * Ws, 0.745 * Hs   # knee
        anx, any_ = (0.500 + s * 0.076) * Ws, 0.930 * Hs  # ankle
        tox, toy = (0.500 + s * 0.090) * Ws, 0.968 * Hs   # toe
        add(_capsule_coverage(gx, gy, hpx, hpy, knx, kny,
                              0.073 * Ws, 0.046 * Ws, aa))  # thigh
        add(_capsule_coverage(gx, gy, knx, kny, anx, any_,
                              0.046 * Ws, 0.030 * Ws, aa))  # calf
        add(_capsule_coverage(gx, gy, anx, any_, tox, toy,
                              0.030 * Ws, 0.022 * Ws, aa))  # foot

    # area-average super-sampled coverage down to the requested resolution
    cov = cov.reshape(height, ss, width, ss).mean(axis=(1, 3))
    return np.clip(cov, 0.0, 1.0).astype(np.float32)


def region_pixel_coords(height: int, width: int):
    """Return arrays ``(rows, cols, sigmas_px)`` giving the pixel-space centre and
    std of every region for a template of the given size (used by Eq. 4)."""
    rows = np.array([r.cy * height for r in REGIONS], dtype=np.float32)
    cols = np.array([r.cx * width for r in REGIONS], dtype=np.float32)
    diag = float(np.sqrt(height * width))
    sig = np.array([max(r.sigma * diag, 1.0) for r in REGIONS], dtype=np.float32)
    return rows, cols, sig


if __name__ == "__main__":  # quick self-check
    tests = {
        "Chest pain, Jaw pain, L Arm pain": ["chest", "face", "left_arm"],
        "MVC/INTUBATED TRAUMA": ["systemic"],
        "R FOOT ULCER/CELLULITIS": ["right_leg", "left_leg"],
        "FACIAL DROOP": ["head", "face"],
        "Abd pain, Dysuria": ["abdomen", "pelvis"],
        "Transfer": ["systemic"],
        "Lower back pain, s/p Fall": ["back", "systemic"],
    }
    for cc, _ in tests.items():
        y = complaint_to_weak_labels(cc)
        fired = [REGION_NAMES[i] for i in np.where(y > 0)[0]]
        print(f"{cc!r:45s} -> {fired}")
    print(f"\nNUM_REGIONS = {NUM_REGIONS}")
    sil = build_silhouette()
    print(f"silhouette {sil.shape}, body fraction = {sil.mean():.3f}")
