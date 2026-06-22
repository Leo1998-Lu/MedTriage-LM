"""MedTriage-LM: Anatomically Grounded Visual Phenotype Synthesis for Interpretable ED Triage.

A faithful, runnable reproduction of the MICCAI-2026 paper. Heavy pretrained
backbones (ClinicalBERT / ViT / Qwen3-VL-8B) are pluggable and default to
lightweight from-scratch equivalents so the full pipeline runs offline.
"""
__version__ = "1.0.0"
