# Coral Reef Early-Damage Detection

Detects two damage patterns on raw underwater coral photos — algae
overgrowth and bleaching — and outputs box/mask-level findings per
photo. See `plans` (or ask for the original plan file) for the full
design rationale; this README covers setup and how to run each stage.

A disease detector was originally planned as a third damage type, meant
to be fine-tuned on an external Roboflow dataset. That dataset turned
out to only have coral genus/species labels (Acropora, Porites,
Sarcophyton, ...), not disease labels, and no other labeled disease
source was available, so **disease detection was dropped from scope**.
`src/models/disease_detector.py` and `src/data/prep_disease_external.py`
no longer exist in this repo for that reason - if disease detection
becomes worth revisiting, it needs an actual disease-labeled dataset
sourced first.

## Why the pipeline is shaped the way it is

- **No damage labels exist in the source data.** `coral_soft` (bbox,
  healthy coral genus photos) and `archive-2` (sparse point labels,
  substrate/genus composition) don't contain a single bleaching or
  damage-region annotation. Each remaining damage type is bootstrapped
  differently:
  - **Algae** — grown automatically from `archive-2`'s algae/macroalgae/
    green_fleshy_algae points into weak masks (`grow_algae_masks.py`).
    CCA and turf are deliberately excluded (not damage indicators).
  - **Bleaching** — not a trained model at all. It's a deterministic
    color-space calculation: how far has a detected colony's color moved
    from its own genus's normal appearance toward white. Calibrated
    against a public Kaggle healthy/bleached set (923 images).
- **Preprocessing lives inside the exported model**, not a separate
  script, because the deployment target is a website/mobile app with no
  Python backend. Concretely: underwater color-cast correction
  (gray-world white balance) and normalization are real tensor ops
  embedded in every exported `.onnx` graph. Resizing an arbitrary photo
  to the network's fixed input size happens once, client-side, before
  the tensor is built — this is a real ONNX-tracing constraint (dynamic
  resize doesn't trace into a portable static graph), not a shortcut,
  and it's exactly how every deployed vision model handles arbitrary
  input resolutions.
- **Each stage exports as its own `.onnx` file**, not one monolithic
  graph for the whole pipeline. Running colony detection, then cropping
  to each detected box, then running algae/bleaching checks on that
  crop, is Python-level control flow that doesn't trace into a single
  static graph. A thin orchestration layer (the equivalent of
  `CoralDamagePipeline` in `src/pipeline/coral_damage_model.py`, ~50
  lines) needs to be reimplemented in JS/Swift/Kotlin at the app layer
  to sequence calls between the three `.onnx` files. Each individual
  file's internal preprocessing is still fully embedded — this only
  affects cross-stage orchestration, not the preprocessing requirement
  itself.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Apple Silicon (M1/M2/...) trains via PyTorch's MPS backend automatically
(`--device mps`); no CUDA GPU is required or used.

## Pipeline, in order

```bash
# 1. Data prep
python3 src/data/coral_soft_to_yolo.py
python3 src/data/grow_algae_masks.py
python3 src/data/prep_bleaching_reference.py
# the last one needs the Kaggle "Healthy and Bleached Corals Image
# Classification" set placed (or symlinked) into data/external/kaggle_bleaching/
# - it prints setup instructions and skips cleanly if that's missing.

# 2. Train (nano-sized models; increase --epochs for real training runs,
#    the numbers below are just enough to prove the mechanics work)
PYTHONPATH=src python3 src/models/colony_detector.py --epochs 30 --device mps
PYTHONPATH=src python3 src/models/algae_segmenter.py --epochs 30 --device mps

# 3. Evaluate everything that's been trained so far
PYTHONPATH=src python3 src/eval/evaluate_all.py

# 4. Export to ONNX (skips any stage that isn't trained yet)
PYTHONPATH=src python3 src/export/export_onnx.py

# 5. Demo end-to-end on a raw photo
PYTHONPATH=src python3 src/inference/run_demo.py path/to/photo.jpg
# proves preprocessing is embedded in the exported graph (onnxruntime only, no manual color-correction):
PYTHONPATH=src python3 src/inference/run_demo.py path/to/photo.jpg --mode onnx_proof --stage colony_detector
```

## Scope note

This build produces a working, runnable pipeline that trains and exports
correctly end-to-end — not fully-converged production-quality models.
Running the training commands above with realistic epoch counts
(potentially hours on this CPU/MPS-only machine) is a follow-up step,
not something already done here.

## Disk note

Only ~8-10GB was free on this machine when this was built. `data/processed/`
re-encodes `coral_soft` (~1.1GB) but symlinks `archive-2` and the Kaggle
bleaching set rather than copying them. Keep an eye on `df -h` before
increasing epoch counts or image sizes materially.
