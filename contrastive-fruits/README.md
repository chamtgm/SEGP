# Contrastive Fruits (CF-SimCLR)

This README applies to the folder `SEGP-main/contrastive-fruits`.

## Overview

This module contains the self-supervised contrastive pipeline and downstream evaluation code:

- HVAE style model training (`train_hvae.py`)
- Contrastive backbone training (`train.py`)
- Linear probing (`linear_probe.py`)
- Fine-tuning (`fine_tune.py`)
- Probe evaluation and plots (`eval_probe.py`, `plot_confusion_matrix.py`)
- Centroid generation (`generate_centroids.py`)

## Quick Setup

From repository root:

```powershell
cd contrastive-fruits
pip install -r requirements.txt
```

## Required Data Paths

Most scripts require dataset paths that are not fixed in this repo. Use your own local paths for:

- fruit dataset root (`--fruit-root`)
- style dataset root (`--style-root`)
- optional train subset (`--train-root`)

## Typical Workflow

Run all commands below from `SEGP-main/contrastive-fruits`.

### 1) Train HVAE (optional but required for `--style-method hvae`)

```powershell
python train_hvae.py --content-root "<path-to-content-images>" --style-root "<path-to-style-images>" --epochs 20 --batch-size 16 --save-dir "hvae_checkpoints" --device cpu
```

Output example: `hvae_checkpoints/hvae_epoch_20.pt`

### 2) Train Contrastive Backbone

```powershell
python train.py --fruit-root "<path-to-fruit-dataset>" --style-root "<path-to-style-images>" --style-method hvae --hvae-ckpt "hvae_checkpoints/hvae_epoch_20.pt" --epochs 100 --batch-size 32 --save-dir "cfsimclr_checkpoints" --device cpu
```

Output example: `cfsimclr_checkpoints/ckpt_epoch_100.pt`

### 3) Train Linear Probe

`--ckpt-dir` can be a checkpoint directory or a single checkpoint file.

```powershell
python linear_probe.py --fruit-root "<path-to-fruit-dataset>" --ckpt-dir "cfsimclr_checkpoints" --epochs 20 --batch-size 64 --lr 0.001 --device cpu
```

Output: `linear_probe_best.pt` (saved into the same location as `--ckpt-dir`)

### 4) Fine-Tune Classifier

```powershell
python fine_tune.py --fruit-root "<path-to-fruit-dataset>" --simclr-ckpt "cfsimclr_checkpoints/ckpt_epoch_100.pt" --probe-ckpt "cfsimclr_checkpoints/linear_probe_best.pt" --epochs 20 --device cpu --save-dir "finetune_checkpoints"
```

Output: `finetune_checkpoints/finetuned_best.pt`

### 5) Evaluate Probe

```powershell
python eval_probe.py --fruit-root "<path-to-fruit-dataset>" --ckpt-dir "cfsimclr_checkpoints" --device cpu
```

Outputs include:

- `confusion_matrix.png`
- `per_class_metrics.csv`

### 6) Generate Centroids

```powershell
python generate_centroids.py --train-root "<path-to-train-dataset>" --ckpt "cfsimclr_checkpoints/ckpt_epoch_100.pt" --out "fruit_centroids.pt" --device cpu
```

## Notes

- Replace `cpu` with `cuda` if CUDA is available.
- Keep paths quoted if they contain spaces.
- Run from this folder (`contrastive-fruits`) to avoid path issues.
