# Object Detection Guide

This README applies to `SEGP-main/Object Detection`.

## Folder Purpose

This folder contains the end-to-end YOLO workflow:

- automatic label generation (`auto_annotate_test.py`)
- YOLO dataset preparation (`prepare_dataset.py`)
- YOLO training (`train_yolo.py`)

## Main Files

- `auto_annotate_test.py`
- `prepare_dataset.py`
- `train_yolo.py`
- `download_roboflow.ipynb`

## Important About Paths

The three Python scripts in this folder currently use hardcoded path variables.
Before running, update these variables to match your machine:

- `auto_annotate_test.py`
	- `input_dir`
	- `output_preview_dir`
	- `output_labels_dir`
- `prepare_dataset.py`
	- `input_images_dir`
	- `input_labels_dir`
	- `output_dataset_dir`
- `train_yolo.py`
	- `dataset_yaml`
	- `project_dir`

If these paths are not updated, the scripts will fail on a different computer.

## Recommended Run Order

From repository root:

```powershell
cd "Object Detection"
```

1. Generate labels:

```powershell
python auto_annotate_test.py
```

2. Build YOLO dataset and `data.yaml`:

```powershell
python prepare_dataset.py
```

3. Train YOLO model:

```powershell
python train_yolo.py
```

## Required Packages

Install required dependencies in your Python environment:

```powershell
pip install ultralytics opencv-python rembg pyyaml
```

## Expected Outputs

- auto annotation previews in the configured preview output directory
- YOLO labels in the configured labels output directory
- prepared dataset with `yolo_dataset/data.yaml`
- training outputs under configured `project_dir` (including `weights/best.pt`)

## Troubleshooting

- `ModuleNotFoundError`:
	install missing package(s) in the active environment.
- `FileNotFoundError`:
	verify all hardcoded paths in the script configs.
- no detections during auto-annotation:
	verify input image folders and class folder names.
    
