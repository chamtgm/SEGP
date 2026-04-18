# Object Detection Folder Guide

This README documents the `Code/Object Detection` folder.

## Folder Purpose

This folder contains the object detection workflow for fruit images, including:
- auto-annotation artifacts and helpers
- YOLO dataset preparation assets
- detector training scripts and training outputs

## Main Contents

- `auto_annotate_test.py`: Generates label candidates from segmented foreground objects.
- `prepare_dataset.py`: Builds train/validation dataset structure and `data.yaml`.
- `train_yolo.py`: Trains the YOLO detector.
- `yolo_dataset/`: Prepared YOLO-format images, labels, and config.
- `runs/`: Training outputs and model weights.

## How To Run `train_yolo.py`

### Prerequisites

- Python environment with required packages installed (especially `ultralytics`).
- Dataset config file exists at `Code/Object Detection/yolo_dataset/data.yaml`.
- Internet access on first run may be needed to download `yolov8n.pt` if it is not already cached.

### 1) Check paths in `train_yolo.py`

The script currently uses fixed Windows paths:

- `dataset_yaml = D:\Study materials/Year 2/SEGP/Code/Object Detection/yolo_dataset/data.yaml`
- `project_dir = D:/Study materials/Year 2/SEGP/Code/Object Detection/runs`

If your project location is different, update these two variables before running.

### 2) Run the script

From inside this folder:

```powershell
cd "D:\Study materials\Year 2\SEGP\Code\Object Detection"
python train_yolo.py
```

Or from project root:

```powershell
cd "D:\Study materials\Year 2\SEGP"
python "Code\Object Detection\train_yolo.py"
```

### 3) What you should see

- Training starts with YOLOv8n and runs up to 100 epochs (with early stopping if applicable).
- Output is written under the folder set by `project_dir`, typically:
	- `Code/Object Detection/runs/fruit_detector*/`
- Model weights are saved under a `weights` subfolder (for example `best.pt` and `last.pt`).

### Quick Troubleshooting

- `ModuleNotFoundError: ultralytics`
	- Install dependencies in your Python environment, then run again.

- `FileNotFoundError` for `data.yaml`
	- Confirm `Code/Object Detection/yolo_dataset/data.yaml` exists and `dataset_yaml` points to it.
    
