# Python Model Service Guide

This README applies to `SEGP-main/scripts/python_model_service.py`.

## What This Folder Contains

- `python_model_service.py`: Flask API for embeddings, prediction, nearest neighbors, and heatmaps.

## Start The Service
For the gallery-root parameter setting, use the "Fruit Images Training" dataset
Run from repository root (`SEGP-main`):

```powershell
python scripts/python_model_service.py --ckpt "<path-to-checkpoint.pt>" --gallery-root "<path-to-gallery-images>" --centroids-path "<path-to-centroids.pt>" --port 8001 --num-classes 10 --device cpu
```

Useful optional flags:

- `--host` (default: `0.0.0.0`)
- `--classifier-path`
- `--embedding-dim`
- `--hidden-dim`

If you need two backends (for side-by-side model comparison), run a second instance on another port:

```powershell
python scripts/python_model_service.py --ckpt "<path-to-second-checkpoint.pt>" --gallery-root "<path-to-gallery-images>" --centroids-path "<path-to-second-centroids.pt>" --port 8002 --num-classes 10 --device cpu
```

## Health Checks

```powershell
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8002/health
```

## API Endpoints

- `GET /health`
- `POST /embed`
- `POST /predict`
- `POST /nn`
- `POST /heatmap`
- `POST /upload-form`
- `POST /reload`

## Frontend Integration

The UI in `webapp1/frontend/main.html` uses `backend1` and `backend2` constants.
Set those URLs to your running service instances, for example:

- `http://127.0.0.1:8001`
- `http://127.0.0.1:8002`

## Optional Public Access (ngrok) if Using Deployed Frontend

If you want public URLs for local ports:

```powershell
ngrok http 8001
ngrok http 8002
```

Then place the generated HTTPS URLs into `backend1` and `backend2` in `webapp1/frontend/main.html`.

## Troubleshooting

- Port already in use:
  stop existing Python processes or use a different `--port`.
- Checkpoint not found:
  verify the `--ckpt` path.
- Missing YOLO weights:
  the service falls back to full-image behavior if `Object Detection/runs/fruit_detector/weights/best.pt` is not present.
- t-SNE unavailable:
  install `scikit-learn` and recheck `/health`.
