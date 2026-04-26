# SEGP Project Startup Guide

This guide matches the current `scripts/python_model_service.py` behavior.

## 🚦 Choose Your Path

- ✅ I only want to use the app now: go to **Path A**
- 🌍 I want public access to my own local backend: go to **Path B (Optional)**

---

## Path A (Part 1): Use The App Now (No Setup)

Open this link and use the program directly:

```text
https://chamtgm.github.io/SEGP-RunService/main.html
```
or do local frontend startup
```text
Run "npm start" in webapp1/frontend directory
```

No local frontend deployment is required.

---

## Path A (Part 2): Run Backend Locally

### Prerequisites

- Python 3 with virtual environment at `Code\.venv`
- Local model artifacts:
  - `fine_tuned_models\CFSIMCLR_finetuned_best.pt`
  - `fine_tuned_models\simCLR_finetuned_best.pt`
  - `fruit_centroids(cfsimclr).pt`
  - `fruit_centroids(simclr).pt`
- Optional but recommended: `Object Detection\runs\fruit_detector\weights\best.pt`

### Install Dependencies

```powershell
cd Code
.\.venv\Scripts\Activate.ps1
python -m pip install -r contrastive-fruits\requirements.txt
```

### Start Local Backends

Terminal 1:

```powershell
cd Code
.\.venv\Scripts\Activate.ps1
python scripts\python_model_service.py --ckpt "fine_tuned_models\CFSIMCLR_finetuned_best.pt" --gallery-root "Datasets - Copy" --centroids-path "fruit_centroids(cfsimclr).pt" --port 8001 --num-classes 10
```

Terminal 2:

```powershell
cd Code
.\.venv\Scripts\Activate.ps1
python scripts\python_model_service.py --ckpt "fine_tuned_models\simCLR_finetuned_best.pt" --gallery-root "Datasets - Copy" --centroids-path "fruit_centroids(simclr).pt" --port 8002 --num-classes 10
```

Quick checks:

```powershell
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8002/health
```

<details>
<summary>Advanced: Optional CLI flags</summary>

- `--device cpu` to force CPU
- `--host 127.0.0.1` to bind locally only
- `--classifier-path <path>` compatibility argument

</details>

---

## Path B: ngrok Tunnels Initialisation for Accessing Deployed Frontend (For Public Access To Local Backends)

Only needed if you want your own local backend to be reachable from the deployed frontend.

### Install ngrok (Windows)

```powershell
winget install ngrok.ngrok
```

or

```powershell
choco install ngrok
```

Verify:

```powershell
ngrok version
```

Set auth token:

```powershell
ngrok config add-authtoken <YOUR_NGROK_AUTHTOKEN>
ngrok config add-authtoken <YOUR_NGROK_AUTHTOKEN> --config ngrok2.yml
```

### Start tunnels

Terminal 3:

```powershell
ngrok http 8001
```

Terminal 4:

```powershell
ngrok http 8002 --config ngrok2.yml
```

Then:

1. Copy ngrok HTTPS forwarding URLs.
2. Update `backend1` and `backend2` in `webapp/frontend/main.html`.
3. Redeploy frontend only if URLs changed and you maintain that deployment.

---

## 📱 Mobile Access

- Using deployed app: open the same link from **Path A**.
- Using a local frontend server instead: `http://<YOUR_IPV4_ADDRESS>:8080/main.html`.

---

## 🔌 API Quick Reference

- `GET /health`: service status, device, checkpoint, t-SNE availability
- `POST /embed`: image bytes; optional `include_raw=1`
- `POST /predict`: image bytes; optional `max_detections`
- `POST /nn`: image bytes; supports `k`, `all_detections`, t-SNE/UMAP query params
- `POST /heatmap`: image bytes; optional `cv`, `colormap`, `alpha`, `labels`
- `POST /upload-form`: form-data field `photo`
- `POST /reload`: JSON body `{"ckpt":"<path>"}`

---

## 🛠️ Troubleshooting

- Port already in use
  - `Get-Process python | Stop-Process -Force`

- Model not loaded or prediction errors
  - Start with `--ckpt` and verify file path from `Code`.

- Centroids warning
  - Verify `--centroids-path` exists and matches the model.

- YOLO model missing
  - Check `Object Detection\runs\fruit_detector\weights\best.pt`.
  - Service still runs with full-image fallback.

- Slow first response
  - Expected while gallery embeddings build in background.

- t-SNE info missing
  - Ensure `scikit-learn` is installed and `/health` shows `have_tsne: true`.

- ngrok issues
  - Install only if using **Path C**.
  - Re-run auth token commands if tunnel start fails.

- Frontend calling wrong backend
  - Recheck `backend1`/`backend2` in `webapp/frontend/main.html`.
