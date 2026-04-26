# SEGP Webapp1 Guide

This README applies to `SEGP-main/webapp1`.

## Folder Contents

| Path | Purpose |
|---|---|
| `backend.ts` | Express + TypeScript gateway service |
| `frontend/main.html` | Main UI for analysis and model comparison |
| `frontend/main.css` | UI styling |
| `frontend/about.html`, `frontend/theme.html`, `frontend/index.html` | Additional frontend pages |
| `package.json` | Backend scripts and dependencies |
| `frontend/package.json` | Frontend local server script |
| `js/` | Alternate JS helper workspace |
| `QUICKSTART.txt`, `STATUS.md` | Extra notes |

## Run Modes

### A) Use Hosted Frontend

Open:

```text
https://chamtgm.github.io/SEGP-RunService/main.html
```

### B) Run Frontend Locally

From repository root:

```powershell
cd webapp1/frontend
npm install
npm start
```

Local URL:

```text
http://127.0.0.1:8080/main.html
```

### C) Run Local Model Backends For Frontend

The frontend reads backend URLs from constants in `webapp1/frontend/main.html`:

- `backend1`
- `backend2`

Set these to your local service URLs (for example `http://127.0.0.1:8001` and `http://127.0.0.1:8002`), then start the Python services from repository root:

```powershell
python scripts/python_model_service.py --ckpt "<path-to-model-1.pt>" --gallery-root "<path-to-gallery>" --centroids-path "<path-to-centroids-1.pt>" --port 8001 --num-classes 10 --device cpu
```

Optional second backend:

```powershell
python scripts/python_model_service.py --ckpt "<path-to-model-2.pt>" --gallery-root "<path-to-gallery>" --centroids-path "<path-to-centroids-2.pt>" --port 8002 --num-classes 10 --device cpu
```

### D) Optional Node Gateway (`backend.ts`)

From repository root:

```powershell
cd webapp1
npm install
npm start
```

Default API server URL:

```text
http://localhost:3000
```

`backend.ts` forwards model calls to `PYTHON_BACKEND_URL` (defaults to `http://127.0.0.1:8001`).

### Optional Public Access (ngrok)

If you need public URLs for local backends:

```powershell
ngrok http 8001
ngrok http 8002
```

Then replace `backend1` and `backend2` in `webapp1/frontend/main.html` with the ngrok HTTPS URLs.
