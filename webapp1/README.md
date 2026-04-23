# SEGP Webapp Mission Guide

This README covers only what exists in the `webapp/` folder.

## 1) Folder Radar (What Is Inside)

| Path | What it does | When you use it |
|---|---|---|
| `backend.ts` | Express + TypeScript API gateway for uploads and proxying model calls | You want local API endpoints |
| `frontend/main.html` | Main UI for image analysis and model comparison | You want to use or tweak the UI |
| `frontend/main.css` | Frontend styling | You want to change visuals |
| `frontend/about.html`, `frontend/theme.html` | Extra UI pages | You want content/theme pages |
| `package.json` | Backend scripts and dependencies | You run backend commands |
| `frontend/package.json` | Frontend local server script | You run frontend locally |
| `main.js` | Camera capture helper script | You test camera capture logic |
| `convert_ckpt.py` | Utility script for checkpoint conversion | You need local conversion tooling |
| `js/` | Additional JS workspace | You are working on alternate JS setup |
| `QUICKSTART.txt`, `STATUS.md` | Supporting notes | You want quick status/context |

## 2) Pick Your Mode (Interactive)

Check one path first:

- [ ] **Path A - Viewer mode**: I only want to open and use the app.
- [ ] **Path B - Public URL mode**: I want my backend URL reachable from outside my machine.
- [ ] **Path C - Run locally**: I want to run both the frontend and backend on my own machine.

---

## Path A - Viewer Mode (No Local Setup)

Open this published page:

```text
https://chamtgm.github.io/SEGP-RunService/main.html
```

Quick self-check:

- [ ] Page loads
- [ ] I can pick/upload a photo
- [ ] UI cards update after processing

---

## Path B - Public URL Mode (Optional ngrok)

Use this only if you want to expose your local backend publicly.

Install ngrok (Windows):

```powershell
winget install ngrok.ngrok
```

or

```powershell
choco install ngrok
```

Set your auth token once:

```powershell
ngrok config add-authtoken <YOUR_NGROK_AUTHTOKEN>
```

Tunnel example:

```powershell
ngrok http 3000
```

Then place that public URL into `backend1` and/or `backend2` inside `frontend/main.html`.

---

## Path C - Run Locally

Use this when you want everything running on your own machine with no external dependencies.

### Prerequisites

- Node.js installed (check: `node -v`)
- Python model service already running on `http://127.0.0.1:8001` (the backend proxies to it)

### Step 1 — Install dependencies

From the `webapp/` root:

```powershell
npm install
```

Then for the frontend:

```powershell
cd frontend
npm install
cd ..
```

### Step 2 — Start the frontend

Then run in terminal:

```powershell
cd frontend
npm start
```

This serves the frontend on **http://127.0.0.1:8080** and opens `main.html` in your browser automatically.

### Step 3 — Open the app

```text
http://127.0.0.1:8080/main.html
```

Quick self-check:

- [ ] Backend terminal shows `Backend listening on http://localhost:3000`
- [ ] Browser opens to `main.html`
- [ ] I can upload a photo and see results

---
