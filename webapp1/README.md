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
