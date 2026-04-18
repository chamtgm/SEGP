# Contrastive Fruits - Image Analysis Service

A Flask-based image analysis service with a web frontend that finds visually similar images using contrastive learning embeddings.

## ✨ Features

- 🖼️ **Image Upload** - Upload images via file picker or camera capture
- 🔍 **Similarity Search** - Finds k-nearest neighbors in a gallery
- 📊 **Visualization** - Displays confidence scores in bar chart
- 🎨 **Theme Support** - Light/Dark/Neon themes
- 📱 **Mobile Ready** - Camera capture on mobile devices
- ⚡ **Fast Processing** - GPU-accelerated embeddings (CPU fallback)

## 🛠️ Setup

### Prerequisites
- Windows 11
- Python 3.12+
- npm (for frontend http-server)

### Installation

1. **Create Python Virtual Environment**
```powershell
cd c:\Users\WINDOWS 11\Desktop\webapp
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. **Install Python Dependencies**
```powershell
pip install -r requirements.txt
# Or install individually:
pip install torch torchvision numpy pillow flask flask-cors scikit-learn opencv-python albumentations timm matplotlib
```

3. **Install Frontend Dependencies**
```powershell
cd frontend
npm install
cd ..
```

4. **Create Gallery** (optional, but recommended)
```powershell
mkdir gallery
# Add image files to gallery/
```

## 🚀 Quick Start

### Option 1: Automated Startup (Recommended)
```powershell
.\start_all.bat
```
This starts both services in separate windows:
- Backend: http://127.0.0.1:8001
- Frontend: http://127.0.0.1:8080/main.html

### Option 2: Manual Startup

**Terminal 1 - Start Backend:**
```powershell
cd c:\Users\WINDOWS 11\Desktop\webapp
.venv\Scripts\Activate.ps1
python scripts\scripts\python_model_service.py `
    --ckpt ckpt_epoch_1000\ckpt_epoch_1000.pt `
    --gallery-root gallery `
    --port 8001
```

**Terminal 2 - Start Frontend:**
```powershell
cd c:\Users\WINDOWS 11\Desktop\webapp\frontend
npm start
```

Then open: http://127.0.0.1:8080/main.html

## 📁 Project Structure

```
webapp/
├── backend.ts                          # TypeScript backend config
├── package.json                        # Root package config
├── tsconfig.json                       # TypeScript config
├── ckpt_epoch_1000/
│   └── ckpt_epoch_1000.pt             # PyTorch checkpoint (required)
├── scripts/
│   └── scripts/
│       └── python_model_service.py    # Flask backend service
├── frontend/
│   ├── main.html                      # Main interface
│   ├── main.css                       # Styling
│   ├── package.json                   # npm config
│   └── picture/                       # Sample images
├── gallery/                           # Image gallery for k-NN search
├── contrastive-fruits/
│   └── contrastive-fruits/
│       ├── ResNet.py                 # Model architecture
│       ├── simclr_service.py         # Model service wrapper
│       └── ...other files
├── start_all.bat                      # One-click startup script
└── README.md                          # This file
```

## 🔧 Configuration

### Backend Command-Line Arguments

```powershell
python scripts\scripts\python_model_service.py `
    --ckpt PATH_TO_CHECKPOINT      # PyTorch checkpoint file
    --gallery-root PATH_TO_GALLERY  # Directory with gallery images
    --port 8001                     # Flask server port
    --host 0.0.0.0                 # Flask server host
```

### Environment Variables
- No environment variables required
- Gallery rebuilds automatically on startup if `--gallery-root` is provided

## 🎯 Usage

### 1. Upload an Image
- Click **Image🖼️** to select a file
- Or click **Capture📸** to capture from camera (mobile/webcam)

### 2. View Results
- JSON response shows top-k matching images from gallery
- Bar chart visualizes confidence scores (0-100%)
- Confidence = dot product similarity (normalized embeddings)

### 3. Manage Gallery
- Add images to `gallery/` folder
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`
- Gallery rebuilds automatically on server restart

### 4. Change Theme
- Click 🖼**Theme** in sidebar
- Options: Light, Dark, Neon
- Theme persists in browser localStorage

## 🔗 API Endpoints

### Health Check
```
GET /health
Response: {"ok": true, "ckpt": "...", "device": "cpu", ...}
```

### Image Upload (Frontend)
```
POST /upload-form
Form Data: photo (binary file)
Response: {
    "ok": true,
    "analysis": {
        "labels": [
            {
                "description": "Match 1: /path/to/image.png",
                "confidence": 0.95
            },
            ...
        ]
    }
}
```

### Embed Image
```
POST /embed
Form Data: photo (binary file)
Response: {
    "ok": true,
    "embedding": [float, ...512 dims]
}
```

### Get K-Nearest Neighbors
```
POST /nn?k=5
Form Data: photo (binary file)
Response: {
    "ok": true,
    "matches": [
        ["path/to/image.png", 0.95, 0.14],
        ...
    ]
}
```

### Reload Checkpoint
```
POST /reload
JSON Body: {"ckpt": "path/to/model.pt"}
Response: {"ok": true, "ckpt": "..."}
```

## 🚨 Troubleshooting

### Backend Won't Start
- **Error**: `checkpoint not found`
  - Solution: Verify checkpoint path: `ckpt_epoch_1000\ckpt_epoch_1000.pt`
  
- **Error**: Connection refused
  - Solution: Check port 8001 is not in use
  - Try: `netstat -ano | findstr 8001`

### Frontend Shows 404
- Ensure backend is running on http://127.0.0.1:8001
- Check browser console for CORS errors
- Verify backend has loaded model successfully

### Gallery Not Loading
- Check directory exists: `gallery/`
- Verify images are valid and readable
- Check logs for: `[GALLERY] Built gallery: X images`

### Slow Image Processing
- First startup is slow (model initialization)
- Subsequent requests should be ~100-500ms
- If GPU available, enable with `--device cuda`

## 📦 Dependencies

### Python (Backend)
- `torch` - PyTorch for neural networks
- `torchvision` - Computer vision models
- `flask` - Web server framework
- `flask-cors` - Cross-origin request handling
- `numpy` - Numerical computing
- `pillow` - Image processing
- `scikit-learn` - Machine learning utilities

### JavaScript (Frontend)
- `http-server` - Simple static file server
- `chart.js` - Data visualization

## 🔐 Security Notes

- This is a development server (single-threaded, no authentication)
- Do NOT use in production without HTTPS and authentication
- Gallery paths are exposed in API responses
- Image files should be treated as sensitive data

## 📝 License

Refer to the Contrastive Fruits project license.

## 🤝 Support

For issues with:
- **Model Loading**: Check checkpoint path and PyTorch version
- **Gallery Building**: Ensure images are valid PNG/JPG files
- **Frontend**: Clear browser cache and check DevTools console
- **CORS Issues**: Verify `flask-cors` is installed and enabled

## ✅ Verification Checklist

- [ ] Python venv activated
- [ ] All pip dependencies installed
- [ ] Checkpoint file exists at `ckpt_epoch_1000\ckpt_epoch_1000.pt`
- [ ] Gallery folder exists (can be empty)
- [ ] Backend starts: `flask running on http://127.0.0.1:8001`
- [ ] Frontend loads: http://127.0.0.1:8080/main.html
- [ ] Can upload image and get results
- [ ] Gallery images show in similarity matches
