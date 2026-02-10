#!/usr/bin/env python3
"""Generate a heatmap PNG and overlay PNG for a single image using ModelService
from `webapp/scripts/scripts/python_model_service.py`.

Usage:
    python scripts/generate_heatmap.py --ckpt webapp/ckpt_epoch_1000/ckpt_epoch_1000.pt --image webapp/gallery/banana.jpg

Outputs saved to `outputs/heatmaps/` by default.
"""
from pathlib import Path
import argparse
import importlib.util
import base64
import sys


def load_service_module(path: Path):
    spec = importlib.util.spec_from_file_location('svc_mod', str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='Path to checkpoint .pt')
    p.add_argument('--image', required=True, help='Path to input image')
    p.add_argument('--out-dir', default='outputs/heatmaps', help='Output directory')
    p.add_argument('--use-cv', action='store_true', help='Use OpenCV colormap if available')
    p.add_argument('--colormap', default='plasma', help='Colormap name (jet, hot, viridis, plasma, etc.)')
    p.add_argument('--alpha', type=float, default=0.7, help='Overlay alpha (0.0-1.0)')
    p.add_argument('--device', default=None, help='Torch device (e.g., cpu or cuda:0)')
    p.add_argument('--service-path', default=None, help='Path to python_model_service.py (optional)')
    args = p.parse_args()

    img_path = Path(args.image)
    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        print('Checkpoint not found:', ckpt_path)
        sys.exit(1)
    if not img_path.exists():
        print('Image not found:', img_path)
        sys.exit(1)

    # discover service file
    if args.service_path:
        svc_path = Path(args.service_path)
    else:
        svc_path = Path(__file__).resolve().parents[1] / 'webapp' / 'scripts' / 'scripts' / 'python_model_service.py'
        if not svc_path.exists():
            svc_path = Path('webapp') / 'scripts' / 'scripts' / 'python_model_service.py'

    if not svc_path.exists():
        print('Could not find python_model_service.py, pass --service-path to locate it')
        sys.exit(1)

    svc_mod = load_service_module(svc_path)

    # instantiate service and load model
    svc = svc_mod.ModelService(ckpt_path=str(ckpt_path), device=args.device)
    if svc.model is None:
        print('Loading model from checkpoint...')
        try:
            svc.load_model(str(ckpt_path))
        except Exception as e:
            print('Failed to load model:', e)
            sys.exit(1)

    # read image bytes
    img_bytes = img_path.read_bytes()

    # compute heatmap (support both simple and rich signatures)
    heat_b64 = None
    overlay_b64 = None
    temp_stats = None
    try:
        try:
            # Rich signature (returns dict)
            out = svc.heatmap_from_bytes(img_bytes, use_cv=bool(args.use_cv), colormap=args.colormap, alpha=float(args.alpha))
            if isinstance(out, dict):
                heat_b64 = out.get('heatmap_base64')
                overlay_b64 = out.get('overlay_base64')
                temp_stats = out.get('temperature_stats')
            else:
                # if it returned a base64 string
                heat_b64 = out
        except TypeError:
            # Fallback: simple signature returning base64 heatmap string
            out_simple = svc.heatmap_from_bytes(img_bytes)
            if isinstance(out_simple, (str, bytes)):
                heat_b64 = out_simple if isinstance(out_simple, str) else out_simple.decode('ascii')
            elif isinstance(out_simple, dict):
                heat_b64 = out_simple.get('heatmap_base64')
                overlay_b64 = out_simple.get('overlay_base64')
                temp_stats = out_simple.get('temperature_stats')
    except Exception as e:
        print('Heatmap generation failed:', e)
        sys.exit(1)

    # decode and write heatmap and overlay (if present)

    if heat_b64:
        heat_bytes = base64.b64decode(heat_b64)
        heat_path = out_dir / (img_path.stem + '_heatmap.png')
        heat_path.write_bytes(heat_bytes)
        print('Saved heatmap to', heat_path)
    else:
        print('No heatmap produced')

    if overlay_b64:
        overlay_bytes = base64.b64decode(overlay_b64)
        overlay_path = out_dir / (img_path.stem + '_overlay.png')
        overlay_path.write_bytes(overlay_bytes)
        print('Saved overlay to', overlay_path)

    if temp_stats:
        print('Temperature stats:', temp_stats)


if __name__ == '__main__':
    main()
