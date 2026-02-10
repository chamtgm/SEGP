#!/usr/bin/env python3
"""Send an image to the running service's /upload-form and /heatmap endpoints,
then save a combined analysis JSON and the returned heatmap image.

Usage:
    python scripts/analyze_with_heatmap.py --image webapp/gallery/banana.jpg --host http://localhost:8001
"""
import argparse
import requests
import base64
from pathlib import Path
import time
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--host', default='http://localhost:8001')
    args = p.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print('Image not found:', img_path)
        return

    # POST multipart to /upload-form
    files = {'photo': open(str(img_path),'rb')}
    print('Posting to /upload-form...')
    resp = requests.post(args.host + '/upload-form', files=files)
    if resp.status_code != 200:
        print('upload-form failed:', resp.status_code, resp.text)
        return
    analysis = resp.json()

    # POST raw bytes to /heatmap
    print('Posting raw image to /heatmap...')
    with open(img_path, 'rb') as f:
        data = f.read()
    resp2 = requests.post(args.host + '/heatmap', data=data)
    if resp2.status_code != 200:
        print('/heatmap failed:', resp2.status_code, resp2.text)
        heatmap = None
        heatmap_path = None
    else:
        j = resp2.json()
        heatmap = j.get('heatmap_base64')
        heatmap_path = j.get('heatmap_path')

    # merge and save
    out = {'analysis': analysis, 'heatmap_base64': heatmap, 'heatmap_path': heatmap_path}
    out_dir = Path('outputs/analysis')
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time()*1000)
    out_path = out_dir / f'analysis_{img_path.stem}_{ts}.json'
    out_path.write_text(json.dumps(out, indent=2))
    print('Saved analysis JSON to', out_path)

    if heatmap:
        hm_bytes = base64.b64decode(heatmap)
        hm_path = out_dir / f'heatmap_{img_path.stem}_{ts}.png'
        hm_path.write_bytes(hm_bytes)
        print('Saved heatmap PNG to', hm_path)

    if heatmap_path:
        print('Server-side heatmap saved at:', heatmap_path)

if __name__ == '__main__':
    main()
