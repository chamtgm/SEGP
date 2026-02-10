#!/usr/bin/env python3
"""Generate synthetic anchors by augmenting gallery images and embedding them using the model.
Saves N embeddings to <out_dir>/embeddings.npy (L2-normalized).
"""
import argparse
import os
from pathlib import Path
import random
import numpy as np
from PIL import Image
import torchvision.transforms as T
import importlib.util

# load ModelService from scripts/python_model_service.py
spec = importlib.util.spec_from_file_location('python_model_service', os.path.join(os.path.dirname(__file__), 'python_model_service.py'))
pmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pmod)
# ModelService is defined as ModelService in that module
ModelService = pmod.ModelService


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, default=os.path.join('webapp', 'ckpt_epoch_1000', 'ckpt_epoch_1000.pt'))
    p.add_argument('--gallery', type=str, default=os.path.join('webapp', 'gallery'))
    p.add_argument('--out-dir', type=str, default='anchors')
    p.add_argument('--n', type=int, default=500)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    gallery = Path(args.gallery)
    if not gallery.exists():
        raise RuntimeError(f'Gallery path not found: {gallery}')
    imgs = [p for p in gallery.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
    if len(imgs) == 0:
        raise RuntimeError(f'No images found in gallery {gallery}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    svc = ModelService(ckpt_path=args.ckpt, gallery_root=None, device=args.device)
    # ensure model is loaded
    if not getattr(svc, 'model', None):
        svc.load_model(args.ckpt)

    # augmentation pipeline (PIL -> PIL)
    aug = T.Compose([
        T.RandomResizedCrop(size=224, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
    ])

    embeddings = []
    meta = []
    target = int(args.n)
    batch = []
    batch_meta = []

    i = 0
    # iterate and repeatedly augment images until we have target count
    while len(embeddings) < target:
        src = random.choice(imgs)
        img = Image.open(src).convert('RGB')
        aug_img = aug(img)
        batch.append(aug_img)
        batch_meta.append((str(src), i))
        i += 1
        # process batch when full or we have enough
        if len(batch) >= args.batch_size or len(embeddings) + len(batch) >= target:
            embs = svc._embed_batch(batch)
            embeddings.append(embs)
            meta.extend(batch_meta)
            batch = []
            batch_meta = []

    embeddings = np.vstack(embeddings)[:target]
    # L2-normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    embeddings = embeddings / norms

    np.save(out_dir / 'embeddings.npy', embeddings)
    # save metadata for debugging
    meta_out = out_dir / 'paths.txt'
    with open(meta_out, 'w', encoding='utf-8') as f:
        for s, idx in meta[:target]:
            f.write(f"{idx}\t{s}\n")

    print(f'Saved {embeddings.shape[0]} embeddings to {out_dir / "embeddings.npy"}')


if __name__ == '__main__':
    main()
