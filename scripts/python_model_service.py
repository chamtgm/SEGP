#!/usr/bin/env python3
import io
import os
import threading
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

import torch
from torchvision import transforms as T
import base64
import time
import subprocess

# optional visualization / projection libs
try:
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAVE_TSNE = True
except Exception:
    HAVE_TSNE = False

# optional OpenCV for colored heatmaps
try:
    import cv2
    HAVE_CV2 = True
except Exception:
    cv2 = None
    HAVE_CV2 = False
import importlib.util

import sys
import re
from collections import Counter


def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ROOT = Path(__file__).resolve().parents[1]
CF_DIR = ROOT / 'contrastive-fruits'

# Load backbone module directly
sys.path.insert(0, str(CF_DIR))
resnet_mod = load_module_from_path('resnet_mod', str(CF_DIR / 'ResNet.py'))
sys.path.remove(str(CF_DIR))

get_backbone = resnet_mod.get_backbone


class ModelService:
    def __init__(self, ckpt_path: str = None, gallery_root: str = None, device: str = None):
        # initialize synchronization and device
        self.lock = threading.Lock()
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        # model placeholder and preprocessing
        self.model = None
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # checkpoint path (no default): only set if provided by caller
        self.ckpt_path = ckpt_path

        # gallery root (no default): only set if provided by caller
        self.gallery_root = gallery_root
        self.gallery_paths: List[str] = []
        self.gallery_embeddings: np.ndarray = np.zeros((0, 512), dtype=np.float32)

        # load model now that ckpt_path and device exist (only if file exists)
        if self.ckpt_path and os.path.exists(self.ckpt_path):
            try:
                self.load_model(self.ckpt_path)
            except Exception as e:
                print(f"Warning: failed to load checkpoint {self.ckpt_path}: {e}")
        else:
            print(f"Warning: checkpoint not found at {self.ckpt_path}; model not loaded.")

        # --- anchor embeddings (optional) ---
        self.anchor_embeddings = None
        anchor_path = str(ROOT / 'anchors' / 'embeddings.npy')
        if os.path.exists(anchor_path):
            try:
                print(f"Loading t-SNE anchors from {anchor_path}...")
                self.anchor_embeddings = np.load(anchor_path)
                # Normalize anchors once on load
                norms = np.linalg.norm(self.anchor_embeddings, axis=1, keepdims=True) + 1e-10
                self.anchor_embeddings = self.anchor_embeddings / norms
                # Limit to 500 points to keep t-SNE fast
                if self.anchor_embeddings.shape[0] > 500:
                    indices = np.random.choice(self.anchor_embeddings.shape[0], 500, replace=False)
                    self.anchor_embeddings = self.anchor_embeddings[indices]
            except Exception:
                print(f"Failed to load anchors from {anchor_path}")
                self.anchor_embeddings = None
        else:
            print("Warning: anchors/embeddings.npy not found. t-SNE will look sparse.")
    # ------------------------

        self._gallery_built = False

    def load_model(self, ckpt_path: str):
        with self.lock:
            print('Loading model from', ckpt_path)
            model = get_backbone(pretrained=False)
            ckpt = torch.load(ckpt_path, map_location='cpu')
            state = ckpt.get('model_state', ckpt)
            model.load_state_dict(state, strict=False)
            model = model.to(self.device)
            model.eval()
            self.model = model
            self.ckpt_path = ckpt_path
            print('Model loaded to', self.device)

    def _embed_batch(self, images: List[Image.Image]) -> np.ndarray:
        tensors = [self.transform(img).unsqueeze(0) for img in images]
        x = torch.cat(tensors, dim=0).to(self.device)
        with torch.no_grad():
            try:
                feats = self.model.encoder(x).view(x.size(0), -1)
            except Exception:
                feats = self.model(x)
            feats = feats.cpu().numpy()
        return feats

    def embed_image_bytes(self, data: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(data)).convert('RGB')
        emb = self._embed_batch([img])[0]
        return emb

    def build_gallery(self, rebuild: bool = False, batch_size: int = 32):
        if self._gallery_built and not rebuild:
            return
        # If no gallery root was provided or the path doesn't exist, skip building
        if not self.gallery_root or not os.path.exists(self.gallery_root):
            print('No gallery root set or path does not exist:', self.gallery_root)
            self.gallery_paths = []
            self.gallery_embeddings = np.zeros((0, 512), dtype=np.float32)
            self._gallery_built = True
            return

        print('Building gallery from', self.gallery_root)
        paths = []
        exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        for root, _, files in os.walk(self.gallery_root):
            for f in files:
                if Path(f).suffix.lower() in exts:
                    paths.append(os.path.join(root, f))
        paths = sorted(paths)
        if len(paths) == 0:
            print('No gallery images found at', self.gallery_root)
            self.gallery_paths = []
            self.gallery_embeddings = np.zeros((0, 512), dtype=np.float32)
            self._gallery_built = True
            return

        embeddings = []
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i+batch_size]
            imgs = [Image.open(p).convert('RGB') for p in batch_paths]
            embs = self._embed_batch(imgs)
            embeddings.append(embs)
        self.gallery_embeddings = np.vstack(embeddings)
        # L2-normalize
        norms = np.linalg.norm(self.gallery_embeddings, axis=1, keepdims=True) + 1e-10
        self.gallery_embeddings = self.gallery_embeddings / norms
        self.gallery_paths = paths
        self._gallery_built = True
        print(f'Built gallery: {len(self.gallery_paths)} images')

    def knn(self, query_emb: np.ndarray, k: int = 5) -> List[Tuple[str, float, float]]:
        # ensure gallery is ready
        if not self._gallery_built:
            self.build_gallery()
        if self.gallery_embeddings.shape[0] == 0:
            return []

        # prepare query
        q = query_emb.reshape(1, -1)
        # normalize
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-10)

        sim = q.dot(self.gallery_embeddings.T).flatten()
        idx = np.argsort(-sim)[:k]

        # compute euclidean distance on normalized vectors: euclid = sqrt(2-2*cos)
        top_matches = []
        for i in idx:
            s = float(sim[i])
            euclid = float((max(0.0, 2.0 - 2.0 * s)) ** 0.5)
            top_matches.append((self.gallery_paths[i], s, euclid))

        # guard: if no matches (shouldn't happen if gallery non-empty), return empty
        if len(top_matches) == 0:
            return []

        # Return top matches (confidence gating handled by caller)
        return top_matches

    def heatmap_from_bytes(self, data: bytes, use_cv: bool = False, colormap: str = 'jet', alpha: float = 0.5):
        """Compute Grad-CAM using the last Conv2d layer if possible.
        Falls back to input-gradient saliency when Grad-CAM cannot be computed.
        Returns dictionary with base64 images and a small temperature map/stats.
        """
        try:
            img = Image.open(io.BytesIO(data)).convert('RGB')
        except Exception as e:
            raise RuntimeError(f'failed to open image: {e}')

        x = self.transform(img).unsqueeze(0).to(self.device)
        # ensure input requires grad for fallback method
        x.requires_grad_(True)

        # Helper storage for hooks
        activations = {}
        gradients = {}

        def find_conv_before_gap(module):
            """Return the Conv2d module immediately before the first adaptive/global pooling
            (e.g., AdaptiveAvgPool2d) if present. Otherwise return the last Conv2d found.
            """
            items = list(module.named_modules())
            conv_idxs = []
            pool_idxs = []
            for idx, (name, m) in enumerate(items):
                if isinstance(m, torch.nn.Conv2d):
                    conv_idxs.append(idx)
                if isinstance(m, (torch.nn.AdaptiveAvgPool2d, torch.nn.AdaptiveMaxPool2d)):
                    pool_idxs.append(idx)

            # If we found a pooling layer, choose the last conv that appears before the first pool
            if pool_idxs:
                first_pool = pool_idxs[0]
                candidates = [i for i in conv_idxs if i < first_pool]
                if candidates:
                    chosen_idx = candidates[-1]
                    return items[chosen_idx][1]

            # Fallback: return last conv if available
            if conv_idxs:
                return items[conv_idxs[-1]][1]
            return None

        target_layer = None
        if self.model is not None:
            try:
                target_layer = find_conv_before_gap(self.model)
            except Exception:
                target_layer = None
        # Log the selected Conv layer for Grad-CAM to the console each time
        try:
            print('Grad-CAM target layer:', target_layer)
        except Exception:
            pass

        hook_handles = []

        def forward_hook(module, inp, out):
            activations['value'] = out.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple; take first
            gradients['value'] = grad_out[0].detach()

        try:
            if target_layer is not None:
                hook_handles.append(target_layer.register_forward_hook(forward_hook))
                hook_handles.append(target_layer.register_full_backward_hook(backward_hook))

            # forward pass
            self.model.zero_grad()
            out = None
            try:
                out = self.model(x)
            except Exception:
                try:
                    out = self.model.encoder(x).view(x.size(0), -1)
                except Exception:
                    out = None

            if out is None:
                raise RuntimeError('model forward failed for heatmap generation')

            # scalar score: use L2 norm of output
            score = out.norm()
            score.backward(retain_graph=False)

            # Try Grad-CAM if activations and gradients were captured
            if 'value' in activations and 'value' in gradients:
                act = activations['value'].cpu().squeeze(0)  # C,H,W
                grad = gradients['value'].cpu().squeeze(0)   # C,H,W
                # global average pooling of gradients
                weights = torch.mean(grad, dim=(1, 2)).numpy()
                act_np = act.numpy()
                cam = np.zeros(act_np.shape[1:], dtype=np.float32)
                for i, w in enumerate(weights):
                    cam += w * act_np[i]
                cam = np.maximum(cam, 0)
                cam = cam - cam.min()
                if cam.max() > 0:
                    cam = cam / (cam.max())
                agg_uint8 = (cam * 255).astype(np.uint8)
            else:
                # Fallback: use input gradients
                if x.grad is None:
                    # ensure we can compute input gradients
                    x.requires_grad_(True)
                    self.model.zero_grad()
                    out = self.model(x) if hasattr(self.model, '__call__') else None
                    if out is None:
                        try:
                            out = self.model.encoder(x).view(x.size(0), -1)
                        except Exception:
                            out = None
                    if out is None:
                        raise RuntimeError('fallback forward failed')
                    out.norm().backward()

                grad = x.grad.detach().cpu().squeeze(0).numpy()  # C,H,W
                agg = np.abs(grad).sum(axis=0)
                agg = agg - agg.min()
                if agg.max() > 0:
                    agg = agg / (agg.max())
                agg_uint8 = (agg * 255).astype(np.uint8)
        finally:
            for h in hook_handles:
                try:
                    h.remove()
                except Exception:
                    pass

        # Apply colormap if requested
        if use_cv and HAVE_CV2:
            try:
                arr = agg_uint8
                cmap_name = (colormap or 'jet').lower()
                cmap_map = {
                    'jet': cv2.COLORMAP_JET,
                    'hot': cv2.COLORMAP_HOT,
                    'viridis': getattr(cv2, 'COLORMAP_VIRIDIS', cv2.COLORMAP_JET),
                    'plasma': getattr(cv2, 'COLORMAP_PLASMA', cv2.COLORMAP_JET),
                    'magma': getattr(cv2, 'COLORMAP_MAGMA', cv2.COLORMAP_JET),
                    'inferno': getattr(cv2, 'COLORMAP_INFERNO', cv2.COLORMAP_JET),
                    'gray': None, 'grey': None,
                }
                cmap = cmap_map.get(cmap_name, cv2.COLORMAP_JET)
                if cmap is None:
                    colored = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
                else:
                    colored = cv2.applyColorMap(arr, cmap)
                    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
                heat = Image.fromarray(colored)
                heat = heat.resize(img.size, resample=Image.BICUBIC)
            except Exception:
                heat = Image.fromarray(agg_uint8, mode='L')
                heat = heat.resize(img.size, resample=Image.BICUBIC)
        else:
            heat = Image.fromarray(agg_uint8, mode='L')
            heat = heat.resize(img.size, resample=Image.BICUBIC)

        try:
            orig = img.convert('RGB')
            heat_rgb = heat.convert('RGB')
            overlay = Image.blend(orig, heat_rgb, alpha=float(alpha))
        except Exception:
            overlay = img

        buf = io.BytesIO()
        heat.save(buf, format='PNG')
        heat_b64 = base64.b64encode(buf.getvalue()).decode('ascii')

        buf2 = io.BytesIO()
        overlay.save(buf2, format='PNG')
        overlay_b64 = base64.b64encode(buf2.getvalue()).decode('ascii')

        # temperature summary
        try:
            arr = np.array(agg_uint8).astype(np.float32) / 255.0
            small = Image.fromarray((arr * 255).astype(np.uint8)).resize((32, 32), resample=Image.BICUBIC)
            small_arr = np.array(small).astype(np.float32) / 255.0
            temp_map = small_arr.tolist()
            temp_stats = {'min': float(arr.min()), 'max': float(arr.max()), 'mean': float(arr.mean())}
        except Exception:
            temp_map = None
            temp_stats = None

        return {'heatmap_base64': heat_b64, 'overlay_base64': overlay_b64, 'temperature_map': temp_map, 'temperature_stats': temp_stats}

    def heatmap_patch_similarity_from_bytes(self, data: bytes, patch_size: int = 64, stride: int = 32,
                                            top_k: int = 1, max_patches: int = 1024,
                                            use_cv: bool = False, colormap: str = 'jet') -> str:
        """Compute a semantic heatmap by sliding-window patch embeddings.
        For each patch we compute its embedding (resized to model input), compare
        to the gallery embeddings and use the top-k similarity as the patch score.
        The resulting patch grid is upsampled to the original image size and
        returned as a PNG base64 string. This highlights image regions whose
        local appearance is most similar to gallery images.
        """
        try:
            img = Image.open(io.BytesIO(data)).convert('RGB')
        except Exception as e:
            raise RuntimeError(f'failed to open image: {e}')

        if self.gallery_embeddings.shape[0] == 0:
            raise RuntimeError('gallery empty: cannot compute similarity heatmap')

        W, H = img.size
        # ensure patch fits at least once
        if patch_size <= 0 or stride <= 0:
            raise ValueError('patch_size and stride must be > 0')

        xs = list(range(0, max(1, W - patch_size + 1), stride))
        ys = list(range(0, max(1, H - patch_size + 1), stride))
        if len(xs) == 0: xs = [0]
        if len(ys) == 0: ys = [0]

        coords = [(x, y) for y in ys for x in xs]
        total = len(coords)

        # Sampling to limit computation
        if total > max_patches:
            idxs = np.linspace(0, total - 1, max_patches, dtype=int).tolist()
            sampled_coords = [coords[i] for i in idxs]
            coords_to_process = sampled_coords
            grid_shape = (len(ys), len(xs))
            sampled_mask = np.zeros((len(ys), len(xs)), dtype=np.bool_)
            for i in idxs:
                ry = i // len(xs)
                rx = i % len(xs)
                sampled_mask[ry, rx] = True
        else:
            coords_to_process = coords
            grid_shape = (len(ys), len(xs))
            sampled_mask = np.ones(grid_shape, dtype=np.bool_)

        # Build patch images
        patches = []
        for (x, y) in coords_to_process:
            patch = img.crop((x, y, min(x + patch_size, W), min(y + patch_size, H)))
            patches.append(patch.resize((224, 224), resample=Image.BICUBIC))

        if len(patches) == 0:
            raise RuntimeError('no patches generated')

        # embed in batches
        batch = 64
        embs = []
        for i in range(0, len(patches), batch):
            embs.append(self._embed_batch(patches[i:i+batch]))
        embs = np.vstack(embs)

        # normalize patch embeddings
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
        embs = embs / norms

        # compute similarity to gallery (gallery already L2-normalized)
        sims = embs.dot(self.gallery_embeddings.T)  # (P, G)

        # for each patch compute top-k average similarity
        if top_k <= 1:
            patch_scores = sims.max(axis=1)
        else:
            topk_idx = np.argsort(-sims, axis=1)[:, :top_k]
            # gather and average
            rows = np.arange(sims.shape[0])[:, None]
            topk_vals = sims[rows, topk_idx]
            patch_scores = topk_vals.mean(axis=1)

        # build score grid (fill zeros for unprocessed patches if sampling)
        score_grid = np.zeros(grid_shape, dtype=np.float32)
        if total > max_patches:
            # fill only sampled positions
            for (x, y), s in zip(coords_to_process, patch_scores):
                ix = xs.index(x)
                iy = ys.index(y)
                score_grid[iy, ix] = float(s)
            # interpolate missing cells by resizing with bilinear to smooth
        else:
            idx = 0
            for iy in range(grid_shape[0]):
                for ix in range(grid_shape[1]):
                    score_grid[iy, ix] = float(patch_scores[idx])
                    idx += 1

        # Upsample the grid to original image size
        # Normalize grid to [0,255]
        g = score_grid.copy()
        g = g - g.min()
        if g.max() > 0:
            g = g / g.max()
        g_uint8 = (g * 255).astype(np.uint8)

        # Convert grid to image by resizing (grid -> HxW)
        grid_img = Image.fromarray(g_uint8, mode='L')
        grid_img = grid_img.resize((W, H), resample=Image.BICUBIC)

        # Optionally apply OpenCV colormap
        if use_cv and HAVE_CV2:
            try:
                arr = np.array(grid_img)
                cmap_name = (colormap or 'jet').lower()
                cmap_map = {
                    'jet': cv2.COLORMAP_JET,
                    'hot': cv2.COLORMAP_HOT,
                    'viridis': getattr(cv2, 'COLORMAP_VIRIDIS', cv2.COLORMAP_JET),
                    'plasma': getattr(cv2, 'COLORMAP_PLASMA', cv2.COLORMAP_JET),
                    'magma': getattr(cv2, 'COLORMAP_MAGMA', cv2.COLORMAP_JET),
                    'inferno': getattr(cv2, 'COLORMAP_INFERNO', cv2.COLORMAP_JET),
                    'gray': None, 'grey': None,
                }
                cmap = cmap_map.get(cmap_name, cv2.COLORMAP_JET)
                if cmap is None:
                    colored = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
                else:
                    colored = cv2.applyColorMap(arr, cmap)
                    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
                heat = Image.fromarray(colored)
            except Exception:
                heat = grid_img
        else:
            heat = grid_img

        # create overlay by blending with original image
        try:
            orig = img.convert('RGB')
            heat_rgb = heat.convert('RGB')
            overlay = Image.blend(orig, heat_rgb, 0.5)
        except Exception:
            overlay = img

        # encode results
        buf = io.BytesIO()
        heat.save(buf, format='PNG')
        heat_b64 = base64.b64encode(buf.getvalue()).decode('ascii')

        buf2 = io.BytesIO()
        overlay.save(buf2, format='PNG')
        overlay_b64 = base64.b64encode(buf2.getvalue()).decode('ascii')

        # temperature outputs: return the coarse normalized score_grid and top locations
        try:
            norm_grid = g.copy()
            temp_grid = norm_grid.tolist()
            # find top patches (up to 5)
            flat = []
            for iy in range(grid_shape[0]):
                for ix in range(grid_shape[1]):
                    flat.append(((ix, iy), float(norm_grid[iy, ix])))
            flat_sorted = sorted(flat, key=lambda x: -x[1])[:5]
            top_patches = []
            for (ix, iy), score in flat_sorted:
                px = int(xs[ix]) if ix < len(xs) else 0
                py = int(ys[iy]) if iy < len(ys) else 0
                top_patches.append({'grid_coord': (ix, iy), 'score': float(score), 'pixel_coord': (px, py)})
        except Exception:
            temp_grid = None
            top_patches = None

        return {'heatmap_base64': heat_b64, 'overlay_base64': overlay_b64, 'temperature_map': temp_grid, 'top_patches': top_patches}


app = Flask('simclr_service')
svc = ModelService()


def _strip_nulls(obj):
    """Recursively remove keys with value None from dicts."""
    if isinstance(obj, dict):
        return {k: _strip_nulls(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_nulls(v) for v in obj]
    return obj


@app.route('/health')
def health():
    return jsonify({'ok': True, 'ckpt': svc.ckpt_path, 'device': str(svc.device)})


@app.route('/embed', methods=['POST'])
def embed():
    try:
        data = request.get_data()
        if not data:
            return jsonify({'error': 'empty body'}), 400
        emb = svc.embed_image_bytes(data)
        # normalize
        norm = float(np.linalg.norm(emb) + 1e-10)
        # By default do NOT include the raw embedding vector in responses
        # to avoid sending large arrays to a frontend. Clients can opt-in
        # by passing ?include_raw=1 (or true/yes).
        include_raw = str(request.args.get('include_raw', '0')).lower() in ('1', 'true', 'yes')
        resp = {'norm': norm, 'ckpt': svc.ckpt_path}
        if include_raw:
            resp['embedding'] = (emb / norm).tolist()
        return jsonify(_strip_nulls(resp))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/nn', methods=['POST'])
def nn():
    try:
        k = int(request.args.get('k', '5'))
        data = request.get_data()
        if not data:
            return jsonify({'error': 'empty body'}), 400
        emb = svc.embed_image_bytes(data)
        results = svc.knn(emb, k=k)
        # results: list of (path, score, euclidean_distance)
        resp = {'nn': [{'path': p, 'score': s, 'euclidean_distance': ed} for p, s, ed in results], 'ckpt': svc.ckpt_path}

        # Gatekeeper: similarity threshold check
        if not results:
            # no neighbors found -> unknown
            resp['nn'] = []
            resp['confidence_level'] = 'unknown'
            resp['confidence_score'] = 0.0
            resp['is_known_class'] = False
            resp['similarity_status'] = 'Unknown'
            return jsonify(_strip_nulls(resp))

        best_score = float(results[0][1])
        # classify confidence and similarity status
        if best_score > 0.85:
            resp['confidence_level'] = 'high'
            resp['confidence_score'] = best_score
            resp['is_known_class'] = True
            resp['similarity_status'] = 'High Match'
        elif best_score >= 0.70:
            resp['confidence_level'] = 'low'
            resp['confidence_score'] = best_score
            resp['is_known_class'] = True
            resp['similarity_status'] = 'Low Match'
        else:
            # Unknown / unseen
            resp['nn'] = []
            resp['confidence_level'] = 'unknown'
            resp['confidence_score'] = 0.0
            resp['is_known_class'] = False
            resp['similarity_status'] = 'Unknown'
            return jsonify(_strip_nulls(resp))

        # Drill-down: derive class labels from neighbor paths and compute distribution
        try:
            def extract_label(p: str) -> str:
                # take immediate parent folder as label source
                try:
                    lbl = Path(p).parent.name
                except Exception:
                    lbl = os.path.basename(os.path.dirname(p))
                # normalize: remove common suffixes like '_done' and non-word chars
                lbl = re.sub(r'(_done$)', '', lbl, flags=re.IGNORECASE)
                lbl = re.sub(r"[^0-9A-Za-z_]+", "_", lbl)
                return lbl

            labels = [extract_label(p) for p, _, _ in results]
            dist = Counter(labels)
            class_distribution = dict(dist)
            majority_class = dist.most_common(1)[0][0] if len(dist) > 0 else None
            resp['class_distribution'] = class_distribution
            resp['majority_class'] = majority_class
            # Calculate weighted_confidence: (votes_for_winner / k_effective) * (average score of winner)
            try:
                k_effective = max(1, len(results))
                votes_for_winner = class_distribution.get(majority_class, 0)
                # collect scores for neighbors belonging to majority_class
                scores_for_winner = [s for (p, s, ed), lbl in zip(results, labels) if lbl == majority_class]
                avg_score_winner = float(sum(scores_for_winner) / len(scores_for_winner)) if scores_for_winner else 0.0
                vote_ratio = float(votes_for_winner) / float(k_effective)
                weighted_confidence = vote_ratio * avg_score_winner
                resp['weighted_confidence'] = float(weighted_confidence)
            except Exception:
                resp['weighted_confidence'] = None
        except Exception:
            # non-fatal: skip distribution on errors
            pass
    
        # Attempt to compute 2D t-SNE projection for input + neighbors
        try:
            # gather neighbor embeddings from gallery
            neigh_paths = [r[0] for r in results]
            # map paths to indices in svc.gallery_paths
            idxs = [svc.gallery_paths.index(p) if p in svc.gallery_paths else None for p in neigh_paths]
            neigh_embs = []
            valid_paths = []
            valid_scores = []
            valid_euclid_dists = []
            for i, idx in enumerate(idxs):
                if idx is not None:
                    neigh_embs.append(svc.gallery_embeddings[idx])
                    valid_paths.append(neigh_paths[i])
                    # results entries: (path, score, euclid_dist)
                    valid_scores.append(results[i][1])
                    valid_euclid_dists.append(results[i][2])
            if len(neigh_embs) > 0 and HAVE_TSNE:
                # Use anchors/background then input then neighbors for t-SNE
                # Normalize input embedding to same scale as gallery (gallery already L2-normalized)
                q = emb / (np.linalg.norm(emb) + 1e-10)

                # 1. Start with Anchors (Background)
                if svc.anchor_embeddings is not None:
                    background = svc.anchor_embeddings
                else:
                    # Fallback: use a subset of gallery if anchors missing (better than nothing)
                    bg_count = min(500, svc.gallery_embeddings.shape[0])
                    background = svc.gallery_embeddings[:bg_count]

                # 2. Stack: [Background] + [Input] + [Neighbors]
                # Note: We stack neighbors last to easily find them
                arr = np.vstack([
                    background,
                    q[np.newaxis, :],
                    np.vstack(neigh_embs)
                ])

                # 3. Run t-SNE
                ts = TSNE(n_components=2, perplexity=30, init='pca', random_state=0)
                coords = ts.fit_transform(arr)

                # 4. Unpack Coordinates
                n_bg = background.shape[0]
                # Background coords (for gray dots)
                bg_coords = coords[:n_bg].tolist()
                # Input coord (The one after background)
                input_coord = coords[n_bg].tolist()
                # Neighbor coords (The rest)
                neigh_coords_list = coords[n_bg+1:].tolist()

                # Do not include background coordinates in the returned analysis payload
                resp['tsne_coordinates'] = {
                    'input': input_coord,
                    'neighbors': [
                        {'path': p, 'score': s, 'euclidean_distance': ed, 'coord': c}
                        for p, s, ed, c in zip(valid_paths, valid_scores, valid_euclid_dists, neigh_coords_list)
                    ]
                }

                # 5. Plotting (Update indices)
                try:
                    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

                    # Plot Background (Gray, small, transparent)
                    bg_x = [c[0] for c in bg_coords]
                    bg_y = [c[1] for c in bg_coords]
                    ax.scatter(bg_x, bg_y, c='lightgray', s=20, alpha=0.5, label='Context')

                    # Plot Neighbors (Colored by score)
                    neigh_x = [c[0] for c in neigh_coords_list]
                    neigh_y = [c[1] for c in neigh_coords_list]
                    ax.scatter(neigh_x, neigh_y, c=valid_scores, cmap='viridis', s=60, edgecolor='k', label='Neighbors')

                    # Plot Input (Red Star)
                    ax.scatter(input_coord[0], input_coord[1], c='red', marker='*', s=150, edgecolor='k', label='Input')

                    ax.set_title('Robustness Analysis (t-SNE)')
                    ax.legend()
                    ax.axis('off')

                    buf = io.BytesIO()
                    plt.tight_layout()
                    fig.savefig(buf, format='png', bbox_inches='tight')
                    plt.close(fig)
                    buf.seek(0)
                    # save PNG to disk instead of returning base64
                    out_dir = Path(ROOT) / 'uploads' / 'tsne_plots'
                    out_dir.mkdir(parents=True, exist_ok=True)
                    fname = f"tsne_{int(time.time()*1000)}.png"
                    fpath = out_dir / fname
                    with open(fpath, 'wb') as f:
                        f.write(buf.getvalue())
                    # return filesystem path and attempt to open automatically
                    resp['tsne_plot_path'] = str(fpath)
                    try:
                        if sys.platform.startswith('win'):
                            os.startfile(str(fpath))
                        elif sys.platform == 'darwin':
                            subprocess.run(['open', str(fpath)], check=False)
                        else:
                            subprocess.run(['xdg-open', str(fpath)], check=False)
                    except Exception:
                        # ignore failures to open viewer
                        pass
                except Exception:
                    # plotting failed, skip image
                    pass
        except Exception:
            # if any step fails, just skip tsne
            pass

        return jsonify(_strip_nulls(resp))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/heatmap', methods=['POST'])
def heatmap():
    try:
        data = request.get_data()
        if not data:
            return jsonify({'error': 'empty body'}), 400
        # support optional OpenCV colored heatmap via ?cv=1 and ?colormap=jet
        # Standardize: always compute the Grad-CAM style heatmap (no mode switching)
        use_cv = str(request.args.get('cv', '0')).lower() in ('1', 'true', 'yes')
        colormap = request.args.get('colormap', 'jet')
        alpha = float(request.args.get('alpha') or 0.5)
        out = svc.heatmap_from_bytes(data, use_cv=use_cv, colormap=colormap, alpha=alpha)

        # Save returned images (heatmap and overlay) if present
        out_dir = Path(ROOT) / 'uploads' / 'heatmaps'
        out_dir.mkdir(parents=True, exist_ok=True)
        heatmap_path = None
        overlay_path = None
        try:
            if isinstance(out, dict):
                if 'heatmap_base64' in out and out['heatmap_base64']:
                    buf = base64.b64decode(out['heatmap_base64'])
                    prefix = 'heatmap_cv' if use_cv else 'heatmap'
                    fname = f"{prefix}_{int(time.time()*1000)}.png"
                    fpath = out_dir / fname
                    with open(fpath, 'wb') as f:
                        f.write(buf)
                    heatmap_path = str(fpath)
                if 'overlay_base64' in out and out['overlay_base64']:
                    buf2 = base64.b64decode(out['overlay_base64'])
                    fname2 = f"overlay_{int(time.time()*1000)}.png"
                    fpath2 = out_dir / fname2
                    with open(fpath2, 'wb') as f:
                        f.write(buf2)
                    overlay_path = str(fpath2)
                # attempt to open overlay for convenience (best-effort)
                try:
                    if overlay_path and sys.platform.startswith('win'):
                        os.startfile(overlay_path)
                except Exception:
                    pass
        except Exception:
            pass

        resp = {'ckpt': svc.ckpt_path}
        if isinstance(out, dict):
            resp.update(out)
            resp['heatmap_path'] = heatmap_path
            resp['overlay_path'] = overlay_path
        else:
            # legacy: out was a base64 string
            resp['heatmap_base64'] = out
        return jsonify(_strip_nulls(resp))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/reload', methods=['POST'])
def reload_ckpt():
    try:
        body = request.get_json(force=True)
        path = body.get('ckpt') if isinstance(body, dict) else None
        if not path:
            return jsonify({'error': 'no ckpt provided'}), 400
        if not os.path.exists(path):
            return jsonify({'error': f'ckpt not found: {path}'}), 400
        svc.load_model(path)
        # rebuild gallery embeddings on reload to ensure consistency
        svc.build_gallery(rebuild=True)
        return jsonify({'ok': True, 'ckpt': svc.ckpt_path})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default=None)
    p.add_argument('--gallery-root', default=None)
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=8001)
    args = p.parse_args()
    if args.ckpt:
        svc.load_model(args.ckpt)
    if args.gallery_root:
        svc.gallery_root = args.gallery_root
    # build gallery in background thread to allow immediate responses
    t = threading.Thread(target=svc.build_gallery, kwargs={'rebuild': False}, daemon=True)
    t.start()
    app.run(host=args.host, port=args.port)
