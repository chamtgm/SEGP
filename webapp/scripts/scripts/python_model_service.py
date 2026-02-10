#!/usr/bin/env python3
import io
import os
import threading
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

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
    TSNE_IMPORT_ERROR = None
except Exception as e:
    HAVE_TSNE = False
    TSNE_IMPORT_ERROR = f"{type(e).__name__}: {e}"

import importlib.util

import sys
import re
from collections import Counter


def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ROOT = Path(__file__).resolve().parents[2]
CF_DIR = ROOT / 'contrastive-fruits' / 'contrastive-fruits'

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
        # Try to generate an anchor t-SNE plot (non-fatal)
        try:
            _ = self.generate_anchor_tsne()
        except Exception:
            pass

        self._gallery_built = False

    def generate_anchor_tsne(self):
        """Generate a t-SNE plot for loaded anchor embeddings (limited to 500) and save to uploads/tsne_plots.
        Non-fatal: failures are logged and will not affect normal operation."""
        try:
            if self.anchor_embeddings is None or self.anchor_embeddings.shape[0] < 2:
                self.anchor_tsne_path = None
                return None
            n = self.anchor_embeddings.shape[0]
            perplexity = min(30, max(1, n - 1))
            print(f"Generating anchor t-SNE for {n} points (perplexity={perplexity})...", flush=True)
            ts = TSNE(n_components=2, perplexity=perplexity, init='pca', random_state=0)
            coords = ts.fit_transform(self.anchor_embeddings)
            xs = coords[:, 0]
            ys = coords[:, 1]
            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
            ax.scatter(xs, ys, c='lightblue', s=20, alpha=0.8)
            ax.set_title(f"Anchors t-SNE ({n} samples)")
            ax.axis('off')

            out_dir = Path(ROOT) / 'uploads' / 'tsne_plots'
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"anchors_tsne_{int(time.time()*1000)}.png"
            fpath = out_dir / fname
            fig.savefig(str(fpath), bbox_inches='tight')
            plt.close(fig)
            self.anchor_tsne_path = str(fpath)
            print(f"Saved anchors t-SNE to {self.anchor_tsne_path}", flush=True)
            try:
                if sys.platform.startswith('win'):
                    os.startfile(str(fpath))
                elif sys.platform == 'darwin':
                    subprocess.run(['open', str(fpath)], check=False)
                else:
                    subprocess.run(['xdg-open', str(fpath)], check=False)
            except Exception:
                pass
            return self.anchor_tsne_path
        except Exception as e:
            print(f"Failed to generate anchor t-SNE: {e}", flush=True)
            self.anchor_tsne_path = None
            return None

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

    def heatmap_from_bytes(self, data: bytes, use_cv: bool = False, colormap: str = 'plasma', alpha: float = 0.7):
        """Compute a gradient-based saliency heatmap (or fallback input-grad) and return
        a dict with base64-encoded heatmap PNG, overlay PNG, and temperature stats.

        The function performs percentile-based contrast stretching, optional smoothing,
        and color mapping (OpenCV if available, otherwise matplotlib).
        """
        try:
            img = Image.open(io.BytesIO(data)).convert('RGB')
        except Exception as e:
            raise RuntimeError(f'failed to open image: {e}')

        # prepare tensor (using the same preprocessing as embeddings)
        x = self.transform(img).unsqueeze(0).to(self.device)
        # ensure gradients for fallback method
        x.requires_grad_(True)

        self.model.zero_grad()
        out = None
        with torch.enable_grad():
            try:
                try:
                    out = self.model.encoder(x).view(x.size(0), -1)
                except Exception:
                    out = self.model(x)
                score = out.norm()
                score.backward()
            except Exception as e:
                raise RuntimeError(f'heatmap forward/backward failed: {e}')

        # Try Grad-CAM style via last conv layer if available (best) - else fallback to input-grad
        try:
            # attempt grad-cam style if hooks present in other code paths - simplified here
            grad = x.grad.detach().cpu().squeeze(0).numpy()  # C,H,W
            agg = np.abs(grad).sum(axis=0)
        except Exception:
            agg = None

        if agg is None or agg.size == 0:
            raise RuntimeError('failed to compute gradients for heatmap')

        # Normalize and enhance contrast using percentiles
        arr = agg.astype(np.float32)
        arr = arr - arr.min()
        if arr.max() > 0:
            arr = arr / (arr.max())
        # percentile stretch to boost contrast (clip extremes)
        try:
            lo = np.percentile(arr, 5)
            hi = np.percentile(arr, 99.5)
            if hi <= lo:
                lo = 0.0
                hi = arr.max() if arr.max() > 0 else 1.0
            arr = (arr - lo) / (hi - lo + 1e-10)
            arr = np.clip(arr, 0.0, 1.0)
        except Exception:
            pass

        # Smooth a bit to reduce speckle
        try:
            if HAVE_CV2:
                arr_uint8 = (arr * 255).astype(np.uint8)
                arr_uint8 = cv2.GaussianBlur(arr_uint8, (0, 0), sigmaX=2)
                arr = arr_uint8.astype(np.float32) / 255.0
            else:
                from PIL import ImageFilter
                tmp = Image.fromarray((arr * 255).astype(np.uint8), mode='L')
                tmp = tmp.filter(ImageFilter.GaussianBlur(radius=2))
                arr = np.array(tmp).astype(np.float32) / 255.0
        except Exception:
            pass

        # Colorize using OpenCV if available, otherwise matplotlib
        colored = None
        cmap_name = (colormap or 'jet').lower()
        try:
            if use_cv and HAVE_CV2:
                cmap_map = {
                    'jet': cv2.COLORMAP_JET,
                    'hot': cv2.COLORMAP_HOT,
                    'viridis': getattr(cv2, 'COLORMAP_VIRIDIS', cv2.COLORMAP_JET),
                    'plasma': getattr(cv2, 'COLORMAP_PLASMA', cv2.COLORMAP_JET),
                    'magma': getattr(cv2, 'COLORMAP_MAGMA', cv2.COLORMAP_JET),
                    'inferno': getattr(cv2, 'COLORMAP_INFERNO', cv2.COLORMAP_JET),
                }
                arr_uint8 = (arr * 255).astype(np.uint8)
                colored = cv2.applyColorMap(arr_uint8, cmap_map.get(cmap_name, cv2.COLORMAP_JET))
                colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            else:
                # use matplotlib colormap
                try:
                    import matplotlib
                    import matplotlib.cm as cm
                    cmap = cm.get_cmap(cmap_name)
                    rgba = cmap(arr)
                    colored = (rgba[:, :, :3] * 255).astype(np.uint8)
                except Exception:
                    colored = (np.stack([arr, arr, arr], axis=2) * 255).astype(np.uint8)
        except Exception:
            colored = (np.stack([arr, arr, arr], axis=2) * 255).astype(np.uint8)

        # Build PIL images
        heat = Image.fromarray(colored)
        heat = heat.resize(img.size, resample=Image.BILINEAR)

        # Overlay with original image
        try:
            orig = img.convert('RGB')
            heat_rgb = heat.convert('RGB')
            overlay = Image.blend(orig, heat_rgb, alpha=float(alpha))
        except Exception:
            overlay = img

        # Encode both heatmap and overlay to base64
        buf = io.BytesIO()
        heat.save(buf, format='PNG')
        heat_b64 = base64.b64encode(buf.getvalue()).decode('ascii')

        buf2 = io.BytesIO()
        overlay.save(buf2, format='PNG')
        overlay_b64 = base64.b64encode(buf2.getvalue()).decode('ascii')

        # temperature summary (coarse)
        try:
            temp_map = (arr * 255).astype(np.uint8).tolist()
            temp_stats = {'min': float(arr.min()), 'max': float(arr.max()), 'mean': float(arr.mean())}
        except Exception:
            temp_map = None
            temp_stats = None

        return {'heatmap_base64': heat_b64, 'overlay_base64': overlay_b64, 'temperature_map': temp_map, 'temperature_stats': temp_stats}


app = Flask('simclr_service')
CORS(app)
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
    return jsonify({
        'ok': True,
        'ckpt': svc.ckpt_path,
        'device': str(svc.device),
        'python_executable': sys.executable,
        'have_tsne': bool(HAVE_TSNE),
        'tsne_import_error': TSNE_IMPORT_ERROR,
    })


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
        out = svc.heatmap_from_bytes(data, use_cv=True, colormap=request.args.get('colormap', 'plasma'), alpha=float(request.args.get('alpha') or 0.7))

        heatmap_b64 = None
        overlay_b64 = None
        fpath = None
        if isinstance(out, dict):
            heatmap_b64 = out.get('heatmap_base64')
            overlay_b64 = out.get('overlay_base64')
        elif isinstance(out, (str, bytes)):
            heatmap_b64 = out if isinstance(out, str) else out.decode('ascii')

        # save heatmap PNG to uploads/heatmaps and attempt to open
        if heatmap_b64:
            try:
                img_bytes = base64.b64decode(heatmap_b64)
                out_dir = Path(ROOT) / 'uploads' / 'heatmaps'
                out_dir.mkdir(parents=True, exist_ok=True)
                fname = f"heatmap_{int(time.time()*1000)}.png"
                fpath = out_dir / fname
                with open(fpath, 'wb') as f:
                    f.write(img_bytes)
                try:
                    if sys.platform.startswith('win'):
                        os.startfile(str(fpath))
                    elif sys.platform == 'darwin':
                        subprocess.run(['open', str(fpath)], check=False)
                    else:
                        subprocess.run(['xdg-open', str(fpath)], check=False)
                except Exception:
                    pass
            except Exception:
                fpath = None

        return jsonify({'heatmap_base64': heatmap_b64, 'overlay_base64': overlay_b64, 'heatmap_path': str(fpath) if fpath is not None else None, 'ckpt': svc.ckpt_path})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upload-form', methods=['POST'])
def upload_form():
    """Handle image upload from frontend form"""
    try:
        print("[UPLOAD] Request received", flush=True)
        
        if 'photo' not in request.files:
            print("[UPLOAD] No photo in request.files", flush=True)
            return jsonify({'error': 'No photo provided'}), 400
        
        print("[UPLOAD] Reading photo bytes", flush=True)
        photo_file = request.files['photo']
        photo_bytes = photo_file.read()
        print(f"[UPLOAD] Photo size: {len(photo_bytes)} bytes", flush=True)
        
        # Check if model is loaded
        if svc.model is None:
            print("[UPLOAD] Model not loaded", flush=True)
            return jsonify({
                'ok': False,
                'error': 'Model not loaded. Please load a checkpoint via /reload endpoint.',
                'analysis': {
                    'labels': []
                }
            }), 400
        
        # Embed the image
        print("[UPLOAD] Starting embedding...", flush=True)
        emb = svc.embed_image_bytes(photo_bytes)
        print(f"[UPLOAD] Embedding complete: shape {emb.shape}", flush=True)
        
        # Get nearest neighbors (k=5)
        print("[UPLOAD] Getting k-nearest neighbors...", flush=True)
        results = svc.knn(emb, k=5)
        print(f"[UPLOAD] Got {len(results)} results", flush=True)

        # Format response for frontend as a JSON-serializable structure
        labels = [
            {
                'description': f"Match {i+1}: {result[0]}",
                'confidence': float(result[1]),
            }
            for i, result in enumerate(results)
        ]

        # Optional: compute t-SNE coordinates for input + neighbors (for frontend visualization)
        tsne_coordinates = None
        tsne_plot_path = None
        tsne_debug = {
            'have_tsne': bool(HAVE_TSNE),
            'results_count': int(len(results)),
            'valid_neighbor_embeddings': 0,
            'n_samples': None,
            'status': 'skipped',
            'reason': None,
            'error': None,
        }
        try:
            print(f"[UPLOAD] HAVE_TSNE={HAVE_TSNE}, results count={len(results)}", flush=True)
            if HAVE_TSNE and results:
                neigh_paths = [r[0] for r in results]
                idxs = [svc.gallery_paths.index(p) if p in svc.gallery_paths else None for p in neigh_paths]
                neigh_embs = []
                valid_paths = []
                valid_scores = []
                valid_euclid_dists = []
                for i, idx in enumerate(idxs):
                    if idx is not None:
                        neigh_embs.append(svc.gallery_embeddings[idx])
                        valid_paths.append(neigh_paths[i])
                        valid_scores.append(float(results[i][1]))
                        valid_euclid_dists.append(float(results[i][2]))

                tsne_debug['valid_neighbor_embeddings'] = int(len(neigh_embs))
                print(f"[UPLOAD] Found {len(neigh_embs)} valid neighbor embeddings", flush=True)

                if len(neigh_embs) > 0:
                    q = emb / (np.linalg.norm(emb) + 1e-10)

                    # Use anchors/background if available, else fall back to a subset of gallery
                    if svc.anchor_embeddings is not None:
                        background = svc.anchor_embeddings
                    else:
                        bg_count = min(500, svc.gallery_embeddings.shape[0])
                        background = svc.gallery_embeddings[:bg_count]

                    # Stack: [Background] + [Input] + [Neighbors]
                    arr = np.vstack([
                        background,
                        q[np.newaxis, :],
                        np.vstack(neigh_embs)
                    ])

                    n_samples = int(arr.shape[0])
                    tsne_debug['n_samples'] = n_samples
                    print(f"[UPLOAD] t-SNE: n_samples={n_samples}", flush=True)

                    # sklearn requires 1 < perplexity < n_samples
                    if n_samples >= 2:
                        perplexity = min(30, max(1, n_samples - 1))
                        print(f"[UPLOAD] Running t-SNE with perplexity={perplexity}...", flush=True)
                        ts = TSNE(n_components=2, perplexity=perplexity, init='pca', random_state=0)
                        coords = ts.fit_transform(arr)

                        # Unpack coordinates respecting background offset
                        n_bg = background.shape[0]
                        bg_coords = coords[:n_bg].tolist()
                        input_coord = coords[n_bg].tolist()
                        neigh_coords_list = coords[n_bg+1:].tolist()

                        tsne_coordinates = {
                            'input': input_coord,
                            'neighbors': [
                                {'path': p, 'score': s, 'euclidean_distance': ed, 'coord': c}
                                for p, s, ed, c in zip(valid_paths, valid_scores, valid_euclid_dists, neigh_coords_list)
                            ],
                        }
                        tsne_debug['status'] = 'ok'
                        print("[UPLOAD] t-SNE completed successfully", flush=True)
                        # Save TSNE plot as PNG to uploads/tsne_plots for visibility (do not change t-SNE computation)
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

                            out_dir = Path(ROOT) / 'uploads' / 'tsne_plots'
                            out_dir.mkdir(parents=True, exist_ok=True)
                            fname = f"upload_tsne_{int(time.time()*1000)}.png"
                            fpath = out_dir / fname
                            with open(fpath, 'wb') as f:
                                f.write(buf.getvalue())
                            tsne_plot_path = str(fpath)

                            try:
                                if sys.platform.startswith('win'):
                                    os.startfile(str(fpath))
                                elif sys.platform == 'darwin':
                                    subprocess.run(['open', str(fpath)], check=False)
                                else:
                                    subprocess.run(['xdg-open', str(fpath)], check=False)
                            except Exception:
                                pass
                        except Exception:
                            tsne_plot_path = None
                    else:
                        tsne_debug['reason'] = f'n_samples={n_samples} < 2'
                        print(f"[UPLOAD] t-SNE skipped: n_samples={n_samples} < 2", flush=True)
                else:
                    tsne_debug['reason'] = 'no valid neighbor embeddings'
                    print("[UPLOAD] t-SNE skipped: no valid neighbor embeddings found", flush=True)
            else:
                if not HAVE_TSNE:
                    tsne_debug['reason'] = 'HAVE_TSNE=False (scikit-learn not available)'
                    print("[UPLOAD] t-SNE skipped: HAVE_TSNE=False (scikit-learn not available)", flush=True)
                elif not results:
                    tsne_debug['reason'] = 'no results from KNN'
                    print("[UPLOAD] t-SNE skipped: no results from KNN", flush=True)
        except Exception as e:
            # Non-fatal: t-SNE is optional; log but ignore failures
            import traceback
            tsne_debug['status'] = 'error'
            tsne_debug['error'] = str(e)
            print(f"[UPLOAD] t-SNE ERROR: {str(e)}", flush=True)
            print(traceback.format_exc(), flush=True)
            tsne_coordinates = None

        # Attempt to compute a heatmap (non-fatal)
        heatmap_b64 = None
        overlay_b64 = None
        heatmap_path = None
        heatmap_error = None
        try:
            print("[UPLOAD] Generating heatmap...", flush=True)
            hm_out = svc.heatmap_from_bytes(photo_bytes)
            if isinstance(hm_out, dict):
                heatmap_b64 = hm_out.get('heatmap_base64')
                overlay_b64 = hm_out.get('overlay_base64')
            elif isinstance(hm_out, (str, bytes)):
                heatmap_b64 = hm_out if isinstance(hm_out, str) else hm_out.decode('ascii')
        except Exception as e:
            heatmap_error = str(e)
            print(f"[UPLOAD] heatmap generation failed: {e}", flush=True)

        # save PNG to uploads/heatmaps and attempt to open (like dedicated endpoint)
        if heatmap_b64:
            try:
                img_bytes = base64.b64decode(heatmap_b64)
                out_dir = Path(ROOT) / 'uploads' / 'heatmaps'
                out_dir.mkdir(parents=True, exist_ok=True)
                fname = f"upload_heatmap_{int(time.time()*1000)}.png"
                fpath = out_dir / fname
                with open(fpath, 'wb') as f:
                    f.write(img_bytes)
                heatmap_path = str(fpath)
                try:
                    if sys.platform.startswith('win'):
                        os.startfile(str(fpath))
                except Exception:
                    pass
            except Exception as e:
                heatmap_error = heatmap_error or str(e)
                print(f"[UPLOAD] Failed to save heatmap: {e}", flush=True)

        response = {
            'analysis': {
                'labels': labels,
            },
            'heatmap_base64': heatmap_b64,
            'overlay_base64': overlay_b64,
            'heatmap_path': heatmap_path,
            'tsne_plot_path': tsne_plot_path,
            'anchor_tsne_plot_path': getattr(svc, 'anchor_tsne_path', None)
        }
        if tsne_coordinates is not None:
            response['tsne_coordinates'] = tsne_coordinates
        response['tsne_debug'] = tsne_debug

        print("[UPLOAD] Success, returning response", flush=True)
        return jsonify(response)
    except Exception as e:
        import traceback
        print(f"[UPLOAD] ERROR: {str(e)}", flush=True)
        print(traceback.format_exc(), flush=True)
        return jsonify({'error': str(e), 'type': type(e).__name__}), 500


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

        # Attempt to (re)load anchor embeddings if present so t-SNE uses them immediately
        try:
            anchor_path = str(ROOT / 'anchors' / 'embeddings.npy')
            if os.path.exists(anchor_path):
                try:
                    print(f"Reload: Loading t-SNE anchors from {anchor_path}...")
                    svc.anchor_embeddings = np.load(anchor_path)
                    norms = np.linalg.norm(svc.anchor_embeddings, axis=1, keepdims=True) + 1e-10
                    svc.anchor_embeddings = svc.anchor_embeddings / norms
                    if svc.anchor_embeddings.shape[0] > 500:
                        indices = np.random.choice(svc.anchor_embeddings.shape[0], 500, replace=False)
                        svc.anchor_embeddings = svc.anchor_embeddings[indices]
                    print(f"Reload: Loaded {svc.anchor_embeddings.shape[0]} anchors")
                    try:
                        svc.generate_anchor_tsne()
                    except Exception:
                        print("Reload: failed to generate anchor t-SNE", flush=True)
                except Exception:
                    svc.anchor_embeddings = None
                    print(f"Reload: Failed to load anchors from {anchor_path}")
            else:
                print("Reload: No anchors/embeddings.npy found; continuing without anchors")
        except Exception:
            print("Reload: anchor reload step failed, continuing")

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
