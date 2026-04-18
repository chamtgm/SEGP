#!/usr/bin/env python3
import io
import os
import threading
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from flask import Flask, request, jsonify, make_response
from ultralytics import YOLO

import torch
from torch import nn
from torchvision import transforms as T
import torchvision.models as tv_models
import base64
import time
import subprocess

# optional plotting libs
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAVE_MATPLOTLIB = True
except Exception:
    HAVE_MATPLOTLIB = False

# optional projection libs
try:
    from sklearn.manifold import TSNE
    HAVE_TSNE = True
    TSNE_IMPORT_ERROR = None
except Exception as e:
    TSNE = None
    HAVE_TSNE = False
    TSNE_IMPORT_ERROR = f"{type(e).__name__}: {e}"


def _project_to_2d(arr: np.ndarray, random_state: int = 42):
    """Project embeddings to 2D using t-SNE."""
    if arr.shape[0] < 2:
        raise ValueError('Need at least 2 samples for 2D projection')

    if HAVE_TSNE and TSNE is not None:
        safe_perplexity = min(30, max(1, arr.shape[0] - 1))
        ts = TSNE(n_components=2, perplexity=safe_perplexity, init='pca', random_state=random_state)
        return ts.fit_transform(arr), 't-SNE'

    raise RuntimeError('No 2D projector available: t-SNE import failed')

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

def _fallback_get_backbone(pretrained: bool = False):
    """Fallback backbone when custom ResNet.py is missing."""
    resnet = tv_models.resnet18(weights=None)
    encoder = nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        resnet.layer3,
        resnet.layer4,
        resnet.avgpool,
    )

    class _Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder

        def forward(self, x):
            return self.encoder(x).view(x.size(0), -1)

    return _Backbone()

get_backbone = _fallback_get_backbone
for cf_dir in (
    ROOT / 'contrastive-fruits' / 'contrastive-fruits',
    ROOT / 'contrastive-fruits',
    ROOT / 'scripts' / 'contrastive-fruits',
):
    resnet_py = cf_dir / 'ResNet.py'
    if resnet_py.exists():
        try:
            sys.path.insert(0, str(cf_dir))
            resnet_mod = load_module_from_path('resnet_mod', str(resnet_py))
            sys.path.remove(str(cf_dir))
            get_backbone = resnet_mod.get_backbone
            break
        except Exception as e:
            print(f"Warning: failed to load custom ResNet backbone from {resnet_py}: {e}")

class PythonModelService:
    def __init__(self, gallery_root=None,
                 device='cpu',
                 embedding_dim=128,
                 hidden_dim=512,
                 num_classes=16,
                 backbone='resnet18',
                 centroids_path='fruit_centroids.pt'):
        self.gallery_root = gallery_root
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available; falling back to CPU.")
            device = 'cpu'
        self.device = torch.device(device)
        self.embedding_dim = embedding_dim
        
        # --- YOLO INTEGRATION ---
        yolo_path = ROOT / 'Object Detection' / 'runs' / 'fruit_detector' / 'weights' / 'best.pt'
        print(f"Loading YOLO Model from {yolo_path}...")
        try:
            self.yolo_model = YOLO(str(yolo_path))
            print("YOLO Model loaded successfully!")
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            self.yolo_model = None
        # ------------------------
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.centroids_path = centroids_path
        self.backbone = None
        self.classifier = None
        self.lock = threading.Lock()

        # model placeholder and preprocessing
        self.model = None
        self.classes = ['Apple', 'Banana', 'Dragonfruit', 'Durian', 'Eggplant', 'Grapes', 'Orange', 'Pineapple', 'Strawberry', 'Watermelon']
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # checkpoint path (no default): only set if provided by caller
        self.ckpt_path = None

        # gallery root (no default): only set if provided by caller
        self.gallery_root = gallery_root
        self.gallery_paths: List[str] = []
        self.gallery_embeddings: np.ndarray = np.zeros((0, 512), dtype=np.float32)

        # --- anchor embeddings (optional) ---
        self.anchor_embeddings = None
        anchor_path = str(ROOT / 'anchors' / 'embeddings.npy')
        if os.path.exists(anchor_path):
            try:
                print(f"Loading projection anchors from {anchor_path}...")
                self.anchor_embeddings = np.load(anchor_path)
                # Normalize anchors once on load
                norms = np.linalg.norm(self.anchor_embeddings, axis=1, keepdims=True) + 1e-10
                self.anchor_embeddings = self.anchor_embeddings / norms
            except Exception:
                print(f"Failed to load anchors from {anchor_path}")
                self.anchor_embeddings = None
        else:
            print("Warning: anchors/embeddings.npy not found. projection context will look sparse.")
    # ------------------------

        self._gallery_built = False

    def load_model(self, ckpt_path: str, classifier_path: str = None):
        with self.lock:
            print('Loading backbone from', ckpt_path)
            model = get_backbone(pretrained=False)
            ckpt = torch.load(ckpt_path, map_location='cpu')
            raw_state = ckpt.get('backbone_state', ckpt.get('model_state_dict', ckpt.get('model_state', ckpt)))
            
            # --- PARAMETER MAPPING: Translator for the Backbone ---
            fixed_state = {}
            for k, v in raw_state.items():
                if k.startswith("0."):
                    fixed_state[k.replace("0.", "encoder.", 1)] = v
                else:
                    fixed_state[k] = v
                    
            load_result = model.load_state_dict(fixed_state, strict=False)
            print(f"\n--- BACKBONE LOAD RESULT ---")
            print(f"Missing keys: {len(load_result.missing_keys)}")
            print("----------------------------\n")
            
            model = model.to(self.device)
            model.eval()
            self.model = model

            # --- CENTROID MAPPING: Load the Master Vectors ---
            centroids_path = self.centroids_path
            if os.path.exists(centroids_path):
                print(f"Loading Master Centroids from {centroids_path}...")
                centroid_data = torch.load(centroids_path, map_location=self.device)
                self.centroids = centroid_data['centroids'].to(self.device)
                self.classes = centroid_data['classes']
                print(f"Success! Loaded {len(self.classes)} centroids perfectly.")
            else:
                print(f"WARNING: {centroids_path} not found in the current folder! Predictions will fail.")
                self.centroids = None

            self.ckpt_path = ckpt_path
            print('Model loaded to', self.device)

    def detect_with_yolo(self, img: Image.Image, padding: int = 15, max_detections: int = 0):
        """Run YOLO once and return all detections sorted by confidence.

        Each detection contains:
          - crop: PIL image crop
          - box: (x1, y1, x2, y2)
          - confidence: detector confidence
          - class_id: YOLO class id (if available)
          - segmentation: polygon points [[x, y], ...] for segmentation models
        """
        if self.yolo_model is None:
            return []

        detections = []
        try:
            results = self.yolo_model(img, verbose=False)
            if not results:
                return []

            result = results[0]
            boxes = getattr(result, 'boxes', None)
            if boxes is None or len(boxes) == 0:
                return []

            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy() if getattr(boxes, 'conf', None) is not None else np.ones((len(xyxy),), dtype=np.float32)
            cls = boxes.cls.cpu().numpy() if getattr(boxes, 'cls', None) is not None else None

            mask_polygons = []
            masks = getattr(result, 'masks', None)
            if masks is not None and getattr(masks, 'xy', None) is not None:
                mask_polygons = masks.xy

            order = np.argsort(-conf)
            if max_detections and max_detections > 0:
                order = order[:max_detections]

            w, h = img.size
            for idx in order:
                x1, y1, x2, y2 = xyxy[idx]
                x1 = int(max(0, x1 - padding))
                y1 = int(max(0, y1 - padding))
                x2 = int(min(w, x2 + padding))
                y2 = int(min(h, y2 + padding))

                if x2 <= x1 or y2 <= y1:
                    continue

                segmentation = None
                if idx < len(mask_polygons):
                    try:
                        poly = np.asarray(mask_polygons[idx], dtype=np.float32)
                        if poly.ndim == 2 and poly.shape[1] >= 2 and poly.shape[0] > 0:
                            segmentation = [[float(p[0]), float(p[1])] for p in poly.tolist()]
                    except Exception:
                        segmentation = None

                class_id = None
                if cls is not None and idx < len(cls):
                    try:
                        class_id = int(cls[idx])
                    except Exception:
                        class_id = None

                detections.append({
                    'crop': img.crop((x1, y1, x2, y2)),
                    'box': (x1, y1, x2, y2),
                    'confidence': float(conf[idx]) if idx < len(conf) else None,
                    'class_id': class_id,
                    'segmentation': segmentation,
                })
        except Exception as e:
            print(f"YOLO multi detection failed: {e}")

        return detections

    def crop_with_yolo_with_coords(self, img: Image.Image) -> Tuple[Image.Image, Optional[Tuple[int, int, int, int]]]:
        """Use YOLO to find the highest-confidence fruit and crop the image. Returns (cropped_img, box_coords) or (img, None)."""
        detections = self.detect_with_yolo(img, max_detections=1)
        if len(detections) > 0:
            det = detections[0]
            return det['crop'], det['box']
        return img, None

    def crop_with_yolo(self, img: Image.Image) -> Image.Image:
        """Use YOLO to find the highest-confidence fruit and crop the image."""
        return self.crop_with_yolo_with_coords(img)[0]

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
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert('RGB')
        img = self.crop_with_yolo(img)
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
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.HEIC'}
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
            imgs = [ImageOps.exif_transpose(Image.open(p)).convert('RGB') for p in batch_paths]
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

    def _compute_gradcam_map(self, img: Image.Image) -> np.ndarray:
        """Compute a normalized Grad-CAM/saliency map for a single crop.

        Returns a float32 HxW numpy array in [0, 1] aligned to img size.
        """
        if self.model is None:
            return np.zeros((img.size[1], img.size[0]), dtype=np.float32)

        x = self.transform(img).unsqueeze(0).to(self.device)
        x.requires_grad_(True)

        activations = {}
        gradients = {}

        def find_conv_before_gap(module):
            items = list(module.named_modules())
            conv_idxs = []
            pool_idxs = []
            for idx, (_, m) in enumerate(items):
                if isinstance(m, torch.nn.Conv2d):
                    conv_idxs.append(idx)
                if isinstance(m, (torch.nn.AdaptiveAvgPool2d, torch.nn.AdaptiveMaxPool2d)):
                    pool_idxs.append(idx)

            if pool_idxs:
                first_pool = pool_idxs[0]
                candidates = [i for i in conv_idxs if i < first_pool]
                if candidates:
                    return items[candidates[-1]][1]
            if conv_idxs:
                return items[conv_idxs[-1]][1]
            return None

        with self.lock:
            target_layer = None
            try:
                target_layer = find_conv_before_gap(self.model)
            except Exception:
                target_layer = None

            hook_handles = []

            def forward_hook(module, inp, out):
                activations['value'] = out.detach()

            def backward_hook(module, grad_in, grad_out):
                gradients['value'] = grad_out[0].detach()

            try:
                if target_layer is not None:
                    hook_handles.append(target_layer.register_forward_hook(forward_hook))
                    hook_handles.append(target_layer.register_full_backward_hook(backward_hook))

                self.model.zero_grad()
                if self.classifier is not None:
                    self.classifier.zero_grad()

                out = None
                try:
                    feats = self.model.encoder(x).view(x.size(0), -1)
                except Exception:
                    feats = self.model(x)
                    if hasattr(feats, 'norm'):
                        out = feats

                if self.classifier is not None and out is None:
                    logits = self.classifier(feats)
                    predicted_class_idx = logits.argmax(dim=1).item()
                    score = logits[0, predicted_class_idx]
                else:
                    if out is None:
                        out = feats
                    score = out.norm()

                score.backward(retain_graph=False)

                if 'value' in activations and 'value' in gradients:
                    act = activations['value'].cpu().squeeze(0)  # C,H,W
                    grad = gradients['value'].cpu().squeeze(0)   # C,H,W
                    weights = torch.mean(grad, dim=(1, 2)).numpy()
                    act_np = act.numpy()
                    cam = np.zeros(act_np.shape[1:], dtype=np.float32)
                    for i, w in enumerate(weights):
                        cam += w * act_np[i]
                    cam = np.maximum(cam, 0)
                    cam = cam - cam.min()
                    if cam.max() > 0:
                        cam = cam / cam.max()
                    agg = cam
                else:
                    if x.grad is None:
                        x.requires_grad_(True)
                        self.model.zero_grad()
                        if self.classifier is not None:
                            self.classifier.zero_grad()
                        try:
                            f = self.model.encoder(x).view(x.size(0), -1)
                        except Exception:
                            f = self.model(x)

                        if self.classifier is not None:
                            lg = self.classifier(f)
                            idx = lg.argmax(dim=1).item()
                            sc = lg[0, idx]
                        else:
                            sc = f.norm()
                        sc.backward()

                    grad = x.grad.detach().cpu().squeeze(0).numpy()  # C,H,W
                    agg = np.abs(grad).sum(axis=0)
                    agg = agg - agg.min()
                    if agg.max() > 0:
                        agg = agg / agg.max()
            finally:
                for h in hook_handles:
                    try:
                        h.remove()
                    except Exception:
                        pass

        agg = np.array(agg, dtype=np.float32)
        agg = np.clip(agg, 0.0, 1.0)
        if agg.shape != (img.size[1], img.size[0]):
            agg_img = Image.fromarray((agg * 255).astype(np.uint8), mode='L')
            agg_img = agg_img.resize(img.size, resample=Image.BICUBIC)
            agg = np.array(agg_img).astype(np.float32) / 255.0
        return agg

    def heatmap_from_bytes(self, data: bytes, use_cv: bool = False, colormap: str = 'plasma', alpha: float = 0.7, labels_str: str = None):
        """Compute a heatmap overlay for all detected fruits.

        For each YOLO detection, computes Grad-CAM on the crop, pastes it back on the
        full image, and draws the detected box (and polygon mask outline when available).
        """
        try:
        
            orig_img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert('RGB')
        except Exception as e:
            raise RuntimeError(f'failed to open image: {e}')

        labels_list = labels_str.split('|') if labels_str else []

        W, H = orig_img.size
        detections = self.detect_with_yolo(orig_img)
        if len(detections) == 0:
            detections = [{
                'crop': orig_img,
                'box': (0, 0, W, H),
                'confidence': None,
                'class_id': None,
                'segmentation': None,
            }]

        overlay = orig_img.copy()
        heat_full = Image.new('RGB', orig_img.size, (0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        draw_heat = ImageDraw.Draw(heat_full)
        full_score = np.zeros((H, W), dtype=np.float32)
        box_width = max(3, int(min(orig_img.size) * 0.005))
        detection_summaries = []

        for i, det in enumerate(detections):
            box = det.get('box') or (0, 0, W, H)
            x1, y1, x2, y2 = [int(v) for v in box]
            x1 = max(0, min(W - 1, x1))
            y1 = max(0, min(H - 1, y1))
            x2 = max(x1 + 1, min(W, x2))
            y2 = max(y1 + 1, min(H, y2))

            crop = det.get('crop')
            if crop is None:
                crop = orig_img.crop((x1, y1, x2, y2))

            if crop.size[0] <= 0 or crop.size[1] <= 0:
                continue

            # Compute activation map in [0, 1] for this crop.
            agg = self._compute_gradcam_map(crop)
            region_w, region_h = (x2 - x1), (y2 - y1)
            if agg.shape != (region_h, region_w):
                agg_img = Image.fromarray((np.clip(agg, 0.0, 1.0) * 255).astype(np.uint8), mode='L')
                agg_img = agg_img.resize((region_w, region_h), resample=Image.BICUBIC)
                agg = np.array(agg_img).astype(np.float32) / 255.0

            agg_uint8 = (np.clip(agg, 0.0, 1.0) * 255).astype(np.uint8)

            # Apply colormap per detection.
            if use_cv and HAVE_CV2:
                try:
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
                        colored = cv2.cvtColor(agg_uint8, cv2.COLOR_GRAY2RGB)
                    else:
                        colored = cv2.applyColorMap(agg_uint8, cmap)
                        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
                    heat_region = Image.fromarray(colored)
                except Exception:
                    heat_region = Image.fromarray(agg_uint8, mode='L')
            else:
                heat_region = Image.fromarray(agg_uint8, mode='L')

            heat_region = heat_region.resize((region_w, region_h), resample=Image.BICUBIC)
            heat_rgb = heat_region.convert('RGB')

            crop_orig = orig_img.crop((x1, y1, x2, y2))
            overlay_crop = Image.blend(crop_orig, heat_rgb, alpha=float(alpha))
            overlay.paste(overlay_crop, (x1, y1))
            heat_full.paste(heat_rgb, (x1, y1))

            # Keep a full-size scalar map for temperature summary using max fusion.
            full_score[y1:y2, x1:x2] = np.maximum(full_score[y1:y2, x1:x2], agg)

            # Draw bounding box for this detection.
            draw_overlay.rectangle([x1, y1, x2, y2], outline="#00FF00", width=box_width)
            draw_heat.rectangle([x1, y1, x2, y2], outline="#00FF00", width=box_width)

            # Draw label if provided
            if i < len(labels_list) and labels_list[i]:
                label_text = labels_list[i]
                font_size = max(12, int(orig_img.size[0] * 0.02))
                try:
                    from PIL import ImageFont
                    font = ImageFont.truetype("arial.ttf", font_size)
                except Exception:
                    font = ImageFont.load_default()
                text_bbox = draw_overlay.textbbox((0, 0), label_text, font=font)
                tw = text_bbox[2] - text_bbox[0]
                th = text_bbox[3] - text_bbox[1]
                
                # Make a background for the label
                bg_rect = [x1, max(0, y1 - th - 8), x1 + tw + 8, max(0, y1 - th - 8) + th + 8]
                draw_overlay.rectangle(bg_rect, fill=(0, 0, 0, int(0.7 * 255)))
                draw_overlay.text((x1 + 4, max(0, y1 - th - 8) + 2), label_text, fill="#00FF00", font=font)
                
                draw_heat.rectangle(bg_rect, fill=(0, 0, 0, int(0.7 * 255)))
                draw_heat.text((x1 + 4, max(0, y1 - th - 8) + 2), label_text, fill="#00FF00", font=font)

            # Draw segmentation polygon outline when available.
            seg_points = det.get('segmentation')
            seg_count = 0
            if isinstance(seg_points, list):
                try:
                    pts = []
                    for p in seg_points:
                        if isinstance(p, (list, tuple)) and len(p) >= 2:
                            px = int(round(float(p[0])))
                            py = int(round(float(p[1])))
                            pts.append((px, py))
                    seg_count = len(pts)
                    if len(pts) >= 2:
                        draw_overlay.line(pts + [pts[0]], fill="#00FF00", width=max(2, box_width // 2))
                        draw_heat.line(pts + [pts[0]], fill="#00FF00", width=max(2, box_width // 2))
                except Exception:
                    seg_count = 0

            detection_summaries.append({
                'index': i,
                'bbox': [x1, y1, x2, y2],
                'detector_confidence': det.get('confidence'),
                'detector_class_id': det.get('class_id'),
                'segmentation_points': seg_count,
            })

        buf = io.BytesIO()
        heat_full.save(buf, format='PNG')
        heat_b64 = base64.b64encode(buf.getvalue()).decode('ascii')

        buf2 = io.BytesIO()
        overlay.save(buf2, format='PNG')
        overlay_b64 = base64.b64encode(buf2.getvalue()).decode('ascii')

        # temperature summary from combined full-image score map
        try:
            arr = np.clip(full_score, 0.0, 1.0).astype(np.float32)
            small = Image.fromarray((arr * 255).astype(np.uint8), mode='L').resize((32, 32), resample=Image.BICUBIC)
            small_arr = np.array(small).astype(np.float32) / 255.0
            temp_map = small_arr.tolist()
            temp_stats = {'min': float(arr.min()), 'max': float(arr.max()), 'mean': float(arr.mean())}
        except Exception:
            temp_map = None
            temp_stats = None

        return {
            'heatmap_base64': heat_b64,
            'overlay_base64': overlay_b64,
            'temperature_map': temp_map,
            'temperature_stats': temp_stats,
            'num_detections': len(detection_summaries),
            'detections': detection_summaries,
        }

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
            orig_img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert('RGB')
            img, box = self.crop_with_yolo_with_coords(orig_img)
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
            heat_rgb = heat.convert('RGB')
            if box is not None:
                from PIL import ImageDraw
                x1, y1, x2, y2 = box
                
                orig_full = orig_img.convert('RGB')
                crop_orig = orig_full.crop(box)
                overlay_crop = Image.blend(crop_orig, heat_rgb, 0.5)
                
                overlay = orig_full.copy()
                overlay.paste(overlay_crop, (x1, y1))
                
                draw = ImageDraw.Draw(overlay)
                draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=max(3, int(min(orig_full.size)*0.005)))
                
                full_heat = Image.new('RGB', orig_full.size, (0, 0, 0))
                full_heat.paste(heat_rgb, (x1, y1))
                draw_heat = ImageDraw.Draw(full_heat)
                draw_heat.rectangle([x1, y1, x2, y2], outline="#00FF00", width=max(3, int(min(orig_full.size)*0.005)))
                heat = full_heat
            else:
                orig = img.convert('RGB')
                overlay = Image.blend(orig, heat_rgb, 0.5)
        except Exception as e:
            print(f"Overlay patch error: {e}")
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
svc = None


@app.after_request
def add_cors_headers(resp):
    """Allow frontend from anywhere to call this API."""
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type, ngrok-skip-browser-warning, bypass-tunnel-reminder'
    resp.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return resp


def _strip_nulls(obj):
    """Recursively remove keys with value None from dicts."""
    if isinstance(obj, dict):
        return {k: _strip_nulls(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_nulls(v) for v in obj]
    return obj


@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return make_response(('', 200))

    if HAVE_TSNE:
        projection_engine = 'tsne'
    else:
        projection_engine = 'none'

    return jsonify({
        'ok': True,
        'ckpt': svc.ckpt_path,
        'device': str(svc.device),
        'python_executable': sys.executable,
        'projection_engine': projection_engine,
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
    
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    try:
        # Handle CORS preflight
        if request.method == 'OPTIONS':
            return make_response(('', 200))
            
        data = request.get_data()
        if not data:
            return jsonify({'error': 'empty body'}), 400

        # Ensure the gallery is built for the KNN check
        if not svc._gallery_built:
            svc.build_gallery()

        def extract_label(p: str) -> str:
            try:
                lbl = Path(p).parent.name
            except Exception:
                lbl = os.path.basename(os.path.dirname(p))
            lbl = re.sub(r'(_done$)', '', lbl, flags=re.IGNORECASE)
            lbl = re.sub(r"[^0-9A-Za-z_]+", "_", lbl)
            return lbl.strip('_').lower()

        # 1. Process the image and get all YOLO detections (boxes + segmentation when available)
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert('RGB')
        try:
            max_detections = int(request.args.get('max_detections', '0') or 0)
        except Exception:
            max_detections = 0

        detections = svc.detect_with_yolo(img, max_detections=max_detections)
        if len(detections) == 0:
            # Fallback: run one prediction on the full image when YOLO finds nothing.
            detections = [{
                'crop': img,
                'box': None,
                'confidence': None,
                'class_id': None,
                'segmentation': None,
            }]

        # 2. Embed all detections in one forward pass
        x = torch.cat([svc.transform(d['crop']).unsqueeze(0) for d in detections], dim=0).to(svc.device)
        with torch.no_grad():
            with svc.lock:
                try:
                    feats_batch = svc.model.encoder(x).view(x.size(0), -1)
                except Exception:
                    feats_batch = svc.model(x)

        THRESHOLD = 0.5
        detection_predictions = []

        # 3. Predict class per detection
        for idx, det in enumerate(detections):
            feats = feats_batch[idx:idx+1]

            # --- PREDICTION via KNN ---
            results = svc.knn(feats.cpu().numpy(), k=5)
            has_majority = False
            confidence_score = 0.0
            raw_class_name = "Unknown / Not a Fruit"

            if results:
                labels = [extract_label(p) for p, _, _ in results]
                dist = Counter(labels)
                majority_class, count = dist.most_common(1)[0]
                has_majority = count >= 3
                
                # map lowercase folder name to proper class name
                class_map = {c.lower(): c for c in svc.classes}
                raw_class_name = class_map.get(majority_class, majority_class.replace('_', ' ').title())

                # The score of the most similar image in the majority class
                majority_scores = [s for l, (_, s, _) in zip(labels, results) if l == majority_class]
                confidence_score = max(majority_scores) if majority_scores else 0.0

            if raw_class_name.lower() == "rubbish dataset":
                predicted_class = "Unknown / Not a Fruit"
                confidence_percent = 0.0
            elif (confidence_score < THRESHOLD or not has_majority):
                predicted_class = "Unknown / Not a Fruit"
                confidence_percent = 0.0
            else:
                predicted_class = raw_class_name
                confidence_percent = round(max(0.0, confidence_score) * 100, 2)

            detection_predictions.append({
                'index': idx,
                'bbox': list(det['box']) if det.get('box') is not None else None,
                'detector_confidence': det.get('confidence'),
                'detector_class_id': det.get('class_id'),
                'segmentation': det.get('segmentation'),
                'predicted_class': predicted_class,
                'confidence': f"{confidence_percent}%",
                'raw_score': float(confidence_score),
                'has_knn_majority': bool(has_majority),
            })

        # Keep backward-compatible top-level fields using the first (highest-confidence) detection
        primary = detection_predictions[0]
        resp = {
            'predicted_class': primary['predicted_class'],
            'confidence': primary['confidence'],
            'raw_score': primary['raw_score'],
            'ckpt': svc.ckpt_path,
            'num_detections': len(detection_predictions),
            'detections': detection_predictions,
        }
        
        return jsonify(_strip_nulls(resp))

    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/nn', methods=['POST', 'OPTIONS'])
def nearest_neighbors():
    try:
        # Handle CORS preflight
        if request.method == 'OPTIONS':
            return make_response(('', 200))
        k = int(request.args.get('k', '5'))
        tsne_plot = str(request.args.get('umap_plot', request.args.get('tsne_plot', '1'))).lower() in ('1', 'true', 'yes')
        all_detections = str(request.args.get('all_detections', '0')).lower() in ('1', 'true', 'yes')
        tsne_global = str(request.args.get('umap_global', request.args.get('tsne_global', '1'))).lower() in ('1', 'true', 'yes')
        try:
            max_detections = int(request.args.get('max_detections', '0') or 0)
        except Exception:
            max_detections = 0
        try:
            tsne_bg_max = int(request.args.get('umap_bg_max', request.args.get('tsne_bg_max', '800')) or 800)
        except Exception:
            tsne_bg_max = 800
        data = request.get_data()
        if not data:
            return jsonify({'error': 'empty body'}), 400
        def extract_label(p: str) -> str:
            try:
                lbl = Path(p).parent.name
            except Exception:
                lbl = os.path.basename(os.path.dirname(p))
            lbl = re.sub(r'(_done$)', '', lbl, flags=re.IGNORECASE)
            lbl = re.sub(r"[^0-9A-Za-z_]+", "_", lbl)
            return lbl

        def analyze_embedding(emb: np.ndarray, plot_tag: str = "", enable_single_plot: bool = True) -> dict:
            results = svc.knn(emb, k=k)
            resp = {'nn': [{'path': p, 'score': s, 'euclidean_distance': ed} for p, s, ed in results], 'ckpt': svc.ckpt_path}

            # Gatekeeper: similarity threshold check
            if not results:
                resp['nn'] = []
                resp['confidence_level'] = 'unknown'
                resp['confidence_score'] = 0.0
                resp['is_known_class'] = False
                resp['similarity_status'] = 'Unknown'
                return resp

            best_score = float(results[0][1])
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
                resp['confidence_level'] = 'unknown'
                resp['confidence_score'] = best_score
                resp['is_known_class'] = False
                resp['similarity_status'] = 'Unknown'

            # Drill-down: derive class labels from neighbor paths and compute distribution
            try:
                labels = [extract_label(p) for p, _, _ in results]
                dist = Counter(labels)
                class_distribution = dict(dist)
                majority_class = dist.most_common(1)[0][0] if len(dist) > 0 else None
                resp['class_distribution'] = class_distribution
                resp['majority_class'] = majority_class

                try:
                    k_effective = max(1, len(results))
                    votes_for_winner = class_distribution.get(majority_class, 0)
                    scores_for_winner = [s for (p, s, ed), lbl in zip(results, labels) if lbl == majority_class]
                    avg_score_winner = float(sum(scores_for_winner) / len(scores_for_winner)) if scores_for_winner else 0.0
                    vote_ratio = float(votes_for_winner) / float(k_effective)
                    weighted_confidence = vote_ratio * avg_score_winner
                    resp['weighted_confidence'] = float(weighted_confidence)
                except Exception:
                    resp['weighted_confidence'] = None
            except Exception:
                pass

            # Attempt to compute 2D projection for input + neighbors 
            try:
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
                        valid_scores.append(results[i][1])
                        valid_euclid_dists.append(results[i][2])

                if len(neigh_embs) > 0 and HAVE_TSNE:
                    q = emb / (np.linalg.norm(emb) + 1e-10)

                    # Prefer gallery embeddings for global maps so we can label clusters by class.
                    background = svc.gallery_embeddings
                    background_paths = list(svc.gallery_paths) if getattr(svc, 'gallery_paths', None) else []
                    if background is None or background.shape[0] == 0:
                        if svc.anchor_embeddings is not None:
                            background = svc.anchor_embeddings
                            background_paths = []
                        else:
                            background = svc.gallery_embeddings
                            background_paths = []

                    # Keep projection latency manageable by capping background points.
                    if tsne_bg_max > 0 and background.shape[0] > tsne_bg_max:
                        bg_idx = np.linspace(0, background.shape[0] - 1, tsne_bg_max, dtype=int)
                        background = background[bg_idx]
                        if len(background_paths) > 0:
                            background_paths = [background_paths[i] for i in bg_idx if i < len(background_paths)]

                    has_centroids = getattr(svc, 'centroids', None) is not None and getattr(svc, 'classes', None) is not None
                    if has_centroids:
                        centroids_np = svc.centroids.cpu().numpy()
                    else:
                        centroids_np = np.zeros((0, q.shape[-1]), dtype=np.float32)

                    arr = np.vstack([
                        background,
                        centroids_np,
                        q[np.newaxis, :],
                        np.vstack(neigh_embs)
                    ])

                    coords, projection_name = _project_to_2d(arr, random_state=0)

                    n_bg = background.shape[0]
                    n_cent = centroids_np.shape[0]
                    bg_coords = coords[:n_bg].tolist()
                    centroid_coordsList = coords[n_bg : n_bg+n_cent].tolist()
                    input_coord = coords[n_bg+n_cent].tolist()
                    neigh_coords_list = coords[n_bg+n_cent+1 :].tolist()

                    centroids_meta = []
                    if has_centroids:
                        # Calculate closest centroid
                        try:
                            t_emb = torch.from_numpy(emb).to(svc.device).unsqueeze(0)
                            t_emb = torch.nn.functional.normalize(t_emb, p=2, dim=1)
                            sims = torch.nn.functional.cosine_similarity(t_emb, svc.centroids)
                            _, c_idx = torch.max(sims, dim=0)
                            closest_class = svc.classes[c_idx.item()]
                        except Exception:
                            closest_class = resp.get('majority_class')

                        for c_name, c_coord in zip(svc.classes, centroid_coordsList):
                            if c_name.lower() == (closest_class or '').lower() or c_name == resp.get('majority_class'):
                                centroids_meta.append({'class': c_name, 'coord': c_coord})

                    resp['tsne_coordinates'] = {
                        'input': input_coord,
                        'neighbors': [
                            {'path': p, 'score': s, 'euclidean_distance': ed, 'coord': c}
                            for p, s, ed, c in zip(valid_paths, valid_scores, valid_euclid_dists, neigh_coords_list)
                        ],
                        'centroids': centroids_meta
                    }

                    if tsne_plot and enable_single_plot:
                        try:
                            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

                            bg_x = [c[0] for c in bg_coords]
                            bg_y = [c[1] for c in bg_coords]
                            ax.scatter(bg_x, bg_y, c='lightgray', s=20, alpha=0.5, label='Context')

                            neigh_x = [c[0] for c in neigh_coords_list]
                            neigh_y = [c[1] for c in neigh_coords_list]
                            ax.scatter(neigh_x, neigh_y, c=valid_scores, cmap='viridis', s=60, edgecolor='k', label='Neighbors')

                            ax.scatter(input_coord[0], input_coord[1], c='red', marker='*', s=200, zorder=10, edgecolor='k', label='Input')
                            ax.set_title(f'Robustness Analysis ({projection_name})\nModel: {os.path.basename(svc.ckpt_path) if getattr(svc, "ckpt_path", None) else "Unknown"}')
                            ax.legend()
                            ax.axis('off')

                            buf = io.BytesIO()
                            plt.tight_layout()
                            fig.savefig(buf, format='png', bbox_inches='tight')
                            plt.close(fig)
                            buf.seek(0)

                            out_dir = Path(ROOT) / 'uploads' / 'tsne_plots'
                            out_dir.mkdir(parents=True, exist_ok=True)
                            suffix = f"_{plot_tag}" if plot_tag else ""
                            fname = f"tsne_{int(time.time()*1000)}{suffix}.png"
                            fpath = (out_dir / fname).resolve()
                            with open(fpath, 'wb') as f:
                                f.write(buf.getvalue())

                            resp['tsne_plot_path'] = str(fpath)
                            resp['tsne_plot_base64'] = base64.b64encode(buf.getvalue()).decode('ascii')
                        except Exception:
                            pass
            except Exception as e:
                print(f"Projection computation failed: {e}")

            return resp

        if all_detections:
            orig_img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert('RGB')
            detections = svc.detect_with_yolo(orig_img, max_detections=max_detections)
            if len(detections) == 0:
                w, h = orig_img.size
                detections = [{
                    'crop': orig_img,
                    'box': (0, 0, w, h),
                    'confidence': None,
                    'class_id': None,
                    'segmentation': None,
                }]

            crops = [d['crop'] for d in detections]
            emb_batch = svc._embed_batch(crops)

            detection_nn = []
            for idx, (det, emb) in enumerate(zip(detections, emb_batch)):
                det_resp = analyze_embedding(emb, plot_tag=f"det{idx}", enable_single_plot=False)
                det_resp['index'] = idx
                det_resp['bbox'] = list(det['box']) if det.get('box') is not None else None
                det_resp['detector_confidence'] = det.get('confidence')
                det_resp['detector_class_id'] = det.get('class_id')
                det_resp['segmentation'] = det.get('segmentation')
                detection_nn.append(det_resp)

            if len(detection_nn) == 0:
                fallback = analyze_embedding(svc.embed_image_bytes(data), plot_tag="det0", enable_single_plot=False)
                fallback['index'] = 0
                detection_nn = [fallback]

            primary = dict(detection_nn[0])
            primary['num_detections'] = len(detection_nn)
            primary['detection_nn'] = detection_nn

            # Build one global projection map containing all detected input fruits.
            if tsne_global and HAVE_TSNE and len(emb_batch) > 0:
                try:
                    q_inputs = np.asarray(emb_batch, dtype=np.float32)
                    q_inputs = q_inputs / (np.linalg.norm(q_inputs, axis=1, keepdims=True) + 1e-10)

                    # Prefer gallery embeddings for global maps so we can label clusters by class.
                    background = svc.gallery_embeddings
                    background_paths = list(svc.gallery_paths) if getattr(svc, 'gallery_paths', None) else []
                    if background is None or background.shape[0] == 0:
                        if svc.anchor_embeddings is not None:
                            background = svc.anchor_embeddings
                            background_paths = []
                        else:
                            background = svc.gallery_embeddings
                            background_paths = []

                    if tsne_bg_max > 0 and background.shape[0] > tsne_bg_max:
                        bg_idx = np.linspace(0, background.shape[0] - 1, tsne_bg_max, dtype=int)
                        background = background[bg_idx]
                        if len(background_paths) > 0:
                            background_paths = [background_paths[i] for i in bg_idx if i < len(background_paths)]

                    # Include union of nearest-neighbor embeddings for richer context.
                    gallery_index_map = {p: i for i, p in enumerate(svc.gallery_paths)}
                    unique_neighbor_paths = []
                    seen_paths = set()
                    for det_resp in detection_nn:
                        for n in (det_resp.get('nn') or []):
                            p = n.get('path') if isinstance(n, dict) else None
                            if p and p in gallery_index_map and p not in seen_paths:
                                seen_paths.add(p)
                                unique_neighbor_paths.append(p)

                    neighbor_embs = [svc.gallery_embeddings[gallery_index_map[p]] for p in unique_neighbor_paths]

                    has_centroids = getattr(svc, 'centroids', None) is not None and getattr(svc, 'classes', None) is not None
                    if has_centroids:
                        centroids_np = svc.centroids.cpu().numpy()
                    else:
                        centroids_np = np.zeros((0, q_inputs.shape[-1]), dtype=np.float32)

                    blocks = [background, centroids_np, q_inputs]
                    if len(neighbor_embs) > 0:
                        blocks.append(np.vstack(neighbor_embs))
                    arr = np.vstack(blocks)

                    if arr.shape[0] >= 2:
                        coords, projection_name = _project_to_2d(arr, random_state=0)

                        n_bg = background.shape[0]
                        n_cent = centroids_np.shape[0]
                        n_in = q_inputs.shape[0]
                        
                        bg_coords = coords[:n_bg]
                        centroid_coords = coords[n_bg : n_bg+n_cent]
                        input_coords = coords[n_bg+n_cent : n_bg+n_cent+n_in]
                        neigh_coords = coords[n_bg+n_cent+n_in : ] if len(neighbor_embs) > 0 else np.zeros((0, 2), dtype=np.float32)

                        centroids_meta = []
                        if has_centroids:
                            active_classes = {det_resp.get('majority_class') for det_resp in detection_nn}
                            try:
                                for q_emb in emb_batch:
                                    t_emb = torch.from_numpy(q_emb).to(svc.device).unsqueeze(0)
                                    t_emb = torch.nn.functional.normalize(t_emb, p=2, dim=1)
                                    sims = torch.nn.functional.cosine_similarity(t_emb, svc.centroids)
                                    _, c_idx = torch.max(sims, dim=0)
                                    active_classes.add(svc.classes[c_idx.item()])
                            except Exception:
                                pass
                                
                            for i, c_name in enumerate(svc.classes):
                                if c_name in active_classes or c_name.lower() in {c.lower() for c in active_classes if c}:
                                    c_coord = centroid_coords[i].tolist()
                                    centroids_meta.append({'class': c_name, 'coord': [float(c_coord[0]), float(c_coord[1])]})

                        # Compute class-cluster label anchor points. If we have master centroids,
                        # anchor the text perfectly on them. Otherwise, fall back to averaging 2D dots.
                        cluster_labels = []
                        if has_centroids:
                            for i, c_name in enumerate(svc.classes):
                                cluster_labels.append({
                                    'label': c_name,
                                    'coord': [float(centroid_coords[i][0]), float(centroid_coords[i][1])],
                                    'count': 1000
                                })
                        elif len(background_paths) == n_bg and n_bg > 0:
                            label_points = {}
                            for pth, coord in zip(background_paths, bg_coords):
                                lbl = extract_label(pth)
                                if not lbl:
                                    continue
                                label_points.setdefault(lbl, []).append(coord)

                            for lbl, pts in label_points.items():
                                # Ignore tiny fragments to reduce text clutter.
                                if len(pts) < 8:
                                    continue
                                arr_pts = np.asarray(pts, dtype=np.float32)
                                center = arr_pts.mean(axis=0)
                                cluster_labels.append({
                                    'label': lbl,
                                    'coord': [float(center[0]), float(center[1])],
                                    'count': int(len(pts)),
                                })

                            cluster_labels.sort(key=lambda x: x['count'], reverse=True)

                        inputs_meta = []
                        for i in range(n_in):
                            det = detections[i] if i < len(detections) else {}
                            det_resp = detection_nn[i] if i < len(detection_nn) else {}
                            inputs_meta.append({
                                'index': i,
                                'bbox': list(det.get('box')) if det.get('box') is not None else None,
                                'detector_confidence': det.get('confidence'),
                                'majority_class': det_resp.get('majority_class'),
                                'coord': [float(input_coords[i][0]), float(input_coords[i][1])],
                            })

                        neighbors_meta = []
                        for i, p in enumerate(unique_neighbor_paths):
                            if i < len(neigh_coords):
                                neighbors_meta.append({
                                    'path': p,
                                    'coord': [float(neigh_coords[i][0]), float(neigh_coords[i][1])],
                                })

                        primary['tsne_global_coordinates'] = {
                            'inputs': inputs_meta,
                            'neighbors': neighbors_meta,
                            'cluster_labels': cluster_labels,
                            'centroids': centroids_meta,
                        }

                        try:
                            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

                            ax.scatter(bg_coords[:, 0], bg_coords[:, 1], c='lightgray', s=16, alpha=0.45, label='Context')
                            if len(neigh_coords) > 0:
                                ax.scatter(neigh_coords[:, 0], neigh_coords[:, 1], c='steelblue', s=32, alpha=0.75, label='NN Pool')

                            for cl in cluster_labels[:30]:
                                x, y = cl['coord']
                                # Draw a dark star for the Master Centroid point if derived from centroids
                                if has_centroids:
                                    ax.scatter(x, y, c='black', marker='*', s=80, alpha=0.6, zorder=5)
                                    
                                ax.text(
                                    float(x),
                                    float(y) + (0.15 if has_centroids else 0.0), # slightly offset so star is visible
                                    str(cl['label']),
                                    fontsize=8,
                                    fontweight='bold',
                                    color='dimgray',
                                    ha='center',
                                    va='center',
                                    zorder=6,
                                    bbox=dict(boxstyle='round,pad=0.2', fc='yellow', ec='none', alpha=0.68),
                                )

                            colors = plt.cm.get_cmap('tab10', max(1, n_in))
                            for i in range(n_in):
                                cls = detection_nn[i].get('majority_class') if i < len(detection_nn) else None
                                cls = cls if cls else f"input_{i+1}"
                                x, y = float(input_coords[i][0]), float(input_coords[i][1])
                                ax.scatter([x], [y], c=[colors(i)], marker='*', s=220, edgecolor='k', linewidths=0.8, zorder=10, label=f"#{i+1} {cls}")
                                ax.text(x + 0.5, y + 0.5, f"#{i+1}", fontsize=8, color='black', zorder=11)

                            ax.set_title(f'Global {projection_name} (All Detected Fruits)\nModel: {os.path.basename(svc.ckpt_path) if getattr(svc, "ckpt_path", None) else "Unknown"}')
                            ax.legend(loc='best', fontsize=7)
                            ax.axis('off')

                            buf = io.BytesIO()
                            plt.tight_layout()
                            fig.savefig(buf, format='png', bbox_inches='tight')
                            plt.close(fig)
                            buf.seek(0)

                            out_dir = Path(ROOT) / 'uploads' / 'tsne_plots'
                            out_dir.mkdir(parents=True, exist_ok=True)
                            fname = f"tsne_global_{int(time.time()*1000)}.png"
                            fpath = (out_dir / fname).resolve()
                            with open(fpath, 'wb') as f:
                                f.write(buf.getvalue())

                            primary['tsne_global_plot_path'] = str(fpath)
                            primary['tsne_global_plot_base64'] = base64.b64encode(buf.getvalue()).decode('ascii')
                            # Keep compatibility with previous tsne_plot_base64 consumers.
                            primary['tsne_plot_base64'] = primary['tsne_global_plot_base64']
                        except Exception as plot_err:
                            print(f"Global projection plot generation failed: {plot_err}")
                except Exception as global_e:
                    print(f"Global projection computation failed: {global_e}")

            return jsonify(_strip_nulls(primary))

        emb = svc.embed_image_bytes(data)
        resp = analyze_embedding(emb)
        return jsonify(_strip_nulls(resp))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/heatmap', methods=['POST', 'OPTIONS'])
def heatmap():
    try:
        # Handle CORS preflight
        if request.method == 'OPTIONS':
            return make_response(('', 200))
        data = request.get_data()
        if not data:
            return jsonify({'error': 'empty body'}), 400
        # support optional OpenCV colored heatmap via ?cv=1 and ?colormap=jet
        # Standardize: always compute the Grad-CAM style heatmap (no mode switching)
        use_cv = str(request.args.get('cv', '0')).lower() in ('1', 'true', 'yes')
        colormap = request.args.get('colormap', 'plasma')
        alpha = float(request.args.get('alpha') or 0.7)
        labels = request.args.get('labels')
        out = svc.heatmap_from_bytes(data, use_cv=use_cv, colormap=colormap, alpha=alpha, labels_str=labels)

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
                    model_name = os.path.basename(svc.ckpt_path) if getattr(svc, 'ckpt_path', None) else "Unknown"
                    try:
                        if HAVE_CV2:
                            img_array = np.frombuffer(buf2, np.uint8)
                            img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                            cv2.putText(img_cv, f"Model: {model_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            _, buf2_encoded = cv2.imencode('.png', img_cv)
                            buf2 = buf2_encoded.tobytes()
                        else:
                            from PIL import ImageDraw
                            tmp_img = Image.open(io.BytesIO(buf2))
                            draw = ImageDraw.Draw(tmp_img)
                            draw.text((10, 10), f"Model: {model_name}", fill=(255, 0, 0))
                            save_buf = io.BytesIO()
                            tmp_img.save(save_buf, format='PNG')
                            buf2 = save_buf.getvalue()
                    except Exception as e:
                        print("Could not draw text on heatmap:", e)

                    fname2 = f"overlay_{int(time.time()*1000)}.png"
                    fpath2 = out_dir / fname2
                    with open(fpath2, 'wb') as f:
                        f.write(buf2)
                    overlay_path = str(fpath2)
                # attempt to open overlay for convenience (best-effort)
                # Removed to prevent popup
                # try:
                #     if overlay_path and sys.platform.startswith('win'):
                #         os.startfile(overlay_path)
                # except Exception:
                #     pass
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


@app.route('/upload-form', methods=['POST', 'OPTIONS'])
def upload_form():
    """Handle image upload from frontend form"""
    try:
        # Handle CORS preflight
        if request.method == 'OPTIONS':
            return make_response(('', 200))

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
                    background = svc.anchor_embeddings if svc.anchor_embeddings is not None else np.zeros((0, emb.shape[1]), dtype=np.float32)
                    arr = np.vstack([background, q[np.newaxis, :], np.vstack(neigh_embs)])
                    n_samples = int(arr.shape[0])
                    tsne_debug['n_samples'] = n_samples
                    print(f"[UPLOAD] t-SNE: n_samples={n_samples}", flush=True)

                    # sklearn requires 1 < perplexity < n_samples
                    if n_samples >= 2:
                        perplexity = min(30, max(1, n_samples - 1))
                        print(f"[UPLOAD] Running t-SNE with perplexity={perplexity}...", flush=True)
                        ts = TSNE(n_components=2, perplexity=perplexity, init='pca', random_state=0)
                        coords = ts.fit_transform(arr)
                        
                        n_bg = background.shape[0]
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

                        # --- PLOT & SAVE TSNE IMAGE ---
                        try:
                            import matplotlib.pyplot as plt
                            import io, time
                            
                            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
                            
                            # Plot Background
                            bg_coords = coords[:n_bg].tolist()
                            if len(bg_coords) > 0:
                                bg_x = [c[0] for c in bg_coords]
                                bg_y = [c[1] for c in bg_coords]
                                ax.scatter(bg_x, bg_y, c='lightgray', s=20, alpha=0.5, label='Context')
                                
                            # Plot Neighbors
                            if len(neigh_coords_list) > 0:
                                neigh_x = [c[0] for c in neigh_coords_list]
                                neigh_y = [c[1] for c in neigh_coords_list]
                                ax.scatter(neigh_x, neigh_y, c=valid_scores, cmap='viridis', s=60, edgecolor='k', label='Neighbors')
                                
                                # Draw dashed lines and distances
                                for i, (neigh_coord, neigh_dist) in enumerate(zip(neigh_coords_list, valid_euclid_dists)):
                                    ax.plot([input_coord[0], neigh_coord[0]], [input_coord[1], neigh_coord[1]], 
                                            color='gray', linestyle='--', linewidth=1.0, alpha=0.7, zorder=4)
                                    
                                    dx = neigh_coord[0] - input_coord[0]
                                    dy = neigh_coord[1] - input_coord[1]
                                    norm = np.hypot(dx, dy) + 1e-5
                                    offset_mag = 20 + (i % 3) * 15 
                                    
                                    ax.annotate(f"{float(neigh_dist):.2f}",
                                                xy=(neigh_coord[0], neigh_coord[1]), 
                                                xytext=((dx/norm) * offset_mag, (dy/norm) * offset_mag), 
                                                textcoords='offset points',
                                                color='black', fontsize=8, ha='center', va='center',
                                                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9, edgecolor='gray', lw=0.5), 
                                                zorder=15 + i)
                            
                            # Plot Input (Red Star)
                            ax.scatter(input_coord[0], input_coord[1], c='red', marker='*', s=200, zorder=10, edgecolor='k', label='Input')
                            
                            ax.set_title(f'Robustness Analysis (t-SNE)\nModel: {os.path.basename(svc.ckpt_path) if getattr(svc, "ckpt_path", None) else "Unknown"}')
                            ax.legend()
                            ax.axis('off')

                            buf = io.BytesIO()
                            plt.tight_layout()
                            fig.savefig(buf, format='png', bbox_inches='tight')
                            plt.close(fig)
                            
                            out_dir = Path(ROOT) / 'uploads' / 'tsne-mapping'
                            out_dir.mkdir(parents=True, exist_ok=True)
                            fname = f"tsne_{int(time.time()*1000)}.png"
                            fpath = out_dir / fname
                            with open(fpath, 'wb') as f:
                                f.write(buf.getvalue())
                                
                            print(f"[UPLOAD] Saved t-SNE plot to: {fpath}", flush=True)
                        except Exception as plot_e:
                            print(f"[UPLOAD] Warning: failed to save t-SNE plot - {plot_e}", flush=True)
                        # ------------------------------

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

        response = {
            'analysis': {
                'labels': labels,
            }
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
                    print(f"Reload: Loaded {svc.anchor_embeddings.shape[0]} anchors")
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
    p.add_argument('--classifier-path', type=str, default=None, help='Path to classifier checkpoint')
    p.add_argument('--centroids-path', type=str, default='fruit_centroids.pt', help='Path to the centroids pt file')
    p.add_argument('--gallery-root', default=None)
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=8001)
    p.add_argument('--device', type=str, default='cuda', help='Device to use')
    p.add_argument('--num-classes', type=int, default=16, help='Number of classes')
    p.add_argument('--embedding-dim', type=int, default=128, help='Dimension of the embedding space')
    p.add_argument('--hidden-dim', type=int, default=512, help='Dimension of the hidden layer')
    p.add_argument('--backbone', type=str, default='resnet18', help='Backbone model')
    args = p.parse_args()

    svc = PythonModelService(gallery_root=args.gallery_root,
                             device=args.device,
                             embedding_dim=args.embedding_dim,
                             hidden_dim=args.hidden_dim,
                             num_classes=args.num_classes,
                             backbone=args.backbone,
                             centroids_path=args.centroids_path)
    if args.ckpt:
        svc.load_model(args.ckpt, classifier_path=args.classifier_path)

    # build gallery in background thread to allow immediate responses
    t = threading.Thread(target=svc.build_gallery, kwargs={'rebuild': False}, daemon=True)
    t.start()
    app.run(host=args.host, port=args.port)