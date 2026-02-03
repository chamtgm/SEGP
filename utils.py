import random
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torch
import cv2


def pil_to_numpy(img: Image.Image):
    return np.array(img).astype(np.float32) / 255.0


def numpy_to_pil(x: np.ndarray):
    x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(x)


def match_channel_stats(content: Image.Image, style: Image.Image):
    """
    Simple AdaIN-like style transfer: match per-channel mean and std from style to content.
    Inputs are PIL Images. Returns a PIL Image.
    This is a lightweight approximation that changes color statistics (useful to emulate camera style differences).
    """
    c = pil_to_numpy(content)
    s = pil_to_numpy(style)

    # if grayscale or different shapes, convert
    if c.ndim == 2:
        c = np.stack([c, c, c], axis=-1)
    if s.ndim == 2:
        s = np.stack([s, s, s], axis=-1)

    # resize style to content size
    if s.shape[:2] != c.shape[:2]:
        style = style.resize((content.width, content.height), Image.BILINEAR)
        s = pil_to_numpy(style)

    return numpy_to_pil(_match_channel_stats_np(c, s))


def _match_channel_stats_np(c: np.ndarray, s: np.ndarray, mask: np.ndarray = None):
    """
    Core numpy implementation of match_channel_stats. If mask is provided it should be a 2D
    float array in [0,1] with same HxW as c; statistics are computed over masked foreground only
    for the content and over the full style image.
    """
    # compute mean/std per channel
    if mask is None:
        c_mean = c.reshape(-1, 3).mean(axis=0)
        c_std = c.reshape(-1, 3).std(axis=0) + 1e-6
    else:
        # mask shaped HxW, select pixels
        m = mask.reshape(-1) > 0.001
        if m.sum() < 10:
            # fallback to global stats if mask too small
            c_mean = c.reshape(-1, 3).mean(axis=0)
            c_std = c.reshape(-1, 3).std(axis=0) + 1e-6
        else:
            pixels = c.reshape(-1, 3)[m]
            c_mean = pixels.mean(axis=0)
            c_std = pixels.std(axis=0) + 1e-6

    s_mean = s.reshape(-1, 3).mean(axis=0)
    s_std = s.reshape(-1, 3).std(axis=0) + 1e-6

    # normalize content and apply style stats
    normalized = (c - c_mean) / c_std
    transferred = normalized * s_std + s_mean

    return transferred


def reinhard_color_transfer(content: Image.Image, style: Image.Image):
    """
    Reinhard color transfer in LAB color space (from Reinhard et al.).
    Transfers mean and std per channel in LAB space.
    Returns a PIL Image.
    """
    c = pil_to_numpy(content)
    s = pil_to_numpy(style)

    # ensure 3-channel
    if c.ndim == 2:
        c = np.stack([c, c, c], axis=-1)
    if s.ndim == 2:
        s = np.stack([s, s, s], axis=-1)

    # resize style to content size
    if s.shape[:2] != c.shape[:2]:
        style = style.resize((content.width, content.height), Image.BILINEAR)
        s = pil_to_numpy(style)

    # convert to 0-255 uint8 BGR for cv2
    c_bgr = (np.clip(c * 255.0, 0, 255).astype(np.uint8))[..., ::-1]
    s_bgr = (np.clip(s * 255.0, 0, 255).astype(np.uint8))[..., ::-1]

    # convert to LAB
    c_lab = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    s_lab = cv2.cvtColor(s_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # compute mean/std per channel
    c_mean, c_std = c_lab.reshape(-1, 3).mean(axis=0), c_lab.reshape(-1, 3).std(axis=0) + 1e-6
    s_mean, s_std = s_lab.reshape(-1, 3).mean(axis=0), s_lab.reshape(-1, 3).std(axis=0) + 1e-6

    # transfer
    result_lab = (c_lab - c_mean) / c_std * s_std + s_mean

    # clip and convert back to BGR
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    # convert back to RGB PIL
    result_rgb = result_bgr[..., ::-1]
    return numpy_to_pil(result_rgb.astype(np.float32) / 255.0)


def feather_mask(mask_pil: Image.Image, radius: int = 15):
    """
    Feather/soften a binary mask using a Gaussian blur. Returns float32 numpy mask in [0,1].
    Accepts PIL mask (L or 1) or numpy array.
    """
    if not isinstance(mask_pil, Image.Image):
        mask = np.array(mask_pil).astype(np.float32)
    else:
        mask = np.array(mask_pil.convert('L')).astype(np.float32)
    mask = mask / 255.0
    if radius <= 0:
        return mask
    # use OpenCV GaussianBlur; convert to 0-255 uint8, blur, and rescale
    mask_u = (np.clip(mask * 255.0, 0, 255).astype(np.uint8))
    k = max(3, int(radius) // 2 * 2 + 1)
    blurred = cv2.GaussianBlur(mask_u, (k, k), sigmaX=radius)
    return blurred.astype(np.float32) / 255.0


def apply_masked_color_transfer(content: Image.Image, style: Image.Image, mask: Image.Image = None, method: str = 'reinhard', feather_radius: int = 15):
    """
    Apply color transfer (reinhard or match_channel_stats) to the content image only on the
    foreground specified by mask. The mask will be feathered with `feather_radius` to soften edges.
    Returns a PIL Image with the same size as content.
    """
    c = pil_to_numpy(content)
    s = pil_to_numpy(style)

    # resize style to content size if needed
    if s.shape[:2] != c.shape[:2]:
        style = style.resize((content.width, content.height), Image.BILINEAR)
        s = pil_to_numpy(style)

    if mask is None:
        mask_arr = None
    else:
        mask_arr = feather_mask(mask, radius=feather_radius)

    if method == 'reinhard':
        # reinhard works in LAB; we can compute full-image transfer then blend using mask
        transferred = pil_to_numpy(reinhard_color_transfer(content, style))
    else:
        transferred = _match_channel_stats_np(c, s, mask=mask_arr)

    if mask_arr is None:
        out = transferred
    else:
        # blend only in masked region
        m = mask_arr[..., None]
        out = transferred * m + c * (1.0 - m)

    return numpy_to_pil(out)


def find_images(root_dir):
    """Recursively find image file paths (jpg/png/jpeg) under root_dir."""
    import os, fnmatch
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    matches = []
    for root, _, files in os.walk(root_dir):
        for pattern in exts:
            for f in fnmatch.filter(files, pattern):
                matches.append(os.path.join(root, f))
    return matches


def ensure_pil(img):
    if isinstance(img, Image.Image):
        return img
    return Image.fromarray(img)
