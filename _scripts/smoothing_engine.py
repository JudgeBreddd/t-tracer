"""
Shared smoothing engine - T-Tracer
Core reusable piece: takes ANY binary mask, returns clean smoothed contours.
Used by both the single-color line tracer and the multi-color pipeline,
so a fix/tune here benefits both instead of drifting into two copies.
"""

import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import splprep, splev


def flatten_to_white(image_path):
    """Composite any transparency onto white, return a flat RGB image.
    Without this, a transparent PNG's alpha gets silently dropped by a plain
    .convert('RGB')/.convert('L'), and transparent pixels (RGBA 0,0,0,0) read
    as solid black - corrupting thresholding on script 1 and poisoning the
    color palette on script 2 (a real background reads as a fake 'black' layer)."""
    im = Image.open(image_path).convert('RGBA')
    bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, im).convert('RGB')


def supersample(mask, scale):
    """Upscale with cubic interpolation, then re-threshold. Preserves true line
    width - fixes the thin-line problem from tracing at native resolution."""
    big = cv2.resize((mask * 255).astype(np.uint8), None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_CUBIC)
    return big > 127


def extract_smoothed_contours(mask, smoothing_per_point=3, min_points=20, sample_density=4):
    """Extract raw pixel contours, fit a smoothing spline through each one.
    Replaces vtracer's own bezier fit, which follows pixel-staircase steps
    too faithfully and produces faceted/jagged curves."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    smoothed = []
    for c in contours:
        pts = c.reshape(-1, 2).astype(float)
        if len(pts) < min_points:
            continue
        x, y = pts[:, 0], pts[:, 1]
        try:
            tck, u = splprep([x, y], s=len(pts) * smoothing_per_point, per=True, k=3)
        except Exception:
            continue
        n_out = max(200, len(pts) // sample_density)
        u_fine = np.linspace(0, 1, n_out)
        x_s, y_s = splev(u_fine, tck)
        smoothed.append((x_s, y_s))
    return smoothed


def smooth_mask(mask, scale=6, smoothing_per_point=3, min_points=20, sample_density=4):
    """One-call convenience: mask in, smoothed contours out (in the
    SUPERSAMPLED coordinate space - caller must track canvas_w/h from mask.shape * scale)."""
    big_mask = supersample(mask, scale)
    contours = extract_smoothed_contours(big_mask, smoothing_per_point, min_points, sample_density)
    canvas_h, canvas_w = big_mask.shape
    return contours, canvas_w, canvas_h


def build_svg(layers, canvas_w, canvas_h, display_w, display_h, background='white'):
    """layers = list of (smoothed_contours, fill_color_hex) tuples, one per material.
    Each layer's contours must already be in the same canvas_w x canvas_h coordinate space."""
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{display_w}" height="{display_h}" viewBox="0 0 {canvas_w} {canvas_h}">'
    svg += f'<rect width="{canvas_w}" height="{canvas_h}" fill="{background}"/>'
    for contours, fill in layers:
        if not contours:
            continue
        paths = []
        for x_s, y_s in contours:
            d = 'M ' + ' L '.join(f'{px:.2f},{py:.2f}' for px, py in zip(x_s, y_s)) + ' Z'
            paths.append(d)
        svg += f'<path fill-rule="evenodd" fill="{fill}" d="' + ' '.join(paths) + '"/>'
    svg += '</svg>'
    return svg


def remove_small_blobs(mask, min_blob_size=100):
    """Drop disconnected regions below min_blob_size - kills scattered
    anti-aliasing speckle that auto-palette can mistake for a real color."""
    from skimage.measure import label
    labels = label(mask, connectivity=2)
    sizes = np.bincount(labels.ravel())
    keep = np.isin(labels, np.where(sizes >= min_blob_size)[0])
    return mask & keep
