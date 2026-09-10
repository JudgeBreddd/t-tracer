#!/usr/bin/env python3
"""
Line-art tracer (Script 1) - T-Tracer
For clean reference art: ropes, banners, stars, single-color line art.
Not for shaded/multi-material logos (use the complex pipeline for those).

Pipeline: threshold -> supersample -> extract contours -> smooth via spline fit -> build SVG
Note: vtracer is NOT used in this version. Direct contour extraction + scipy spline
fitting outperformed vtracer's own curve fit for clean line art in testing.

Drop-folder usage: no arguments needed. Put image(s) next to this script and run it —
each one gets traced to a same-named .svg right beside it.
    python lineart_trace_v1.py
    python lineart_trace_v1.py --threshold 200 --scale 6 --smoothing 3
"""

import argparse
from pathlib import Path
import numpy as np
from smoothing_engine import supersample, extract_smoothed_contours, flatten_to_white

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def threshold_mask(image_path, threshold):
    gray = np.array(flatten_to_white(image_path).convert('L'))
    return gray < threshold


def build_svg(smoothed_contours, canvas_w, canvas_h, display_w, display_h, fill='black'):
    """canvas_w/h = internal (supersampled) coordinate space.
    display_w/h = final on-disk size (original image size)."""
    paths = []
    for x_s, y_s in smoothed_contours:
        d = 'M ' + ' L '.join(f'{px:.2f},{py:.2f}' for px, py in zip(x_s, y_s)) + ' Z'
        paths.append(d)

    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{display_w}" height="{display_h}" viewBox="0 0 {canvas_w} {canvas_h}">'
    svg += f'<rect width="{canvas_w}" height="{canvas_h}" fill="white"/>'
    svg += f'<path fill-rule="evenodd" fill="{fill}" d="' + ' '.join(paths) + '"/>'
    svg += '</svg>'
    return svg


def run(input_path, output_path, threshold=200, scale=6, smoothing=3, min_points=20):
    display_w, display_h = flatten_to_white(input_path).size

    mask = threshold_mask(input_path, threshold)
    big_mask = supersample(mask, scale)
    smoothed = extract_smoothed_contours(big_mask, smoothing_per_point=smoothing, min_points=min_points)

    canvas_h, canvas_w = big_mask.shape
    svg = build_svg(smoothed, canvas_w, canvas_h, display_w, display_h)

    with open(output_path, 'w') as f:
        f.write(svg)

    print(f'{len(smoothed)} smoothed contours written -> {output_path}')
    print(f'display size: {display_w}x{display_h}  internal scale: {scale}x')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Line-art tracer (Script 1) - drop image(s) next to this script and run')
    ap.add_argument('--threshold', type=int, default=200, help='brightness cutoff for line vs background (0-255)')
    ap.add_argument('--scale', type=int, default=6, help='supersample factor before contour extraction')
    ap.add_argument('--smoothing', type=float, default=3, help='spline smoothing strength per point (higher = smoother, less exact)')
    ap.add_argument('--min-points', type=int, default=20, help='discard contours with fewer raw points than this (noise filter)')
    args = ap.parse_args()

    folder = Path(__file__).resolve().parent
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    if not images:
        print(f'No image files found next to this script ({folder}). Drop a .png/.jpg here and rerun.')
    for img_path in images:
        out_path = img_path.with_suffix('.svg')
        print(f'-- {img_path.name} -> {out_path.name}')
        run(str(img_path), str(out_path), args.threshold, args.scale, args.smoothing, args.min_points)
