#!/usr/bin/env python3
"""
Multi-color line-art tracer - T-Tracer
Fixes Script 1's single-threshold limit: quantizes into N color buckets,
then runs the proven smoothing engine once PER bucket, so each color
gets its own clean smoothed shape instead of collapsing into one mask.

Drop-folder usage: no arguments needed. Put image(s) next to this script and run it —
each one gets traced to a same-named .svg right beside it.
    python multicolor_trace_v1.py
    python multicolor_trace_v1.py --colors 4
"""

import argparse
from pathlib import Path
import numpy as np
from smoothing_engine import smooth_mask, build_svg, remove_small_blobs, flatten_to_white

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def auto_palette(img_rgb, n_colors, merge_dist=24.0):
    flat = img_rgb.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    order = np.argsort(-counts)
    colors, counts = colors[order], counts[order]
    palette = []
    for c in colors:
        if len(palette) >= n_colors:
            break
        c = c.astype(float)
        if all(np.linalg.norm(c - np.array(p, dtype=float)) > merge_dist for p in palette):
            palette.append(tuple(int(round(x)) for x in c))
    return np.array(palette)


def quantize(img_rgb, palette):
    h, w, _ = img_rgb.shape
    flat = img_rgb.reshape(-1, 3).astype(int)
    dists = np.linalg.norm(flat[:, None, :] - palette[None, :, :], axis=2)
    nearest = np.argmin(dists, axis=1)
    return palette[nearest].reshape(h, w, 3).astype(np.uint8)


def run(input_path, output_path, n_colors=4, scale=6, smoothing=3, min_points=20, background_color=None):
    src = flatten_to_white(input_path)
    img_rgb = np.array(src)
    display_w, display_h = src.size

    palette = auto_palette(img_rgb, n_colors)
    quant = quantize(img_rgb, palette)

    # background = most frequent color (assume it's the base/canvas), skip tracing it
    flat = quant.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    bg_color = tuple(colors[np.argmax(counts)]) if background_color is None else background_color

    layers = []
    canvas_w = canvas_h = None
    for color in palette:
        color_t = tuple(int(c) for c in color)
        if color_t == bg_color:
            continue
        mask = np.all(quant == np.array(color_t), axis=-1)
        if mask.sum() < 30:
            continue
        mask = remove_small_blobs(mask, min_blob_size=100)
        if mask.sum() < 30:
            print(f'  color #{color_t}: discarded as noise (all blobs < 100px)')
            continue
        contours, cw, ch = smooth_mask(mask, scale=scale, smoothing_per_point=smoothing, min_points=min_points)
        canvas_w, canvas_h = cw, ch
        hexcolor = '#%02X%02X%02X' % color_t
        layers.append((contours, hexcolor))
        print(f'  color {hexcolor}: {mask.sum()} px -> {len(contours)} contours')

    bg_hex = '#%02X%02X%02X' % bg_color
    svg = build_svg(layers, canvas_w, canvas_h, display_w, display_h, background=bg_hex)

    with open(output_path, 'w') as f:
        f.write(svg)
    print(f'{len(layers)} color layers written -> {output_path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Multi-color line-art tracer - drop image(s) next to this script and run')
    ap.add_argument('--colors', type=int, default=4)
    ap.add_argument('--scale', type=int, default=6)
    ap.add_argument('--smoothing', type=float, default=3)
    ap.add_argument('--min-points', type=int, default=20)
    args = ap.parse_args()

    folder = Path(__file__).resolve().parent
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    if not images:
        print(f'No image files found next to this script ({folder}). Drop a .png/.jpg here and rerun.')
    for img_path in images:
        out_path = img_path.with_suffix('.svg')
        print(f'-- {img_path.name} -> {out_path.name}')
        run(str(img_path), str(out_path), args.colors, args.scale, args.smoothing, args.min_points)
