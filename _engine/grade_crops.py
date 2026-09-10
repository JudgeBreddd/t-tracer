#!/usr/bin/env python3
"""grade_crops.py - T-Tracer

Label small crops of a traced result DEFECT or CLEAN, so a measurement can be
checked against Tyler's eye instead of against three regions Claude guessed.

Why this exists. Five different width/waviness measures were built on 2026-09-08.
Every one passed its synthetics and every one failed to separate the three spots
Tyler circled on the winged-roundel emblem from the spots he did not. The regions being
tested were BOXES CLAUDE DREW after reading circles off a compressed screenshot
- so a disagreement could equally mean the metric was wrong or the boxes were.
No amount of tuning resolves that, and tuning against three guessed boxes is
overfitting to n=3.

This produces real labels at real coordinates. Same interaction as pick.py: one
contact sheet opened in the image viewer, answers typed in the terminal.

    ./.venv/bin/python _engine/grade_crops.py <image.png|.svg> [--n 40]

Output: <out>/labels.jsonl - one row per crop, with its bbox, so any metric can
be scored against it later. Crops are sampled ON THE SKELETON so every tile
contains linework; an unbiased grid would mostly show blank stock.
"""
import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np


def _open(path):
    for cmd in (['xdg-open', str(path)], ['open', str(path)]):
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            return
        except FileNotFoundError:
            continue


def load_ink(path, width=1280):
    """Render an SVG, or load a raster, and return (bgr, ink_mask)."""
    path = Path(path)
    if path.suffix.lower() == '.svg':
        tmp = path.with_suffix('.grade.png')
        subprocess.run(['rsvg-convert', '-w', str(width), str(path),
                        '-o', str(tmp)], check=True)
        im = cv2.imread(str(tmp), cv2.IMREAD_UNCHANGED)
        tmp.unlink(missing_ok=True)
    else:
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        sys.exit(f'could not read {path}')
    if im.ndim == 3 and im.shape[2] == 4:
        a = im[:, :, 3:4] / 255.0
        im = (im[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return im, cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) < 128


def sample_centres(ink, n, min_sep, seed=0):
    """Well-spread points on the medial axis - every crop contains a line."""
    from skimage.morphology import medial_axis
    skel = medial_axis(ink)
    ys, xs = np.nonzero(skel)
    if len(ys) == 0:
        sys.exit('no linework found')
    idx = np.random.default_rng(seed).permutation(len(ys))
    picked = []
    for i in idx:
        p = (int(xs[i]), int(ys[i]))
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > min_sep ** 2
               for q in picked):
            picked.append(p)
        if len(picked) >= n:
            break
    return picked


def build_sheet(im, centres, size, zoom, cols):
    tiles = []
    half = size // 2
    h, w = im.shape[:2]
    boxes = []
    for k, (cx, cy) in enumerate(centres, 1):
        x0 = max(0, min(w - size, cx - half))
        y0 = max(0, min(h - size, cy - half))
        boxes.append((x0, y0, x0 + size, y0 + size))
        t = im[y0:y0 + size, x0:x0 + size]
        t = cv2.resize(t, (size * zoom, size * zoom),
                       interpolation=cv2.INTER_NEAREST)
        t = cv2.copyMakeBorder(t, 46, 10, 10, 10, cv2.BORDER_CONSTANT,
                               value=(255, 255, 255))
        cv2.rectangle(t, (10, 46), (t.shape[1] - 10, t.shape[0] - 10),
                      (200, 200, 200), 1)
        cv2.putText(t, str(k), (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, 0, 220), 3)
        tiles.append(t)
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        while len(row) < cols:
            row.append(np.full_like(tiles[0], 255))
        rows.append(np.hstack(row))
    return np.vstack(rows), boxes


def parse_list(s, n):
    out = set()
    for tok in s.replace(',', ' ').split():
        if '-' in tok[1:]:
            a, _, b = tok.partition('-')
            try:
                out.update(range(int(a), int(b) + 1))
            except ValueError:
                continue
        elif tok.isdigit():
            out.add(int(tok))
    return {i for i in out if 1 <= i <= n}


def grade_one(src, out, args, label=None):
    """Build one sheet, ask, append labels. Shared by single and batch modes."""
    im, ink = load_ink(src, args.width)
    centres = sample_centres(ink, args.n, args.min_sep, args.seed)
    sheet, boxes = build_sheet(im, centres, args.size, args.zoom, args.cols)
    name = label or src.stem
    sheet_path = out / f'{name}-crops.png'
    cv2.imwrite(str(sheet_path), sheet)
    _open(sheet_path)

    n = len(boxes)
    print(f'  {n} crops  ({im.shape[1]}x{im.shape[0]})   sheet: {sheet_path.name}')
    print('  DEFECT = would stop you shipping.  UNSURE = would ship, not perfect.')
    def ask(prompt):
        try:
            return input(prompt)
        except EOFError:                      # Ctrl-D ends the session cleanly
            raise KeyboardInterrupt
    bad = parse_list(ask('  DEFECT tiles > '), n)
    unsure = parse_list(ask('  UNSURE tiles > '), n)
    bad -= unsure
    note = ask('  note (optional) > ').strip()

    rows = [{
        'date': date.today().isoformat(), 'image': name,
        'candidate': src.stem, 'crop': k, 'bbox': list(box), 'centre': list(c),
        'label': 'unsure' if k in unsure else ('defect' if k in bad else 'clean'),
        'source': 'owner',
    } for k, (box, c) in enumerate(zip(boxes, centres), 1)]
    with (out / 'labels.jsonl').open('a') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
        if note:
            fh.write(json.dumps({'date': date.today().isoformat(),
                                 'image': name, 'candidate': src.stem,
                                 'note': note, 'source': 'owner'}) + '\n')
    d = sum(r['label'] == 'defect' for r in rows)
    u = sum(r['label'] == 'unsure' for r in rows)
    print(f'  -> {d} defect / {n - d - u} clean / {u} unsure\n')
    return rows


def top_candidate(job_dir):
    """The SVG the app would have put in front of Tyler first."""
    import candidates as C
    mf = job_dir / 'metrics.json'
    if not mf.is_file():
        return None
    try:
        metrics = json.loads(mf.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    ranked = C.rank_candidates(metrics)
    for name, _ in ranked:
        svg = job_dir / f'{name}.svg'
        if svg.is_file():
            return svg
    return None


def run_batch(args):
    """Grade every image in a candidates output folder, one sheet at a time."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    root = Path(args.batch)
    jobs = sorted(d for d in root.iterdir()
                  if d.is_dir() and not d.name.startswith('_'))
    picks = [(d, top_candidate(d)) for d in jobs]
    picks = [(d, s) for d, s in picks if s]
    if not picks:
        sys.exit(f'no traced candidates found under {root}')

    out = Path(args.out) if args.out else root / '_grading'
    out.mkdir(parents=True, exist_ok=True)
    print(f'\n{len(picks)} images to grade. Ctrl-C stops; everything already '
          f'answered is saved.\n')
    for i, (job, svg) in enumerate(picks, 1):
        print(f'--- {i}/{len(picks)}  {job.name}   [{svg.stem}] ---')
        try:
            grade_one(svg, out, args, label=job.name)
        except KeyboardInterrupt:
            print('\n  stopped.')
            break
    print(f'\nlabels -> {out / "labels.jsonl"}')


def main():
    ap = argparse.ArgumentParser(description='Label traced crops defect/clean.')
    ap.add_argument('image', nargs='?', help='traced .svg or raster')
    ap.add_argument('--batch', metavar='DIR',
                    help='a candidates output folder: grade every image in it, '
                         'using each image\'s TOP-RANKED candidate. Ranking comes '
                         'from candidates.rank_candidates, the same one the '
                         'contact sheet and pick.py use, so the tile you grade '
                         'is the one the app would have shown you first.')
    ap.add_argument('--n', type=int, default=40, help='number of crops')
    ap.add_argument('--size', type=int, default=150, help='crop size in px')
    ap.add_argument('--zoom', type=int, default=3)
    ap.add_argument('--cols', type=int, default=8)
    ap.add_argument('--min-sep', type=int, default=90,
                    help='minimum spacing between crop centres')
    ap.add_argument('--width', type=int, default=1280,
                    help='render width for SVG input (the ship size)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    if args.batch:
        return run_batch(args)
    if not args.image:
        ap.error('give an image, or --batch DIR')

    src = Path(args.image)
    out = Path(args.out) if args.out else src.parent / 'grading'
    out.mkdir(parents=True, exist_ok=True)
    try:
        grade_one(src, out, args)
    except KeyboardInterrupt:
        print('\n  stopped.')
    print(f'labels -> {out / "labels.jsonl"}')


if __name__ == '__main__':
    main()
