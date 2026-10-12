#!/usr/bin/env python3
"""
defects.py - T-Tracer

Hygiene at the size the part is actually burned, plus the named-defect legend.

Why this exists (2026-10-02): hygiene scored the 1600px trace raster with no
physical scale, while the reviewer judged a differently-sized preview - the
two were literally looking at different pixels. This module renders each
emitted SVG ONCE at RENDER_PX on its long edge, scores that raster with
hygiene's own measurements as if the long edge were 70 mm and again as if it
were 150 mm (300 lines/cm engraving; 200x200 mm bed, 150 mm practical max),
and writes the same raster out as the full-size preview the reviewer judges.

The weights are hygiene's, unchanged on purpose: this round is the baseline
the per-defect scoreboard measures against.
"""

import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import hygiene

RENDER_PX = 4500                 # 150 mm at 300 lines/cm; timed 2026-10-02, ~6 s/candidate
SIZES_MM = (70, 150)

# One word, one meaning. "blob" and "wavy" each meant two things once and cost
# a week (2026-09-08); every box here has exactly one definition.
LEGEND = [
    ('gap', 'Gap', 'a break in a line, or shapes that touch in the source do not touch'),
    ('stray', 'Stray', 'a detached black mark that is not in the source'),
    ('blotch', 'Blotch', 'a black lump stuck ON a line (was "blob")'),
    ('fused', 'Fused', 'two separate things merged, no white between them'),
    ('edge', 'Edge', 'uneven black/white boundary: swelling, width jumps, kinks (was "wavy")'),
    ('stairstep', 'Stairstep', 'fine pixel steps along a curve that should be smooth'),
    # Split from Stray 2026-10-02: the NMRTU anchor's "strays" were the gold
    # anchor's shading turned into black slivers - in the source, inside one shape.
    ('shading', 'Shading', 'marks where the original has a shadow, highlight or gradient inside one shape'),
    ('missing', 'Missing', 'a whole element of the source is absent'),
    ('thin', 'Thin', 'a line too thin to burn, or only partly drawn'),
    # Split from Polarity 2026-10-02 (Council tier 1 + Tyler): a whole-image
    # inversion is NOT a defect - he shipped Asset 10 both ways.
    ('collapse', 'Contrast collapse', 'two neighbouring parts that differ in the original came out the same colour, so one disappears into the other'),
    ('fill', 'Fill loss', 'a solid shape came out as an outline only'),
    ('other', 'Other', 'anything else - say what in the text box'),
]

# Which hygiene penalty answers which box. Boxes not listed have no detector yet.
DETECTORS = {'gap': 'gaps', 'stray': 'strays', 'fused': 'fused', 'stairstep': 'jagged'}

IMAGE_TYPES = ['bw', 'multicolour', 'mid-luminance', 'low-res', 'photo-3d']


# ---------------------------------------------------------------------------
# Reading our own SVGs back
# ---------------------------------------------------------------------------

_PATH = re.compile(r'<path\b[^>]*?fill="(#[0-9A-Fa-f]{6})"[^>]*?\bd="([^"]*)"')
_NUM = re.compile(r'-?\d+(?:\.\d+)?')


def read_svg(svg_text):
    """(w, h, paths) from an SVG written by vector.paths_to_svg[_layered].

    Only the dark fills are kept - the laser burns ink, and a layered file's
    light layers are not ink. Parses exactly the M/C/Z form our writer emits;
    it is not a general SVG reader and does not need to be.
    """
    vb = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg_text)
    w, h = int(float(vb.group(1))), int(float(vb.group(2)))
    paths = []
    for fill, d in _PATH.findall(svg_text):
        r, g, b = (int(fill[i:i + 2], 16) for i in (1, 3, 5))
        if 0.299 * r + 0.587 * g + 0.114 * b >= 128:
            continue
        for sub in d.split('M')[1:]:
            nums = np.array([float(x) for x in _NUM.findall(sub)]).reshape(-1, 2)
            p, segs = nums[0], []
            for i in range(1, len(nums) - 2, 3):
                c1, c2, p3 = nums[i], nums[i + 1], nums[i + 2]
                segs.append((p, c1, c2, p3))
                p = p3
            if segs:
                paths.append(segs)
    return w, h, paths


def rasterize_at(paths, w, h, long_px=RENDER_PX):
    """vector.rasterize with a fractional scale: the delivered Beziers drawn
    onto a canvas whose long edge is `long_px`. Even-odd, like the SVG."""
    from vector import _sample_beziers
    s = long_px / max(w, h)
    H, W = int(round(h * s)), int(round(w * s))
    canvas = np.zeros((H, W), np.uint8)
    layer = np.zeros_like(canvas)
    for segs in paths:
        pts = _sample_beziers(segs, per_seg=16) * s
        if len(pts) < 3:
            continue
        layer.fill(0)
        cv2.fillPoly(layer, [pts.astype(np.int32)], 255)
        cv2.bitwise_xor(canvas, layer, dst=canvas)
    return canvas > 0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_raster(ink, src_rgb, paths, xing, sizes_mm=SIZES_MM):
    """hygiene's report for one rendered candidate, once per physical size.

    The size-free measurements run once; only the ones with a physical
    threshold (gap width, nodes per mm) are repeated per size.
    """
    src_rgb = cv2.resize(src_rgb, (ink.shape[1], ink.shape[0]), interpolation=cv2.INTER_AREA)
    src_gray = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2GRAY)
    comps = hygiene.measure_components(ink)
    fuse = hygiene.measure_fusions(ink, src_rgb)
    jag = hygiene.measure_jaggedness(ink)
    fid = hygiene.measure_fidelity(ink, src_gray)
    xing = {'self_intersecting_paths': int(xing or 0)}
    nodes = sum(len(segs) for segs in paths)
    out = {}
    for mm in sizes_mm:
        mm_per_px = mm / max(ink.shape)
        gap_px = max(1, int(round(0.12 / mm_per_px)))     # hygiene.score's 0.12 mm
        gaps = hygiene.measure_gaps(ink, src_gray, gap_px)
        out[f'{mm}mm'] = hygiene._assemble(ink, paths, nodes, comps, gaps, fuse,
                                           jag, xing, fid, mm_per_px, 1)
    return out


def image_type(rgb, src_long):
    """One coarse category per source image, by fixed rules.

    ponytail: hand-set thresholds, a guess until the reviewer's overrides say
    how often it is wrong; replace with whatever those overrides support.
    """
    small = cv2.resize(rgb, (256, max(1, int(256 * rgb.shape[0] / rgb.shape[1]))),
                       interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0] * 100 / 255
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    q = (small // 32).reshape(-1, 3)
    _, counts = np.unique(q, axis=0, return_counts=True)
    top8 = float(np.sort(counts)[::-1][:8].sum() / counts.sum())
    ev = {'long_px': int(src_long), 'top8_colour_share': round(top8, 3),
          'grey_share': round(float((chroma < 12).mean()), 3),
          'mid_lum_share': round(float(((L > 30) & (L < 70)).mean()), 3)}
    if top8 < 0.7:
        t = 'photo-3d'
    elif src_long < 600:
        t = 'low-res'
    elif ev['grey_share'] > 0.95 and ev['mid_lum_share'] < 0.10:
        t = 'bw'
    elif ev['mid_lum_share'] > 0.25:
        t = 'mid-luminance'
    else:
        t = 'multicolour'
    return t, ev


def _cache_dir():
    # Beside the app's work dir, never in the synced repo. TT_SCORE_CACHE=off disables it.
    if env := os.environ.get('TT_SCORE_CACHE'):
        return None if env == 'off' else Path(env)
    if env := os.environ.get('TT_WORK_DIR'):
        return Path(env).expanduser().parent / 'score-cache'
    base = (os.environ.get('LOCALAPPDATA', Path.home() / 'AppData/Local') if sys.platform == 'win32'
            else os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
    return Path(base) / 't-tracer' / 'score-cache'


# The detector code is part of the key, so editing a detector re-scores everything.
_CODE = hashlib.sha256(b''.join(Path(m.__file__).read_bytes() for m in (sys.modules[__name__], hygiene))
                       ).hexdigest()[:16]
CACHE_KEEP = 3000  # ponytail: newest-N prune by mtime; switch to size-based if hi.png files get big


def _prune(cache):
    entries = sorted(cache.glob('*.json'), key=lambda p: p.stat().st_mtime)
    for j in entries[:max(0, len(entries) - CACHE_KEEP)]:
        j.unlink(missing_ok=True)
        j.with_suffix('.png').unlink(missing_ok=True)


def score_image(out_dir, rgb, src_long, metrics):
    """Write <out_dir>/defects.json and <name>.hi.png for every emitted
    candidate. `rgb` is the traced source; `metrics` is run_one's results.
    A candidate whose SVG, source, render size and detector code are all
    unchanged is read from the score cache instead of re-rendered (Wiki audit K13:
    scoring added 76 s to an 85 s trace)."""
    out_dir = Path(out_dir)
    t0 = time.perf_counter()
    t, ev = image_type(rgb, src_long)
    per = {f'{mm}mm': {} for mm in SIZES_MM}
    cache = _cache_dir()
    src_hash = hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
    hits = 0
    for name, sc in metrics.items():
        svg = out_dir / f'{name}.svg'
        if 'hygiene' not in sc or not svg.is_file():
            continue
        text = svg.read_text()
        xing = sc.get('self_intersecting_paths', 0)
        key = hashlib.sha256(f'{_CODE}|{RENDER_PX}|{SIZES_MM}|{xing}|{src_hash}|{text}'.encode()).hexdigest()
        hit = cache and cache / f'{key}.json'
        if hit and hit.is_file() and hit.with_suffix('.png').is_file():
            rep = json.loads(hit.read_text())
            shutil.copyfile(hit.with_suffix('.png'), out_dir / f'{name}.hi.png')
            hits += 1
        else:
            w, h, paths = read_svg(text)
            ink = rasterize_at(paths, w, h)
            cv2.imwrite(str(out_dir / f'{name}.hi.png'), np.where(ink, 0, 255).astype(np.uint8))
            rep = score_raster(ink, rgb, paths, xing)
            if cache:
                try:
                    cache.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(out_dir / f'{name}.hi.png', hit.with_suffix('.png'))
                    hit.write_text(json.dumps(rep))   # json last: it is what marks the entry complete
                except OSError:
                    pass                               # a cache is never allowed to fail a trace
        for k, v in rep.items():
            per[k][name] = v
    for k in per:
        hygiene.finalize(per[k])
    if cache and hits < len(per['70mm']):
        _prune(cache)
    doc = {'render_px': RENDER_PX, 'image_type': t, 'type_evidence': ev, 'sizes': per,
           'scoring_s': round(time.perf_counter() - t0, 2), 'cache_hits': hits}
    (out_dir / 'defects.json').write_text(json.dumps(doc, indent=1))
    return doc


if __name__ == '__main__':
    # self-check: our own writer round-trips, a solid square scores clean
    from vector import paths_to_svg
    sq = [[(np.array(a, float), np.array(a, float), np.array(b, float), np.array(b, float))
           for a, b in [((10, 10), (90, 10)), ((90, 10), (90, 90)),
                        ((90, 90), (10, 90)), ((10, 90), (10, 10))]]]
    w, h, p = read_svg(paths_to_svg(sq, 100, 100))
    assert (w, h, len(p), len(p[0])) == (100, 100, 1, 4)
    ink = rasterize_at(p, w, h, long_px=1000)
    assert abs(ink.mean() - 0.64) < 0.01, ink.mean()
    src = np.where(ink[..., None], 0, 255).astype(np.uint8).repeat(3, 2)
    rep = score_raster(ink, src, p, 0)
    assert set(rep) == {'70mm', '150mm'} and rep['70mm']['strays'] == 0
    assert image_type(src, 1000)[0] == 'bw'
    print('defects.py self-check ok')
