#!/usr/bin/env python3
"""vector.py - T-Tracer

Mask -> contours -> cubic Beziers -> SVG, plus rasterising a result back to
pixels for the preview and the contact sheet.

Split out of candidates.py 2026-09-22; `git log --follow` carries the history.
"""

import argparse
import contextlib
import hashlib
import os
import time
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import paths

import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
from skimage.filters import threshold_multiotsu, threshold_sauvola
from skimage.measure import label
from skimage.morphology import skeletonize
from scipy.ndimage import binary_fill_holes, distance_transform_edt, find_objects

from tracestate import _progress



# ---------------------------------------------------------------------------
# Mask cleanup
# ---------------------------------------------------------------------------

def clean_mask(mask, src_gray=None, close_px=2, open_px=1, min_area_frac=0.0004):
    """Close hairline breaks, remove speckle, drop stray islands.

    The close is PROTECTED, and that protection is load-bearing. A plain
    morphological close fills every white region narrower than its radius -
    which on a 386px source silently ate the white the banner lettering out of
    the banner, because those strokes are about 3px wide and the radius was 2.
    The candidate still looked clean and still scored well; the text was just
    gone. Reversed letters in a customer's squadron name is not a defect that
    gets caught downstream, so it gets prevented here.

    So a white region is only allowed to be filled if it is small enough to be
    a genuine hairline break rather than real artwork. The size rule is derived
    from the close radius itself, so it scales with whatever the caller asks
    for instead of being another magic number to keep in sync.

    min_area_frac is likewise a fraction of total ink area, not a pixel count,
    so the same rule works on a 400px source and a 4000px one.
    """
    m = mask.astype(bool)

    if close_px > 0 and src_gray is not None:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1,) * 2)
        closed = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
        newly = closed & ~m

        # Size alone cannot separate a hairline seam from a letter stroke - both
        # are small and thin. What separates them is the SOURCE: a seam sits
        # over dark pixels that should have been ink, a letter sits over light
        # pixels that are supposed to stay bare. So each white region the close
        # wants to fill is put to that question, and only the ones the source
        # agrees about are filled.
        if newly.any() and m.any() and (~m).any():
            ink_mean = float(src_gray[m].mean())
            bg_mean = float(src_gray[~m].mean())
            lab_w = label(newly, connectivity=2)
            n = lab_w.max()
            if n > 0 and abs(ink_mean - bg_mean) > 1e-6:
                sizes = np.bincount(lab_w.ravel(), minlength=n + 1)
                sums = np.bincount(lab_w.ravel(), weights=src_gray.ravel(),
                                   minlength=n + 1)
                allow = np.zeros(n + 1, dtype=bool)
                for i in range(1, n + 1):
                    if sizes[i] == 0:
                        continue
                    lum = sums[i] / sizes[i]
                    allow[i] = abs(lum - ink_mean) < abs(lum - bg_mean)
                m = m | allow[lab_w]
            else:
                m = closed
        else:
            m = closed

    if open_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_px + 1,) * 2)
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k) > 0
    total = int(m.sum())
    if total == 0:
        return m

    lab = label(m, connectivity=2)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = sizes >= max(2, total * min_area_frac)

    # NESTED ISLANDS ARE PROTECTED, for the same reason the close is protected.
    # A letter counter -- the dark centre of an O, the bowl of an R in knocked-
    # out text -- is a SMALL INK ISLAND, and min_area_frac was deleting them:
    # measured 53 -> 18 islands on the 719th DUI, 22 -> 12 on LCS-Squadron-1.
    # the reviewer named it three separate times ("missing the small dots in the
    # letters", "missing letter dots", "the letters are missing the middle
    # pieces like in the R") before it was found, because the result still looks
    # clean and still scores well -- the same signature as the unit banner
    # text and the gold shield outline before it.
    #
    # Size cannot separate a counter from a speck; both are small. NESTING can.
    # A speck sits in the open background. A counter sits inside a non-ink
    # region that is ITSELF enclosed by ink, so it is two levels deep. Only
    # islands that are NOT nested face the size test.
    holes = label(~m, connectivity=1)
    border = np.unique(np.concatenate([holes[0], holes[-1], holes[:, 0], holes[:, -1]]))
    outer = set(int(b) for b in border)          # non-ink touching the frame
    if holes.max() > 0:
        # For each ink island, which non-ink region surrounds it? Dilate by one
        # and read the labels that appear alongside it.
        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # Each island is tested inside its own bounding box (one pixel of
        # margin, so the 3x3 dilate cannot leave it) rather than on a full
        # image comparison per island. Same answer; O(island) not O(image).
        boxes = find_objects(lab)
        for i in np.where(~keep)[0]:
            if i == 0 or sizes[i] == 0 or boxes[i - 1] is None:
                continue
            sl = tuple(slice(max(0, b.start - 1), b.stop + 1) for b in boxes[i - 1])
            comp = (lab[sl] == i).astype(np.uint8)
            ring = (cv2.dilate(comp, k3, iterations=1) > 0) & ~comp.astype(bool)
            around = np.unique(holes[sl][ring])
            around = around[around != 0]
            # nested == every surrounding non-ink region is an interior one
            if around.size and not (set(int(a) for a in around) & outer):
                keep[i] = True
    return np.isin(lab, np.where(keep)[0])


# ---------------------------------------------------------------------------
# Vectorization: contour -> smoothed spline -> cubic Beziers
# ---------------------------------------------------------------------------

def _bezier_from_hermite(p0, p1, m0, m1):
    """Hermite segment (endpoints + parametric tangents) to cubic Bezier
    control points. Exact conversion, not an approximation."""
    c1 = p0 + m0 / 3.0
    c2 = p1 - m1 / 3.0
    return c1, c2


def _sample_beziers(segs, per_seg=24):
    """Dense polyline through a list of (p0, c1, c2, p3) cubic segments."""
    t = np.linspace(0, 1, per_seg, endpoint=False)[:, None]
    out = []
    for p0, c1, c2, p3 in segs:
        mt = 1 - t
        pts = (mt ** 3) * p0 + 3 * (mt ** 2) * t * c1 + 3 * mt * (t ** 2) * c2 + (t ** 3) * p3
        out.append(pts)
    return np.vstack(out) if out else np.zeros((0, 2))


def fit_beziers(pts, smooth_per_point=3.0, tol=0.6, max_nodes=400):
    """Fit a closed cubic-Bezier path to a raw pixel contour.

    Two stages, deliberately separated:
      1. A periodic smoothing spline removes the pixel staircase. This is what
         kills jagged edges, and it is the same idea smoothing_engine already
         uses - the difference is what happens next.
      2. Knots are placed by EQUAL CUMULATIVE CURVATURE, so straight runs spend
         almost no nodes and tight corners get all of them, then the node count
         is raised only until max deviation from the smooth curve falls under
         `tol` pixels. That yields the minimum nodes for a given accuracy
         instead of a fixed sample rate that is simultaneously too coarse on
         corners and absurdly wasteful on straights.

    Returns a list of (p0, c1, c2, p3) segments, or None if the contour is
    too small to fit.
    """
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 8:
        return None

    try:
        tck, _ = splprep([pts[:, 0], pts[:, 1]],
                         s=len(pts) * smooth_per_point, per=True, k=3)
    except Exception:
        return None

    dense_u = np.linspace(0, 1, 2000)
    dx, dy = splev(dense_u, tck)
    dense = np.column_stack([dx, dy])

    # Curvature along the smoothed curve, used purely to distribute knots.
    d1x, d1y = splev(dense_u, tck, der=1)
    d2x, d2y = splev(dense_u, tck, der=2)
    num = np.abs(d1x * d2y - d1y * d2x)
    den = np.power(d1x ** 2 + d1y ** 2, 1.5) + 1e-9
    curv = num / den
    # A flat floor guarantees long straight runs still receive some nodes.
    weight = curv + np.percentile(curv, 60) + 1e-6
    cum = np.concatenate([[0], np.cumsum(weight[:-1])])
    cum /= cum[-1]

    tree = cKDTree(dense)
    n = 8
    while True:
        knot_u = np.interp(np.linspace(0, 1, n, endpoint=False), cum, dense_u)
        knot_u = np.append(knot_u, 1.0)

        kx, ky = splev(knot_u, tck)
        kdx, kdy = splev(knot_u, tck, der=1)
        knots = np.column_stack([kx, ky])
        tangents = np.column_stack([kdx, kdy])

        segs = []
        for i in range(n):
            du = knot_u[i + 1] - knot_u[i]
            p0, p1 = knots[i], knots[i + 1]
            m0, m1 = tangents[i] * du, tangents[i + 1] * du
            c1, c2 = _bezier_from_hermite(p0, p1, m0, m1)
            segs.append((p0, c1, c2, p1))

        err = tree.query(_sample_beziers(segs))[0].max()
        if err <= tol or n >= max_nodes:
            return segs
        n = int(n * 1.6) + 1


def mask_to_paths(mask, scale=6, smooth=3.0, tol=0.6, min_points=16):
    """Supersample, extract contours with hierarchy, fit Beziers to each.

    Supersampling before contour extraction is inherited from smoothing_engine
    and is load-bearing: it preserves true stroke width on thin features that
    would otherwise be quantized away at native resolution.

    Returns paths in ORIGINAL image coordinates.
    """
    m = (mask.astype(np.uint8) * 255)
    big = cv2.resize(m, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    big = (big > 127).astype(np.uint8)

    contours, _ = cv2.findContours(big, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    paths = []
    for c in contours:
        pts = c.reshape(-1, 2).astype(float)
        if len(pts) < min_points:
            continue
        segs = fit_beziers(pts, smooth_per_point=smooth, tol=tol * scale)
        if segs:
            paths.append([(p0 / scale, c1 / scale, c2 / scale, p3 / scale)
                          for p0, c1, c2, p3 in segs])
            # Live view: the fitted curves so far, drawn over a faint mask.
            _progress('trace', lambda: _progress_canvas(mask, paths))
    return paths


def _progress_canvas(mask, paths, long_edge=480):
    """A small RGB frame: the mask in light grey with the Bezier paths fitted
    so far stroked in black. Built only when a live frame is due."""
    h, w = mask.shape
    s = long_edge / max(h, w)
    small = cv2.resize(mask.astype(np.uint8) * 255, (max(1, int(w * s)), max(1, int(h * s))),
                       interpolation=cv2.INTER_AREA)
    canvas = np.full((small.shape[0], small.shape[1], 3), 255, np.uint8)
    canvas[small > 127] = (215, 215, 215)
    for segs in paths:
        pts = (_sample_beziers(segs, per_seg=6) * s).astype(np.int32)
        if len(pts) >= 2:
            cv2.polylines(canvas, [pts], True, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# SVG emission and rasterization
# ---------------------------------------------------------------------------

def paths_to_svg(paths, w, h, ink='#000000', background=None, height_mm=None):
    """Emit one <path> holding every subpath, fill-rule evenodd so interior
    holes knock out correctly. Exactly one fill color - the geometry IS the
    job, per _engine/laser-output-rules.md.

    A background <rect> is omitted by default: in XCS a white rectangle is
    another closed shape to reason about, not 'nothing'.
    """
    if height_mm:
        dims = f'width="{w / h * height_mm:.3f}mm" height="{height_mm:.3f}mm"'
    else:
        dims = f'width="{w}" height="{h}"'

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" {dims} viewBox="0 0 {w} {h}">']
    if background:
        out.append(f'<rect width="{w}" height="{h}" fill="{background}"/>')

    d = []
    for segs in paths:
        p0 = segs[0][0]
        d.append(f'M{p0[0]:.3f},{p0[1]:.3f}')
        for _, c1, c2, p3 in segs:
            d.append(f'C{c1[0]:.3f},{c1[1]:.3f} {c2[0]:.3f},{c2[1]:.3f} {p3[0]:.3f},{p3[1]:.3f}')
        d.append('Z')

    out.append(f'<path fill-rule="evenodd" fill="{ink}" d="{" ".join(d)}"/>')
    out.append('</svg>')
    return '\n'.join(out)


def _hex(color):
    r, g, b = (int(round(float(c))) for c in color[:3])
    return f'#{r:02X}{g:02X}{b:02X}'


def paths_to_svg_layered(layers, w, h, background=None, height_mm=None):
    """One <path> PER COLOUR, each with its own fill and fill-rule evenodd so a
    colour's interior holes still knock out within that colour. Layers are
    written in the order given: first is the bottom of the stack.

    This is the kmeans-layered deliverable: a customer colour change becomes
    'edit one fill attribute', not 're-run the tracer'. It deliberately breaks
    the one-fill rule in _engine/laser-output-rules.md - a layered file is a
    different kind of output, and the pick step decides which kind the job
    wants.
    """
    if height_mm:
        dims = f'width="{w / h * height_mm:.3f}mm" height="{height_mm:.3f}mm"'
    else:
        dims = f'width="{w}" height="{h}"'

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" {dims} viewBox="0 0 {w} {h}">']
    if background:
        out.append(f'<rect width="{w}" height="{h}" fill="{background}"/>')
    for paths, color in layers:
        d = []
        for segs in paths:
            p0 = segs[0][0]
            d.append(f'M{p0[0]:.3f},{p0[1]:.3f}')
            for _, c1, c2, p3 in segs:
                d.append(f'C{c1[0]:.3f},{c1[1]:.3f} {c2[0]:.3f},{c2[1]:.3f} {p3[0]:.3f},{p3[1]:.3f}')
            d.append('Z')
        out.append(f'<path id="layer-{_hex(color)[1:]}" fill-rule="evenodd" '
                   f'fill="{_hex(color)}" d="{" ".join(d)}"/>')
    out.append('</svg>')
    return '\n'.join(out)


def rasterize(paths, w, h, ss=2):
    """Render the emitted geometry back to a boolean mask.

    This deliberately rasterizes the Beziers we actually wrote out, not the
    mask we started from - so the hygiene score describes the delivered SVG,
    including anything the curve fit changed.
    """
    canvas = np.zeros((h * ss, w * ss), np.uint8)
    layer = np.zeros_like(canvas)
    for segs in paths:
        pts = _sample_beziers(segs, per_seg=16) * ss
        if len(pts) < 3:
            continue
        # XOR each subpath in turn - this IS the even-odd rule, so a contour
        # nested inside another knocks a hole rather than filling it solid.
        # One scratch layer, cleared and reused: at 1600px x ss=2 a fresh
        # 10 MB allocation per subpath was the largest single cost in scoring.
        layer.fill(0)
        cv2.fillPoly(layer, [pts.astype(np.int32)], 255)
        cv2.bitwise_xor(canvas, layer, dst=canvas)
    return canvas > 0


def count_nodes(paths):
    return sum(len(segs) for segs in paths)


def composite_layers(rendered, w, h):
    """Paint (raster, paths, colour) layers bottom-first over white and return
    an HxWx3 RGB uint8 preview at the source size."""
    out = np.full((rendered[0][0].shape[0], rendered[0][0].shape[1], 3), 255, np.uint8)
    for r, _, color in rendered:
        out[r] = color
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)
