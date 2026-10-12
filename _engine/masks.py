#!/usr/bin/env python3
"""masks.py - T-Tracer

Mask-domain helpers shared by the strategies. No strategy is defined here,
and nothing here may import `strategies` -- that edge is what keeps the
import graph acyclic.

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



# ---------------------------------------------------------------------------
# Binarization strategies
#
# Each returns a boolean mask where True == INK (the geometry that ends up in
# the SVG). They are ordered cheapest-and-most-literal first.
# ---------------------------------------------------------------------------

def _background_color(rgb, border=4):
    """Median color of the image border. The border is background far more
    often than not, and unlike 'most frequent color' it is not fooled by a
    large foreground fill that happens to dominate the histogram."""
    h, w, _ = rgb.shape
    b = min(border, h // 4, w // 4) or 1
    edge = np.concatenate([
        rgb[:b, :, :].reshape(-1, 3), rgb[-b:, :, :].reshape(-1, 3),
        rgb[:, :b, :].reshape(-1, 3), rgb[:, -b:, :].reshape(-1, 3),
    ])
    return np.median(edge, axis=0)


def _kmeans_lab(flat, k, attempts=4):
    """Seeded k-means on an Nx3 float32 Lab array. cv2.kmeans draws its
    initial centres from a process-global RNG, so the seed is reset
    immediately before EVERY call - see the determinism note in main()."""
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    cv2.setRNGSeed(0)
    compact, labels, centers = cv2.kmeans(flat, k, None, crit, attempts,
                                          cv2.KMEANS_PP_CENTERS)
    return compact, labels.ravel(), centers


def _pick_k(flat, k_min=2, k_max=8, sample=20000, min_gain=0.12):
    """Choose the number of colour clusters by the elbow of within-cluster
    scatter on a fixed 20k-pixel sample. Stops at the first k whose extra
    cluster cuts the scatter by less than `min_gain` of the previous value.

    Insignia are flat-colour art: a 3-colour patch has a sharp elbow at 3, a
    7-colour one keeps gaining until 7. A fixed k=5 over-splits the first into
    anti-aliasing bands and under-splits the second into muddy merges.
    """
    if len(flat) > sample:
        idx = np.random.default_rng(0).choice(len(flat), sample, replace=False)
        flat = flat[idx]
    n_unique = len(np.unique(flat, axis=0))
    k_max = max(k_min, min(k_max, n_unique))
    prev = None
    best = k_min
    for k in range(k_min, k_max + 1):
        compact, _, _ = _kmeans_lab(flat, k, attempts=3)
        if prev is not None and (prev - compact) < min_gain * prev:
            break
        best = k
        prev = compact
    return best


def _true_lab(centers):
    """OpenCV's 8-bit Lab packs L as 0-255 and a/b as offset 128, so a
    Euclidean distance on it weights lightness 2.55x over chroma - light grey
    came out 'nearer' to green than to white. Every colour DECISION here is
    made in real CIELAB units; only the clustering itself runs on the packed
    values (that is what `kmeans` does, and the assignment is the same)."""
    c = np.asarray(centers, dtype=np.float64).copy()
    c[:, 0] *= 100.0 / 255.0
    c[:, 1:] -= 128.0
    return c


def _between(c, others, slack=0.3, want_pair=False):
    """If colour `c` lies on the straight line between some pair of `others`
    (within `slack` of that pair's separation), return the index into
    `others` of the nearer endpoint - or, with `want_pair`, the tuple
    (a, b, nearer) - else None. True for the mixed pixels of an anti-aliased
    edge between two flat colours."""
    best = None
    for a in range(len(others)):
        for b in range(a + 1, len(others)):
            pa, pb = others[a], others[b]
            ab = pb - pa
            L = float(np.dot(ab, ab))
            if L < 1e-6:
                continue
            t = float(np.dot(c - pa, ab)) / L
            if 0.05 < t < 0.95:
                off = float(np.linalg.norm(c - (pa + t * ab)))
                if off < slack * np.sqrt(L) and (best is None or off < best[0]):
                    best = (off, a, b, a if t < 0.5 else b)
    if best is None:
        return None
    return best[1:] if want_pair else best[3]


def _keep_white_channels(lines, fill, line_px):
    """Stop the outline stroke from closing a bare channel narrower than
    itself.

    Tyler, 2026-09-15, shown the stroke at 4, 6 and 8px: "none - left closest
    but no". No width can be right, because the defect is not the width. A
    stroke `line_px` wide swallows every bare gap narrower than `line_px`, and
    on the LCS Squadron One lighthouse the railing's gaps are about 3px at the
    1600px working size. A thinner stroke only moves which detail dies.

    So the medial axis of the bare area is carved back out of the stroke
    WHERE THE BARE AREA IS NARROW - within `line_px` of ink, which is exactly
    where the stroke would have closed it. Each such channel keeps a
    one-pixel white spine and reads as two edges rather than one blob.

    Restricting it to narrow places is what makes it safe. In open space the
    medial axis runs far from any ink, the stroke never reaches it, and the
    stroke comes back untouched - so this cannot punch holes in a legitimate
    boundary line, which is the defect it would otherwise trade for. It also
    covers a channel that is open at both ends (a stripe between two fields),
    which an earlier enclosed-region-only version missed.
    """
    bare = ~fill
    if not bare.any() or not lines.any():
        return lines
    dist = cv2.distanceTransform(bare.astype(np.uint8), cv2.DIST_L2, 3)
    spine = skeletonize(bare) & (dist <= float(line_px))
    return lines & ~spine


def _silhouette_of(rgb, alpha=None, lab_tol=14.0):
    """The artwork's extent. Alpha when the source has real transparency,
    otherwise a small Lab distance from the detected background."""
    if alpha is not None and alpha.min() < 250:
        return alpha > 128
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    return np.linalg.norm(lab - _background_color(lab), axis=2) > lab_tol


def _lightness(rgb):
    """CIELAB L* on a real 0..100 scale. OpenCV packs L into 0..255."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)[..., 0] * (100.0 / 255.0)


def _grow_counters(mask, min_px):
    """Grow NESTED ink islands -- letter counters -- to a minimum visible size.

    The counter-side twin of the stroke floor, and it exists because of one
    measurement. Tyler on the small patch: *"silhouette is right there. i
    mean right there. its missing the dot in the A in VFA."* Zooming the source
    to individual pixels, **that counter is ONE PIXEL** -- the source is
    199x254 and the whole letter is about 15px tall.

    So this is not a resolution problem and no upscaler invents it. The
    information is there; it is one pixel wide, and it dies twice on the way to
    the plate: Lanczos blurs it, and then `mask_to_paths`'s INTER_CUBIC
    supersample erases an island of 3x3 or smaller before contouring. Measured
    at a 2048px working size, the counter survives thresholding as a 7px island
    and contributes **zero** pixels to the final raster.

    The honest fix is the same one the stroke floor makes: a feature below the
    minimum visible size cannot reach the plate, so grow it to that size or
    accept it is lost. Nesting is what makes this safe -- only islands that sit
    inside an enclosed non-ink region are grown, which is a counter by
    construction and never a speck in the open background. Same test
    `clean_mask` already uses to protect counters from `min_area_frac`.
    """
    if min_px < 1:
        return mask
    lab = label(mask, connectivity=2)
    if lab.max() == 0:
        return mask
    sizes = np.bincount(lab.ravel(), minlength=lab.max() + 1)
    sizes[0] = 0
    holes = label(~mask, connectivity=1)
    border = set(int(b) for b in np.unique(np.concatenate(
        [holes[0], holes[-1], holes[:, 0], holes[:, -1]])))
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    out = mask.copy()
    for i in range(1, lab.max() + 1):
        if sizes[i] == 0 or sizes[i] >= min_px * min_px:
            continue
        comp = (lab == i).astype(np.uint8)
        ring = (cv2.dilate(comp, k3) > 0) & ~comp.astype(bool)
        around = set(int(a) for a in np.unique(holes[ring]) if a != 0)
        if not around or (around & border):
            continue                              # not nested: a speck, leave it
        r = max(1, int(np.ceil((min_px - np.sqrt(sizes[i])) / 2)))
        kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
        out |= cv2.dilate(comp, kk) > 0
    return out
