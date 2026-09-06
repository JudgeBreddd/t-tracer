#!/usr/bin/env python3
"""
candidates.py - T-Tracer

One source image in, N candidate laser-ready SVGs out - each produced by a
genuinely DIFFERENT binarization mechanism, not by nudging one threshold.

Why N candidates instead of one tuned pass: there is no single threshold that
works across flat vector logos, alpha-transparent PNGs, and photographs of
embroidered patches. Rather than guess, emit the honest spread and let the
operator pick in four seconds. Every candidate is scored by hygiene.py against
the criteria that actually predict a good burn (see below), so the pick is
informed rather than a beauty contest.

Target job (drives the defaults): black-coated brass plate, xTool F2 Ultra UV.
Cold process, tiny spot, no thermal bloom - so minimum-feature-size is NOT the
binding constraint here. Vector hygiene is. the reviewer's stated acceptance test:
    "if the lines are all touching and the svg is clean, no jagged edges,
     no gaps, no random black lines, it will burn good."
Each of those four is a measurable quantity; hygiene.py measures them.

Two things this fixes versus lineart_trace_v1 / multicolor_trace_v1:
  1. Output is real cubic Beziers, not a spline sampled into hundreds of `L`
     segments. Same curve, ~20x fewer nodes, no faceting.
  2. Single-mask output, so there are no independently-smoothed adjacent color
     layers to drift apart and leave gaps along a shared edge.

Usage (drop-folder, same convention as the other scripts - no args needed):
    ../.venv/bin/python candidates.py
    ../.venv/bin/python candidates.py --only "logo.png" --scale 8
    ../.venv/bin/python candidates.py --invert          # swap ink/background

Output, per input image:
    candidates/<image-stem>/<strategy>.svg     the laser file
    candidates/<image-stem>/<strategy>.png     preview, rendered from the SVG geometry
    candidates/<image-stem>/metrics.json       hygiene scores for every strategy
    candidates/_sheets/<image-stem>.png        contact sheet - ALL sheets in one folder
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

import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
from skimage.filters import threshold_multiotsu, threshold_sauvola
from skimage.measure import label
from scipy.ndimage import binary_fill_holes, distance_transform_edt

from smoothing_engine import flatten_to_white

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


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


def strat_otsu(rgb):
    """Global Otsu on luminance. The baseline every tracer starts with.
    Fails when two important regions share a lightness (red on navy)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binv > 0


def strat_sauvola(rgb, window=51, k=0.25):
    """Sauvola local adaptive threshold. The small-text and uneven-lighting
    specialist - keeps a 6pt motto that global Otsu erases. Pays for it in
    speckle, which the cleanup pass then has to earn back."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    win = max(3, window | 1)
    return gray < threshold_sauvola(gray, window_size=win, k=k)


def strat_bgdist(rgb):
    """Perceptual distance from the detected background color, Otsu'd.
    This is the one that handles a saturated logo on a flat field - including
    an alpha PNG flattened to white - because it asks 'how far from the
    background is this pixel' rather than 'how dark is this pixel'."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg = _background_color(lab)
    dist = np.linalg.norm(lab - bg, axis=2)
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, b = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b > 0


def strat_kmeans(rgb, k=5):
    """K-means in Lab, then a figure/ground solve: the cluster owning the most
    border pixels is background, everything else is ink. Separates regions that
    differ in hue but not in lightness, which luminance thresholding fuses."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w, _ = lab.shape
    flat = lab.reshape(-1, 3)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    cv2.setRNGSeed(0)                  # see the determinism note in main()
    _, labels, _ = cv2.kmeans(flat, min(k, len(np.unique(flat, axis=0))),
                              None, crit, 4, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(h, w)

    b = max(1, min(4, h // 4, w // 4))
    border = np.concatenate([
        labels[:b, :].ravel(), labels[-b:, :].ravel(),
        labels[:, :b].ravel(), labels[:, -b:].ravel(),
    ])
    bg_label = np.bincount(border).argmax()
    return labels != bg_label


def strat_inotsu(rgb, alpha=None, lab_tol=14.0):
    """Otsu computed over the ARTWORK ONLY, not the whole frame.

    The smallest change that answers the project's most common rejection:
    saturated mid-luminance colour read as the wrong side of the split
    ("missing flour de le", "the cross in the center is missing", "missing the
    chevrons (originally red)").

    Every previous attempt at that failure attacked the COLOUR SPACE --
    `bgdist` measuring Lab distance, `neural` clustering per region, `nested`
    walking a containment tree. `red-eye logo` showed the problem is one stage earlier
    and much simpler: not what you measure, but WHICH PIXELS YOU MEASURE OVER.
    A logo is mostly background, and background is the largest, most
    extreme-valued population in the frame, so it drags a global Otsu split
    toward its own end.

    Measured on `red-eye logo`, whose eye is red on a black head:
        background                     82% of the image
        whole-image Otsu cut           128   -> eye (gray 80) is ink, and
                                              vanishes into the head
        artwork-only Otsu cut           71   -> eye survives as a knockout

    This is exactly why `silhouette` has been quietly winning since 09-04
    without anyone explaining it -- it computes its interior threshold over
    `gray[sil]`. `silhouette` then adds a morphological outline RING on top,
    which is a second, separable idea. This strategy isolates the population
    fix alone, so the two can be judged apart on the sheet.
    """
    if alpha is not None and alpha.min() < 250:
        art = alpha > 128
    else:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        art = np.linalg.norm(lab - _background_color(lab), axis=2) > lab_tol

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    inside = gray[art]
    if inside.size < 16:                     # no artwork found -> plain Otsu
        return strat_otsu(rgb)

    cut, _ = cv2.threshold(inside.reshape(-1, 1), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return art & (gray < cut)


def strat_triotsu(rgb, alpha=None, lab_tol=14.0):
    """THREE-class Otsu, with the middle class joining the ink.

    Aimed straight at the project's most common rejection. the reviewer's own words
    for it, three separate times: "missing flour de le" (gold), "the cross in
    the center is missing", "missing the chevrons (origonally red)".

    Why a second cut and not a better first one. These logos have THREE tonal
    populations -- a dark navy or black field, a mid-luminance saturated red or
    gold, and a pale background. A two-class split has to put the middle
    population on one side or the other, and it consistently files red and gold
    with the background, which is exactly the reported failure. No amount of
    moving a single threshold fixes a three-population problem.

    Measured, and this is why `inotsu` was not the answer: restricting Otsu to
    the artwork does move the cut (-15 to -57 gray levels across the test set)
    but changes only 0.15-3.4% of pixels, because almost nothing lives in the
    band between the two thresholds. The population was never the problem --
    the NUMBER OF CLASSES was.

    Multi-Otsu gives two cuts and three classes; ink is everything below the
    UPPER cut, so the mid tone burns with the dark tone instead of dropping out
    with the background.

    TESTED 2026-09-06 AND IT FAILS. Kept off the sheet, and kept in the file so
    nobody spends another afternoon rediscovering this. On `gold-shield crest` the gold
    shield -- mid tone -- joined the ink and became a SOLID BLACK BLOB that
    swallowed the entire interior design; detail 0.44 of best. `anchor crest` lost
    the anchor the same way, detail 0.70. It re-creates the "big black mess"
    failure while trying to fix the colour one.

    The reason is worth more than the strategy: **the same mid-luminance gold
    is INK on one logo and a FIELD on another.** The fleur-de-lis crest and
    another crest's chevrons are shapes drawn ON something. A third crest's
    gold shield is the thing being drawn on. They are indistinguishable
    by luminance, at any number of classes, over any population -- the
    difference is structural, not tonal. That is precisely the question
    `nested`'s containment tree asks, and it is why the region route, not the
    threshold route, is the one that can work here.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    if alpha is not None and alpha.min() < 250:
        art = alpha > 128
    else:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        art = np.linalg.norm(lab - _background_color(lab), axis=2) > lab_tol
    if art.sum() < 16:
        return strat_otsu(rgb)

    try:
        cuts = threshold_multiotsu(gray[art], classes=3)
    except ValueError:
        return strat_otsu(rgb)                   # too few distinct levels
    return art & (gray < float(cuts[-1]))


def strat_edges(rgb, min_stroke=2, max_hole_frac=0.05):
    """Canny ridges closed to a minimum stroke width, then SMALL enclosed
    regions filled - an outline/line-art reading of the source.

    The first version flood-filled everything the border could not reach, which
    on any emblem with a closed outer ring means 'everything', and it produced
    a solid black square that then topped the leaderboard. Only holes below
    `max_hole_frac` of the canvas are filled now, so letter counters and small
    enclosed shapes go solid while the emblem interior stays open.

    This is the strategy that still says something useful about a photograph of
    a physical embroidered patch, where every flat-field assumption is dead.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 60, 60)
    med = float(np.median(gray))
    edges = cv2.Canny(gray, int(max(0, 0.66 * med)), int(min(255, 1.33 * med)))

    r = max(1, int(min_stroke))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kern) > 0

    h, w = closed.shape
    lab = label(~closed, connectivity=1)
    sizes = np.bincount(lab.ravel())
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))

    fill = np.zeros_like(closed)
    limit = h * w * max_hole_frac
    for i in np.where(sizes <= limit)[0]:
        if i == 0 or i in border:
            continue
        fill |= (lab == i)

    return closed | fill


def strat_silhouette(rgb, alpha=None, lab_tol=14.0):
    """Everything that is not background - the artwork's outline shape.

    Added after a real miss: on gold-shield crest the gold shield and its scroll are
    bright, so every luminance-driven strategy read them as background and
    dropped them. All five candidates lost the same element, which meant the
    relative fidelity score could not flag it either - nothing to compare
    against. That is the failure mode where a slate of candidates silently
    agrees on being wrong.

    Two sources of truth, cheapest first. If the source is a real alpha PNG,
    the alpha channel already IS the artwork silhouette and no thresholding can
    beat it - and this pipeline was throwing that away in flatten_to_white.
    Otherwise, fall back to a small fixed Lab distance from the detected
    background, which unlike an Otsu split does not require the artwork to be
    darker than its surroundings.

    Emitted as OUTLINE + INTERIOR, not as a filled blob. A bare silhouette is
    one closed shape with no internal structure - technically the shape, but
    useless on a plate. Ringing the silhouette and OR-ing the interior's dark
    detail back in gives the shield's edge AND everything drawn inside it.
    """
    if alpha is not None and alpha.min() < 250:
        sil = alpha > 128
    else:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        bg = _background_color(lab)
        sil = np.linalg.norm(lab - bg, axis=2) > lab_tol

    if not sil.any():
        return sil

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ring = cv2.morphologyEx(sil.astype(np.uint8), cv2.MORPH_GRADIENT, k) > 0

    # Interior detail, thresholded against only the pixels inside the artwork -
    # so a gold field does not drag the split the way whole-image Otsu does.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    inside = gray[sil]
    if inside.size:
        cut, _ = cv2.threshold(inside.reshape(-1, 1), 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inner = sil & (gray < cut)
    else:
        inner = np.zeros_like(sil)

    return ring | inner


def _silhouette_of(rgb, alpha=None, lab_tol=14.0):
    """The artwork's extent. Alpha when the source has real transparency,
    otherwise a small Lab distance from the detected background."""
    if alpha is not None and alpha.min() < 250:
        return alpha > 128
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    return np.linalg.norm(lab - _background_color(lab), axis=2) > lab_tol


def strat_linework(rgb, alpha=None, k=6):
    """Fill the lines, outline the fields, fill the dark elements.

    Built to the reviewer's description of what a gold-shield-style insignia should
    become on black-coated brass: "the gold outline becomes black and the blue
    elements become black... it should also capture the scroll outline, not
    just the shield."

    `silhouette` failed that because it outlined everything uniformly - one
    ring around the whole artwork - so the scroll's own border never appeared
    and the lettering came out hollow instead of solid. The distinction it was
    missing is not colour, it is THICKNESS:

      * a thin region is already a line (the gold shield border, the gold
        scroll border) -> fill it, and the line draws itself
      * a thick region is a field (the gold shield interior, the white scroll
        interior) -> take its boundary, and the field becomes an outline
      * anything clearly darker than the artwork's median (the blue elements,
        the blue lettering) is the primary subject -> fill it solid

    Thickness is tested by erosion: a region that mostly disappears under a
    small erode was never more than a stroke.
    """
    sil = _silhouette_of(rgb, alpha)
    if not sil.any():
        return sil

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    pts = lab[sil]
    kk = int(min(k, max(2, len(np.unique(pts.astype(np.int16), axis=0)))))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    cv2.setRNGSeed(0)                  # see the determinism note in main()
    _, lbl, centers = cv2.kmeans(pts, kk, None, crit, 4, cv2.KMEANS_PP_CENTERS)

    labels = np.full(sil.shape, -1, np.int32)
    labels[sil] = lbl.ravel()

    diag = float(np.hypot(*sil.shape))
    r_thin = max(1, int(round(diag * 0.005)))
    r_stroke = max(1, int(round(diag * 0.003)))
    k_thin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_thin + 1,) * 2)
    k_str = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_stroke + 1,) * 2)
    med_L = float(np.median(lab[:, :, 0][sil]))

    # Pass 1: everything that gets FILLED.
    fills = np.zeros_like(sil)
    fields = []
    for i in range(kk):
        m = labels == i
        area = int(m.sum())
        if area < 20:
            continue
        mu = m.astype(np.uint8)
        line_like = int(cv2.erode(mu, k_thin).sum()) < area * 0.30
        dark = float(centers[i][0]) < med_L - 12.0

        if dark or line_like:
            fills |= m
        else:
            fields.append(mu)

    # Per-pixel backstop for small dark detail. Clustering allocates centres by
    # pixel count, so a few hundred anti-aliased pixels of the unit lettering get
    # absorbed into whichever larger, lighter cluster is nearest - and the
    # lettering then came out hollow, drawn only by the scroll's boundary.
    fills |= sil & (lab[:, :, 0] < med_L - 25.0)

    # Pass 2: field boundaries, but NOT where they hug something already
    # filled. The white scroll is a field, so its boundary runs around every
    # letter sitting on it - and drawing that halo on top of the filled letter
    # grew each glyph by a stroke width on all sides, closing the counters and
    # welding "WING" into a blob. An outline is only wanted where a field meets
    # something that is not already ink.
    guard = cv2.dilate(fills.astype(np.uint8), k_str) > 0
    rings = np.zeros_like(sil)
    for mu in fields:
        rings |= cv2.morphologyEx(mu, cv2.MORPH_GRADIENT, k_str) > 0
    rings &= ~guard

    return (fills | rings) & sil


def strat_plate(rgb, alpha=None, k=6, border_frac=0.015):
    """Borders by inward offset, interior detail filled solid.

    This is the reviewer's own hand method, mechanised. Describing what he would build
    manually for a gold-shield plate: "take the shield only, offset it inward and
    set to fill. So the shield would have a border instead of being completely
    filled black... take the graphic from linework and put it inside."

    Two things make this better than `linework`'s morphological gradient:

    1. An inward offset (`shape & ~erode(shape)`) keeps the whole border INSIDE
       the shape. A gradient straddles the boundary, half in and half out, so
       it both bloats the silhouette and eats into the artwork.

    2. Holes are filled BEFORE the offset. A field like the white scroll has
       holes where the lettering sits, and eroding it directly grows a border
       around every glyph - which is what closed the counters and welded "WING"
       into a blob. Filling first means the border can only appear at the outer
       perimeter, so interior detail is never touched. `linework` needed an
       explicit guard to undo that damage; here it cannot happen.

    Border width is a fraction of image height: 0.015 is about 0.5mm on a 32mm
    plate, which is roughly the weight the reviewer offsets by hand.
    """
    sil = _silhouette_of(rgb, alpha)
    if not sil.any():
        return sil

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    pts = lab[sil]
    kk = int(min(k, max(2, len(np.unique(pts.astype(np.int16), axis=0)))))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    cv2.setRNGSeed(0)                  # see the determinism note in main()
    _, lbl, centers = cv2.kmeans(pts, kk, None, crit, 4, cv2.KMEANS_PP_CENTERS)
    labels = np.full(sil.shape, -1, np.int32)
    labels[sil] = lbl.ravel()

    h = sil.shape[0]
    r_thin = max(1, int(round(float(np.hypot(*sil.shape)) * 0.005)))
    w_border = max(1, int(round(h * border_frac)))
    k_thin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_thin + 1,) * 2)
    k_bord = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * w_border + 1,) * 2)
    med_L = float(np.median(lab[:, :, 0][sil]))

    # Figure/ground, decided before anything else. A large DARK region can be
    # either foreground artwork (gold-shield crest's blue band) or the field everything
    # sits on (VFC-204's red disc, NIWDC's navy globe). Both are dark and both
    # are thick, so darkness alone cannot tell them apart - and treating the
    # second kind as ink filled it solid and swallowed every element inside it,
    # which is what turned six of these insignia into black discs.
    #
    # What separates them is position, not colour: the ground is the region
    # that owns the silhouette's rim. Whichever cluster covers most of that rim
    # is forced to be a field and outlined, however dark it is.
    r_rim = max(2, int(round(h * 0.01)))
    k_rim = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_rim + 1,) * 2)
    rim = sil & ~(cv2.erode(sil.astype(np.uint8), k_rim) > 0)
    rim_n = max(1, int(rim.sum()))
    rim_share = [float((labels[rim] == i).sum()) / rim_n for i in range(kk)]
    ground = int(np.argmax(rim_share)) if max(rim_share) > 0.40 else -1

    # Pass 1: what the source itself draws as ink.
    fills = np.zeros_like(sil)
    fields = []
    for i in range(kk):
        m = labels == i
        area = int(m.sum())
        if area < 20:
            continue
        mu = m.astype(np.uint8)
        line_like = int(cv2.erode(mu, k_thin).sum()) < area * 0.30
        dark = float(centers[i][0]) < med_L - 12.0
        if i == ground:
            fields.append(m)
        elif dark or line_like:
            fills |= m
        else:
            fields.append(m)
    backstop = sil & (lab[:, :, 0] < med_L - 25.0)
    if ground >= 0:
        backstop &= labels != ground
    fills |= backstop

    # Pass 2: borders. Where the source already rules an edge with its own
    # hairline, that hairline IS the border and only needs thickening inward to
    # the target weight. Synthesizing a second border beside it produced two
    # parallel rules with a ragged sliver between them, and every attempt to
    # weld them afterwards either left the sliver or bridged across the field
    # and filled the shield solid. Growing the real line cannot produce a
    # sliver, because there is only ever one line.
    borders = np.zeros_like(sil)
    k_near = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for m in fields:
        solid = binary_fill_holes(m)
        su = solid.astype(np.uint8)
        edge = (su - cv2.erode(su, k_near)) > 0          # the field's own rim
        existing = fills & cv2.dilate(edge.astype(np.uint8), k_near).astype(bool)

        if existing.sum() > edge.sum() * 0.35:
            grown = cv2.dilate(existing.astype(np.uint8), k_bord) > 0
            borders |= (grown & solid) | existing
        else:
            borders |= solid & ~(cv2.erode(su, k_bord) > 0)

    return (fills | borders) & sil


# ---------------------------------------------------------------------------
# `neural` -- learned boundaries + per-region colour clustering
#
# Every strategy above decides ink from LUMINANCE, which is why red and gold,
# both mid-scale, get filed as background. `bgdist` was meant to fix that and
# measures Lab distance from the background correctly, but it then decides PER
# PIXEL against a global Otsu split of its own distance map, which puts red on
# the background side.
#
# This one changes the unit of decision, not the threshold. A learned lineart
# annotator (Informative Drawings, SIGGRAPH 2022) supplies closed boundaries;
# those boundaries partition the image into regions; each region votes with its
# MEDIAN colour, which is far more stable than any pixel. The split itself comes
# from 2-means over the regions' own lightness, so each logo sets its own
# cut rather than importing a global constant.
#
# Requires the separate lineart venv (torch). Absent it, the strategy raises and
# run_one skips it -- the other six are unaffected. See lineart_backend.py.
# ---------------------------------------------------------------------------

def _lineart_python() -> Path:
    """Interpreter for the annotator venv.

    Was hardcoded to the Linux layout (~/.venvs/.../bin/python), which is wrong
    on Windows twice over: the folder is Scripts, not bin, and the installer
    puts the venv under LOCALAPPDATA rather than the home directory. LINEART_PY
    lets the launcher say exactly where it went.
    """
    if env := os.environ.get('LINEART_PY'):
        return Path(env)
    if sys.platform == 'win32':
        return (Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData/Local'))
                / 't-tracer' / 'lineart-venv' / 'Scripts' / 'python.exe')
    return Path.home() / '.venvs' / 'png2svg-lineart' / 'bin' / 'python'


def _cache_dir() -> Path:
    """Activation-map cache, kept OUT of the project.

    It lived in _scripts/.lineart_cache, which is inside a OneDrive tree - the
    same mistake the app's .work folder made. These are .npy blobs, one per
    image per detector resolution, and syncing them is pure waste.
    """
    if env := os.environ.get('TT_CACHE_DIR'):
        return Path(env).expanduser()
    if sys.platform == 'win32':
        base = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData/Local'))
    else:
        base = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
    return base / 't-tracer' / 'lineart-cache'


LINEART_PY = _lineart_python()
LINEART_BACKEND = Path(__file__).resolve().parent / 'lineart_backend.py'
LINEART_CACHE = _cache_dir()
NEURAL_MAX_DIM = 1024          # region labelling at 8000px is pointless and slow


# ---------------------------------------------------------------------------
# Resource safety
#
# MEASURED on this machine, and these numbers drive everything below:
#   candidates.py worker, steady state ....  831 MB
#   torch annotator subprocess, peak ...... 3197 MB
#
# A naive "one worker per core" would therefore ask for 12 x 4 GB = ~48 GB on a
# 14 GB laptop. That does not thrash, it OOM-kills -- and this tool is meant to
# run alongside Bambu Studio, XCS, Affinity, LightBurn and Chrome, and later on
# machines whose specs nobody here controls. So: workers are sized by MEMORY,
# never by core count, and the annotator is serialised regardless of how many
# workers there are.
# ---------------------------------------------------------------------------

WORKER_MB = 900          # one candidates.py worker, steady state (measured 831)
ANNOTATOR_MB = 3400      # one torch subprocess, peak (measured 3197)
HEADROOM_MB = 3500       # left for the user's own applications. Not negotiable
                         # downward: the whole point is not being the process
                         # that kills their slicer mid-print-prep.


def _available_mb():
    """Free memory the OS will actually give us, or None if we cannot tell.

    Deliberately dependency-free -- psutil is not installed and this has to
    behave the same on the reviewer's Windows desktop as on this laptop.
    """
    try:                                            # Linux
        with open('/proc/meminfo') as fh:
            for line in fh:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    try:                                            # Windows
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong),
                        ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong),
                        ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong),
                        ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong),
                        ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
        st = _MS(); st.dwLength = ctypes.sizeof(_MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return int(st.ullAvailPhys) // (1024 * 1024)
    except Exception:
        pass
    return None


def auto_jobs(n_images, verbose=True):
    """How many workers this machine can afford RIGHT NOW.

    Binding constraint is memory, not cores. Cores only cap the answer -- and
    only at half of them, so the machine stays usable while this runs.
    """
    cores = max(1, (os.cpu_count() or 2) // 2)
    avail = _available_mb()
    if avail is None:                               # cannot measure -> be timid
        jobs, why = min(2, cores, n_images), 'memory unreadable, assuming 2'
    else:
        budget = avail - HEADROOM_MB - ANNOTATOR_MB
        jobs = max(1, budget // WORKER_MB)
        jobs = min(jobs, cores, n_images)
        why = (f'{avail} MB free, minus {HEADROOM_MB} MB for your apps '
               f'and {ANNOTATOR_MB} MB for the annotator')
    if verbose:
        print(f'-- {jobs} worker(s): {why}')
    return int(jobs)


@contextlib.contextmanager
def _annotator_slot():
    """Only ONE torch subprocess may exist at a time, machine-wide.

    3.4 GB each. Two concurrent is already most of this laptop; the worker
    count must not be able to multiply it. A lock file rather than a semaphore
    so it holds across separate `candidates.py` invocations too -- two terminals
    running batches should not be able to OOM each other.
    """
    LINEART_CACHE.mkdir(parents=True, exist_ok=True)
    lock = LINEART_CACHE / '.annotator.lock'
    fh = open(lock, 'w')
    try:
        try:
            import fcntl
            fcntl.flock(fh, fcntl.LOCK_EX)
        except ImportError:                         # Windows
            import msvcrt
            while True:
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1); break
                except OSError:
                    time.sleep(0.5)
        yield
    finally:
        try:
            fh.close()
        except Exception:
            pass


def _lineart_activation(rgb, res):
    """Line-strength map for `rgb`, cached by content hash across runs.

    `res` is part of the key on purpose: the map genuinely differs by detector
    resolution, and a key without it silently serves a stale 512px map after the
    resolution is raised -- which looks like the change having no effect.
    """
    key = hashlib.sha1(rgb.tobytes() + repr(rgb.shape).encode()
                       + f'r{res}'.encode()).hexdigest()[:16]
    cached = LINEART_CACHE / f'{key}.npy'
    if cached.exists():
        return np.load(cached).astype(np.float32)

    if not LINEART_PY.exists():
        raise RuntimeError(
            f'lineart venv missing at {LINEART_PY} -- `neural` needs it. '
            'Create it with torch + controlnet_aux, or drop `neural` from --strategies.')

    LINEART_CACHE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / 'in.png'
        cv2.imwrite(str(tmp), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        # Write to a PID-unique path and rename into place. os.replace is
        # atomic on POSIX and Windows, so a reader never sees a half-written
        # .npy and two workers racing on the same key both produce a valid
        # file. Without this, parallel runs can serve a truncated map -- the
        # same silent-wrong-data failure mode this cache already caused once
        # when the key was missing the detector resolution.
        staging = cached.with_suffix(f'.{os.getpid()}.tmp.npy')
        with _annotator_slot():
            # Re-check inside the lock: while we waited, another worker may
            # have computed this exact map. Recomputing it would cost 6.7s and
            # 3.4 GB for nothing.
            if cached.exists():
                return np.load(cached).astype(np.float32)
            r = subprocess.run([str(LINEART_PY), str(LINEART_BACKEND), str(tmp),
                                str(staging), str(res)],
                               capture_output=True, text=True, timeout=600)
        try:
            if r.returncode != 0 or not staging.exists():
                raise RuntimeError(
                    f'lineart backend failed: {r.stderr.strip()[-300:]}')
            os.replace(staging, cached)
        finally:
            staging.unlink(missing_ok=True)
    return np.load(cached).astype(np.float32)



def _lineart_lines(rgb, hi=0.15, lo=0.04, blur=1.0, stroke=1):
    """Working-size image + its hysteresis line mask, shared by every strategy
    built on the annotator.

    BLUR + HYSTERESIS, not a single hard cut. the reviewer on the hard-cut version:
    the lines "are rough, jagged things" with "little blobs hanging off
    everywhere". A hard threshold on a soft activation map lets the boundary
    wander wherever the signal sits near the cut, and any isolated pixel
    clearing it becomes its own component. Hysteresis makes a STRONG pixel seed
    a line before following it down to `lo`, so strays seed nothing.
    """
    h0, w0 = rgb.shape[:2]
    scale = min(1.0, NEURAL_MAX_DIM / max(h0, w0))
    small = (cv2.resize(rgb, (max(1, int(w0 * scale)), max(1, int(h0 * scale))),
                        interpolation=cv2.INTER_AREA) if scale < 1.0 else rgb)

    # Detect at the working size, not the 512 default -- see _lineart_activation.
    act = _lineart_activation(small, max(small.shape[:2]))
    if blur > 0:
        act = cv2.GaussianBlur(act, (0, 0), blur)
    strong, weak = act > hi, act > lo
    _, lbl_w = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    keep = np.unique(lbl_w[strong])
    keep = keep[keep != 0]
    lines = np.isin(lbl_w, keep).astype(np.uint8)
    if stroke > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * stroke + 1,) * 2)
        lines = cv2.dilate(lines, k, iterations=1)
    return small, lines, scale


def strat_neural(rgb, alpha=None, hi=0.15, lo=0.04, blur=1.0, stroke=1,
                 min_region=12):
    """Learned boundaries + per-region colour clustering.

    The line mask uses BLUR + HYSTERESIS, not a single hard cut. the reviewer on the
    hard-cut version: the lines "are rough, jagged things" with "little blobs
    hanging off everywhere". That is what a hard threshold does to a soft
    activation map -- the boundary wanders wherever the signal sits near the
    cut, and any isolated pixel clearing it becomes its own component.
    Hysteresis requires a STRONG pixel to seed a line before following it down
    to `lo`, so strays seed nothing; a 1px blur settles the boundary first.

    Measured on two logos: connected components 57 -> 18 and 30 -> 19, edges
    solid instead of dotted. A separate small-component filter changed nothing
    -- hysteresis already subsumes it. blur 2.0 broke the seal ring into dashes.
    """
    small, lines, scale = _lineart_lines(rgb, hi, lo, blur, stroke)
    h0, w0 = rgb.shape[:2]

    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0                  # OpenCV packs L into 0..255
    n, labels = cv2.connectedComponents((1 - lines).astype(np.uint8), connectivity=4)

    ids, Ls, sizes = [], [], []
    for rid in range(1, n):
        sel = labels == rid
        c = int(sel.sum())
        if c < min_region:
            continue
        ids.append(rid); Ls.append(float(np.median(lab[..., 0][sel]))); sizes.append(c)

    ink = np.zeros(lines.shape, dtype=bool)
    if ids:
        L = np.asarray(Ls, dtype=np.float32)
        wgt = np.asarray(sizes, dtype=np.float64)
        lo, hi = float(L.min()), float(L.max())
        if hi - lo > 1e-3:
            c = np.array([lo, hi], dtype=np.float32)
            for _ in range(25):                    # 1-D 2-means, area weighted
                a = np.argmin(np.abs(L[:, None] - c[None, :]), axis=1)
                for j in (0, 1):
                    m = a == j
                    if m.any():
                        c[j] = np.average(L[m], weights=wgt[m])
            dark = np.argmin(np.abs(L[:, None] - c[None, :]), axis=1) == 0
            for rid, d in zip(ids, dark):
                if d:
                    ink[labels == rid] = True

    ink |= lines.astype(bool)                      # the boundaries themselves burn
    if scale < 1.0:
        ink = cv2.resize(ink.astype(np.uint8), (w0, h0),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
    return ink


def strat_nested(rgb, alpha=None, hi=0.15, lo=0.04, blur=1.0, stroke=1,
                 min_region=12, dE=12.0):
    """Ink decided by NESTING: each region is judged against the region that
    contains it, never against a global split.

    Why this exists. Every threshold-driven strategy fails the same way on
    `olive-disc logo`: the olive disc is mid-luminance, so it lands on the ink
    side, and the black bird sitting on top of it is already ink -- the two
    merge and the bird is destroyed IN THE MASK, before the tracer runs. That
    is the "big black blob". Thickening lines cannot recover it; there is
    nothing left to recover. `neural` aims at this with per-region colour but
    still forces ONE global 2-way split, and this image has four tonal levels
    (black / olive / tan / white).

    The correct assignment is not a threshold at all -- it is a 2-COLOURING of
    the containment tree. Background is bare. A region flips relative to its
    parent when it is a genuinely different colour from its parent, and
    inherits when it is not. Contrast is therefore preserved by construction at
    every nesting depth: a child that differs from its parent can never come
    out the same as its parent, whatever its absolute luminance is.

    Third defect in this project solved by the same move -- nesting carries
    information that size and threshold cannot. See also the letter counters in
    `clean_mask`, and the protected close.

    `dE` is the CIELAB distance at which two regions count as different
    colours. It is a perceptual "is this the same paint" question, not a
    tuning knob fitted to any image.
    """
    small, lines, scale = _lineart_lines(rgb, hi, lo, blur, stroke)
    h0, w0 = rgb.shape[:2]

    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0                   # OpenCV packs L into 0..255
    n, labels = cv2.connectedComponents((1 - lines).astype(np.uint8), connectivity=4)
    if n <= 1:
        return cv2.resize(lines, (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool)

    # Region colours. Small regions keep a label but are never given their own
    # decision -- they inherit, so speckle cannot flip a field.
    sizes = np.bincount(labels.ravel(), minlength=n)
    colour = np.zeros((n, 3), dtype=np.float32)
    for rid in range(1, n):
        sel = labels == rid
        colour[rid] = np.median(lab[sel], axis=0)

    # Line pixels belong to no region, so no two regions ever touch directly.
    # Assign every line pixel to its nearest region first, then adjacency is
    # just "different labels side by side".
    _, idx = distance_transform_edt(lines.astype(bool), return_indices=True)
    filled = labels[idx[0], idx[1]]

    adj = [set() for _ in range(n)]
    for a, b in ((filled[:, :-1], filled[:, 1:]), (filled[:-1, :], filled[1:, :])):
        d = a != b
        for u, v in np.unique(np.stack([a[d], b[d]], axis=1), axis=0):
            if u and v:
                adj[u].add(v); adj[v].add(u)

    # Root at the background: the region owning the most border pixels.
    border = np.concatenate([filled[0], filled[-1], filled[:, 0], filled[:, -1]])
    root = int(np.bincount(border, minlength=n)[1:].argmax() + 1)

    # BFS out from the background. The discovering neighbour IS the containing
    # region, because the only way to reach a nested region is through it.
    ink_of = np.zeros(n, dtype=bool)
    seen = np.zeros(n, dtype=bool)
    seen[root] = True
    queue = [root]
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if seen[v]:
                continue
            seen[v] = True
            differs = (sizes[v] >= min_region
                       and float(np.linalg.norm(colour[v] - colour[u])) > dE)
            ink_of[v] = (not ink_of[u]) if differs else ink_of[u]
            queue.append(v)

    # What to do with the annotator's own lines. Both obvious answers are wrong,
    # and they fail in opposite directions:
    #
    #   `ink |= lines`      -- burn every boundary. A 3px dilated line is
    #                          invisible on a large shape and a BLOB on a
    #                          letterform. the reviewer on this version: "the turkey
    #                          foot is exactly right and the interior circle.
    #                          the words all have the blobs". The identical
    #                          `ink |= lines` in `neural` is why its letters
    #                          have read as fattened and "nasty" since 09-03.
    #   `ink = ink_of[...]` -- burn none. On thin artwork the dilated line and
    #                          the artwork are THE SAME PIXELS, so the letter
    #                          has no interior region of its own and vanishes
    #                          entirely. Measured: the banner lettering reduced to
    #                          disconnected fragments.
    #
    # Being a boundary was never the right question. The right one is whether
    # there is actually dark paint at that pixel in the source. A line drawn
    # along a real black stroke is artwork and burns; a line marking a mere
    # colour change (olive meeting tan) is scaffolding and inherits its host
    # region's state. This also undoes the dilation's damage for free: the
    # pixels a 3x3 dilation pushes out into white stock are white in the
    # source, so they fail the test and never burn.
    ink = ink_of[filled]
    line_px = lines.astype(bool)
    if line_px.any():
        host = colour[filled]                      # each pixel's region colour
        differs = np.linalg.norm(lab - host, axis=-1) > dE
        darker = lab[..., 0] < host[..., 0]
        ink = ink | (line_px & differs & darker)
    if scale < 1.0:
        ink = cv2.resize(ink.astype(np.uint8), (w0, h0),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
    return ink


def strat_composite(rgb, alpha=None, solid_frac=0.015, knock_frac=0.0006,
                    max_knock_frac=0.5, **kw):
    """`otsu` keeps everything; `nested` is allowed to punch holes in it.

    the reviewer's design, and the reason it works: the two masks disagree in exactly
    ONE interesting way. `nested` knows a mid-luminance field is a field;
    `otsu` reads it as ink and swallows whatever sits on top of it. This takes
    that single piece of knowledge and nothing else.

    So `nested` is consulted only INSIDE a large solid `otsu` region, and only
    to remove ink, never to add it. Everything `otsu` already does well -- the
    lettering above all -- it keeps untouched, which is what sidesteps the
    blob problem rather than solving it: the annotator's 3px line and a 3px
    letter stroke are the same pixels, and no rule over that mask separates
    them (three attempts, three different failures, 2026-09-05).

    A knockout must be a MINORITY of the region it sits in. That is the
    premise restated, not an extra rule: the composite exists because `nested`
    can see something INSIDE an `otsu` blob. When `nested` wants to remove most
    of a field, it is not finding a detail -- it is re-classifying the field
    itself, and on art whose background is genuinely dark (NSWU-2, VX-30, the
    NOPD SWAT patch, the Saints fleur-de-lis) that strips the design to hollow
    outlines and `otsu` was right all along.
    Measured over the first batch of 10 plus `olive-disc logo`, as
    largest-knockout / its-host: the wins sit at 0.14 and 0.23, the
    destructions at 0.70, 0.84, 0.99, 0.99 and 1.00. `max_knock_frac` is the
    ordinary majority boundary sitting in that 0.47-wide gap, not a value
    fitted to those numbers.

    KNOWN CEILING, and it is not a small one: this can only ever REMOVE ink
    from `otsu`. It is aimed at "big black mess" -- 7 of 17 rejections, the
    largest single failure mode -- and at nothing else. Where `otsu` has
    already dropped saturated mid-luminance colour (red chevrons, a gold
    fleur-de-lis), the composite cannot bring it back. That failure mode needs
    a different answer.
    """
    base = strat_otsu(rgb)
    field = strat_nested(rgb, alpha=alpha, **kw)

    total = float(base.size)
    n_b, lbl_b = cv2.connectedComponents(base.astype(np.uint8), connectivity=8)
    sizes_b = np.bincount(lbl_b.ravel(), minlength=n_b)

    # Candidate knockouts: burned by otsu, left bare by nested.
    knock = base & ~field
    n_k, lbl_k = cv2.connectedComponents(knock.astype(np.uint8), connectivity=8)
    sizes_k = np.bincount(lbl_k.ravel(), minlength=n_k)

    ink = base.copy()
    for k in range(1, n_k):
        if sizes_k[k] < knock_frac * total:
            continue                                # speckle, not a knockout
        sel = lbl_k == k
        hosts = np.bincount(lbl_b[sel], minlength=n_b)
        hosts[0] = 0
        if not hosts.any():
            continue
        host = int(hosts.argmax())
        if sizes_b[host] < solid_frac * total:      # not a large solid field
            continue
        if sizes_k[k] > max_knock_frac * sizes_b[host]:
            continue                                # re-classifying the field, not a detail
        ink[sel] = False
    return ink


STRATEGIES = {
    'otsu':       strat_otsu,
    'bgdist':     strat_bgdist,
    'kmeans':     strat_kmeans,
    'linework':   strat_linework,
    'silhouette': strat_silhouette,
    'nested':     strat_nested,
    'composite':  strat_composite,
}

# `plate` demoted 2026-09-05 at the reviewer's call. It was the most elaborate
# strategy and the most iterated on, and it never won a single pick in the
# project's history; once rows could hold a SET, it appeared in only 2 of 17
# shippable sets. Sophistication has not been paying here. Same treatment as
# `neural` -- still reachable via --strategies, off the default sheet.
#
# `edges` is still reachable with --strategies but is off the default sheet:
# it never won a pick, and it exists for photographs of physical patches,
# which is not the work in front of this pipeline right now. Six tiles is
# already the limit of a glance-and-choose decision.
# `inotsu` isolates the artwork-only-population idea. Kept reachable because it
# proved the mechanism on `red-eye logo`, but off the sheet: it differs from plain
# `otsu` by 0.15-3.4% of pixels, which is sheet clutter, not a candidate.
OPTIONAL = {'edges': strat_edges, 'sauvola': strat_sauvola,
            'neural': strat_neural, 'plate': strat_plate,
            'inotsu': strat_inotsu, 'triotsu': strat_triotsu}
ALL_STRATEGIES = {**STRATEGIES, **OPTIONAL}

# Strategies that can use the source's alpha channel when one exists.
# `composite` was listed here when it was added, and that was WRONG: it is
# `strat_otsu` (not alpha-aware) plus `strat_nested`, which accepts an `alpha`
# argument and never reads it. Nothing broke -- the value was passed and
# dropped -- but the set is a claim about behaviour, and a false claim here is
# how someone later concludes transparency is handled when it is not.
# Making `nested` genuinely alpha-aware is a real option, not a bug fix: alpha
# identifies the background region exactly, where the BFS currently infers it
# from whichever region owns the most border pixels.
ALPHA_AWARE = {'silhouette', 'linework', 'plate', 'inotsu', 'triotsu'}


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
        for i in np.where(~keep)[0]:
            if i == 0 or sizes[i] == 0:
                continue
            comp = (lab == i).astype(np.uint8)
            ring = (cv2.dilate(comp, k3, iterations=1) > 0) & ~comp.astype(bool)
            around = np.unique(holes[ring])
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
    return paths


# ---------------------------------------------------------------------------
# SVG emission and rasterization
# ---------------------------------------------------------------------------

def paths_to_svg(paths, w, h, ink='#000000', background=None, height_mm=None):
    """Emit one <path> holding every subpath, fill-rule evenodd so interior
    holes knock out correctly. Exactly one fill color - the geometry IS the
    job, per _shared/laser-output-rules.md.

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


def rasterize(paths, w, h, ss=2):
    """Render the emitted geometry back to a boolean mask.

    This deliberately rasterizes the Beziers we actually wrote out, not the
    mask we started from - so the hygiene score describes the delivered SVG,
    including anything the curve fit changed.
    """
    canvas = np.zeros((h * ss, w * ss), np.uint8)
    for segs in paths:
        pts = _sample_beziers(segs, per_seg=16) * ss
        if len(pts) < 3:
            continue
        # XOR each subpath in turn - this IS the even-odd rule, so a contour
        # nested inside another knocks a hole rather than filling it solid.
        layer = np.zeros_like(canvas)
        cv2.fillPoly(layer, [pts.astype(np.int32)], 255)
        canvas = cv2.bitwise_xor(canvas, layer)
    return canvas > 0


def count_nodes(paths):
    return sum(len(segs) for segs in paths)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_one(img_path, out_root, args):
    import hygiene

    from PIL import Image
    raw = Image.open(img_path)
    alpha = np.array(raw.convert('RGBA'))[:, :, 3] if 'A' in raw.getbands() else None

    rgb = np.array(flatten_to_white(img_path))
    # Order-catalogue art runs to 134 MP. With --scale supersampling on top that
    # is billions of pixels and the run never returns. Engraved artwork is ~32mm
    # tall, so detail past a couple of thousand pixels cannot reach the plate.
    # Every image in the original 15-image test set is <=2000px, so the 2048
    # default is a no-op on all prior calibration.
    if getattr(args, 'max_dim', 0) and max(rgb.shape[:2]) > args.max_dim:
        _s = args.max_dim / max(rgb.shape[:2])
        rgb = cv2.resize(rgb, (max(1, int(rgb.shape[1] * _s)), max(1, int(rgb.shape[0] * _s))),
                         interpolation=cv2.INTER_AREA)
        if alpha is not None:
            alpha = cv2.resize(alpha, (rgb.shape[1], rgb.shape[0]),
                               interpolation=cv2.INTER_AREA)
    h, w, _ = rgb.shape
    src_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    out_dir = out_root / img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    names = args.strategies.split(',') if args.strategies else list(STRATEGIES)

    # `nested` and `composite` both need the same activation map. Computing it
    # once up front means the second one is always a cache hit, and -- once
    # images run in parallel -- that two workers on the SAME image can never
    # both pay the 6.7s cold cost or race on the same cache key.
    if any(n.strip() in ('nested', 'composite', 'neural') for n in names):
        try:
            _scale = min(1.0, NEURAL_MAX_DIM / max(rgb.shape[:2]))
            _small = (cv2.resize(rgb, (max(1, int(rgb.shape[1] * _scale)),
                                       max(1, int(rgb.shape[0] * _scale))),
                                 interpolation=cv2.INTER_AREA)
                      if _scale < 1.0 else rgb)
            _lineart_activation(_small, max(_small.shape[:2]))
        except Exception as e:
            print(f'  ! annotator unavailable ({type(e).__name__}), '
                  f'lineart strategies will fail')

    results = {}
    previews = []

    for name in names:
        fn = ALL_STRATEGIES.get(name.strip())
        if fn is None:
            print(f'  ! unknown strategy {name!r}, skipping')
            continue

        try:
            mask = fn(rgb, alpha) if name.strip() in ALPHA_AWARE else fn(rgb)
        except Exception as e:
            print(f'  {name:<8} FAILED: {type(e).__name__}: {e}')
            continue

        if args.invert:
            mask = ~mask
        mask = clean_mask(mask, src_gray, args.close, args.open, args.min_area_frac)

        ink_frac = float(mask.mean())
        if ink_frac < 0.001 or ink_frac > 0.98:
            print(f'  {name:<8} degenerate ({ink_frac:.1%} ink) - not emitted')
            results[name] = {'rejected': f'degenerate ink fraction {ink_frac:.3f}'}
            continue

        paths = mask_to_paths(mask, scale=args.scale, smooth=args.smoothing,
                              tol=args.tol)
        if not paths:
            print(f'  {name:<8} produced no usable contours')
            results[name] = {'rejected': 'no contours'}
            continue

        svg = paths_to_svg(paths, w, h,
                           background='#FFFFFF' if args.bg else None,
                           height_mm=args.height_mm)
        (out_dir / f'{name}.svg').write_text(svg)

        rendered = rasterize(paths, w, h, ss=args.raster_ss)
        sc = hygiene.score(rendered, src_gray, paths, count_nodes(paths),
                           height_mm=args.height_mm, ss=args.raster_ss, src_h=h,
                           src_rgb=rgb)
        results[name] = sc

        prev = (~rendered * 255).astype(np.uint8)
        prev = cv2.resize(prev, (w, h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f'{name}.png'), prev)
        previews.append((name, prev))

    # Fidelity only means something compared across candidates from the same
    # image, so ranking happens once every candidate exists - never inside the
    # loop, where a strategy would be graded against itself.
    hygiene.finalize(results)

    for name, sc in sorted(results.items(),
                           key=lambda kv: -kv[1].get('overall', -1)):
        if 'hygiene' not in sc:
            print(f'  {name:<8} rejected: {sc["rejected"]}')
            continue
        print(f'  {name:<8} {sc["overall"]:>5.1f} overall  '
              f'(hygiene {sc["hygiene"]:>5.1f}, detail {sc["fidelity_rel"]:.2f})  '
              f'{sc["nodes"]:>5} nodes  {sc["paths"]:>3} paths  -- {sc["headline"]}')

    (out_dir / 'metrics.json').write_text(json.dumps(results, indent=2))
    if previews:
        # All sheets land in one folder. Per-image folders are right for the
        # deliverables, but reviewing means flipping through every sheet in a
        # row, and that should be one directory of images, not fifteen clicks.
        sheets = out_root / '_sheets'
        sheets.mkdir(parents=True, exist_ok=True)
        _contact_sheet([(n, p, results[n]) for n, p in previews],
                       rgb, sheets / f'{img_path.stem}.png')
    return results


def rank_candidates(metrics):
    """The ONE ordering. Returns [(name, score_dict), ...] best first.

    Both the contact sheet and pick.py must number candidates identically --
    the reviewer reads a number off a tile and types it into the picker, so any
    disagreement silently records the wrong strategy. They used to differ: the
    sheet was a stable sort on -overall (ties kept run order) while pick.py
    sorted tuples in reverse (ties broken by name, DESCENDING). On an image
    like a high-node logo, where five candidates all score 0.0, those are different
    orders. Name ascending as the tie-break, defined once, used by both.
    """
    return sorted(((k, v) for k, v in metrics.items() if 'overall' in v),
                  key=lambda kv: (-kv[1].get('overall', -1), kv[0]))


def _contact_sheet(previews, source_rgb, path, tile_h=340):
    """Source first, then every candidate ranked best hygiene first.

    Tiles carry their PICK NUMBER, large, at the left of the label. Without it
    the reviewer has to hold a name-to-number mapping in their head and glance
    between the sheet and the terminal for every choice -- on a batch of ten
    that is a hundred context switches and a real source of misclicks.
    """
    def fit(img, is_color=False):
        ih, iw = img.shape[:2]
        s = tile_h / ih
        r = cv2.resize(img, (max(1, int(iw * s)), tile_h),
                       interpolation=cv2.INTER_AREA)
        return r if is_color else cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)

    order = [n for n, _ in rank_candidates({n: s for n, _, s in previews})]
    by_name = {n: (p, s) for n, p, s in previews}
    tiles = [('', 'SOURCE', fit(cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR), True), None)]
    tiles += [(str(i), n, fit(by_name[n][0]), by_name[n][1])
              for i, n in enumerate(order, 1)]

    pad, label_h = 12, 46
    total_w = sum(t[2].shape[1] for t in tiles) + pad * (len(tiles) + 1)
    sheet = np.full((tile_h + label_h + pad * 2, total_w, 3), 245, np.uint8)

    x = pad
    for num, name, img, score in tiles:
        sheet[label_h:label_h + tile_h, x:x + img.shape[1]] = img
        cv2.rectangle(sheet, (x, label_h), (x + img.shape[1], label_h + tile_h),
                      (200, 200, 200), 1)
        nx = x
        if num:
            cv2.putText(sheet, num, (x, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.95, (10, 10, 10), 2)
            nx = x + (34 if len(num) == 1 else 48)
        cv2.putText(sheet, name, (nx, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (20, 20, 20), 2)
        if score and 'overall' in score:
            cv2.putText(sheet, f'{score["overall"]:.0f}/100  {score["headline"]}',
                        (nx, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1)
        x += img.shape[1] + pad

    cv2.imwrite(str(path), sheet)


def _worker_init():
    """Each worker keeps its own libraries single-threaded -- see --jobs.

    Also de-prioritised: this is a background batch, and the user is expected
    to be actively working in Bambu Studio / XCS / Affinity / LightBurn while
    it runs. Losing a few percent of throughput to keep their UI responsive is
    the right trade every time.
    """
    try:
        os.nice(10)
    except Exception:
        try:                                        # Windows
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
        except Exception:
            pass
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    cv2.setNumThreads(1)


def _run_one_quiet(img_path, out_root, args):
    """run_one with its per-strategy chatter suppressed.

    Interleaved progress lines from N workers are unreadable, and worse,
    misleading -- they no longer correspond to the order anything finished.
    The parent prints one line per completed image instead. metrics.json still
    holds every number.
    """
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return run_one(img_path, out_root, args)


def main():
    ap = argparse.ArgumentParser(
        description='Emit N candidate laser SVGs per image, scored for vector hygiene.')
    ap.add_argument('--only', help='process just this filename')
    ap.add_argument('--strategies', help='comma-separated subset of: ' + ','.join(ALL_STRATEGIES))
    ap.add_argument('--invert', action='store_true',
                    help='swap ink/background (black-coated brass: ink = ablated = bright)')
    ap.add_argument('--scale', type=int, default=6, help='supersample factor before tracing')
    ap.add_argument('--smoothing', type=float, default=3.0, help='spline smoothing per point')
    ap.add_argument('--tol', type=float, default=0.6,
                    help='max Bezier deviation in source px (lower = more nodes, tighter fit)')
    ap.add_argument('--close', type=int, default=2, help='morphological close radius (joins hairline breaks)')
    # Default 0. An opening of radius 1 removes anything thinner than about
    # 3px - which on gold-shield crest silently erased the gold shield and scroll
    # outlines, the exact linework the job is about. Speckle is already handled
    # by --min-area-frac, which drops small islands without touching long thin
    # ones, so the opening was costing real detail to solve a solved problem.
    ap.add_argument('--open', type=int, default=0,
                    help='morphological open radius (0 = off; >0 erases strokes thinner than ~2*r+1 px)')
    ap.add_argument('--min-area-frac', type=float, default=0.0004,
                    help='drop islands smaller than this fraction of total ink')
    ap.add_argument('--height-mm', type=float, default=32.0,
                    help='engraved height, for per-mm reporting (0 to disable)')
    ap.add_argument('--raster-ss', type=int, default=2, help='supersample for scoring raster')
    ap.add_argument('--src-dir', default=None,
                    help='where --resheet should look for source images '
                         '(default: corpus/)')
    ap.add_argument('--resheet', action='store_true',
                    help='rebuild contact sheets from an existing --out folder '
                         'and exit. Re-renders labels/numbering without '
                         're-running any strategy.')
    ap.add_argument('--jobs', '-j', type=int, default=0,
                    help='parallel workers across IMAGES (not within one '
                         'image). Default 0 = auto: sized by FREE MEMORY, '
                         'capped at half the cores, leaving headroom for your '
                         'other applications. 1 = serial. An explicit N above '
                         'the auto value is honoured but is your own risk.')
    ap.add_argument('--bg', action='store_true', help='include a white background rect')
    ap.add_argument('--in', dest='input_dir', default=None,
                    help='read images from this directory instead of the drop folder')
    ap.add_argument('--max-dim', type=int, default=2048,
                    help='downscale sources longer than this on the long edge '
                         '(0 = off). Guards against 100+ MP catalogue art.')
    ap.add_argument('--out', default='candidates', help='output folder name')
    args = ap.parse_args()

    if args.height_mm <= 0:
        args.height_mm = None


    # Determinism before speed, unconditionally and in every path.
    #
    # `cv2.kmeans` results depend on OpenCV's thread count: the parallel
    # reduction order changes the floating-point sums, which flips pixels that
    # sit on a cluster boundary. Measured 2026-09-06: running the same 10
    # images at -j 1 and -j 4 produced 3 differing SVGs out of 39, all in
    # `kmeans` and `linework`.
    #
    # That is not a cosmetic difference. `picks.jsonl` is this project's only
    # ground truth and every row points at a specific SVG; if re-running the
    # same image can produce different geometry depending on a scheduling flag,
    # a recorded pick cannot be reproduced and the calibration set quietly
    # stops meaning what it says. Worth more than the threading gains.
    #
    # The RNG seed is the other half. `cv2.kmeans` is called with
    # KMEANS_PP_CENTERS, and kmeans++ picks its initial centres RANDOMLY from
    # OpenCV's global RNG. Fixing the thread count alone still left 4 of 20
    # outputs differing between -j 1 and -j 4, because a worker process does
    # not start with the parent's RNG state. Seeding it makes `kmeans` and
    # `linework` reproducible across processes, machines and job counts.
    #
    # Seeding once per PROCESS is not enough and the measurement says so: two
    # parallel runs with identical flags still differed, because one worker
    # handles several images and the RNG advances between them, so a result
    # depends on what that worker happened to process first. The seed has to be
    # set immediately before each `cv2.kmeans` call -- see `strat_kmeans`,
    # `strat_linework` and `strat_plate`.
    cv2.setNumThreads(1)

    folder = Path(__file__).resolve().parent
    out_root = folder / args.out

    if args.resheet:
        n = 0
        for d in sorted(x for x in out_root.iterdir()
                        if x.is_dir() and not x.name.startswith('_')):
            mp = d / 'metrics.json'
            if not mp.exists():
                continue
            metrics = json.loads(mp.read_text())
            previews = []
            for name, sc in metrics.items():
                png = d / f'{name}.png'
                if 'overall' in sc and png.exists():
                    previews.append((name, cv2.imread(str(png),
                                                      cv2.IMREAD_GRAYSCALE), sc))
            if not previews:
                continue
            src = None
            for cand in (folder / 'corpus').glob(d.name + '.*'):
                src = cand
                break
            if src is None:
                for cand in Path(args.src_dir or folder).glob(d.name + '.*') \
                        if args.src_dir else []:
                    src = cand
                    break
            rgb = (np.array(flatten_to_white(src)) if src
                   else np.full(previews[0][1].shape + (3,), 255, np.uint8))
            sheets = out_root / '_sheets'
            sheets.mkdir(parents=True, exist_ok=True)
            _contact_sheet(previews, rgb, sheets / f'{d.name}.png')
            n += 1
        print(f'rebuilt {n} sheet(s) in {out_root / "_sheets"}')
        return

    # --in points the drop-folder convention at any directory; without it the
    # behaviour is exactly as before (folder, then "converter script testers/").
    if getattr(args, 'input_dir', None):
        src_dir = Path(args.input_dir)
        if not src_dir.is_absolute():
            src_dir = folder / src_dir
        if not src_dir.is_dir():
            print(f'--in: no such directory {src_dir}')
            return
        images = sorted(p for p in src_dir.iterdir()
                        if p.suffix.lower() in IMAGE_EXTS and p.is_file())
        if args.only:
            images = [p for p in images if p.name == args.only]
        if not images:
            print(f'No images in {src_dir}.')
            return
    else:
      images = sorted(p for p in folder.iterdir()
                    if p.suffix.lower() in IMAGE_EXTS and p.is_file())
      if args.only:
        images = [p for p in images if p.name == args.only]
      if not images:
        # Fall back to the existing test-image folder rather than doing nothing.
        testers = folder / 'converter script testers'
        if testers.is_dir():
            images = sorted(p for p in testers.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if args.only:
                images = [p for p in images if p.name == args.only]

    if not images:
        print(f'No images found next to {folder} or in "converter script testers/".')
        return

    if args.jobs == 0:
        jobs = auto_jobs(len(images))
    else:
        jobs = max(1, min(args.jobs, len(images)))
        if jobs > 1:
            safe = auto_jobs(len(images), verbose=False)
            if jobs > safe:
                print(f'-- WARNING: -j {jobs} exceeds what this machine can '
                      f'safely afford right now ({safe}). Each worker holds '
                      f'~{WORKER_MB} MB and the annotator peaks at '
                      f'~{ANNOTATOR_MB} MB. Continuing as asked.')
    if jobs > 1:
        # Images share no state, so this is the axis that actually parallelises.
        # It does NOT speed up a single image -- that would need the strategy
        # loop split instead, which is a separate change.
        #
        # OMP_NUM_THREADS=1 in the workers is load-bearing: OpenCV and NumPy
        # each spawn their own thread pools, and N workers x M threads
        # oversubscribes the CPU and runs SLOWER than serial.
        import concurrent.futures as _cf
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        cv2.setNumThreads(1)
        print(f'-- {len(images)} images across {jobs} workers')
        with _cf.ProcessPoolExecutor(max_workers=jobs,
                                     initializer=_worker_init) as ex:
            futs = {ex.submit(_run_one_quiet, img, out_root, args): img
                    for img in images}
            for i, f in enumerate(_cf.as_completed(futs), 1):
                img = futs[f]
                try:
                    f.result()
                    print(f'  [{i}/{len(images)}] {img.name}')
                except Exception as e:
                    print(f'  [{i}/{len(images)}] {img.name} FAILED: '
                          f'{type(e).__name__}: {e}')
    else:
        for img in images:
            print(f'-- {img.name}')
            run_one(img, out_root, args)

    print(f'\nWrote {out_root}')


if __name__ == '__main__':
    main()
