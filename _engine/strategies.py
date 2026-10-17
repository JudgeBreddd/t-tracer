#!/usr/bin/env python3
"""strategies.py - T-Tracer

The binarization strategies: RGB in, boolean ink mask out.

One function per strategy, plus the registry at the foot of the file. To add
one, write the function here and register it in `STRATEGIES` (on the sheet) or
`OPTIONAL` (CLI only). Everything before and after a strategy -- loading,
cleanup, vectorising, scoring -- is shared and lives elsewhere; see
`_engine/CONTEXT.md`.

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

from tracestate import _memo
from masks import (_background_color, _between, _keep_white_channels,
                   _kmeans_lab, _lightness, _pick_k, _silhouette_of,
                   _true_lab)
from lineart import _lineart_lines



def strat_otsu(rgb):
    """Global Otsu on luminance. The baseline every tracer starts with.
    Fails when two important regions share a lightness (red on navy)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binv > 0


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


def strat_kmeans_layered(rgb, k=0, min_frac=0.015, k_max=8, halo_px=3.0,
                         merge_de=15.0, core_r=None, min_core=None,
                         _return_labels=False):
    """`kmeans`, but the colour split is KEPT all the way to the SVG.

    Same Lab k-means, same border-majority background solve. Instead of
    collapsing every non-background cluster into one boolean mask, each
    cluster becomes its own layer with its own representative colour, so the
    output is one <path> per colour and a recolour is an attribute edit.

    Returns a list of (bool mask, (r, g, b)) in stacking order: lightest first,
    darkest last (drawn on top) - the same 'darker is ink' reading `kmeans`
    already makes, so where clusters meet at an anti-aliased edge the darker
    colour wins the boundary.

    Differences from `kmeans`, both deliberate:
      * k is chosen per image (`_pick_k`) unless given; fixed k=5 over-splits
        a 3-colour patch into anti-aliasing bands and under-splits a 7-colour one.
      * clusters under `min_frac` of the artwork are noise, not colours: their
        pixels are merged into the nearest surviving cluster in Lab.

    This does NOT decide which layers are 'ink' for a single-colour burn. That
    is a discrete call over 2-7 cluster colours instead of a per-pixel one,
    and the per-layer metadata in metrics.json (`layers`) is what a later
    classifier or a human would decide it from.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w, _ = lab.shape
    flat = lab.reshape(-1, 3)
    if k <= 0:
        k = _pick_k(flat, k_max=k_max)
    k = min(k, len(np.unique(flat[::max(1, len(flat) // 50000)], axis=0)))
    if k < 2:
        return []
    _, labels, centers = _kmeans_lab(flat, k)
    labels = labels.reshape(h, w)

    b = max(1, min(4, h // 4, w // 4))
    border = np.concatenate([
        labels[:b, :].ravel(), labels[-b:, :].ravel(),
        labels[:, :b].ravel(), labels[:, -b:].ravel(),
    ])
    bg_label = int(np.bincount(border, minlength=k).argmax())

    # Three kinds of cluster are NOT colours, and each is folded into the
    # nearest surviving centre (Lab distance). No second k-means, so nothing
    # new to seed.
    #   1. too small      - under `min_frac` of the artwork
    #   2. a halo         - thin (mean local thickness under ~3px) AND its
    #                       colour lies between two other clusters' colours:
    #                       the anti-aliasing ramp where two flat colours meet.
    #                       Thin alone is not enough - a gold keyline is thin
    #                       and is a colour; gold is not between black and white.
    #   3. a near-twin    - within `merge_de` of another centre (shading,
    #                       JPEG noise), folded into the larger of the two
    areas = np.bincount(labels.ravel(), minlength=k).astype(np.float64)
    art_area = areas.sum() - areas[bg_label]
    if art_area <= 0:
        return []
    alive = list(range(k))
    tlab = _true_lab(centers)

    def fold(i, into):
        nonlocal labels
        labels = np.where(labels == i, into, labels)
        areas[into] += areas[i]
        areas[i] = 0
        alive.remove(i)

    def nearest(i):
        cand = [j for j in alive if j != i]
        d = np.linalg.norm(tlab[cand] - tlab[i], axis=1)
        return cand[int(np.argmin(d))]

    for i in sorted(range(k), key=lambda j: areas[j]):
        if i == bg_label or len(alive) <= 2:
            continue
        if areas[i] / art_area < min_frac:
            fold(i, nearest(i))
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for i in sorted([j for j in alive if j != bg_label], key=lambda j: areas[j]):
        if len(alive) <= 2:
            break
        m = labels == i
        dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
        thin = 2.0 * float(dt[m].mean()) < halo_px
        if thin:
            others = [j for j in alive if j != i]
            end = _between(tlab[i], tlab[others], want_pair=True)
            if end is not None:
                a, b, nearer = end
                # A halo sits BETWEEN its two colours in the image too: its
                # ring touches both. Thin grey detail drawn inside black (a
                # lighthouse railing) is between black and white in colour
                # but touches only black - that is artwork, and it stays.
                ring = (cv2.dilate(m.astype(np.uint8), k3) > 0) & ~m
                nb = np.bincount(labels[ring], minlength=k) if ring.any() else np.zeros(k)
                tot = max(1, nb.sum())
                if nb[others[a]] / tot >= 0.2 and nb[others[b]] / tot >= 0.2:
                    fold(i, others[nearer])
    changed = True
    while changed and len(alive) > 2:
        changed = False
        for i in sorted([j for j in alive if j != bg_label], key=lambda j: areas[j]):
            j = nearest(i)
            if j != bg_label and np.linalg.norm(tlab[i] - tlab[j]) < merge_de:
                fold(i, j)
                changed = True
                break
    if len(alive) < 2:
        return []

    # Anti-aliased boundaries: where two flat colours meet, the mixed pixels
    # get assigned to whichever cluster is nearest in Lab - often a THIRD
    # colour (blue|yellow -> the green cluster), which draws a one-pixel
    # sliver of that colour along the whole edge. Thousands of paths, every
    # penalty pinned at its cap. So each cluster keeps only its eroded core,
    # and every remaining pixel goes to the nearest core: boundaries land on
    # the midline between the two real regions and slivers have no core to
    # survive from. A line 3px wide keeps a 1px core, so keylines survive.
    # A core fragment smaller than ~(0.4% of the long edge)^2 - 40px at 1600 -
    # is the same anti-aliasing seen end-on (blue|green mixes to a dark blob
    # the black cluster claims); it is dropped so the propagation fills it
    # from the real regions around it. Real detail keeps cores far larger.
    if core_r is None:
        core_r = max(1, int(round(0.001 * max(h, w))))
    if min_core is None:
        min_core = max(4, int(round((0.004 * max(h, w)) ** 2)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * core_r + 1,) * 2)
    core = np.full((h, w), -1, np.int32)
    for i in alive:
        m = cv2.erode((labels == i).astype(np.uint8), ker)
        n_c, lbl_c, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        small = np.where(stats[:, cv2.CC_STAT_AREA] < min_core)[0]
        if small.size:
            m[np.isin(lbl_c, small[small != 0])] = 0
        core[m > 0] = i
    if (core >= 0).any():
        _, (iy, ix) = distance_transform_edt(core < 0, return_indices=True)
        labels = core[iy, ix]

    layers = []
    for i in alive:
        if i == bg_label:
            continue
        mask = labels == i
        if not mask.any():
            continue
        # Representative colour: the cluster centre, back through OpenCV's
        # 8-bit Lab so the swatch matches what the pixels actually were.
        c8 = np.clip(centers[i], 0, 255).astype(np.uint8).reshape(1, 1, 3)
        rgb_c = cv2.cvtColor(c8, cv2.COLOR_LAB2RGB).reshape(3)
        layers.append((float(centers[i][0]), mask, tuple(int(v) for v in rgb_c)))
    if len(layers) < 2:
        return ([], None, None) if _return_labels else []   # one colour: that is `kmeans`
    layers.sort(key=lambda t: -t[0])               # lightest first, darkest on top
    out = [(m, c) for _, m, c in layers]
    if _return_labels:
        return out, labels, bg_label
    return out


def strat_layerlines(rgb, line_frac=0.00125, fill_L=40.0, protect_white=True,
                     **kw):
    """Black-and-white output built ON the colour layering.

    Tyler, 2026-09-15, on seeing `kmeans-layered` in colour: gaps between
    colours, blobbed, no lines - "we need strong black lines with a white
    background". The colour split is still the right instrument (it is the
    only one that sees structure luminance thresholding fuses); what the
    laser wants is the DRAWING of that split:

      * every boundary between two different colour regions - including a
        colour against the background - becomes a black line `line_frac` of
        the long edge wide (2px at 1600 - Tyler's pick, 2026-09-15, once the
        channel fix below made every width keep its detail), with any white
        channel the stroke would have closed carved back out - see
        `_keep_white_channels`, and the note there for why no width alone
        could be right. This is what the hairline seams
        between separately-traced colour layers turn into: one stroke that
        covers them, instead of a gap;
      * every region darker than `fill_L` (CIELAB L*) is filled black - the
        navy rope, the black ring - so the plate still has its solid darks;
      * light and mid colours (gold, red, white) stay bare, outlined.

    Detail survives because the boundary clean-up is turned down to a 1px
    core with a tiny minimum fragment: on this path a sliver along an edge
    is not a wrong colour, it is part of the line.
    """
    h, w = rgb.shape[:2]
    res = strat_kmeans_layered(rgb, core_r=1, min_core=4, _return_labels=True, **kw)
    layers, labels, bg_label = res
    if not layers:
        return np.zeros((h, w), dtype=bool)        # abstain: one colour

    # Boundaries: a pixel whose right or lower neighbour has a different
    # label. Marked on both sides so the line is centred on the edge.
    edge = np.zeros((h, w), dtype=bool)
    dif = labels[:, 1:] != labels[:, :-1]
    edge[:, 1:] |= dif
    edge[:, :-1] |= dif
    dif = labels[1:, :] != labels[:-1, :]
    edge[1:, :] |= dif
    edge[:-1, :] |= dif

    line_px = max(2, int(round(line_frac * max(h, w))))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (line_px, line_px))
    lines = cv2.dilate(edge.astype(np.uint8), ker) > 0

    fill = np.zeros((h, w), dtype=bool)
    for mask, color in layers:
        L = cv2.cvtColor(np.asarray(color, np.uint8).reshape(1, 1, 3),
                         cv2.COLOR_RGB2LAB)[0, 0, 0] * 100.0 / 255.0
        if L < fill_L:
            fill |= mask

    if protect_white:
        lines = _keep_white_channels(lines, fill, line_px)
    return lines | fill


def strat_fieldflip(rgb, alpha=None, lab_tol=14.0, **kw):
    """The artwork-aware inverse of :func:`strat_layerlines`.

    ``--invert`` is deliberately not used here: it inverts the whole canvas,
    which turns a white source field into a giant black rectangle.  A Field
    Flip inverts only the detected artwork envelope, leaving the frame white.
    This is a second interpretation of the same colour structure, useful when
    the operator knows the field/foreground polarity is opposite to the
    darker-is-ink reading used by ``layerlines``.

    The envelope comes from alpha when the source has meaningful transparency,
    otherwise from Lab distance to the border background.  The current
    layerlines result supplies the structural mask; keeping that mechanism
    unchanged makes this candidate a reversible polarity experiment rather
    than another luminance threshold.  A one-colour image has no meaningful
    opposite interpretation and therefore abstains.

    This function returns a single boolean mask like every non-layered
    strategy.  The driver clips the cleaned result to the same envelope too,
    because morphological cleanup is allowed to grow a mask by a pixel or two.
    """
    envelope = _silhouette_of(rgb, alpha, lab_tol)
    if int(envelope.sum()) < 16:
        return np.zeros(envelope.shape, dtype=bool)

    # When both candidates are requested in one run, reuse the expensive
    # colour segmentation that layerlines already computed.
    base = _memo('layerlines', strat_layerlines, rgb, **kw)
    if not base.any():
        return np.zeros(envelope.shape, dtype=bool)

    # Preserve a narrow inner rim around the artwork. Without it, flipping a
    # dark outline makes the outside edge disappear entirely; the operator
    # would get a floating field instead of a recognisable mark. Keep only the
    # inside half of the ring so the frame remains white by construction.
    ring_px = max(1, int(round(0.00125 * max(rgb.shape[:2]))))
    ring_ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (2 * ring_px + 1,) * 2)
    ring = (cv2.morphologyEx(envelope.astype(np.uint8), cv2.MORPH_GRADIENT,
                             ring_ker) > 0) & envelope

    # True means black/ink. Flip only inside artwork; the outside is always
    # false (white) even when the source field occupies most of the canvas.
    opposite = envelope & ~base
    if int(opposite.sum()) < 16:
        return np.zeros(envelope.shape, dtype=bool)
    return opposite | ring


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
    sil = _silhouette_of(rgb, alpha, lab_tol)

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


def strat_keyline(rgb, alpha=None, l_black=20.0, lab_tol=14.0, ring=True):
    """Recover the black line drawing the colour was poured into.

    THE PREMISE, and it is a claim about how this artwork is MADE rather than
    about pixels. Vector-derived insignia are not colour fields that happen to
    have edges -- they are a line drawing that was then colour-filled. The
    artist outlined every element in near-black and poured colour inside. If
    that is true then the drawing is still in the file, and the ink-vs-field
    question never has to be answered: every colour is fill, and fill is bare.

    WHY THIS IS NOT A THIRD THRESHOLD VARIANT. `inotsu` and `triotsu` both
    failed by MOVING the split -- a different cut through the same population,
    which the 2026-09-06 finding proved cannot work, because the same
    mid-luminance gold is INK on one logo and a FIELD on another and no cut
    over any population separates them. This does not move a split. It asks an
    absolute perceptual question with no reference to the image's own
    histogram: "is this paint actually black?" A data-derived cut must land
    somewhere between the modes. An absolute anchor need not land anywhere at
    all, and on art with no black in it, it correctly returns nothing.

    Measured on the two 2026-09-07 order failures before any of this was built:
        pixels below L*20      17.5% (vector)   14.1% (embroidery)
        pixels `otsu` burns      ~40%             ~44%
    `otsu` was burning the fills AND the keylines together.

    THREE PARTS, and each one was earned by a failure in the probe:

    1. FLATTEN FIRST (bilateral, twice). The remaining defects after the bare
       threshold were both shading, not classification: the airbrushed gradient
       at the bottom of each USN letter dips below the anchor and burns as a
       black wedge INSIDE the letter, and the embroidered patch's thread
       texture breaks its navy star outline into dashes. Both are one piece of
       paint whose lightness varies, and flattening is the operation that says
       so -- `nested`'s "is this the same paint" question asked BEFORE the
       threshold instead of after it. Measured: path count fell 2-7x
       (embroidery 1027 -> 142, vector 413 -> 198).

    2. SILHOUETTE RING, scaled to the image. A shape whose outer edge carries
       no keyline still needs one -- the embroidered patch's red star sits
       directly on white and came out as open Vs. This is the same
       MORPH_GRADIENT ring `strat_silhouette` uses, with two corrections the
       probe forced. It is computed on the FLATTENED image, because on raw
       thread texture the silhouette fragments and the ring came back as
       stipple over the whole design. And its radius scales with the image
       instead of being a fixed 5x5: on the 199px patch a fixed 5x5
       is proportionally enormous, `clean_mask`'s protected close then filled
       every white channel it created, ink went 51.5% -> 68.1% and the bird
       disappeared. That is the FIFTH instance of this project's oldest
       pattern -- a cleanup step destroying the fine detail that matters.

    3. TWO ABSTAINS, returning an empty mask so the pipeline's own degenerate
       guard drops the candidate rather than putting junk on the sheet.
       Keyline art is a real category and this is honest about not being in it.
       Verified on the corpus: `NSWU-2` (black field by design) abstains on the
       first, `anchors` (three gold anchors, no black anywhere -- the image
       Tyler rejected with "detail lost on all options") abstains on the
       second.

    `l_black` is a perceptual constant, not a fitted knob: sRGB ~48/255, the
    boundary where paint stops reading as "a dark colour" and starts reading as
    "black". A sweep of 15/20/25/30/35/40 over the three failures showed 20 is
    not sitting on a cliff.
    """
    empty = np.zeros(rgb.shape[:2], dtype=bool)

    flat = cv2.bilateralFilter(cv2.bilateralFilter(rgb, 9, 75, 75), 9, 75, 75)
    dark = _lightness(flat) < l_black
    sil = _silhouette_of(flat, alpha, lab_tol)

    # Abstain 1: the stock itself is dark. "Is this paint black" is not a
    # question about ink when the background is black -- every dark-field brand
    # asset (NSWU-2, VX-30, the SWAT patch, the Saints fleur-de-lis) would come
    # back as a solid burn. `otsu` is right on those and this must not compete.
    if (~sil).any() and float(np.median(_lightness(flat)[~sil])) < 40.0:
        return empty

    # Abstain 2: there is no black drawing in here. Flat colour art with no
    # outlines yields a few percent of speckle, which traces into junk and
    # still clears the pipeline's 0.1% degenerate floor.
    art = int(sil.sum())
    if not art or (dark & sil).sum() / art < 0.02:
        return empty

    if not ring:
        return dark

    r = max(1, round(0.0015 * max(rgb.shape[:2])))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return dark | (cv2.morphologyEx(sil.astype(np.uint8), cv2.MORPH_GRADIENT, k) > 0)


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
    base = _memo('otsu', strat_otsu, rgb)
    field = _memo('nested', strat_nested, rgb, alpha, **kw)

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
    'composite':  strat_composite,
    'layerlines': strat_layerlines,
    # ponytail: TRIAL 2026-09-22, Tyler is judging fieldflip on the sheet.
    # Revert with `git checkout -- _engine/candidates.py` when the trial ends.
    'fieldflip':  strat_fieldflip,
}

# THE SHEET WAS TRIMMED FROM 10 TO 5 ON 2026-09-15, on evidence, at Tyler's
# call ("3-5 total strats is the goal"). The evidence, from 95 judged images
# (79 with a full shippable set; the clean all-ten pool is n=41, 2026-09-08):
#
#   * coverage saturates at TWO: otsu + bgdist alone give at least one
#     shippable candidate on 40/41 images, and the 41st had nothing shippable
#     from anyone. Every other strategy has marginal coverage value 0 - no
#     image loses its only shippable candidate when it is removed.
#   * favourite-pick retention (which tile Tyler actually starred, n=40):
#     {otsu, bgdist, kmeans, composite} keeps 34/40 = 85%, the best four.
#     composite is his favourite on 9/41 despite a median overall of 0.0 -
#     the scorer disagrees with him, and the scorer is the suspect.
#   * near-duplicates by mask IoU (n=58): otsu/bgdist median 0.947 and
#     identical (>0.95) on half of all images; bgdist/kmeans 0.831. They are
#     kept as a pair because they are also the ONLY two with coverage value.
#   * cut from the sheet: linework (0 favourites, 75% of runs under 20/100),
#     nested (0 favourites; still runs INSIDE composite), keyflip (1
#     favourite, 2 of 3 runs unscorable), keyfill (strict subset of keyline's
#     shippable images in both pools), keyline and silhouette (2 favourites
#     each in the all-ten pool; silhouette was sole-cover on 5 of 16 OLD
#     single-pick rows judged against a 7-strategy roster, so it is the one
#     cut to revisit if a job ever has nothing shippable).
#   * layerlines is new (2026-09-15): black outlines of every colour boundary
#     plus dark fills, from the kmeans colour split. Abstains on one-colour
#     art, where the other four already do the job.
#
# NINE STRATEGIES WERE DELETED ON 2026-09-22, at Tyler's call, after a
# whole-repo over-engineering audit: sauvola, inotsu, triotsu, edges, keyfill,
# keyflip, linework, plate and neural, plus the three helpers only keyfill and
# keyflip called (_thicken_thin, _flip_fields, _floor_px).
#
# They are NOT archived in-tree. Their method-index.md entries survive with the
# steps and every parameter, which is the part a new strategy is designed
# against; the code itself is one command away:
#
#     git show 6f1e1d4:_engine/candidates.py
#
# silhouette and keyline were deliberately NOT cut. The pick record puts
# silhouette at 13 -- second most-picked in the project -- and keyline at 4
# with a shipped emblem; the n=40 pool that justified the 2026-09-15 trim
# undercounts both because they were judged against the old 7-strategy roster.
#
# Everything below is still reachable with --strategies for a calibration run.
OPTIONAL = {'silhouette': strat_silhouette, 'keyline': strat_keyline,
            'nested': strat_nested,
            'kmeans-layered': strat_kmeans_layered}

# Strategies that return LAYERS - a list of (bool mask, (r, g, b)) - instead of
# one boolean mask. They go through `_emit_layered`, not the single-mask path.
LAYERED = {'kmeans-layered'}
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
ALPHA_AWARE = {'silhouette', 'keyline', 'fieldflip'}
