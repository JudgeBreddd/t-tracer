#!/usr/bin/env python3
"""
hygiene.py - T-Tracer

Turns the reviewer's stated acceptance test into numbers:

    "if the lines are all touching and the svg is clean, no jagged edges,
     no gaps, no random black lines, it will burn good."

Five clauses, five measurements:

    lines all touching  ->  fragmentation   (how many separate ink islands)
    no gaps             ->  gaps            (narrow white channels between ink)
    no random black lines -> strays         (tiny islands far below the main mass)
    no jagged edges     ->  jaggedness      (staircase residual on the emitted curve)
    svg is clean        ->  nodes/mm + self-intersections + degenerate subpaths

Deliberately NOT measured: minimum feature size against beam kerf. The job this
serves runs on an F2 Ultra UV - cold process, tiny spot, no thermal bloom - so
sub-kerf feature loss is not the binding constraint. If this ever moves to the
MOPA fiber for a heat-marked job, that gate has to come back.

Two separate axes, on purpose:

  hygiene   0-100  is the geometry clean?      (does it burn well)
  fidelity  0-1    is it still the right art?  (does it look like the source)

Kept apart because a solid black rectangle scores a perfect hygiene and is
worthless. `overall` combines them, and is what ranks the contact sheet.
"""

import numpy as np
import cv2
from skimage.measure import label


# ---------------------------------------------------------------------------
# Individual measurements
# ---------------------------------------------------------------------------

def measure_components(ink, stray_frac=0.002):
    """Ink islands, and how many of them are debris.

    A 'stray' is an island below `stray_frac` of total ink area. Using a
    fraction rather than an absolute pixel count means the same rule works on a
    400px source and a 4000px one.
    """
    total = int(ink.sum())
    if total == 0:
        return {'components': 0, 'strays': 0, 'stray_area_frac': 0.0}

    lab = label(ink, connectivity=2)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    sizes = sizes[sizes > 0]

    cutoff = max(1.0, total * stray_frac)
    strays = sizes[sizes < cutoff]

    return {
        'components': int(len(sizes)),
        'strays': int(len(strays)),
        'stray_area_frac': float(strays.sum() / total),
    }


def measure_gaps(ink, src_gray, gap_px, min_gap_area=6):
    """Narrow white channels between ink regions - the 'no gaps' clause.

    Closing the ink by `gap_px` and subtracting it leaves the white slivers
    narrower than that radius. But a sliver is NOT automatically a defect: in
    this artwork the thin white channel around a panther, or between a star and
    its field, is the design. Counting those was the scorer's first real bug -
    it reported 43 "gaps" on a candidate whose gaps were all intentional.

    So each sliver is put to a question the trace itself cannot bias: is the
    SOURCE dark or light where this sliver sits? Compare the source luminance
    under the sliver against the mean luminance under the rendered ink and
    under the rendered background. A sliver sitting over source pixels that
    look like ink is a seam the trace opened by mistake. A sliver over source
    pixels that look like background is the artwork, and is left alone.
    """
    if gap_px < 1 or ink.sum() == 0:
        return {'gaps': 0, 'gap_area_px': 0, 'design_gaps': 0}

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap_px + 1,) * 2)
    closed = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
    sliver = closed & ~ink
    if not sliver.any():
        return {'gaps': 0, 'gap_area_px': 0, 'design_gaps': 0}

    g = _match_shape(src_gray, ink)
    ink_mean = float(g[ink].mean())
    bg_mean = float(g[~ink].mean()) if (~ink).any() else 255.0
    if abs(ink_mean - bg_mean) < 1e-6:
        return {'gaps': 0, 'gap_area_px': 0, 'design_gaps': 0}

    lab = label(sliver, connectivity=2)
    n = lab.max()
    if n == 0:
        return {'gaps': 0, 'gap_area_px': 0, 'design_gaps': 0}

    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    sums = np.bincount(lab.ravel(), weights=g.ravel(), minlength=n + 1)

    # A sliver lies along a region boundary, so its pixels are often the
    # source's anti-aliased blend - luminance lands mid-scale and a nearest-mean
    # test becomes a coin flip. That produced 24-28 "gaps" on candidates that
    # were visually clean, which is the exact false-alarm rate that teaches an
    # operator to ignore the gate. So the sliver has to sit clearly on the ink
    # side of the scale, and anything ambiguous is called design.
    mid = (ink_mean + bg_mean) / 2.0
    margin = abs(bg_mean - ink_mean) * 0.25
    ink_is_dark = ink_mean < bg_mean
    threshold = (mid - margin) if ink_is_dark else (mid + margin)

    defects, defect_area, design = 0, 0, 0
    for i in range(1, n + 1):
        if sizes[i] < min_gap_area:
            continue
        mean_lum = sums[i] / sizes[i]
        is_defect = (mean_lum < threshold) if ink_is_dark else (mean_lum > threshold)
        if is_defect:
            defects += 1
            defect_area += int(sizes[i])
        else:
            design += 1

    return {'gaps': defects, 'gap_area_px': defect_area, 'design_gaps': design}


def measure_fusions(ink, src_rgb, min_frac=0.12, delta_e=22.0, min_comp_px=400):
    """Distinct elements that binarized into one black mass.

    Named directly by the reviewer, on a unit insignia: "the
    sword hilt is merged with the black background." The sword's brown grip and
    the shield's dark blue quadrant are different objects that happen to share a
    lightness, so every luminance split welds them together. The result still
    has clean curves, no strays and no gaps - it scores well and it is wrong.

    Gaps and fusions are opposite failures and need opposite tests. A gap is
    ink that should be joined; a fusion is ink that should be split. Nothing
    measured before this looked for the second one.

    Detection uses the colour the binarization threw away. Inside each ink
    component, split the SOURCE pixels into two clusters in Lab. If the
    centroids are far apart (> `delta_e`) and the minority cluster is a real
    share of the component (> `min_frac`), then that one black shape is
    covering two visually distinct things, and they fused.
    """
    total = int(ink.sum())
    if total == 0 or src_rgb is None:
        return {'fusions': 0, 'fused_area_frac': 0.0}

    rgb = src_rgb
    if rgb.shape[:2] != ink.shape[:2]:
        rgb = cv2.resize(rgb, (ink.shape[1], ink.shape[0]),
                         interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    labels = label(ink, connectivity=2)
    n = labels.max()
    fused, fused_area = 0, 0
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    for i in range(1, n + 1):
        sel = labels == i
        area = int(sel.sum())
        if area < min_comp_px:
            continue
        pts = lab[sel]
        if len(pts) > 20000:                       # subsample large components
            pts = pts[np.random.default_rng(0).choice(len(pts), 20000, replace=False)]
        try:
            _, lbl, centers = cv2.kmeans(pts, 2, None, crit, 3,
                                         cv2.KMEANS_PP_CENTERS)
        except Exception:
            continue
        counts = np.bincount(lbl.ravel(), minlength=2)
        minority = counts.min() / counts.sum()
        sep = float(np.linalg.norm(centers[0] - centers[1]))
        if sep > delta_e and minority > min_frac:
            fused += 1
            fused_area += area

    return {'fusions': int(fused), 'fused_area_frac': round(fused_area / total, 3)}


def measure_jaggedness(ink, window=9):
    """Staircase residual: how far the rendered outline sits from its own
    locally-smoothed self.

    A pixel staircase oscillates around the true edge by roughly half a pixel,
    so it leaves a large residual at small scales. A properly fitted curve
    tracks its own smoothing almost exactly. Reported in source pixels, so it
    is comparable across strategies on the same image.
    """
    contours, _ = cv2.findContours(ink.astype(np.uint8), cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_NONE)
    resid, weight = [], []
    for c in contours:
        p = c.reshape(-1, 2).astype(float)
        if len(p) < window * 3:
            continue
        # Circular moving average - contours are closed.
        pad = np.vstack([p[-window:], p, p[:window]])
        kern = np.ones(window) / window
        sx = np.convolve(pad[:, 0], kern, mode='same')[window:-window]
        sy = np.convolve(pad[:, 1], kern, mode='same')[window:-window]
        d = np.hypot(p[:, 0] - sx, p[:, 1] - sy)
        resid.append(d.mean())
        weight.append(len(p))

    if not resid:
        return {'jaggedness_px': 0.0}
    return {'jaggedness_px': float(np.average(resid, weights=weight))}


def measure_self_intersections(paths, sample=160):
    """Subpaths that cross themselves. These are what produce overlapping
    geometry, which is the single biggest cause of laser stutter and
    double-passes (see _shared/laser-output-rules.md).

    Checked on a decimated polyline - a fitted curve that crosses itself does
    so grossly, not by a sub-pixel sliver, so coarse sampling finds it.
    """
    from candidates import _sample_beziers

    bad = 0
    for segs in paths:
        pts = _sample_beziers(segs, per_seg=8)
        if len(pts) < 8:
            continue
        step = max(1, len(pts) // sample)
        p = pts[::step]
        n = len(p)
        if n < 8:
            continue

        a, b = p, np.roll(p, -1, axis=0)
        r = b - a
        # Pairwise orientation tests, vectorized over all segment pairs.
        d = r[:, None, :]
        e = r[None, :, :]
        diff = a[None, :, :] - a[:, None, :]
        denom = d[..., 0] * e[..., 1] - d[..., 1] * e[..., 0]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (diff[..., 0] * e[..., 1] - diff[..., 1] * e[..., 0]) / denom
            u = (diff[..., 0] * d[..., 1] - diff[..., 1] * d[..., 0]) / denom
        hit = (denom != 0) & (t > 1e-9) & (t < 1 - 1e-9) & (u > 1e-9) & (u < 1 - 1e-9)
        # Ignore neighbours, which share an endpoint by construction.
        idx = np.arange(n)
        adjacent = (np.abs(idx[:, None] - idx[None, :]) <= 1) | \
                   (np.abs(idx[:, None] - idx[None, :]) >= n - 1)
        if (hit & ~adjacent).any():
            bad += 1
    return {'self_intersecting_paths': int(bad)}


def _match_shape(arr, like):
    """Resize `arr` to `like`'s shape. Nearest for masks, area for grayscale."""
    if arr.shape[:2] == like.shape[:2]:
        return arr
    interp = cv2.INTER_NEAREST if arr.dtype == bool else cv2.INTER_AREA
    out = cv2.resize(arr.astype(np.uint8), (like.shape[1], like.shape[0]),
                     interpolation=interp)
    return out > 0 if arr.dtype == bool else out


def measure_fidelity(rendered, src_gray, tol_frac=0.004):
    """Does the delivered geometry still describe the SOURCE artwork?

    The first version of this compared the render against the mask it was
    traced from - which measured nothing, because a strategy that produced a
    solid black square matched its own solid black square perfectly and scored
    0.999. It ranked first on the very first real image. This is the same trap
    the design session flagged: the dangerous failure is not a bad file, it is
    a confident metric blessing one.

    The fix is a reference the strategy cannot influence: the edges of the
    source image itself. Symmetric edge agreement (F1) between the source's
    Canny edges and the rendered outline, with a distance tolerance so a
    slightly-moved edge still counts.

      recall    - how much of the source's structure survived
      precision - how much of the output's structure is actually in the source

    A solid blob has one outline and no interior structure, so recall collapses
    and it can no longer win. Speckle inflates output edges the source does not
    have, so precision collapses. Both failure modes are now visible.
    """
    g = _match_shape(src_gray, rendered)
    med = float(np.median(g))
    src_edges = cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0),
                          int(max(0, 0.66 * med)), int(min(255, 1.33 * med))) > 0

    ink = rendered.astype(np.uint8)
    out_edges = cv2.morphologyEx(ink, cv2.MORPH_GRADIENT,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))) > 0

    if not src_edges.any() or not out_edges.any():
        return 0.0

    tol = max(2.0, tol_frac * float(np.hypot(*rendered.shape)))
    d_to_src = cv2.distanceTransform((~src_edges).astype(np.uint8), cv2.DIST_L2, 3)
    d_to_out = cv2.distanceTransform((~out_edges).astype(np.uint8), cv2.DIST_L2, 3)

    precision = float((d_to_src[out_edges] <= tol).mean())
    recall = float((d_to_out[src_edges] <= tol).mean())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

def score(rendered, src_gray, paths, nodes, height_mm=None, ss=2, src_h=None,
          src_rgb=None):
    """Full hygiene report for one candidate.

    `rendered` is the rasterized emitted geometry (supersampled by `ss`).
    `src_gray` is the SOURCE image in grayscale - deliberately not the mask the
    candidate was traced from, so no strategy can grade its own homework.
    """
    src_h = src_h or src_gray.shape[0]
    mm_per_px = (height_mm / src_h) if height_mm else None

    # A gap narrower than ~0.2mm reads as "these should be touching".
    gap_px = max(1, int(round(0.12 / mm_per_px * ss))) if mm_per_px else 2 * ss

    comps = measure_components(rendered)
    gaps = measure_gaps(rendered, src_gray, gap_px)
    fuse = measure_fusions(rendered, src_rgb)
    jag = measure_jaggedness(rendered)
    xing = measure_self_intersections(paths)
    fidelity = measure_fidelity(rendered, src_gray)

    # Jaggedness is measured on the supersampled raster; report in source px.
    jag_px = jag['jaggedness_px'] / ss

    path_len_mm = None
    nodes_per_mm = None
    if mm_per_px:
        per = sum(cv2.arcLength(c, True) for c, in
                  [(c,) for c in cv2.findContours(rendered.astype(np.uint8),
                                                  cv2.RETR_LIST,
                                                  cv2.CHAIN_APPROX_NONE)[0]])
        path_len_mm = float(per / ss * mm_per_px)
        nodes_per_mm = float(nodes / path_len_mm) if path_len_mm > 0 else None

    # ---- penalties, each capped so one bad axis cannot drive the score
    # negative and hide the others -------------------------------------------
    p_strays = min(25.0, comps['strays'] * 3.0 + comps['stray_area_frac'] * 300)
    p_gaps = min(25.0, gaps['gaps'] * 4.0)
    p_jag = min(25.0, max(0.0, (jag_px - 0.08)) * 90)
    p_xing = min(15.0, xing['self_intersecting_paths'] * 7.5)
    # Fusion is weighted heavily: it is the one defect that looks clean.
    #
    # The per-fusion weight is deliberately small against a high cap. The first
    # version used 10.0 against a cap of 30, which saturated at three fusions -
    # so a candidate with 4 and one with 17 scored identically, and the count
    # carried no information exactly where it mattered. Real candidates land
    # between about 2 and 35, so the scale has to stay linear across that range.
    p_fuse = min(40.0, fuse['fusions'] * 2.5 + fuse['fused_area_frac'] * 10)
    p_nodes = 0.0
    if nodes_per_mm is not None and nodes_per_mm > 6:
        p_nodes = min(10.0, (nodes_per_mm - 6) * 1.5)

    hygiene = max(0.0, 100.0 - (p_strays + p_gaps + p_jag + p_xing + p_nodes + p_fuse))

    penalties = {
        'strays': p_strays, 'gaps': p_gaps, 'jagged': p_jag,
        'self_intersect': p_xing, 'node_bloat': p_nodes, 'fused': p_fuse,
    }
    worst = max(penalties, key=penalties.get)

    # Absolute edge-F1 depends on how busy the source art is, so a fixed
    # "too low" threshold would be meaningless. `overall` and the artwork-loss
    # wording are finalized by the caller, which can compare candidates that
    # all came from the same image. What is decided here is only the geometry.
    if penalties[worst] < 3:
        headline = 'clean'
    else:
        headline = {
            'strays': f'{comps["strays"]} stray marks',
            'gaps': f'{gaps["gaps"]} gaps between regions',
            'jagged': f'jagged edges ({jag_px:.2f}px staircase)',
            'self_intersect': f'{xing["self_intersecting_paths"]} self-crossing paths',
            'node_bloat': f'node bloat ({nodes_per_mm:.1f}/mm)',
            'fused': f'{fuse["fusions"]} fused shapes (distinct elements merged)',
        }[worst]

    return {
        'hygiene': round(hygiene, 1),
        'fidelity': round(fidelity, 3),
        'headline': headline,
        'nodes': int(nodes),
        'paths': len(paths),
        'nodes_per_mm': round(nodes_per_mm, 2) if nodes_per_mm else None,
        'path_length_mm': round(path_len_mm, 1) if path_len_mm else None,
        **comps, **gaps, **jag, **xing, **fuse,
        'jaggedness_px': round(jag_px, 3),
        'penalties': {k: round(v, 1) for k, v in penalties.items()},
    }


def finalize(results, loss_ratio=0.75):
    """Second pass over every candidate for ONE image.

    Edge-F1 has no meaningful absolute scale - a busy insignia and a simple
    roundel produce different numbers for equally good traces. What IS
    meaningful is the spread within one image: if the best candidate recovered
    0.62 of the source's structure and another managed 0.21, the second one
    threw the artwork away, and no amount of clean geometry redeems it.

    So `overall` is hygiene scaled by fidelity relative to the best candidate
    on the same image. Anything under `loss_ratio` of the leader is called out
    as having lost the artwork, which is what a solid blob now earns.
    """
    scored = {k: v for k, v in results.items() if 'hygiene' in v}
    if not scored:
        return results

    best_fid = max(v['fidelity'] for v in scored.values()) or 1.0

    for v in scored.values():
        rel = v['fidelity'] / best_fid if best_fid else 0.0
        v['fidelity_rel'] = round(rel, 3)
        v['overall'] = round(v['hygiene'] * min(1.0, rel / loss_ratio), 1)
        if rel < loss_ratio:
            v['headline'] = f'lost the artwork ({int(rel * 100)}% of best detail)'
    return results
