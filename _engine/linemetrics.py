#!/usr/bin/env python3
"""linemetrics.py - T-Tracer

The measurements Tyler actually looks at when he picks, which hygiene.py does
not measure. Built BESIDE hygiene rather than inside it, so the existing score
keeps working unchanged while these are calibrated.

ONE sampling pass, FOUR independent outputs, per his own constraint: "i dont
want to hear in 3 hours that #3 tool silently overrode the output from tool
#4." Each metric returns its own named value; none writes another's.

  waviness    - the CENTRELINE wanders        (a snake has constant width)
  thickness   - the width values themselves
  consistency - how much the width varies
  blotch      - local width SPIKES

Width variation and waviness are different failures. A perfectly even-width
line can be wavy; a dead-straight line can have terrible width variation.

--------------------------------------------------------------------------
WAVINESS: measured on the runs that are SUPPOSED to be straight
--------------------------------------------------------------------------
hygiene.measure_jaggedness compares a contour to itself smoothed over a 9px
window. That is the PIXEL STAIRCASE - half-pixel oscillation, wavelength 1-3px.
Tyler's complaint is a different animal: a star arm whose top edge should be a
ruler line instead bows over its whole 220px length. One number for both
dilutes both.

TWO ATTEMPTS WERE THROWN AWAY. Both are recorded because both look obviously
right until they are measured, and neither is.

ATTEMPT 1 - contour vs a heavily smoothed copy of itself, measure the gap.
  * a clean rectangle scored 0.95px: heavy smoothing ROUNDS ITS FOUR CORNERS
    and the rounding read as wander. It punished the corners Tyler had just
    said were fine.
  * response collapsed past ~100px wavelength (4px of wobble read 2.75px at
    wl=40, 0.95px at wl=260), so the long slow bow that IS the complaint was
    the one thing it could not see.

ATTEMPT 2 - use curvature to find the runs that are "supposed to be straight",
then measure those against a fitted ruler. It read 0.00 on EVERY wavy input.
The reason is not a bad threshold, it is the idea: WAVINESS IS CURVATURE. A
4px/60px-wavelength wobble carries curvature 0.044, an order of magnitude above
any "is this straight" cutoff, so the gate discarded exactly the signal it was
meant to select. A test that removes what it is looking for cannot be tuned
into working.

WHAT ACTUALLY SEPARATES THEM IS NOT HOW MUCH A CURVE BENDS BUT WHETHER IT KEEPS
CHANGING ITS MIND. A drawn arc curves consistently; a traced wobble alternates.
So fit a QUADRATIC in the local tangent frame over a sliding window:

  straight run  -> quadratic fits exactly            -> residual 0
  true arc      -> quadratic IS an arc, to 2nd order -> residual 0
  wavy run      -> no quadratic follows an oscillation -> residual = the wobble
  corner        -> excluded outright, see below

Corners are detected as a turning-angle spike and any window containing one is
dropped rather than scored. A corner is a feature, not a defect, and Tyler has
explicitly said the corners are fine.

The window length sets what counts as wander rather than shape: anything
oscillating faster than the window is caught, anything slower is treated as the
intended form. That is a real limit and it is stated rather than hidden.
"""
import numpy as np
import cv2
from skimage.measure import label


def _resample_closed(pts, step=1.0):
    """Uniform arc-length resampling of a closed polyline."""
    p = np.asarray(pts, dtype=float)
    ring = np.vstack([p, p[:1]])
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(ring, axis=0).T))]
    total = d[-1]
    if total < step * 8:
        return None
    n = max(8, int(round(total / step)))
    u = np.linspace(0, total, n, endpoint=False)
    x = np.interp(u, d, ring[:, 0])
    y = np.interp(u, d, ring[:, 1])
    return np.column_stack([x, y])


def _smooth_closed(p, sigma):
    """Circular Gaussian smoothing of a closed curve, sigma in samples."""
    if sigma <= 0:
        return p.copy()
    rad = max(1, int(round(sigma * 3)))
    k = np.exp(-0.5 * (np.arange(-rad, rad + 1) / sigma) ** 2)
    k /= k.sum()
    n = len(p)
    if n <= 2 * rad:
        return p.copy()
    out = np.empty_like(p)
    for j in (0, 1):
        ext = np.r_[p[-rad:, j], p[:, j], p[:rad, j]]
        out[:, j] = np.convolve(ext, k, mode='same')[rad:rad + n]
    return out


def _curvature(p, sigma):
    """Signed curvature of a closed curve, smoothed at `sigma` samples."""
    q = _smooth_closed(p, sigma)
    d1 = np.gradient(q, axis=0)
    d2 = np.gradient(d1, axis=0)
    num = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    den = np.power(d1[:, 0] ** 2 + d1[:, 1] ** 2, 1.5) + 1e-12
    return num / den



def _corner_flags(p, half=4, deg=35.0):
    """True where the tangent turns sharply - a corner, not a defect."""
    n = len(p)
    a = p[np.arange(n) - half]
    b = p[(np.arange(n) + half) % n]
    v1, v2 = p - a, b - p
    d1 = np.hypot(*v1.T) + 1e-9
    d2 = np.hypot(*v2.T) + 1e-9
    cos = np.clip((v1 * v2).sum(1) / (d1 * d2), -1, 1)
    return np.degrees(np.arccos(cos)) > deg


def _quad_residual(seg):
    """RMS deviation of a run from the best quadratic through it.

    Fitted in the run's own principal frame, so it is orientation-free. A line
    and a gentle arc are both exactly representable; an oscillation is not.
    """
    c = seg - seg.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    x = c @ vt[0]
    y = c @ vt[1]
    A = np.column_stack([np.ones_like(x), x, x * x])
    try:
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    r = y - A @ coef
    # Sagitta of the FITTED curve across this window: how far the intended
    # form departs from a straight line. Computed from the fit, never from the
    # raw points, so wobble lands in the residual and not in here. This is what
    # lets a straight run and an arc be reported apart WITHOUT gating away the
    # signal -- the mistake that made attempt 2 read 0.00 on everything.
    half = (x.max() - x.min()) / 2.0
    sag = abs(float(coef[2])) * half * half
    return (float(np.sqrt((r ** 2).mean())), float(np.abs(r).max()), sag)


def contour_waviness(pts, sigma_small=2.0, window=120, step=None,
                     corner_deg=35.0, max_turn_deg=100.0):
    """Sliding-window waviness for one contour.

    `max_turn_deg` is what keeps the metric honest on DETAIL. A 120px window
    laid over a rope bead or a hatching tick wraps most of the way around a
    small closed blob, and no quadratic can follow a loop -- so the window
    scored high for being a SHAPE rather than for being wavy. Measured on the
    the star-and-anchor emblem: the beaded rope and the snake scales lit up as hard as the a line-art emblem
    letterforms, which is a false positive on intended detail.

    A run that is meant to read as a line barely turns. A bead turns through
    most of a circle. Capping total turning keeps windows that are runs and
    discards windows that are shapes, without needing to know which is which
    in advance.

    Returns [(rms, window_len, midpoint_xy, max_dev), ...].
    """
    p = _resample_closed(pts, step=1.0)
    if p is None or len(p) < window:
        return []
    sm = _smooth_closed(p, sigma_small)
    corner = _corner_flags(sm, half=max(2, int(sigma_small * 2)), deg=corner_deg)
    n = len(sm)
    step = step or max(8, window // 4)
    out = []
    for a in range(0, n, step):
        idx = (np.arange(a, a + window) % n)
        if corner[idx].any():
            continue                       # a corner is a feature, not wander
        seg = sm[idx]
        d = np.diff(seg, axis=0)
        ang = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        if np.degrees(abs(ang[-1] - ang[0])) > max_turn_deg:
            continue                       # this is a shape, not a run
        got = _quad_residual(seg)
        if got is None:
            continue
        rms, mx, sag = got
        mid = seg[window // 2]
        out.append((rms, window, (float(mid[0]), float(mid[1])), mx, sag))
    return out


def measure_waviness(ink, sigma_small=2.0, window=120, corner_deg=35.0,
                     max_turn_deg=100.0, straight_sag=3.0, min_perimeter=None):
    """Waviness in source pixels: how far a run departs from any smooth curve.

    Separate from hygiene.measure_jaggedness by construction - that measures the
    1-3px pixel staircase, this measures wander above it. Neither dilutes the
    other.
    """
    ink = np.asarray(ink)
    if ink.dtype != np.uint8:
        ink = ink.astype(np.uint8)
    if min_perimeter is None:
        min_perimeter = window
    cs, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    vals, straight, worst, worst_at = [], [], 0.0, None
    for c in cs:
        q = c.reshape(-1, 2).astype(float)
        if len(q) < min_perimeter:
            continue
        for rms, L, mid, mx, sag in contour_waviness(
                q, sigma_small, window, corner_deg=corner_deg,
                max_turn_deg=max_turn_deg):
            vals.append(rms)
            if sag < straight_sag:
                straight.append(rms)
                # The headline number is the STRAIGHT-run one, so the worst
                # offender reported is a ruler line that wanders -- not an
                # organic shape doing what it was drawn to do.
                if rms > worst:
                    worst, worst_at = rms, mid
    if not vals:
        return {'waviness_px': 0.0, 'waviness_windows': 0,
                'waviness_worst_px': 0.0, 'waviness_worst_at': None,
                'waviness_p90_px': 0.0, 'waviness_curved_px': 0.0,
                'waviness_straight_windows': 0}
    v = np.asarray(vals)
    sv = np.asarray(straight) if straight else np.zeros(0)
    curved = np.asarray([x for x in vals if x not in straight]) if straight else v
    # THE HEADLINE IS p90, NOT THE MEAN. Tyler rejects on the worst visible
    # defect, never on an average: "if the strait line on the letter A is wavy
    # its not shipping." One bad run kills the part, and a mean over a busy
    # emblem buries it -- on the the star-and-anchor emblem the worst straight run is 5.3px
    # while the mean is 0.26px. p90 rather than max so a single stray window
    # cannot decide a part on its own.
    return {
        'waviness_px': round(float(np.percentile(sv, 90)), 3) if len(sv) else 0.0,
        'waviness_mean_px': round(float(sv.mean()), 3) if len(sv) else 0.0,
        'waviness_worst_px': round(float(worst), 3),
        'waviness_worst_at': worst_at,
        'waviness_straight_windows': int(len(sv)),
        # reported, never mixed in: organic/curved runs are a different question
        'waviness_curved_px': round(float(curved.mean()), 3) if len(curved) else 0.0,
        'waviness_windows': int(len(v)),
    }


# ---------------------------------------------------------------------------
# CORNER-TO-CORNER RUNS - Tyler's design, 2026-09-08
#
# The sliding window above can only see runs LONGER than the window, so short
# strokes were skipped entirely rather than scored. That is not a tuning miss,
# it is a blind spot, and he found it from the output alone: the star tip read
# 0.00 waviness at every smoothing level while he could see it change between
# 2.5, 3 and 3.5. It was never measured.
#
# Everything he circled was a SHORT angled stroke - the S's lower-left and
# middle diagonals, the left edge of the star spike, one bulge on the snake.
# No long run was circled at all.
#
# His rule, verbatim: "so if line start after corner a is 3px wide and line
# ends at corner b is 3.1px wide, the max width on that line should be close
# to that."
#
# So the unit of measurement is the run BETWEEN TWO CORNERS, however short, and
# the test is local: a run declares its own expected width from its two ends,
# and anything much fatter in the middle is a bulge. No global threshold, no
# tuning against a handful of images - each run is judged against itself.
#
# ONE pass, FOUR separate outputs, never merged:
#   width_px      - item 3, is the stroke thick enough
#   width_cv      - item 4, is it consistent along its length
#   bulge_ratio   - item 6, a LOCAL swelling against this run's own ends
#   wander_px     - item 1, does the centre of the run leave a straight path
# ---------------------------------------------------------------------------

def _tls_residual(seg):
    """(rms, max) perpendicular distance from a run to its best-fit LINE.

    Used for short corner-to-corner runs, where the question is simply whether
    the stroke ran straight between its two corners. The quadratic fit above is
    for long windows, where an intended arc has to be allowed for.
    """
    c = seg - seg.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    d = c @ vt[-1]
    return float(np.sqrt((d ** 2).mean())), float(np.abs(d).max())


def _inward_normals(p):
    """Unit normals for a contour, pointing into the shape (OpenCV CCW)."""
    t = np.gradient(p, axis=0)
    ln = np.hypot(t[:, 0], t[:, 1]) + 1e-9
    return np.column_stack([t[:, 1] / ln, -t[:, 0] / ln])


def _widths_along(p, ink, max_w=80):
    """Local stroke thickness at each contour point.

    Marches inward along the normal until it leaves the ink - which is the
    literal reading of "measure the line perpendicular to itself". Points whose
    march never exits inside `max_w` sit on a large filled region rather than a
    line, and are returned as NaN so they cannot pollute a width statistic.
    """
    h, w = ink.shape
    n = _inward_normals(p)
    out = np.full(len(p), np.nan)
    for i in range(len(p)):
        x0, y0 = p[i]
        nx, ny = n[i]
        for t in range(1, max_w + 1):
            xi = int(round(x0 + nx * t))
            yi = int(round(y0 + ny * t))
            if xi < 0 or yi < 0 or xi >= w or yi >= h or not ink[yi, xi]:
                out[i] = t
                break
    return out


def measure_runs(ink, sigma_small=2.0, corner_deg=32.0, corner_half=4,
                 min_run=12, max_w=80, end_frac=0.25):
    """Corner-to-corner runs, with four independent measurements each.

    `end_frac` is how much of each end defines the run's own expected width.
    """
    ink = np.asarray(ink).astype(bool)
    cs, _ = cv2.findContours(ink.astype(np.uint8), cv2.RETR_LIST,
                             cv2.CHAIN_APPROX_NONE)
    runs = []
    for c in cs:
        q = c.reshape(-1, 2).astype(float)
        p = _resample_closed(q, step=1.0)
        if p is None or len(p) < min_run * 2:
            continue
        sm = _smooth_closed(p, sigma_small)
        corner = _corner_flags(sm, half=corner_half, deg=corner_deg)
        idx = np.flatnonzero(corner)
        n = len(sm)
        if len(idx) < 2:
            spans = [(0, n)]
        else:
            spans = []
            for a, b in zip(idx, np.r_[idx[1:], idx[0] + n]):
                L = b - a
                if L >= min_run:
                    spans.append((a % n, int(L)))
        wid = _widths_along(sm, ink, max_w)
        for a, L in spans:
            sel = [(a + t) % n for t in range(L)]
            wv = wid[sel]
            good = np.isfinite(wv)
            if good.sum() < max(6, L // 3):
                continue                        # a filled region, not a line
            wv = wv[good]
            k = max(2, int(len(wv) * end_frac))
            ends = np.r_[wv[:k], wv[-k:]]
            base = float(np.median(ends))
            if base <= 0:
                continue
            seg = sm[sel]
            wander, _ = _tls_residual(seg) if L >= 8 else (0.0, 0.0)
            runs.append({
                'length_px': int(L),
                'width_px': float(np.median(wv)),
                'width_end_px': base,
                'width_max_px': float(wv.max()),
                'width_cv': float(wv.std() / max(wv.mean(), 1e-6)),
                'bulge_ratio': float(wv.max() / base),
                'wander_px': float(wander),
                'at': (float(seg[L // 2][0]), float(seg[L // 2][1])),
            })
    return runs


def summarise_runs(runs, bulge_flag=1.6):
    """Headline numbers. Worst-driven: one bad run can sink a part."""
    if not runs:
        return {'runs': 0}
    L = np.array([r['length_px'] for r in runs], float)
    br = np.array([r['bulge_ratio'] for r in runs])
    wd = np.array([r['width_px'] for r in runs])
    cvv = np.array([r['width_cv'] for r in runs])
    wan = np.array([r['wander_px'] for r in runs])
    worst = runs[int(np.argmax(br))]
    return {
        'runs': len(runs),
        'width_px': round(float(np.average(wd, weights=L)), 2),
        'width_min_px': round(float(wd.min()), 2),
        'width_cv': round(float(np.average(cvv, weights=L)), 3),
        'bulge_p95': round(float(np.percentile(br, 95)), 3),
        'bulge_max': round(float(br.max()), 3),
        'bulge_flagged': int((br > bulge_flag).sum()),
        'bulge_worst_at': worst['at'],
        'wander_p95_px': round(float(np.percentile(wan, 95)), 3),
    }


# ---------------------------------------------------------------------------
# MEDIAL AXIS STROKE MEASUREMENT - the instrument that matches what Tyler sees
#
# 2026-09-08. Two wrong instruments preceded this one and both are kept above,
# because the reason each failed is worth more than the code was.
#
# THE MISUNDERSTANDING THAT COST THE DAY, and it was a real one on both sides:
# Tyler kept saying "wavy". He meant THE BOUNDARY WHERE BLACK MEETS WHITE looks
# uneven. I built a tool that measures whether the CENTRELINE of a run wanders.
# Same word, two different objects. His three circled defects on the winged-roundel emblem
# were, measured:
#
#   * the ellipse's upper stroke SWELLING from ~15px to ~25px along its length,
#     top edge dead straight
#   * an abrupt STEP in stroke width at the lower star
#   * a single angular KINK in the star point's taper
#
# Not one of them is centreline wander. In two of them the centreline is
# perfectly straight and the WIDTH is the defect. That is also why the smoothing
# sweep never moved his verdict: smoothing changes the path, and the path was
# already straight.
#
# WIDTH IS MEASURED ON THE MEDIAL AXIS, never by marching in from the contour.
# The contour march was tried and it broke on the letter S: from the boundary of
# a filled glyph the march walks into a large solid region and returns 40-80px,
# which then becomes "the width of a line". Bulge ratios of 27x on a 12px stroke
# were the symptom. On the medial axis the distance transform IS the half-width
# by definition, with no direction to guess.
#
# BOTH POLARITIES ARE MEASURED. The S's defect is the WHITE CHANNEL between the
# glyph's outline and its fill changing width. To Tyler's eye a white channel is
# a line exactly as a black stroke is, so ink-only measurement would have missed
# the letter that started this.
#
# Four outputs, never merged (his constraint: "i dont want to hear in 3 hours
# that #3 tool silently overrode the output from tool #4"):
#     width_px  item 3 | width_cv item 4 | bulge_ratio item 6 | kink_deg (new)
# ---------------------------------------------------------------------------

def _branch_paths(skel):
    """Split a skeleton into branches, each returned as an ordered pixel path."""
    from scipy.ndimage import convolve
    s = skel.astype(np.uint8)
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
    deg = convolve(s, k, mode='constant') * s
    nodes = (deg >= 3) | ((deg == 1) & (s > 0))     # junctions and endpoints
    interior = (s > 0) & ~((deg >= 3))
    lab = label(interior, connectivity=2)
    paths = []
    for i in range(1, lab.max() + 1):
        ys, xs = np.nonzero(lab == i)
        if len(ys) < 3:
            continue
        pts = list(zip(ys.tolist(), xs.tolist()))
        pset = set(pts)
        ends = [p for p in pts
                if sum((p[0] + dy, p[1] + dx) in pset
                       for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                       if (dy or dx)) <= 1]
        start = ends[0] if ends else pts[0]
        order, seen, cur = [start], {start}, start
        while True:
            nxt = None
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not (dy or dx):
                        continue
                    c = (cur[0] + dy, cur[1] + dx)
                    if c in pset and c not in seen:
                        nxt = c
                        break
                if nxt:
                    break
            if nxt is None:
                break
            order.append(nxt); seen.add(nxt); cur = nxt
        if len(order) >= 3:
            paths.append(np.array(order, dtype=float))   # (row, col)
    return paths


def _kink_deg(path, span=6):
    """Largest single direction break along a path, in degrees.

    A DESIGNED corner sits where the source drew one. A kink is a direction
    break the tracer invented in the middle of what should be one edge. This
    reports the break; deciding which kind it is needs the source, and that
    comparison is not made here.
    """
    if len(path) < span * 2 + 2:
        return 0.0
    p = path[:, ::-1]                                  # to (x, y)
    a = p[span:-span] - p[:-2 * span]
    b = p[2 * span:] - p[span:-span]
    na = np.hypot(*a.T) + 1e-9
    nb = np.hypot(*b.T) + 1e-9
    cos = np.clip((a * b).sum(1) / (na * nb), -1, 1)
    return float(np.degrees(np.arccos(cos)).max())


def _junction_mask(skel, dist, reach=1.6):
    """Skeleton points too close to a junction to be judged as line width.

    Where two strokes meet, the largest inscribed circle is genuinely bigger
    than either stroke - that is what a junction IS, not a tracing defect.
    Measured on the winged-roundel emblem: the wing feathers join the wing body at dozens of
    junctions and every one read as a swelling, scoring 1.81 in regions Tyler
    called clean while his actual defects scored 1.39-1.74. Judging junctions
    as line width does not just add noise, it inverts the ranking.

    Excluded out to `reach` x the local width, which is the distance over which
    the merge inflates the inscribed circle.
    """
    from scipy.ndimage import convolve
    from scipy.spatial import cKDTree
    s = skel.astype(np.uint8)
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
    deg = convolve(s, k, mode='constant') * s
    jy, jx = np.nonzero(deg >= 3)
    ys, xs = np.nonzero(skel)
    if len(jy) == 0:
        return np.zeros(len(ys), bool)
    pts = np.column_stack([ys, xs]).astype(float)
    w = 2.0 * dist[ys, xs]
    tree = cKDTree(np.column_stack([jy, jx]).astype(float))
    d, _ = tree.query(pts)
    return d < (w * reach)


def _local_bulge(skel, dist, radius_mult=4.0, min_r=12.0, drop_junctions=True):
    """Width of every skeleton point against the LOCAL trend around it.

    Branch segmentation cannot carry this measure and three attempts proved it:
    a local swelling puts junctions in the medial axis, the run fragments there,
    and the fragment's own ends land INSIDE the defect - so it compares the
    bulge to itself and reports ~1.0. Measured: a true 2.0x swell read 1.22.

    So no segmentation. Each skeleton point is compared to the median width of
    the skeleton within a radius around it, which is exactly "is this bit fatter
    than the line it belongs to" and cannot be defeated by the defect splitting
    its own run.

    Returns (coords Nx2 as (y, x), widths, ratios).
    """
    from scipy.spatial import cKDTree
    ys, xs = np.nonzero(skel)
    if len(ys) < 8:
        return None
    w = 2.0 * dist[ys, xs]
    pts = np.column_stack([ys, xs]).astype(float)
    keep = ~_junction_mask(skel, dist) if drop_junctions else np.ones(len(w), bool)
    if keep.sum() < 8:
        return None
    r = max(min_r, float(np.median(w)) * radius_mult)
    tree = cKDTree(pts)
    ratios = np.ones(len(w))
    for i, nb in enumerate(tree.query_ball_point(pts, r)):
        if len(nb) >= 5:
            base = float(np.median(w[nb]))
            if base > 0:
                ratios[i] = w[i] / base
    # Junction neighbourhoods still inform the local MEDIAN (they are part of
    # the line) but are never themselves reported as bulges.
    return pts[keep], w[keep], ratios[keep]


def measure_strokes(ink, min_len=10, end_frac=0.25, both_polarities=True):
    """Per-stroke width, consistency, bulge and kink, on the medial axis.

    Runs on the ink AND on its enclosed white channels, because a white gap
    between two black shapes reads as a line to the eye just as a stroke does.
    """
    from skimage.morphology import medial_axis
    ink = np.asarray(ink).astype(bool)
    masks = [('ink', ink)]
    if both_polarities:
        holes = ~ink
        hl = label(holes, connectivity=1)
        border = set(np.unique(np.r_[hl[0], hl[-1], hl[:, 0], hl[:, -1]]))
        enclosed = np.isin(hl, [i for i in range(1, hl.max() + 1)
                                if i not in border])
        masks.append(('white', enclosed))

    out = []
    for pol, m in masks:
        if not m.any():
            continue
        skel, dist = medial_axis(m, return_distance=True)

        # Local bulge, computed over the whole skeleton and independent of how
        # it happens to fragment into branches.
        lb = _local_bulge(skel, dist)
        if lb is not None:
            lp, lw, lr = lb
            from scipy.spatial import cKDTree as _KD
            ltree = _KD(lp)
        else:
            ltree = None

        for path in _branch_paths(skel):
            if len(path) < min_len:
                continue
            rr = path[:, 0].astype(int); cc = path[:, 1].astype(int)
            w = 2.0 * dist[rr, cc]

            # TRIM THE ENDS BY ONE STROKE WIDTH. The medial axis of a blunt end
            # forks toward the two corners, and those spurs are an artefact of
            # the transform, not a feature of the artwork. Measured on a
            # perfectly uniform 20px bar: untrimmed it reported an 18.9 degree
            # kink and a 1.9x bulge, both entirely spurious. Trimming by the
            # local width removes the fork region and leaves the real stroke.
            trim = int(round(np.median(w) * 0.5))
            if trim > 0 and len(w) > 4 * trim + 4:
                w = w[trim:-trim]
                path = path[trim:-trim]
                rr, cc = rr[trim:-trim], cc[trim:-trim]
            if len(w) < min_len:
                continue
            # A medial-axis fork at a blunt end produces two SPURS roughly one
            # stroke-width long. They are an artefact of the transform and they
            # faked an 18.9 degree kink on a perfectly straight bar.
            if len(w) < 2.0 * float(np.median(w)):
                continue

            k = max(2, int(len(w) * end_frac))
            # The run's own expected width, Tyler's rule: "if line start after
            # corner a is 3px wide and line ends at corner b is 3.1px wide, the
            # max width on that line should be close to that."
            #
            # Interpolated between the two ends rather than a flat median of
            # them, so a deliberate TAPER is not read as a bulge. A flat
            # baseline scored a clean 20->36 taper at 1.89x, indistinguishable
            # from a real defect.
            w0 = float(np.median(w[:k])); w1 = float(np.median(w[-k:]))
            base_prof = np.linspace(w0, w1, len(w))
            base = float(np.median(np.r_[w[:k], w[-k:]]))
            if base <= 0 or np.any(base_prof <= 0):
                continue
            rel = w / base_prof
            # local-trend ratios for the points on this branch
            if ltree is not None:
                _, ii = ltree.query(np.column_stack([rr, cc]).astype(float))
                brat = lr[ii]
            else:
                brat = rel
            mid = path[len(path) // 2]
            out.append({
                'polarity': pol,
                'length_px': int(len(path)),
                'width_px': float(np.median(w)),
                'width_min_px': float(w.min()),
                'width_end_px': base,
                'width_max_px': float(w.max()),
                'width_cv': float(w.std() / max(w.mean(), 1e-6)),
                'bulge_ratio': float(brat.max()),
                'pinch_ratio': float(brat.min()),
                'bulge_end_ratio': float(rel.max()),
                'kink_deg': _kink_deg(path),
                'at': (float(mid[1]), float(mid[0])),
            })
    return out


def summarise_strokes(strokes, bulge_flag=1.35, pinch_flag=0.70,
                      kink_flag=25.0):
    """Headline numbers. Worst-driven - one bad stroke can sink a part."""
    if not strokes:
        return {'strokes': 0}
    L = np.array([s['length_px'] for s in strokes], float)
    br = np.array([s['bulge_ratio'] for s in strokes])
    pr = np.array([s['pinch_ratio'] for s in strokes])
    cv = np.array([s['width_cv'] for s in strokes])
    wd = np.array([s['width_px'] for s in strokes])
    kk = np.array([s['kink_deg'] for s in strokes])
    worst = strokes[int(np.argmax(br))]
    return {
        'strokes': len(strokes),
        'width_px': round(float(np.average(wd, weights=L)), 2),   # item 3
        'width_min_px': round(float(wd.min()), 2),                # item 3
        'width_cv': round(float(np.average(cv, weights=L)), 3),   # item 4
        'bulge_p95': round(float(np.percentile(br, 95)), 3),      # item 6
        'bulge_max': round(float(br.max()), 3),                   # item 6
        'bulge_flagged': int((br > bulge_flag).sum()),
        'pinch_flagged': int((pr < pinch_flag).sum()),
        'kink_max_deg': round(float(kk.max()), 1),                # new
        'kink_flagged': int((kk > kink_flag).sum()),
        'worst_at': worst['at'],
    }


def measure_bulge(ink, both_polarities=True, radius_mult=4.0):
    """Local width swellings and pinches, computed with NO branch segmentation.

    Kept as its own entry point because every attempt to derive it from runs
    failed the same way, three times:

      contour march  - walked into filled regions, reported 27x on a 12px stroke
      corner runs    - the defect's own shoulders split the run, so its "ends"
                       landed inside the defect and it compared to itself
      medial branches- the spur prune that removes end-forks also removed the
                       short fat segments that ARE the bulge (a real 1.5x swell
                       read 1.00 with cv 0.00)

    Every one of those is the same mistake in a different costume: using a
    segmentation that the defect itself disturbs. So this measure uses none.
    Each skeleton point is compared to the median width of skeleton within a
    radius, which is literally "is this bit fatter than the line it is part of".

    Both polarities, because a white channel changing width reads as a defect
    exactly as a black stroke does - that is a letterform.
    """
    from skimage.morphology import medial_axis
    ink = np.asarray(ink).astype(bool)
    masks = [('ink', ink)]
    if both_polarities:
        hl = label(~ink, connectivity=1)
        border = set(np.unique(np.r_[hl[0], hl[-1], hl[:, 0], hl[:, -1]]))
        masks.append(('white', np.isin(
            hl, [i for i in range(1, hl.max() + 1) if i not in border])))

    res = {}
    for pol, m in masks:
        if not m.any():
            res[pol] = None
            continue
        skel, dist = medial_axis(m, return_distance=True)
        lb = _local_bulge(skel, dist, radius_mult=radius_mult)
        if lb is None:
            res[pol] = None
            continue
        pts, w, r = lb
        i = int(np.argmax(r))
        j = int(np.argmin(r))
        res[pol] = {
            'width_px': round(float(np.median(w)), 2),
            'width_min_px': round(float(w.min()), 2),
            'width_cv': round(float(w.std() / max(w.mean(), 1e-6)), 3),
            'bulge_max': round(float(r.max()), 3),
            'bulge_p99': round(float(np.percentile(r, 99)), 3),
            'bulge_at': (float(pts[i][1]), float(pts[i][0])),
            'pinch_min': round(float(r.min()), 3),
            'pinch_at': (float(pts[j][1]), float(pts[j][0])),
            'points': int(len(w)),
            '_pts': pts, '_w': w, '_r': r,
        }
    return res
