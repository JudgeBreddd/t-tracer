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
# candidates.py was split into siblings on 2026-09-22 (2,370 lines -> ~900).
# This file is the FACADE and the driver: source loading, the per-image run,
# the CLI, and the public name every other module imports.
#
#   tracestate.py  per-run memo + live-view sink   (assign THROUGH the module)
#   masks.py       mask-domain helpers
#   lineart.py     the learned annotator + its memory budget
#   strategies.py  the strategies and the registry
#   vector.py      mask -> Beziers -> SVG, and rasterising back
#
# The re-export below is deliberate: `import candidates as C` is the public
# interface for server.py, pick.py, hygiene.py, burnmap.py, region_graph.py,
# grade_crops.py and the tests, and it stays source-compatible. New code should
# import the sibling directly. See _engine/CONTEXT.md.
# ---------------------------------------------------------------------------
import tracestate
from tracestate import _memo, _progress
from masks import (_background_color, _between, _grow_counters, _keep_white_channels,
                   _kmeans_lab, _lightness, _pick_k, _silhouette_of, _true_lab)
from lineart import (ANNOTATOR_MB, HEADROOM_MB, LINEART_BACKEND, LINEART_CACHE,
                     LINEART_PY, NEURAL_MAX_DIM, WORKER_MB, _annotator_slot,
                     _cache_dir, _lineart_activation, _lineart_lines, _lineart_python)
from strategies import (ALL_STRATEGIES, ALPHA_AWARE, LAYERED, OPTIONAL, STRATEGIES,
                        strat_bgdist, strat_composite, strat_fieldflip, strat_keyline,
                        strat_kmeans, strat_kmeans_layered, strat_layerlines,
                        strat_nested, strat_otsu, strat_silhouette)
from vector import (_bezier_from_hermite, _hex, _progress_canvas, _sample_beziers,
                    clean_mask, composite_layers, count_nodes, fit_beziers,
                    mask_to_paths, paths_to_svg, paths_to_svg_layered, rasterize)

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


# ---------------------------------------------------------------------------
# Source loading
#
# The source used to be DECODED TWICE and buffered five times over before the
# `--max-dim` resize ever ran. On the two 12396x12396 USMC images in the
# calibration set that is ~3.5 GB of peak allocation for ONE image, and workers
# multiply it. Measured, per full-size RGBA buffer, on those: 614 MB.
#
#   raw.convert('RGBA') -> np.array(...)[:, :, 3]   the [:,:,3] is a VIEW, so
#                                                   the whole RGBA array stays
#                                                   alive just to hold alpha
#   flatten_to_white(path)                          opens and decodes AGAIN,
#                                                   then Image.new + composite
#                                                   + convert = 3 more buffers
#
# One decode, alpha copied out so the big buffer can be freed, and the same
# `Image.alpha_composite` call as before so the pixels are provably unchanged.
#
# The megapixel cap is an ABSURDITY guard, not a working limit. 200 MP is set
# above the largest real calibration image (134 MP) on purpose: a cap that
# rejects ground truth would quietly invalidate rows in picks.jsonl. Override
# with TT_MAX_MEGAPIXELS.
# ---------------------------------------------------------------------------

MAX_MEGAPIXELS = max(1.0, float(os.environ.get('TT_MAX_MEGAPIXELS', '200')))
# Tyler locked 1600 on 2026-09-08 and turned it on by default 2026-09-14: it
# produced 55/58 shippable. The app and the CLI both read this one value.
WORK_DIM_DEFAULT = 1600
# Sources this small keep their native size. Tyler, 2026-09-14: a 385px JPEG
# enlarged to 1600 traced with rippled edges; at native size it was clean.
NATIVE_MAX_PX = 400


def _align_pillow_limit():
    """Pillow's own bomb guard warns at 89.5 MP and raises at 2x that, which is
    both noisier and looser than ours. Point it at the same number so there is
    one limit, not two that disagree."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = int(MAX_MEGAPIXELS * 1_000_000)


_align_pillow_limit()


class SourceTooLarge(ValueError):
    """Raised before any decode, so the pixels are never allocated."""


def load_source(img_path, work_dim=0):
    """Return (rgb_HxWx3, alpha_HxW or None), resized to `work_dim` FIRST.

    RESIZE IS STEP ONE. Everything downstream then runs at one agreed size
    instead of each stage picking its own -- which is what produced a pipeline
    that downscaled to 2048, then to 1024 for the annotator, then supersampled
    x6, then resampled x2 again for scoring.

    Doing it here rather than in run_one is the memory fix as well as the
    architecture fix. The 134 MP USMC sources needed ~3.2 GB of full-size
    buffers to produce a 2048px working image, because the composite ran before
    the resize. Resizing first, the only full-size allocation left is the decode
    itself, and JPEG sources avoid even that.

    `work_dim` 0 keeps the original size (the default is WORK_DIM_DEFAULT).
    Sources whose long edge is NATIVE_MAX_PX or less are never resized.
    """
    from PIL import Image
    with Image.open(img_path) as probe:
        w, h = probe.size
        fmt = probe.format
    if w * h > MAX_MEGAPIXELS * 1_000_000:
        raise SourceTooLarge(
            f'{w}x{h} = {w * h / 1e6:.1f} MP exceeds the {MAX_MEGAPIXELS:.0f} MP '
            f'limit ({fmt}). Raise TT_MAX_MEGAPIXELS to allow it.')

    with Image.open(img_path) as im:
        # Which alpha to RETURN is decided by the band names, exactly as before.
        # A palette PNG carrying `transparency` in info reports bands ('P',) and
        # returns None here even though it is genuinely transparent. That is
        # pre-existing behaviour the strategies are calibrated against.
        report_alpha = 'A' in im.getbands()
        # JPEG only: ask libjpeg to decode at 1/2, 1/4 or 1/8 scale directly.
        # The full-size buffer is then never allocated at all.
        if work_dim and fmt == 'JPEG':
            im.draft('RGB', (work_dim, work_dim))
        rgba = im.convert('RGBA')

    # THE RESIZE. Before the composite, before anything.
    long_edge = max(rgba.size)
    if work_dim and long_edge > NATIVE_MAX_PX and long_edge != work_dim:
        sc = work_dim / long_edge
        tgt = (max(1, round(rgba.width * sc)), max(1, round(rgba.height * sc)))
        # BOX on the way down, BICUBIC on the way up. Matches the cv2
        # INTER_AREA / INTER_LANCZOS4 split the pipeline already used, and
        # avoids the aliasing a plain BICUBIC downscale leaves on fine art.
        rgba = rgba.resize(tgt, Image.BOX if sc < 1 else Image.BICUBIC)

    alpha = np.array(rgba.getchannel('A'))

    # Whether to COMPOSITE is asked of the real alpha channel, never the band
    # names -- convert('RGBA') expands palette transparency into a true alpha,
    # so a 'P'-mode PNG needs the composite despite reporting no 'A' band.
    # Skipping it let transparent pixels come through as their underlying
    # palette colour: 649,189 wrong pixels on a palette-mode PNG in the corpus, the
    # exact failure flatten_to_white was written to prevent.
    if int(alpha.min()) == 255:
        rgb = np.array(rgba.convert('RGB'))
        rgba.close()
        return rgb, (alpha if report_alpha else None)

    bg = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
    flat = Image.alpha_composite(bg, rgba)
    bg.close(); rgba.close()
    rgb = np.array(flat.convert('RGB'))
    flat.close()
    return rgb, (alpha if report_alpha else None)
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


_BATCH_LOCK_FH = None          # module-global: holds the lock for the
                               # life of the process. The OS releases it
                               # on exit, including a crash or Ctrl-C.


def _acquire_batch_lock(verbose=True):
    """Only ONE tracing batch may run at a time, machine-wide.

    The annotator has been serialised since 09-06, but the WORKERS never were:
    each invocation sized its own pool from free memory at the instant it
    started, so two batches launched seconds apart both measured a quiet machine
    and both committed. Measured failure, 09-08: `-j 6` (5.4 GB of workers) plus
    a second run's worker plus one annotator came to ~9.7 GB on a 14 GB laptop
    with an editor and a browser already resident, and the OOM killer took the
    editor.

    Per-process resource awareness cannot fix that -- the processes cannot see
    each other. A lock file can, and it is the same fcntl/msvcrt pattern
    `_annotator_slot` already proved across separate invocations.
    """
    LINEART_CACHE.mkdir(parents=True, exist_ok=True)
    try:
        fh = open(LINEART_CACHE / '.batch.lock', 'w')
        try:
            import fcntl
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                if verbose:
                    print('-- another T-Tracer batch is running; waiting for it '
                          'to finish (this is what stops the two of them from '
                          'OOM-killing each other)')
                fcntl.flock(fh, fcntl.LOCK_EX)
        except ImportError:                          # Windows
            import msvcrt
            waited = False
            while True:
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1); break
                except OSError:
                    if verbose and not waited:
                        print('-- another T-Tracer batch is running; waiting')
                        waited = True
                    time.sleep(0.5)
    except Exception:
        return None                     # never let locking break a real run
    global _BATCH_LOCK_FH
    _BATCH_LOCK_FH = fh
    return fh


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_one(img_path, out_root, args):
    import hygiene

    rgb, alpha = load_source(img_path, getattr(args, 'work_dim', WORK_DIM_DEFAULT))
    # Order-catalogue art runs to 134 MP. With --scale supersampling on top that
    # is billions of pixels and the run never returns. Engraved artwork is ~32mm
    # tall, so detail past a couple of thousand pixels cannot reach the plate.
    # With --work-dim on (the default) the source is already 1600px here and
    # this never fires; it only guards --work-dim 0. It is NOT a no-op on
    # calibration at native size: 26 of 82 picked images exceed 2048px.
    if getattr(args, 'max_dim', 0) and max(rgb.shape[:2]) > args.max_dim:
        _s = args.max_dim / max(rgb.shape[:2])
        rgb = cv2.resize(rgb, (max(1, int(rgb.shape[1] * _s)), max(1, int(rgb.shape[0] * _s))),
                         interpolation=cv2.INTER_AREA)
        if alpha is not None:
            alpha = cv2.resize(alpha, (rgb.shape[1], rgb.shape[0]),
                               interpolation=cv2.INTER_AREA)
    # Small sources: trace BIGGER than they arrived, not smaller.
    #
    # `IMG_2379` is 199x254 and Tyler rejected it with "words look rough".
    # That is not a classification failure -- at 199px a letter stroke is about
    # 6px and the threshold quantises its curve into a staircase before the
    # tracer ever sees it. The anti-aliased edges in the source carry sub-pixel
    # information about where that curve actually runs, and thresholding at
    # native size throws it away. Upsampling first recovers it: measured on
    # that image, the lettering go from ragged to clean.
    #
    # DEFAULT 0 (OFF), and that is deliberate rather than timid. 11 of the 82
    # images with a recorded pick are under 800px, so turning this on by
    # default would change the geometry that 11 rows of `picks.jsonl` point at,
    # and a pick that cannot be reproduced stops being ground truth. Same
    # standard the 2026-09-06 determinism work was held to. Turning it on is
    # Tyler's call, not a side effect.
    if getattr(args, 'min_dim', 0) and max(rgb.shape[:2]) < args.min_dim:
        _s = args.min_dim / max(rgb.shape[:2])
        rgb = cv2.resize(rgb, (max(1, int(rgb.shape[1] * _s)), max(1, int(rgb.shape[0] * _s))),
                         interpolation=cv2.INTER_LANCZOS4)
        if alpha is not None:
            alpha = cv2.resize(alpha, (rgb.shape[1], rgb.shape[0]),
                               interpolation=cv2.INTER_LANCZOS4)

    h, w, _ = rgb.shape
    src_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    out_dir = out_root / img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    names = args.strategies.split(',') if args.strategies else list(STRATEGIES)

    # `nested` and `composite` both need the same activation map. Computing it
    # once up front means the second one is always a cache hit, and -- once
    # images run in parallel -- that two workers on the SAME image can never
    # both pay the 6.7s cold cost or race on the same cache key.
    if any(n.strip() in ('nested', 'composite') for n in names):
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

    # Optional per-step observer: on_step(strategy, step, image) with `image`
    # an HxW bool mask, an HxWx3 RGB image, or a list of (mask, rgb) layers.
    # Costs one attribute read when unset. The app's live view plugs in here;
    # the CLI never sets it.
    on_step = getattr(args, 'on_step', None)

    tracestate._MEMO = {}
    try:
        _run_strategies(names, rgb, alpha, src_gray, args, out_dir,
                        results, previews, on_step)
    finally:
        tracestate._MEMO = None

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
    # Hygiene again at the burned size (70 and 150 mm), on the same full-size
    # raster the reviewer is shown - see defects.py. Never allowed to cost the
    # trace itself.
    try:
        import defects
        from PIL import Image
        with Image.open(img_path) as im:
            src_long = max(im.size)
        defects.score_image(out_dir, rgb, src_long, results)
    except Exception as e:
        print(f'  ! defect scoring failed: {type(e).__name__}: {e}')
    if previews:
        # All sheets land in one folder. Per-image folders are right for the
        # deliverables, but reviewing means flipping through every sheet in a
        # row, and that should be one directory of images, not fifteen clicks.
        sheets = out_root / '_sheets'
        sheets.mkdir(parents=True, exist_ok=True)
        _contact_sheet([(n, p, results[n]) for n, p in previews],
                       rgb, sheets / f'{img_path.stem}.png')
    return results


def _run_strategies(names, rgb, alpha, src_gray, args, out_dir, results,
                    previews, on_step=None):
    """The strategy loop of run_one. One entry in `results` per requested
    strategy: a score dict, or {'rejected': why} - a strategy that raises is
    recorded too, so a missing tile in the app always has a stated reason."""
    import hygiene
    h, w = rgb.shape[:2]

    for name in names:
        name = name.strip()
        fn = ALL_STRATEGIES.get(name)
        if fn is None:
            print(f'  ! unknown strategy {name!r}, skipping')
            continue

        try:
            if name in LAYERED:
                out = fn(rgb)
            elif name in ALPHA_AWARE:
                out = _memo(name, fn, rgb, alpha)
            else:
                out = _memo(name, fn, rgb)
        except Exception as e:
            print(f'  {name:<8} FAILED: {type(e).__name__}: {e}')
            results[name] = {'rejected': f'failed: {type(e).__name__}: {e}'}
            continue

        if name in LAYERED:
            _emit_layered(name, out, rgb, src_gray, args, out_dir, results,
                          previews, on_step)
            continue

        mask = out
        if args.invert:
            mask = ~mask
        if on_step:
            on_step(name, 'mask', mask)
            tracestate._PROGRESS = lambda step, img, _n=name: on_step(_n, step, img)
        mask = clean_mask(mask, src_gray, args.close, args.open, args.min_area_frac)
        if getattr(args, 'min_island', 0):
            mask = _grow_counters(mask, args.min_island)
        # Field Flip is an artwork-local polarity change. Cleanup is allowed
        # to close a hairline and min-island can grow counters, but neither is
        # allowed to turn the white frame into ink.
        if name == 'fieldflip':
            mask &= _silhouette_of(rgb, alpha)
        if on_step:
            on_step(name, 'clean', mask)

        ink_frac = float(mask.mean())
        if ink_frac < 0.001 or ink_frac > 0.98:
            print(f'  {name:<8} degenerate ({ink_frac:.1%} ink) - not emitted')
            results[name] = {'rejected': f'degenerate ink fraction {ink_frac:.3f}'}
            tracestate._PROGRESS = None
            if on_step:
                on_step(name, 'scored', mask)
            continue

        paths = mask_to_paths(mask, scale=args.scale, smooth=args.smoothing,
                              tol=args.tol)
        if not paths:
            print(f'  {name:<8} produced no usable contours')
            results[name] = {'rejected': 'no contours'}
            tracestate._PROGRESS = None
            continue

        svg = paths_to_svg(paths, w, h,
                           background='#FFFFFF' if args.bg else None,
                           height_mm=args.height_mm)
        (out_dir / f'{name}.svg').write_text(svg)

        rendered = rasterize(paths, w, h, ss=args.raster_ss)
        if on_step:
            on_step(name, 'render', rendered)
        sc = hygiene.score(rendered, src_gray, paths, count_nodes(paths),
                           height_mm=args.height_mm, ss=args.raster_ss, src_h=h,
                           src_rgb=rgb)
        results[name] = sc

        prev = (~rendered * 255).astype(np.uint8)
        prev = cv2.resize(prev, (w, h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f'{name}.png'), prev)
        previews.append((name, prev))
        if on_step:
            tracestate._PROGRESS = None
            on_step(name, 'scored', rendered)


def _emit_layered(name, layers, rgb, src_gray, args, out_dir, results,
                  previews, on_step=None):
    """Emission for a LAYERED strategy: one cleaned mask -> one <path> per
    colour, stacked in the order the strategy returned them (bottom first).
    Reuses clean_mask / mask_to_paths unchanged per layer; only the SVG
    writer, the raster, the preview and the scorer are the layered variants."""
    import hygiene
    h, w = rgb.shape[:2]
    if on_step:
        on_step(name, 'mask', layers)

    cleaned = []
    for mask, color in layers:
        m = clean_mask(mask, src_gray, args.close, args.open, args.min_area_frac)
        if getattr(args, 'min_island', 0):
            m = _grow_counters(m, args.min_island)
        if m.any():
            cleaned.append((m, color))
    if on_step:
        on_step(name, 'clean', cleaned)

    union = np.zeros((h, w), dtype=bool)
    for m, _ in cleaned:
        union |= m
    ink_frac = float(union.mean())
    if len(cleaned) < 2 or ink_frac < 0.001 or ink_frac > 0.98:
        why = ('abstained: fewer than two colours' if len(cleaned) < 2
               else f'degenerate ink fraction {ink_frac:.3f}')
        print(f'  {name:<8} {why} - not emitted')
        results[name] = {'rejected': why}
        return

    traced = []                                    # (paths, color)
    for m, color in cleaned:
        paths = mask_to_paths(m, scale=args.scale, smooth=args.smoothing,
                              tol=args.tol)
        if paths:
            traced.append((paths, color))
    if len(traced) < 2:
        results[name] = {'rejected': 'fewer than two layers produced contours'}
        return

    svg = paths_to_svg_layered(traced, w, h,
                               background='#FFFFFF' if args.bg else None,
                               height_mm=args.height_mm)
    (out_dir / f'{name}.svg').write_text(svg)

    rendered = [(rasterize(paths, w, h, ss=args.raster_ss), paths, color)
                for paths, color in traced]
    sc = hygiene.score_layered(rendered, src_gray, rgb, height_mm=args.height_mm,
                               ss=args.raster_ss, src_h=h)
    sc['layers'] = [
        {'fill': _hex(color), 'area_frac': round(float(r.mean()), 4),
         'paths': len(paths), 'nodes': count_nodes(paths)}
        for r, paths, color in rendered]
    results[name] = sc

    prev = composite_layers(rendered, w, h)
    cv2.imwrite(str(out_dir / f'{name}.png'), cv2.cvtColor(prev, cv2.COLOR_RGB2BGR))
    previews.append((name, prev))
    if on_step:
        on_step(name, 'scored', prev)


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
        if r.ndim == 3:                     # RGB preview (a layered candidate)
            return r if is_color else cv2.cvtColor(r, cv2.COLOR_RGB2BGR)
        return cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)

    order = [n for n, _ in rank_candidates({n: s for n, _, s in previews})]
    by_name = {n: (p, s) for n, p, s in previews}
    tiles = [('', 'SOURCE', fit(cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR), True), None)]
    tiles += [(str(i), n, fit(by_name[n][0]), by_name[n][1])
              for i, n in enumerate(order, 1)]

    pad, label_h = 12, 46
    crops = _native_crops(source_rgb, [('', cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR))]
                          + [(t[0], _bgr(by_name[t[1]][0])) for t in tiles[1:]])
    crop_w = sum(c.shape[1] for c in crops) + pad * (len(crops) + 1)
    crop_h = max(c.shape[0] for c in crops) + label_h // 2 + pad
    total_w = max(sum(t[2].shape[1] for t in tiles) + pad * (len(tiles) + 1), crop_w)
    sheet = np.full((tile_h + label_h + pad * 2 + crop_h, total_w, 3), 245, np.uint8)

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

    # Second row: the same detailed spot in every tile at NATIVE resolution, each pixel
    # shown as a 2x2 block. Two defects passed review because the only views were
    # downsampled previews (C178), and a wrong call came from judging fine detail at
    # contact-sheet size (C450). This row is the view those calls needed.
    y0 = label_h + tile_h + pad * 2
    cv2.putText(sheet, 'native pixels, 2x: same spot in every tile', (pad, y0 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
    x = pad
    for (num, _, _, _), c in zip(tiles, crops):
        top = y0 + label_h // 2
        sheet[top:top + c.shape[0], x:x + c.shape[1]] = c
        cv2.rectangle(sheet, (x, top), (x + c.shape[1], top + c.shape[0]), (200, 200, 200), 1)
        if num:
            cv2.putText(sheet, num, (x + 4, top + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2)
        x += c.shape[1] + pad

    cv2.imwrite(str(path), sheet)


def _bgr(img):
    """Previews are grayscale, or RGB for a layered candidate; the sheet is written as BGR."""
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR if img.ndim == 2 else cv2.COLOR_RGB2BGR)


CROP_PX = 120   # ponytail: one fixed window; a second window or a picker if one spot misses defects


def _native_crops(source_rgb, images, size=CROP_PX, zoom=2):
    """The most detailed CROP_PX square of the source (densest edges), cut from every
    image at its own pixels and enlarged with nearest-neighbour so nothing is smoothed.
    All images share the source's pixel grid (previews are resized to it)."""
    gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    s = min(size, h, w)
    density = cv2.boxFilter(cv2.Canny(gray, 60, 160).astype(np.float32), -1, (s, s),
                            normalize=False, borderType=cv2.BORDER_CONSTANT)
    cy, cx = np.unravel_index(int(np.argmax(density)), density.shape)
    y, x = min(max(cy - s // 2, 0), h - s), min(max(cx - s // 2, 0), w - s)
    out = []
    for _, img in images:
        c = img[y:y + s, x:x + s]
        c = c if c.ndim == 3 else cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
        out.append(cv2.resize(c, (s * zoom, s * zoom), interpolation=cv2.INTER_NEAREST))
    return out


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


def build_parser():
    """The CLI's argparse parser. Separate from main() so the app can take
    the DEFAULTS from it (server.cli_defaults) instead of re-typing them."""
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
    ap.add_argument('--min-island', type=int, default=0,
                    help='grow nested ink islands (letter counters) to at least '
                         'this many px across (0 = off). A counter smaller than '
                         'the eye can resolve is lost on the plate whatever the '
                         'tracer does. OFF by default: it changes geometry.')
    ap.add_argument('--min-dim', type=int, default=0,
                    help='upscale sources whose long edge is under this before '
                         'tracing (0 = off). Small art quantises its letterforms '
                         'into a staircase at native size; 800 is a good value. '
                         'OFF by default because it changes the geometry that '
                         '11 recorded picks point at.')
    ap.add_argument('--max-dim', type=int, default=2048,
                    help='downscale sources longer than this on the long edge '
                         '(0 = off). Guards against 100+ MP catalogue art.')
    ap.add_argument('--force-jobs', action='store_true',
                    help='allow -j above the memory-safe worker count. The '
                         'safe count exists because exceeding it OOM-kills '
                         'whatever else is running on the machine.')
    ap.add_argument('--work-dim', type=int, default=WORK_DIM_DEFAULT,
                    help='resize EVERY source to this long edge as the very '
                         'first step, so every stage downstream runs at one '
                         f'agreed size (default {WORK_DIM_DEFAULT}; 0 = off, '
                         f'keep native). Sources of {NATIVE_MAX_PX}px or less '
                         'always keep native size.')
    ap.add_argument('--out', default='candidates', help='output folder name')
    return ap


def main():
    args = build_parser().parse_args()

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
    # set immediately before each `cv2.kmeans` call -- see `strat_kmeans`.
    cv2.setNumThreads(1)

    # Every use of `folder` below is a PRIVATE-data path - the corpus, the
    # drop folder, the output root. It points at _private/, not at the code
    # directory, so nothing this script reads or writes can land in the
    # publishable tree. See paths.py for why.
    folder = paths.PRIVATE
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
            rgb = (load_source(src)[0] if src
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
        testers = folder / 'runs' / 'converter script testers'
        if testers.is_dir():
            images = sorted(p for p in testers.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if args.only:
                images = [p for p in images if p.name == args.only]

    if not images:
        print(f'No images found next to {folder} or in "converter script testers/".')
        return

    # Machine-wide: one batch at a time. Acquired BEFORE the pool is sized,
    # so the memory reading that sizes it is taken with no other batch live.
    _acquire_batch_lock()

    if args.jobs == 0:
        jobs = auto_jobs(len(images))
    else:
        jobs = max(1, min(args.jobs, len(images)))
        if jobs > 1:
            safe = auto_jobs(len(images), verbose=False)
            if jobs > safe:
                # CLAMPED, not warned. It used to print this and continue, and
                # on 09-08 it was overridden to -j 6 on a machine that could
                # afford 1 -- which is how the OOM killer got the user's editor.
                # An advisory that the tool ignores is not a safety feature.
                # --force-jobs is the deliberate escape hatch.
                if getattr(args, 'force_jobs', False):
                    print(f'-- WARNING: -j {jobs} exceeds the safe count '
                          f'({safe}); --force-jobs given, continuing as asked.')
                else:
                    print(f'-- -j {jobs} exceeds what this machine can safely '
                          f'afford right now ({safe}); using {safe}. Each worker '
                          f'holds ~{WORKER_MB} MB and the annotator peaks at '
                          f'~{ANNOTATOR_MB} MB. Pass --force-jobs to override.')
                    jobs = safe
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
