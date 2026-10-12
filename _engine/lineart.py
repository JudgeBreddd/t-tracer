#!/usr/bin/env python3
"""lineart.py - T-Tracer

The learned lineart annotator, and the memory budget its subprocess forces.

Load-bearing for a DEFAULT run: `strategies.strat_nested` calls
`_lineart_lines`, and `strat_composite` -- on the sheet -- calls `strat_nested`.
The torch venv is not an optional extra.

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
# The lineart annotator -- learned boundaries, shared by `nested`
#
# `strat_neural` was deleted 2026-09-22; this infrastructure stayed because
# `nested` calls `_lineart_lines`, and `composite` -- on the sheet -- calls
# `nested`. The torch venv is therefore load-bearing for a DEFAULT run, not
# an optional extra.
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
# Requires the separate lineart venv (torch). Absent it, `nested` raises, and
# `composite` raises with it. See lineart_backend.py.
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

    It lived in _engine/.lineart_cache, which is inside the synced workspace - the
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
            f'lineart venv missing at {LINEART_PY} -- `nested` needs it, and '
            '`composite` is on the sheet and calls `nested`, so the default '
            'run needs it too. Create it with torch + controlnet_aux.')

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
