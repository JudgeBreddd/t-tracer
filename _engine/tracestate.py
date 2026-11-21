#!/usr/bin/env python3
"""tracestate.py - T-Tracer

Per-run mutable state for one trace: the strategy memo and the live-view sink.

ONE HOME. These were module globals in candidates.py; `_memo`/`_progress` read
them and the driver wrote them with `global`. Split across modules, `global`
would bind to the writer's module and the reader would never see it -- the memo
silently stops (composite recomputes nested, 6.7s and 3.4 GB per image) and the
live view goes dark, with no error and no failing test. Writers must assign
through this module: `tracestate._MEMO = {}`, never `global _MEMO`.

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
# Per-image memo. run_one() opens it before the strategy loop and closes it
# after; while it is open, a strategy that is built on another strategy's
# mask (keyfill -> keyline, composite -> otsu + nested) gets the mask that
# strategy already produced for the same image instead of recomputing it.
# Outside run_one the memo is None and every call computes, so the strategies
# stay ordinary functions. Only default-argument calls are memoised - a call
# with overrides is a different computation and always runs.
# ---------------------------------------------------------------------------

_MEMO = None

# Live-view progress sink (see run_one's on_step). `_PROGRESS` is set per
# strategy by _run_strategies to a callable(step, image) and is None for the
# CLI; `_progress()` is the only thing long loops call, and when the sink is
# None it is one comparison. Frames are rate-limited HERE, at the source, to
# PROGRESS_HZ so a tight loop cannot flood the display or slow itself down.
_PROGRESS = None
_PROGRESS_T = 0.0
PROGRESS_HZ = 3.0


def _progress(step, make_image):
    """Emit a live frame if one is due. `make_image` is a zero-arg callable
    so the frame is only built when it will be shown."""
    global _PROGRESS_T
    if _PROGRESS is None:
        return
    now = time.monotonic()
    if now - _PROGRESS_T < 1.0 / PROGRESS_HZ:
        return
    _PROGRESS_T = now
    _PROGRESS(step, make_image())


def _memo(name, fn, *args, **kw):
    if _MEMO is None or kw:
        return fn(*args, **kw)
    if name not in _MEMO:
        _MEMO[name] = fn(*args, **kw)
    return _MEMO[name].copy()
