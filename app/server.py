#!/usr/bin/env python3
"""T-Tracer - local backend.

Wraps the existing pipeline rather than reimplementing any of it. `run_one` in
`_engine/candidates.py` is imported directly and called with exactly the CLI's
default arguments, so the app and the command line produce byte-identical SVGs.
That matters: `_private/candidates/picks.jsonl` is the project's calibration
ground truth, and it would stop meaning anything if the app traced differently
from the tool the picks were recorded with.

Serves a small JSON API on 127.0.0.1 to a static frontend. No auth and no
external binding on purpose - this is a single-user desktop app, and the port is
bound to loopback only.
"""
from __future__ import annotations

import json
import secrets
import shutil
import sys
import threading
import uuid
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

APP_DIR = Path(__file__).resolve().parent
PROJECT = APP_DIR.parent
# The tracing engine. Renamed from _scripts/ on 2026-09-10: the folder holds
# the algorithm, and app/ holds the interface, so "scripts" said nothing.
ENGINE = PROJECT / '_engine'
SCRIPTS = ENGINE          # back-compat alias; ENGINE is the name to use
STATIC = APP_DIR / 'static'

# Working files live OUTSIDE the project, and this is not optional.
#
# This project sits inside a OneDrive tree. Every traced job writes a folder of
# PNGs and SVGs - roughly 15 files per image - and with .work/ under app/ that
# is a sync storm on every run. the reviewer, watching it happen: "one drive is having
# a stroke." It is the same trap the lineart venv already hit: OneDrive's skip
# list matches ".venv" exactly and nothing else, so a dot-prefixed name inside
# the project buys nothing.
#
# XDG cache on Linux/macOS, LOCALAPPDATA on Windows. Override with TT_WORK_DIR.
def _work_dir() -> Path:
    import os
    if env := os.environ.get('TT_WORK_DIR'):
        return Path(env).expanduser()
    if sys.platform == 'win32':
        base = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData/Local'))
    else:
        base = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
    return base / 't-tracer' / 'work'


WORK = _work_dir()

sys.path.insert(0, str(ENGINE))
import candidates as C                                   # noqa: E402
import paths as _paths                                   # noqa: E402

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}

# One VERSION file at the repo root is the single source of truth - the
# installer's own AppVersion and this string used to be set independently
# (a hardcoded literal here vs. the git tag passed to Inno Setup) and drifted:
# a machine on installer v0.1.1 kept reporting "0.1.0" in the app itself,
# which is exactly the kind of thing that makes the update checker lie.
# installer.iss ships this file next to {app} (one level above app/, same
# place requirements.txt lands); the release workflow overwrites it with the
# tag being built, so both numbers can only ever come from that one push.
def _read_app_version() -> str:
    try:
        return (Path(__file__).resolve().parent.parent / 'VERSION').read_text().strip()
    except OSError:
        return '0.0.0-dev'                                   # running from source, untagged


APP_VERSION = _read_app_version()
# Update checks ask GitHub about this repo. Overridable so a fork does not
# report someone else's releases as its own updates.
import os as _os
REPO = _os.environ.get('TT_REPO', 'JudgeBreddd/t-tracer')

# Item 8: a random per-launch token, required on every /api/ request.
#
# 127.0.0.1-only binding was the entire access control until now. That is fine
# against another machine on the network, but not against another local-web
# origin on the SAME machine - a malicious page in a normal browser tab can
# already reach 127.0.0.1 (that is the whole "localhost is not a security
# boundary" class of bug), and this API deletes files and runs CPU-heavy jobs.
#
# Generated once per process, never written to disk, never logged except at
# direct `python server.py` startup where there is no other way to learn it.
# Regenerates every launch by construction - there is nowhere it could be
# cached between runs.
SESSION_TOKEN = secrets.token_urlsafe(32)
TOKEN_HEADER = 'X-T-Tracer-Token'
# <img src> / <a href> requests (SVG/PNG previews, the source image) cannot
# set a custom header, so those two endpoints accept the token as a query
# param instead - everything else must use the header. app.js appends this
# param when it builds those URLs; it never appends it to a real fetch() call,
# so the token does not end up in fetch's Referer/logs for anything else.
TOKEN_QUERY_PARAM = 'tt_token'

# Settings and the app's own pick log live beside the work directory, i.e.
# under ~/.cache (or LOCALAPPDATA), NEVER inside the project. Same reason the
# work directory moved there: this repo sits in a OneDrive tree, and a file
# rewritten on every save is a sync storm.
SETTINGS_FILE = WORK.parent / 'settings.json'

# The app's picks are a SEPARATE file from _private/candidates/picks.jsonl, and
# that separation is the point rather than an implementation detail.
#
# picks.jsonl is the calibration ground truth: a reviewer at a contact sheet,
# over a corpus chosen to represent the work. This file is whatever happened to
# be dropped into the app, which is not necessarily a logo at all - an out-of-
# focus photograph is a perfectly ordinary thing for someone to try. Useful
# signal, but not the same KIND of signal, so it is not treated as one.
#
# So app rows are kept, reported and shared - but they are never merged into
# the calibration set, and /api/stats returns the two side by side rather than
# averaging them into one misleading number. The one field-tier number that IS
# trustworthy is the share of runs where nothing was usable, because that does
# not depend on the input being a logo.
APP_PICKS = WORK.parent / 'app-picks.jsonl'

# Ground truth, read-only here. Absent on an installed copy (the installer
# ships the engine but not the corpus), which is expected, not an error.
# Resolved through paths.py, which owns the location of everything private.
CALIB_PICKS = _paths.PICKS

# Uploads stream to disk in fixed chunks instead of through one bytes object.
#
# `dest.write_bytes(await f.read())` materialised the ENTIRE file in RAM before
# a single byte reached disk, and several files upload concurrently - so a
# handful of 130 MP catalogue TIFFs is that many full copies resident at once.
# On a machine whose tracing workers are already budgeted to the megabyte
# (see candidates.auto_jobs / WORKER_MB), that is the one unbounded allocation
# left in the request path. Chunking makes upload memory constant in file size.
#
# The cap REJECTS rather than truncates: a half-written PNG still decodes far
# enough to trace, and it traces into junk that reads as a pipeline bug rather
# than as a bad upload. Override with TT_MAX_UPLOAD_MB.
UPLOAD_CHUNK = 1 << 20                       # 1 MiB per read
MAX_UPLOAD_MB = max(1, int(_os.environ.get('TT_MAX_UPLOAD_MB', '64')))

RETENTION_DAYS = {'90d': 90, '30d': 30}    # 'forever' has no entry - never expires

DEFAULT_SETTINGS = {
    'dest': '',            # last folder saved to - survives a restart
    'share_stats': False,  # opt-IN. Nothing ever leaves this machine unasked.
    'role': 'user',        # 'owner' marks the reviewer's own install in shared rows
    'auto_update': False,  # item 3.5. Off by default; Windows-only regardless
                            # of this value (server enforces it, not just the UI).
    'retention': 'forever', # item 7: 'forever' | '90d' | '30d'
}


def load_settings() -> dict:
    out = dict(DEFAULT_SETTINGS)
    try:
        out.update(json.loads(SETTINGS_FILE.read_text()))
    except (OSError, json.JSONDecodeError, ValueError):
        pass                        # a corrupt settings file is not fatal
    if role := _os.environ.get('TT_ROLE'):
        out['role'] = role
    return out


def write_settings(d: dict) -> dict:
    cur = load_settings()
    cur.update({k: v for k, v in d.items() if k in DEFAULT_SETTINGS})
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(cur, indent=2))
    except OSError:
        pass
    return cur


def read_jsonl(path: Path) -> list[dict]:
    """Last row wins per image, matching how pick.py reads its own log."""
    if not path.is_file():
        return []
    by_image: dict[str, dict] = {}
    try:
        text = path.read_text()
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get('image'):
            by_image[row['image']] = row
    return list(by_image.values())


def summarise(rows: list[dict]) -> dict:
    """Aggregate a pick log the way pick.py reports it at the end of a session.

    Deliberately reports each denominator alongside its numerator. Rows written
    before 2026-09-05 carry a single `pick` and no `shippable` set, so a
    per-strategy rate computed over ALL rows would silently mix two different
    questions. Only rows that recorded a full set can answer "how often is this
    strategy shippable", and `sets` says how many those were.
    """
    judged = len(rows)
    nothing = [r for r in rows if not r.get('pick')]
    with_set = [r for r in rows if r.get('shippable')]
    ranked = [r for r in rows if r.get('rank')]

    strategies: dict[str, dict] = {}
    for name in C.STRATEGIES:
        strategies[name] = {'shippable': 0, 'starred': 0}
    for r in rows:
        for nm in (r.get('shippable') or []):
            strategies.setdefault(nm, {'shippable': 0, 'starred': 0})
            strategies[nm]['shippable'] += 1
        if pick := r.get('pick'):
            strategies.setdefault(pick, {'shippable': 0, 'starred': 0})
            strategies[pick]['starred'] += 1

    top1_ok = len([r for r in with_set
                   if 1 in (r.get('shippable_ranks') or [])])
    ships = sum(len(r['shippable']) for r in with_set)

    # A lifetime success rate is actively misleading here, and the log proves
    # it: over 2026-09-02..04 the tracer failed outright on 63% of images, and
    # over 09-05..06 on 13%. The difference is the pipeline getting better, not
    # the images getting easier. Reporting one blended 44% would understate the
    # CURRENT tracer by roughly four times, so the date breakdown ships
    # alongside the total and the UI leads with the recent window.
    by_date: dict[str, dict] = {}
    for r in rows:
        d = by_date.setdefault(str(r.get('date', 'unknown')),
                               {'judged': 0, 'nothing_usable': 0})
        d['judged'] += 1
        if not r.get('pick'):
            d['nothing_usable'] += 1

    recent = sorted(rows, key=lambda r: str(r.get('date', '')))[-30:]
    return {
        'recent': {
            'window': len(recent),
            'judged': len(recent),
            'nothing_usable': len([r for r in recent if not r.get('pick')]),
        },
        'by_date': dict(sorted(by_date.items())),
        'judged': judged,
        'nothing_usable': len(nothing),
        'usable': judged - len(nothing),
        'sets': len(with_set),
        'ranked': len(ranked),
        'top1_agreed': len([r for r in ranked if r.get('rank') == 1]),
        'top1_shippable': top1_ok,
        'avg_shippable': round(ships / len(with_set), 1) if with_set else None,
        'strategies': strategies,
    }


def cli_defaults(**over) -> Namespace:
    """Exactly the argparse defaults from candidates.py, overridable.

    Kept in one place so a change to the CLI's defaults cannot silently leave
    the app tracing with stale settings.
    """
    base = dict(only=None, strategies=None, invert=False, scale=6,
                smoothing=3.0, tol=0.6, close=2, open=0, min_area_frac=0.0004,
                height_mm=32.0, raster_ss=2, jobs=1, bg=False, max_dim=2048,
                work_dim=C.WORK_DIM_DEFAULT)
    base.update(over)
    return Namespace(**base)


# --------------------------------------------------------------------------
# Job state, persisted.
#
# Originally in-memory, on the reasoning that a closed app has no jobs to
# resume. the reviewer asked for history and gave the reason that overturns it: "in
# case the one i download doesnt work, i can grab the another one from a
# previous run." The traced SVGs already survive on disk in .work/; only the
# index of them was being thrown away on exit.
#
# A `job.json` per job folder is the whole mechanism. Rebuilt by scanning
# .work/ at startup, so the folder itself stays the source of truth and a
# deleted folder simply disappears from history.
# --------------------------------------------------------------------------
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()

# Only ONE tracing batch runs at a time, process-wide.
#
# Every submitted job used to start its own daemon thread, and each thread sized
# its own ProcessPoolExecutor from free memory at the moment it started. Two
# submissions seconds apart therefore both measured a quiet machine and both
# committed to a full pool -- individually resource-aware, collectively over
# budget. Someone dropping 15 images in, watching nothing happen, and dropping
# them in again is the ordinary way to trigger it, not an edge case.
#
# candidates._acquire_batch_lock() covers the same failure ACROSS processes
# (a CLI run alongside the app). This one covers it within this process, and
# gives the UI something honest to show while a job waits.
BATCH = threading.Lock()


def _job_file(job_id: str) -> Path:
    return WORK / job_id / 'job.json'


def save_job(job: dict) -> None:
    try:
        _job_file(job['id']).write_text(json.dumps(job))
    except OSError:
        pass                      # history is a convenience, never fatal


def load_jobs() -> None:
    if not WORK.is_dir():
        return
    for d in sorted(WORK.iterdir()):
        f = d / 'job.json'
        if not f.is_file():
            continue
        try:
            job = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue                     # corrupt/truncated job.json: skip it
        if not isinstance(job, dict):
            continue                     # valid JSON, wrong shape - same idea
        # Rebuild the paths from where the folder actually IS rather than
        # trusting what was written. job.json used to carry absolute paths, so
        # moving the work directory (out of OneDrive, for one) silently
        # orphaned every past job - history came back empty with the files
        # still sitting right there.
        job['src_dir'] = str(d / 'in')
        job['out_dir'] = str(d / 'out')
        job['id'] = d.name
        # No thread survives a restart to finish either of these, so both
        # would otherwise sit in History forever looking like they are still
        # running - 'tracing' was already fixed; 'queued' (killed while
        # waiting on the batch lock, before it ever started) is the same
        # failure and was missed - error-path audit, item 3.55.
        if job.get('status') in ('tracing', 'queued'):
            job['status'] = 'interrupted'
        JOBS[job['id']] = job


def _trace_one(img_path: str, out_root: str) -> dict:
    """Worker body. Top level so ProcessPoolExecutor can pickle it."""
    import candidates as _C
    return _C.run_one(Path(img_path), Path(out_root), cli_defaults())


def _run_job(job_id: str) -> None:
    job = JOBS[job_id]
    # Queued is a real, reportable state - not a job that looks hung.
    if not BATCH.acquire(blocking=False):
        job['status'] = 'queued'
        save_job(job)
        BATCH.acquire()
    try:
        job['status'] = 'tracing'
        save_job(job)
        _run_job_inner(job_id)
    finally:
        BATCH.release()


def _run_job_inner(job_id: str) -> None:
    job = JOBS[job_id]
    src_dir, out_dir = Path(job['src_dir']), Path(job['out_dir'])
    images = sorted(p for p in src_dir.iterdir()
                    if p.suffix.lower() in IMAGE_EXTS)
    workers = C.auto_jobs(len(images), verbose=False)
    try:
        if workers > 1 and len(images) > 1:
            with ProcessPoolExecutor(max_workers=workers,
                                     initializer=C._worker_init) as ex:
                futs = {ex.submit(_trace_one, str(p), str(out_dir)): p
                        for p in images}
                for f in futs:
                    pass
                for f in list(futs):
                    try:
                        f.result()
                    except Exception as e:
                        job['errors'].append(f'{futs[f].name}: {e}')
                    with LOCK:
                        job['done'] += 1
        else:
            for p in images:
                try:
                    _trace_one(str(p), str(out_dir))
                except Exception as e:
                    job['errors'].append(f'{p.name}: {e}')
                with LOCK:
                    job['done'] += 1
        job['status'] = 'ready'
    except Exception as e:                                # noqa: BLE001
        job['status'] = 'failed'
        job['errors'].append(str(e))
    save_job(job)


app = FastAPI(title='T-Tracer')


@app.middleware('http')
async def _require_session_token(request: Request, call_next):
    """Every /api/ request needs the launch token; the static frontend does not.

    The static mount (index.html, app.js, app.css, icons) is intentionally
    left open: the page itself has to load before it can learn the token, and
    it carries no capability - reading app.js is not the same risk as being
    able to call POST /api/jobs or DELETE /api/jobs/{id}.
    """
    if request.url.path.startswith('/api/'):
        token = (request.headers.get(TOKEN_HEADER)
                 or request.query_params.get(TOKEN_QUERY_PARAM))
        if token != SESSION_TOKEN:
            return JSONResponse({'detail': 'missing or invalid session token'},
                                status_code=401)
    return await call_next(request)


async def _save_upload(f: UploadFile, dest: Path, max_bytes: int) -> str:
    """Stream one upload to `dest`. Returns 'ok', 'oversized' or 'error'.

    The partial file is removed on ANY exit that is not a clean complete write,
    including a disconnect mid-upload. A truncated image in the job folder would
    be picked up by the tracer as though it were a real input.

    'oversized' and 'error' used to be the same outcome (both just `False`),
    which meant a disk-full write (error-path audit, item 3.55: simulated with
    OSError(ENOSPC)) was reported to the user as "over the upload size limit" -
    true of nothing on their end and actively misleading about what to fix.
    """
    written = 0
    try:
        with dest.open('wb') as out:
            while chunk := await f.read(UPLOAD_CHUNK):
                written += len(chunk)
                if written > max_bytes:
                    dest.unlink(missing_ok=True)
                    return 'oversized'
                out.write(chunk)
    except Exception:                                      # noqa: BLE001
        dest.unlink(missing_ok=True)
        return 'error'
    return 'ok'


@app.post('/api/jobs')
async def create_job(files: list[UploadFile]):
    accepted, rejected, oversized, failed = [], [], [], []
    job_id = uuid.uuid4().hex[:12]
    src = WORK / job_id / 'in'
    out = WORK / job_id / 'out'
    src.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    for f in files:
        name = Path(f.filename or 'unnamed').name
        if Path(name).suffix.lower() not in IMAGE_EXTS:
            rejected.append(name)
            continue
        # Stem collisions overwrite each other downstream (a known pipeline
        # bug: "DC Insignia.jpg" and "DC Insignia.png" share an output folder).
        # Disambiguate on the way in rather than losing one silently.
        stem, suf, n = Path(name).stem, Path(name).suffix, 1
        while (src / f'{stem}{suf}').exists():
            n += 1
            stem = f'{Path(name).stem} ({n})'
        dest = src / f'{stem}{suf}'
        result = await _save_upload(f, dest, MAX_UPLOAD_MB * 1024 * 1024)
        if result == 'oversized':
            oversized.append(name)
            continue
        if result == 'error':
            failed.append(name)
            continue
        accepted.append(dest.name)

    if not accepted:
        shutil.rmtree(WORK / job_id, ignore_errors=True)
        detail = []
        if oversized:
            detail.append(f'over the {MAX_UPLOAD_MB} MB limit: '
                          + ', '.join(oversized))
        if rejected:
            detail.append('not an image: ' + ', '.join(rejected))
        if failed:
            # Distinct from "oversized" on purpose (error-path audit, item
            # 3.55): telling someone their file is too big when the real
            # problem is the destination disk sends them fixing the wrong thing.
            detail.append('could not be saved (disk full or unwritable): '
                          + ', '.join(failed))
        raise HTTPException(400, 'No usable images. ' + '; '.join(detail))

    from datetime import datetime
    JOBS[job_id] = {'id': job_id, 'status': 'queued', 'done': 0,
                    'total': len(accepted), 'src_dir': str(src),
                    'out_dir': str(out), 'errors': [], 'rejected': rejected,
                    'created': datetime.now().isoformat(timespec='seconds'),
                    'names': accepted, 'oversized': oversized, 'failed': failed}
    save_job(JOBS[job_id])
    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()
    return {'job_id': job_id, 'accepted': accepted, 'rejected': rejected,
            'oversized': oversized, 'failed': failed}


@app.get('/api/jobs/{job_id}')
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, 'unknown job')
    # .get() with defaults, not job[k]: a job.json truncated by a kill mid-save
    # (or mid-write to disk-full) can be valid JSON missing some of these keys
    # entirely, and a plain KeyError here used to turn that into a 500 on
    # every poll of that job - error-path audit, item 3.55.
    return {'id': job.get('id', job_id), 'status': job.get('status', 'unknown'),
            'done': job.get('done', 0), 'total': job.get('total', 0),
            'errors': job.get('errors', []), 'rejected': job.get('rejected', [])}


@app.get('/api/jobs/{job_id}/results')
def job_results(job_id: str):
    """One entry per image, candidates ranked by the shared ordering.

    Uses `rank_candidates` so the app, the contact sheets and pick.py all number
    candidates identically - they diverged once already and it silently recorded
    the wrong strategy.
    """
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, 'unknown job')
    out = Path(job['out_dir'])
    images = []
    for d in sorted(p for p in out.iterdir()
                    if p.is_dir() and not p.name.startswith('_')):
        mp = d / 'metrics.json'
        if not mp.exists():
            continue
        # A trace killed mid-write (process killed, or disk full) can leave
        # metrics.json truncated - valid on disk, not valid JSON. That used to
        # be an unhandled JSONDecodeError, turning ONE bad image into a 500 for
        # the whole job's results, including every OTHER image that traced
        # fine. Skip just this image instead, same as the missing-file case
        # right above it - error-path audit, item 3.55.
        try:
            metrics = json.loads(mp.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        cands = []
        for i, (name, sc) in enumerate(C.rank_candidates(metrics), 1):
            if not (d / f'{name}.png').exists():
                continue
            cands.append({
                'n': i, 'name': name,
                'score': round(sc.get('overall', 0), 1),
                'headline': sc.get('headline', ''),
                'nodes': sc.get('nodes'), 'paths': sc.get('paths'),
                'preview': f'/api/jobs/{job_id}/preview/{d.name}/{name}.png',
            })
        src = next((p for p in Path(job['src_dir']).iterdir()
                    if p.stem == d.name), None)
        images.append({
            'stem': d.name,
            'source': f'/api/jobs/{job_id}/source/{src.name}' if src else None,
            'candidates': cands,
        })
    return {'images': images}


@app.get('/api/jobs/{job_id}/preview/{stem}/{fname}')
def preview(job_id: str, stem: str, fname: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, 'unknown job')
    p = (Path(job['out_dir']) / stem / Path(fname).name).resolve()
    if not p.is_file() or Path(job['out_dir']).resolve() not in p.parents:
        raise HTTPException(404, 'not found')
    return FileResponse(p)


@app.get('/api/jobs/{job_id}/source/{fname}')
def source(job_id: str, fname: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, 'unknown job')
    p = (Path(job['src_dir']) / Path(fname).name).resolve()
    if not p.is_file():
        raise HTTPException(404, 'not found')
    return FileResponse(p)


class SaveReq(BaseModel):
    dest: str
    picks: dict[str, str]          # image stem -> favourite strategy name


@app.post('/api/jobs/{job_id}/save')
def save(job_id: str, req: SaveReq):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, 'unknown job')
    dest = Path(req.dest).expanduser()
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise HTTPException(400, f'Cannot write to {dest}: {e}') from e

    written, missing = [], []
    for stem, strategy in req.picks.items():
        src = Path(job['out_dir']) / stem / f'{strategy}.svg'
        if not src.is_file():
            missing.append(stem)
            continue
        target = dest / f'{stem}.svg'
        n = 1
        while target.exists():
            n += 1
            target = dest / f'{stem} ({n}).svg'
        shutil.copy2(src, target)
        written.append(target.name)
    # Remember where it went, so the next launch opens on the same folder
    # (item 7). Written here as well as on edit, so a folder typed straight
    # into the box and saved to is remembered even if the field never fired
    # a change event.
    write_settings({'dest': str(dest)})
    return {'written': written, 'missing': missing, 'dest': str(dest)}


@app.get('/api/history')
def history():
    """Every past job still on disk, newest first."""
    out = []
    for job in JOBS.values():
        if not Path(job.get('out_dir', '')).is_dir():
            continue
        out.append({'id': job['id'], 'created': job.get('created', ''),
                    'total': job.get('total', 0),
                    'status': job.get('status', 'ready'),
                    'names': job.get('names', [])})
    out.sort(key=lambda j: j['created'], reverse=True)
    return {'jobs': out}


@app.delete('/api/jobs/{job_id}')
def forget_job(job_id: str):
    job = JOBS.pop(job_id, None)
    if job:
        shutil.rmtree(Path(job['out_dir']).parent, ignore_errors=True)
    return {'ok': True}


# --------------------------------------------------------------------------
# Item 7: history/storage controls.
#
# Everything a job writes lives under WORK/<job_id>/ - settings.json and
# app-picks.jsonl sit one level up, at WORK.parent, and the calibration
# picks.jsonl (CALIB_PICKS) lives entirely outside WORK, under _private/. A
# cleanup that only ever does shutil.rmtree(WORK / job_id) therefore CANNOT
# reach any of those, structurally, not by convention - "cleanup must never
# destroy settings/picks/field statistics" holds even if this code has a bug
# in which job_ids it picks.
# --------------------------------------------------------------------------

# A job in either of these states has a thread (or a restart-recovery path)
# that still owns its folder. Cleanup must never touch it out from under
# that - "active/queued jobs are never removed by cleanup".
ACTIVE_STATUSES = {'queued', 'tracing'}


def select_jobs_to_clean(jobs: dict[str, dict], cutoff: str | None = None) -> list[str]:
    """Which job ids a cleanup pass may delete. Pure - no disk I/O, easy to test.

    `cutoff`: an ISO 'created' timestamp. None means "every finished job,
    regardless of age" (the manual Clear button); given, only jobs created
    strictly before it qualify (the automatic retention sweep).
    """
    return [job_id for job_id, job in jobs.items()
            if job.get('status') not in ACTIVE_STATUSES
            and (cutoff is None or str(job.get('created', '')) < cutoff)]


def _dir_size(path: Path) -> int:
    if not path.is_dir():
        return 0
    total = 0
    for p in path.rglob('*'):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass                      # a file that vanished mid-scan is not fatal
    return total


def _clean_jobs(job_ids: list[str]) -> int:
    """Delete the given job folders. Returns bytes freed."""
    freed = 0
    for job_id in job_ids:
        freed += _dir_size(WORK / job_id)
        shutil.rmtree(WORK / job_id, ignore_errors=True)
        JOBS.pop(job_id, None)
    return freed


def _apply_retention() -> None:
    """Automatic cleanup by the configured retention window, run once at launch.

    After load_jobs() so JOBS reflects what is actually on disk, and before
    the server starts accepting requests, so an install nobody opens Settings
    on still shrinks rather than growing forever at 'forever' by default doing
    nothing until someone changes it.
    """
    days = RETENTION_DAYS.get(load_settings().get('retention', 'forever'))
    if not days:
        return                            # 'forever' (or an unknown value): no-op
    from datetime import datetime, timedelta
    cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec='seconds')
    _clean_jobs(select_jobs_to_clean(JOBS, cutoff))


@app.get('/api/storage')
def storage():
    """Disk used by everything History can show - the number the Settings
    tab's "X used" line reports, and what Clear/retention actually reclaim."""
    return {'bytes': _dir_size(WORK), 'jobs': len(JOBS)}


@app.post('/api/history/clear')
def clear_history():
    """The manual 'Clear generated history' button: every FINISHED job, now,
    regardless of age. Active/queued jobs are skipped - see select_jobs_to_clean."""
    ids = select_jobs_to_clean(JOBS)
    freed = _clean_jobs(ids)
    return {'cleared': len(ids), 'bytes_freed': freed}


@app.get('/api/pick-folder')
def pick_folder():
    """Native folder chooser, server-side.

    The pywebview bridge only exists when pywebview is the window backend, and
    that is no longer the default on either Linux (no GTK/Qt renderer) or
    Windows (Chrome/Edge --app mode is now first choice there - see main.py).
    Doing it here means Browse works the same regardless of which window
    backend actually rendered the page.
    tkinter would be the obvious cross-platform choice and is deliberately not
    used on Linux: it is present but broken on that machine (missing
    libtk8.6), which is exactly the kind of thing that turns a button into a
    crash. On Windows it ships with the standard python.org installer, so it
    is tried there; PowerShell's own folder dialog is the fallback for a
    Python build that dropped it.
    """
    import shutil
    import subprocess

    if sys.platform == 'win32':
        try:
            import tkinter
            from tkinter import filedialog
            root = tkinter.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            path = filedialog.askdirectory()
            root.destroy()
            return {'path': path or None}
        except Exception:                                  # noqa: BLE001
            pass
        ps_script = (
            "Add-Type -AssemblyName System.Windows.Forms | Out-Null;"
            "$d = New-Object System.Windows.Forms.FolderBrowserDialog;"
            "if ($d.ShowDialog() -eq 'OK') { Write-Output $d.SelectedPath }"
        )
        try:
            r = subprocess.run(
                ['powershell', '-NoProfile', '-Command', ps_script],
                capture_output=True, text=True, timeout=180,
            )
        except Exception:                                  # noqa: BLE001
            return {'path': None, 'unavailable': True}
        return {'path': r.stdout.strip() or None}

    for cmd in (['kdialog', '--getexistingdirectory', str(Path.home())],
                ['zenity', '--file-selection', '--directory']):
        if not shutil.which(cmd[0]):
            continue
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except Exception:                                  # noqa: BLE001
            continue
        if r.returncode == 0 and r.stdout.strip():
            return {'path': r.stdout.strip()}
        return {'path': None}                              # user cancelled
    return {'path': None, 'unavailable': True}


class Settings(BaseModel):
    dest: str | None = None
    share_stats: bool | None = None
    role: str | None = None
    auto_update: bool | None = None
    retention: str | None = None


@app.get('/api/settings')
def get_settings():
    # 'platform' is computed, never persisted - it rides along so the
    # Settings tab can hide/disable the auto-update toggle on anything that
    # is not Windows without a second round trip. It is dropped again by
    # write_settings() (only DEFAULT_SETTINGS keys survive a PUT).
    return {**load_settings(), 'platform': sys.platform}


@app.put('/api/settings')
def put_settings(req: Settings):
    """Persisted server-side, and that is the fix rather than a preference.

    The destination folder was already being kept in localStorage, and it was
    still lost on every restart, because main.py binds a FREE PORT each launch
    - so the page's origin changes from run to run and localStorage comes back
    empty against the new one. Storing it next to the work directory makes it
    independent of whichever port the app happened to get.
    """
    patch = req.model_dump(exclude_none=True)
    if 'retention' in patch and patch['retention'] not in ('forever', *RETENTION_DAYS):
        raise HTTPException(400, "retention must be 'forever', '90d' or '30d'")
    return write_settings(patch)


class JudgeRow(BaseModel):
    stem: str
    pick: str | None = None
    shippable: list[str] = []
    shippable_ranks: list[int] = []
    of: int = 0
    failed: bool = False


@app.post('/api/jobs/{job_id}/judge')
def judge(job_id: str, rows: list[JudgeRow]):
    """Record what the user actually decided, in pick.py's row shape.

    Called on Save and whenever an image is marked unusable - NOT only on save,
    because an image where nothing worked is never saved and that is precisely
    the row worth having. Append-only; the last row for an image wins, so a
    changed mind just appends again.
    """
    role = load_settings().get('role', 'user')
    from datetime import date
    try:
        APP_PICKS.parent.mkdir(parents=True, exist_ok=True)
        with APP_PICKS.open('a') as f:
            for r in rows:
                f.write(json.dumps({
                    'date': date.today().isoformat(),
                    'image': r.stem,
                    'pick': None if r.failed else r.pick,
                    'shippable': [] if r.failed else r.shippable,
                    'shippable_ranks': [] if r.failed else r.shippable_ranks,
                    'rank': (r.shippable_ranks[0]
                             if r.shippable_ranks and not r.failed else None),
                    'of': r.of,
                    'source': role,
                    'origin': 'app',
                    'job': job_id,
                }) + '\n')
    except OSError:
        return {'ok': False}            # never let logging break a save
    return {'ok': True, 'recorded': len(rows)}


@app.get('/api/stats')
def stats():
    """Two populations, reported separately and never averaged together.

    `calibration` is picks.jsonl - the reviewer at a contact sheet over a curated
    corpus. `app` is whatever was dropped into this install. Merging them would
    let a stranger's blurry phone photo move a number that is supposed to
    describe how the tracer does on logos.
    """
    return {
        'version': APP_VERSION,
        'calibration': {
            'available': CALIB_PICKS.is_file(),
            **summarise(read_jsonl(CALIB_PICKS)),
        },
        'app': summarise(read_jsonl(APP_PICKS)),
        'strategy_order': list(C.STRATEGIES),
    }


@app.get('/api/stats/report')
def stats_report():
    """The opt-in share payload. Aggregates only - built by exclusion.

    No filenames, no image data, no paths, no notes, no dates, no machine
    identifiers. Only counts, plus the platform and version needed to read them.
    The user sees this exact JSON before anything is sent, and the app never
    transmits it - it is copied into a GitHub issue by hand.
    """
    cfg = load_settings()
    if not cfg.get('share_stats'):
        raise HTTPException(403, 'Stats sharing is off. Turn it on first.')
    a = summarise(read_jsonl(APP_PICKS))
    return {
        'report': 't-tracer usage',
        'version': APP_VERSION,
        'platform': sys.platform,
        'role': cfg.get('role', 'user'),
        'images_judged': a['judged'],
        'nothing_usable': a['nothing_usable'],
        'rows_with_full_set': a['sets'],
        'avg_shippable_per_image': a['avg_shippable'],
        'top_scored_was_shippable': a['top1_shippable'],
        'per_strategy': a['strategies'],
    }


# --------------------------------------------------------------------------
# Item 3.5: auto-update. Shared by the reporting-only check above and the
# actual downloader below, so a change to how "newer" is decided cannot
# silently diverge between what the UI shows and what the updater acts on.
# --------------------------------------------------------------------------

def _version_parts(v: str) -> list[int]:
    out = []
    for chunk in v.split('.'):
        digits = ''.join(c for c in chunk if c.isdigit())
        out.append(int(digits) if digits else 0)
    return out


def _fetch_latest_release() -> dict | None:
    """The full GitHub release payload (assets included), or None on any
    failure - offline, rate limited, no releases yet. Never raises."""
    import urllib.request
    url = f'https://api.github.com/repos/{REPO}/releases/latest'
    try:
        req = urllib.request.Request(
            url, headers={'Accept': 'application/vnd.github+json',
                          'User-Agent': f't-tracer/{APP_VERSION}'})
        with urllib.request.urlopen(req, timeout=4) as r:
            return json.loads(r.read().decode())
    except Exception:                                      # noqa: BLE001
        return None


@app.get('/api/update-check')
def update_check():
    """Ask GitHub for the newest release. Fails silently and never blocks.

    Reports only; it does not download or install anything. Offline, rate
    limited, or a repo with no releases yet all return the same quiet
    'no update' rather than an error the user has to dismiss.
    """
    data = _fetch_latest_release()
    if data is None:
        return {'current': APP_VERSION, 'checked': False}

    tag = str(data.get('tag_name') or '').lstrip('vV')
    newer = bool(tag) and _version_parts(tag) > _version_parts(APP_VERSION)
    return {'current': APP_VERSION, 'checked': True, 'latest': tag or None,
            'update': newer, 'url': data.get('html_url'),
            'notes': (data.get('body') or '')[:400]}


# --------------------------------------------------------------------------
# The install half of item 3.5. "Do item 6 (SHA-256 verification) first - an
# auto-updater that runs an unverified download is strictly worse than no
# auto-updater" - so this never runs the downloaded .exe without a checksum
# that matches, ships with build-installer.yml computing and publishing that
# checksum as its own release asset (T-Tracer-Setup.exe.sha256).
# --------------------------------------------------------------------------

def _release_assets(data: dict) -> dict[str, str]:
    """name -> browser_download_url for every asset on a release payload."""
    return {a['name']: a['browser_download_url']
            for a in (data.get('assets') or []) if a.get('name') and a.get('browser_download_url')}


def verify_sha256(path: Path, expected: str) -> bool:
    """True iff `path` hashes to `expected`. Accepts either a bare hex digest
    (Get-FileHash's own format, matching app/install.ps1's PyInstallerSha256)
    or a `sha256sum`-style 'hex  filename' line - only the first token is used."""
    import hashlib
    digest = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    token = expected.strip().split()[0] if expected.strip() else ''
    return bool(token) and digest.hexdigest().lower() == token.lower()


def _download(url: str, dest: Path) -> None:
    """Stream a URL to disk. Raises on any failure - the caller decides what
    'could not update' means; this function never swallows an error, because
    swallowing it here is how a truncated .exe would end up looking verified."""
    import urllib.request
    req = urllib.request.Request(url, headers={'User-Agent': f't-tracer/{APP_VERSION}'})
    with urllib.request.urlopen(req, timeout=120) as r, dest.open('wb') as out:
        while chunk := r.read(UPLOAD_CHUNK):
            out.write(chunk)


# Read by GET /api/update-status so the frontend can surface a checksum
# failure or a missing release asset - the one case that is NOT silent, per
# item 3.5's own "mismatch or missing checksum => do not run, surface a clear
# error". Never written to disk; a restart clears it, same as SESSION_TOKEN.
UPDATE_STATE: dict = {'status': 'idle', 'detail': ''}


def _auto_update_once() -> None:
    """Runs once per launch, in a background thread - see serve(). Every
    return before the download is a SILENT no-op by design (item 3.5: offline
    / no update / declined must not be an error the user has to dismiss)."""
    global UPDATE_STATE
    if sys.platform != 'win32':
        return                            # never attempt an install off Windows,
                                           # regardless of what settings.json says
    if not load_settings().get('auto_update'):
        return
    data = _fetch_latest_release()
    if data is None:
        return                            # offline / rate limited: silent
    tag = str(data.get('tag_name') or '').lstrip('vV')
    if not tag or _version_parts(tag) <= _version_parts(APP_VERSION):
        return                            # no update available: silent
    assets = _release_assets(data)
    exe_name = next((n for n in assets if n.lower().endswith('.exe')), None)
    sha_name = next((n for n in assets if n.lower().endswith('.sha256')), None)
    if not exe_name or not sha_name:
        UPDATE_STATE = {'status': 'error',
                        'detail': f'Release {tag} is missing the installer or its '
                                  'checksum file. Not downloading.'}
        return
    tmp = WORK.parent / 'update'
    try:
        tmp.mkdir(parents=True, exist_ok=True)
        exe_path, sha_path = tmp / exe_name, tmp / sha_name
        _download(assets[exe_name], exe_path)
        _download(assets[sha_name], sha_path)
        if not verify_sha256(exe_path, sha_path.read_text()):
            exe_path.unlink(missing_ok=True)
            UPDATE_STATE = {'status': 'error',
                            'detail': f'{exe_name} failed its checksum check after '
                                      'download. Not run - try again later.'}
            return
        import subprocess
        subprocess.Popen([str(exe_path)], close_fds=True)
        UPDATE_STATE = {'status': 'installing', 'detail': f'Installing {tag}…'}
    except Exception as e:                                 # noqa: BLE001
        UPDATE_STATE = {'status': 'error', 'detail': f'Auto-update failed: {e}'}
        return
    # The running app and the installer it just launched cannot both hold the
    # install directory - item 3.5: "run it, exit the app". os._exit() rather
    # than a normal return: uvicorn is serving on another thread and would
    # otherwise keep the process (and this window) alive underneath the
    # installer.
    _os._exit(0)                                           # noqa: SLF001


@app.get('/api/update-status')
def update_status():
    return UPDATE_STATE


@app.get('/api/health')
def health():
    return {'ok': True, 'strategies': list(C.STRATEGIES)}


app.mount('/', StaticFiles(directory=str(STATIC), html=True), name='static')


def migrate_legacy_dir() -> None:
    """Carry ~/.cache/stickbird-vectorizer/ over to ~/.cache/t-tracer/.

    The rename to T-Tracer moved the state directory, and simply leaving the
    old one behind would empty History and throw away the annotator cache -
    exactly the failure that already happened once when job.json stored
    absolute paths. Moves the whole parent, so `work/`, `settings.json`,
    `app-picks.jsonl` and `lineart-cache/` all come with it. Runs once: after
    the move the old path no longer exists.
    """
    legacy = WORK.parent.parent / 'stickbird-vectorizer'
    if legacy.is_dir() and not WORK.parent.exists():
        try:
            legacy.rename(WORK.parent)
        except OSError:
            pass                      # a failed migration must not block start


def serve(host='127.0.0.1', port=8765):
    import uvicorn
    migrate_legacy_dir()
    WORK.mkdir(parents=True, exist_ok=True)
    load_jobs()
    _apply_retention()
    # main.py reads SESSION_TOKEN off the module directly and puts it in the
    # window URL, so the normal launch path never needs this. It is printed
    # here only for `python server.py` directly - the "Try: python server.py"
    # fallback main.py itself suggests - where there is no other way to learn
    # the token the API now requires.
    print(f'T-Tracer: session token for direct API use: {SESSION_TOKEN}')
    # Backgrounded so a slow/offline GitHub check never delays the window
    # opening - main.py is already polling wait_for(port) for exactly that.
    threading.Thread(target=_auto_update_once, daemon=True).start()
    uvicorn.run(app, host=host, port=port, log_level='warning')


if __name__ == '__main__':
    serve()
