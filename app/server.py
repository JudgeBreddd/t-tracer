#!/usr/bin/env python3
"""T-Tracer - local backend.

Wraps the existing pipeline rather than reimplementing any of it. `run_one` in
`_scripts/candidates.py` is imported directly and called with exactly the CLI's
default arguments, so the app and the command line produce byte-identical SVGs.
That matters: `_scripts/candidates/picks.jsonl` is the project's calibration
ground truth, and it would stop meaning anything if the app traced differently
from the tool the picks were recorded with.

Serves a small JSON API on 127.0.0.1 to a static frontend. No auth and no
external binding on purpose - this is a single-user desktop app, and the port is
bound to loopback only.
"""
from __future__ import annotations

import json
import shutil
import sys
import threading
import uuid
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

APP_DIR = Path(__file__).resolve().parent
PROJECT = APP_DIR.parent
SCRIPTS = PROJECT / '_scripts'
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

sys.path.insert(0, str(SCRIPTS))
import candidates as C                                   # noqa: E402

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}

APP_VERSION = '0.1.0'
# Update checks ask GitHub about this repo. Overridable so a fork does not
# report someone else's releases as its own updates.
import os as _os
REPO = _os.environ.get('TT_REPO', 'JudgeBreddd/t-tracer')

# Settings and the app's own pick log live beside the work directory, i.e.
# under ~/.cache (or LOCALAPPDATA), NEVER inside the project. Same reason the
# work directory moved there: this repo sits in a OneDrive tree, and a file
# rewritten on every save is a sync storm.
SETTINGS_FILE = WORK.parent / 'settings.json'

# The app's picks are a SEPARATE file from _scripts/candidates/picks.jsonl, and
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
# ships the scripts but not the corpus), which is expected, not an error.
CALIB_PICKS = SCRIPTS / 'candidates' / 'picks.jsonl'

DEFAULT_SETTINGS = {
    'dest': '',            # last folder saved to - survives a restart
    'share_stats': False,  # opt-IN. Nothing ever leaves this machine unasked.
    'role': 'user',        # 'owner' marks the reviewer's own install in shared rows
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
                height_mm=32.0, raster_ss=2, jobs=1, bg=False, max_dim=2048)
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
            continue
        # Rebuild the paths from where the folder actually IS rather than
        # trusting what was written. job.json used to carry absolute paths, so
        # moving the work directory (out of OneDrive, for one) silently
        # orphaned every past job - history came back empty with the files
        # still sitting right there.
        job['src_dir'] = str(d / 'in')
        job['out_dir'] = str(d / 'out')
        job['id'] = d.name
        # A job interrupted by a quit would otherwise sit at "tracing" forever.
        if job.get('status') == 'tracing':
            job['status'] = 'interrupted'
        JOBS[job['id']] = job


def _trace_one(img_path: str, out_root: str) -> dict:
    """Worker body. Top level so ProcessPoolExecutor can pickle it."""
    import candidates as _C
    return _C.run_one(Path(img_path), Path(out_root), cli_defaults())


def _run_job(job_id: str) -> None:
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


@app.post('/api/jobs')
async def create_job(files: list[UploadFile]):
    accepted, rejected = [], []
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
        dest.write_bytes(await f.read())
        accepted.append(dest.name)

    if not accepted:
        shutil.rmtree(WORK / job_id, ignore_errors=True)
        raise HTTPException(400, f'No usable images. Rejected: {rejected}')

    from datetime import datetime
    JOBS[job_id] = {'id': job_id, 'status': 'tracing', 'done': 0,
                    'total': len(accepted), 'src_dir': str(src),
                    'out_dir': str(out), 'errors': [], 'rejected': rejected,
                    'created': datetime.now().isoformat(timespec='seconds'),
                    'names': accepted}
    save_job(JOBS[job_id])
    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()
    return {'job_id': job_id, 'accepted': accepted, 'rejected': rejected}


@app.get('/api/jobs/{job_id}')
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, 'unknown job')
    return {k: job[k] for k in
            ('id', 'status', 'done', 'total', 'errors', 'rejected')}


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
        metrics = json.loads(mp.read_text())
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


@app.get('/api/settings')
def get_settings():
    return load_settings()


@app.put('/api/settings')
def put_settings(req: Settings):
    """Persisted server-side, and that is the fix rather than a preference.

    The destination folder was already being kept in localStorage, and it was
    still lost on every restart, because main.py binds a FREE PORT each launch
    - so the page's origin changes from run to run and localStorage comes back
    empty against the new one. Storing it next to the work directory makes it
    independent of whichever port the app happened to get.
    """
    return write_settings(req.model_dump(exclude_none=True))


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


@app.get('/api/update-check')
def update_check():
    """Ask GitHub for the newest release. Fails silently and never blocks.

    Reports only; it does not download or install anything. Offline, rate
    limited, or a repo with no releases yet all return the same quiet
    'no update' rather than an error the user has to dismiss.
    """
    import urllib.request
    url = f'https://api.github.com/repos/{REPO}/releases/latest'
    try:
        req = urllib.request.Request(
            url, headers={'Accept': 'application/vnd.github+json',
                          'User-Agent': f't-tracer/{APP_VERSION}'})
        with urllib.request.urlopen(req, timeout=4) as r:
            data = json.loads(r.read().decode())
    except Exception:                                      # noqa: BLE001
        return {'current': APP_VERSION, 'checked': False}

    tag = str(data.get('tag_name') or '').lstrip('vV')

    def parts(v):
        out = []
        for chunk in v.split('.'):
            digits = ''.join(c for c in chunk if c.isdigit())
            out.append(int(digits) if digits else 0)
        return out

    newer = bool(tag) and parts(tag) > parts(APP_VERSION)
    return {'current': APP_VERSION, 'checked': True, 'latest': tag or None,
            'update': newer, 'url': data.get('html_url'),
            'notes': (data.get('body') or '')[:400]}


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
    uvicorn.run(app, host=host, port=port, log_level='warning')


if __name__ == '__main__':
    serve()
