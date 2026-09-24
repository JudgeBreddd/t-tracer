"""Shared fixtures for the T-Tracer test suite.

Tests exercise real logic in app/server.py against a throwaway work directory
- never `~/.cache/t-tracer/`, and never a directory under OneDrive. server.py
reads TT_WORK_DIR once, at import time (`WORK = _work_dir()`), so it has to be
set before the module is first imported anywhere in the test session.
"""
import os
import sys
import tempfile
from pathlib import Path

_TMP_WORK = Path(tempfile.mkdtemp(prefix='t-tracer-tests-'))
os.environ.setdefault('TT_WORK_DIR', str(_TMP_WORK / 'work'))

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'app'))
sys.path.insert(0, str(ROOT / '_engine'))

import pytest

import server as server_module  # noqa: E402  (path must be set up first)


@pytest.fixture
def server(tmp_path, monkeypatch):
    """The server module, pointed at a fresh, empty work directory per test.

    Reassigning `server.WORK` (rather than only the env var) is what actually
    isolates tests from each other and from a real install - every function
    under test reads the module-level `WORK`/`SETTINGS_FILE`/`APP_PICKS`
    constants directly, not a re-read of the environment.
    """
    work = tmp_path / 'work'
    work.mkdir()
    monkeypatch.setattr(server_module, 'WORK', work)
    monkeypatch.setattr(server_module, 'SETTINGS_FILE', work.parent / 'settings.json')
    monkeypatch.setattr(server_module, 'APP_PICKS', work.parent / 'app-picks.jsonl')
    monkeypatch.setattr(server_module, 'JOBS', {})
    monkeypatch.setattr(server_module, 'REFINE_STATES', server_module.OrderedDict())
    monkeypatch.setattr(server_module, 'REFINE_SNAPSHOTS', server_module.OrderedDict())
    return server_module
