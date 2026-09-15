"""Error-path audit (3.55): disk-full writes, a killed process, corrupt state.

Each test exercises the actual fix in app/server.py, not a mock of it - the
goal is proving the failure used to break something and now does not.
"""
import asyncio
import json

from fastapi.testclient import TestClient


def run(coro):
    """Plain asyncio.run() instead of pytest-asyncio - one less test-only dep."""
    return asyncio.run(coro)


class _FakeUpload:
    """Just enough of fastapi.UploadFile for _save_upload: async .read()."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def read(self, _n):
        return self._chunks.pop(0) if self._chunks else b''


# --------------------------------------------------------------------------
# Disk full during an upload write
# --------------------------------------------------------------------------

def test_save_upload_disk_full_is_not_reported_as_oversized(server, tmp_path, monkeypatch):
    """OSError(ENOSPC) mid-write must come back as 'error', not 'oversized'.

    Before this fix both outcomes returned False and create_job() labelled
    every failure "over the upload size limit" - true of nothing when the
    real problem is a full disk, and it sends the user to fix the wrong thing.
    """
    dest = tmp_path / 'out.png'
    import pathlib
    real_open = pathlib.Path.open

    def _open_then_enospc(self, mode='r', *a, **kw):
        handle = real_open(self, mode, *a, **kw)
        handle.write = lambda data: (_ for _ in ()).throw(
            OSError(28, 'No space left on device'))
        return handle

    monkeypatch.setattr(pathlib.Path, 'open', _open_then_enospc)
    result = run(server._save_upload(_FakeUpload([b'not-really-a-png']),
                                      dest, max_bytes=10_000_000))

    assert result == 'error'
    assert not dest.exists()          # partial file must not survive


def test_save_upload_oversized_is_distinct_from_error(server, tmp_path):
    dest = tmp_path / 'out.png'
    result = run(server._save_upload(_FakeUpload([b'x' * 100]), dest, max_bytes=10))
    assert result == 'oversized'
    assert not dest.exists()


def test_save_upload_normal_write_succeeds(server, tmp_path):
    dest = tmp_path / 'out.png'
    result = run(server._save_upload(_FakeUpload([b'hello', b'world']), dest,
                                      max_bytes=10_000_000))
    assert result == 'ok'
    assert dest.read_bytes() == b'helloworld'


# --------------------------------------------------------------------------
# Process killed mid-trace, then the app restarts
# --------------------------------------------------------------------------

def _write_job(work_dir, job_id, payload):
    d = work_dir / job_id
    (d / 'in').mkdir(parents=True, exist_ok=True)
    (d / 'out').mkdir(parents=True, exist_ok=True)
    (d / 'job.json').write_text(payload if isinstance(payload, str)
                                else json.dumps(payload))


def test_restart_recovers_a_job_killed_mid_trace(server):
    """status 'tracing' at the moment of a kill must not sit 'tracing' forever."""
    _write_job(server.WORK, 'jobA', {'id': 'jobA', 'status': 'tracing',
                                      'done': 1, 'total': 3, 'errors': [],
                                      'rejected': [], 'names': ['a.png']})
    server.load_jobs()
    assert server.JOBS['jobA']['status'] == 'interrupted'


def test_restart_recovers_a_job_killed_while_queued(server):
    """A job killed before it ever started tracing must not stay 'queued'
    forever either - nothing on restart will ever pick it back up and run it."""
    _write_job(server.WORK, 'jobB', {'id': 'jobB', 'status': 'queued',
                                      'done': 0, 'total': 2, 'errors': [],
                                      'rejected': [], 'names': ['b.png']})
    server.load_jobs()
    assert server.JOBS['jobB']['status'] == 'interrupted'


def test_restart_leaves_a_finished_job_alone(server):
    _write_job(server.WORK, 'jobC', {'id': 'jobC', 'status': 'ready',
                                      'done': 2, 'total': 2, 'errors': [],
                                      'rejected': [], 'names': ['c.png']})
    server.load_jobs()
    assert server.JOBS['jobC']['status'] == 'ready'


# --------------------------------------------------------------------------
# Corrupt / truncated job.json on startup
# --------------------------------------------------------------------------

def test_truncated_job_json_is_skipped_not_fatal(server):
    """A write cut off mid-flush (disk full, kill -9) can leave job.json as
    invalid JSON entirely. load_jobs() must skip it, not raise."""
    _write_job(server.WORK, 'jobBad', '{"id": "jobBad", "status": "tra')
    server.load_jobs()                 # must not raise
    assert 'jobBad' not in server.JOBS


def test_job_json_valid_json_wrong_shape_is_skipped(server):
    """Valid JSON that is not an object (e.g. an empty array from a write that
    was cut off after the opening bracket) must not crash load_jobs()."""
    _write_job(server.WORK, 'jobList', '[]')
    server.load_jobs()                 # must not raise
    assert 'jobList' not in server.JOBS


def test_job_json_missing_fields_does_not_500_on_status(server):
    """A job.json truncated AFTER valid JSON started (e.g. {"id": "jobX"})
    used to reach job_status() and KeyError on the missing fields - a 500
    where every other endpoint already degraded gracefully."""
    _write_job(server.WORK, 'jobX', {'id': 'jobX'})
    server.load_jobs()
    client = TestClient(server.app)
    r = client.get('/api/jobs/jobX',
                    headers={'X-T-Tracer-Token': server.SESSION_TOKEN})
    assert r.status_code == 200
    body = r.json()
    assert body['status'] == 'unknown'
    assert body['errors'] == []
    assert body['total'] == 0


def test_job_results_survives_one_corrupt_metrics_json(server):
    """One image's metrics.json truncated by a kill mid-write must not 500 the
    whole job's results - the other images in the same batch traced fine and
    should still be visible."""
    job_id = 'jobMixed'
    out = server.WORK / job_id / 'out'
    good, bad = out / 'good-image', out / 'bad-image'
    good.mkdir(parents=True)
    bad.mkdir(parents=True)
    (server.WORK / job_id / 'in').mkdir(parents=True)
    good_metrics = {'otsu': {'overall': 90.0, 'headline': 'clean'}}
    (good / 'metrics.json').write_text(json.dumps(good_metrics))
    (good / 'otsu.png').write_bytes(b'fake-png')
    (bad / 'metrics.json').write_text('{"otsu": {"overall": 5')   # truncated

    server.JOBS[job_id] = {'id': job_id, 'status': 'ready', 'done': 2,
                            'total': 2, 'src_dir': str(server.WORK / job_id / 'in'),
                            'out_dir': str(out), 'errors': [], 'rejected': [],
                            'names': []}
    client = TestClient(server.app)
    r = client.get(f'/api/jobs/{job_id}/results',
                    headers={'X-T-Tracer-Token': server.SESSION_TOKEN})
    assert r.status_code == 200
    stems = [img['stem'] for img in r.json()['images']]
    assert stems == ['good-image']       # the corrupt one is skipped, not fatal
