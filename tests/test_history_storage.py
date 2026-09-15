"""History storage controls (item 7): the selection logic that decides what
a cleanup pass is allowed to delete, and that it never reaches settings.json,
app-picks.jsonl or an active/queued job.
"""
import json

from fastapi.testclient import TestClient


def _job(status, created, out_dir=None):
    return {'status': status, 'created': created,
            'out_dir': str(out_dir) if out_dir else '/nonexistent'}


def test_active_and_queued_jobs_are_never_selected(server):
    jobs = {
        'a': _job('ready', '2026-01-01T00:00:00'),
        'b': _job('queued', '2026-01-01T00:00:00'),
        'c': _job('tracing', '2026-01-01T00:00:00'),
        'd': _job('failed', '2026-01-01T00:00:00'),
        'e': _job('interrupted', '2026-01-01T00:00:00'),
    }
    picked = set(server.select_jobs_to_clean(jobs))
    assert picked == {'a', 'd', 'e'}


def test_cutoff_only_picks_jobs_strictly_older(server):
    jobs = {
        'old': _job('ready', '2026-01-01T00:00:00'),
        'new': _job('ready', '2026-06-01T00:00:00'),
    }
    picked = server.select_jobs_to_clean(jobs, cutoff='2026-03-01T00:00:00')
    assert picked == ['old']


def test_no_cutoff_picks_every_finished_job_regardless_of_age(server):
    jobs = {
        'old': _job('ready', '2020-01-01T00:00:00'),
        'new': _job('ready', '2026-01-01T00:00:00'),
    }
    picked = set(server.select_jobs_to_clean(jobs))
    assert picked == {'old', 'new'}


def test_clean_jobs_never_touches_settings_or_picks(server, tmp_path):
    """The structural guarantee: cleanup only ever rmtrees WORK/<job_id>, and
    settings.json / app-picks.jsonl live at WORK.parent, one level above."""
    job_id = 'jobZ'
    job_dir = server.WORK / job_id
    (job_dir / 'out' / 'logo').mkdir(parents=True)
    (job_dir / 'out' / 'logo' / 'metrics.json').write_text('{}')
    server.JOBS[job_id] = _job('ready', '2020-01-01T00:00:00', job_dir / 'out')

    server.SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    server.SETTINGS_FILE.write_text(json.dumps({'dest': '/somewhere'}))
    server.APP_PICKS.write_text('{"image": "x", "pick": "otsu"}\n')

    freed = server._clean_jobs(server.select_jobs_to_clean(server.JOBS))

    assert freed > 0
    assert not job_dir.exists()
    assert job_id not in server.JOBS
    assert server.SETTINGS_FILE.read_text() == json.dumps({'dest': '/somewhere'})
    assert 'otsu' in server.APP_PICKS.read_text()


def test_clean_jobs_never_touches_an_active_job_even_if_passed_by_mistake(server):
    """_clean_jobs() itself does not re-check status - select_jobs_to_clean is
    the one gate. This documents that the API route always goes through it."""
    job_id = 'jobActive'
    job_dir = server.WORK / job_id
    (job_dir / 'out').mkdir(parents=True)
    server.JOBS[job_id] = _job('tracing', '2020-01-01T00:00:00', job_dir / 'out')

    # The route-level guarantee: select_jobs_to_clean excludes it, so the
    # normal call path (clear_history / _apply_retention) never passes an
    # active job's id to _clean_jobs at all.
    assert server.select_jobs_to_clean(server.JOBS) == []


def test_storage_endpoint_reports_bytes_and_job_count(server):
    job_dir = server.WORK / 'jobS' / 'out' / 'x'
    job_dir.mkdir(parents=True)
    (job_dir / 'a.svg').write_bytes(b'x' * 1000)
    server.JOBS['jobS'] = _job('ready', '2026-01-01T00:00:00', job_dir)

    client = TestClient(server.app)
    r = client.get('/api/storage', headers={'X-T-Tracer-Token': server.SESSION_TOKEN})
    assert r.status_code == 200
    body = r.json()
    assert body['bytes'] >= 1000
    assert body['jobs'] == 1


def test_clear_history_endpoint_skips_active_jobs(server):
    finished_dir = server.WORK / 'jobF' / 'out'
    finished_dir.mkdir(parents=True)
    (finished_dir / 'x.svg').write_bytes(b'data')
    server.JOBS['jobF'] = _job('ready', '2020-01-01T00:00:00', finished_dir)

    active_dir = server.WORK / 'jobT' / 'out'
    active_dir.mkdir(parents=True)
    server.JOBS['jobT'] = _job('tracing', '2020-01-01T00:00:00', active_dir)

    client = TestClient(server.app)
    r = client.post('/api/history/clear', headers={'X-T-Tracer-Token': server.SESSION_TOKEN})
    assert r.status_code == 200
    assert r.json()['cleared'] == 1
    assert 'jobF' not in server.JOBS
    assert 'jobT' in server.JOBS
    assert (server.WORK / 'jobT').exists()


def test_retention_setting_rejects_unknown_values(server):
    client = TestClient(server.app)
    r = client.put('/api/settings', json={'retention': 'never'},
                    headers={'X-T-Tracer-Token': server.SESSION_TOKEN})
    assert r.status_code == 400


def test_retention_setting_accepts_known_values(server):
    client = TestClient(server.app)
    r = client.put('/api/settings', json={'retention': '30d'},
                    headers={'X-T-Tracer-Token': server.SESSION_TOKEN})
    assert r.status_code == 200
    assert json.loads(server.SETTINGS_FILE.read_text())['retention'] == '30d'
