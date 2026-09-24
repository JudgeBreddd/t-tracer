"""Calibration intake queue: never serve an image that already has a verdict.

The defect this exists to prevent, measured 2026-09-15: 282 images in the
intake folder, 111 already judged, so every batch re-traced work that was
already done.
"""
import json

from fastapi.testclient import TestClient


def _img(path, colour=(20, 30, 200)):
    import cv2
    import numpy as np
    a = np.full((40, 40, 3), 255, np.uint8)
    cv2.circle(a, (20, 20), 12, colour, -1)
    cv2.imwrite(str(path), a)


def _intake(server, tmp_path, names):
    folder = tmp_path / 'new to test'
    folder.mkdir()
    for n in names:
        _img(folder / n)
    server._paths.INTAKE = folder
    return folder


def test_judged_images_are_never_served(server, tmp_path, monkeypatch):
    _intake(server, tmp_path, ['a.png', 'b.png', 'c.png'])
    picks = tmp_path / 'picks.jsonl'
    picks.write_text(json.dumps({'image': 'a.png', 'pick': 'otsu'}) + '\n'
                     + json.dumps({'image': 'b', 'pick': None}) + '\n')
    monkeypatch.setattr(server._paths, 'PICKS', picks)
    pool, counts = server._intake_pool()
    # 'b' was judged as "nothing shippable" - that is still a judgement.
    assert [p.name for p in pool] == ['c.png']
    assert counts == {**counts, 'total': 3, 'judged': 2, 'remaining': 1}


def test_duplicate_stems_are_served_once(server, tmp_path):
    _intake(server, tmp_path, ['dup.png', 'dup.jpg', 'solo.png'])
    pool, counts = server._intake_pool()
    assert sorted(p.stem for p in pool) == ['dup', 'solo']


def test_images_already_in_a_job_are_not_served_again(server, tmp_path):
    _intake(server, tmp_path, ['x.png', 'y.png'])
    live = tmp_path / 'work' / 'job1' / 'in'
    live.mkdir(parents=True)
    _img(live / 'x.png')
    server.JOBS['job1'] = {'id': 'job1', 'src_dir': str(live)}
    pool, counts = server._intake_pool()
    assert [p.name for p in pool] == ['y.png']
    assert counts['in_flight'] == 1


def test_next_copies_rather_than_moves_and_starts_a_job(server, tmp_path, monkeypatch):
    folder = _intake(server, tmp_path, ['a.png', 'b.png', 'c.png'])
    started = []
    monkeypatch.setattr(server, '_start_job',
                        lambda jid, src, out, acc, *a, **k: started.append((jid, acc)) or {'job_id': jid, 'accepted': acc})
    c = TestClient(server.app)
    h = {'X-T-Tracer-Token': server.SESSION_TOKEN}
    r = c.post('/api/intake/next', json={'count': 2}, headers=h)
    assert r.status_code == 200, r.text
    assert r.json()['accepted'] == ['a.png', 'b.png']
    assert r.json()['remaining'] == 1
    # the originals are still there - the intake folder is the only copy
    assert sorted(p.name for p in folder.iterdir()) == ['a.png', 'b.png', 'c.png']
    assert len(started) == 1


def test_empty_queue_is_a_clear_error_not_an_empty_job(server, tmp_path, monkeypatch):
    _intake(server, tmp_path, ['a.png'])
    picks = tmp_path / 'picks.jsonl'
    picks.write_text(json.dumps({'image': 'a.png', 'pick': 'otsu'}) + '\n')
    monkeypatch.setattr(server._paths, 'PICKS', picks)
    c = TestClient(server.app)
    h = {'X-T-Tracer-Token': server.SESSION_TOKEN}
    r = c.post('/api/intake/next', json={'count': 5}, headers=h)
    assert r.status_code == 400
    assert 'already judged' in r.json()['detail']


def test_status_route_reports_a_missing_folder_without_failing(server, tmp_path):
    server._paths.INTAKE = tmp_path / 'nope'
    c = TestClient(server.app)
    h = {'X-T-Tracer-Token': server.SESSION_TOKEN}
    j = c.get('/api/intake', headers=h).json()
    assert j['exists'] is False and j['remaining'] == 0
    assert c.post('/api/intake/next', json={'count': 5}, headers=h).status_code == 404
