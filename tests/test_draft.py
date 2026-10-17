"""K02: star, selection and defect ticks are written per click and come back on reopen."""
import json

from fastapi.testclient import TestClient


def _auth(server):
    return {'X-T-Tracer-Token': server.SESSION_TOKEN}


def _job(server, tmp_path):
    src = tmp_path / 'in'; out = tmp_path / 'out'
    src.mkdir(); out.mkdir()
    (src / 'logo.png').write_bytes(b'')
    d = out / 'logo'; d.mkdir()
    d.joinpath('metrics.json').write_text(json.dumps(
        {'otsu': {'overall': 9}, 'bgdist': {'overall': 5}}))
    for n in ('otsu', 'bgdist'):
        d.joinpath(f'{n}.png').write_bytes(b'')
    server.JOBS['j1'] = {'id': 'j1', 'status': 'ready', 'src_dir': str(src),
                         'out_dir': str(out), 'created': '2026-01-01T00:00:00',
                         'names': ['logo.png']}
    return out


def test_draft_round_trips_through_results(server, tmp_path):
    out = _job(server, tmp_path)
    client = TestClient(server.app)
    body = {'fav': 'bgdist', 'sel': ['bgdist'], 'touched': True,
            'reasons': {'otsu': {'defects': ['gap'], 'note': 'left edge'}}}
    r = client.put('/api/jobs/j1/draft/logo', json=body, headers=_auth(server))
    assert r.json() == {'ok': True}
    # On disk, not in memory: this is what survives a kill.
    assert json.loads((out / '_draft.json').read_text())['logo']['fav'] == 'bgdist'
    img = client.get('/api/jobs/j1/results', headers=_auth(server)).json()['images'][0]
    assert img['draft'] == body
    # The draft file is not mistaken for an image.
    assert len(client.get('/api/jobs/j1/results', headers=_auth(server)).json()['images']) == 1


def test_no_draft_and_corrupt_draft_both_mean_none(server, tmp_path):
    out = _job(server, tmp_path)
    client = TestClient(server.app)
    assert client.get('/api/jobs/j1/results', headers=_auth(server)).json()['images'][0]['draft'] is None
    (out / '_draft.json').write_text('{"logo": {"fav"')   # truncated mid-write
    assert client.get('/api/jobs/j1/results', headers=_auth(server)).json()['images'][0]['draft'] is None


def test_draft_rejects_path_stems_and_unknown_jobs(server, tmp_path):
    _job(server, tmp_path)
    client = TestClient(server.app)
    assert client.put('/api/jobs/nope/draft/logo', json={}, headers=_auth(server)).status_code == 404
    assert client.put('/api/jobs/j1/draft/a%5Cb', json={}, headers=_auth(server)).status_code == 400
