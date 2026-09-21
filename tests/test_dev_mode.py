"""Developer mode: excluding a test run, and purging an image completely.

Both exist because this app is also the instrument that MEASURES the tracer.
A run made to watch the live view, or on a file dropped in by mistake, lands
in the same log as a real verdict and moves the hit rate. Exclude retracts it
softly; purge erases it. Purge is the only destructive thing in the server, so
its blast radius is pinned here rather than trusted.
"""
import json

import pytest
from fastapi.testclient import TestClient


def _auth(server):
    return {'X-T-Tracer-Token': server.SESSION_TOKEN}


def _row(**kw):
    return json.dumps({'image': 'a', 'pick': 'otsu', 'date': '2026-09-16', **kw})


def test_excluded_row_drops_out_of_stats_and_the_judged_set(server, tmp_path):
    server.APP_PICKS.write_text(
        _row() + '\n'
        + _row(image='b', pick=None) + '\n'
        + _row(image='b', pick=None, excluded=True, note='wrong file') + '\n')
    rows = server.read_jsonl(server.APP_PICKS)
    assert len(rows) == 2                       # last row wins per image
    live = server.active_rows(rows)
    assert [r['image'] for r in live] == ['a']  # the tombstone retracts 'b'
    assert server.summarise(live)['judged'] == 1
    assert 'b' not in server._judged_stems()    # so 'b' is queued again


def test_judge_writes_the_exclusion_and_the_reason(server, tmp_path):
    client = TestClient(server.app)
    server.JOBS['j1'] = {'id': 'j1', 'status': 'done'}
    r = client.post('/api/jobs/j1/judge', headers=_auth(server), json=[
        {'stem': 'a', 'pick': None, 'failed': True, 'of': 5,
         'note': '  source is a photo of a patch  '},
        {'stem': 'b', 'pick': 'otsu', 'of': 5, 'excluded': True},
    ])
    assert r.status_code == 200
    rows = [json.loads(l) for l in server.APP_PICKS.read_text().splitlines()]
    assert rows[0]['note'] == 'source is a photo of a patch'   # stripped
    assert 'excluded' not in rows[0]            # absent, not False
    assert rows[1]['excluded'] is True
    assert 'note' not in rows[1]                # a blank reason writes nothing


def test_purge_endpoints_refuse_while_dev_mode_is_off(server, tmp_path):
    client = TestClient(server.app)
    a = _auth(server)
    assert client.post('/api/purge', headers=a, json={'stem': 'a'}).status_code == 403
    assert client.post('/api/purge/plan', headers=a, json={'stem': 'a'}).status_code == 403


@pytest.mark.parametrize('stem', ['../x', 'a/b', '', '.', '..'])
def test_purge_rejects_a_stem_that_could_escape(server, stem):
    server.write_settings({'dev_mode': True})
    with pytest.raises(Exception):
        server._purge_plan(stem)


def _tree(server, tmp_path):
    """A stand-in for every place an image can exist, all pointed at tmp."""
    corpus, intake, cands = (tmp_path / 'corpus', tmp_path / 'new to test',
                             tmp_path / 'candidates')
    for d in (corpus, intake, cands):
        d.mkdir()
    (corpus / 'a.png').write_bytes(b'x')
    (intake / 'a.png').write_bytes(b'x')
    (cands / 'a').mkdir()
    (cands / 'a' / 'otsu.svg').write_text('<svg/>')
    (corpus / 'keep.png').write_bytes(b'x')          # a bystander
    server._paths.CORPUS, server._paths.INTAKE = corpus, intake
    server._paths.CANDIDATES = cands
    server._paths.RUNS = tmp_path / 'runs'           # absent on purpose
    server._paths.INBOX = tmp_path / 'inbox'
    server.CALIB_PICKS = tmp_path / 'picks.jsonl'
    server.CALIB_PICKS.write_text(_row() + '\n' + _row(image='keep') + '\n')
    server.APP_PICKS.write_text(_row(excluded=True) + '\n' + _row(image='keep') + '\n')
    # two jobs: one holding only 'a', one holding 'a' alongside another image
    solo, mixed = server.WORK / 'solo', server.WORK / 'mixed'
    for j in (solo, mixed):
        (j / 'in').mkdir(parents=True)
        (j / 'out').mkdir(parents=True)
        (j / 'in' / 'a.png').write_bytes(b'x')
        (j / 'out' / 'a').mkdir()
        (j / 'job.json').write_text(json.dumps({'id': j.name}))
    (mixed / 'in' / 'other.png').write_bytes(b'x')
    (mixed / 'out' / 'other').mkdir()
    server.JOBS.update({'solo': {'id': 'solo'}, 'mixed': {'id': 'mixed'}})
    return corpus, intake, cands, solo, mixed


def test_purge_plan_lists_exactly_what_will_go(server, tmp_path):
    corpus, intake, cands, solo, mixed = _tree(server, tmp_path)
    server.write_settings({'dev_mode': True})
    plan = server._purge_plan('a')
    paths = {p['path'] for p in plan['paths']}
    assert str(solo) in paths                        # whole job: only held 'a'
    assert str(mixed) not in paths                   # shared job survives
    assert str(mixed / 'in' / 'a.png') in paths
    assert str(mixed / 'out' / 'a') in paths
    assert str(corpus / 'a.png') in paths
    assert str(intake / 'a.png') in paths
    assert str(cands / 'a') in paths
    assert str(corpus / 'keep.png') not in paths
    assert plan['rows'] == {'app': 1, 'calibration': 1}


def test_purge_erases_everything_and_leaves_the_bystanders(server, tmp_path):
    corpus, intake, cands, solo, mixed = _tree(server, tmp_path)
    server.write_settings({'dev_mode': True})
    client = TestClient(server.app)
    r = client.post('/api/purge', headers=_auth(server), json={'stem': 'a'}).json()
    assert r['ok'] and not r['failed']
    assert r['rows'] == {'app': 1, 'calibration': 1}
    for gone in (corpus / 'a.png', intake / 'a.png', cands / 'a', solo,
                 mixed / 'in' / 'a.png', mixed / 'out' / 'a'):
        assert not gone.exists(), gone
    for kept in (corpus / 'keep.png', mixed / 'in' / 'other.png',
                 mixed / 'out' / 'other', mixed / 'job.json'):
        assert kept.exists(), kept
    assert [json.loads(l)['image'] for l in
            server.CALIB_PICKS.read_text().splitlines()] == ['keep']
    assert [json.loads(l)['image'] for l in
            server.APP_PICKS.read_text().splitlines()] == ['keep']
    assert 'solo' not in server.JOBS and 'mixed' in server.JOBS


def test_purge_leaves_an_unreadable_line_alone(server, tmp_path):
    _tree(server, tmp_path)
    server.write_settings({'dev_mode': True})
    server.APP_PICKS.write_text('not json\n' + _row() + '\n')
    assert server._strip_rows(server.APP_PICKS, 'a') == 1
    assert server.APP_PICKS.read_text() == 'not json\n'


def test_purge_keeps_log_rows_when_a_file_cannot_be_deleted(server, tmp_path,
                                                              monkeypatch):
    """A partial filesystem purge must not also erase its audit trail."""
    _, _, cands, _, _ = _tree(server, tmp_path)
    server.write_settings({'dev_mode': True})
    real_rmtree = server.shutil.rmtree

    def fail_one(path, *args, **kwargs):
        if path == cands / 'a':
            raise OSError('locked')
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(server.shutil, 'rmtree', fail_one)
    client = TestClient(server.app)
    result = client.post('/api/purge', headers=_auth(server),
                         json={'stem': 'a'}).json()

    assert result['ok'] is False
    assert result['rows'] == {'app': 0, 'calibration': 0}
    assert any(json.loads(line)['image'] == 'a'
               for line in server.CALIB_PICKS.read_text().splitlines())
    assert any(json.loads(line)['image'] == 'a'
               for line in server.APP_PICKS.read_text().splitlines())
