"""Per-candidate verdicts (2026-10-02): every candidate of a judged image gets a
row carrying the reviewer's call, the named defects behind a rejection, and the
hygiene numbers that candidate got at trace size, 70 mm and 150 mm."""
import json

import numpy as np
from fastapi.testclient import TestClient

import defects
from vector import paths_to_svg


def _auth(server):
    return {'X-T-Tracer-Token': server.SESSION_TOKEN}


def test_svg_round_trips_and_scores_at_both_sizes():
    sq = [[(np.array(a, float), np.array(a, float), np.array(b, float), np.array(b, float))
           for a, b in [((10, 10), (90, 10)), ((90, 10), (90, 90)),
                        ((90, 90), (10, 90)), ((10, 90), (10, 10))]]]
    w, h, paths = defects.read_svg(paths_to_svg(sq, 100, 100))
    ink = defects.rasterize_at(paths, w, h, long_px=500)
    src = np.where(ink[..., None], 0, 255).astype(np.uint8).repeat(3, 2)
    rep = defects.score_raster(ink, src, paths, 0)
    assert set(rep) == {'70mm', '150mm'}
    assert rep['70mm']['strays'] == 0 and rep['70mm']['nodes'] == 4


def test_judge_writes_one_verdict_row_per_candidate(server, tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'VERDICTS', tmp_path / 'verdicts.jsonl')
    server.write_settings({'dev_mode': True})
    out = tmp_path / 'out'
    (out / 'logo').mkdir(parents=True)
    rep = {'hygiene': 90.0, 'overall': 90.0, 'fidelity': 0.9}
    (out / 'logo' / 'metrics.json').write_text(json.dumps(
        {'otsu': {**rep, 'overall': 95.0}, 'kmeans': rep}))
    (out / 'logo' / 'defects.json').write_text(json.dumps({
        'render_px': 4500, 'image_type': 'bw',
        'sizes': {'70mm': {'kmeans': {'gaps': 3}}, '150mm': {'kmeans': {'gaps': 1}}}}))
    server.JOBS['j1'] = {'id': 'j1', 'status': 'done', 'out_dir': str(out)}
    r = TestClient(server.app).post('/api/jobs/j1/judge', headers=_auth(server), json=[{
        'stem': 'logo', 'pick': 'otsu', 'shippable': ['otsu'], 'of': 2,
        'reasons': {'kmeans': {'defects': ['gap', 'blotch'], 'note': ' lumps '},
                    'otsu': {'defects': ['stray']}},
        'image_type': 'multicolour'}])
    assert r.status_code == 200
    rows = {x['candidate']: x for x in map(json.loads, server.VERDICTS.read_text().splitlines())}
    assert rows['otsu']['shippable'] and rows['otsu']['picked']
    assert rows['otsu']['defects'] == []            # a shippable one carries no defects
    assert rows['kmeans']['defects'] == ['gap', 'blotch'] and rows['kmeans']['note'] == 'lumps'
    assert rows['kmeans']['hygiene_70mm'] == {'gaps': 3}
    assert rows['kmeans']['image_type_auto'] == 'bw'
    assert rows['kmeans']['image_type'] == 'multicolour'


def test_no_verdict_rows_without_dev_mode(server, tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'VERDICTS', tmp_path / 'verdicts.jsonl')
    server.JOBS['j1'] = {'id': 'j1', 'status': 'done', 'out_dir': str(tmp_path)}
    TestClient(server.app).post('/api/jobs/j1/judge', headers=_auth(server),
                                json=[{'stem': 'logo', 'pick': None, 'failed': True}])
    assert not server.VERDICTS.exists()


def test_reopened_job_gets_its_saved_ticks_back(server, tmp_path, monkeypatch):
    # Wiki audit K02: ticks lived only in the page. Last row per candidate wins.
    monkeypatch.setattr(server, 'VERDICTS', tmp_path / 'verdicts.jsonl')
    rows = [{'job': 'j', 'image': 'logo', 'candidate': 'a', 'defects': ['gap'], 'note': None},
            {'job': 'j', 'image': 'logo', 'candidate': 'b', 'defects': ['stray'], 'note': 'x'},
            {'job': 'j', 'image': 'logo', 'candidate': 'b', 'defects': [], 'note': None},
            {'job': 'other', 'image': 'logo', 'candidate': 'a', 'defects': ['gap'], 'note': None}]
    server.VERDICTS.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    assert server._saved_reasons('j') == {'logo': {'a': {'defects': ['gap'], 'note': ''}}}
    assert server._saved_reasons('none') == {}


def test_unchanged_candidate_is_scored_from_cache(tmp_path, monkeypatch):
    # Wiki audit K13: same SVG + source + detector code -> no second render.
    monkeypatch.setenv('TT_SCORE_CACHE', str(tmp_path / 'cache'))
    monkeypatch.setattr(defects, 'RENDER_PX', 400)
    sq = [[(np.array(a, float), np.array(a, float), np.array(b, float), np.array(b, float))
           for a, b in [((10, 10), (90, 10)), ((90, 10), (90, 90)),
                        ((90, 90), (10, 90)), ((10, 90), (10, 10))]]]
    (tmp_path / 'a.svg').write_text(paths_to_svg(sq, 100, 100))
    src = np.full((100, 100, 3), 255, np.uint8); src[10:90, 10:90] = 0
    first = defects.score_image(tmp_path, src, 100, {'a': {'hygiene': 1}})
    again = defects.score_image(tmp_path, src, 100, {'a': {'hygiene': 1}})
    assert (first['cache_hits'], again['cache_hits']) == (0, 1)
    assert first['sizes'] == again['sizes'] and (tmp_path / 'a.hi.png').is_file()
    src[0, 0] = 0                                   # source changed: must re-score
    assert defects.score_image(tmp_path, src, 100, {'a': {'hygiene': 1}})['cache_hits'] == 0
