"""Live view: send-latest semantics, zero disk writes unless saved, and the
job-loop bookkeeping around it."""
import json
import numpy as np
from fastapi.testclient import TestClient


def test_observer_keeps_only_the_latest_frame_per_step(server):
    lv = server.LiveView(keep=False)
    on = lv.observer('logo')
    on('otsu', 'mask', np.zeros((8, 8), bool))
    on('otsu', 'mask', np.ones((8, 8), bool))
    on('otsu', 'clean', np.ones((8, 8), bool))
    frames = lv.frames()
    assert [(f['strategy'], f['step']) for f in frames] == [('otsu', 'mask'), ('otsu', 'clean')]
    assert frames[0]['seq'] == 2                 # the first mask frame was replaced
    assert lv.retained == []


def test_keep_retains_every_frame_as_small_png(server):
    lv = server.LiveView(keep=True)
    on = lv.observer('logo')
    on('otsu', 'mask', np.zeros((2000, 1000), bool))
    on('otsu', 'mask', np.ones((2000, 1000), bool))
    assert len(lv.retained) == 2
    assert all(png.startswith(b'\x89PNG') for *_, png in lv.retained)


def test_encode_handles_mask_rgb_and_layers(server):
    assert server._live_encode(np.zeros((4, 4), bool)).startswith(b'\x89PNG')
    assert server._live_encode(np.zeros((4, 4, 3), np.uint8)).startswith(b'\x89PNG')
    layers = [(np.ones((4, 4), bool), (1, 2, 3))]
    assert server._live_encode(layers).startswith(b'\x89PNG')
    assert server._live_encode([]) == b''


def test_live_endpoints_and_save_writes_only_on_request(server, tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    server.JOBS['j1'] = {'id': 'j1', 'status': 'ready', 'out_dir': str(out)}
    lv = server.LiveView(keep=True)
    lv.observer('logo')('otsu', 'mask', np.zeros((4, 4), bool))
    server.LIVE['j1'] = lv
    c = TestClient(server.app)
    h = {'X-T-Tracer-Token': server.SESSION_TOKEN}
    j = c.get('/api/jobs/j1/live', headers=h).json()
    assert j['enabled'] and j['retained'] == 1 and j['frames'][0]['strategy'] == 'otsu'
    assert not (out / '_live').exists()
    r = c.post('/api/jobs/j1/live/save', headers=h).json()
    assert r['saved'] == 1
    assert len(list((out / '_live').glob('*.png'))) == 1
    assert 'j1' not in server.LIVE
    assert c.get('/api/jobs/j1/live', headers=h).json()['enabled'] is False


def test_live_is_off_by_default_in_settings(server):
    assert server.DEFAULT_SETTINGS['live_view'] is False
    assert server.DEFAULT_SETTINGS['live_keep'] is False


def test_cli_defaults_come_from_the_parser(server):
    ns = server.cli_defaults()
    for k in ('min_dim', 'min_island', 'work_dim', 'smoothing', 'strategies'):
        assert hasattr(ns, k)
    assert ns.jobs == 1
    assert server.cli_defaults(on_step=print).on_step is print
