"""Focused tests for the opt-in, job-scoped Burn Map refinement path."""
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import burnmap as burnmap_engine


def _auth(server):
    return {'X-T-Tracer-Token': server.SESSION_TOKEN}


def _job(server, tmp_path):
    src = tmp_path / 'in'; out = tmp_path / 'out'
    src.mkdir(); out.mkdir()
    image = np.full((180, 240, 3), 255, np.uint8)
    cv2.rectangle(image, (22, 25), (215, 150), (20, 40, 110), -1)
    cv2.circle(image, (120, 88), 30, (255, 255, 255), -1)
    cv2.imwrite(str(src / 'sample.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    server.JOBS['j1'] = {'id': 'j1', 'status': 'ready', 'src_dir': str(src),
                         'out_dir': str(out), 'created': '2026-01-01T00:00:00',
                         'names': ['sample.png']}
    return src, out


def test_read_rgb_composites_partial_alpha_over_white(tmp_path):
    source = tmp_path / 'alpha.png'
    rgba = np.array([[[255, 0, 0, 128], [0, 0, 0, 0]]], dtype=np.uint8)
    Image.fromarray(rgba, 'RGBA').save(source)
    rgb, alpha = burnmap_engine.read_rgb(source)
    assert alpha.tolist() == [[128, 0]]
    assert np.allclose(rgb[0, 0], [255, 127, 127], atol=1)
    assert rgb[0, 1].tolist() == [255, 255, 255]


def test_read_rgb_rejects_unbounded_source_dimensions(tmp_path, monkeypatch):
    source = tmp_path / 'large.png'
    Image.new('RGB', (4, 4), 'white').save(source)
    monkeypatch.setattr(burnmap_engine, 'MAX_SOURCE_PIXELS', 15)
    with pytest.raises(ValueError, match='refinement is limited'):
        burnmap_engine.read_rgb(source)


def test_refine_state_is_job_scoped_and_assets_require_token(server, tmp_path):
    _job(server, tmp_path)
    client = TestClient(server.app)
    assert client.get('/api/jobs/j1/refine/sample').status_code == 401
    headers = _auth(server)
    response = client.get('/api/jobs/j1/refine/sample', headers=headers)
    assert response.status_code == 200
    payload = response.json()
    assert payload['width'] <= 800 and payload['height'] <= 800
    assert payload['backend'] in {'slic', 'legacy', 'legacy-fallback'}
    assert len(payload['proposals']) == 3
    asset = payload['assets']['result.png']
    assert client.get(asset).status_code == 401
    assert client.get(asset, params={'tt_token': server.SESSION_TOKEN}).headers['content-type'].startswith('image/png')


def test_refine_actions_change_preview_and_save_corrected_svg(server, tmp_path):
    _job(server, tmp_path)
    client = TestClient(server.app)
    headers = _auth(server)
    before = client.get('/api/jobs/j1/refine/sample', headers=headers).json()
    region = before['regions'][0]['region_id']
    changed = client.post('/api/jobs/j1/refine/sample/action', headers=headers,
                          json={'region_id': region}).json()
    assert changed['metrics']['changed_regions'] >= 1
    point = before['regions'][1]['centroid']
    clicked = client.post('/api/jobs/j1/refine/sample/action', headers=headers,
                          json={'x': round(point[0]), 'y': round(point[1])})
    assert clicked.status_code == 200
    download = client.get('/api/jobs/j1/refine/sample/download', headers=headers)
    assert download.status_code == 200
    assert download.content.startswith(b'<svg')
    assert 'filename*=' in download.headers['content-disposition']
    query_download = client.get('/api/jobs/j1/refine/sample/download',
                                params={'tt_token': server.SESSION_TOKEN})
    assert query_download.status_code == 200
    dest = tmp_path / 'saved'
    saved = client.post('/api/jobs/j1/refine/sample/save', headers=headers,
                        json={'dest': str(dest)})
    assert saved.status_code == 200
    assert (dest / 'sample-corrected.svg').is_file()


def test_refine_cache_is_bounded_and_cleared_with_history(server, tmp_path):
    _job(server, tmp_path)
    headers = _auth(server)
    client = TestClient(server.app)
    first = client.get('/api/jobs/j1/refine/sample', headers=headers).json()
    first_region = first['regions'][0]['region_id']
    edited = client.post('/api/jobs/j1/refine/sample/action', headers=headers,
                         json={'region_id': first_region}).json()
    edited_value = next(row['solved_ink'] for row in edited['regions']
                        if row['region_id'] == first_region)
    for index in range(server.REFINE_MAX_STATES + 2):
        stem = f'sample-{index}'
        source = Path(server.JOBS['j1']['src_dir']) / f'{stem}.png'
        source.write_bytes((Path(server.JOBS['j1']['src_dir']) / 'sample.png').read_bytes())
        client.get(f'/api/jobs/j1/refine/{stem}', headers=headers)
    assert len(server.REFINE_STATES) == server.REFINE_MAX_STATES
    reopened = client.get('/api/jobs/j1/refine/sample', headers=headers).json()
    assert next(row['solved_ink'] for row in reopened['regions']
                if row['region_id'] == first_region) == edited_value
    client.delete('/api/jobs/j1', headers=headers)
    assert not server.REFINE_STATES
    assert not server.REFINE_SNAPSHOTS
