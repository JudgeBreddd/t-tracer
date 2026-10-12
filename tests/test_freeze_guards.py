"""2026-10-02: two batches froze on phone photos. One image may now take at most
IMAGE_TIME_LIMIT_S, and images a frozen batch never reached go back in the queue."""
import json
import time

import pytest


def _job(server, tmp_path, status, traced=(), untraced=(), timed_out=()):
    src, out = tmp_path / 'in', tmp_path / 'out'
    src.mkdir(); out.mkdir()
    for s in (*traced, *untraced, *timed_out):
        (src / f'{s}.png').write_bytes(b'x')
    for s in traced:
        (out / s).mkdir(); (out / s / 'metrics.json').write_text('{}')
    server.JOBS['j'] = {'id': 'j', 'status': status, 'src_dir': str(src), 'out_dir': str(out),
                        'timed_out': list(timed_out)}


def test_interrupted_job_releases_images_it_never_reached(server, tmp_path):
    _job(server, tmp_path, 'interrupted', traced=['a'], untraced=['b'], timed_out=['c'])
    assert server._in_flight_stems() == {'a', 'c'}


def test_running_job_still_holds_every_image(server, tmp_path):
    _job(server, tmp_path, 'tracing', traced=['a'], untraced=['b'])
    assert server._in_flight_stems() == {'a', 'b'}


def test_restart_mid_batch_puts_unreached_images_back(server):
    # Wiki audit K11: the real recovery path - app killed mid-trace, then started again.
    d = server.WORK / 'j'
    (d / 'in').mkdir(parents=True); (d / 'out' / 'a').mkdir(parents=True)
    for s in ('a', 'b'):
        (d / 'in' / f'{s}.png').write_bytes(b'x')
    (d / 'out' / 'a' / 'metrics.json').write_text('{}')
    (d / 'job.json').write_text(json.dumps({'id': 'j', 'status': 'tracing'}))
    server.load_jobs()
    assert server.JOBS['j']['status'] == 'interrupted'
    assert server._in_flight_stems() == {'a'}


def _slow(img, out, on_step=None):
    time.sleep(5)


def test_image_over_the_time_limit_is_skipped(server, tmp_path, monkeypatch):
    monkeypatch.setattr(server, '_trace_one', _slow)
    import multiprocessing as mp                  # fork, so the child sees the patched _trace_one
    fork = mp.get_context('fork')
    monkeypatch.setattr(mp, 'get_context', lambda *a: fork)
    t = time.time()
    with pytest.raises(TimeoutError):
        server._trace_with_limit(tmp_path / 'x.png', tmp_path, 1)
    assert time.time() - t < 4
