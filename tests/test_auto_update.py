"""Auto-update toggle (item 3.5): the parts that can be tested without a real
Windows machine or a real GitHub release - version comparison, asset lookup,
and checksum verification. The download/launch/exit path itself is NOT
covered here; see the PR description for what stays unverified.
"""
import hashlib

from fastapi.testclient import TestClient


def test_version_parts_orders_numerically_not_lexically(server):
    # Lexical string comparison would get this backwards: '0.10.0' < '0.2.0'
    # as strings, but 0.10.0 is the NEWER version.
    assert server._version_parts('0.10.0') > server._version_parts('0.2.0')
    assert server._version_parts('1.2.0') > server._version_parts('0.9.9')
    assert server._version_parts('0.1.2') == server._version_parts('0.1.2')


def test_release_assets_maps_name_to_url(server):
    payload = {'assets': [
        {'name': 'T-Tracer-Setup.exe', 'browser_download_url': 'https://x/exe'},
        {'name': 'T-Tracer-Setup.exe.sha256', 'browser_download_url': 'https://x/sha'},
        {'name': 'source.zip'},          # no browser_download_url: must be skipped
    ]}
    assets = server._release_assets(payload)
    assert assets == {'T-Tracer-Setup.exe': 'https://x/exe',
                      'T-Tracer-Setup.exe.sha256': 'https://x/sha'}


def test_release_assets_handles_no_assets_key(server):
    assert server._release_assets({}) == {}


def test_verify_sha256_matches_bare_hex_digest(server, tmp_path):
    """Get-FileHash's own format - what app/install.ps1's PyInstallerSha256
    already uses for the Python installer check this mirrors."""
    p = tmp_path / 'file.bin'
    p.write_bytes(b'hello world')
    expected = hashlib.sha256(b'hello world').hexdigest()
    assert server.verify_sha256(p, expected.upper())   # case-insensitive
    assert server.verify_sha256(p, expected)


def test_verify_sha256_matches_sha256sum_style_line(server, tmp_path):
    p = tmp_path / 'file.bin'
    p.write_bytes(b'hello world')
    expected = hashlib.sha256(b'hello world').hexdigest()
    assert server.verify_sha256(p, f'{expected}  T-Tracer-Setup.exe\n')


def test_verify_sha256_rejects_a_mismatch(server, tmp_path):
    p = tmp_path / 'file.bin'
    p.write_bytes(b'hello world')
    assert not server.verify_sha256(p, '0' * 64)


def test_verify_sha256_rejects_empty_checksum_file(server, tmp_path):
    """A checksum asset that failed to download (empty/truncated) must not
    read as 'no expected value, anything passes'."""
    p = tmp_path / 'file.bin'
    p.write_bytes(b'hello world')
    assert not server.verify_sha256(p, '')
    assert not server.verify_sha256(p, '   \n')


def test_auto_update_never_runs_off_windows(server, monkeypatch):
    """Even with the setting on and a real newer release available, the
    non-Windows guard must return before any network call - patch
    _fetch_latest_release to explode if it is ever reached."""
    monkeypatch.setattr(server.sys, 'platform', 'linux')
    server.write_settings({'auto_update': True})

    def _boom():
        raise AssertionError('must not fetch a release on non-Windows')
    monkeypatch.setattr(server, '_fetch_latest_release', _boom)

    server._auto_update_once()           # must return quietly, not raise


def test_auto_update_is_a_noop_when_setting_is_off(server, monkeypatch):
    monkeypatch.setattr(server.sys, 'platform', 'win32')
    server.write_settings({'auto_update': False})

    def _boom():
        raise AssertionError('must not fetch a release when the toggle is off')
    monkeypatch.setattr(server, '_fetch_latest_release', _boom)

    server._auto_update_once()


def test_auto_update_records_an_error_for_a_release_missing_the_checksum(server, monkeypatch):
    """Missing checksum => do not run, surface a clear error (item 3.5)."""
    monkeypatch.setattr(server.sys, 'platform', 'win32')
    server.write_settings({'auto_update': True})
    monkeypatch.setattr(server, '_fetch_latest_release', lambda: {
        'tag_name': 'v99.0.0',
        'assets': [{'name': 'T-Tracer-Setup.exe', 'browser_download_url': 'https://x/exe'}],
    })
    server.UPDATE_STATE = {'status': 'idle', 'detail': ''}
    server._auto_update_once()
    assert server.UPDATE_STATE['status'] == 'error'
    assert 'checksum' in server.UPDATE_STATE['detail'].lower()


def test_auto_update_status_endpoint_reports_current_state(server):
    server.UPDATE_STATE = {'status': 'error', 'detail': 'boom'}
    client = TestClient(server.app)
    r = client.get('/api/update-status', headers={'X-T-Tracer-Token': server.SESSION_TOKEN})
    assert r.status_code == 200
    assert r.json() == {'status': 'error', 'detail': 'boom'}
    server.UPDATE_STATE = {'status': 'idle', 'detail': ''}   # leave clean for other tests
