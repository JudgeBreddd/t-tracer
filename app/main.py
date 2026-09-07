#!/usr/bin/env python3
"""T-Tracer - desktop launcher.

Double-click entry point. Starts the local API on a free loopback port, then
opens it in a real application window.

THREE WINDOW BACKENDS, tried in order, because none of them is available
everywhere:

1. **Chrome/Edge in --app mode** - a chromeless standalone window: no tabs,
   no address bar, its own taskbar entry. Indistinguishable from a native app
   window for this purpose, and the DEFAULT on Windows (see below).
2. **pywebview** - wraps the OS's own webview. Used on macOS (Cocoa), and as
   a last resort on Windows if no Chromium browser is found at all.
3. **The default browser** - a normal tab. Ugly but functional; better than
   refusing to start.

Why the fallback chain exists, learned the hard way: on Linux pywebview needs
PyGObject+WebKit2GTK or Qt+QtWebEngine, and **neither is pip-installable in
practice** - they are system packages. `pip install pywebview` therefore
succeeds on a bare Linux box and then `webview.start()` has nothing to render
into, so the app appears to launch and simply never opens a window. That is
exactly what happened on this machine. Detecting the backend BEFORE trying to
use it turns a silent hang into a working window.

**Windows-specific reason pywebview is no longer tried first, found on a real
install:** pywebview's WinForms host window hit a documented upstream bug
where Windows 11's Snap Layout hover triggers infinite recursion walking the
window's accessibility tree (`window.native.AccessibilityObject.Bounds.Empty.
Empty...` -> `RecursionError`). pywebview swallows the error per-event rather
than raising it, so nothing here can catch it - the window silently fails to
render (or renders and then vanishes) while the Python process and its local
server keep running orphaned in the background. Reported as "it opens and
then closes." Chrome/Edge `--app` mode hosts its own window and never touches
pywebview's WinForms code at all, which is why it moved to first choice on
Windows - every Windows 10/11 machine ships Edge, so this is not a
"hope a browser is installed" fallback the way it is on Linux.
"""
from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

# pythonw.exe (a GUI-subsystem executable - what every shortcut here launches,
# and what "no console window" requires) has NO console at all unless
# something redirects it, and CPython then sets sys.stdout/sys.stderr to
# None rather than a no-op stream. A bare print() under that condition raises
# AttributeError: 'NoneType' object has no attribute 'write' - instantly, with
# no console to show it, so the whole process dies before doing anything
# visible. Found the hard way: this is what "the app opens and then closes"
# (or "nothing happens at all") turned out to actually be, once Chrome/Edge
# became the default Windows window backend and its one print() call sat on
# that path unconditionally. Route every write in this file through here
# instead of the stdlib streams directly, and give both a real sink up front
# so nothing else - server.py, a library, a future print - can hit the same
# crash by writing to sys.stdout/sys.stderr elsewhere in the process.
if sys.stdout is None or sys.stderr is None:
    import os
    _devnull = open(os.devnull, 'w')                        # noqa: SIM115
    sys.stdout = sys.stdout or _devnull
    sys.stderr = sys.stderr or _devnull


def log(*args, **kwargs) -> None:
    """print(), but safe under pythonw with no console (see the guard above)."""
    try:
        print(*args, **kwargs)
    except Exception:                                        # noqa: BLE001
        pass


def free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class Api:
    """Bridge exposed to the page as `window.pywebview.api`."""

    def __init__(self):
        self.window = None

    def pick_folder(self):
        import webview
        res = self.window.create_file_dialog(webview.FOLDER_DIALOG)
        return res[0] if res else None


def _has_webview_backend() -> bool:
    """True only if pywebview can actually open a window on this machine.

    pywebview installs cleanly without a renderer on Linux, so importing it
    proves nothing. Windows and macOS always have one.
    """
    try:
        import webview                                    # noqa: F401
    except ImportError:
        return False
    if sys.platform != 'linux':
        return True
    import importlib.util
    return any(importlib.util.find_spec(m) is not None
               for m in ('gi', 'PyQt6', 'PyQt5', 'PySide6', 'PySide2'))


def find_chrome() -> str | None:
    """A Chromium-family browser that supports --app windows.

    Edge is checked because it ships with every Windows 10/11 install - this
    is the primary Windows path now, not a last-resort guess.
    """
    import shutil
    names = ['google-chrome-stable', 'google-chrome', 'chromium',
             'chromium-browser', 'brave-browser', 'microsoft-edge', 'msedge']
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    fixed = [
        '/opt/google/chrome/chrome',
        '/usr/lib/chromium/chromium',
        '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        r'C:\Program Files\Google\Chrome\Application\chrome.exe',
        r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
        r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
        r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
    ]
    return next((p for p in fixed if Path(p).exists()), None)


def wait_for(port: int, timeout: float = 45.0) -> bool:
    """Uvicorn plus the first torch import can take a few seconds; poll rather
    than sleeping a guessed amount and opening a window on a dead port."""
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            s.settimeout(0.4)
            if s.connect_ex(('127.0.0.1', port)) == 0:
                return True
        time.sleep(0.2)
    return False


def main() -> int:
    import server

    port = free_port()
    threading.Thread(target=server.serve,
                     kwargs={'host': '127.0.0.1', 'port': port},
                     daemon=True).start()

    url = f'http://127.0.0.1:{port}/'
    if not wait_for(port):
        log(f'Backend did not start. Try: python server.py', file=sys.stderr)
        return 1

    def run_webview() -> int:
        import webview
        api = Api()
        api.window = webview.create_window(
            'T-Tracer', url,
            width=1280, height=860, min_size=(900, 620),
            background_color='#090A0D', js_api=api,
        )
        webview.start()
        return 0

    def run_chrome(chrome: str) -> int:
        import subprocess
        profile = APP_DIR / '.chrome-profile'      # keeps it out of the user's
        profile.mkdir(exist_ok=True)               # normal Chrome session

        # A brand-new profile makes Chrome show its first-run/welcome tab
        # alongside the app window - the reviewer: "it opens an empty chrom tab".
        # --no-first-run alone does not suppress it; Chrome looks for this
        # sentinel file in the profile directory and skips the whole first-run
        # flow when it exists.
        (profile / 'First Run').touch(exist_ok=True)

        log('Opening in an app window…')
        subprocess.run([
            chrome, f'--app={url}',
            f'--user-data-dir={profile}',
            '--window-size=1280,860',
            '--no-first-run',
            '--no-default-browser-check',
            '--disable-background-networking',   # kills the GCM registration
            '--disable-sync',                    # noise in the terminal too
            '--disable-features=Translate,ChromeWhatsNewUI',
        ], stderr=subprocess.DEVNULL)
        return 0

    # Windows tries Chrome/Edge FIRST - see the module docstring for why
    # pywebview's WinForms host is not trusted there. Every other platform
    # keeps the original preference: pywebview when it can actually render.
    chrome = find_chrome()
    if sys.platform == 'win32' and chrome:
        return run_chrome(chrome)
    if _has_webview_backend():
        return run_webview()
    if chrome:
        return run_chrome(chrome)

    import webbrowser
    log(f'No app-window backend found - opening a browser tab: {url}')
    log('Close this terminal window to quit.')
    webbrowser.open(url)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
