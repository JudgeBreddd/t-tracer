#!/usr/bin/env python3
"""T-Tracer - desktop launcher.

Double-click entry point. Starts the local API on a free loopback port, then
opens it in a real application window.

THREE WINDOW BACKENDS, tried in order, because none of them is available
everywhere:

1. **pywebview** - wraps the OS's own webview. Ideal on Windows (WinForms) and
   macOS (Cocoa), where a backend always exists.
2. **Chrome/Chromium in --app mode** - a chromeless standalone window: no tabs,
   no address bar, its own taskbar entry. Indistinguishable from a native app
   window for this purpose.
3. **The default browser** - a normal tab. Ugly but functional; better than
   refusing to start.

Why the fallback chain exists, learned the hard way: on Linux pywebview needs
PyGObject+WebKit2GTK or Qt+QtWebEngine, and **neither is pip-installable in
practice** - they are system packages. `pip install pywebview` therefore
succeeds on a bare Linux box and then `webview.start()` has nothing to render
into, so the app appears to launch and simply never opens a window. That is
exactly what happened on this machine. Detecting the backend BEFORE trying to
use it turns a silent hang into a working window.
"""
from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))


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
    """A Chromium-family browser that supports --app windows."""
    import shutil
    names = ['google-chrome-stable', 'google-chrome', 'chromium',
             'chromium-browser', 'brave-browser', 'microsoft-edge']
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
        print(f'Backend did not start. Try: python server.py', file=sys.stderr)
        return 1

    if _has_webview_backend():
        import webview
        api = Api()
        api.window = webview.create_window(
            'T-Tracer', url,
            width=1280, height=860, min_size=(900, 620),
            background_color='#090A0D', js_api=api,
        )
        webview.start()
        return 0

    chrome = find_chrome()
    if chrome:
        import subprocess
        profile = APP_DIR / '.chrome-profile'      # keeps it out of the user's
        profile.mkdir(exist_ok=True)               # normal Chrome session

        # A brand-new profile makes Chrome show its first-run/welcome tab
        # alongside the app window - the reviewer: "it opens an empty chrom tab".
        # --no-first-run alone does not suppress it; Chrome looks for this
        # sentinel file in the profile directory and skips the whole first-run
        # flow when it exists.
        (profile / 'First Run').touch(exist_ok=True)

        print('Opening in an app window…')
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

    import webbrowser
    print(f'No app-window backend found - opening a browser tab: {url}')
    print('Close this terminal window to quit.')
    webbrowser.open(url)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
