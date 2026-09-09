#!/usr/bin/env bash
# T-Tracer - one-shot install.
#
# Creates both venvs and pulls every dependency. Safe to re-run; it skips work
# that is already done.
#
# Two venvs, not one, and this is deliberate. The tracing pipeline is light
# (~500 MB). The lineart annotator needs torch (~1.4 GB) and is only used by
# the `nested` and `composite` strategies. Keeping them apart means the app
# starts fast and a torch problem cannot break plain tracing.
#
# The annotator venv lives OUTSIDE this folder on purpose: this project sits in
# a OneDrive tree, and OneDrive's skip list matches the name ".venv" exactly, so
# an in-project ".venv-lineart" would sync 1.4 GB to the desktop.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$HERE")"
LINEART_VENV="${LINEART_VENV:-$HOME/.venvs/png2svg-lineart}"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null || { echo "Need python3 on PATH."; exit 1; }
echo "Using $($PY --version)"

# Fail loudly and early rather than three minutes into a Rust build. The app's
# own deps are wheel-only below, but the tracing stack (numpy, scipy, opencv,
# scikit-image) needs wheels for this interpreter too, and a brand-new Python
# release routinely has none for weeks.
$PY - <<'VERCHECK' || exit 1
import sys
maj, mi = sys.version_info[:2]
if (maj, mi) < (3, 10):
    sys.exit(f"Python {maj}.{mi} is too old - need 3.10+.")
print(f"    Python {maj}.{mi} OK")
VERCHECK

# ---------------------------------------------------------------- main venv
if [ ! -x "$PROJECT/.venv/bin/python" ]; then
  echo "==> Creating tracing venv"
  "$PY" -m venv "$PROJECT/.venv"
fi
echo "==> Installing tracing + app dependencies"
"$PROJECT/.venv/bin/python" -m pip install --upgrade pip --quiet

# --only-binary=pydantic-core is scoped deliberately. pydantic-core is the one
# dependency here that builds through Rust/PyO3, and when no wheel matches the
# interpreter pip silently starts a compiler and fails minutes later inside a
# 125-line cargo log. This turns that into an immediate, readable "no wheel
# available". A blanket --only-binary=:all: is wrong: proxy_tools, which
# pywebview needs, is a pure-Python sdist and would be rejected.
if ! "$PROJECT/.venv/bin/python" -m pip install \
        -r "$PROJECT/requirements.txt" -r "$HERE/requirements.txt" \
        --only-binary=pydantic-core --quiet; then
  echo
  echo "Dependency install failed."
  echo "If it mentions a missing wheel, this Python is likely too new for one"
  echo "of the packages. Re-run against an older interpreter, e.g.:"
  echo "    PYTHON=python3.13 $0"
  exit 1
fi

# ------------------------------------------------------------ annotator venv
if [ ! -x "$LINEART_VENV/bin/python" ]; then
  echo "==> Creating annotator venv at $LINEART_VENV (this one is large)"
  "$PY" -m venv "$LINEART_VENV"
fi
echo "==> Installing torch (CPU) + controlnet_aux"
"$LINEART_VENV/bin/python" -m pip install --upgrade pip --quiet
# CPU wheels unless the machine has CUDA. The annotator is the biggest per-image
# cost (~6.7s on CPU, well under a second on CUDA) and lineart_backend.py
# autodetects, so a CUDA box gets the speedup with no config.
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "    NVIDIA GPU detected - installing CUDA torch"
  "$LINEART_VENV/bin/python" -m pip install torch torchvision --quiet
else
  "$LINEART_VENV/bin/python" -m pip install torch torchvision --quiet \
      --index-url https://download.pytorch.org/whl/cpu
fi
"$LINEART_VENV/bin/python" -m pip install controlnet_aux --quiet

# ------------------------------------------------------- warm the model cache
echo "==> Downloading the lineart model (~17 MB, once)"
"$LINEART_VENV/bin/python" - <<'PYEOF' || echo "    (model will download on first use instead)"
from controlnet_aux import LineartDetector
LineartDetector.from_pretrained("lllyasviel/Annotators")
print("    model ready")
PYEOF

# ----------------------------------------------------------------- launcher
cat > "$HERE/T-Tracer" <<EOF
#!/usr/bin/env bash
exec "$PROJECT/.venv/bin/python" "$HERE/main.py" "\$@"
EOF
chmod +x "$HERE/T-Tracer"

DESKTOP="$HOME/.local/share/applications"
if [ -d "$HOME/.local/share" ]; then
  mkdir -p "$DESKTOP"
  # Exec MUST be quoted. The freedesktop spec splits Exec on whitespace, so an
  # unquoted path containing spaces - and this project lives under
  # "(6) Workspace" - makes the menu try to run "/home/.../Documents/(6)" and
  # fail silently. It worked from a shell and did nothing from the launcher.
  cat > "$DESKTOP/t-tracer.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=T-Tracer
Comment=Turn a customer logo into a laser-ready SVG
Exec="$HERE/T-Tracer"
# Absolute path, and a PNG rather than the .ico: freedesktop icon themes do not
# read .ico, so pointing at icon.ico here leaves the menu entry blank. Without
# any Icon= line at all the launcher and the taskbar both fall back to a
# generic placeholder, which is half of why the app showed up as a globe.
Icon=$HERE/static/icon.png
Terminal=false
Categories=Graphics;
StartupNotify=true
EOF
  update-desktop-database "$DESKTOP" 2>/dev/null || true
  echo "==> Added to your applications menu"
fi

echo
echo "Done. Launch it with:  $HERE/T-Tracer"
echo "Or find 'T-Tracer' in your applications menu."
