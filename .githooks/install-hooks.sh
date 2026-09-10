#!/bin/sh
# Point git at the tracked hooks in .githooks/ instead of .git/hooks/.
#
# Run once per clone:   sh .githooks/install-hooks.sh
#
# core.hooksPath is LOCAL config and cannot be committed, which is the one
# thing about hooks that always bites. Keeping the hooks themselves in a
# tracked directory means a fresh clone gets the hook FILES for free and only
# has to run this line to arm them - and the same line works on Windows,
# because Git for Windows ships the bash these hooks are written for.

set -eu

root=$(git rev-parse --show-toplevel)
cd "$root"

if [ ! -d .githooks ]; then
    echo "install-hooks: no .githooks/ directory at $root" >&2
    exit 1
fi

chmod +x .githooks/* 2>/dev/null || true
git config core.hooksPath .githooks

echo "Hooks armed. core.hooksPath -> $(git config core.hooksPath)"
echo
echo "Active:"
for h in .githooks/*; do
    [ -f "$h" ] || continue
    # Skip this installer. It lives alongside the hooks but is not one - git
    # only ever invokes files named after a real hook event.
    [ "$(basename "$h")" = "install-hooks.sh" ] && continue
    echo "  $(basename "$h")"
done
echo
echo "pre-push refuses to publish anything not on its allowlist. This repo is"
echo "public and the calibration corpus is customer artwork; three separate"
echo "leaks got through .gitignore before this existed."
