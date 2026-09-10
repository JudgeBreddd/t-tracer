"""paths.py - where private material lives, resolved in exactly one place.

THE RULE
--------
Everything this project must never publish lives under ONE directory,
`_private/`, and `.gitignore` denies that directory by default. Nothing else
in the tree is private.

WHY IT IS SHAPED THIS WAY
-------------------------
`.gitignore` is a DENYLIST: it names what must not be published, so anything
NEW is publishable until somebody remembers to add a rule. On this repo that
was not remembered three times:

    2026-09-06  _scripts/redgold/     31 customer logos   caught by hand
    2026-09-08  _scripts/_not-logos/  47 files, 27.7 MB   went public
    2026-09-08  HANDOFF.md            internal notes      went public

Each was a NEW name no existing rule described. Adding another rule fixes the
last incident and never the next one. A single default-denied directory fixes
every future one, because a file created inside it is covered the moment it
exists - nobody has to predict its name.

The pattern is borrowed from the reviewer's dashboard project, which hit the
identical bug three times with OAuth token files and settled on exactly this
fix: ignore the whole data/ directory rather than naming each new token file.

WHAT STAYS OUT OF _private/
---------------------------
`CLAUDE.md` and `CONTEXT.md` stay at the project root, denied by exact name.
They are fixed, well-known filenames that never change - the `.env` case,
where a denylist genuinely works. The directory rule exists for material whose
names churn, which is everything else.

OVERRIDE
--------
`TT_PRIVATE_DIR` points the whole private tree somewhere else - another disk,
a location outside the OneDrive sync, or a throwaway path in a test. Matches
the TT_* convention already used by TT_WORK_DIR, TT_REPO and TT_MAX_UPLOAD_MB.
"""

import os
from pathlib import Path

# _engine/paths.py -> _engine -> the project root
ENGINE = Path(__file__).resolve().parent
PROJECT = ENGINE.parent

PRIVATE = Path(os.environ.get('TT_PRIVATE_DIR', PROJECT / '_private'))

# The named areas inside it. Kept as constants so a later move is one edit here
# rather than a grep across the codebase - which is the whole point of the file.
CORPUS = PRIVATE / 'corpus'              # the calibration corpus, customer artwork
CANDIDATES = PRIVATE / 'candidates'      # traced output + picks.jsonl
PICKS = CANDIDATES / 'picks.jsonl'       # calibration ground truth
RUNS = PRIVATE / 'runs'                  # per-batch output, the churn
STAGES = PRIVATE / 'stages'
DELIVERABLES = STAGES / '06_deliverables'
INBOX = PRIVATE / 'inbox'
NOTES = PRIVATE / 'notes'                # HANDOFF.md, the fix list, working notes

__all__ = [
    'ENGINE', 'PROJECT', 'PRIVATE', 'CORPUS', 'CANDIDATES', 'PICKS',
    'RUNS', 'STAGES', 'DELIVERABLES', 'INBOX', 'NOTES',
]
