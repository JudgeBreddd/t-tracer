#!/usr/bin/env python3
"""
pick.py - T-Tracer

Review the contact sheets, choose the candidate to ship, and record why.

    ../.venv/bin/python candidates.py     # generate
    ../.venv/bin/python pick.py           # choose

For each image it opens the contact sheet, lists the candidates ranked by the
(provisional) score, and waits for a number. The chosen SVG is copied into
stages/06_deliverables/ under the job name. The decision is appended to
candidates/picks.jsonl.

Why the log matters more than the copy
--------------------------------------
The hygiene score has been wrong every time it has disagreed with the reviewer, and
each disagreement exposed a real defect in the metric rather than in his eye.
These rows are the only ground truth that exists for fixing that, so the log is
the point and the file copy is a convenience.

Two rules, both load-bearing:

1. **"none of these" is a first-class answer.** On a hard input every candidate
   can be wrong, and a grid invites picking anyway. If least-bad is logged as a
   win, the ranking gets calibrated on "least bad" as though it meant "good",
   and it will confidently start recommending work that would never be sent to
   a customer. `0` records null and ships nothing.

2. **A rejection wants a reason.** A null says the slate failed; a reason says
   which detector was wrong. Even one word ("hilt fused", "text mush") turns a
   rejection into a metric fix instead of just a low score.

3. **Several candidates can be shippable, and the log has to say so.** Added
   2026-09-05, when the pipeline got good enough that this became the normal
   case. Forcing one winner out of five good outputs records four false
   losses, and the reviewer was compensating by picking strategically -- "1, 2, 3 are
   all shippable. picking 3 so it gets recognized" -- which corrupts the
   signal in the other direction. Enter every shippable candidate; the first
   one entered is the one to ship. `pick` still holds that single choice so
   the 49 rows recorded before this change stay readable, and `shippable`
   carries the full set.
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANDIDATES = HERE / 'candidates'
SHEETS = CANDIDATES / '_sheets'
# The pick log is the project's only calibration ground truth, so it stays in
# ONE canonical file no matter which run produced the tiles being reviewed.
# --dir moves where candidates are read from; it never moves where picks land.
PICKS = HERE / 'candidates' / 'picks.jsonl'
DELIVERABLES = HERE.parent / 'stages' / '06_deliverables'


def load_picks():
    if not PICKS.exists():
        return {}
    out = {}
    for line in PICKS.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                row = json.loads(line)
                out[row['image']] = row
            except json.JSONDecodeError:
                continue
    return out


def append_pick(row):
    with PICKS.open('a') as f:
        f.write(json.dumps(row) + '\n')


def open_sheet(path):
    """Best effort - if no viewer is available the path is still printed, and
    the sheet can be opened by hand. Never let a missing viewer block a pick."""
    for cmd in (['xdg-open', str(path)], ['open', str(path)]):
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            return True
        except FileNotFoundError:
            continue
    return False


def parse_choice(ans, n, names=None):
    """Accept `3`, `1,2,3`, `1-4`, `all`, `all -6`, and strategy NAMES.

    Names matter as much as numbers: the reviewer is reading a contact sheet
    where each tile is captioned with its strategy, so `otsu,bgdist` is what
    they actually have in front of them. Typing the name also survives a
    re-ranking, which a number does not.

    Returns a list of 1-based indexes in the order given, or None if the input
    does not parse. Order matters: the first entry is the one to ship.
    """
    names = [x.lower() for x in (names or [])]

    def as_index(tok):
        tok = tok.strip().lower()
        if tok.isdigit():
            return int(tok)
        if tok in names:
            return names.index(tok) + 1
        hits = [i for i, nm in enumerate(names, 1) if nm.startswith(tok)]
        return hits[0] if len(hits) == 1 else None
    ans = ans.replace(' except ', ' -').strip().lower()
    drop = []
    if ans.startswith('all') and '-' in ans:
        head, _, tail = ans.partition('-')
        ans = head.strip()
        drop = [i for i in (as_index(x) for x in tail.replace(',', ' ').split())
                if i]
    if ans in ('all', 'a'):
        picks = [i for i in range(1, n + 1) if i not in drop]
        return picks or None
    out = []
    for part in ans.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part and all(x.strip().isdigit()
                               for x in part.split('-', 1)):
            lo, _, hi = part.partition('-')
            rng = range(int(lo), int(hi) + 1)
            if not all(1 <= i <= n for i in rng):
                return None
            out.extend(i for i in rng if i not in out)
        else:
            i = as_index(part)
            if i is None or not 1 <= i <= n or i in out:
                return None
            out.append(i)
    return out or None


def review(stem, out_dir, redo=False):
    metrics_path = out_dir / 'metrics.json'
    if not metrics_path.exists():
        return None
    metrics = json.loads(metrics_path.read_text())

    # Imported, not reimplemented: the sheet prints these numbers ON the tiles
    # now, so any divergence records the wrong strategy. See rank_candidates().
    sys.path.insert(0, str(HERE))
    from candidates import rank_candidates
    scored = [(sc.get('overall', -1), nm, sc)
              for nm, sc in rank_candidates(metrics)]
    if not scored:
        print(f'  {stem}: no scored candidates')
        return None

    sheet = SHEETS / f'{stem}.png'
    print(f'\n{"=" * 72}\n{stem}\n{"=" * 72}')
    if sheet.exists():
        open_sheet(sheet)
        print(f'  sheet: {sheet}')
    for i, (ov, name, v) in enumerate(scored, 1):
        print(f'   {i}. {name:<11} {ov:>5.1f}  {v["headline"]}')
    print('   0. none of these are shippable')
    print('   s. skip   q. quit')
    print('  Enter EVERY shippable one, best first. Numbers or names:')
    print('    3  |  1,2,3  |  otsu,bgdist  |  1-4  |  all  |  all -6  '
          '|  all except otsu')

    while True:
        try:
            ans = input('  pick > ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 'quit'
        if ans == 'q':
            return 'quit'
        if ans == 's':
            return None
        if ans == '0':
            note = input('  what was wrong with all of them? > ').strip()
            append_pick({'date': date.today().isoformat(), 'image': stem,
                         'pick': None, 'note': note, 'source': 'owner'})
            print('  recorded: none shippable')
            return 'none'
        idxs = parse_choice(ans, len(scored), [nm for _, nm, _ in scored])
        if idxs:
            names = [scored[i - 1][1] for i in idxs]
            if len(idxs) > 1:
                print(f'  {len(idxs)} shippable: {", ".join(names)}')
                first = input(f'  which would you reach for first? '
                              f'[{names[0]}] > ').strip().lower()
                if first:
                    match = [j for j, nm in enumerate(names)
                             if nm == first or (first.isdigit()
                                                and idxs[j] == int(first))]
                    if match:
                        j = match[0]
                        idxs.insert(0, idxs.pop(j))
                        names.insert(0, names.pop(j))
            note = input('  note (optional) > ').strip()
            name = names[0]
            DELIVERABLES.mkdir(parents=True, exist_ok=True)
            dst = DELIVERABLES / f'{stem}.svg'
            shutil.copy2(out_dir / f'{name}.svg', dst)
            append_pick({'date': date.today().isoformat(), 'image': stem,
                         # `pick` stays the single one to ship so every row
                         # written before 2026-09-05 reads the same way.
                         'pick': name, 'shippable': names,
                         'shippable_ranks': idxs,
                         'note': note, 'source': 'owner',
                         'rank': idxs[0], 'of': len(scored)})
            extra = (f'  (+{len(names) - 1} more shippable)'
                     if len(names) > 1 else '')
            print(f'  recorded: {name}  ->  {dst}{extra}')
            return name
        print('  ? enter e.g. 3 | 1,2,3 | 1-4 | all | all -6 | 0 | s | q')


def main():
    ap = argparse.ArgumentParser(description='Pick the candidate to ship.')
    ap.add_argument('--only', help='review just this image stem')
    ap.add_argument('--redo', action='store_true',
                    help='also review images that already have a recorded pick')
    ap.add_argument('--dir', dest='cand_dir', default=None,
                    help='review a different candidates folder (e.g. '
                         'candidates-redgold). Picks still log to '
                         'candidates/picks.jsonl.')
    args = ap.parse_args()

    global CANDIDATES, SHEETS
    if args.cand_dir:
        CANDIDATES = Path(args.cand_dir)
        if not CANDIDATES.is_absolute():
            CANDIDATES = HERE / CANDIDATES
        SHEETS = CANDIDATES / '_sheets'

    if not CANDIDATES.is_dir():
        print(f'No {CANDIDATES.name}/ folder. Run candidates.py first.')
        return
    PICKS.parent.mkdir(parents=True, exist_ok=True)

    done = load_picks()
    dirs = sorted(d for d in CANDIDATES.iterdir()
                  if d.is_dir() and not d.name.startswith('_'))
    if args.only:
        dirs = [d for d in dirs if d.name == args.only]

    reviewed = agreed = 0
    for d in dirs:
        if d.name in done and not args.redo:
            continue
        result = review(d.name, d)
        if result == 'quit':
            break
        if result:
            reviewed += 1
            if result != 'none':
                row = load_picks().get(d.name, {})
                if row.get('rank') == 1:
                    agreed += 1

    rows = [r for r in load_picks().values() if r.get('rank')]
    print(f'\n{reviewed} reviewed this session.')
    if not rows:
        return
    top1 = len([r for r in rows if r.get('rank') == 1])
    print(f'Ranking agreement: {top1}/{len(rows)} picks were the top-scored '
          f'candidate.')

    # The better diagnostic, available only once a row can hold a SET. Asking
    # "was the top-scored candidate shippable at all" separates a scorer that
    # is merely mis-ordering good options from one that is promoting bad ones.
    # The first is nearly harmless on a contact sheet; the second is not.
    multi = [r for r in rows if r.get('shippable')]
    if multi:
        ok = len([r for r in multi if 1 in (r.get('shippable_ranks') or [])])
        ships = sum(len(r['shippable']) for r in multi)
        print(f'Of {len(multi)} rows recording the full set: the top-scored '
              f'candidate was shippable in {ok}, and {ships/len(multi):.1f} '
              f'candidates were shippable per image on average.')
    print('Treat these as diagnostics on the score, not on the picks.')


if __name__ == '__main__':
    main()
