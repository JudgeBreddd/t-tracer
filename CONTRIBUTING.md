# Contributing

## The most valuable bug report

**An image it fails on.** Roughly 1 logo in 14 defeats every strategy, and
those are the ones worth having. Include:

- the source image (only if you own it or it is freely distributable)
- which strategy came closest
- what is wrong with it, in plain words — "the chevrons vanished", "the centre
  went solid black", "the letters have blobs on them"

Plain description beats precise terminology here. Every real defect this
project has fixed started as a sentence like one of those.

## Running it from source

```bash
git clone https://github.com/JudgeBreddd/t-tracer.git
cd t-tracer/app
./install.sh          # Linux/macOS
./T-Tracer
```

Windows: `powershell -ExecutionPolicy Bypass -File app\install.ps1`

Two virtual environments get created. The project `.venv` runs the tracing
pipeline; a second one holds PyTorch and the ControlNet lineart annotator,
which only the `nested` and `composite` strategies need. They are separate so
the app starts fast and a broken torch cannot take plain tracing down with it.

## Arm the hooks first

```bash
sh .githooks/install-hooks.sh
```

Once per clone. It points `core.hooksPath` at the tracked `.githooks/`
directory, which git cannot do for you — hook config is local by design.

`pre-push` then refuses to publish anything not on its allowlist. This repo is
public and the calibration corpus it is developed against is real customer
artwork, so the check is deliberately strict: a path allowlist, an image test
by magic bytes rather than extension, and a 2 MB size cap. If you add a genuine
new source file, add its path to the allowlist in `.githooks/pre-push` in the
same commit.

## Adding a strategy

A strategy is one function: RGB array in, boolean ink mask out.

```python
def strat_mine(rgb, alpha=None):
    ...
    return mask          # True where the laser should burn
```

Register it in `STRATEGIES` in `_engine/candidates.py`. Add it to `OPTIONAL`
instead if it is experimental — the default sheet is capped at what a person
can compare at a glance, and strategies get demoted off it when they stop
winning.

**Document what it is for, and document it when it fails.** The source carries
several strategies that lost, with the measurements that killed them, because
a recorded negative result is worth more than a deleted one.

## Things worth knowing before you change anything

- **Determinism is load-bearing.** `cv2.kmeans` uses a random init, so the seed
  is set immediately before every call. Per-process seeding is not enough: a
  worker handles several images and the RNG advances between them. If you add a
  call that uses OpenCV's RNG, seed it at the call site.
- **The score is a hint, not a verdict.** It finds the right neighbourhood and
  is wrong about the winner roughly one time in five. Do not tune anything to
  make the score agree with a handful of images — that destroys the only signal
  there is.
- **Cleanup steps are where detail dies.** Four separate defects in this
  project were a cleanup stage silently deleting real artwork — letter
  counters, a bird's toe, banner text, a gold outline — while the output still
  looked clean and still scored well. If something is missing, suspect
  `clean_mask` before the strategy.
- **Nesting carries information that size and threshold cannot.** It is the
  move behind three of the fixes here.

## Style

Match the surrounding code. Comments explain *why*, especially when the obvious
approach was tried first and failed — several comments in this codebase exist
to stop someone re-running an experiment that has already been settled.
