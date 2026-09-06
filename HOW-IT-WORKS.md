# How T-Tracer works

A technical tour, for anyone curious about the internals or thinking about
contributing. The short version lives in the [README](README.md); this is the
long one.

The design has one governing idea: **tracing a logo for engraving has no single
correct answer, so the tool does not pretend to have one.** Everything below
follows from that.

---

## 1. Why seven answers instead of one

The recurring failure in this problem space is a mid-luminance colour — a gold,
a red, an olive — that a threshold has to call either *ink* or *background*.

Both calls are correct, on different images:

- On one crest, a gold emblem is a **shape drawn on something**. It must
  become ink.
- On another, the gold shield is **the thing being drawn on**. It must stay
  bare, or the entire interior design is swallowed by a black blob.

Those two golds are indistinguishable by luminance — at any number of threshold
classes, over any population of pixels. This was tested rather than assumed:
a three-class Otsu variant (`triotsu`) is kept in the source *with its failure
documented*, because it looks like the obvious fix and is not one.

The difference between them is **structural**, not tonal. Some strategies below
attack the structure; others attack the colour; none wins everywhere. So the
pipeline runs all of them and shows you the results.

---

## 2. The pipeline

One image in, one folder of candidates out. Per strategy:

```
source image
  ├─ decode, flatten alpha onto white
  ├─ cap the long edge at --max-dim (default 2048)
  ▼
STRATEGY  ──────────────────►  boolean ink mask
  ▼
clean_mask()   protected close · speckle removal · stray-island drop
  ▼
mask_to_paths()   supersample ×6 · contour extraction · Bézier fitting
  ▼
paths_to_svg()   one <path>, fill-rule="evenodd", one fill colour
  ▼
rasterize()  ──►  hygiene.score()  ──►  metrics.json
```

### 2.1 Decode and cap

Alpha is flattened onto white before anything else — a transparent PNG and a
white-background JPEG of the same artwork should trace identically.

The long edge is capped at 2048 px by default. The catalogue this was built
against contains sources up to 134 megapixels (10807 × 12396), and with
supersampling enabled those never finish. The cap took one such image from
*unrunnable* to about ten seconds. Every image in the original calibration set
is under 2000 px, so the cap is a no-op on all prior calibration data.

### 2.2 The strategies

Each is a pure function from RGB to a boolean mask. Adding one means adding a
function and a dictionary entry.

| Strategy | Mechanism |
|---|---|
| `otsu` | Global luminance threshold. The baseline, and still the most reliable single strategy — shippable in 24 of 26 judged images. |
| `bgdist` | CIELAB distance from the detected background colour. Does not require the artwork to be darker than its surroundings. Quietly excellent: also 24 of 26. |
| `kmeans` | Colour clustering, k=5, darker clusters become ink. |
| `linework` | Decides by **stroke thickness, not colour**. A thin region already *is* a line, so fill it; a thick region is a field, so outline it. Built for crests where a gold outline must become a black outline. |
| `silhouette` | Everything that is not background, thresholded **inside the artwork only**. |
| `nested` | Ink as a 2-colouring of the region containment tree. |
| `composite` | `otsu` as the base, with `nested` allowed to remove ink from large solid fields. |

Reachable via `--strategies` but off the default sheet: `plate`, `neural`,
`edges`, `sauvola`, `inotsu`, `triotsu`. Each lost its place by failing to win
picks over real artwork, and the reasons are recorded in the source.

### 2.3 Two strategies worth explaining properly

**`silhouette` — the population matters more than the threshold.**

It computes its Otsu split over `gray[silhouette]` — only pixels *inside* the
detected artwork — rather than over the whole frame. On a wolf logo with a red
eye inside a black head:

| | value |
|---|---|
| red eye, mean grey | 80 |
| whole-image Otsu cut | **128** → 80 < 128, the eye becomes ink and vanishes into the head |
| interior-only Otsu cut | **71** → 80 > 71, the eye survives as a knockout |
| share of frame that is background | 82% |

The background is usually the largest and most extreme-valued region in a logo,
so including it drags a global threshold toward one end. Excluding it moved the
cut by 57 grey levels. The lesson generalises further than the strategy does:
**not what you measure, but which pixels you let into the population.**

**`nested` — ink as a graph 2-colouring.**

Instead of asking "is this pixel dark", it asks "is this region a different
colour from the region containing it". Background starts bare. A region flips
relative to its parent when their CIELAB distance exceeds a tolerance, and
inherits its parent's state when it does not. Contrast is then preserved *by
construction* at every nesting depth, whatever a region's absolute luminance is.

Region boundaries come from a ControlNet lineart annotator (Informative
Drawings, SIGGRAPH 2022 — deterministic and non-generative; nothing here
invents detail). Line pixels are assigned to their nearest region so adjacency
is well defined, and a BFS outward from the background region makes the
discovering neighbour the containing one.

`composite` then uses exactly one piece of that knowledge: `nested` can tell
that a mid-luminance field is a field, so it is allowed to punch holes in large
solid `otsu` regions — **and never to add ink**. A knockout must also be a
*minority* of the region it sits in, which is the premise restated rather than
an extra rule: wanting to remove most of a field means you are re-classifying
the field, not finding a detail inside it. Measured across judged images, the
wins sit at 0.14 and 0.23 of their host and the destructions at 0.70 to 1.00,
so the 0.5 boundary falls in a wide empty gap rather than being fitted.

### 2.4 Mask cleanup, and the bug that shaped it

`clean_mask()` closes hairline breaks, removes speckle, and drops stray
islands. The close is **protected**, and that protection is load-bearing.

A plain morphological close fills every white region narrower than its radius.
On a 386 px source that silently ate the white lettering out of a unit banner — those strokes are about 3 px wide and the radius was 2. The candidate
still looked clean and still scored well. The text was simply gone.

Size alone cannot separate a hairline seam from a letter stroke; both are small
and thin. What separates them is the **source**: a seam sits over dark pixels
that should have been ink, a letter sits over light pixels meant to stay bare.
So every white region the close wants to fill is put to that question, and only
the ones the source agrees about get filled.

This same failure — *a cleanup step destroying the fine detail that matters* —
has appeared four times in this project's history. It is the single most
recurrent bug class here, which is why `--open` defaults to **0** and speckle
is handled by a minimum-area fraction instead.

Thresholds are fractions of total ink area, not pixel counts, so the same rules
work on a 400 px source and a 4000 px one.

### 2.5 Vector emission

`mask_to_paths()` supersamples the mask ×6 with cubic interpolation *before*
extracting contours. This is load-bearing rather than cosmetic: it preserves
true stroke width on thin features that would otherwise be quantised away at
native resolution.

Contours come from `findContours` with `RETR_CCOMP` and `CHAIN_APPROX_NONE`
(every boundary point kept — the fitter wants them), then each contour is fitted
with cubic Béziers under a distance tolerance, and the control points are scaled
back into original image coordinates.

The SVG is deliberately minimal:

- **One `<path>`** holding every subpath.
- **`fill-rule="evenodd"`** so interior holes knock out correctly — this is what
  makes letter counters and knockouts work.
- **Exactly one fill colour.** The geometry *is* the job.
- **No background rectangle.** In XCS a white rectangle is another closed shape
  to reason about, not "nothing".
- Dimensions emitted in **millimetres** when `--height-mm` is given, so the
  artwork arrives at physical size instead of needing to be scaled by hand.

---

## 3. Scoring, and why you should not trust it too much

Each candidate is rasterised back from its **emitted geometry** and scored
against the **source image in greyscale** — deliberately not against the mask it
was traced from, so no strategy can grade its own homework.

Six measurements, matching the acceptance test this was built to:

| Measurement | Question |
|---|---|
| `components` | stray islands that will burn as specks |
| `gaps` | white channels where the source says two shapes should touch |
| `fusions` | distinct shapes merged into one, detected from the **source colour** the binarisation discarded |
| `jaggedness` | edge roughness |
| `self_intersections` | paths that cross themselves |
| `fidelity` | how much of the source artwork survived at all |

`finalize()` then scales hygiene by fidelity **relative to the best candidate on
that image**, so a candidate that scores beautifully by deleting most of the
artwork is caught rather than rewarded.

### The honest part

Measured over 26 judged images:

| | |
|---|---|
| top-scored candidate was the human's first pick | **16 of 42** |
| top-scored candidate was *shippable at all* | **21 of 26 (81%)** |
| shippable candidates per image | **4.9 average** |

Those two numbers together say something specific: **the scorer mis-orders good
options; it does not promote bad ones.** On a contact sheet where you are
looking at every tile anyway, that is nearly harmless — which is why known
defects in the ranking have been left unrepaired rather than tuned. The picture
is the decision; the score is a hint.

The second number was invisible until the pick log could record a **set**
instead of a single winner. Before that, four good candidates were being logged
as four losses on every image.

---

## 4. Determinism

Every recorded pick points at a specific SVG. If re-running an image can produce
different geometry, the calibration set quietly stops meaning what it says.

`cv2.kmeans` seeds its initial centres from OpenCV's global RNG, which made
`kmeans` and `linework` non-deterministic run to run — two identical parallel
runs differed on 2 of 10 images.

Two fixes failed before the right one. Setting the seed **once per process** is
not enough: a worker handles several images and the RNG advances between them,
so results depended on what a worker happened to process first. The seed is now
set **immediately before each `cv2.kmeans` call**. Verified at 18/18 identical
outputs across three runs.

---

## 5. Parallelism sized by memory, not cores

`--jobs 0` (the default) means *auto*, and auto reads free memory rather than
core count.

Measured footprints: a tracing worker holds **831 MB** steady; the torch
annotator subprocess peaks at **3197 MB**. The first version of this defaulted
to one worker per core — on a 12-core / 14 GB laptop that asks for roughly
48 GB. That is not thrashing, that is an OOM kill, and this tool is expected to
run alongside a slicer and a laser controller.

So `auto_jobs()` reads `MemAvailable` (Linux) or `GlobalMemoryStatusEx`
(Windows — dependency-free on purpose), subtracts a reserve for the user's own
applications and one annotator, divides by the per-worker footprint, and caps at
**half the cores** so the machine stays usable. On that same laptop with 9.8 GB
free it picks **3 workers, not 12**.

Alongside that:

- The annotator is **serialised machine-wide** by a lock file, so worker count
  can never multiply its 3.4 GB. A file lock rather than an in-process
  semaphore, so two separate terminal invocations cannot OOM each other.
- `OMP_NUM_THREADS=1` per worker. OpenCV and NumPy each spawn their own thread
  pools; N workers × M threads oversubscribes and runs *slower* than serial.
- Workers are `nice`d. Losing a few percent of throughput to keep the UI
  responsive is the right trade for a background batch.
- An explicit `--jobs N` above the safe value **warns and then obeys**. It is
  your machine.

Speedup measured at **3.0×** on a batch of ten (8:00 → 2:41), not 10×, because
wall time is bounded by the single slowest image. This does **not** speed up a
single image — that would need the strategy loop split instead, which is not
built.

---

## 6. The app

A local FastAPI service on loopback plus a static frontend — one HTML file, one
CSS file, one JS file, no build step.

```
main.py     window: pywebview → Chrome --app → browser tab
server.py   FastAPI on 127.0.0.1, imports run_one from candidates.py
static/     the UI
```

**The backend imports the pipeline rather than reimplementing it**, and calls it
with the CLI's exact defaults held in a single `cli_defaults()`. All seven
strategies are verified byte-identical between app and CLI. This matters because
the pick log is calibration ground truth and would stop meaning anything if the
app traced differently from the tool the picks were recorded with.

**Three window backends, tried in order.** On Linux, `pip install pywebview`
succeeds on a bare machine and then has no renderer — it needs GTK/WebKit or
Qt/QtWebEngine, neither of which is pip-installable in practice. The app
therefore *detects* a usable backend before trying to use it, falling through to
a chromeless Chrome `--app` window and finally a browser tab. Without the
detection the app starts and silently never opens a window.

**All state lives outside the repo**, in `~/.cache/t-tracer/`
(`%LOCALAPPDATA%` on Windows, `TT_WORK_DIR` to override): traced jobs,
`settings.json`, and the app's own pick log. This is not tidiness — the
development copy sits in a OneDrive tree, and every traced job writes roughly
fifteen files per image.

**History** is a `job.json` per job folder, rebuilt by scanning the work
directory at startup, so the folder itself stays the source of truth and a
deleted folder simply disappears. Paths are rebuilt from where the folder
actually *is* rather than trusted from the file — storing absolute paths once
orphaned every past job the moment the work directory moved.

---

## 7. Two rules about recording judgement

Both exist because the pick log is the only ground truth this project has.

**"None of these" is a first-class answer.** On a hard input every candidate can
be wrong, and a grid invites picking anyway. If least-bad is logged as a win,
the ranking gets calibrated on "least bad" as though it meant "good". Both the
CLI and the app can record that nothing was usable.

**Several candidates can be shippable, and the log has to say so.** When the
schema allowed only one winner, the reviewer compensated by picking
*strategically* — choosing a candidate specifically so a strategy would "get
recognised" — which is a rational response to a forced choice and poison for
anything learning from the file.

A related consequence in the app: statistics from app usage are kept in a
**separate file** from the calibration corpus and are never merged into it. What
gets dropped into a desktop app is not necessarily a logo at all, and a blurry
phone photo should not be allowed to move a number that describes how the tracer
performs on insignia.

---

## Contributing

The most useful bug report is **an image it fails on**, plus which strategy got
closest and what is wrong with it. Roughly 1 logo in 14 still defeats every
strategy; those are the interesting ones.

See [CONTRIBUTING.md](CONTRIBUTING.md).
