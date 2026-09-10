# Method Index — T-Tracer

One entry per strategy: the ordered steps it runs, and every parameter that
touches its result. Built to drive the parameter sweep, not to explain the code
— for *why* a method exists, read its docstring in `_scripts/candidates.py`.

**Complete:** 10 default + 6 optional + 3 archived. Written 2026-09-08.

---

## The headline finding

**A strategy is one function: RGB in, boolean ink mask out. Everything before
and after it is shared by all sixteen.** The shared pipeline carries ~12
parameters. Most methods own 0–4.

Total tunable surface: **12 shared + 34 per-method = 46 parameters**, and only
**8 shared + 34 per-method** change geometry at all.

Two more things the index surfaced:

- **Every method already resizes before it runs** (step 3, ahead of all
  strategies). The problem is not that some skip it — it is that `--min-dim`
  **defaults to 0, off.**
- **The exception is `neural` / `nested` / `composite`**, which hit
  `NEURAL_MAX_DIM = 1024` inside step 5 and never see the upscale.

---

## The shared pipeline (runs for EVERY strategy)

`run_one`, [candidates.py:1592](_scripts/candidates.py). In execution order:

| # | step | parameter | default | geometry? |
|---|---|---|---|---|
| 1 | load, flatten alpha onto white | — | — | — |
| 2 | downscale if long edge exceeds cap (INTER_AREA) | `--max-dim` | 2048 | **yes** |
| 3 | upscale if long edge below floor (LANCZOS4) | `--min-dim` | **0 = OFF** | **yes** |
| 4 | grayscale copy kept for scoring | — | — | — |
| 5 | **THE STRATEGY** → boolean ink mask | per-method | — | **yes** |
| 6 | global invert | `--invert` | off | **yes** |
| 7a | protected morphological close | `--close` | 2 | **yes** |
| 7b | morphological open | `--open` | 0 | **yes** |
| 7c | drop components below area fraction | `--min-area-frac` | 0.0004 | **yes** |
| 8 | grow nested counters to min size | `--min-island` | 0 = off | **yes** |
| 9 | degenerate gate: reject mask | *hardcoded* | <0.001 or >0.98 ink | gate |
| 10a | supersample (INTER_CUBIC) | `--scale` | 6 | **yes** |
| 10b | spline smoothing per point | `--smoothing` | 3.0 | **yes** |
| 10c | curve-fit tolerance | `--tol` | 0.6 | **yes** |
| 10d | discard contours under N points | *hardcoded* `min_points` | 16 | **yes** |
| 11 | emit SVG | `--height-mm` / `--bg` | 32.0 / off | no |
| 12 | rasterize emitted Beziers back | `--raster-ss` | 2 | no |
| 13 | score the render (not the mask) | — | — | no |
| 14 | rank across candidates, fidelity relative | *hardcoded* `loss_ratio` | 0.75 | no |

**Sweep targets: steps 2, 3, 7a–c, 8, 10a–d.** Steps 11–14 are output and
measurement — moving them changes the score without changing the artwork, which
is the dangerous kind of tuning.

### Shared helpers most methods call

| helper | what it does | parameter | default |
|---|---|---|---|
| `_background_color` | median colour of an N-px border ring | `border` | 4 |
| `_silhouette_of` | alpha if real, else Lab distance from bg | `lab_tol` | 14.0 |
| `_lightness` | CIELAB L* rescaled to a true 0–100 | — | — |
| `_floor_px` | stroke floor as a fraction of long edge | `frac` | 0.0022 |
| `_thicken_thin` | grow only sub-floor strokes, gap-protected | `keep_frac` | 0.5 |
| `_grow_counters` | grow nested islands to min visible size | `min_px` | via `--min-island` |
| `_flip_fields` | turn a filled field inside-out | 5 params, below | — |
| `NEURAL_MAX_DIM` | hidden downscale before the annotator | *constant* | **1024** |

---

# DEFAULT SHEET — 10 strategies

## otsu — 21 picks (most-picked in the project)

1. RGB → grayscale (BT.601)
2. Global Otsu on the histogram, minimising intra-class variance
3. Invert: dark = ink

**Own parameters: none.** The threshold is derived, not set. This makes otsu
the cleanest possible vehicle for the shared sweep — nothing of its own can
confound the reading.

**Fails when** two important regions share a lightness (red on navy). Not
fixable from any parameter; that is why `bgdist`/`nested` exist.

## silhouette — 13 picks

1. Silhouette: alpha if real, else Lab distance from background > `lab_tol`
2. MORPH_GRADIENT ring around it — **fixed 5×5 kernel, not scaled to image**
3. Otsu over `gray[sil]` only — interior detail, artwork-population threshold
4. Return `ring | inner`

| parameter | default |
|---|---|
| `lab_tol` | 14.0 |
| ring kernel | **hardcoded 5×5** |

## composite — 8 picks

1. `base = strat_otsu(rgb)`
2. `field = strat_nested(rgb, alpha)` — inherits every `nested` parameter
3. Candidate knockouts = `base & ~field`, per connected component
4. Skip components below `knock_frac` of canvas (speckle)
5. Skip unless host otsu region exceeds `solid_frac` of canvas (large field)
6. Skip if knockout exceeds `max_knock_frac` of its host (re-classifying, not detail)
7. Otherwise remove that ink from `base`

| parameter | default |
|---|---|
| `solid_frac` | 0.015 |
| `knock_frac` | 0.0006 |
| `max_knock_frac` | 0.5 |
| + all of `nested` | — |

**Ceiling: can only REMOVE ink from otsu.** Cannot recover dropped saturated
colour.

## kmeans — 6 picks

1. RGB → Lab, flatten to Nx3
2. K-means, `KMEANS_PP_CENTERS`, 4 attempts, 30 iters / eps 0.5, seed 0
3. Border band `b = max(1, min(4, h//4, w//4))`; majority cluster there = background
4. Ink = every other cluster

| parameter | default |
|---|---|
| `k` | 5 |
| attempts / iters / eps | 4 / 30 / 0.5 (hardcoded) |
| border band `b` | ≤4 px (hardcoded) |

## bgdist — 5 picks

1. RGB → Lab
2. `bg = _background_color(lab)`
3. Per-pixel Euclidean Lab distance from `bg`
4. Normalise 0–255, global Otsu on the distance map
5. Ink = above the cut

**Own parameters: none** (inherits `_background_color`'s `border=4`).

**Known defect:** decides per pixel against a *global* Otsu of its own distance
map, which puts red back on the background side — the failure it was built to fix.

## keyline — 4 picks (shipped a line-art emblem)

1. **Bilateral filter twice** (d=9, σ=75/75) — flatten shading before thresholding
2. `dark = _lightness(flat) < l_black`
3. `sil = _silhouette_of(flat, alpha, lab_tol)` — on the *flattened* image
4. **Abstain 1:** median lightness outside silhouette < 40 → dark stock, return empty
5. **Abstain 2:** `(dark & sil) / sil < 0.02` → no black drawing, return empty
6. Ring radius `r = max(1, round(0.0015 × long_edge))` — **scales with image**
7. Return `dark | MORPH_GRADIENT(sil, r)`

| parameter | default |
|---|---|
| `l_black` | 20.0 |
| `lab_tol` | 14.0 |
| `ring` | True |
| bilateral d / σcolor / σspace | 9 / 75 / 75 (hardcoded, applied **twice**) |
| dark-stock abstain threshold | 40.0 L* (hardcoded) |
| no-drawing abstain threshold | 0.02 (hardcoded) |
| ring radius fraction | 0.0015 (hardcoded) |

## keyfill — 0 picks (new 2026-09-07)

1. `ink = strat_keyline(...)`; if it abstained, pass the abstention through
2. `_thicken_thin(ink, _floor_px(rgb, floor_frac))`
   - distance-transform ink, select strokes with half-width < floor
   - dilate only those
   - **gap protection:** no white region may lose more than `keep_frac` of itself

| parameter | default |
|---|---|
| `floor_frac` | 0.0022 |
| `keep_frac` | 0.5 |
| + all of `keyline` | — |

## keyflip — 0 picks (new 2026-09-07)

1. `ink = strat_keyline(...)`; abstain passthrough
2. `_flip_fields(ink, ring_px = max(1, round(0.0015 × long_edge)))`
3. **Abstain 3:** if the flip changed nothing, return empty (never duplicates `keyfill`)
4. `_thicken_thin(...)` as `keyfill`

| parameter | default |
|---|---|
| `min_field_frac` | 0.02 |
| `min_knock_frac` | 0.05 |
| `ring_px` | scaled, 0.0015 × long edge |
| `core_frac` | 0.004 |
| `min_core_share` | 0.40 |
| + all of `keyfill` | — |

## nested — 0 picks

1. `_lineart_lines()` — **downscale to ≤1024**, annotator, Gaussian blur, hysteresis, dilate
2. Connected components of `~lines` → regions
3. Median Lab colour per region; regions under `min_region` inherit, never decide
4. Assign line pixels to nearest region (EDT) → build adjacency
5. Root at the region owning the most border pixels = background
6. **BFS the containment tree.** A region flips vs its parent iff Lab distance > `dE`; else inherits
7. Line pixels burn only where they differ from their host **and** are darker
8. **INTER_NEAREST upsample** back to working size

| parameter | default |
|---|---|
| `hi` / `lo` (hysteresis) | 0.15 / 0.04 — **no CLI flag** |
| `blur` | 1.0 — **no CLI flag** |
| `stroke` (dilation) | 1 — **no CLI flag** |
| `min_region` | 12 |
| `dE` | 12.0 |
| `NEURAL_MAX_DIM` | **1024, constant** |

## linework — 1 pick

1. `sil = _silhouette_of(rgb, alpha)`
2. K-means in Lab over silhouette pixels only, seed 0
3. Radii scale with the diagonal: `r_thin = 0.005×diag`, `r_stroke = 0.003×diag`
4. **Pass 1 (fill):** a cluster is filled if `dark` (centre L < median−12) **or**
   `line_like` (erosion leaves < 30% of its area)
5. Per-pixel backstop: also fill anything below median L − 25
6. **Pass 2 (outline):** MORPH_GRADIENT of each field, **minus** a guard band
   dilated from `fills` — stops halos closing letter counters
7. Return `(fills | rings) & sil`

| parameter | default |
|---|---|
| `k` | 6 |
| `r_thin` frac | 0.005 |
| `r_stroke` frac | 0.003 |
| dark threshold | median L − 12.0 |
| backstop threshold | median L − 25.0 |
| line_like erosion ratio | 0.30 |
| min cluster area | 20 px |

---

# OFF THE SHEET — 6 optional (`--strategies` only)

## neural — 1 pick. Demoted; `nested` supersedes it

1. `_lineart_lines()` — identical to `nested` step 1
2. Regions from `~lines`; per-region median L*
3. **1-D 2-means over region lightness, area-weighted, 25 iterations**
4. Darker cluster → ink
5. `ink |= lines` — **every boundary burns.** This is why its letters read
   fattened; `nested` step 7 is the fix
6. INTER_NEAREST upsample

Same parameters as `nested`, minus `dE`.

## plate — 1 pick. Demoted 2026-09-05: most elaborate, never won

1. Silhouette, then K-means in Lab
2. **Figure/ground first:** the cluster owning most of the silhouette *rim* is
   forced to be a field, however dark — position, not colour
3. Pass 1 fills: `dark or line_like`, excluding the ground cluster
4. **Pass 2 borders by inward offset** (`solid & ~erode(solid)`), with holes
   filled *first* so borders appear only at the outer perimeter
5. Where the source already draws a hairline at that rim (>35% overlap),
   thicken the real line instead of synthesizing a second one

| parameter | default |
|---|---|
| `k` | 6 |
| `border_frac` | 0.015 (≈0.5 mm at 32 mm) |
| existing-hairline ratio | 0.35 |

## inotsu — 0 picks. Off-sheet: differs from otsu by 0.15–3.4% of pixels

1. Artwork mask: alpha if real, else Lab distance > `lab_tol`
2. Otsu computed over `gray[art]` **only** — not the whole frame
3. Return `art & (gray < cut)`

`lab_tol` 14.0. Falls back to plain otsu if artwork < 16 px.

## triotsu — 0 picks. **TESTED 2026-09-06 AND IT FAILS**

1. Artwork mask as `inotsu`
2. `threshold_multiotsu(gray[art], classes=3)`
3. Ink = everything below the **upper** cut — mid tone joins the dark tone

`lab_tol` 14.0. Falls back to otsu on ValueError.

**Why it fails, and this is worth more than the strategy:** the same
mid-luminance gold is *ink* on one logo and a *field* on another. Indistinguishable
by luminance at any number of classes. The difference is structural — which is
`nested`'s question.

## sauvola — 0 picks

1. Grayscale
2. Sauvola local adaptive threshold (window forced odd)
3. Ink = below the local surface

`window` 51, `k` 0.25. Keeps 6pt text global otsu erases; pays in speckle.

## edges — 0 picks. Exists for photographs of physical patches

1. Grayscale, bilateral filter (d=7, σ=60/60)
2. Canny with median-relative thresholds (0.66×med, 1.33×med)
3. MORPH_CLOSE at `min_stroke` radius
4. Label `~closed`; fill only enclosed regions **below** `max_hole_frac` of
   canvas and not touching the border
5. Return `closed | fill`

| parameter | default |
|---|---|
| `min_stroke` | 2 |
| `max_hole_frac` | 0.05 |
| bilateral d / σ | 7 / 60 / 60 |
| Canny med multipliers | 0.66 / 1.33 |

---

# ARCHIVED — `_scripts/_archive/`

Superseded by `candidates.py`. Kept for the parameter history: **`--scale 6`
and `--smoothing 3` have been the defaults since v1 and have never been swept.**

## lineart trace v1.py

Fixed-threshold → supersample → contours → spline → SVG. No vtracer (direct
contour extraction beat it on clean line art).

`--threshold` 200, `--scale` 6, `--smoothing` 3, `--min-points` 20.

## multicolor trace v1.py

Auto-palette by frequency with a merge distance, quantize to N buckets, run the
smoothing engine **once per bucket** so each colour keeps its own shape.

`--colors` 4, `merge_dist` 24.0, `--scale` 6, `--smoothing` 3, `--min-points` 20.

## smoothing engine.py

Shared back end for both. Contains one parameter the current pipeline dropped:
**`sample_density=4`** in `extract_smoothed_contours`. `min_blob_size` 100.

---

# Sweep order (derived from the above)

Shared parameters swept once on **otsu**, which has none of its own:

1. `--min-dim` — currently OFF, and the only one with a measured effect so far
2. `--scale` / `--smoothing` / `--tol` — unswept since v1
3. `--close` / `--open` / `--min-area-frac`
4. `--max-dim`
5. `--min-island`

Then per-method knobs, cheapest first: `bgdist` (0) → `sauvola` (2) →
`inotsu`/`triotsu` (1) → `kmeans` (1) → `edges` (2) → `silhouette` (2) →
`linework` (7) → `plate` (3) → `nested` (6) → `keyline` (7) → `keyfill` (+2) →
`keyflip` (+5) → `composite` (3 + nested's).
