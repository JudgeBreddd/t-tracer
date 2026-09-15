# Laser output rules — the XCS/LightBurn contract

The standing constraints every stage's output has to satisfy. Referenced from `02_decompose` (shapes the vocabulary of what counts as a "part"), `03_build-library` and `04_assemble` (what "clean" actually means), and implicitly checked at `05_verify`.

## Contrast and fill

- Exactly two colors, absolute: `#000000` (black — engrave/vaporize) and `#FFFFFF` (white — masked/knockout/bare material). No grays, no anti-aliased edges, no gradients.
- Every filled region is a closed path. No open strokes standing in for fills.
- **Which element gets black vs. white is decided per-job in `04_assemble`'s `color-map.md`**, not fixed here and not baked into `library/` shapes (those stay color-agnostic geometry — see `library/CONTEXT.md`). This rule says the output must reduce to exactly two colors; it doesn't say which elements land on which side of that split.

## Geometry

- No overlapping or duplicate paths — each region traced/drawn exactly once. Overlapping geometry is the single biggest cause of laser stutter and double-passes.
- Low node count relative to the shape's actual complexity — a circle is one clean arc-based path, not a 200-point polygon approximation.
- Layer/z-order matters: elements are ordered furthest-background to closest-foreground so nothing clips incorrectly on import.

## Text

- Text is always flattened to real vector paths (font glyph outlines) before it reaches `assembled.svg` — never a live `<text>`/`<textPath>` element. XCS/LightBurn handle outlined text far more reliably than live text objects, and this makes the output font-independent.

## Source

Carried forward from the original idea draft's "Contrast & Layering Strategy" (see `../_archive/original-idea-draft.md`) — the rule itself held up under this pipeline's redesign even though the generation approach around it changed.
