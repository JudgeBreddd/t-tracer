"""Small, bounded Burn Map refinement engine used by the optional UI.

The normal candidate slate remains the default.  This module supplies a
human-in-the-loop rescue path for flat or layered artwork: spatial regions are
editable independently, while a few palette interpretations provide useful
starting points when the source polarity is ambiguous.
"""
from __future__ import annotations

import io
import itertools
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from candidates import mask_to_paths, paths_to_svg
from region_graph import RegionEvidence, RegionGraphResult, solve_region_polarity
from spatial_regions import build_spatial_regions


MAX_SOURCE_PIXELS = 40_000_000


def read_rgb(path: Path, max_dim: int = 800) -> tuple[np.ndarray, np.ndarray | None]:
    with Image.open(path) as image:
        w, h = image.size
        if int(w) * int(h) > MAX_SOURCE_PIXELS:
            raise ValueError(
                f"image has {int(w) * int(h):,} pixels; refinement is limited "
                f"to {MAX_SOURCE_PIXELS:,} source pixels"
            )
        if max_dim > 0 and max(h, w) > max_dim:
            # JPEG decoders can use this hint to avoid materialising the full
            # source. thumbnail() handles every format and preserves aspect.
            # JPEG decoders understand RGB draft mode; alpha is restored by
            # the explicit conversion below for formats that carry it.
            image.draft("RGB", (max_dim, max_dim))
            image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    alpha = rgba[..., 3]
    alpha_fraction = alpha[..., None].astype(np.float32) / 255.0
    rgb = (rgba[..., :3].astype(np.float32) * alpha_fraction
           + 255.0 * (1.0 - alpha_fraction)).round().astype(np.uint8)
    return rgb, None if np.all(alpha == 255) else alpha


def _png(rgb: np.ndarray) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), "RGB").save(stream, format="PNG", optimize=True)
    return stream.getvalue()


def mask_png(mask: np.ndarray) -> bytes:
    image = np.full((*mask.shape, 3), 255, np.uint8)
    image[mask] = 0
    return _png(image)


def _palette(n: int) -> np.ndarray:
    out = np.zeros((max(1, n), 3), np.uint8)
    for i in range(n):
        hsv = np.uint8([[[int((i * 0.61803398875 % 1) * 179), 180, 235]]])
        out[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return out


def structural_lines(region_map: np.ndarray, envelope: np.ndarray, width: int = 2) -> np.ndarray:
    edge = np.zeros(region_map.shape, bool)
    left, right = region_map[:, :-1], region_map[:, 1:]
    different = left != right
    edge[:, :-1] |= different & (left >= 0)
    edge[:, 1:] |= different & (right >= 0)
    top, bottom = region_map[:-1], region_map[1:]
    different = top != bottom
    edge[:-1] |= different & (top >= 0)
    edge[1:] |= different & (bottom >= 0)
    if width > 1:
        edge = cv2.dilate(edge.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))) > 0
    return edge & envelope


def _fallback(rgb: np.ndarray, alpha: np.ndarray | None) -> RegionGraphResult:
    """Connected quantised components for simple one-colour artwork."""
    quant = (rgb // 8).astype(np.uint8)
    white = np.all(rgb >= 248, axis=2)
    if alpha is not None:
        white |= alpha < 8
    region_map = np.full(rgb.shape[:2], -1, np.int32)
    records: list[RegionEvidence] = []
    next_id = 0
    for colour in np.unique(quant.reshape(-1, 3), axis=0):
        active = np.all(quant == colour, axis=2) & ~white
        count, labels, stats, _ = cv2.connectedComponentsWithStats(active.astype(np.uint8), 8)
        for component in range(1, count):
            area = int(stats[component, cv2.CC_STAT_AREA])
            if area < 4:
                continue
            pixels = labels == component
            rid = next_id
            next_id += 1
            region_map[pixels] = rid
            mean = cv2.cvtColor(np.asarray(np.mean(rgb[pixels], axis=0), np.uint8).reshape(1, 1, 3), cv2.COLOR_RGB2LAB)[0, 0]
            ink = float(mean[0]) < 35.0
            records.append(RegionEvidence(rid, -1, area, tuple(float(v) for v in mean),
                float(area / max(1, (~white).sum())), 0.0, float(ink), float(ink), False, False,
                0.2 if ink else 2.2, 2.2 if ink else 0.2, ink, 0.1))
    envelope = region_map >= 0
    mask = np.zeros(envelope.shape, bool)
    for row in records:
        if row.solved_ink:
            mask[region_map == row.region_id] = True
    return RegionGraphResult(mask, None, envelope, region_map, tuple(records), tuple(), tuple(), 0.1, True,
        {"regions": len(records), "confidence": 0.1, "abstained": True, "fallback": True,
         "ink_fraction": float(mask.mean()), "envelope_fraction": float(envelope.mean())})


def _spatial(rgb: np.ndarray, alpha: np.ndarray | None) -> tuple[RegionGraphResult, str]:
    try:
        legacy = solve_region_polarity(rgb, alpha)
    except (ValueError, cv2.error, TypeError):
        legacy = _fallback(rgb, alpha)
    if min(rgb.shape[:2]) < 24 or not legacy.envelope.any():
        return legacy, "legacy"
    try:
        spatial = build_spatial_regions(rgb, legacy.envelope, target_regions=96)
    except (ValueError, cv2.error, TypeError, RuntimeError, ImportError):
        return legacy, "legacy-fallback"
    if not spatial.regions:
        return legacy, "legacy-fallback"
    region_map = spatial.region_map.astype(np.int32, copy=True)
    region_map[region_map > 0] -= 1
    region_map[~spatial.envelope] = -1
    means = np.asarray([region.mean_lab for region in spatial.regions], dtype=np.float32)
    if len(means) <= 1:
        labels = np.zeros(len(means), dtype=np.int32)
    else:
        k = min(7, max(2, round(np.sqrt(len(means) / 2))))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        cv2.setRNGSeed(17)
        _, labels, _ = cv2.kmeans(means, k, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
        labels = labels.reshape(-1).astype(np.int32)
    seed = np.asarray(legacy.mask, dtype=bool)
    rows: list[RegionEvidence] = []
    for index, region in enumerate(spatial.regions):
        pixels = region_map == index
        fraction = float(np.mean(seed[pixels])) if pixels.any() else 0.0
        solved = fraction >= 0.5
        confidence = float(np.clip(abs(fraction - 0.5) * 2.0, 0.05, 0.99))
        rows.append(RegionEvidence(index, int(labels[index]), int(pixels.sum()), tuple(float(v) for v in region.mean_lab),
            float(pixels.sum() / max(1, spatial.envelope.sum())), float(region.frame_fraction),
            float(region.dark_fraction), float(region.dark_fraction), bool(region.touches_envelope),
            bool(region.frame_fraction > 0.45), float(1.0 - fraction), float(fraction), solved, confidence))
    mask = np.zeros(region_map.shape, bool)
    for row in rows:
        if row.solved_ink:
            mask[region_map == row.region_id] = True
    metrics = dict(legacy.metrics)
    metrics.update(spatial.metrics)
    metrics.update({"backend": "slic", "regions": len(rows), "confidence": float(np.mean([r.confidence for r in rows])),
                    "abstained": bool(legacy.abstained), "ink_fraction": float(mask.mean()),
                    "envelope_fraction": float(spatial.envelope.mean())})
    return RegionGraphResult(mask, legacy.paired_mask, spatial.envelope, region_map, tuple(rows), spatial.adjacency,
                             tuple(), float(metrics["confidence"]), bool(legacy.abstained), metrics), "slic"


class BurnMap:
    """Thread-safe per-image editor state.  All arrays are capped by ``max_dim``."""
    def __init__(self, source: Path, max_dim: int = 800):
        self.source = source.resolve()
        self.rgb, alpha = read_rgb(self.source, max_dim=max_dim)
        self.result, self.backend = _spatial(self.rgb, alpha)
        self.region_map = self.result.region_map
        self.lines = structural_lines(self.region_map, self.result.envelope)
        self.fill_state = {r.region_id: bool(r.solved_ink) for r in self.result.regions}
        self.initial_fill_state = dict(self.fill_state)
        self.overrides: set[int] = set()
        self.mask = self._rebuild()
        self.initial = self.mask.copy()
        self.palette = _palette(len(self.result.regions))
        self.regions = self._rows()
        self.proposals = self._build_proposals()
        import threading
        self.lock = threading.RLock()

    @property
    def height(self): return int(self.region_map.shape[0])
    @property
    def width(self): return int(self.region_map.shape[1])

    def _rebuild(self):
        mask = self.lines.copy()
        for rid, filled in self.fill_state.items():
            if filled:
                mask[self.region_map == rid] = True
        return mask & self.result.envelope

    def _rows(self):
        valid = self.region_map >= 0
        ids = self.region_map[valid]
        yy, xx = np.nonzero(valid)
        n = len(self.result.regions)
        count = np.bincount(ids, minlength=n)
        sx, sy = np.bincount(ids, weights=xx, minlength=n), np.bincount(ids, weights=yy, minlength=n)
        rows = []
        for region in self.result.regions:
            area = max(1, int(count[region.region_id]))
            row = asdict(region)
            row.update(centroid=[float(sx[region.region_id] / area), float(sy[region.region_id] / area)],
                       swatch=[int(v) for v in self.palette[region.region_id]])
            rows.append(row)
        return rows

    def _build_proposals(self):
        labels = sorted({int(r["colour_label"]) for r in self.regions})
        if not 2 <= len(labels) <= 7:
            return []
        by_label = {label: [r for r in self.regions if int(r["colour_label"]) == label] for label in labels}
        features = {}
        for label, rows in by_label.items():
            area = max(1, sum(int(r["area"]) for r in rows))
            light = sum(float(r["mean_lab"][0]) * int(r["area"]) for r in rows) / area
            dark = sum(float(r["dark_fraction"]) * int(r["area"]) for r in rows) / area
            frame = sum(float(r["frame_fraction"]) * int(r["area"]) for r in rows) / area
            features[label] = (light, dark, frame)
        options = []
        for bits in itertools.product((False, True), repeat=len(labels)):
            states = dict(zip(labels, bits)); score = 0.0; black = 0
            for label, rows in by_label.items():
                light, dark, frame = features[label]
                for row in rows:
                    area = int(row["area"])
                    score += area * ((max(0.0, (48.0 - light) / 48.0) + dark + 0.2 * dark - 1.2 * frame)
                                   if states[label] else (max(0.0, (light - 42.0) / 58.0) + frame + 0.8 * frame))
                    black += area if states[label] else 0
            options.append((score, states, black))
        options.sort(key=lambda item: (-item[0], tuple(sorted(item[1].items()))))
        out = []; seen = set(); envelope = max(1, int(self.result.envelope.sum()))
        for score, states, black in options:
            key = tuple(bool(states[label]) for label in labels)
            if key in seen: continue
            seen.add(key)
            out.append({"id": f"palette-{len(out) + 1}", "label": f"Interpretation {len(out) + 1}",
                        "score": round(float(score) / envelope, 4), "black_fraction": round(float(black) / envelope, 4),
                        "group_states": {str(label): bool(states[label]) for label in labels},
                        "changed_regions": sum(bool(states[int(r["colour_label"])]) != bool(r["solved_ink"]) for r in self.regions)})
            if len(out) == 3: break
        return out

    def region_at(self, x: int, y: int) -> int:
        if not (0 <= int(x) < self.width and 0 <= int(y) < self.height): return -1
        return int(self.region_map[int(y), int(x)])

    def toggle(self, rid: int):
        with self.lock:
            if rid not in self.fill_state: raise KeyError(rid)
            self.fill_state[rid] = not self.fill_state[rid]; self.overrides.add(rid); self.mask = self._rebuild()
            for row in self.regions:
                if row["region_id"] == rid: row["solved_ink"] = self.fill_state[rid]
            return self.fill_state[rid]

    def toggle_group(self, label: int):
        with self.lock:
            ids = [int(r["region_id"]) for r in self.regions if int(r["colour_label"]) == int(label)]
            if not ids: raise KeyError(label)
            target = not (sum(bool(self.fill_state[rid]) for rid in ids) >= len(ids) / 2)
            for rid in ids:
                if rid in self.overrides: continue
                self.fill_state[rid] = target
                for row in self.regions:
                    if row["region_id"] == rid: row["solved_ink"] = target
            self.mask = self._rebuild(); return target

    def apply_proposal(self, proposal_id: str):
        with self.lock:
            proposal = next((p for p in self.proposals if p["id"] == proposal_id), None)
            if proposal is None: raise KeyError(proposal_id)
            states = {int(k): bool(v) for k, v in proposal["group_states"].items()}
            for row in self.regions:
                rid = int(row["region_id"])
                if rid not in self.overrides and int(row["colour_label"]) in states:
                    self.fill_state[rid] = states[int(row["colour_label"])]
                    row["solved_ink"] = self.fill_state[rid]
            self.mask = self._rebuild(); return proposal

    def snapshot(self) -> dict[str, Any]:
        """Return the tiny mutable part of an editor for safe LRU eviction."""
        with self.lock:
            return {
                "fill_state": {int(rid): bool(value) for rid, value in self.fill_state.items()},
                "overrides": sorted(int(rid) for rid in self.overrides),
            }

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore edits only when every referenced region still exists."""
        with self.lock:
            saved = {int(rid): bool(value)
                     for rid, value in dict(snapshot.get("fill_state", {})).items()}
            if set(saved) != set(self.fill_state):
                return
            self.fill_state.update(saved)
            self.overrides = {int(rid) for rid in snapshot.get("overrides", ())
                              if int(rid) in self.fill_state}
            by_id = {int(row["region_id"]): row for row in self.regions}
            for rid, value in self.fill_state.items():
                by_id[rid]["solved_ink"] = bool(value)
            self.mask = self._rebuild()

    def groups(self):
        with self.lock:
            groups = {}
            for row in self.regions:
                label = int(row["colour_label"]); g = groups.setdefault(label, {"colour_label": label, "area": 0, "regions": 0, "black": 0, "mean_lab": np.zeros(3, float)})
                area = int(row["area"]); g["area"] += area; g["regions"] += 1; g["black"] += int(bool(row["solved_ink"]))
                g["mean_lab"] += np.asarray(row["mean_lab"], float) * area
            out = []
            for g in groups.values():
                g["mean_lab"] = (g["mean_lab"] / max(1, g["area"])).tolist()
                g["state"] = "BLACK" if g["black"] == g["regions"] else "WHITE" if not g["black"] else "MIXED"
                out.append(g)
            return sorted(out, key=lambda g: (-g["area"], g["colour_label"]))

    def state(self):
        with self.lock:
            metrics = dict(self.result.metrics); metrics.update({"backend": self.backend, "regions": len(self.regions),
                "ink_fraction": float(self.mask.mean()), "changed_regions": sum(self.fill_state[r] != self.initial_fill_state[r] for r in self.fill_state),
                "proposals": len(self.proposals)})
            return {"width": self.width, "height": self.height, "backend": self.backend, "metrics": metrics,
                    "proposals": [dict(item) for item in self.proposals], "groups": self.groups(),
                    "regions": [dict(row) for row in self.regions]}

    def assets(self):
        with self.lock:
            valid = self.region_map >= 0; colours = _palette(max(1, len(self.regions)))
            over = self.rgb.astype(np.float32); tint = np.zeros_like(over); tint[valid] = colours[self.region_map[valid]]
            over = np.where(valid[..., None], over * .48 + tint * .52, over).clip(0, 255).astype(np.uint8)
            edge = np.zeros(self.region_map.shape, np.uint8)
            edge[:, 1:] |= ((self.region_map[:, 1:] >= 0) & (self.region_map[:, 1:] != self.region_map[:, :-1])).astype(np.uint8)
            edge[1:] |= ((self.region_map[1:] >= 0) & (self.region_map[1:] != self.region_map[:-1])).astype(np.uint8)
            over[edge > 0] = 20
            overlay = Image.fromarray(over, "RGB")
            draw = ImageDraw.Draw(overlay); font = ImageFont.load_default()
            for row in sorted(self.regions, key=lambda r: -int(r["area"]))[:120]:
                x, y = row["centroid"]; draw.text((x + 3, y - 7), str(row["region_id"]), fill=(0, 0, 0), font=font)
            over = np.asarray(overlay)
            diff = self.rgb.copy(); diff[self.initial ^ self.mask] = (225, 55, 30)
            return {"source.png": _png(self.rgb), "overlay.png": _png(over), "result.png": mask_png(self.mask), "diff.png": _png(diff)}

    def svg(self) -> bytes:
        with self.lock:
            return paths_to_svg(mask_to_paths(self.mask, scale=6, smooth=3.0, tol=0.6, min_points=16),
                                self.width, self.height, ink="#000000").encode("utf-8")
