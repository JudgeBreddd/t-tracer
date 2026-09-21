"""Review-only spatial colour regions for shaded and photographic artwork.

This module deliberately does not change the production candidates.  It is a
small SLIC + region-adjacency prototype intended to replace global k-means
components when local spatial context is more useful than a palette label.

The public API is :func:`build_spatial_regions`.  It returns a compact region
map, region statistics, and a deterministic adjacency graph.  A future Burn
Map UI can use ``region_map`` to toggle one connected region without changing
any other region with the same colour.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import time
from typing import Any

import cv2
import numpy as np
from skimage.color import rgb2lab
from skimage.segmentation import slic


@dataclass(frozen=True)
class SpatialRegion:
    """Summary for one final connected region."""

    region_id: int
    area: int
    mean_lab: tuple[float, float, float]
    mean_rgb: tuple[float, float, float]
    dark_fraction: float
    frame_fraction: float
    touches_envelope: bool
    initial_segments: int


@dataclass(frozen=True)
class SpatialRegionsResult:
    """SLIC/RAG output suitable for a review UI or polarity solver."""

    region_map: np.ndarray
    envelope: np.ndarray
    initial_labels: np.ndarray
    regions: tuple[SpatialRegion, ...]
    adjacency: tuple[tuple[int, int, int, float], ...]
    metrics: dict[str, Any]


def _normalise_envelope(rgb: np.ndarray, envelope: np.ndarray | None) -> np.ndarray:
    if envelope is None:
        # A conservative colour envelope for callers that do not already have
        # the candidate's alpha-aware silhouette.  This is intentionally not a
        # production replacement for candidates._silhouette_of.
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        border = np.concatenate((lab[0], lab[-1], lab[:, 0], lab[:, -1]), axis=0)
        dist = np.linalg.norm(lab - np.median(border, axis=0), axis=2)
        tol = max(10.0, float(np.percentile(dist, 72)))
        mask = dist > tol
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return mask.astype(bool)
    mask = np.asarray(envelope, dtype=bool)
    if mask.shape != rgb.shape[:2]:
        raise ValueError("envelope must have the same height and width as rgb")
    return mask


def _choose_segments(area: int, requested: int, target: int) -> int:
    if requested > 0:
        return int(max(target, requested))
    # A moderate oversegmentation gives the RAG enough local pieces to keep
    # lettering and holes, while the merge pass collapses smooth shading.
    return int(np.clip(max(target * 3, area // 2600), 180, 720))


def _initial_slic(rgb: np.ndarray, envelope: np.ndarray, n_segments: int,
                  compactness: float, sigma: float) -> np.ndarray:
    # SLIC is run in Lab and masked to the artwork extent.  ``start_label=1``
    # leaves zero available for pixels outside the envelope.
    labels = slic(
        rgb,
        n_segments=n_segments,
        compactness=float(compactness),
        sigma=float(sigma),
        start_label=1,
        mask=envelope,
        channel_axis=-1,
        convert2lab=True,
        enforce_connectivity=True,
        min_size_factor=0.25,
        max_size_factor=3.0,
    )
    labels = np.asarray(labels, dtype=np.int32)
    labels[~envelope] = 0
    # SLIC can leave holes in a masked image.  Giving each isolated pixel its
    # own initial label is safer than leaking a neighbouring region over it.
    if np.any(envelope & (labels == 0)):
        missing = envelope & (labels == 0)
        start = int(labels.max()) + 1
        labels[missing] = np.arange(start, start + int(missing.sum()), dtype=np.int32)
    return labels


def _adjacency(labels: np.ndarray) -> dict[tuple[int, int], int]:
    pairs: dict[tuple[int, int], int] = {}
    for first, second in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        valid = (first > 0) & (second > 0) & (first != second)
        if not valid.any():
            continue
        a = first[valid].astype(np.int64)
        b = second[valid].astype(np.int64)
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        unique, counts = np.unique(np.stack((lo, hi), axis=1), axis=0, return_counts=True)
        for pair, count in zip(unique, counts):
            key = (int(pair[0]), int(pair[1]))
            pairs[key] = pairs.get(key, 0) + int(count)
    return pairs


def _merge_rag(labels: np.ndarray, rgb: np.ndarray, envelope: np.ndarray,
               target_regions: int, max_lab_distance: float,
               preserve_dark: bool) -> tuple[np.ndarray, dict[str, Any], dict[tuple[int, int], int]]:
    """Merge the lowest-cost RAG edges until the target or safety floor."""
    ids = sorted(int(v) for v in np.unique(labels) if v > 0)
    lab = rgb2lab(rgb.astype(np.float32) / 255.0).astype(np.float32)
    flat_labels = labels.ravel()
    flat_lab = lab.reshape(-1, 3)
    flat_rgb = rgb.reshape(-1, 3).astype(np.float64)
    areas: dict[int, int] = {}
    sum_lab: dict[int, np.ndarray] = {}
    sum_rgb: dict[int, np.ndarray] = {}
    sum_dark: dict[int, int] = {}
    sum_frame: dict[int, int] = {}
    frame = np.zeros(envelope.shape, bool)
    band = max(1, min(4, min(envelope.shape) // 80))
    frame[:band] = True; frame[-band:] = True; frame[:, :band] = True; frame[:, -band:] = True
    flat_frame = frame.ravel()
    dark = lab[..., 0] < 28.0
    flat_dark = dark.ravel()
    for rid in ids:
        m = flat_labels == rid
        areas[rid] = int(m.sum())
        sum_lab[rid] = flat_lab[m].sum(axis=0)
        sum_rgb[rid] = flat_rgb[m].sum(axis=0)
        sum_dark[rid] = int(flat_dark[m].sum())
        sum_frame[rid] = int(flat_frame[m].sum())
    edges = _adjacency(labels)
    parent = {rid: rid for rid in ids}
    active = set(ids)
    members = {rid: 1 for rid in ids}

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != x:
            nxt = parent[x]; parent[x] = root; x = nxt
        return root

    def means(a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
        return sum_lab[a] / areas[a], sum_lab[b] / areas[b]

    def score(a: int, b: int, contact: int) -> float:
        ma, mb = means(a, b)
        distance = float(np.linalg.norm(ma - mb))
        dark_a = sum_dark[a] / max(1, areas[a]); dark_b = sum_dark[b] / max(1, areas[b])
        # Thin dark lettering is a semantic boundary even when its spatial
        # cell shares a long edge with a nearby shaded field.
        barrier = 0.0
        if preserve_dark and abs(dark_a - dark_b) > 0.35:
            barrier = 26.0
        contact_bonus = min(5.0, np.log1p(contact))
        return distance + barrier - 0.55 * contact_bonus

    heap: list[tuple[float, int, int, int]] = []
    for (a, b), contact in edges.items():
        heapq.heappush(heap, (score(a, b, contact), a, b, contact))
    merges = 0
    stop_score = 0.0
    while len(active) > max(1, int(target_regions)) and heap:
        candidate, a, b, contact = heapq.heappop(heap)
        a, b = find(a), find(b)
        if a == b or a not in active or b not in active:
            continue
        # An edge's contact can have changed after earlier merges; stale
        # entries are discarded and requeued with the current statistics.
        key = (min(a, b), max(a, b))
        current_contact = edges.get(key, 0)
        if current_contact != contact:
            heapq.heappush(heap, (score(a, b, current_contact), a, b, current_contact))
            continue
        if candidate > max_lab_distance:
            stop_score = candidate
            break
        # Merge into the lower id for deterministic region ids.
        keep, drop = (a, b) if a < b else (b, a)
        parent[drop] = keep
        active.remove(drop)
        areas[keep] += areas[drop]
        sum_lab[keep] += sum_lab[drop]; sum_rgb[keep] += sum_rgb[drop]
        sum_dark[keep] += sum_dark[drop]; sum_frame[keep] += sum_frame[drop]
        members[keep] += members[drop]
        del areas[drop], sum_lab[drop], sum_rgb[drop], sum_dark[drop], sum_frame[drop], members[drop]
        # Rewire the keep node's adjacency.  Stale heap entries are cheap and
        # are validated above, which keeps this loop simpler and deterministic.
        incident = [(x, c) for (x, y), c in list(edges.items()) if y == drop for x in [x]]
        incident += [(y, c) for (x, y), c in list(edges.items()) if x == drop for y in [y]]
        for other, c in incident:
            other = find(other)
            if other == keep or other not in active:
                continue
            old_key = (min(drop, other), max(drop, other)); edges.pop(old_key, None)
            new_key = (min(keep, other), max(keep, other))
            edges[new_key] = edges.get(new_key, 0) + int(c)
        for key in [key for key in list(edges) if drop in key]:
            edges.pop(key, None)
        for (x, y), c in list(edges.items()):
            if keep in (x, y):
                heapq.heappush(heap, (score(x, y, c), x, y, c))
        merges += 1
    roots = {rid: find(rid) for rid in ids}
    compact_roots = {root: index + 1 for index, root in enumerate(sorted(active))}
    compact = np.zeros(labels.shape, np.int32)
    inside = labels > 0
    compact[inside] = np.array([compact_roots[roots[int(rid)]] for rid in labels[inside]], dtype=np.int32)
    # Recompute final adjacency in compact ids for the caller.
    final_edges = _adjacency(compact)
    info = {"initial_regions": len(ids), "final_regions": len(active), "merges": merges,
            "merge_stop_score": float(stop_score), "max_lab_distance": float(max_lab_distance)}
    return compact, info, final_edges


def build_spatial_regions(rgb: np.ndarray, envelope: np.ndarray | None = None, *,
                          n_segments: int = 0, target_regions: int = 96,
                          compactness: float = 8.0, sigma: float = 0.4,
                          max_lab_distance: float = 42.0,
                          preserve_dark: bool = True) -> SpatialRegionsResult:
    """Build local, mergeable regions from an RGB image.

    ``target_regions`` is a ceiling, not a promise: very strong colour
    boundaries remain separate when merging would cross ``max_lab_distance``.
    The output never assigns pixels outside ``envelope`` to a region.
    """
    started = time.perf_counter()
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be an HxWx3 uint8 array")
    env = _normalise_envelope(rgb, envelope)
    if not env.any():
        empty = np.zeros(rgb.shape[:2], np.int32)
        return SpatialRegionsResult(empty, env, empty.copy(), tuple(), tuple(),
                                    {"initial_regions": 0, "final_regions": 0, "runtime_ms": 0.0})
    requested = _choose_segments(int(env.sum()), int(n_segments), int(target_regions))
    initial = _initial_slic(rgb, env, requested, compactness, sigma)
    region_map, merge_info, final_edges = _merge_rag(
        initial, rgb, env, int(target_regions), float(max_lab_distance), bool(preserve_dark)
    )
    lab = rgb2lab(rgb.astype(np.float32) / 255.0).astype(np.float32)
    regions: list[SpatialRegion] = []
    final_ids = [int(v) for v in np.unique(region_map) if v > 0]
    frame = np.zeros(env.shape, bool); band = max(1, min(4, min(env.shape) // 80))
    frame[:band] = True; frame[-band:] = True; frame[:, :band] = True; frame[:, -band:] = True
    for rid in final_ids:
        mask = region_map == rid; area = int(mask.sum())
        mean_l = np.mean(lab[mask], axis=0); mean_r = np.mean(rgb[mask], axis=0)
        regions.append(SpatialRegion(
            rid, area, tuple(float(x) for x in mean_l), tuple(float(x) for x in mean_r),
            float(np.mean(lab[..., 0][mask] < 28.0)), float(np.mean(frame[mask])),
            bool(np.any(mask & (env ^ cv2.erode(env.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)))),
            0,
        ))
    adjacency = tuple((a, b, c, float(np.linalg.norm(lab[region_map == a].mean(0) - lab[region_map == b].mean(0))))
                      for (a, b), c in sorted(final_edges.items()))
    metrics = dict(merge_info)
    metrics.update({"envelope_area": int(env.sum()), "envelope_fraction": float(env.mean()),
                    "smallest_region": min((r.area for r in regions), default=0),
                    "largest_region": max((r.area for r in regions), default=0),
                    "runtime_ms": round((time.perf_counter() - started) * 1000.0, 2),
                    "slic_segments_requested": requested})
    return SpatialRegionsResult(region_map, env, initial, tuple(regions), adjacency, metrics)


def render_region_map(result: SpatialRegionsResult) -> np.ndarray:
    """Render deterministic false-colour regions for review contact sheets."""
    palette = np.array([
        [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40],
        [148, 103, 189], [140, 86, 75], [227, 119, 194], [127, 127, 127],
        [23, 190, 207], [188, 189, 34],
    ], dtype=np.uint8)
    out = np.full((*result.region_map.shape, 3), 255, np.uint8)
    for rid in (int(v) for v in np.unique(result.region_map) if v > 0):
        out[result.region_map == rid] = palette[(rid - 1) % len(palette)]
    edge = np.zeros(result.region_map.shape, bool)
    edge[:, 1:] |= (result.region_map[:, 1:] != result.region_map[:, :-1]) & (result.region_map[:, 1:] > 0) & (result.region_map[:, :-1] > 0)
    edge[1:, :] |= (result.region_map[1:, :] != result.region_map[:-1, :]) & (result.region_map[1:, :] > 0) & (result.region_map[:-1, :] > 0)
    out[edge] = 0
    return out
