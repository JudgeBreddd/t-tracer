"""Experimental connected-region polarity solver.

This module is deliberately separate from :mod:`candidates`.  It keeps the
colour segmentation from ``kmeans-layered`` but makes the black/white choice
per connected region instead of per colour cluster.  The solver is a small
binary graph cut with deterministic unary anchors (frame/background and
near-black keyline) and Lab-aware adjacency costs.

The return value is useful to a review UI before it becomes an engine
strategy: it includes the envelope, region map, confidence, and a paired
interpretation when the evidence is weak.  A customer image never gets
written by this module; callers choose where review artifacts go.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import networkx as nx
import numpy as np

from candidates import (
    _lightness,
    _silhouette_of,
    _true_lab,
    strat_keyline,
    strat_kmeans_layered,
)


SOURCE = -1
SINK = -2
MAX_FULL_MASK_REGIONS = 192


class RegionLimitError(RuntimeError):
    """Raised before per-region full-frame masks could exhaust memory."""


@dataclass(frozen=True)
class RegionEvidence:
    """Features and solve-time evidence for one connected colour region."""

    region_id: int
    colour_label: int
    area: int
    mean_lab: tuple[float, float, float]
    area_fraction: float
    frame_fraction: float
    dark_fraction: float
    keyline_fraction: float
    touches_envelope: bool
    is_border_colour: bool
    unary_ink: float
    unary_background: float
    solved_ink: bool
    confidence: float


@dataclass(frozen=True)
class RegionGraphResult:
    """Output of :func:`solve_region_polarity`."""

    mask: np.ndarray
    paired_mask: np.ndarray | None
    envelope: np.ndarray
    region_map: np.ndarray
    regions: tuple[RegionEvidence, ...]
    adjacency: tuple[tuple[int, int, int], ...]
    containment: tuple[tuple[int, int], ...]
    confidence: float
    abstained: bool
    metrics: dict[str, Any]


def _components(labels: np.ndarray, envelope: np.ndarray, bg_label: int):
    """Split each k-means label into full-resolution connected regions."""
    region_map = np.full(labels.shape, -1, np.int32)
    records: list[dict[str, Any]] = []
    next_id = 0
    colour_labels = sorted(int(v) for v in np.unique(labels) if v >= 0)
    # Every record below owns a full-frame boolean mask. Preflight the number
    # of components before allocating any of them: JPEG noise can otherwise
    # create tens of thousands of masks and exhaust the process. Burn Map can
    # still use the returned envelope with its bounded SLIC backend.
    component_total = 0
    for colour_label in colour_labels:
        active = (labels == colour_label) & envelope
        if not active.any():
            continue
        count, _ = cv2.connectedComponents(active.astype(np.uint8), connectivity=8)
        component_total += max(0, int(count) - 1)
        if component_total > MAX_FULL_MASK_REGIONS:
            raise RegionLimitError(
                f"{component_total} connected regions exceed the safe limit "
                f"of {MAX_FULL_MASK_REGIONS}"
            )
    for colour_label in colour_labels:
        active = (labels == colour_label) & envelope
        if not active.any():
            continue
        n, component_labels, stats, centroids = cv2.connectedComponentsWithStats(
            active.astype(np.uint8), connectivity=8
        )
        for component_id in range(1, n):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area == 0:
                continue
            mask = component_labels == component_id
            region_map[mask] = next_id
            records.append(
                {
                    "id": next_id,
                    "colour_label": colour_label,
                    "area": area,
                    "centroid": tuple(float(v) for v in centroids[component_id]),
                    "mask": mask,
                    "is_border_colour": colour_label == bg_label,
                }
            )
            next_id += 1
    return region_map, records


def _merge_tiny_regions(
    rgb: np.ndarray,
    region_map: np.ndarray,
    records: list[dict[str, Any]],
    envelope: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Fold tiny Lab-near components into a neighbouring real region.

    Photographic gradients and JPEG ringing can make one colour cluster break
    into thousands of one-pixel islands.  Treating each island as a semantic
    decision makes the graph both slow and noisy.  Only a component below a
    resolution-scaled area floor and within a small Lab distance of a larger
    neighbour is folded; thin, strongly contrasting artwork remains a region.
    """
    if len(records) < 2:
        return region_map, records
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    means = {
        int(record["id"]): _true_lab(
            np.mean(lab[record["mask"]], axis=0, keepdims=True)
        )[0]
        for record in records
    }
    areas = {int(record["id"]): int(record["area"]) for record in records}
    floor = max(12, int(round(float(envelope.sum()) * 0.00008)))
    parent = {int(record["id"]): int(record["id"]) for record in records}
    neighbours: dict[int, list[tuple[int, int]]] = {int(record["id"]): [] for record in records}
    for first, second, contact in _adjacency(region_map):
        neighbours[first].append((second, contact))
        neighbours[second].append((first, contact))

    def find(item: int) -> int:
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != item:
            item, parent[item] = parent[item], root
        return root

    for rid in sorted(parent, key=lambda item: (areas[item], item)):
        if areas[rid] >= floor:
            continue
        candidates = []
        for other, contact in neighbours[rid]:
            root = find(other)
            if root == rid or areas.get(root, 0) < floor:
                continue
            distance = float(np.linalg.norm(means[rid] - means[root]))
            if distance <= 20.0:
                candidates.append((distance, -contact, -areas[root], root))
        if candidates:
            _, _, _, target = min(candidates)
            parent[rid] = target

    groups: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(find(int(record["id"])), []).append(record)
    if len(groups) == len(records):
        return region_map, records

    remap = {old: new for new, old in enumerate(sorted(groups))}
    compact = np.full(region_map.shape, -1, np.int32)
    old_to_new = {old: remap[find(old)] for old in parent}
    compact[region_map >= 0] = np.array(
        [old_to_new[int(value)] for value in region_map[region_map >= 0]], dtype=np.int32
    )
    merged: list[dict[str, Any]] = []
    for old_root in sorted(groups):
        members = groups[old_root]
        mask = np.zeros(region_map.shape, bool)
        for member in members:
            mask |= member["mask"]
        largest = max(members, key=lambda member: int(member["area"]))
        merged.append(
            {
                "id": remap[old_root],
                "colour_label": int(largest["colour_label"]),
                "area": int(mask.sum()),
                "centroid": tuple(float(v) for v in np.argwhere(mask).mean(axis=0)[::-1]),
                "mask": mask,
                "is_border_colour": bool(largest["is_border_colour"]),
            }
        )
    return compact, merged


def _adjacency(region_map: np.ndarray) -> tuple[tuple[int, int, int], ...]:
    """Return 4-neighbour contact counts, sorted for deterministic solving."""
    pairs: dict[tuple[int, int], int] = {}
    for first, second in (
        (region_map[:, :-1], region_map[:, 1:]),
        (region_map[:-1, :], region_map[1:, :]),
    ):
        valid = (first >= 0) & (second >= 0) & (first != second)
        if not valid.any():
            continue
        a = first[valid].astype(np.int64)
        b = second[valid].astype(np.int64)
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        for key, count in zip(
            *np.unique(np.stack([lo, hi], axis=1), axis=0, return_counts=True)
        ):
            pair = (int(key[0]), int(key[1]))
            pairs[pair] = pairs.get(pair, 0) + int(count)
    return tuple((a, b, pairs[(a, b)]) for a, b in sorted(pairs))


def _containment(region_map: np.ndarray, records: list[dict[str, Any]]) -> tuple[tuple[int, int], ...]:
    """Find region pairs where one region is a true hole inside another.

    ``RETR_CCOMP`` is used per connected component so an enclosed white island
    is not confused with the image frame.  The sampled hole pixel is looked up
    in the full region map; that makes the relation useful even when the
    enclosing and enclosed regions came from different k-means labels.
    """
    out: set[tuple[int, int]] = set()
    for record in records:
        rid = int(record["id"])
        ys, xs = np.where(record["mask"])
        if not ys.size:
            continue
        # Work inside the component's bounding box.  The original prototype
        # allocated and scanned a full-image hole mask for every region; on a
        # shaded 1600 px badge that made Burn Map initialization take minutes.
        # One pixel of padding preserves the exterior required by RETR_CCOMP.
        y0, y1 = max(0, int(ys.min()) - 1), min(region_map.shape[0], int(ys.max()) + 2)
        x0, x1 = max(0, int(xs.min()) - 1), min(region_map.shape[1], int(xs.max()) + 2)
        component = record["mask"][y0:y1, x0:x1].astype(np.uint8)
        contours, hierarchy = cv2.findContours(
            component, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        if hierarchy is None:
            continue
        local_regions = region_map[y0:y1, x0:x1]
        for contour_id, contour in enumerate(contours):
            if int(hierarchy[0, contour_id, 3]) < 0:
                continue
            hole = np.zeros(component.shape, np.uint8)
            cv2.drawContours(hole, contours, contour_id, 1, thickness=-1)
            child_ids = np.unique(local_regions[(hole > 0) & (local_regions != rid)])
            for child in child_ids:
                child = int(child)
                if child >= 0:
                    out.add((rid, child))
    return tuple(sorted(out))


def _edge_weights(
    rgb: np.ndarray,
    records: list[dict[str, Any]],
    adjacency: tuple[tuple[int, int, int], ...],
    containment: tuple[tuple[int, int], ...],
):
    """Build Lab-aware smoothness terms for the binary graph cut."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    means = {
        int(record["id"]): np.mean(lab[record["mask"]], axis=0)
        for record in records
    }
    areas = {int(record["id"]): int(record["area"]) for record in records}
    weights: dict[tuple[int, int], float] = {}
    for a, b, contact in adjacency:
        distance = float(np.linalg.norm(_true_lab(np.vstack([means[a], means[b]]))[0]
                                       - _true_lab(np.vstack([means[a], means[b]]))[1]))
        # Similar colours are usually anti-aliasing/shading and should share a
        # polarity. Strong colour boundaries still communicate region structure,
        # but do not force a black/white flip when no anchor supports it.
        colour_factor = 1.8 if distance < 12.0 else (0.85 if distance < 30.0 else 0.2)
        contact_factor = min(3.0, max(0.15, contact / max(1.0, np.sqrt(areas[a] * areas[b]))))
        weights[(a, b)] = 0.7 * colour_factor * contact_factor
    for parent, child in containment:
        key = tuple(sorted((parent, child)))
        # A hole boundary is meaningful but less reliable than a long shared
        # boundary, so it nudges rather than dominates the cut.
        weights[key] = max(weights.get(key, 0.0), 0.22)
    return tuple((a, b, float(weights[(a, b)])) for a, b in sorted(weights))


def _unary_evidence(
    rgb: np.ndarray,
    alpha: np.ndarray | None,
    envelope: np.ndarray,
    region_map: np.ndarray,
    records: list[dict[str, Any]],
):
    """Compute anchor costs without turning lightness into a hard threshold."""
    flat = cv2.bilateralFilter(cv2.bilateralFilter(rgb, 9, 75, 75), 9, 75, 75)
    lightness = _lightness(flat)
    dark = (lightness < 20.0) & envelope
    try:
        keyline = strat_keyline(rgb, alpha=alpha, ring=False) & envelope
    except (cv2.error, ValueError):
        keyline = np.zeros(envelope.shape, bool)
    image_area = float(envelope.sum() or 1)
    h, w = envelope.shape
    frame = np.zeros(envelope.shape, bool)
    frame[: max(1, min(3, h // 20)), :] = True
    frame[-max(1, min(3, h // 20)) :, :] = True
    frame[:, : max(1, min(3, w // 20))] = True
    frame[:, -max(1, min(3, w // 20)) :] = True
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    for record in records:
        mask = record["mask"]
        area = max(1, int(record["area"]))
        frame_fraction = float((mask & frame).sum()) / area
        dark_fraction = float((mask & dark).sum()) / area
        keyline_fraction = float((mask & keyline).sum()) / area
        mean_lab = _true_lab(np.mean(lab[mask], axis=0, keepdims=True))[0]
        light_score = float(np.clip((50.0 - mean_lab[0]) / 25.0, -1.0, 1.0))

        # Anchor magnitudes are deliberately bounded.  Mid-luminance colour
        # remains undecided unless shape and graph evidence support it.
        ink_evidence = 0.28 * light_score + 4.5 * dark_fraction + 2.0 * keyline_fraction
        background_evidence = 0.0
        if record["is_border_colour"]:
            # The border-majority k-means label is the strongest available
            # background prior even when the silhouette has trimmed the
            # literal frame away.  A frame contact adds the hard part.
            background_evidence += 1.1
            if frame_fraction > 0.02:
                background_evidence += 5.0 * min(1.0, frame_fraction * 4.0)
        if frame_fraction > 0.40 and not record["is_border_colour"]:
            background_evidence += 0.6
        # NetworkX receives costs: high ink evidence must make the ink side
        # cheap, not expensive.  A neutral region costs 2.5 either way; this
        # leaves the Lab-aware graph terms able to propagate an anchor without
        # turning a mid-luminance colour into a hard threshold.
        ink_cost = max(0.05, 2.5 - float(max(0.0, ink_evidence)))
        background_cost = max(0.05, 2.5 - float(max(0.0, background_evidence)))
        record.update(
            {
                "mean_lab": tuple(float(v) for v in mean_lab),
                "frame_fraction": frame_fraction,
                "dark_fraction": dark_fraction,
                "keyline_fraction": keyline_fraction,
                "area_fraction": area / image_area,
                "touches_envelope": bool((mask & (envelope ^ cv2.erode(envelope.astype(np.uint8), np.ones((3, 3), np.uint8).astype(np.uint8))).astype(bool)).any()),
                "unary_ink": ink_cost,
                "unary_background": background_cost,
            }
        )


def _solve_cut(records: list[dict[str, Any]], edge_weights):
    """Solve a binary submodular energy using NetworkX's deterministic cut."""
    graph = nx.DiGraph()
    graph.add_node(SOURCE)
    graph.add_node(SINK)
    for record in records:
        rid = int(record["id"])
        graph.add_edge(SOURCE, rid, capacity=float(record["unary_background"]))
        graph.add_edge(rid, SINK, capacity=float(record["unary_ink"]))
    for a, b, weight in edge_weights:
        graph.add_edge(a, b, capacity=float(weight))
        graph.add_edge(b, a, capacity=float(weight))
    _, partition = nx.minimum_cut(graph, SOURCE, SINK, capacity="capacity")
    source_side, _ = partition
    return {int(record["id"]): int(record["id"]) in source_side for record in records}


def _confidence(records, edge_weights, solved):
    """Return per-region and area-weighted confidence from the chosen energy."""
    neighbours: dict[int, list[tuple[int, float]]] = {int(r["id"]): [] for r in records}
    for a, b, weight in edge_weights:
        neighbours[a].append((b, weight))
        neighbours[b].append((a, weight))
    total_area = float(sum(int(r["area"]) for r in records) or 1)
    weighted = 0.0
    for record in records:
        rid = int(record["id"])
        score_ink = float(record["unary_ink"])
        score_bg = float(record["unary_background"])
        for other, weight in neighbours[rid]:
            if solved[other]:
                score_ink += weight
            else:
                score_bg += weight
        margin = abs(score_ink - score_bg)
        confidence = float(np.clip(margin / (score_ink + score_bg + 1e-6), 0.0, 1.0))
        record["confidence"] = confidence
        record["solved_ink"] = bool(solved[rid])
        weighted += confidence * int(record["area"]) / total_area
    return float(weighted)


def solve_region_polarity(
    rgb: np.ndarray,
    alpha: np.ndarray | None = None,
    *,
    k: int = 0,
    confidence_floor: float = 0.34,
) -> RegionGraphResult:
    """Assign black/white polarity to connected full-resolution regions.

    Low-confidence images abstain and expose ``paired_mask`` as the opposite
    envelope interpretation.  This is intentional: a binary image cannot
    infer an artist's semantic foreground from colour alone in every case.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be an HxWx3 uint8 array")
    if alpha is not None and alpha.shape != rgb.shape[:2]:
        raise ValueError("alpha must have the same height and width as rgb")
    envelope = _silhouette_of(rgb, alpha)
    layered = strat_kmeans_layered(rgb, k=k, _return_labels=True)
    if not isinstance(layered, tuple) or len(layered) != 3:
        labels = bg_label = None
    else:
        _, labels, bg_label = layered
    if labels is None or bg_label is None or not envelope.any():
        empty = np.zeros(rgb.shape[:2], bool)
        return RegionGraphResult(empty, None, envelope, np.full(envelope.shape, -1, np.int32),
                                  tuple(), tuple(), tuple(), 0.0, True,
                                  {"reason": "no_colour_structure"})
    try:
        region_map, records = _components(labels, envelope, int(bg_label))
    except RegionLimitError:
        empty = np.zeros(rgb.shape[:2], bool)
        return RegionGraphResult(
            empty, None, envelope, np.full(envelope.shape, -1, np.int32),
            tuple(), tuple(), tuple(), 0.0, True,
            {"reason": "component_limit", "regions": 0,
             "envelope_fraction": float(envelope.mean()), "abstained": True},
        )
    region_map, records = _merge_tiny_regions(rgb, region_map, records, envelope)
    if not records:
        empty = np.zeros(rgb.shape[:2], bool)
        return RegionGraphResult(empty, None, envelope, region_map, tuple(), tuple(), tuple(),
                                  0.0, True, {"reason": "no_regions"})
    adjacency = _adjacency(region_map)
    containment = _containment(region_map, records)
    _unary_evidence(rgb, alpha, envelope, region_map, records)
    edge_weights = _edge_weights(rgb, records, adjacency, containment)
    solved = _solve_cut(records, edge_weights)
    confidence = _confidence(records, edge_weights, solved)
    total_area = float(sum(int(record["area"]) for record in records) or 1)
    uncertain_area = sum(
        int(record["area"])
        for record in records
        if float(record["confidence"]) < confidence_floor
    ) / total_area

    mask = np.zeros(envelope.shape, bool)
    for record in records:
        if solved[int(record["id"])]:
            mask[record["mask"]] = True
    mask &= envelope
    paired = envelope & ~mask if confidence < confidence_floor or uncertain_area > 0.25 else None
    abstained = confidence < confidence_floor or uncertain_area > 0.25
    region_evidence = tuple(
        RegionEvidence(
            region_id=int(r["id"]),
            colour_label=int(r["colour_label"]),
            area=int(r["area"]),
            mean_lab=tuple(float(v) for v in r["mean_lab"]),
            area_fraction=float(r["area_fraction"]),
            frame_fraction=float(r["frame_fraction"]),
            dark_fraction=float(r["dark_fraction"]),
            keyline_fraction=float(r["keyline_fraction"]),
            touches_envelope=bool(r["touches_envelope"]),
            is_border_colour=bool(r["is_border_colour"]),
            unary_ink=float(r["unary_ink"]),
            unary_background=float(r["unary_background"]),
            solved_ink=bool(r["solved_ink"]),
            confidence=float(r["confidence"]),
        )
        for r in records
    )
    metrics = {
        "regions": len(records),
        "adjacency_edges": len(adjacency),
        "containment_edges": len(containment),
        "ink_fraction": float(mask.mean()),
        "envelope_fraction": float(envelope.mean()),
        "confidence": confidence,
        "uncertain_area_fraction": float(uncertain_area),
        "abstained": abstained,
        "paired": paired is not None,
    }
    return RegionGraphResult(
        mask,
        paired,
        envelope,
        region_map,
        region_evidence,
        adjacency,
        containment,
        confidence,
        abstained,
        metrics,
    )


def render_region_debug(result: RegionGraphResult) -> np.ndarray:
    """Render a compact false-colour region/polarity diagnostic."""
    h, w = result.region_map.shape
    out = np.full((h, w, 3), 255, np.uint8)
    palette = np.array(
        [[31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40],
         [148, 103, 189], [140, 86, 75], [227, 119, 194], [127, 127, 127]],
        np.uint8,
    )
    by_id = {r.region_id: r for r in result.regions}
    for rid, evidence in by_id.items():
        pix = result.region_map == rid
        colour = palette[rid % len(palette)] if evidence.solved_ink else np.array([240, 240, 240], np.uint8)
        out[pix] = colour
    edges = np.zeros((h, w), bool)
    edges[:, 1:] |= result.region_map[:, 1:] != result.region_map[:, :-1]
    edges[1:, :] |= result.region_map[1:, :] != result.region_map[:-1, :]
    out[edges] = (0, 0, 0)
    return out
