"""Focused checks for the isolated connected-region polarity prototype."""

import cv2
import numpy as np

from region_graph import RegionLimitError, _components, solve_region_polarity


def _nested_logo():
    image = np.full((180, 220, 3), 255, np.uint8)
    cv2.rectangle(image, (50, 40), (170, 140), (0, 0, 0), -1)
    cv2.rectangle(image, (70, 60), (150, 120), (30, 100, 200), -1)
    cv2.circle(image, (110, 90), 12, (240, 180, 40), -1)
    return image


def _ambiguous_logo():
    image = np.full((200, 200, 3), 255, np.uint8)
    cv2.circle(image, (100, 100), 60, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(image, (100, 100), 50, (30, 120, 210), -1, cv2.LINE_AA)
    cv2.rectangle(image, (65, 90), (135, 110), (250, 200, 20), -1, cv2.LINE_AA)
    return image


def test_dark_anchor_is_ink_and_solver_reports_nested_structure():
    result = solve_region_polarity(_nested_logo())

    dark = [region for region in result.regions
            if region.area > 100 and region.mean_lab[0] < 8]
    assert dark and all(region.solved_ink for region in dark)
    assert result.adjacency
    assert result.containment
    assert result.mask.shape == result.envelope.shape
    assert not (result.mask & ~result.envelope).any()


def test_ambiguous_midtones_expose_a_paired_interpretation():
    result = solve_region_polarity(_ambiguous_logo())

    assert result.abstained
    assert result.paired_mask is not None
    assert not (result.mask & result.paired_mask).any()
    assert np.array_equal(result.mask | result.paired_mask, result.envelope)
    assert result.metrics["paired"] is True


def test_solver_is_deterministic_and_regions_are_connected():
    image = _nested_logo()
    first = solve_region_polarity(image)
    second = solve_region_polarity(image)

    assert np.array_equal(first.mask, second.mask)
    assert np.array_equal(first.region_map, second.region_map)
    assert first.metrics == second.metrics
    for region in first.regions:
        pixels = first.region_map == region.region_id
        count, labels = cv2.connectedComponents(pixels.astype(np.uint8), connectivity=8)
        assert count == 2, (region.region_id, labels.max())


def test_noisy_component_explosion_abstains_before_allocating_region_masks():
    envelope = np.zeros((72, 72), bool)
    envelope[2:-2:3, 2:-2:3] = True
    labels = np.zeros(envelope.shape, np.int32)
    with np.testing.assert_raises(RegionLimitError):
        _components(labels, envelope, bg_label=-1)
