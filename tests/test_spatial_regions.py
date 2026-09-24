"""Focused tests for the isolated spatial SLIC/RAG review stage."""

import numpy as np

from spatial_regions import build_spatial_regions


def _fixture():
    rgb = np.full((180, 240, 3), 255, np.uint8)
    rgb[25:155, 30:210] = (38, 100, 180)
    rgb[53:124, 74:165] = (242, 196, 54)
    rgb[76:101, 92:148] = (20, 20, 20)
    env = np.zeros(rgb.shape[:2], bool)
    env[25:155, 30:210] = True
    return rgb, env


def test_regions_respect_envelope_and_target():
    rgb, env = _fixture()
    result = build_spatial_regions(rgb, env, target_regions=32)
    assert result.metrics["initial_regions"] > result.metrics["final_regions"]
    assert result.metrics["final_regions"] <= 32
    assert np.all(result.region_map[~env] == 0)
    assert len(result.regions) == result.metrics["final_regions"]


def test_regions_are_deterministic_and_keep_dark_keyline_separate():
    rgb, env = _fixture()
    first = build_spatial_regions(rgb, env, target_regions=32)
    second = build_spatial_regions(rgb, env, target_regions=32)
    assert np.array_equal(first.region_map, second.region_map)
    dark = [region for region in first.regions if region.dark_fraction > 0.7]
    assert dark, "dark keyline/detail should survive as a local region"


def test_distinct_local_regions_can_share_a_colour():
    rgb, env = _fixture()
    rgb[30:45, 40:58] = (242, 196, 54)
    rgb[130:145, 180:198] = (242, 196, 54)
    result = build_spatial_regions(rgb, env, target_regions=32)
    yellow = [region.region_id for region in result.regions if region.mean_rgb[0] > 180 and region.mean_rgb[1] > 130 and region.mean_rgb[2] < 120]
    assert len(yellow) >= 2
