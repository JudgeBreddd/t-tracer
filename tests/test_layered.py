"""kmeans-layered: the colour split survives to the SVG, and the pieces the
strategy is built from behave as the docstrings claim."""
import numpy as np
import cv2
import pytest

import candidates as C
import hygiene as H


def _flat_logo(size=320):
    """White field, blue disc, yellow bar across it, black outline ring - four
    flat colours plus the anti-aliasing cv2 draws between them."""
    img = np.full((size, size, 3), 255, np.uint8)
    c = size // 2
    cv2.circle(img, (c, c), size // 3, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img, (c, c), size // 3 - 8, (60, 120, 220), -1, cv2.LINE_AA)   # L* ~52
    cv2.rectangle(img, (c - size // 4, c - 14), (c + size // 4, c + 14),
                  (250, 200, 20), -1, cv2.LINE_AA)
    return img


def test_layers_are_disjoint_and_one_per_colour():
    layers = C.strat_kmeans_layered(_flat_logo())
    assert len(layers) == 3, [C._hex(c) for _, c in layers]
    union = np.zeros(layers[0][0].shape, bool)
    for m, _ in layers:
        assert not (union & m).any(), 'layers overlap'
        union |= m
    fills = {C._hex(c) for _, c in layers}
    # black outline, blue disc, yellow bar - white is background, not a layer
    assert any(f.startswith('#0') for f in fills)
    assert '#FFFFFF' not in fills


def test_darkest_layer_is_drawn_last():
    layers = C.strat_kmeans_layered(_flat_logo())
    L = [cv2.cvtColor(np.asarray(c, np.uint8).reshape(1, 1, 3),
                      cv2.COLOR_RGB2LAB)[0, 0, 0] for _, c in layers]
    assert L == sorted(L, reverse=True)


def test_fixed_k_is_honoured_and_halo_bands_are_folded():
    # k=6 on a 4-colour image: the extra clusters are anti-aliasing ramps and
    # must fold away rather than become 1px layers.
    layers = C.strat_kmeans_layered(_flat_logo(), k=6)
    assert len(layers) == 3
    for m, _ in layers:
        assert m.sum() > 200


def test_two_colour_image_abstains():
    img = np.full((200, 200, 3), 255, np.uint8)
    cv2.circle(img, (100, 100), 60, (0, 0, 0), -1)
    assert C.strat_kmeans_layered(img) == []


def test_layered_svg_has_one_path_per_layer_with_its_fill():
    img = _flat_logo()
    layers = C.strat_kmeans_layered(img)
    traced = [(C.mask_to_paths(m, scale=2, smooth=3.0, tol=0.6), c) for m, c in layers]
    svg = C.paths_to_svg_layered(traced, 320, 320)
    assert svg.count('<path ') == 3
    for _, c in layers:
        assert f'fill="{C._hex(c)}"' in svg
    assert 'fill-rule="evenodd"' in svg


def test_between_finds_the_nearer_endpoint_and_ignores_off_axis():
    black = np.array([0.0, 0.0, 0.0])
    white = np.array([100.0, 0.0, 0.0])
    grey = np.array([70.0, 0.0, 0.0])
    green = np.array([50.0, -60.0, 40.0])
    assert C._between(grey, [black, white]) == 1        # nearer white
    assert C._between(grey, [black, green]) is None
    assert C._between(green, [black, white]) is None


def test_true_lab_undoes_opencv_packing():
    t = C._true_lab(np.array([[255.0, 128.0, 128.0], [0.0, 128.0, 128.0]]))
    assert np.allclose(t[0], [100, 0, 0]) and np.allclose(t[1], [0, 0, 0])


def test_score_layered_matches_single_score_shape():
    img = _flat_logo()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    layers = C.strat_kmeans_layered(img)
    rendered = []
    for m, c in layers:
        paths = C.mask_to_paths(m, scale=2, smooth=3.0, tol=0.6)
        rendered.append((C.rasterize(paths, 320, 320, ss=2), paths, c))
    sc = H.score_layered(rendered, gray, img, height_mm=32.0, ss=2, src_h=320)
    single = H.score(rendered[0][0], gray, rendered[0][1], 10, height_mm=32.0,
                     ss=2, src_h=320, src_rgb=img)
    assert set(single) <= set(sc)
    assert sc['layer_count'] == 3
    assert sc['fused'] if 'fused' in sc else sc['fusions'] == 0
    assert 0.0 <= sc['fidelity_color'] <= 1.0
    assert sc['penalties']['fused'] == 0.0


def test_layer_strays_ignore_islands_touching_another_colour():
    a = np.zeros((60, 60), bool)
    b = np.zeros((60, 60), bool)
    a[10:50, 10:30] = True            # big block
    b[10:50, 30:50] = True            # big block touching a
    a[52:54, 52:54] = True            # tiny island in open background -> stray
    b[20:22, 28:30] = True            # tiny island glued to block a -> design
    union = a | b
    ca = H._layer_components(a, union, cutoff_px=50)
    cb = H._layer_components(b, union, cutoff_px=50)
    assert ca['strays'] == 1
    assert cb['strays'] == 0


def test_memo_reuses_within_run_and_computes_outside():
    calls = []

    def fn(x, flag=False):
        calls.append(flag)
        return np.array([True])

    assert C._MEMO is None
    C._memo('t', fn, 1); C._memo('t', fn, 1)
    assert len(calls) == 2                      # no memo open -> computes
    C._MEMO = {}
    try:
        r1 = C._memo('t', fn, 1); r2 = C._memo('t', fn, 1)
        assert len(calls) == 3
        assert r1 is not r2                     # copies, never the cached array
        C._memo('t', fn, 1, flag=True)          # overrides always compute
        assert len(calls) == 4
    finally:
        C._MEMO = None


def test_rasterize_evenodd_hole_survives_buffer_reuse():
    outer = [(np.array([10., 10.]), np.array([90., 10.]), np.array([90., 90.]), np.array([10., 90.]))]
    # a 4-segment square, then a smaller square inside it
    def square(a, b):
        pts = [np.array([a, a]), np.array([b, a]), np.array([b, b]), np.array([a, b])]
        return [(pts[i], pts[i], pts[(i + 1) % 4], pts[(i + 1) % 4]) for i in range(4)]
    r = C.rasterize([square(10., 90.), square(30., 70.)], 100, 100, ss=1)
    assert r[20, 20] and not r[50, 50]


def test_layerlines_outlines_every_boundary_and_fills_dark():
    img = _flat_logo()
    ink = C.strat_layerlines(img)
    assert ink.dtype == bool and ink.shape == img.shape[:2]
    c = img.shape[0] // 2
    assert ink[c, c - img.shape[0] // 3 + 2]          # the black ring is filled
    assert ink[c - 14, c]                              # bar/disc boundary is a line
    assert not ink[c, c]                                # yellow bar interior stays bare
    assert not ink[c - 60, c]                           # blue disc interior stays bare (L* 40+)


def test_layerlines_abstains_on_one_colour_art():
    img = np.full((200, 200, 3), 255, np.uint8)
    cv2.circle(img, (100, 100), 60, (0, 0, 0), -1)
    assert not C.strat_layerlines(img).any()


def test_default_sheet_is_five():
    assert list(C.STRATEGIES) == ['otsu', 'bgdist', 'kmeans', 'composite', 'layerlines']
    assert 'silhouette' in C.OPTIONAL and 'kmeans-layered' in C.OPTIONAL
    assert C.LAYERED == {'kmeans-layered'}


def test_progress_sink_is_silent_when_unset_and_throttled_when_set():
    seen = []
    assert C._PROGRESS is None
    C._progress('x', lambda: seen.append(1) or 'img')
    assert seen == []                                   # image never built
    C._PROGRESS = lambda step, img: seen.append((step, img))
    C._PROGRESS_T = 0.0
    try:
        C._progress('a', lambda: 'A'); C._progress('b', lambda: 'B')
        assert seen == [('a', 'A')]                      # second frame inside 1/3 s dropped
    finally:
        C._PROGRESS = None


def test_fused_extent_is_the_minority_population_not_the_whole_component():
    """One big component whose second colour is a small minority must report a
    SMALL fused area. Before 2026-09-15 it reported the whole component, so
    any single-blob trace scored 1.000 no matter how little was wrong."""
    img = np.zeros((200, 200, 3), np.uint8)
    img[:, :] = (240, 240, 240)
    img[40:160, 40:160] = (20, 20, 20)          # one big dark square
    img[40:160, 40:64] = (230, 30, 30)          # 20% of it a different colour
    ink = np.zeros((200, 200), bool)
    ink[40:160, 40:160] = True                   # traced as ONE component
    out = H.measure_fusions(ink, img)
    assert out['fusions'] == 1
    # the red strip is 1/5 of the square, so the fused extent is about 0.2
    assert 0.1 < out['fused_area_frac'] < 0.3, out
    # and a component with no second colour is not a fusion at all
    plain = np.zeros((200, 200), bool); plain[40:160, 64:160] = True
    assert H.measure_fusions(plain, img)['fusions'] == 0


def test_fusion_penalty_is_area_led_with_a_capped_count_term():
    """A little artwork eaten by many small fusions must cost less than a lot
    of artwork eaten by one big one - the reverse of the old count-led form."""
    many_small = {'fusions': 20, 'fused_area_frac': 0.02}
    one_big = {'fusions': 1, 'fused_area_frac': 0.60}
    pen = lambda f: min(40.0, f['fused_area_frac'] * 40 + min(5.0, f['fusions'] * 0.5))
    assert pen(many_small) < pen(one_big)
    assert pen(many_small) <= 5.8               # count term cannot sink it alone
    assert pen({'fusions': 0, 'fused_area_frac': 1.0}) == 40.0


def test_stroke_never_closes_a_white_channel_narrower_than_itself():
    """Two dark blocks with a 3px bare channel between them, outlined with a
    stroke wide enough to swallow it. The channel must survive as at least a
    one-pixel white spine, or the two blocks read as one."""
    fill = np.zeros((60, 60), bool)
    fill[10:50, 10:28] = True
    fill[10:50, 31:50] = True                 # 3px bare channel at x=28..30
    edge = np.zeros((60, 60), bool)
    edge[10:50, 27:32] = True                 # the boundary between them
    lines = cv2.dilate(edge.astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    assert not (~fill & ~lines)[20:40, 28:31].any()      # unprotected: gone
    kept = C._keep_white_channels(lines, fill, line_px=7)
    channel = (~fill & ~kept)[15:45, 27:32]
    assert channel.any(), 'the channel was closed anyway'


def test_protection_leaves_a_stroke_in_open_space_alone():
    """A stroke that closes nothing must come back byte-identical - otherwise
    protection would punch holes in legitimate boundary lines."""
    fill = np.zeros((60, 60), bool)
    fill[10:50, 10:30] = True                 # one block, open white around it
    lines = cv2.dilate(fill.astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    assert np.array_equal(C._keep_white_channels(lines, fill, line_px=5), lines)


def test_layerlines_keeps_a_thin_light_stripe_between_two_dark_fields():
    img = np.full((240, 240, 3), 255, np.uint8)
    img[60:180, 40:118] = (18, 18, 18)
    img[60:180, 122:200] = (22, 22, 60)       # second dark field, different hue
    ink = C.strat_layerlines(img, line_frac=0.03)   # ~7px stroke, channel is 4px
    assert not ink[110:130, 118:122].all(), 'the light stripe was filled in'
