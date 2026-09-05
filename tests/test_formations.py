"""Formations: rings, annuli and text, on the grid."""

from __future__ import annotations

import pytest

from swarmplan.assignment.formations import (
    FONT,
    GLYPH_HEIGHT,
    GLYPH_WIDTH,
    annulus_points,
    bounding_box,
    grid_points,
    resample,
    ring_points,
    text_points,
    text_width,
)


def test_font_glyphs_are_all_five_by_seven():
    for ch, glyph in FONT.items():
        assert len(glyph) == GLYPH_HEIGHT, ch
        assert all(len(row) == GLYPH_WIDTH for row in glyph), ch
        assert set("".join(glyph)) <= {"#", "."}, ch


def test_font_covers_letters_digits_and_space():
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ":
        assert ch in FONT


def test_text_points_are_distinct_and_placed_left_to_right():
    pts = text_points("AB", origin=(0, 0, 0))
    assert len(set(pts)) == len(pts)
    (x0, _, _), (x1, _, _) = bounding_box(pts)
    assert x1 - x0 <= text_width("AB")
    assert text_width("AB") == 2 * GLYPH_WIDTH + 1


def test_unknown_character_is_rejected():
    with pytest.raises(ValueError):
        text_points("!", origin=(0, 0, 0))


def test_ring_returns_exactly_the_requested_count():
    pts = ring_points(24, (10, 2, 10), 6.0)
    assert len(pts) == 24
    assert len(set(pts)) == 24
    for x, y, z in pts:
        assert y == 2
        r = ((x - 10) ** 2 + (z - 10) ** 2) ** 0.5
        assert 5.0 <= r <= 7.0


def test_ring_refuses_a_radius_that_cannot_hold_the_count():
    with pytest.raises(ValueError):
        ring_points(200, (10, 2, 10), 2.0)


def test_annulus_fills_inwards_until_the_count_is_met():
    pts = annulus_points(120, (20, 2, 20), 9.0, depth=2)
    assert len(pts) == 120
    assert len(set(pts)) == 120


def test_grid_block_is_dense_and_the_right_size():
    pts = grid_points(10, (0, 0, 0), width=4)
    assert len(pts) == 10
    assert len(set(pts)) == 10
    (x0, _, z0), (x1, _, z1) = bounding_box(pts)
    assert x1 - x0 == 3 and z1 - z0 == 2


def test_resample_spreads_and_never_repeats():
    seq = [(i, 0, 0) for i in range(20)]
    out = resample(seq, 5)
    assert len(out) == len(set(out)) == 5
    assert out[0] == seq[0] and out[-1] == seq[-1]
    assert resample(seq, 20) == seq
    with pytest.raises(ValueError):
        resample(seq, 21)


def test_bounding_box_of_nothing_is_an_error():
    with pytest.raises(ValueError):
        bounding_box([])
