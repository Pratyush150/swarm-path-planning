"""Parsing benchmark scenarios and building instances from them."""

from __future__ import annotations

import pytest

from swarmplan.graph import GridMap
from swarmplan.scenarios import (
    instance_from_coords,
    make_instance,
    parse_scen,
)

SCEN = """version 1
0\ttiny.map\t5\t3\t0\t0\t4\t2\t4.82842712
1\ttiny.map\t5\t3\t4\t0\t0\t2\t4.82842712
2\ttiny.map\t5\t3\t2\t0\t2\t2\t2.00000000
"""


def test_scen_columns_are_col_row_and_get_swapped():
    scen = parse_scen(SCEN, name="tiny-1")
    assert len(scen) == 3
    first = scen.entries[0]
    # File says x=0 y=0 -> goal x=4 y=2, which is (row 2, col 4).
    assert first.start == (0, 0)
    assert first.goal == (2, 4)
    assert first.map_name == "tiny.map"
    assert first.octile_optimal == pytest.approx(4.82842712)


def test_take_returns_the_first_k_entries():
    scen = parse_scen(SCEN)
    assert len(scen.take(2)) == 2
    assert scen.take(2).entries == scen.entries[:2]
    with pytest.raises(ValueError):
        scen.take(9)


def test_malformed_line_is_rejected():
    with pytest.raises(ValueError):
        parse_scen("version 1\n0\tbad.map\t5\n")


def test_make_instance_maps_to_node_ids():
    grid = GridMap.from_rows(["....."] * 3, name="tiny")
    scen = parse_scen(SCEN)
    instance = make_instance(grid, scen, 3)
    assert instance.n_agents == 3
    starts, goals = instance.coords()
    assert starts[0] == (0, 0) and goals[0] == (2, 4)
    assert instance.octile_lower_bound == pytest.approx(4.82842712 * 2 + 2.0)
    assert instance.graph is grid.graph


def test_make_instance_rejects_blocked_endpoints():
    grid = GridMap.from_rows(["@....", ".....", "....."], name="tiny")
    scen = parse_scen(SCEN)
    with pytest.raises(ValueError):
        make_instance(grid, scen, 1)


def test_instance_from_coords_checks_lengths():
    grid = GridMap.from_rows(["...", "..."])
    inst = instance_from_coords(grid, [(0, 0)], [(1, 2)])
    assert inst.n_agents == 1
    with pytest.raises(ValueError):
        instance_from_coords(grid, [(0, 0)], [(1, 2), (0, 1)])
