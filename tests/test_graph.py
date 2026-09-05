"""Grid parsing and the search-graph abstraction."""

from __future__ import annotations

import numpy as np
import pytest

from swarmplan.graph import Grid3D, GridMap

OCTILE = """type octile
height 4
width 5
map
.....
.@@..
...@.
.....
"""


def test_parses_movingai_octile_format():
    grid = GridMap.from_text(OCTILE, name="tiny")
    assert (grid.height, grid.width) == (4, 5)
    assert grid.n_free == 17
    assert grid.blocked[1, 1] and grid.blocked[2, 3]
    assert not grid.blocked[0, 0]
    assert grid.name == "tiny"


def test_rejects_bad_header_and_unknown_characters():
    with pytest.raises(ValueError):
        GridMap.from_text("height 2\nwidth 2\n..\n..\n")
    with pytest.raises(ValueError):
        GridMap.from_text("type octile\nheight 1\nwidth 2\nmap\n.x\n")


def test_node_ids_round_trip():
    grid = GridMap.from_rows(["...", ".@.", "..."])
    for r in range(3):
        for c in range(3):
            if grid.passable((r, c)):
                assert grid.rc(grid.node((r, c))) == (r, c)
    assert not grid.passable((1, 1))
    assert not grid.passable((-1, 0))
    assert not grid.passable((0, 99))


def test_neighbours_are_four_connected_and_exclude_obstacles():
    grid = GridMap.from_rows(["...", ".@.", "..."])
    centre_top = grid.node((0, 1))
    nbrs = {grid.rc(v) for v in grid.graph.neighbours[centre_top]}
    assert nbrs == {(0, 0), (0, 2)}
    corner = grid.node((0, 0))
    assert {grid.rc(v) for v in grid.graph.neighbours[corner]} == {(0, 1), (1, 0)}
    # No diagonal moves anywhere.
    for v in range(grid.graph.n_nodes):
        r, c = grid.rc(v)
        for u in grid.graph.neighbours[v]:
            ur, uc = grid.rc(u)
            assert abs(ur - r) + abs(uc - c) == 1


def test_wait_is_not_an_edge():
    grid = GridMap.from_rows(["..", ".."])
    for v in range(grid.graph.n_nodes):
        assert v not in grid.graph.neighbours[v]


def test_manhattan_and_coord_array():
    grid = GridMap.from_rows(["....", "....."])
    a, b = grid.node((0, 0)), grid.node((1, 3))
    assert grid.graph.manhattan(a, b) == 4
    arr = grid.graph.coord_array()
    assert arr.shape == (grid.n_free, 2)
    assert arr.dtype == np.int32


def test_grid3d_six_connected():
    box = Grid3D((3, 3, 3))
    assert box.n_free == 27
    middle = box.node((1, 1, 1))
    assert len(box.graph.neighbours[middle]) == 6
    corner = box.node((0, 0, 0))
    assert len(box.graph.neighbours[corner]) == 3
    assert box.xyz(middle) == (1, 1, 1)


def test_grid3d_four_connected_ignores_depth():
    box = Grid3D((3, 3, 3), connectivity=4)
    middle = box.node((1, 1, 1))
    for v in box.graph.neighbours[middle]:
        assert box.xyz(v)[1] == 1
    with pytest.raises(ValueError):
        Grid3D((2, 2, 2), connectivity=8)


def test_blocked_cells_are_not_nodes():
    box = Grid3D((2, 2, 2), blocked=np.array([[[True, False], [False, False]],
                                              [[False, False], [False, False]]]))
    assert box.n_free == 7
    assert not box.graph.has((0, 0, 0))


def test_search_graph_rejects_unknown_coordinate():
    grid = GridMap.from_rows([".."])
    with pytest.raises(KeyError):
        grid.node((5, 5))
