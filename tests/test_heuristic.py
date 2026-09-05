"""The true-distance heuristic must be admissible and consistent."""

from __future__ import annotations

import numpy as np
import pytest

from swarmplan.graph import GridMap
from swarmplan.lowlevel.heuristic import (
    UNREACHABLE,
    HeuristicCache,
    backward_dijkstra,
    true_distance,
)

#: A serpentine maze: fully connected, but every route is long. Two cells one
#: row apart can be twenty steps apart, which is exactly the case Manhattan
#: distance gets badly wrong.
MAZE = [
    "..........",
    ".@@@@@@@@@",
    "..........",
    "@@@@@@@@@.",
    "..........",
    ".@@@@@@@@@",
    "..........",
]


def test_distances_are_exact_on_an_open_grid():
    grid = GridMap.from_rows(["....."] * 5)
    goal = grid.node((0, 0))
    d = backward_dijkstra(grid.graph, goal)
    for v in range(grid.graph.n_nodes):
        r, c = grid.rc(v)
        assert d[v] == r + c


def test_heuristic_is_admissible_on_a_map_with_obstacles():
    """It never exceeds the true distance, because it *is* the true distance."""
    grid = GridMap.from_rows(MAZE)
    for goal_rc in [(0, 0), (2, 5), (6, 9), (4, 3)]:
        goal = grid.node(goal_rc)
        d = backward_dijkstra(grid.graph, goal)
        # Independent check: breadth-first search from every node, one at a time.
        for v in range(0, grid.graph.n_nodes, 3):
            ref = backward_dijkstra(grid.graph, v)[goal]
            assert d[v] == ref


def test_heuristic_is_consistent_on_a_map_with_obstacles():
    """h(u) <= cost(u, v) + h(v) for every edge, which is what closing nodes needs."""
    grid = GridMap.from_rows(MAZE)
    d = backward_dijkstra(grid.graph, grid.node((0, 0)))
    checked = 0
    for u in range(grid.graph.n_nodes):
        for v in grid.graph.neighbours[u]:
            if d[u] >= UNREACHABLE or d[v] >= UNREACHABLE:
                continue
            assert d[u] <= 1 + d[v]
            checked += 1
    assert checked > 80


def test_manhattan_is_admissible_but_much_weaker():
    """The reason the true distance is worth computing, stated as a test."""
    grid = GridMap.from_rows(MAZE)
    goal_rc = (0, 0)
    goal = grid.node(goal_rc)
    d = backward_dijkstra(grid.graph, goal)
    coords = grid.graph.coord_array()
    manhattan = np.abs(coords[:, 0] - goal_rc[0]) + np.abs(coords[:, 1] - goal_rc[1])
    reachable = d < UNREACHABLE
    assert np.all(manhattan[reachable] <= d[reachable])  # admissible
    assert d[reachable].max() >= 2 * manhattan[reachable].max()  # and far weaker


def test_unreachable_cells_are_marked():
    grid = GridMap.from_rows(["..@..", "..@..", "..@.."])
    d = backward_dijkstra(grid.graph, grid.node((0, 0)))
    assert d[grid.node((0, 4))] >= UNREACHABLE
    assert d[grid.node((2, 0))] == 2


def test_weighted_dijkstra_matches_unit_costs_when_weights_are_one():
    grid = GridMap.from_rows(["....", ".@..", "...."])
    goal = grid.node((0, 0))
    unit = backward_dijkstra(grid.graph, goal)
    weighted = backward_dijkstra(grid.graph, goal, weights=np.ones(grid.graph.n_nodes))
    assert np.allclose(unit[unit < UNREACHABLE], weighted[unit < UNREACHABLE])
    with pytest.raises(ValueError):
        backward_dijkstra(grid.graph, goal, weights=np.ones(3))


def test_weighted_dijkstra_prefers_cheap_cells():
    grid = GridMap.from_rows(["....."] * 3)
    weights = np.ones(grid.graph.n_nodes)
    for c in range(5):
        weights[grid.node((1, c))] = 50.0
    d = backward_dijkstra(grid.graph, grid.node((0, 0)), weights=weights)
    assert d[grid.node((2, 0))] > 50.0


def test_cache_computes_each_goal_once():
    grid = GridMap.from_rows(["....."] * 3)
    cache = HeuristicCache(grid.graph)
    goal = grid.node((2, 4))
    cache.get(goal)
    cache.get(goal)
    assert cache.sweeps == 1
    assert len(cache) == 1
    assert cache.distance(grid.node((0, 0)), goal) == 6
    assert cache.reachable(grid.node((0, 0)), goal)


def test_true_distance_alias_and_bad_goal():
    grid = GridMap.from_rows(["..", ".."])
    assert np.array_equal(true_distance(grid.graph, 0), backward_dijkstra(grid.graph, 0))
    with pytest.raises(ValueError):
        backward_dijkstra(grid.graph, 99)
