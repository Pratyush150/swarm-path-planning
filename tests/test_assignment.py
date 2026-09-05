"""Anonymous MAPF: the goal assignment is where most of the cost is decided."""

from __future__ import annotations

import pytest

from swarmplan.assignment.anonymous import (
    assign_goals,
    assignment_lower_bound,
    distance_matrix,
    identity_assignment,
)
from swarmplan.assignment.formations import annulus_points, text_points
from swarmplan.graph import Grid3D, GridMap
from swarmplan.lowlevel.heuristic import HeuristicCache


def test_distance_matrix_holds_true_obstacle_aware_distances():
    grid = GridMap.from_rows([".....", ".@@@.", "....."])
    starts = [grid.node((0, 1)), grid.node((0, 3))]
    goals = [grid.node((2, 1)), grid.node((2, 3))]
    d = distance_matrix(grid.graph, starts, goals)
    assert d.shape == (2, 2)
    # Straight down is blocked; every route goes round the wall.
    assert d[0, 0] == 4
    assert d[0, 1] == 6


def test_optimal_assignment_beats_the_arbitrary_one_on_a_formation_morph():
    """The claim the light show rests on, measured on a swap-the-line instance."""
    grid = GridMap.from_rows(["." * 16] * 3)
    starts = [grid.node((0, c)) for c in range(12)]
    goals = [grid.node((2, 11 - c)) for c in range(12)]  # reversed order
    identity = identity_assignment(grid.graph, starts, goals)
    optimal = assign_goals(grid.graph, starts, goals, objective="sum")
    assert optimal.total_distance < identity.total_distance
    assert optimal.total_distance == 24  # every agent simply drops two rows
    assert identity.total_distance == 96  # four times as far, for the same formation
    assert optimal.order != list(range(12))


def test_makespan_objective_minimises_the_longest_flight():
    grid = GridMap.from_rows(["." * 12] * 2)
    starts = [grid.node((0, 0)), grid.node((0, 1))]
    goals = [grid.node((1, 0)), grid.node((1, 11))]
    by_sum = assign_goals(grid.graph, starts, goals, objective="sum")
    by_max = assign_goals(grid.graph, starts, goals, objective="makespan")
    assert by_max.max_distance <= by_sum.max_distance
    assert by_sum.total_distance <= by_max.total_distance


def test_assignment_is_a_permutation_and_can_be_applied():
    grid = GridMap.from_rows(["....."] * 5)
    starts = [grid.node((0, c)) for c in range(4)]
    goals = [grid.node((4, c)) for c in range(4)]
    res = assign_goals(grid.graph, starts, goals)
    assert sorted(res.order) == list(range(4))
    applied = res.apply(goals)
    assert sorted(applied) == sorted(goals)
    assert res.n_agents == 4
    assert assignment_lower_bound(res) == res.total_distance


def test_unreachable_goals_are_penalised_not_crashed():
    grid = GridMap.from_rows(["..@..", "..@..", "..@.."])
    starts = [grid.node((0, 0)), grid.node((0, 4))]
    goals = [grid.node((2, 0)), grid.node((2, 4))]
    d = distance_matrix(grid.graph, starts, goals)
    assert d[0, 1] > 1e8
    res = assign_goals(grid.graph, starts, goals)
    assert res.order == [0, 1]  # the only feasible pairing


def test_assignment_in_three_dimensions_uses_the_whole_airspace():
    space = Grid3D((10, 4, 10))
    text = [space.node(p) for p in text_points("A", origin=(2, 2, 1), depth=2)]
    ring = [space.node(p) for p in annulus_points(len(text), (5, 2, 5), 4.0, depth=2)]
    cache = HeuristicCache(space.graph)
    identity = identity_assignment(space.graph, ring, text, cache)
    optimal = assign_goals(space.graph, ring, text, cache=cache)
    assert optimal.total_distance <= identity.total_distance
    assert optimal.matrix_time >= 0.0 and optimal.solve_time >= 0.0


def test_bad_inputs_are_rejected():
    grid = GridMap.from_rows(["..."])
    with pytest.raises(ValueError):
        assign_goals(grid.graph, [0], [1, 2])
    with pytest.raises(ValueError):
        assign_goals(grid.graph, [0], [1], objective="cheapest")
