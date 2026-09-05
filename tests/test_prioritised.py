"""Prioritised planning: fast, useful, and provably incomplete."""

from __future__ import annotations

import pytest

from swarmplan.cbs.solver import solve_cbs
from swarmplan.conflicts import validate_plan
from swarmplan.graph import GridMap
from swarmplan.prioritised.planner import PPConfig, PrioritisedPlanner, solve_pp
from swarmplan.solution import FAILED, SOLVED


def alcove_instance():
    """Three cells of corridor with one alcove; agents must swap ends.

        A B C      agent 0: A -> C
        . D .      agent 1: C -> A
    """
    grid = GridMap.from_rows(["...", "@.@"], name="alcove")
    s = [grid.node((0, 0)), grid.node((0, 2))]
    g = [grid.node((0, 2)), grid.node((0, 0))]
    return grid, s, g


def test_prioritised_planning_fails_where_cbs_succeeds():
    """The headline failure mode, demonstrated rather than described."""
    grid, s, g = alcove_instance()
    planner = PrioritisedPlanner(grid.graph, s, g)
    assert planner.plan_with_order([0, 1]) is None
    assert planner.plan_with_order([1, 0]) is None

    restarts = solve_pp(grid.graph, s, g, restarts=16, time_limit=5.0)
    assert restarts.status == FAILED  # no priority order works, so restarts cannot help

    optimal = solve_cbs(grid.graph, s, g, time_limit=10.0)
    assert optimal.status == SOLVED
    assert optimal.cost == 7
    assert validate_plan(optimal.paths, s, g, grid.graph) == []


def test_failure_is_never_reported_as_unsolvable():
    """An incomplete planner must not make claims about the instance."""
    grid, s, g = alcove_instance()
    result = solve_pp(grid.graph, s, g, restarts=4, time_limit=5.0)
    assert result.status != "unsolvable"
    assert result.paths is None


def test_it_solves_ordinary_instances_and_the_plan_is_valid():
    grid = GridMap.from_rows([".....", "..@..", "....."])
    s = [grid.node(x) for x in [(0, 0), (0, 4), (2, 0)]]
    g = [grid.node(x) for x in [(0, 4), (0, 0), (2, 4)]]
    result = solve_pp(grid.graph, s, g)
    assert result.status == SOLVED
    assert validate_plan(result.paths, s, g, grid.graph) == []


def test_it_is_much_faster_than_optimal_search_on_the_same_instance():
    grid = GridMap.from_rows([".........."] * 6)
    s = [grid.node((0, i)) for i in range(6)]
    g = [grid.node((5, 9 - i)) for i in range(6)]
    pp = solve_pp(grid.graph, s, g, time_limit=20.0)
    assert pp.status == SOLVED
    assert pp.high_level_expanded == 0  # there is no high level at all
    assert pp.low_level_calls == len(s)


def test_priority_orders_are_built_as_documented():
    grid = GridMap.from_rows(["....."] * 5)
    s = [grid.node((0, 0)), grid.node((0, 3)), grid.node((4, 0))]
    g = [grid.node((0, 4)), grid.node((0, 4)), grid.node((0, 4))]
    planner = PrioritisedPlanner(grid.graph, s, g, PPConfig(order="longest_first"))
    assert planner.priority_order()[0] == 2
    planner = PrioritisedPlanner(grid.graph, s, g, PPConfig(order="shortest_first"))
    assert planner.priority_order()[0] == 1
    planner = PrioritisedPlanner(grid.graph, s, g, PPConfig(order="random", seed=3))
    assert sorted(planner.priority_order()) == [0, 1, 2]


def test_random_restarts_recover_orders_a_fixed_order_misses():
    """One ordering fails, another works; restarts find the one that works."""
    grid = GridMap.from_rows([".....", ".@@@.", "....."])
    s = [grid.node((0, 2)), grid.node((0, 0))]
    g = [grid.node((0, 0)), grid.node((0, 4))]
    planner = PrioritisedPlanner(grid.graph, s, g)
    orders = [planner.plan_with_order(o) is not None for o in ([0, 1], [1, 0])]
    assert any(orders)
    result = solve_pp(grid.graph, s, g, restarts=8, seed=1, time_limit=5.0)
    assert result.status == SOLVED
    assert result.notes["attempts"] >= 1


def test_unreachable_goal_is_unsolvable():
    grid = GridMap.from_rows(["..@..", "..@..", "..@.."])
    result = solve_pp(grid.graph, [grid.node((0, 0))], [grid.node((0, 4))])
    assert result.status == "unsolvable"


def test_configuration_is_validated_and_labelled():
    with pytest.raises(ValueError):
        PPConfig(order="cleverest").validate()
    with pytest.raises(ValueError):
        PPConfig(restarts=0).validate()
    assert PPConfig().label() == "PP"
    assert PPConfig(restarts=8).label() == "PP(restarts=8)"
    assert PPConfig(order="longest_first").label() == "PP(longest_first)"


def test_mismatched_inputs_are_rejected():
    grid = GridMap.from_rows(["..."])
    with pytest.raises(ValueError):
        PrioritisedPlanner(grid.graph, [0], [1, 2])
