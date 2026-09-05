"""ECBS must respect its suboptimality bound, and w=1 must be optimal."""

from __future__ import annotations

import pytest

from reference import brute_force_sum_of_costs
from swarmplan.conflicts import validate_plan
from swarmplan.ecbs.solver import ECBS, ECBSConfig, solve_ecbs
from swarmplan.graph import GridMap
from swarmplan.solution import SOLVED

W_VALUES = [1.0, 1.05, 1.2, 1.5, 2.0, 3.0]


def bay_instance():
    """Three agents, a corridor and one passing bay."""
    grid = GridMap.from_rows([".....", "..@..", "....."])
    s = [grid.node(x) for x in [(0, 0), (0, 4), (2, 0)]]
    g = [grid.node(x) for x in [(0, 4), (0, 0), (2, 4)]]
    return grid, s, g


@pytest.mark.parametrize("w", W_VALUES)
def test_solution_is_within_w_times_the_true_optimum(w):
    """The bound is a guarantee, checked against a brute-force optimum."""
    grid, s, g = bay_instance()
    optimal = brute_force_sum_of_costs(grid.graph, s, g)
    result = solve_ecbs(grid.graph, s, g, w=w, time_limit=30.0)
    assert result.status == SOLVED
    assert result.cost <= w * optimal + 1e-9
    assert validate_plan(result.paths, s, g, grid.graph) == []
    assert result.suboptimality_bound == w


def test_w_of_one_is_optimal():
    grid, s, g = bay_instance()
    optimal = brute_force_sum_of_costs(grid.graph, s, g)
    result = solve_ecbs(grid.graph, s, g, w=1.0, time_limit=30.0)
    assert result.cost == optimal


def test_lower_bound_never_exceeds_the_optimum():
    """The reported bound has to be a real lower bound or the guarantee is void."""
    grid, s, g = bay_instance()
    optimal = brute_force_sum_of_costs(grid.graph, s, g)
    for w in (1.0, 1.2, 2.0):
        result = solve_ecbs(grid.graph, s, g, w=w, time_limit=30.0)
        assert result.lower_bound <= optimal


def test_larger_w_never_expands_more_nodes_here():
    grid, s, g = bay_instance()
    tight = solve_ecbs(grid.graph, s, g, w=1.0, time_limit=30.0)
    loose = solve_ecbs(grid.graph, s, g, w=2.0, time_limit=30.0)
    assert loose.high_level_expanded <= tight.high_level_expanded


VARIANTS = [{}, {"prioritise_conflicts": True}, {"heuristic": "cg"}, {"heuristic": "dg"}]


@pytest.mark.parametrize("options", VARIANTS)
def test_variants_stay_within_the_bound(options):
    grid, s, g = bay_instance()
    optimal = brute_force_sum_of_costs(grid.graph, s, g)
    result = solve_ecbs(grid.graph, s, g, w=1.5, time_limit=30.0, **options)
    assert result.status == SOLVED
    assert result.cost <= 1.5 * optimal + 1e-9


def test_unreachable_goal_is_unsolvable():
    grid = GridMap.from_rows(["..@..", "..@..", "..@.."])
    result = solve_ecbs(grid.graph, [grid.node((0, 0))], [grid.node((0, 4))])
    assert result.status == "unsolvable"


def test_configuration_is_validated_and_labelled():
    with pytest.raises(ValueError):
        ECBSConfig(w=0.5).validate()
    with pytest.raises(ValueError):
        ECBSConfig(heuristic="wdg").validate()
    assert ECBSConfig(w=1.1).label() == "ECBS(w=1.1)"
    assert ECBSConfig(w=1.5, prioritise_conflicts=True).label() == "ECBS(w=1.5)+PC"
    assert ECBSConfig(w=1.2, heuristic="dg").label() == "ECBS(w=1.2)+DG"


def test_mismatched_inputs_are_rejected():
    grid = GridMap.from_rows(["..."])
    with pytest.raises(ValueError):
        ECBS(grid.graph, [0], [1, 2])


def test_focal_search_reports_low_level_work():
    grid, s, g = bay_instance()
    result = solve_ecbs(grid.graph, s, g, w=1.2, time_limit=30.0)
    assert result.low_level_calls >= len(s)
    assert result.low_level_expanded > 0
