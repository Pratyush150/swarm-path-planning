"""CBS must return a provably optimal plan, and every variant must agree."""

from __future__ import annotations

import pytest

from reference import best_prioritised_cost, brute_force_sum_of_costs
from swarmplan.cbs.solver import CBS, CBSConfig, solve_cbs
from swarmplan.conflicts import validate_plan
from swarmplan.graph import GridMap
from swarmplan.solution import SOLVED, TIMEOUT

#: Every switchable combination we claim to support. Each must be optimal.
VARIANTS = [
    {},
    {"prioritise_conflicts": True},
    {"bypass": True},
    {"disjoint": True},
    {"heuristic": "cg"},
    {"heuristic": "dg"},
    {"prioritise_conflicts": True, "bypass": True},
    {"prioritise_conflicts": True, "heuristic": "cg"},
    {"prioritise_conflicts": True, "bypass": True, "heuristic": "dg"},
    {"prioritise_conflicts": True, "disjoint": True, "heuristic": "dg"},
]


def instances():
    """Small instances with a hand-checkable structure, one per scenario."""
    bay = GridMap.from_rows([".....", "..@..", "....."], name="bay")
    out = [
        (
            "head-on with a passing bay",
            bay,
            [(0, 0), (0, 4)],
            [(0, 4), (0, 0)],
        ),
        (
            "three agents through a bay",
            bay,
            [(0, 0), (0, 4), (2, 0)],
            [(0, 4), (0, 0), (2, 4)],
        ),
        (
            "corridor with one alcove",
            GridMap.from_rows(["...", "@.@"], name="alcove"),
            [(0, 0), (0, 2)],
            [(0, 2), (0, 0)],
        ),
        (
            "open room, crossing diagonals",
            GridMap.from_rows(["...."] * 4, name="room"),
            [(0, 0), (0, 3), (3, 0)],
            [(3, 3), (3, 0), (0, 3)],
        ),
    ]
    return out


@pytest.mark.parametrize("name,grid,starts,goals", instances(), ids=[i[0] for i in instances()])
def test_cbs_matches_brute_force_optimum(name, grid, starts, goals):
    """The independent joint-space search agrees on the optimal sum-of-costs."""
    s = [grid.node(x) for x in starts]
    g = [grid.node(x) for x in goals]
    optimal = brute_force_sum_of_costs(grid.graph, s, g)
    assert optimal is not None, "brute force hit its state cap; instance is too big"
    result = solve_cbs(grid.graph, s, g, time_limit=30.0)
    assert result.status == SOLVED
    assert result.cost == optimal
    assert validate_plan(result.paths, s, g, grid.graph) == []


@pytest.mark.parametrize("options", VARIANTS, ids=[str(sorted(v)) or "plain" for v in VARIANTS])
def test_every_variant_returns_the_same_optimal_cost(options):
    """Prioritising, bypass, disjoint splitting and the heuristics preserve optimality."""
    grid = GridMap.from_rows([".....", "..@..", "....."])
    s = [grid.node(x) for x in [(0, 0), (0, 4), (2, 0)]]
    g = [grid.node(x) for x in [(0, 4), (0, 0), (2, 4)]]
    optimal = brute_force_sum_of_costs(grid.graph, s, g)
    result = solve_cbs(grid.graph, s, g, time_limit=30.0, **options)
    assert result.status == SOLVED
    assert result.cost == optimal
    assert validate_plan(result.paths, s, g, grid.graph) == []


def test_improvements_do_not_expand_more_nodes_than_plain_cbs():
    """On an instance with real branching, the improvements pay for themselves."""
    grid = GridMap.from_rows(
        ["..........", ".@@@@@@@@.", "..........", ".@@@@@@@@.", ".........."]
    )
    s = [grid.node(x) for x in [(0, 0), (0, 9), (2, 0), (2, 9)]]
    g = [grid.node(x) for x in [(0, 9), (0, 0), (2, 9), (2, 0)]]
    plain = solve_cbs(grid.graph, s, g, time_limit=60.0)
    improved = solve_cbs(
        grid.graph, s, g, time_limit=60.0,
        prioritise_conflicts=True, bypass=True, heuristic="dg",
    )
    assert plain.status == improved.status == SOLVED
    assert plain.cost == improved.cost
    assert improved.high_level_expanded <= plain.high_level_expanded


def test_no_conflict_instance_is_solved_at_the_root():
    grid = GridMap.from_rows(["....."] * 3)
    s = [grid.node((0, 0)), grid.node((2, 0))]
    g = [grid.node((0, 4)), grid.node((2, 4))]
    result = solve_cbs(grid.graph, s, g)
    assert result.high_level_expanded == 1
    assert result.cost == 8


def test_agent_with_no_route_is_reported_unsolvable():
    grid = GridMap.from_rows(["..@..", "..@..", "..@.."])
    result = solve_cbs(grid.graph, [grid.node((0, 0))], [grid.node((0, 4))])
    assert result.status == "unsolvable"
    assert result.paths is None
    assert "cannot reach" in result.notes["reason"]


def test_unsolvable_swap_is_reported_as_a_timeout_not_a_solution():
    """CBS cannot prove a corridor swap impossible; it must not claim success.

    Two agents swapping ends of a one-wide corridor has no solution. CBS is
    complete for solvable instances but does not terminate on this one -- it
    keeps adding constraints forever -- so the honest outcome is a timeout, and
    the status must never be SOLVED or UNSOLVABLE.
    """
    grid = GridMap.from_rows(["....."])
    result = solve_cbs(
        grid.graph, [grid.node((0, 0)), grid.node((0, 4))],
        [grid.node((0, 4)), grid.node((0, 0))], time_limit=2.0,
    )
    assert result.status == TIMEOUT
    assert result.paths is None


def test_cbs_is_never_worse_than_the_best_priority_order():
    """Exhaustive search over priority orders still cannot beat an optimal solver."""
    grid = GridMap.from_rows([".....", "..@..", "....."])
    s = [grid.node(x) for x in [(0, 0), (0, 4), (2, 2)]]
    g = [grid.node(x) for x in [(0, 4), (0, 0), (0, 2)]]
    optimal = solve_cbs(grid.graph, s, g, time_limit=30.0)
    best_pp = best_prioritised_cost(grid.graph, s, g)
    assert optimal.status == SOLVED
    assert best_pp is None or optimal.cost <= best_pp


def test_node_limit_is_honoured():
    grid = GridMap.from_rows([".....", "..@..", "....."])
    s = [grid.node(x) for x in [(0, 0), (0, 4)]]
    g = [grid.node(x) for x in [(0, 4), (0, 0)]]
    result = solve_cbs(grid.graph, s, g, node_limit=1, time_limit=30.0)
    assert result.status in ("budget", SOLVED)


def test_configuration_is_validated_and_labelled():
    with pytest.raises(ValueError):
        CBSConfig(heuristic="wdg").validate()
    with pytest.raises(ValueError):
        CBSConfig(time_limit=0).validate()
    assert CBSConfig().label() == "CBS"
    assert CBSConfig(prioritise_conflicts=True, bypass=True).label() == "CBS+PC+BP"
    assert CBSConfig(heuristic="dg", disjoint=True).label() == "CBS+DS+DG"
    assert CBSConfig(name="custom").label() == "custom"


def test_mismatched_start_and_goal_counts_are_rejected():
    grid = GridMap.from_rows(["..."])
    with pytest.raises(ValueError):
        CBS(grid.graph, [0, 1], [2])
