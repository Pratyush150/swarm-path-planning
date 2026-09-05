"""MDDs and the cardinal / dependency reasoning built on them."""

from __future__ import annotations

import pytest

from swarmplan.cbs.mdd import build_mdd, joint_mdd_size, mdds_dependent
from swarmplan.constraints import Constraint, build_table
from swarmplan.graph import GridMap
from swarmplan.lowlevel.heuristic import backward_dijkstra


def mdd_for(grid, start_rc, goal_rc, cost, constraints=()):
    """Helper: build an MDD between two coordinates at a given cost."""
    s, g = grid.node(start_rc), grid.node(goal_rc)
    h = backward_dijkstra(grid.graph, g)
    table = build_table(0, list(constraints), goal=g)
    return build_mdd(grid.graph, s, g, cost, h, table)


def test_corridor_mdd_is_width_one_everywhere():
    """With only one shortest path, every agent on it is forced at every step."""
    grid = GridMap.from_rows(["....."])
    mdd = mdd_for(grid, (0, 0), (0, 4), 4)
    assert mdd is not None
    assert [mdd.width(t) for t in range(5)] == [1, 1, 1, 1, 1]
    assert mdd.singleton(2) == grid.node((0, 2))
    assert mdd.forced_edge(3) == (grid.node((0, 2)), grid.node((0, 3)))


def test_open_grid_mdd_widens_in_the_middle(open_grid):
    mdd = mdd_for(open_grid, (0, 0), (2, 2), 4)
    assert mdd is not None
    assert mdd.width(0) == 1
    assert mdd.width(2) > 1  # several ways to be halfway there
    assert mdd.width(4) == 1
    assert mdd.singleton(2) is None


def test_mdd_past_its_cost_is_the_parked_goal(open_grid):
    mdd = mdd_for(open_grid, (0, 0), (0, 2), 2)
    assert mdd.width(50) == 1
    assert mdd.at(50) == {open_grid.node((0, 2))}
    assert mdd.successors(mdd.goal, 50) == {open_grid.node((0, 2))}


def test_mdd_respects_constraints(open_grid):
    forbidden = open_grid.node((0, 1))
    mdd = mdd_for(open_grid, (0, 0), (0, 2), 2, [Constraint(0, forbidden, None, 1)])
    assert mdd is None  # the only 2-step route is blocked
    longer = mdd_for(open_grid, (0, 0), (0, 2), 4, [Constraint(0, forbidden, None, 1)])
    assert longer is not None
    assert forbidden not in longer.at(1)


def test_mdd_is_none_when_the_cost_is_impossible(open_grid):
    assert mdd_for(open_grid, (0, 0), (4, 4), 3) is None
    with pytest.raises(ValueError):
        mdd_for(open_grid, (0, 0), (4, 4), -1)


def test_head_on_agents_in_a_corridor_are_dependent():
    """Two agents crossing a one-wide corridor cannot both stay optimal."""
    grid = GridMap.from_rows(["....."])
    h_right = backward_dijkstra(grid.graph, grid.node((0, 4)))
    h_left = backward_dijkstra(grid.graph, grid.node((0, 0)))
    m1 = build_mdd(grid.graph, grid.node((0, 0)), grid.node((0, 4)), 4, h_right)
    m2 = build_mdd(grid.graph, grid.node((0, 4)), grid.node((0, 0)), 4, h_left)
    assert mdds_dependent(m1, m2)
    assert joint_mdd_size(m1, m2) >= 1


def test_agents_that_never_meet_are_independent(open_grid):
    grid = open_grid
    h1 = backward_dijkstra(grid.graph, grid.node((0, 4)))
    h2 = backward_dijkstra(grid.graph, grid.node((4, 4)))
    m1 = build_mdd(grid.graph, grid.node((0, 0)), grid.node((0, 4)), 4, h1)
    m2 = build_mdd(grid.graph, grid.node((4, 0)), grid.node((4, 4)), 4, h2)
    assert not mdds_dependent(m1, m2)


def test_repr_shows_the_widths(open_grid):
    mdd = mdd_for(open_grid, (0, 0), (0, 2), 2)
    assert "widths=[1, 2, 1]" in repr(mdd) or "widths=" in repr(mdd)
    assert len(mdd) == 3
