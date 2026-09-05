"""Space-time A*: optimal, constraint-respecting, and correct about waiting."""

from __future__ import annotations

import pytest

from swarmplan.constraints import Constraint, build_table
from swarmplan.graph import GridMap
from swarmplan.lowlevel.astar import location_at, path_cost, space_time_astar
from swarmplan.lowlevel.heuristic import backward_dijkstra


def plan(grid, start_rc, goal_rc, constraints=(), **kw):
    """Helper: run space-time A* between two coordinates."""
    s, g = grid.node(start_rc), grid.node(goal_rc)
    h = backward_dijkstra(grid.graph, g)
    table = build_table(0, list(constraints), goal=g)
    return space_time_astar(grid.graph, s, g, h, table, **kw)


def test_finds_the_hand_checkable_optimum(open_grid):
    res = plan(open_grid, (0, 0), (4, 4))
    assert res.found
    assert res.cost == 8
    assert len(res.path) == 9
    assert res.path[0] == open_grid.node((0, 0))
    assert res.path[-1] == open_grid.node((4, 4))


def test_every_step_is_a_legal_move_or_a_wait(wall_grid):
    res = plan(wall_grid, (0, 0), (2, 4))
    graph = wall_grid.graph
    for t in range(1, len(res.path)):
        u, v = res.path[t - 1], res.path[t]
        assert u == v or v in graph.neighbours[u]


def test_detours_round_a_wall(wall_grid):
    """The wall makes the path longer than the Manhattan estimate."""
    res = plan(wall_grid, (0, 1), (2, 1))
    assert res.cost == 4  # not 2: it has to go round the end of the wall


def test_vertex_constraint_forces_a_wait(open_grid):
    blocked = Constraint(0, open_grid.node((0, 1)), None, 1)
    res = plan(open_grid, (0, 0), (0, 2), [blocked])
    assert res.cost == 3
    assert res.path[1] != open_grid.node((0, 1))


def test_edge_constraint_blocks_only_that_move(open_grid):
    """An edge constraint forbids a transition, not the cell itself."""
    a, b = open_grid.node((0, 0)), open_grid.node((0, 1))
    res = plan(open_grid, (0, 0), (0, 1), [Constraint(0, b, a, 1)])
    assert res.found
    assert res.cost > 1
    assert res.path[-1] == b  # the cell is still reachable, just not that way


def test_goal_is_not_available_until_the_last_constraint_passes(open_grid):
    goal_rc = (0, 1)
    goal = open_grid.node(goal_rc)
    constraints = [Constraint(0, goal, None, t) for t in range(1, 6)]
    res = plan(open_grid, (0, 0), goal_rc, constraints)
    assert res.cost == 6
    assert res.path[-1] == goal
    assert all(loc != goal for loc in res.path[:-1])


def test_returns_nothing_when_the_goal_is_walled_off():
    grid = GridMap.from_rows(["..@..", "..@..", "..@.."])
    res = plan(grid, (0, 0), (0, 4))
    assert not res.found
    assert res.path is None


def test_start_blocked_at_time_zero_fails(open_grid):
    start = open_grid.node((0, 0))
    res = plan(open_grid, (0, 0), (0, 2), [Constraint(0, start, None, 0)])
    assert not res.found


def test_node_budget_stops_the_search(open_grid):
    res = plan(open_grid, (0, 0), (4, 4), node_budget=2)
    assert not res.found
    assert res.expanded <= 3


def test_agent_already_on_its_goal_returns_a_single_state(open_grid):
    res = plan(open_grid, (2, 2), (2, 2))
    assert res.cost == 0
    assert res.path == [open_grid.node((2, 2))]


def test_path_cost_and_location_helpers(open_grid):
    res = plan(open_grid, (0, 0), (0, 3))
    assert path_cost(res.path) == res.cost
    assert path_cost(None) == 0
    assert location_at(res.path, 99) == res.path[-1]
    with pytest.raises(ValueError):
        location_at(res.path, -1)


def test_search_without_a_table_is_unconstrained(open_grid):
    s, g = open_grid.node((0, 0)), open_grid.node((4, 4))
    h = backward_dijkstra(open_grid.graph, g)
    res = space_time_astar(open_grid.graph, s, g, h)
    assert res.cost == 8


def test_statistics_are_reported(open_grid):
    res = plan(open_grid, (0, 0), (4, 4))
    assert res.expanded > 0
    assert res.generated >= res.expanded
    assert len(res) == res.cost + 1
