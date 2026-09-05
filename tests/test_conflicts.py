"""Conflict detection, including the swap case naive implementations miss."""

from __future__ import annotations

import pytest

from swarmplan.conflicts import (
    Conflict,
    ConflictType,
    count_conflicts,
    find_all_conflicts,
    find_first_conflict,
    is_valid_plan,
    validate_plan,
)
from swarmplan.graph import GridMap


def test_vertex_conflict_is_found():
    paths = [[0, 1, 2], [4, 1, 6]]
    c = find_first_conflict(paths)
    assert c is not None
    assert (c.a1, c.a2, c.loc1, c.loc2, c.time) == (0, 1, 1, None, 1)
    assert not c.is_edge


def test_swap_conflict_is_found_although_no_cell_is_shared():
    """Agents 0 and 1 exchange cells between t=0 and t=1 and pass through each other.

    At no timestep are they in the same cell, so a per-timestep position check
    calls this plan valid. It is not.
    """
    paths = [[3, 4], [4, 3]]
    assert not any(a == b for a, b in zip(*paths))  # no shared cell at any time
    c = find_first_conflict(paths)
    assert c is not None
    assert c.is_edge
    assert (c.loc1, c.loc2, c.time) == (3, 4, 1)


def test_following_one_step_behind_is_not_a_conflict():
    """Moving into a cell the moment its occupant leaves is legal MAPF."""
    paths = [[0, 1, 2], [9, 0, 1]]
    assert find_first_conflict(paths) is None


def test_agents_parked_on_their_goals_still_block():
    paths = [[0, 1], [5, 6, 7, 1]]
    c = find_first_conflict(paths)
    assert c is not None and c.time == 3 and c.loc1 == 1


def test_find_all_conflicts_returns_every_pair():
    paths = [[0, 1, 2], [4, 1, 2], [8, 1, 9]]
    conflicts = find_all_conflicts(paths)
    assert count_conflicts(paths) == len(conflicts)
    pairs = {(c.a1, c.a2) for c in conflicts}
    assert pairs == {(0, 1), (0, 2), (1, 2)}


def test_valid_plan_reports_no_conflicts():
    assert is_valid_plan([[0, 1, 2], [9, 8, 7]])


def test_vertex_conflict_branches_into_two_constraints():
    c = Conflict(2, 5, 11, None, 7)
    a, b = c.constraints()
    assert (a.agent, a.loc, a.prev, a.time) == (2, 11, None, 7)
    assert (b.agent, b.loc, b.prev, b.time) == (5, 11, None, 7)


def test_edge_conflict_branches_into_opposite_moves():
    c = Conflict(1, 3, 4, 5, 9)
    a, b = c.constraints()
    assert (a.agent, a.prev, a.loc) == (1, 4, 5)
    assert (b.agent, b.prev, b.loc) == (3, 5, 4)


def test_disjoint_split_produces_a_positive_and_a_negative_constraint():
    c = Conflict(1, 3, 4, None, 9)
    pos, neg = c.disjoint_constraints()
    assert pos.positive and not neg.positive
    assert pos.agent == neg.agent == 1
    pos2, neg2 = c.disjoint_constraints(3)
    assert pos2.agent == 3
    with pytest.raises(ValueError):
        c.disjoint_constraints(7)


def test_disjoint_split_of_an_edge_conflict_uses_the_right_direction():
    c = Conflict(1, 3, 4, 5, 9)
    pos, _ = c.disjoint_constraints(3)
    assert (pos.prev, pos.loc) == (5, 4)


def test_conflict_types_are_ordered_by_how_much_they_cost():
    assert ConflictType.CARDINAL > ConflictType.SEMI_CARDINAL > ConflictType.NON_CARDINAL


def test_validate_plan_catches_teleports_and_wrong_endpoints():
    grid = GridMap.from_rows(["....."])
    starts = [grid.node((0, 0))]
    goals = [grid.node((0, 4))]
    teleport = [[grid.node((0, 0)), grid.node((0, 3)), grid.node((0, 4))]]
    problems = validate_plan(teleport, starts, goals, grid.graph)
    assert any("illegal move" in p for p in problems)
    wrong_end = [[grid.node((0, 0)), grid.node((0, 1))]]
    assert any("ends at" in p for p in validate_plan(wrong_end, starts, goals, grid.graph))
    assert validate_plan([[]], starts, goals, grid.graph) == ["agent 0: empty path"]


def test_validate_plan_accepts_a_correct_plan():
    grid = GridMap.from_rows(["....."])
    starts = [grid.node((0, 0))]
    goals = [grid.node((0, 2))]
    good = [[grid.node((0, 0)), grid.node((0, 1)), grid.node((0, 2))]]
    assert validate_plan(good, starts, goals, grid.graph) == []
