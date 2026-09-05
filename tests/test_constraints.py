"""Constraint tables: negatives, positives, reservations, and goal release."""

from __future__ import annotations

import pytest

from swarmplan.constraints import (
    Constraint,
    ConstraintTable,
    build_table,
    constraints_for,
)


def test_vertex_constraint_blocks_only_its_own_agent_and_time():
    table = build_table(1, [Constraint(1, 7, None, 3)])
    assert table.blocked(7, 3)
    assert not table.blocked(7, 4)
    assert not table.blocked(8, 3)
    other = build_table(2, [Constraint(1, 7, None, 3)])
    assert not other.blocked(7, 3)


def test_edge_constraint_is_directional():
    table = build_table(0, [Constraint(0, 5, 4, 2)])
    assert table.blocked_move(4, 5, 2)
    assert not table.blocked_move(5, 4, 2)
    assert not table.blocked_move(4, 5, 3)
    assert not table.blocked(5, 2)


def test_positive_constraint_pins_its_own_agent():
    table = build_table(0, [Constraint(0, 9, None, 4, positive=True)])
    assert not table.blocked(9, 4)
    assert table.blocked(8, 4)  # everything else at that timestep is forbidden
    assert not table.blocked(8, 5)


def test_positive_constraint_excludes_every_other_agent():
    table = build_table(3, [Constraint(0, 9, None, 4, positive=True)])
    assert table.blocked(9, 4)


def test_positive_edge_constraint_pins_both_ends():
    table = build_table(0, [Constraint(0, 9, 8, 4, positive=True)])
    assert table.landmark_edge[4] == (8, 9)
    assert not table.blocked(8, 3)
    assert table.blocked(7, 3)
    assert table.blocked_move(7, 9, 4)


def test_contradictory_positive_constraints_are_rejected():
    table = ConstraintTable(0)
    table.add(Constraint(0, 1, None, 2, positive=True))
    with pytest.raises(ValueError):
        table.add(Constraint(0, 5, None, 2, positive=True))


def test_reserved_path_blocks_cells_and_head_on_swaps():
    table = ConstraintTable(1)
    table.reserve_path([10, 11, 12])
    assert table.blocked(11, 1)
    assert table.blocked_move(11, 10, 1)  # the swap against their move 10 -> 11
    assert not table.blocked_move(10, 11, 1)
    assert table.blocked(12, 99)  # they parked on their goal


def test_goal_release_waits_for_the_last_constraint_on_the_goal():
    table = build_table(0, [Constraint(0, 4, None, 6)], goal=4)
    assert table.goal_release == 7
    empty = build_table(0, [], goal=4)
    assert empty.goal_release == 0


def test_goal_release_accounts_for_landmarks_elsewhere():
    table = build_table(0, [Constraint(0, 9, None, 5, positive=True)], goal=4)
    assert table.goal_release == 6


def test_constraints_for_selects_own_and_positive_constraints():
    cs = [
        Constraint(0, 1, None, 1),
        Constraint(1, 2, None, 1),
        Constraint(1, 3, None, 2, positive=True),
    ]
    picked = constraints_for(cs, 0)
    assert len(picked) == 2
    assert picked[0].agent == 0 and picked[1].positive


def test_blocked_forever_reports_permanently_occupied_cells():
    table = ConstraintTable(0)
    table.reserve_path([7])
    assert table.blocked_forever(7)
    assert not table.blocked_forever(8)


def test_repr_and_max_time_track_the_contents():
    table = build_table(0, [Constraint(0, 1, None, 5), Constraint(0, 2, 1, 9)])
    assert table.max_time == 9
    assert "a0" in repr(table)
    assert str(Constraint(0, 1, None, 5)).startswith("-(a0")
    assert str(Constraint(0, 2, 1, 5, positive=True)).startswith("+(a0")
