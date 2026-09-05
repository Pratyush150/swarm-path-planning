"""The drone-light-show pipeline: formations, assignment, routing, concatenation."""

from __future__ import annotations

import pytest

from swarmplan.conflicts import validate_plan
from swarmplan.lightshow import (
    block_formation,
    build_airspace,
    plan_show,
    ring_formation,
    text_formation,
)


def small_show():
    """A deliberately small airspace so the test runs in a second."""
    space = build_airspace((16, 5, 12))
    n = 24
    block = block_formation(space, n, width=8)
    ring = ring_formation(space, n, radius=4.0)
    return space, n, block, ring


def test_formations_have_distinct_slots_inside_the_airspace():
    space, n, block, ring = small_show()
    for formation in (block, ring):
        assert len(formation) == n
        assert len(set(formation)) == n
        for node in formation:
            x, y, z = space.xyz(node)
            assert 0 <= x < space.size[0] and 0 <= y < space.size[1] and 0 <= z < space.size[2]


def test_text_formation_is_thinned_to_the_requested_count():
    space = build_airspace((36, 5, 14))
    letters = text_formation(space, "SWARM", count=40)
    assert len(letters) == 40
    assert len(set(letters)) == 40


def test_a_morph_is_planned_conflict_free_and_ends_in_formation():
    space, n, block, ring = small_show()
    show = plan_show(space, [block, ring], algorithm="ecbs:w=1.5", time_limit=60.0)
    assert show.solved
    assert show.n_agents == n
    assert validate_plan(show.paths, graph=space.graph) == []
    assert sorted(p[-1] for p in show.paths) == sorted(ring)


def test_optimal_assignment_is_recorded_against_the_arbitrary_one():
    space, n, block, ring = small_show()
    show = plan_show(space, [block, ring], algorithm="ecbs:w=1.5", time_limit=60.0)
    transition = show.transitions[0]
    assert transition.assignment.total_distance <= transition.identity_total
    assert transition.improvement >= 1.0


def test_several_transitions_concatenate_into_one_plan():
    space, n, block, ring = small_show()
    letters = text_formation(space, "AB", count=n)
    show = plan_show(space, [block, ring, letters], algorithm="ecbs:w=1.5", time_limit=90.0)
    assert show.solved
    assert len(show.transitions) == 2
    assert validate_plan(show.paths, graph=space.graph) == []
    assert show.makespan >= max(t.solution.makespan for t in show.transitions)
    assert all(p[0] in block for p in show.paths)


def test_shows_need_two_formations_of_equal_size():
    space, n, block, ring = small_show()
    with pytest.raises(ValueError):
        plan_show(space, [block])
    with pytest.raises(ValueError):
        plan_show(space, [block, ring[:-1]])
