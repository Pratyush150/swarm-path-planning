"""The Action Dependency Graph must keep a plan safe when an agent runs late."""

from __future__ import annotations

import pytest

from swarmplan.cbs.solver import solve_cbs
from swarmplan.execution.adg import ActionDependencyGraph, fixed_schedule_execution
from swarmplan.graph import GridMap


def crossing_plan():
    """A plan where one agent follows another through a shared corridor cell."""
    grid = GridMap.from_rows([".....", "..@..", "....."])
    s = [grid.node(x) for x in [(0, 0), (0, 4), (2, 0)]]
    g = [grid.node(x) for x in [(0, 4), (0, 0), (2, 4)]]
    result = solve_cbs(grid.graph, s, g, time_limit=20.0)
    assert result.solved
    return grid, result.paths


def test_undelayed_execution_reproduces_the_plan():
    grid, paths = crossing_plan()
    trace = ActionDependencyGraph(paths).execute()
    assert trace.ticks == max(len(p) - 1 for p in paths)
    assert trace.is_safe()
    for i, p in enumerate(paths):
        assert trace.positions[trace.ticks][i] == p[-1]


def test_a_delayed_agent_does_not_cause_a_collision():
    """The headline property: late is late, not unsafe."""
    grid, paths = crossing_plan()
    adg = ActionDependencyGraph(paths)
    for step in range(1, min(len(p) for p in paths)):
        for delay in (1, 3, 7):
            trace = adg.execute({(0, step): delay})
            assert trace.is_safe(), f"collision after delaying agent 0 at step {step}"
            assert not trace.deadlocked


def test_the_fixed_timetable_does_collide_on_the_same_plan():
    """Without the dependency graph, the same delay produces collisions."""
    grid, paths = crossing_plan()
    collided = False
    for step in range(1, min(len(p) for p in paths)):
        trace = fixed_schedule_execution(paths, {(0, step): 3})
        if trace.collisions():
            collided = True
    assert collided, "expected the naive timetable execution to break somewhere"


def test_several_agents_delayed_at_once_stays_safe():
    grid, paths = crossing_plan()
    adg = ActionDependencyGraph(paths)
    trace = adg.execute({(0, 1): 4, (1, 2): 2, (2, 1): 6})
    assert trace.is_safe()
    assert trace.ticks >= max(len(p) - 1 for p in paths)


def test_delay_pushes_out_the_makespan_but_only_for_the_agents_behind():
    grid, paths = crossing_plan()
    adg = ActionDependencyGraph(paths)
    base = adg.execute()
    delayed = adg.execute({(0, 1): 5})
    assert delayed.ticks > base.ticks
    assert delayed.completed_at[0] > base.completed_at[0]


def test_dependencies_are_built_from_shared_cells():
    paths = [[0, 1, 2], [3, 0, 1]]
    adg = ActionDependencyGraph(paths)
    # Agent 1 enters cell 0 at step 1, which agent 0 leaves at step 1.
    assert (0, 1) in adg.predecessors((1, 1))
    assert adg.n_cross_agent >= 1
    assert adg.n_dependencies >= adg.n_cross_agent


def test_a_plan_with_a_collision_is_rejected():
    with pytest.raises(ValueError):
        ActionDependencyGraph([[0, 1], [4, 1]])


def test_a_plan_where_an_agent_parks_on_another_route_is_rejected():
    with pytest.raises(ValueError):
        ActionDependencyGraph([[0, 1], [5, 6, 7, 1]])


def test_no_following_mode_adds_separation():
    paths = [[0, 1, 2], [3, 0, 1]]
    tight = ActionDependencyGraph(paths, allow_following=True).execute()
    spaced = ActionDependencyGraph(paths, allow_following=False).execute()
    assert tight.is_safe() and spaced.is_safe()
    assert spaced.ticks >= tight.ticks


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        ActionDependencyGraph([])


def test_trace_reports_collisions_when_asked():
    from swarmplan.execution.adg import ExecutionTrace

    trace = ExecutionTrace(positions=[[0, 1], [1, 0]], completed_at=[1, 1], ticks=1)
    kinds = {k for *_, k in trace.collisions()}
    assert "swap" in kinds
    assert not trace.is_safe()
