"""Smoothing to a flyable trajectory, and separation in continuous time."""

from __future__ import annotations

import numpy as np
import pytest

from swarmplan.cbs.solver import solve_cbs
from swarmplan.execution.separation import (
    min_separation,
    pairwise_min_separation,
    separation_report,
    separation_violations,
)
from swarmplan.execution.smoothing import (
    path_coordinates,
    smooth_plan,
    smooth_waypoints,
)
from swarmplan.graph import GridMap


def crossing_plan():
    """Three agents crossing a corridor with a passing bay."""
    grid = GridMap.from_rows([".....", "..@..", "....."])
    s = [grid.node(x) for x in [(0, 0), (0, 4), (2, 0)]]
    g = [grid.node(x) for x in [(0, 4), (0, 0), (2, 4)]]
    result = solve_cbs(grid.graph, s, g, time_limit=20.0)
    assert result.solved
    return grid, result.paths


@pytest.mark.parametrize("v_max,a_max", [(1.0, 1.0), (2.0, 0.5), (5.0, 4.0)])
def test_smoothed_trajectory_respects_both_limits(v_max, a_max):
    grid, paths = crossing_plan()
    plan = smooth_plan(grid.graph, paths, cell_size=1.0, v_max=v_max, a_max=a_max)
    assert plan.peak_speed() <= v_max + 1e-6
    assert plan.peak_acceleration() <= a_max + 1e-6
    assert plan.within_limits()
    assert plan.duration > 0


def test_tighter_limits_take_longer():
    grid, paths = crossing_plan()
    fast = smooth_plan(grid.graph, paths, v_max=4.0, a_max=4.0)
    slow = smooth_plan(grid.graph, paths, v_max=0.5, a_max=0.5)
    assert slow.duration > fast.duration


def test_smoothing_never_moves_a_waypoint_beyond_its_bound():
    """The cap is what keeps the smoothed path inside the corridor CBS cleared."""
    wps = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [2.0, 2.0]])
    for cap in (0.0, 0.1, 0.35, 1.0):
        out = smooth_waypoints(wps, passes=6, max_deviation=cap)
        assert np.all(np.linalg.norm(out - wps, axis=1) <= cap + 1e-9)
    assert np.allclose(smooth_waypoints(wps, passes=0), wps)


def test_endpoints_are_never_moved():
    wps = np.array([[0.0, 0.0], [0.0, 3.0], [3.0, 3.0]])
    out = smooth_waypoints(wps, passes=4, max_deviation=1.0)
    assert np.allclose(out[0], wps[0])
    assert np.allclose(out[-1], wps[-1])


def test_trajectory_starts_and_ends_on_the_planned_cells():
    grid, paths = crossing_plan()
    plan = smooth_plan(grid.graph, paths, cell_size=2.0)
    for i, path in enumerate(paths):
        start = np.array(grid.graph.coord(path[0]), dtype=float) * 2.0
        end = np.array(grid.graph.coord(path[-1]), dtype=float) * 2.0
        assert np.allclose(plan.positions[i][0], start)
        assert np.allclose(plan.positions[i][-1], end)


def test_separation_is_exact_between_samples_not_just_at_them():
    """Two agents crossing are closest halfway through a step, never at one."""
    grid = GridMap.from_rows(["...", "...", "..."])
    paths = [
        [grid.node((0, 0)), grid.node((1, 0)), grid.node((2, 0))],
        [grid.node((2, 0)), grid.node((1, 0)), grid.node((0, 0))],
    ]
    # That plan is a vertex conflict on purpose: the separation check must see
    # the agents meet even though it only looks at continuous positions.
    plan = smooth_plan(grid.graph, paths, cell_size=1.0, samples_per_step=4)
    assert min_separation(plan) < 1e-6


def test_grid_legal_plans_can_still_violate_a_separation_requirement():
    """Adjacent cells are one cell apart, which is not always far enough."""
    grid, paths = crossing_plan()
    plan = smooth_plan(grid.graph, paths, cell_size=1.0)
    worst = min_separation(plan)
    assert worst < 1.5
    assert separation_violations(plan, 1.5)
    assert not separation_violations(plan, worst * 0.5)


def test_pairwise_matrix_is_symmetric_with_infinite_diagonal():
    grid, paths = crossing_plan()
    plan = smooth_plan(grid.graph, paths)
    m = pairwise_min_separation(plan)
    assert np.allclose(m, m.T, equal_nan=True)
    assert np.all(np.isinf(np.diag(m)))


def test_report_mentions_the_limits_and_the_closest_approach():
    grid, paths = crossing_plan()
    plan = smooth_plan(grid.graph, paths, cell_size=2.0, v_max=3.0, a_max=2.0)
    text = separation_report(plan, 1.0)
    assert "peak speed" in text and "closest approach" in text
    assert "violations" in text


def test_path_coordinates_scale_with_cell_size():
    grid = GridMap.from_rows(["...", "..."])
    path = [grid.node((0, 0)), grid.node((0, 1))]
    coords = path_coordinates(grid.graph, path, cell_size=3.0)
    assert np.allclose(coords, [[0.0, 0.0], [0.0, 3.0]])


def test_invalid_limits_are_rejected():
    grid, paths = crossing_plan()
    with pytest.raises(ValueError):
        smooth_plan(grid.graph, paths, v_max=0.0)
    with pytest.raises(ValueError):
        smooth_plan(grid.graph, [])
