"""Metrics, lower bounds and the aggregation the benchmark tables use."""

from __future__ import annotations

import math

import pytest

from swarmplan.graph import GridMap
from swarmplan.metrics import (
    RunRecord,
    cost_ratio,
    group_by,
    makespan,
    mean_ratio,
    runtime_stats,
    singleton_lower_bound,
    success_rate,
    sum_of_costs,
)
from swarmplan.solution import SOLVED, TIMEOUT, Solution


def test_sum_of_costs_and_makespan_are_different_numbers():
    paths = [[0, 1, 2, 3], [4, 5]]
    assert sum_of_costs(paths) == 4
    assert makespan(paths) == 3
    assert makespan([]) == 0


def test_singleton_lower_bound_sums_individual_optima():
    grid = GridMap.from_rows(["....."] * 3)
    starts = [grid.node((0, 0)), grid.node((2, 4))]
    goals = [grid.node((0, 4)), grid.node((2, 0))]
    assert singleton_lower_bound(grid.graph, starts, goals) == 8


def test_lower_bound_uses_the_true_distance_not_manhattan():
    grid = GridMap.from_rows([".....", ".@@@.", "....."])
    starts = [grid.node((0, 1))]
    goals = [grid.node((2, 1))]
    assert singleton_lower_bound(grid.graph, starts, goals) == 4  # not 2


def test_unreachable_goal_is_an_error():
    grid = GridMap.from_rows(["..@..", "..@..", "..@.."])
    with pytest.raises(ValueError):
        singleton_lower_bound(grid.graph, [grid.node((0, 0))], [grid.node((0, 4))])


def test_solution_reports_cost_makespan_and_conflicts():
    sol = Solution(paths=[[0, 1, 2], [9, 8]], status=SOLVED, algorithm="X")
    assert sol.solved
    assert sol.cost == 3
    assert sol.makespan == 2
    assert sol.n_agents == 2
    assert sol.conflicts() == []
    assert "soc=3" in sol.summary()
    unsolved = Solution(status=TIMEOUT, algorithm="X", runtime=1.5)
    assert not unsolved.solved
    assert unsolved.cost == -1 and unsolved.makespan == -1
    assert "timeout" in unsolved.summary()


def test_cost_ratio_needs_a_solved_run_and_a_bound():
    sol = Solution(paths=[[0, 1, 2]], status=SOLVED)
    assert cost_ratio(sol, 2) == pytest.approx(1.0)
    assert cost_ratio(sol, 0) is None
    assert cost_ratio(Solution(status=TIMEOUT), 5) is None


def records():
    """A handful of runs to aggregate."""
    return [
        RunRecord("m", "s1", 10, "A", SOLVED, 1.0, cost=110, lower_bound=100),
        RunRecord("m", "s2", 10, "A", TIMEOUT, 30.0),
        RunRecord("m", "s1", 10, "B", SOLVED, 0.5, cost=100, lower_bound=100),
        RunRecord("m", "s2", 10, "B", SOLVED, 2.5, cost=120, lower_bound=100),
    ]


def test_success_rate_counts_only_solved_runs():
    rows = records()
    assert success_rate(rows) == 0.75
    assert success_rate([r for r in rows if r.algorithm == "A"]) == 0.5
    assert success_rate([]) == 0.0


def test_runtime_stats_exclude_failures_by_default():
    rows = records()
    stats = runtime_stats(rows)
    assert stats["n"] == 3
    assert stats["max"] == 2.5  # not the 30 s timeout
    with_failures = runtime_stats(rows, solved_only=False)
    assert with_failures["max"] == 30.0
    assert math.isnan(runtime_stats([])["median"])


def test_ratio_to_lower_bound():
    rows = records()
    assert rows[0].ratio == pytest.approx(1.1)
    assert rows[1].ratio is None
    assert mean_ratio(rows) == pytest.approx((1.1 + 1.0 + 1.2) / 3)
    assert mean_ratio([rows[1]]) is None


def test_grouping_and_csv_row():
    rows = records()
    groups = group_by(rows, "algorithm")
    assert set(groups) == {("A",), ("B",)}
    row = rows[0].as_row()
    assert row["algorithm"] == "A" and row["ratio_to_lb"] == 1.1
    assert row["status"] == SOLVED
