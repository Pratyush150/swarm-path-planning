"""Tests that need the movingai benchmark files. Skipped when they are absent."""

from __future__ import annotations

import pytest

from conftest import requires_data
from swarmplan import datasets
from swarmplan.conflicts import validate_plan
from swarmplan.graph import GridMap
from swarmplan.lowlevel.heuristic import HeuristicCache
from swarmplan.metrics import singleton_lower_bound
from swarmplan.planners import solve
from swarmplan.scenarios import load_scen, make_instance
from swarmplan.solution import SOLVED

pytestmark = requires_data


def test_registry_matches_the_files_on_disk():
    for info in datasets.MAPS:
        grid = GridMap.from_file(datasets.map_path(info.name))
        assert (grid.height, grid.width) == (info.height, info.width), info.name
        assert grid.n_free == info.free_cells, info.name


def test_missing_reports_nothing_when_everything_is_present():
    assert datasets.missing() == []
    assert datasets.available()


@pytest.mark.parametrize("map_name", ["random-32-32-20", "empty-32-32", "maze-32-32-2"])
def test_scenarios_reference_only_free_cells(map_name):
    grid = GridMap.from_file(datasets.map_path(map_name))
    scen = load_scen(datasets.scen_path(map_name, 1, "random"))
    assert len(scen) > 100
    for entry in scen.entries[:200]:
        assert grid.passable(entry.start)
        assert grid.passable(entry.goal)


def test_octile_column_is_a_weaker_bound_than_the_true_four_connected_distance():
    """The .scen column allows diagonal moves, so it sits below the 4-connected optimum.

    This is why the benchmark tables measure the ratio against the sum of true
    single-agent distances and only carry the octile figure for reference.
    """
    grid = GridMap.from_file(datasets.map_path("random-32-32-20"))
    scen = load_scen(datasets.scen_path("random-32-32-20", 1, "random"))
    cache = HeuristicCache(grid.graph)
    strictly_below = 0
    for entry in scen.entries[:120]:
        true_d = cache.distance(grid.node(entry.start), grid.node(entry.goal))
        assert entry.octile_optimal <= true_d + 1e-6
        if entry.octile_optimal < true_d - 1e-6:
            strictly_below += 1
    assert strictly_below > 40  # and it is usually strictly below, not equal


def test_ecbs_solves_a_real_instance_and_the_plan_validates():
    grid = GridMap.from_file(datasets.map_path("random-32-32-20"))
    scen = load_scen(datasets.scen_path("random-32-32-20", 1, "random"))
    instance = make_instance(grid, scen, 15)
    cache = HeuristicCache(grid.graph)
    lb = singleton_lower_bound(grid.graph, instance.starts, instance.goals, cache)
    result = solve("ecbs:w=1.1", grid.graph, instance.starts, instance.goals,
                   time_limit=60.0, cache=cache)
    assert result.status == SOLVED
    assert validate_plan(result.paths, instance.starts, instance.goals, grid.graph) == []
    assert lb <= result.cost <= 1.1 * lb + 1e-9


def test_cbs_and_ecbs_agree_within_the_bound_on_a_real_instance():
    grid = GridMap.from_file(datasets.map_path("random-32-32-20"))
    scen = load_scen(datasets.scen_path("random-32-32-20", 1, "random"))
    instance = make_instance(grid, scen, 8)
    cache = HeuristicCache(grid.graph)
    optimal = solve("cbs:pc,bp,dg", grid.graph, instance.starts, instance.goals,
                    time_limit=60.0, cache=cache)
    bounded = solve("ecbs:w=1.2", grid.graph, instance.starts, instance.goals,
                    time_limit=60.0, cache=cache)
    assert optimal.status == bounded.status == SOLVED
    assert optimal.cost <= bounded.cost <= 1.2 * optimal.cost + 1e-9


def test_the_warehouse_map_is_solvable_at_a_realistic_fleet_size():
    """The layout a fulfilment centre actually has: long aisles, no room to pass."""
    grid = GridMap.from_file(datasets.map_path("warehouse-10-20-10-2-1"))
    scen = load_scen(datasets.scen_path("warehouse-10-20-10-2-1", 1, "random"))
    instance = make_instance(grid, scen, 20)
    result = solve("ecbs:w=1.5", grid.graph, instance.starts, instance.goals, time_limit=60.0)
    assert result.status == SOLVED
    assert validate_plan(result.paths, instance.starts, instance.goals, grid.graph) == []
