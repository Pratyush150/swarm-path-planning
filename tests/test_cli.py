"""The command line, including the no-data demo path."""

from __future__ import annotations

import pytest

from conftest import requires_data
from swarmplan.cli import DEMO_MAP, build_parser, main
from swarmplan.graph import GridMap


def test_demo_runs_without_any_data(capsys):
    """The quickstart claim: one command, no downloads, real output."""
    assert main(["demo", "--time-limit", "20"]) == 0
    out = capsys.readouterr().out
    assert "sum-of-costs" in out
    assert "0 problems found" in out
    assert "dependency-graph execution: 0 collisions" in out
    assert "closest approach" in out


def test_demo_map_is_two_rooms_with_one_door():
    grid = GridMap.from_rows(DEMO_MAP)
    doors = [c for c in range(grid.width) if grid.passable((4, c))]
    assert len(doors) == 1


def test_data_command_reports_status(capsys):
    assert main(["data"]) == 0
    out = capsys.readouterr().out
    assert "data directory" in out
    assert "random-32-32-20" in out


def test_solve_without_data_fails_cleanly(tmp_path, capsys):
    code = main(["solve", "--data-dir", str(tmp_path), "--agents", "2"])
    assert code == 2
    assert "benchmark data not found" in capsys.readouterr().err


@requires_data
def test_solve_runs_on_a_real_instance(capsys):
    assert main(["solve", "--agents", "6", "--algorithm", "ecbs:w=1.2", "--time-limit", "20"]) == 0
    out = capsys.readouterr().out
    assert "lower bound" in out
    assert "ok" in out


@requires_data
def test_execute_command_reports_both_execution_models(capsys):
    """The delayed execution has to be safe, and the naive one has to be shown."""
    code = main([
        "execute", "--agents", "10", "--delayed", "2", "--delay-ticks", "4",
        "--algorithm", "ecbs:w=1.2", "--time-limit", "30",
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "ordering edges" in out
    assert "dependency-graph execution:   0 collisions" in out
    assert "continuous-time check" in out


def test_execute_without_data_fails_cleanly(tmp_path, capsys):
    assert main(["execute", "--data-dir", str(tmp_path), "--agents", "4"]) == 2
    assert "benchmark data not found" in capsys.readouterr().err


def test_show_command_plans_a_small_show(capsys):
    assert main(["show", "--agents", "20", "--text", "AB", "--algorithm", "ecbs:w=1.5",
                 "--time-limit", "90"]) == 0
    out = capsys.readouterr().out
    assert "aircraft" in out
    assert "assignment:" in out


def test_parser_requires_a_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
