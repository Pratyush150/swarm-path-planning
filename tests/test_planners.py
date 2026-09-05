"""The algorithm spec strings that name every planner in tables and figures."""

from __future__ import annotations

import pytest

from swarmplan.graph import GridMap
from swarmplan.planners import DEFAULT_SUITE, label_for, parse_spec, solve, solver_for
from swarmplan.solution import SOLVED


def test_plain_specs_parse():
    assert parse_spec("cbs") == {"kind": "cbs", "options": {}}
    assert parse_spec("pp")["kind"] == "pp"


def test_flags_and_values_parse():
    parsed = parse_spec("cbs:pc,bp,ds,dg")
    assert parsed["options"] == {
        "prioritise_conflicts": True,
        "bypass": True,
        "disjoint": True,
        "heuristic": "dg",
    }
    assert parse_spec("ecbs:w=1.25")["options"] == {"w": 1.25}
    assert parse_spec("pp:restarts=8")["options"] == {"restarts": 8}


def test_unknown_specs_are_rejected():
    with pytest.raises(ValueError):
        parse_spec("astar")
    with pytest.raises(ValueError):
        parse_spec("cbs:magic")


def test_labels_match_the_solver_labels():
    assert label_for("cbs") == "CBS"
    assert label_for("cbs:pc,bp,dg") == "CBS+PC+BP+DG"
    assert label_for("ecbs:w=1.1") == "ECBS(w=1.1)"
    assert label_for("pp:restarts=8") == "PP(restarts=8)"


def test_every_spec_in_the_default_suite_is_valid_and_uniquely_labelled():
    labels = [label_for(s) for s in DEFAULT_SUITE]
    assert len(set(labels)) == len(labels)


def test_solve_dispatches_to_each_family():
    grid = GridMap.from_rows([".....", "..@..", "....."])
    s = [grid.node((0, 0)), grid.node((0, 4))]
    g = [grid.node((0, 4)), grid.node((0, 0))]
    for spec in ("cbs", "cbs:pc,bp,dg", "ecbs:w=1.5", "pp"):
        result = solve(spec, grid.graph, s, g, time_limit=15.0)
        assert result.status == SOLVED, spec
        assert result.algorithm == label_for(spec)


def test_solver_for_binds_a_spec():
    grid = GridMap.from_rows(["....."])
    run = solver_for("ecbs:w=1.2")
    assert "ECBS" in run.__doc__
    result = run(grid.graph, [grid.node((0, 0))], [grid.node((0, 4))])
    assert result.status == SOLVED
