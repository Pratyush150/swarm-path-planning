"""Figure generation. Skipped when matplotlib is absent; nothing else imports it."""

from __future__ import annotations

import pytest

from conftest import requires_matplotlib
from swarmplan.cbs.solver import CBS, CBSConfig, solve_cbs
from swarmplan.conflicts import find_first_conflict
from swarmplan.graph import GridMap
from swarmplan.lightshow import block_formation, build_airspace, plan_show, ring_formation
from swarmplan.metrics import RunRecord
from swarmplan.solution import SOLVED, TIMEOUT


def small_plan():
    """A two-agent plan on a corridor with a passing bay."""
    grid = GridMap.from_rows([".....", "..@..", "....."])
    s = [grid.node((0, 0)), grid.node((0, 4))]
    g = [grid.node((0, 4)), grid.node((0, 0))]
    return grid, solve_cbs(grid.graph, s, g, time_limit=20.0)


def test_viz_module_imports_without_matplotlib_installed():
    """The guard exists so the package works in a bare environment."""
    from swarmplan import viz

    assert isinstance(viz.HAVE_MATPLOTLIB, bool)


@requires_matplotlib
def test_animation_and_filmstrip_are_written(tmp_path):
    from swarmplan import viz

    grid, result = small_plan()
    gif = viz.animate_plan(grid, result.paths, tmp_path / "plan.gif", fps=4, dpi=50)
    assert gif.exists() and gif.stat().st_size > 0
    strip = viz.filmstrip(grid, result.paths, tmp_path / "strip.png", n_frames=3, dpi=60)
    assert strip.exists() and strip.stat().st_size > 0


@requires_matplotlib
def test_space_time_figure_is_written(tmp_path):
    from swarmplan import viz

    grid = GridMap.from_rows(["......", ".@@@@.", "......"])
    s = [grid.node((0, 0)), grid.node((0, 5))]
    g = [grid.node((0, 5)), grid.node((0, 0))]
    cbs = CBS(grid.graph, s, g, CBSConfig(time_limit=20.0))
    root = cbs.initial_paths()
    conflict = find_first_conflict(root)
    result = cbs.solve()
    assert conflict is not None and result.status == SOLVED
    out = viz.plot_space_time(grid, root, result.paths, conflict, tmp_path / "st.png", dpi=60)
    assert out.exists()


@requires_matplotlib
def test_light_show_renders_both_projections(tmp_path):
    from swarmplan import viz

    space = build_airspace((14, 5, 10))
    block = block_formation(space, 12, width=6)
    ring = ring_formation(space, 12, radius=3.5)
    show = plan_show(space, [block, ring], algorithm="ecbs:w=1.5", time_limit=60.0)
    assert show.solved
    stills = viz.plot_light_show(space, show.paths, tmp_path / "show.png", dpi=60)
    assert stills.exists()
    gif = viz.animate_light_show(space, show.paths, tmp_path / "show.gif", fps=4, dpi=50)
    assert gif.exists()


@requires_matplotlib
def test_charts_are_written_from_records(tmp_path):
    from swarmplan import viz

    records = [
        RunRecord("m", "s1", n, alg, SOLVED if n < 30 else TIMEOUT, 0.1 * n, cost=10 * n,
                  lower_bound=9 * n)
        for n in (10, 20, 30)
        for alg in ("CBS", "ECBS(w=1.1)")
    ]
    success = viz.plot_success_rate(records, tmp_path / "success.png", dpi=60)
    grid = viz.plot_success_rate_grid(records, ["m"], tmp_path / "success-grid.png", dpi=60)
    assert grid.exists()
    runtime = viz.plot_runtime_distribution(records, tmp_path / "runtime.png", dpi=60)
    assert success.exists() and runtime.exists()
    bars = viz.plot_assignment_comparison(
        ["a", "b"], [100.0, 60.0], [12.0, 8.0], tmp_path / "assign.png", dpi=60
    )
    assert bars.exists()


@requires_matplotlib
def test_isometric_projection_is_a_linear_map():
    import numpy as np

    from swarmplan import viz

    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    out = viz.isometric(pts)
    assert out.shape == (4, 2)
    # x and y move the projected point in opposite horizontal directions, and z
    # only moves it vertically.
    assert out[1][0] > 0 > out[2][0]
    assert out[3][0] == pytest.approx(0.0)
    assert out[3][1] == pytest.approx(1.0)
