#!/usr/bin/env python3
"""Render every figure and animation the README displays.

Separate from ``run.py`` because these are illustrations rather than
measurements: they are produced from plans this script computes on the spot, and
they are the artefacts that get committed to the repository, so they are kept
small on purpose (quantised GIFs, modest dpi).

Usage::

    python3 benchmarks/make_figures.py --all
    python3 benchmarks/make_figures.py --animation --morph
    python3 benchmarks/make_figures.py --charts        # needs results.csv

Nothing here is hand-drawn or hand-edited. Every pixel comes from a plan
produced by the solvers in ``src/swarmplan``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swarmplan import datasets, viz  # noqa: E402
from swarmplan.assignment.anonymous import assign_goals, identity_assignment  # noqa: E402
from swarmplan.cbs.solver import CBS, CBSConfig  # noqa: E402
from swarmplan.conflicts import find_first_conflict, validate_plan  # noqa: E402
from swarmplan.graph import GridMap  # noqa: E402
from swarmplan.lightshow import (  # noqa: E402
    block_formation,
    build_airspace,
    plan_show,
    ring_formation,
    text_formation,
)
from swarmplan.lowlevel.heuristic import HeuristicCache  # noqa: E402
from swarmplan.metrics import singleton_lower_bound  # noqa: E402
from swarmplan.planners import solve  # noqa: E402
from swarmplan.scenarios import load_scen, make_instance  # noqa: E402

OUT = ROOT / "benchmarks" / "output"


def optimise_gif(path: Path, colours: int = 96) -> int:
    """Quantise a GIF in place and make it loop. Returns the new size in bytes.

    Matplotlib writes full-colour frames; a grid animation only needs a small
    palette, and quantising it typically takes a megabyte down to a couple of
    hundred kilobytes with no visible loss.
    """
    try:
        from PIL import Image, ImageSequence
    except ImportError:  # pragma: no cover - depends on the environment
        return path.stat().st_size
    with Image.open(path) as im:
        frames = [
            frame.convert("RGB").quantize(colors=colours, method=Image.MEDIANCUT)
            for frame in ImageSequence.Iterator(im)
        ]
        duration = im.info.get("duration", 100)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration,
        optimize=True,
        disposal=2,
    )
    return path.stat().st_size


def to_video(gif: Path, suffix: str = ".mp4") -> Optional[Path]:
    """Transcode a GIF to MP4/WebM if ffmpeg is available, else return ``None``."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    out = gif.with_suffix(suffix)
    cmd = [
        ffmpeg, "-y", "-loglevel", "error", "-i", str(gif),
        "-movflags", "faststart", "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(out),
    ]
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, OSError):  # pragma: no cover
        return None
    return out


def benchmark_animation(args) -> None:
    """The money shot: agents crossing a real benchmark map without colliding."""
    data_dir = datasets.default_data_dir()
    grid = GridMap.from_file(datasets.map_path(args.map, data_dir))
    scen = load_scen(datasets.scen_path(args.map, args.scen, "random", data_dir))
    instance = make_instance(grid, scen, args.agents)
    cache = HeuristicCache(grid.graph)
    lb = singleton_lower_bound(grid.graph, instance.starts, instance.goals, cache)
    started = time.perf_counter()
    result = solve(args.algorithm, grid.graph, instance.starts, instance.goals,
                   time_limit=args.time_limit, cache=cache)
    if not result.solved:
        print(f"animation: {args.algorithm} did not solve {args.map} with {args.agents} agents")
        return
    problems = validate_plan(result.paths, instance.starts, instance.goals, grid.graph)
    print(
        f"animation: {args.map}, {args.agents} agents, {result.algorithm}, "
        f"sum-of-costs {result.cost} ({result.cost / lb:.3f}x lower bound), "
        f"makespan {result.makespan}, {time.perf_counter() - started:.1f}s, "
        f"{len(problems)} validation problems"
    )
    gif = viz.animate_plan(
        grid,
        result.paths,
        OUT / "swarm-demo.gif",
        title=f"{args.map}: {args.agents} agents, {result.algorithm}, makespan {result.makespan}",
        fps=args.fps,
        dpi=args.dpi,
        figsize=(4.6, 4.8),
    )
    size = optimise_gif(gif, args.gif_colours)
    print(f"  {gif.name}: {size / 1024:.0f} KB")
    video = to_video(gif)
    if video:
        print(f"  {video.name}: {video.stat().st_size / 1024:.0f} KB")
    strip = viz.filmstrip(
        grid,
        result.paths,
        OUT / "swarm-filmstrip.png",
        n_frames=6,
        title=f"{args.map}, {args.agents} agents, {result.algorithm}",
    )
    print(f"  {strip.name}: {strip.stat().st_size / 1024:.0f} KB")


def _interleave(times: List[int]) -> List[int]:
    """Formation timesteps with the midpoint of each transition between them.

    A still taken only where the swarm is in formation shows no traffic at all:
    in formation every aircraft is in the display plane. The midpoints are where
    the depth axis is being used, which is the half of the problem the audience
    never sees.
    """
    out: List[int] = []
    for a, b in zip(times, times[1:]):
        out.extend([a, (a + b) // 2])
    out.append(times[-1])
    return out


def morph_figures(args) -> None:
    """Drone-light-show morph: block, ring, text, and back, assigned optimally."""
    size = (36, 7, 18)
    space = build_airspace(size)
    n = args.show_agents
    block = block_formation(space, n, width=16)
    ring = ring_formation(space, n, radius=min(size[0], size[2]) / 2.6)
    letters = text_formation(space, args.text, count=n)
    # Returning to the launch block makes the animation loop seamlessly.
    formations = [block, ring, letters, block]
    started = time.perf_counter()
    show = plan_show(space, formations, args.show_algorithm, time_limit=args.time_limit)
    if not show.solved:
        print("morph: at least one transition was not solved; nothing rendered")
        for tr in show.transitions:
            print(f"  transition {tr.index}: {tr.solution.status}")
        return
    problems = validate_plan(show.paths, graph=space.graph)
    print(
        f"morph: {n} aircraft, airspace {size}, {len(formations) - 1} transitions, "
        f"makespan {show.makespan}, planned in {time.perf_counter() - started:.1f}s, "
        f"{len(problems)} validation problems"
    )
    for tr in show.transitions:
        print(
            f"  transition {tr.index}: sum-of-costs {tr.solution.cost:5d}  "
            f"optimal assignment {tr.assignment.total_distance:6.0f} cells vs "
            f"arbitrary {tr.identity_total:6.0f} ({tr.improvement:.2f}x)"
        )
    gif = viz.animate_light_show(
        space, show.paths, OUT / "formation-morph.gif", fps=args.fps, dpi=args.dpi,
        title=f"{n} aircraft: launch block -> ring -> {args.text}",
    )
    size_kb = optimise_gif(gif, args.gif_colours) / 1024
    print(f"  {gif.name}: {size_kb:.0f} KB")
    video = to_video(gif)
    if video:
        print(f"  {video.name}: {video.stat().st_size / 1024:.0f} KB")
    stills = viz.plot_light_show(
        space, show.paths, OUT / "formation-morph.png",
        times=_interleave(show.in_formation_at[:3]),
        title=f"formation morph, {n} aircraft, optimal assignment "
              f"(in formation, and mid-transition)",
    )
    print(f"  {stills.name}: {stills.stat().st_size / 1024:.0f} KB")

    labels = ["arbitrary\n(agent i -> slot i)", "min-sum\nassignment", "min-max\nassignment"]
    totals, maxima = [], []
    current = block
    ident = identity_assignment(space.graph, current, ring)
    total_sum = assign_goals(space.graph, current, ring, objective="sum")
    total_max = assign_goals(space.graph, current, ring, objective="makespan")
    for res in (ident, total_sum, total_max):
        totals.append(res.total_distance)
        maxima.append(res.max_distance)
    chart = viz.plot_assignment_comparison(
        labels, totals, maxima, OUT / "assignment-comparison.png",
        title="unlabelled MAPF: what the goal assignment is worth (block to ring, "
              f"{n} aircraft)",
    )
    print(
        f"  assignment: arbitrary {totals[0]:.0f}/{maxima[0]:.0f}, "
        f"min-sum {totals[1]:.0f}/{maxima[1]:.0f}, min-max {totals[2]:.0f}/{maxima[2]:.0f} "
        f"(total/worst cells)"
    )
    print(f"  {chart.name}: {chart.stat().st_size / 1024:.0f} KB")


def conflict_figure(args) -> None:
    """Space-time diagram of one conflict, before and after CBS resolves it."""
    grid = GridMap.from_rows(
        [
            "..........",
            ".@@@@@@@@.",
            "..........",
        ],
        name="passing-bay",
    )
    starts = [grid.node((0, 0)), grid.node((0, 9))]
    goals = [grid.node((0, 9)), grid.node((0, 0))]
    cache = HeuristicCache(grid.graph)
    cbs = CBS(grid.graph, starts, goals, CBSConfig(time_limit=10.0), cache)
    root = cbs.initial_paths()
    conflict = find_first_conflict(root) if root else None
    result = cbs.solve()
    if root is None or not result.solved or conflict is None:
        print("conflict figure: no conflict or no solution; nothing rendered")
        return
    print(
        f"conflict: unconstrained plans collide as {conflict}; CBS resolves it with "
        f"sum-of-costs {result.cost} in {result.high_level_expanded} high-level nodes"
    )
    path = viz.plot_space_time(
        grid, root, result.paths, conflict, OUT / "conflict-spacetime.png"
    )
    print(f"  {path.name}: {path.stat().st_size / 1024:.0f} KB")


def charts(args) -> None:
    """Success-rate and runtime figures from the sweep results."""
    from run import read_csv  # local import: same directory

    csv_path = OUT / "results.csv"
    if not csv_path.exists():
        print("charts: benchmarks/output/results.csv not found; run benchmarks/run.py first")
        return
    records = [r for r in read_csv(csv_path) if r.status != "skipped"]
    maps = [m for m in args.chart_maps.split(",") if any(r.map_name == m for r in records)]
    if maps:
        p = viz.plot_success_rate_grid(
            records, maps, OUT / "success-rate.png",
            title="success rate vs number of agents (3 scenarios per point, 20 s budget, "
                  "one CPU-only Python process)",
        )
        print(f"  {p.name}: {p.stat().st_size / 1024:.0f} KB")
    p = viz.plot_runtime_distribution(
        records, OUT / "runtime-distribution.png",
        title="runtime per solved instance, all maps (log scale)",
    )
    print(f"  {p.name}: {p.stat().st_size / 1024:.0f} KB")


def main(argv=None) -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--all", action="store_true")
    p.add_argument("--animation", action="store_true")
    p.add_argument("--morph", action="store_true")
    p.add_argument("--conflict", action="store_true")
    p.add_argument("--charts", action="store_true")
    p.add_argument("--map", default="random-32-32-20")
    p.add_argument("--chart-maps", default="random-32-32-20,warehouse-10-20-10-2-1")
    p.add_argument("--scen", type=int, default=1)
    p.add_argument("--agents", type=int, default=30)
    p.add_argument("--algorithm", default="ecbs:w=1.1")
    p.add_argument("--show-agents", type=int, default=87,
                   help="87 is the number of cells in a 5x7 rendering of SWARM")
    p.add_argument("--show-algorithm", default="ecbs:w=1.2")
    p.add_argument("--gif-colours", type=int, default=64)
    p.add_argument("--text", default="SWARM")
    p.add_argument("--time-limit", type=float, default=120.0)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--dpi", type=int, default=66)
    args = p.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    if not viz.HAVE_MATPLOTLIB:
        print("matplotlib is required: pip install matplotlib pillow", file=sys.stderr)
        return 2
    any_flag = args.animation or args.morph or args.conflict or args.charts
    if args.all or not any_flag:
        args.animation = args.morph = args.conflict = args.charts = True
    if args.conflict:
        conflict_figure(args)
    if args.animation:
        benchmark_animation(args)
    if args.morph:
        morph_figures(args)
    if args.charts:
        charts(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
