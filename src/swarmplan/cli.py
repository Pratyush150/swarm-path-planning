"""Command line entry point: ``swarmplan <command>``.

``swarmplan demo`` runs end to end with no data, no network and no plotting
dependency -- it plans a small instance on a map defined in this file and prints
the plan as ASCII frames, so a reader can confirm the package works before
downloading an 8 MB benchmark archive.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from . import datasets
from .conflicts import validate_plan
from .execution.adg import ActionDependencyGraph, fixed_schedule_execution
from .execution.separation import separation_report
from .execution.smoothing import smooth_plan
from .graph import GridMap
from .lowlevel.heuristic import HeuristicCache
from .metrics import singleton_lower_bound
from .planners import DEFAULT_SUITE, solve
from .scenarios import load_scen, make_instance

#: A small map with two rooms and a single connecting door, used by ``demo``.
#: Every agent that crosses has to negotiate the same one-cell doorway, which is
#: the smallest instance that makes the point.
DEMO_MAP = [
    "..........",
    "....@@....",
    "....@@....",
    "..........",
    "@@@@.@@@@@",
    "..........",
    "....@@....",
    "....@@....",
    "..........",
]


def _render(grid: GridMap, paths: Sequence[Sequence[int]], t: int) -> List[str]:
    """One ASCII frame of a plan."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    rows = [
        ["#" if grid.blocked[r, c] else "." for c in range(grid.width)]
        for r in range(grid.height)
    ]
    for i, p in enumerate(paths):
        r, c = grid.rc(p[t] if t < len(p) else p[-1])
        rows[r][c] = letters[i % len(letters)]
    return ["".join(row) for row in rows]


def cmd_demo(args: argparse.Namespace) -> int:
    """Plan a small built-in instance and print it, with no external data."""
    grid = GridMap.from_rows(DEMO_MAP, name="two-rooms")
    starts = [grid.node(rc) for rc in [(3, 2), (5, 6), (0, 0), (0, 9), (8, 3)]]
    goals = [grid.node(rc) for rc in [(5, 6), (3, 2), (8, 9), (8, 0), (0, 6)]]
    cache = HeuristicCache(grid.graph)
    lb = singleton_lower_bound(grid.graph, starts, goals, cache)

    print(
        f"map: {grid.height}x{grid.width}, {grid.n_free} free cells, "
        "one door between the rooms"
    )
    print(f"agents: {len(starts)}   sum of individual optimal distances (lower bound): {lb}")
    print()
    for spec in ("cbs", "cbs:pc,bp,dg", "ecbs:w=1.2", "pp"):
        result = solve(spec, grid.graph, starts, goals, time_limit=args.time_limit, cache=cache)
        if result.solved:
            print(
                f"{result.algorithm:16s} {result.status:8s} sum-of-costs={result.cost:4d} "
                f"({result.cost / lb:.3f} x lower bound)  makespan={result.makespan:3d}  "
                f"high-level nodes={result.high_level_expanded:5d}  {result.runtime:.3f}s"
            )
        else:
            print(f"{result.algorithm:16s} {result.status}")

    result = solve(
        "cbs:pc,bp,dg", grid.graph, starts, goals, time_limit=args.time_limit, cache=cache
    )
    problems = validate_plan(result.paths, starts, goals, grid.graph)
    print(f"\nindependent validation of the optimal plan: {len(problems)} problems found")
    horizon = max(len(p) for p in result.paths)
    steps = [int(round(i * (horizon - 1) / 5)) for i in range(6)]
    print("\noptimal plan, sampled (agents a-e, # obstacle):\n")
    frames = [_render(grid, result.paths, t) for t in steps]
    print("   " + "   ".join(f"t={t:<8d}" for t in steps))
    for r in range(grid.height):
        print("   " + "   ".join(f[r] for f in frames))
    print()

    adg = ActionDependencyGraph(result.paths)
    delay = {(0, 2): 3}
    safe = adg.execute(delay)
    naive = fixed_schedule_execution(result.paths, delay)
    print("agent 0 delayed 3 ticks at step 2:")
    print(f"  fixed timetable execution: {len(naive.collisions())} collisions, {naive.ticks} ticks")
    print(f"  dependency-graph execution: {len(safe.collisions())} collisions, {safe.ticks} ticks")

    plan = smooth_plan(grid.graph, result.paths, cell_size=2.0, v_max=3.0, a_max=2.0)
    print("\ncontinuous-time execution check (2 m cells, 3 m/s, 2 m/s^2):")
    print("  " + separation_report(plan, 1.0).replace("\n", "\n  "))
    return 0


def cmd_solve(args: argparse.Namespace) -> int:
    """Solve one benchmark instance."""
    data_dir = Path(args.data_dir) if args.data_dir else datasets.default_data_dir()
    map_path = datasets.map_path(args.map, data_dir)
    scen_path = datasets.scen_path(args.map, args.scen, args.scen_kind, data_dir)
    if not map_path.exists() or not scen_path.exists():
        print(
            f"benchmark data not found ({map_path}). Run tools/fetch_benchmarks.py first.",
            file=sys.stderr,
        )
        return 2
    grid = GridMap.from_file(map_path)
    scen = load_scen(scen_path)
    instance = make_instance(grid, scen, args.agents)
    cache = HeuristicCache(grid.graph)
    lb = singleton_lower_bound(grid.graph, instance.starts, instance.goals, cache)
    print(f"{args.map} scen {args.scen} ({args.scen_kind}), {args.agents} agents")
    print(
        f"lower bound (sum of individual optima): {lb}    "
        f".scen octile column: {instance.octile_lower_bound:.1f}"
    )
    specs = args.algorithm or DEFAULT_SUITE
    for spec in specs:
        result = solve(
            spec, grid.graph, instance.starts, instance.goals,
            time_limit=args.time_limit, cache=cache,
        )
        if result.solved:
            problems = validate_plan(result.paths, instance.starts, instance.goals, grid.graph)
            flag = "ok" if not problems else f"INVALID ({len(problems)})"
            print(
                f"  {result.algorithm:18s} {result.cost:6d}  {result.cost / lb:6.3f}x  "
                f"makespan {result.makespan:4d}  nodes {result.high_level_expanded:6d}  "
                f"{result.runtime:7.2f}s  {flag}"
            )
        else:
            print(f"  {result.algorithm:18s} {result.status:>8s}  after {result.runtime:.2f}s")
    return 0


def cmd_execute(args: argparse.Namespace) -> int:
    """Plan a benchmark instance, then execute it with agents running late.

    Reports the two things the execution layer exists to separate: what a
    fixed timetable does when a vehicle slips, and what the dependency graph
    does with the same plan and the same delays.
    """
    data_dir = Path(args.data_dir) if args.data_dir else datasets.default_data_dir()
    map_path = datasets.map_path(args.map, data_dir)
    scen_path = datasets.scen_path(args.map, args.scen, args.scen_kind, data_dir)
    if not map_path.exists() or not scen_path.exists():
        print(
            f"benchmark data not found ({map_path}). Run tools/fetch_benchmarks.py first.",
            file=sys.stderr,
        )
        return 2
    grid = GridMap.from_file(map_path)
    scen = load_scen(scen_path)
    instance = make_instance(grid, scen, args.agents)
    result = solve(
        args.algorithm, grid.graph, instance.starts, instance.goals, time_limit=args.time_limit
    )
    if not result.solved:
        print(f"{result.algorithm}: {result.status} on {args.map} with {args.agents} agents")
        return 1

    adg = ActionDependencyGraph(result.paths)
    rng = random.Random(args.seed)
    delays = {}
    for agent in rng.sample(range(len(result.paths)), min(args.delayed, len(result.paths))):
        last = len(result.paths[agent]) - 1
        if last < 1:
            continue
        delays[(agent, rng.randint(1, last))] = args.delay_ticks

    safe = adg.execute(delays)
    naive = fixed_schedule_execution(result.paths, delays)
    print(
        f"{args.map}, {args.agents} agents, {result.algorithm}: "
        f"sum-of-costs {result.cost}, makespan {result.makespan}"
    )
    print(
        f"dependency graph: {adg.n_dependencies} ordering edges, "
        f"{adg.n_cross_agent} of them between different agents"
    )
    print(
        f"delaying {len(delays)} agents by {args.delay_ticks} ticks each "
        f"(seed {args.seed}):"
    )
    print(
        f"  fixed timetable execution:  {len(naive.collisions()):3d} collisions, "
        f"{naive.ticks} ticks"
    )
    print(
        f"  dependency-graph execution: {len(safe.collisions()):3d} collisions, "
        f"{safe.ticks} ticks"
    )
    flown = [[row[a] for row in safe.positions] for a in range(len(result.paths))]
    plan = smooth_plan(
        grid.graph, flown, cell_size=args.cell_size, v_max=args.v_max, a_max=args.a_max
    )
    print(
        f"\ncontinuous-time check of the delayed execution "
        f"({args.cell_size:g} m cells, {args.v_max:g} m/s, {args.a_max:g} m/s^2):"
    )
    print("  " + separation_report(plan, args.min_separation).replace("\n", "\n  "))
    return 0 if safe.is_safe() else 1


def cmd_show(args: argparse.Namespace) -> int:
    """Plan the drone-light-show demonstration and report the assignment gain."""
    from .lightshow import default_show

    show = default_show(
        n_agents=args.agents, text=args.text, algorithm=args.algorithm, time_limit=args.time_limit
    )
    print(f"airspace {show.space.size}, {show.n_agents} aircraft, text {args.text!r}")
    for tr in show.transitions:
        s = tr.solution
        print(
            f"  transition {tr.index}: {s.status:8s} sum-of-costs {s.cost:6d}  "
            f"makespan {s.makespan:4d}  {s.runtime:6.2f}s"
        )
        print(
            f"    assignment: optimal total {tr.assignment.total_distance:.0f} cells vs "
            f"arbitrary {tr.identity_total:.0f} cells "
            f"({tr.improvement:.2f}x further without it)"
        )
    if show.solved:
        problems = validate_plan(show.paths, graph=show.space.graph)
        print(f"  whole show: makespan {show.makespan}, validation problems {len(problems)}")
    return 0 if show.solved else 1


def cmd_data(args: argparse.Namespace) -> int:
    """Report which benchmark files are present."""
    data_dir = Path(args.data_dir) if args.data_dir else datasets.default_data_dir()
    print(f"data directory: {data_dir}")
    print(f"available: {datasets.available(data_dir)}")
    for m in datasets.MAPS:
        path = datasets.map_path(m.name, data_dir)
        state = "present" if path.exists() else "missing"
        print(f"  {m.name:24s} {state:8s} {m.height}x{m.width}  {m.why}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    p = argparse.ArgumentParser(prog="swarmplan", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="run a self-contained demo, no data needed")
    d.add_argument("--time-limit", type=float, default=15.0)
    d.set_defaults(func=cmd_demo)

    s = sub.add_parser("solve", help="solve one benchmark instance")
    s.add_argument("--map", default="random-32-32-20")
    s.add_argument("--scen", type=int, default=1)
    s.add_argument("--scen-kind", choices=("random", "even"), default="random")
    s.add_argument("--agents", type=int, default=20)
    s.add_argument("--algorithm", action="append", help="algorithm spec, repeatable")
    s.add_argument("--time-limit", type=float, default=30.0)
    s.add_argument("--data-dir", default=None)
    s.set_defaults(func=cmd_solve)

    w = sub.add_parser("show", help="plan the drone-light-show demonstration")
    w.add_argument("--agents", type=int, default=80)
    w.add_argument("--text", default="SWARM")
    w.add_argument("--algorithm", default="ecbs:w=1.1")
    w.add_argument("--time-limit", type=float, default=120.0)
    w.set_defaults(func=cmd_show)

    e = sub.add_parser("execute", help="plan a benchmark instance and fly it with delays")
    e.add_argument("--map", default="random-32-32-20")
    e.add_argument("--scen", type=int, default=1)
    e.add_argument("--scen-kind", choices=("random", "even"), default="random")
    e.add_argument("--agents", type=int, default=30)
    e.add_argument("--algorithm", default="ecbs:w=1.1")
    e.add_argument("--delayed", type=int, default=2, help="how many agents run late")
    e.add_argument("--delay-ticks", type=int, default=5)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--cell-size", type=float, default=2.0, help="metres per grid cell")
    e.add_argument("--v-max", type=float, default=3.0)
    e.add_argument("--a-max", type=float, default=2.0)
    e.add_argument("--min-separation", type=float, default=1.5)
    e.add_argument("--time-limit", type=float, default=30.0)
    e.add_argument("--data-dir", default=None)
    e.set_defaults(func=cmd_execute)

    v = sub.add_parser("data", help="report which benchmark files are present")
    v.add_argument("--data-dir", default=None)
    v.set_defaults(func=cmd_data)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the ``swarmplan`` console script."""
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
