#!/usr/bin/env python3
"""The benchmark sweep: maps x agent counts x algorithms, with a time budget.

Every number in the README comes from this script. It runs the standard
Sturtevant/movingai MAPF instances -- the same maps and the same start/goal
pairs the literature reports on -- writes one CSV row per run, and renders the
success-rate and runtime figures from that CSV.

Usage::

    python3 benchmarks/run.py --quick              # a few minutes, smoke test
    python3 benchmarks/run.py                      # the full sweep
    python3 benchmarks/run.py --figures-only       # re-render from results.csv

Honest reading of the output
----------------------------
* Runtimes are from **one CPU-only laptop-class machine running CPython**, with
  the timeout stated in the CSV. They are not comparable with a tuned C++
  implementation and are not offered as such.
* Runtime includes building the true-distance heuristic tables for the instance,
  because a deployed planner pays that too. Each run gets a fresh cache so no
  algorithm benefits from a previous one's work.
* An algorithm that solves none of the instances at some agent count is not run
  at higher counts on that map; the curve simply ends. Those points are recorded
  in the CSV as ``skipped`` and are excluded from every average.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swarmplan import datasets  # noqa: E402
from swarmplan.conflicts import validate_plan  # noqa: E402
from swarmplan.graph import GridMap  # noqa: E402
from swarmplan.lowlevel.heuristic import HeuristicCache  # noqa: E402
from swarmplan.metrics import (  # noqa: E402
    RunRecord,
    group_by,
    mean_ratio,
    runtime_stats,
    singleton_lower_bound,
    success_rate,
)
from swarmplan.planners import label_for, solve  # noqa: E402
from swarmplan.scenarios import load_scen, make_instance  # noqa: E402
from swarmplan.solution import SOLVED  # noqa: E402

DEFAULT_CONFIG = ROOT / "config" / "benchmark_default.json"
QUICK_CONFIG = ROOT / "config" / "benchmark_quick.json"


def load_config(path: Path) -> dict:
    """Read a sweep configuration."""
    with path.open() as fh:
        return json.load(fh)


def run_sweep(config: dict, data_dir: Path, out_dir: Path, verbose: bool = True) -> List[RunRecord]:
    """Execute every (map, scenario, agent count, algorithm) run in the config."""
    timeout = float(config.get("timeout", 30.0))
    scen_kind = config.get("scenario_kind", "random")
    validate = bool(config.get("validate", True))
    records: List[RunRecord] = []

    for sweep in config["sweeps"]:
        map_name = sweep["map"]
        map_path = datasets.map_path(map_name, data_dir)
        if not map_path.exists():
            print(f"skipping {map_name}: {map_path} not found", file=sys.stderr)
            continue
        grid = GridMap.from_file(map_path)
        algorithms = sweep.get("algorithms", config.get("algorithms", []))
        scen_ids = sweep.get("scenarios", config.get("scenarios", [1]))
        agent_counts = sweep["agents"]
        dead: Dict[str, bool] = {a: False for a in algorithms}

        for n_agents in agent_counts:
            for spec in algorithms:
                if dead[spec]:
                    records.append(
                        RunRecord(
                            map_name=map_name,
                            scenario="-",
                            n_agents=n_agents,
                            algorithm=label_for(spec),
                            status="skipped",
                            runtime=0.0,
                        )
                    )
                    continue
                solved_here = 0
                attempted = 0
                for scen_id in scen_ids:
                    scen_path = datasets.scen_path(map_name, scen_id, scen_kind, data_dir)
                    if not scen_path.exists():
                        continue
                    scen = load_scen(scen_path)
                    if len(scen) < n_agents:
                        continue
                    instance = make_instance(grid, scen, n_agents)
                    cache = HeuristicCache(grid.graph)
                    started = time.perf_counter()
                    lb = singleton_lower_bound(grid.graph, instance.starts, instance.goals, cache)
                    result = solve(
                        spec,
                        grid.graph,
                        instance.starts,
                        instance.goals,
                        time_limit=timeout,
                        cache=cache,
                    )
                    elapsed = time.perf_counter() - started
                    attempted += 1
                    status = result.status
                    if result.solved and validate:
                        problems = validate_plan(
                            result.paths, instance.starts, instance.goals, grid.graph
                        )
                        if problems:
                            status = "invalid"
                            print(
                                f"  INVALID PLAN {map_name} {spec} {n_agents} agents: "
                                f"{problems[:2]}",
                                file=sys.stderr,
                            )
                    if status == SOLVED:
                        solved_here += 1
                    records.append(
                        RunRecord(
                            map_name=map_name,
                            scenario=f"{scen_kind}-{scen_id}",
                            n_agents=n_agents,
                            algorithm=result.algorithm,
                            status=status,
                            runtime=elapsed,
                            cost=result.cost,
                            makespan=result.makespan,
                            lower_bound=lb,
                            octile_lower_bound=instance.octile_lower_bound or -1.0,
                            high_level_expanded=result.high_level_expanded,
                            low_level_expanded=result.low_level_expanded,
                            suboptimality_bound=result.suboptimality_bound,
                        )
                    )
                if verbose:
                    print(
                        f"{map_name:24s} {n_agents:4d} agents  {label_for(spec):18s} "
                        f"{solved_here}/{attempted} solved",
                        flush=True,
                    )
                if attempted and solved_here == 0:
                    dead[spec] = True
    return records


def write_csv(records: List[RunRecord], path: Path) -> None:
    """Write one row per run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.as_row() for r in records]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[RunRecord]:
    """Read records back, so figures can be re-rendered without re-running."""
    out: List[RunRecord] = []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            out.append(
                RunRecord(
                    map_name=row["map"],
                    scenario=row["scenario"],
                    n_agents=int(row["agents"]),
                    algorithm=row["algorithm"],
                    status=row["status"],
                    runtime=float(row["runtime_s"]),
                    cost=int(row["sum_of_costs"]),
                    makespan=int(row["makespan"]),
                    lower_bound=int(row["lower_bound"]),
                    octile_lower_bound=float(row["octile_lower_bound"]),
                    high_level_expanded=int(row["high_level_expanded"]),
                    low_level_expanded=int(row["low_level_expanded"]),
                    suboptimality_bound=float(row["w"]),
                )
            )
    return out


def markdown_tables(records: List[RunRecord], timeout: float) -> str:
    """Render the tables that go into the README."""
    live = [r for r in records if r.status != "skipped"]
    lines: List[str] = []

    lines.append("## Success rate by map and agent count\n")
    for (map_name,), rows in group_by(live, "map_name").items():
        counts = sorted({r.n_agents for r in rows})
        algs = sorted({r.algorithm for r in rows})
        lines.append(f"\n### {map_name}\n")
        lines.append("| algorithm | " + " | ".join(str(c) for c in counts) + " |")
        lines.append("|---" * (len(counts) + 1) + "|")
        for alg in algs:
            cells = []
            for c in counts:
                sub = [r for r in rows if r.algorithm == alg and r.n_agents == c]
                cells.append("-" if not sub else f"{100 * success_rate(sub):.0f}%")
            lines.append(f"| {alg} | " + " | ".join(cells) + " |")

    lines.append("\n## Runtime and solution quality (solved runs only)\n")
    lines.append(
        "| map | algorithm | solved | median s | p90 s | max s | mean cost / lower bound |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for (map_name, alg), rows in group_by(live, "map_name", "algorithm").items():
        stats = runtime_stats(rows)
        ratio = mean_ratio(rows)
        n_solved = sum(1 for r in rows if r.solved)
        if not n_solved:
            continue
        lines.append(
            f"| {map_name} | {alg} | {n_solved}/{len(rows)} | {stats['median']:.2f} | "
            f"{stats['p90']:.2f} | {stats['max']:.2f} | "
            + (f"{ratio:.3f} |" if ratio else "- |")
        )
    lines.append(f"\nTime budget per instance: {timeout:.0f} s.\n")
    return "\n".join(lines)


def make_figures(records: List[RunRecord], out_dir: Path) -> List[Path]:
    """Success-rate and runtime figures, one per map plus one overall."""
    from swarmplan import viz

    if not viz.HAVE_MATPLOTLIB:
        print("matplotlib not installed: skipping figures", file=sys.stderr)
        return []
    live = [r for r in records if r.status != "skipped"]
    made: List[Path] = []
    for (map_name,), rows in group_by(live, "map_name").items():
        if len({r.n_agents for r in rows}) < 2:
            continue
        made.append(
            viz.plot_success_rate(
                rows,
                out_dir / f"success_{map_name}.png",
                title=f"success rate vs agents -- {map_name}",
            )
        )
    made.append(
        viz.plot_runtime_distribution(
            live, out_dir / "runtime_distribution.png", title="runtime per solved instance"
        )
    )
    return made


def main(argv=None) -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=None)
    p.add_argument("--quick", action="store_true", help="use the small smoke-test sweep")
    p.add_argument("--out", default=str(ROOT / "benchmarks" / "output"))
    p.add_argument("--data-dir", default=None)
    p.add_argument("--figures-only", action="store_true", help="re-render from results.csv")
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    default_config = QUICK_CONFIG if args.quick else DEFAULT_CONFIG
    config_path = Path(args.config) if args.config else default_config
    config = load_config(config_path)
    csv_path = out_dir / "results.csv"

    if args.figures_only:
        if not csv_path.exists():
            print(f"{csv_path} not found; run the sweep first", file=sys.stderr)
            return 2
        records = read_csv(csv_path)
    else:
        data_dir = Path(args.data_dir) if args.data_dir else datasets.default_data_dir()
        if not datasets.available(data_dir):
            print(
                "benchmark data not found. Run: python3 tools/fetch_benchmarks.py --all",
                file=sys.stderr,
            )
            return 2
        started = time.perf_counter()
        records = run_sweep(config, data_dir, out_dir)
        if not records:
            print("no runs executed", file=sys.stderr)
            return 1
        write_csv(records, csv_path)
        print(f"\n{len(records)} runs in {time.perf_counter() - started:.0f}s -> {csv_path}")

    tables = markdown_tables(records, float(config.get("timeout", 30.0)))
    (out_dir / "tables.md").write_text(tables)
    print(f"tables -> {out_dir / 'tables.md'}")
    for path in make_figures(records, out_dir):
        print(f"figure -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
