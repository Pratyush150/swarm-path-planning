#!/usr/bin/env python3
"""Measure how much the true-distance heuristic beats Manhattan distance.

The claim in :mod:`swarmplan.lowlevel.heuristic` is that Manhattan distance,
though admissible, is so weak on maps with obstacles that an A* guided by it
degenerates towards breadth-first search. This script measures it on the real
benchmark maps, using the real scenario start/goal pairs, and prints the table
quoted in the README.

Usage::

    python3 tools/heuristic_report.py
    python3 tools/heuristic_report.py --maps maze-32-32-2 room-32-32-4 --pairs 400
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swarmplan import datasets  # noqa: E402
from swarmplan.graph import GridMap  # noqa: E402
from swarmplan.lowlevel.astar import space_time_astar  # noqa: E402
from swarmplan.lowlevel.heuristic import UNREACHABLE, HeuristicCache  # noqa: E402
from swarmplan.scenarios import load_scen  # noqa: E402

import numpy as np  # noqa: E402


def report(map_names, pairs: int, data_dir: Path) -> int:
    """Print the ratio table and the A* expansion comparison."""
    print(
        f"{'map':24s} {'pairs':>6s} {'mean true/manhattan':>20s} {'p95':>7s} {'max':>7s} "
        f"{'A* nodes (true)':>16s} {'A* nodes (manhattan)':>21s}"
    )
    status = 0
    for name in map_names:
        map_path = datasets.map_path(name, data_dir)
        scen_path = datasets.scen_path(name, 1, "random", data_dir)
        if not map_path.exists() or not scen_path.exists():
            print(f"{name:24s} missing benchmark data")
            status = 1
            continue
        grid = GridMap.from_file(map_path)
        scen = load_scen(scen_path)
        entries = scen.entries[:pairs]
        cache = HeuristicCache(grid.graph)
        coords = grid.graph.coord_array()

        ratios = []
        exp_true = exp_manhattan = 0
        for e in entries:
            s = grid.node(e.start)
            g = grid.node(e.goal)
            true_table = cache.get(g)
            d = int(true_table[s])
            if d >= UNREACHABLE or d == 0:
                continue
            gr, gc = grid.rc(g)
            manhattan = abs(coords[s][0] - gr) + abs(coords[s][1] - gc)
            if manhattan > 0:
                ratios.append(d / manhattan)
            exp_true += space_time_astar(grid.graph, s, g, true_table).expanded
            man_table = np.abs(coords[:, 0] - gr) + np.abs(coords[:, 1] - gc)
            exp_manhattan += space_time_astar(grid.graph, s, g, man_table).expanded
        if not ratios:
            continue
        ratios.sort()
        p95 = ratios[min(len(ratios) - 1, int(0.95 * (len(ratios) - 1)))]
        print(
            f"{name:24s} {len(ratios):6d} {statistics.fmean(ratios):20.2f} {p95:7.2f} "
            f"{ratios[-1]:7.2f} {exp_true:16d} {exp_manhattan:21d}"
        )
    return status


def main(argv=None) -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--maps", nargs="*", default=[m.name for m in datasets.MAPS[:6]])
    p.add_argument("--pairs", type=int, default=200)
    p.add_argument("--data-dir", default=None)
    args = p.parse_args(argv)
    data_dir = Path(args.data_dir) if args.data_dir else datasets.default_data_dir()
    return report(args.maps, args.pairs, data_dir)


if __name__ == "__main__":
    raise SystemExit(main())
