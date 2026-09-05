"""Metrics: the numbers a MAPF result is judged on, and their lower bounds.

Sum-of-costs and makespan are the two objectives the literature reports.
They are not interchangeable and optimising one can badly hurt the other -- a
plan that lets one agent take a long detour so the other 99 arrive quickly is
good for makespan and bad for sum-of-costs. Everything in this package
optimises **sum-of-costs**, which is the standard MAPF objective, and reports
makespan alongside because that is what a light show actually cares about (the
show is over when the last drone is in place).

The lower bound matters as much as the cost. Two are computed here:

``singleton_lower_bound``
    Sum over agents of the true single-agent distance, ignoring every other
    agent. Provably a lower bound on the optimal sum-of-costs, tight on empty
    maps, loose when the map is congested. This is what the ratio in the
    benchmark tables is measured against, so a ratio of 1.00 means *provably
    optimal* and 1.10 means "at most 10% above optimal, probably less".

``octile_lower_bound``
    The last column of the ``.scen`` file, summed. Reported for cross-reference
    only; it is an 8-connected distance and so is systematically below the
    4-connected one. See :mod:`swarmplan.scenarios`.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from .graph import SearchGraph
from .lowlevel.heuristic import UNREACHABLE, HeuristicCache
from .solution import SOLVED, Solution


def sum_of_costs(paths: Sequence[Sequence[int]]) -> int:
    """Total timesteps spent by all agents before parking on their goals."""
    return sum(len(p) - 1 for p in paths)


def makespan(paths: Sequence[Sequence[int]]) -> int:
    """Timestep at which the last agent arrives."""
    return max((len(p) - 1 for p in paths), default=0)


def singleton_lower_bound(
    graph: SearchGraph,
    starts: Sequence[int],
    goals: Sequence[int],
    cache: Optional[HeuristicCache] = None,
) -> int:
    """Sum of individual optimal distances: an admissible bound on sum-of-costs.

    Each agent must travel at least its own obstacle-aware shortest distance,
    and the other agents can only ever force it to travel further, so the sum is
    a lower bound on the optimal sum-of-costs. Raises if an agent cannot reach
    its goal at all.
    """
    cache = cache or HeuristicCache(graph)
    total = 0
    for s, g in zip(starts, goals):
        d = int(cache.get(g)[s])
        if d >= UNREACHABLE:
            raise ValueError(f"no path exists from {s} to {g} on {graph.name}")
        total += d
    return total


def cost_ratio(solution: Solution, lower_bound: int) -> Optional[float]:
    """Solution cost divided by the lower bound, or ``None`` if unsolved."""
    if not solution.solved or lower_bound <= 0:
        return None
    return solution.cost / lower_bound


@dataclass
class RunRecord:
    """One (instance, algorithm) benchmark measurement, ready to tabulate."""

    map_name: str
    scenario: str
    n_agents: int
    algorithm: str
    status: str
    runtime: float
    cost: int = -1
    makespan: int = -1
    lower_bound: int = -1
    octile_lower_bound: float = -1.0
    high_level_expanded: int = 0
    low_level_expanded: int = 0
    suboptimality_bound: float = 1.0

    @property
    def solved(self) -> bool:
        """True if this run produced a valid plan."""
        return self.status == SOLVED

    @property
    def ratio(self) -> Optional[float]:
        """Cost relative to the sum-of-individual-optima lower bound."""
        if not self.solved or self.lower_bound <= 0:
            return None
        return self.cost / self.lower_bound

    def as_row(self) -> Dict[str, object]:
        """Flat dict for CSV output."""
        return {
            "map": self.map_name,
            "scenario": self.scenario,
            "agents": self.n_agents,
            "algorithm": self.algorithm,
            "status": self.status,
            "runtime_s": round(self.runtime, 4),
            "sum_of_costs": self.cost,
            "makespan": self.makespan,
            "lower_bound": self.lower_bound,
            "octile_lower_bound": round(self.octile_lower_bound, 3),
            "ratio_to_lb": None if self.ratio is None else round(self.ratio, 5),
            "high_level_expanded": self.high_level_expanded,
            "low_level_expanded": self.low_level_expanded,
            "w": self.suboptimality_bound,
        }


def success_rate(records: Iterable[RunRecord]) -> float:
    """Fraction of runs that produced a plan within the time budget."""
    records = list(records)
    if not records:
        return 0.0
    return sum(1 for r in records if r.solved) / len(records)


def runtime_stats(records: Iterable[RunRecord], solved_only: bool = True) -> Dict[str, float]:
    """Median / mean / p90 / max runtime over a set of runs, in seconds.

    Failed runs are excluded by default: counting a run that hit the time budget
    as "took exactly the budget" flatters a slow algorithm, and averages a number
    that describes the budget rather than the algorithm.
    """
    times = [r.runtime for r in records if r.solved or not solved_only]
    if not times:
        nan = float("nan")
        return {"n": 0.0, "median": nan, "mean": nan, "p90": nan, "max": nan}
    times.sort()
    p90 = times[min(len(times) - 1, int(round(0.9 * (len(times) - 1))))]
    return {
        "n": float(len(times)),
        "median": statistics.median(times),
        "mean": statistics.fmean(times),
        "p90": p90,
        "max": times[-1],
    }


def mean_ratio(records: Iterable[RunRecord]) -> Optional[float]:
    """Mean cost/lower-bound ratio over the solved runs."""
    ratios = [r.ratio for r in records if r.ratio is not None]
    if not ratios:
        return None
    return statistics.fmean(ratios)


def group_by(records: Iterable[RunRecord], *keys: str) -> Dict[tuple, List[RunRecord]]:
    """Group records by attribute names, preserving insertion order."""
    out: Dict[tuple, List[RunRecord]] = {}
    for r in records:
        k = tuple(getattr(r, key) for key in keys)
        out.setdefault(k, []).append(r)
    return out
