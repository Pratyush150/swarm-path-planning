"""Unlabelled (anonymous) MAPF: which drone should fly to which slot.

A drone light show is not a labelled MAPF instance. When the swarm morphs from a
ring into a logo, the choreography specifies *the set of occupied positions*,
not which airframe occupies which one. Any drone may take any slot. That freedom
is worth an enormous amount, and throwing it away by assigning slots in index
order is one of the most expensive mistakes available in this problem:

* with an arbitrary assignment, drones cross the formation to reach slots on the
  far side, so the swarm has to route hundreds of long, mutually conflicting
  paths;
* with a minimum-cost assignment, most drones move to a nearby slot, the paths
  are short, and far fewer of them interact at all.

The reduction is exact for the relaxed problem: ignoring inter-agent conflicts,
the cheapest way to cover the goal set is a minimum-cost bipartite matching
between starts and goals with edge weights equal to true obstacle-aware
distances. That is a Hungarian-algorithm problem, and it gives a **lower bound**
on sum-of-costs for the unlabelled instance as well as a good labelling to hand
to CBS or ECBS.

Two objectives are offered, and they are not the same:

``"sum"``
    Minimise total distance. Standard assignment, minimises the sum-of-costs
    lower bound.
``"makespan"``
    Minimise the *largest* distance. A show is not in formation until the last
    drone arrives, so this is usually the one an operator cares about. Solved as
    a bottleneck assignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from ..graph import SearchGraph
from ..lowlevel.heuristic import UNREACHABLE, HeuristicCache
from .hungarian import bottleneck_assignment, hungarian

#: Cost used for a start/goal pair with no path between them. Large enough that
#: the matching avoids it whenever any feasible alternative exists, finite so
#: the dual potentials stay bounded.
UNREACHABLE_PENALTY = 1e9


@dataclass
class AssignmentResult:
    """Outcome of an anonymous-MAPF goal assignment."""

    order: List[int]
    total_distance: float
    max_distance: float
    objective: str
    matrix_time: float = 0.0
    solve_time: float = 0.0

    @property
    def n_agents(self) -> int:
        """Number of agents assigned."""
        return len(self.order)

    def apply(self, goals: Sequence[int]) -> List[int]:
        """Reorder ``goals`` so that agent *i* is assigned ``result[i]``."""
        return [goals[j] for j in self.order]


def distance_matrix(
    graph: SearchGraph,
    starts: Sequence[int],
    goals: Sequence[int],
    cache: Optional[HeuristicCache] = None,
) -> np.ndarray:
    """``(len(starts), len(goals))`` matrix of true obstacle-aware distances.

    One backward Dijkstra per goal fills a whole column, so the matrix costs
    ``len(goals)`` sweeps regardless of the number of agents. The same cache is
    then reused as the A* heuristic for whichever assignment wins, so none of
    that work is thrown away.
    """
    cache = cache or HeuristicCache(graph)
    starts_arr = np.asarray(starts, dtype=np.int64)
    out = np.empty((len(starts), len(goals)), dtype=np.float64)
    for j, g in enumerate(goals):
        col = cache.get(g)[starts_arr].astype(np.float64)
        col[col >= UNREACHABLE] = UNREACHABLE_PENALTY
        out[:, j] = col
    return out


def assign_goals(
    graph: SearchGraph,
    starts: Sequence[int],
    goals: Sequence[int],
    objective: str = "sum",
    cache: Optional[HeuristicCache] = None,
    use_scipy: bool = False,
) -> AssignmentResult:
    """Assign each agent a goal, minimising total or maximum travel distance."""
    import time

    if len(starts) != len(goals):
        raise ValueError("anonymous MAPF needs as many goals as agents")
    if objective not in ("sum", "makespan"):
        raise ValueError("objective must be 'sum' or 'makespan'")
    t0 = time.perf_counter()
    cost = distance_matrix(graph, starts, goals, cache)
    t1 = time.perf_counter()
    if objective == "sum":
        if use_scipy:
            from .hungarian import solve_assignment

            rows, cols = solve_assignment(cost, use_scipy=True)
        else:
            rows, cols = hungarian(cost)
    else:
        rows, cols, _ = bottleneck_assignment(cost)
    t2 = time.perf_counter()
    order = [0] * len(starts)
    for r, c in zip(rows, cols):
        order[int(r)] = int(c)
    chosen = cost[np.arange(len(starts)), order]
    return AssignmentResult(
        order=order,
        total_distance=float(chosen.sum()),
        max_distance=float(chosen.max()) if len(chosen) else 0.0,
        objective=objective,
        matrix_time=t1 - t0,
        solve_time=t2 - t1,
    )


def identity_assignment(
    graph: SearchGraph,
    starts: Sequence[int],
    goals: Sequence[int],
    cache: Optional[HeuristicCache] = None,
) -> AssignmentResult:
    """The do-nothing baseline: agent *i* takes goal *i*.

    This is what "we just gave each drone a slot" costs, and it is the number
    the optimal assignment is compared against in the README figure.
    """
    if len(starts) != len(goals):
        raise ValueError("anonymous MAPF needs as many goals as agents")
    cost = distance_matrix(graph, starts, goals, cache)
    order = list(range(len(starts)))
    chosen = cost[np.arange(len(starts)), order]
    return AssignmentResult(
        order=order,
        total_distance=float(chosen.sum()),
        max_distance=float(chosen.max()) if len(chosen) else 0.0,
        objective="identity",
    )


def assignment_lower_bound(result: AssignmentResult) -> float:
    """Sum-of-costs lower bound implied by an assignment.

    Each agent must travel at least its assigned distance, so the total is a
    lower bound on the sum-of-costs of any conflict-free plan realising that
    assignment. For the minimum-sum assignment it is also a lower bound for the
    *unlabelled* problem as a whole.
    """
    return result.total_distance
