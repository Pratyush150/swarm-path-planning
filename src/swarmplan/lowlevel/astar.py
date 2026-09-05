"""Space-time A*: optimal single-agent search in the time-expanded graph.

The state is ``(vertex, timestep)``, not ``vertex``. That is the whole point: an
agent that is forbidden from a cell at t=7 may pass through it at t=9, so the
search cannot collapse the time dimension. The available actions from
``(v, t)`` are the graph moves ``v -> u`` and the **wait** action ``v -> v``,
each costing one timestep.

Two details are easy to get wrong and are handled explicitly here.

**Waiting is an action with a cost.** Under sum-of-costs, an agent that waits
three steps and then walks pays for those three steps. Only waiting *at the goal
after final arrival* is free, and that is modelled by ending the path at arrival
and treating the agent as parked from then on.

**An agent cannot always stop when it first reaches its goal.** If a constraint
forbids the goal cell at t=12, an agent that arrives at t=9 has to leave and come
back. The search therefore only accepts a goal state at or after
``table.goal_release``, which :class:`~swarmplan.constraints.ConstraintTable`
computes from the constraints that touch the goal.

With a consistent heuristic (which the backward-Dijkstra table is) the first
expansion of a state is optimal, so a closed set is sound and the returned path
is a minimum-cost path subject to the constraints.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..constraints import ConstraintTable
from ..graph import SearchGraph
from .heuristic import UNREACHABLE


@dataclass
class LowLevelResult:
    """Outcome of one low-level search.

    ``cost`` is the arrival timestep, which is the agent's contribution to
    sum-of-costs. ``lower_bound`` equals ``cost`` for the optimal search and is
    smaller for the focal (bounded-suboptimal) variant.
    """

    path: Optional[List[int]]
    cost: int = -1
    lower_bound: int = -1
    expanded: int = 0
    generated: int = 0

    @property
    def found(self) -> bool:
        """True if a path satisfying every constraint was found."""
        return self.path is not None

    def __len__(self) -> int:
        return 0 if self.path is None else len(self.path)


def default_horizon(graph: SearchGraph, table: ConstraintTable) -> int:
    """A safe upper bound on the length of an optimal constrained path.

    Any optimal path visits at most one state per (vertex, timestep) below the
    last constrained timestep, and after the last constraint it never has a
    reason to wait, so ``last_constraint + n_vertices`` bounds it. The bound is
    only used to guarantee termination on unsolvable instances; searches that
    succeed stop far earlier.
    """
    return int(table.max_time + graph.n_nodes + 1)


def space_time_astar(
    graph: SearchGraph,
    start: int,
    goal: int,
    heuristic: np.ndarray,
    table: Optional[ConstraintTable] = None,
    max_time: Optional[int] = None,
    node_budget: Optional[int] = None,
) -> LowLevelResult:
    """Minimum-cost constrained path from ``start`` to ``goal``.

    Parameters
    ----------
    graph:
        The movement graph.
    start, goal:
        Node ids.
    heuristic:
        Admissible, consistent distance-to-goal table, normally from
        :class:`~swarmplan.lowlevel.heuristic.HeuristicCache`.
    table:
        Constraints to respect. ``None`` means an unconstrained search.
    max_time:
        Horizon. Defaults to :func:`default_horizon`.
    node_budget:
        Optional cap on expansions, so a caller with a deadline can bail out
        rather than block. Returns a not-found result when hit.

    Returns
    -------
    LowLevelResult
        ``path`` is the sequence of vertices at timesteps ``0 .. cost``, or
        ``None`` if no constrained path exists (or the budget was exhausted).
    """
    if table is None:
        table = ConstraintTable(-1)
        table.set_goal(goal)
    if max_time is None:
        max_time = default_horizon(graph, table)

    h0 = int(heuristic[start])
    if h0 >= UNREACHABLE:
        return LowLevelResult(None)
    if table.blocked(start, 0):
        return LowLevelResult(None)

    nbrs = graph.neighbours
    release = table.goal_release

    counter = 0
    open_list: List[Tuple[int, int, int, int, int]] = [(h0, h0, counter, start, 0)]
    parents: Dict[Tuple[int, int], Tuple[int, int]] = {}
    best_g: Dict[Tuple[int, int], int] = {(start, 0): 0}
    closed = set()
    expanded = 0
    generated = 1

    while open_list:
        f, h, _, loc, t = heapq.heappop(open_list)
        state = (loc, t)
        if state in closed:
            continue
        closed.add(state)
        expanded += 1
        if node_budget is not None and expanded > node_budget:
            return LowLevelResult(None, expanded=expanded, generated=generated)

        if loc == goal and t >= release:
            path = reconstruct_path(parents, state)
            return LowLevelResult(
                path=path,
                cost=t,
                lower_bound=t,
                expanded=expanded,
                generated=generated,
            )

        nt = t + 1
        if nt > max_time:
            continue
        for nxt in nbrs[loc] + (loc,):
            hn = int(heuristic[nxt])
            if hn >= UNREACHABLE or nt + hn > max_time:
                continue
            if table.blocked(nxt, nt):
                continue
            if nxt != loc and table.blocked_move(loc, nxt, nt):
                continue
            ns = (nxt, nt)
            if ns in closed:
                continue
            prior = best_g.get(ns)
            if prior is not None and prior <= nt:
                continue
            best_g[ns] = nt
            parents[ns] = state
            counter += 1
            generated += 1
            heapq.heappush(open_list, (nt + hn, hn, counter, nxt, nt))

    return LowLevelResult(None, expanded=expanded, generated=generated)


def reconstruct_path(
    parents: Dict[Tuple[int, int], Tuple[int, int]], state: Tuple[int, int]
) -> List[int]:
    """Walk the parent pointers back to the start, returning vertices in time order."""
    out = []
    cur: Optional[Tuple[int, int]] = state
    while cur is not None:
        out.append(cur[0])
        cur = parents.get(cur)
    out.reverse()
    return out


def path_cost(path: Optional[List[int]]) -> int:
    """Sum-of-costs contribution of one path: its arrival timestep."""
    if not path:
        return 0
    return len(path) - 1


def location_at(path: List[int], t: int) -> int:
    """Where an agent is at timestep ``t``, parking on its goal after arrival."""
    if t < 0:
        raise ValueError("timestep must be non-negative")
    return path[t] if t < len(path) else path[-1]
