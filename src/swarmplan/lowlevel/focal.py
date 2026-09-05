"""Focal search: bounded-suboptimal single-agent search, the low level of ECBS.

A* expands the node with the lowest ``f``. Focal search keeps a second list --
FOCAL -- holding every open node whose ``f`` is within a factor ``w`` of the
current minimum ``f``, and expands the *best* of those by a different, cheating
criterion. Any solution it returns still costs at most ``w`` times the optimum,
because every node it ever expands satisfies ``f <= w * f_min`` and ``f_min`` is
a lower bound on the optimum.

Here the cheating criterion is the number of conflicts the partial path already
has with the other agents' current paths. So within its cost budget the search
actively prefers a route that nobody else is using. That is what makes ECBS
fast: the low level hands the high level paths that are already nearly
compatible, so the constraint tree stays small.

The open list is bucketed by integer ``f`` rather than kept as a heap. Costs are
integers and, with a consistent heuristic, ``f_min`` never decreases, so a
bucketed list gives O(1) access to ``f_min`` and makes the "move everything that
now qualifies into FOCAL" step a slice rather than a scan.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..constraints import ConstraintTable
from ..graph import SearchGraph
from .astar import LowLevelResult, _reconstruct, default_horizon
from .heuristic import UNREACHABLE


class ConflictAvoidanceTable:
    """Where the other agents are, so the low level can prefer to avoid them.

    This is a soft constraint: entries here never make a move illegal, they only
    make it less attractive. Hard constraints live in
    :class:`~swarmplan.constraints.ConstraintTable`.
    """

    __slots__ = ("vertex", "edge", "parked", "horizon", "n_paths")

    def __init__(self, paths: Sequence[Optional[Sequence[int]]] = (), skip: int = -1) -> None:
        self.vertex: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.edge: Dict[int, Dict[Tuple[int, int], int]] = defaultdict(lambda: defaultdict(int))
        self.parked: Dict[int, int] = {}
        self.horizon = 0
        self.n_paths = 0
        for i, p in enumerate(paths):
            if p is None or i == skip:
                continue
            self.add(p)

    def add(self, path: Sequence[int]) -> None:
        """Register one agent's path."""
        self.n_paths += 1
        for t, loc in enumerate(path):
            self.vertex[t][loc] += 1
            if t > 0:
                self.edge[t][(path[t - 1], loc)] += 1
        self.horizon = max(self.horizon, len(path) - 1)
        end = path[-1]
        last = len(path) - 1
        if end not in self.parked or self.parked[end] > last:
            self.parked[end] = last

    def count(self, loc: int, t: int) -> int:
        """How many other agents occupy ``loc`` at timestep ``t``."""
        n = self.vertex.get(t, {}).get(loc, 0) if t in self.vertex else 0
        parked = self.parked.get(loc)
        if parked is not None and t > parked:
            n += 1
        return n

    def count_move(self, frm: int, to: int, t: int) -> int:
        """How many other agents make the opposing move (a head-on swap)."""
        if frm == to:
            return 0
        e = self.edge.get(t)
        if not e:
            return 0
        return e.get((to, frm), 0)

    def __bool__(self) -> bool:
        return self.n_paths > 0


def focal_space_time_astar(
    graph: SearchGraph,
    start: int,
    goal: int,
    heuristic: np.ndarray,
    table: Optional[ConstraintTable] = None,
    cat: Optional[ConflictAvoidanceTable] = None,
    w: float = 1.5,
    max_time: Optional[int] = None,
    node_budget: Optional[int] = None,
) -> LowLevelResult:
    """Bounded-suboptimal constrained path, cost at most ``w`` times optimal.

    Returns a :class:`~swarmplan.lowlevel.astar.LowLevelResult` whose
    ``lower_bound`` is the ``f_min`` at the moment the goal was reached. ECBS
    sums those lower bounds across agents to get an admissible bound on the
    sum-of-costs of the node, which is what its own focal list is bounded
    against.
    """
    if w < 1.0:
        raise ValueError("suboptimality factor w must be >= 1")
    if table is None:
        table = ConstraintTable(-1)
        table.set_goal(goal)
    if cat is None:
        cat = ConflictAvoidanceTable()
    if max_time is None:
        max_time = default_horizon(graph, table)

    h0 = int(heuristic[start])
    if h0 >= UNREACHABLE or table.blocked(start, 0):
        return LowLevelResult(None)

    nbrs = graph.neighbours
    release = table.goal_release

    buckets: Dict[int, List[Tuple[int, int, int, int]]] = defaultdict(list)
    alive: Dict[int, int] = defaultdict(int)
    focal: List[Tuple[int, int, int, int, int]] = []
    parents: Dict[Tuple[int, int], Tuple[int, int]] = {}
    best_g: Dict[Tuple[int, int], int] = {(start, 0): 0}
    closed = set()
    counter = 0
    expanded = 0
    generated = 1

    buckets[h0].append((0, counter, start, 0))
    alive[h0] += 1
    f_min = h0
    moved = h0 - 1
    max_f = h0

    while True:
        while alive[f_min] == 0 and f_min <= max_f:
            f_min += 1
        if f_min > max_f:
            return LowLevelResult(None, expanded=expanded, generated=generated)
        limit = int(np.floor(w * f_min))
        while moved < limit:
            moved += 1
            for conf, cnt, loc, t in buckets.pop(moved, ()):
                heapq.heappush(focal, (conf, moved, cnt, loc, t))
        if not focal:
            # Unreachable while the invariant holds (a live node at f_min has
            # always been moved into FOCAL, because moved >= floor(w*f_min) >=
            # f_min). Kept as a hard stop rather than a spin.
            return LowLevelResult(None, expanded=expanded, generated=generated)

        conf, f, _, loc, t = heapq.heappop(focal)
        alive[f] -= 1
        state = (loc, t)
        if state in closed or best_g.get(state, -1) != t:
            continue
        closed.add(state)
        expanded += 1
        if node_budget is not None and expanded > node_budget:
            return LowLevelResult(None, expanded=expanded, generated=generated)

        if loc == goal and t >= release:
            return LowLevelResult(
                path=_reconstruct(parents, state),
                cost=t,
                lower_bound=f_min,
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
            nf = nt + hn
            nconf = conf + cat.count(nxt, nt) + cat.count_move(loc, nxt, nt)
            max_f = max(max_f, nf)
            alive[nf] += 1
            if nf <= moved:
                heapq.heappush(focal, (nconf, nf, counter, nxt, nt))
            else:
                buckets[nf].append((nconf, counter, nxt, nt))
