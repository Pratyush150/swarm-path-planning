"""Multi-value decision diagrams: every optimal path of a given cost, at once.

An MDD for agent *i* at cost *c* is the layered graph of exactly those states
``(vertex, t)`` that lie on at least one constraint-respecting path from the
start to the goal of length exactly *c*. It is built with one forward sweep
(reachable in *t* steps, and close enough to the goal to still finish in
``c - t``) and one backward sweep (can actually reach the goal from here in the
time remaining).

MDDs are what turn CBS from "resolve conflicts in the order they happen" into
something that reasons about *cost*:

* if the MDD has **width 1** at timestep ``t``, agent *i* is on that cell in
  every optimal path it has, so forbidding it there necessarily makes the agent
  more expensive. A conflict where that holds for both agents is **cardinal**:
  both children of the split cost strictly more than the parent, which is
  exactly the information the CBS heuristic needs.
* the joint search over two MDDs answers "can these two agents both stay
  optimal simultaneously?" -- the dependency test behind the DG heuristic.

Building an MDD is not free (it is another BFS over the time-expanded graph),
which is why prioritising conflicts and the CBS heuristics are switchable: on
easy instances they cost more than they save, and the benchmark tables in the
README show where the crossover is.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..constraints import ConstraintTable
from ..graph import SearchGraph
from ..lowlevel.heuristic import UNREACHABLE


class MDD:
    """Layered set of states on optimal-cost paths, plus their edges."""

    __slots__ = ("levels", "edges", "cost", "goal", "start")

    def __init__(
        self,
        levels: List[Set[int]],
        edges: List[Dict[int, Set[int]]],
        cost: int,
        start: int,
        goal: int,
    ) -> None:
        self.levels = levels
        self.edges = edges
        self.cost = cost
        self.start = start
        self.goal = goal

    def width(self, t: int) -> int:
        """Number of distinct cells the agent could be on at timestep ``t``.

        Past the end of the MDD the agent is parked on its goal, so the width is
        1 -- which is why a conflict with a parked agent is always cardinal for
        the parked one.
        """
        if t < 0:
            raise ValueError("timestep must be non-negative")
        if t > self.cost:
            return 1
        return len(self.levels[t])

    def at(self, t: int) -> Set[int]:
        """The set of cells reachable at timestep ``t`` on an optimal path."""
        if t > self.cost:
            return {self.goal}
        return self.levels[t]

    def singleton(self, t: int) -> Optional[int]:
        """The forced cell at ``t`` if the width is 1, else ``None``."""
        s = self.at(t)
        return next(iter(s)) if len(s) == 1 else None

    def forced_edge(self, t: int) -> Optional[Tuple[int, int]]:
        """The forced move arriving at ``t`` if there is exactly one, else ``None``."""
        if t <= 0:
            return None
        a = self.singleton(t - 1)
        b = self.singleton(t)
        return None if a is None or b is None else (a, b)

    def successors(self, loc: int, t: int) -> Set[int]:
        """Cells the agent can be on at ``t+1`` given it is on ``loc`` at ``t``."""
        if t >= self.cost:
            return {self.goal}
        return self.edges[t].get(loc, set())

    def __len__(self) -> int:
        return self.cost + 1

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<MDD cost={self.cost} widths={[len(l) for l in self.levels]}>"


def build_mdd(
    graph: SearchGraph,
    start: int,
    goal: int,
    cost: int,
    heuristic: np.ndarray,
    table: Optional[ConstraintTable] = None,
) -> Optional[MDD]:
    """Build the MDD of all constraint-respecting paths of length exactly ``cost``.

    Returns ``None`` if no such path exists (which, called with the cost the
    low-level search just returned, should not happen and is treated as a bug
    signal by the callers).
    """
    if cost < 0:
        raise ValueError("cost must be non-negative")
    nbrs = graph.neighbours
    forward: List[Set[int]] = [set() for _ in range(cost + 1)]
    if table is not None and table.blocked(start, 0):
        return None
    forward[0].add(start)
    for t in range(cost):
        nt = t + 1
        layer = forward[nt]
        for loc in forward[t]:
            for nxt in nbrs[loc] + (loc,):
                h = int(heuristic[nxt])
                if h >= UNREACHABLE or nt + h > cost:
                    continue
                if table is not None:
                    if table.blocked(nxt, nt):
                        continue
                    if nxt != loc and table.blocked_move(loc, nxt, nt):
                        continue
                layer.add(nxt)
        if not layer:
            return None
    if goal not in forward[cost]:
        return None

    # Backward pass: keep only states from which the goal is still reachable
    # along an edge that survives the forward pass.
    levels: List[Set[int]] = [set() for _ in range(cost + 1)]
    edges: List[Dict[int, Set[int]]] = [dict() for _ in range(cost)]
    levels[cost].add(goal)
    for t in range(cost - 1, -1, -1):
        nxt_level = levels[t + 1]
        layer = levels[t]
        edge_layer = edges[t]
        for loc in forward[t]:
            succ = set()
            for cand in nbrs[loc] + (loc,):
                if cand not in nxt_level:
                    continue
                if table is not None:
                    if table.blocked(cand, t + 1):
                        continue
                    if cand != loc and table.blocked_move(loc, cand, t + 1):
                        continue
                succ.add(cand)
            if succ:
                layer.add(loc)
                edge_layer[loc] = succ
        if not layer:
            return None
    if start not in levels[0]:
        return None
    return MDD(levels, edges, cost, start, goal)


def mdds_dependent(m1: MDD, m2: MDD) -> bool:
    """True if two agents cannot both stay at their own optimal cost.

    Searches the cross product of the two MDDs for a joint path with no vertex
    or edge conflict. If none exists, at least one of the two has to get more
    expensive, so the pair contributes 1 to the DG heuristic. This is a
    *pairwise* statement -- it says nothing about what a third agent forces --
    which is why the heuristic is a vertex cover of these pairs and not their
    count.
    """
    horizon = max(m1.cost, m2.cost)
    start = (m1.start, m2.start)
    if start[0] == start[1]:
        return True
    frontier = {start}
    for t in range(horizon):
        nxt: Set[Tuple[int, int]] = set()
        for u, v in frontier:
            for nu in m1.successors(u, t):
                for nv in m2.successors(v, t):
                    if nu == nv:
                        continue
                    if nu == v and nv == u:
                        continue
                    nxt.add((nu, nv))
        if not nxt:
            return True
        frontier = nxt
    return (m1.goal, m2.goal) not in frontier


def joint_mdd_size(m1: MDD, m2: MDD) -> int:
    """Number of joint states explored by :func:`mdds_dependent`.

    Exposed for the ablation table: this is the cost that buys the DG heuristic,
    and on wide MDDs it is quadratic in the width.
    """
    horizon = max(m1.cost, m2.cost)
    frontier = {(m1.start, m2.start)}
    total = 1
    for t in range(horizon):
        nxt: Set[Tuple[int, int]] = set()
        for u, v in frontier:
            for nu in m1.successors(u, t):
                for nv in m2.successors(v, t):
                    if nu != nv and not (nu == v and nv == u):
                        nxt.add((nu, nv))
        total += len(nxt)
        frontier = nxt
        if not frontier:
            break
    return total
