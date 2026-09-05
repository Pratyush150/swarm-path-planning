"""The true-distance heuristic: backward Dijkstra from each goal.

Why not Manhattan distance
--------------------------
Manhattan distance is admissible on a 4-connected grid, so it is *correct*. It
is also, on any map with real obstacles, badly uninformed, and in MAPF that
matters far more than usual because the low-level search is re-run thousands of
times inside the constraint tree.

On ``maze-32-32-2`` -- a benchmark maze with one-cell corridors -- two cells a
few columns apart can be tens of steps apart through the maze. Measured over 150
benchmark start/goal pairs on that map (``tools/heuristic_report.py``), the true
distance averages 2.87x the Manhattan estimate and reaches 17.3x, and the
consequence for the search is not subtle: A* guided by the true distance expands
**8,508** states over those 150 queries, and the same A* guided by Manhattan
distance expands **893,969**. Two orders of magnitude, on the same queries, for
the same answers.

The fix is standard and cheap: for each goal, run one backward Dijkstra over the
static graph and store the exact distance from every cell to that goal. It costs
one O(V log V) sweep per goal, it is reused by every low-level call for that
agent for the whole run, and it is *perfect* on the obstacle-only problem -- the
only thing it ignores is the other agents, which is exactly the relaxation CBS
is built on.

Both properties the search needs are consequences of it being an exact distance
in the same graph:

* **admissible** -- it never overestimates, because a real path of that length
  exists and no shorter one does;
* **consistent** -- ``h(u) <= 1 + h(v)`` for every edge ``u -> v``, because
  distances in a unit-cost graph satisfy the triangle inequality. Consistency is
  what lets the search close a node on first expansion.

Both are asserted directly in the test suite on a map with obstacles.
"""

from __future__ import annotations

import heapq
from collections import deque
from typing import Dict, Optional, Sequence

import numpy as np

from ..graph import SearchGraph

#: Distance value used for "no path exists". Large but still an int, so
#: ``g + h`` never overflows into a float comparison.
UNREACHABLE = np.iinfo(np.int32).max // 4


def backward_dijkstra(
    graph: SearchGraph,
    goal: int,
    weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Exact distance from every node to ``goal``.

    The graph is undirected, so a backward search from the goal and a forward
    search to it are the same sweep; the name follows the MAPF literature, where
    the distinction matters for directed roadmaps.

    ``weights`` (per-node arrival cost) is accepted for graphs where cells cost
    more than one step -- a congestion penalty, or a slow zone over a crop row.
    With the default unit cost the function uses a BFS, which is the same
    algorithm with the priority queue replaced by a FIFO and is several times
    faster.
    """
    n = graph.n_nodes
    if not 0 <= goal < n:
        raise ValueError(f"goal node {goal} out of range for graph with {n} nodes")
    dist = np.full(n, UNREACHABLE, dtype=np.int64)
    nbrs = graph.neighbours

    if weights is None:
        dist[goal] = 0
        queue = deque([goal])
        while queue:
            u = queue.popleft()
            du = dist[u] + 1
            for v in nbrs[u]:
                if dist[v] > du:
                    dist[v] = du
                    queue.append(v)
        return dist

    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n,):
        raise ValueError("weights must have one entry per node")
    fdist = np.full(n, float(UNREACHABLE), dtype=np.float64)
    fdist[goal] = 0.0
    pq = [(0.0, goal)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > fdist[u]:
            continue
        for v in nbrs[u]:
            nd = d + w[u]
            if nd < fdist[v]:
                fdist[v] = nd
                heapq.heappush(pq, (nd, v))
    return fdist


def true_distance(graph: SearchGraph, goal: int) -> np.ndarray:
    """Alias for :func:`backward_dijkstra` with unit costs, read as a heuristic."""
    return backward_dijkstra(graph, goal)


class HeuristicCache:
    """One true-distance table per goal, computed once and shared.

    CBS re-plans a single agent thousands of times; every one of those calls
    uses the same heuristic table, so computing it per call would dominate the
    runtime. The cache is keyed on the goal node, so agents that share a goal
    (common in the light-show demos before assignment) share the sweep.
    """

    def __init__(self, graph: SearchGraph) -> None:
        self.graph = graph
        self._tables: Dict[int, np.ndarray] = {}
        self.sweeps = 0

    def get(self, goal: int) -> np.ndarray:
        """Distance-to-``goal`` table, computing it on first request."""
        table = self._tables.get(goal)
        if table is None:
            table = backward_dijkstra(self.graph, goal)
            self._tables[goal] = table
            self.sweeps += 1
        return table

    def distance(self, start: int, goal: int) -> int:
        """Exact obstacle-aware distance between two nodes."""
        return int(self.get(goal)[start])

    def reachable(self, start: int, goal: int) -> bool:
        """True if a path exists between the two nodes."""
        return self.distance(start, goal) < UNREACHABLE

    def __len__(self) -> int:
        return len(self._tables)
