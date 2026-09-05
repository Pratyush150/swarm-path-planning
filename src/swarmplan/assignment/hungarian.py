"""The Hungarian algorithm, implemented here rather than imported.

Given an ``n x m`` cost matrix (``n <= m``), find the assignment of every row to
a distinct column that minimises the total cost. This is the Kuhn-Munkres
algorithm in its O(n^2 m) shortest-augmenting-path form (Jonker-Volgenant
style): maintain dual potentials ``u`` and ``v`` such that every reduced cost
``c[i][j] - u[i] - v[j]`` is non-negative, and grow an alternating tree along
zero-reduced-cost edges, adjusting the potentials when it gets stuck. The
potentials are the certificate of optimality -- when the last row is matched,
``sum(u) + sum(v)`` equals the assignment cost, and no assignment can be
cheaper.

The inner loop is vectorised over columns with NumPy, which is what makes a
few-hundred-agent light show assignment take milliseconds in Python.

``scipy.optimize.linear_sum_assignment`` does the same job and is faster still.
It is not a dependency: :func:`solve_assignment` uses this implementation by
default, and the test suite asserts that the two agree in *total cost* on random
matrices (the assignments themselves can differ when there are ties, and both
are then equally optimal).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def hungarian(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Minimum-cost assignment of rows to distinct columns.

    Parameters
    ----------
    cost:
        ``(n, m)`` array of finite costs. If ``n > m`` the problem is solved
        transposed, so the *columns* are the ones fully assigned.

    Returns
    -------
    (row_ind, col_ind):
        Index arrays such that ``cost[row_ind, col_ind].sum()`` is minimal, with
        ``row_ind`` sorted ascending. Matches the calling convention of
        ``scipy.optimize.linear_sum_assignment``.
    """
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError("cost matrix must be 2D")
    if cost.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    if not np.isfinite(cost).all():
        raise ValueError(
            "cost matrix contains non-finite entries; substitute a large finite "
            "penalty for forbidden assignments so the dual potentials stay bounded"
        )
    transposed = cost.shape[0] > cost.shape[1]
    if transposed:
        cost = cost.T
    n, m = cost.shape

    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(m + 1, dtype=np.float64)
    match = np.zeros(m + 1, dtype=np.int64)  # column -> row (1-based), 0 = free
    way = np.zeros(m + 1, dtype=np.int64)

    for i in range(1, n + 1):
        match[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf, dtype=np.float64)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = int(match[j0])
            free = ~used[1:]
            cur = cost[i0 - 1] - u[i0] - v[1:]
            better = free & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0
            cand = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(cand)) + 1
            delta = float(cand[j1 - 1])
            if not np.isfinite(delta):
                raise ValueError("assignment infeasible: no finite augmenting path")
            u[match[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if match[j0] == 0:
                break
        while j0:
            j1 = int(way[j0])
            match[j0] = match[j1]
            j0 = j1

    col_for_row = np.zeros(n, dtype=np.int64)
    for j in range(1, m + 1):
        if match[j]:
            col_for_row[int(match[j]) - 1] = j - 1
    rows = np.arange(n, dtype=np.int64)
    if transposed:
        return col_for_row, rows
    return rows, col_for_row


def assignment_cost(cost: np.ndarray, row_ind: np.ndarray, col_ind: np.ndarray) -> float:
    """Total cost of an assignment."""
    return float(np.asarray(cost)[row_ind, col_ind].sum())


def solve_assignment(
    cost: np.ndarray, use_scipy: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Minimum-cost assignment, optionally via SciPy.

    ``use_scipy=True`` asks for ``scipy.optimize.linear_sum_assignment`` and
    falls back to :func:`hungarian` if SciPy is not installed, so the call site
    never has to care.
    """
    if use_scipy:
        try:
            from scipy.optimize import linear_sum_assignment

            r, c = linear_sum_assignment(np.asarray(cost, dtype=np.float64))
            return np.asarray(r), np.asarray(c)
        except ImportError:
            pass
    return hungarian(cost)


def _hopcroft_karp(adj: list, n_left: int, n_right: int) -> int:
    """Maximum bipartite matching size, used by the bottleneck solver."""
    INF = float("inf")
    match_l = [-1] * n_left
    match_r = [-1] * n_right
    result = 0
    while True:
        dist = [INF] * n_left
        queue = []
        for u in range(n_left):
            if match_l[u] == -1:
                dist[u] = 0
                queue.append(u)
        found = False
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                w = match_r[v]
                if w == -1:
                    found = True
                elif dist[w] == INF:
                    dist[w] = dist[u] + 1
                    queue.append(w)
        if not found:
            return result

        def try_augment(u: int) -> bool:
            for v in adj[u]:
                w = match_r[v]
                if w == -1 or (dist[w] == dist[u] + 1 and try_augment(w)):
                    match_l[u] = v
                    match_r[v] = u
                    return True
            dist[u] = INF
            return False

        for u in range(n_left):
            if match_l[u] == -1 and try_augment(u):
                result += 1


def bottleneck_assignment(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Assignment minimising the **largest** individual cost, not the sum.

    For a drone light show this is usually the objective that matters: the
    formation is not complete until the slowest drone is in place, so the show's
    dead time is the *maximum* travel distance, not the average. Minimising the
    sum can leave one drone crossing the whole airspace while everyone else
    waits.

    Solved by binary search on the threshold: keep only the edges no more
    expensive than the candidate, and test with a maximum bipartite matching
    whether a complete assignment still exists.
    """
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    if n > m:
        raise ValueError("bottleneck assignment needs at least as many columns as rows")
    values = np.unique(cost)
    lo, hi = 0, len(values) - 1
    best: Optional[float] = None
    while lo <= hi:
        mid = (lo + hi) // 2
        thresh = values[mid]
        adj = [np.flatnonzero(cost[i] <= thresh).tolist() for i in range(n)]
        if _hopcroft_karp(adj, n, m) == n:
            best = float(thresh)
            hi = mid - 1
        else:
            lo = mid + 1
    if best is None:
        raise ValueError("assignment infeasible")
    # Among assignments meeting the bottleneck, take the cheapest by total cost:
    # a large penalty makes over-threshold edges unusable by the Hungarian pass.
    penalty = float(values[-1]) * (n + 1) + 1.0
    masked = np.where(cost <= best, cost, penalty)
    r, c = hungarian(masked)
    return r, c, best
