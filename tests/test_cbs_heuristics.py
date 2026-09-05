"""Minimum vertex cover, the admissible core of the CG and DG heuristics."""

from __future__ import annotations

import itertools
import random

from swarmplan.cbs.heuristics import (
    DependencyCache,
    cg_heuristic,
    dg_heuristic,
    minimum_vertex_cover,
)


def brute_force_cover(n, edges):
    """Exhaustive minimum vertex cover, for checking the branch-and-bound one."""
    for k in range(n + 1):
        for combo in itertools.combinations(range(n), k):
            s = set(combo)
            if all(u in s or v in s for u, v in edges):
                return k
    return n


def test_known_small_covers():
    assert minimum_vertex_cover(4, []) == 0
    assert minimum_vertex_cover(2, [(0, 1)]) == 1
    assert minimum_vertex_cover(3, [(0, 1), (1, 2), (0, 2)]) == 2  # triangle
    assert minimum_vertex_cover(4, [(0, 1), (1, 2), (2, 3)]) == 2  # path
    assert minimum_vertex_cover(5, [(0, 1), (2, 3)]) == 2  # two components
    assert minimum_vertex_cover(4, [(1, 1)]) == 0  # self loops are ignored


def test_star_graph_cover_is_one():
    assert minimum_vertex_cover(6, [(0, i) for i in range(1, 6)]) == 1


def test_matches_brute_force_on_random_graphs():
    rng = random.Random(20260904)
    for _ in range(60):
        n = rng.randint(2, 8)
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.4]
        assert minimum_vertex_cover(n, edges) == brute_force_cover(n, edges)


def test_cg_and_dg_deduplicate_pairs():
    assert cg_heuristic([(1, 0), (0, 1)], 3) == 1
    assert dg_heuristic([(0, 1), (1, 2)], 3) == 1


def test_dependency_cache_counts_hits_and_misses():
    cache = DependencyCache()
    assert cache.get(("a",)) is None
    cache.put(("a",), True)
    assert cache.get(("a",)) is True
    assert cache.hits == 1 and cache.misses == 1
    assert len(cache) == 1
