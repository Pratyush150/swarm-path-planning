"""Admissible heuristics for the CBS **high level**.

Plain CBS treats the constraint tree as a blind best-first search on cost: it
knows what a node costs, but nothing about what it will still have to pay. That
is the same mistake as running Dijkstra where A* was available, and it is why
plain CBS collapses so early as agents are added.

The fix (Felner et al. 2018, Li et al. 2019) is to look at the *conflict graph*
of a node -- one vertex per agent, one edge per unresolved conflict -- and ask
how much extra cost is unavoidable:

``CG`` (cardinal conflict graph)
    Put an edge between two agents only when their conflict is **cardinal**:
    both MDDs are width 1 there, so any resolution makes one of the two more
    expensive. Resolving a set of cardinal conflicts requires raising the cost
    of at least one agent per conflict, so the **minimum vertex cover** of that
    graph is a lower bound on the extra cost. Admissible, and cheap.

``DG`` (dependency graph)
    Same vertex cover, but the edge test is broader: two agents are joined if
    they cannot *both* keep their current cost, whether or not they currently
    conflict at a width-1 state. Strictly stronger than CG, strictly more
    expensive -- it runs a joint search over the two MDDs per pair.

Both are added to the node's ``f`` value. Neither ever overestimates, so CBS
stays optimal; what changes is how much of the tree it has to build.

We do not implement WDG (the weighted variant, which solves a small MAPF
instance per pair to get an edge *weight* and then an edge-weighted MVC). It is
the strongest of the family and it is the obvious next step; leaving it out is a
scope decision, not an oversight, and the ablation table says what CG and DG are
worth without it.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


def minimum_vertex_cover(n: int, edges: Sequence[Tuple[int, int]]) -> int:
    """Exact minimum vertex cover size, by branch and bound.

    MVC is NP-hard in general, but a CBS conflict graph has one vertex per agent
    and only the *conflicting* agents have edges, so the instances are tiny --
    a few dozen vertices, usually a handful of edges. Branching on an endpoint
    of an uncovered edge is exact: any cover must contain one of the two.
    """
    if not edges:
        return 0
    adj: Dict[int, Set[int]] = {}
    for u, v in edges:
        if u == v:
            continue
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    if not adj:
        return 0

    # Split into connected components: the cover size is additive over them and
    # the branching depth drops accordingly.
    total = 0
    seen: Set[int] = set()
    for root in adj:
        if root in seen:
            continue
        comp: List[int] = []
        stack = [root]
        seen.add(root)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comp_edges = [(u, v) for u in comp for v in adj[u] if u < v]
        total += _mvc_component(comp_edges)
    return total


def _mvc_component(edges: List[Tuple[int, int]]) -> int:
    """MVC of one connected component, by depth-first branch and bound."""
    best = [len(set([u for u, _ in edges] + [v for _, v in edges]))]

    def recurse(remaining: List[Tuple[int, int]], taken: int) -> None:
        if taken >= best[0]:
            return
        if not remaining:
            best[0] = taken
            return
        # Branch on the endpoint with the highest degree first: it is the most
        # likely to be in an optimal cover, so the bound tightens sooner.
        degree: Dict[int, int] = {}
        for u, v in remaining:
            degree[u] = degree.get(u, 0) + 1
            degree[v] = degree.get(v, 0) + 1
        u, v = max(remaining, key=lambda e: degree[e[0]] + degree[e[1]])
        for pick in (u, v) if degree[u] >= degree[v] else (v, u):
            recurse([e for e in remaining if pick not in e], taken + 1)

    recurse(edges, 0)
    return best[0]


def cg_heuristic(cardinal_pairs: Iterable[Tuple[int, int]], n_agents: int) -> int:
    """CG heuristic: MVC over the cardinal-conflict graph."""
    return minimum_vertex_cover(n_agents, list({tuple(sorted(p)) for p in cardinal_pairs}))


def dg_heuristic(dependent_pairs: Iterable[Tuple[int, int]], n_agents: int) -> int:
    """DG heuristic: MVC over the dependency graph."""
    return minimum_vertex_cover(n_agents, list({tuple(sorted(p)) for p in dependent_pairs}))


class DependencyCache:
    """Memoises pairwise dependency answers within one solver run.

    The key is (agent pair, both costs, both constraint fingerprints). Two nodes
    deep in different subtrees frequently ask the same question, and the joint
    MDD search is the most expensive thing the heuristic does.
    """

    def __init__(self) -> None:
        self._store: Dict[tuple, bool] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple) -> Optional[bool]:
        """Cached answer for a key, or ``None``."""
        val = self._store.get(key)
        if val is None:
            self.misses += 1
        else:
            self.hits += 1
        return val

    def put(self, key: tuple, value: bool) -> None:
        """Store an answer."""
        self._store[key] = value

    def __len__(self) -> int:
        return len(self._store)
