"""Conflict-Based Search: the standard optimal MAPF algorithm, plus the
improvements that make it usable.

The reason MAPF is hard is easy to state. With *k* agents on a graph of *V*
vertices the joint state space has *V^k* states; twenty agents on the 819 free
cells of ``random-32-32-20`` is 10^58 joint states. Searching that directly is
hopeless, and A* over the joint space -- even with operator decomposition -- runs
out of memory long before it runs out of patience.

CBS (Sharon et al., AIJ 2015) avoids ever building it. It searches a **tree of
constraints** instead:

1. Plan every agent on its own, ignoring the others. That is *k* independent
   single-agent searches, and the sum of their costs is a lower bound on the
   optimal joint cost.
2. Simulate the joint plan. If nobody collides, that lower bound was achievable
   and the plan is optimal -- done.
3. Otherwise take one collision, say agents 3 and 7 both on cell 91 at t=12,
   and split: one child forbids agent 3 that cell at that time, the other
   forbids agent 7. Every valid joint plan satisfies at least one of the two, so
   nothing is lost.
4. Replan only the constrained agent in each child, and expand the tree
   best-first on cost.

The high level never enumerates joint states; the low level never sees more than
one agent. That decomposition is the whole idea, and it is why CBS scales to
agent counts that joint-space A* cannot approach.

It still blows up. The tree is exponential in the number of conflicts, so the
four switchable improvements here matter more than the base algorithm:

``prioritise_conflicts``
    Split on a *cardinal* conflict when one exists -- one where both children
    provably cost more. Splitting on a conflict that costs nothing to resolve
    just widens the tree.
``bypass``
    If a child has the same cost as its parent and fewer conflicts, adopt the
    child's path into the parent instead of branching at all. Trades a branch
    for a strictly better plan at the same cost.
``disjoint``
    Split on "agent *a* **must** be at *v* at *t*" against "must not", rather
    than "*a* must not" against "*b* must not". The classic split's children
    overlap; the disjoint split partitions.
``heuristic``
    ``cg`` or ``dg``: an admissible estimate of the cost still to be paid, from
    the conflict graph (see :mod:`swarmplan.cbs.heuristics`). It shrinks the tree
    the most and costs the most per node; whether that trade pays depends on the
    map, and the ablation table in the README says where it does.

Every one of them is off by default and independently switchable, because the
point of implementing them is to be able to measure them separately.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Dict, List, Optional, Sequence, Tuple

from ..conflicts import Conflict, ConflictType, find_all_conflicts
from ..constraints import Constraint, ConstraintTable, constraints_for
from ..graph import SearchGraph
from ..lowlevel.astar import space_time_astar
from ..lowlevel.heuristic import HeuristicCache
from ..solution import BUDGET, SOLVED, TIMEOUT, UNSOLVABLE, Solution
from .heuristics import DependencyCache, minimum_vertex_cover
from .mdd import MDD, build_mdd, mdds_dependent


@dataclass
class CBSConfig:
    """Which CBS variant to run.

    Defaults are plain CBS, so ``CBSConfig()`` is the textbook algorithm and
    every improvement has to be asked for by name.
    """

    prioritise_conflicts: bool = False
    bypass: bool = False
    disjoint: bool = False
    heuristic: str = "none"  # "none" | "cg" | "dg"
    time_limit: float = 30.0
    node_limit: Optional[int] = None
    name: Optional[str] = None

    def label(self) -> str:
        """Short name for tables and figures, derived from what is enabled."""
        if self.name:
            return self.name
        bits = []
        if self.prioritise_conflicts:
            bits.append("PC")
        if self.bypass:
            bits.append("BP")
        if self.disjoint:
            bits.append("DS")
        if self.heuristic != "none":
            bits.append(self.heuristic.upper())
        return "CBS" if not bits else "CBS+" + "+".join(bits)

    def validate(self) -> None:
        """Reject impossible configurations early."""
        if self.heuristic not in ("none", "cg", "dg"):
            raise ValueError("heuristic must be one of 'none', 'cg', 'dg'")
        if self.time_limit <= 0:
            raise ValueError("time_limit must be positive")


@dataclass
class CBSNode:
    """One node of the constraint tree."""

    constraints: List[Constraint]
    paths: List[List[int]]
    cost: int
    h: int = 0
    conflicts: Optional[List[Conflict]] = None
    mdds: Dict[int, Optional[MDD]] = field(default_factory=dict)
    depth: int = 0

    @property
    def f(self) -> int:
        """Cost plus admissible estimate of the cost still to be paid."""
        return self.cost + self.h


class CBS:
    """Conflict-Based Search over a :class:`~swarmplan.graph.SearchGraph`."""

    def __init__(
        self,
        graph: SearchGraph,
        starts: Sequence[int],
        goals: Sequence[int],
        config: Optional[CBSConfig] = None,
        cache: Optional[HeuristicCache] = None,
    ) -> None:
        if len(starts) != len(goals):
            raise ValueError("starts and goals must have the same length")
        self.graph = graph
        self.starts = list(starts)
        self.goals = list(goals)
        self.config = config or CBSConfig()
        self.config.validate()
        self.cache = cache or HeuristicCache(graph)
        self.n_agents = len(starts)
        self.htables = [self.cache.get(g) for g in self.goals]
        self.low_level_expanded = 0
        self.low_level_calls = 0
        self.high_level_expanded = 0
        self.high_level_generated = 0
        self._dep_cache = DependencyCache()
        self._deadline = float("inf")

    # -- low level -------------------------------------------------------
    def _plan(self, agent: int, constraints: Sequence[Constraint]) -> Optional[List[int]]:
        """Optimal path for one agent under the node's constraints."""
        table = ConstraintTable(agent)
        table.add_all(constraints_for(constraints, agent))
        table.set_goal(self.goals[agent])
        res = space_time_astar(
            self.graph,
            self.starts[agent],
            self.goals[agent],
            self.htables[agent],
            table,
        )
        self.low_level_calls += 1
        self.low_level_expanded += res.expanded
        return res.path

    def _mdd(self, node: CBSNode, agent: int) -> Optional[MDD]:
        """MDD of ``agent`` at its cost in ``node``, built on demand and cached."""
        if agent in node.mdds:
            return node.mdds[agent]
        table = ConstraintTable(agent)
        table.add_all(constraints_for(node.constraints, agent))
        table.set_goal(self.goals[agent])
        mdd = build_mdd(
            self.graph,
            self.starts[agent],
            self.goals[agent],
            len(node.paths[agent]) - 1,
            self.htables[agent],
            table,
        )
        node.mdds[agent] = mdd
        return mdd

    # -- conflict handling ----------------------------------------------
    def _classify(self, node: CBSNode, c: Conflict) -> Conflict:
        """Tag a conflict cardinal / semi-cardinal / non-cardinal using MDDs."""
        m1 = self._mdd(node, c.a1)
        m2 = self._mdd(node, c.a2)
        if m1 is None or m2 is None:
            return c
        if c.loc2 is None:
            f1 = m1.singleton(c.time) == c.loc1
            f2 = m2.singleton(c.time) == c.loc1
        else:
            f1 = m1.forced_edge(c.time) == (c.loc1, c.loc2)
            f2 = m2.forced_edge(c.time) == (c.loc2, c.loc1)
        kind = ConflictType(int(f1) + int(f2))
        return Conflict(c.a1, c.a2, c.loc1, c.loc2, c.time, kind)

    def _select_conflict(self, node: CBSNode) -> Conflict:
        """Pick the conflict to branch on."""
        conflicts = node.conflicts or []
        if not self.config.prioritise_conflicts:
            return conflicts[0]
        best = None
        for c in conflicts:
            tagged = self._classify(node, c)
            if best is None or tagged.kind > best.kind:
                best = tagged
            if best.kind == ConflictType.CARDINAL:
                break
        return best if best is not None else conflicts[0]

    def _heuristic(self, node: CBSNode) -> int:
        """Admissible high-level heuristic for a node (0 if disabled)."""
        mode = self.config.heuristic
        if mode == "none" or not node.conflicts:
            return 0
        pairs = set()
        seen_pairs = set()
        for c in node.conflicts:
            pair = (min(c.a1, c.a2), max(c.a1, c.a2))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            if mode == "cg":
                if self._classify(node, c).kind == ConflictType.CARDINAL:
                    pairs.add(pair)
            else:
                m1 = self._mdd(node, pair[0])
                m2 = self._mdd(node, pair[1])
                if m1 is None or m2 is None:
                    continue
                key = (
                    pair,
                    m1.cost,
                    m2.cost,
                    tuple(sorted(constraints_for(node.constraints, pair[0]), key=repr)),
                    tuple(sorted(constraints_for(node.constraints, pair[1]), key=repr)),
                )
                cached = self._dep_cache.get(key)
                if cached is None:
                    cached = mdds_dependent(m1, m2)
                    self._dep_cache.put(key, cached)
                if cached:
                    pairs.add(pair)
        return minimum_vertex_cover(self.n_agents, sorted(pairs))

    # -- children --------------------------------------------------------
    def _child_classic(
        self, node: CBSNode, constraint: Constraint
    ) -> Optional[CBSNode]:
        """Child with one extra negative constraint; replan that agent only."""
        constraints = node.constraints + [constraint]
        path = self._plan(constraint.agent, constraints)
        if path is None:
            return None
        paths = list(node.paths)
        paths[constraint.agent] = path
        cost = sum(len(p) - 1 for p in paths)
        return CBSNode(constraints, paths, cost, depth=node.depth + 1)

    def _child_positive(
        self, node: CBSNode, constraint: Constraint
    ) -> Optional[CBSNode]:
        """Disjoint-splitting child: one agent is pinned, everyone else replans.

        The pinned agent already satisfies the landmark (that is where the
        conflict was), so only the agents whose current paths violate it need a
        new search -- which is still the expensive half of disjoint splitting.
        """
        constraints = node.constraints + [constraint]
        paths = list(node.paths)
        t = constraint.time
        for other in range(self.n_agents):
            if other == constraint.agent:
                continue
            table = ConstraintTable(other)
            table.add_all(constraints_for(constraints, other))
            table.set_goal(self.goals[other])
            p = paths[other]
            violates = False
            for step in (t - 1, t):
                if step < 0:
                    continue
                loc = p[step] if step < len(p) else p[-1]
                if table.blocked(loc, step):
                    violates = True
                    break
            if not violates and t < len(p) and t > 0:
                if table.blocked_move(p[t - 1], p[t], t):
                    violates = True
            if not violates:
                continue
            new_path = self._plan(other, constraints)
            if new_path is None:
                return None
            paths[other] = new_path
        cost = sum(len(p) - 1 for p in paths)
        return CBSNode(constraints, paths, cost, depth=node.depth + 1)

    def _expand(self, node: CBSNode, conflict: Conflict) -> List[CBSNode]:
        """The children of a node for one conflict."""
        if self.config.disjoint:
            agent = self._disjoint_agent(node, conflict)
            pos, neg = conflict.disjoint_constraints(agent)
            children = [self._child_positive(node, pos), self._child_classic(node, neg)]
        else:
            c1, c2 = conflict.constraints()
            children = [self._child_classic(node, c1), self._child_classic(node, c2)]
        return [c for c in children if c is not None]

    def _disjoint_agent(self, node: CBSNode, conflict: Conflict) -> int:
        """Which agent to pin in a disjoint split: the more constrained one.

        Pinning the agent whose MDD is narrower at that timestep removes more of
        the search space in the positive child, because that agent had fewer
        alternatives to begin with.
        """
        m1 = node.mdds.get(conflict.a1)
        m2 = node.mdds.get(conflict.a2)
        if m1 is None or m2 is None:
            return conflict.a1
        return conflict.a1 if m1.width(conflict.time) <= m2.width(conflict.time) else conflict.a2

    # -- main loop -------------------------------------------------------
    def initial_paths(self) -> Optional[List[List[int]]]:
        """Each agent's optimal path ignoring every other agent: the CBS root.

        Exposed because it is the "before" picture: the sum of these costs is
        the lower bound the whole search starts from, and the collisions between
        them are exactly what the constraint tree exists to resolve. Returns
        ``None`` if some agent cannot reach its goal at all.
        """
        paths = []
        for a in range(self.n_agents):
            p = self._plan(a, [])
            if p is None:
                return None
            paths.append(p)
        return paths

    def solve(self) -> Solution:
        """Run the search. Returns a :class:`~swarmplan.solution.Solution`."""
        cfg = self.config
        started = time.perf_counter()
        self._deadline = started + cfg.time_limit

        root_paths: List[List[int]] = []
        for a in range(self.n_agents):
            p = self._plan(a, [])
            if p is None:
                return Solution(
                    status=UNSOLVABLE,
                    algorithm=cfg.label(),
                    runtime=time.perf_counter() - started,
                    notes={"reason": f"agent {a} cannot reach its goal"},
                )
            root_paths.append(p)
        root = CBSNode([], root_paths, sum(len(p) - 1 for p in root_paths))
        root.conflicts = find_all_conflicts(root.paths)
        root.h = self._heuristic(root)
        lower_bound = root.f

        counter = 0
        open_list: List[Tuple[int, int, int, int, CBSNode]] = [
            (root.f, len(root.conflicts), root.cost, counter, root)
        ]
        self.high_level_generated = 1

        while open_list:
            if time.perf_counter() > self._deadline:
                return self._result(TIMEOUT, None, started, lower_bound)
            if cfg.node_limit is not None and self.high_level_expanded >= cfg.node_limit:
                return self._result(BUDGET, None, started, lower_bound)

            _, _, _, _, node = heappop(open_list)
            if node.conflicts is None:
                node.conflicts = find_all_conflicts(node.paths)
            lower_bound = max(lower_bound, node.f)
            self.high_level_expanded += 1

            if not node.conflicts:
                return self._result(SOLVED, node.paths, started, node.cost)

            conflict = self._select_conflict(node)
            children = self._expand(node, conflict)

            if cfg.bypass and not cfg.disjoint:
                bypassed = False
                for child in children:
                    child.conflicts = find_all_conflicts(child.paths)
                    if child.cost == node.cost and len(child.conflicts) < len(node.conflicts):
                        node.paths = child.paths
                        node.conflicts = child.conflicts
                        node.mdds = {}
                        node.h = self._heuristic(node)
                        counter += 1
                        heappush(
                            open_list,
                            (node.f, len(node.conflicts), node.cost, counter, node),
                        )
                        self.high_level_generated += 1
                        bypassed = True
                        break
                if bypassed:
                    continue

            for child in children:
                if child.conflicts is None:
                    child.conflicts = find_all_conflicts(child.paths)
                child.h = self._heuristic(child)
                # The heuristic is admissible, so a child's f can never be below
                # its parent's; clamping keeps the open list monotone.
                if child.f < node.f:
                    child.h = node.f - child.cost
                counter += 1
                heappush(
                    open_list, (child.f, len(child.conflicts), child.cost, counter, child)
                )
                self.high_level_generated += 1

        return self._result(UNSOLVABLE, None, started, lower_bound)

    def _result(
        self,
        status: str,
        paths: Optional[List[List[int]]],
        started: float,
        lower_bound: int,
    ) -> Solution:
        """Package the run into a :class:`~swarmplan.solution.Solution`."""
        return Solution(
            paths=paths,
            status=status,
            algorithm=self.config.label(),
            runtime=time.perf_counter() - started,
            high_level_expanded=self.high_level_expanded,
            high_level_generated=self.high_level_generated,
            low_level_expanded=self.low_level_expanded,
            low_level_calls=self.low_level_calls,
            lower_bound=lower_bound,
            suboptimality_bound=1.0,
            notes={
                "heuristic": self.config.heuristic,
                "prioritise_conflicts": self.config.prioritise_conflicts,
                "bypass": self.config.bypass,
                "disjoint": self.config.disjoint,
                "dependency_cache": len(self._dep_cache),
            },
        )


def solve_cbs(
    graph: SearchGraph,
    starts: Sequence[int],
    goals: Sequence[int],
    **kwargs,
) -> Solution:
    """Convenience wrapper: build a :class:`CBS` from keyword config and run it."""
    config = CBSConfig(**kwargs)
    return CBS(graph, starts, goals, config).solve()
