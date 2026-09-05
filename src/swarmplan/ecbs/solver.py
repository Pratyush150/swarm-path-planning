"""ECBS: bounded-suboptimal CBS, with focal search at both levels.

Optimal CBS is the algorithm you quote in a paper. ECBS (Barer, Sharon, Stern,
Felner, SoCS 2014) is the one that actually flies a warehouse or a light show,
because the last few percent of solution cost is worth nothing to an operator
and is worth *everything* to the search: dropping the optimality requirement
from "exact" to "within 5%" moves the tractable agent count up by a large
factor, and this repository measures how large on the standard benchmarks.

How the bound is kept
---------------------
Both levels run a focal search.

**Low level.** For each agent, focal space-time A* returns a path costing at
most ``w`` times the constrained optimum, chosen from within that budget to
minimise conflicts with the other agents' current paths. It also returns
``lb_i``, the ``f_min`` at the moment it stopped -- a genuine lower bound on
that agent's constrained optimum.

**High level.** ``LB(N) = sum_i lb_i(N)`` is a lower bound on the cost of the
best solution below node ``N``. OPEN is ordered by ``LB``; FOCAL holds every
open node whose *cost* is at most ``w * LB_min`` and is ordered by number of
conflicts. Expanding from FOCAL means the first conflict-free node found costs
at most ``w * LB_min <= w * C*``.

So the suboptimality factor is a hard guarantee, not a hope, and
``Solution.suboptimality_bound`` carries it. In practice the solutions come out
much closer to optimal than the bound allows, which the benchmark tables show
by reporting measured cost against the true lower bound rather than against
``w``.

Relationship to EECBS
---------------------
EECBS (Li, Ruml, Koenig, AAAI 2021) is ECBS plus an *inadmissible* high-level
heuristic learned online, used to order FOCAL, with the admissible bound kept
separately. We implement ECBS with an optional **admissible** CG/DG heuristic
added to ``LB`` (which tightens ``LB_min`` and therefore the bound), and we do
not implement the online-learned inadmissible estimate. That is the main gap
between this and a full EECBS, and it is why the reference implementation this
was checked against is cited in the README rather than claimed to be matched.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Dict, List, Optional, Sequence, Tuple

from ..conflicts import Conflict, ConflictType, find_all_conflicts
from ..constraints import Constraint, ConstraintTable, constraints_for
from ..graph import SearchGraph
from ..lowlevel.focal import ConflictAvoidanceTable, focal_space_time_astar
from ..lowlevel.heuristic import HeuristicCache
from ..solution import BUDGET, SOLVED, TIMEOUT, UNSOLVABLE, Solution
from ..cbs.heuristics import minimum_vertex_cover
from ..cbs.mdd import build_mdd, mdds_dependent


@dataclass
class ECBSConfig:
    """Configuration for :class:`ECBS`."""

    w: float = 1.5
    prioritise_conflicts: bool = False
    heuristic: str = "none"  # "none" | "cg" | "dg"
    time_limit: float = 30.0
    node_limit: Optional[int] = None
    name: Optional[str] = None

    def label(self) -> str:
        """Short name for tables and figures."""
        if self.name:
            return self.name
        tag = f"ECBS(w={self.w:g})"
        extras = []
        if self.prioritise_conflicts:
            extras.append("PC")
        if self.heuristic != "none":
            extras.append(self.heuristic.upper())
        return tag if not extras else tag + "+" + "+".join(extras)

    def validate(self) -> None:
        """Reject impossible configurations early."""
        if self.w < 1.0:
            raise ValueError("w must be >= 1")
        if self.heuristic not in ("none", "cg", "dg"):
            raise ValueError("heuristic must be one of 'none', 'cg', 'dg'")
        if self.time_limit <= 0:
            raise ValueError("time_limit must be positive")


@dataclass
class ECBSNode:
    """A high-level node: constraints, paths, and the per-agent lower bounds."""

    constraints: List[Constraint]
    paths: List[List[int]]
    lbs: List[int]
    cost: int
    conflicts: List[Conflict] = field(default_factory=list)
    h: int = 0
    expanded: bool = False
    in_focal: bool = False
    depth: int = 0

    @property
    def lb(self) -> int:
        """Admissible lower bound on the best solution below this node."""
        return sum(self.lbs) + self.h


class ECBS:
    """Bounded-suboptimal Conflict-Based Search."""

    def __init__(
        self,
        graph: SearchGraph,
        starts: Sequence[int],
        goals: Sequence[int],
        config: Optional[ECBSConfig] = None,
        cache: Optional[HeuristicCache] = None,
    ) -> None:
        if len(starts) != len(goals):
            raise ValueError("starts and goals must have the same length")
        self.graph = graph
        self.starts = list(starts)
        self.goals = list(goals)
        self.config = config or ECBSConfig()
        self.config.validate()
        self.cache = cache or HeuristicCache(graph)
        self.n_agents = len(starts)
        self.htables = [self.cache.get(g) for g in self.goals]
        self.low_level_expanded = 0
        self.low_level_calls = 0
        self.high_level_expanded = 0
        self.high_level_generated = 0

    # -- low level -------------------------------------------------------
    def _plan(
        self, agent: int, constraints: Sequence[Constraint], paths: Sequence[List[int]]
    ) -> Tuple[Optional[List[int]], int]:
        """Focal search for one agent, avoiding the other agents' current paths."""
        table = ConstraintTable(agent)
        table.add_all(constraints_for(constraints, agent))
        table.set_goal(self.goals[agent])
        cat = ConflictAvoidanceTable(
            [p for i, p in enumerate(paths) if i != agent and p is not None]
        )
        res = focal_space_time_astar(
            self.graph,
            self.starts[agent],
            self.goals[agent],
            self.htables[agent],
            table,
            cat,
            w=self.config.w,
        )
        self.low_level_calls += 1
        self.low_level_expanded += res.expanded
        return res.path, res.lower_bound

    def _node_heuristic(self, node: ECBSNode) -> int:
        """Optional admissible CG/DG heuristic on the node's conflict graph."""
        mode = self.config.heuristic
        if mode == "none" or not node.conflicts:
            return 0
        mdds: Dict[int, object] = {}

        def mdd(agent: int):
            if agent not in mdds:
                table = ConstraintTable(agent)
                table.add_all(constraints_for(node.constraints, agent))
                table.set_goal(self.goals[agent])
                mdds[agent] = build_mdd(
                    self.graph,
                    self.starts[agent],
                    self.goals[agent],
                    len(node.paths[agent]) - 1,
                    self.htables[agent],
                    table,
                )
            return mdds[agent]

        pairs = set()
        seen = set()
        for c in node.conflicts:
            pair = (min(c.a1, c.a2), max(c.a1, c.a2))
            if pair in seen:
                continue
            seen.add(pair)
            m1, m2 = mdd(pair[0]), mdd(pair[1])
            if m1 is None or m2 is None:
                continue
            if mode == "cg":
                if c.loc2 is None:
                    forced = m1.singleton(c.time) == c.loc1 and m2.singleton(c.time) == c.loc1
                else:
                    forced = m1.forced_edge(c.time) == (c.loc1, c.loc2) and m2.forced_edge(
                        c.time
                    ) == (c.loc2, c.loc1)
                if forced:
                    pairs.add(pair)
            elif mdds_dependent(m1, m2):
                pairs.add(pair)
        if not pairs:
            return 0
        # The heuristic bounds the extra cost of the *optimal* solution below
        # this node, so it is added to the lower bound and never to the cost.
        return minimum_vertex_cover(self.n_agents, sorted(pairs))

    def _select_conflict(self, node: ECBSNode) -> Conflict:
        """First conflict, or the earliest cardinal one when PC is enabled."""
        if not self.config.prioritise_conflicts:
            return node.conflicts[0]
        table_cache: Dict[int, ConstraintTable] = {}

        def mdd(agent: int):
            if agent not in table_cache:
                t = ConstraintTable(agent)
                t.add_all(constraints_for(node.constraints, agent))
                t.set_goal(self.goals[agent])
                table_cache[agent] = t
            return build_mdd(
                self.graph,
                self.starts[agent],
                self.goals[agent],
                len(node.paths[agent]) - 1,
                self.htables[agent],
                table_cache[agent],
            )

        best = node.conflicts[0]
        best_kind = ConflictType.NON_CARDINAL
        for c in node.conflicts:
            m1, m2 = mdd(c.a1), mdd(c.a2)
            if m1 is None or m2 is None:
                continue
            if c.loc2 is None:
                f1 = m1.singleton(c.time) == c.loc1
                f2 = m2.singleton(c.time) == c.loc1
            else:
                f1 = m1.forced_edge(c.time) == (c.loc1, c.loc2)
                f2 = m2.forced_edge(c.time) == (c.loc2, c.loc1)
            kind = ConflictType(int(f1) + int(f2))
            if kind > best_kind:
                best, best_kind = c, kind
            if best_kind == ConflictType.CARDINAL:
                break
        return best

    # -- main loop -------------------------------------------------------
    def solve(self) -> Solution:
        """Run the search. Returns a :class:`~swarmplan.solution.Solution`."""
        cfg = self.config
        started = time.perf_counter()
        deadline = started + cfg.time_limit

        paths: List[List[int]] = []
        lbs: List[int] = []
        for a in range(self.n_agents):
            p, lb = self._plan(a, [], paths)
            if p is None:
                return Solution(
                    status=UNSOLVABLE,
                    algorithm=cfg.label(),
                    runtime=time.perf_counter() - started,
                    suboptimality_bound=cfg.w,
                    notes={"reason": f"agent {a} cannot reach its goal"},
                )
            paths.append(p)
            lbs.append(lb)
        root = ECBSNode([], paths, lbs, sum(len(p) - 1 for p in paths))
        root.conflicts = find_all_conflicts(root.paths)
        root.h = self._node_heuristic(root)

        counter = 0
        open_list: List[Tuple[int, int, ECBSNode]] = []
        focal: List[Tuple[int, int, int, ECBSNode]] = []
        heappush(open_list, (root.lb, counter, root))
        heappush(focal, (len(root.conflicts), root.cost, counter, root))
        root.in_focal = True
        self.high_level_generated = 1
        lb_min = root.lb
        best_lb = lb_min

        while focal or open_list:
            if time.perf_counter() > deadline:
                return self._result(TIMEOUT, None, started, best_lb)
            if cfg.node_limit is not None and self.high_level_expanded >= cfg.node_limit:
                return self._result(BUDGET, None, started, best_lb)

            while open_list and open_list[0][2].expanded:
                heappop(open_list)
            if not open_list:
                break
            new_lb_min = open_list[0][2].lb
            if new_lb_min > lb_min:
                lb_min = new_lb_min
                bound = cfg.w * lb_min
                for _, cnt, node in open_list:
                    if not node.expanded and not node.in_focal and node.cost <= bound:
                        node.in_focal = True
                        heappush(focal, (len(node.conflicts), node.cost, cnt, node))
            best_lb = max(best_lb, lb_min)

            while focal and focal[0][3].expanded:
                heappop(focal)
            if not focal:
                # Nothing within the suboptimality bound is unexpanded; widen it
                # by expanding the best node in OPEN instead.
                node = open_list[0][2]
            else:
                node = heappop(focal)[3]
            node.expanded = True
            self.high_level_expanded += 1

            if not node.conflicts:
                return self._result(SOLVED, node.paths, started, max(best_lb, node.lb))

            conflict = self._select_conflict(node)
            for constraint in conflict.constraints():
                child = self._child(node, constraint)
                if child is None:
                    continue
                counter += 1
                heappush(open_list, (child.lb, counter, child))
                if child.cost <= cfg.w * lb_min:
                    child.in_focal = True
                    heappush(focal, (len(child.conflicts), child.cost, counter, child))
                self.high_level_generated += 1

        return self._result(UNSOLVABLE, None, started, best_lb)

    def _child(self, node: ECBSNode, constraint: Constraint) -> Optional[ECBSNode]:
        """One child node: add a constraint and replan its agent under focal search."""
        constraints = node.constraints + [constraint]
        path, lb = self._plan(constraint.agent, constraints, node.paths)
        if path is None:
            return None
        paths = list(node.paths)
        lbs = list(node.lbs)
        paths[constraint.agent] = path
        # A child can never have a smaller lower bound than its parent: the
        # constraint set only grew.
        lbs[constraint.agent] = max(lb, node.lbs[constraint.agent])
        child = ECBSNode(
            constraints,
            paths,
            lbs,
            sum(len(p) - 1 for p in paths),
            depth=node.depth + 1,
        )
        child.conflicts = find_all_conflicts(paths)
        child.h = self._node_heuristic(child)
        return child

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
            suboptimality_bound=self.config.w,
            notes={"w": self.config.w, "heuristic": self.config.heuristic},
        )


def solve_ecbs(
    graph: SearchGraph,
    starts: Sequence[int],
    goals: Sequence[int],
    **kwargs,
) -> Solution:
    """Convenience wrapper: build an :class:`ECBS` from keyword config and run it."""
    return ECBS(graph, starts, goals, ECBSConfig(**kwargs)).solve()
