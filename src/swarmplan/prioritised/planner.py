"""Prioritised planning: fast, simple, ships in real systems, and incomplete.

Give the agents a total order. Plan agent 1 alone. Plan agent 2 treating agent
1's path as a moving obstacle. Plan agent 3 avoiding both. Continue. There is no
tree, no backtracking, and *k* single-agent searches solve the whole instance,
so it is orders of magnitude faster than CBS and it is what a great many
deployed multi-robot systems actually run.

The catch is not that it is suboptimal. The catch is that it is **incomplete**:
there are solvable instances it cannot solve under *any* priority order, so no
amount of random restarting rescues it. The canonical example is three cells of
corridor with one alcove::

    A B C        agent 1: A -> C
    . D .        agent 2: C -> A   (D is the only alcove, below B)

CBS solves this in a few nodes -- one agent ducks into the alcove and lets the
other past. Prioritised planning cannot solve it in either order: whichever
agent plans first walks straight down the corridor and *parks on the other's
start cell*, and the second agent has nowhere to be. ``tests/test_prioritised.py``
runs exactly this instance, asserts that both orders fail, and asserts that CBS
succeeds on it, because a baseline is only honest if its failure mode is
demonstrated rather than described.

What it is genuinely good for: large open maps, low density, and any system that
would rather have a plan in 50 ms and re-plan on failure than a provably optimal
plan in 20 s. The random-restart variant here recovers a good fraction of the
instances a single fixed order fails on -- how large a fraction is in the
benchmark tables, measured, not assumed.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

from ..constraints import ConstraintTable
from ..graph import SearchGraph
from ..lowlevel.astar import space_time_astar
from ..lowlevel.heuristic import UNREACHABLE, HeuristicCache
from ..solution import FAILED, SOLVED, TIMEOUT, UNSOLVABLE, Solution

#: Priority orderings the planner knows how to build.
ORDERS = ("index", "longest_first", "shortest_first", "random")


@dataclass
class PPConfig:
    """Configuration for :class:`PrioritisedPlanner`."""

    order: str = "index"
    restarts: int = 1
    seed: int = 0
    time_limit: float = 30.0
    horizon_slack: int = 0
    name: Optional[str] = None

    def label(self) -> str:
        """Short name for tables and figures."""
        if self.name:
            return self.name
        if self.restarts > 1:
            return f"PP(restarts={self.restarts})"
        return "PP" if self.order == "index" else f"PP({self.order})"

    def validate(self) -> None:
        """Reject impossible configurations early."""
        if self.order not in ORDERS:
            raise ValueError(f"order must be one of {ORDERS}")
        if self.restarts < 1:
            raise ValueError("restarts must be >= 1")


class PrioritisedPlanner:
    """Plan agents one at a time, each avoiding the ones already planned."""

    def __init__(
        self,
        graph: SearchGraph,
        starts: Sequence[int],
        goals: Sequence[int],
        config: Optional[PPConfig] = None,
        cache: Optional[HeuristicCache] = None,
    ) -> None:
        if len(starts) != len(goals):
            raise ValueError("starts and goals must have the same length")
        self.graph = graph
        self.starts = list(starts)
        self.goals = list(goals)
        self.config = config or PPConfig()
        self.config.validate()
        self.cache = cache or HeuristicCache(graph)
        self.n_agents = len(starts)
        self.htables = [self.cache.get(g) for g in self.goals]
        self.low_level_expanded = 0
        self.low_level_calls = 0

    def priority_order(self, rng: Optional[random.Random] = None) -> List[int]:
        """Build the priority order named in the config."""
        order = self.config.order
        agents = list(range(self.n_agents))
        if order == "index":
            return agents
        if order == "random":
            rng = rng or random.Random(self.config.seed)
            rng.shuffle(agents)
            return agents
        dists = [int(self.htables[a][self.starts[a]]) for a in agents]
        reverse = order == "longest_first"
        return sorted(agents, key=lambda a: dists[a], reverse=reverse)

    def plan_with_order(self, order: Sequence[int]) -> Optional[List[List[int]]]:
        """Plan in the given priority order. ``None`` if any agent gets stuck.

        The horizon matters here in a way it does not in CBS: a low-priority
        agent may have to wait a long time for the traffic ahead to clear, so
        the search is allowed to run past the last reservation by the number of
        free cells (plus any configured slack) before it gives up.
        """
        reserved: List[List[int]] = []
        paths: List[Optional[List[int]]] = [None] * self.n_agents
        for agent in order:
            table = ConstraintTable(agent)
            for p in reserved:
                table.reserve_path(p)
            table.set_goal(self.goals[agent])
            horizon = table.max_time + self.graph.n_nodes + 1 + self.config.horizon_slack
            res = space_time_astar(
                self.graph,
                self.starts[agent],
                self.goals[agent],
                self.htables[agent],
                table,
                max_time=horizon,
            )
            self.low_level_calls += 1
            self.low_level_expanded += res.expanded
            if res.path is None:
                return None
            paths[agent] = res.path
            reserved.append(res.path)
        return [p for p in paths if p is not None]

    def solve(self) -> Solution:
        """Run prioritised planning, with random restarts if configured."""
        cfg = self.config
        started = time.perf_counter()
        deadline = started + cfg.time_limit

        for a in range(self.n_agents):
            if int(self.htables[a][self.starts[a]]) >= UNREACHABLE:
                return Solution(
                    status=UNSOLVABLE,
                    algorithm=cfg.label(),
                    runtime=time.perf_counter() - started,
                    notes={"reason": f"agent {a} cannot reach its goal"},
                )

        rng = random.Random(cfg.seed)
        attempts = 0
        for attempt in range(cfg.restarts):
            if time.perf_counter() > deadline:
                break
            attempts += 1
            first = attempt == 0 and cfg.order != "random"
            order = self.priority_order() if first else self.priority_order(rng)
            if attempt > 0 and cfg.order != "random":
                order = list(order)
                rng.shuffle(order)
            paths = self.plan_with_order(order)
            if paths is not None:
                return Solution(
                    paths=paths,
                    status=SOLVED,
                    algorithm=cfg.label(),
                    runtime=time.perf_counter() - started,
                    low_level_expanded=self.low_level_expanded,
                    low_level_calls=self.low_level_calls,
                    lower_bound=-1,
                    notes={"attempts": attempts, "order": list(order)},
                )
        # Prioritised planning that fails has proved nothing about the instance:
        # the status says "no plan from this planner", never "no plan exists".
        timed_out = time.perf_counter() > deadline
        return Solution(
            status=TIMEOUT if timed_out else FAILED,
            algorithm=cfg.label(),
            runtime=time.perf_counter() - started,
            low_level_expanded=self.low_level_expanded,
            low_level_calls=self.low_level_calls,
            notes={"attempts": attempts, "reason": "no priority order tried succeeded"},
        )


def solve_pp(
    graph: SearchGraph,
    starts: Sequence[int],
    goals: Sequence[int],
    **kwargs,
) -> Solution:
    """Convenience wrapper: build a :class:`PrioritisedPlanner` and run it."""
    return PrioritisedPlanner(graph, starts, goals, PPConfig(**kwargs)).solve()
