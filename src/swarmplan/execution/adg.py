"""Executing a MAPF plan on real vehicles that do not keep perfect time.

A MAPF plan is a *timetable*: agent 4 is on cell 91 at timestep 12. Real drones
and real AMRs do not hit timesteps. One takes an extra 300 ms to spin up, one
slows down in a gust, one stops because a person walked in front of it. The
moment a single agent is late, the timetable is a lie -- and the failure is not
"the plan finishes late", it is **collision**, because the agent that was
supposed to have vacated a cell is still in it while the next agent arrives on
schedule.

The fix is the Action Dependency Graph (equivalently the Temporal Plan Graph;
Hönig et al., "Multi-Agent Path Finding with Kinematic Constraints", ICAPS
2016). Throw away the absolute times and keep only the **ordering** the plan
implies:

* *type 1* -- agent *i* reaches waypoint *k* only after it reached *k-1*;
* *type 2* -- if the plan has agent *j* entering cell *v* after agent *i* leaves
  it, then agent *j* may not enter *v* until agent *i* has actually left.

Execute by advancing each agent as soon as its dependencies are satisfied, and
the plan degrades gracefully: a late agent delays exactly the agents that were
scheduled behind it, everyone else carries on, and **no ordering is ever
violated, so no collision the planner ruled out can occur**. That property is
what makes the plan safe to fly rather than merely correct on paper.

The test suite delays one agent by an arbitrary number of ticks and asserts
that the ADG execution stays collision-free while a fixed-timetable execution
of the same plan does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

Node = Tuple[int, int]  # (agent, step)


@dataclass
class ExecutionTrace:
    """What actually happened when a plan was executed."""

    positions: List[List[int]]  # positions[tick][agent]
    completed_at: List[int]
    ticks: int
    deadlocked: bool = False

    @property
    def makespan(self) -> int:
        """Tick at which the last agent reached its goal."""
        return max(self.completed_at) if self.completed_at else 0

    def collisions(self) -> List[Tuple[int, int, int, str]]:
        """Every collision in the trace, as ``(tick, agent_a, agent_b, kind)``."""
        out = []
        for t, row in enumerate(self.positions):
            seen: Dict[int, int] = {}
            for a, loc in enumerate(row):
                if loc in seen:
                    out.append((t, seen[loc], a, "vertex"))
                else:
                    seen[loc] = a
            if t > 0:
                prev = self.positions[t - 1]
                for a in range(len(row)):
                    for b in range(a + 1, len(row)):
                        if row[a] == prev[b] and row[b] == prev[a] and row[a] != row[b]:
                            out.append((t, a, b, "swap"))
        return out

    def is_safe(self) -> bool:
        """True if the execution never put two agents in the same place."""
        return not self.collisions()


class ActionDependencyGraph:
    """Orderings extracted from a MAPF plan, and an executor that respects them."""

    def __init__(self, paths: Sequence[Sequence[int]], allow_following: bool = True) -> None:
        if not paths:
            raise ValueError("no paths")
        self.paths = [list(p) for p in paths]
        self.n_agents = len(self.paths)
        self.allow_following = allow_following
        self.dependencies: Dict[Node, List[Node]] = {}
        self._build()

    # -- construction ----------------------------------------------------
    def _intervals(self, agent: int) -> List[Tuple[int, int, int]]:
        """Compressed occupancy of one agent: ``(cell, first_step, last_step)``."""
        path = self.paths[agent]
        out: List[Tuple[int, int, int]] = []
        start = 0
        for k in range(1, len(path) + 1):
            if k == len(path) or path[k] != path[start]:
                out.append((path[start], start, k - 1))
                start = k
        return out

    def _build(self) -> None:
        """Add type-1 (sequential) and type-2 (cross-agent) dependencies."""
        for i, path in enumerate(self.paths):
            for k in range(1, len(path)):
                self.dependencies[(i, k)] = [(i, k - 1)]

        occupancy: Dict[int, List[Tuple[int, int, int]]] = {}
        for i in range(self.n_agents):
            for cell, first, last in self._intervals(i):
                occupancy.setdefault(cell, []).append((first, last, i))

        for cell, entries in occupancy.items():
            entries.sort()
            for (first_a, last_a, a), (first_b, _, b) in zip(entries, entries[1:]):
                if a == b:
                    continue
                if first_b <= last_a:
                    raise ValueError(
                        f"plan is not conflict-free: agents {a} and {b} overlap on cell {cell}"
                    )
                exit_step = last_a + 1
                if exit_step >= len(self.paths[a]):
                    raise ValueError(
                        f"plan is not conflict-free: agent {a} parks on cell {cell} "
                        f"which agent {b} later needs"
                    )
                if first_b == 0:
                    raise ValueError("an agent's start cell cannot be entered later")
                self.dependencies.setdefault((b, first_b), []).append((a, exit_step))

    def predecessors(self, node: Node) -> List[Node]:
        """Nodes that must complete before ``node`` may."""
        return self.dependencies.get(node, [])

    @property
    def n_dependencies(self) -> int:
        """Total number of ordering edges."""
        return sum(len(v) for v in self.dependencies.values())

    @property
    def n_cross_agent(self) -> int:
        """Ordering edges that couple two different agents."""
        return sum(
            1
            for (agent, _), deps in self.dependencies.items()
            for d in deps
            if d[0] != agent
        )

    # -- execution -------------------------------------------------------
    def execute(
        self,
        delays: Optional[Dict[Node, int]] = None,
        max_ticks: Optional[int] = None,
    ) -> ExecutionTrace:
        """Run the plan, holding each agent until its dependencies are met.

        ``delays`` maps ``(agent, step)`` to the number of extra ticks that step
        takes -- a stalled motor, a slow spin-up, a person in the aisle. Nothing
        else about the plan changes: the ordering does the work.
        """
        delays = dict(delays or {})
        progress = [0] * self.n_agents
        finals = [len(p) - 1 for p in self.paths]
        remaining = dict(delays)
        completed_at = [0] * self.n_agents
        positions = [[p[0] for p in self.paths]]
        limit = max_ticks if max_ticks is not None else (
            sum(finals) + sum(delays.values()) + self.n_agents + 2
        )

        tick = 0
        while any(progress[i] < finals[i] for i in range(self.n_agents)):
            tick += 1
            if tick > limit:
                return ExecutionTrace(positions, completed_at, tick, deadlocked=True)
            moved_this_tick: Set[int] = set()
            changed = True
            while changed:
                changed = False
                for i in range(self.n_agents):
                    if i in moved_this_tick or progress[i] >= finals[i]:
                        continue
                    nxt = progress[i] + 1
                    node = (i, nxt)
                    if remaining.get(node, 0) > 0:
                        remaining[node] -= 1
                        moved_this_tick.add(i)
                        continue
                    ready = True
                    for dep_agent, dep_step in self.predecessors(node):
                        if dep_agent == i:
                            continue
                        if progress[dep_agent] < dep_step:
                            ready = False
                            break
                        if not self.allow_following and dep_agent in moved_this_tick:
                            ready = False
                            break
                    if not ready:
                        continue
                    progress[i] = nxt
                    moved_this_tick.add(i)
                    changed = True
                    if nxt == finals[i]:
                        completed_at[i] = tick
            positions.append([self.paths[i][progress[i]] for i in range(self.n_agents)])
            if not moved_this_tick and all(
                remaining.get((i, progress[i] + 1), 0) == 0
                for i in range(self.n_agents)
                if progress[i] < finals[i]
            ):
                return ExecutionTrace(positions, completed_at, tick, deadlocked=True)
        return ExecutionTrace(positions, completed_at, tick)


def fixed_schedule_execution(
    paths: Sequence[Sequence[int]],
    delays: Optional[Dict[Node, int]] = None,
) -> ExecutionTrace:
    """Execute the plan as a timetable: everyone advances every tick regardless.

    This is what happens when a plan is dispatched as absolute times and one
    vehicle is late. It is included so the ADG's guarantee can be demonstrated
    against something, not asserted: on the same plan with the same delay, this
    produces collisions and :meth:`ActionDependencyGraph.execute` does not.
    """
    delays = dict(delays or {})
    paths = [list(p) for p in paths]
    n = len(paths)
    finals = [len(p) - 1 for p in paths]
    progress = [0] * n
    remaining = dict(delays)
    completed_at = [0] * n
    positions = [[p[0] for p in paths]]
    tick = 0
    limit = sum(finals) + sum(delays.values()) + n + 2
    while any(progress[i] < finals[i] for i in range(n)):
        tick += 1
        if tick > limit:
            return ExecutionTrace(positions, completed_at, tick, deadlocked=True)
        for i in range(n):
            if progress[i] >= finals[i]:
                continue
            node = (i, progress[i] + 1)
            if remaining.get(node, 0) > 0:
                remaining[node] -= 1
                continue
            progress[i] += 1
            if progress[i] == finals[i]:
                completed_at[i] = tick
        positions.append([paths[i][progress[i]] for i in range(n)])
    return ExecutionTrace(positions, completed_at, tick)
