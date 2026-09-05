"""An independent brute-force MAPF solver, used only to check the real ones.

This is deliberately naive: uniform-cost search over the **joint** state space,
where a state is the tuple of every agent's position and the cost increment per
timestep is one per agent that is not sitting still on its goal. That is the
definition of sum-of-costs, implemented directly, with none of the machinery the
package under test uses -- no constraints, no conflict tree, no MDDs, no
heuristic. If CBS and this agree on the optimal cost of an instance, they are
unlikely to be wrong in the same way.

It is exponential in the number of agents, which is exactly the point CBS
exists to avoid, so it is only usable on the tiny hand-checkable instances in
the test suite.
"""

from __future__ import annotations

import heapq
import itertools
from typing import Dict, List, Optional, Sequence, Tuple

State = Tuple[int, ...]


def joint_successors(graph, state: State, goals: Sequence[int]):
    """Every legal joint move from ``state``, with its sum-of-costs increment."""
    options = []
    for i, loc in enumerate(state):
        moves = list(graph.neighbours[loc]) + [loc]
        options.append(moves)
    for combo in itertools.product(*options):
        if len(set(combo)) != len(combo):
            continue  # vertex conflict
        swap = False
        for i in range(len(combo)):
            for j in range(i + 1, len(combo)):
                if combo[i] == state[j] and combo[j] == state[i]:
                    swap = True
                    break
            if swap:
                break
        if swap:
            continue  # edge (swap) conflict
        step = 0
        for i, nxt in enumerate(combo):
            if not (state[i] == goals[i] and nxt == goals[i]):
                step += 1
        yield combo, step


def brute_force_sum_of_costs(
    graph, starts: Sequence[int], goals: Sequence[int], max_states: int = 400000
) -> Optional[int]:
    """Optimal sum-of-costs by exhaustive joint-space search, or ``None`` if capped."""
    start = tuple(starts)
    goal = tuple(goals)
    best: Dict[State, int] = {start: 0}
    pq: List[Tuple[int, State]] = [(0, start)]
    seen = 0
    while pq:
        cost, state = heapq.heappop(pq)
        if cost > best.get(state, cost + 1):
            continue
        if state == goal:
            return cost
        seen += 1
        if seen > max_states:
            return None
        for nxt, step in joint_successors(graph, state, goals):
            nc = cost + step
            if nc < best.get(nxt, 1 << 60):
                best[nxt] = nc
                heapq.heappush(pq, (nc, nxt))
    return None


def best_prioritised_cost(graph, starts, goals, cache=None) -> Optional[int]:
    """Cheapest plan prioritised planning can produce over *all* priority orders.

    Used to show that even an exhaustive search over orderings does not make
    prioritised planning optimal, and does not make it complete.
    """
    from swarmplan.prioritised.planner import PPConfig, PrioritisedPlanner

    planner = PrioritisedPlanner(graph, starts, goals, PPConfig(), cache)
    best = None
    for order in itertools.permutations(range(len(starts))):
        paths = planner.plan_with_order(order)
        if paths is None:
            continue
        cost = sum(len(p) - 1 for p in paths)
        best = cost if best is None else min(best, cost)
    return best
