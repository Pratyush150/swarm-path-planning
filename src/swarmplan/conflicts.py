"""Conflict detection and classification.

Two kinds of collision exist on a grid, and the second is the one naive
implementations miss:

**Vertex conflict** -- two agents occupy the same cell at the same timestep.
Easy to spot, easy to remember.

**Edge conflict (swap)** -- at timestep ``t`` agent *i* moves ``u -> v`` while
agent *j* moves ``v -> u``. Neither agent is ever in the same cell as the other
at the same timestep, so a check that only compares positions per timestep
declares this plan collision-free. Two quadrotors flying it pass through each
other. Every conflict check in this package tests both cases, and the test suite
contains the exact two-agent corridor instance that separates them.

Agents also **park on their goals**: after an agent arrives it stays there
forever, and a later agent that walks over that cell is in conflict. Paths are
therefore compared out to the longest path in the plan, not to their own length.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Sequence, Tuple

from .constraints import Constraint


class ConflictType(IntEnum):
    """How much a conflict is going to cost to resolve.

    The ordering matters: CBS with prioritising conflicts branches on the most
    expensive conflict first, because resolving it raises the cost of both
    children immediately and prunes the subtree that a cheap conflict would have
    let it wander into.
    """

    NON_CARDINAL = 0
    SEMI_CARDINAL = 1
    CARDINAL = 2


@dataclass(frozen=True)
class Conflict:
    """A collision between two agents in a joint plan.

    ``loc2`` is ``None`` for a vertex conflict. For an edge conflict the pair
    ``(loc1, loc2)`` is the move agent ``a1`` makes, and ``a2`` makes the
    reverse move at the same timestep.
    """

    a1: int
    a2: int
    loc1: int
    loc2: Optional[int]
    time: int
    kind: ConflictType = ConflictType.NON_CARDINAL

    @property
    def is_edge(self) -> bool:
        """True if this is a swap conflict rather than a shared cell."""
        return self.loc2 is not None

    def constraints(self) -> Tuple[Constraint, Constraint]:
        """The two negative constraints classic CBS branches on.

        One child forbids agent ``a1`` the conflicting vertex/move, the other
        forbids ``a2``. Between them they cover every solution of the parent, so
        the split loses nothing: that is why CBS stays optimal.
        """
        if self.loc2 is None:
            return (
                Constraint(self.a1, self.loc1, None, self.time),
                Constraint(self.a2, self.loc1, None, self.time),
            )
        return (
            Constraint(self.a1, self.loc2, self.loc1, self.time),
            Constraint(self.a2, self.loc1, self.loc2, self.time),
        )

    def disjoint_constraints(self, agent: Optional[int] = None) -> Tuple[Constraint, Constraint]:
        """The disjoint-splitting pair: one agent is forced *onto* the conflict.

        Classic splitting produces two children whose solution sets overlap --
        a plan where neither agent uses the vertex is legal in both -- and CBS
        can therefore explore the same joint plan twice. Disjoint splitting
        picks one agent and branches on "``a`` must be here" against "``a`` must
        not be here", which partitions the space instead of covering it. The
        positive child is more expensive (every *other* agent has to be
        replanned) but the tree is provably smaller.
        """
        a = self.a1 if agent is None else agent
        if a not in (self.a1, self.a2):
            raise ValueError("agent must be one of the two in the conflict")
        if self.loc2 is None:
            return (
                Constraint(a, self.loc1, None, self.time, positive=True),
                Constraint(a, self.loc1, None, self.time, positive=False),
            )
        frm, to = (self.loc1, self.loc2) if a == self.a1 else (self.loc2, self.loc1)
        return (
            Constraint(a, to, frm, self.time, positive=True),
            Constraint(a, to, frm, self.time, positive=False),
        )

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        what = f"{self.loc1}" if self.loc2 is None else f"{self.loc1}<->{self.loc2}"
        return f"conflict(a{self.a1},a{self.a2} {what} t={self.time} {self.kind.name})"


def at(path: Sequence[int], t: int) -> int:
    """Where an agent is at timestep ``t``; it parks on its goal after arrival."""
    return path[t] if t < len(path) else path[-1]


def find_first_conflict(
    paths: Sequence[Sequence[int]], start_agent: int = 0
) -> Optional[Conflict]:
    """The earliest conflict in a joint plan, or ``None`` if it is valid.

    Scanning by timestep and returning the first hit keeps the constraint tree
    shallow: resolving an early conflict often removes the later ones for free.
    """
    n = len(paths)
    horizon = max((len(p) for p in paths), default=0)
    for t in range(horizon):
        for i in range(start_agent, n):
            pi = paths[i]
            li_prev = at(pi, t - 1) if t > 0 else None
            li = at(pi, t)
            for j in range(i + 1, n):
                pj = paths[j]
                lj = at(pj, t)
                if li == lj:
                    return Conflict(i, j, li, None, t)
                if t > 0:
                    lj_prev = at(pj, t - 1)
                    if li == lj_prev and lj == li_prev:
                        return Conflict(i, j, li_prev, li, t)
    return None


def find_all_conflicts(paths: Sequence[Sequence[int]]) -> List[Conflict]:
    """Every conflict in a joint plan.

    Used by the CBS heuristics (which need the whole conflict graph) and by
    bypassing (which compares conflict counts between siblings).
    """
    out: List[Conflict] = []
    n = len(paths)
    horizon = max((len(p) for p in paths), default=0)
    for t in range(horizon):
        for i in range(n):
            pi = paths[i]
            li = at(pi, t)
            li_prev = at(pi, t - 1) if t > 0 else None
            for j in range(i + 1, n):
                pj = paths[j]
                lj = at(pj, t)
                if li == lj:
                    out.append(Conflict(i, j, li, None, t))
                elif t > 0:
                    lj_prev = at(pj, t - 1)
                    if li == lj_prev and lj == li_prev:
                        out.append(Conflict(i, j, li_prev, li, t))
    return out


def count_conflicts(paths: Sequence[Sequence[int]]) -> int:
    """Number of conflicts in a joint plan. 0 means the plan is executable."""
    return len(find_all_conflicts(paths))


def is_valid_plan(paths: Sequence[Sequence[int]]) -> bool:
    """True if no two agents ever share a cell or swap across an edge."""
    return find_first_conflict(paths) is None


def validate_plan(
    paths: Sequence[Sequence[int]],
    starts: Optional[Sequence[int]] = None,
    goals: Optional[Sequence[int]] = None,
    graph=None,
) -> List[str]:
    """Full plan check: endpoints, single-step moves, and collisions.

    Returns a list of human-readable problems; an empty list means the plan is
    executable. This is deliberately independent of the solvers -- it re-derives
    everything from the paths alone, so it can catch a solver that convinced
    itself it was right.
    """
    problems: List[str] = []
    for i, p in enumerate(paths):
        if not p:
            problems.append(f"agent {i}: empty path")
            continue
        if starts is not None and p[0] != starts[i]:
            problems.append(f"agent {i}: path starts at {p[0]}, expected {starts[i]}")
        if goals is not None and p[-1] != goals[i]:
            problems.append(f"agent {i}: path ends at {p[-1]}, expected {goals[i]}")
        if graph is not None:
            for t in range(1, len(p)):
                if p[t] != p[t - 1] and p[t] not in graph.neighbours[p[t - 1]]:
                    problems.append(f"agent {i}: illegal move {p[t-1]} -> {p[t]} at t={t}")
    for c in find_all_conflicts(paths):
        problems.append(str(c))
    return problems
