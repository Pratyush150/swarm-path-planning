"""Constraints, constraint tables, and reservations.

A constraint is what the CBS high level hands down to the low level: "agent 3
may not be at vertex 91 at timestep 7". The low-level search has to respect it
exactly, including the two cases naive implementations get wrong:

* an **edge constraint** forbids a specific *move* ``u -> v`` arriving at time
  ``t``, not the vertex ``v`` itself -- the agent may still sit on ``v`` if it
  got there another way;
* a **positive constraint** (disjoint splitting) forbids everything *except* a
  particular vertex or move at that timestep, and simultaneously forbids that
  vertex to every other agent.

:class:`ConstraintTable` compiles a list of constraints, plus any fixed
reservations from already-planned agents, into the O(1) lookups the inner loop
needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class Constraint:
    """A single CBS constraint.

    Parameters
    ----------
    agent:
        Index of the agent the constraint applies to.
    loc:
        Node id the constraint is about.
    prev:
        ``None`` for a vertex constraint. For an edge constraint, the node the
        agent would be moving *from*, i.e. the constraint forbids the move
        ``prev -> loc`` that arrives at ``time``.
    time:
        Timestep the agent would be *at* ``loc``.
    positive:
        ``False`` (default) is the classic prohibition. ``True`` is a disjoint
        splitting landmark: this agent *must* be at ``loc`` (or make the move
        ``prev -> loc``) at ``time``, and no other agent may be.
    """

    agent: int
    loc: int
    prev: Optional[int]
    time: int
    positive: bool = False

    @property
    def is_edge(self) -> bool:
        """True if this constrains a move rather than a vertex."""
        return self.prev is not None

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        sign = "+" if self.positive else "-"
        if self.prev is None:
            return f"{sign}(a{self.agent} @ {self.loc} t={self.time})"
        return f"{sign}(a{self.agent} {self.prev}->{self.loc} t={self.time})"


class ConstraintTable:
    """Compiled constraints for one agent's low-level search.

    Built once per low-level call. Lookups are dict-of-set membership tests,
    which is the hot path of the whole package.
    """

    __slots__ = (
        "agent",
        "vertex",
        "edge",
        "landmark_vertex",
        "landmark_edge",
        "permanent",
        "max_time",
        "goal_release",
    )

    def __init__(self, agent: int) -> None:
        self.agent = agent
        self.vertex: Dict[int, Set[int]] = {}
        self.edge: Dict[int, Set[Tuple[int, int]]] = {}
        self.landmark_vertex: Dict[int, int] = {}
        self.landmark_edge: Dict[int, Tuple[int, int]] = {}
        # loc -> timestep from which the location is blocked forever (an agent
        # that has parked on its goal). Used by prioritised planning and by the
        # execution layer, not by CBS.
        self.permanent: Dict[int, int] = {}
        self.max_time = -1
        self.goal_release = 0

    # -- construction ----------------------------------------------------
    def add(self, c: Constraint) -> None:
        """Add one constraint, translating positives for other agents."""
        if c.positive:
            if c.agent == self.agent:
                if c.prev is None:
                    prior = self.landmark_vertex.get(c.time)
                    if prior is not None and prior != c.loc:
                        raise ValueError("contradictory positive vertex constraints")
                    self.landmark_vertex[c.time] = c.loc
                else:
                    self.landmark_edge[c.time] = (c.prev, c.loc)
                    self.landmark_vertex[c.time] = c.loc
                    self.landmark_vertex.setdefault(c.time - 1, c.prev)
            else:
                # Another agent must occupy loc at time: we must not, and we
                # must not swap across the move it is making either.
                self.vertex.setdefault(c.time, set()).add(c.loc)
                if c.prev is not None:
                    self.vertex.setdefault(c.time - 1, set()).add(c.prev)
                    self.edge.setdefault(c.time, set()).add((c.loc, c.prev))
        else:
            if c.agent != self.agent:
                return
            if c.prev is None:
                self.vertex.setdefault(c.time, set()).add(c.loc)
            else:
                self.edge.setdefault(c.time, set()).add((c.prev, c.loc))
        self.max_time = max(self.max_time, c.time)

    def add_all(self, constraints: Iterable[Constraint]) -> None:
        """Add many constraints."""
        for c in constraints:
            self.add(c)

    def reserve_path(self, path: Sequence[int], permanent_from_end: bool = True) -> None:
        """Reserve another agent's whole path as a moving obstacle.

        This is what prioritised planning does instead of branching: higher
        priority agents are baked into the table as hard reservations, including
        the fact that an agent parked on its goal blocks that cell forever.
        """
        for t, loc in enumerate(path):
            self.vertex.setdefault(t, set()).add(loc)
            if t > 0:
                # Forbid the head-on swap: we may not move loc -> path[t-1]
                # while they move path[t-1] -> loc.
                self.edge.setdefault(t, set()).add((loc, path[t - 1]))
            self.max_time = max(self.max_time, t)
        if permanent_from_end and path:
            end = path[-1]
            first = len(path) - 1
            if end not in self.permanent or self.permanent[end] > first:
                self.permanent[end] = first

    def set_goal(self, goal: int) -> None:
        """Record the goal so the search knows when it may stop there.

        An agent may not stop on its goal while a later constraint still forbids
        that cell: it has to leave and come back. ``goal_release`` is the first
        timestep from which parking on the goal is legal forever.
        """
        release = 0
        for t, locs in self.vertex.items():
            if goal in locs:
                release = max(release, t + 1)
        for t, loc in self.landmark_vertex.items():
            if loc != goal:
                release = max(release, t + 1)
        if goal in self.permanent:
            release = max(release, self.permanent[goal] + 1)
            # A parked higher-priority agent makes the goal permanently
            # unusable; the caller has to detect the resulting failure.
        self.goal_release = release

    # -- queries ---------------------------------------------------------
    def blocked(self, loc: int, t: int) -> bool:
        """True if the agent may not occupy ``loc`` at timestep ``t``."""
        s = self.vertex.get(t)
        if s is not None and loc in s:
            return True
        lm = self.landmark_vertex.get(t)
        if lm is not None and lm != loc:
            return True
        p = self.permanent.get(loc)
        if p is not None and t >= p:
            return True
        return False

    def blocked_move(self, frm: int, to: int, t: int) -> bool:
        """True if the move ``frm -> to`` arriving at ``t`` is forbidden."""
        s = self.edge.get(t)
        if s is not None and (frm, to) in s:
            return True
        lm = self.landmark_edge.get(t)
        if lm is not None and lm != (frm, to):
            return True
        return False

    def blocked_forever(self, loc: int) -> bool:
        """True if ``loc`` is unusable at every timestep (a parked agent sits there)."""
        p = self.permanent.get(loc)
        return p is not None and p <= 0

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        nv = sum(len(v) for v in self.vertex.values())
        ne = sum(len(v) for v in self.edge.values())
        return (
            f"<ConstraintTable a{self.agent} vertex={nv} edge={ne} "
            f"landmarks={len(self.landmark_vertex)} max_t={self.max_time}>"
        )


def build_table(
    agent: int,
    constraints: Iterable[Constraint],
    goal: Optional[int] = None,
) -> ConstraintTable:
    """Convenience: compile ``constraints`` into a table for ``agent``."""
    table = ConstraintTable(agent)
    table.add_all(constraints)
    if goal is not None:
        table.set_goal(goal)
    return table


def constraints_for(constraints: Iterable[Constraint], agent: int) -> List[Constraint]:
    """The subset of ``constraints`` that affect ``agent``.

    Negative constraints affect only their own agent; positive constraints
    affect every agent (their own by compulsion, the others by exclusion).
    """
    return [c for c in constraints if c.agent == agent or c.positive]
