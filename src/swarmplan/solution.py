"""The object every solver returns.

One shape for CBS, ECBS and prioritised planning, so the benchmark harness can
treat them interchangeably and so a failure carries as much information as a
success -- ``status`` says *why* there is no plan, which is the difference
between "this instance is unsolvable" and "we ran out of time", and those two
must never be averaged together in a success-rate table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .conflicts import find_all_conflicts

#: Solved: ``paths`` is a valid, conflict-free joint plan.
SOLVED = "solved"
#: The deadline passed before the search finished. Says nothing about the
#: instance -- a longer budget may well solve it.
TIMEOUT = "timeout"
#: Proven to have no solution (the search space was exhausted).
UNSOLVABLE = "unsolvable"
#: A node or expansion budget was exhausted. Same caveat as ``TIMEOUT``.
BUDGET = "budget"
#: This planner could not find a plan. Says nothing about the instance -- an
#: incomplete planner (prioritised planning) fails on solvable instances by
#: construction, so its failures must never be recorded as ``UNSOLVABLE``.
FAILED = "failed"


@dataclass
class Solution:
    """Result of one solver run on one instance."""

    paths: Optional[List[List[int]]] = None
    status: str = TIMEOUT
    algorithm: str = ""
    runtime: float = 0.0
    high_level_expanded: int = 0
    high_level_generated: int = 0
    low_level_expanded: int = 0
    low_level_calls: int = 0
    lower_bound: int = -1
    suboptimality_bound: float = 1.0
    notes: Dict[str, Any] = field(default_factory=dict)

    @property
    def solved(self) -> bool:
        """True if a joint plan was produced."""
        return self.status == SOLVED and self.paths is not None

    @property
    def cost(self) -> int:
        """Sum-of-costs: the total number of timesteps agents spend moving."""
        if not self.paths:
            return -1
        return sum(len(p) - 1 for p in self.paths)

    @property
    def makespan(self) -> int:
        """Timestep at which the last agent arrives."""
        if not self.paths:
            return -1
        return max(len(p) - 1 for p in self.paths)

    @property
    def n_agents(self) -> int:
        """Number of agents in the plan."""
        return len(self.paths) if self.paths else 0

    def conflicts(self) -> List:
        """Conflicts remaining in the returned plan. Should always be empty."""
        return find_all_conflicts(self.paths) if self.paths else []

    def summary(self) -> str:
        """One-line human-readable summary."""
        if not self.solved:
            return f"{self.algorithm}: {self.status} after {self.runtime:.2f}s"
        return (
            f"{self.algorithm}: soc={self.cost} makespan={self.makespan} "
            f"nodes={self.high_level_expanded} in {self.runtime:.2f}s"
        )
