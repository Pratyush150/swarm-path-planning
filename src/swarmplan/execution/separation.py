"""Minimum separation in continuous time, not just on grid cells.

Grid conflict checking answers "were two agents ever in the same cell at the
same timestep?". That is not the question a safety case asks. Two quadrotors on
*adjacent* cells passing each other in opposite directions are, halfway through
the step, a cell-width apart and closing -- perfectly legal on the grid, and
much too close if a cell is 1 m and the rotor tips need 2 m.

So the separation check here runs on the continuous trajectories, and it is
exact rather than sampled. Between two consecutive samples both agents move
linearly, so the relative position is linear in the interpolation parameter and
the minimum distance over the interval has a closed form: minimise the quadratic
``|d0 + s*(d1 - d0)|^2`` over ``s`` in ``[0, 1]``. Sampling the distance at the
sample points alone would miss exactly the case that matters -- the closest
approach in the middle of a crossing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .smoothing import ContinuousPlan


@dataclass(frozen=True)
class SeparationViolation:
    """Two agents closer than the required minimum, and when."""

    agent_a: int
    agent_b: int
    time: float
    distance: float

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"agents {self.agent_a}/{self.agent_b} at t={self.time:.2f}s "
            f"separated by {self.distance:.3f} m"
        )


def _segment_min_distance(d0: np.ndarray, d1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Exact minimum of ``|d0 + s(d1-d0)|`` over ``s`` in ``[0,1]``, vectorised.

    Returns ``(distance, s)`` for every row of the inputs.
    """
    delta = d1 - d0
    denom = np.einsum("ij,ij->i", delta, delta)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(denom > 0, -np.einsum("ij,ij->i", d0, delta) / np.maximum(denom, 1e-18), 0.0)
    s = np.clip(s, 0.0, 1.0)
    closest = d0 + s[:, None] * delta
    return np.linalg.norm(closest, axis=1), s


def pairwise_min_separation(plan: ContinuousPlan) -> np.ndarray:
    """``(n_agents, n_agents)`` matrix of closest approach distances, in metres.

    The diagonal is ``inf``. Costs O(n^2 * samples), which is fine for a few
    hundred agents and is not intended for thousands.
    """
    n = plan.n_agents
    out = np.full((n, n), np.inf)
    pos = plan.positions
    for a in range(n):
        for b in range(a + 1, n):
            d = pos[a] - pos[b]
            dist, _ = _segment_min_distance(d[:-1], d[1:])
            m = float(dist.min()) if dist.size else float(np.linalg.norm(d[0]))
            out[a, b] = out[b, a] = m
    return out


def separation_violations(
    plan: ContinuousPlan, min_distance: float, ignore_shared_endpoints: bool = False
) -> List[SeparationViolation]:
    """Every pair that comes closer than ``min_distance``, with the time it happens."""
    n = plan.n_agents
    pos = plan.positions
    times = plan.times
    out: List[SeparationViolation] = []
    for a in range(n):
        for b in range(a + 1, n):
            d = pos[a] - pos[b]
            if d.shape[0] < 2:
                continue
            dist, s = _segment_min_distance(d[:-1], d[1:])
            k = int(np.argmin(dist))
            if dist[k] < min_distance:
                if ignore_shared_endpoints and np.allclose(pos[a][-1], pos[b][-1]):
                    continue
                t = float(times[k] + s[k] * (times[k + 1] - times[k]))
                out.append(SeparationViolation(a, b, t, float(dist[k])))
    return out


def min_separation(plan: ContinuousPlan) -> float:
    """Closest approach between any two agents over the whole flight, in metres."""
    if plan.n_agents < 2:
        return float("inf")
    return float(pairwise_min_separation(plan).min())


def separation_report(plan: ContinuousPlan, min_distance: float) -> str:
    """One-paragraph summary suitable for printing after a run."""
    worst = min_separation(plan)
    bad = separation_violations(plan, min_distance)
    lines = [
        f"agents: {plan.n_agents}",
        f"duration: {plan.duration:.2f} s (time scale {plan.time_scale:.2f})",
        f"peak speed: {plan.peak_speed():.2f} m/s (limit {plan.v_max:.2f})",
        f"peak acceleration: {plan.peak_acceleration():.2f} m/s^2 (limit {plan.a_max:.2f})",
        f"closest approach: {worst:.3f} m (required {min_distance:.3f})",
        f"violations: {len(bad)}",
    ]
    for v in bad[:5]:
        lines.append(f"  {v}")
    return "\n".join(lines)
