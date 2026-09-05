"""From grid steps to something an airframe can actually fly.

A MAPF plan says "cell (14, 7) at t=9, cell (14, 8) at t=10". A quadrotor cannot
fly that: it is a sequence of instantaneous jumps between cell centres with
right-angle corners and infinite acceleration at every one of them.

This module turns the discrete plan into a sampled continuous trajectory that
respects a velocity and an acceleration limit:

1. **Corner smoothing.** Waypoints are averaged with their neighbours, a few
   passes, with a hard cap on how far any waypoint may move from its grid cell.
   The cap is what keeps the smoothed path inside the corridor the planner
   cleared -- shortcut a path freely and you have quietly discarded the
   collision guarantee you paid CBS for.
2. **Catmull-Rom interpolation** through the smoothed waypoints, so position is
   continuous and velocity is continuous at the waypoints.
3. **Uniform time scaling.** Sample once in normalised time, measure the peak
   speed and acceleration, then stretch the time axis by the single factor that
   brings both under their limits. Stretching time by ``s`` divides speed by
   ``s`` and acceleration by ``s^2``, so the correct factor is exact and one
   pass is enough -- no iteration, no search.

The result is deliberately *conservative*: every agent is slowed by the same
factor, so the relative ordering of the plan is untouched and the separation
guarantee survives. A time-optimal per-segment retiming would be faster and
would need its own safety argument; that is called out in the README's
limitations rather than half-done here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..graph import SearchGraph


@dataclass
class ContinuousPlan:
    """A time-sampled continuous trajectory for every agent.

    ``positions`` has shape ``(n_agents, n_samples, ndim)`` in metres and
    ``times`` has shape ``(n_samples,)`` in seconds.
    """

    times: np.ndarray
    positions: np.ndarray
    v_max: float
    a_max: float
    time_scale: float
    cell_size: float

    @property
    def n_agents(self) -> int:
        """Number of agents in the plan."""
        return self.positions.shape[0]

    @property
    def duration(self) -> float:
        """Wall-clock length of the trajectory, in seconds."""
        return float(self.times[-1]) if len(self.times) else 0.0

    def velocities(self) -> np.ndarray:
        """Finite-difference velocities, shape ``(n_agents, n_samples-1, ndim)``."""
        dt = np.diff(self.times)[None, :, None]
        return np.diff(self.positions, axis=1) / dt

    def accelerations(self) -> np.ndarray:
        """Finite-difference accelerations, shape ``(n_agents, n_samples-2, ndim)``."""
        v = self.velocities()
        dt = np.diff(self.times)[None, :-1, None]
        return np.diff(v, axis=1) / dt

    def peak_speed(self) -> float:
        """Largest speed of any agent at any sample, in m/s."""
        v = self.velocities()
        return float(np.linalg.norm(v, axis=2).max()) if v.size else 0.0

    def peak_acceleration(self) -> float:
        """Largest acceleration magnitude of any agent, in m/s^2."""
        a = self.accelerations()
        return float(np.linalg.norm(a, axis=2).max()) if a.size else 0.0

    def within_limits(self, tol: float = 1e-6) -> bool:
        """True if the trajectory respects both kinematic limits."""
        return (
            self.peak_speed() <= self.v_max + tol
            and self.peak_acceleration() <= self.a_max + tol
        )


def path_coordinates(
    graph: SearchGraph, path: Sequence[int], cell_size: float = 1.0
) -> np.ndarray:
    """Grid path as an ``(n_steps, ndim)`` array of metric coordinates."""
    return np.asarray([graph.coord(v) for v in path], dtype=np.float64) * cell_size


def smooth_waypoints(
    waypoints: np.ndarray, passes: int = 2, max_deviation: float = 0.35
) -> np.ndarray:
    """Round off the corners without leaving the cleared corridor.

    Endpoints are pinned (the agent must actually start and finish where the
    plan says) and every interior waypoint is clamped to ``max_deviation``
    metres of its original position.
    """
    if passes <= 0 or len(waypoints) < 3:
        return waypoints.copy()
    original = waypoints
    out = waypoints.copy()
    for _ in range(passes):
        smoothed = out.copy()
        smoothed[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        delta = smoothed - original
        dist = np.linalg.norm(delta, axis=1, keepdims=True)
        scale = np.where(dist > max_deviation, max_deviation / np.maximum(dist, 1e-12), 1.0)
        out = original + delta * scale
        out[0] = original[0]
        out[-1] = original[-1]
    return out


def _catmull_rom(points: np.ndarray, samples_per_segment: int) -> np.ndarray:
    """Catmull-Rom spline through ``points``, sampled uniformly per segment."""
    n = len(points)
    if n == 1:
        return points.copy()
    if n == 2:
        s = np.linspace(0.0, 1.0, samples_per_segment + 1)[:, None]
        return points[0][None, :] * (1 - s) + points[1][None, :] * s
    padded = np.vstack([points[0], points, points[-1]])
    out = []
    s = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)[:, None]
    s2, s3 = s * s, s * s * s
    for i in range(n - 1):
        p0, p1, p2, p3 = padded[i], padded[i + 1], padded[i + 2], padded[i + 3]
        out.append(
            0.5
            * (
                (2 * p1)
                + (-p0 + p2) * s
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * s3
            )
        )
    out.append(points[-1][None, :])
    return np.vstack(out)


def smooth_plan(
    graph: SearchGraph,
    paths: Sequence[Sequence[int]],
    cell_size: float = 1.0,
    v_max: float = 2.0,
    a_max: float = 2.0,
    samples_per_step: int = 8,
    smoothing_passes: int = 2,
    max_deviation: float = 0.35,
) -> ContinuousPlan:
    """Turn a grid plan into a kinematically feasible sampled trajectory.

    All agents share one time axis and one scaling factor, so their relative
    timing -- and therefore every ordering the planner established -- is
    preserved exactly.
    """
    if v_max <= 0 or a_max <= 0:
        raise ValueError("v_max and a_max must be positive")
    if not paths:
        raise ValueError("no paths to smooth")
    horizon = max(len(p) for p in paths)
    tracks = []
    for path in paths:
        padded = list(path) + [path[-1]] * (horizon - len(path))
        wp = path_coordinates(graph, padded, cell_size)
        wp = smooth_waypoints(wp, smoothing_passes, max_deviation * cell_size)
        tracks.append(_catmull_rom(wp, samples_per_step))
    positions = np.stack(tracks, axis=0)
    n_samples = positions.shape[1]
    # Nominal timing: one second per grid step, then rescaled below.
    times = np.linspace(0.0, float(horizon - 1), n_samples)
    if n_samples < 3:
        return ContinuousPlan(times, positions, v_max, a_max, 1.0, cell_size)

    dt = times[1] - times[0]
    vel = np.diff(positions, axis=1) / dt
    acc = np.diff(vel, axis=1) / dt
    peak_v = float(np.linalg.norm(vel, axis=2).max())
    peak_a = float(np.linalg.norm(acc, axis=2).max()) if acc.size else 0.0
    scale = max(1e-9, peak_v / v_max, float(np.sqrt(peak_a / a_max)) if peak_a > 0 else 0.0)
    scale = max(scale, 1e-9)
    return ContinuousPlan(times * scale, positions, v_max, a_max, scale, cell_size)
