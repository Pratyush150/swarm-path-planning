"""The drone-light-show case: formations, assignment, and the morph between them.

A show is a sequence of formations. Each transition is an **unlabelled** MAPF
instance: the choreography fixes the set of occupied positions and says nothing
about which aircraft goes where. Two decisions have to be made, in this order:

1. **Assignment.** Which drone flies to which slot. Solved as a minimum-cost
   bipartite matching over true obstacle-aware distances
   (:mod:`swarmplan.assignment`). Getting this wrong is the single most
   expensive mistake in the problem -- an arbitrary assignment has drones
   crossing the whole formation past each other, which is both slow and a
   conflict-generating machine.
2. **Routing.** Given the assignment, plan collision-free paths. That is
   ordinary labelled MAPF, so ECBS does it.

The airspace is a 3D grid, 6-connected: drones can climb and use depth to get
out of each other's way, which is exactly what a real show does behind the
façade the audience sees. The formations themselves are built in the x-z plane
(the plane facing the audience) at a fixed depth.

Everything here is discrete. Real shows also have to respect downwash
separation, battery state, radio scheduling and geofences; none of that is
modelled, and the README says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .assignment.anonymous import AssignmentResult, assign_goals, identity_assignment
from .assignment.formations import annulus_points, grid_points, text_points, text_width
from .graph import Grid3D
from .lowlevel.heuristic import HeuristicCache
from .planners import solve
from .solution import Solution


@dataclass
class ShowTransition:
    """One formation-to-formation morph: how it was assigned and how it flew."""

    index: int
    assignment: AssignmentResult
    solution: Solution
    identity_total: float = 0.0
    identity_max: float = 0.0

    @property
    def improvement(self) -> Optional[float]:
        """Ratio of arbitrary-assignment travel to optimal-assignment travel."""
        if self.assignment.total_distance <= 0:
            return None
        return self.identity_total / self.assignment.total_distance


@dataclass
class Show:
    """A planned show: the airspace, the formations, and the concatenated plan."""

    space: Grid3D
    formations: List[List[int]]
    transitions: List[ShowTransition] = field(default_factory=list)
    paths: List[List[int]] = field(default_factory=list)
    formation_times: List[int] = field(default_factory=list)

    @property
    def n_agents(self) -> int:
        """Number of aircraft in the show."""
        return len(self.paths)

    @property
    def in_formation_at(self) -> List[int]:
        """Timesteps at which the swarm is in each of its formations."""
        return list(self.formation_times)

    @property
    def makespan(self) -> int:
        """Total show length in timesteps."""
        return max((len(p) - 1 for p in self.paths), default=0)

    @property
    def solved(self) -> bool:
        """True if every transition was planned successfully."""
        return bool(self.transitions) and all(t.solution.solved for t in self.transitions)


def build_airspace(size: Tuple[int, int, int] = (34, 7, 16)) -> Grid3D:
    """An empty box of airspace, 6-connected. ``size`` is ``(x, y, z)``."""
    return Grid3D(size, name="airspace")


def ring_formation(
    space: Grid3D, count: int, radius: float = 6.0, depth: Optional[int] = None
) -> List[int]:
    """A ring of ``count`` slots facing the audience, as node ids.

    More slots than one circle holds turns the formation into concentric rings,
    which is what a real show does with a large fleet anyway.
    """
    cx, cy, cz = space.size[0] // 2, space.size[1] // 2, space.size[2] // 2
    y = cy if depth is None else depth
    pts = annulus_points(count, (cx, y, cz), radius, depth=y)
    return [space.node(p) for p in pts]


def text_formation(
    space: Grid3D, text: str, count: Optional[int] = None, depth: Optional[int] = None
) -> List[int]:
    """A text formation, centred in the airspace, as node ids.

    If ``count`` is given the glyph cells are thinned to exactly that many slots,
    so a show can hold its aircraft count constant across formations of
    different natural sizes.
    """
    from .assignment.formations import resample

    width = text_width(text)
    x0 = max(0, (space.size[0] - width) // 2)
    y = space.size[1] // 2 if depth is None else depth
    z0 = max(0, (space.size[2] - 7) // 2)
    pts = text_points(text, origin=(x0, y, z0), depth=y)
    pts = [p for p in pts if all(0 <= c < s for c, s in zip(p, space.size))]
    if count is not None:
        pts = resample(pts, count)
    return [space.node(p) for p in pts]


def block_formation(space: Grid3D, count: int, width: int = 12) -> List[int]:
    """A compact rectangular block of ``count`` slots, as node ids."""
    x0 = max(0, (space.size[0] - width) // 2)
    y = space.size[1] // 2
    pts = grid_points(count, (x0, y, 1), width, depth=1)
    return [space.node(p) for p in pts]


def plan_show(
    space: Grid3D,
    formations: Sequence[Sequence[int]],
    algorithm: str = "ecbs:w=1.1",
    objective: str = "sum",
    time_limit: float = 60.0,
    cache: Optional[HeuristicCache] = None,
) -> Show:
    """Assign and route every transition, and concatenate them into one plan.

    ``formations[0]`` is where the aircraft start. Each subsequent formation is
    assigned against the positions the swarm actually reached, so an aircraft's
    slot in transition 2 depends on where the assignment put it in transition 1
    -- which is how a real show accumulates.
    """
    if len(formations) < 2:
        raise ValueError("a show needs at least two formations")
    counts = {len(f) for f in formations}
    if len(counts) != 1:
        raise ValueError(f"every formation needs the same number of slots, got {sorted(counts)}")

    cache = cache or HeuristicCache(space.graph)
    show = Show(space=space, formations=[list(f) for f in formations])
    show.formation_times = [0]
    current = list(formations[0])
    combined: List[List[int]] = [[v] for v in current]

    for i, target in enumerate(formations[1:]):
        assignment = assign_goals(
            space.graph, current, list(target), objective=objective, cache=cache
        )
        identity = identity_assignment(space.graph, current, list(target), cache=cache)
        goals = assignment.apply(list(target))
        result = solve(algorithm, space.graph, current, goals, time_limit=time_limit, cache=cache)
        show.transitions.append(
            ShowTransition(
                index=i,
                assignment=assignment,
                solution=result,
                identity_total=identity.total_distance,
                identity_max=identity.max_distance,
            )
        )
        if not result.solved:
            show.paths = combined
            return show
        horizon = max(len(p) for p in result.paths)
        for a, p in enumerate(result.paths):
            padded = list(p) + [p[-1]] * (horizon - len(p))
            combined[a].extend(padded[1:])
        show.formation_times.append(show.formation_times[-1] + horizon - 1)
        current = [p[-1] for p in result.paths]

    show.paths = combined
    return show


def default_show(
    n_agents: int = 80,
    text: str = "SWARM",
    size: Tuple[int, int, int] = (36, 7, 18),
    algorithm: str = "ecbs:w=1.1",
    objective: str = "sum",
    time_limit: float = 120.0,
) -> Show:
    """The demonstration used in the README: launch block, then ring, then text.

    Each transition stresses something different. Block to ring is a spread --
    everyone moves outwards and the traffic is mostly radial. Ring to text is a
    dense rearrangement where most of the fleet has to cross the middle, which
    is where the assignment earns its keep and where the conflicts are.
    """
    space = build_airspace(size)
    start = block_formation(space, n_agents, width=min(size[0] - 2, 16))
    ring = ring_formation(space, n_agents, radius=min(size[0], size[2]) / 2.6)
    letters = text_formation(space, text, count=n_agents)
    return plan_show(space, [start, ring, letters], algorithm, objective, time_limit)
