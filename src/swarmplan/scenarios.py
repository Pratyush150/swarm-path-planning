"""Reading the movingai ``.scen`` benchmark scenario files.

A scenario file is a list of (start, goal) pairs for one map, in a fixed order.
The MAPF convention (Stern et al., "Multi-Agent Pathfinding: Definitions,
Variants, and Benchmarks", SoCS 2019) is that an instance with *k* agents is the
**first k lines** of the file, so every paper that quotes "success rate at 60
agents on random-32-32-20" is talking about the same instances. We follow that
convention exactly, which is the only reason our numbers can be compared with
anyone else's.

Two things about the last column are worth stating plainly, because they are a
common source of wrong "optimality ratio" numbers:

* It is the **8-connected octile** optimal distance (diagonal moves allowed, at
  cost sqrt(2)), inherited from Sturtevant's grid-pathfinding benchmark.
* MAPF is normally solved **4-connected**. A 4-connected path is also a legal
  8-connected path of the same cost, so the octile value is a valid lower bound
  on the 4-connected single-agent optimum -- but a loose one, typically 15-30%
  below it on open maps.

So :func:`swarmplan.metrics` reports the ratio against the sum of true
4-connected single-agent distances (computed by backward Dijkstra, tight and
provably a lower bound on sum-of-costs) and carries the octile column alongside
for reference rather than pretending they are the same number.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .graph import GridMap


@dataclass(frozen=True)
class ScenarioEntry:
    """One (start, goal) pair from a ``.scen`` file.

    Coordinates are stored as ``(row, col)``, already swapped from the
    ``(col, row)`` order used on disk.
    """

    bucket: int
    map_name: str
    width: int
    height: int
    start: Tuple[int, int]
    goal: Tuple[int, int]
    octile_optimal: float


@dataclass(frozen=True)
class Scenario:
    """A parsed ``.scen`` file."""

    name: str
    map_name: str
    entries: List[ScenarioEntry]

    def __len__(self) -> int:
        return len(self.entries)

    def take(self, n_agents: int) -> "Scenario":
        """The first ``n_agents`` entries, the standard way to size an instance."""
        if n_agents > len(self.entries):
            raise ValueError(
                f"scenario {self.name} has {len(self.entries)} entries, asked for {n_agents}"
            )
        return Scenario(self.name, self.map_name, self.entries[:n_agents])


def parse_scen(text: str, name: str = "scen") -> Scenario:
    """Parse the movingai ``.scen`` text format (version 1)."""
    entries: List[ScenarioEntry] = []
    map_name = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.lower().startswith("version"):
            continue
        parts = line.split()
        if len(parts) < 9:
            raise ValueError(f"malformed .scen line: {raw!r}")
        bucket = int(parts[0])
        map_name = parts[1]
        width, height = int(parts[2]), int(parts[3])
        sx, sy, gx, gy = (int(v) for v in parts[4:8])
        entries.append(
            ScenarioEntry(
                bucket=bucket,
                map_name=map_name,
                width=width,
                height=height,
                start=(sy, sx),
                goal=(gy, gx),
                octile_optimal=float(parts[8]),
            )
        )
    return Scenario(name=name, map_name=map_name, entries=entries)


def load_scen(path) -> Scenario:
    """Load a ``.scen`` file from disk."""
    path = Path(path)
    return parse_scen(path.read_text(), name=path.stem)


@dataclass
class MapfInstance:
    """A concrete MAPF problem: one graph, k starts, k goals.

    ``starts`` and ``goals`` are node ids in ``graph``. ``octile_lower_bound``
    is the sum of the ``.scen`` file's last column over the selected agents,
    kept for reference only (see the module docstring).
    """

    name: str
    grid: GridMap
    starts: List[int]
    goals: List[int]
    octile_lower_bound: Optional[float] = None

    @property
    def graph(self):
        """The :class:`~swarmplan.graph.SearchGraph` the planners search."""
        return self.grid.graph

    @property
    def n_agents(self) -> int:
        """Number of agents in this instance."""
        return len(self.starts)

    def coords(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Start and goal coordinates as ``(row, col)`` lists."""
        return (
            [self.grid.rc(s) for s in self.starts],
            [self.grid.rc(g) for g in self.goals],
        )


def make_instance(
    grid: GridMap, scen: Scenario, n_agents: int, name: Optional[str] = None
) -> MapfInstance:
    """Build a :class:`MapfInstance` from the first ``n_agents`` scenario entries."""
    sub = scen.take(n_agents)
    starts, goals = [], []
    for e in sub.entries:
        if not grid.passable(e.start) or not grid.passable(e.goal):
            raise ValueError(
                f"scenario {scen.name} references a blocked cell on map {grid.name}: "
                f"{e.start} -> {e.goal}"
            )
        starts.append(grid.node(e.start))
        goals.append(grid.node(e.goal))
    lb = sum(e.octile_optimal for e in sub.entries)
    return MapfInstance(
        name=name or f"{scen.name}-{n_agents}",
        grid=grid,
        starts=starts,
        goals=goals,
        octile_lower_bound=lb,
    )


def instance_from_coords(
    grid: GridMap,
    starts: Sequence[Tuple[int, int]],
    goals: Sequence[Tuple[int, int]],
    name: str = "instance",
) -> MapfInstance:
    """Build an instance directly from ``(row, col)`` coordinate lists."""
    if len(starts) != len(goals):
        raise ValueError("starts and goals must have the same length")
    return MapfInstance(
        name=name,
        grid=grid,
        starts=[grid.node(s) for s in starts],
        goals=[grid.node(g) for g in goals],
    )
