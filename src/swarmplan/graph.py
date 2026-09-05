"""Search graphs: the movement model every planner in this package sits on.

Everything above this module -- space-time A*, CBS, ECBS, prioritised planning,
the assignment layer -- talks to a :class:`SearchGraph` and never to a grid
directly. A location is an ``int`` node id, not a coordinate tuple, because the
inner loops of a multi-agent search do tens of millions of set and dict
operations on locations and interning them as small integers is worth roughly a
factor of three in Python.

Two concrete graphs are provided:

``GridMap``
    A 4-connected 2D occupancy grid, parsed from the ``.map`` files of the
    Sturtevant / movingai.com benchmark set.

``Grid3D``
    A 4- or 6-connected 3D box, used for the drone-light-show demonstrations
    where the swarm has altitude and depth to route through.

Both are just adjacency lists plus a coordinate mapping, so anything that can
produce those (a roadmap, a lattice, a warehouse aisle graph) can be planned on
without touching the planners.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Characters the movingai benchmark format uses for traversable cells. The MAPF
# subset only ever contains '.' and '@', but the parser accepts the full octile
# alphabet so it can also read the wider grid-pathfinding benchmark set.
PASSABLE_CHARS = frozenset(".G S")
BLOCKED_CHARS = frozenset("@OTW")


class SearchGraph:
    """An undirected unit-cost graph with named coordinates.

    Attributes
    ----------
    n_nodes:
        Number of traversable nodes. Node ids are ``0 .. n_nodes - 1``.
    neighbours:
        ``neighbours[v]`` is a tuple of node ids adjacent to ``v``. Waiting is
        *not* included here; the planners add the wait action themselves,
        because a wait is a different kind of action (it can be forbidden by a
        vertex constraint but never by an edge constraint).
    """

    __slots__ = ("n_nodes", "neighbours", "_coords", "_index", "dims", "name")

    def __init__(
        self,
        coords: Sequence[Tuple[int, ...]],
        neighbours: Sequence[Sequence[int]],
        dims: Tuple[int, ...],
        name: str = "graph",
    ) -> None:
        self.n_nodes = len(coords)
        self._coords = list(coords)
        self.neighbours: List[Tuple[int, ...]] = [tuple(n) for n in neighbours]
        self._index = {c: i for i, c in enumerate(self._coords)}
        self.dims = tuple(dims)
        self.name = name

    def coord(self, node: int) -> Tuple[int, ...]:
        """Coordinate tuple of a node id."""
        return self._coords[node]

    def index(self, coord: Tuple[int, ...]) -> int:
        """Node id of a coordinate tuple. Raises ``KeyError`` if blocked."""
        return self._index[tuple(coord)]

    def has(self, coord: Tuple[int, ...]) -> bool:
        """True if the coordinate is a traversable node of this graph."""
        return tuple(coord) in self._index

    def coords(self) -> List[Tuple[int, ...]]:
        """All node coordinates, indexed by node id."""
        return list(self._coords)

    def coord_array(self) -> np.ndarray:
        """``(n_nodes, ndim)`` integer array of node coordinates."""
        return np.asarray(self._coords, dtype=np.int32)

    def manhattan(self, a: int, b: int) -> int:
        """Manhattan distance between two nodes, ignoring obstacles."""
        ca, cb = self._coords[a], self._coords[b]
        return int(sum(abs(x - y) for x, y in zip(ca, cb)))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<SearchGraph {self.name} nodes={self.n_nodes} dims={self.dims}>"


def _adjacency(passable: np.ndarray, offsets: Sequence[Tuple[int, ...]]) -> SearchGraph:
    """Build a :class:`SearchGraph` from a boolean passability array."""
    shape = passable.shape
    ids = np.full(shape, -1, dtype=np.int64)
    idx = np.argwhere(passable)
    for i, cell in enumerate(idx):
        ids[tuple(cell)] = i
    coords = [tuple(int(v) for v in cell) for cell in idx]
    neighbours: List[Tuple[int, ...]] = []
    for cell in coords:
        adj = []
        for off in offsets:
            nb = tuple(c + o for c, o in zip(cell, off))
            if all(0 <= v < s for v, s in zip(nb, shape)) and passable[nb]:
                adj.append(int(ids[nb]))
        neighbours.append(tuple(adj))
    return SearchGraph(coords, neighbours, shape)


class GridMap:
    """A 2D 4-connected occupancy grid parsed from a movingai ``.map`` file.

    Coordinates are ``(row, col)``. The benchmark ``.scen`` files store
    ``(col, row)``; :mod:`swarmplan.scenarios` does that swap once, at parse
    time, so nothing else in the package has to remember it.
    """

    def __init__(self, blocked: np.ndarray, name: str = "grid") -> None:
        if blocked.ndim != 2:
            raise ValueError("a GridMap needs a 2D occupancy array")
        self.blocked = np.asarray(blocked, dtype=bool)
        self.height, self.width = self.blocked.shape
        self.name = name
        self.graph = _adjacency(~self.blocked, ((-1, 0), (1, 0), (0, -1), (0, 1)))
        self.graph.name = name

    # -- construction ----------------------------------------------------
    @classmethod
    def from_text(cls, text: str, name: str = "grid") -> "GridMap":
        """Parse the movingai octile ``.map`` text format."""
        lines = text.splitlines()
        height = width = None
        body_start = None
        for i, line in enumerate(lines[:8]):
            low = line.strip().lower()
            if low.startswith("height"):
                height = int(low.split()[1])
            elif low.startswith("width"):
                width = int(low.split()[1])
            elif low == "map":
                body_start = i + 1
                break
        if height is None or width is None or body_start is None:
            raise ValueError("not a movingai .map file: missing height/width/map header")
        rows = lines[body_start : body_start + height]
        if len(rows) != height:
            raise ValueError(f"expected {height} map rows, found {len(rows)}")
        blocked = np.ones((height, width), dtype=bool)
        for r, row in enumerate(rows):
            if len(row) < width:
                row = row.ljust(width, "@")
            for c in range(width):
                ch = row[c]
                if ch in PASSABLE_CHARS:
                    blocked[r, c] = False
                elif ch in BLOCKED_CHARS:
                    blocked[r, c] = True
                else:
                    raise ValueError(f"unknown map character {ch!r} at row {r} col {c}")
        return cls(blocked, name=name)

    @classmethod
    def from_file(cls, path) -> "GridMap":
        """Parse a ``.map`` file from disk. The stem becomes the map name."""
        path = Path(path)
        return cls.from_text(path.read_text(), name=path.stem)

    @classmethod
    def from_rows(cls, rows: Iterable[str], name: str = "grid") -> "GridMap":
        """Build from plain ASCII rows (``.`` free, anything else blocked).

        This is what the unit tests use to write hand-checkable maps inline.
        """
        rows = [r for r in rows]
        width = max(len(r) for r in rows)
        blocked = np.ones((len(rows), width), dtype=bool)
        for r, row in enumerate(rows):
            for c, ch in enumerate(row):
                blocked[r, c] = ch != "."
        return cls(blocked, name=name)

    # -- queries ---------------------------------------------------------
    @property
    def n_free(self) -> int:
        """Number of traversable cells."""
        return self.graph.n_nodes

    def passable(self, rc: Tuple[int, int]) -> bool:
        """True if ``(row, col)`` is inside the map and free."""
        r, c = rc
        return 0 <= r < self.height and 0 <= c < self.width and not self.blocked[r, c]

    def node(self, rc: Tuple[int, int]) -> int:
        """Node id of ``(row, col)``."""
        return self.graph.index(rc)

    def rc(self, node: int) -> Tuple[int, int]:
        """``(row, col)`` of a node id."""
        return self.graph.coord(node)  # type: ignore[return-value]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<GridMap {self.name} {self.height}x{self.width} free={self.n_free}>"


class Grid3D:
    """A 3D box graph, 6-connected by default. Coordinates are ``(x, y, z)``.

    Used for the light-show demonstrations: a swarm morphing between formations
    has the whole airspace to route through, and the depth and altitude axes are
    where most of the conflict resolution actually happens.
    """

    def __init__(
        self,
        size: Tuple[int, int, int],
        blocked: Optional[np.ndarray] = None,
        connectivity: int = 6,
        name: str = "airspace",
    ) -> None:
        self.size = tuple(int(v) for v in size)
        if blocked is None:
            blocked = np.zeros(self.size, dtype=bool)
        self.blocked = np.asarray(blocked, dtype=bool)
        if self.blocked.shape != self.size:
            raise ValueError("blocked array shape does not match size")
        if connectivity == 6:
            offsets = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
        elif connectivity == 4:
            offsets = ((-1, 0, 0), (1, 0, 0), (0, 0, -1), (0, 0, 1))
        else:
            raise ValueError("connectivity must be 4 or 6")
        self.connectivity = connectivity
        self.name = name
        self.graph = _adjacency(~self.blocked, offsets)
        self.graph.name = name

    @property
    def n_free(self) -> int:
        """Number of traversable cells."""
        return self.graph.n_nodes

    def node(self, xyz: Tuple[int, int, int]) -> int:
        """Node id of ``(x, y, z)``."""
        return self.graph.index(xyz)

    def xyz(self, node: int) -> Tuple[int, int, int]:
        """``(x, y, z)`` of a node id."""
        return self.graph.coord(node)  # type: ignore[return-value]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Grid3D {self.name} {self.size} free={self.n_free}>"
