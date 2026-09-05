"""Formations for the light-show demonstrations: rings, text, blocks.

A show is a sequence of formations plus the transitions between them. This
module builds the formations as sets of grid cells; :mod:`swarmplan.assignment`
decides which drone flies to which cell, and the MAPF solvers plan the
transition. Nothing here knows about drones -- it produces integer coordinates.

The 5x7 bitmap font is written out in full rather than rendered with a text
library, so a formation is reproducible without a font file, a rasteriser, or a
matplotlib import.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

Point3 = Tuple[int, int, int]

#: 5x7 bitmap font. Rows run top to bottom; '#' is an occupied cell.
FONT: Dict[str, Tuple[str, ...]] = {
    "A": (".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "B": ("####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."),
    "C": (".####", "#....", "#....", "#....", "#....", "#....", ".####"),
    "D": ("####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."),
    "E": ("#####", "#....", "#....", "####.", "#....", "#....", "#####"),
    "F": ("#####", "#....", "#....", "####.", "#....", "#....", "#...."),
    "G": (".####", "#....", "#....", "#..##", "#...#", "#...#", ".###."),
    "H": ("#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "I": ("#####", "..#..", "..#..", "..#..", "..#..", "..#..", "#####"),
    "J": ("####.", "...#.", "...#.", "...#.", "...#.", "#..#.", ".##.."),
    "K": ("#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"),
    "L": ("#....", "#....", "#....", "#....", "#....", "#....", "#####"),
    "M": ("#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"),
    "N": ("#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#"),
    "O": (".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "P": ("####.", "#...#", "#...#", "####.", "#....", "#....", "#...."),
    "Q": (".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"),
    "R": ("####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"),
    "S": (".####", "#....", "#....", ".###.", "....#", "....#", "####."),
    "T": ("#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."),
    "U": ("#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "V": ("#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."),
    "W": ("#...#", "#...#", "#...#", "#.#.#", "#.#.#", "##.##", "#...#"),
    "X": ("#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"),
    "Y": ("#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."),
    "Z": ("#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"),
    "0": (".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."),
    "1": ("..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."),
    "2": (".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"),
    "3": ("#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."),
    "4": ("...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."),
    "5": ("#####", "#....", "####.", "....#", "....#", "#...#", ".###."),
    "6": ("..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."),
    "7": ("#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."),
    "8": (".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."),
    "9": (".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."),
    " ": (".....", ".....", ".....", ".....", ".....", ".....", "....."),
}

GLYPH_WIDTH = 5
GLYPH_HEIGHT = 7


def text_points(
    text: str,
    origin: Tuple[int, int, int] = (0, 0, 0),
    depth: int = 0,
    spacing: int = 1,
    scale: int = 1,
) -> List[Point3]:
    """Cells forming ``text`` in the x-z plane at a fixed depth ``y``.

    The audience is looking along +y, so text is drawn across x (left to right)
    and up z. Returns points in reading order.
    """
    out: List[Point3] = []
    x0, y0, z0 = origin
    cursor = x0
    for ch in text.upper():
        glyph = FONT.get(ch)
        if glyph is None:
            raise ValueError(f"no glyph for character {ch!r}")
        for row, bits in enumerate(glyph):
            for col, bit in enumerate(bits):
                if bit != "#":
                    continue
                for sx in range(scale):
                    for sz in range(scale):
                        x = cursor + col * scale + sx
                        z = z0 + (GLYPH_HEIGHT - 1 - row) * scale + sz
                        out.append((x, depth if depth else y0, z))
        cursor += GLYPH_WIDTH * scale + spacing
    return out


def text_width(text: str, spacing: int = 1, scale: int = 1) -> int:
    """Width in cells of a rendered string."""
    n = len(text)
    return n * GLYPH_WIDTH * scale + max(0, n - 1) * spacing


def _ring_cells(
    centre: Tuple[int, int, int],
    radius: float,
    depth: Optional[int] = None,
    oversample: int = 16,
) -> List[Point3]:
    """Distinct grid cells on one circle, in angular order."""
    cx, cy, cz = centre
    y = cy if depth is None else depth
    steps = max(8, int(round(2.0 * math.pi * radius * oversample)))
    seen: Set[Point3] = set()
    ordered: List[Point3] = []
    for i in range(steps):
        a = 2.0 * math.pi * i / steps
        p = (int(round(cx + radius * math.cos(a))), y, int(round(cz + radius * math.sin(a))))
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def ring_points(
    count: int,
    centre: Tuple[int, int, int],
    radius: float,
    depth: Optional[int] = None,
    oversample: int = 16,
) -> List[Point3]:
    """``count`` distinct cells on a circle in the x-z plane.

    Rounding a circle onto a grid collides points near the flat parts of the
    curve, so the circle is oversampled and thinned to exactly ``count`` distinct
    cells spread as evenly as the grid allows.
    """
    if count <= 0:
        return []
    ordered = _ring_cells(centre, radius, depth, oversample)
    if len(ordered) < count:
        raise ValueError(
            f"a radius-{radius:.2f} ring holds {len(ordered)} distinct cells, {count} requested"
        )
    return resample(ordered, count)


def annulus_points(
    count: int,
    centre: Tuple[int, int, int],
    outer_radius: float,
    ring_spacing: float = 1.8,
    depth: Optional[int] = None,
) -> List[Point3]:
    """``count`` cells on concentric rings, filled from the outside in.

    One circle only holds so many grid cells (roughly ``5.7 * radius``), so a
    formation with more slots than that becomes an annulus: rings are added
    inwards until the count is met, and the innermost ring is thinned so the
    total is exact.
    """
    if count <= 0:
        return []
    out: List[Point3] = []
    seen: Set[Point3] = set()
    radius = outer_radius
    while len(out) < count and radius > 0.4:
        ring = [p for p in _ring_cells(centre, radius, depth) if p not in seen]
        remaining = count - len(out)
        if len(ring) > remaining:
            ring = resample(ring, remaining)
        for p in ring:
            seen.add(p)
            out.append(p)
        radius -= ring_spacing
    if len(out) < count:
        cx, cy, cz = centre
        y = cy if depth is None else depth
        if (cx, y, cz) not in seen and len(out) < count:
            out.append((cx, y, cz))
    if len(out) < count:
        raise ValueError(
            f"an annulus of outer radius {outer_radius:.2f} holds {len(out)} cells, "
            f"{count} requested"
        )
    return out


def grid_points(
    count: int,
    origin: Tuple[int, int, int],
    width: int,
    depth: int = 1,
) -> List[Point3]:
    """``count`` cells in a compact block, filled row by row."""
    x0, y0, z0 = origin
    out: List[Point3] = []
    z = 0
    while len(out) < count:
        for dy in range(depth):
            for x in range(width):
                if len(out) >= count:
                    break
                out.append((x0 + x, y0 + dy, z0 + z))
        z += 1
    return out


def resample(points: Sequence[Point3], count: int) -> List[Point3]:
    """Take ``count`` points spread evenly through a longer ordered sequence."""
    n = len(points)
    if count > n:
        raise ValueError(f"cannot take {count} points from {n}")
    if count == n:
        return list(points)
    idx = [int(round(i * (n - 1) / (count - 1))) if count > 1 else 0 for i in range(count)]
    # Rounding can repeat an index near the ends; walk forward to keep them
    # distinct without disturbing the spread.
    used: Set[int] = set()
    out: List[Point3] = []
    for i in idx:
        while i in used:
            i = (i + 1) % n
        used.add(i)
        out.append(points[i])
    return out


def fit_in_box(points: Sequence[Point3], size: Tuple[int, int, int]) -> bool:
    """True if every point lies inside a box of the given size."""
    return all(all(0 <= c < s for c, s in zip(p, size)) for p in points)


def bounding_box(points: Sequence[Point3]) -> Tuple[Point3, Point3]:
    """Axis-aligned bounds of a formation, as ``(min, max)`` corners."""
    if not points:
        raise ValueError("no points")
    xs, ys, zs = zip(*points)
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))
