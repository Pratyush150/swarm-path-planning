"""Figures and animations. Everything committed to the repository comes from here.

Matplotlib is an optional dependency: this module imports cleanly without it and
every entry point raises a clear error instead of failing on an attribute deep
inside a plotting call. Nothing in the solver packages imports this module.

The colour choices are deliberate rather than decorative. Agent colours come
from a cyclic map so that neighbouring agents are distinguishable on a crowded
grid; the light-show renders are on black because that is what the subject
actually looks like, and because a white background makes 200 overlapping
trails unreadable.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - exercised implicitly by the figure scripts
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.collections import LineCollection

    HAVE_MATPLOTLIB = True
except ImportError:  # pragma: no cover - depends on the environment
    HAVE_MATPLOTLIB = False


def _require() -> None:
    """Fail with an actionable message when matplotlib is absent."""
    if not HAVE_MATPLOTLIB:
        raise RuntimeError(
            "matplotlib is required for swarmplan.viz; install it with "
            "'pip install matplotlib pillow'"
        )


def agent_colours(n: int, cmap: str = "turbo") -> np.ndarray:
    """``n`` visually distinct colours, spread so neighbours differ."""
    _require()
    base = plt.get_cmap(cmap)
    # Golden-ratio stride around the colour wheel: adjacent agent indices land
    # far apart, which matters when agents cluster in one corridor.
    phi = 0.6180339887
    return np.array([base((i * phi) % 1.0) for i in range(n)])


def at(path: Sequence[int], t: int) -> int:
    """Location of an agent at timestep ``t``, parked on its goal after arrival."""
    return path[t] if t < len(path) else path[-1]


def path_xy(grid, path: Sequence[int]) -> np.ndarray:
    """``(len(path), 2)`` array of ``(x, y)`` plot coordinates for a 2D grid path."""
    rc = np.array([grid.rc(v) for v in path], dtype=float)
    return np.column_stack([rc[:, 1], rc[:, 0]])


# ---------------------------------------------------------------------------
# 2D grid plans
# ---------------------------------------------------------------------------
def draw_map(ax, grid, obstacle_colour: str = "#2b2f38", free_colour: str = "#f4f5f7") -> None:
    """Draw the occupancy grid as an image with the origin at the top left."""
    _require()
    img = np.where(grid.blocked, 0.0, 1.0)
    ax.imshow(
        img,
        cmap=matplotlib.colors.ListedColormap([obstacle_colour, free_colour]),
        interpolation="nearest",
        origin="upper",
        extent=(-0.5, grid.width - 0.5, grid.height - 0.5, -0.5),
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def animate_plan(
    grid,
    paths: Sequence[Sequence[int]],
    out_path,
    title: str = "",
    fps: int = 6,
    trail: int = 6,
    dpi: int = 80,
    figsize: Tuple[float, float] = (5.2, 5.4),
    marker_size: float = 42.0,
) -> Path:
    """Animate a joint plan on a 2D grid and write it as a GIF.

    This is the asset that shows the whole point of the repository: dozens of
    agents crossing the same map, on the same timesteps, never colliding.
    """
    _require()
    out_path = Path(out_path)
    horizon = max(len(p) for p in paths)
    colours = agent_colours(len(paths))
    xy = [path_xy(grid, p) for p in paths]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    draw_map(ax, grid)
    if title:
        ax.set_title(title, fontsize=9, color="#1b1e24", pad=6)
    # The scatter is created with its final colours: a scatter built from an
    # empty ``c=[]`` is treated as colour-mapped, and every later call to
    # set_facecolors is silently overwritten at draw time by the (empty) colour
    # array. Set the offsets per frame and leave the colours alone.
    start = np.array([track[0] for track in xy])
    scat = ax.scatter(
        start[:, 0], start[:, 1], s=marker_size, c=colours,
        edgecolors="white", linewidths=0.4, zorder=3,
    )
    trails = LineCollection([], linewidths=1.1, alpha=0.55, zorder=2)
    ax.add_collection(trails)
    stamp = ax.text(
        0.015, 0.02, "", transform=ax.transAxes, va="bottom", ha="left", fontsize=8,
        color="#1b1e24",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none"),
    )

    def frame(t: int):
        pts = np.array([xy[i][min(t, len(xy[i]) - 1)] for i in range(len(paths))])
        scat.set_offsets(pts)
        segs, seg_colours = [], []
        for i, track in enumerate(xy):
            lo = max(0, min(t, len(track) - 1) - trail)
            hi = min(t, len(track) - 1) + 1
            if hi - lo >= 2:
                segs.append(track[lo:hi])
                seg_colours.append(colours[i])
        trails.set_segments(segs)
        trails.set_color(seg_colours)
        stamp.set_text(f"t = {t}/{horizon - 1}   agents = {len(paths)}")
        return scat, trails, stamp

    interval = 1000 // max(fps, 1)
    anim = animation.FuncAnimation(fig, frame, frames=horizon, interval=interval)
    anim.save(str(out_path), writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def filmstrip(
    grid,
    paths: Sequence[Sequence[int]],
    out_path,
    n_frames: int = 6,
    title: str = "",
    dpi: int = 110,
) -> Path:
    """A row of stills from a plan, for readers who cannot see the animation."""
    _require()
    out_path = Path(out_path)
    horizon = max(len(p) for p in paths)
    steps = [int(round(i * (horizon - 1) / max(n_frames - 1, 1))) for i in range(n_frames)]
    colours = agent_colours(len(paths))
    xy = [path_xy(grid, p) for p in paths]

    fig, axes = plt.subplots(1, n_frames, figsize=(2.05 * n_frames, 2.35), dpi=dpi)
    for ax, t in zip(np.atleast_1d(axes), steps):
        draw_map(ax, grid)
        pts = np.array([xy[i][min(t, len(xy[i]) - 1)] for i in range(len(paths))])
        for i, track in enumerate(xy):
            lo = max(0, min(t, len(track) - 1) - 8)
            hi = min(t, len(track) - 1) + 1
            if hi - lo >= 2:
                ax.plot(track[lo:hi, 0], track[lo:hi, 1], color=colours[i], lw=0.9, alpha=0.5)
        ax.scatter(
            pts[:, 0], pts[:, 1], s=14, c=colours, edgecolors="white", linewidths=0.3, zorder=3
        )
        ax.set_title(f"t = {t}", fontsize=8, pad=3)
    if title:
        fig.suptitle(title, fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def plot_space_time(
    grid,
    before: Sequence[Sequence[int]],
    after: Sequence[Sequence[int]],
    conflict,
    out_path,
    agents: Optional[Sequence[int]] = None,
    dpi: int = 130,
) -> Path:
    """Space-time diagram of one conflict, before and after CBS resolves it.

    Time runs along the horizontal axis and the vertical axis is the cell, in
    row-major order, so a wait is a horizontal segment, a detour into another
    row is a jump to a different band, and a collision is two trajectories
    meeting at a point. It is the clearest picture of what a single CBS
    constraint actually does to a plan.
    """
    _require()
    out_path = Path(out_path)
    agents = list(agents) if agents is not None else [conflict.a1, conflict.a2]
    colours = agent_colours(max(agents) + 1)

    def label(cell: int) -> str:
        r, c = grid.rc(cell)
        return f"({r},{c})"

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), dpi=dpi, sharey=True)
    panels = (
        (axes[0], before, "before: single-agent plans ignore each other"),
        (axes[1], after, "after: CBS adds one constraint and replans"),
    )
    for ax, paths, title in panels:
        horizon = max(len(p) for p in paths)
        for a in agents:
            cells = [at(paths[a], t) for t in range(horizon)]
            ax.plot(
                range(horizon), cells, "-", color=colours[a], lw=1.8, marker="o", ms=3.4,
                label=f"agent {a}",
            )
        ax.set_xlabel("timestep")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("cell (row-major index)")
    cells_shown = sorted({at(p, t) for p in list(before) + list(after) for t in range(len(p))})
    ticks = cells_shown[:: max(1, len(cells_shown) // 8)]
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels([label(c) for c in ticks], fontsize=7)
    if conflict is not None:
        axes[0].scatter(
            [conflict.time], [conflict.loc1], s=170, facecolors="none",
            edgecolors="#d1495b", linewidths=2.0, zorder=5,
        )
        kind = "swap (edge) conflict" if conflict.is_edge else "vertex conflict"
        axes[0].annotate(
            kind, (conflict.time, conflict.loc1), textcoords="offset points",
            xytext=(-6, 16), color="#d1495b", fontsize=8, ha="right",
        )
    axes[1].legend(fontsize=8, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 3D light show
# ---------------------------------------------------------------------------
def isometric(points: np.ndarray) -> np.ndarray:
    """Project ``(n, 3)`` ``(x, y, z)`` points to 2D with a 30-degree isometric view."""
    c, s = math.cos(math.radians(30.0)), math.sin(math.radians(30.0))
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return np.column_stack([(x - y) * c, (x + y) * s + z])


def _show_positions(space, paths: Sequence[Sequence[int]], t: int) -> np.ndarray:
    """``(n_agents, 3)`` coordinates of every agent at timestep ``t``."""
    return np.array([space.xyz(at(p, t)) for p in paths], dtype=float)


def plot_light_show(
    space,
    paths: Sequence[Sequence[int]],
    out_path,
    times: Optional[Sequence[int]] = None,
    title: str = "",
    dpi: int = 120,
) -> Path:
    """Stills of a formation morph in three projections.

    The audience view is what the show looks like from the ground, the isometric
    view shows the depth the swarm is using to get out of its own way, and the
    top-down view is where the traffic actually is. The depth axis is a fraction
    of the width, so the top-down panel is drawn without an equal aspect ratio
    and its label says so.
    """
    _require()
    out_path = Path(out_path)
    horizon = max(len(p) for p in paths)
    times = list(times) if times is not None else [0, horizon // 3, 2 * horizon // 3, horizon - 1]
    colours = agent_colours(len(paths))

    fig, axes = plt.subplots(3, len(times), figsize=(2.5 * len(times), 6.0), dpi=dpi)
    frames = [_show_positions(space, paths, t) for t in times]
    rows = [
        [(f[:, 0], f[:, 2]) for f in frames],
        [(isometric(f)[:, 0], isometric(f)[:, 1]) for f in frames],
        [(f[:, 0], f[:, 1]) for f in frames],
    ]
    fig.patch.set_facecolor("#07080c")
    for row, series in enumerate(rows):
        xs = np.concatenate([x for x, _ in series])
        ys = np.concatenate([y for _, y in series])
        pad = 1.5
        for col, (x, y) in enumerate(series):
            ax = axes[row, col]
            ax.scatter(x, y, s=13, c=colours, edgecolors="none")
            ax.set_xlim(xs.min() - pad, xs.max() + pad)
            ax.set_ylim(ys.min() - pad, ys.max() + pad)
            ax.set_facecolor("#07080c")
            ax.set_xticks([])
            ax.set_yticks([])
            if row < 2:
                ax.set_aspect("equal")
            for spine in ax.spines.values():
                spine.set_color("#232733")
            if row == 0:
                ax.set_title(f"t = {times[col]}", fontsize=8, color="#dfe3ea")
    for row, label in enumerate(("audience view", "isometric", "top down (depth exaggerated)")):
        axes[row, 0].set_ylabel(label, color="#dfe3ea", fontsize=8)
    if title:
        fig.suptitle(title, color="#dfe3ea", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="#07080c")
    plt.close(fig)
    return out_path


def animate_light_show(
    space,
    paths: Sequence[Sequence[int]],
    out_path,
    fps: int = 8,
    trail: int = 5,
    dpi: int = 80,
    title: str = "",
) -> Path:
    """Animate a formation morph, isometric next to top-down, on black."""
    _require()
    out_path = Path(out_path)
    horizon = max(len(p) for p in paths)
    colours = agent_colours(len(paths))
    tracks = [np.array([space.xyz(v) for v in p], dtype=float) for p in paths]
    iso_tracks = [isometric(t) for t in tracks]

    all_iso = np.vstack(iso_tracks)
    all_top = np.vstack([t[:, [0, 1]] for t in tracks])

    fig, axes = plt.subplots(
        2, 1, figsize=(5.6, 5.0), dpi=dpi, gridspec_kw={"height_ratios": [5, 1]}
    )
    fig.patch.set_facecolor("#07080c")
    for ax, data, name, equal in (
        (axes[0], all_iso, "isometric", True),
        (axes[1], all_top, "top down (depth exaggerated)", False),
    ):
        ax.set_facecolor("#07080c")
        pad = 2.0
        ax.set_xlim(data[:, 0].min() - pad, data[:, 0].max() + pad)
        ax.set_ylim(data[:, 1].min() - pad, data[:, 1].max() + pad)
        if equal:
            ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name, color="#98a0b3", fontsize=8)
        for spine in ax.spines.values():
            spine.set_color("#1b1f2a")

    # See the note in animate_plan: colours are set once, at creation.
    iso_start = np.array([t[0] for t in iso_tracks])
    top_start = np.array([t[0, [0, 1]] for t in tracks])
    iso_scat = axes[0].scatter(
        iso_start[:, 0], iso_start[:, 1], s=19, c=colours, edgecolors="none", zorder=3
    )
    top_scat = axes[1].scatter(
        top_start[:, 0], top_start[:, 1], s=16, c=colours, edgecolors="none", zorder=3
    )
    iso_trails = LineCollection([], linewidths=0.8, alpha=0.45, zorder=2)
    top_trails = LineCollection([], linewidths=0.8, alpha=0.45, zorder=2)
    axes[0].add_collection(iso_trails)
    axes[1].add_collection(top_trails)
    stamp = fig.text(0.5, 0.97, "", ha="center", color="#98a0b3", fontsize=9)

    def frame(t: int):
        idx = [min(t, len(tr) - 1) for tr in tracks]
        iso_pts = np.array([iso_tracks[i][idx[i]] for i in range(len(tracks))])
        top_pts = np.array([tracks[i][idx[i], [0, 1]] for i in range(len(tracks))])
        iso_scat.set_offsets(iso_pts)
        top_scat.set_offsets(top_pts)
        iso_segs, top_segs, cols = [], [], []
        for i in range(len(tracks)):
            lo = max(0, idx[i] - trail)
            hi = idx[i] + 1
            if hi - lo >= 2:
                iso_segs.append(iso_tracks[i][lo:hi])
                top_segs.append(tracks[i][lo:hi][:, [0, 1]])
                cols.append(colours[i])
        iso_trails.set_segments(iso_segs)
        iso_trails.set_color(cols)
        top_trails.set_segments(top_segs)
        top_trails.set_color(cols)
        stamp.set_text(f"{title}   t = {t}/{horizon - 1}")
        return iso_scat, top_scat, iso_trails, top_trails, stamp

    interval = 1000 // max(fps, 1)
    anim = animation.FuncAnimation(fig, frame, frames=horizon, interval=interval)
    anim.save(
        str(out_path),
        writer=animation.PillowWriter(fps=fps),
        savefig_kwargs={"facecolor": "#07080c"},
    )
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Benchmark charts
# ---------------------------------------------------------------------------
def plot_success_rate(records, out_path, title: str = "", dpi: int = 130) -> Path:
    """Success rate against number of agents, one line per algorithm."""
    _require()
    out_path = Path(out_path)
    from .metrics import group_by, success_rate

    by_alg: Dict[str, Dict[int, float]] = {}
    for (alg, n), rows in group_by(records, "algorithm", "n_agents").items():
        by_alg.setdefault(alg, {})[n] = success_rate(rows)

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=dpi)
    cmap = plt.get_cmap("viridis")
    names = sorted(by_alg)
    for i, alg in enumerate(names):
        series = sorted(by_alg[alg].items())
        xs = [n for n, _ in series]
        ys = [100.0 * v for _, v in series]
        ax.plot(xs, ys, marker="o", ms=4, lw=1.7, label=alg, color=cmap(i / max(len(names) - 1, 1)))
    ax.set_xlabel("number of agents")
    ax.set_ylabel("success rate within the time budget (%)")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def plot_runtime_distribution(records, out_path, title: str = "", dpi: int = 130) -> Path:
    """Runtime distribution per algorithm on a log axis, solved instances only."""
    _require()
    out_path = Path(out_path)
    from .metrics import group_by

    groups = {k[0]: v for k, v in group_by(records, "algorithm").items()}
    names = sorted(groups)
    data = [[r.runtime for r in groups[k] if r.solved] for k in names]
    keep = [i for i, d in enumerate(data) if d]
    names = [names[i] for i in keep]
    data = [data[i] for i in keep]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=dpi)
    parts = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=True)
    cmap = plt.get_cmap("viridis")
    for i, box in enumerate(parts["boxes"]):
        box.set_facecolor(cmap(i / max(len(names) - 1, 1)))
        box.set_alpha(0.75)
        box.set_edgecolor("#333333")
    for whisker in parts["whiskers"] + parts["caps"]:
        whisker.set_color("#333333")
    for median in parts["medians"]:
        median.set_color("#111111")
    for flier in parts["fliers"]:
        flier.set_markersize(2.5)
        flier.set_markeredgecolor("#777777")
    ax.set_yscale("log")
    ax.set_xticklabels([n.replace("+", "\n+") for n in names], fontsize=7.5)
    ax.set_ylabel("runtime, solved instances (s, log scale)")
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def plot_assignment_comparison(
    labels: Sequence[str],
    totals: Sequence[float],
    maxima: Sequence[float],
    out_path,
    title: str = "",
    dpi: int = 130,
) -> Path:
    """Bar chart of total and worst-case travel distance per assignment strategy."""
    _require()
    out_path = Path(out_path)
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), dpi=dpi)
    axes[0].bar(x, totals, width=0.6, color="#3b6ea5")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("total travel distance (cells)")
    axes[0].set_title("sum of distances", fontsize=9)
    axes[1].bar(x, maxima, width=0.6, color="#a5533b")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("longest single flight (cells)")
    axes[1].set_title("makespan driver", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5, axis="y")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
