"""One place to name a planner, so the benchmark harness and the CLI agree.

Algorithms are named by a small spec string::

    cbs                     plain CBS
    cbs:pc,bp,ds,dg         CBS with prioritising conflicts, bypass,
                            disjoint splitting and the DG heuristic
    ecbs:w=1.1              ECBS with suboptimality factor 1.1
    ecbs:w=1.5,pc           ECBS with prioritised conflicts
    pp                      prioritised planning, agents in scenario order
    pp:restarts=8           prioritised planning with 8 random restarts

Everything in ``benchmarks/`` and every figure label comes from these strings,
so a table row can be reproduced by pasting its algorithm name back into the
CLI.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

from .cbs.solver import CBS, CBSConfig
from .ecbs.solver import ECBS, ECBSConfig
from .graph import SearchGraph
from .lowlevel.heuristic import HeuristicCache
from .prioritised.planner import PPConfig, PrioritisedPlanner
from .solution import Solution

#: Flags accepted after ``cbs:`` and ``ecbs:``.
FLAGS = {
    "pc": "prioritise_conflicts",
    "bp": "bypass",
    "ds": "disjoint",
}

#: The sweep used in the README tables, in increasing order of sophistication.
DEFAULT_SUITE: List[str] = [
    "cbs",
    "cbs:pc",
    "cbs:pc,bp",
    "cbs:pc,bp,dg",
    "ecbs:w=1.02",
    "ecbs:w=1.1",
    "ecbs:w=1.5",
    "pp",
    "pp:restarts=8",
]


def parse_spec(spec: str) -> Dict[str, object]:
    """Parse an algorithm spec string into ``{"kind": ..., "options": {...}}``."""
    if ":" in spec:
        kind, rest = spec.split(":", 1)
        parts = [p.strip() for p in rest.split(",") if p.strip()]
    else:
        kind, parts = spec, []
    kind = kind.strip().lower()
    options: Dict[str, object] = {}
    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
            key = key.strip().lower()
            options[key] = float(value) if key in ("w",) else int(value)
        elif part.lower() in FLAGS:
            options[FLAGS[part.lower()]] = True
        elif part.lower() in ("cg", "dg"):
            options["heuristic"] = part.lower()
        else:
            raise ValueError(f"unknown option {part!r} in algorithm spec {spec!r}")
    if kind not in ("cbs", "ecbs", "pp"):
        raise ValueError(f"unknown algorithm {kind!r}; expected cbs, ecbs or pp")
    return {"kind": kind, "options": options}


def label_for(spec: str) -> str:
    """The label a spec will produce in tables and figures."""
    parsed = parse_spec(spec)
    kind, options = parsed["kind"], dict(parsed["options"])  # type: ignore[assignment]
    if kind == "cbs":
        return CBSConfig(**options).label()  # type: ignore[arg-type]
    if kind == "ecbs":
        return ECBSConfig(**options).label()  # type: ignore[arg-type]
    return PPConfig(**options).label()  # type: ignore[arg-type]


def solve(
    spec: str,
    graph: SearchGraph,
    starts: Sequence[int],
    goals: Sequence[int],
    time_limit: float = 30.0,
    cache: Optional[HeuristicCache] = None,
) -> Solution:
    """Run the planner named by ``spec`` on one instance."""
    parsed = parse_spec(spec)
    kind = parsed["kind"]
    options = dict(parsed["options"])  # type: ignore[arg-type]
    options["time_limit"] = time_limit
    if kind == "cbs":
        cbs = CBS(graph, starts, goals, CBSConfig(**options), cache)  # type: ignore[arg-type]
        return cbs.solve()
    if kind == "ecbs":
        ecbs = ECBS(graph, starts, goals, ECBSConfig(**options), cache)  # type: ignore[arg-type]
        return ecbs.solve()
    config = PPConfig(**options)  # type: ignore[arg-type]
    return PrioritisedPlanner(graph, starts, goals, config, cache).solve()


def solver_for(spec: str) -> Callable[..., Solution]:
    """A callable bound to one spec, for code that runs the same planner repeatedly."""

    def run(graph, starts, goals, time_limit: float = 30.0, cache=None) -> Solution:
        return solve(spec, graph, starts, goals, time_limit=time_limit, cache=cache)

    run.__name__ = f"solve_{spec.replace(':', '_').replace(',', '_').replace('=', '')}"
    run.__doc__ = f"Run the {label_for(spec)} planner."
    return run
