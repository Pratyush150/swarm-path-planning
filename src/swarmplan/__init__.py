"""swarmplan -- multi-agent path finding for drone swarms and robot fleets.

Plan collision-free paths for many vehicles at once: Conflict-Based Search for
provably optimal plans, ECBS for bounded-suboptimal plans at useful scale,
prioritised planning as the fast incomplete baseline, Hungarian goal assignment
for unlabelled formations, and an execution layer that keeps the plan safe when
a vehicle runs late.

Typical use::

    from swarmplan import GridMap, solve
    grid = GridMap.from_file("data/maps/random-32-32-20.map")
    result = solve("ecbs:w=1.1", grid.graph, starts, goals, time_limit=30)
    print(result.summary())
"""

from .graph import Grid3D, GridMap, SearchGraph
from .constraints import Constraint, ConstraintTable
from .conflicts import (
    Conflict,
    ConflictType,
    find_all_conflicts,
    find_first_conflict,
    validate_plan,
)
from .solution import BUDGET, FAILED, SOLVED, TIMEOUT, UNSOLVABLE, Solution
from .lowlevel import HeuristicCache, backward_dijkstra, focal_space_time_astar, space_time_astar
from .cbs import CBS, CBSConfig, solve_cbs
from .ecbs import ECBS, ECBSConfig, solve_ecbs
from .prioritised import PPConfig, PrioritisedPlanner, solve_pp
from .planners import DEFAULT_SUITE, label_for, parse_spec, solve, solver_for
from .metrics import makespan, singleton_lower_bound, sum_of_costs
from .scenarios import MapfInstance, Scenario, load_scen, make_instance

__version__ = "0.1.0"

__all__ = [
    "Grid3D",
    "GridMap",
    "SearchGraph",
    "Constraint",
    "ConstraintTable",
    "Conflict",
    "ConflictType",
    "find_all_conflicts",
    "find_first_conflict",
    "validate_plan",
    "Solution",
    "SOLVED",
    "TIMEOUT",
    "UNSOLVABLE",
    "BUDGET",
    "FAILED",
    "HeuristicCache",
    "backward_dijkstra",
    "space_time_astar",
    "focal_space_time_astar",
    "CBS",
    "CBSConfig",
    "solve_cbs",
    "ECBS",
    "ECBSConfig",
    "solve_ecbs",
    "PPConfig",
    "PrioritisedPlanner",
    "solve_pp",
    "solve",
    "solver_for",
    "parse_spec",
    "label_for",
    "DEFAULT_SUITE",
    "sum_of_costs",
    "makespan",
    "singleton_lower_bound",
    "MapfInstance",
    "Scenario",
    "load_scen",
    "make_instance",
    "__version__",
]
