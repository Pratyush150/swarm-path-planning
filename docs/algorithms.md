# The algorithms, and why each one exists

This is the reasoning behind the code in `src/swarmplan`. It assumes you can
read A* and want to know what MAPF adds on top.

## The problem in one paragraph

Given a graph, *k* agents, a start and a goal for each, find a path for every
agent such that no two agents occupy the same vertex at the same timestep and no
two agents swap across an edge, minimising the **sum of costs** (the total
number of timesteps agents spend before parking on their goals). Agents may
wait. An agent that has arrived stays put and still occupies its cell.

## Why it is hard

The joint state space is the product of the individual ones: *V^k*. Twenty
agents on the 819 free cells of `random-32-32-20` is about 10^58 joint states.
A* over that space is not slow, it is impossible -- the branching factor alone
is 5^20 (four moves plus a wait, per agent). Finding an optimal sum-of-costs
solution is NP-hard.

The escape is that the coupling between agents is usually *sparse*. Most pairs
of agents never interact. An algorithm that plans each agent separately and only
pays for the interactions that actually occur is doing exponentially less work
than one that treats every agent as coupled to every other. That is the idea
behind everything below.

## Space-time A* (`swarmplan/lowlevel`)

The single-agent search, in the time-expanded graph: a state is
`(vertex, timestep)`, actions are the graph moves plus `wait`, each costing one
timestep.

Two details that are easy to get wrong:

* **A wait costs.** Only waiting at the goal *after final arrival* is free, and
  that is handled by ending the path at arrival and treating the agent as parked
  from then on.
* **An agent cannot always stop when it arrives.** If a constraint forbids the
  goal cell at t=12, an agent arriving at t=9 has to leave and come back. The
  search accepts a goal state only at or after the last constrained timestep on
  the goal.

### The heuristic

The heuristic is the exact distance to the goal in the static graph, computed
once per goal by a backward Dijkstra sweep (a BFS, since costs are unit) and
reused by every one of the thousands of low-level calls CBS makes for that
agent.

Manhattan distance is admissible but weak. Measured over the first 150
start/goal pairs of scenario 1 for each map, by
`python3 tools/heuristic_report.py --pairs 150`:

| map | mean true/Manhattan | max | A* states expanded, true | with Manhattan |
|---|---|---|---|---|
| empty-32-32 | 1.00 | 1.00 | 3,381 | 3,381 |
| random-32-32-20 | 1.09 | 2.11 | 3,635 | 11,022 |
| room-32-32-4 | 1.22 | 3.80 | 3,857 | 35,382 |
| maze-32-32-2 | 2.87 | 17.33 | 8,508 | 893,969 |
| den312d | 1.34 | 8.82 | 7,951 | 327,166 |
| warehouse-10-20-10-2-1 | 1.00 | 1.10 | 12,719 | 13,295 |

Two orders of magnitude on the maze, for the same answers. On the open maps it
costs nothing and gains nothing, which is the honest summary: the true-distance
heuristic matters exactly where the map has structure.

The heuristic is also *consistent* (`h(u) <= 1 + h(v)` across every edge), which
is what makes the closed set sound.

## Conflict-Based Search (`swarmplan/cbs`)

Two levels.

**Low level:** plan one agent, respecting a set of constraints.

**High level:** search a binary tree of constraint sets, best-first on cost.
The root has no constraints. Simulate the joint plan; if two agents collide,
create two children, one forbidding each agent from the conflicting
vertex/edge at that timestep, and replan just that agent. Every valid joint plan
satisfies at least one of the two constraints, so the split loses nothing, and
because the search is best-first on cost the first conflict-free node found is
optimal.

### The four improvements, all switchable

| flag | what it does | why it helps |
|---|---|---|
| `pc` | prioritise conflicts | split on a **cardinal** conflict (one where both children provably cost more) when one exists. Splitting on a free conflict just widens the tree. |
| `bp` | bypass conflicts | if a child has the same cost as its parent and fewer conflicts, adopt its path into the parent instead of branching. A better plan for the price of a branch that never happens. |
| `ds` | disjoint splitting | split on "agent *a* **must** be here" against "must not", instead of "*a* must not" against "*b* must not". The classic split's children overlap, so CBS can explore the same joint plan twice; the disjoint split partitions. |
| `cg` / `dg` | high-level heuristic | an admissible estimate of the cost still to be paid, from the conflict graph. Shrinks the tree the most and costs the most per node; the README ablation says where that pays. |

### Cardinal conflicts and MDDs

An MDD for agent *i* at cost *c* is the layered graph of every state on some
constraint-respecting path of length exactly *c*. If the MDD has **width 1** at
timestep *t*, the agent is on that cell in all of its optimal paths, so
forbidding it there necessarily costs more. A conflict where that holds for both
agents is **cardinal**, for one agent **semi-cardinal**, for neither
**non-cardinal**.

### The CG and DG heuristics

Build a graph with one vertex per agent. In **CG**, join two agents if their
conflict is cardinal. Resolving each such conflict requires raising the cost of
at least one of its two agents, so the **minimum vertex cover** of that graph is
a lower bound on the extra cost -- admissible, so CBS stays optimal. **DG** uses
a broader edge test: join two agents if they cannot both keep their current cost,
decided by a joint search over their two MDDs. Strictly stronger, strictly more
expensive.

MVC is NP-hard in general and trivial here: the graphs have one vertex per agent
and only the conflicting agents have edges, so exact branch and bound over the
connected components is instant. `tests/test_cbs_heuristics.py` checks it
against exhaustive enumeration on random graphs.

**Not implemented:** WDG, the edge-weighted variant that solves a two-agent MAPF
instance per pair to get an edge weight. It is the strongest of the family and
it is the obvious next step.

## ECBS (`swarmplan/ecbs`)

Optimal is expensive and rarely worth it. ECBS trades a stated factor *w* of
solution cost for a large amount of runtime, using focal search at both levels:

* **Low level:** among all open states whose `f` is within `w` of the minimum,
  expand the one with fewest conflicts against the other agents' current paths.
  Returns a path costing at most `w x optimum` and a genuine lower bound `lb_i`.
* **High level:** `LB(N) = sum_i lb_i(N)` bounds the best solution below `N`.
  OPEN is ordered by `LB`; FOCAL holds the open nodes whose **cost** is at most
  `w * LB_min`, ordered by conflict count. The first conflict-free node found
  therefore costs at most `w * LB_min <= w * C*`.

The guarantee is hard, and `tests/test_ecbs.py` checks it against a brute-force
optimum for w in {1.0, 1.05, 1.2, 1.5, 2.0, 3.0}.

**Relationship to EECBS.** EECBS adds an *inadmissible*, online-learned
high-level heuristic to order FOCAL while keeping the admissible bound separate.
We implement ECBS plus an optional **admissible** CG/DG heuristic added to `LB`.
That is the gap between this and a full EECBS. We have not run the EECBS
reference implementation and make no claim about matching it; it is cited as
what a serious comparison would be made against.

## Prioritised planning (`swarmplan/prioritised`)

Order the agents, plan each one avoiding the paths already fixed. *k*
single-agent searches, no tree, no backtracking, milliseconds. It is what a
large number of deployed multi-robot systems actually run.

It is **incomplete**. Three cells of corridor with one alcove:

```
A B C      agent 1: A -> C
. D .      agent 2: C -> A
```

Whichever agent plans first walks down the corridor and parks on the other's
start cell; the second agent has nowhere to be. Both orders fail. CBS solves it
in a few nodes with a sum-of-costs of 7. `tests/test_prioritised.py` asserts all
of that, including that random restarts cannot help, because no order works.

A failure of prioritised planning is therefore reported as `failed`, never as
`unsolvable`: an incomplete planner has proved nothing about the instance.

## What none of these do

* No kinematic constraints inside the search. A grid step is a grid step.
* No continuous-time reasoning (no SIPP, no CCBS).
* No large-neighbourhood search (MAPF-LNS), which is what the current
  state of the art uses for very large instances.
* No lifelong or online MAPF: goals are fixed at planning time.
