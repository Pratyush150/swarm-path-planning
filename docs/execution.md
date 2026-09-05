# From a plan to a flight

A MAPF plan is a timetable, and a timetable is a lie the moment a vehicle is
late. This document covers the three layers between "the planner says agent 4 is
on cell 91 at t=12" and "the fleet flew it".

## 1. Ordering, not timing: the Action Dependency Graph

If agent B is scheduled to enter a cell after agent A leaves it, then what
matters is not the absolute times -- it is the *order*. Keep the order and throw
away the clock:

* **type 1** -- agent *i* reaches waypoint *k* only after it reached *k-1*;
* **type 2** -- if the plan has agent *j* entering cell *v* after agent *i*
  leaves it, then *j* may not enter *v* until *i* has actually left.

Executing by "advance each agent as soon as its dependencies are satisfied" gives
a plan that degrades gracefully. A late agent delays exactly the agents scheduled
behind it, everyone else carries on, and no ordering is ever violated, so no
collision the planner ruled out can happen.

The alternative -- dispatching absolute times and hoping -- produces collisions
as soon as anything slips. Both are implemented
(`ActionDependencyGraph.execute` and `fixed_schedule_execution`) so the
difference can be measured rather than asserted; the demo prints it:

```
agent 0 delayed 3 ticks at step 2:
  fixed timetable execution: 3 collisions, 19 ticks
  dependency-graph execution: 0 collisions, 21 ticks
```

Two ticks later, and safe. That is the trade.

`allow_following=False` additionally forbids an agent from entering a cell in
the same tick its predecessor leaves, which buys a tick of physical separation
for vehicles that need it.

## 2. Smoothing: corners a quadrotor can fly

Grid plans have right-angle corners and instantaneous stops. `smooth_plan`:

1. averages each waypoint with its neighbours, a few passes, with a hard cap on
   how far any waypoint may move from its grid cell -- the cap is what keeps the
   smoothed path inside the corridor the planner cleared;
2. interpolates a Catmull-Rom spline through the smoothed waypoints;
3. scales the time axis by the single factor that brings peak speed and peak
   acceleration under their limits. Stretching time by `s` divides speed by `s`
   and acceleration by `s^2`, so the factor is exact and one pass is enough.

Every agent is slowed by the same factor, so the relative ordering the planner
established is untouched. A per-segment time-optimal retiming would be faster
and would need its own safety argument.

## 3. Separation in continuous time

Grid conflict checking asks "were two agents in the same cell at the same
timestep?". A safety case asks "how close did they ever get?". Two agents on
adjacent cells passing in opposite directions are one cell apart mid-step --
legal on the grid, too close if a cell is 1 m and the rotors need 2 m.

`separation_violations` works on the continuous trajectories and is exact rather
than sampled: between two samples both agents move linearly, so the relative
position is linear and the minimum distance over the interval has a closed form.
Sampling distances at the sample points would miss the closest approach, which
is precisely in the middle of a crossing.

The demo shows the check firing on a plan that is perfectly legal on the grid:

```
continuous-time execution check (2 m cells, 3 m/s, 2 m/s^2):
  peak speed: 1.47 m/s (limit 3.00)
  peak acceleration: 2.00 m/s^2 (limit 2.00)
  closest approach: 1.299 m (required 1.000)
  violations: 0
```

Raise the requirement to 1.5 m on 2 m cells and the same plan has violations.
Grid legality is not separation.

## What this layer does not model

No aerodynamics and no downwash: a real multirotor descending through another's
wake is a hazard the grid knows nothing about, and vertical separation rules in
a light show exist for that reason. No battery state, no radio scheduling, no
wind, no localisation error, and no yaw or attitude -- the vehicles are points
with a speed and an acceleration limit.
