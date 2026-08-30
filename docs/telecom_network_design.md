# Telecom Network Design

Generates multicommodity telecom network design instances with binary link installation, continuous routed traffic, capacities, and an installation budget.

## Overview

This generator represents telecommunications network planning. It creates a geographic network of potential undirected physical links, directed routing arcs in both directions, and traffic commodities with source, sink, and demand. The model chooses which physical links to install and how to route every commodity while minimizing installation and routing cost.

## Generator Data and Sizing

`target_variables` is interpreted as:

```text
n_arcs * (2 * n_commodities + 1)
```

For each physical arc, the model has one installation variable and one flow variable for each commodity in each of the two directions.

Sizing is exact-product driven rather than band driven. The constructor samples a realistic commodities-per-link ratio (log-uniform in `[0.35, 1.4]`), derives the implied arc count from `target = n_arcs * (2 * ratio * n_arcs + 1)`, and then scores every commodity count in a wide window around it by

```text
|n_arcs * (2 * n_commodities + 1) - target| / target  +  0.02 * |log(n_commodities / hint)|
```

picking the best. Because the exactness term dominates, the realised variable count tracks the target continuously: over a 25-point logarithmic sweep from 50 to 20000 variables (all three feasibility statuses, five seeds each) the mean relative error is 0.3% and the worst case 1.6%. Node count is then derived from arc count with a random density factor between 1.5 and 2.5 arcs per node, clamped so the topology is realisable (connected and simple).

Requests above `TELECOM_MAX_VARIABLES = 1_000_000` raise an `ArgumentError` rather than silently undersizing, matching the convention used by `supply_chain/network_planning`.

Geographic scale and unit costs are derived from the node count rather than from hard-coded size bands:

| Quantity | Value |
| --- | --- |
| Grid span | `clamp(120 * sqrt(n_nodes), 150, 9000)` times `U(0.8, 1.25)` |
| Base install cost | `12000 * sqrt(n_nodes)` times `U(0.8, 1.3)` |
| Cost per km | `U(60, 400)` |
| Flow cost per unit | `U(0.005, 0.08)` |
| Capacity modules | 155, 622, 2488, 9953, 39813 (OC-3 to OC-768) |

Random data generation:

- Node locations are clustered around `max(1, div(n_nodes, 4))` random centers with normal offsets and clamping to the grid.
- Topology is a Euclidean minimum spanning tree (guaranteeing connectivity) plus the shortest remaining node pairs, ranked with a lognormal perturbation so a few long-haul links appear. Exactly `n_arcs` links are produced.
- Physical arcs are stored canonically with `i < j`; `directed_arcs` contains both `(i, j)` and `(j, i)`.
- Distances are Euclidean and stored for both directions.
- Commodities are population-weighted source-sink pairs with gravity-style lognormal volume *shares* that sum to one. The absolute demand scale is fixed later, once the topology's routing capacity is known.
- Link capacity modules are sized from the traffic each link actually carries: a first cheapest-path routing pass gives per-link reference loads, and each link takes the smallest module covering its load (with a 12% floor of the peak load so unused links stay installable and useful for reroutes).
- Installation costs equal a base term adjusted by the chosen capacity module plus distance cost with 10% random noise.
- Flow costs are proportional to distance and stored for both directions.

## Joint Calibration

Topology, demand, capacity and budget are derived from one another through a **planted nominal design** instead of being sampled on independent schedules:

1. The unit-demand traffic matrix is routed over the topology by a Frank-Wolfe congestion-balancing routing (iteration 1 is the plain cheapest-path routing; later iterations take Frank-Wolfe steps on `sum_a (load_a / cap_a)^4`). Small topologies get more steps, since they are cheap to route.
2. `routable_scale = 1 / max_a (load_a / cap_a)` is the largest total demand this planted routing carries exactly. Solving the max-concurrent-flow LP shows the planted routing is within 0-11% of the true routing threshold at every network size.
3. A family of cuts (every singleton, geometric sweep cuts along random directions, nearest-neighbour balls) yields `cut_bound_scale`, the smallest total demand that provably cannot be routed. By construction `routable_scale <= cut_bound_scale`.
4. `nominal_cost` is the installation cost of the links the planted routing uses; the budget is set relative to it.

The stored struct fields are:

- `n_nodes`, `n_arcs`, `n_commodities`
- `arcs`, `directed_arcs`, `node_locations`, `distances`
- `installation_costs`, `link_capacities`, `flow_costs`
- `commodities`, `budget`, `outgoing_arcs`, `incoming_arcs`
- `total_demand`, `routable_scale`, `cut_bound_scale`, `nominal_cost`
- `feasible_witness::Union{Nothing,TelecomRouteWitness}`
- `infeasibility_certificate::Union{Nothing,TelecomCapacityCutCertificate,TelecomBudgetCertificate}`
- `feasibility_status`

All randomness runs through a local `MersenneTwister(seed)`, so generation is reproducible for a fixed seed and leaves Julia's global RNG untouched.

## LP Formulation

Sets and indices:

- `N = {1, ..., n_nodes}`: network nodes.
- `A`: physical undirected arcs, stored as canonical tuples.
- `D`: directed arcs, containing both directions for each physical arc.
- `K = {1, ..., n_commodities}`: traffic commodities.

For a commodity `k`, let `s_k` be its source, `t_k` its sink, and `d_k` its demand.

Decision variables:

```text
y_a in {0, 1}          for a in A
f_{k,u,v} >= 0         for k in K, (u, v) in D
```

`y_a = 1` means physical link `a` is installed. `f_{k,u,v}` is flow of commodity `k` on directed arc `(u, v)`.

Objective:

```text
minimize
    sum_{a in A} installation_cost_a y_a
  + sum_{(u,v) in D} flow_cost_{u,v} sum_{k in K} f_{k,u,v}
```

Flow conservation:

```text
sum_{(n,j) in outgoing(n)} f_{k,n,j}
  - sum_{(i,n) in incoming(n)} f_{k,i,n}
    = d_k       if n = s_k
    = -d_k      if n = t_k
    = 0         otherwise
```

Capacity on each physical link `a = (i, j)`:

```text
sum_{k in K} (f_{k,i,j} + f_{k,j,i}) <= link_capacity_a y_a
```

Budget:

```text
sum_{a in A} installation_cost_a y_a <= budget
```

Bounds:

```text
y_a binary
f_{k,u,v} >= 0
```

At the package API level, `generate_problem(...; relax_integer=true)` is the default, so installation variables are relaxed unless the caller sets `relax_integer=false`.

## Feasibility Controls

All three profiles place the total demand relative to the two calibration anchors above.

- `feasible`: total demand is 55-90% of `routable_scale`, so the planted design routes every commodity inside the installed capacities, and the budget is 1.02-1.35 times that design's installation cost. The design is stored as a typed `TelecomRouteWitness` (installed links, per-commodity `(node path, flow)` routes, resulting link loads, installation cost) and is an exact feasible point of the integer model - hence also of the LP relaxation.
- `infeasible`: one of two modes, chosen at random.
  - *Capacity shortfall*: total demand is pushed 15-80% past `cut_bound_scale`, so the tightest cut cannot carry the traffic that must cross it. Stored as `TelecomCapacityCutCertificate` with the node set, crossing links, crossing demand and crossing capacity.
  - *Budget shortfall*: demand stays comfortably routable (40-80% of `routable_scale`), but the budget is set to 45-85% of the cut-implied minimum spend `crossing_demand * min_{a in cut} (c_a / cap_a)`. Stored as `TelecomBudgetCertificate`.
  Both arguments use only flow conservation, `sum_k f_k(a) <= cap_a * y_a` and `0 <= y <= 1`, so **the infeasibility survives `relax_integrality`** - it is not integer-deep. This matters because the package API defaults to `relax_integer=true`.
- `unknown`: total demand is placed in a +-35% log band just above `routable_scale`, which brackets the true routing threshold at every scale. The position inside the band is a golden-ratio (low-discrepancy) function of the seed, so consecutive seeds sweep the band evenly and any block of seeds is a genuine mix. Measured with HiGHS on the default LP relaxation (30 seeds per size), the feasible share is 53%, 50%, 43%, 47%, 43% and 43% at 50, 100, 500, 1000, 5000 and 20000 variables.

The witness and the certificates are exact arithmetic statements about the stored fields, not solver calls; `test/problem_types/telecom_network_design.jl` recomputes both from the struct.

## Model Characteristics

- Variables: `n_arcs` installation variables plus `2 * n_arcs * n_commodities` flow variables.
- Constraints: `n_commodities * n_nodes` flow-conservation equalities, `n_arcs` capacity constraints, and one budget constraint.
- Density: flow-conservation rows are sparse network incidence rows; capacity rows touch both directions for every commodity on one physical link plus the corresponding install variable.
- Intended model class: mixed-integer multicommodity network design.
- Default generated LP: with the package default `relax_integer=true`, installation decisions become continuous in `[0, 1]`, producing the LP relaxation.

## Practical Notes

These instances are useful for testing large sparse network matrices, multicommodity flow structure, and fixed-charge design relaxations. The generated physical network is undirected for capacity and installation, but flow variables are directed. The `feasible` and `infeasible` profiles are backed by a planted design and by cut certificates respectively, so they are proofs rather than heuristics; only the `unknown` profile is left genuinely undecided.
