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

Scale-dependent ranges:

| Scale condition | Nodes | Physical arcs | Commodities |
| --- | ---: | ---: | ---: |
| `target_variables <= 100` | 4-12 | 5-25 | 3-15 |
| `target_variables <= 1000` | 8-35 | 15-120 | 10-80 |
| otherwise | 20-120 | 50-600 | 20-300 |

The constructor searches over `n_arcs` and the implied `n_commodities`. If no combination is within 10% error, it uses a square-root heuristic. Node count is then derived from arc count using a random density factor between 1.5 and 2.5 arcs per node, subject to scale bounds and complete-graph feasibility.

Scale-dependent parameter ranges:

| Scale | Grid | Base install cost | Cost/km | Flow cost/unit | Demand range | Budget factor | Capacity modules |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| small | 100-500 | 10000-50000 | 50-150 | 0.005-0.02 | sampled 1-5 to 50-150 | 0.4-0.7 | 155, 622, 2488 |
| medium | 500-2000 | 30000-100000 | 80-200 | 0.01-0.05 | sampled 5-15 to 100-300 | 0.5-0.8 | 155, 622, 2488, 9953 |
| large | 2000-10000 | 100000-500000 | 150-500 | 0.02-0.1 | sampled 10-30 to 200-1000 | 0.6-0.9 | 622, 2488, 9953, 39813 |

Random data generation:

- Node locations are clustered around `max(1, div(n_nodes, 3))` random centers with normal offsets and clamping to the grid.
- Topology starts with a nearest-neighbor spanning tree to ensure connectivity, then adds mostly short arcs, then random remaining arcs if needed.
- Physical arcs are stored canonically with `i < j`; `directed_arcs` contains both `(i, j)` and `(j, i)`.
- Distances are Euclidean and stored for both directions.
- Link capacities are selected from optical capacity modules. Longer links bias toward larger modules.
- Installation costs equal a base term adjusted by capacity module plus distance cost with 10% random noise.
- Flow costs are proportional to distance and stored for both directions.
- Commodities are random source-sink pairs. Hub-to-hub traffic, where both endpoints are among the first quarter of nodes, receives higher log-normal demand.
- Initial budget is `sum(installation_costs) * budget_factor`.

The stored struct fields are:

- `n_nodes`
- `n_arcs`
- `n_commodities`
- `arcs`
- `directed_arcs`
- `node_locations`
- `distances`
- `installation_costs`
- `link_capacities`
- `flow_costs`
- `commodities`
- `budget`
- `outgoing_arcs`
- `incoming_arcs`

The constructor calls `Random.seed!(seed)`, so generation is reproducible for a fixed seed but resets Julia's global RNG state.

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

- `feasible`: The generator first checks whether all demands can be greedily routed on the full network using `can_route_demands`. If not, it repeatedly scales link capacities up by factors up to a maximum cumulative scale of 50, with one final doubling attempt. It then greedily selects a high capacity-per-cost subset of links, first building connectivity and then adding links until `can_route_demands` succeeds. Budget is set to at least the selected-link installation cost times `1.05` to `1.20`.
- `infeasible`: The generator tightens budget below a greedy spanning-tree cost estimate, using a multiplier from `0.45` to `0.75`. It also targets the busiest source and sink nodes and reduces incident link capacities to only `25%` to `60%` of their associated traffic. If routing still appears possible, it globally halves capacities up to three times, then may directly cap the busiest source's incident capacity to about 10% of outgoing demand.
- `unknown`: No explicit feasibility adjustment is applied; the original budget and sampled capacities are used.

The feasibility checks are constructive heuristics, not calls to the final JuMP solver. They route commodities one at a time along shortest paths with remaining undirected capacity.

## Model Characteristics

- Variables: `n_arcs` installation variables plus `2 * n_arcs * n_commodities` flow variables.
- Constraints: `n_commodities * n_nodes` flow-conservation equalities, `n_arcs` capacity constraints, and one budget constraint.
- Density: flow-conservation rows are sparse network incidence rows; capacity rows touch both directions for every commodity on one physical link plus the corresponding install variable.
- Intended model class: mixed-integer multicommodity network design.
- Default generated LP: with the package default `relax_integer=true`, installation decisions become continuous in `[0, 1]`, producing the LP relaxation.

## Practical Notes

These instances are useful for testing large sparse network matrices, multicommodity flow structure, and fixed-charge design relaxations. The helper `can_route_demands` is greedy and capacity-based, so it is a generation heuristic rather than a proof procedure for all cases. The generated physical network is undirected for capacity and installation, but flow variables are directed.
