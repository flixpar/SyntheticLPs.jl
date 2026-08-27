# Traveling Salesperson

The `tsp` category generates routing LPs and MIPs over realistic clustered or
street-network geography. Its eight variants separate two kinds of diversity:
alternative formulations of a single tour, and operational extensions found in
delivery, field service, sales, and pickup-and-delivery planning.

## Variants

| Variant | Application | Natural formulation |
| --- | --- | --- |
| `standard` | Symmetric courier tour | Lifted MTZ |
| `asymmetric` | Courier routing on one-way streets | Lifted MTZ with directed shortest-path times |
| `flow` | Symmetric courier tour | Single-commodity flow |
| `assignment_relaxation` | Fast lower bound / LP test instance | Continuous degree relaxation with pairwise two-cycle cuts |
| `time_windows` | Appointment delivery | Time propagation, route budget, shift return time |
| `prize_collecting` | Optional sales/service calls | Visit binaries, prize quota, omission penalties, single-commodity flow |
| `multiple_salespersons` | Balanced shared-depot fleet | Lifted route ordering with exact stop-count limits |
| `precedence` | Pickup-before-delivery or ordered service tasks | Lifted MTZ plus precedence rows |

The default is `tsp/standard`. Select another formulation with, for example,
`generate_problem("tsp/flow", 500, feasible, 1)`.

## Geography and data

The symmetric variants share `_tsp_stops` and `_tsp_distance`. A depot is near
the center of a scale-tiered service region; most customers are drawn from town
clusters and roughly 20% are rural outliers. One per-instance road-circuity
factor converts straight-line distance into a positive symmetric road metric.

`asymmetric` instead places stops on an explicit street grid. Odd horizontal
streets run west, even streets run east, vertical avenues are two-way, and every
street has a sampled congestion weight. Costs between stops are directed
shortest-path times, so they satisfy the directed triangle inequality and model
one-way detours without independent pairwise noise.

Prize values are log-normal with correlated omission penalties. Precedence
pairs form a sampled acyclic task graph in natural instances. Multiple-
salesperson instances choose a modest fleet and balanced route-size limits.

## Formulations and sizing

With `n` total stops, including the depot:

| Variant | Variable count |
| --- | ---: |
| `standard`, `asymmetric`, `precedence`, `multiple_salespersons` | `n^2 - 1` |
| `flow` | `2n(n-1)` |
| `assignment_relaxation` | `n(n-1)` |
| `time_windows` | `n^2` |
| `prize_collecting` | `2n(n-1) + (n-1)` |

All natural MIP variants declare binary arc or visit variables. The package
defaults to `relax_integer=true`, producing their LP relaxations. Only
`assignment_relaxation` is continuous by construction even when integrality is
not relaxed.

The standard, asymmetric, and precedence variants use lifted MTZ rows. The flow
and prize-collecting variants source one unit per selected stop from the depot,
with `f[i,j] <= (n-1)x[i,j]`. Time windows eliminate customer-only subtours by
strictly increasing service times along selected arcs. Multiple salespersons
anchor every route's first order to one and use lifted rows that increment the
order exactly on selected customer arcs; the returning stop's order is therefore
the route's customer count.

## Feasibility controls

Every requested status is valid for the model returned by the default relaxed
API:

- `feasible` plants or exhibits an integer witness: a complete tour, a schedule
  enclosed by its windows, full prize collection, an acyclic precedence order,
  or a balanced partition across the fleet.
- `infeasible` uses an algebraic certificate that survives relaxation. Core arc
  formulations use a Hall-deficit access restriction: `k` blocked stops can
  receive arcs only from `k-1` gate nodes, contradicting the degree rows.
  Time windows use a travel budget below the sum of each node's cheapest
  outgoing arc. Prize collection requests more than the total available prize.
  Multiple salespersons set aggregate fleet route capacity below the customer
  count. Precedence creates a directed three-task cycle.
- `unknown` samples natural operational settings without promising a status.

These constructions avoid empty degree rows and contradictory variable bounds,
so infeasible instances retain meaningful routing structure for presolve and LP
solver experiments.
