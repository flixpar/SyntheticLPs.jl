# Traveling Salesperson

The `tsp` category generates routing LPs and MIPs over realistic clustered or
street-network geography. Its eight variants separate two kinds of diversity:
alternative formulations of a single tour, and operational extensions found in
delivery, field service, sales, and pickup-and-delivery planning.

## Variants

| Variant | Application | Natural formulation |
| --- | --- | --- |
| `standard` | Symmetric courier tour | Lifted MTZ |
| `asymmetric` | Large urban courier route, one-way streets and hills | Sparse candidate-arc graph (thousands of stops), lifted MTZ |
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

`asymmetric` is the category's sparse, large-`n` member. Instead of pricing
every ordered pair of a few hundred stops, it keeps a **candidate-arc graph**
the way large-scale routing does: each stop keeps its `m ∈ 6:10` cheapest
outgoing legs among its `2m` geometrically nearest neighbours (found by grid
bucketing, so construction is near-linear). Travel times are directional for
physical reasons — a smooth terrain of a few hills charges extra minutes per
metre of ascent, and about a quarter of neighbouring pairs are joined by a
one-way street that costs a 1.3–2.0× detour against the flow — so the support
itself is asymmetric. Stops with fewer than two incoming candidates receive arcs
from their nearest neighbours. At 100k variables this gives about 10,000 stops
(versus about 316 for `standard`), very sparse degree rows, and an MTZ block
whose big-M equals the large stop count — a different LP from `standard`, not
the same dense model with another cost matrix.

Prize values are log-normal with correlated omission penalties. Precedence
pairs form a sampled acyclic task graph in natural instances.
Multiple-salesperson instances choose a modest fleet and balanced route-size
limits.

## Formulations and sizing

With `n` total stops, including the depot:

| Variant | Variable count |
| --- | ---: |
| `standard`, `precedence`, `multiple_salespersons` | `n^2 - 1` |
| `asymmetric` | `|arcs| + n - 1` (≈ `(m + 1.15) n`) |
| `flow` | `2n(n-1)` |
| `assignment_relaxation` | `n(n-1)` |
| `time_windows` | `n^2` |
| `prize_collecting` | `2n(n-1) + (n-1)` |

Infeasible Hall-district instances delete `k(n-k)` arcs (and their flow
variables where the formulation has them) and size `n` against the delivered
count. `time_windows` omits propagation rows whose big-M would be non-positive
(the row is then implied by the window bounds — roughly half of all stop pairs
when windows are narrow), so its row count is data-dependent.

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

- `feasible` plants or exhibits an integer witness: a complete tour (for `asymmetric`, a Hilbert-curve tour whose legs are added to the candidate graph), a schedule enclosed by its windows, full prize collection, an acyclic precedence order, or a balanced partition across the fleet.
- `infeasible` uses an algebraic certificate that survives relaxation. Core arc formulations use a Hall-deficit **district**: the `k` stops nearest a random anchor can be entered only from the next `k-1` nearest stops (the gateways), contradicting the degree rows (`k = Σ_S indeg ≤ Σ_T outdeg = k-1`). The district scales with the instance (`k ≈ 6–12%` of the nodes, at least 3; `≈ 0.4–0.8·√n` in the sparse `asymmetric` variant), so the deficit is spread over `2k-1` dense degree rows and HiGHS presolve does not detect it — refuting the instance takes simplex work (with the former `k ∈ {2,3}` presolve alone proved infeasibility). Multiple salespersons and prize collection use the district too in about 75% of infeasible instances (for prize collection the quota is set between the LP maximum `total − min_{j∈S} prize_j` and the total prize); the rest keep their classical certificates — fleet route capacity below the customer count, or a quota above the total prize — which presolve does detect. Time windows use a travel budget below the sum of each node's cheapest outgoing arc. Precedence creates a directed three-task cycle.
- `unknown` samples natural operational settings without promising a status (for `asymmetric`, the bare candidate graph without a planted tour; whether it is Hamiltonian is unknown, although its LP relaxation is almost always feasible).

These constructions avoid empty degree rows and contradictory variable bounds,
so infeasible instances retain meaningful routing structure for presolve and LP
solver experiments.
