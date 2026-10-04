# Network Flow

Single-commodity flow over sparse, capacitated, geographically embedded
networks. The category contains three variants: `standard`, the classical
minimum-cost flow (NETGEN-style transshipment) LP whose feasibility is placed by
an exact max-flow computation; `generalized_flow`, a lossy-flow variant whose
arcs attenuate what they carry; and `time_expanded`, evacuation planning as a
dynamic (time-expanded) network flow. All build pure continuous LPs, scale
from a handful of variables to the documented 1,000,000 cap, and use a local
RNG — calling a generator neither reseeds nor consumes Julia's global RNG, and
`build_model` does no sampling.

## Variants

| Variant | Model class | Key structure | Domain grounding |
|---|---|---|---|
| `standard` | continuous LP (TU) | min-cost flow with supply, demand and transit nodes; arc capacities as bounds; exact max-flow feasibility boundary | freight and commodity distribution, pipelines, regional road networks |
| `generalized_flow` | continuous LP (non-TU) | per-arc gains in (0, 1): distance-decaying losses; same balance-row structure | transmission/line losses, pipeline leakage, evaporation, spoilage |
| `time_expanded` | continuous LP (TU) | network copies per period, travel-time movement arcs, waiting arcs, shelter intake rates, deadline; total evacuation time | evacuation planning, quickest transshipment, dynamic traffic assignment |

## Shared network model

Both variants draw their network from the shared geographic machinery in
`src/problem_types/network_flow/geo_network.jl` (also used by the other flow
categories):

- **Nodes** are scattered over a square region whose side is `12 * sqrt(n)`
  (constant node density, so arc lengths and costs keep realistic magnitudes
  at every size) in one of three shapes: `:uniform` (rural grids, mesh
  distribution), `:clustered` (metropolitan clusters with Zipf-like sizes and
  spreads growing with the cluster's share; each cluster's centre node carries
  a 5-20x activity boost), and `:corridor` (a bent band: river valleys,
  coastal strips, interstate corridors). Every node gets a heavy-tailed
  (lognormal) activity weight.
- **Arcs**: a geometric spanning tree (Kruskal over each node's 8 nearest
  neighbours, with disconnected clusters joined by inter-regional trunk links
  along a minimum spanning tree of the cluster representatives) in both
  directions — the network is strongly connected and these `trunk` arcs are
  flagged — then a second two-way link for every leaf of the tree (dead ends
  are rare in real networks, and a dangling node turns into
  presolve-reducible doubleton rows), then local links drawn short-first (with
  lognormal noise) from the nearest-neighbour candidates, two-way with
  probability 0.85, until the arc count equals the target **exactly**. The result has 3.2-4.6 arcs per node,
  like real road and pipeline networks.
- **Roles**: 4%-12% of nodes are supply sites (drawn uniformly — resources sit
  where they sit); 25%-50% of the rest are demand nodes, drawn in proportion
  to activity weight; nominal demands follow the weights (mean 50 units,
  heavy-tailed). The remaining nodes are transit junctions.
- **Costs**: arc length times a lognormal route factor (spread drawn per
  instance), 20% cheaper per km on trunk links, plus a small handling charge —
  strongly correlated with distance rather than iid noise.

## `standard` formulation

```text
flow[a] in [0, capacity[a]]                       for each arc a  (bounds)

sum_{a out of v} flow[a] - sum_{a into v} flow[a] <= supply[v]    supply node v
sum_{a into v} flow[a] - sum_{a out of v} flow[a]  = demand[v]    demand node v
sum_{a into v} flow[a] - sum_{a out of v} flow[a]  = 0            transit node v

minimize sum_a cost[a] * flow[a]
```

Capacities are variable bounds, not singleton rows. Capacities come from a
"historical" routing of the nominal demand over noisy route lengths
(infrastructure is built where traffic used to go), multiplied by a wide
lognormal provisioning factor (median 1.2, some arcs under-built), with a tiered
lognormal floor (trunk links 1.6x fatter). Because the cost-optimal routing and
the historical routing disagree, many arcs bind at the optimum.

### Feasibility control

By Gale's theorem the demands `lambda * d0` are deliverable iff every node set
`T` satisfies `lambda * d0(T) <= supply(T) + cap_in(T)`. The constructor
computes the **exact** largest such `lambda*` with Dinkelbach's min-ratio-cut
iteration over exact Dinic max flows on the extended network (super source ->
supply sites, demand nodes -> super sink): first with unlimited supply (the
network's own limit), then — after sizing total supply at 1.15-1.6x that limit,
spread over sites by lognormal weights — with the real supplies. Demands are
`load_factor * lambda* * d0` (rounded to cents) and a final exact max flow is
stored as `max_flow_value` next to `total_demand`:

- `feasible`: `load_factor` in [0.55, 0.9]; the witness is the final max-flow
  plan (every demand arc saturated).
- `infeasible`: `load_factor` in [1.08, 1.3]; the certificate is the min-cut
  region `T` read off the residual graph, with
  `region_demand > region_supply + inbound_capacity`. Summing the balance rows
  over `T` refutes the model from rows and bounds alone. `T` is typically a
  whole under-connected region (a city behind a few bridges), not one starved
  node, so HiGHS presolve does not detect it — simplex has to work.
- `unknown`: `load_factor` in [0.85, 1.15], a natural instance on either side
  of the exact boundary; `max_flow_value >= total_demand` decides it.

## `generalized_flow` formulation

```text
flow[a] in [0, capacity[a]]                       (flow SENT on a)

sum_{out} flow - sum_{in} gain * flow <= supply[v]     supply node v
sum_{in} gain * flow - sum_{out} flow  = demand[v]     demand node v
sum_{in} gain * flow - sum_{out} flow  = 0             transit node v

minimize sum_a cost[a] * flow[a]
```

Gains decay exponentially in arc length times a lognormal line-quality factor,
calibrated per instance so the median best-route delivery efficiency to the
demand nodes is 72%-90%; they are stored to 4 digits and kept in
[0.5, 0.9995] (never exactly lossless). Gains below one destroy total
unimodularity: vertices are genuinely fractional, the classic hard family for
simplex. Every gain is below one, so no cycle amplifies flow.

### Feasibility control

A planted lossy routing — a shortest-path tree from all supply sites on loss
lengths `-log(gain)` with mild noise, routed with `_geo_tree_flows` so the flow
sent on each tree arc is what its head needs divided by the gain — is a concrete
feasible flow; capacities always cover it (provisioning 1.05-1.55x plus a tiered
floor).

- `feasible`: each site's supply is 1.05-1.6x what the plan draws from it; the
  plan is the stored witness.
- `infeasible`: let `efficiency[v]` be the best achievable product of gains
  from any site to `v` (Dijkstra on `-log(gain)`). Weighting node `v`'s row by
  `1 / efficiency[v]` makes every column coefficient nonpositive, so any
  feasible flow satisfies `sum_v demand[v] / efficiency[v] <= total supply`.
  Total supply is 3%-8% below that loss-adjusted requirement (and, whenever
  the losses allow, above the lossless total demand, so the naive supply >=
  demand check passes). Every site starts at its planted draw and the
  shortfall is taken in proportion to draw x out-degree^2 — the
  best-connected hubs run short, not a district's only source — and a local
  repair keeps every demand node deliverable under one step of bound
  propagation with 30% slack (total unchanged), so presolve rarely sees it.
  The repair keeps the total only up to rounding, site floors and early
  stops, so it is applied only while the repaired total stays below the
  requirement (small networks otherwise keep the plain cut). The certificate
  stores the potentials and both sides, the supply side being the total the
  model actually carries.
- `unknown`: every site holds a common reserve factor in [0.9, 1.1] of its
  planted draw (4% site noise), starved sites topped up: below 1 the routing
  must beat the planted (near-efficient, noisy) paths, above 1 it has slack.

## `time_expanded` formulation

A road network (`_geo_network`) gets integer travel times (1-4 periods,
distance / speed) and per-period capacities (lognormal, 1.8x on trunk roads).
Zones (12%-20% of nodes, by population) hold evacuees at time 0; shelters
(4%-8%, spread by a randomised farthest-point placement over low-density
nodes) take in a lognormal number per period; staging areas (20% of other
nodes) can hold a lognormal amount of traffic, and evacuees can wait at home
without limit. With horizon `H`:

```text
minimize    sum_{shelter s, t} t * intake[s,t] + travel_weight * sum road_cost * move
subject to  supply[v]*[t == 0] + arrivals(v,t) + wait(v,t-1)
                = departures(v,t) + wait(v,t) + intake(v,t)      every useful copy (v, t)
            0 <= move[a,t] <= road capacity, 0 <= wait <= holding, 0 <= intake <= rate
```

Only useful copies are generated: `(v, t)` exists iff a zone reaches `v` by
`t` and a shelter is reachable from `v` by `H` (zones too far from every
shelter are dropped), so no variable is dead on arrival. The objective is the
total evacuation time (quickest transshipment). The time-expanded LP is a
network LP but a long, layered and highly degenerate one: at 100k variables
HiGHS needs 100k-280k dual simplex iterations.

Feasibility uses the same exact machinery as `standard`: the largest
population scale `lambda*` that can be evacuated by `H` (Dinkelbach over exact
Dinic max flows on the time-expanded network), populations
`load_factor * lambda*` (feasible 0.6-0.9 with the max-flow plan as witness;
infeasible 1.06-1.25 with a **trapped space-time region** certificate — the
region's time-0 population exceeds the capacity of every road, waiting and
intake copy leaving it; unknown 0.85-1.15, decided by `max_flow_value` vs
`total_supply`).

Sizing: `H = clamp(round(2.5 target^0.25), 6, 48)`; the node count is redrawn
until the pruned variable count is within 8% of the target (closest of up to
twelve draws; tiny instances can miss by up to ~30%). Rows = useful node
copies (about a quarter of the columns).

## Sizing (`standard`, `generalized_flow`)

Variables equal the target exactly (= arcs). `n_nodes ~ target / U(3.2, 4.6)`
(adjusted so the bidirectional spanning tree fits); rows are one balance row
per node, i.e. about a quarter of the columns at every size. Targets below 2
round up to 2 and a target of 3 to 4. Build time is near-linear (the
generalized variant about a second at 100k arcs; the standard variant a few
seconds, dominated by the handful of exact max flows). Requests above
`NETWORK_FLOW_MAX_ARCS = 1_000_000` raise an `ArgumentError`.

## References

- Ahuja, R.K., Magnanti, T.L., Orlin, J.B. (1993). Network Flows: Theory,
  Algorithms, and Applications. Prentice Hall.
- Gale, D. (1957). A theorem on flows in networks. Pacific Journal of
  Mathematics 7(2).
- Klingman, D., Napier, A., Stutz, J. (1974). NETGEN: A program for generating
  large scale capacitated assignment, transportation, and minimum cost flow
  network problems. Management Science 20(5).
- Goldberg, A.V., Plotkin, S.A., Tardos, É. (1991). Combinatorial algorithms
  for the generalized circulation problem. Mathematics of Operations Research
  16(2).
- Dinic, E.A. (1970). Algorithm for solution of a problem of maximum flow in
  networks with power estimation. Soviet Mathematics Doklady 11.
- Dinkelbach, W. (1967). On nonlinear fractional programming. Management
  Science 13(7).
- Ford, L.R., Fulkerson, D.R. (1958). Constructing maximal dynamic flows from
  static flows. Operations Research 6(3).
- Hamacher, H.W., Tjandra, S.A. (2002). Mathematical modelling of evacuation
  problems: a state of the art. In Pedestrian and Evacuation Dynamics.
