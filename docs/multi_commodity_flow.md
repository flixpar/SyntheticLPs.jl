# Multi-Commodity Flow

Commodities competing for shared arc capacity on sparse geographic networks.
The bundle (shared-capacity) rows couple every commodity, so — unlike a
single-commodity network LP — the constraint matrix is not totally unimodular
and simplex has to trade capacity between commodities. Both variants build on
the shared network machinery of `network_flow` (`geo_network.jl`) and use a
constructor-local RNG (`build_model` does no sampling).

## Variants

| Variant | Model class | Commodities | Key structure |
|---|---|---|---|
| `standard` (default) | LP | origin-aggregated (one origin, many gravity destinations) | min-cost routing, bundle rows `sum_k x[a,k] <= u[a]` |
| `binary_capacity` | MIP (strong LP relaxation) | origin-destination pairs | 3 capacity modules per arc, module choice rows, strong linking `x[a,k] <= min(d_k, U_a) sum_m y[a,m]` |

`integer_flow` was removed: its general-integer arc flows disappear under the
default `relax_integer=true`, leaving an LP structurally identical to
`standard`.

## Shared data model

- **Network**: `_geo_network` — strongly connected, 3.2-4.6 arcs per node, an
  exact arc budget, dead ends closed by a second link, region side
  `12 sqrt(n)`, geography `:clustered` / `:uniform` / `:corridor`.
- **Commodities**: origins (ports, plants, hubs) drawn by heavy-tailed activity
  weight; destinations drawn without replacement with gravity weights
  `w_v / (1 + dist / L)^1.5` (`L` = 5x the median arc length); demands
  `w_o^0.5 w_v / (1 + dist / L)^1.5`, scaled to a mean of 50 units.
- **Costs**: arc length x arc noise (15% cheaper on trunk arcs for
  `standard`) x a lognormal commodity value factor, plus a small handling
  charge — commodities differ, so the LP is not symmetric across them.
- **Planted routing**: every commodity is routed along a shortest-path tree on
  its own noisy lengths (`_geo_dijkstra` + `_geo_tree_flows`); the aggregate
  per-arc load sizes the capacities.

### Metric-inequality certificates

For any arc lengths `l >= 0`, a feasible routing satisfies (Onaga-Kakusho,
the "Japanese theorem")

```text
sum_a capacity[a] * l[a]  >=  sum_k sum_v demand[k][v] * dist_l(origin[k], v)
```

because each commodity's flow decomposes into paths no shorter than the
shortest one. An infeasible instance stores
`MultiCommodityFlowMetricCertificate(mode, lengths, region, capacity_length,
required)` with `capacity_length < required`:

- `:length` (default, ~60%): `l` = geographic arc length — a network-wide
  capacity shortage that only shows when all commodities' path lengths are
  weighed together;
- `:regional_cut` (~40%): `l = 1` on the arcs leaving a region of 5%-25% of
  the nodes around the largest origin — the region's exit capacity is below
  the demand that must leave it.

Capacities are squeezed to 80%-93% of the requirement (`_mcf_squeeze!`),
while `_mcf_local_repair!` keeps every node open (its in-arcs carry 1.15x the
demand ending there, an origin's out-arcs 1.15x the demand leaving it).
Together with `_geo_network` closing dead ends, this keeps infeasibility out of
presolve's single-row / doubleton reach: the HiGHS presolve keeps the whole
model and simplex has to prove infeasibility. (On tiny networks, where
destinations sit a hop or two from their origins, the repair slack is relaxed
so the certificate can separate at all.)

## `standard`

```text
minimize    sum_{a,k} cost[a,k] x[a,k]
subject to  sum_k x[a,k] <= capacity[a]                       every arc
            sum_out x[.,k] - sum_in x[.,k] = b[v,k]           every node, commodity
            x >= 0
```

`b[origin_k, k]` = commodity k's total demand, `-demand` at its destinations,
0 elsewhere. Each commodity serves 10%-35% of the nodes. Capacities are
`max(floor, rho * load)` with a tiered lognormal floor (trunk arcs 1.6x).

- `feasible`: `rho` in [1.05, 1.5]; witness = the planted routing.
- `infeasible`: as `feasible`, then a metric certificate is enforced.
- `unknown`: capacities as `feasible`, then every demand grows by a common
  factor in [1.0, 1.15] (traffic growth since provisioning): whether the
  commodities can reroute through the remaining spare capacity is left open.

Sizing: `n_commodities = clamp(round(0.4 * target^0.35), 2, 60)` (about 4 at
1k, 10 at 10k, 22 at 100k), `n_arcs ~ target / n_commodities`; variables
`n_arcs * n_commodities` (within half a commodity of the target); rows
`n_nodes * n_commodities + n_arcs`.

## `binary_capacity`

Multicommodity capacitated network design with modules (the strong MCND
formulation):

```text
minimize    sum routing_cost[a,k] x[a,k] + sum module_cost[a,m] y[a,m]
subject to  conservation per (node, OD commodity)
            sum_k x[a,k] <= sum_m module_capacity[a,m] y[a,m]          every arc
            sum_m y[a,m] <= 1                                          every arc
            x[a,k] <= min(demand[k], max_m module_capacity[a,m]) * sum_m y[a,m]
            x >= 0, y in {0,1}
```

Modules are 1x, 2.5x and 6x a lognormal base unit (trunk arcs 1.6x); module
cost = (length + 1) x capacity^0.7 / 2 + an installation charge (economies of
scale). The strong linking rows are what keep the LP relaxation meaningful:
without them a relaxed design installs a sliver of a module per unit of flow.

- `feasible`: on every loaded arc the base unit is raised so the largest module
  covers 1.05-1.5x the planted load; witness = planted routing plus the
  smallest adequate module per used arc (MIP-feasible).
- `infeasible`: metric certificate on the LARGEST module capacities (valid in
  the relaxation because `sum_m y <= 1`); all modules of a squeezed arc are
  rescaled together.
- `unknown`: modules as `feasible`, demands grown by a common factor in
  [1.0, 1.8] (the largest modules leave more headroom).

Sizing: `n_commodities = clamp(round(0.5 * target^0.3), 3, 40)`,
`n_arcs ~ target / (n_commodities + 3)`; variables
`n_arcs * (n_commodities + 3)`; rows
`n_nodes * n_commodities + 2 n_arcs + n_arcs * n_commodities`.

Both variants reject targets above `MCF_MAX_VARIABLES = 1_000_000` with an
`ArgumentError`.

## References

- Ahuja, R.K., Magnanti, T.L., Orlin, J.B. (1993). Network Flows, ch. 17.
- Onaga, K., Kakusho, O. (1971). On feasibility conditions of multicommodity
  flows in networks. IEEE Transactions on Circuit Theory 18(4).
- Iri, M. (1971). On an extension of the maximum-flow minimum-cut theorem to
  multicommodity flows. Journal of the Operations Research Society of Japan
  13(3).
- Gendron, B., Crainic, T.G., Frangioni, A. (1999). Multicommodity capacitated
  network design. In Telecommunications Network Planning, Springer.
