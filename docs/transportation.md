# Transportation

Distribution LPs on sparse geographic lane networks: shipping from plants and
distribution centres to many customers over the lanes a real freight network
contracts — each customer's nearest sources plus occasional long-haul lanes —
instead of a complete bipartite graph between a few dozen nodes. Rows therefore
grow with the instance (one per customer and source, roughly a fifth of the
columns), and every variant scales from a handful of lanes to the documented
1,000,000-variable cap (`TRANSPORTATION_MAX_VARIABLES`; larger requests raise
an `ArgumentError`).

All variants use a constructor-local RNG (`build_model` does no sampling) and
place every feasibility profile against an **exact** boundary computed with the
`network_flow` max-flow machinery.

## Variants

| Variant | Model class | Key structure |
|---|---|---|
| `standard` (default) | LP (TU) | capacitated transportation: supply rows, demand rows, truck-allotment lane caps as bounds |
| `fixed_charge` | MIP (strong LP relaxation) | lane opening `y` with strong linking `x <= min(s, d, u) y` and per-source lane-count rows |
| `emission_constrained` | LP (non-TU) | truck/rail/intermodal options per lane, rail-terminal rows, regional and corporate CO2 caps |
| `transshipment` | LP (TU) | plants -> DCs -> customers plus direct lanes; DC conservation and throughput rows |

The former `balanced` and `capacitated` variants were removed: both were
complete-bipartite TU transportation LPs differing from `standard` only by an
equality supply row or per-lane capacity rows, and `standard` now carries
lane capacities (as bounds) itself.

## Shared data model (`common.jl`)

- **Geography**: sources and customers are drawn from one `_geo_positions`
  population (clustered metropolitan areas, uniform coverage, or a corridor;
  region side `12 * sqrt(n)`), so plants sit among their customers.
- **Lanes** (`_tp_lanes`): every customer gets a lognormal number of lanes
  (at least 2) to its nearest sources, scaled by its activity weight^0.3 (big
  customers contract more carriers); with probability 0.25 one lane is a
  long-haul lane to a farther source. The lane count equals the budget
  exactly.
- **Demand** follows a heavy-tailed activity weight (mean 100 units nominal).
- **Supply** (`_tp_market_supply`): each source's capacity is its historical
  market (half of each customer's demand attributed to its nearest source, half
  spread over its other lanes) times a lognormal build factor (median 1.5) and
  2-6 **regional under-build shocks** (discs in which plants were built at
  45%-80% of their market). Total supply comfortably exceeds demand; shortages
  are regional.
- **Costs**: landed cost = source production cost (lognormal, ~20) + distance
  x freight rate x lognormal lane noise + handling.

### Exact feasibility boundary

The lane network is mapped onto a max-flow network (super source -> sources at
their supply, lanes at their caps, customers -> super sink at their demand),
and `_network_flow_max_scale` computes the exact largest uniform demand scale
`lambda*` (Dinkelbach min-ratio-cut iterations over exact Dinic max flows —
by Gale's theorem `lambda* = min_T (supply(T) + cap_in(T)) / demand(T)`).
Demands are `load_factor * lambda*` times the nominal profile; a final max flow
on the final data gives the witness (feasible) or the min-cut region
certificate (infeasible), and `max_flow_value` vs `total_demand` decides
`unknown`.

## `standard`

```text
minimize    sum_l cost[l] x[l]
subject to  sum_{l from i} x[l] <= supply[i]     every source i
            sum_{l to j}   x[l] >= demand[j]     every customer j
            0 <= x[l] <= capacity[l]             (bound only on capped lanes)
```

Each customer's primary lane (to its nearest source) is uncapped; 30%-70% of
the others carry truck allotments of 25%-100% of the customer's nominal demand.

- `feasible`: `load_factor` in [0.6, 0.92]; witness = the exact max-flow plan.
- `infeasible`: `load_factor` in [1.06, 1.25]; certificate
  (`TransportationCutCertificate`) = a region of sources and customers whose
  demand exceeds the region's supply plus the caps of the lanes reaching in
  from outside. It is typically an under-built region of dozens of sources and
  hundreds of customers (sometimes the whole network when the aggregate is
  short) — never detectable by presolve's single-row reasoning.
- `unknown`: `load_factor` in [0.85, 1.15].

Sizing: variables = lanes = `max(target, 2)`; rows = `n_sources + n_customers`.

## `fixed_charge`

```text
minimize    sum_l unit_cost[l] x[l] + fixed_cost[l] y[l]
subject to  supply rows, demand rows (as in standard)
            x[l] <= link_bound[l] * y[l]                 every lane l
            sum_{l from i} y[l] <= max_lanes[i]          every source i
            x >= 0, y in {0,1}
```

`link_bound = min(supply[i], demand[j], lane_capacity[l])` is the strong
linking coefficient, so the default LP relaxation does not collapse to a
transportation LP with rescaled costs (as the previous big-M version did):
the lane-count rows become weighted capacity rows
`sum_l x[l] / link_bound[l] <= max_lanes[i]` with heterogeneous coefficients,
and every `y` sits in two rows. Fixed cost = lane setup (lognormal, median
150) + a distance-proportional deadhead term.

- `feasible`: `load_factor` in [0.6, 0.88]; `max_lanes` = 1.1-1.5x the lanes
  the max-flow plan uses; witness = the plan with its lanes opened — feasible
  for the unrelaxed MIP too.
- `infeasible`: `load_factor` in [1.06, 1.25]; the region certificate holds in
  the relaxation because each lane cap binds through `x <= link_bound * y`,
  `y <= 1`.
- `unknown`: `load_factor` in [0.85, 1.1] and `max_lanes` 0.75-1.3x the
  plan's usage (the budgets may or may not leave enough routing flexibility).

Sizing: variables = `2 * max(round(target / 2), 2)`; rows =
`2 * n_sources + n_customers + n_lanes`.

## `emission_constrained`

Each lane offers truck (always), rail (source has a siding, lane long enough)
and intermodal (longer lanes). Per unit: truck costs `rate * d + 2` and emits
`0.062 d`; rail costs `0.45 rate d + 8` and emits `0.022 d + 0.3`; intermodal
costs `0.6 rate d + 5` and emits `0.035 d + 0.6` (lognormal noise on all).

```text
minimize    sum_k cost[k] x[k]                         k = (lane, mode) option
subject to  supply rows, demand rows
            sum_{rail k from i} x[k] <= rail_capacity[i]           sources with rail
            sum_{k from region r} emission[k] x[k] <= region_cap[r]   2-12 sales regions
            sum_k emission[k] x[k] <= global_cap
```

The shipping part is always feasible (`load_factor` 0.6-0.95 of the exact
`lambda*`); the caps decide. The planted plan ships the max-flow plan on each
lane's lowest-emission mode; rail-terminal capacities are 1.1-1.5x its rail
volume.

- `feasible`: region caps 1.03-1.25x and the global cap 1.0-1.08x the plan's
  emissions; the plan is the witness.
- `infeasible`: the global cap is 85%-95% of the demand-weighted minimum
  emission rate `sum_j demand[j] * min_rate[j]`, a lower bound on any plan's
  emissions (`EmissionTransportationCertificate`). The emission row alone is
  satisfiable; only its combination with the demand rows refutes the model.
- `unknown`: the global cap is drawn across the lower 70% of the gap between
  that lower bound and the plan's emissions (the true minimum lies in the
  gap), region caps 0.9-1.2x the plan's.

Sizing: variables = options = `max(target, 2)` exactly (about `target / 1.7`
lanes, optional modes added/removed at random to hit the count); rows =
`n_sources + n_customers + (#sources with a rail option) + n_regions + 1`.

## `transshipment`

Plants -> DCs -> customers, plus direct full-truckload lanes for the largest
customers. Each DC is fed by 2-5 of its nearest plants; each customer by 2-4
of its nearest DCs. DCs are placed at high-activity nodes.

```text
minimize    sum cost * flow (inbound, outbound, direct lanes)
subject to  sum_{inbound+direct from p} flow <= supply[p]         plants
            sum_{inbound to h} flow = sum_{outbound from h} flow   DCs
            sum_{inbound to h} flow <= throughput[h]               DCs
            sum_{outbound+direct to c} flow >= demand[c]           customers
            lane caps as bounds (40% of linehaul lanes, all direct lanes)
```

DC throughput is market-sized with regional under-build shocks; plants hold
1.3-1.6x nominal demand. Linehaul runs at half the per-km rate, final mile at
1.4x plus a DC handling fee. The exact boundary is computed on the DC-split
network (DC receiving side -> shipping side arc at the throughput); the
infeasible certificate (`TransshipmentCutCertificate`) lists the region's
plants, DC sides and customers and every capacitated entry into it
(linehaul lanes, direct lanes, DC throughput rows). Load factors as in
`standard`.

Sizing: variables = lanes = `max(target, 6)` (about 7% inbound, 8% direct,
the rest outbound); rows = `n_plants + 2 * n_dcs + n_customers`.

## References

- Hitchcock, F.L. (1941). The distribution of a product from several sources
  to numerous localities. Journal of Mathematics and Physics 20.
- Gale, D. (1957). A theorem on flows in networks. Pacific Journal of
  Mathematics 7(2).
- Balinski, M.L. (1961). Fixed-cost transportation problems. Naval Research
  Logistics Quarterly 8(1).
- McKinnon, A., Piecyk, M. (2011). Measuring and managing CO2 emissions of
  European chemical transport. (Emission factors by mode.)
