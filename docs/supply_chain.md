# Supply Chain

The `supply_chain` category generates production–distribution network models.
Three variants share a **multi-echelon, multi-product, multi-period network
core** (plants → distribution centers → customers) and differ in the coupling
family they add on top of the flow-balance backbone; two further variants are
stand-alone generators.

| Variant | Model class | Key structure |
|---|---|---|
| `standard` (default) | MIP (binary DC opening), relaxed by default | multi-echelon network design with per-arc linking, truck/rail/intermodal linehaul with modal capacity |
| `carbon` | MIP, relaxed by default | the `standard` network under per-period carbon caps on production, linehaul, and last-mile emissions |
| `multi_product` | pure LP | 2–12 products, specialized plants with product-line capacity, lane bundle capacity shared by products |
| `network_planning` | pure LP | two-echelon multi-period production/inventory/shipment planning with sparse period-specific lanes and structural profiles |
| `single_source` | MIP, relaxed by default | single-period facility location with single-sourcing assignment |

All constructors use a local `MersenneTwister(seed)`; `build_model` is
deterministic.

The previous `standard`, `carbon`, and `multi_product` generators were
single-period facility-location models with a few hundred rows at 10k–100k
columns (rows = customers + facilities + modes) and, for `standard`, a
heuristic "capacity smoothing" feasibility step without a witness. They were
rebuilt on the shared network core described below.

## Shared network core (`standard`, `carbon`, `multi_product`)

### Data

- **Geography**: customers cluster in regions on a 100 × 100 map with lognormal
  sizes; candidate DCs sit near regions; plants are spread out. Customers grow
  with the target; DCs ≈ customers / 9 (+2), plants ≈ DCs / 3.
- **Products and periods**: periods grow slowly with the target (2 at tiny
  targets, 4 at 1k, 6 at 10k, 8 at 100k). `standard`/`carbon` carry 1–3 products
  ordered by every customer; `multi_product` 2–12 products, each customer
  ordering about 60%. Demand has product-specific seasonality and noise.
- **Lanes** (linehaul, plant → DC): each DC from its 2–3 nearest plants plus the
  nearest source of any product they miss (so every DC can receive every
  product). Truck lanes always exist; rail (distance > 25, probability 0.6) and
  intermodal (distance > 18, probability 0.5) where available (`standard` uses
  truck/rail below 2k variables, truck/rail/intermodal above; `carbon` always all
  three; `multi_product` truck only). Unit cost = plant–product production
  cost + mode terminal charge + mode rate × distance.
- **Arcs** (last mile, DC → customer): every customer's two nearest DCs, plus
  further near DCs (up to five) until the variable count reaches the target.
- **Plants** are specialized in `multi_product` (each makes a random ~45% of the
  products, every product has a source).

### Variables

- `open[d]` (binary, `standard`/`carbon` only): operate DC `d` (fixed cost over
  the horizon, scaled to its size);
- `ship[lane, product, period]` for every product the lane's plant makes;
- `stock[d, product, period]`: DC inventory;
- `deliver[arc, product, period]` for every product the arc's customer orders.

Exact count: `(design ? n_dcs : 0) + Σ_lanes |products(plant)| T + n_dcs K T +
Σ_arcs |products(customer)| T` (`_scn_num_variables`). The arc fill lands within
half an arc's worth (`K T / 2`) of the target; targets below 30 are treated as 30.

### Rows

```text
Σ_{lanes from p, k} resource_use[p,k] ship ≤ plant_capacity[p,t]               ∀ p, t
Σ_{lanes from p} ship[·,k,t] ≤ line_capacity[p,k]                ∀ p, k, t   (multi_product)
Σ_k ship[lane,k,t] ≤ lane_capacity[lane]                          ∀ lane, t  (multi_product)
Σ_{lanes of mode m} ship[·,·,t] ≤ mode_capacity[m,t]          ∀ rail/intermodal, t  (standard, carbon)
stock[d,k,t−1] + Σ_{lanes into d} ship[·,k,t] − Σ_{arcs from d} deliver[·,k,t] = stock[d,k,t]
Σ_{arcs from d, k} deliver ≤ dc_throughput[d] · open[d]                         ∀ d, t
Σ_k stock[d,k,t] ≤ dc_storage[d] · open[d]                                      ∀ d, t
Σ_{arcs into c} deliver[·,k,t] ≥ demand[c,k,t]                    ∀ c, ordered k, t
Σ_k deliver[a,k,t] ≤ (Σ_k demand[c,k,t]) · open[d(a)]                ∀ arc a, t  (design)
```

(`open ≡ 1` without design.) The per-arc linking rows are the disaggregated
formulation: under the default relaxation a fractional `open[d]` still has to
cover each customer's flow through `d`, so DC opening keeps its meaning instead
of being absorbed by one throughput row. Rows are ≈ 0.4–0.65 of the columns at
10k–100k, and HiGHS presolve keeps essentially all of them.

### Planted plan (shared by all statuses)

A random set of DCs is opened (each customer keeps at least one open DC);
deliveries split each customer's demand over its open DCs by distance weights;
DCs follow a cover-stock policy (stock = 10–45% of next period's outflow);
inbound flow is split over the DC's lanes carrying each product. Every capacity
(plant, line, lane, mode, DC throughput and storage) is then sized 8–40% above
the plan's usage; closed DCs get typical-size capacities. The plan is stored as
`SupplyChainNetworkWitness` for `feasible` requests and satisfies every row of
the unrelaxed model (the test file checks it with `primal_feasibility_report`).

## `standard`

- `feasible`: the planted plan.
- `infeasible`: one region (with at least four customers whenever the network
  has one — a one-customer region is refuted by presolve bound propagation
  alone) has its DCs (the union of its customers' arcs) lose
  throughput until their combined capacity is 8–18% below the region's
  peak-period demand (`SupplyChainRegionalCertificate`: region, customers, DCs,
  period, demand, throughput, margin). The proof sums the region's demand rows
  and those DCs' throughput rows with `open ≤ 1`; it needs simplex work.
- `unknown`: every capacity is multiplied by one network-wide supply factor
  `U(0.60, 1.05)` (`capacity_factor`); the network may or may not cope by
  rerouting, opening more DCs, and prebuilding stock (both outcomes occur over
  seed blocks).

## `carbon`

Emissions: plant-specific production intensity per unit, plus linehaul mode
factor (truck 0.10, intermodal 0.05, rail 0.03 per unit-km) × distance, plus
last-mile truck factor × distance. The horizon allowance `carbon_budget` is
issued per period in proportion to the planted plan's emission profile
(`period_budget`, no banking), one cap row per period. The planted plan weights
rail/intermodal lanes three times as heavily as trucks, so it is a low-emission
plan, and the truck-heavy cost optimum must shift modes and sources.

- `feasible`: budget 1.00–1.05 × the plan's emissions (active).
- `infeasible`: budget 7–15% below a valid emission lower bound
  (`SupplyChainCarbonCertificate`): each customer's demand costs at least its
  cheapest arc emission plus the cleanest lane into that arc's DC, minus a
  credit for initial DC stock (which needs no linehaul). Summing the period caps
  gives the contradiction.
- `unknown`: an active cap (0.97–1.05 × plan) with all capacities scaled by
  `U(0.60, 1.05)`. Caps drawn near the minimum achievable emissions were tried
  and rejected: on the barely-infeasible side HiGHS's dual simplex stalled or
  returned an unknown status.

## `multi_product`

- `feasible`: the planted plan.
- `infeasible`: the product lines of one product are cut so its cumulative
  supply through a late period — initial DC stock plus, for every capable plant
  and period, `min(line capacity, plant capacity / resource use)` — is 10–20%
  short of its cumulative demand (`SupplyChainProductCertificate`).
- `unknown`: capacities scaled by `U(0.60, 1.05)` (`capacity_factor`).

## `network_planning`

Multi-period, multi-product two-echelon planning LP: plants produce under
product and shared resource capacity, hold inventory, and ship on sparse,
period-specific plant → customer arcs with exact (equality) demand. Three
structural profiles (`:regional_stable`, `:seasonal_prebuild`, `:disruption`)
are selected by `seed mod 3`. Exact variable count
`2 · n_plants · n_products · n_periods + length(shipment_arcs)`; targets above
1,000,000 raise `ArgumentError`.

Dimension search: the arc degree per demand node is an **absolute** range
(3–5 regional, 2.8–4.6 seasonal, 2.6–4.2 disruption), plants grow with the
target (≈15 at 1k, ≈30 at 10k, ≈55 at 100k), and every demand node first
receives its two best lanes. The old search measured density as a fraction of
the plant count, so at 50k it chose 3 plants and 4,868 customers with ≈1.1
lanes per demand row; almost every demand row was a singleton and HiGHS
presolve removed the whole model.

- `feasible`: planted production/inventory/shipment plan
  (`NetworkPlanningWitness`).
- `infeasible` (default, 70%): a network-wide capacity crunch — the shared
  resource capacity of every plant through period τ is scaled to 78–92% of the
  resource the cumulative demand of all products provably needs
  (`NetworkPlanningResourceCertificate`); otherwise a single product's
  cumulative supply cut (`NetworkPlanningInfeasibilityCertificate`).
- `unknown`: a correlated network-wide supply factor `U(0.66, 1.14)` with small
  plant/product/period effects; local lane service is preserved so no singleton
  demand cut arises (about 50–60% feasible).

## `single_source`

Single-period capacitated facility location with single sourcing: open
facilities `y`, assign each customer to exactly one facility `z[f,c]`, and
ship over available modes `x[(f,c,m)] ≤ demand[c] · z[f,c]`. Feasible requests
plant an explicit capacity-respecting single-source assignment; infeasible
requests drive total facility capacity below total demand with a margin.

Customers are generated one at a time (location, demand, and lane availability
per facility and mode), stopping at the customer count whose exact column total
`n_facilities·(1 + n_customers) + n_lanes` is closest to the target — sizes
land within ~2% of the target from a few hundred variables up (previously
0.86–0.98× because the lane density was only estimated).

## Measured behavior (wave-2 audit, seeds 0–1, HiGHS dual simplex, 60 s limit)

| Variant | Target | Cols | Rows | Presolve kept (cols/rows) | Statuses |
|---|---|---|---|---|---|
| `standard` | 10k | 10,009 | ≈5.9–6.0k | 1.00 / 1.00 | feasible OPTIMAL; infeasible needs 0.9k–5.9k iterations |
| `standard` | 100k | ≈100k | ≈62k | 1.00 / 1.00 | |
| `carbon` | 10k | 10,009 | ≈5.9–6.0k | 1.00 / 1.00 | infeasible needs ≈9–10k iterations |
| `carbon` | 100k | ≈100k | ≈62k | 1.00 / 1.00 | hits the 60 s limit |
| `multi_product` | 10k | ≈10k | ≈4.2k | 1.00 / 0.99 | |
| `multi_product` | 100k | 100k | ≈39k | 1.00 / 0.99 | feasible solves in 20–30 s |
| `network_planning` | 10k | 10,000 | ≈2.9–3.2k | 0.97 / 0.84–0.90 | |
| `network_planning` | 100k | 100,000 | ≈26–29k | 0.94–0.97 / 0.9 | was presolved to empty at 50k |
