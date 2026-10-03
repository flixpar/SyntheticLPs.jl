# Facility Location

The `facility_location` category generates distribution-network design
instances: which sites to open (and how big), and how customer demand is routed
through them. All three variants are MIPs whose defining structure survives the
default `relax_integer=true` because they use strong (disaggregated) linking.

## Variants

| Variant | Application | Natural formulation |
| --- | --- | --- |
| `standard` (default) | Budgeted capacitated facility location (CFLP) | Dense shipments, aggregate capacity, strong `x ≤ d·y` linking, budget row |
| `p_median` | Capacitated p-median (CPMP) | Exactly `p` sites, single assignment, `y ≤ z` linking, demand-weighted capacity |
| `two_echelon` | Plant → DC → customer network with DC sizing | Sparse nearest-site lanes, discrete size ladder, cross-dock conservation, strong linking |

Every variant stores a typed `feasible_witness` for `feasible` requests and a
typed `infeasibility_certificate` for `infeasible` ones (neither for
`unknown`). Neither mechanism is a single impossible row: each infeasibility
proof aggregates many demand/assignment rows with capacity rows, so HiGHS
presolve does not detect it and simplex has to work.

## `standard`

Facilities are uniform over a square region, customers clustered in towns,
demands log-normal, shipping cost = distance × per-km rate × lane noise, fixed
costs grow with capacity and location (OR-Library `cap`-style).

Sizing: `total = F · (C + 1)` with a sampled customer/facility ratio
`r ∈ 4..15`, `F = max(2, round(sqrt(target / r)))`,
`C = max(1, round(target / F) − 1)` — within `F/2` of the target, no cap.

```text
min  Σ_w fixed_w y_w + Σ_{w,c} ship_{w,c} x_{w,c}
s.t. Σ_w x_{w,c} ≥ d_c                    ∀c   (demand)
     Σ_c x_{w,c} ≤ cap_w y_w              ∀w   (capacity)
     x_{w,c} ≤ d_c y_w                    ∀w,c (strong linking)
     Σ_w fixed_w y_w ≤ budget                  (budget)
     y ∈ {0,1}, x ≥ 0
```

Rows ≈ columns. The strong linking rows (Cornuéjols, Sridharan & Thizy 1991)
keep `y` binding in the relaxation; the aggregate-only model lets `y_w` shrink
to `throughput / cap_w`.

- `feasible`: capacities lifted to ≥ 1.05× demand if needed; a greedy
  capacity/cost-ordered subset covering demand is opened, the budget is ≥
  1.02–1.25× its cost, and customers are served from their nearest open
  facilities with spare capacity → `FacilityLocationWitness(open, shipments)`.
- `infeasible`: budget = 75–95% of the fractional-knapsack cost of reaching
  total demand → `FacilityBudgetCertificate(budget, fundable_capacity,
  total_demand)`; summing demand and capacity rows and bounding `Σ cap_w y_w`
  by the budget row gives `total_demand ≤ fundable_capacity`, false.
- `unknown`: budget 60–95% of total fixed cost and capacity 1.3–2.0× demand as
  drawn — usually feasible, sometimes not.

## `p_median`

Capacitated p-median (Osman–Christofides / Lorena–Senne family). Sites uniform,
customers clustered, log-normal demands, Euclidean distances.

Sizing: `total = F · (C + 1)` with ratio `r ∈ 1..3` (CPMP benchmarks use about
as many candidate sites as customers) and `p ∈ [F/10, F/4]`.

```text
min  Σ_{w,c} dist_{w,c} d_c y_{w,c}
s.t. Σ_w y_{w,c} = 1                      ∀c
     y_{w,c} ≤ z_w                        ∀w,c
     Σ_w z_w = p
     Σ_c d_c y_{w,c} ≤ Q_w z_w            ∀w
     z, y ∈ {0,1}
```

Capacities `Q_w = total_demand / p · ρ · U(0.8, 1.25)` with tightness
`ρ ∈ [0.8, 1.3]` (≈ 0.83 is the LP threshold).

- `feasible`: greedy weighted p-median seeds are opened, every customer goes to
  its nearest planted site, and a planted site whose load exceeds its drawn
  capacity is expanded to 1.02–1.12× that load (capacities stay tight) →
  `PMedianWitness(open, assignment)`.
- `infeasible`: capacities scaled so the `p` largest sum to
  `total_demand / (1.05..1.25)` → `PMedianCapacityCertificate(top_p_capacity,
  total_demand)`.
- `unknown`: `ρ` as drawn; the integer model additionally faces a bin-packing
  question.

## `two_echelon`

Plants (suppliers) in a few manufacturing zones, candidate DCs near metro
clusters (35% rural), customers in metros. Lanes are sparse and local: each
customer has `K_c ∈ 3..6` delivery lanes to its nearest DCs, each DC `L ∈ 2..3`
inbound lanes from its nearest plants (bucket-grid nearest-site search, so the
build is near-linear). Each DC has a size ladder of `K ∈ {3,4}` capacities
scaled to its forecast catchment, with concave-plus-noise installation costs.
Inbound (full-truckload) and delivery (less-than-truckload) costs are
distance-based; handling cost is charged per unit delivered.

Sizing: `total = W(1 + K) + W·L + Σ_c K_c`, exact for every target above ~15
(customer count and per-customer lane counts are solved for). No size cap
(previously silently capped at 20,600 columns).

```text
min  Σ fixed_w y_w + Σ sizecost_{w,k} z_{w,k} + Σ in_l f1_l + Σ (out_l + handling_w) f2_l
s.t. Σ_k z_{w,k} = y_w                                    ∀w
     Σ_{l∈in(w)} f1_l ≤ Σ_k cap_{w,k} z_{w,k}             ∀w (throughput)
     Σ_{l∈in(w)} f1_l = Σ_{l∈out(w)} f2_l                 ∀w (cross-dock)
     Σ_{l∈out(s)} f1_l ≤ supply_s                         ∀s
     Σ_{l∈lanes(c)} f2_l ≥ d_c                            ∀c
     f2_l ≤ d_c y_w        for every delivery lane l = (w,c)
```

- `feasible`: 50–80% of DCs open (plus each customer's nearest DC if none of
  its lanes is open); customers fully served by their nearest open DC; each
  open DC buys the smallest size covering its throughput with a 5–15% margin;
  each DC is replenished from its nearest plant, whose capacity is lifted to
  1.1–1.3× its load where needed → `TwoEchelonWitness(open, size_choice,
  supply_flow, delivery_flow)`.
- `infeasible`: a compact region `R` (4–12% of customers around a random
  customer) gets a 30–80% demand surge and zoning limits on the DCs `N(R)`
  that can reach it, so `Σ_R d ≥ 1.1–1.3 × Σ_{N(R)} max_k cap` →
  `TwoEchelonRegionalDeficit(customers, warehouses, region_demand,
  max_capacity)`. The proof sums `|R|` demand rows and the throughput,
  cross-dock and size rows of `N(R)` — a Hall-type regional deficit, not a
  single row.
- `unknown`: ladders and plant capacities are planned against a forecast;
  realized demand has 1–3 regional shocks of 1.2–2.8×. About 80% of instances
  are feasible across seeds and sizes.

## Notes

- With the default `relax_integer=true`, `y`, `z` (and `p_median`'s `y`) are
  relaxed; the strong linking rows are what keep the relaxation non-trivial.
- All randomness is drawn from a constructor-local `MersenneTwister(seed)`.
- Tests: `test/problem_types/facility_location.jl` checks the sizing formulas,
  lane structure, witnesses row by row via `primal_feasibility_report`, the
  certificate arithmetic, and HiGHS contracts.
