# Revenue Management

The `revenue_management` category generates continuous network
revenue-management LPs. Both variants allocate perishable capacity on a
hub-and-spoke network, distinguish fare classes, and preserve the requested
`feasible`/`infeasible` status through a constructive witness or a mathematical
infeasibility certificate.

## Variants

| Variant | Planning setting | Main decisions |
| --- | --- | --- |
| `standard` (default) | Choice-based network RM (sales-based LP, MNL choice) over a multi-hub banked schedule | Expected sales per product and no-purchase volume per market-day |
| `stochastic_overbooking` | Scenario-based show-ups with service recovery | Advance bookings, served customers, and denied customers in every scenario |

Both constructors use their own `MersenneTwister`. A fixed seed reproduces all
data without resetting or consuming Julia's process-global random stream.
`build_model` uses only stored data and is deterministic.

## Choice-based network model (`standard`)

The default variant is the **sales-based linear program (SBLP)** of Gallego,
Ratliff & Shebalov (2015) for network revenue management under
multinomial-logit (MNL) customer choice. It replaced an independent-demand
deterministic LP whose fare-class columns on the same itinerary were parallel:
HiGHS presolve merged them and kept only 5–32% of the columns at 10k–50k, with
80 rows at any size.

### Schedule and markets

- Airports on a 2500 × 2500 km map: 1–3 hubs near the center and many spokes.
  Each spoke is served from its nearest hub and, with probability 0.35, a second
  hub; hubs are fully connected. Routes operate in daily departure **banks**
  (2–4), more frequently from larger cities, over 1–14 days.
- A **market** is an origin–destination pair on one day, with a gravity-model
  size `Λ_m` (populations over distance, day-of-week factor, lognormal noise).
  Its **products** are its nonstop flights and one-stop connections (same-bank
  connection at a hub, at most four itineraries) in 3–5 fare classes
  (`Y, B, M, Q, V`). Markets are taken in a gravity-weighted random order
  (Efraimidis–Spirakis keys) until the variable target is reached.
- MNL attraction `v_j = exp(quality_class − β_m · fare / base_fare − 0.7 · stops
  − 0.8 · bank_gap + noise)`, with market-specific price sensitivity `β_m`;
  the no-purchase attraction `v0_m` makes the full-offer purchase probability
  35–75%.
- Flight capacities are aircraft sizes (50–300 seats, or multiples of 50) just
  above the planted plan's loads, so capacity binds and the LP must decide
  which classes to close.

### Formulation

```text
max   Σ_j fare_j sales_j
s.t.  Σ_{j∈m} sales_j + no_purchase_m = Λ_m              (market balance)
      sales_j − (v_j / v0_m) no_purchase_m ≤ 0            (one scale row per product)
      Σ_{j uses f} sales_j ≤ capacity_f                    (flight capacity)
      Σ_{j uses f} sales_j ≥ min_load_f, f contracted      (minimum-load contracts)
      sales, no_purchase ≥ 0
```

Variables are exactly `n_products + n_markets` (the last market-day is trimmed
so the count equals the target; targets below 2 give 2). Rows are
`n_products + n_markets + n_flights + n_contracts`, so rows grow one-for-one
with columns, and every product column has its own scale row. About 6% of the
flights carry a minimum-load contract (charter or public-service guarantees).

### Feasibility artifacts

- `feasible`: every market offers all products and sells a fraction
  `θ_m ∈ [0.45, 0.85]` of its full-offer MNL sales; with `θ ≤ 1` every scale row
  holds. Capacities sit above, and contracts at 70–95% of, the plan's flight
  loads. Stored as `RMChoiceWitness` (sales, no-purchase volumes, θ).
- `infeasible`: on one contracted flight `f`, every feeding market `m` can sell at
  most `Λ_m V_{m,f} / (V_{m,f} + v0_m)` of the flight's products (sum its scale rows
  and use the balance row). The contract is set 5–12% above the sum of these
  bounds (the aircraft is up-gauged if needed so the contract fits the cabin).
  Stored as `RMChoiceCertificate`. The proof combines the contract row with the
  balance and scale rows of every feeding market, so presolve does not see it.
- `unknown`: all contracts at an instance-wide tightness `U(0.9, 2.0)` times the
  plan's loads (up-gauging when needed): below one the plan meets them; above it,
  whether all contracts can be met together under the choice model, the shared
  connecting demand, and other flights' capacities is left to the instance
  (about 50–70% feasible in a 16-seed sample at 2k and 10k).

The variant does not use the shared hub-and-spoke helpers below; those now
serve `stochastic_overbooking` only (`src/problem_types/revenue_management/common.jl`).

## Shared network and demand data (`stochastic_overbooking`)

Capacity resources are directed legs in a compact hub-and-spoke network. Odd and
even resource indices form outbound and inbound legs for successive spokes. Every
leg first receives a local product; the remaining products are sampled as either
local trips or coherent two-leg spoke-hub-spoke connections. Thus every resource
appears in at least one itinerary, and a connection's first destination is the
second leg's origin.

Each product is stored as a `RevenueManagementProduct` containing its integer ID,
origin, destination, fare class, and consumed resource indices. The parallel
`product_resources` and `resource_products` fields provide both incidence
directions for convenient downstream use.

An instance samples one of three operating profiles:

- `regional_airline`: smaller aircraft, mostly local traffic, and moderate fares;
- `network_airline`: larger aircraft and a higher connecting-passenger share;
- `intercity_rail`: larger capacity, lower fares, and stronger local demand.

Economy, premium, and business products have different fare and demand scales.
Demand uses a capped log-normal distribution to retain skew without producing
pathological values. A sparse subset of products receives a positive contractual
floor representing protected allotments or group blocks.

## Stochastic network overbooking (`stochastic_overbooking`)

### Sizing and scenario mix

For `P` products and `S` scenarios, the variant creates:

```text
P booking variables + P*S served variables + P*S denied variables
    = P * (1 + 2*S) variables.
```

The dimension planner searches nearby product/scenario combinations instead of
silently dropping recourse variables. It uses 3--5 scenarios below 150 requested
variables, 4--8 below 1,200, and 6--12 thereafter. The smallest formulation is
two products and three scenarios, or 14 variables. At ordinary scales the selected
count is the closest representable count in the applicable band.

Scenario probabilities are positive and normalized. Show-up rates reflect both
fare-class behavior and one of three scenario profiles:

- `stable_business` has a narrow scenario range;
- `mixed_leisure` has moderately dispersed show-up rates;
- `disruption_prone` includes a deliberately low-show scenario.

Rates are kept in `[0.55, 0.995]`. Denied-service compensation and service
standards vary by fare class: higher classes receive larger compensation and a
smaller allowed denial fraction.

### Formulation

Let `x[j]` be first-stage bookings. For each scenario `s`, `served[j,s]` and
`denied[j,s]` allocate realized show-ups, `q[j,s]` is the show rate, `pi[s]` is the
scenario probability, `a[j]` is the product denial limit, and `K[s]` is the
aggregate denial cap. The objective maximizes expected realized service revenue
minus denied-service compensation:

```math
\max \sum_s \pi_s \sum_j
  \left(f_j served_{j,s} - c^{deny}_j denied_{j,s}\right).
```

Bookings retain the deterministic demand and commitment bounds:

```math
l_j \le x_j \le d_j.
```

Scenario recourse exactly accounts for realized show-ups:

```math
served_{j,s} + denied_{j,s} = q_{j,s}x_j.
```

Product-level and system-wide service promises are explicit:

```math
denied_{j,s} \le a_j q_{j,s}x_j,
```

```math
\sum_j denied_{j,s} \le K_s.
```

Served customers consume each itinerary leg in every scenario:

```math
\sum_{j \in P(r)} served_{j,s} \le C_r
\qquad r \in R,\ s \in S.
```

This is an LP with shared first-stage decisions and scenario-dependent continuous
recourse; it does not require integer relaxation.

### Feasibility artifacts

For feasible requests, the stored witness books exactly each product's commitment,
serves all corresponding show-ups, and denies nobody. Capacities are constructed
above the maximum scenario load of that point, making feasibility independent of
an optimizer or retry loop. The helper
`SyntheticLPs._stochastic_overbooking_witness_is_valid(problem)` checks bounds,
show-up balance, denial limits, aggregate service caps, and every scenario-leg
capacity row.

For infeasible requests, consider any product using a selected leg. Show-up
balance and its denial cap imply

```math
served_{j,s} \ge (1-a_j)q_{j,s}x_j
                 \ge (1-a_j)q_{j,s}l_j.
```

The generator selects the scenario with the largest sum of these mandatory served
loads and puts the selected leg's capacity strictly below that sum. The stored
`StochasticOverbookingCertificate` records the resource, scenario, mandatory
load, capacity, and positive excess. The helper
`SyntheticLPs._stochastic_overbooking_certificate_is_valid(problem)` verifies the
certificate without solving the LP.

For either variant, an `unknown` request resolves reproducibly to a feasible or
infeasible profile. The actual choice is recorded in `resolved_status`, and exactly
one corresponding audit artifact is present.
