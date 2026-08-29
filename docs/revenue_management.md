# Revenue Management

The `revenue_management` category generates continuous network
revenue-management LPs. Both variants allocate perishable capacity on a coherent
hub-and-spoke network, distinguish fare classes, and preserve the status requested
through a constructive witness or a mathematical infeasibility certificate.

## Variants

| Variant | Planning setting | Main decisions |
| --- | --- | --- |
| `standard` (default) | Deterministic network revenue management | Accepted demand by itinerary |
| `stochastic_overbooking` | Scenario-based show-ups with service recovery | Advance bookings, served customers, and denied customers in every scenario |

Both constructors use their own `MersenneTwister`. A fixed seed reproduces all
data without resetting or consuming Julia's process-global random stream.
`build_model` uses only stored data and is deterministic.

## Shared network and demand data

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

## Deterministic network model (`standard`)

### Sizing

There is one acceptance variable per product, so the delivered variable count is
exactly

```text
max(2, target_variables).
```

The two-variable minimum keeps a meaningful capacity-allocation model even for tiny
or non-positive requests. The number of resources is scale-dependent and is capped
at 80 to keep very large formulations network-oriented rather than nearly diagonal.

### Formulation

For products `j in P` and resources `r in R`, let `x[j]` be accepted demand,
`f[j]` its fare, `d[j]` its forecast demand, `l[j]` its contractual floor, and
`P(r)` the products consuming resource `r`:

```math
\max \sum_{j \in P} f_j x_j
```

subject to

```math
l_j \le x_j \le d_j \qquad j \in P,
```

```math
\sum_{j \in P(r)} x_j \le C_r \qquad r \in R.
```

Connecting itineraries consume one unit of capacity on both constituent legs.
This is the classic deterministic network LP used for bid-price and displacement-
cost analysis.

### Feasibility artifacts

A requested-feasible instance sets every acceptance to its contractual floor and
constructs each capacity above that point's resource load. This vector is stored
in `feasible_witness` and attached to the JuMP variables as start values. The
solver-independent helper
`SyntheticLPs._revenue_management_witness_is_valid(problem)` checks all bounds and
resource rows.

A requested-infeasible instance selects a leg, raises the floors of every product
using it, and places that leg's capacity strictly below the resulting mandatory
load. The stored `RevenueManagementCapacityCertificate` records the leg,
committed load, capacity, and positive excess. Since all feasible points must
satisfy `x[j] >= l[j]`, the contradiction survives any objective choice. The
helper `SyntheticLPs._revenue_management_certificate_is_valid(problem)` recomputes
the proof directly from the generated data.

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
