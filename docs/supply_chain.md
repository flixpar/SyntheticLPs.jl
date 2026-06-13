# Supply Chain

`SupplyChainProblem` generates a facility-location and transportation model with geographic customer clusters, mode-specific route availability, fixed facility-opening costs, and capacity constraints.

## Overview

This generator represents strategic supply chain network design. A planner decides which facilities to open and how much demand to ship from open facilities to customers over available transportation modes. Costs include fixed facility costs and distance-based transportation costs.

## Generator Data and Sizing

The code comments describe variables as approximately facility-open variables plus shipment variables:

```text
n_facilities + n_facilities * n_customers * n_transport_modes * infrastructure_density
```

In practice, `target_variables` selects one of three size regimes rather than solving dimensions to match the target exactly:

- `target_variables <= 250`: `n_facilities` from `3:8`, `n_customers` from `15:35`, modes from `1:2`, grid dimensions `200-800`, infrastructure density roughly `0.7-1.0`, clustering factor roughly `0.25-0.85`.
- `target_variables <= 1000`: `n_facilities` from `6:18`, `n_customers` from `25:65`, modes from `2:3`, grid dimensions `800-2000`, infrastructure density roughly `0.5-0.9`, clustering factor roughly `0.2-0.7`.
- larger targets: `n_facilities` from `12:40`, `n_customers` from `60:200`, modes from `3:4`, grid dimensions `2000-5000`, infrastructure density roughly `0.4-0.8`, clustering factor roughly `0.15-0.55`.

Transport modes are sampled from `truck`, `rail`, `ship`, and `air`. Base transport costs use gamma distributions: truck `Gamma(4, 0.25)`, rail `Gamma(3, 0.2)`, ship `Gamma(2, 0.15)`, air `Gamma(6, 0.5)`.

Customers are drawn around cluster centers. Cluster weights come from a Dirichlet distribution. Facilities are more dispersed, but 40 percent are placed near a cluster center. Fixed facility costs are lognormal-scale costs correlated with market potential and location. Customer demands are lognormal multipliers applied to cluster-influenced base demand. Capacities are based on total demand per facility, a random capacity factor `1.2-2.2`, relative fixed cost, and a gamma multiplier.

Route availability depends on mode:

- Truck is almost always available before density scaling, with base probability `0.98`.
- Rail availability increases with distance, capped at `0.8`.
- Ship is available only when either endpoint lies near the bottom grid edge, otherwise probability 0.
- Air is more likely for long distances.

Mode capacities are generated as total-demand fractions using `mode_capacity_factor` from `0.25-0.65` and mode-specific gamma multipliers.

The struct stores:

- `n_facilities`, `n_customers`
- `transport_modes`
- `facility_locs`, `customer_locs`
- `cluster_centers`, `cluster_weights`
- `fixed_costs`, `demands`, `capacities`
- `transport_costs`, keyed by `(facility, customer, mode)` only for available routes
- `mode_capacities`
- `total_demand`

The constructor calls `Random.seed!(seed)`, resetting Julia's global RNG.

## LP Formulation

The implemented model is a mixed-integer linear model, not a pure LP, because facility-open variables are binary.

Sets:

- `F = {1, ..., n_facilities}` facilities
- `C = {1, ..., n_customers}` customers
- `M` selected transport modes
- `A = {(f,c,m): transport_costs has key (f,c,m)}` available route-mode arcs

Decision variables:

- `y_f in {0,1}`: whether facility `f` is open
- `x_{fcm} >= 0`: shipment from facility `f` to customer `c` by mode `m` on available arcs

Objective:

```math
\min \sum_{f \in F} F_f y_f + \sum_{(f,c,m) \in A} q_{fcm} x_{fcm}
```

Customer demand satisfaction:

```math
\sum_{(f,c,m) \in A: c \text{ fixed}} x_{fcm} \ge d_c \quad \forall c \in C
```

Facility capacity gated by opening:

```math
\sum_{(f,c,m) \in A: f \text{ fixed}} x_{fcm} \le K_f y_f \quad \forall f \in F
```

Mode capacity:

```math
\sum_{(f,c,m) \in A: m \text{ fixed}} x_{fcm} \le U_m \quad \forall m \in M
```

## Feasibility Controls

For `feasible`, the generator strengthens route connectivity and capacities after initial random data. It selects a fallback mode, preferring truck when present. For each customer, it ensures arcs from the `K` nearest facilities exist in the fallback mode, where `K = min(max(3, ceil(n_facilities / 3)), n_facilities)`. It then computes an approximate demand share for each facility from those nearest-facility links and raises capacities to at least `1.05` times that share. It also ensures the fallback mode can carry all demand with 5 percent slack, scales aggregate mode capacity if needed, and scales aggregate facility capacity if needed.

For `infeasible`, it creates a transport-capacity shortfall. It picks a desired total mode capacity ratio from `Uniform(0.7, 0.95)` times total demand and scales mode capacities downward if the current total exceeds that value. Since all demand must ship through modes, aggregate mode capacity below total demand makes the model infeasible.

For `unknown`, no special feasibility branch is applied. The randomly generated route availability, facility capacity, and mode capacity data are returned as generated and may be feasible or infeasible.

## Model Characteristics

The model has `n_facilities` binary variables plus one continuous shipment variable per available route-mode combination. Constraint count is `n_customers + n_facilities + length(transport_modes)`. Shipment variables appear in demand, facility, and mode rows. Route density is controlled by infrastructure availability and can be much less than the full facility-customer-mode product, especially for ship and air.

Because `y` is binary, this is a MILP. The LP relaxation would replace `y_f in {0,1}` with `0 <= y_f <= 1`, but the implementation uses JuMP `Bin`.

## Practical Notes

This generator is useful for network design, sparse transportation arcs, fixed-charge facility opening, and aggregate-capacity infeasibility. `target_variables` is only a regime selector; actual variable count can vary substantially with random dimensions and route density. The local variable `avg_density` is defined but not used in the constructor's sizing logic.
