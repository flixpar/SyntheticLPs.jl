# Inventory

`InventoryProblem` generates a multi-period production and inventory planning model with optional backlogging, capacity limits, stochastic demand patterns, and controlled feasibility.

## Overview

This generator represents single-item inventory control over a planning horizon. A planner chooses production in each period, carries inventory when production exceeds demand, and may incur backlog costs if backorders are allowed. When backlogging is disabled, every prefix of demand must be coverable by initial inventory plus cumulative production capacity.

## Generator Data and Sizing

The generator first selects a scale from `target_variables`:

- `<= 250`: small
- `<= 1000`: medium
- larger: large

Backlogging is enabled with probability `0.2`, `0.4`, or `0.6` for small, medium, or large instances. The horizon is chosen from the variable-budget formula:

- With backlogging: variables are `x[1:T]`, `I_plus[0:T]`, and `I_minus[0:T]`, so `3T + 2`.
- Without backlogging: variables are `x[1:T]` and `I[0:T]`, so `2T + 1`.

The initial `n_periods` is rounded from that formula, bounded to `[2, 5000]`, then adjusted for up to 10 iterations until the variable count is within 10 percent of the target or cannot improve.

Scale-specific distributions:

- Small: production capacity `Uniform(50, 500)`, demand base `Uniform(10, 100)`, demand volatility `0.2-0.5`, initial inventory `10-50` percent of average demand, production cost base `10-100`, production cost spread `0.1-0.3`, annual holding rate `0.05-0.25` divided by 12.
- Medium: production capacity `Uniform(200, 2000)`, demand base `Uniform(50, 1000)`, demand volatility `0.15-0.4`, initial inventory `5-40` percent of average demand, production cost base `5-200`, production cost spread `0.05-0.25`, annual holding rate `0.03-0.20` divided by 12.
- Large: production capacity `Uniform(1000, 50000)`, demand base `Uniform(100, 10000)`, demand volatility `0.1-0.3`, initial inventory `2-30` percent of average demand, production cost base `1-500`, production cost spread `0.02-0.20`, annual holding rate `0.01-0.15` divided by 12.

Demands are sampled from a normal distribution centered at the demand range midpoint with standard deviation one quarter of the range, then clamped. The generator may add annual, weekly, and quarterly sinusoidal seasonality; exponential trends in production or holding costs; and occasional demand disruptions from a Poisson count. Backlog costs equal production costs multiplied by `Uniform(1.5, 5.0)`.

The struct stores:

- `n_periods`
- `prod_capacity`
- `initial_inventory`
- `backlog_allowed`
- `demands`
- `production_costs`
- `holding_costs`
- `backlog_costs`

The constructor calls `Random.seed!(seed)`, resetting Julia's global RNG.

## LP Formulation

Sets:

- `T = {1, ..., n_periods}` periods

Decision variables without backlogging:

- `x_t >= 0`: production in period `t`
- `I_t >= 0`: ending inventory in period `t`, including `I_0`

Objective without backlogging:

```math
\min \sum_{t \in T} p_t x_t + h_t I_t
```

Constraints without backlogging:

```math
I_0 = I^{init}
```

```math
I_{t-1} + x_t - d_t = I_t \quad \forall t \in T
```

```math
x_t \le C \quad \forall t \in T
```

Decision variables with backlogging:

- `x_t >= 0`: production in period `t`
- `I^+_t >= 0`: positive ending inventory
- `I^-_t >= 0`: backlog

Objective with backlogging:

```math
\min \sum_{t \in T} p_t x_t + h_t I^+_t + b_t I^-_t
```

Constraints with backlogging:

```math
I^+_0 = I^{init}
```

```math
I^-_0 = 0
```

```math
I^+_{t-1} - I^-_{t-1} + x_t - d_t = I^+_t - I^-_t \quad \forall t \in T
```

```math
x_t \le C \quad \forall t \in T
```

The implemented model is continuous. Production, inventory, and backlog can be fractional.

## Feasibility Controls

The constructor maps `feasibility_status` to `:feasible`, `:infeasible`, or `:all`. For `unknown`, it leaves `:all` and applies no feasibility enforcement branch, so the initially generated stochastic instance is returned after base generation.

For feasible instances, if backlogging is disabled, the generator may take realistic operator actions: increase production capacity by `10-25` percent, raise initial inventory with safety stock, enable backlogging, or smooth peak demands while slightly increasing capacity. It then performs a surgical feasibility pass. If no-backlog cumulative demand has a prefix shortfall, it computes the minimum capacity needed per prefix and either raises capacity, adds enough initial inventory, or enables backlogging.

For infeasible instances, the generator forces `backlog_allowed = false`, then creates one of several disruptions: sustained high demand, reduced capacity and lower starting inventory, supplier disruption effects, or very low starting stock with high variability. Afterward it recomputes cumulative demand and guarantees prefix infeasibility. If the instance is still feasible, it lowers production capacity below the minimum prefix capacity needed.

The key feasibility test is the maximum prefix shortfall:

```math
\max_t \left(\sum_{\tau=1}^t d_\tau - I^{init} - tC\right)
```

For no-backlog models, a positive value implies infeasibility. With backlogging allowed, demand balance can always carry shortage forward subject only to production capacity and nonnegative backlog variables.

## Model Characteristics

Without backlogging, variables are `2T + 1`; constraints are one initial-inventory equality plus `T` flow equalities and `T` capacity inequalities. With backlogging, variables are `3T + 2`; constraints are two initial equalities plus `T` flow equalities and `T` capacity inequalities. The constraint matrix is very sparse and banded over adjacent periods.

The model is a continuous LP. It does not include setup decisions, lot sizes, fixed ordering costs, or integer production quantities.

## Practical Notes

This generator is useful for testing long, sparse, time-linked LPs and prefix-feasibility behavior. `unknown` is unusual here compared with several other generators: it does not randomly choose feasible or infeasible; it returns the base stochastic data without explicit feasibility repair or sabotage. Feasible requests may switch a no-backlog instance into a backlog-allowed instance if that is the selected or necessary repair.
