# Facility Location

Generates capacitated facility location instances with fixed opening costs, customer demand, shipping costs, and an opening budget.

## Overview

This generator represents a distribution network design problem. A planner chooses which candidate facilities to open and how much demand to serve from each open facility. The model minimizes fixed facility costs plus customer shipping costs while satisfying all customer demand, respecting facility capacities, and staying within an opening budget.

## Generator Data and Sizing

`target_variables` is interpreted as:

```text
n_facilities * (n_customers + 1)
```

This matches one open variable per facility plus one shipping variable for every facility-customer pair.

Scale-dependent ranges:

| Scale condition | Facilities | Customers | Grid size | Transport cost/km | Capacity factor | Budget factor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `target_variables <= 100` | 2-20 | 1-40 | 200-800 by 200-800 | 0.5-1.2 | 1.1-1.6 | 0.4-0.8 |
| `target_variables <= 1000` | 3-100 | 5-200 | 500-2000 by 500-2000 | 0.8-1.8 | 1.2-1.8 | 0.5-0.9 |
| otherwise | 5-500 | 10-2000 | 1000-5000 by 1000-5000 | 1.0-3.0 | 1.3-2.0 | 0.6-0.95 |

The constructor searches over facility counts and chooses the customer count with the lowest relative variable-count error. If no combination is within 10%, it uses a square-root heuristic.

Random data generation:

- Facility locations are uniform on the sampled rectangle.
- Customer locations are clustered around `max(2, div(n_customers, 20))` random centers with normal offsets and clamping to the grid.
- Customer demands are log-normal around the midpoint of sampled min/max demand ranges.
- Average facility capacity is `(total_demand / n_facilities) * capacity_factor`.
- Individual capacities vary by a factor in `[0.8, 1.2]`.
- Fixed costs depend on sampled fixed-cost ranges, relative capacity, and a location factor.
- Shipping costs are Euclidean distance times sampled transport cost per km times random noise in `[0.9, 1.1]`.
- Initial budget is `sum(fixed_costs) * budget_factor`.

The stored struct fields are:

- `n_facilities`
- `n_customers`
- `facility_locs`
- `customer_locs`
- `demands`
- `fixed_costs`
- `capacities`
- `shipping_costs`
- `budget`

The constructor calls `Random.seed!(seed)`, so generation is reproducible for a fixed seed but resets Julia's global RNG state.

## LP Formulation

Sets and indices:

- `W = {1, ..., n_facilities}`: candidate facilities.
- `C = {1, ..., n_customers}`: customers.

Decision variables:

```text
y_w in {0, 1}
x_{w,c} >= 0
```

`y_w = 1` means facility `w` is opened. `x_{w,c}` is the amount shipped from facility `w` to customer `c`.

Objective:

```text
minimize
    sum_{w in W} fixed_cost_w y_w
  + sum_{w in W} sum_{c in C} shipping_cost_{w,c} x_{w,c}
```

Constraints:

Demand:

```text
sum_{w in W} x_{w,c} >= demand_c    for each c in C
```

Facility capacity:

```text
sum_{c in C} x_{w,c} <= capacity_w y_w    for each w in W
```

Opening budget:

```text
sum_{w in W} fixed_cost_w y_w <= budget
```

Bounds:

```text
y_w binary
x_{w,c} >= 0
```

At the package API level, `generate_problem(...; relax_integer=true)` is the default, so the binary open variables are relaxed unless the caller sets `relax_integer=false`.

## Feasibility Controls

- `feasible`: If total capacity is below total demand, all capacities are scaled so total capacity is at least `1.05 * total_demand`. Facilities are sorted by capacity-per-fixed-cost ratio. A greedy integer subset is selected until total demand can be covered, and budget is set to at least that subset cost times a random slack factor from `1.02` to `1.25`.
- `infeasible`: The same ratio ordering is used to compute a fractional lower bound on the minimum opening cost needed to reach total demand capacity. If the bound is finite, budget is tightened to `75%` to `95%` of that threshold. If the bound is infinite, budget is set below a fraction of total fixed cost.
- `unknown`: The original sampled budget is retained without capacity scaling or budget tightening guarantees.

## Model Characteristics

- Variables: `n_facilities * n_customers` shipment variables plus `n_facilities` open variables.
- Constraints: `n_customers` demand constraints, `n_facilities` capacity constraints, and one budget constraint.
- Density: demand rows touch all facilities for one customer; capacity rows touch all customer shipments for one facility plus that facility's open variable; the budget row touches only open variables.
- Intended model class: mixed-integer capacitated facility location.
- Default generated LP: with the package default `relax_integer=true`, facility open decisions become continuous in `[0, 1]`, yielding a capacitated facility-location relaxation.

## Practical Notes

The model allows overserving customers because demand constraints are `>=`, not equality. Shipping variables represent quantities rather than assignment fractions. The infeasible mode is designed around a fractional capacity-cost threshold, so it is especially relevant to the default LP relaxation as well as to the integer model.
