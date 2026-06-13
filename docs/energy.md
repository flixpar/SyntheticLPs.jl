# Energy

Generates energy generation mix LPs that dispatch multiple power sources over time to meet demand at minimum generation cost.

## Overview

This generator represents short-horizon power generation planning. It creates conventional and renewable generation sources, time-varying demand, source capacities, source costs, emissions rates, and a minimum renewable share. The optimization model decides generation by source and time period.

## Generator Data and Sizing

`target_variables` is used to tune a requested `n_sources * n_periods` product. The constructor starts from scale-dependent source and period ranges, initializes `n_sources = min_sources + 2` and `n_periods = min_periods + 12`, then iteratively adjusts those requested dimensions for up to 15 iterations.

| Scale condition | Sources | Periods | Peak demand |
| --- | ---: | ---: | ---: |
| `target_variables < 250` | 3-8 | 12-48 | 10-100 |
| `target_variables < 1000` | 5-12 | 24-72 | 100-1000 |
| otherwise | 8-20 | 48-200 | 1000-10000 |

Potential source types are:

| Source | Renewable | Availability | Capacity factor | Cost factor |
| --- | --- | ---: | ---: | ---: |
| coal | no | 0.95 | 0.90 | 1.0 |
| gas | no | 0.98 | 0.85 | 1.2 |
| nuclear | no | 0.92 | 0.95 | 0.8 |
| solar | yes | 0.99 | 0.25 | 0.3 |
| wind | yes | 0.95 | 0.35 | 0.4 |
| hydro | yes | 0.90 | 0.50 | 0.6 |
| biomass | yes | 0.88 | 0.75 | 1.1 |

The built JuMP model indexes source variables over the selected `sources` vector,
not over `1:n_sources`. Because there are only seven hard-coded source types,
the actual number of modeled sources is `length(sources)` and can be smaller
than the stored `n_sources` metadata when the requested source count exceeds the
available renewable and conventional type pools.

Random data generation:

- Renewable share target is beta-distributed: small `Beta(2,3)`, medium `Beta(3,4)`, large `Beta(4,5)`.
- Demand variation is beta-distributed and decreases with scale.
- Base generation cost is log-normal around 60, 45, or 35 depending on scale.
- Renewable cost factor is gamma-distributed.
- Capacity margin is clipped normal: small about `1.35`, medium about `1.25`, large about `1.15`.
- Source costs combine base cost, source cost factor, renewable cost factor if applicable, and source-specific random variation.
- Capacity shares use source-specific gamma, beta, or log-normal samples, normalized to total required capacity.
- Demand follows residential, commercial, or industrial 24-hour profiles depending on peak demand, with weather, economic, random, and seasonal multiplicative noise.
- Emissions are `0.0` for renewable sources and nuclear, `emission_limit` for coal, and half that for gas.

The stored struct fields are:

- `n_sources`
- `n_periods`
- `sources`
- `time_periods`
- `generation_costs`
- `capacities`
- `demands`
- `emission_limits`
- `renewable_fraction`

The constructor calls `Random.seed!(seed)`, so generation is reproducible for a fixed seed but resets Julia's global RNG state.

## LP Formulation

Sets and indices:

- `S`: selected generation sources.
- `T = {1, ..., n_periods}`: time periods.
- `R = {s in S: emission_limits[s] == 0}`: zero-emission sources used as renewables in the model.

Decision variables:

```text
0 <= x_{s,t} <= capacity_s
```

`x_{s,t}` is the generation from source `s` in period `t`.

Objective:

```text
minimize sum_{s in S} sum_{t in T} cost_s x_{s,t}
```

Constraints:

Demand satisfaction:

```text
sum_{s in S} x_{s,t} >= demand_t    for each t in T
```

Emissions:

```text
sum_{s in S} emission_s x_{s,t}
    <= max_emission * sum_{s in S} x_{s,t}    for each t in T
```

where `max_emission = maximum(values(emission_limits))`.

Renewable fraction:

```text
sum_{s in R} x_{s,t}
    >= renewable_fraction * sum_{s in S} x_{s,t}    for each t in T
```

Bounds:

All variables are continuous and bounded above by source capacity in every period.

## Feasibility Controls

- `feasible`: The generator checks total nameplate capacity against `peak_demand * capacity_margin` and scales all capacities up if needed. It does not solve the final LP, but it tries to ensure adequate capacity before returning.
- `infeasible`: One of four scenarios is sampled:
  - Capacity crisis: all capacities are scaled so total capacity is only `60%` to `80%` of peak demand.
  - Renewable intermittency: `renewable_fraction` is raised to `0.7` to `0.9`, then renewable capacity is reduced to about `40%` to `60%` of the required renewable capacity.
  - Emission impossibility: fossil emissions are reduced near zero and clean capacity may be capped around `50%` to `70%` of peak demand.
  - Demand surge: demands are multiplied so peak demand exceeds current total capacity by roughly `10%` to `30%`.
- `unknown`: The generator samples feasible behavior with probability `0.7`; otherwise it samples one of the infeasible scenarios.

## Model Characteristics

- Variables: `length(sources) * n_periods` in the built model. The stored
  `n_sources` field records the requested adjusted count, but variables are
  created only for the selected named source types.
- Constraints: `3 * n_periods`: one demand, one emissions, and one renewable-share constraint per period.
- Bounds: one upper bound from capacity for each selected source-period pair.
- Density: each period constraint touches all source variables for that period, so the matrix is block diagonal by time period.
- Model class: continuous LP; there are no unit commitment, startup, ramping, storage, or integer investment decisions.

## Practical Notes

These instances are useful for testing dispatch-style LPs with repeated time-period blocks, upper-bounded variables, and renewable-share constraints. The emissions constraint is unusual: because it uses the maximum emissions rate among sources as the right-hand coefficient, it is generally nonbinding for nonnegative generation. Infeasibility is therefore mostly driven by capacity, demand, and renewable-fraction interactions rather than by the emissions row itself.
