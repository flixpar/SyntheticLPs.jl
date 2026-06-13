# Airline Crew

Generates airline crew pairing set-partitioning instances where each flight must be covered by exactly one crew pairing at minimum cost.

## Overview

This generator represents the crew pairing step in airline operations planning. A flight network is generated over base and non-base airports, then feasible-looking pairings are built as sequences of connected flights. The optimization model chooses pairings so that every flight in the planning horizon is covered exactly once.

## Generator Data and Sizing

`target_variables` is interpreted as the target number of pairing variables. The constructor first samples a scale:

| Scale condition | Airports | Bases | Flights | Max pairings | Max flights per pairing |
| --- | ---: | ---: | ---: | ---: | ---: |
| `target_variables <= 300` | 5-12 | 2-4 | 40-80 | 50-150 | 3-5 |
| `target_variables <= 1500` | 10-25 | 3-6 | 80-200 | 150-500 | 4-6 |
| otherwise | 20-60 | 5-15 | 150-600 | 400-2000 | 5-8 |

The sampled flight and pairing counts are iteratively adjusted for up to 15 iterations. The sizing target used during adjustment is:

```text
current_vars = min(max_pairings, num_flights * 2)
```

The final number of pairings is `min(max_pairings, actual_num_flights * 2)`, so the realized variable count can differ from `target_variables`.

Generated data:

- Airports `1:num_bases` are crew bases; the rest are non-base airports.
- Flights are first generated in hub-and-spoke patterns from bases to non-bases with probability `0.7`, and from non-bases to bases with probability `0.6`.
- Any remaining flights are filled with random point-to-point origin-destination pairs with distinct endpoints.
- Flight costs are sampled from truncated normals: small `Normal(1200:2000, 300:600)` truncated to `[500, 10000]`, medium `Normal(2000:3500, 600:1200)`, large `Normal(3000:6000, 1000:2500)`.
- Base-to-non-base and point-to-point costs receive additional multiplicative beta noise.
- Pairing overhead means are sampled from truncated normals around `0.2`, `0.25`, or `0.35` by scale; overhead standard deviations are similarly sampled and truncated.
- Pairing length is sampled with weights `[0.4, 0.3, 0.15, 0.1, 0.04, 0.01]`, truncated to the maximum allowed length.

The stored struct fields are:

- `num_flights`
- `flight_origins`
- `flight_destinations`
- `pairing_costs`
- `flights_in_pairing`

The constructor calls `Random.seed!(seed)`, so generation is reproducible for a fixed seed but resets Julia's global RNG state.

## LP Formulation

Sets and indices:

- `F = {1, ..., num_flights}`: flights.
- `P = {1, ..., length(pairing_costs)}`: generated pairings.
- `A_p subset F`: flights contained in pairing `p`.

Decision variables:

```text
x_p in {0, 1}
```

`x_p = 1` means pairing `p` is selected.

Objective:

```text
minimize sum_{p in P} c_p x_p
```

where `c_p` is `pairing_costs[p]`.

Constraints:

```text
sum_{p in P: f in A_p} x_p = 1    for each f in F
```

Each flight must be covered exactly once. There are no explicit connection, duty-time, crew-base, or rest constraints in the JuMP model; those are represented only indirectly through how pairings are generated.

Bounds:

```text
x_p binary
```

At the package API level, `generate_problem(...; relax_integer=true)` is the default, so these binary variables are relaxed unless the caller sets `relax_integer=false`.

## Feasibility Controls

- `feasible`: The generator builds an exact cover first. It repeatedly assigns unassigned flights to pairings, removes covered flights from the unassigned set, then adds extra random pairings until the target count is reached. The initial partition provides a feasible integer solution before relaxation.
- `infeasible`: The generator chooses one random flight and puts it in an `avoid_set`. All generated pairings avoid that flight. The model still includes the equality constraint for the avoided flight, so its covering sum is empty and the constraint is `0 == 1`.
- `unknown`: The generator randomly selects feasible behavior with probability `0.5`; otherwise it uses the infeasible behavior above.

## Model Characteristics

- Variables: `length(pairing_costs)`, approximately `min(max_pairings, 2 * num_flights)`.
- Constraints: exactly `num_flights` covering equalities.
- Nonzeros: driven by total flight appearances across all pairings. Pairing lengths are short, so the matrix is sparse.
- Intended model class: binary set partitioning.
- Default generated LP: with the package default `relax_integer=true`, the binary pairing choices are converted to continuous variables, yielding the LP relaxation.

## Practical Notes

These instances are useful for testing set-partitioning structure, sparse exact-cover constraints, and binary-to-LP relaxation behavior. Costs encode some operational realism, including base start/end discounts and length penalties, but the generated pairings are not validated against real airline duty rules. In infeasible mode the infeasibility is structural and direct: at least one flight has no covering pairing.
