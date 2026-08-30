# Airline Crew

Generates airline crew pairing set-partitioning instances where each flight must be covered by exactly one *operationally legal* crew pairing at minimum cost.

## Overview

This generator represents the crew pairing step in airline operations planning. A dated flight schedule is generated over a hub-and-spoke airport network, and crew pairings are built as time-and-airport-respecting walks through that schedule. The optimization model chooses pairings so that every flight in the planning horizon is covered exactly once.

Every generated column is a pairing a crew could actually fly. Pairings are grown leg by leg under the legality rules, so the following hold by construction, and no downstream step ever edits a pairing's leg set (filtering only ever drops whole columns):

- **Airport continuity**: each leg departs from the airport where the previous leg arrived.
- **Time feasibility**: each leg departs at least `min_connect` minutes after the previous leg arrives.
- **Base return**: the pairing starts at a crew base and ends back at that same base.
- **Duty rules**: legs group into duty periods bounded by `max_legs_per_duty`, `max_block_minutes` (flight time) and `max_duty_minutes` (elapsed time), with at most `max_duties` duties per pairing.
- **Rest rules**: consecutive duties are separated by a rest in `[min_rest, max_rest]`.

Because `max_sit < min_rest`, the duty structure of a pairing is uniquely recoverable from its leg times: a ground time of at most `max_sit` is an in-duty connection, anything longer is an overnight rest.

## Generator Data and Sizing

`target_variables` is interpreted as the number of pairing columns, and the generator emits **exactly** that many. Derived sizes:

| Quantity | Value |
| --- | --- |
| pairing columns | `target_variables` (exact) |
| flights (covering rows) | about `0.55 * target_variables`, grown further whenever the pairing sampler stalls |
| airports | `clamp(round(5 + flights/7), 6, 60)` |
| crew bases | `clamp(round(airports/6), 2, 10)`, the airports `1:num_bases` |
| schedule horizon | `clamp(round(flights / (2.5 * airports)), 2, 12)` days, seven departure waves per day |

Legality rules are sampled per instance in ranges typical of a domestic narrowbody operation:

| Rule | Range (minutes unless noted) |
| --- | --- |
| `min_connect` | `30:5:45` |
| `max_sit` | `180:30:300` |
| `max_legs_per_duty` | `3:5` legs |
| `max_duty_minutes` | `690:30:840` |
| `max_block_minutes` | `480:30:570` |
| `min_rest` | `600:30:660` |
| `max_rest` | `960:60:1200` |
| `max_duties` | `2:4` duties |

Airports sit on a 2000 x 1400 km map with the bases spread around the centre; scheduled block time is `35 + distance / 12` minutes rounded to five minutes and clamped to `[45, 240]`.

### Schedule construction

The schedule is produced by *planting lines of flying*: each planted line is a legal pairing whose legs are **created as it is flown** (base -> ... -> base, with sit times, duty limits and overnight rests). Every flight therefore belongs to exactly one planted line, so the planted lines partition the flight set. A line is buffered and re-checked against the legality rules before it is committed; on repeated failure a two-leg out-and-back (always legal under the sampled rules) is committed instead.

Additional columns are then sampled by randomized depth-first walks over the resulting schedule: each expansion is restricted to legal successors (connection or rest window, duty leg/block/elapsed limits, duty count), and a walk is accepted only while standing at its base. Sampled columns freely mix legs from different lines. If the walk sampler stalls, another line is planted, which both enlarges the schedule and contributes a column, so the loop always reaches the target column count.

### Cost

Pairing cost follows standard airline crew pay:

```text
credit_p = max( block_p , duty_guarantee * duty_time_p , min_daily_credit * duties_p )
c_p      = pay_rate * credit_p / 60 + per_diem_rate * tafb_p / 60 + hotel_cost * (duties_p - 1)
```

where `block_p` is total flight time, `duty_time_p` is total elapsed duty time, `tafb_p` is time away from base and `duties_p` is the number of duty periods. `pay_rate in [180, 320]` per credit hour, `duty_guarantee in [0.50, 0.60]`, `min_daily_credit in 240:15:315` minutes, `per_diem_rate in [2.0, 3.5]` per hour, `hotel_cost in [90, 160]` per overnight. Costs are a deterministic function of the pairing's schedule.

### Struct fields

- `num_flights`, `num_airports`, `bases`, `airport_locations`, `block_minutes`
- `flight_origins`, `flight_destinations`, `departure_times`, `arrival_times`
- `rules::CrewPairingRules`
- `pairing_costs`, `flights_in_pairing`, `pairing_bases`
- `pay_rate`, `duty_guarantee`, `min_daily_credit`, `per_diem_rate`, `hotel_cost`
- `feasible_witness::Union{Nothing,CrewPairingCoverWitness}`
- `infeasibility_certificate::Union{Nothing,UncoverableFlightCertificate}`
- `feasibility_status`

The constructor draws from a local `MersenneTwister(seed)`, so generation is reproducible and leaves Julia's global RNG untouched.

## LP Formulation

Sets and indices:

- `F = {1, ..., num_flights}`: flights.
- `P = {1, ..., length(pairing_costs)}`: generated pairings.
- `A_p subset F`: flights contained in pairing `p`.

Decision variables:

```text
x_p in {0, 1}
```

`x_p = 1` means pairing `p` is flown.

Objective:

```text
minimize sum_{p in P} c_p x_p
```

Constraints:

```text
sum_{p in P: f in A_p} x_p = 1    for each f in F
```

Each flight must be covered exactly once. Connection, duty-time, crew-base and rest rules are not rows of the model: as in real crew pairing solvers, they are enforced inside the columns.

At the package API level, `generate_problem(...; relax_integer=true)` is the default, so these binary variables are relaxed unless the caller sets `relax_integer=false`.

## Feasibility Controls

- `feasible`: every planted line is kept as a column, so those columns partition the flight set. The partition is recorded in `feasible_witness::CrewPairingCoverWitness` (the column indices), and setting them to one is an integral exact cover - feasible for the MIP and its LP relaxation.
- `infeasible`: the same construction plus one extra flight departing from a **non-base** airport, scheduled beyond every other arrival so nothing can connect into it. It can neither open a pairing (its origin is not a base) nor follow another leg (no flight arrives at its origin inside the sit or rest window), so no legal pairing contains it and its covering row is `0 == 1`. `infeasibility_certificate::UncoverableFlightCertificate` records the flight, its endpoints and the (zero) predecessor count. Every other flight is still covered, so the infeasibility is minimal and structural, and it refutes the LP relaxation as well.
- `unknown`: a genuine three-way mix - the planted partition is kept intact (feasible), only a random subset of the planted lines is kept as columns (genuinely undecided: the surviving columns may or may not still admit an exact cover), or an uncoverable flight is planted (infeasible). Metadata is status specific, so `unknown` instances carry neither a witness nor a certificate.

## Model Characteristics

- Variables: exactly `target_variables`, one per pairing.
- Constraints: exactly `num_flights` covering equalities (`num_flights + 1` rows worth of flights in infeasible mode, since the orphan flight adds a row).
- Nonzeros: total flight appearances across all columns; pairings average roughly five to six legs, so the matrix is sparse.
- Intended model class: binary set partitioning.
- Default generated LP: with the package default `relax_integer=true`, the binary pairing choices become continuous, yielding the set-partitioning LP relaxation.

## Practical Notes

These instances are useful for testing set-partitioning structure, sparse exact-cover constraints, degenerate LP relaxations, and column-oriented solvers. Unlike a random 0/1 covering matrix, the columns here are genuine crew pairings: the sparsity pattern is induced by a real connection network in space and time, and the costs are the credit-hour costs a crew planning system would pay.
