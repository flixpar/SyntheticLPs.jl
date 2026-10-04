# Airline Crew

Generates airline crew pairing instances: each flight must be covered by exactly one *operationally legal* crew pairing at minimum cost, subject to base crew availability and base block-hour balance.

## Overview

This generator represents the crew pairing step in airline operations planning. A dated flight schedule is generated over a regional hub-and-spoke airport network, and crew pairings are built as time-and-airport-respecting walks through that schedule. The optimization model chooses pairings so that every flight in the planning horizon is covered exactly once, no base has more crews away on a day than it rosters, and the block hours flown out of each base stay inside a negotiated band.

Every generated column is a pairing a crew could actually fly. Pairings are grown leg by leg under the legality rules, so the following hold by construction, and no downstream step ever edits a pairing's leg set:

- **Airport continuity**: each leg departs from the airport where the previous leg arrived.
- **Time feasibility**: each leg departs at least `min_connect` minutes after the previous leg arrives.
- **Base return**: the pairing starts at a crew base and ends back at that same base.
- **Duty rules**: legs group into duty periods bounded by `max_legs_per_duty`, `max_block_minutes` (flight time) and `max_duty_minutes` (elapsed time), with at most `max_duties` duties per pairing.
- **Rest rules**: consecutive duties are separated by a rest in `[min_rest, max_rest]`.

Because `max_sit < min_rest`, the duty structure of a pairing is uniquely recoverable from its leg times: a ground time of at most `max_sit` is an in-duty connection, anything longer is an overnight rest.

## Generator Data and Sizing

`target_variables` is interpreted as the number of pairing columns, and the generator emits **exactly** that many (for targets of at least 4). Derived sizes, with `F = round(0.35 * target_variables)` the target flight count:

| Quantity | Value |
| --- | --- |
| pairing columns | `target_variables` (exact) |
| flights (covering rows) | about `F`, grown further only when a tiny schedule cannot be flown enough ways |
| airports | `clamp(round(3 + sqrt(F) / 1.5), 6, 80)` |
| crew bases (hubs) | `clamp(round(airports / 6), 2, 12)`, the airports `1:num_bases` |
| schedule horizon | `clamp(round(F / (10 * airports)), 2, 28)` days (a monthly problem at scale), departure waves every 90 minutes from 06:00 |
| crew-availability rows | one per `(base, day)` whose column count exceeds the base roster, at most `bases * days` |
| block-hour band rows | one ranged row per base that owns a column |

Each flight is covered by about 20 pairings on average (median 13-21 at 10k-100k columns) and by at least six whenever the column budget and the network allow. At 10k columns there are about 3,600 rows, at 100k about 35,400; a 100k-column instance builds in 2-3 seconds.

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

Airports sit on a 2000 x 1400 km map: the hubs (crew bases) are spread around the centre and every spoke is placed within ~550 km of a home hub. Scheduled block time is `35 + distance / 12` minutes rounded to five minutes and clamped to `[45, 240]`. Flying is regional hub-and-spoke: from a hub, crews mostly fly to that hub's own spokes (sometimes to another hub); from a spoke, almost always back to its home hub.

### Schedule construction

The schedule is produced by *planting lines of flying*: each planted line is a legal pairing whose legs are **created as it is flown** (base -> ... -> base, with sit times, duty limits and overnight rests). Every flight therefore belongs to exactly one planted line, so the planted lines partition the flight set. A line is buffered and re-checked against the legality rules before it is committed; on repeated failure a two-leg out-and-back (always legal under the sampled rules) is committed instead.

### Pairing generation

The remaining columns are *through-flight* samples, the way production pairing generators enumerate: pick a flight, walk **backwards** through legal predecessors (same-duty connections inside the sit window, or an earlier duty across a legal rest) until the walk stands at a crew base - which becomes the pairing's base - then walk **forwards** by randomized depth-first search over legal successors until the crew is home again. Every step checks the connection or rest window and the duty leg/block/elapsed and duty-count limits, so every column is legal by construction. Sampling first gives every flight at least six covering pairings, then draws flights uniformly; when a tiny schedule stalls the sampler, another line is planted.

Dense coverage is what keeps the LP intact under presolve. The previous generator covered some flights by only one or two columns; HiGHS presolve fixed those singleton columns to one, the resulting forcing rows fixed every competing column to zero, and the cascade reduced every 10k and 50k instance to an empty problem. With every flight in several pairings no such seed exists: presolve keeps essentially all columns and 85-95% of rows (it removes linearly dependent covering rows of flights that are always flown together).

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
- `pairing_first_day`, `pairing_last_day` (calendar days, `minute ÷ 1440 + 1`), `pairing_block_hours`
- `crew_rows::Vector{Tuple{Int,Int}}` (`(base, day)`), `crew_capacity`
- `base_block_lower`, `base_block_upper`
- `pay_rate`, `duty_guarantee`, `min_daily_credit`, `per_diem_rate`, `hotel_cost`
- `feasible_witness::Union{Nothing,CrewPairingCoverWitness}`
- `infeasibility_certificate::Union{Nothing,CrewShortageCertificate}`
- `feasibility_status`

The constructor draws from a local `MersenneTwister(seed)`, so generation is reproducible and leaves Julia's global RNG untouched.

## LP Formulation

Sets and indices:

- `F = {1, ..., num_flights}`: flights.
- `P = {1, ..., length(pairing_costs)}`: generated pairings, with base `b(p)`, away days `D_p = first_day(p):last_day(p)` and block hours `h_p`.
- `A_p subset F`: flights contained in pairing `p`.
- `R`: emitted crew-availability rows `(b, d)` with capacity `cap_{b,d}`.

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
sum_{p in P: f in A_p} x_p = 1                       for each f in F        (flight covering)
sum_{p in P: b(p) = b, d in D_p} x_p <= cap_{b,d}     for each (b, d) in R   (crew availability)
lo_b <= sum_{p in P: b(p) = b} h_p x_p <= hi_b       for each base b         (block-hour balance)
```

Connection, duty-time and rest rules are not rows of the model: as in real crew pairing solvers, they are enforced inside the columns.

At the package API level, `generate_problem(...; relax_integer=true)` is the default, so these binary variables are relaxed unless the caller sets `relax_integer=false`.

## Feasibility Controls

The planted lines are always kept as columns, so the covering rows alone are always satisfiable. Each base rosters a fixed number of crews per day. The planted lines turn out to be close to crew-efficient on their peak days (cutting every base to 95% of its planted peak made almost every probed instance infeasible), so rosters are drawn relative to the planted peak `peak_b = max_d usage_{b,d}`.

- `feasible`: rosters `ceil(peak_b * U(1.00, 1.15))` (at least 2) and block-hour bands `[U(0.80, 0.95), U(1.05, 1.25)]` times the planted block hours of each base. The planted partition satisfies every row and is recorded in `feasible_witness::CrewPairingCoverWitness` (column indices) - an integral solution of the binary model and of its LP relaxation.
- `infeasible`: as `feasible`, except for a crew shortage on the busiest day `d`: the total capacity of that day's crew rows is cut to `floor(0.9 * F_d / L_d)`, where `F_d` is the number of flights departing on `d` and `L_d` the most of them any single generated pairing flies (spread over the bases by planted usage, at least one crew per base when possible). Summing the day's covering equalities gives `sum_p n_p x_p = F_d` with `n_p <= L_d`, and every pairing with `n_p > 0` sits in its base's crew row for `d`, so `F_d <= L_d * capacity <= 0.9 F_d` - a contradiction for any `x >= 0`, valid for the LP relaxation, with a 10% margin. `infeasibility_certificate::CrewShortageCertificate` records the day, `F_d`, `L_d`, the capacity and the row indices. The refutation aggregates every covering row and every crew row of the day, so HiGHS presolve does not detect it: the simplex has to (thousands of iterations at 10k columns).
- `unknown`: a natural instance with rosters `round(peak_b * U(0.92, 1.08))` - above or below the planted peak, so the instance is feasible or not depending on how well the generated pairings absorb the peak days (both outcomes occur at every size). No metadata is attached.

## Model Characteristics

- Variables: exactly `target_variables`, one per pairing.
- Constraints: `num_flights` covering equalities, plus the emitted crew-availability rows (tens to a few hundred), plus one ranged block-hour row per base.
- Nonzeros: flight appearances (five to eight legs per pairing on average), plus each pairing's away days in its base's crew rows, plus one block-hour coefficient per pairing.
- Intended model class: binary set partitioning with side constraints.
- Default generated LP: with the package default `relax_integer=true`, the binary pairing choices become continuous, yielding the crew pairing LP relaxation.

## Practical Notes

These instances are useful for testing set-partitioning structure, sparse exact-cover constraints, degenerate LP relaxations, and column-oriented solvers. Unlike a random 0/1 covering matrix, the columns here are genuine crew pairings: the sparsity pattern is induced by a real connection network in space and time, and the costs are the credit-hour costs a crew planning system would pay.

Expect these LPs to be hard, as production crew pairing LP relaxations are: highly primal- and dual-degenerate (most of a pairing's cost is block time, which every exact cover pays in full), with dense basis factors. HiGHS dual simplex needs about 1 s at 2k columns, 4 s at 5k and 40-60+ s at 10k; 100k-column instances are far beyond a one-minute budget.
