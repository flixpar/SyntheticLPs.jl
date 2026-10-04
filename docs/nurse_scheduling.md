# Nurse Scheduling

Generates multi-ward nurse rostering instances: assign nurses to wards and shifts over a planning horizon of whole weeks, meeting coverage and skill-mix requirements under realistic labor-contract rules, at minimum labor cost.

## Overview

A hospital runs several wards. Each nurse has a home ward; float-pool nurses also serve one or two neighbouring wards, which couples the wards' rosters. Each nurse has a contract type (`:core`, `:float_pool`, `:part_time`, `:day_only`), a set of skills (skill 1 is the base skill every nurse has; specialty skills are sampled), night qualification, a personal availability pattern, a consecutive-day limit and a post-night rest requirement.

Only *available* assignment slots get a variable. A nurse who is off, not night-qualified, or not attached to a ward simply has no column there. (The previous generator created a variable for every `(nurse, day, shift)` and fixed the unavailable ones to zero with one-variable rows; HiGHS presolve removed those - 43% of columns kept, 28% of rows - and the leftover rows it touched.)

## Generator Data and Sizing

| Quantity | Value |
| --- | --- |
| variables | exactly `target_variables` (tiny targets raised to two per ward-day-shift) |
| horizon | 7 days (target <= 600), 14 days (<= 2000), else 28 days |
| shift types | 2 (`:day`, `:night`) for target <= 150, else 3 (`:day`, `:evening`, `:night`) |
| wards | `clamp(round(target / (32 * days * shifts * 0.5)), 1, ...)` (~32 home nurses each) |
| skills | 3, or 4 for targets above 4,000 |

Nurses are sampled one at a time (type, skills, ward attachments, availability) until their available slots reach the target. Every `(ward, day, shift)` is then topped up to at least two variables (night slots only from night-qualified nurses, promoting a ward nurse to night duty if necessary), and surplus slots are trimmed at random - never below that minimum - to land exactly on the target. A 100k-variable instance has ~44 wards, ~1,300 nurses and ~89k rows, and builds in under a second.

Availability follows a personal Beta(7, 2) density times shift propensities (day 0.85, evening 0.65, night 0.42), with weekend effects by contract type, a strong night/evening reduction for day-only nurses and a reduction for part-timers. Costs are base rate x shift premium (night 1.28, evening 1.12) x weekend premium (1.08) x contract penalties (part-time nights, day-only off-shifts) x a 5% premium for working away from the home ward.

## LP Formulation

Variables: `x[v] in {0, 1}` for each assignment slot `v = (nurse, ward, day, shift)`.

Objective: minimize `sum_v cost[v] * x[v]`.

Constraints (built by `nurse_model_rows`; rows that cannot bind are omitted):

```text
sum_{v at (w,d,s)} x[v] >= demand[w,d,s]                       coverage
sum_{v at (w,d,s), nurse skilled in k} x[v] >= req[w,d,s,k]     skill mix, k >= 2
sum_{v of (n,d)} x[v] <= 1                                     one shift per day (2+ variables)
min_shifts[n] <= sum_{v of n} x[v] <= max_shifts[n]            contracted shifts (ranged)
lo_n <= sum_{weekend v of n} x[v] <= hi_n                      weekend shifts (ranged)
sum_{night v of n} x[v] <= night_limits[n]                     night limit
sum_{v of n on days t..t+L} x[v] <= L                          consecutive days, L = max_consecutive_days[n]
x[night of (n,d)] + sum x[early shifts of (n,d+o)] <= 1        rest after night, o = 1..rest
```

Early shifts are shift 1, plus shift 2 when there are three shift types. With the package default `relax_integer=true` the binaries become `[0, 1]` continuous variables; `relax_integer=false` returns the natural MIP.

## Feasibility Controls

All three statuses share one construction: a greedy integral roster is built day by day, shift by shift, ward by ward, from the nurses holding a variable at each slot, respecting one shift per day, consecutive-day limits and post-night rest. Demand is then set at 85-98% of the roster's coverage, skill requirements are capped at the skilled nurses it fields, total-shift bounds at `[0.7 x, x + 15% of the horizon]` around its totals, weekend bounds at its weekend count +-1, night limits at its night count plus 0-1, and consecutive-day limits are raised to its longest run.

- `feasible`: the roster satisfies every row of the integer model and is stored in `feasible_witness::NurseRosterWitness` (`assigned` variable indices plus per-nurse totals, night and weekend counts and longest runs).
- `infeasible`: a hospital-wide night shortage. Nurses' night limits are cut (largest first, keeping one night per nurse while possible) until their sum is at most `min(0.95 * night_demand, night_demand - 1)`. Summing every night coverage row and every night-limit row (each night variable appears in exactly one of each) gives `night_demand <= sum_v x[v] <= night_capacity`, a contradiction for any `x >= 0`. `infeasibility_certificate::NurseNightShortageCertificate` records both totals. Every coverage row stays individually satisfiable, so the refutation needs simplex work rather than presolve; at 10k variables HiGHS needs 18-42k iterations.
- `unknown`: natural, two-sided perturbations of the same instance - demand is the planted coverage times a global census factor in `[1.00, 1.20]` with ±3% per-slot noise (capped one below the slot's variable count, so no slot is contradictory on its own), maximum totals are tightened by 0-2 shifts and night limits by 0-1 (never below one for a nurse who worked nights). Both outcomes occur. No metadata is attached.

## Model Characteristics

- Presolve keeps essentially the whole model (measured 99.9% of columns and rows at 10k).
- Rows are about 90% of the columns (coverage and skill rows per ward-slot, plus the per-nurse rule rows).
- These are genuinely hard rostering LPs: at 10k variables HiGHS dual simplex needs 4-5 s for feasible instances, at 50k about 100 s, and 100k exceeds a two-minute budget.
