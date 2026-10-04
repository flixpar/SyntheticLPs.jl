# Unit Commitment

`unit_commitment/standard` generates multi-period power-system scheduling MILPs with
heterogeneous generators, time-varying availability, ramping, unit-level operating
reserve, startup and shutdown decisions, and minimum up/down times.

## Operational setting

The model schedules a fleet containing nuclear, coal, combined-cycle gas,
combustion-turbine, hydro, and wind units. Every archetype has its own ranges for
nameplate capacity, stable minimum output, ramp rates, operating costs, startup and
shutdown costs, and minimum up/down times. Thermal units can be derated by planned
outages, gas turbines can have short forced outages, hydro follows a daily/seasonal
availability profile, and wind follows a noisy diurnal profile.

The natural model uses binary commitment, startup, and shutdown decisions. As in
the rest of the package, `generate_problem` defaults to `relax_integer=true`, so
the ordinary returned model is its LP relaxation. Set `relax_integer=false` to
retain an implementable unit-commitment MILP.

## Sizing

There are five variable families indexed by unit and period:

```text
generation, commitment, startup, shutdown, reserve
```

Consequently, the delivered variable count is exactly

```text
5 * n_units * n_periods,
```

and, because available capacity is a variable bound rather than a singleton row,
the row count is exactly `9 * n_units * n_periods + 2 * n_periods` (headroom,
reserve capability, stable minimum, two ramp rows, state transition,
startup/shutdown exclusivity and the two minimum-time windows per unit-period;
demand balance and the reserve requirement per period).

The constructor searches within scale-appropriate dimension ranges:

| Requested size | Units | Periods |
| --- | ---: | ---: |
| below 240 | 2–6 | 6–24 |
| 240–1,199 | 4–9 | 12–36 |
| 1,200–4,799 | 10–22 | 24–72 |
| 4,800 and above | 20 or more | 48–168 |

Band boundaries match the smallest formulation in the next band, avoiding a
variable-count jump at the threshold. Large requests grow the fleet rather than
silently saturating at a fixed 48-unit cap. Ordinary targets are within about 10%
of the request (100k requests land within 10% with ~140 units over a week).
Targets below 60 clamp to the smallest useful formulation: two units over six
periods, or 60 variables.

## Generated data

For each unit `u`, the problem stores:

- `unit_types[u]`: one of `:nuclear`, `:coal`, `:ccgt`, `:gas_ct`, `:hydro`, or
  `:wind`;
- `max_output[u]` and `min_output[u]`;
- `ramp_up[u]` and `ramp_down[u]`;
- variable, no-load, startup, and shutdown costs;
- minimum up and down times;
- one availability factor per period;
- initial commitment and generation;
- `reserve_capability[u]` (half the hourly ramp-up limit: what the unit can
  deliver within 30 minutes) and `reserve_costs[u]` (a small holding cost that
  grows with the unit's energy cost).

The load shape is sampled from three daily system profiles, with seasonal,
day-to-day, and short-term noise. The reserve requirement is normally 8–18% of
demand. In the constructive feasible profile it is capped at 85% of the reserve
the stored dispatch holds; in the other profiles it never exceeds 90% of what
reserve offers could ever cover (a requirement no offer could meet would be a
single infeasible row, not a natural shortage).

All randomness uses a constructor-local `MersenneTwister`. Generating an instance is
reproducible for a fixed seed and does not reset or consume Julia's global RNG.

## Formulation

For units `u in U` and periods `t in T`, define:

```text
g[u,t]          generation (MW), g >= 0
on[u,t]         binary commitment
startup[u,t]    binary startup indicator
shutdown[u,t]   binary shutdown indicator
reserve[u,t]    operating reserve held (MW), 0 <= reserve <= capability_u
```

Under the default API relaxation, the last three domains become `[0, 1]`.

The objective minimizes variable generation cost plus no-load, startup,
shutdown, and reserve-holding costs:

```math
\min \sum_{u,t}
  c^{var}_u g_{u,t}
  + c^{nl}_u on_{u,t}
  + c^{su}_u startup_{u,t}
  + c^{sd}_u shutdown_{u,t}
  + c^{res}_u reserve_{u,t}.
```

Availability is a variable bound on generation. Generation and held reserve share
the committed available capacity, reserve is capped by ramp capability while
online, and online units respect a stable minimum:

```math
g_{u,t} \le \bar g_u a_{u,t},
\qquad
g_{u,t} + reserve_{u,t} \le \bar g_u a_{u,t} on_{u,t},
\qquad
reserve_{u,t} \le R_u on_{u,t},
\qquad
g_{u,t} \ge \underline g_u on_{u,t}.
```

Demand balance is an equality:

```math
\sum_u g_{u,t} = d_t.
```

This prevents the model from economically or physically over-generating merely to
satisfy other rows. The system reserve requirement is met by unit-level reserve:

```math
\sum_u reserve_{u,t} \ge r_t.
```

(An earlier formulation measured reserve as aggregate headroom
`Σ (ḡ a on − g)`; since that row has the same `g` coefficients as demand balance,
a single presolve substitution collapsed it into `Σ ḡ a on ≥ d + r`, exposing the
planted contradiction without any simplex work. Unit-level reserve variables are
also what production UC models use.)

Ramping between adjacent periods includes startup and shutdown allowances:

```math
g_{u,t} - g_{u,t-1}
  \le RU_u on_{u,t-1} + \bar g_u startup_{u,t},
```

```math
g_{u,t-1} - g_{u,t}
  \le RD_u on_{u,t} + \bar g_u shutdown_{u,t}.
```

The first period uses the stored initial commitment and generation. State changes
obey

```math
on_{u,t} - on_{u,t-1} = startup_{u,t} - shutdown_{u,t},
```

with an analogous initial-period equation. A unit cannot start and stop
simultaneously:

```math
startup_{u,t} + shutdown_{u,t} \le 1.
```

Rolling startup and shutdown windows enforce minimum up and down times:

```math
\sum_{k=\max(1,t-UT_u+1)}^t startup_{u,k} \le on_{u,t},
```

```math
\sum_{k=\max(1,t-DT_u+1)}^t shutdown_{u,k} \le 1-on_{u,t}.
```

The boundary convention assumes the initial state has already satisfied any
pre-horizon minimum-time obligation; minimum up/down clocks start with transitions
that occur inside the modeled horizon.

## Feasibility profiles and audit artifacts

`resolved_status` records the requested status. `unknown` is a natural instance —
natural availability (outages, wind and hydro profiles, no floors), a natural
initial state, and a peak load of 50–82% of nameplate capacity with an 8–18%
reserve — and carries neither a witness nor a certificate. Whether the peak hours
fit is genuinely undetermined: across seeds both outcomes occur (most infeasible
ones need thousands of simplex iterations to disprove).

### Feasible

Feasible instances are constructed from a primal trajectory rather than from a
capacity-margin heuristic:

1. Availability remains heterogeneous, but is floored above stable minimum output
   for the always-online witness.
2. A smooth dispatch is built sequentially for every unit within availability and
   ramp limits.
3. Initial generation is set to the witness's first-period dispatch.
4. Demand is defined as the exact sum of witness generation in each period.
5. Reserve is set below the witness's available online headroom.
6. Commitment is one throughout, with zero startups and shutdowns, which satisfies
   transition and minimum up/down rows.

The complete integral point is stored in `feasible_witness` as five
unit-by-period matrices:
`generation`, `commitment`, `startup`, `shutdown`, and `reserve` (each unit holds
reserve up to its headroom and capability). `build_model` installs these
values as JuMP starts. The solver-independent helper
`SyntheticLPs._unit_commitment_witness_is_valid(problem)` checks the witness against
every model constraint family.

### Infeasible

Infeasible instances retain diverse stress scenarios—demand spikes, outages, or
tighter reserve—then force a relaxation-proof aggregate contradiction in one
period: demand plus reserve exceeds all available nameplate capacity by 3–8%.
The contradiction is split so that no single row is violated: demand stays at
95% of available capacity, the requirement within 90% of what reserve offers
could cover, and every other period keeps demand plus reserve within 97% of its
capacity. Exposing it needs the balance row, every unit's headroom row and the
requirement row together.

Demand balance, the headroom rows, `on <= 1`, and the reserve requirement imply
the necessary cut

```math
d_t + r_t \le \sum_u \bar g_u a_{u,t}.
```

The selected period violates this inequality. `infeasibility_certificate` stores
the period, available capacity, required capacity, and positive excess. The helper
`SyntheticLPs._unit_commitment_certificate_is_valid(problem)` recomputes and checks
the certificate without a solver.

Exactly one of `feasible_witness` and `infeasibility_certificate` is present for
every `feasible` or `infeasible` instance, and neither for `unknown`.

## Practical notes

- The stored witness proves feasibility of both the natural MILP and its LP
  relaxation; it is not intended to be optimal or representative of the solver's
  final commitment schedule.
- The infeasibility certificate survives integrality relaxation because it uses only
  demand balance, headroom, the reserve requirement, availability, and the upper
  bound `on <= 1`.
- The LP relaxation stays meaningful: at the relaxed optimum of a 3k-variable
  feasible instance about 10% of commitment values are fractional and the LP bound
  is within ~0.4% of the MILP optimum (the tight-ish 3-binary formulation with
  minimum up/down windows).
- Cost and fleet data remain random even though feasibility is constructive, so
  seeds still provide materially different objective coefficients, fleet mixes,
  availability profiles, and time-series loads.
