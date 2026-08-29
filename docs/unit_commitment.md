# Unit Commitment

`unit_commitment/standard` generates multi-period power-system scheduling MILPs with
heterogeneous generators, time-varying availability, ramping, spinning reserve,
startup and shutdown decisions, and minimum up/down times.

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

There are four variable families indexed by unit and period:

```text
generation, commitment, startup, shutdown
```

Consequently, the delivered variable count is exactly

```text
4 * n_units * n_periods.
```

The constructor searches within scale-appropriate dimension ranges:

| Requested size | Units | Periods |
| --- | ---: | ---: |
| below 192 | 2–6 | 6–24 |
| 192–959 | 4–9 | 12–36 |
| 960–3,839 | 10–22 | 24–72 |
| 3,840 and above | 20 or more | 48–168 |

Band boundaries match the smallest formulation in the next band, avoiding a
variable-count jump at the threshold. Large requests grow the fleet rather than
silently saturating at a fixed 48-unit cap. Ordinary targets are within about 10%
of the request. Targets below 48 clamp to the smallest useful formulation: two
units over six periods, or 48 variables.

## Generated data

For each unit `u`, the problem stores:

- `unit_types[u]`: one of `:nuclear`, `:coal`, `:ccgt`, `:gas_ct`, `:hydro`, or
  `:wind`;
- `max_output[u]` and `min_output[u]`;
- `ramp_up[u]` and `ramp_down[u]`;
- variable, no-load, startup, and shutdown costs;
- minimum up and down times;
- one availability factor per period;
- initial commitment and generation.

The load shape is sampled from three daily system profiles, with seasonal,
day-to-day, and short-term noise. Spinning reserve is normally 8–18% of demand. In
the constructive feasible profile it is capped below the online headroom of the
stored dispatch, so reserve remains meaningful without invalidating the witness.

All randomness uses a constructor-local `MersenneTwister`. Generating an instance is
reproducible for a fixed seed and does not reset or consume Julia's global RNG.

## Formulation

For units `u in U` and periods `t in T`, define:

```text
g[u,t]          generation (MW), g >= 0
on[u,t]         binary commitment
startup[u,t]    binary startup indicator
shutdown[u,t]   binary shutdown indicator
```

Under the default API relaxation, the last three domains become `[0, 1]`.

The objective minimizes variable generation cost plus no-load, startup, and
shutdown costs:

```math
\min \sum_{u,t}
  c^{var}_u g_{u,t}
  + c^{nl}_u on_{u,t}
  + c^{su}_u startup_{u,t}
  + c^{sd}_u shutdown_{u,t}.
```

Availability and commitment limit generation, while online units respect a stable
minimum:

```math
g_{u,t} \le \bar g_u a_{u,t},
\qquad
g_{u,t} \le \bar g_u on_{u,t},
\qquad
g_{u,t} \ge \underline g_u on_{u,t}.
```

Demand balance is an equality:

```math
\sum_u g_{u,t} = d_t.
```

This prevents the model from economically or physically over-generating merely to
satisfy other rows. Spinning reserve is available headroom from committed,
available units:

```math
\sum_u \left(\bar g_u a_{u,t} on_{u,t} - g_{u,t}\right) \ge r_t.
```

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

The constructor records the status it actually built in `resolved_status`. An
`unknown` request is resolved deterministically from the instance seed to either a
feasible or infeasible profile, so downstream analysis does not have to infer what
the random branch selected.

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

The complete integral point is stored in `feasible_witness` as four
unit-by-period matrices:
`generation`, `commitment`, `startup`, and `shutdown`. `build_model` installs these
values as JuMP starts. The solver-independent helper
`SyntheticLPs._unit_commitment_witness_is_valid(problem)` checks the witness against
every model constraint family.

### Infeasible

Infeasible instances retain diverse stress scenarios—demand spikes, outages, or
tighter reserve—then force a relaxation-proof aggregate contradiction in one
period. Demand remains below available generation when capacity is positive, but
demand plus reserve strictly exceeds all available nameplate capacity.

Demand balance and the reserve row imply the necessary cut

```math
d_t + r_t \le \sum_u \bar g_u a_{u,t}.
```

The selected period violates this inequality. `infeasibility_certificate` stores
the period, available capacity, required capacity, and positive excess. The helper
`SyntheticLPs._unit_commitment_certificate_is_valid(problem)` recomputes and checks
the certificate without a solver.

Exactly one of `feasible_witness` and `infeasibility_certificate` is present for
every generated instance.

## Practical notes

- The stored witness proves feasibility of both the natural MILP and its LP
  relaxation; it is not intended to be optimal or representative of the solver's
  final commitment schedule.
- The infeasibility certificate survives integrality relaxation because it uses only
  demand balance, reserve, availability, and the upper bound `on <= 1`.
- Cost and fleet data remain random even though feasibility is constructive, so
  seeds still provide materially different objective coefficients, fleet mixes,
  availability profiles, and time-series loads.
