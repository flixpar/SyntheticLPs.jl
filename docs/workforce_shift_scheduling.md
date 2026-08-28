# Workforce Shift Scheduling

## `covering`

This variant is a continuous multi-skill shift-pattern covering LP for
aggregate workforce planning. A decision variable assigns a number of workers
from labor pool `q` to shift pattern `r` and served skill `s`. Only qualified
pool-skill and availability-compatible pool-pattern combinations become
columns.

The model minimizes full-shift labor cost:

```text
min  sum[q,r,s] assignment_cost[q,r,s] * assigned_workers[q,r,s]
```

subject to:

- effective skill-period staffing meeting time-varying demand; and
- total assignments from each labor pool not exceeding its available worker
  capacity.

Pool productivity is skill-specific, so cross-trained workers provide genuine
substitution without counting one worker simultaneously toward several served
skills. Costs combine hourly wages, paid shift duration, skill premiums, and
premiums for late, closing, or night work. There are no undercoverage
variables: unmet demand is a real infeasibility rather than an expensive but
always-available escape.

## Structural profiles

The selected `problem.profile` is stored and inspectable:

- `contact_center`: 24 half-hour periods over a 12-hour service day, call-type
  skills, morning/evening peaks, and 4/6/8-hour shifts.
- `retail`: 14 hourly periods over an opening day, sales/checkout/inventory
  skills, a strong closing peak, and 4/6/8/10-hour shifts.
- `continuous_operations`: 24 hourly periods, operator/maintenance/quality
  skills, around-the-clock pools, and 6/8/10/12-hour shifts including
  wraparound night patterns.

Long shifts include an unpaid break. Labor pools have distinct skill
qualifications, productivities, availability windows, eligible pattern menus,
wages, and capacities. Duplicate `(pool, skill, coverage support)` columns are
not emitted.

## Feasibility and sizing

- `feasible`: demand is covered by the stored `feasible_staffing` witness,
  and pool capacities are set above that witness's usage. This field is
  `nothing` for `infeasible` and `unknown` instances.
- `infeasible`: one skill's demand curve is scaled above an aggregate capacity
  certificate. For each pool, the certificate permits its entire capacity to
  use its longest selected pattern serving the certified skill, so violating
  this upper bound proves infeasibility even for the continuous LP.
  `infeasible_skill` and `infeasibility_capacity_bound` store this certificate
  and are `nothing` for the other statuses.
- `unknown`: independently sampled workload and labor-market shocks are
  applied without forcing either status; no witness or certificate is exposed.

The only variables are selected staffing columns. Normal targets are therefore
matched exactly, including targets above 1,000 variables; only very small
targets (including `target_variables=1`) are raised to the compact skill-period
cover plus representative generated labor-pool columns.
