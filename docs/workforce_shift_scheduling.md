# Workforce Shift Scheduling

## `covering`

This variant is a continuous multi-site, multi-day, multi-skill shift-pattern
covering LP for aggregate workforce planning. A decision variable assigns a
number of workers from labor pool `q` to site `s`, day `d` of the planning
week, shift pattern `r` and served skill `k`. Only qualified pool-skill,
availability-compatible pool-pattern, working-day and served-site combinations
become columns.

The model minimizes full-shift labor cost:

```text
min  sum[q,s,d,r,k] assignment_cost[q,s,d,r,k] * assigned_workers[q,s,d,r,k]
```

subject to:

- effective skill staffing meeting time-varying demand in every
  site-day-period (`skill_coverage[site, day, period, skill]`);
- pool-day capacity: the workers a pool fields on one day, across all its
  sites, patterns and skills, do not exceed its daily capacity
  (`pool_day_capacity`; a pool-day with a single column is emitted as that
  column's upper bound rather than a one-variable row); and
- weekly pool capacity: the worker-shifts a pool works over the horizon stay
  below its weekly capacity, which is less than the sum of its daily
  capacities - rest days (`pool_week_capacity`, pools working two or more
  days).

Pool productivity is skill-specific, so cross-trained workers provide genuine
substitution without counting one worker simultaneously toward several served
skills. Labor pools belong to a home site; *float* pools (remote agents,
seasonal store staff, contractors) also serve one or two neighbouring sites,
coupling the sites' staffing problems. Costs combine hourly wages, paid shift
duration, skill premiums, premiums for late, closing, or night work, a 12%
weekend premium and a 6% travel premium for working away from the home site.
There are no undercoverage variables: unmet demand is a real infeasibility
rather than an expensive but always-available escape.

## Structural profiles

The selected `problem.profile` is stored and inspectable:

- `contact_center`: 24 half-hour periods over a 12-hour service day, call-type
  skills, morning/evening peaks, 4/6/8-hour shifts, quiet weekends, remote
  agents as float pools.
- `retail`: 14 hourly periods over an opening day, sales/checkout/inventory
  skills, a strong closing peak, 4/6/8/10-hour shifts, a Saturday peak, and
  seasonal staff floating between stores.
- `continuous_operations`: 24 hourly periods, operator/maintenance/quality/
  control-room skills, around-the-clock pools, 6/8/10/12-hour shifts including
  wraparound night patterns, a flat week, and floating contractors.

Long shifts include an unpaid break. Labor pools have distinct skill
qualifications, productivities, availability windows, working days (weekend
pools work the weekend plus a few weekdays), eligible pattern menus, wages,
and capacities. The first pool of every site is a flexible anchor so every
site-day-period-skill stays coverable. Duplicate
`(pool, site, day, skill, coverage support)` columns are not emitted. Sites
differ in demand scale and peak timing; days follow the profile's weekly curve.

## Sizing

| Quantity | Value |
| --- | --- |
| variables | `target_variables` (exact, except tiny targets raised to the cover size) |
| days | `clamp(round(target / 300), 1, 7)` |
| sites | `clamp(round(target / 5000), 1, 1000)` |
| coverage rows | `sites * days * periods * skills` |
| capacity rows | one per pool-day with two or more columns, one weekly row per pool working two or more days |

Pools are added round-robin over the sites until there are at least
`1.3 * target` candidate columns; the selected columns are a greedy cover of
every period of every `(site, day, skill)` group, one column for any pool
still unrepresented, and a uniform sample of the remaining candidates. Rows
are about 10-15% of the columns at every scale (e.g. ~1,000-1,550 rows at 10k
columns, ~9,000-15,000 at 100k; the previous single-day, single-site model had
~350 at 10k). A 100k-variable instance builds in about a second.

## Feasibility

- `feasible`: demand is covered by the stored `feasible_staffing` witness, a
  greedy per-`(site, day, skill)` construction with 1.5% slack; pool-day
  capacities are set 5-13% (at least 0.35 workers) above the witness's
  pool-day usage, and weekly capacities 3-9% above its weekly usage. This
  field is `nothing` for `infeasible` and `unknown` instances.
- `infeasible`: one `(site, day, skill)` group's demand curve is scaled above
  an aggregate capacity certificate. For each pool the certificate lets its
  whole pool-day capacity work its longest selected pattern serving that
  group, at its productivity for the skill; the summed coverage rows of the
  group must exceed this bound, which they do by at least 3% (or 0.5
  worker-periods), so the contradiction holds for the continuous LP and is
  robust to tolerances. The group with the smallest bound-to-demand ratio is
  chosen. `infeasible_group` and `infeasibility_capacity_bound` store the
  certificate and are `nothing` for the other statuses. The refutation needs
  the group's coverage rows and pool-day rows together, so presolve does not
  detect it.
- `unknown`: capacities are scaled by a labor-market factor in `[0.50, 0.95]`
  with independent per-pool noise, and demand receives per-period load
  shocks. Because the greedy witness is generous, the critical uniform
  capacity factor (the smallest keeping the LP feasible) was measured at
  0.53-0.88 across profiles and sizes, so the market factor lands on both
  sides of it: both statuses occur at every size. No witness or certificate is
  exposed.
