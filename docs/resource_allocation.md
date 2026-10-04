# Resource Allocation

Multi-period allocation of skilled resource pools (teams, compute clusters,
maintenance crews) to a portfolio of windowed activities. The category has one
variant, `standard`, a pure continuous LP.

The previous `standard` was a single-period profit-maximisation over up to 96
knapsack rows with per-activity floors; every column touched only those few
rows, and HiGHS presolve reduced *every* instance to an empty model (10k and
50k, all statuses). The rebuilt model gives every column three rows with
heterogeneous efficiencies, so it keeps ~80–95% of its columns and ~90–96% of
its rows through presolve.

## Variants

| Variant | Model class | Key structure | Domain grounding |
|---|---|---|---|
| `standard` | continuous LP | pool-period capacity, ranged commitment/scope rows with efficiencies, absorption-rate rows | engineering portfolios, cloud capacity, maintenance crews |

## Formulation

Activities `a` have a window `[release[a], deadline[a]]`, a home department and
a small set of eligible pools `P_a` with efficiency `eff[a,p]` (output per
hour). One column per eligible (activity, pool, in-window period):

```text
y[a, p, t] >= 0      hours of pool p spent on activity a in period t

sum_a y[a,p,t] <= capacity[p,t]                                (pool, period)
floor[a] <= sum_{p,t} eff[a,p] y[a,p,t] <= workload[a]         (scope; ranged when committed)
sum_p eff[a,p] y[a,p,t] <= rate_cap[a]                         (absorption rate, per period;
                                                                a variable bound if |P_a| = 1)
```

Objective (maximize): `value[a] * discount^(t - release[a]) * eff[a,p]` per hour
(earlier delivery is worth more) minus `pool_cost[p] * period_cost_factor[t]`
(seasonal overtime premium).

## Data Grounding

- Regimes: `:engineering_portfolio` (teams with strong skill differences),
  `:cloud_capacity` (near-interchangeable clusters, longer horizons),
  `:maintenance_crews` (craft specialisation, many commitments).
- Pools: grouped in departments of 3–7; pool count chosen so each pool-period
  row is shared by ~5–20 activities; lognormal cost and quality.
- Activities: 2–6 eligible pools in the home department, plus a cross-trained
  or contracted pool elsewhere 20–45% of the time (efficiency × 0.65); windows
  span a Beta-distributed fraction of the horizon.
- Workloads exceed the planted output (35–85% of the workload is planned), and
  values are lognormal around ~2× an average hour's cost, so most columns are
  profitable and pools, scopes, and rates all bind somewhere.

## Feasibility Control

A plan is planted first (per-period output split across eligible pools with
Dirichlet shares); capacities are `max(planned hours × (1 + headroom), flat base ×
availability)` with holiday/outage dips, and commitment floors are 30–90% of the
planned output.

- `feasible`: `ResourceAllocationPlanWitness(allocation, pool_hours, output)` —
  strict slack on every pool row.
- `infeasible`: the department whose local (department-only) activities carry
  the most committed hours has its whole local portfolio put under contract;
  floors grow part of the way (never past a single activity's own workload or
  cumulative rate cap) and the department's pools lose staff until the floors
  need 12–35% more hours than exist. A floor the cut would put out of reach
  of its own activity (above what its eligible pools and rate cap could deliver
  even if it had them to itself) is trimmed to 90% of that standalone maximum
  and the cut redone: otherwise that single row is a contradiction HiGHS
  presolve finds by bound propagation, which it did on 1k–100k instances
  (including `unknown` ones) before the trim. `DepartmentOvercommitCertificate(department,
  pools, activities, max_efficiency, required_hours, available_hours)`: divide
  each commitment row by the activity's best efficiency and sum with all the
  department's pool rows. Department `0` (the whole organisation) is the
  fallback when no department has two local activities.
- `unknown`: the same mechanism with ratio `1 ± U(0.03, 0.30)`.

## Sizing

Columns are exactly `max(target, 4)`: activities are generated until the column
budget is used, the last one's window (then pool set) truncated. Rows are the
used (pool, period) pairs, one scope row per activity, and one rate row per
in-window period of every multi-pool activity — about 33–40% of the columns.
There is no dense data and no size cap; 100k columns build in ~0.2 s and solve
in a few seconds.

## References

- Ibaraki, T., Katoh, N. (1988). Resource Allocation Problems: Algorithmic
  Approaches. MIT Press.
- Gutjahr, W.J., Katzensteiner, S., Reiter, P., Stummer, C., Denk, M. (2008).
  Competence-driven project portfolio selection, scheduling and staff
  assignment. Central European Journal of Operations Research 16.
