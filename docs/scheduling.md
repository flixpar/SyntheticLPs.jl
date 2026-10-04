# Scheduling

Multi-department staff rostering with cross-training, shift templates, contract
hours, and rest rules (retail stores, hospital support services, contact
centers). The category has one variant, `standard`. Assignment variables are
binary (a natural MIP), returned as the LP relaxation under the default
`relax_integer=true`.

Differentiation from the neighbouring generators: `nurse_scheduling` rosters a
single ward with skill mix and night/weekend fairness; `workforce_shift_scheduling`
covers demand intervals with anonymous shift-pattern pools. Here named
employees are assigned to (day, shift template, department) with
productivity-weighted coverage, hour-banded contracts, and rest rules.

The previous `standard` created one binary per (worker, shift) and forced
unavailable pairs to zero with singleton `x == 0` rows (stripped by presolve,
which kept ~56% of columns and ~40% of rows), never stored its constructed
roster, and its infeasible modes were presolve-detectable. It was rebuilt.

## Variants

| Variant | Model class | Key structure | Domain grounding |
|---|---|---|---|
| `standard` | MIP (relaxed by default) | productivity-weighted coverage, one-shift-per-day, ranged weekly hours, consecutive-day windows, quick-return rule | retail, hospital support, contact centers |

## Formulation

Columns exist only for available (worker, day, template) combinations and the
worker's eligible departments:

```text
x[w, d, k, m] in {0, 1}

sum_w efficiency[w,m] x[w,d,k,m] >= requirement[d,k,m]          (coverage)
sum_{k,m} x[w,d,k,m] <= 1                                       (one shift per day)
min_hours[w] <= sum_{d in week, k, m} length[k] x[w,d,k,m] <= max_hours[w]
                                                                (contract hours, ranged)
sum_{d in window, k, m} x[w,d,k,m] <= max_consecutive           (every window of
                                                                 max_consecutive + 1 days)
sum_{closing k} x[w,d,k,·] + sum_{opening k} x[w,d+1,k,·] <= 1  (quick-return rest rule)
```

Rows that can never bind (one-column day rows, windows with too few available
days) are omitted. Objective (minimize): wage × hours × (1 + night/weekend
premiums) minus a small home-department preference bonus.

## Data Grounding

- Profiles choose the shift templates: retail (early/day/late/evening),
  hospital support (early/late/night), contact center (all five). Templates
  differ in length (4–10 h) and in rest-rule class (late/night close, early/day
  open).
- Workers: home department (productivity ~ LogNormal(0, 0.12)), cross-trained
  for one or two more departments at 60–90% efficiency half of the time;
  full-time (32–40 h/week, 80% of templates, 12% days off) or part-time
  (8 h up to 16/20/24 h/week, 50% of templates, 30% days off); lognormal wages.
- Departments sized to crews of 8–25 workers; 1–4 weeks by scale.

## Feasibility Control

- `feasible`: a roster is planted first — each worker picks days, templates, and
  (mostly home) departments respecting availability, contract hours, the
  consecutive-day limit, and the rest rule — and requirements are 80–100% of the
  productivity it supplies per slot. Stored as
  `ScheduleRosterWitness(assignment)`; a contract floor is lowered to what the
  roster achieves when requested days off make it unreachable.
- `infeasible`: one department-week (a peak event) gets requirements raised so
  the department needs 10–35% more productive shifts than its eligible staff can
  supply. `DepartmentWeekCertificate(department, week, workers, caps, required,
  available)`: each worker can work at most `cap_w = min(available days,
  max_hours / shortest template)` shifts that week (one-per-day and hours rows,
  valid in the LP), so coverage is at most `sum efficiency × cap`. The surge is
  spread over the week's slots, never above 90% of a slot's own attainable
  coverage, so no single row is unattainable and presolve does not decide it.
- `unknown`: the same surge with ratio `1 ± U(0.03, 0.30)`.

## Sizing

Columns are `sum_w (available days × templates × eligible departments)`;
workers are added until the budget is reached, so the count lands within one
worker's columns of the target (exact to ~1% from 1k up). Rows ≈ 45% of
columns. 100k columns build in well under a second.

## References

- Ernst, A.T., Jiang, H., Krishnamoorthy, M., Sier, D. (2004). Staff scheduling
  and rostering: a review of applications, methods and models. European Journal
  of Operational Research 153.
- Van den Bergh, J., Beliën, J., De Bruecker, P., Demeulemeester, E.,
  De Boeck, L. (2013). Personnel scheduling: a literature review. European
  Journal of Operational Research 226.
