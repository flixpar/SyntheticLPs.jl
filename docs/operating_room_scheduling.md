# Operating Room Scheduling

This category contains six complementary MILP generators rather than several
names for the same formulation:

| Variant | Planning level | Main decisions |
|---|---|---|
| `master_surgical_schedule` | tactical, repeating 5/10-day cycle | assign surgical services' blocks to compatible rooms and level expected specialty-ward/ICU beds |
| `elective_assignment` | advance, finite horizon | assign waiting-list cases to MSS blocks or postpone them |
| `robust_elective` | robust advance scheduling | elective assignment protected against duration overruns with a Γ budget |
| `weekly_planning` | advance, aggregate OR capacity | assign cases to days and constrain their sequential ICU-to-ward paths |
| `case_sequencing` | operational, one week | time-indexed: room, day and 15-minute start slot per case |
| `benchmark_loading` | empirical benchmark abstraction | load empirical surgery types into specialty OR-day blocks within scheduling windows |

All constructors use a local RNG. Calling a generator does not reseed or
consume Julia's global RNG, and `build_model` does no sampling.

Every variant scales to 100k+ variables within ~10% of the target (100k builds
in about a second), HiGHS presolve keeps essentially the whole model, and
every `infeasible` certificate aggregates many rows, so the simplex - not
presolve - has to refute it.

## Leeftink--Hans empirical profile

The public [Leeftink--Hans benchmark page](https://www.utwente.nl/en/choir/research/benchmark-orscheduling/)
provides a 2019 archive associated with the [Journal of Scheduling
paper](https://doi.org/10.1007/s10951-017-0539-8). Its real-life database was
constructed from roughly 200,000 realized procedures at five Dutch hospitals
and contains more than 1,000 surgery types. Every type has a frequency and a
fitted three-parameter lognormal distribution:

```text
duration = gamma + LogNormal(mu, sigma)
```

The upstream archive is about 91 MB. SyntheticLPs does not download data at
generation time or vendor the archive. `leeftink_hans_data.jl` contains a
transparent compact derivative:

1. normalize the frequencies within each of the 11 specialty files;
2. compute each type's expected duration;
3. order types by expected duration; and
4. retain the representatives at weighted quantiles 1/12, 3/12, ..., 11/12.

Repeated IDs are intentional and preserve a high-frequency type. The file also
records full-file, frequency-weighted specialty means and coefficients of
variation. Generated cases expose the source type ID and/or fitted parameters,
so the calibration is auditable. These are compressed, modified profiles, not
copies of the published benchmark instances.

The benchmark normalizes each specialty separately. It therefore does not
supply hospital-wide specialty shares, and it has no patient urgency,
deadline, surgeon, ward, or ICU fields. The relative service volumes and all
clinical/downstream fields in the general generators are labeled synthetic
assumptions. This boundary is important: empirical duration calibration does
not make those other fields empirical.

### `benchmark_loading`

This is the closest generator to the published instance design. It uses:

- a 480-minute capacity per OR-day;
- load factors `0.80:0.05:1.20`;
- empirical three-parameter-lognormal type means as planning coefficients;
- one independent realized duration per case as out-of-sample metadata; and
- a generated case list within 0.025 of the requested load.

Published OR-day counts are marked `:published_or_days`; smaller or larger
counts selected to honor the package's variable-count contract are marked
`:scaled_or_days`. Cancellation and overtime costs are package extensions and
are not attributed to the benchmark.

A second package extension makes the model sparse: the OR-days are laid out as
`rooms_per_day` rooms over `n_calendar_days` days and dealt to specialties as
blocks in proportion to workload (smaller suites run fewer services, about one
per three OR-days), and a case may only be loaded into blocks of its own
specialty inside its scheduling window `[case_release, case_due]` - about ten
admissible blocks per case (8-12 of its team's blocks when a one-day window
holds more; windows are widened until a case has at least three blocks, so no
mandatory case is pinned to one block). The dense `cases x OR-days` model it
replaces had only
`cases + OR-days` rows (327 rows at 10k columns); the sparse one keeps rows at
10-15% of the columns. Mandatory cases have no cancellation column.

## Formulation notes

### Tactical master surgical schedule

Blocks are assigned to surgical *services* (surgeon groups, each belonging to
a specialty) rather than to the 11 specialties directly, so the model scales
by adding services and rooms instead of saturating (the old generator topped
out at ~15.6k variables). Rooms form specialty clusters in proportion to the
number of services, and each service is compatible with 4-8 rooms of its
cluster; only compatible `(service, room, day)` columns are created. Room
exclusivity, ranged minimum/maximum quotas, a soft target quota, and daily
room ceilings define the block plan. Each specialty has its own ward (expected
post-ICU and direct ward occupancy) and the ICU is hospital-wide; both
profiles are cyclically convolved with the repeating schedule, the bed
capacities are bounds on the occupancy variables, and zero profile
coefficients are omitted. ICU stays use the same discrete 1--2 day
distribution as weekly planning; each cohort enters the ward only on its ICU
discharge day, and LOS tails from prior cycles are periodized into the current
cycle. Feasible instances store the planted block plan (admissible-block
indices).

### Elective and robust assignment

The elective formulation creates variables only for triples that match the
MSS specialty, surgeon availability, and deadline. The hospital grows with the
target (about `sqrt(target * specialties / 150)` rooms from 2,500 variables)
and the surgeon pool with the waiting list, so the list stays near one times
OR capacity instead of piling thousands of cases onto 16 rooms. Feasible
instances first plant a capacity-respecting schedule and then designate
mandatory cases from the scheduled set; the generator never downgrades
clinical urgency to make the instance feasible. Mandatory cases have no
postponement column (rather than one fixed to zero by a one-variable row).

`robust_elective` adds one `mu` variable per admissible triple and one `theta`
per open block. Its capacity row is the linear robust counterpart from
[Bertsimas and Sim, “The Price of
Robustness”](https://doi.org/10.1287/opre.1030.0065):

```text
nominal load + Gamma[q] * theta[q] + sum(mu[a]) <= capacity + overtime[q]
theta[q] + mu[a] >= deviation[i] * assign[a]
```

Deviation magnitudes are calibrated from the empirical fitted standard
deviations. A feasible robust witness is checked against the exact fractional
Γ-budget (largest deviations first), not a proxy average.

### Weekly downstream beds

Cases needing critical care occupy ICU first and enter the ward only after ICU
discharge. Direct ward admissions start on their surgery day. Capacity arrays
extend beyond the surgery horizon through the latest possible discharge, so a
last-day case cannot evade downstream constraints through horizon truncation.
The hospital grows linearly with the target (about `target / 190` rooms).

### Case sequencing (time-indexed)

The previous big-M disjunctive model (allocation plus one ordering binary per
shared-resource pair) had an empty LP relaxation: with fractional ordering
variables every disjunction is slack and the relaxed optimum starts every case
at time zero - the same collapse as `job_shop_scheduling`. It was rebuilt as a
time-indexed model over a week: a column per `(case, room, day, start slot)`
(15-minute slots, 480-minute session, completion by 600 minutes), one
assignment row per case, and per-slot capacity rows for every room (duration
plus room turnover) and every surgeon (duration plus surgeon turnover).
Surgeons have a specialty, 1-3 operating days, and a full-day, morning or
afternoon window with 0-60 minutes of overtime; rooms form specialty clusters
that grow as needed. The schedule is planted (each surgeon-day fills one room
from its window start), each case is eligible for its planted room plus 1-3
others of the cluster and its planted day plus each other operating day with
probability 1/2, and surplus columns (never a planted one) are dropped so the
variable count equals the target. The objective is weighted tardiness past
the regular close plus a small completion-time term.

## Feasibility status contract

- `feasible` stores a witness revalidated against every relevant capacity and
  compatibility family.
- `infeasible` stores a structural certificate that aggregates many rows, so
  presolve cannot refute it and the simplex has to:
  - `elective_assignment`, `robust_elective`, `weekly_planning`:
    `SurgeonOverloadCertificate` - three or more of one surgeon's cases, each
    admissible only on (at least two of) two or three shared days and fitting
    the room/specialty capacity on each, become mandatory while each of those
    days is budgeted only the longest case, so the budgets total at most 90%
    of the cases' minutes; no single row is contradictory and no variable
    bound tightens (three-day sets are preferred because presolve can
    aggregate two-day doubleton assignment rows);
  - `master_surgical_schedule`: `MSSWardShortageCertificate` - the patient-days
    the busiest specialty ward receives from its services' minimum quotas
    exceed the ward's cycle capacity by more than 10%;
  - `case_sequencing`: `SurgeonDayOverbookingCertificate` - add-on cases
    restricted to one surgeon-day keep the surgeon busy at least 5% longer
    than the window allows;
  - `benchmark_loading`: total expected minutes exceed `480 * n_or_days` with
    no overtime (load factor 1.05-1.20).

  Each contradiction also holds in the LP relaxation. (The previous
  certificates - a surgeon budget too small for a single case, a completion
  deadline below a case's own duration, a quota above a specialty's
  compatible room-days - were single-row contradictions presolve found
  without a simplex iteration.)
- `unknown` is a natural, two-sided instance; both outcomes occur at every
  size. (Previously, elective/robust/weekly `unknown` instances were always
  presolve-infeasible because urgent cases without any admissible slot were
  mandatory.)
  - elective/robust/weekly: urgent cases come from a greedy plan (for every
    status, so no urgent case is stranded on a single impossible slot), plus
    urgent referrals of a random 0-100% of the cases the plan could not place
    that fit at least two days on their own (robust: each block's overtime cap
    is raised where a mandatory case would not fit any block on its own under
    its robust load);
  - MSS: quotas loosen or tighten around the plan and a hospital-wide bed
    pressure factor in `[0.70, 1.05]` scales capacities (critical ~0.85);
  - case sequencing: up to ~20 surgeons receive one short add-on case
    bookable on any of their operating days;
  - benchmark loading: 60-100% of cases are mandatory and overtime caps are a
    global factor in `[0.3, 1.3]` times a reference LPT plan's excess.

The test suite checks helper properties over hundreds of seeds, exact sparse
variable formulas, all witness resources, certificate arithmetic without a
solver, field-level determinism, global-RNG isolation, build time at 60k, and
HiGHS-solved status contracts, simplex work on infeasible instances, and
two-sided `unknown` for every variant.
