# Operating Room Scheduling

This category contains six complementary MILP generators rather than several
names for the same formulation:

| Variant | Planning level | Main decisions |
|---|---|---|
| `master_surgical_schedule` | tactical, repeating 5/10-day cycle | assign specialty blocks to compatible rooms and level expected ICU/ward beds |
| `elective_assignment` | advance, finite horizon | assign waiting-list cases to MSS blocks or postpone them |
| `robust_elective` | robust advance scheduling | elective assignment protected against duration overruns with a Γ budget |
| `weekly_planning` | advance, aggregate OR capacity | assign cases to days and constrain their sequential ICU-to-ward paths |
| `case_sequencing` | operational, one day | assign rooms/surgeons and sequence every shared-resource pair |
| `benchmark_loading` | empirical benchmark abstraction | load empirical surgery types across identical 480-minute OR-days |

All constructors use a local RNG. Calling a generator does not reseed or
consume Julia's global RNG, and `build_model` does no sampling.

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

## Formulation notes

### Tactical master surgical schedule

Only compatible `(specialty, room, day)` columns are created. Room exclusivity,
minimum/maximum service quotas, and daily room ceilings define the block plan.
Separate expected ICU and post-ICU ward profiles are cyclically convolved with
the repeating schedule. ICU stays use the same discrete 1--2 day distribution
as weekly planning; each cohort enters the ward only on its ICU discharge day,
and LOS tails from prior cycles are periodized into the current cycle. Feasible
instances store the complete planted block array. Infeasible instances require
one specialty to receive more blocks than all of its compatible room-days, an
LP-level certificate.

### Elective and robust assignment

The elective formulation creates variables only for triples that match the
MSS specialty, surgeon availability, and deadline. Feasible instances first
plant a capacity-respecting schedule and then designate mandatory cases from
the scheduled set; the generator never downgrades clinical urgency to make the
instance feasible.

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

### Daily sequencing

Room and surgeon eligibility variables are sparse. An ordering binary is
created for each unordered case pair and each room/surgeon they could share.
Big-M disjunctions enforce turnover-separated no-overlap, while hard surgeon
windows and soft target completion times retain operational structure.

## Feasibility status contract

- `feasible` stores a witness revalidated against every relevant capacity and
  compatibility family.
- `infeasible` stores a structural certificate: surgeon-minute shortage,
  impossible completion deadline, compatible-block quota excess, or total
  OR-day workload excess, depending on the variant. Each contradiction also
  holds in the LP relaxation.
- `unknown` applies natural sampled conditions and stores neither a witness nor
  a forced certificate.

The test suite checks helper properties over hundreds of seeds, exact sparse
variable formulas, all witness resources, certificate inequalities,
field-level determinism, global-RNG isolation, and HiGHS-solved status contracts
for every variant.
