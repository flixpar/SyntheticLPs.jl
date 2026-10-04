# Assignment

Field-service assignment of jobs to workers (technicians, crews, drivers) over
**sparse** compatibility graphs: each job is reachable only by its nearest
workers holding the skill it needs. Only compatible pairs become variables —
the previous dense `n_workers x n_tasks` matrix with `x[i,j] == 0` rows for
forbidden pairs (dead columns plus singleton rows; HiGHS presolve kept 27% of
the columns and 2% of the rows, and the infeasible branch overshot the size
target by up to 95%) is gone. Both variants scale to the documented
1,000,000-variable cap (`ASSIGNMENT_MAX_VARIABLES`) and use a constructor-local
RNG; `build_model` does no sampling.

## Variants

| Variant | Model class | Key structure |
|---|---|---|
| `standard` (default) | binary (TU LP relaxation) | sparse linear assignment: job rows `= 1`, worker rows `<= 1` |
| `workload_balance` | binary + continuous makespan | unrelated workers: worker-specific processing times, availability-scaled makespan rows, overtime cap, cost term |

## Shared data model (`assignment.jl`)

- **Geography**: workers and jobs drawn from one `_geo_positions` population
  (metro clusters, uniform coverage or a corridor; region side `12 sqrt(n)`).
- **Skills**: Zipf-like skill popularity; every job needs one skill; every
  worker holds a primary skill plus up to two cross-trained ones.
- **Eligibility**: per-skill grid kNN (`_asg_skill_candidates`) finds each
  job's nearest skill holders in near-linear time; per-job edge counts are
  lognormal and summed exactly to the variable budget (`_asg_edge_counts`).

## `standard`

```text
minimize    sum_e cost[e] x[e]
subject to  sum_{e serving j} x[e]  = 1        every job
            sum_{e of w}      x[e] <= 1        every worker with an edge
            x binary
```

Each job has a lognormal number (mean 5-12, at least 2) of edges to its
nearest skill holders, followed by the nearest other workers at a 40
cross-skill premium. Cost = worker wage (lognormal, median 30/h) x job duration
(lognormal, median 2 h) + 0.8 x travel distance (+ premium).

The constraint matrix is a bipartite incidence matrix, so the LP relaxation is
totally unimodular and integral: this is the classic assignment LP, valuable
for its heavy primal degeneracy at scale on realistic sparse graphs rather
than for fractional structure (use `workload_balance` for that).

- `feasible`: 3%-25% more workers than jobs; a planted matching (jobs in
  random order to their nearest free candidate) is forced into the edge set
  and stored as `AssignmentWitness`.
- `infeasible`: a skill group with at least 4 jobs (and at least 3% of all
  jobs) is served only by holders of that skill, and their number is cut to
  70%-90% of the group's jobs — a Hall violation
  (`AssignmentHallCertificate(jobs, workers)` with fewer workers than jobs)
  that spans a whole trade, so presolve does not see it. Tiny instances
  without such a group fall back to a worker shortfall.
- `unknown`: 97%-120% as many workers as jobs, every job restricted to
  qualified workers, no planting — scarce trades may or may not be coverable.

Sizing: variables = edges, exactly `max(target, 2)` (tiny instances may offer
fewer compatible pairs); rows = jobs + workers with at least one edge.

## `workload_balance`

```text
minimize    makespan_weight * L + sum_e cost[e] x[e]
subject to  sum_{e serving t} x[e] = 1                                   every task
            sum_{e of w} processing_time[e] x[e] - availability[w] L <= 0  every worker
            0 <= L <= max_makespan,  x binary
```

Each task is eligible for a lognormal number (mean 3-7) of its nearest skill
holders (only a skill nobody holds falls back to the nearest workers, 40%
slower). Processing time = task base duration (lognormal, median 4 h, clipped
to 0.5-10 h) / worker speed (lognormal) x skill fit (1.0 primary trade, 1.15
cross-trained) x noise; availability is 1.0, 0.75 or 0.5 (full/part time);
cost = wage x time + travel. Worker-specific times make the relaxation a
genuine unrelated-machines LP (the old identical-load version relaxed to the
trivial `L = total / n_workers`), and the cost term breaks the degeneracy of a
pure makespan objective. The weight balances the two terms (0.5-2x the ratio
of typical assignment cost to makespan).

- `feasible`: `max_makespan` (the overtime cap) is 1.05-1.3x the makespan of a
  planted greedy plan (longest tasks first, each to the eligible worker whose
  availability-scaled load grows least) — `WorkloadBalanceWitness`.
- `infeasible`: a skill group (at least 4 tasks, at least 3%) has its
  durations surged until its fastest-possible workload exceeds its workforce's
  `sum availability * max_makespan` by 5%-15%
  (`WorkloadBalanceCertificate(tasks, workers, required, available)`, valid in
  the relaxation). The group is chosen so the surge leaves every single task
  doable within 90% of some eligible worker's cap — no single row refutes the
  model. Without such a group, all tasks and workers are used.
- `unknown`: `max_makespan` is 0.75-1.15x the greedy makespan; the fractional
  optimum may sit below the greedy one, so it may or may not fit.

Sizing: variables = edges + 1, exactly `max(target, 3)`; rows = tasks +
workers.

## References

- Burkard, R., Dell'Amico, M., Martello, S. (2012). Assignment Problems,
  revised reprint. SIAM.
- Lenstra, J.K., Shmoys, D.B., Tardos, É. (1990). Approximation algorithms for
  scheduling unrelated parallel machines. Mathematical Programming 46.
- Hall, P. (1935). On representatives of subsets. Journal of the London
  Mathematical Society 10.
