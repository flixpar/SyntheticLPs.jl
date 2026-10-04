# Job Shop Scheduling

Time-indexed job shop scheduling with parallel-machine work centers, release
dates, hard deadlines, and weighted tardiness. The category has one variant,
`standard`. Start variables are binary (a natural MIP); under the default
`relax_integer=true` the model is returned as its LP relaxation, which — unlike
the big-M disjunctive formulation this replaced — keeps machine contention.
The generator uses a constructor-local RNG and `build_model` does no sampling.

## Variants

| Variant | Model class | Key structure | Domain grounding |
|---|---|---|---|
| `standard` | MIP (relaxed by default) | time-indexed start variables, work-center/slot capacity rows, aggregated routing precedence | dynamic job shop: arrivals over time, flow-dominant routings, bottleneck centers |

## Formulation

Data: jobs `j` with ordered operations `o` (one per visited work center),
processing time `proc[o]` in slots, start window `[earliest[o], latest[o]]`;
work center `w` holds `capacity[w]` identical machines.

```text
x[o, s] in {0, 1}        operation o starts in slot s,  s in [earliest[o], latest[o]]

sum_s x[o, s] = 1                                          (assignment, per operation)
sum_s s x[q, s] - sum_s s x[o, s] >= proc[o]               (precedence, consecutive o -> q)
sum_{o at w} sum_{s = t - proc[o] + 1}^{t} x[o, s] <= capacity[w]
                                                           (capacity, per work center and slot)
```

Capacity rows are emitted only for slots in which more than `capacity[w]`
operations could possibly be running (the others can never bind). Windows come
from the job's release plus routing head (`earliest`) and the job's hard
deadline minus routing tail (`latest`).

Objective (minimize): weighted tardiness of each job's last operation against
its due date — a piecewise-linear cost placed directly on the last operation's
start columns — plus a small work-in-process cost `wip_cost * weight[j] *
(s - earliest[o])` on every operation.

Relaxation strength: the old big-M model relaxed to the no-contention closed
form exactly (every disjunction slack at fractional order variables). Here
the capacity rows bind wherever operations compete for a slot; the test suite
checks that dropping them strictly lowers the relaxed optimum on most
instances.

## Data Grounding

- Work centers: 4–20 (by scale), 75% single-machine, the rest 2–3 machines;
  lognormal mean processing time per center with ~20% slow bottleneck centers.
- Routings: a random subset of 2–8 distinct centers visited in the shop's
  dominant flow order, with one adjacent swap 30% of the time.
- Arrivals: a Poisson stream whose rate puts the bottleneck at 75–92%
  utilisation; operations are list-scheduled (earliest feasible slot with
  backfilling) as jobs arrive.
- Due dates: total-work-content rule `release + F * total processing`,
  `F ~ U(1.3, 3.5)`; weights from priority classes (1, 2, 4) with jitter.
- Deadlines: the planted completion plus slack that fills the column budget
  (window widths vary across jobs like real flow allowances).

## Feasibility Control

- `feasible`: deadlines are at or after the planted list schedule's completions,
  so the schedule is a 0/1 point of the model — stored as
  `JobShopScheduleWitness(start, completion)`.
- `infeasible`: an expedited batch of 3–8 orders (per machine) that all visit
  the bottleneck center gets releases and deadlines confining their bottleneck
  operations to one interval `[a, b]` whose machine-slots are 10–35% short of
  their total processing time. `JobShopEnergyCertificate(work_center,
  interval_start, interval_end, operations, required_load,
  available_capacity)`: summing the center's capacity rows over the interval
  and the batch's assignment rows gives `required_load <= available_capacity`,
  a contradiction. The argument is an LP-row combination (it survives
  relaxation) and needs `interval length + |batch|` rows at once, so presolve
  does not detect it; the batch grows (or a machine of a multi-machine center
  is taken down) until a margin of at least 8% is reached.
- `unknown`: the same expedited batch, with the interval sized so the
  bottleneck's *total* planted load in it (batch plus ordinary traffic) sits at
  `1 ± U(0.03, 0.30)` of capacity — both outcomes occur.

## Sizing

Columns are `sum_o (latest[o] - earliest[o] + 1)`. Jobs are generated until
about `target / mean_window` operations exist (mean window 8–30 slots by
scale); ordinary jobs arriving last are dropped if queueing already exceeds the
budget, and the remainder is distributed as deadline slack in whole-job
increments, so the column count lands within one routing length (≤ 8) of the
target. Rows are `n_ops` assignment rows, `sum_j (ops_j - 1)` precedence rows,
and the binding-capable capacity rows (≈ 20–25% of the column count). Nonzeros
are about `(mean proc + 3)` per column. 100k columns build in well under a
second.

The README example

```julia
generate_problem(:job_shop_scheduling, 2000, feasible, 2;
                 relax_integer=false, optimizer=HiGHS.Optimizer,
                 feasibility_timeout=120.0)
```

works unchanged (the unrelaxed time-indexed MIP solves quickly at that size).

## References

- Pritsker, A.A.B., Watters, L.J., Wolfe, P.M. (1969). Multiproject scheduling
  with limited resources: a zero-one programming approach. Management Science
  16(1).
- Sousa, J.P., Wolsey, L.A. (1992). A time indexed formulation of
  non-preemptive single machine scheduling problems. Mathematical Programming 54.
- Baptiste, P., Le Pape, C., Nuijten, W. (2001). Constraint-Based Scheduling
  (energetic reasoning). Kluwer.
