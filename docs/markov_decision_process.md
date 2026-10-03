# Markov Decision Process

Linear programs of finite Markov decision processes in the occupation-measure
(dual) form of Manne (1960), with the budget rows of Altman's constrained MDPs.
This is a genuinely important LP class: the LPs are sparse, structured entirely
by the transition kernel, scale to millions of state-action pairs, and have a
special relationship with the simplex method — Howard's policy iteration is a
block-pivoting simplex method on exactly this LP, and Ye (2011) showed the
simplex method with Dantzig's rule is strongly polynomial on discounted MDPs.

All four variants are pure continuous LPs (`relax_integer` is a no-op), build
the model from struct data only, and draw all randomness from a constructor-
local RNG. The MDPs come from operational models rather than random kernels.

## Variants

| Variant | Underlying model | State | Action | Budget rows |
|---|---|---|---|---|
| `inventory_control` (default) | seasonal joint pricing and replenishment with lead time | net inventory, order pipeline, season phase | order quantity × price level | optional shortage (service-level) row |
| `queueing_control` | overloaded two-station tandem queue, uniformized CTMC | queue lengths at both stations | admit/reject × service-rate levels | optional blocking-rate SLA row |
| `machine_maintenance` | condition-based maintenance with spares and a production calendar | condition level, spares on hand, calendar phase | run at a speed / repair / replace / expedite / wait × spare order | optional downtime (availability) row |
| `constrained` | a constrained MDP over one of the three models | as base | as base | 2-3 dense budget rows on conflicting streams |

## Formulation (shared)

Sets and data: states `S`, state-action pairs `k ∈ A(s)` (one LP column each),
transition probabilities `P(t | k)`, per-pair primary cost `c_k` and secondary
nonnegative metric streams `d^j_k`.

```text
x_k >= 0                                     (frequency of pair k)

Discounted (γ < 1):
  Σ_{k∈A(s)} x_k - γ Σ_k P(s|k) x_k = rhs_s   for every state s
  rhs = (1-γ) · |S| · μ,   μ = 0.7·(domain start profile) + 0.3·uniform

Average cost (γ = 1):
  Σ_{k∈A(s)} x_k - Σ_k P(s|k) x_k = 0         for every state s
  Σ_k x_k = |S|                               (normalization row)

Budget rows (when present):
  Σ_k d^j_k x_k <= B_j

Objective: minimize Σ_k c_k x_k
```

Both criteria are scaled so the occupation measure sums to `|S|` (average state
occupancy 1), which keeps column values well above solver tolerances; under the
discounted criterion the state-relevance weights have full support, so no
balance row is forced to zero. The criterion is sampled per instance
(discounted 65%, with an effective horizon `1/(1-γ)` drawn log-uniformly; average
cost 35%). Rows: `|S|` balance rows, `+1` normalization row under the average
criterion, `+` the budget rows; columns: exactly the number of state-action
pairs.

Each pair's successors are a handful of states (demand support, a few
uniformized events, a few degradation increments), so nonzeros grow linearly.
Action sets never offer two actions with the same transition law in a state
(e.g. an idle station has no rate menu), so there are no parallel columns.

## `inventory_control`

Single item, periodic review, backlogging (beyond `max_backlog` demand is lost),
replenishment lead time `ℓ ∈ {0,1,2}` periods, and a position cap
`i + Σ pipeline <= max_inventory`. The review cadence is weekly (cycle of
`{1,2,4,13,26,52}` phases) or daily (`{7,...,364}`, with a weekday pattern), and
demand is negative binomial with a seasonal mean and a price elasticity across
at most three price levels. Costs: fixed and variable ordering, holding,
backorder and lost-sale penalties, minus revenue. Streams: `:shortage`,
`:on_hand`, `:orders`. The position cap is set at 1.1-1.6 × the peak lead-time
demand, so some shortage is unavoidable under every policy. Reference policy:
base-stock on the inventory position at a 90-98% service quantile, at the price
level nearest the base price.

## `queueing_control`

Two stations in tandem with finite buffers and blocking after service. The
controller admits or rejects arrivals at station 1 (forced reject when full)
and picks a service-rate level at each busy station from a 2-3 level menu with
convex power-law energy cost. Uniformization at `Λ = λ + max μ1 + max μ2`.
Peak load exceeds the fastest bottleneck rate (`ρ ∈ [1.03, 1.3]`), the regime
where admission control matters, so some rejection is unavoidable. Streams:
`:rejection`, `:energy`, `:congestion`. Reference policy: admission threshold,
fast service above a queue-length threshold.

## `machine_maintenance`

Condition levels `0..C` (`C` = failed) with speed-dependent Poisson wear that
accelerates with condition, speed-dependent shock failures, imperfect repair,
preventive and corrective replacement (needing a spare), emergency replacement
with an expedited part, or waiting; spare parts on hand with replenishment
orders; and a production calendar whose margin and load vary seasonally.
Streams: `:downtime`, `:labour`, `:spares_held`. Reference policy: control-limit
replacement with base-stock spares.

## `constrained`

Builds one of the three operational models without its own service row and
adds 2-3 budget rows: the service stream (`:shortage` / `:rejection` /
`:downtime`) plus a conflicting stream (inventory: `:on_hand`; otherwise a
random one of the other two), or all three. Each budget row has a nonzero on
most columns — dense coupling rows over the sparse balance rows, which make
the optimal policy randomized in a few states.

## Feasibility control

The unconstrained MDP LP is always feasible and bounded (every stationary
policy induces a feasible occupation measure), so feasibility is governed by
budget rows. The operational variants carry their service row with probability
1/2 (always for `infeasible`); the `constrained` variant always has 2-3 rows.

- `feasible`: the witness (`MDPOccupationWitness`) is the exact occupation
  measure of the reference policy, computed by one sparse LU solve
  (`(I - γ P_π)ᵀ y = rhs`, or the stationary distribution with a recurrent state
  pinned); for several rows it is mixed with the occupation measure of the
  policy optimal for a random weighting of the budget streams, so the budgets
  sit near the Pareto frontier. Each budget is the witness's value plus a 5-25%
  margin. Tests check every balance, normalization, and budget row against the
  witness without a solver.
- `infeasible`: an `MDPDualCertificate` — a Farkas certificate built from LP
  rows alone. For budget weights `w >= 0` and combined metric `d_w = Σ w_j d^j`,
  the stored per-state potential `v` (and gain `g` for average cost) satisfies
  `g + v_s - γ Σ_t P(t|k) v_t <= d_w(k)` at every pair; multiplying the balance
  rows by `v` (and normalization by `g`) shows every feasible `x` has
  `Σ d_w x >= lower_bound`, while `Σ w_j B_j = (1 - m)·lower_bound` with
  `m ∈ [0.08, 0.25]`. The potential is the optimal value function of `d_w` from
  policy iteration (sparse LU evaluations), made rigorously dual feasible by the
  MacQueen (discounted) or Odoni (average, via a `1 - 1e-4` surrogate discount)
  correction, so the lower bound is essentially the true minimum — refuting the
  budgets needs the whole transition structure, not an aggregate presolve
  argument (`Σx` times the smallest coefficient is typically zero here). In the
  `constrained` variant the deficit is spread so that, whenever the streams
  conflict, each budget individually exceeds its own stream's optimum: no
  single row is infeasible, only the combination.
- `unknown`: single row — `B = L + u·max(R - L, 0.3 L)` with `u ∈ [-0.5, 1]`
  between the certified optimum `L` and the reference value `R`; several rows
  — `u_j ∈ [0.4, 1.4]`, decided by the rows' conflict. Instances without a row
  are the plain (always feasible) MDP LP.

## Sizing

`target_variables` is the number of state-action pairs. Each variant searches
its discrete dimensions (lead time / cycle / price menu / demand scale; buffers;
condition levels / spares / calendar) for the exact pair count closest to the
target, mildly preferring per-seed structural draws, so the count is typically
within 1-5% of the target (tiny targets round to the smallest model). Rows grow
with size: the inventory search penalizes more than ~16 columns per state, the
queue has 8-18 actions per interior state, maintenance ~10-20. Targets above
`MDP_MAX_PAIRS = 1_000_000` raise an `ArgumentError`.

## Solver notes

- HiGHS presolve leaves these LPs essentially intact (no dominated columns, no
  empty rows: every state has inflow and full-support right-hand sides).
- These LPs are hard for simplex: thousands of iterations at 10k columns, with
  heavy degeneracy under the average criterion.
- HiGHS's default dual simplex sometimes cannot finish the infeasibility proof
  on infeasible instances (it reports "possibly dual unbounded", its proof check
  rejects the ray, and it returns `OTHER_ERROR`), because MDP bases are
  ill-conditioned (`≈ 1/(1-γ)`). The planted certificates are exact; HiGHS IPM
  and primal simplex confirm infeasibility. When verifying `infeasible`
  instances through `generate_problem(...; optimizer=...)`, pass e.g.
  `optimizer_with_attributes(HiGHS.Optimizer, "solver" => "ipm")`.

## References

- A. S. Manne, "Linear programming and sequential decisions", Management
  Science 6 (1960).
- E. Altman, *Constrained Markov Decision Processes*, Chapman & Hall (1999).
- Y. Ye, "The simplex and policy-iteration methods are strongly polynomial for
  the Markov decision problem with a fixed discount rate", Math. of OR 36 (2011).
- M. L. Puterman, *Markov Decision Processes*, Wiley (1994) — MacQueen and
  Odoni bounds, uniformization.
- A. Federgruen, A. Heching, "Combined pricing and inventory control under
  uncertainty", Operations Research 47 (1999).
