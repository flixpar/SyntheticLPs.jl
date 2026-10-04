# Inverse Optimization

The `inverse_optimization` category generates LPs whose variables are parameters
of a forward optimization model. Observed decisions, routes, or dispatches are
data; the inverse model recovers objective coefficients that explain them.

The family deliberately spans exact, noisy, structured, and application-specific
inverse optimization. The distributions are calibrated synthetic profiles, not
claims that one fitted distribution represents every inverse-optimization
application.

| Variant | Evidence | Identification and loss |
| --- | --- | --- |
| `standard` (default) | One exact solution of a sparse covering LP | Box-bounded costs; weighted L1 distance from a prior |
| `classical_normalized` | One exact solution of a sparse packing LP | `sum(c) = n` (unit mean cost); weighted L1 distance |
| `linf` | One exact covering-LP solution | Box-bounded costs; weighted L-infinity distance |
| `noisy_observations` | Multiple feasible, imperfect packing decisions | Normalized mean duality gap plus weighted L1 regularization |
| `restricted_optimal_value` | An exact plan and a target value | Keep the plan optimal while attaining the target |
| `shortest_path` | Multiple routes on one spatial road network | Box-bounded arc times; weighted L1 distance |
| `shortest_path_layered` | One route on a controlled layered DAG | Box-bounded arc costs; weighted L1 distance |
| `market_clearing` | Multi-period copper-plate dispatch | Generator offers; weighted L1 distance |

All constructors own a local `MersenneTwister`, store every sampled datum, and
leave `build_model` deterministic. Requests are capped at 250,000 variables.
Every variant hits its target exactly (within a few variables), keeps rows
growing with columns (roughly 0.5–1.9 rows per variable), stays sparse (2–8
nonzeros per variable), and builds a 100k-variable model in well under ten
seconds. HiGHS presolve keeps essentially the whole model (≥ 85% of columns
and rows) for every variant and status except where an infeasible instance is
refuted outright.

## Exact inverse LPs

`standard` uses the forward model

```math
\min_{x \ge 0} c^T x \quad\text{s.t.}\quad Ax \ge b.
```

An observed feasible `x_hat` is optimal precisely when a dual vector `y >= 0`
satisfies

```math
A^T y \le c, \qquad b^T y = c^T x_{hat}.
```

The cost box is bounded away from zero, preventing the uninformative solution
`c = y = 0`. A planted active-row set, shadow prices, sparse activity support,
and a corrupted prior create diverse normal cones.

`classical_normalized` provides the complementary packing convention

```math
\max_{x \ge 0} c^T x \quad\text{s.t.}\quad Ax \le b,
```

with `sum(c) = n` — the simplex normalization rescaled to unit *mean* cost,
which `noisy_observations` shares. (A unit-sum normalization makes every cost
`O(1/n)` and every deviation weight `O(n)`; at 10k variables that put column
bounds near `1e-6` against objective weights near `1e4`, and HiGHS dual
simplex aborted on some instances.) Its exact observation is positive, so the stationarity
equalities `A' y = c` are valid. The two variants make the important modeling
choice between prior-centered interval identification and scale normalization
explicit.

`linf` shares the `standard` data mechanism but minimizes one epigraph variable
subject to `w_j * abs(c_j - prior_j) <= t`. It uses the direct two-inequality
form, without redundant positive/negative deviation variables.

## Noisy observation panels

`noisy_observations` draws several context-specific capacity vectors. Latent
optimal plans are reduced according to routine, heterogeneous, or
outlier-contaminated utilization profiles. Every recorded decision remains
forward-feasible but has a positive duality gap.

For each observation `k`, the inverse LP creates its own dual prices and gap:

```math
A^T y_k \ge c,
\qquad
g_k = b_k^T y_k - c^T x_k \ge 0.
```

The objective combines the mean gap, normalized by the mean observed plan
value per activity (so both terms scale linearly with size), with a sampled weighted-L1
regularizer. This is the tractable absolute-suboptimality model; it should not
be interpreted as the nonconvex statistically consistent estimator for general
measurement noise.

## Inverse shortest paths

The default `shortest_path` variant constructs a connected spatial network,
adds both directions of each physical link, samples road classes and
distance/speed-based travel times, and records multiple separated OD routes.
The prior is explicitly conditioned so at least one route is no longer optimal,
preventing zero-adjustment instances.

For every route, node potentials enforce reduced-cost inequalities, with the
destination potential equal to the route cost. `shortest_path_layered` retains
a smaller controlled family in which route length, topology, and reduced-cost
margins are transparent. Its prior is conditioned in the same way.

## Restricted optimal value and market inference

The general inverse optimal-value problem is NP-hard. The
`restricted_optimal_value` LP implements the tractable restriction in which a
specified plan must remain optimal while its value reaches a target. Separate
rows pin both the primal and dual value to that target.

Its infeasible profile is a target the forward model cannot certify: the
forward rows admit a perturbed *alternative plan* `x_bar` that, even with
every cost at its upper bound, is cheaper than the target (by 4–10%). Weak
duality then refutes every candidate,

```math
\tau = b^T y \le (A \bar x)^T y \le \bar x^T c \le \bar x^T u < \tau .
```

The target deliberately stays well inside the observed plan's pricing range
`[l' x_hat, u' x_hat]` and below the dual-row activity bound, so no single row
is out of range: HiGHS presolve cannot detect the contradiction and simplex
has to aggregate the dual-feasibility rows (`CheaperPlanCertificate`).

`market_clearing` infers time-invariant generator offer costs from a
multi-period merit-order dispatch with capacity and ramp constraints. It is a
copper-plate economic-dispatch benchmark, not a network-constrained LMP model.
The fleet mixes baseload, intermediate, and peaking units, with type-dependent
capacity and offer distributions and a diurnal demand profile. Feasible priors
are conditioned so the recorded dispatch is strictly suboptimal under the
prior: ordinary measurement-noise draws are resampled first, and if they all
preserve the merit order, the closest actionable offer pair is crossed around
its midpoint. This avoids zero-adjustment inverse instances while minimizing
the fallback distortion from the planted offers.

## Feasibility profiles

Status requests alter the inverse evidence rather than adding unrelated
contradictory rows.

- `feasible` stores an independently checkable ground-truth witness.
- `infeasible` stores a certificate tied to the observation:
  - a strictly interior positive decision under positive normalized costs;
  - a panel fit tolerance below a lower bound implied by feasible improvements;
  - a target value above the cost of a feasible alternative plan priced at the
    upper cost bounds (weak duality);
  - an alternative route cheaper under every admissible arc-cost vector;
  - or a dispatch that reverses every admissible merit order.
- `unknown` stores neither witness nor certificate. It samples exact, near
  conflict, and ambiguous mechanisms so both solver outcomes occur across a
  corpus.

Because noisy absolute-suboptimality is feasible without a fit requirement,
its infeasible profile includes a meaningful maximum mean-gap tolerance. The
certificate derives a lower bound from the known feasible latent plans and the
componentwise cost floors. Whenever a tolerance row is present, the model also
states the shadow-price upper bounds it implies (each gap is at most `K` times
the tolerance, so `y_ki <= (K * tol + x_k' u) / b_ki`); they cut nothing but
let HiGHS certify the infeasible profile reliably instead of stalling with an
unknown status on an unbounded dual block.

## Validation

The focused tests check:

- exact or near-exact target sizing and deterministic rebuilds;
- sparse-matrix, topology, fleet, and observation invariants;
- witness and certificate arithmetic recomputed from raw fields;
- feasible and infeasible solver contracts across variants and scales;
- informative priors for both path families and market dispatch; and
- end-to-end semantics by solving the inverse model and then independently
  solving or evaluating the recovered forward model.

## References

- R. K. Ahuja and J. B. Orlin, *Inverse Optimization*, Operations Research
  49(5), 2001.
- A. Aswani, Z.-J. M. Shen, and A. Siddiq, *Inverse Optimization with Noisy
  Data*, Operations Research 66(3), 2018.
- T. C. Y. Chan, R. Mahmood, and I. Y. Zhu, *Inverse Optimization: Theory and
  Applications*, Operations Research 73(2), 2025.
- J. Jia, X. Guan, X. Qian, and P. M. Pardalos, *Restricted Inverse Optimal
  Value Problem on Linear Programming under Weighted L1 Norm*, 2023.
- C. Ruiz, A. J. Conejo, and D. Bertsimas, *Revealing Rival Marginal Offer
  Prices via Inverse Optimization*, IEEE Transactions on Power Systems 28(3),
  2013.
