# Inverse Optimization

The `inverse_optimization` category generates LPs that infer an unknown forward
objective from observed decisions. It deliberately spans three important parts
of the inverse-optimization literature rather than treating inverse optimization
as one generic random matrix family.

| Variant | Evidence supplied to the inverse model | Inverse loss |
| --- | --- | --- |
| `classical` (default) | One exactly optimal production plan | Weighted L1 distance from a prior cost |
| `noisy_observations` | Multiple feasible, context-specific, suboptimal plans | Mean absolute suboptimality plus weighted L1 regularization |
| `shortest_path` | Multiple optimal routes on one spatial road network | Weighted L1 distance from prior arc times |

All three are pure continuous LPs. Their constructors own a local
`MersenneTwister`, store every sampled datum, and plant an independently
checkable feasible point. `build_model` is deterministic.

## Research basis and scope

The design follows the classical definition of Ahuja and Orlin: minimally
perturb a given cost vector so that an observed feasible decision becomes
optimal. They show that inverse LPs under L1 and L-infinity losses remain LPs
and identify shortest path, assignment, cut, and flow as canonical structured
cases [Ahuja and Orlin (2001)](https://doi.org/10.1287/opre.49.5.771.10607).

Modern inverse optimization distinguishes exact, classical models from
data-driven models that allow imperfect fit. Strong-duality reformulations and
absolute suboptimality are especially useful because they preserve linearity;
normalizing the inferred cost rules out the uninformative all-zero objective
[Chan, Mahmood, and Zhu (2023)](https://doi.org/10.1287/opre.2022.0382).
Noisy observations are not a corner case: statistically motivated inverse
methods explicitly treat measured optimizer decisions as corrupted or
imperfect [Aswani, Shen, and Siddiq (2018)](https://doi.org/10.1287/opre.2017.1705).

The generated numbers are calibrated synthetic proxies, not claims that one
parametric distribution fits every application. Sparse positive technological
matrices represent activities consuming small resource bundles. Positive
activity volumes, shadow prices, technological coefficients, context shocks,
and travel-time multipliers use moderate log-normal dispersion to capture
right-skew while avoiding solver-pathological tails. Panel utilization losses
use bounded beta distributions, with a separate contaminated-panel profile.
Road networks are sparse, spatial, bidirectional graphs with local, arterial,
regional, and highway speed bands. These choices represent the qualitative
structure of production-planning, preference-learning, and route-choice
applications reviewed in the literature; users should not interpret them as
fitted confidential operational datasets.

## Shared packing forward problem

The `classical` and `noisy_observations` variants use the forward packing LP

```math
\max_{x \ge 0}\ c^\mathsf{T}x
\quad \text{s.t.}\quad Ax \le b,
```

whose dual is

```math
\min_{y \ge 0}\ b^\mathsf{T}y
\quad \text{s.t.}\quad A^\mathsf{T}y \ge c.
```

`A` is sparse and strictly positive on its support. Every activity consumes one
to four resources, every activity has a nonzero column, and every resource is
used. A positive planted shadow-price vector gives
`c_true = A' * y_true`, after which both are scaled so `sum(c_true) = 1`.
The simplex normalization resolves the scale ambiguity inherent in learning a
linear objective. The prior cost is a multiplicatively perturbed and
renormalized version of the truth, and all true and prior costs lie strictly
inside broad admissible bounds.

### Classical exact variant

The observed plan `x_hat` is positive and `b = A * x_hat`, so every resource is
tight. The planted pair obeys primal feasibility, dual feasibility, and

```math
c_{true}^\mathsf{T}\hat{x} = b^\mathsf{T}y_{true}.
```

The inverse LP uses positive/negative deviations from the prior and minimizes a
weighted L1 distance. Its main rows are

```math
A^\mathsf{T}y = c,\qquad
c^\mathsf{T}\hat{x} = b^\mathsf{T}y,\qquad
\mathbf{1}^\mathsf{T}c = 1.
```

Equality in stationarity is valid here because every component of the observed
decision is strictly positive. There are exactly `3*n_activities + n_resources`
variables: inferred costs, dual prices, and two deviation blocks. The dimension
planner normally matches the target exactly.

### Noisy multi-observation variant

Each observation `k` has a latent optimal plan `x_star[k]` and capacity
`b[k] = A*x_star[k]`. The stored observation reduces each activity by a bounded
utilization loss, which guarantees `A*x_hat[k] <= b[k]` while creating positive
suboptimality. There are three reproducibly sampled behavior profiles:

- `routine`: small, concentrated losses;
- `heterogeneous`: broader activity-level losses;
- `outlier_contaminated`: routine observations plus one strongly under-utilized
  context.

For every observation the model creates dual prices and a nonnegative gap:

```math
A^\mathsf{T}y_k \ge c,
\qquad
g_k = b_k^\mathsf{T}y_k-c^\mathsf{T}\hat{x}_k \ge 0.
```

It minimizes the mean gap, divided by a planted decision-value scale, plus a
sampled weighted-L1 regularizer toward the prior cost. This is the tractable
absolute-suboptimality approach for known-feasible observations, rather than a
nonconvex minimum-distance estimator. Its variable count is
`3*n_activities + K*n_resources + K`.

## Inverse shortest path

The network variant builds a connected spatial graph, adds both directions of
each physical link, and samples asymmetric positive travel times from link
distance, road-class speed, and congestion. Multiple well-separated OD pairs
are selected, and Dijkstra's algorithm records their exact shortest paths under
the planted costs. The prior arc times contain multiplicative measurement error;
the draw is conditioned on making at least one observed route nonoptimal, so the
inverse instance always requires a nonzero calibration rather than admitting the
unmodified prior.

For observation `k`, node potentials implement the shortest-path dual:

```math
\pi_{k,v}-\pi_{k,u} \le c_{uv}\quad (u,v)\in E,
```

with the source potential anchored at zero. Equality between the destination
potential and observed route cost imposes strong duality and makes the route
shortest. A weighted L1 loss minimally changes prior arc times. The model has
`3*n_arcs + K*n_nodes` variables and typically lands within a few variables of
the requested size.

## Feasibility artifacts

For `feasible` requests:

- `classical` stores `ClassicalInverseWitness` with the true cost and shadow
  prices;
- `noisy_observations` stores `NoisyInverseWitness` with the common cost,
  observation-specific dual prices, and exact positive gaps;
- `shortest_path` stores `InverseShortestPathWitness` with true arc times and
  shortest-distance potentials.

The internal audit helpers are, respectively,
`_classical_inverse_witness_is_valid`, `_noisy_inverse_witness_is_valid`, and
`_inverse_shortest_path_witness_is_valid`.

For `infeasible` requests, the admissible parameter set contains explicit lower
and upper total-cost rows with `total_lower > total_upper`. The stored
`InverseCostSetCertificate` records that contradiction, checked by
`_inverse_cost_certificate_is_valid`. This represents incompatible calibration
requirements and proves infeasibility independently of the KKT system or a
solver. An `unknown` request reproducibly resolves to a realistic mixture and
stores exactly the corresponding artifact.
