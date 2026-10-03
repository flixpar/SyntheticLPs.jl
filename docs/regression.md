# Regression

Statistical-estimation LPs: robust, sparse, and minimax data fitting plus a
1-norm linear classifier. Five variants, each with its own data profile and its
own infeasibility mechanism. All are pure LPs (no integer variables).

| Variant | Model | Data profile | Infeasibility certificate |
| --- | --- | --- | --- |
| `lad` (default) | two-row L1 regression + absolute-error budget | dense correlated covariates, one-hot fixed effects, heavy tails, outliers, replicates | replicate pairs: `Σ e ≥ Σ (y_a − y_b)` |
| `quantile` | L1-penalized quantile regression + prediction bands | sparse indicator codes (Zipf), heteroscedastic skewed cost | cohort-mean band above its members' mean upper band |
| `chebyshev` | weighted minimax B-spline surface fit + accuracy cap | scattered survey points, terrain surface, bounded instrument errors | local null vector of one knot cell |
| `basis_pursuit` | weighted L1 sparse recovery `Ax = b` | Gaussian / coherent / sparse measurement matrices (column-capped) | one measurement row is a combination of 2–4 others with an inconsistent RHS |
| `l1_svm` | 1-norm SVM + hinge-loss budget | tf-idf bag-of-words, sparse planted classifier, label noise | conflicting duplicate documents: `ξ_a + ξ_b ≥ 2` |

## Scale and footprint

Every variant keeps nonzeros linear in the target, and model building is
row-wise with pre-sized affine expressions (no dense `n × m` products, no
quadratic expression construction). Measured at 100k variables:

| Variant | Rows | Nonzeros | Build |
| --- | ---: | ---: | ---: |
| `lad` | ≈ 1.7 × cols | 1.8–3.6M | < 1 s |
| `quantile` | ≈ 0.25 × cols | 0.6–0.9M | < 1 s |
| `chebyshev` | ≈ 4–5 × cols | 2.3–3.3M | ≈ 1 s |
| `basis_pursuit` | ≈ 0.2 × cols | ≤ 3M | ≈ 2.5 s |
| `l1_svm` | ≈ 0.65 × cols | 2.6–3.9M | 1–2 s |

The previous dense formulations needed `n_features × n_samples` nonzeros
(`chebyshev` took 80 s to build at 1k variables and 10M nonzeros; `lad` 287 s
and 27M nonzeros at 10k; `basis_pursuit` 19M nonzeros at 10k).

HiGHS presolve keeps ≥ 93% of rows and columns on every variant at 10k, and the
infeasible instances are disproved by simplex, not presolve.

## `lad` — robust fixed-effects LAD regression

**Data.** `n_samples ≈ 80–90%` of the target; the rest are coefficients:
an intercept, 6–20 dense continuous covariates (driven by 2–4 latent factors,
30% log-normally skewed, each with its own unit scale), and 1–3 categorical
factors with Zipf-distributed level frequencies, reference-coded as fixed
effects (every level is observed). Noise is Student-t (ν ∈ [1.5, 5]); 2–10% of
samples are gross vertical outliers (15–60σ), half of them also bad leverage
points. 10–25% of samples are *replicates*: exact repeats of an earlier design
row with fresh noise. Samples carry log-normal survey weights.

**Formulation.**

```text
min  Σ_i w_i e_i
s.t. e_i + ŷ_i ≥ y_i,   e_i − ŷ_i ≥ −y_i              (two rows per sample)
     Σ_i e_i ≤ L                                       (absolute-error budget)
     ŷ_i = β₀ + x_i·β + Σ_f γ_{f, level_f(i)},  β₀, β, γ free,  e ≥ 0
```

**Sizing.** Variables `= 1 + n_continuous + Σ_f (levels_f − 1) + n_samples`,
exact for targets ≥ 12. Rows `= 2·n_samples + 1`.

**Feasibility.**

- `feasible`: `L = (1.05–1.25) × loss(β*)`; `feasible_witness::LADWitness`
  holds the planted coefficients and their loss.
- `infeasible`: `L = (0.70–0.90) × Σ_pairs |y_a − y_b|`;
  `infeasibility_certificate::LADCertificate` lists the oriented replicate
  pairs. Adding each pair's two residual rows cancels the shared fit, so
  `e_a + e_b ≥ y_a − y_b`; summing contradicts the budget row. The proof needs
  `2·|pairs| + 1` rows.
- `unknown`: `L = (0.86–1.00) × loss(β*)`; the LAD optimum is typically
  92–97% of the planted loss, so both outcomes occur.

## `quantile` — penalized quantile regression with prediction bands

**Data.** Claims-style: each sample has `1 + Poisson(4–12)` distinct indicator
codes from a Zipf vocabulary (every code in ≥ 5 samples) plus 3–6 standardized
demographics. 5–15% of codes drive the cost (mostly positive log-normal
effects). The response noise is right-skewed (centred Gamma) with a scale that
grows with the expected cost (heteroscedastic). `τ ∈ {0.1, 0.25, 0.5, 0.75, 0.9}`.
The L1 penalty `λ` is small enough relative to the minimum code frequency that
presolve cannot fix code coefficients to zero by dual arguments.

**Reference profiles.** About 1% of samples (≤ 200) plus up to 25 *cohorts*:
3–6 member patients and their exact average profile (fractional code
prevalence, mean demographics). Every reference has a ranged plausibility band
on its predicted quantile.

**Formulation.**

```text
min  Σ_i (τ u_i + (1−τ) v_i) + λ Σ_j (β⁺_j + β⁻_j)
s.t. u_i − v_i + b₀ + z_i·δ + x_i·(β⁺ − β⁻) = y_i          (one row per sample)
     ℓ_r ≤ b₀ + z_r·δ + x_r·(β⁺ − β⁻) ≤ h_r                 (one ranged row per reference)
     u, v, β± ≥ 0;  b₀, δ free
```

**Sizing.** Variables `= 1 + n_demographic + 2·n_codes + 2·n_samples`, exact
for targets ≥ 30.

**Feasibility.** The residual split absorbs any fit, so feasibility is decided
by the band system.

- `feasible`: band centres are within half a half-width of the planted
  predictions (`QuantileWitness`).
- `infeasible`: one cohort's lower band is set above its members' mean upper
  band (`QuantileCertificate`: weight member rows by `1/k`, subtract the cohort
  row, get `0 ≥ gap > 0`).
- `unknown`: band centres are the planted predictions plus independent noise,
  with the noise scale calibrated (by bisection on the analytic overlap
  probability) so that about half of the instances have mutually consistent
  cohort bands.

## `chebyshev` — weighted minimax spline surface fit

**Data.** A tensor-product uniform B-spline space of degree 1–3 (degree 3 only
up to 30k variables) on an `nx × ny` knot grid over the unit square. Measurement
sites: one per knot cell, `(degree+1)² + 1` in one designated cell, and the
rest a mix of uniform coverage and Gaussian survey clusters (`n_samples ≈
1.8–3×` the coefficient count). The surface is a terrain-like sum of Gaussian
bumps, a ridge, and a trend, represented by its quasi-interpolant. Errors are
bounded (U-shaped Beta on `±η/w`), with precision weights `w ∈ {1, 2, 4}`.
Basis values below `1e-3` are stored as zeros (coefficient-range hygiene).

**Formulation.**

```text
min  t
s.t. w_i (y_i − B(p_i)·c) ≤ t,   −w_i (y_i − B(p_i)·c) ≤ t
     0 ≤ t ≤ t_max,   |c_j| ≤ C
```

Each row has only `(degree+1)² + 1` nonzeros. The coefficient box `C` (twice
the larger of the planted coefficients and the data range) keeps simplex bases
well conditioned; without it HiGHS hit solve errors from near-singular bases.

**Sizing.** Variables `= (nx+degree)(ny+degree) + 1`, within a few percent of
the target. Rows `= 2·n_samples`.

**Feasibility.**

- `feasible`: `t_max = (1.1–1.4) ×` the planted surface's maximum weighted
  residual (`ChebyshevWitness`).
- `infeasible`: same specification, then the designated cell's measurements get
  a localized oscillation along the left null vector `λ` of their local basis
  block. Since `Σ λ_k B(p_k) = 0`, every fit satisfies
  `|Σ λ_k y_k| ≤ t_max Σ |λ_k|/w_k`, which the data violate by ≥ 25%
  (`ChebyshevCertificate`).
- `unknown`: `t_max = (0.88–1.02) ×` the planted maximum residual; the minimax
  optimum is typically 0.90–0.99× it.

## `basis_pursuit` — weighted sparse recovery

`min Σ_j w_j |x_j|  s.t.  A x = b` in split form `x = x⁺ − x⁻`.

**Matrix profiles.** `gaussian` (row-whitened when dense, otherwise sparse
Gaussian columns), `correlated_columns` (groups of highly coherent columns
sharing a prototype and support), `sparse_measurements` (≈12% signed entries
per column). Each column holds at most
`max(8, BASIS_PURSUIT_NNZ_BUDGET ÷ n_features)` entries (budget 1.5M, so ≤ 3M
LP nonzeros); instances up to ≈1.5k features stay dense. Entries below `1e-3`
of their column's largest magnitude are dropped.

**Sizing.** Variables `= 2·n_features`: even targets exact, odd targets round
up, minimum 2. Measurements are 30–50% of the features.

**Feasibility.** `feasible`: `b = A·source_signal` (sparse planted signal).
`infeasible`: one measurement row is overwritten by a combination of 2–4 other
rows with an inconsistent right-hand side (`BasisPursuitCertificate` stores
rows, multipliers, and the nonzero `rhs_gap`); the contradiction needs ≥ 3 rows,
so presolve's parallel-row check does not see it. `unknown`: planted feasible
with probability 0.8, certified infeasible otherwise (`resolved_status`) — an
underdetermined full-row-rank system has no natural borderline.

## `l1_svm` — 1-norm SVM text classification

**Data.** Documents have log-normal lengths (mean 15–45 distinct terms) drawn
from a Zipf vocabulary (each term in ≥ 3 documents); entries are
`(1 + log tf)·idf`, L2-normalized per document. Labels come from a sparse
planted classifier on mid-frequency terms with logistic label noise and 15–50%
minority share. 1–3% of documents are exact duplicates re-posted with the
opposite label. Documents outnumber terms 3–6×, so the data are not separable.

**Formulation.**

```text
min  λ Σ_j (w⁺_j + w⁻_j) + Σ_i c_i ξ_i
s.t. y_i (x_i·(w⁺ − w⁻) + b) + ξ_i ≥ 1,    Σ_i ξ_i ≤ B
     w±, ξ ≥ 0;  b free;  c_i class-balancing weights
```

**Sizing.** Variables `= 2·n_terms + 1 + n_documents`, exact for targets ≥ 20.

**Feasibility.** `feasible`: `B = (1.05–1.2) ×` the planted classifier's hinge
loss (`L1SVMWitness`). `infeasible`: `B = (0.7–0.9) × 2·|pairs|`
(`L1SVMCertificate`: each conflicting pair forces `ξ_a + ξ_b ≥ 2`).
`unknown`: `B` lies 25–75% of the way from `2·|pairs|` to the planted hinge
loss; the hinge minimum typically sits 25–70% of the way.
