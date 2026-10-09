# Project Selection

`project_selection/standard` generates a multi-year, multi-division capital
portfolio selection model (R&D or capital-program planning). One binary
variable funds each project; the default `relax_integer=true` solves its
`0 <= x <= 1` relaxation.

## Data

- **Organisation.** `n_divisions = clamp(round(n / 60), 1, n)` divisions of
  uneven size (Gamma weights, every division non-empty) and
  `n_themes = clamp(round(sqrt(n) / 3), 2, 12)` strategic themes.
- **Timing.** A 3-5 year horizon for `n <= 200`, otherwise 5-8 years. Each
  project lasts `1 + Poisson(1.3)` years (at most 4) and starts so it finishes
  inside the horizon.
- **Spending.** Lifetime cost is lognormal around $2M times a per-division
  programme scale (clamped to $50k-$200M), spread over the active years with a
  noisy front-loaded hump. Headcount (FTE) per year is spend times a
  project-specific labour share (25-75%) divided by a loaded FTE cost
  ($160k-$240k).
- **Platforms and prerequisites.** About 20% of projects are platform
  (infrastructure) projects. Each non-platform project needs 0, 1 or 2
  platforms (35/45/20%), preferring its own division, never one that starts
  later; 30% of platforms build on an earlier-indexed platform, so the
  prerequisite graph is a DAG with in-degree at most 2.
- **Exclusive alternatives.** Within each (division, theme), non-platform
  projects are grouped into 2-3 alternative scopes of one initiative with
  probability 0.3 per position; at most one per group may be funded.
- **Value and risk.** NPV is lifetime cost times an ROI drawn by class
  (platforms 0.6-1.2: they exist to enable others; low/medium/high classes
  1.1-1.6, 1.4-2.4, 1.8-4.0) times lognormal noise. Risk scores lie in
  `[1, 10]`, correlated with the ROI class; projects above 7.0 are high-risk.

## Model

```text
max  sum_p NPV_p x_p
s.t. sum_p spend_pt x_p            <= B_t        (corporate budget, each year)
     sum_{p in d} spend_pt x_p     <= B_dt       (division budget, each active division-year)
     sum_{p in d} fte_pt x_p       <= E_dt       (division headcount, same index set)
     x_p <= x_q                                  (q is a prerequisite of p)
     sum_{p in G} x_p <= 1                       (exclusive alternatives)
     sum_{p in d} x_p >= m_d                     (division delivery mandate, when m_d > 0)
     sum_p risk_p x_p <= R                       (portfolio risk)
     sum_{p high-risk} x_p <= H                  (high-risk count, when any exist)
     x binary
```

Rows: `n_years + 2 * (#active division-years) + #prerequisites + #groups +
#(m_d > 0) + 1 + [any high-risk]` — about 1.0-1.4 rows per column, with about
10-14 nonzeros per column. The pre-2026-10 generator drew an `O(n^2)`
dependency matrix (77k rows at 1k projects, 10M rows at 10k).

## Feasibility

- **`feasible`.** A portfolio is planted (35-45% of free-standing projects, a
  60% chance of one alternative per group, then closed under prerequisites).
  Division budgets and headcount are its usage times `U(1.04, 1.25)` plus a
  small allowance (capped at the division-year demand), the corporate budget is
  the larger of `U(1.01, 1.05)` times planted spend and `U(0.82, 0.92)` times
  the summed division budgets, mandates are `U(0.55, 0.95)` of the planted
  count, and the risk and high-risk caps sit above the planted values. The
  planted 0/1 point is stored as `ProjectPortfolioWitness`.
- **`infeasible`.** Same construction, then the largest division's mandate is
  raised to the smallest count whose cheapest lifetime costs reach
  `U(1.06, 1.20)` times its summed yearly budgets (the budgets are first shrunk
  to 60% of the division's total cost if they are too generous). Stored as
  `DivisionMandateCertificate`: summing that division's yearly budget rows
  gives `sum C_p x_p <= sum_t B_dt`, while any fractional `x` meeting the
  mandate costs at least the `m` cheapest projects. The proof spans the
  mandate row and all of the division's budget rows, so presolve does not
  detect it; simplex needs a few hundred iterations at 10k.
- **`unknown`.** A natural capital review: division budgets are 30-60% and
  headcount 35-65% of division demand, mandates 15-40% of division size, and
  the corporate budget is `kappa = exp(N(-0.40, 0.35))` times the spend of a
  heuristic mandate portfolio (cheapest projects plus prerequisite closure),
  capped at the summed division budgets. The LP decides; across 10 seeds both
  outcomes occur at 300, 3k and 30k projects.

## Scale

Exact variable count (`n = target_variables`); build is linear (about 0.1 s at
10k, under a second at 100k); HiGHS presolve keeps essentially all rows and
columns.
