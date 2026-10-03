# Portfolio

Scenario-based portfolio construction LPs on a shared factor-structured market.
Two variants, both pure LPs:

| Variant | Mandate | Risk measure | Default infeasibility |
| --- | --- | --- | --- |
| `cvar` (default) | multi-asset allocation vs a cap-weighted benchmark | Rockafellar–Uryasev CVaR limit | CVaR limit below the crash-tail loss bound |
| `tracking_error` | enhanced indexing under an ESG exclusion list | mean absolute active return (MAD tracking error) | commodity-factor band unattainable after a fossil-fuel exclusion |

## Shared scenario market (`portfolio.jl`)

Asset returns in scenario `s` are

```text
r[s, i] = Σ_k F[s, k] · B[i, k] + E[i, s]
```

- **Factors `F`** (monthly): a fat-tailed market factor (Student-t, ν = 5) with
  a crash regime (2–8% of scenarios, mean −12%), 3–6 style factors, and one
  industry factor per sector; style and industry volatility rise 1.8× in a
  crash.
- **Exposures `B`**: market beta and style loadings are dense; industry
  exposure is the 0/1 sector indicator.
- **Idiosyncratic shocks `E`** are *sparse*: each scenario draws Student-t
  shocks for a random subset of `J` assets (12–32), scaled by `√(n/J)` so each
  asset's idiosyncratic variance is preserved in expectation.
- Forecast returns are factor premia plus a small alpha; the benchmark is
  cap-weighted (log-normal caps).

The LPs use **factor-exposure variables** `f = Bᵀx` (one definition row per
factor), so a scenario row has `K + J + O(1)` nonzeros instead of `n`. The
previous formulation materialized the dense `n_scenarios × n_assets` matrix
(8M nonzeros at 10k variables for `cvar`, 32M for `tracking_error`, 380 s to
build; ≈ 6 GB at 100k). Now 100k-variable instances have 2.4M (`cvar`) to 5.5M
(`tracking_error`) nonzeros and build in a few seconds.

Helpers: `_portfolio_waterfill` (cap-respecting allocation), `_portfolio_cvar`
(exact CVaR of a loss vector), `_portfolio_extreme_on_capped_simplex`, and
`_portfolio_floor_bound` (Lagrangian lower bound for the crash-tail
certificate).

## `cvar`

**Universe.** 3–5 asset classes with class-specific beta, idiosyncratic
volatility, style sensitivity, and trading cost: developed equity, government
bonds (slightly negative beta — they rally in a crash), emerging equity,
credit, alternatives. 3–6 regions, 8–12 sectors.

**Formulation.**

```text
max  Σ_i μ_i x_i − Σ_i c_i (buy_i + sell_i)
s.t. style_k − Σ_i B[i,k] x_i = 0            market and style exposures (banded by bounds)
     sector_g − Σ_{i∈g} x_i = 0               sector weights (capped by bounds)
     z_s + F_s·f + Σ_{i∈J_s} E[i,s] x_i + α ≥ 0       shortfall rows
     α + Σ_s z_s / ((1−β) S) ≤ cvar_limit
     Σ_i x_i = 1
     Σ_{i∈region} x_i ≤ region_upper
     class_lower ≤ Σ_{i∈class} x_i ≤ class_upper
     buy_i − sell_i − x_i = −b_i,   Σ (buy + sell) ≤ turnover_limit
     0 ≤ x_i ≤ max_position_i,  z, buy, sell ≥ 0,  α free
```

`β ∈ {0.90, 0.95, 0.975, 0.99}`.

**Sizing.** Variables `= 3·n_assets + n_factors + n_scenarios + 1`, exact for
targets ≥ 40; `n_assets ≈ 10–18%` of the target.

**Feasibility.**

- `feasible`: the benchmark is water-filled under 90% of the position caps
  (every cap respected exactly — the old clip-then-renormalize reference could
  exceed caps) and every limit is widened around it. `CVaRWitness` stores the
  weights, exposures, VaR level `α`, CVaR, and turnover.
- `infeasible` (`infeasibility_mode`):
  - `:crash_tail` (≈ 50%, default): `cvar_limit` is set 15–40% below a lower
    bound on the crash-tail loss. Uniform multipliers on the worst
    `⌈(1−β)S⌉` market scenarios turn the shortfall rows plus the CVaR row into
    `cvar_limit ≥ ℓ·x` (mean tail loss per asset); the budget row, the class
    floor rows, and the position bounds then bound `ℓ·x` below by a Lagrangian
    dual value (`CVaRTailCertificate` stores the tail, `ℓ`, and the
    multipliers). The class floors are essential — without them the bond
    sleeve could gain in the crash. This replaces the old heuristic "shrink the
    limit by 1–15%" mode, which had no proof.
  - `:class_floor` (≈ 25%): class floors sum to 1.05–1.2 (`ClassFloorCertificate`).
  - `:turnover_sector` (≈ 25%): the 1–3 largest sectors are capped at 30–60%
    of their benchmark weight, forcing turnover ≥ 2× the deficit, above the
    turnover limit (`TurnoverSectorCertificate`).
- `unknown`: the natural mandate (all limits drawn relative to the benchmark;
  CVaR limit 0.55–0.95× the benchmark's CVaR, while the attainable minimum is
  typically 0.55–0.9× it) with no repair — both outcomes occur.

## `tracking_error`

**Universe.** An equity index: sector 1 is *energy* (5–10% of names, 8–12% of
benchmark weight). The last style factor is a *commodity* factor with loadings
1.5–3 for energy names and −0.01 to 0.1 elsewhere. An ESG exclusion list removes
names from the investable universe (the benchmark still holds them).

**Formulation.**

```text
max  Σ_{i investable} α_i x_i
s.t. f_k − Σ_i B[i,k] x_i = 0                   all factors; |f_k − f_k^b| ≤ δ_k as bounds
     u_s ≥ ±(F_s·f + Σ_{i∈J_s} E[i,s] x_i − r_s·b)    two rows per scenario
     (1/S) Σ_s u_s ≤ te_budget
     Σ_i x_i = 1,   0 ≤ x_i ≤ max_position_i
```

Sectors with no investable names keep no lower band (their active weight is
the exclusion itself).

**Sizing.** Variables `= n_investable + n_factors + n_scenarios`, exact for
targets ≥ 60; `n_assets ≈ 15–30%` of the target. Rows
`= 2·n_scenarios + n_factors + 2`.

**Feasibility.**

- `feasible`: a small random exclusion list (1–4% of names); the investable benchmark is
  water-filled under 90% of the caps and the bands and TE budget are widened
  around it (`TrackingErrorWitness`).
- `infeasible`: the whole energy sector is excluded while the commodity band's
  lower side exceeds every investable name's commodity loading
  (`TrackingErrorCertificate`): the factor row and the budget row give
  `f ≤ max loading < band_lower`. The band is also kept below the factor
  row's own activity bound under the position caps, so the contradiction needs
  the budget row and is not visible to presolve's single-row checks.
- `unknown`: random exclusions and the natural mandate (bands relative to the
  benchmark, TE budget 0.4–1.1× the naive exclusion-renormalized portfolio's
  TE, floored at 5% of the benchmark's scenario MAD) with no repair; both outcomes occur.

## Practical notes

Both variants have a free/banded factor-exposure block that couples every
scenario row to the asset weights, a large block of shortfall/absolute-value
rows, and overlapping policy rows — non-trivial for simplex at every size.
