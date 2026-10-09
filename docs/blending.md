# Blending

Secondary-aluminium alloy blending: casthouses charge furnaces with scrap,
primary metal and master alloys so every melt lands inside its alloy's
chemical-composition window — the classic large-scale blending LP. The three
variants share one catalog (`common.jl`) and are structurally different LPs.
All are pure continuous LPs, draw from a constructor-local
`MersenneTwister(seed)` (the global RNG is untouched) and build
deterministically.

## Variants

| Variant | Key structure | Scale comes from |
|---|---|---|
| `standard` (default) | max-margin order blending: element-mass window rows, output windows, scrap-lot and primary/master-alloy availability shared across orders and plants | customer orders × compatible materials |
| `multi_period` | multi-plant, multi-period production planning: purchases on a moving market, raw and finished-goods inventory staircases, melt and yard capacities | plants × periods × grade portfolio |
| `robust` | `standard` with Bertsimas–Sim budgeted protection of the tramp-element limits against scrap-assay error (auxiliary protection variables, three-term rows) | orders × scrap lots × uncertain elements |

The former `multi_product` and `equipment_batches` variants were near-duplicates
(allocation + per-destination quality bands + shared supply + one capacity
row, iid-uniform attributes, 8–16 simplex iterations at 50k) and were
replaced by `multi_period` and `robust`; the former single-blend `standard`
silently capped at 500 variables.

## Catalog

- **Elements** (wt%): Si, Fe, Cu, Mn, Mg, Zn, Cr, Ti.
- **Alloy grades** (12, Aluminum Association windows): AA1050, 3003, 3104,
  5052, 5182, 6061, 6063, 6082, 2024, 7075 and casting alloys A356, A380, with
  prices and order frequencies.
- **Materials**: primary metal (P1020, P0406); master alloys and alloying
  metals (AlSi50, Al-50Cu, AlMn20, magnesium ingot, zinc ingot, AlCr10,
  Al-10Ti), allowed only in grades with a minimum on their element; 13 ISRI-style scrap classes (foil, clips, used
  beverage cans, taint/tabor, 6063 extrusions, 5xxx clips, 2xxx/7xxx turnings,
  painted siding, twitch, zorba, mixed cast, A356 wheels) with segregation
  rules deciding which alloy families they may enter.
- **Scrap lots** scatter around their class chemistry (lognormal) with a
  lot-wide contamination factor on Fe/Cu/Zn that also discounts the price —
  dirty scrap is cheap, dilution with primary metal is expensive, which is the
  trade-off that makes the LP non-trivial. Assays are reported at three
  significant digits with a 0.005 wt% detection limit.

## `standard`

Variables `x[i, o] ≥ 0` (tonnes of material `i` in order `o`, compatible pairs
only: a plant's scrap lots reach only its own orders) and the charge mass
`charge[o]`. Maximize `Σ (price[o]·yield[i] − cost[i]) x[i, o]`. Rows per order:
charge balance `Σ_i x = charge`, output window
`demand_min ≤ Σ_i yield_i x ≤ demand_max` (two rows), and composition rows in
element-mass form `Σ_i comp[e,i] x ≥ lo[e,o]·charge`,
`Σ_i comp[e,i] x ≤ hi[e,o]·charge` (only when some candidate material lies
outside the bound); per material `Σ_o x[i, o] ≤ availability[i]`.

The element-mass form (instead of homogeneous `Σ (comp − lo) x ≥ 0`) keeps the
assay values as coefficients and avoids near-cancelling differences.

Sizing: plants `≈ target/8000` (1–30) with `≈ target^0.4` lots each; orders are
added round-robin until pairs + charge variables reach the target (within one
order, a few dozen variables).

Feasibility: `feasible` plants a hand-written recipe per order (`_blend_recipe`:
1–3 scrap lots capped so no tramp element exceeds 85% of its limit, primary
metal, master alloys added by fixed point to a target inside each window),
widens windows only where the recipe falls outside, and sets availabilities
1.02–1.40× its use; `blend_charge_satisfies` checks it. `infeasible`
(`BlendInfeasibilityCertificate`): **element shortage** (40%) — the carriers of
one order's alloying element are cut until the exact covering-knapsack
optimum of `Σ (comp − lo) x` over charges meeting its minimum output is
negative; else a **family metal shortage** — all materials of one alloy family
are cut until the family's minimum output exceeds the recoverable metal by
8–25%, choosing a family in which every order could still be filled alone (so
no single output row is contradicted by implied bounds). `unknown`: registered
windows; scrap and primary metal drawn against the contracted output with one
instance-wide market tightness (about 40–50% of instances are infeasible).

## `multi_period`

Indices: materials (primaries, needed master alloys, 2–13 scrap classes at
class chemistry), plants with 1–8-grade portfolios, periods (1–104). Variables:
`x[i,g,k,t]`, `charge[g,k,t]`, `buy[i,k,t]`, `stock[i,k,t]`, `fg[g,k,t]`.
Minimize purchases at a mean-reverting market price + holding + melting. Rows:
charge balance and element-mass composition rows per `(g,k,t)`; finished-goods
balance `fg[t] = fg[t−1] + Σ yield·x − demand`; raw-stock balance
`stock[t] = stock[t−1] + buy − Σ_g x`; melt capacity and scrap-yard capacity
per plant and period; regional market volume per material and period (master
alloys unlimited). Sizing: plants are added until one period's variables times
a nominal horizon reach the target, then `T = round(target / per_period)`.

Feasibility: `feasible` plants a recipe per `(g,k,t)` producing exactly that
period's demand bought in the same period (`MultiPeriodBlendingPlan`,
`mp_plan_satisfies`); `infeasible` cuts one plant's melt capacity below the
cumulative charge its horizon demand needs at the best yield
(`MultiPeriodBlendingCertificate` — an aggregation over all its periods);
`unknown` draws melt capacity and market volumes around nominal need
(≈ 40% infeasible).

## `robust`

`standard` plus, for each order and uncertain tramp element (Si, Fe, Cu, Zn)
carried by its scrap lots, the robust counterpart of the maximum row:

```text
Σ_i comp[e,i] x[i,o] + Γ_o z[o,e] + Σ_{i∈scrap(o)} p[o,e,i] ≤ hi[e,o] · charge[o]
z[o,e] + p[o,e,i] ≥ deviation[i] · comp[e,i] · x[i,o]      (each scrap lot i)
```

with lot deviations 8–35% of the assay and a budget `Γ ∈ [0.5, 3]`
(`Γ_o = min(Γ, |scrap(o)|)`). The planted witness carries the exact optimal
`z`/`p` (`_robust_protection`: the top-Γ deviations with a fractional
remainder); windows are widened only where the recipe's worst case exceeds
them. Infeasible instances reuse `standard`'s nominal certificates, which
remain valid because the robust feasible set is contained in the nominal one.

## Solver note

HiGHS 1.13's dual simplex sometimes ends a requested-infeasible blending
instance with `OTHER_ERROR`/`UNKNOWN` (failed cleanup after cost perturbation
on these degenerate, many-row infeasibilities; about 5–10% at 2k–4k, after
switching silicon, manganese and chromium additions from 75–98% pure
briquettes to AlSi50/AlMn20/AlCr10 master alloys halved the rate). The
typed certificate is still a valid proof; setting
`dual_simplex_cost_perturbation_multiplier = 0` resolved most cases in a
20-seed probe. Note that the optional feasibility-contract backstop
(`generate_problem(...; optimizer=...)`) treats such a status as inconclusive
and raises, as it does for any non-certifying status.
