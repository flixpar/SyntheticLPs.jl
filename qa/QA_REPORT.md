# SyntheticLPs.jl — QA Report

**Date:** 2026-07-26
**Method:** Sample every registered variant across (target size, feasibility status, seed) matrices; build with `generate_problem`; solve with HiGHS simplex; record termination/primal/dual status, simplex iterations, var/con counts, objective, and coefficient-magnitude stats. ~1,600 instances total (default LP-relaxation path), plus dedicated feasibility-contract, edge-size, true-MIP, and `bounds_to_constraints` audits.

**Headline:** The corpus is broadly healthy — no build errors in the default size range, no unbounded instances, no `ALMOST_OPTIMAL`, fully deterministic, and the MIP variants are genuine (realistic integrality gaps). But there are **real deficiencies** in 4 areas, listed below by severity.

---

## P0 — Feasibility-contract violations (correctness bugs)

The `FeasibilityStatus` enum promises that requesting `feasible`/`infeasible` yields that status. **5 variants violate this.** The common root cause is *heuristic infeasibility injection with no final verification solve* (contrast with `diet_problem/standard`, which has an explicit "FINAL VERIFICATION" block and is the gold standard).

| Variant | Violation | Rate | File:line |
|---|---|---|---|
| `crop_planning/standard` | infeasible-request → OPTIMAL | ~17% (2/12) | `crop_planning/standard.jl:389–413`, model `:498` |
| `energy/standard` | infeasible-request → OPTIMAL | ~17% @ n=300 (size-dependent; 0/12 @ n=120) | `energy/standard.jl:293–354`, `:400–405` |
| `blending/standard` | infeasible-request → OPTIMAL | ~8–17% (size-dependent) | `blending/standard.jl:183–300` |
| `feed_blending/standard` | infeasible-request → OPTIMAL | ~8% | `feed_blending/standard.jl:408–474` |
| `unit_commitment/standard` | feasible-request → INFEASIBLE | ~8% (documented as "overwhelmingly likely") | `unit_commitment/standard.jl:301–325` |

### Root causes (confirmed by reproduction)

**`crop_planning/standard` — the "fallow-land" hole.** The infeasibility proof computes a lower bound on water/labor usage assuming **all** land must be planted:
```
remaining_land_required = max(0.0, total_land - sum(min_area_per_crop))   # standard.jl:389
min_water_bound += water_requirements[min_water_crop] * remaining_land_required
water_capacity = min_water_bound * violation_factor                         # 0.75–0.95
```
But the model's land constraint is `sum(x) <= total_land` — an **upper** bound (`standard.jl:498`). The farmer may leave land fallow, so the true minimum resource usage is just `sum(water_req .* min_area)` (often **0**, when no crop is mandatory). Reproduced instance (seed 2): `total_land=51`, `sum(min_area)=0`, `water_capacity=18866`, true min water usage `= 0` ⇒ `x = 0` is feasible ⇒ OPTIMAL.

**`energy/standard` — (a) dead constraints + (b) thin margins.**
- (a) The emissions row is `Σ em_s·x_s ≤ max_emission·Σ x_s` (`standard.jl:400–405`). Since `em_s ≤ max_emission` for every source by construction, the LHS (a weighted average times total) is **always ≤ RHS** — the constraint can never bind. Every `energy/standard` instance carries `n_periods` of these no-op rows (24 of 72 affine constraints in the reproduced instance), inflating the constraint count and weakening the infeasibility scenarios.
- (b) Scenarios 1 and 4 use thin capacity/demand margins that the per-period demand random multiplier `max(0.7, min(1.4, …))` can erase, so at larger sizes (more periods ⇒ wider demand spread) the "infeasible" instance becomes feasible. No final guard asserts infeasibility.

**`blending/standard` & `feed_blending/standard`** — all infeasibility scenarios are heuristic; the ratio-form quality constraints are only infeasible under specific (occasionally unmet) attribute-matrix conditions. No final verification solve.

**Suggested fix pattern (applies to all five):** after constructing an `infeasible` instance, solve it once and, if it returns OPTIMAL, either re-tighten the contradiction or fall back to a deterministic impossibility (à la `diet_problem`'s final-verification block). For `crop_planning`, additionally either make the land constraint `sum(x) == total_land` when infeasibility is requested, or bound resources against `sum(water .* min_area)` only. For `energy`, replace the tautological emissions row with a real emissions budget (e.g. `Σ em_s·x_s ≤ emission_budget`, a fixed scalar) or drop it.

---

## P1 — Build crashes at edge sizes

These don't fire in default dataset generation (`var_min=50`), but any user requesting the affected (size, variant) gets a hard crash.

| Variant | Trigger | Error | File:line |
|---|---|---|---|
| `portfolio/cvar` | `target_variables > ~1250` (n_assets > 250) | `ArgumentError: Uniform: the condition a < b is not satisfied` | `portfolio/cvar.jl:186` |
| `land_use/standard` | tiny targets (n_parcels == 2) | `ArgumentError: collection must be non-empty` | `land_use/standard.jl:212` |

- `cvar.jl:186`: `max_position = rand(Uniform(max(2.0/n_assets, 0.02), min(0.3, 5.0/n_assets)))`. For `n_assets > 250`, lower bound `0.02` exceeds upper bound `5/n_assets`. Verified: target 1300 (n=260) invalid, target 2000 (n=400) invalid.
- `land_use/standard.jl:212`: `n_neighbors = rand(2:min(4, n_parcels-1))`. When `n_parcels == 2`, the range is `2:1` (empty).

---

## P1 — Structurally trivial instances

Many variants produce instances that solve in ≤ 2 simplex iterations, making them weak LP-solver test material. (The existing `check_quality` filter rejects `≤ min_iterations`, but these are still emitted by default and pass without the filter.)

| Variant | Problem | Evidence |
|---|---|---|
| `knapsack/standard`, `knapsack/bounded` | Single-constraint LP (`nc = 1`); fractional knapsack solves greedily | sweep: `iters = 0–1`, `nc = 1` at n=50 and n=300 |
| `production_planning/standard` | `nc = 1` whenever `n_resources == 1` (`n_resources = rand(1:50)`) | `production_planning/standard.jl:42`; sweep `iters = 0`, `nc = 1` (seed 2) |
| `resource_allocation/standard` | `nc = 2` consistently | sweep `iters = 2` |
| `load_balancing/standard` | **No flow conservation** (acknowledged in its docstring); objective is just `Min u` (1 nonzero objective coeff); each link's flow is independent ⇒ collapses to `u* = max_link(required_flow/capacity)`, solved in presolve | `load_balancing/standard.jl:238–269`; sweep `iters = 0`, `obj_nnz = 1` |
| `cutting_stock/standard` | Severe undersizing + tiny instances | `nv = 6` for `target = 50`; `nc = 3`, `iters = 2` (see also P2) |

---

## P2 — Variable-count targeting drift

`|actual − target| / target > 0.5` for these variants (ratios are `actual/target`):

| Variant | ratio | note |
|---|---|---|
| `cutting_stock/standard` | 0.02–0.32 (mean 0.18) | `n_piece_types ≈ target/10` (capped); pattern count ≈ piece types ⇒ `nv ≪ target` (`standard.jl:50,58,66`) |
| `load_balancing/standard` | ~0.42 | post-`unique(links)` collapses the spanning-tree additions |
| `network_flow/standard` | ~0.44 | — |
| `scheduling/standard` | ~1.7 (overshoot) | — |
| `supply_chain/{standard,multi_product}` | 0.40–2.48 (wildly variable) | — |

---

## P2 — Numerical scaling (coefficient magnitude span > 1e8)

No `ALMOST_OPTIMAL` actually resulted (HiGHS handled them), but these mix coefficients across 8–9 orders of magnitude, which risks numerical warnings under stricter solver settings/tolerances.

| Variant | max/min coef ratio | cause |
|---|---|---|
| `blending/standard` | 2.9e9 | tiny `(attribute − target_pct)` differences from tight quality bands alongside costs ~1e2 and `min_blend_amount` up to 2e4 |
| `land_use/standard` | 4.6e8 | objective coeffs `parcel_size × (revenue − cost)` up to ~1e7 mixed with resource-consumption coeffs ~0.01 and near-zero `(revenue − cost)` cells |

---

## Strengths (affirmed)

- **No build errors** across 744 default-path instances; **no unbounded**; **no `ALMOST_OPTIMAL`/numerical errors**; **fully deterministic** (same seed ⇒ identical size) for every variant.
- **No all-zero objectives.**
- **MIP variants are genuine.** Solved as true MIPs (`relax_integer=false`) with realistic integrality gaps: `job_shop_scheduling` 94%, `vehicle_routing/cvrp` 27%, `bin_packing` 17%, `facility_location/two_echelon` 14%, `inventory/lot_sizing` 14%, `assignment/workload_balance` 14%, `cutting_stock/setup_cost` 1.8%, `knapsack/*` ~0% (tight).
- **`bounds_to_constraints` is correct** — feasibility/objective sense preserved across all spot checks; non-nonnegativity bounds become genuine rows (e.g. `knapsack/bounded` 1→121 rows, `unit_commitment` 252→336) while variable counts are unchanged.
- **Sound feasibility logic** in `transportation/*`, `diet_problem/standard` (final-verification gold standard), `portfolio/*`, `vehicle_routing/cvrp`, `nurse_scheduling/standard`, `revenue_management/standard`, `facility_location/p_median`, `multi_commodity_flow/standard`.

---

## Reproduction

All scripts are in `qa/`:
- `qa_sweep.jl` — broad sweep + anomaly summary (`qa_results.json`).
- `feas_contract.jl` — per-variant feasibility-contract violation rates (12 seeds).
- `drill.jl`, `drill2.jl` — root-cause evidence for crop_planning/energy/blending.
- `edge_mip.jl` — edge sizes, true-MIP solves, `bounds_to_constraints`.

Run any with `julia --project=scripts qa/<script>.jl`.
