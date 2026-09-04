# Process Planning

Multi-period production planning for petroleum refineries and chemical
process plants: crude assays, conversion units, blending, quality
specifications, campaigns, contracts, and inventory.

## Overview

The `process_planning` category covers the two classical planning workhorses
of the process industries. The `refinery` variant is the classical "refinery
LP" - the industrial LP application, in continuous industrial use since the
1950s: buy crude under term and spot contracts, distill it on the crude
distillation unit, upgrade intermediates on conversion units, blend finished
products from assay-origin streams under quality specifications, and sell
against seasonal demand while carrying tank inventories. The `campaign`
variant is the medium-term planning model of a petrochemical complex in
state-task-network form, where product grades run as campaigns on shared
trains with changeovers.

Both variants are multi-period LPs (the campaign model is a natural MIP whose
selectors relax to `[0,1]` under the default `relax_integer=true`), with
seasonal demand, storage that couples the periods, and a profit-maximisation
objective.

## Generator Data and Sizing

**`refinery`.** A configuration ladder determines the processing structure,
sampled with the target's scale:

| configuration | units | products |
|---|---|---|
| `:topping_reform` | CDU + reformer | PGAS, RGL, DIES, HFO (+PGL, LNAPH) |
| `:hydroskimming` | + naphtha/kero/diesel hydrotreaters, isomeriser | + JET |
| `:catalytic` | + vacuum distillation, FCC (± hydrocracker) | + ASPH |
| `:deep_conversion` | + hydrocracker, delayed coker | + COKE |

Crude assays are sampled from a lightness/sweetness slate: distillation-cut
yields interpolate between heavy and light assay tables (renormalised to one
barrel), sulphur inherits from the crude with cut-dependent concentration
factors, and the crude price carries lightness and sweetness premia.
Conversion units run in modes with fixed yield vectors - reformer severity
pairs with mode-specific reformate pools, FCC and hydrocracker in
max-gasoline/max-distillate operation - and swing cuts shift bounded volume
between adjacent CDU cuts. The exact variable count is

```text
n_periods * (3*n_crudes + n_crudes*n_swing + n_crudes*n_modes +
             n_crudes*n_blend_pairs + 2*n_products)
```

with the horizon solved in closed form and scanned (`3..126` periods); the
row count is `n_periods * (n_crudes + n_crudes*n_streams + n_units +
n_products + n_active_specs + 2)`. Volumes are in thousand barrels per
period (a 60-400 kbpd refinery), which keeps bounds near `1e2-1e4` for
solver scaling. Targets through
`MAX_REFINERY_PLANNING_VARIABLES = 200_000` are supported; larger targets
raise `ArgumentError` before any assay data is allocated. Measured over the
sweep in the category tests, realised sizes land within a few percent of the
request across four decades.

**`campaign`.** The complex is assembled from a chain library - vinyls
(EDC/VCM/PVC grades), aromatics (cumene/phenol/acetone with a co-product and
two-input bisphenol and PET steps), polyolefins (three trains sharing
monomer markets), polyester (PX-PTA-PET), C1 chemistry
(methanol/acetic acid/vinyl acetate/PVOH), and nitrogen fertilisers - with
1-6 chains sampled with the target and shared raw materials (ethylene,
propylene, natural gas) merging into common feed markets. The exact variable
count is

```text
n_periods * (n_tasks + 2*n_campaign_tasks + n_raws*n_tiers +
             n_materials + n_finals)
```

with raw purchasing in 1-3 price tiers (contract quota, spot, premium - a
convex piecewise-linear cost the LP exploits directly) and the row count
`n_periods * (n_materials + n_units + n_campaign_units +
4*n_campaign_tasks) - n_campaign_tasks*(campaign_length - 1)`. Targets
through `MAX_CAMPAIGN_PLANNING_VARIABLES = 20_000` are supported.

## Reference Plan and Specifications

Every instance is planted around a reference plan computed inside the
constructor from a constructor-local MersenneTwister. The refinery plan runs
the crude mix through the network at realistic unit utilisations (isomerise
the light naphtha, reform nearly all heavy naphtha, crack most of the vacuum
gas oil, cap coker naphtha at a tenth of the gasoline pool), splits each
stream's net availability across its eligible product pools with
affinity-biased shares, and sizes contracts, capacities, and tankage above
the plan's peaks. Product specifications are then selected as the tightest
industry band the planted recipe clears with margin in every period - octane
84-87/89-93 AKI, sulphur from 15 ppm to 3.5 wt %, cetane 40-51, fuel-oil
viscosity 180-700 cSt, and a seasonal RVP window - so specifications are
always realistic and always satisfiable by the refinery as planted. RVP
blends linearly in the Chevron index `RVP^1.25` and viscosity in the Walter
index `cSt^(1/3)`, exactly as in commercial refinery LPs. The campaign plan
schedules campaign blocks of at least the minimum campaign length per train,
propagates material flows in feed-topological order, splits tiered purchases
around consumption, and sizes unit capacities and tankage above the
resulting peaks.

## LP Formulation

**`refinery`** (pure LP, profit maximisation). Variables per (crude, period):
purchases, tank levels, CDU feed, swing displacements, per-mode unit feed by
crude origin, assay-origin blend allocations, and product sales and tank
levels. Rows: crude inventory balance and aggregate tankage, CDU capacity,
per-origin stream balances (production equals unit feed plus blending - no
free dumping), per-unit mode-aggregated capacity with turnaround windows,
product inventory balance, and per-period quality rows
`sum_s (q_s - rhs) * blend_s ⋚ 0` with index-transformed coefficients for
RVP and viscosity.

**`campaign`** (MIP, margin maximisation). Variables: task rates, binary
campaign selectors and start indicators, tiered purchases, material
inventories, and final sales. Rows: material balance per material and
period, unit capacity, train exclusivity, rate linking (`rate <= capacity *
active`), minimum turndown (`rate >= fraction * capacity * active`),
minimum campaign length
`sum_{k=t..t+L-1} active_k >= L * (active_t - active_{t-1})`, and start
definitions charged a changeover cost. With `relax_integer=true` (the
default) the selectors relax to `[0,1]` and the model is a pure LP.

## Feasibility Controls

- `feasible`: the reference plan itself is the stored witness
  (`RefineryPlanWitness` / `CampaignScheduleWitness`) - a complete primal
  point, re-verified by pure arithmetic and by `primal_feasibility_report`
  in the category tests, including the unrelaxed campaign MIP.
- `infeasible`: a cumulative supply-demand cut. The refinery certificate
  (`RefinerySupplyCertificate`) limits a product's cumulative term demand
  above `initial stock + yield_bound * min(CDU horizon capacity, initial
  crude stock + purchase ceilings)`, where `yield_bound` is the best-mode
  conversion fraction of crude into that product's blendstocks plus the
  swing-cut allowance; the campaign
  certificate (`CampaignCapacityCertificate`) bounds cumulative output by
  the producing task's unit capacity (`rate <= capacity * active <=
  capacity`, valid for relaxed selectors) and its raw-tier supply. Both are
  aggregations of linear rows and variable bounds only - **the refutations
  survive `relax_integrality`** - and the certificates are recomputed from
  struct fields in the tests.
- `unknown`: a correlated market scenario (`RefineryMarketScenario` /
  `CampaignMarketScenario`) scales the supply side (purchase ceilings, unit
  capacities) by `supply_factor` and term sales floors by `demand_factor`,
  positioned along the band by the golden-ratio sequence of the seed so any
  block of seeds is a genuine feasibility mix rather than a binomial
  accident; measured mixes land near 50/50 for the refinery and 25-40%
  infeasible for the campaign.

## Model Characteristics

| variant | sense | class | typical size range |
|---|---|---|---|
| `refinery` | Max | pure LP | 60 - 200,000 variables |
| `campaign` | Max | MIP (LP-relaxed by default) | 48 - 20,000 variables |

Structural density comes from the assay-origin blend block
(`n_crudes * n_blend_pairs * n_periods` refinery variables), the swing-cut
freedom at CDU cut points, the mode splits on conversion units, and the
campaign scheduling block. Coefficient magnitudes stay near `1e-3..1e2`
(qualities are unitless numbers, volumes are in thousand barrels or
kilotonnes), so the models are well-conditioned for simplex and interior
point alike.

## Practical Notes

- The refinery is a pure LP, so `relax_integer` has no effect; the campaign
  model's binaries are the campaign selectors only.
- Infeasibility *proofs* on very large refinery instances (tens of thousands
  of variables of equality network) can exceed short solver time limits even
  though the certificate refutes the model instantly by arithmetic; the
  category tests check solver contracts at moderate sizes.
- The witness and the certificates are exact arithmetic statements about the
  stored fields, not solver calls;
  `test/problem_types/process_planning.jl` recomputes both from the struct
  and additionally hands the witness to JuMP through
  `primal_feasibility_report`.
