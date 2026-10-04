# Forest Planning

Forest harvest scheduling (timber supply) LPs, the classic source of very large
LPs in natural-resource management: the USDA Forest Service's FORPLAN and
Spectrum models, and the Woodstock/Remsoft models used across industry, are
all built on the two formulations of Johnson & Scheurman (1977). Both variants
share one landscape, yield and economics model (`common.jl`), build pure
continuous LPs (nothing to relax), and use a local RNG. A generator call never
reseeds or consumes Julia's global RNG, and `build_model` does no sampling.

## Variants

| Variant | Model class | Key structure | Typical shape |
|---|---|---|---|
| `model_i` (default) | continuous LP | one column per whole-horizon prescription of a stratum; GUB-like stratum rows plus dense accounting and coupling rows | wide: about 30 columns per row |
| `model_ii` | continuous LP | rotation-level area flows from strata through watershed regeneration nodes, a period-ordered DAG network with side constraints | about 15 columns per row; node rows outnumber stratum rows |

Both are of type `ForestPlanningProblem{F}`, aliased as `ForestModelIProblem` /
`ForestModelIIProblem`, and have the same fields.

## Landscape and data

A **region** is drawn first. It fixes the period length `L`, the forest types,
the plantation type used for conversions, and the price and cost ranges
(US dollars, real):

| Region | `L` | Forest types (weights) | Conversion |
|---|---:|---|---|
| `pacific_northwest` | 10 | Douglas-fir .55, western hemlock .30, red alder .15 | alder → Douglas-fir |
| `southeast` | 5 | loblolly pine .50, slash pine .25, upland hardwood .25 | hardwood → loblolly |
| `lake_states` | 10 | aspen .35, northern hardwood .30, red pine .20, spruce-fir .15 | aspen → red pine |
| `interior_west` | 10 | ponderosa pine .40, lodgepole pine .30, mixed conifer .30 | lodgepole → ponderosa |

The landscape is a sequence of **watersheds** (management zones), each with
5–14 **strata** (analysis areas) defined by a unique
`(forest type, site class, age class)` combination. A watershed draws its own
type mix (the regional weights with lognormal noise), a valley-versus-ridge
site tendency, a disturbance-history age shift, and a green-up limit of
25–40%. Stratum ages come from an instance-level mixture of three age-structure
profiles: mature legacy stands, a plantation-era bulge, and a balanced
structure. Areas are lognormal (median 45 ha, 3–600 ha). The first stratum of
every watershed is merchantable in period 1, and every stratum can be
clearcut at least once in the horizon, so no stratum carries a single,
presolve-fixed column.

**Yield.** Standing volume follows a Chapman–Richards curve per stand model
(forest type × site class × regime):

```text
V(a) = M * (1 - exp(-k * (a - lag)))^c        (m³/ha; 0 for a <= lag)
```

`M` and `k` take site multipliers (site 1/2/3: `M` ×1.25/1.0/0.75, `k`
×1.10/1.0/0.90). Planted improved stock gets a genetic gain of 10–25% on `M`.
Natural regeneration has a type-specific establishment lag. The sawlog share
rises logistically with age to a type-specific maximum (0.25 for aspen, 0.85
for Douglas-fir and ponderosa pine). The non-sawlog volume is pulpwood at 85%
utilization. Sawlogs go to the softwood or hardwood sawlog product by species,
so `products` is `[:softwood_sawlog, (:hardwood_sawlog,) :pulpwood]`.
A merchantability threshold of 5 m³/ha (`FOREST_MIN_MERCH`) applies. A
smaller sawlog assortment goes to pulpwood, a smaller pulpwood remainder is
left on site, and standing volume below the threshold does not count toward
ending inventory. This also keeps sub-m³/ha coefficients out of the matrix.

**Silviculture.** Plantation softwoods (Douglas-fir, western hemlock,
loblolly, slash, red pine, ponderosa pine, mixed conifer) have a
commercial-thinning window. A thinning removes 25–35% of the standing volume,
mostly pulpwood, sold at 75% of the stumpage price. The removal recovers at
2–4%/yr, and the final harvest gets a +0.10 sawlog-share bonus. Regeneration
after a clearcut can be natural (cheap, lagged), planted (costly, faster), or,
for convertible types, a conversion to the regional plantation type.

**Economics.** Stumpage prices are drawn per product from the regional ranges
and scaled by species premiums. Each clearcut has a fixed cost of 80–200
USD/ha and each thinning one of 60–120 USD/ha, plus the regeneration cost. The
discount rate is 3–6% real, and cash flows are discounted from the period
midpoint.

**Horizon** (`T` periods), by target size:

| `L` | target ≤ 300 | ≤ 3000 | larger |
|---:|---|---|---|
| 10 | 4–5 | 6–8 | 8–12 |
| 5 | 6–8 | 8–12 | 12–18 |

## Columns

**Model I.** Per stratum, in this order:

1. do nothing (grow to the end);
2. clearcut in any merchantable period `h1` and regenerate with option `r1`;
3. the same, followed by a second rotation: clearcut the regenerated stand in
   `h2 ∈ h1 + ρ0 + {0,1,2,3}` (`ρ0` = its minimum rotation in periods) and
   replant under the same regime;
4. a commercial thinning in the first period inside the thinning window,
   followed by nothing or by a clearcut `h1` with regeneration `r1`.

**Model II.** Per source, where a source is a stratum or a regeneration node
`(stand model, watershed, period i)`: grow to the end, or clearcut in a
merchantable period `j` and regenerate with option `r`. Both choices are also
available after a commercial thinning. Regenerated area flows into node
`(regenerated model, same watershed, j)` when that stand can still be
clearcut before `T` (`j + ρ0 <= T`). Otherwise the column ends there, and its
ending inventory counts the young stand. Area regenerated in the same
watershed, period and stand model merges into one node whatever stratum it
came from. This merging is what makes Model II compact at long horizons.

**Dominance pruning.** After a clearcut whose regenerated stand is not
harvested again (Model I's single-rotation and thinning prescriptions, and
Model II columns that end at the horizon), the regeneration options share
every harvest and green-up coefficient. They differ only in cost and in
ending inventory. An option is therefore dropped when a sibling costs no more
and ends with at most 20 m³/ha less standing volume
(`FOREST_REGEN_EI_TOL`), for example planting improved stock in the last
period. Such columns are economically pointless and nearly parallel.

Each column stores its harvest-volume coefficients sparsely
(`vol_ptr`/`vol_period`/`vol_product`/`vol_amount`), its discounted NPV
(`col_npv`), its ending standing volume (`col_ending_inventory`), and its
prescription description (`col_thin`, `col_cut1`, `col_regen1`, `col_cut2`,
`col_dest`).

**Units.** Areas are in ha. Every volume (coefficients, `harvest`, supply
bounds, inventory) is in **thousand m³**, and money is in **thousand USD**.
This keeps bounds and right-hand sides within a few orders of magnitude of the
unit area coefficients.

## Formulation

```text
max  Σ_j col_npv[j] * area[j]

Σ_{j: src(j)=s} area[j]                                   = stratum_area[s]   ∀ strata s
Σ_{j: src(j)=n} area[j] - Σ_{j: dst(j)=n} area[j]         = 0                 ∀ nodes n (Model II)
Σ_j v[j,t,k] * area[j] - harvest[t,k]                     = 0                 ∀ t, k
Σ_k harvest[t,k] - (1-δ) Σ_k harvest[t-1,k]               ≥ 0                 t = 2..T
Σ_k harvest[t,k] - (1+δ) Σ_k harvest[t-1,k]               ≤ 0                 t = 2..T
Σ_{j in z, clearcut in t-w+1..t} area[j]                  ≤ γ_z * zone_area[z] ∀ z, t (non-empty rows)
Σ_j col_ending_inventory[j] * area[j]                     ≥ min_ending_inventory
min_supply[k] ≤ harvest[t,k] ≤ max_supply[k],  area ≥ 0
```

The rows are, in order: area accounting, the Model II node balance, the
harvest-volume definitions, two-sided even flow on total volume (tolerance
`δ ∈ [0.05, 0.20]`), watershed green-up with a 20-year window
(`w = ceil(20/L)` periods), and the ending-inventory floor. Mill-supply
contracts (`min_supply`) and mill capacities (`max_supply`) are bounds on the
accounting variables.

## Feasibility control

The constructor first plants an **area-control schedule**. This heuristic
clearcuts the highest-volume eligible parcels first, uses fractional areas,
and keeps every watershed's window clearcut area at most 90% of its green-up
limit. A bisection finds the largest flat flow it can sustain in every period.
The planted schedule runs at 70–95% of that flow, cuts exactly the same total
volume every period, and regenerates harvested area into second rotations or
nodes that later periods may harvest again. The requirements are then derived
from this schedule with margins:

- `base_min_supply[k]` is 75–92% of the schedule's smallest per-period
  product-`k` harvest.
- `base_min_ending_inventory` is 85–95% of its ending inventory.
- `max_supply[k]` is 1.3–1.8× its largest product-`k` harvest, plus 0.3× the
  flat flow.

**Certificate.** For an inventory weight `μ ≥ 0`, the dynamic-programming
values `π[src] = max_j (CH[j] + μ EI[j] + π[dst(j)])` are computed over the
period-ordered sources (`CH` is the column's total harvest volume). They are
dual-feasible multipliers for the area and node rows. Summing the
harvest-definition rows, the `T*K` supply lower bounds, and `μ` times the
inventory row shows that every feasible point has
`Σ_s area[s] π[s] ≥ T Σ_k min_supply[k] + μ min_ending_inventory`. The
*Lagrangian scale* `θ* = min_μ B(μ) / (T Σ base_min_supply + μ base_EI)`
(with `B(μ) = Σ_s area[s] π[s]`) is therefore the smallest common scale-up of
the planted contracts that the certificate refutes. A golden-section search
over `μ` finds it. By construction `θ* ≥ 1`.

- `feasible`: the contracts are at the planted level (`supply_scale =
  inventory_scale = 1`). The typed `feasible_witness` holds the column areas,
  the per-product harvest matrix, the flat flow and the ending inventory. It
  satisfies every row: the flow is perfectly even, the green-up rows have a
  ≥ 10% margin, and the supply contracts have at least 8% margins.
- `infeasible`: the contracts are scaled by `θ*(1 + m)`, `m ∈ [0.04, 0.12]`.
  The typed `infeasibility_certificate` stores `μ`, `π`, the bound and the
  requirement, with `bound * (1 + m/2) <= required`. If joint scaling would
  push the inventory floor above 75% of the maximum attainable inventory (the
  never-harvest level), the floor is held there and the supply contracts
  absorb the gap. This keeps the floor satisfiable on its own, so presolve
  sees no single-row contradiction. It also keeps the floor clear of the
  degenerate sliver near the maximum, where the dual simplex was observed to
  stall on infeasibility proofs. The certificate uses
  only the area, node, harvest-definition and inventory rows and the supply
  bounds, so it is independent of even flow and green-up. Measured with HiGHS
  (joint scaling, 2,000 variables), the true feasibility boundary sits at
  0.75–0.9 θ*, so infeasible instances lie 15–45% beyond it.
- `unknown`: the contracts are scaled by `s ~ U(1, 1.08 θ*)` (the inventory
  floor capped as above), a continuum from
  the planted level to just past the certified level, which straddles the true
  boundary. The instance stores no witness and no certificate. About half of
  the instances are feasible at 300–2,000 variables.

No mode is detectable from a single row. Every infeasible instance needs the
aggregated, multi-row argument above, and HiGHS presolve leaves the instance
intact.

## Sizing

Variables are the area columns plus the `T*K` accounting variables. Strata
(and in Model II the nodes they reach) are streamed until the target is met.
The last source's option list is truncated: Model I lists single-rotation
options first, and Model II offers one clearcut per period first and reserves
one column for every node it opens. The count is therefore exact up to one
column, and every source keeps at least two columns. A floor of 30 area
columns keeps tiny instances schedulable, so targets below `30 + T*K` round up
to it. Rows grow linearly: there is one stratum row per ~70 Model I columns
and one node row per ~15–20 Model II columns.

`target_variables` is capped at `FOREST_PLANNING_MAX_VARIABLES = 1_000_000`
(`ArgumentError` above it). At the cap, a constructor takes about 2–4 s, and a
JuMP build about 3–4 s.

## Solver profile

Presolve barely touches either variant: at 1k–100k variables HiGHS keeps
97–100% of columns and 85–100% of rows. These are hard LPs. The dense
harvest-definition and inventory rows make the dual simplex take about 7–10×
as many iterations as there are rows. At 10k variables, a solve takes about
3–7k iterations (~1 s). At 50k, it takes 12–46k iterations (13–60 s).
Instances at 100k variables exceed a 60 s single-thread budget.

**Known limitation: dual-simplex infeasibility proofs.** An `infeasible`
instance is refuted by a *global* Farkas ray: one multiplier per stratum (and
node) row, aggregated over every period. HiGHS's dual simplex reaches it, but
on roughly 15–25% of `infeasible` instances at 10k–50k variables it fails to
verify the dual ray and returns `UNKNOWN` / `OTHER_ERROR` instead of
`INFEASIBLE`. This was measured over 32 direct solves and 8 MPS-roundtrip
audit solves, and the rate did not change
with the infeasibility depth, the even-flow tolerance, mill capacities, cost
perturbation, or scaling strategy. HiGHS's IPM proves the same instances
infeasible in seconds, and the stored certificate is exact (verified
arithmetically in the tests). Instances up to a few thousand variables solve
cleanly. The framework's verification backstop
(`generate_problem(...; optimizer=...)`) treats `OTHER_ERROR` as
inconclusive and raises, so verify large `infeasible` instances with an
optimizer configured for IPM (for example, HiGHS with `solver = "ipm"`).

## References

- Johnson, K. N., and H. L. Scheurman (1977). *Techniques for prescribing
  optimal timber harvest and investment under different objectives —
  discussion and synthesis.* Forest Science Monograph 18.
- Johnson, K. N., D. B. Jones, and B. M. Kent (1986). *FORPLAN Version 2:
  User's Guide.* USDA Forest Service.
- Davis, L. S., K. N. Johnson, P. S. Bettinger, and T. E. Howard (2001).
  *Forest Management: To Sustain Ecological, Economic, and Social Values*,
  4th ed. McGraw-Hill. (Model I / Model II, even flow, ending inventory.)
- Pienaar, L. V., and K. J. Turnbull (1973). The Chapman–Richards
  generalization of Von Bertalanffy's growth model for basal area growth and
  yield in even-aged stands. *Forest Science* 19(1).
