# Diet Problem

Least-cost diet planning on a role-correlated food-composition table with
Dietary Reference Intake (DRI) requirements. The three variants are
structurally different LPs built on one shared catalog (`common.jl`): a
multi-cohort population diet, a multi-week menu plan with a perishable
inventory, and a humanitarian ration-and-sourcing network. All are pure
continuous LPs; every generator draws from a constructor-local
`MersenneTwister(seed)` — it neither reseeds nor consumes Julia's global RNG —
and `build_model` does no sampling.

## Variants

| Variant | Key structure | Scale comes from | Domain grounding |
|---|---|---|---|
| `standard` (default) | per-cohort DRI rows, energy band, guideline share rows; shared supply rows | cohorts (`≈ target / 4√target`) | institutional / regional population diets |
| `food_groups` | daily food-group bands, weekly nutrient targets, variety caps, perishable inventory with weekly delivery and spot top-ups | days (up to a year) × foods | school / hospital / care-home menu planning |
| `food_aid` | NutVal ration rows per distribution site + source → hub → site procurement network | distribution sites | WFP Optimus-style food-basket design |

The former `nutrient_bounds` variant (an 85% verbatim copy of `standard` with a
same-row max < min contradiction) was removed: upper limits (sodium, saturated
fat, added sugar, total fat band) are now part of every variant.

## Shared catalog

- **Nutrients** (18): energy, protein, fat, saturated fat, carbohydrate,
  fiber, added sugar, sodium, calcium, iron, potassium, magnesium, zinc,
  vitamins A, C, D, B12 and folate. Energy is computed from the
  macronutrients with Atwater factors (4/9/4 kcal/g), saturated fat is a
  fraction of fat and added sugar a fraction of carbohydrate, so columns are
  correlated exactly as in real composition data.
- **Foods** belong to 13 categories (grains, vegetables, fruits, dairy,
  meat & poultry, fish & seafood, eggs, legumes, nuts & seeds, fats & oils,
  sweets & snacks, beverages, mixed dishes) with category median profiles and
  per-nutrient presence probabilities rounded from USDA FoodData Central; a
  per-food portion factor scales all nutrients and the price together.
  Vitamin D, B12, vitamin C are carried by few categories, which is what makes
  them bind.
- **Demographics** (11 DRI groups, children to lactating women) give the EER,
  protein, fiber, micronutrient minimums and the sodium limit.
- **Guideline limits**: energy within 90–110% of EER, saturated fat and added
  sugar ≤ 10% of energy, total fat 20–35% of energy — written as
  homogeneous rows with mixed signs, e.g. `Σ_f (9·satfat_f − 0.1·kcal_f) x_f ≤ 0`.

## `standard`: population diet

Decision `x[f, g] ∈ [0, upper[f, g]]`, servings per person per day of food `f`
for cohort `g` (portion limit × cohort appetite). Minimize
`Σ_g headcount[g] Σ_f cost[f] x[f, g]` subject to, per cohort, a ranged energy
row, minimum rows for protein, fiber and 6–10 tracked micronutrients, a sodium
ceiling and the share rows; and, for foods with limited regional supply,
`Σ_g headcount[g] x[f, g] ≤ supply[f]`.

Sizing: `n_foods * n_cohorts` variables, foods `≈ 4√target` (8–200),
`n_cohorts = round(target / n_foods_nominal)`, so the count is within
`n_cohorts / 2` of the target. Rows `= n_cohorts · (3 + |min_nutrients| +
sugar + 2·fat_band) + |limited foods|` (≈ 8% of columns); ≈ 14 nonzeros per
column.

Feasibility:

- `feasible`: each cohort gets a guideline-pattern diet (1–3 foods per
  category, ±25%); requirements are the DRIs lowered only where that diet falls
  short, limits are raised only where it exceeds them, and supplies of the foods
  it uses are 1.02–1.30× its consumption. The servings matrix is stored as
  `feasible_witness` (`diet_plan_satisfies` checks it row by row).
- `infeasible` (`DietInfeasibilityCertificate`): **supply shortage** (60%) —
  the carriers of one scarce micronutrient are cut so that, even used up to
  `min(supply, Σ headcount·portion limit)`, they deliver ≤ 1/1.08 of the summed
  cohort requirement; or **energy squeeze** — one cohort's minimum for a
  nutrient exceeds the exact fractional-knapsack maximum under its energy
  ceiling and portion limits. Both aggregate several rows; presolve does not
  see them. `diet_certificate_holds` recomputes them.
- `unknown`: DRI requirements and supplies drawn around a nominal consumption
  with an instance-wide market tightness and per-category supply shocks
  (≈ 10–25% of instances are infeasible).

## `food_groups`: menu planning with inventory

Per food `f`, day `d`, week `w` (deliveries at the start of each week):
servings `s[f,d] ∈ [0, upper[f]]`, weekly purchases `b[f,w] ≥ 0`, spot top-ups
`spot[f,d] ≥ 0` on non-delivery days at a 30–80% retail markup, and end-of-day
stock `I[f,d] ∈ [0, shelf_limit[f]]`, all per person. Minimize
`headcount · (purchases at seasonal prices + spot purchases + holding)`.

Rows: daily energy band, protein minimum, sodium ceiling, saturated-fat share
and a servings band per present food group (grains, vegetables, fruits, dairy,
protein foods); weekly fiber/micronutrient minimums on the week's intake and a
variety cap per food and week; the inventory balance
`I[f,d] = (1 − decay[f]) I[f,d−1] + delivery/spot − s[f,d]` (perishables lose
1–15% a day); storage capacity per class (dry, refrigerated, frozen) and day.

Spot purchases and per-item shelf limits are what keep the inventory columns
from being aggregated away by presolve (without them the balance chain is
eliminated and only ~35% of rows survive).

Sizing: `3 · n_foods · n_days` variables; foods `≈ 3√target` (13–250),
`n_days = round(target / 3 n_foods)` (1–364).

Feasibility: `feasible` plants a day-by-day pattern menu bought weekly to cover
consumption after spoilage (`MenuPlan`, checked by `menu_plan_satisfies`);
`infeasible` either tightens one food group's variety caps below a week's group
minimum (60%) or raises one day's protein minimum above the energy-capped
knapsack maximum (`MenuInfeasibilityCertificate`); `unknown` uses DRI weekly
targets, group-calibrated variety caps and nominal storage.

## `food_aid`: ration design with sourcing

Distribution sites belong to a programme (general distribution, school meals,
child supplementary feeding, pregnant/lactating women) with NutVal targets,
allowed commodities (20-commodity catalog: cereals, pulses, oil,
SuperCereal/SuperCereal Plus, LNS, sugar, milk powder, canned fish, dates,
biscuits) and ration limits. Variables: rations `r[c, j]` (g/person/day),
delivery arcs `y[c, h, j]` for sites served by two hubs, procurement arcs
`q[c, s, h]` from international, regional and local sources.

Rows: per site an energy band and minimums for protein, fat, calcium, iron,
zinc, vitamins A and C plus a minimum fat share of energy; per two-hub ration
pair `Σ_h y = 30·10⁻⁶·beneficiaries·r`; per (commodity, hub) the balance
`Σ_s q ≥ Σ y + direct single-hub demand`; per (commodity, source) capacity; per
hub throughput. A single-hub site's delivery is fixed by its ration, so it has
no delivery variable (otherwise presolve aggregates the doubleton away).

Sizing: hubs `≈ √target / 6`, sources and commodities by scale, then sites are
added until the variable count reaches the target (within one site).

Feasibility: `feasible` plants each programme's reference basket and splits it
over hubs and sources (`AidPlan`, `aid_plan_satisfies`); `infeasible` cuts a
hub's throughput below what its single-hub sites need to reach their energy
minimum at the most energy-dense ration (65%, when such sites exist) or makes
the whole pipeline short of a micronutrient (`AidInfeasibilityCertificate`);
`unknown` keeps NutVal targets (micronutrients at 60–95%) and draws market and
logistics capacity around last cycle's flows (two-sided, ≈ 30% infeasible).

## Presolve and difficulty (HiGHS, seed 0)

| Variant | 10k: rows, presolve kept (cols/rows), iterations | 100k: rows, kept, iterations |
|---|---|---|
| `standard` | 924, 1.00/1.00, ~1k | 7.6k, 1.00/1.00, ~10k |
| `food_groups` | 4.0k, 0.97/0.94, ~6k | 40k, 1.00/0.99, ~120k |
| `food_aid` | 5.9k, 1.00/0.96, ~9k | 54k, 1.00/0.96, ~105k |
