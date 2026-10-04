# Crop Planning

Regional multi-year crop-rotation planning: farms in several irrigation
districts / market regions decide the crop mix of every field for a horizon of
years. Years are linked by rotation breaks and nitrogen carry-over; fields are
linked by seasonal family and hired labour, district water allocations, tiered
regional markets and processing / food-security contracts. A pure continuous
LP; generation uses a constructor-local `MersenneTwister(seed)` and
`build_model` is deterministic.

The previous generator (one area variable per crop option against three
knapsack rows plus singleton market and minimum-area rows) was reduced to an
empty model by HiGHS presolve at every size.

## Data

- **Crops** (18): winter wheat, maize, barley, sorghum, rice, oats, soybean,
  field pea, chickpea, lentil, canola, sunflower, cotton, sugar beet, potato,
  processing tomato, onion, alfalfa — with irrigated yield, rainfed factor,
  price, variable cost, nitrogen need and residual credit, irrigation water and
  seasonal labour (spring/summer/autumn/winter). Rice, cotton, potato, tomato
  and onion are irrigation-only. An instance uses `≈ 2 + target^0.25` of them
  (always one cereal and one legume).
- **Fields** (2–300 ha, lognormal) carry a soil class (loam, clay, sandy, silt)
  with family suitability factors, are irrigable with a regional probability,
  and allow ~85% of the suitable crops. Field yields = crop yield × soil ×
  (irrigated or rainfed) × lognormal field noise.

## Formulation

Variables (all ≥ 0): `a[j,c,t]` ha of crop `c` on field `j` in year `t`
(allowed pairs), `fert[j,t]` purchased N (kg), `hire[farm,m,t] ≤ hire_cap`
seasonal hired labour (h), `sell[c,r,t,k] ≤ tier_width` sales in price tier `k`
(1.0/0.85/0.65 of the price; one tier for small targets). Maximize revenue −
crop costs − fertilizer − hired labour. Rows:

```text
Σ_c a[j,c,t] ≤ A_j                                             land
Σ_{c∈F} (a[j,c,t] + a[j,c,t+1]) ≤ A_j                           rotation, F ∈ {legume, oilseed, root, vegetable}
fert[j,t] + Σ_c credit_c a[j,c,t−1] − Σ_c need_c a[j,c,t] ≥ −N0_j·[t=1]   nitrogen
Σ_{j∈farm} fert[j,t] ≤ quota[farm,t]                           fertilizer quota
Σ labour_cm a − hire[farm,m,t] ≤ family_labour[farm,m,t]        seasonal labour
Σ_m hire[farm,m,t] ≤ hire_budget[farm,t]                       seasonal-hire budget
Σ_{irrigable j∈r} water_c share_m a ≤ allocation[r,m,t]         district water (spring, summer)
Σ_k sell[c,r,t,k] ≤ Σ_{j∈r} yield_jc a[j,c,t]                   market
Σ_{j∈r} yield_jc a[j,c,t] ≥ contract[c,r,t]                     contracts
```

The fertilizer quota and the hire budget give the `fert` and `hire` columns a
second row, so presolve cannot eliminate the nitrogen and labour rows as
column singletons.

## Sizing

Years `≈ 1.5 + log10(target)` (2–8), regions `≈ target / 12000` (1–12). Farms of
3–15 fields are added until the variable count (area pairs + fertilizer +
hire + sales tiers) reaches the target; the last field's crop list is trimmed
so the count lands within about one year-block of it. Rows ≈ 40% of columns,
≈ 8 nonzeros per column.

## Feasibility

- `feasible`: a planted rotation (each field's crop sequence avoids repeating a
  break family, and break-family areas in consecutive years fit the field),
  exact fertilizer and hired labour; capacities, quotas and allocations
  1.05–1.30× its use and contracts 50–90% of its production. Stored as a
  `CropPlan`; `crop_plan_satisfies` checks every bound and row.
- `infeasible` (`CropInfeasibilityCertificate`): contracts in one region and
  year are raised until they need more land than the region has
  (`crop_land_shortage`, at the best field yield of each crop) or more water
  than the district allocation (`crop_water_shortage`, when ≥ 2
  irrigation-only crops compete). Each contract stays below 80% of what that
  crop alone could produce within the implied column bounds, so only the
  aggregate over several crops is contradictory — not visible to presolve
  (≥ 95% of instances in a 40-seed probe).
- `unknown`: labour, quotas, water and contracts drawn around a nominal plan;
  two-sided.
