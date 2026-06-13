# Crop Planning

Generates agricultural land-allocation LPs that maximize crop profit subject to land, water, labor, market, minimum-area, and optional diversity constraints.

## Overview

This generator represents farm or regional crop planning. A planner allocates hectares across crop options with different yields, prices, production costs, water needs, labor needs, market limits, and crop-type diversity requirements. The model chooses continuous planted area for each crop.

## Generator Data and Sizing

`target_variables` is interpreted directly as the number of crop variables:

```text
n_crops = max(2, target_variables)
```

Scale-dependent ranges:

| Scale condition | Total land | Water availability factor | Labor availability factor | Market demand factor | Diversity probability |
| --- | ---: | ---: | ---: | ---: | ---: |
| `target_variables <= 250` | 50-500 ha | 0.6-0.8 | 0.7-0.9 | 1.0-1.3 | 0.5-0.8 |
| `target_variables <= 1000` | 500-5000 ha | 0.65-0.85 | 0.75-0.95 | 1.1-1.5 | 0.6-0.9 |
| otherwise | 5000-50000 ha | 0.7-0.9 | 0.8-1.0 | 1.2-2.0 | 0.7-0.95 |

The first 25 crops use fixed names and types:

- 1-5: cereals: Wheat, Corn, Rice, Barley, Oats.
- 6-10: vegetables: Tomatoes, Peppers, Lettuce, Carrots, Onions.
- 11-15: legumes: Soybeans, Peas, Lentils, Beans, Chickpeas.
- 16-20: industrial crops: Cotton, Sugarcane, Tobacco, Hemp, Flax.
- 21-25: oilseeds: Sunflower, Canola, Safflower, Sesame, Peanuts.

Additional crops are named `Crop_i` and assigned a random type from `:cereal`, `:vegetable`, `:legume`, `:industrial`, and `:oilseed`.

Random data generation:

- Yields are type-specific log-normal samples with clamps, e.g. cereals 3-10 tons/ha, vegetables 15-40, legumes 2-4, industrial 3-8, oilseeds 1.5-4.
- Prices are type-specific log-normal samples with clamps, e.g. cereals 150-350 dollars/ton, vegetables 400-900, legumes 300-600, industrial 500-1200, oilseeds 400-700.
- Production costs are type-specific normal samples with clamps, e.g. cereals 400-900 dollars/ha and vegetables 1200-2500.
- Water requirements are type-specific normal samples with clamps; Rice and Sugarcane receive special high-water ranges.
- Labor requirements are type-specific gamma samples with clamps.
- Net profit per hectare is `prices .* yields .- production_costs`.
- Market demand limits are sampled as `total_land * Uniform(0.1, 0.4) * market_demand_factor`.
- With probability `0.85`, minimum area requirements are created for 30%-50% of crops, preferring cereals and legumes. Minimums are 2%-8% of total land, capped by market demand, and scaled down if their sum exceeds total land.
- Optional diversity constraints require 5%-15% of total land in crop-type groups with at least two crops.

The stored struct fields are:

- `n_crops`
- `total_land`
- `crop_types`
- `crop_names`
- `yields`
- `prices`
- `production_costs`
- `water_requirements`
- `labor_requirements`
- `net_profit_per_ha`
- `market_demand_limits`
- `min_area_per_crop`
- `water_capacity`
- `labor_capacity`
- `diversity_constraints`

The constructor calls `Random.seed!(seed)`, so generation is reproducible for a fixed seed but resets Julia's global RNG state.

## LP Formulation

Sets and indices:

- `I = {1, ..., n_crops}`: crops.
- `D`: optional diversity constraints, each containing a crop type, minimum area, and crop index set.

Decision variables:

```text
x_i >= 0
```

`x_i` is hectares allocated to crop `i`.

Objective:

```text
maximize sum_{i in I} (price_i * yield_i - production_cost_i) x_i
```

Constraints:

Land:

```text
sum_{i in I} x_i <= total_land
```

Water:

```text
sum_{i in I} water_requirement_i x_i <= water_capacity
```

Labor:

```text
sum_{i in I} labor_requirement_i x_i <= labor_capacity
```

Market demand:

```text
x_i <= market_demand_limit_i    for each i in I
```

Minimum area, only where `min_area_per_crop[i] > 0`:

```text
x_i >= min_area_i
```

Diversity, for each generated tuple `(crop_type, min_type_area, crop_indices)`:

```text
sum_{i in crop_indices} x_i >= min_type_area
```

Bounds:

All variables are continuous and nonnegative.

## Feasibility Controls

- `feasible`: The generator constructs a baseline allocation by first satisfying minimum crop areas, then distributing remaining land by nonnegative profit weights subject to market limits. Water and labor capacities are set to baseline usage times a slack factor from `1.1` to `1.3`, and also at least `1.2` times the resource usage of minimum areas. Diversity constraints are added only when the baseline allocation satisfies them within a 5% tolerance.
- `infeasible`: The generator computes water and labor lower bounds from minimum crop areas plus an assumed allocation of all remaining land to the least-resource crop. It then randomly makes either water or labor capacity `75%` to `95%` of that lower bound, while giving the other resource `10%` to `40%` slack. Diversity constraints are not added in infeasible mode.
- `unknown`: Water and labor capacities are sampled from estimated average use times scale-specific availability factors and additional `Uniform(0.6, 1.4)` noise. Diversity constraints may be added without a feasibility check.

## Model Characteristics

- Variables: `n_crops`.
- Constraints: one land row, one water row, one labor row, `n_crops` market upper-bound rows, up to `n_crops` minimum-area rows, and optional diversity rows.
- Density: land, water, and labor rows touch all crop variables; market and minimum-area rows are singleton bounds represented as constraints; diversity rows touch crop subsets by type.
- Model class: continuous LP.

## Practical Notes

These instances are useful for testing dense resource rows mixed with many simple bound-like constraints. One implementation caveat is that the infeasible lower-bound calculation assumes all remaining land must be allocated, but the actual land constraint is `<= total_land`, not equality. If minimum-area requirements alone are small, reducing water or labor below the "use all land" lower bound may not always prove infeasibility of the implemented LP.
