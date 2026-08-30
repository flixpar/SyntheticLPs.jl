# Feed Blending

`feed_blending/standard` generates a continuous least-cost feed formulation with
a fixed batch mass, role-correlated ingredient data, nutrient floors and caps,
ingredient availability, and typed average-content constraints.

## Application and data model

The decision is how much of each ingredient to include in one production batch.
The generated ingredient catalog is synthetic, but its four roles create useful
economic and nutritional correlations:

- `feed_energy_source`: inexpensive commodity ingredients, moderate major-nutrient
  concentration, and comparatively broad availability;
- `feed_protein_source`: higher major-nutrient concentration and moderately higher
  cost;
- `feed_mineral_supplement`: concentrated mineral and trace content, higher unit
  cost, and tighter inclusion availability;
- `feed_specialty_additive`: the highest and most dispersed unit cost, frequent
  trace content, and tight availability.

Every instance with at least four ingredients contains all four roles; larger
instances draw additional roles randomly. Unit costs are positive lognormal draws
whose medians increase from commodity energy sources through specialty additives.
Availability limits are more frequent and tighter for supplements and additives.

Nutrient rows likewise have typed semantics:

- `feed_major_nutrient`: dense concentrations, with protein sources typically
  richer than energy sources;
- `feed_mineral`: concentrated in mineral supplements;
- `feed_trace_nutrient`: sparse in commodity ingredients and concentrated in
  supplements/additives;
- `feed_restricted_compound`: usually upper-limited and more prevalent in specialty
  additives.

`nutrient_content[j, i]` is the concentration of nutrient or quality metric `j`
in ingredient `i`. Different nutrient rows may use different domain units; totals
and limits for a row always use that row's unit consistently. Empty nutrient rows
and ingredient columns are repaired with role-aware positive content.

## Sizing and reproducibility

The decision-variable count is exact except for the smallest requests:

```text
num_ingredients = max(3, target_variables)
```

The remaining dimensions scale as follows:

| Requested variables | Nutrients | Batch-size distribution |
| ---: | ---: | --- |
| `<= 250` | `4:8` | `Normal(500, 200)`, truncated to `[100, 2,000]` |
| `251:1,000` | `6:12` | `Normal(2,000, 800)`, truncated to `[500, 10,000]` |
| `> 1,000` | `8:20` | `Normal(10,000, 5,000)`, truncated to `[2,000, 50,000]` |

All random operations receive a constructor-local `MersenneTwister`. Generating a
feed blend therefore does not seed or advance Julia's global RNG.

The stored fields are:

- dimensions: `num_ingredients`, `num_nutrients`, and `batch_size`;
- ingredient data: `ingredient_types`, `costs`, and `availabilities`;
- nutrient data: `nutrient_content`, `nutrient_types`, `min_requirements`, and
  `max_limits`;
- `ratio_constraints::Vector{FeedRatioConstraint}`;
- status metadata: `feasible_witness`, `infeasibility_certificate`, and
  `requested_status`.

## Formulation

For ingredients `i in I`, let `x_i >= 0` be the ingredient mass, `c_i` its unit
cost, `A_i` a finite availability when one applies, and `B` the batch mass.

```math
\min \sum_{i \in I} c_i x_i
```

```math
\sum_{i \in I} x_i = B
```

For nutrient `j`, coefficient `a_{ji}`, total minimum `L_j`, and total maximum
`U_j`, active nutrient rows are:

```math
\sum_{i \in I} a_{ji}x_i \ge L_j,
\qquad
\sum_{i \in I} a_{ji}x_i \le U_j.
```

Finite ingredient availability is enforced by:

```math
x_i \le A_i.
```

Each `FeedRatioConstraint` stores a nutrient index, concentration target `p`, and
a `FeedRatioSense`. Because total mass is fixed, an average-content minimum is the
linear row

```math
\sum_{i \in I}(a_{ji} - p)x_i \ge 0,
```

while an average-content maximum is

```math
\sum_{i \in I}(a_{ji} - p)x_i \le 0.
```

The model branches explicitly on `feed_ratio_minimum` versus
`feed_ratio_maximum`. Constraint direction is not inferred from a diagnostic
string. In particular, a certificate described as a “maximum below achievable
minimum” remains a maximum row.

## Feasible requests and their witness

For `feasible` and `infeasible` requests, generation first constructs a complete
baseline recipe. A Dirichlet composition is clipped to finite availability and
then filled in a cost-biased randomized order. If sampled availability cannot fill
the batch, one commodity ingredient is made sufficiently available before the
recipe is built.

Nutrient and ratio bounds are placed around this recipe with positive randomized
slack. A requested-feasible instance stores the recipe in `feasible_witness`.
`feed_recipe_satisfies(problem)` checks, without a solver:

- nonnegativity and the batch equality;
- every finite ingredient availability;
- every active nutrient floor and cap;
- every typed ratio minimum and maximum.

Requested-feasible instances have no infeasibility certificate.

## Infeasible requests and certificates

A requested-infeasible instance begins from the same feasible baseline and then
applies exactly one certified contradiction. Its
`FeedInfeasibilityCertificate` records the certificate kind, relevant nutrient and
ratio-row index, the exact achievable bound, and the conflicting required bound.

The four mechanisms are:

1. `feed_minimum_ratio_above_achievable_maximum`: a minimum average target is
   strictly above the availability-aware maximum concentration.
2. `feed_maximum_ratio_below_achievable_minimum`: a maximum average target is
   strictly below the availability-aware minimum concentration.
3. `feed_minimum_nutrient_above_achievable_maximum`: a total nutrient minimum is
   strictly above the maximum possible total contribution.
4. `feed_insufficient_ingredient_capacity`: the sum of every ingredient's usable
   availability is strictly below the fixed batch mass.

The availability-aware nutrient extrema are exact for this model: sorting one
nutrient row and greedily filling ingredient capacities solves the corresponding
one-row continuous knapsack. `feed_infeasibility_certificate_holds(problem)`
recomputes the bound from stored model data, verifies that metadata matches it,
checks that the referenced row has the correct typed sense, and confirms a strict
contradiction. It does not call a solver.

Requested-infeasible instances have no feasible witness.

## Unknown requests

For `unknown`, each active nutrient or ratio target is drawn inside its own
availability-aware attainable interval, but no common recipe is planted. Multiple
individually attainable rows can still conflict, so joint feasibility is genuinely
unspecified. Unknown instances store neither a witness nor a certificate.

## Model characteristics

The generator produces a continuous LP. Ingredient amounts are divisible; it does
not model package sizes, discrete mixer batches, or integer purchase lots. The
batch equality and availability rows are sparse, while nutrient and ratio rows can
be dense. Trace-nutrient sparsity and role-correlated concentrations create a mix
of row densities and coefficient scales suitable for LP benchmarking.
