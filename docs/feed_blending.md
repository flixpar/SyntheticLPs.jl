# Feed Blending

Least-cost feed formulation for a network of feed mills. Every mill produces a
book of formulas — species and growth phase (broiler starter/grower/finisher,
layer, swine starter/grower/finisher, dairy concentrate, beef finisher,
aquaculture grower) — in fixed batch tonnages from the ingredients it stocks,
and mills draw on shared supplier contracts. The model is a pure continuous
LP; generation uses a constructor-local `MersenneTwister(seed)` and
`build_model` is deterministic.

A commercial mill makes hundreds of formulas from a few dozen ingredients, so
large instances come from many formulas and mills, not from thousands of
ingredients in one recipe (the previous generator: one recipe with
`target` ingredients, ~10–20 nutrient rows and singleton availability rows,
which presolve reduced to 0% of its rows and solved in 6–14 iterations).

## Data

- **Ingredients** (32, NRC-style feed-table means, as fed): grains (corn,
  wheat, barley, sorghum), protein meals (soybean 48/44, canola, sunflower,
  peas, cottonseed, corn gluten), animal proteins (fish meal, meat & bone
  meal), by-products (DDGS, middlings, bran, rice bran, palm kernel), alfalfa,
  molasses, fats (soybean oil, tallow), minerals (limestone, di/monocalcium
  phosphate, salt, bicarbonate), synthetic amino acids (lysine, methionine,
  threonine), urea and premix.
- **Nutrients** (12): metabolizable energy, crude protein, fat, fiber,
  calcium, available phosphorus, sodium, digestible lysine, methionine,
  methionine+cysteine, threonine, NDF.
- **Lots**: each ingredient is offered by 1–3 suppliers with their own quality
  scatter (±4%) and price; each mill stocks the staples (corn, soybean meal 48,
  soybean oil, limestone, dicalcium phosphate, salt, premix) and most other
  ingredients from one supplier.
- **Species rules**: maximum inclusion fraction by species group and
  ingredient class with ingredient overrides (gossypol, glucosinolates,
  palatability); 0 excludes the pair — no animal protein for ruminants, urea
  only for ruminants, no synthetic amino acids for ruminants. Premix has a
  0.2% minimum inclusion.

## Formulation

Variables `x[k] ∈ [lower[k], upper[k]]`, tonnes of lot `i` in formula `f` for
every allowed pair (`upper = inclusion cap × batch`). Minimize ingredient
cost. Rows per formula with batch `D_f`:

```text
Σ_i x[i,f] = D_f                                          (batch)
Σ_i a[j,i] x[i,f] ≥ lo[j,f] · D_f                         (nutrient minimum, sparse)
Σ_i a[j,i] x[i,f] ≤ hi[j,f] · D_f                         (only if some lot exceeds hi)
Σ_i (Ca_i − r_hi·P_i) x[i,f] ≤ 0,  Σ_i (Ca_i − r_lo·P_i) x[i,f] ≥ 0   (Ca : avP band)
```

Coupling rows: mill stock `Σ_{f at mill} x[i,f] ≤ stock[i]` per lot, and
supplier contracts `Σ_{lots of s} Σ_f x ≤ contract[s]` shared by the mills
buying from `s` (a contract covering a single lot is folded into its stock).
Nutrient specifications use the sparse absolute form because the batch is
fixed — the deliberate contrast with `blending`, whose production is variable.

## Sizing

Mills `≈ target / 2500` (1–40); formulas are added round-robin over mills
until the number of allowed pairs reaches the target, and the last formula
drops optional (non-staple) lots so the count lands within a few pairs of it.
About 16 nonzeros per column; rows ≈ 85% of columns.

## Feasibility

- `feasible`: each formula gets its species' reference recipe (grain, protein,
  by-product, fat, mineral, amino-acid and premix shares) mapped onto its
  allowed lots and clipped to the inclusion caps; specifications are the
  reference ones, relaxed only where the recipe falls outside; stock and
  contracts are 1.02–1.30× its use. Stored as `feasible_witness`;
  `feed_formulation_satisfies` checks every bound and row.
- `infeasible` (`FeedInfeasibilityCertificate`):
  - `feed_nutrient_unreachable` (60%): a customer specification set 5–12% above
    the exact fractional-knapsack maximum of one nutrient over a batch within
    the inclusion caps, stock and contracts — chosen only where that maximum is
    limited by the batch equality, not by the carriers' caps (so the nutrient
    row alone is not contradicted by column bounds and presolve cannot see it);
  - `feed_mill_short`: one mill's stock cut below its whole batch book.
  `feed_certificate_holds` recomputes either from the data.
- `unknown`: the same reference specifications (every formula is makeable on
  its own); stock and contracts drawn around the reference use with an
  instance-wide tightness, so a mill may or may not make its whole book
  (about a third of the instances are infeasible, at every size).
