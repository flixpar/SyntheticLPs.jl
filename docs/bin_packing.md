# Bin Packing

The `bin_packing` category models one-dimensional packing with operational
handling categories. It provides two formulations:

- `standard` (default): identical bins, minimizing the number used.
- `heterogeneous`: a finite fleet of bin types with different capacities,
  fixed costs, availability, and category eligibility.

These models pack a scalar item size. They are distinct from
`container_loading`, which models multiple capacity dimensions or geometric
two-dimensional non-overlap.

## Common Sizing and Data

Both variants use assignment variables `x[i,b]`, bin-use variables `y[b]`, and
category-presence variables `z[c,b]`. Their emitted variable count is therefore

```math
n_{bins}(n_{items}+n_{categories}+1).
```

The constructor searches directly over integer dimensions that produce this
count, typically with two to six items per bin (the smallest model has three
items and two bins), two to eight nonempty categories, and at least two
candidate bins. It stores both `target_variables` and
`actual_variables`; status-specific generation never changes the dimensions.
For targets below the smallest 12-variable formulation, the generator returns
that smallest valid model.

Representative sizing results are:

| Target | Actual variables |
| ---: | ---: |
| 12 | 12 |
| 49 | 48 |
| 50 | 51 |
| 99-101 | 100 |
| 249 | 248 |
| 250 | 250 |
| 251 | 252 |
| 999 | 1000 |
| 1000 | 1000 |
| 1001 | 1001 |
| 5000 | 5000 |
| 10000 | 10000 |

Every category is represented by at least one item. The category catalog is:

1. Food Grade
2. Hazardous
3. Fragile
4. Odorous
5. Chilled
6. Frozen
7. High Value
8. Ambient

Each profile has its own mean size fraction and dispersion. Items are sampled
around those fractions, with common carton/pallet modules rounded near multiples
of five percent of nominal capacity. Generated conflict pairs are sampled from
interpretable restrictions such as food-grade versus hazardous or odorous,
hazardous versus fragile or temperature-controlled, and odorous versus chilled
or frozen.

All random calls use `MersenneTwister(seed)` local to the constructor. Neither
variant resets or consumes Julia's process-wide random stream.

## Standard Identical-Bin Formulation

### Data and variables

`BinPackingProblem` stores:

- `n_items`, `n_bins`, and `n_categories`;
- `item_sizes` and common `bin_capacity`;
- integer `item_categories`, readable `category_names`, and canonical
  `incompatible_pairs`;
- sizing, requested-status, and named `load_profile` metadata;
- either `feasible_witness`, `infeasibility_certificate`, or neither.

The binary variables are:

```math
x_{ib}=1 \quad\text{if item }i\text{ is assigned to bin }b,
```

```math
y_b=1 \quad\text{if bin }b\text{ is used},
```

```math
z_{cb}=1 \quad\text{if category }c\text{ is present in bin }b.
```

The objective is

```math
\min \sum_b y_b.
```

Each item is assigned exactly once:

```math
\sum_b x_{ib}=1 \qquad \forall i.
```

Capacity also activates a bin:

```math
\sum_i s_i x_{ib}\le C y_b \qquad \forall b.
```

Category presence uses a two-sided linear envelope:

```math
x_{ib}\le z_{c(i),b},
```

```math
z_{cb}\le \sum_{i:c(i)=c}x_{ib},
```

```math
z_{cb}\le y_b.
```

Thus presence is exact for integer assignments and meaningfully linked in both
directions in the LP relaxation; it is not claimed to be a unique fractional
value. The chain `x[i,b] <= z[c(i),b] <= y[b]` also makes a separate
`x[i,b] <= y[b]` row redundant in both the MILP and its LP relaxation. For every
incompatible pair `(c,d)`:

```math
z_{cb}+z_{db}\le1.
```

### Symmetry

Identical bin labels create many equivalent solutions. The model opens a prefix
of bins,

```math
y_b\ge y_{b+1},
```

and uses the standard triangular canonical labeling

```math
x_{ib}=0 \qquad b>i.
```

Any unlabeled packing can be relabeled by the smallest item index in each bin to
satisfy both rules. The stored feasible witness is canonicalized the same way.

### Status construction

For a requested-feasible instance, a conflict-aware first-fit-decreasing pass
plants a packing. Sizes are adjusted only when necessary to fit the fixed
target-sized bin budget, then tightened to avoid an excessively slack witness.
`feasible_witness[i]` records the bin for item `i`.

For a requested-infeasible instance, sizes retain their category-profile
heterogeneity but are scaled so that

```math
\sum_i s_i > n_{bins}C.
```

Summing all capacity rows and using `y_b <= 1` proves this contradictory even
in the LP relaxation. `BinPackingCapacityCertificate` records total item size,
aggregate capacity, and their positive difference.

Unknown mode intentionally mixes three named workload regimes rather than
inheriting an accidental scale-dependent feasibility bias:

- `light`: conflict-aware packing followed by an aggregate-load ceiling of 68%;
- `nominal`: calibrated conflict-aware packing at normal utilization;
- `surge`: a modest 3%-9% aggregate-capacity overload.

Every ten consecutive seeds contain three light, four nominal, and three surge
profiles, with a deterministic target-dependent offset. The problem remains
tagged `unknown` and stores neither a witness nor a certificate: the profile is
workload metadata, not a requested feasibility contract.

## Heterogeneous Fleet Formulation

`HeterogeneousBinPackingProblem` represents individual available fleet slots.
Each slot has a fixed type; the number of slots of each type equals its
`type_availability`. This enforces availability directly without a redundant
aggregate row.

Depending on fleet size, types include:

- Standard;
- Controlled Specialty or Refrigerated;
- Hazmat;
- High Cube.

The struct stores type names, capacities, fixed costs, availability counts,
the type of every candidate bin, a named `load_profile`, and a type-by-category
compatibility matrix. General, temperature-controlled, regulated, and oversized
equipment therefore have materially different feasible assignments and
economics.

The item-assignment, activation, presence, and conflict constraints have the
same meaning as in `standard`, with two additions. Ineligible assignments are
fixed to zero:

```math
x_{ib}=0 \quad\text{if type}(b)\text{ cannot handle category}(i),
```

and capacity depends on the slot type:

```math
\sum_i s_i x_{ib}\le C_{type(b)}y_b.
```

The objective minimizes fleet cost rather than bin count:

```math
\min \sum_b f_{type(b)}y_b.
```

Only slots of the same type are interchangeable, so prefix symmetry is applied
within each type rather than globally across unequal equipment.

Requested-feasible instances use compatibility-aware best-fit-decreasing,
prioritizing items with fewer eligible fleet slots before size, then remap used
slots to the within-type prefixes. The witness is checked for capacity,
eligibility, conflicts, availability, and symmetry.

Requested-infeasible instances satisfy

```math
\sum_i s_i > \sum_t availability_t C_t.
```

This is again valid for the LP relaxation because every candidate slot can
contribute at most its type capacity. Certificate validation recomputes this
capacity directly from `bin_types` and rejects metadata whose observed type
counts disagree with `type_availability`.

Unknown typed-fleet instances use the same deterministic light/nominal/surge
mix as the standard variant. Their calibration is compatibility-aware, and they
likewise store no feasibility artifact.

## Evidence Validation and API Behavior

The solver-free methods

```text
SyntheticLPs.validate_bin_packing_witness(problem)
SyntheticLPs.validate_bin_packing_certificate(problem)
```

support both variants. Constructors assert the applicable validator before
returning a requested-feasible or requested-infeasible instance. These checks
cover the full planted structure, not just a stored scalar flag.

For requested-feasible instances, `build_model` also attaches a complete JuMP
start for every `x`, `y`, and category-presence variable reconstructed from the
validated witness. Unknown and requested-infeasible instances do not receive
these starts.

Both native formulations are binary MILPs. The package API defaults to
`relax_integer=true`, producing their LP relaxations. Set
`relax_integer=false` to solve the original packing MILP. Feasible witnesses are
integer-valid, and aggregate infeasibility certificates remain valid under
either setting.
