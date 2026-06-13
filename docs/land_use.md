# Land Use

`LandUseProblem` generates a parcel zoning assignment model with resource capacities, environmental restrictions, minimum zoning counts, and residential-industrial adjacency rules.

## Overview

This generator represents land-use planning. A planner assigns each parcel to one zoning type, such as residential, commercial, industrial, agricultural, or conservation. Assignments create economic value from revenue minus development cost, consume infrastructure resources, and may be blocked by environmental restrictions or adjacency rules.

## Generator Data and Sizing

`target_variables` is interpreted as approximately `n_parcels * n_zoning_types`. The generator first selects zoning and resource counts by size regime, then sets:

```text
n_parcels = max(2, round(Int, target_variables / n_zoning_types))
```

Size regimes:

- `target_variables <= 250`: `n_zoning_types` from `3:5`, `n_resources` from `3:5`, development cost scale `50000:150000`, revenue scale `20000:80000`, infrastructure capacity factor `0.6-0.8`, environmental restriction probability `0.2-0.4`.
- `target_variables <= 1000`: `n_zoning_types` from `4:8`, `n_resources` from `4:6`, development cost scale `75000:250000`, revenue scale `40000:120000`, infrastructure capacity factor `0.65-0.85`, environmental restriction probability `0.25-0.45`.
- larger targets: `n_zoning_types` from `5:12`, `n_resources` from `5:8`, development cost scale `100000:500000`, revenue scale `60000:200000`, infrastructure capacity factor `0.7-0.9`, environmental restriction probability `0.3-0.5`.

Adjacency constraints are enabled with probability `0.8`; minimum zoning requirements with probability `0.9`.

Parcel sizes follow `LogNormal(log(5), 0.8)` and are floored at `0.1`. Development costs and revenues are generated per parcel-zoning pair using zoning-specific multipliers, parcel-specific gamma location factors, and normal noise. Resource consumption is generated per zoning-resource pair from hand-coded patterns for the first five zoning types and random `Uniform(0.5, 3.0)` bases for additional types, multiplied by `Gamma(2, 0.5)`.

Initial resource capacities are based on total parcel size times mean consumption, scaled by the infrastructure capacity factor and `Uniform(0.8, 1.2)` noise. Environmental restrictions randomly forbid some parcel-zoning pairs, but the constructor repairs any parcel with no allowed zoning type. If minimum zoning requirements are active, the first up to three zoning types receive minimum parcel counts, usually about 10 percent of parcels each, adjusted to fit the parcel count. Environmental restrictions are relaxed when needed so enough parcels allow required zoning types.

The struct stores:

- `n_parcels`, `n_zoning_types`, `n_resources`
- `parcel_sizes`
- `development_costs`, `revenues`
- `resource_consumption`
- `resource_capacities`
- `environmental_restrictions`
- `adjacency_matrix`
- `zoning_names`, `resource_names`
- `min_counts_by_type`
- `zoning_adjacency_constraints`
- `minimum_zoning_requirements`

The constructor calls `Random.seed!(seed)`, resetting Julia's global RNG.

## LP Formulation

The implemented model is a binary assignment MILP, not a pure LP.

Sets:

- `P = {1, ..., n_parcels}` parcels
- `Z = {1, ..., n_zoning_types}` zoning types
- `R = {1, ..., n_resources}` resources

Decision variable:

- `x_{ij} in {0,1}`: parcel `i` is assigned to zoning type `j`

Objective:

```math
\max \sum_{i \in P} \sum_{j \in Z} s_i (rev_{ij} - cost_{ij}) x_{ij}
```

Each parcel receives exactly one zoning type:

```math
\sum_{j \in Z} x_{ij} = 1 \quad \forall i \in P
```

Resource capacities:

```math
\sum_{i \in P} \sum_{j \in Z} s_i r_{jk} x_{ij} \le C_k \quad \forall k \in R
```

Environmental restrictions:

```math
x_{ij} = 0 \quad \text{for restricted parcel-zoning pairs}
```

Minimum zoning requirements, when enabled:

```math
\sum_{i \in P} x_{ij} \ge m_j
```

Adjacency constraints, when enabled and at least three zoning types exist, prohibit residential type 1 next to industrial type 3:

```math
x_{i,1} + x_{i',3} \le 1
```

```math
x_{i,3} + x_{i',1} \le 1
```

for adjacent parcels `i` and `i'`.

## Feasibility Controls

For `feasible`, the constructor builds a concrete witness assignment. It computes allowed zoning sets and neighbor lists, then tries to satisfy minimum counts for type 1, type 3, and type 2. Residential type 1 and industrial type 3 assignment is adjacency-aware. If it cannot find enough nonconflicting parcels, it prunes adjacency edges to make the witness possible. Commercial type 2 has additional fallbacks, including swaps and final environmental relaxation when needed.

Remaining parcels are assigned to the allowed zoning type with the best parcel-level net benefit, excluding residential-industrial adjacency conflicts when possible. The witness is then repaired: adjacency edges that conflict with the witness are pruned, minimum zoning count violations are addressed by swaps, and as a last resort minimum requirements are reduced to the achieved count. Finally, resource usage of the witness is computed and capacities are raised to at least usage times a slack factor from `1.05` to `1.25`.

For `infeasible`, the constructor computes a resource lower bound by assigning each parcel its minimum possible consumption for each resource over currently allowed zoning types. It then sets every resource capacity below that lower bound by `5-25` percent, floored at `1e-6`. Since every parcel must be assigned exactly one allowed zoning, this creates a resource-capacity contradiction.

For `unknown`, no feasibility enforcement branch is applied. The base random restrictions, capacities, adjacency matrix, and minimum counts are returned after only the general repairs that ensure each parcel has an allowed zoning and required types are not blocked solely by environmental restrictions.

## Model Characteristics

Variable count is `n_parcels * n_zoning_types`, all binary. Constraint count is driven by:

- `n_parcels` assignment equalities
- `n_resources` resource capacity inequalities
- one equality for each restricted parcel-zoning pair
- up to `length(min_counts_by_type)` minimum zoning rows
- two adjacency inequalities for each true ordered adjacency entry when adjacency constraints are active and `n_zoning_types >= 3`

The assignment and environmental rows are sparse. Resource rows are dense across all parcel-zoning variables. Adjacency rows are sparse two-variable rows. Because the adjacency matrix is symmetric and the model loops over all ordered pairs, each undirected adjacency can produce duplicated directional constraints.

## Practical Notes

This generator is useful for testing binary assignment, resource-capacity infeasibility, environmental exclusions, and graph-like adjacency restrictions. The feasible path may modify generated environmental restrictions, minimum counts, and adjacency edges to admit the witness. The built model uses binary variables, so solving it requires MILP support; the LP relaxation would allow fractional parcel-zoning assignments but is not what `build_model` creates.
