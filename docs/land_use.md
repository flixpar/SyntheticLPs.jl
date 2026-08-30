# Land Use

`LandUseProblem` generates spatial parcel-zoning assignment models with
infrastructure capacities, parcel-specific environmental exclusions, minimum
zoning counts, and residential-industrial adjacency rules. The native model is
a binary MILP; the package API relaxes those binaries by default unless
`relax_integer=false` is requested.

## Planning Setting

Each parcel must receive exactly one zoning type. A zoning decision produces a
parcel-size-weighted economic benefit, consumes infrastructure resources, and
may be unavailable because of an environmental restriction. Residential and
industrial uses cannot occupy adjacent parcels when adjacency rules are active.

Unlike a generic random graph, parcel adjacency comes from geography. Parcels
are placed on a jittered two-dimensional grid, parcel identifiers are shuffled
over the cells, and horizontal and vertical neighboring cells form a connected
spatial graph. The generator stores both the symmetric adjacency matrix and a
canonical list of undirected edges `(i, j)` with `i < j`.

## Sizing and Catalogs

`target_variables` approximates `n_parcels * n_zoning_types`:

```text
n_parcels = max(2, round(Int, target_variables / n_zoning_types))
```

| Target regime | Zoning types | Resources | Development-cost scale | Revenue scale |
| --- | ---: | ---: | ---: | ---: |
| `target_variables <= 250` | 3-5 | 3-5 | 50,000-150,000 | 20,000-80,000 |
| `250 < target_variables <= 1000` | 4-8 | 4-6 | 75,000-250,000 | 40,000-120,000 |
| `target_variables > 1000` | 5-12 | 5-8 | 100,000-500,000 | 60,000-200,000 |

The complete zoning catalog, in order, is:

1. Residential
2. Commercial
3. Industrial
4. Agricultural
5. Conservation
6. Mixed Use
7. Recreational
8. Institutional
9. Transportation
10. Special
11. Utilities
12. Open Space

Every catalog entry defines its name, cost multiplier, revenue multiplier, and
an eight-resource consumption profile. The resource catalog is Water, Sewage,
Transportation, Power, Internet, Gas, Environmental, and Emergency. Because the
catalogs are the source of both sampling limits and generated metadata, every
sampled large instance has complete names and parameter profiles.

## Generated Data

- Parcel sizes follow `LogNormal(log(5), 0.75)` and are floored at 0.1 acres.
- Coordinates lie inside the unit square on a jittered grid. Grid-cell
  assignment to parcel identifiers is shuffled.
- Development cost and revenue combine zoning-specific catalog multipliers,
  distance to the urban center, and log-normal parcel noise. Urban uses favor
  accessible central parcels; agricultural, conservation, recreational, and
  open-space uses favor more rural parcels.
- Zoning-resource rates apply log-normal noise to the catalog profiles while
  remaining strictly positive.
- Adjacency constraints are active with probability 0.8. The spatial graph is
  still retained when the rule is inactive.
- Minimum zoning counts are active with probability 0.9. The first up to three
  types—Residential, Commercial, and Industrial—each normally require about 10
  percent of parcels, with small-instance counts reduced to fit.
- Environmental restrictions affect a random subset of parcels and forbid one
  to three zoning types. They never forbid the generator's planted reference
  assignment, so restrictions cannot accidentally invalidate a requested
  feasible witness.

The constructor uses a local `MersenneTwister(seed)`. It neither resets nor
consumes Julia's process-wide random stream. Identical arguments produce
field-identical problem data for the same package version.

## Stored Evidence and Spatial Fields

In addition to the economic, resource, and restriction data, the problem stores:

- `parcel_coordinates::Matrix{Float64}`: `n_parcels x 2` coordinates.
- `adjacency_matrix::Matrix{Bool}`: symmetric parcel adjacency.
- `adjacency_edges::Vector{Tuple{Int,Int}}`: unique undirected edges with
  `i < j`.
- `feasible_witness::Union{Nothing,Vector{Int}}`: zoning index selected for each
  parcel in a requested-feasible instance.
- `infeasibility_certificate::Union{Nothing,LandUseInfeasibilityCertificate}`:
  a resource lower-bound certificate for a requested-infeasible instance.

Unknown instances expose neither status claim.

## MILP Formulation

Let `P` be parcels, `Z` zoning types, `R` resources, and `E` undirected spatial
edges. The binary variable

```math
x_{iz} = 1
```

means that parcel `i` receives zoning type `z`.

The objective maximizes parcel-size-weighted net benefit:

```math
\max \sum_{i \in P}\sum_{z \in Z}
s_i(\mathit{revenue}_{iz}-\mathit{cost}_{iz})x_{iz}.
```

Each parcel receives one zoning:

```math
\sum_{z \in Z} x_{iz}=1 \qquad \forall i \in P.
```

Resource use is limited by capacity:

```math
\sum_{i \in P}\sum_{z \in Z}s_i r_{zk}x_{iz}\le C_k
\qquad \forall k \in R.
```

Environmentally forbidden assignments are fixed to zero:

```math
x_{iz}=0 \qquad \forall (i,z)\text{ restricted}.
```

When minimum zoning requirements are active:

```math
\sum_{i \in P}x_{iz}\ge m_z.
```

For every edge `{i,j}` when adjacency rules are active:

```math
x_{i,\mathrm{Residential}}+x_{j,\mathrm{Industrial}}\le 1,
```

```math
x_{i,\mathrm{Industrial}}+x_{j,\mathrm{Residential}}\le 1.
```

Each undirected edge is visited once, so these are the two distinct
orientations rather than duplicated rows from a symmetric matrix traversal.

## Feasibility Controls

### Requested feasible

The generator plants an integer reference assignment before environmental
restrictions are sampled. Required residential and industrial parcels are
chosen from one parity class of the bipartite grid graph, so they cannot be
adjacent. Required commercial parcels and all remaining parcels are then
assigned while preserving the residential-industrial rule. Environmental
restrictions exclude the chosen zoning from their candidates.

Resource capacities are finally set to at least the witness consumption with
5-20 percent slack. The resulting `feasible_witness` directly satisfies parcel
assignment, environmental, minimum-count, adjacency, and capacity constraints.

### Requested infeasible

One resource `k*` receives a solver-independent lower-bound certificate. For
each parcel, let `A_i` be its environmentally allowed zoning types and define

```math
b_i=s_i\min_{z\in A_i}r_{z,k^*}.
```

Even in the LP relaxation, the exactly-one assignment and environmental rows
imply resource use of at least

```math
B=\sum_i b_i.
```

The generator sets `C[k*]` to 72-92 percent of `B`, hence `C[k*] < B`. The
certificate stores `k*`, every `b_i`, `B`, and the capacity. All other resource
capacities admit the planted reference assignment, making the recorded
contradiction easy to inspect.

### Requested unknown

Unknown mode returns nominal capacities derived from total parcel area and
average zoning consumption with additional scenario noise. It makes no status
claim and stores neither a witness nor a certificate.

## Model Size and Use

- Variables: `n_parcels * n_zoning_types`, binary before optional relaxation.
- Assignment rows: `n_parcels`.
- Resource rows: `n_resources`.
- Environmental rows: one per restricted parcel-zoning pair.
- Minimum-count rows: up to three.
- Adjacency rows: exactly `2 * length(adjacency_edges)` when enabled.

The package-level default `relax_integer=true` converts the binaries to
continuous variables in `[0,1]`. Use `relax_integer=false` for the native zoning
MILP. The infeasibility certificate is valid for both formulations, while the
feasible witness is an integer solution to both.
