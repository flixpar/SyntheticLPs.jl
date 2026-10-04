# Land Use

Spatial zoning plan for a growing city: every parcel of a planar parcel graph
takes exactly one land-use zone, at maximum net development value, subject to
district infrastructure capacities, district housing and employment targets,
green-space accessibility around homes, and residential–industrial buffer
rules. The variables are binary; under the default `relax_integer=true` the
model is a fractional land-allocation LP whose district, accessibility and
buffer rows all stay meaningful. Generation uses a constructor-local
`MersenneTwister(seed)`; `build_model` is deterministic.

## Data

- **Parcels** sit on a jittered grid (ids shuffled over cells) with a
  four-neighbour edge list (`adjacency_edges`, sorted, `i < j`; no dense
  adjacency matrix is stored). Sizes are lognormal around 5 ha.
- **Zones** (3–12 from the catalog: residential, commercial, industrial,
  agricultural, conservation, mixed use, recreational, institutional,
  transportation, special, utilities, open space) with cost/revenue profiles
  that depend on accessibility to the centre, and per-zone consumption of 3–8
  infrastructure resources (water, sewage, transport, power, internet, gas,
  environmental, emergency).
- **Service districts**: square grid blocks of ≈ 80 parcels.
- **Environmental exclusions** remove 1–3 zones from 20–50% of the parcels;
  excluded pairs have no variable at all (no singleton `x == 0` rows).

## Formulation

Variables `x[k] ∈ {0,1}` for allowed parcel–zone pairs `k = (i, z)`.
Maximize `Σ size_i (revenue[i,z] − cost[i,z]) x[k]`. Rows:

```text
Σ_z x[i,z] = 1                                                     assignment
Σ_{i∈d} size_i consumption[z,r] x[i,z] ≤ capacity[d,r]               district infrastructure
Σ_{i∈d} size_i housing_density_z x[i,z] ≥ housing_target[d]          residential 20, mixed use 15 dwellings/ha
Σ_{i∈d} size_i job_density_z x[i,z] ≥ jobs_target[d]                 commercial, industrial, mixed use, institutional
Σ_{j∈N[i]} size_j x[j,green] − ρ_i size_i x[i,residential] ≥ 0      green-space accessibility
x[i,residential] + x[j,industrial] ≤ 1                              buffer, both orientations per edge
```

Green zones are conservation, recreational and open space (accessibility
rows exist when the instance has at least five zones).

## Sizing

`n_parcels = round(target / E[allowed zones per parcel])`, so the number of
pairs is within a few percent of the target; build is linear (100k variables
in well under a second).

## Feasibility

- `feasible`: a greedy reference plan (best net value per parcel, never
  residential next to industrial, homes given a green neighbour where possible)
  is drawn before the environmental exclusions, which never remove its zones;
  capacities are 1.03–1.20× its use, housing/jobs targets 85–97% of what it
  provides, and accessibility ratios at most what it achieves. Stored as
  `feasible_witness` (zone per parcel); `land_use_plan_satisfies` checks it.
- `infeasible`: one district's capacity for one resource is cut to 75–93% of
  the least any allowed zoning of its parcels consumes
  (`LandUseInfeasibilityCertificate`, from the assignment equalities and the
  capacity row only, so it also proves the LP relaxation infeasible).
- `unknown`: nominal capacities and targets; two-sided.
