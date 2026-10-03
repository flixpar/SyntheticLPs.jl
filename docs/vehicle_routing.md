# Vehicle Routing

The `vehicle_routing` category generates capacitated vehicle routing problems
(CVRP): a homogeneous fleet based at one depot serves customers with known
demands, every route starting and ending at the depot and carrying at most the
vehicle capacity.

## Variants

| Variant | Application | Natural formulation |
| --- | --- | --- |
| `cvrp` (default) | Daily delivery from a single depot | Two-index arcs with strengthened single-commodity (Gavish–Graves) load flow |

## Data

- Depot near the centre of a scale-tiered square region; customers in 2–8
  Gaussian neighbourhoods.
- Demands log-normal between scale-dependent bounds (few large shipments, many
  small).
- Vehicle capacity `Q` ≈ 3–6 average customers per route, at least 1.1× the
  largest demand; fleet `K` sized from total demand with 15–30% slack, `K ≤ N`.
- Arc costs = Euclidean distance × per-km rate × a small asymmetric per-arc
  factor (0.95–1.05).

## Sizing

The model has one binary `x` and one load variable `f` per directed arc of the
complete graph over the depot and `N` customers:

```text
total = 2 · N · (N + 1),   N = max(3, round((sqrt(1 + 2·target) − 1) / 2))
```

so the count is within `2N` of the target (`target = 500` → 480 variables,
`target = 100_000` → 100,128). Rows:
`2N + 2` degree rows, `N + 1` load-balance rows, `N(N + 1)` upper and `N²`
lower coupling rows. Build time is a fraction of a second at 100k variables.

## Formulation

Node 1 is the depot, nodes `2..N+1` customers, `d_1 = 0`.

```text
min  Σ_{i≠j} c_ij x_ij
s.t. Σ_i x_ij = 1, Σ_k x_jk = 1                 ∀ customers j   (degree)
     Σ_j x_1j = K, Σ_i x_i1 = K                                (depot degree)
     Σ_i f_ij − Σ_k f_jk = d_j                  ∀ customers j   (load balance)
     Σ_j f_1j − Σ_i f_i1 = Σ_j d_j                             (depot load)
     d_j x_ij ≤ f_ij ≤ (Q − d_i) x_ij           ∀ arcs          (coupling)
     x ∈ {0,1}, f ≥ 0
```

The coupling rows are the strengthened Gavish–Graves bounds: an arc entering
`j` carries at least `j`'s demand, an arc leaving customer `i` at most what is
left after serving `i`. Because load originates at the depot and only travels
on used arcs, the LP relaxation cannot close free fractional subtours; it is a
genuine depot-anchored routing relaxation (mixed 0/1 and fractional arcs, not
the all-½ point of a pure two-index model). A fractional `x` is still not an
implementable set of routes.

## Feasibility

- `feasible`: `K·Q ≥ 1.15 · total_demand`, every demand ≤ `Q`, and `Q` is raised
  until a first-fit-decreasing packing fits the demands in ≤ `K` bins. The bins
  are split into exactly `K` non-empty routes and ordered by a nearest-neighbour
  walk → `CVRPWitness(routes)`. Setting `x = 1` along the routes and `f` to the
  remaining route load satisfies every row of the MIP and of the relaxation.
- `infeasible`: vehicles are out of service — the fleet shrinks to
  `K = ⌊total_demand / (Q · overload)⌋` with `overload ∈ [1.1, 1.3]` and `Q`
  is raised to `total_demand / (K · overload)`, so every customer still fits a
  vehicle but `total_demand = overload · K·Q` →
  `CVRPFleetCapacityCertificate(total_demand, fleet_capacity)`. The proof chains
  the depot load row, the depot-arc coupling rows and the depot degree row
  (`total_demand ≤ Σ_j f_1j ≤ Q Σ_j x_1j = Q·K`), so presolve does not see it
  and simplex needs hundreds of iterations at 10k variables. (Tiny instances
  that cannot fill one overloaded vehicle inflate demands instead.)
- `unknown`: the natural instance as sampled. Fleet sizing keeps the LP
  relaxation feasible; integer feasibility (a bin-packing question) is left
  undetermined.

## Notes

- With the default `relax_integer=true`, `x` is relaxed to `[0, 1]`.
- All randomness is drawn from a constructor-local `MersenneTwister(seed)`.
- Tests: `test/problem_types/vehicle_routing.jl` checks the sizing and row
  counts, replays the route witness through `primal_feasibility_report`, checks
  the certificate arithmetic, reproducibility, and HiGHS contracts (including
  the unrelaxed MIP on a small feasible instance).
