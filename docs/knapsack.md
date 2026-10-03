# Knapsack

The `knapsack` category holds knapsack-family resource-allocation LPs whose
row count grows with the instance. Every variant stores typed feasibility
evidence: a planted witness for `feasible` requests and a relaxation-valid
certificate for `infeasible` ones that spans many rows (so HiGHS presolve does
not detect it and simplex has to work); `unknown` instances are natural and
come out both ways across seeds.

| Variant | Structure | Columns | Rows |
| --- | --- | --- | --- |
| `multiple_choice` (default) | one configuration per workload (GUB rows) + cluster and shared resources | exactly `n` | `G + 3C + 2` ≈ 0.2 n |
| `multidimensional` | sparse 0/1 MDKP: window resources, 3 shared budgets, program floors | exactly `n` | ≈ n/12 + n/40 + 3 |
| `bounded` | bounded multiple knapsack: stock lots to vehicles, commitments | exactly `n` | ≈ 0.32 n |
| `mixed_integer_set` | HEM-MIK-style knapsack set: many sparse local rows, ≤16 dense rows, profit floor | exactly `n` | ≈ 0.35-0.6 n |

**Removed:** the single-row fractional `standard` knapsack. A one-row knapsack
LP is solved by Dantzig's greedy and HiGHS presolve reduced it to nothing
(SimplexRL found the whole category presolved to empty). The single-row
`bounded` variant was rebuilt as the bounded multiple knapsack below.

## `multiple_choice` — multiple-choice multi-dimensional knapsack (MMKP)

Workloads (classes) choose exactly one of 3-8 service configurations. A class
is homed on one of `C = clamp(round(G / 25), 1, G)` clusters; a configuration
uses the cluster's CPU, memory and storage-I/O plus two shared site resources
(power, WAN). Usage is workload size × resource mix × service level × a
lognormal tilt normalised per option (so configurations trade resources and
no option is cheapest in every resource); value is concave in the level
(`level^0.6`).

```text
max sum v_gk x_gk   s.t.  sum_k x_gk = 1 (each class),
                          cluster rows <= C_cr,  shared rows <= S_s,  x binary
```

- `feasible`: a planted configuration per class (`MultipleChoiceWitness`);
  cluster capacities `U(1.03, 1.20)` and shared `U(1.02, 1.10)` times its usage.
- `infeasible`: one shared capacity below the sum of per-class minimum usage
  by `U(4%, 12%)` (`MinimumLevelCertificate`: the shared row plus the minimum
  usage times every class row).
- `unknown`: capacities around the all-minimal-service plan — cluster
  `U(1.0, 1.4)`, shared `U(0.90, 1.12)` times its usage.

## `multidimensional` — sparse MDKP with program commitments

`n_local = clamp(round(n / 12), 2, n)` machine-week resources on a ring; each
item occupies a window of 2-6 consecutive ones and consumes three shared
budgets (capital, labour, energy). Usage = latent size × resource intensity ×
lognormal noise (correlated columns); values correlate with usage. Items
belong to `clamp(round(n / 40), 1, n)` programs with minimum selection counts
(covering rows), so the LP is not solved by a density greedy.

- `feasible`: 35-50% of items planted (`MultidimensionalSelectionWitness`);
  window capacities `U(1.03, 1.25)` × planted use plus an allowance (below the
  row total), shared `U(1.02, 1.10)`, floors `U(0.5, 0.95)` × planted count.
- `infeasible`: floors raised round-robin until the lightest items meeting them
  need `U(1.04, 1.12)` × one shared capacity (`ProgramFloorCertificate`, a
  combination of every floor row and that budget row).
- `unknown`: capacities are fractions of row totals (window `U(0.35, 0.65)`,
  shared `U(0.30, 0.55)`), floors a per-instance `U(0.35, 0.85)` of each program.

## `bounded` — bounded multiple knapsack with commitments

Stock lots (`u_i ∈ 1:12` units, lognormal unit weight and price) are loaded on
`K = clamp(round(n / 28), 1, n)` vehicles in `clamp(round(K / 40), 1, 6)`
regions. An item may use 2-5 vehicles of its region; column `(i, k)` has a
vehicle-specific unit weight (packaging factor) and margin (lane cost), and is
a general integer in `[0, u_i]`. About a third of the items carry a contracted
minimum delivery, giving ranged item rows `l_i <= sum_k x_ik <= u_i`. Each
column is in two rows with non-unit weights: a generalized-assignment LP.

- `feasible`: planted shipments `round(u_i × U(0.3, 0.9))` split over lanes
  (`BoundedAllocationWitness`); vehicle capacity `U(1.04, 1.25)` × planted
  load plus an allowance (below the full-stock load).
- `infeasible`: the largest region's commitments raised until their
  lightest-lane weight is `U(1.04, 1.12)` × the region's capacity
  (`RegionalCommitmentCertificate`).
- `unknown`: a per-instance share `U(0.2, 0.9)` of items committed at
  `U(0.5, 1.0)` of stock, capacities `U(0.30, 0.65)` × expected full-stock load.

## `mixed_integer_set` — HEM-MIK-style knapsack set

Bounded general integers (`2:10`) plus a continuous block of
`clamp(round(0.04 n), 2, 20)`; `round(n × U(0.35, 0.60))` packing rows with 4-24
nonzeros, of which `clamp(round(rows / 4), 1, 16)` evenly spaced dense rows
cover `U(0.45, 0.80)` of `min(n, 40_000)` columns; plus a profit floor. Sparse rows draw from a local window of related columns. The
previous generator had `O(n^2)` nonzeros (15M at 10k, an 18.7 GB MPS at 50k).
Capacities are built around a nonzero planted point.

- `feasible`: profit floor `U(0.70, 0.90)` × the planted profit.
- `infeasible`: floor `U(1.01, 1.05)` × a Lagrangian dual bound computed by
  projected subgradient (`LagrangianBoundCertificate` stores the row
  multipliers; the bound is recomputed in O(nnz) by `mik_dual_bound`). It is far
  below the box bound, so presolve cannot see it.
- `unknown`: floor `U(0.45, 1.0)` of the way from the planted profit to that
  dual bound.

All variants build in O(nnz) (well under a second at 100k columns).
