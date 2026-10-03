# Graph Optimization

The `graph_optimization` category generates packing, covering, coloring, and
dense-subgraph models on graphs drawn from the applications that produce them:
hotspot unit-disk interference graphs (wireless networks, siting), scale-free
router topologies, map-label overlap graphs, and heavy-tailed community
networks. Every variant uses a formulation whose LP relaxation still depends on
the graph — clique rows instead of edge rows, capacity linking instead of plain
covering, per-channel cliques instead of the collapsing assignment model.

## Variants

| Variant | Application | Formulation |
| --- | --- | --- |
| `independent_set` (default) | Simultaneously active wireless transmitters / dispersed sites | Weighted MIS, clique formulation, service floor |
| `generalized_independent_set` | Wind-farm layout | Spacing cliques + wake-loss penalty pairs (GISP), capacity floor |
| `vertex_cover` | Link-monitoring probe placement on a router topology | Capacitated vertex cover with one orientation variable per link |
| `vertex_coloring` | WLAN channel assignment | Min-interference list coloring, per-channel clique rows |
| `map_labeling` | Point-feature map labeling | Label-candidate packing, Helly clique rows, coverage floor |
| `quasi_clique` | Community extraction in social / protein networks | Densest-`k`-subgraph with edge activations and a density floor |

## Graphs and data

All builders are near-linear (grid bucketing, no all-pairs scans), so 100k
variables build in about two seconds.

- **Hotspot unit-disk graphs** (`independent_set`, `generalized_independent_set`,
  `vertex_coloring`): sites in a square sized for a target mean degree, with
  roughly half of them in Gaussian hotspots 2–5× denser than the background.
  Edges join sites within the unit interference / spacing radius. Hotspots
  give heterogeneous degrees and natural cliques of 20–40 sites.
- **Clique cover**: grid cells of side `1/√2` are cliques; each is grown to a
  maximal clique, then every still-uncovered edge seeds a further greedy
  clique. The packing rows `Σ_{v∈K} x_v ≤ 1` imply every edge row and give the
  classical strong clique formulation, whose LP optimum is not the all-½ point
  of the edge relaxation. The LP optimum of `Σ x` sits 12–19% above a greedy
  independent set.
- **Scale-free graph** (`vertex_cover`): preferential attachment, 80%
  degree-proportional and 20% uniform, with exactly `m` edges.
- **Map labels** (`map_labeling`): towns with lognormal importance. The top
  10% get a larger font. Each town has four label boxes (NE, NW, SE, SW;
  a few towns get a fifth, centred above, so the count is exact), with widths
  set by name length. Two boxes conflict when they overlap. Pairwise
  overlapping boxes share a common point (Helly), so the clique rows are stacks
  of labels in crowded map regions.
- **Community network** (`quasi_clique`): heavy-tailed communities (8–120
  members) take 60–80% of the edges. The rest join vertices weighted by
  lognormal degree propensities (Chung–Lu), which creates hubs.
- **Values**: lognormal traffic demand that grows with local crowding
  (`independent_set`); wind yield from a smooth ridge field, with wake penalties
  that fall with distance along the prevailing wind (`generalized_independent_set`);
  probe cost of installation plus ports (`vertex_cover`); channel costs from a
  spatial field of external interferers, with DFS channels costlier
  (`vertex_coloring`); importance × cartographic position preference, cut to a
  quarter when a box hides another town's symbol (`map_labeling`).

## Formulations and sizing

| Variant | Variables | Rows |
| --- | --- | --- |
| `independent_set` | `n = target` | clique rows + floor |
| `generalized_independent_set` | `n + #wake pairs = target` (strongest pairs kept) | hard clique rows + one row per wake pair + floor |
| `vertex_cover` | `n + m = target` (`x` per vertex, `z` per link) | `2m` linking + `n` capacity |
| `vertex_coloring` | `Σ_v |D_v| = target` (domains adjusted exactly) | `n` assignment + one per (clique, channel) with ≥ 2 members |
| `map_labeling` | candidate boxes `= target` | clique rows + floor |
| `quasi_clique` | `n + m = target` (`y` only on real edges) | `2m` linking + cardinality + density |

`vertex_cover` models link `e = (u, v)` with one variable `z_e`, the share
monitored from `u`. It has rows `z_e ≤ x_u`, `1 − z_e ≤ x_v` and
`Σ_{e∋v} load_e(v) ≤ cap_v x_v`. A two-variable assignment `y_eu + y_ev = 1`
is a doubleton equality that presolve would immediately substitute out.

`vertex_coloring` minimizes `Σ cost[v,c] x[v,c]` subject to
`Σ_{c∈D_v} x[v,c] = 1` and `Σ_{v∈K} x[v,c] ≤ 1` for every clique `K` and
channel `c`. The compact model it replaces (assignment variables, color-use
variables, edge rows) collapses under relaxation: `x = 1/k` satisfies every
edge row, so the graph has no effect on the bound.

## Feasibility

Each variant stores a typed `feasible_witness` for `feasible` requests and a
typed `infeasibility_certificate` for `infeasible` ones. Both are built from LP
rows alone, so they hold under the default `relax_integer=true`. Every
certificate aggregates hundreds to thousands of rows, so HiGHS presolve does not
detect any of these infeasible instances. At 10k variables they take 300–10,000
simplex iterations.

| Variant | `feasible` witness | `infeasible` certificate | `unknown` |
| --- | --- | --- | --- |
| `independent_set`, `generalized_independent_set` | greedy (hard-graph) independent set; floor 85–100% of its size | `CliquePartitionCertificate`: vertices partitioned into cliques, each inside a model row, prove `Σx ≤ #parts`; floor = bound + 2% | floor between greedy size and the partition bound |
| `vertex_cover` | greedy orientation of every link toward the endpoint with spare capacity (upgrading a card when both are full) | `VertexCoverDeficitCertificate`: the top-5% rich club's links exceed its total card capacity by ≥ 10% (Hakimi) | capacities a random 25–75% of each degree |
| `vertex_coloring` | largest-degree-first coloring, planted in the domains | `OvercrowdedCliqueCertificate`: the largest clique row (≥ 5 access points in practice) shares a plan with one channel fewer than its size | no planted coloring; channel count = greedy colors −2..+1 |
| `map_labeling` | greedy labeling by importance | clique partition of the candidates (at most `#features`) | floor between greedy count and bound |
| `quasi_clique` | hidden community of `k` members with density ≥ γ + 0.05 | `DegeneracyBoundCertificate`: charging each edge to its earlier-peeled endpoint gives `Σy ≤ Σ_v load_v x_v ≤ top-k(load)`; requirement = bound + 3% | requirement between the greedy-peeled value and the bound |

The `unknown` thresholds sit between an integral lower reference and an LP-valid
upper bound. The LP optimum falls in between, so over 8 seeds every variant
returns both feasible and infeasible relaxations at 2k and 20k variables.

## Notes

- All variants are MIPs. The public API returns their LP relaxations by
  default, and the clique and capacity rows keep the relaxed models
  graph-dependent.
- At 100k variables, presolve keeps 98–100% of columns for every variant except
  `vertex_coloring`, which keeps 68–90%. There, channels no other clique member
  can use are dominated by the access point's cheapest such channel.
