# Load Balancing

Minimise the worst-case load: traffic engineering on an ISP-style backbone
(`standard`) and service placement with workload routing across machines
(`discrete_placement`). Both generators use a constructor-local RNG;
`build_model` does no sampling.

## Variants

| Variant | Model class | Key structure |
|---|---|---|
| `standard` (default) | LP | path-based min-max-utilization TE: OD demand rows, link rows `sum_{p uses a} x_p - c_a U <= -background_a`, SLA bound on `U` |
| `discrete_placement` | MIP | binary service placements, continuous per-class workload routing linked to placements, machine-load definitions, makespan |

## `standard`: path-based traffic engineering

```text
minimize    U + latency_weight * sum_p latency[p] * x[p]
subject to  sum_{p in P(k)} x[p] = demand[k]                         every TE OD pair k
            sum_{p uses a} x[p] - capacity[a] * U <= -background[a]   every link in use
            0 <= U <= max_utilization,  x >= 0
```

The previous version routed a single aggregated injection vector (the
per-pair demand dictionary it stored was unused), so its "multi-commodity"
story collapsed to one commodity; its unknown profile was a 70/30 coin flip,
its infeasible profile zeroed a source's links (refuted by presolve) and it
presolved to nothing on downstream pipelines. The rebuild is a genuine
path-based TE LP — structurally distinct from the arc-based
`multi_commodity_flow`: long path columns (about 14 links each at 100k), one
row per OD pair and per link, and the utilization column `U` in every link row.

### Data grounding

- **Backbone**: `n ~ 1.2 sqrt(target / 3)` PoPs from `_geo_positions`
  (metro clusters dominate), links from `_geo_network` (4.5-6.5 directed links
  per PoP, strongly connected, no dead ends). Link latency =
  distance / 200 + 0.1 per hop.
- **Traffic**: OD pairs drawn by gravity weight `w_o w_d / (1 + dist / L)`,
  demands the same gravity times lognormal noise (mean 10).
- **Candidate paths**: 5-8 successive shortest-path trees per origin, link
  lengths multiplied by `exp(0.7 * uses)` on links earlier trees used (plus
  10% noise), deduplicated — the diverse k-path sets TE tools precompute. OD
  pairs with three or more distinct paths are TE pairs (paths trimmed at
  random, never below three, to hit the size); pairs with fewer stay on their
  shortest path as fixed **background** traffic (otherwise their demand rows
  would be presolve-substitutable doubletons).
- **Capacities** are standard router ports (1, 2.5, 10, 40, 100, 400 units;
  bundles of 400 beyond), provisioned so the planted routing runs at a target
  utilization of 45%-80% of the SLA; the SLA `max_utilization` is 0.8, 0.9 or
  1.0.

### Feasibility control

The planted routing puts every TE demand on its first (shortest) path.

- `feasible`: witness = that routing and its utilization (at most the SLA).
- `infeasible`: TE traffic grows until a **latency-metric certificate**
  separates: with `l` = link latency on the links in the model,
  `sum_k demand_k * min_{p in P(k)} l(p) + sum_a l_a background_a`
  exceeds `max_utilization * sum_a l_a capacity_a` (capacity-length 80%-93% of
  the requirement). A **local repair** with 15% slack keeps every link able to
  carry its forced load (background plus the demands of pairs all of whose
  paths use it) and every pair's summed path bottlenecks above its demand, by
  upgrading ports — so no single link or demand row is refutable and presolve
  keeps the whole model. (On tiny backbones the slack is relaxed so the
  certificate can separate.)
- `unknown`: traffic grows by 0.6-1.2x the factor that would bring the planted
  routing to the SLA (never shrinking), with the same repair: rerouting over
  the alternative paths may or may not absorb it.

### Sizing

Variables = TE candidate paths + 1, within 2 of the target (paths are trimmed
in steps that keep at least three per pair). Rows = TE OD pairs + links
carrying a path or background (about a third of the columns). Build is about
half a second at 100k variables and ten seconds at the 1,000,000 cap
(`LOAD_BALANCING_MAX_VARIABLES`); nonzeros grow like the mean path length
(about 1.4M at 100k).

## `discrete_placement`

```text
minimize    makespan + 1e-4 * sum placement
subject to  1 <= sum_m placement[s,m] <= max_replicas[s]                    every service
            sum_m workload[k,s,m] = demand[k,s]                              every class, service
            workload[k,s,m] <= demand[k,s] * placement[s,m]                  every class, service, machine
            machine_load[m] = sum_{k,s} processing_time[s,m] * workload[k,s,m]
            machine_load[m] <= makespan,  0 <= machine_load[m] <= capacity[m]
            placement binary
```

The routing upper bounds use each class's own demand (not a big-M), so they
stay tight after relaxation; the replica cardinality rows keep placement
decisions meaningful. A `grid_side x grid_side` placement grid (3-20) with the
number of traffic classes absorbing the rest of the target gives
`S*M + K*S*M + M + 1` variables.

- `feasible`: a permutation places every service on one machine
  (`DiscretePlacementWitness`, MIP-feasible), capacities 1.10-1.35x the planted
  loads.
- `infeasible`: total capacity is 65%-85% of the workload lower bound
  `sum demand * min_m processing_time` (`DiscretePlacementCertificate`) —
  valid when the placement binaries are relaxed.
- `unknown`: every machine is sized at a common factor in [0.65, 1.1] times
  0.85-1.15 of its planted load: some machines are short, and replicas on
  other machines may or may not absorb the gap.

## References

- Fortz, B., Thorup, M. (2000). Internet traffic engineering by optimizing
  OSPF weights. INFOCOM.
- Wang, Y., Wang, Z. (1999). Explicit routing algorithms for Internet traffic
  engineering. ICCCN.
- Onaga, K., Kakusho, O. (1971). On feasibility conditions of multicommodity
  flows in networks. IEEE Transactions on Circuit Theory 18(4).
