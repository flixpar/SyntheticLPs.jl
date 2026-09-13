# Network Flow

Single-commodity flow over capacitated directed networks. The category contains
two variants: `standard`, the classical max-flow / min-cost-flow LP with an
exact max-flow-based feasibility control, and `generalized_flow`, a lossy-flow
variant whose arcs attenuate what they carry. Both build pure continuous LPs,
and both use a local RNG — calling a generator neither reseeds nor consumes
Julia's global RNG, and `build_model` does no sampling.

## Variants

| Variant | Model class | Key structure | Domain grounding |
|---|---|---|---|
| `standard` | continuous LP | max-flow or min-cost flow with a contracted volume; exact Dinic max flow drives feasibility | freight corridors, regional grids, mesh distribution |
| `generalized_flow` | continuous LP | per-arc gain multipliers in (0, 1]; required *delivered* amount at the sink | transmission losses, evaporation, conversion yield |

## `standard` Formulation

Sets and data:

- `V = {1, ..., n_nodes}`, `s = source_node = 1`, `t = sink_node = n_nodes`
- `A = arcs`: forward arcs `(i, j)` with `i < j`, always containing the backbone
  path `1 -> 2 -> ... -> n_nodes`; the network is a DAG ordered by node index
- `cap[a]`, `cost[a]`: per-arc capacity and per-unit routing cost

Decision variable:

```text
flow[a] >= 0 = flow on arc a        (one variable per arc)
```

Capacity rows, one per arc:

```text
flow[a] <= cap[a]                   for each a in A
```

Flow conservation at every intermediate node:

```text
sum_{(u,v) in A} flow[u,v] = sum_{(v,w) in A} flow[v,w]
    for each v in V \ {s, t}
```

Forward-only arcs are what make the feasibility control airtight: nothing can
re-enter the source or leave the sink, so source outflow, sink inflow, and
cross-cut flow all coincide and classical max-flow / min-cut theory applies
verbatim to the built LP.

The objective is sampled per instance:

- `:max_flow` — maximize source outflow:

  ```text
  maximize sum_{(s,j) in A} flow[s,j]
  ```

- `:min_cost` — route a *contracted volume* `target_flow` at minimum cost,
  enforced by one extra equality on the source's out-arcs:

  ```text
  minimize sum_{a in A} cost[a] * flow[a]
  sum_{(s,j) in A} flow[s,j] = target_flow
  ```

The struct stores `arcs` (sorted), aligned `capacities`/`costs` vectors, node
`positions` and the sampled `geography` shape, `flow_objective`,
`target_flow`, the exact `max_flow_value`, a typed `feasible_witness` /
`infeasibility_certificate` (or `nothing`), and the requested
`feasibility_status`.

## `standard` Data Grounding

Nodes are scattered over a 100 x 100 region in one of three sampled geography
shapes:

- `:corridor` — nodes strung along a slightly bent transport corridor *in index
  order*, so the source and sink sit at opposite ends and the backbone follows
  the corridor (pipelines, rail, river navigation);
- `:clustered` — nodes grouped around a handful of well-separated cluster
  centers with Gaussian spread (regional road networks);
- `:uniform` — uniform coverage of the region (mesh-like distribution grids).

Per-unit routing cost is proportional to the Euclidean distance between the
arc's endpoints, multiplied by a lognormal route factor whose spread is drawn
once per instance — long-haul arcs cost more, with route-dependent
heterogeneity on top, so cost is strongly correlated with distance rather than
uniform noise. Capacities are lognormal, in tiers that grow with the requested
scale (roughly feeder networks -> trunk lines -> backbone grids): median around
40 for targets up to 100 arcs, up to around 600 near the 1,000,000-arc cap,
with tier-dependent spread.

## `standard` Feasibility Control

The constructor computes the TRUE maximum source-sink flow (Dinic's algorithm,
deterministic, on the planted capacities) and derives the min cut from the
residual graph, so every profile is placed relative to an exact boundary
rather than an estimate:

- `feasible`: `target_flow` is 25%-85% of the max flow. The stored witness is
  the max-flow assignment scaled by `target_flow / max_flow_value` (`:min_cost`)
  or the unscaled assignment (`:max_flow` — a feasible point that is in fact
  optimal); scaling preserves conservation and keeps every capacity row
  satisfied by construction.
- `infeasible`: a max-flow objective is always feasible (the zero flow is
  admissible), so the request becomes a min-cost instance whose contracted
  volume is 115%-160% of the max flow — a realistic "contracted volume cannot
  be routed" scenario. The stored certificate is a minimum cut: the cut arcs
  are exactly the arcs whose tail lies on the source side and whose head does
  not, and their total capacity equals the max flow and is strictly below the
  contract. Any feasible flow must push at least the contracted volume across
  the cut (nothing re-enters the source), which the cut cannot carry — the
  source-outflow equality is refuted using LP rows alone.
- `unknown`: the objective is kept. A `:min_cost` contract is drawn as
  60%-140% of the max flow — a genuine coin flip on either side of the
  feasibility boundary at every problem size (routable exactly when the
  contract does not exceed the max flow); `:max_flow` instances stay
  unconstrained (trivially feasible, no claim either way). Neither a witness
  nor a certificate is stored.

The model is a pure continuous LP, so the certificate refutes the model as
built and survives every transform.

## `standard` Sizing and Variable Counts

Variables are the arcs, and the arc count equals the target exactly. `n_nodes`
is the smallest `n >= 4` with `n * (n - 1) / 2 >= target_variables` (plus the
backbone, source/sink shortcuts, and a shuffled fill of the remaining forward
candidates up to the target), so:

```text
n_nodes = max(4, ceil((1 + sqrt(1 + 8 * target_variables)) / 2))
variables = target_variables
rows = target_variables + (n_nodes - 2) + [1 if :min_cost]
```

Targets below 3 round up to the 3-arc backbone of the minimum 4-node network.
The maximum supported target is 1,000,000 arcs (about 1415 nodes, two million
candidate arcs); larger requests raise an `ArgumentError` instead of silently
undersizing, the same convention as `telecom_network_design` and
`supply_chain/network_planning`.

## `generalized_flow`

Models lossy flow: each arc `(i, j)` has a multiplicative gain `g in (0.85,
1.0]`; flow sent on an arc arrives at its head multiplied by `g`
(transmission losses, evaporation, conversion yield), making conservation
multiplicative. Node 1 is the source, node `n_nodes` the sink, and the model
must DELIVER a required amount at the sink (post-gain inflow on the sink's
in-arcs) while minimizing routing cost subject to per-arc capacities and a
source-supply cap. The topology mirrors `standard` (backbone path, forward
arcs, shortcuts).

Feasibility profiles:

- `feasible`: the backbone path is guaranteed to deliver the demand — backbone
  capacities and the source supply cover `demand / prod(g over backbone)` with
  slack.
- `infeasible`: the post-gain inflow capacity into the sink is capped so that
  `sum over sink in-arcs of g * cap = demand * alpha` with `alpha in [0.7,
  0.9] < 1`; delivered flow can never reach `demand` regardless of the rest of
  the network, an LP-relaxation-proof pigeonhole bound.
- `unknown`: a natural instance biased toward feasible.

Variables are again one per arc (`length(arcs)`).

## References

- Ahuja, R.K., Magnanti, T.L., Orlin, J.B. (1993). Network Flows: Theory,
  Algorithms, and Applications. Prentice Hall. (Max-flow/min-cut theory,
  min-cost flow formulations.)
- Dinic, E.A. (1970). Algorithm for solution of a problem of maximum flow in
  networks with power estimation. Soviet Mathematics Doklady 11.
