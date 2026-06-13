# Network Flow

The network flow generator creates continuous single-commodity flow LPs over a directed graph, either maximizing source outflow or minimizing cost for a required flow.

## Overview

This generator represents routing through a capacitated directed network. Node 1 is the source, the last node is the sink, arcs have capacities and per-unit costs, and intermediate nodes conserve flow. Depending on the sampled objective type, the model either sends as much flow as possible from source to sink or sends a specified target amount at minimum cost.

## Generator Data and Sizing

`target_variables` maps to the number of directed arcs:

```text
target_variables ~= length(arcs)
```

The constructor seeds Julia's global RNG with `Random.seed!(seed)`. The same target, feasibility status, and seed reproduce the same graph, capacities, costs, objective choice, and target flow.

Problem scale controls node search ranges, density, capacities, and costs:

| Target variables | Node search range | Target density | Capacity range | Cost range |
| --- | --- | --- | --- | --- |
| `<= 100` | `4:15` | `0.4` | `10.0` to `100.0` | `1.0` to `10.0` |
| `<= 500` | `10:35` | `0.2` | `10.0` to `500.0` | `1.0` to `25.0` |
| `> 500` | `20:100` | `0.1` | `50.0` to `2000.0` | `1.0` to `50.0` |

The node count is the first value whose rounded dense-arc estimate reaches at least 90% of the target. `source_node` is always 1 and `sink_node` is always `n_nodes`.

Topology generation first creates the path `(1,2), (2,3), ..., (n_nodes-1,n_nodes)` to ensure source-to-sink connectivity. It then optionally adds arcs from the source to intermediate nodes and from intermediate nodes to the sink with probability 0.3 each, followed by shuffled random directed arcs without self-loops until the requested arc count is reached. If the path alone already exceeds `target_variables`, the returned graph can have more arcs than the target.

Each arc gets:

- capacity sampled uniformly from the scale-specific floating-point interval and rounded to 2 digits
- cost sampled uniformly from the scale-specific floating-point interval and rounded to 2 digits

The struct stores:

- `n_nodes::Int`
- `source_node::Int`
- `sink_node::Int`
- `arcs::Vector{Tuple{Int,Int}}`
- `capacities::Dict{Tuple{Int,Int},Float64}`
- `costs::Dict{Tuple{Int,Int},Float64}`
- `flow_objective::Symbol`, either `:max_flow` or `:min_cost`
- `target_flow::Union{Float64,Nothing}`

Objective selection depends on target size: targets `<= 100` choose `:max_flow` with probability 0.7; larger targets choose `:max_flow` with probability 0.4. The remaining cases use `:min_cost`.

## LP Formulation

Sets:

- `V = {1, ..., n_nodes}`
- `A = arcs`
- `s = source_node`
- `t = sink_node`

Decision variable:

```text
flow[a] >= 0 = flow on directed arc a
```

Capacity bounds are modeled as constraints:

```text
flow[i,j] <= capacities[(i,j)]    for each (i,j) in A
```

Intermediate-node flow conservation:

```text
sum_{(i,v) in A} flow[i,v] = sum_{(v,j) in A} flow[v,j]
    for each v in V \ {s,t}
```

For `:max_flow`, the objective is:

```text
maximize sum_{(s,j) in A} flow[s,j]
```

For `:min_cost`, the objective and source-flow requirement are:

```text
minimize sum_{a in A} costs[a] * flow[a]
sum_{(s,j) in A} flow[s,j] = target_flow
```

The model does not add a corresponding sink inflow requirement. Conservation at intermediate nodes plus the graph structure usually forces flow to terminate at the sink when there are no useful cycles, but cycles and arcs into the source or out of the sink are allowed by topology generation.

## Feasibility Controls

The generator estimates maximum available flow as the sum of capacities on arcs leaving the source. This is an upper bound, not an exact max-flow solve.

- `feasible`: for `:min_cost`, sets `target_flow` to 10% to 40% of the source-out capacity estimate. For `:max_flow`, no extra requirement is added.
- `infeasible`: if the sampled objective is `:min_cost`, sets `target_flow` to 120% to 170% of the source-out capacity estimate. If the sampled objective is `:max_flow`, the constructor switches to `:min_cost` and uses the same impossible target rule.
- `unknown`: randomly maps to feasible with probability 0.7 or infeasible with probability 0.3 before applying the rules above.

Because the target-flow logic is based only on source-out capacity, feasible `:min_cost` instances are intended to be feasible but are not verified by an exact max-flow computation. Infeasible target-flow instances are guaranteed with respect to the model's source-out equality and source capacity constraints when the estimate is positive.

## Model Characteristics

The model has one continuous nonnegative variable per arc. It has one capacity constraint per arc, plus one conservation constraint for each intermediate node with incident arcs. A `:min_cost` instance with a target flow adds one source-flow equality.

The matrix is sparse: each arc variable appears in its own capacity row and in up to two conservation rows, plus possibly the source-flow row. The formulation is continuous. Network flow LPs often have integral extreme points under integral capacities for standard source-sink formulations, but this generator does not declare integer variables.

## Practical Notes

These instances are useful for testing sparse network LP behavior, objective switching, and capacity-conservation structure. The implementation permits arbitrary directed arcs, including arcs into the source, arcs out of the sink, and cycles. The `max_flow_estimate` is deliberately simple, so it should be treated as a data-generation heuristic rather than a certified graph-theoretic max flow.
