# Multi-Commodity Flow

The multi-commodity flow generator creates continuous minimum-cost routing LPs in which several commodities share capacities on the same directed network.

## Overview

This generator represents shared infrastructure planning: different freight classes, products, or traffic demands each need to move from their own source to their own sink, while all commodities compete for capacity on common arcs. The model minimizes total routing cost subject to arc capacity limits and per-commodity flow conservation.

## Generator Data and Sizing

`target_variables` maps to the product of commodities and arcs:

```text
target_variables ~= n_commodities * n_arcs
```

The constructor seeds Julia's global RNG with `Random.seed!(seed)`. `sample_parameters_mcf` also calls `Random.seed!(seed)`, so parameter selection and subsequent data generation are reproducible for a given input.

Parameter selection searches for a realistic combination of commodity count, node count, and arc count. If no combination gets within 10% relative error, it falls back to a heuristic using a random commodity count and target density.

| Target variables | Commodities | Nodes | Density band | Capacity range | Demand range | Cost range | Utilization |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `<= 100` | `2:5` | `5:15` | `0.2` to `0.6` | `20.0` to `200.0` | `5.0` to `50.0` | `1.0` to `15.0` | `0.6` to `0.8` |
| `<= 500` | `3:15` | `10:30` | `0.15` to `0.5` | `50.0` to `500.0` | `10.0` to `100.0` | `1.0` to `25.0` | `0.5` to `0.7` |
| `> 500` | `8:50` | `15:100` | `0.1` to `0.4` | `100.0` to `2000.0` | `20.0` to `500.0` | `1.0` to `50.0` | `0.4` to `0.6` |

The generated network starts with a directed cycle through all nodes, which gives every node reachability around the cycle. It then adds a limited number of reverse cycle arcs, shortcut arcs, and finally shuffled random directed arcs without self-loops until the target arc count is reached. `n_arcs` is capped by `n_nodes * (n_nodes - 1)`, but if the requested arc count is below `n_nodes`, the initial cycle can still produce more actual arcs than `n_arcs`.

Arc capacities are sampled from a log-normal distribution centered on the geometric mean of the capacity range, clamped to the range, and rounded to 2 digits. Demands are similarly log-normal over the demand range and rounded to 2 digits. Costs are sampled uniformly from the cost range and multiplied by a congestion factor from 1.0 to about 1.3, where lower-capacity arcs become more expensive.

Commodity pairs are generated as source-sink pairs with a 40% short-haul and 60% long-haul pattern, avoiding duplicate pairs during the first 100 attempts per commodity. If uniqueness attempts fail, additional non-identical source and sink pairs are added without checking duplication.

The struct stores:

- `n_nodes::Int`
- `n_arcs::Int`
- `n_commodities::Int`
- `arcs::Vector{Tuple{Int,Int}}`
- `capacities::Dict{Tuple{Int,Int},Float64}`
- `demands::Dict{Int,Float64}`
- `costs::Dict{Tuple{Int,Int},Float64}`
- `commodities::Vector{Tuple{Int,Int}}`

## LP Formulation

Sets:

- `V = {1, ..., n_nodes}`
- `A = arcs`
- `K = {1, ..., n_commodities}`
- commodity `k` has source `s_k`, sink `t_k`, and demand `d_k`

Decision variable:

```text
flow[k,a] >= 0 = flow of commodity k on arc a
```

Objective:

```text
minimize sum_{a in A} costs[a] * sum_{k in K} flow[k,a]
```

Shared arc capacities:

```text
sum_{k in K} flow[k,a] <= capacities[a]    for each a in A
```

Flow conservation and demand satisfaction:

```text
outflow(k,s_k) - inflow(k,s_k) = d_k
inflow(k,t_k) - outflow(k,t_k) = d_k
inflow(k,v) = outflow(k,v)                 for v not in {s_k,t_k}
```

All variables are continuous and nonnegative. There are no commodity-specific arc eligibility restrictions; every commodity can use every generated arc.

## Feasibility Controls

The constructor maps statuses to internal symbols: `feasible` becomes `:feasible`, `infeasible` becomes `:infeasible`, and `unknown` becomes `:all`.

- `feasible`: computes total demand and total capacity. If total capacity is below `total_demand / capacity_utilization`, all capacities are scaled up proportionally. It then checks reachability for each commodity and adds a direct or two-hop path if needed, assigning new arcs random capacities and costs.
- `infeasible`: first either scales all capacities down to 30% to 70% of their current values or scales all demands up to 150% to 250%. It then calls `enforce_infeasibility_mcf!`, which selects up to three commodities and reduces either total outgoing capacity at the commodity source or total incoming capacity at the commodity sink below that commodity's demand. If targeted bottlenecks fail, it inflates one demand beyond total network capacity.
- `unknown`: leaves the sampled data in its natural random state, except for the base strongly connected topology.

The feasible path is based on aggregate capacity scaling plus reachability checks. It is intended to improve feasibility but does not solve a multi-commodity feasibility LP. The infeasible path is stronger because a source or sink cut with total capacity below a commodity's demand is a direct certificate of infeasibility.

## Model Characteristics

The intended variable count is `n_commodities * length(arcs)`. The struct's `n_arcs` field records the target/capped arc count, while `length(arcs)` is the actual number of arcs used by the model. The actual count can differ when the initial connectivity construction adds more arcs or feasible repair adds paths.

Constraint counts are driven by:

- one shared capacity constraint per actual arc
- one flow-balance constraint for every commodity-node pair

The resulting matrix is sparse but larger than single-commodity flow: each commodity-arc variable appears in one shared capacity row and in two node-balance rows for that commodity. The model is a continuous LP relaxation of a routing problem; it does not enforce unsplittable flows or integer path choices.

## Practical Notes

These instances are useful for testing large sparse LPs with coupling constraints, because arc capacities couple otherwise separate commodity flows. They also exercise denser conservation structure than single-commodity flow. The main caveat is that the feasible generator uses sufficient-looking heuristics rather than optimization-based verification, so generated `feasible` data should be interpreted as constructed-to-be-feasible rather than formally certified by the constructor.
