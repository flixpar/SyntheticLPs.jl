# Load Balancing

Load balancing generates a continuous network LP that minimizes maximum link utilization while routing fixed traffic demands along preselected paths.

## Overview

This generator represents network traffic load balancing. A directed network has link capacities and source-target traffic demands. Each demand is assigned a single generated path. The optimization sets link flows and minimizes the maximum utilization factor needed to carry all path demands.

Unlike a full multicommodity flow model, this implementation does not choose paths or split flow. The paths are generated before the model is built, and each link on a demand path must carry at least that demand amount.

## Generator Data and Sizing

The constructor documents `target_variables` as `1 + number of links`, matching the model variables `u` plus one flow variable per link. It sets:

```text
target_links = max(1, target_variables - 1)
n_links = target_links
```

Network scale parameters depend on `target_variables`.

For `target_variables <= 250`:

- `base_nodes`: rounded `Uniform(5, 20)`.
- `density_dist`: `Uniform(0.4, 0.7)`.
- `capacity_mean`: `500.0`.
- `capacity_std`: `150.0`.
- `demand_mean`: `50.0`.
- `demand_std`: `20.0`.
- `max_path_length`: `DiscreteUniform(2, 4)`.

For `250 < target_variables <= 1000`:

- `base_nodes`: rounded `Uniform(20, 60)`.
- `density_dist`: `Uniform(0.25, 0.5)`.
- `capacity_mean`: `2000.0`.
- `capacity_std`: `600.0`.
- `demand_mean`: `150.0`.
- `demand_std`: `60.0`.
- `max_path_length`: `DiscreteUniform(3, 6)`.

For `target_variables > 1000`:

- `base_nodes`: rounded `Uniform(50, 150)`.
- `density_dist`: `Uniform(0.15, 0.35)`.
- `capacity_mean`: `8000.0`.
- `capacity_std`: `2000.0`.
- `demand_mean`: `500.0`.
- `demand_std`: `200.0`.
- `max_path_length`: `DiscreteUniform(4, 8)`.

`link_density = rand(density_dist)` is sampled but is not used when building links.

Topology generation:

- All directed non-self links are candidates.
- If `n_links` is at least the number of possible links, all possible directed links are used.
- Otherwise, the generator first builds a bidirectional spanning structure by adding `(from_node, to_node)` and `(to_node, from_node)` while connecting all nodes.
- It then appends random remaining links up to the target count and applies `unique`.

Capacity generation:

- `min_capacity`: truncated normal around `capacity_mean * 0.3`, lower-bounded at `10.0`.
- `max_capacity`: `min_capacity` plus a truncated normal around `capacity_mean * 1.2`.
- Each link capacity is sampled from a truncated normal centered between `min_capacity` and `max_capacity`.

Demand generation:

- Possible demands are all ordered source-target node pairs with `i != j`.
- A demand ratio is sampled from `Uniform(0.3, 0.7)`.
- `n_demands` is that ratio times `n_nodes * (n_nodes - 1)`, rounded and clamped to available pairs.
- Demand amounts are sampled from a truncated gamma distribution whose support is based on sampled `min_demand` and `max_demand`.

Path generation:

- For each demand, the generator attempts a random walk from source to target with length up to `max_path_length`.
- It prefers unvisited outgoing neighbors, but may revisit if needed.
- If the random walk reaches the target, the path is stored.
- Otherwise, if a direct link exists, the direct one-link path is stored.
- If no path is found, the demand is deleted.

The stored struct fields are:

- `n_nodes::Int`
- `links::Vector{Tuple{Int,Int}}`
- `capacities::Dict{Tuple{Int,Int},Float64}`
- `demands::Dict{Tuple{Int,Int},Float64}`
- `paths::Dict{Tuple{Int,Int},Vector{Tuple{Int,Int}}}`
- `max_utilization::Union{Float64,Nothing}`

The constructor calls `Random.seed!(seed)`, so generation is reproducible for the same inputs and resets Julia's global RNG.

## LP Formulation

Sets and indices:

- Directed links `a in L`.
- Demands `k = (source, target) in K`.
- Generated path `P_k subseteq L` for each demand with a stored path.

Decision variables:

```text
u >= 0      maximum link utilization
f_a >= 0    flow on link a
```

Objective:

```math
\min u
```

Link utilization constraints:

```math
f_a \le u \cdot capacity_a \quad \forall a \in L
```

Demand/path constraints:

```math
f_a \ge demand_k \quad \forall k \in K, a \in P_k
```

Optional maximum utilization constraint, used for forced infeasibility:

```math
u \le max\_utilization
```

Bounds:

- `u` and all link-flow variables are continuous and nonnegative.

Interpretation: `u` is the largest capacity multiplier needed for any link flow. Each selected path link must be able to carry the full demand amount for every demand whose generated path uses that link.

## Feasibility Controls

The constructor sets `actual_status = feasibility_status`. If `feasibility_status == unknown`, it randomly chooses `feasible` with probability `0.7` and `infeasible` with probability `0.3`.

For `feasible`, no extra cap on `u` is added. Because `u` is unbounded above and all capacities are positive, the model can choose a sufficiently large `u` to satisfy all path lower bounds.

For `infeasible`, if there are nonempty demands and paths, the generator:

1. Initializes a minimum required flow per link to zero.
2. For each demand path, updates each path link's required flow to the maximum demand using that link.
3. Computes:

   ```text
   min_u = maximum(required_flow[link] / capacities[link])
   ```

4. Sets `max_utilization = min_u * (0.5 + 0.3 * rand())`.

The model then enforces `u <= max_utilization`, which is below the utilization needed by at least one demanded path link.

If the infeasible branch has empty demands or paths, `max_utilization` remains `nothing`, so no infeasibility constraint is added.

For `unknown`, the chosen actual status follows the same feasible or infeasible path above.

## Model Characteristics

Variable count:

```text
1 + length(links)
```

Constraint count drivers:

- One utilization constraint per link.
- One lower-bound flow constraint for each pair `(demand, link)` where the link lies on that demand's generated path.
- Optional one upper bound on `u` for forced infeasibility.

The link-flow model is sparse with respect to demand constraints because each demand only constrains links on its generated path. However, link utilization constraints cover every link.

The model is a continuous LP. There are no integer variables.

## Practical Notes

This generator is useful for continuous minimax-style network LPs and for testing models with one global utilization variable coupled to many link variables.

The formulation is not a conservation-based routing model: `f[link]` is a single aggregate flow variable, and each demand path imposes lower bounds on those shared link variables. Multiple demands on the same link do not add in the model; the binding requirement is the maximum lower bound among those constraints, not the sum of all traffic on the link.

For small `target_variables`, the connectivity-building step can create more links than `target_variables - 1` because it first adds bidirectional tree links for all sampled nodes. In that case the final variable count can exceed the target. The sampled `link_density` is currently unused.
