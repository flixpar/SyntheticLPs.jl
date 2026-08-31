using JuMP
using Random
using Distributions

"""
    LoadBalancingProblem <: ProblemGenerator

Generator for load balancing / traffic-engineering problems on a directed network.

# Overview
Models min-max-utilization routing. The decisions are an aggregate flow on each
directed link plus a global maximum-utilization variable `u`. The objective
minimizes the maximum link utilization `max_link(f / capacity)`. Link capacity
rows tie flow to `u`, and **flow-conservation** rows balance in- and out-flow at
every node against that node's net demand injection (supply at traffic sources,
withdrawal at sinks). Conservation is what couples the links, so the model is a
genuine routing LP rather than a collection of independent per-link bounds.

Feasibility:
- `feasible`: the network is connected and `u` is unbounded above, so any
  routable demand is feasible; capacities are scaled so the optimum keeps `u`
  modest (realistic utilization).
- `infeasible`: a traffic source's outgoing link capacities are zeroed, so
  flow conservation at that node cannot hold (it must push out more than the
  zero available outgoing capacity) — a structural contradiction independent
  of `u`.

# Fields
- `n_nodes::Int`: Number of nodes in the network
- `links::Vector{Tuple{Int,Int}}`: List of directed links in the network
- `capacities::Dict{Tuple{Int,Int},Float64}`: Capacity of each link
- `demands::Dict{Tuple{Int,Int},Float64}`: Traffic demand `(source, target) => amount`
- `net_injection::Vector{Float64}`: Per-node net supply (outflow minus inflow required)
"""
struct LoadBalancingProblem <: ProblemGenerator
    n_nodes::Int
    links::Vector{Tuple{Int,Int}}
    capacities::Dict{Tuple{Int,Int},Float64}
    demands::Dict{Tuple{Int,Int},Float64}
    net_injection::Vector{Float64}
end

"""
    LoadBalancingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a load balancing problem instance.

# Arguments
- `target_variables`: Target number of variables (`1 + n_links`)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function LoadBalancingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    rng = MersenneTwister(seed)

    # Scale-dependent magnitude ranges (capacities/demands only).
    if target_variables <= 250
        capacity_mean = 500.0; capacity_std = 150.0
        demand_mean = 50.0; demand_std = 20.0
    elseif target_variables <= 1000
        capacity_mean = 2000.0; capacity_std = 600.0
        demand_mean = 150.0; demand_std = 60.0
    else
        capacity_mean = 8000.0; capacity_std = 2000.0
        demand_mean = 500.0; demand_std = 200.0
    end

    # Variables = 1 (the utilization u) + n_links. Size n_nodes so the complete
    # digraph has at least `target` links, then create exactly target-1 links.
    n_nodes = max(4, ceil(Int, (1 + sqrt(1 + 4 * target_variables)) / 2))
    n_links = max(1, target_variables - 1)

    # Build a strongly connected directed network: a bidirectional spanning tree
    # (so every node can reach every other) plus random links up to n_links.
    possible_links = [(i, j) for i in 1:n_nodes for j in 1:n_nodes if i != j]
    links = Tuple{Int,Int}[]
    connected = [1]
    remaining = collect(2:n_nodes)
    while !isempty(remaining)
        from = rand(rng, connected)
        idx = rand(rng, 1:length(remaining))
        to = remaining[idx]
        deleteat!(remaining, idx)
        push!(links, (from, to))
        push!(links, (to, from))
        push!(connected, to)
    end
    links = unique(links)
    extras = shuffle!(rng, [a for a in possible_links if !(a in links)])
    for a in extras
        length(links) >= n_links && break
        push!(links, a)
    end

    # Capacities (truncated normal).
    min_cap = max(10.0, rand(rng, truncated(Normal(capacity_mean * 0.3, capacity_std * 0.2), 10.0, capacity_mean)))
    max_cap = min_cap + rand(rng, truncated(Normal(capacity_mean * 1.2, capacity_std), capacity_mean * 0.5, capacity_mean * 3.0))
    cap_dist = truncated(Normal((min_cap + max_cap) / 2, (max_cap - min_cap) / 6), min_cap, max_cap)
    capacities = Dict(a => rand(rng, cap_dist) for a in links)

    # Demands: a random set of (source, target) => amount pairs.
    n_demands = max(1, round(Int, n_nodes * (n_nodes - 1) * rand(rng, Uniform(0.3, 0.7))))
    demand_pairs = unique([(rand(rng, 1:n_nodes), rand(rng, 1:n_nodes)) for _ in 1:n_demands])
    filter!(p -> p[1] != p[2], demand_pairs)

    dmin = max(1.0, rand(rng, truncated(Normal(demand_mean * 0.2, demand_std * 0.1), 1.0, demand_mean * 0.5)))
    dmax = dmin + rand(rng, truncated(Normal(demand_mean * 0.8, demand_std), demand_mean * 0.3, demand_mean * 2.0))
    dmean = (dmin + dmax) / 2
    dscale = max((dmax - dmin) / 6, 1e-6)
    dshape = max(dmean / dscale, 1.0)
    demand_dist = truncated(Gamma(dshape, dscale), dmin, dmax)

    demands = Dict{Tuple{Int,Int},Float64}()
    for p in demand_pairs
        demands[p] = rand(rng, demand_dist)
    end

    # Net injection per node (supply at sources, withdrawal at sinks).
    net_injection = zeros(Float64, n_nodes)
    for ((s, t), amount) in demands
        net_injection[s] += amount
        net_injection[t] -= amount
    end

    actual_status = feasibility_status == unknown ? (rand(rng) < 0.7 ? feasible : infeasible) : feasibility_status

    if actual_status == feasible
        # Keep the optimal utilization realistic: every net-source node's outgoing
        # capacity should comfortably exceed the flow it must push out.
        for n in 1:n_nodes
            if net_injection[n] > 0
                out_links = [a for a in links if a[1] == n]
                isempty(out_links) && continue
                out_cap = sum(capacities[a] for a in out_links)
                needed = net_injection[n] * (1.5 + 0.5 * rand(rng))
                if out_cap < needed
                    scale = needed / max(out_cap, 1e-6)
                    for a in out_links
                        capacities[a] *= scale
                    end
                end
            end
        end
    elseif actual_status == infeasible && any(>(0), net_injection)
        # Structural infeasibility: zero a traffic source's outgoing capacities so
        # flow conservation at that node is impossible (outflow is forced to 0 but
        # the node must emit its net injection). Independent of u, so it cannot be
        # "solved away" by growing utilization.
        sources = [n for n in 1:n_nodes if net_injection[n] > 0]
        n = rand(rng, sources)
        for a in links
            if a[1] == n
                capacities[a] = 0.0
            end
        end
    end

    return LoadBalancingProblem(n_nodes, links, capacities, demands, net_injection)
end

"""
    build_model(prob::LoadBalancingProblem)

Build a JuMP model for the load balancing problem. Deterministic — uses only data
from the struct fields.
"""
function build_model(prob::LoadBalancingProblem)
    model = Model()

    # Variables: one utilization variable plus a flow on each link.
    @variable(model, u >= 0)
    @variable(model, f[prob.links] >= 0)

    # Objective: minimize the maximum link utilization.
    @objective(model, Min, u)

    # Capacity coupling: flow on a link bounded by u times its capacity.
    for a in prob.links
        @constraint(model, f[a] <= u * prob.capacities[a])
    end

    # Flow conservation at every node: outflow - inflow == net demand injection.
    for node in 1:prob.n_nodes
        inflow = isempty(prob.links) ? 0.0 : sum(f[a] for a in prob.links if a[2] == node; init = 0.0)
        outflow = isempty(prob.links) ? 0.0 : sum(f[a] for a in prob.links if a[1] == node; init = 0.0)
        @constraint(model, outflow - inflow == prob.net_injection[node])
    end

    return model
end

# Register the variant
register_variant(
    :load_balancing,
    :standard,
    LoadBalancingProblem,
    "Load balancing / traffic-engineering problem that routes demand to minimize the maximum link utilization, with per-node flow conservation",
)
