using JuMP
using Random

"""
    GeneralizedFlowProblem <: ProblemGenerator

Generator for generalized (lossy) network-flow problems with per-arc gain
multipliers.

# Overview
Models single-commodity flow on a connected directed network in which each arc
`(i, j)` has a multiplicative *gain* `g[i,j] ∈ (0, 1]`. Flow *sent* on an arc is
`f[i,j]`, but only `g[i,j] * f[i,j]` *arrives* at the head node — capturing
transmission/line losses, evaporation, spoilage, or conversion yield. This makes
conservation multiplicative rather than the pure 1:1 balance of the standard
network-flow variant.

Node 1 is the source and node `n_nodes` is the sink. The model must DELIVER a
required amount `demand` at the sink, where delivered flow is the post-gain inflow
on the sink's in-arcs. Subject to per-arc capacities and a source-supply cap, the
objective MINIMIZES total routing cost `sum cost[arc] * f[arc]`. (We deliberately
do not maximize source outflow: with gains that objective is degenerate.)

Generalized conservation at each intermediate node `v` (not source, not sink):

    sum over in-arcs (u,v) of g[u,v] * f[u,v]  ==  sum over out-arcs (v,w) of f[v,w]

# Fields
- `n_nodes::Int`: Number of nodes (node 1 = source, node `n_nodes` = sink)
- `source_node::Int`: Source node index (always 1)
- `sink_node::Int`: Sink node index (always `n_nodes`)
- `arcs::Vector{Tuple{Int,Int}}`: Directed arcs
- `backbone::Vector{Tuple{Int,Int}}`: The source→…→sink backbone path arcs
- `capacities::Dict{Tuple{Int,Int},Float64}`: Per-arc flow (sent) capacity
- `costs::Dict{Tuple{Int,Int},Float64}`: Per-unit-sent routing cost
- `gains::Dict{Tuple{Int,Int},Float64}`: Per-arc gain multiplier in (0, 1]
- `source_supply::Float64`: Cap on total flow sent out of the source
- `demand::Float64`: Required delivered (post-gain) amount at the sink
"""
struct GeneralizedFlowProblem <: ProblemGenerator
    n_nodes::Int
    source_node::Int
    sink_node::Int
    arcs::Vector{Tuple{Int,Int}}
    backbone::Vector{Tuple{Int,Int}}
    capacities::Dict{Tuple{Int,Int},Float64}
    costs::Dict{Tuple{Int,Int},Float64}
    gains::Dict{Tuple{Int,Int},Float64}
    source_supply::Float64
    demand::Float64
end

"""
    _generalized_flow_topology(n_nodes::Int, n_arcs::Int)

Build a connected directed network on `1:n_nodes` with the source→…→sink backbone
path `1→2→…→n_nodes` always present, plus extra random arcs until roughly `n_arcs`
arcs exist. Returns `(arcs, backbone)` where `backbone` is the ordered list of
backbone arcs. Named distinctly so it does not clash with `standard.jl`'s
`generate_connected_network`.
"""
function _generalized_flow_topology(n_nodes::Int, n_arcs::Int)
    arcs = Set{Tuple{Int,Int}}()
    backbone = Tuple{Int,Int}[]

    # Backbone path 1 -> 2 -> ... -> n_nodes (guarantees source->sink connectivity)
    for i in 1:(n_nodes - 1)
        a = (i, i + 1)
        push!(arcs, a)
        push!(backbone, a)
    end

    source = 1
    sink = n_nodes

    # Forward "shortcut" arcs (i -> j with i < j) for realism; keep DAG-like to
    # avoid trivial gain cycles (gains <= 1 already preclude unbounded cycles).
    for i in 2:(n_nodes - 1)
        if rand() < 0.35
            push!(arcs, (source, i))
        end
        if rand() < 0.35
            push!(arcs, (i, sink))
        end
    end

    # Fill in additional forward arcs until we reach the target arc count.
    forward_candidates = [(i, j) for i in 1:n_nodes for j in 1:n_nodes if i < j]
    shuffle!(forward_candidates)
    for arc in forward_candidates
        length(arcs) >= n_arcs && break
        push!(arcs, arc)
    end

    return collect(arcs), backbone
end

"""
    GeneralizedFlowProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a generalized (lossy) network-flow instance.

Variable-count formula (decision variables created by `build_model`):

    total = length(arcs)        # one nonnegative flow variable f[arc] per arc

The constructor sizes `n_nodes` and the arc density so `length(arcs)` lands near
`target_variables`. There is exactly one variable block (per-arc flow), so the
arc count IS the variable count.

# Feasibility
- `feasible`: the backbone path 1→…→n is guaranteed to deliver `demand`. Let
  `P = prod(g over backbone)`. Sending `s = demand / P` at the source arrives as
  `demand` at the sink. Every backbone arc capacity is set `>= s * slack` and
  `source_supply >= s * slack`, so the backbone alone is an admissible delivery —
  finite optimum in the LP relaxation.
- `infeasible`: the post-gain inflow capacity into the sink is capped strictly
  below `demand`: `sum over sink in-arcs of g[u,sink] * cap[u,sink] = demand * α`
  with `α ∈ [0.7, 0.9] < 1`. Since delivered ≤ that aggregate bound regardless of
  the rest of the network, `delivered >= demand` is unsatisfiable. This pigeonhole
  bound holds in the LP relaxation (no integrality used).
- `unknown`: a natural instance biased toward feasible (backbone sized to deliver
  a modest `demand`, no infeasibility forcing).

# Arguments
- `target_variables`: Target number of decision variables (= number of arcs)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function GeneralizedFlowProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # --- Scale-tiered parameter ranges ---
    if target_variables <= 100
        min_nodes, max_nodes = 5, 16
        target_density = 0.4
        cap_range = (20.0, 150.0)
        cost_range = (1.0, 12.0)
    elseif target_variables <= 500
        min_nodes, max_nodes = 12, 40
        target_density = 0.2
        cap_range = (40.0, 600.0)
        cost_range = (1.0, 30.0)
    else
        min_nodes, max_nodes = 25, 110
        target_density = 0.1
        cap_range = (100.0, 2500.0)
        cost_range = (1.0, 60.0)
    end

    # --- Choose node count so #arcs ≈ target_variables ---
    # A network on n nodes restricted to forward arcs has n*(n-1)/2 candidates.
    n_nodes = min_nodes
    for n in min_nodes:max_nodes
        possible_arcs = n * (n - 1) ÷ 2
        if round(Int, possible_arcs * target_density) >= round(Int, target_variables * 0.95)
            n_nodes = n
            break
        end
        n_nodes = n
    end
    n_nodes = max(n_nodes, 4)

    source_node = 1
    sink_node = n_nodes

    # --- Topology (backbone path is known explicitly) ---
    arcs, backbone = _generalized_flow_topology(n_nodes, target_variables)

    # --- Gains in (0.85, 1.0]: lossy arcs, never amplifying (no unbounded cycles) ---
    gains = Dict{Tuple{Int,Int},Float64}()
    for arc in arcs
        gains[arc] = round(0.85 + 0.15 * rand(), digits=4)
    end

    # --- Costs ---
    cmin, cmax = cost_range
    costs = Dict{Tuple{Int,Int},Float64}()
    for arc in arcs
        costs[arc] = round(cmin + (cmax - cmin) * rand(), digits=3)
    end

    # --- Baseline capacities (sampled; refined below per feasibility intent) ---
    capmin, capmax = cap_range
    capacities = Dict{Tuple{Int,Int},Float64}()
    for arc in arcs
        capacities[arc] = round(capmin + (capmax - capmin) * rand(), digits=2)
    end

    # Product of gains along the backbone (always > 0, <= 1).
    backbone_gain_product = prod(gains[a] for a in backbone)

    sink_in_arcs = [arc for arc in arcs if arc[2] == sink_node]

    # A natural demand scale: roughly what the backbone can comfortably deliver.
    # The smallest backbone capacity bounds how much can be pushed end-to-end.
    min_backbone_cap = minimum(capacities[a] for a in backbone)
    nominal_deliverable = min_backbone_cap * backbone_gain_product

    # Resolve feasibility intent.
    status = feasibility_status

    if status == feasible
        # Pick a modest demand, then guarantee the backbone can deliver it.
        demand = round(nominal_deliverable * (0.4 + 0.4 * rand()), digits=2)
        demand = max(demand, capmin)  # keep it meaningfully positive

        # Sent amount on the backbone to deliver `demand`: s = demand / P.
        slack = 1.25
        send_amount = demand / backbone_gain_product * slack
        for a in backbone
            if capacities[a] < send_amount
                capacities[a] = round(send_amount, digits=2)
            end
        end
        source_supply = round(send_amount * 1.5, digits=2)

    elseif status == infeasible
        # Cap the post-gain inflow into the sink strictly below demand.
        # Choose demand first (any positive scale), then set sink in-arc caps so
        # sum(g * cap) = demand * alpha with alpha < 1.
        demand = round(max(nominal_deliverable, capmin) * (0.5 + 0.5 * rand()), digits=2)
        alpha = 0.7 + 0.2 * rand()  # 0.7 .. 0.9
        target_inflow_cap = demand * alpha

        if isempty(sink_in_arcs)
            # Backbone always provides at least one sink in-arc, so this should not
            # happen; guard defensively by forcing a single tiny in-arc cap.
            sink_in_arcs = [backbone[end]]
        end

        # Distribute the allowed post-gain inflow budget across sink in-arcs.
        weights = [rand() for _ in sink_in_arcs]
        wsum = sum(weights)
        for (k, arc) in enumerate(sink_in_arcs)
            share = (weights[k] / wsum) * target_inflow_cap
            # cap chosen so g * cap == share  =>  cap = share / g
            capacities[arc] = round(share / gains[arc], digits=4)
        end
        # Source supply is generous; the binding constraint is the sink inflow cap.
        source_supply = round(demand / backbone_gain_product * 2.0, digits=2)

    else  # unknown: natural instance, biased feasible (backbone sized to deliver).
        demand = round(nominal_deliverable * (0.3 + 0.3 * rand()), digits=2)
        demand = max(demand, capmin)
        send_amount = demand / backbone_gain_product * 1.1
        for a in backbone
            if capacities[a] < send_amount
                capacities[a] = round(send_amount, digits=2)
            end
        end
        source_supply = round(send_amount * 1.4, digits=2)
    end

    return GeneralizedFlowProblem(
        n_nodes, source_node, sink_node,
        arcs, backbone,
        capacities, costs, gains,
        source_supply, demand,
    )
end

"""
    build_model(prob::GeneralizedFlowProblem)

Build the JuMP model for the generalized (lossy) network-flow problem.
Deterministic — uses only data from the struct fields.

Decision variables:
- `f[arc] >= 0`: flow *sent* on each arc (post-gain arrival at the head is
  `g[arc] * f[arc]`).

# Returns
- `model`: The JuMP model
"""
function build_model(prob::GeneralizedFlowProblem)
    model = Model()

    arcs = prob.arcs
    src = prob.source_node
    snk = prob.sink_node

    # One nonnegative flow variable per arc (total = length(arcs)).
    @variable(model, f[arc in arcs] >= 0)

    # Per-arc capacities (also keep the LP bounded).
    for arc in arcs
        @constraint(model, f[arc] <= prob.capacities[arc])
    end

    # Source supply cap on total sent flow out of the source.
    src_out = [arc for arc in arcs if arc[1] == src]
    if !isempty(src_out)
        @constraint(model, sum(f[arc] for arc in src_out) <= prob.source_supply)
    end

    # Generalized (multiplicative) conservation at intermediate nodes:
    #   sum_in g*f  ==  sum_out f
    for v in 1:prob.n_nodes
        v == src && continue
        v == snk && continue
        in_arcs = [arc for arc in arcs if arc[2] == v]
        out_arcs = [arc for arc in arcs if arc[1] == v]
        (isempty(in_arcs) && isempty(out_arcs)) && continue
        inflow = isempty(in_arcs) ? 0.0 : sum(prob.gains[arc] * f[arc] for arc in in_arcs)
        outflow = isempty(out_arcs) ? 0.0 : sum(f[arc] for arc in out_arcs)
        @constraint(model, inflow == outflow)
    end

    # Delivered at sink = post-gain inflow on sink in-arcs; must meet demand.
    sink_in = [arc for arc in arcs if arc[2] == snk]
    if !isempty(sink_in)
        @constraint(model, sum(prob.gains[arc] * f[arc] for arc in sink_in) >= prob.demand)
    end

    # Objective: minimize total routing cost over sent flow.
    @objective(model, Min, sum(prob.costs[arc] * f[arc] for arc in arcs))

    return model
end

# Register the variant
register_variant(
    :network_flow,
    :generalized_flow,
    GeneralizedFlowProblem,
    "Generalized (lossy) min-cost network flow with per-arc gain multipliers in (0,1] delivering a required amount at the sink under multiplicative conservation",
)
