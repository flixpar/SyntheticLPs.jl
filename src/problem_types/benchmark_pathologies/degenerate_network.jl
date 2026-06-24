using JuMP
using Random
using Distributions

"""
    DegenerateNetworkProblem <: ProblemGenerator

Layered network-flow LP with many parallel, nearly equal-cost alternatives. It
creates large optimal faces and highly degenerate bases.
"""
struct DegenerateNetworkProblem <: ProblemGenerator
    n_layers::Int
    width::Int
    arcs::Vector{Tuple{Int,Int}}
    cost::Vector{Float64}
    capacity::Vector{Float64}
    supply::Vector{Float64}
end

function DegenerateNetworkProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)
    best = (Inf, 4, 3)
    for layers in 4:20, width in 2:25
        arcs = (layers - 1) * width^2
        err = abs(arcs - target_variables)
        if err < best[1]
            best = (err, layers, width)
        end
    end
    n_layers, width = best[2], best[3]
    node(layer, pos) = (layer - 1) * width + pos
    arcs = Tuple{Int,Int}[]
    cost = Float64[]
    capacity = Float64[]
    flow_value = rand(Uniform(50.0, 200.0))
    for l in 1:(n_layers-1), u in 1:width, v in 1:width
        push!(arcs, (node(l,u), node(l+1,v)))
        push!(cost, 1.0 + 1e-5 * randn())
        cap_mult = feasibility_status == infeasible && l == 1 ? rand(Uniform(0.1, 0.6)) : rand(Uniform(0.8, 2.0))
        push!(capacity, cap_mult * flow_value / width)
    end
    n_nodes = n_layers * width
    supply = zeros(Float64, n_nodes)
    for u in 1:width
        supply[node(1,u)] = flow_value / width
        supply[node(n_layers,u)] = -flow_value / width
    end
    return DegenerateNetworkProblem(n_layers, width, arcs, cost, capacity, supply)
end

function build_model(prob::DegenerateNetworkProblem)
    model = Model()
    n_arcs = length(prob.arcs)
    n_nodes = prob.n_layers * prob.width
    @variable(model, 0 <= x[a=1:n_arcs] <= prob.capacity[a])
    @objective(model, Min, sum(prob.cost[a] * x[a] for a in 1:n_arcs))
    for v in 1:n_nodes
        @constraint(model,
            sum(x[a] for a in 1:n_arcs if prob.arcs[a][1] == v) -
            sum(x[a] for a in 1:n_arcs if prob.arcs[a][2] == v) == prob.supply[v])
    end
    return model
end

register_variant(:benchmark_pathologies, :degenerate_network, DegenerateNetworkProblem,
    "Layered network-flow LP with many parallel nearly tied routes, zero slacks, and degenerate optima.")
