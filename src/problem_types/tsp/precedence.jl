using JuMP
using Random

"""
    TSPPrecedenceProblem <: ProblemGenerator

TSP with precedence constraints for pickup-before-delivery, inspection-before-
repair, and other ordered field-service tasks. A lifted MTZ formulation uses the
same order variables for connectivity and task precedence.
"""
struct TSPPrecedenceProblem <: ProblemGenerator
    n_stops::Int
    locations::Vector{Tuple{Float64,Float64}}
    dist::Matrix{Float64}
    precedence_pairs::Vector{Tuple{Int,Int}}
end

function TSPPrecedenceProblem(target_variables::Int,
                              feasibility_status::FeasibilityStatus,
                              seed::Int)
    Random.seed!(seed)
    n = max(5, round(Int, sqrt(target_variables + 1)))
    locations = _tsp_stops(n)
    dist = _tsp_distance(locations)

    pairs = Tuple{Int,Int}[]
    if feasibility_status == infeasible
        cycle_nodes = shuffle(collect(2:n))[1:3]
        push!(pairs, (cycle_nodes[1], cycle_nodes[2]))
        push!(pairs, (cycle_nodes[2], cycle_nodes[3]))
        push!(pairs, (cycle_nodes[3], cycle_nodes[1]))
    else
        witness_order = shuffle(collect(2:n))
        candidates = [(witness_order[a], witness_order[b])
                      for a in 1:length(witness_order)-1
                      for b in a+1:length(witness_order)]
        density = feasibility_status == feasible ? 0.25 : 0.15 + 0.35 * rand()
        n_pairs = clamp(round(Int, density * (n - 1)), 1, length(candidates))
        pairs = shuffle(candidates)[1:n_pairs]
    end

    return TSPPrecedenceProblem(n, locations, dist, pairs)
end

function build_model(prob::TSPPrecedenceProblem)
    model = Model()
    n = prob.n_stops
    nodes = 1:n
    stops = 2:n

    @variable(model, x[i in nodes, j in nodes; i != j], Bin)
    @variable(model, 1 <= u[j in stops] <= n - 1)
    @objective(model, Min,
        sum(prob.dist[i, j] * x[i, j] for i in nodes, j in nodes if i != j))

    for j in nodes
        @constraint(model, sum(x[i, j] for i in nodes if i != j) == 1)
        @constraint(model, sum(x[j, k] for k in nodes if k != j) == 1)
    end
    for i in stops, j in stops
        i == j && continue
        @constraint(model,
            u[i] - u[j] + (n - 1) * x[i, j] + (n - 3) * x[j, i] <= n - 2)
    end
    for (before, after) in prob.precedence_pairs
        @constraint(model, u[before] + 1 <= u[after])
    end

    return model
end

register_variant(
    :tsp,
    :precedence,
    TSPPrecedenceProblem,
    "Precedence-constrained TSP for ordered field-service tasks using lifted MTZ sequencing",
)
