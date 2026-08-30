using JuMP
using Random
using Distributions

"""All-candidate opening witness for a feasible hub set-covering instance."""
struct HubCoveringWitness
    open_hubs::Vector{Int}
    threshold::Float64
end

"""An OD pair whose cheapest two-hub path exceeds the service threshold."""
struct HubCoveringCertificate
    origin::Int
    destination::Int
    minimum_route_cost::Float64
    threshold::Float64
end

"""
    HubSetCoveringProblem <: ProblemGenerator

Multiple-allocation hub set-covering location problem.  Every ordered OD pair
must select a collection/transfer/distribution path whose generalized cost is
within `service_threshold`; the objective minimizes hub opening cost.
"""
struct HubSetCoveringProblem <: ProblemGenerator
    n_nodes::Int
    profile::Symbol
    chi::Float64
    alpha::Float64
    delta::Float64
    locations::Vector{Tuple{Float64,Float64}}
    dist::Matrix{Float64}
    fixed_cost::Vector{Float64}
    service_threshold::Float64
    covering_sets::Dict{Tuple{Int,Int},Vector{Tuple{Int,Int}}}
    feasible_witness::Union{Nothing,HubCoveringWitness}
    infeasibility_certificate::Union{Nothing,HubCoveringCertificate}
    feasibility_status::FeasibilityStatus
end

_hub_covering_variable_count(prob::HubSetCoveringProblem) =
    prob.n_nodes + sum(length, values(prob.covering_sets); init=0)

function _build_hub_covering(n::Int, target_variables::Int,
                             feasibility_status::FeasibilityStatus,
                             rng::AbstractRNG)
    profile = rand(rng, (:passenger, :freight, :express))
    shape = rand(rng, (:clustered, :corridor, :archipelago))
    locations = _hub_city_locations(rng, n, shape)
    dist = _hub_distance_matrix(locations)
    if profile == :freight
        chi, alpha, delta = 3.0, 0.75, 2.0
    elseif profile == :passenger
        chi, alpha, delta = 1.0, rand(rng, (0.2, 0.4, 0.6, 0.8)), 1.0
    else
        chi = rand(rng, Uniform(1.2, 2.0))
        alpha = rand(rng, Uniform(0.35, 0.65))
        delta = rand(rng, Uniform(1.2, 2.0))
    end

    mean_dist = sum(dist) / max(n^2 - n, 1)
    base_fixed = mean_dist * n * rand(rng, Uniform(1.5, 4.0))
    fixed_cost = [base_fixed * rand(rng, Uniform(0.7, 1.3)) for _ in 1:n]

    min_route = fill(Inf, n, n)
    route_costs = Float64[]
    sizehint!(route_costs, n^3 * (n - 1))
    route_cost(i, j, k, m) =
        chi * dist[i, k] + alpha * dist[k, m] + delta * dist[m, j]
    for i in 1:n, j in 1:n
        i == j && continue
        for k in 1:n, m in 1:n
            c = route_cost(i, j, k, m)
            push!(route_costs, c)
            min_route[i, j] = min(min_route[i, j], c)
        end
    end
    sort!(route_costs)
    desired = max(target_variables - n, 0)
    worst_minimum = maximum(min_route[i, j] for i in 1:n for j in 1:n if i != j)

    if feasibility_status == feasible
        minimum_rank = searchsortedlast(route_costs, worst_minimum)
        rank = clamp(desired, minimum_rank, length(route_costs))
        threshold = route_costs[rank]
    elseif feasibility_status == infeasible
        maximum_rank = searchsortedfirst(route_costs, worst_minimum) - 1
        rank = clamp(desired, 0, maximum_rank)
        threshold = rank == 0 ? prevfloat(first(route_costs)) : route_costs[rank]
        threshold < worst_minimum || (threshold = prevfloat(worst_minimum))
    else
        rank = clamp(desired, 0, length(route_costs))
        threshold = rank == 0 ? prevfloat(first(route_costs)) : route_costs[rank]
    end

    covering_sets = Dict{Tuple{Int,Int},Vector{Tuple{Int,Int}}}()
    for i in 1:n, j in 1:n
        i == j && continue
        paths = Tuple{Int,Int}[]
        for k in 1:n, m in 1:n
            route_cost(i, j, k, m) <= threshold && push!(paths, (k, m))
        end
        covering_sets[(i, j)] = paths
    end

    witness = feasibility_status == feasible ?
              HubCoveringWitness(collect(1:n), threshold) : nothing
    certificate = nothing
    if feasibility_status == infeasible
        uncovered = nothing
        for od in keys(covering_sets)
            if isempty(covering_sets[od])
                uncovered = od
                break
            end
        end
        uncovered === nothing && error("Failed to construct an uncovered OD pair.")
        i, j = uncovered
        certificate = HubCoveringCertificate(i, j, min_route[i, j], threshold)
    end
    return HubSetCoveringProblem(
        n, profile, chi, alpha, delta, locations, dist, fixed_cost, threshold,
        covering_sets, witness, certificate, feasibility_status,
    )
end

function HubSetCoveringProblem(target_variables::Int,
                               feasibility_status::FeasibilityStatus, seed::Int)
    target = max(target_variables, 1)
    hint = clamp(round(Int, 1.25 * target^0.25), 4, 45)
    best = nothing
    best_score = (1, Inf)
    for attempt in 1:16
        rng = MersenneTwister(seed + 67867967 * attempt)
        candidate = _build_hub_covering(hint, target, feasibility_status, rng)
        total = _hub_covering_variable_count(candidate)
        gap = abs(total - target) / target
        score = (gap <= 0.25 || total <= 50 ? 0 : 1, gap)
        if score < best_score
            best, best_score = candidate, score
        end
        gap <= 0.03 && break
        ratio = clamp((target / max(total, 1))^0.25, 0.65, 1.5)
        next_hint = round(Int, hint * ratio)
        next_hint == hint && (next_hint += total < target ? 1 : -1)
        hint = clamp(next_hint, 4, 45)
    end
    return best::HubSetCoveringProblem
end

function build_model(prob::HubSetCoveringProblem)
    model = Model()
    n = prob.n_nodes
    routes = NTuple{4,Int}[]
    for ((i, j), paths) in prob.covering_sets, (k, m) in paths
        push!(routes, (i, j, k, m))
    end
    @variable(model, y[1:n], Bin)
    @variable(model, 0 <= x[routes] <= 1)
    @objective(model, Min, sum(prob.fixed_cost[k] * y[k] for k in 1:n))

    for ((i, j), paths) in prob.covering_sets
        if isempty(paths)
            @constraint(model, 0 >= 1)
        else
            @constraint(model, sum(x[(i, j, k, m)] for (k, m) in paths) >= 1)
            for (k, m) in paths
                @constraint(model, x[(i, j, k, m)] <= y[k])
                @constraint(model, x[(i, j, k, m)] <= y[m])
            end
        end
    end
    return model
end

register_variant(
    :hub_location,
    :hub_covering,
    HubSetCoveringProblem,
    "Multiple-allocation hub set covering: minimize opening cost while every ordered OD pair meets a two-hub service threshold",
)
