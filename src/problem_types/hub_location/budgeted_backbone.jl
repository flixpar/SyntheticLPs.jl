using JuMP
using Random
using Distributions

"""Planted exact-p hub assignment and complete selected-hub backbone."""
struct BudgetedBackboneWitness
    open_hubs::Vector{Int}
    assignment::Vector{Int}
    links::Vector{Tuple{Int,Int}}
end

"""Degree-implied LP lower bound that exceeds the available link budget."""
struct LinkBudgetCertificate
    implied_minimum::Float64
    budget::Float64
end

"""
    BudgetedBackboneHubProblem <: ProblemGenerator

Exact-p single-allocation hub location with a complete candidate link graph,
binary physical-link installation, a link-investment budget, shared
both-direction capacities, and origin-indexed flows that may traverse multiple
installed links.
"""
struct BudgetedBackboneHubProblem <: ProblemGenerator
    n_nodes::Int
    hubs::Vector{Int}
    p::Int
    chi::Float64
    alpha::Float64
    delta::Float64
    locations::Vector{Tuple{Float64,Float64}}
    dist::Matrix{Float64}
    flow::Matrix{Float64}
    outvolume::Vector{Float64}
    involume::Vector{Float64}
    fixed_cost::Vector{Float64}
    link_cost::Matrix{Float64}
    link_capacity::Matrix{Float64}
    link_budget::Float64
    feasible_witness::Union{Nothing,BudgetedBackboneWitness}
    infeasibility_certificate::Union{Nothing,LinkBudgetCertificate}
    feasibility_status::FeasibilityStatus
end

function _budgeted_backbone_dimensions(target_variables::Int)
    target = max(target_variables, 1)
    min_h = target >= 40 ? 3 : 2
    best = (n=max(3, min_h), h=min_h)
    best_gap = Inf
    max_h = max(min_h, ceil(Int, cbrt(2 * target)) + 2)
    for h in min_h:max_h
        fixed = h + h * (h - 1) ÷ 2
        n_estimate = round(Int, (target - fixed) / h^2)
        for n in max(3, h, n_estimate - 2):max(3, h, n_estimate + 2)
            n > 5 * h && continue
            total = n * h^2 + fixed
            if abs(total - target) < best_gap
                best, best_gap = (n=n, h=h), abs(total - target)
            end
        end
    end
    return best
end

function BudgetedBackboneHubProblem(target_variables::Int,
                                    feasibility_status::FeasibilityStatus,
                                    seed::Int)
    rng = MersenneTwister(seed)
    dims = _budgeted_backbone_dimensions(target_variables)
    n, h = dims.n, dims.h
    locations = _hub_city_locations(rng, n, rand(rng, (:clustered, :corridor, :archipelago)))
    dist = _hub_distance_matrix(locations)
    populations = _hub_populations(rng, n)
    flow = _hub_gravity_flows(
        rng, n, populations, dist, rand(rng, Uniform(0.5, 1.0)),
        rand(rng, Uniform(0.6, 1.0)); symmetric=false,
        scale=rand(rng, Uniform(0.8, 3.0)),
    )
    outvolume = vec(sum(flow; dims=2))
    involume = vec(sum(flow; dims=1))
    hubs = _hub_candidate_sites(rng, n, outvolume .+ involume, h)
    p = clamp(round(Int, rand(rng, Uniform(0.55, 0.82)) * h), min(3, h), h)
    chi = delta = rand(rng, Uniform(1.0, 2.5))
    alpha = rand(rng, Uniform(0.05, 0.4))

    total_flow = sum(outvolume)
    average_distance = sum(dist) / max(n^2 - n, 1)
    fixed_cost = [total_flow * average_distance * rand(rng, Uniform(0.015, 0.055))
                  for _ in 1:h]
    link_cost = zeros(Float64, h, h)
    link_capacity = zeros(Float64, h, h)
    for k in 1:h, m in (k + 1):h
        d = dist[hubs[k], hubs[m]]
        cost = total_flow * (0.006 + 0.018 * d / max(average_distance, 1.0)) *
               rand(rng, Uniform(0.85, 1.15))
        link_cost[k, m] = link_cost[m, k] = round(cost; digits=3)
        capacity = total_flow * rand(rng, Uniform(0.35, 0.75))
        link_capacity[k, m] = link_capacity[m, k] = round(capacity; digits=3)
    end

    witness = nothing
    certificate = nothing
    min_link_cost = minimum(link_cost[k, m] for k in 1:h for m in (k + 1):h)
    if feasibility_status == feasible
        chosen = sort(sortperm(fixed_cost)[1:p])
        assignment = zeros(Int, n)
        candidate_position = Dict(node => k for (k, node) in enumerate(hubs))
        for i in 1:n
            own = get(candidate_position, i, 0)
            assignment[i] = own in chosen ? own :
                chosen[argmin([dist[i, hubs[k]] for k in chosen])]
        end
        links = [(k, m) for (pos, k) in enumerate(chosen)
                 for m in chosen[(pos + 1):end]]
        for (k, m) in links
            required = sum(flow[i, j] for i in 1:n, j in 1:n
                           if (assignment[i] == k && assignment[j] == m) ||
                              (assignment[i] == m && assignment[j] == k);
                           init=0.0)
            link_capacity[k, m] = link_capacity[m, k] =
                max(link_capacity[k, m], required * rand(rng, Uniform(1.08, 1.25)))
        end
        link_budget = sum(link_cost[k, m] for (k, m) in links) *
                      rand(rng, Uniform(1.03, 1.15))
        witness = BudgetedBackboneWitness(chosen, assignment, links)
    elseif feasibility_status == infeasible
        implied_minimum = min_link_cost * p / 2
        link_budget = implied_minimum * rand(rng, Uniform(0.55, 0.82))
        certificate = LinkBudgetCertificate(implied_minimum, link_budget)
    else
        link_budget = min_link_cost * p * rand(rng, Uniform(0.8, 1.8))
    end

    return BudgetedBackboneHubProblem(
        n, hubs, p, chi, alpha, delta, locations, dist, flow, outvolume,
        involume, fixed_cost, link_cost, link_capacity, link_budget, witness,
        certificate, feasibility_status,
    )
end

function build_model(prob::BudgetedBackboneHubProblem)
    model = Model()
    n, h = prob.n_nodes, length(prob.hubs)
    @variable(model, y[1:h], Bin)
    @variable(model, z[1:n, 1:h], Bin)
    @variable(model, b[k=1:h, m=1:h; k < m], Bin)
    @variable(model, q[i=1:n, k=1:h, m=1:h; k != m] >= 0)

    @constraint(model, single_allocation[i=1:n], sum(z[i, k] for k in 1:h) == 1)
    @constraint(model, allocation_open[i=1:n, k=1:h], z[i, k] <= y[k])
    @constraint(model, candidate_self_allocation[k=1:h], z[prob.hubs[k], k] == y[k])
    @constraint(model, hub_count, sum(y) == prob.p)
    @constraint(model, link_tail_open[k=1:h, m=1:h; k < m], b[k, m] <= y[k])
    @constraint(model, link_head_open[k=1:h, m=1:h; k < m], b[k, m] <= y[m])
    @constraint(model, active_hub_degree[k=1:h],
        sum(b[min(k, m), max(k, m)] for m in 1:h if m != k) >= y[k])
    @constraint(model, link_budget,
        sum(prob.link_cost[k, m] * b[k, m] for k in 1:h, m in (k + 1):h) <=
        prob.link_budget)
    @constraint(model, flow_balance[i=1:n, k=1:h],
        sum(q[i, k, m] for m in 1:h if m != k) -
        sum(q[i, m, k] for m in 1:h if m != k) ==
        prob.outvolume[i] * z[i, k] -
        sum(prob.flow[i, j] * z[j, k] for j in 1:n))
    @constraint(model, physical_link_capacity[k=1:h, m=1:h; k < m],
        sum(q[i, k, m] + q[i, m, k] for i in 1:n) <=
        prob.link_capacity[k, m] * b[k, m])

    @objective(model, Min,
        sum(prob.fixed_cost[k] * y[k] for k in 1:h) +
        sum(prob.link_cost[k, m] * b[k, m] for k in 1:h, m in (k + 1):h) +
        sum(prob.chi * prob.dist[i, prob.hubs[k]] * prob.outvolume[i] * z[i, k]
            for i in 1:n, k in 1:h) +
        sum(prob.delta * prob.dist[prob.hubs[k], j] * prob.involume[j] * z[j, k]
            for j in 1:n, k in 1:h) +
        sum(prob.alpha * prob.dist[prob.hubs[k], prob.hubs[m]] * q[i, k, m]
            for i in 1:n, k in 1:h, m in 1:h if k != m))
    return model
end

register_variant(
    :hub_location,
    :budgeted_backbone,
    BudgetedBackboneHubProblem,
    "Exact-p single-allocation hub location with budgeted capacitated physical links over a complete candidate backbone",
)
