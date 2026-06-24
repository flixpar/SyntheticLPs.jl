using JuMP
using Random
using Distributions

"""
    TimeExpandedServiceNetworkProblem <: ProblemGenerator

Time-expanded service-network LP with loaded commodity flows, holding arcs,
scheduled service arcs, shared arc capacities, and unmet-demand penalties.
"""
struct TimeExpandedServiceNetworkProblem <: ProblemGenerator
    n_locations::Int
    n_periods::Int
    n_commodities::Int
    arcs::Vector{Tuple{Int,Int,Int,Int}}
    arc_cost::Vector{Float64}
    arc_capacity::Vector{Float64}
    commodity_origin::Vector{Int}
    commodity_destination::Vector{Int}
    commodity_release::Vector{Int}
    commodity_due::Vector{Int}
    demand::Vector{Float64}
    unmet_penalty::Vector{Float64}
    max_unmet::Vector{Float64}
end

function _choose_service_network_dims(target_variables::Int)
    best = (Inf, 3, 4, 2, 0)
    for loc in 3:15, periods in 4:16, commodities in 2:30
        service_arcs = (periods - 1) * (loc + max(loc, round(Int, 0.35 * loc * (loc - 1))))
        vars = service_arcs * commodities + commodities
        err = abs(vars - target_variables)
        if err < best[1]
            best = (err, loc, periods, commodities, service_arcs)
        end
    end
    return best[2], best[3], best[4]
end

function TimeExpandedServiceNetworkProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)
    n_locations, n_periods, n_commodities = _choose_service_network_dims(target_variables)
    x = rand(Uniform(0.0, 100.0), n_locations)
    y = rand(Uniform(0.0, 100.0), n_locations)

    arcs = Tuple{Int,Int,Int,Int}[]
    arc_cost = Float64[]
    arc_capacity = Float64[]
    for t in 1:(n_periods-1)
        for i in 1:n_locations
            push!(arcs, (i, i, t, t + 1))
            push!(arc_cost, rand(Uniform(0.05, 0.35)))
            push!(arc_capacity, rand(Uniform(80.0, 220.0)))
        end
        for i in 1:n_locations
            distances = [(j, sqrt((x[i]-x[j])^2 + (y[i]-y[j])^2)) for j in 1:n_locations if j != i]
            sort!(distances, by=z -> z[2])
            keep = min(length(distances), max(1, round(Int, 0.35 * (n_locations - 1))))
            for (j, dist) in distances[1:keep]
                transit = rand() < 0.75 ? 1 : 2
                if t + transit <= n_periods
                    push!(arcs, (i, j, t, t + transit))
                    push!(arc_cost, 0.18 * dist * rand(Uniform(0.75, 1.35)) + rand(Uniform(1.0, 4.0)))
                    push!(arc_capacity, rand(Uniform(25.0, 120.0)))
                end
            end
        end
    end

    origin = Int[]; destination = Int[]; release = Int[]; due = Int[]; demand = Float64[]
    for _ in 1:n_commodities
        o = rand(1:n_locations)
        d = rand(setdiff(1:n_locations, [o]))
        r = rand(1:max(1, n_periods - 2))
        dd = rand((r+1):n_periods)
        push!(origin, o); push!(destination, d); push!(release, r); push!(due, dd)
        push!(demand, rand(Uniform(8.0, 45.0)))
    end
    if feasibility_status == infeasible
        demand .*= rand(Uniform(4.0, 8.0))
    elseif feasibility_status == feasible
        arc_capacity .*= rand(Uniform(1.4, 2.4))
    end
    unmet_penalty = rand(Uniform(250.0, 700.0), n_commodities)
    max_unmet = fill(Inf, n_commodities)
    if feasibility_status == infeasible
        max_unmet .= 0.0
    end

    return TimeExpandedServiceNetworkProblem(n_locations, n_periods, n_commodities,
        arcs, arc_cost, arc_capacity, origin, destination, release, due, demand, unmet_penalty, max_unmet)
end

function build_model(prob::TimeExpandedServiceNetworkProblem)
    model = Model()
    A, K = length(prob.arcs), prob.n_commodities
    @variable(model, flow[1:A, 1:K] >= 0)
    @variable(model, 0 <= unmet[k=1:K] <= prob.max_unmet[k])
    @objective(model, Min,
        sum(prob.arc_cost[a] * flow[a,k] for a in 1:A, k in 1:K) +
        sum(prob.unmet_penalty[k] * unmet[k] for k in 1:K))
    for a in 1:A
        @constraint(model, sum(flow[a,k] for k in 1:K) <= prob.arc_capacity[a])
    end
    for k in 1:K, loc in 1:prob.n_locations, t in 1:prob.n_periods
        rhs = 0.0
        if loc == prob.commodity_origin[k] && t == prob.commodity_release[k]
            rhs += prob.demand[k]
        end
        if loc == prob.commodity_destination[k] && t == prob.commodity_due[k]
            rhs -= prob.demand[k]
        end
        outgoing = sum(flow[a,k] for a in 1:A if prob.arcs[a][1] == loc && prob.arcs[a][3] == t)
        incoming = sum(flow[a,k] for a in 1:A if prob.arcs[a][2] == loc && prob.arcs[a][4] == t)
        if loc == prob.commodity_destination[k] && t == prob.commodity_due[k]
            @constraint(model, outgoing - incoming - unmet[k] == rhs)
        else
            @constraint(model, outgoing - incoming == rhs)
        end
    end
    return model
end

register_variant(:service_network_design, :time_expanded, TimeExpandedServiceNetworkProblem,
    "Time-expanded service-network LP with scheduled arcs, holding arcs, shared capacities, commodities, and unmet-demand penalties.")
