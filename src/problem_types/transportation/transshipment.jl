using JuMP
using Random
using Distributions

"""
    Transshipment <: ProblemGenerator

Generator for transshipment problems with intermediate storage and routing.

This problem extends transportation by allowing intermediate transshipment nodes
that can receive, store, and forward goods between sources and destinations.

# Fields
- `n_sources::Int`: Number of supply sources
- `n_transshipment::Int`: Number of transshipment nodes
- `n_destinations::Int`: Number of demand destinations
- `supplies::Vector{Float64}`: Supply at each source
- `demands::Vector{Float64}`: Demand at each destination
- `transship_capacities::Vector{Float64}`: Storage capacity at each transshipment node
- `source_costs::Matrix{Float64}`: Cost from source to transshipment node
- `transship_costs::Matrix{Float64}`: Cost between transshipment nodes
- `destination_costs::Matrix{Float64}`: Cost from transshipment to destination
- `holding_costs::Vector{Float64}`: Per-unit holding cost at transshipment nodes
- `direct_source_dest_costs::Matrix{Float64}`: Direct shipping cost (bypassing transshipment)
"""
struct Transshipment <: ProblemGenerator
    n_sources::Int
    n_transshipment::Int
    n_destinations::Int
    supplies::Vector{Float64}
    demands::Vector{Float64}
    transship_capacities::Vector{Float64}
    source_costs::Matrix{Float64}
    transship_costs::Matrix{Float64}
    destination_costs::Matrix{Float64}
    holding_costs::Vector{Float64}
    direct_source_dest_costs::Matrix{Float64}
end

"""
    Transshipment(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a transshipment problem instance.

# Arguments
- `target_variables`: Target number of variables
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility

# Details
Variables: flows on all arcs (source→transship, transship→transship, transship→dest, source→dest)
Target: n_sources×n_trans + n_trans² + n_trans×n_dest + n_sources×n_dest
"""
function Transshipment(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Determine scale
    if target_variables <= 150
        min_sources, max_sources = 2, 10
        min_trans, max_trans = 2, 8
        min_dest, max_dest = 2, 12
        supply_range = (50.0, 500.0)
        demand_range = (30.0, 300.0)
        cost_range = (5.0, 50.0)
    elseif target_variables <= 800
        min_sources, max_sources = 3, 20
        min_trans, max_trans = 3, 15
        min_dest, max_dest = 3, 25
        supply_range = (100.0, 1000.0)
        demand_range = (80.0, 800.0)
        cost_range = (10.0, 100.0)
    else
        min_sources, max_sources = 5, 40
        min_trans, max_trans = 5, 30
        min_dest, max_dest = 5, 50
        supply_range = (200.0, 3000.0)
        demand_range = (150.0, 2000.0)
        cost_range = (20.0, 200.0)
    end

    # Solve for dimensions
    # target_vars ≈ n_s×n_t + n_t² + n_t×n_d + n_s×n_d
    best_config = (min_sources, min_trans, min_dest)
    best_error = Inf

    for n_trans in min_trans:max_trans
        for n_sources in min_sources:max_sources
            # Given n_trans and n_sources, solve for n_dest
            # target = n_s×n_t + n_t² + n_t×n_d + n_s×n_d
            # target = n_t×(n_s + n_t + n_d) + n_s×n_d
            # This is complex, so we use heuristic
            target_remaining = target_variables - n_sources * n_trans - n_trans * n_trans
            if target_remaining <= 0
                continue
            end

            # target_remaining ≈ n_t×n_d + n_s×n_d = n_d×(n_t + n_s)
            n_dest_approx = round(Int, target_remaining / (n_trans + n_sources))
            n_dest = clamp(n_dest_approx, min_dest, max_dest)

            actual_vars = n_sources * n_trans + n_trans * n_trans + n_trans * n_dest + n_sources * n_dest
            error = abs(actual_vars - target_variables) / target_variables

            if error < best_error
                best_error = error
                best_config = (n_sources, n_trans, n_dest)
            end
        end
    end

    n_sources, n_transshipment, n_destinations = best_config

    # Generate supplies and demands
    min_supply, max_supply = supply_range
    supplies = [rand(Uniform(min_supply, max_supply)) for _ in 1:n_sources]
    supplies = round.(supplies, digits=2)

    min_demand, max_demand = demand_range
    demands = [rand(Uniform(min_demand, max_demand)) for _ in 1:n_destinations]
    demands = round.(demands, digits=2)

    # Balance supply and demand
    total_supply = sum(supplies)
    total_demand = sum(demands)

    if total_supply < total_demand
        # Scale up supplies
        scale = (total_demand * 1.1) / total_supply
        supplies .*= scale
        supplies = round.(supplies, digits=2)
    end

    # Transshipment capacities (storage limits)
    total_demand = sum(demands)
    avg_trans_capacity = (total_demand / n_transshipment) * rand(1.2:0.1:2.0)
    transship_capacities = [round(avg_trans_capacity * (0.8 + 0.4 * rand()), digits=2)
                            for _ in 1:n_transshipment]

    # Generate costs
    min_cost, max_cost = cost_range

    # Source to transshipment costs
    source_costs = [round(rand(Uniform(min_cost, max_cost)), digits=2)
                    for _ in 1:n_sources, _ in 1:n_transshipment]

    # Transshipment to transshipment costs (usually cheaper due to consolidation)
    transship_cost_discount = rand(0.7:0.05:0.9)
    transship_costs = zeros(n_transshipment, n_transshipment)
    for i in 1:n_transshipment
        for j in 1:n_transshipment
            if i != j
                transship_costs[i, j] = round(rand(Uniform(min_cost, max_cost)) * transship_cost_discount, digits=2)
            end
        end
    end

    # Transshipment to destination costs
    destination_costs = [round(rand(Uniform(min_cost, max_cost)), digits=2)
                         for _ in 1:n_transshipment, _ in 1:n_destinations]

    # Direct source to destination costs (usually more expensive to encourage transshipment use)
    direct_cost_premium = rand(1.3:0.1:2.0)
    direct_source_dest_costs = [round(rand(Uniform(min_cost, max_cost)) * direct_cost_premium, digits=2)
                                for _ in 1:n_sources, _ in 1:n_destinations]

    # Holding costs at transshipment nodes
    holding_cost_range = (min_cost * 0.05, min_cost * 0.2)
    holding_costs = [round(rand(Uniform(holding_cost_range...)), digits=2)
                     for _ in 1:n_transshipment]

    # Adjust for feasibility
    if feasibility_status == feasible
        # Ensure sufficient transshipment capacity
        total_trans_capacity = sum(transship_capacities)
        if total_trans_capacity < total_demand
            scale = (total_demand * 1.2) / total_trans_capacity
            transship_capacities .*= scale
            transship_capacities = round.(transship_capacities, digits=2)
        end

        # Ensure balanced supply/demand
        total_supply = sum(supplies)
        total_demand = sum(demands)
        if total_supply < total_demand
            scale = (total_demand * 1.05) / total_supply
            supplies .*= scale
            supplies = round.(supplies, digits=2)
        end

    elseif feasibility_status == infeasible
        # Create infeasibility
        choice = rand()
        if choice < 0.4
            # Reduce transshipment capacity
            scale = rand(0.3:0.05:0.7)
            transship_capacities .*= scale
            transship_capacities = round.(transship_capacities, digits=2)
        elseif choice < 0.7
            # Reduce total supply
            scale = rand(0.6:0.05:0.9)
            supplies .*= scale
            supplies = round.(supplies, digits=2)
        else
            # Increase demand
            scale = rand(1.3:0.1:2.0)
            demands .*= scale
            demands = round.(demands, digits=2)
        end
    end

    return Transshipment(
        n_sources,
        n_transshipment,
        n_destinations,
        supplies,
        demands,
        transship_capacities,
        source_costs,
        transship_costs,
        destination_costs,
        holding_costs,
        direct_source_dest_costs
    )
end

"""
    build_model(prob::Transshipment)

Build a JuMP model for the transshipment problem.

# Arguments
- `prob`: Transshipment instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::Transshipment)
    model = Model()

    S = prob.n_sources
    T = prob.n_transshipment
    D = prob.n_destinations

    # Decision variables

    # f_st[s,t] = flow from source s to transshipment node t
    @variable(model, f_st[1:S, 1:T] >= 0)

    # f_tt[t1,t2] = flow between transshipment nodes
    @variable(model, f_tt[1:T, 1:T] >= 0)

    # f_td[t,d] = flow from transshipment node t to destination d
    @variable(model, f_td[1:T, 1:D] >= 0)

    # f_sd[s,d] = direct flow from source s to destination d
    @variable(model, f_sd[1:S, 1:D] >= 0)

    # Objective: minimize total cost
    @objective(model, Min,
        # Source to transshipment
        sum(prob.source_costs[s, t] * f_st[s, t] for s in 1:S, t in 1:T) +
        # Transshipment to transshipment
        sum(prob.transship_costs[t1, t2] * f_tt[t1, t2] for t1 in 1:T, t2 in 1:T if t1 != t2) +
        # Transshipment to destination
        sum(prob.destination_costs[t, d] * f_td[t, d] for t in 1:T, d in 1:D) +
        # Direct source to destination
        sum(prob.direct_source_dest_costs[s, d] * f_sd[s, d] for s in 1:S, d in 1:D) +
        # Holding costs (proportional to throughput)
        sum(prob.holding_costs[t] * sum(f_st[s, t] for s in 1:S) for t in 1:T)
    )

    # Constraints

    # Source capacity constraints
    for s in 1:S
        @constraint(model,
            sum(f_st[s, t] for t in 1:T) + sum(f_sd[s, d] for d in 1:D) <= prob.supplies[s]
        )
    end

    # Destination demand constraints
    for d in 1:D
        @constraint(model,
            sum(f_td[t, d] for t in 1:T) + sum(f_sd[s, d] for s in 1:S) >= prob.demands[d]
        )
    end

    # Transshipment node flow balance
    for t in 1:T
        # Inflow = outflow
        inflow = sum(f_st[s, t] for s in 1:S) + sum(f_tt[t2, t] for t2 in 1:T if t2 != t)
        outflow = sum(f_td[t, d] for d in 1:D) + sum(f_tt[t, t2] for t2 in 1:T if t2 != t)
        @constraint(model, inflow == outflow)
    end

    # Transshipment capacity constraints
    for t in 1:T
        @constraint(model,
            sum(f_st[s, t] for s in 1:S) + sum(f_tt[t2, t] for t2 in 1:T if t2 != t) <=
            prob.transship_capacities[t]
        )
    end

    # No self-loops in transshipment network
    for t in 1:T
        @constraint(model, f_tt[t, t] == 0)
    end

    return model
end

# Register the problem type
register_problem(
    :transshipment,
    Transshipment,
    "Transshipment problem with intermediate storage and routing optimization"
)
