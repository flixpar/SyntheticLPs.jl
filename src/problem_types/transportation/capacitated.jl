using JuMP
using Random
using Distributions

"""
    CapacitatedTransportationProblem <: ProblemGenerator

Generator for capacitated transportation problems that optimize shipping goods from
sources to destinations at minimum cost subject to per-lane capacity limits.

# Overview
Extends the classic transportation problem with an explicit capacity on every
individual source-destination lane (route). The decisions are shipment amounts on
each lane. The objective minimizes total shipping cost. Source constraints limit
outbound shipments by available supply, destination constraints require inbound
shipments to meet demand, and each lane `x[i, j]` is additionally bounded above by
its route capacity `route_capacities[i, j]`. A capacity of `Inf` denotes an
unlimited lane; finite capacities are real upper bounds.

# Fields
- `n_sources::Int`: Number of supply sources
- `n_destinations::Int`: Number of demand destinations
- `supplies::Vector{Int}`: Supply at each source
- `demands::Vector{Int}`: Demand at each destination
- `costs::Matrix{Float64}`: Transportation cost from each source to each destination
- `route_capacities::Matrix{Float64}`: Per-lane shipment capacity (may be `Inf` for unlimited lanes)
"""
struct CapacitatedTransportationProblem <: ProblemGenerator
    n_sources::Int
    n_destinations::Int
    supplies::Vector{Int}
    demands::Vector{Int}
    costs::Matrix{Float64}
    route_capacities::Matrix{Float64}
end

"""
    CapacitatedTransportationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a capacitated transportation problem instance.

The only decision variables are the lane shipments `x[i, j]`, so the variable count
is exactly `n_sources * n_destinations`; the dimensions are sized so this lands near
`target_variables`.

# Arguments
- `target_variables`: Target number of variables (n_sources × n_destinations)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function CapacitatedTransportationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # --- Dimension sizing ---
    # Variable count formula: n_sources * n_destinations (only x[i, j]).
    sqrt_target = sqrt(target_variables)
    ratio = 0.5 + rand() * 1.0  # ratio between 0.5 and 1.5

    n_sources = max(2, round(Int, sqrt_target * ratio))
    n_destinations = max(2, round(Int, target_variables / n_sources))

    # Fine-tune to get closer to target
    current_vars = n_sources * n_destinations
    if current_vars < target_variables * 0.9
        if n_sources >= n_destinations
            n_sources = max(n_sources, round(Int, target_variables / n_destinations))
        else
            n_destinations = max(n_destinations, round(Int, target_variables / n_sources))
        end
    elseif current_vars > target_variables * 1.1
        if n_sources >= n_destinations
            n_sources = max(2, round(Int, target_variables / n_destinations))
        else
            n_destinations = max(2, round(Int, target_variables / n_sources))
        end
    end

    # Set realistic parameter ranges based on problem size
    total_vars = n_sources * n_destinations
    if total_vars <= 250
        supply_range = (rand(50:100), rand(200:500))
        demand_range = (rand(30:80), rand(150:300))
        cost_range = (rand(5.0:0.5:15.0), rand(25.0:0.5:60.0))
    elseif total_vars <= 1000
        supply_range = (rand(100:500), rand(1000:5000))
        demand_range = (rand(80:300), rand(800:3000))
        cost_range = (rand(10.0:0.5:30.0), rand(50.0:0.5:150.0))
    else
        supply_range = (rand(500:2000), rand(5000:50000))
        demand_range = (rand(300:1500), rand(3000:30000))
        cost_range = (rand(20.0:0.5:100.0), rand(100.0:0.5:500.0))
    end

    # Generate random data
    min_supply, max_supply = supply_range
    supplies = rand(min_supply:max_supply, n_sources)

    min_demand, max_demand = demand_range
    demands = rand(min_demand:max_demand, n_destinations)

    # Generate geographic positions for realistic distance-based costs
    source_positions = [(rand() * 100.0, rand() * 100.0) for _ in 1:n_sources]
    dest_positions = [(rand() * 100.0, rand() * 100.0) for _ in 1:n_destinations]

    distances = zeros(n_sources, n_destinations)
    for i in 1:n_sources, j in 1:n_destinations
        distances[i, j] = sqrt((source_positions[i][1] - dest_positions[j][1])^2 +
                               (source_positions[i][2] - dest_positions[j][2])^2)
    end

    # Generate costs based on distances with variation
    min_cost, max_cost = cost_range
    cost_per_distance = (max_cost - min_cost) / max(maximum(distances), 1.0)
    costs = zeros(n_sources, n_destinations)
    for i in 1:n_sources, j in 1:n_destinations
        base_cost = min_cost + distances[i, j] * cost_per_distance
        costs[i, j] = base_cost * (0.8 + 0.4 * rand())
    end

    # --- Route capacities ---
    # Some lanes are capacitated (finite) and some are unlimited (Inf).
    avg_flow = sum(demands) / (n_sources * n_destinations)
    route_capacities = fill(Inf, n_sources, n_destinations)
    for i in 1:n_sources, j in 1:n_destinations
        if rand() < 0.7  # 70% of routes have capacity limits
            route_capacities[i, j] = avg_flow * rand(Uniform(0.5, 3.0))
        end
    end

    # Helper function to distribute additions across a vector
    function distribute_additions!(vec::Vector{Int}, amount::Int)
        if amount <= 0
            return
        end
        w = rand(length(vec))
        w_sum = sum(w)
        base = floor.(Int, (w ./ w_sum) .* amount)
        remainder = amount - sum(base)
        if remainder > 0
            for idx in randperm(length(vec))[1:min(remainder, length(vec))]
                base[idx] += 1
            end
        end
        vec .+= base
    end

    # Resolve the effective status: for `unknown` we leave the natural instance
    # untouched (no forced infeasibility, no feasibility guarantees).
    total_supply = sum(supplies)
    total_demand = sum(demands)

    if feasibility_status == feasible
        # 1) Ensure total supply can cover total demand (with no margin needed; equality OK).
        if total_supply < total_demand
            distribute_additions!(supplies, total_demand - total_supply)
            total_supply = sum(supplies)
        end

        # 2) Ensure lane capacities admit a feasible routing.
        #    Sufficient condition: for every destination j, the sum of finite lane
        #    capacities into j is >= demand[j] (Inf lanes trivially satisfy this).
        #    We also keep at least some Inf lanes so the instance is not all-finite.
        for j in 1:n_destinations
            finite_idx = [i for i in 1:n_sources if isfinite(route_capacities[i, j])]
            if length(finite_idx) == n_sources
                # No unlimited lane into j: scale finite caps up to cover demand with margin.
                cap_in = sum(route_capacities[i, j] for i in finite_idx)
                needed = demands[j] * 1.25
                if cap_in < needed
                    scale = needed / max(cap_in, 1e-9)
                    for i in finite_idx
                        route_capacities[i, j] *= scale
                    end
                end
            end
        end

    elseif feasibility_status == infeasible
        # Make infeasibility come ONLY from lane capacities (do not also inflate demand):
        # force every lane finite (remove all unlimited escapes) and scale the total
        # finite capacity so it is provably below total demand with a clear margin.
        # Then no routing can deliver total_demand units, so demand constraints cannot
        # all be met -> infeasible. Also bump supply well above demand so supply is not
        # the binding/limiting reason (capacity is the sole cause).
        for i in 1:n_sources, j in 1:n_destinations
            if !isfinite(route_capacities[i, j])
                route_capacities[i, j] = avg_flow * rand(Uniform(0.5, 3.0))
            end
        end

        total_finite_cap = sum(route_capacities)
        # Target total capacity at 70% of demand: a clear, deterministic margin below demand.
        target_cap = 0.7 * total_demand
        scale = target_cap / max(total_finite_cap, 1e-9)
        route_capacities .*= scale

        # Ensure supply is not the binding constraint masking the capacity infeasibility.
        if total_supply < total_demand
            distribute_additions!(supplies, (total_demand - total_supply) + n_sources)
        end
    end
    # For `unknown`, leave supplies, demands, and capacities exactly as generated.

    return CapacitatedTransportationProblem(n_sources, n_destinations, supplies, demands, costs, route_capacities)
end

"""
    build_model(prob::CapacitatedTransportationProblem)

Build a JuMP model for the capacitated transportation problem. Deterministic — uses
only data from the struct fields.

# Returns
- `model`: The JuMP model
"""
function build_model(prob::CapacitatedTransportationProblem)
    model = Model()

    # Variables: one shipment per lane -> n_sources * n_destinations variables.
    @variable(model, x[1:prob.n_sources, 1:prob.n_destinations] >= 0)

    # Objective: minimize total shipping cost.
    @objective(model, Min, sum(prob.costs[i, j] * x[i, j]
                               for i in 1:prob.n_sources, j in 1:prob.n_destinations))

    # Supply constraints.
    for i in 1:prob.n_sources
        @constraint(model, sum(x[i, j] for j in 1:prob.n_destinations) <= prob.supplies[i])
    end

    # Demand constraints.
    for j in 1:prob.n_destinations
        @constraint(model, sum(x[i, j] for i in 1:prob.n_sources) >= prob.demands[j])
    end

    # Per-lane route capacity constraints (only for finite capacities).
    for i in 1:prob.n_sources, j in 1:prob.n_destinations
        if isfinite(prob.route_capacities[i, j])
            @constraint(model, x[i, j] <= prob.route_capacities[i, j])
        end
    end

    return model
end

# Register the variant
register_variant(
    :transportation,
    :capacitated,
    CapacitatedTransportationProblem,
    "Capacitated transportation problem with per-lane route capacity limits on each source-destination pair",
)
