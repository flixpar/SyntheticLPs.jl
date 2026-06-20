using JuMP
using Random
using Distributions

"""
    EmissionConstrainedTransportationProblem <: ProblemGenerator

Generator for transportation problems with a global CO2 emission budget.

# Overview
Models the classic transportation planning problem augmented with a
sustainability constraint. The decisions are shipment amounts on every
source-destination lane. The objective minimizes total shipping cost, subject to
source supply limits, destination demand requirements, and a single global
emission budget `sum_{i,j} e_{ij} x_{ij} <= emission_limit`, where `e_{ij}` is the
CO2 emitted per unit shipped on lane `(i, j)` (proportional to geographic
distance with vehicle-type variation).

The only decision variables are the lane flows `x[i, j]`, so the total variable
count is `n_sources * n_destinations`, sized to match `target_variables`.

# Fields
- `n_sources::Int`: Number of supply sources
- `n_destinations::Int`: Number of demand destinations
- `supplies::Vector{Int}`: Supply at each source
- `demands::Vector{Int}`: Demand at each destination
- `costs::Matrix{Float64}`: Transportation cost from each source to each destination
- `emission_rates::Matrix{Float64}`: CO2 emitted per unit shipped on each lane
- `emission_limit::Float64`: Maximum total emissions allowed across all shipments
"""
struct EmissionConstrainedTransportationProblem <: ProblemGenerator
    n_sources::Int
    n_destinations::Int
    supplies::Vector{Int}
    demands::Vector{Int}
    costs::Matrix{Float64}
    emission_rates::Matrix{Float64}
    emission_limit::Float64
end

"""
    EmissionConstrainedTransportationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct an emission-constrained transportation problem instance.

Variables: x[i, j] (lane flows). Total = n_sources * n_destinations, sized to
`target_variables`.

# Arguments
- `target_variables`: Target number of variables (n_sources × n_destinations)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function EmissionConstrainedTransportationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # --- Dimension sizing ---
    # Var-count formula: total variables = n_sources * n_destinations.
    # Size n_sources ~ sqrt(target) and n_destinations to hit the target product.
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

    # --- Scale-dependent parameter ranges ---
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

    # --- Base supplies and demands ---
    min_supply, max_supply = supply_range
    supplies = rand(min_supply:max_supply, n_sources)

    min_demand, max_demand = demand_range
    demands = rand(min_demand:max_demand, n_destinations)

    # --- Geographic positions for distance-based costs and emissions ---
    source_positions = [(rand() * 100.0, rand() * 100.0) for _ in 1:n_sources]
    dest_positions = [(rand() * 100.0, rand() * 100.0) for _ in 1:n_destinations]

    distances = zeros(n_sources, n_destinations)
    for i in 1:n_sources, j in 1:n_destinations
        distances[i, j] = sqrt((source_positions[i][1] - dest_positions[j][1])^2 +
                               (source_positions[i][2] - dest_positions[j][2])^2)
    end

    # --- Costs based on distances with variation ---
    min_cost, max_cost = cost_range
    max_dist = maximum(distances)
    cost_per_distance = max_dist > 0 ? (max_cost - min_cost) / max_dist : 0.0
    costs = zeros(n_sources, n_destinations)
    for i in 1:n_sources, j in 1:n_destinations
        base_cost = min_cost + distances[i, j] * cost_per_distance
        costs[i, j] = base_cost * (0.8 + 0.4 * rand())
    end

    # --- Emission rates: proportional to distance with vehicle-type variation ---
    # Ensure a strictly positive floor so emissions scale with shipped volume.
    emission_rates = zeros(n_sources, n_destinations)
    for i in 1:n_sources, j in 1:n_destinations
        emission_rates[i, j] = (distances[i, j] + 1.0) * rand(Uniform(0.01, 0.05))
    end

    # --- Resolve feasibility status ---
    actual_status = feasibility_status
    if feasibility_status == unknown
        actual_status = rand() < 0.7 ? feasible : infeasible
    end

    # Helper: distribute integer additions across a vector
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

    # --- Balance supply/demand for the requested status ---
    total_supply = sum(supplies)
    total_demand = sum(demands)

    if actual_status == feasible || actual_status == unknown
        # Ensure enough total supply exists to meet demand
        if total_supply < total_demand
            distribute_additions!(supplies, total_demand - total_supply)
        end
    elseif actual_status == infeasible
        # The emission constraint is what forces infeasibility below; keep
        # supply/demand satisfiable so infeasibility is unambiguously from
        # the emission budget (ensure total_supply >= total_demand).
        if total_supply < total_demand
            distribute_additions!(supplies, total_demand - total_supply)
        end
    end

    # --- Compute the emissions of a CONCRETE feasible flow (respects supply caps) ---
    # For each destination, greedily fill demand from the lowest-emission sources
    # that still have remaining supply. This yields a genuinely achievable flow,
    # so its total emissions is a valid feasible reference (unlike taking the
    # column-wise minimum emission rate, which ignores supply capacity).
    function feasible_flow_emissions()
        remaining = Float64.(copy(supplies))
        total_emissions = 0.0
        feasible = true
        for j in 1:n_destinations
            need = Float64(demands[j])
            # sources sorted by emission rate for this destination (cheapest first)
            order = sortperm(emission_rates[:, j])
            for i in order
                need <= 0 && break
                ship = min(need, remaining[i])
                if ship > 0
                    total_emissions += emission_rates[i, j] * ship
                    remaining[i] -= ship
                    need -= ship
                end
            end
            if need > 1e-9
                feasible = false  # not enough supply (should not happen when balanced)
            end
        end
        return total_emissions, feasible
    end

    ref_emissions, ref_ok = feasible_flow_emissions()

    # --- Set the emission limit ---
    if actual_status == infeasible
        # Force infeasibility: set the budget strictly below the emissions of the
        # MINIMUM-emission feasible flow. Any feasible transportation flow emits at
        # least `ref_emissions` (greedy min-emission flow is a lower-emission
        # achievable point), so a budget below it makes the model infeasible with
        # a clear margin.
        # Lower bound on emissions of ANY feasible flow: each unit of total demand
        # must travel on some lane, costing at least the global minimum rate.
        min_rate = minimum(emission_rates)
        hard_lower_bound = min_rate * sum(demands)
        emission_limit = min(ref_emissions, hard_lower_bound) * 0.5
        # Guard against a degenerate zero limit.
        emission_limit = max(emission_limit, 1e-6)
    else
        # Feasible / unknown: budget must accommodate the known feasible flow.
        # Use the achievable greedy-flow emissions inflated by a comfortable
        # margin so the feasible point provably satisfies the budget.
        base_ref = ref_ok ? ref_emissions : maximum(emission_rates) * sum(demands)
        emission_limit = base_ref * rand(Uniform(1.3, 2.0))
    end

    return EmissionConstrainedTransportationProblem(
        n_sources, n_destinations, supplies, demands, costs, emission_rates, emission_limit
    )
end

"""
    build_model(prob::EmissionConstrainedTransportationProblem)

Build a JuMP model for the emission-constrained transportation problem.
Deterministic — uses only data from the struct fields.

# Arguments
- `prob`: EmissionConstrainedTransportationProblem instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::EmissionConstrainedTransportationProblem)
    model = Model()

    # Variables: lane flows. Total = n_sources * n_destinations.
    @variable(model, x[1:prob.n_sources, 1:prob.n_destinations] >= 0)

    # Objective: minimize total shipping cost
    @objective(model, Min, sum(prob.costs[i, j] * x[i, j]
                               for i in 1:prob.n_sources, j in 1:prob.n_destinations))

    # Supply constraints
    for i in 1:prob.n_sources
        @constraint(model, sum(x[i, j] for j in 1:prob.n_destinations) <= prob.supplies[i])
    end

    # Demand constraints
    for j in 1:prob.n_destinations
        @constraint(model, sum(x[i, j] for i in 1:prob.n_sources) >= prob.demands[j])
    end

    # Global emission budget
    @constraint(model, sum(prob.emission_rates[i, j] * x[i, j]
                           for i in 1:prob.n_sources, j in 1:prob.n_destinations) <= prob.emission_limit)

    return model
end

# Register the variant
register_variant(
    :transportation,
    :emission_constrained,
    EmissionConstrainedTransportationProblem,
    "Transportation problem with a global CO2 emission budget constraint on total shipments",
)
