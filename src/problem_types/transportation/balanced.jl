using JuMP
using Random

"""
    BalancedTransportationProblem <: ProblemGenerator

Generator for *balanced* transportation problems, where total supply must exactly
equal total demand and both supply and demand constraints are enforced as equalities.

# Overview
Models the classic balanced transportation problem. The decisions are shipment
amounts on every source-destination lane. The objective minimizes total shipping
cost, computed from distance-based per-lane costs. Unlike the standard variant
(inequality supply/demand constraints), the balanced variant requires that every
source ships out *exactly* its supply (`sum_j x_ij == supply_i`) and every
destination receives *exactly* its demand (`sum_i x_ij == demand_j`). Such a model
is feasible if and only if total supply equals total demand, so the constructor
controls feasibility purely through the supply/demand totals.

# Fields
- `n_sources::Int`: Number of supply sources
- `n_destinations::Int`: Number of demand destinations
- `supplies::Vector{Int}`: Supply at each source
- `demands::Vector{Int}`: Demand at each destination
- `costs::Matrix{Float64}`: Distance-based transportation cost from each source to each destination
"""
struct BalancedTransportationProblem <: ProblemGenerator
    n_sources::Int
    n_destinations::Int
    supplies::Vector{Int}
    demands::Vector{Int}
    costs::Matrix{Float64}
end

"""
    BalancedTransportationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a balanced transportation problem instance.

The model creates exactly `n_sources * n_destinations` decision variables (one per
source-destination lane), so the dimensions are sized to land near
`target_variables`.

Feasibility handling (a balanced TP with equality constraints is feasible iff
total supply == total demand):
- `feasible`: supplies/demands are adjusted so the two totals are made *exactly*
  equal (rounding off-by-one guarded), guaranteeing a feasible point exists.
- `infeasible`: total demand is forced to exceed total supply by a clear positive
  margin, making the equalities mutually contradictory.
- `unknown`: totals are left as generated (a natural instance, balanced only by
  chance), with no forced infeasibility.

# Arguments
- `target_variables`: Target number of variables (n_sources × n_destinations)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function BalancedTransportationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # --- Dimension sizing ---
    # Variable count = n_sources * n_destinations, sized to target_variables.
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

    # --- Base supply/demand data ---
    min_supply, max_supply = supply_range
    supplies = rand(min_supply:max_supply, n_sources)

    min_demand, max_demand = demand_range
    demands = rand(min_demand:max_demand, n_destinations)

    # --- Distance-based cost generation ---
    source_positions = [(rand() * 100.0, rand() * 100.0) for _ in 1:n_sources]
    dest_positions = [(rand() * 100.0, rand() * 100.0) for _ in 1:n_destinations]

    distances = zeros(n_sources, n_destinations)
    for i in 1:n_sources, j in 1:n_destinations
        distances[i, j] = sqrt((source_positions[i][1] - dest_positions[j][1])^2 +
                               (source_positions[i][2] - dest_positions[j][2])^2)
    end

    min_cost, max_cost = cost_range
    max_dist = maximum(distances)
    cost_per_distance = max_dist > 0 ? (max_cost - min_cost) / max_dist : 0.0
    costs = zeros(n_sources, n_destinations)
    for i in 1:n_sources, j in 1:n_destinations
        base_cost = min_cost + distances[i, j] * cost_per_distance
        costs[i, j] = base_cost * (0.8 + 0.4 * rand())
    end

    # Helper: distribute a positive integer `amount` across `vec` entries.
    # The total added is EXACTLY `amount` (remainder distributed one-by-one),
    # which is essential for guaranteeing exact balance.
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

    # --- Feasibility handling ---
    total_supply = sum(supplies)
    total_demand = sum(demands)

    if feasibility_status == feasible
        # Make totals EXACTLY equal so the equality-constrained model is feasible.
        # distribute_additions! adds exactly the requested deficit (off-by-one
        # remainder is fully distributed), so afterwards sum(supplies) == sum(demands).
        if total_supply > total_demand
            distribute_additions!(demands, total_supply - total_demand)
        elseif total_demand > total_supply
            distribute_additions!(supplies, total_demand - total_supply)
        end
        # Hard guard against any residual mismatch (rounding off-by-one safety).
        diff = sum(supplies) - sum(demands)
        if diff > 0
            demands[end] += diff
        elseif diff < 0
            supplies[end] += -diff
        end
    elseif feasibility_status == infeasible
        # Force total demand strictly greater than total supply with a clear margin,
        # making sum_i sum_j x_ij = total_supply and = total_demand contradictory.
        target_margin = max(1, round(Int, (0.05 + 0.10 * rand()) * max(total_supply, 1)))
        needed = (total_supply + target_margin) - total_demand
        if needed > 0
            distribute_additions!(demands, needed)
        end
        # Guarantee strict inequality even if base data already had demand > supply.
        if sum(demands) <= sum(supplies)
            demands[end] += (sum(supplies) - sum(demands)) + target_margin
        end
    end
    # For unknown, leave supplies/demands as generated (no forced infeasibility).

    return BalancedTransportationProblem(n_sources, n_destinations, supplies, demands, costs)
end

"""
    build_model(prob::BalancedTransportationProblem)

Build a JuMP model for the balanced transportation problem. Completely
deterministic — uses only the data stored in `prob`.

Creates `n_sources * n_destinations` nonnegative shipment variables, minimizes
total distance-based shipping cost, and enforces equality on both supply
(`sum_j x_ij == supply_i`) and demand (`sum_i x_ij == demand_j`).

# Arguments
- `prob`: BalancedTransportationProblem instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::BalancedTransportationProblem)
    model = Model()

    # Variables: one shipment amount per source-destination lane.
    @variable(model, x[1:prob.n_sources, 1:prob.n_destinations] >= 0)

    # Objective: minimize total shipping cost.
    @objective(model, Min, sum(prob.costs[i, j] * x[i, j]
                               for i in 1:prob.n_sources, j in 1:prob.n_destinations))

    # Supply equalities: each source ships out exactly its supply.
    for i in 1:prob.n_sources
        @constraint(model, sum(x[i, j] for j in 1:prob.n_destinations) == prob.supplies[i])
    end

    # Demand equalities: each destination receives exactly its demand.
    for j in 1:prob.n_destinations
        @constraint(model, sum(x[i, j] for i in 1:prob.n_sources) == prob.demands[j])
    end

    return model
end

# Register the variant
register_variant(
    :transportation,
    :balanced,
    BalancedTransportationProblem,
    "Balanced transportation problem with equality supply and demand constraints (total supply = total demand)",
)
