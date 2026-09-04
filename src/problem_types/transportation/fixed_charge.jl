using JuMP
using Random

"""
    FixedChargeTransportationProblem <: ProblemGenerator

Transportation with a fixed cost for opening each source-destination lane.
Continuous shipments are gated by binary lane-use decisions, giving the classic
fixed-charge transportation formulation rather than a continuous capacitated
transportation LP.
"""
struct FixedChargeTransportationProblem <: ProblemGenerator
    n_sources::Int
    n_destinations::Int
    supplies::Vector{Int}
    demands::Vector{Int}
    variable_costs::Matrix{Float64}
    fixed_costs::Matrix{Float64}
    lane_capacities::Matrix{Int}
end

function FixedChargeTransportationProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    target_lanes = max(4, round(Int, max(target_variables, 1) / 2))
    n_sources = max(2, round(Int, sqrt(target_lanes)))
    n_destinations = max(2, round(Int, target_lanes / n_sources))

    demands = rand(rng, 8:35, n_destinations)
    supplies = rand(rng, 15:55, n_sources)
    variable_costs = [
        round(1.0 + 14.0 * rand(rng); digits=3) for _ in 1:n_sources, _ in 1:n_destinations
    ]
    fixed_costs = [
        round(20.0 + 180.0 * rand(rng); digits=2) for _ in 1:n_sources, _ in 1:n_destinations
    ]
    lane_capacities = rand(rng, 10:45, n_sources, n_destinations)

    if feasibility_status == feasible
        # Assign every destination to a planted source lane. The corresponding
        # source supply and lane capacity then support an explicit shipment plan.
        planted_outflow = zeros(Int, n_sources)
        for j in 1:n_destinations
            i = 1 + (j - 1) % n_sources
            planted_outflow[i] += demands[j]
            lane_capacities[i, j] = max(lane_capacities[i, j], demands[j] + rand(rng, 1:5))
        end
        for i in 1:n_sources
            supplies[i] = max(supplies[i], planted_outflow[i] + rand(rng, 1:6))
        end
    elseif feasibility_status == infeasible
        # Aggregate demand exceeds all source supply, independent of lane-use
        # integrality or capacities.
        total_supply = sum(supplies)
        scale = (total_supply + rand(rng, 5:20)) / sum(demands)
        demands .= ceil.(Int, demands .* scale)
        while sum(demands) <= total_supply
            demands[rand(rng, 1:n_destinations)] += 1
        end
    end

    return FixedChargeTransportationProblem(
        n_sources, n_destinations, supplies, demands, variable_costs, fixed_costs, lane_capacities
    )
end

function build_model(prob::FixedChargeTransportationProblem)
    model = Model()
    I = prob.n_sources
    J = prob.n_destinations

    @variable(model, shipment[1:I, 1:J] >= 0)
    @variable(model, lane_open[1:I, 1:J], Bin)
    @objective(
        model,
        Min,
        sum(
            prob.variable_costs[i, j] * shipment[i, j] + prob.fixed_costs[i, j] * lane_open[i, j]
            for i in 1:I, j in 1:J
        )
    )

    for i in 1:I
        @constraint(model, sum(shipment[i, j] for j in 1:J) <= prob.supplies[i])
    end
    for j in 1:J
        @constraint(model, sum(shipment[i, j] for i in 1:I) >= prob.demands[j])
    end
    for i in 1:I, j in 1:J
        @constraint(model, shipment[i, j] <= prob.lane_capacities[i, j] * lane_open[i, j])
    end
    return model
end

register_variant(
    :transportation,
    :fixed_charge,
    FixedChargeTransportationProblem,
    "Fixed-charge transportation with continuous shipments gated by binary lane-opening decisions",
)
