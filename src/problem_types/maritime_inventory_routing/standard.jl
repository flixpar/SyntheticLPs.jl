using JuMP
using Random

"""
    MaritimeInventoryRoutingProblem <: ProblemGenerator

A time-expanded maritime inventory-routing problem. Binary vessel locations and
movements determine which ports can receive deliveries; onboard load evolves
with depot pickups and customer deliveries; and customer inventories evolve
over multiple periods under exogenous consumption.
"""
struct MaritimeInventoryRoutingProblem <: ProblemGenerator
    n_ports::Int
    n_customers::Int
    n_vessels::Int
    n_periods::Int
    vessel_capacity::Vector{Float64}
    initial_load::Vector{Float64}
    initial_inventory::Vector{Float64}
    consumption::Matrix{Float64}
    depot_supply::Vector{Float64}
    travel_cost::Matrix{Float64}
    holding_cost::Vector{Float64}
end

function _mirp_variable_count(P::Int, V::Int, T::Int)
    C = P - 1
    return V * P * (T + 1) +       # vessel locations
           V * P * P * T +         # vessel movements, including stays
           V * C * T +             # deliveries
           V * T +                 # depot pickups
           V * (T + 1) +           # onboard loads
           C * (T + 1)             # customer inventories
end

function MaritimeInventoryRoutingProblem(
    target_variables::Int,
    feasibility_status::FeasibilityStatus,
    seed::Int,
)
    rng = MersenneTwister(seed)
    target = max(target_variables, 1)

    # Search small, operationally meaningful dimensions. Every customer must
    # fit in at least one planted odd-period visit slot.
    best = (3, 1, 3)
    best_error = typemax(Int)
    for P in 3:12, V in 1:3, T in 2:30
        V * cld(T, 2) >= P - 1 || continue
        actual = _mirp_variable_count(P, V, T)
        error = abs(actual - target)
        if error < best_error || (error == best_error && P > best[1])
            best = (P, V, T)
            best_error = error
        end
    end
    n_ports, n_vessels, n_periods = best
    n_customers = n_ports - 1

    consumption = Float64.(rand(rng, 1:5, n_customers, n_periods))
    travel_cost = zeros(Float64, n_ports, n_ports)
    for i in 1:n_ports, j in 1:n_ports
        if i != j
            travel_cost[i, j] = round(15.0 + 90.0 * rand(rng), digits=2)
        end
    end
    holding_cost = [round(0.1 + 1.5 * rand(rng), digits=3)
                    for _ in 1:n_customers]

    total_consumption = sum(consumption)
    average_per_vessel = total_consumption / max(n_vessels, 1)
    vessel_capacity = [max(10.0, average_per_vessel * rand(rng, 0.45:0.05:0.75))
                       for _ in 1:n_vessels]
    initial_load = copy(vessel_capacity)
    initial_inventory = Float64[
        rand(rng, 0:round(Int, sum(consumption[c, :]) * 0.5))
        for c in 1:n_customers
    ]
    depot_supply = [total_consumption / n_periods * rand(rng, 0.7:0.1:1.3)
                    for _ in 1:n_periods]

    if feasibility_status == feasible
        # Plant alternating depot/customer schedules. A customer is replenished
        # on its first visit with all consumption from that period onward; its
        # initial inventory covers exactly the preceding periods.
        first_visit = fill(0, n_customers)
        first_vessel = fill(0, n_customers)
        visit_slot = 0
        for t in 1:2:n_periods, v in 1:n_vessels
            visit_slot += 1
            c = 1 + (visit_slot - 1) % n_customers
            if first_visit[c] == 0
                first_visit[c] = t
                first_vessel[c] = v
            end
        end

        planted_delivery = zeros(Float64, n_vessels, n_customers, n_periods)
        for c in 1:n_customers
            t = first_visit[c]
            v = first_vessel[c]
            initial_inventory[c] = t == 1 ? 0.0 : sum(consumption[c, 1:(t - 1)])
            planted_delivery[v, c, t] = sum(consumption[c, t:n_periods])
        end
        for v in 1:n_vessels
            largest_delivery = maximum(planted_delivery[v, :, :])
            vessel_capacity[v] = max(vessel_capacity[v], 1.10 * largest_delivery + 1.0)
            initial_load[v] = vessel_capacity[v]
        end
        # At a depot period each vessel can refill completely, which is enough
        # to support the planted alternating schedule.
        depot_supply .= sum(vessel_capacity)
    elseif feasibility_status == infeasible
        # Aggregate material certificate. Deliveries only transfer material from
        # vessels to customer inventories, while depot pickup is the sole source.
        # Available initial material plus all depot supply is strictly below
        # cumulative consumption, even after relaxing route binaries.
        initial_inventory .= 0.0
        initial_load .= 0.0
        depot_supply .= 0.65 * total_consumption / n_periods
    end

    return MaritimeInventoryRoutingProblem(
        n_ports, n_customers, n_vessels, n_periods,
        vessel_capacity, initial_load, initial_inventory,
        consumption, depot_supply, travel_cost, holding_cost,
    )
end

function build_model(prob::MaritimeInventoryRoutingProblem)
    model = Model()
    P = prob.n_ports
    C = prob.n_customers
    V = prob.n_vessels
    T = prob.n_periods

    @variable(model, location[1:V, 1:P, 0:T], Bin)
    @variable(model, move[1:V, 1:P, 1:P, 1:T], Bin)
    @variable(model, delivery[1:V, 1:C, 1:T] >= 0)
    @variable(model, pickup[1:V, 1:T] >= 0)
    @variable(model, 0 <= load[v in 1:V, 0:T] <= prob.vessel_capacity[v])
    @variable(model, inventory[1:C, 0:T] >= 0)

    @objective(model, Min,
        sum(prob.travel_cost[i, j] * move[v, i, j, t]
            for v in 1:V, i in 1:P, j in 1:P, t in 1:T) +
        sum(prob.holding_cost[c] * inventory[c, t] for c in 1:C, t in 1:T)
    )

    for v in 1:V
        @constraint(model, location[v, 1, 0] == 1)
        for p in 2:P
            @constraint(model, location[v, p, 0] == 0)
        end
        @constraint(model, load[v, 0] == prob.initial_load[v])
    end
    for c in 1:C
        @constraint(model, inventory[c, 0] == prob.initial_inventory[c])
    end

    for v in 1:V, t in 1:T
        for i in 1:P
            @constraint(model, sum(move[v, i, j, t] for j in 1:P) == location[v, i, t - 1])
        end
        for j in 1:P
            @constraint(model, sum(move[v, i, j, t] for i in 1:P) == location[v, j, t])
        end
        @constraint(model, pickup[v, t] <= prob.vessel_capacity[v] * location[v, 1, t])
        @constraint(model,
            load[v, t] == load[v, t - 1] + pickup[v, t] -
                          sum(delivery[v, c, t] for c in 1:C)
        )
        for c in 1:C
            @constraint(model,
                delivery[v, c, t] <= prob.vessel_capacity[v] * location[v, c + 1, t]
            )
        end
    end
    for t in 1:T
        @constraint(model, sum(pickup[v, t] for v in 1:V) <= prob.depot_supply[t])
    end
    for c in 1:C, t in 1:T
        @constraint(model,
            inventory[c, t] == inventory[c, t - 1] +
                               sum(delivery[v, c, t] for v in 1:V) -
                               prob.consumption[c, t]
        )
    end

    return model
end

register_variant(
    :maritime_inventory_routing,
    :standard,
    MaritimeInventoryRoutingProblem,
    "Time-expanded maritime inventory routing with binary vessel movement, onboard cargo, depot pickup, deliveries, and customer inventories",
)
