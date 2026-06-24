using JuMP
using Random
using Distributions

"""
    SupplyChainNetworkPlanningProblem <: ProblemGenerator

Multi-period, multi-product supply-chain planning LP with production, customer
shipments, plant inventory, and soft unmet-demand penalties. The formulation
creates the repeated block-angular time structure common in industrial network
planning models.
"""
struct SupplyChainNetworkPlanningProblem <: ProblemGenerator
    n_plants::Int
    n_customers::Int
    n_products::Int
    n_periods::Int
    production_cost::Array{Float64,3}
    shipment_cost::Array{Float64,4}
    inventory_cost::Array{Float64,3}
    unmet_penalty::Array{Float64,3}
    max_unmet::Array{Float64,3}
    demand::Array{Float64,3}
    plant_capacity::Array{Float64,2}
    product_capacity_use::Vector{Float64}
    initial_inventory::Array{Float64,2}
    arc_open::Array{Bool,3}
end

function _choose_supply_network_dims(target_variables::Int)
    best = (Inf, 2, 3, 2, 3)
    for p in 2:12, c in 3:40, k in 1:5, t in 3:12
        vars = p*k*t + p*c*k*t + p*k*t + c*k*t
        err = abs(vars - target_variables)
        if err < best[1]
            best = (err, p, c, k, t)
        end
    end
    return best[2], best[3], best[4], best[5]
end

function SupplyChainNetworkPlanningProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)
    n_plants, n_customers, n_products, n_periods = _choose_supply_network_dims(target_variables)

    plant_x = rand(Uniform(0.0, 100.0), n_plants)
    plant_y = rand(Uniform(0.0, 100.0), n_plants)
    cust_x = rand(Uniform(0.0, 100.0), n_customers)
    cust_y = rand(Uniform(0.0, 100.0), n_customers)

    base_prod = rand(LogNormal(log(18.0), 0.35), n_plants, n_products, n_periods)
    production_cost = base_prod .* reshape(rand(Uniform(0.85, 1.25), n_periods), 1, 1, n_periods)
    inventory_cost = rand(Uniform(0.25, 1.5), n_plants, n_products, n_periods)
    unmet_penalty = rand(Uniform(120.0, 250.0), n_customers, n_products, n_periods)
    max_unmet = fill(Inf, n_customers, n_products, n_periods)
    if feasibility_status == infeasible
        max_unmet .= 0.0
    end
    product_capacity_use = rand(Uniform(0.6, 1.8), n_products)

    demand = zeros(Float64, n_customers, n_products, n_periods)
    for j in 1:n_customers, k in 1:n_products, t in 1:n_periods
        seasonal = 1.0 + 0.25 * sin(2π * t / max(n_periods, 2) + rand())
        demand[j, k, t] = rand(Uniform(15.0, 60.0)) * seasonal
    end

    arc_open = falses(n_plants, n_customers, n_products)
    shipment_cost = zeros(Float64, n_plants, n_customers, n_products, n_periods)
    for j in 1:n_customers, k in 1:n_products
        distances = [sqrt((plant_x[i] - cust_x[j])^2 + (plant_y[i] - cust_y[j])^2) for i in 1:n_plants]
        order = sortperm(distances)
        keep = min(n_plants, max(2, ceil(Int, 0.35 * n_plants)))
        for ii in 1:keep
            i = order[ii]
            arc_open[i, j, k] = true
        end
        for i in 1:n_plants
            if arc_open[i, j, k]
                for t in 1:n_periods
                    shipment_cost[i, j, k, t] = 0.4 * distances[i] * rand(Uniform(0.8, 1.3)) + rand(Uniform(2.0, 8.0))
                end
            end
        end
    end

    initial_inventory = rand(Uniform(0.0, 25.0), n_plants, n_products)
    total_weighted_demand = [sum(demand[:, k, t]) * product_capacity_use[k] for k in 1:n_products, t in 1:n_periods]
    period_need = [sum(total_weighted_demand[:, t]) for t in 1:n_periods]
    multiplier = feasibility_status == infeasible ? rand(Uniform(0.35, 0.75)) : rand(Uniform(1.10, 1.55))
    if feasibility_status == unknown && rand() < 0.15
        multiplier = rand(Uniform(0.75, 1.05))
    end
    plant_capacity = zeros(Float64, n_plants, n_periods)
    plant_shares = rand(Dirichlet(n_plants, 2.5))
    for t in 1:n_periods, i in 1:n_plants
        plant_capacity[i, t] = multiplier * period_need[t] * plant_shares[i] + rand(Uniform(5.0, 25.0))
    end

    return SupplyChainNetworkPlanningProblem(n_plants, n_customers, n_products, n_periods,
        production_cost, shipment_cost, inventory_cost, unmet_penalty, max_unmet, demand, plant_capacity,
        product_capacity_use, initial_inventory, arc_open)
end

function build_model(prob::SupplyChainNetworkPlanningProblem)
    model = Model()
    P, C, K, T = prob.n_plants, prob.n_customers, prob.n_products, prob.n_periods
    @variable(model, produce[1:P, 1:K, 1:T] >= 0)
    @variable(model, ship[1:P, 1:C, 1:K, 1:T] >= 0)
    @variable(model, inventory[1:P, 1:K, 1:T] >= 0)
    @variable(model, 0 <= unmet[j=1:C, k=1:K, t=1:T] <= prob.max_unmet[j,k,t])

    @objective(model, Min,
        sum(prob.production_cost[i,k,t] * produce[i,k,t] for i in 1:P, k in 1:K, t in 1:T) +
        sum(prob.shipment_cost[i,j,k,t] * ship[i,j,k,t] for i in 1:P, j in 1:C, k in 1:K, t in 1:T) +
        sum(prob.inventory_cost[i,k,t] * inventory[i,k,t] for i in 1:P, k in 1:K, t in 1:T) +
        sum(prob.unmet_penalty[j,k,t] * unmet[j,k,t] for j in 1:C, k in 1:K, t in 1:T))

    for i in 1:P, k in 1:K, t in 1:T
        previous = t == 1 ? prob.initial_inventory[i,k] : inventory[i,k,t-1]
        @constraint(model, previous + produce[i,k,t] - sum(ship[i,j,k,t] for j in 1:C) == inventory[i,k,t])
    end
    for j in 1:C, k in 1:K, t in 1:T
        @constraint(model, sum(ship[i,j,k,t] for i in 1:P) + unmet[j,k,t] >= prob.demand[j,k,t])
    end
    for i in 1:P, t in 1:T
        @constraint(model, sum(prob.product_capacity_use[k] * produce[i,k,t] for k in 1:K) <= prob.plant_capacity[i,t])
    end
    for i in 1:P, j in 1:C, k in 1:K, t in 1:T
        if !prob.arc_open[i,j,k]
            @constraint(model, ship[i,j,k,t] == 0)
        end
    end
    return model
end

register_variant(:supply_chain, :network_planning, SupplyChainNetworkPlanningProblem,
    "Multi-period, multi-product supply-chain network planning with production, shipments, inventory, capacity, sparse arcs, and soft unmet demand.")
