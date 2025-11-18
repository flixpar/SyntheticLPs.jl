using JuMP
using Random
using Distributions

"""
    CrossDocking <: ProblemGenerator

Generator for cross-docking problems with transfer optimization.

Cross-docking involves receiving goods from suppliers and immediately shipping
them to customers with minimal storage time. The challenge is coordinating
inbound and outbound flows through limited dock capacity.

# Fields
- `n_suppliers::Int`: Number of suppliers (inbound shipments)
- `n_customers::Int`: Number of customers (outbound destinations)
- `n_docks::Int`: Number of loading/unloading docks
- `n_products::Int`: Number of product types
- `inbound_volumes::Dict{Tuple{Int,Int},Float64}`: Volume from supplier s of product p
- `outbound_demands::Dict{Tuple{Int,Int},Float64}`: Demand from customer c for product p
- `dock_capacities::Vector{Float64}`: Throughput capacity of each dock
- `inbound_costs::Matrix{Float64}`: Cost to receive from supplier at dock
- `outbound_costs::Matrix{Float64}`: Cost to ship from dock to customer
- `transfer_costs::Matrix{Float64}`: Cost to transfer between docks (sorting)
- `storage_costs::Vector{Float64}`: Cost per unit for temporary storage at dock
- `max_storage_time::Float64`: Maximum allowed storage time (hours)
- `dock_processing_times::Vector{Float64}`: Processing time per unit at each dock
"""
struct CrossDocking <: ProblemGenerator
    n_suppliers::Int
    n_customers::Int
    n_docks::Int
    n_products::Int
    inbound_volumes::Dict{Tuple{Int,Int},Float64}
    outbound_demands::Dict{Tuple{Int,Int},Float64}
    dock_capacities::Vector{Float64}
    inbound_costs::Matrix{Float64}
    outbound_costs::Matrix{Float64}
    transfer_costs::Matrix{Float64}
    storage_costs::Vector{Float64}
    max_storage_time::Float64
    dock_processing_times::Vector{Float64}
end

"""
    CrossDocking(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a cross-docking problem instance.

# Arguments
- `target_variables`: Target number of variables
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility

# Details
Variables: inbound assignments + outbound assignments + transfers
Target: (n_suppliers×n_products)×n_docks + (n_customers×n_products)×n_docks + transfers
"""
function CrossDocking(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Determine scale
    if target_variables <= 150
        min_supp, max_supp = 2, 10
        min_cust, max_cust = 2, 12
        min_docks, max_docks = 2, 6
        n_products = rand(2:4)
        volume_range = (50.0, 500.0)
        demand_range = (30.0, 300.0)
        cost_range = (5.0, 50.0)
    elseif target_variables <= 800
        min_supp, max_supp = 3, 20
        min_cust, max_cust = 3, 25
        min_docks, max_docks = 3, 10
        n_products = rand(3:6)
        volume_range = (100.0, 1000.0)
        demand_range = (80.0, 800.0)
        cost_range = (10.0, 100.0)
    else
        min_supp, max_supp = 5, 40
        min_cust, max_cust = 5, 50
        min_docks, max_docks = 5, 15
        n_products = rand(4:10)
        volume_range = (200.0, 3000.0)
        demand_range = (150.0, 2000.0)
        cost_range = (20.0, 200.0)
    end

    # Solve for dimensions
    # Variables: (S×P)×D (inbound) + (C×P)×D (outbound) + D² (transfers)
    # Simplified: target ≈ (S×P + C×P)×D + D²
    best_config = (min_supp, min_cust, min_docks)
    best_error = Inf

    for n_docks in min_docks:max_docks
        # Given docks, solve for suppliers and customers
        # target ≈ (S×P + C×P)×D + D²
        # target - D² ≈ (S×P + C×P)×D
        target_remaining = target_variables - n_docks * n_docks
        if target_remaining <= 0
            continue
        end

        flows_per_dock = target_remaining / n_docks
        total_sp_cp = flows_per_dock / 1  # (S×P + C×P)

        # Split between suppliers and customers
        ratio = rand(0.4:0.1:0.6)
        n_supp_approx = round(Int, (total_sp_cp * ratio) / n_products)
        n_cust_approx = round(Int, (total_sp_cp * (1 - ratio)) / n_products)

        n_supp = clamp(n_supp_approx, min_supp, max_supp)
        n_cust = clamp(n_cust_approx, min_cust, max_cust)

        actual_vars = (n_supp * n_products + n_cust * n_products) * n_docks + n_docks * n_docks
        error = abs(actual_vars - target_variables) / target_variables

        if error < best_error
            best_error = error
            best_config = (n_supp, n_cust, n_docks)
        end
    end

    n_suppliers, n_customers, n_docks = best_config

    # Generate inbound volumes (sparse - not all suppliers have all products)
    min_vol, max_vol = volume_range
    inbound_volumes = Dict{Tuple{Int,Int},Float64}()
    supply_density = rand(0.4:0.05:0.7)  # 40-70% of supplier-product pairs

    for s in 1:n_suppliers
        for p in 1:n_products
            if rand() < supply_density
                volume = rand(Uniform(min_vol, max_vol))
                inbound_volumes[(s, p)] = round(volume, digits=2)
            end
        end
    end

    # Generate outbound demands (sparse)
    min_dem, max_dem = demand_range
    outbound_demands = Dict{Tuple{Int,Int},Float64}()
    demand_density = rand(0.4:0.05:0.7)

    for c in 1:n_customers
        for p in 1:n_products
            if rand() < demand_density
                demand = rand(Uniform(min_dem, max_dem))
                outbound_demands[(c, p)] = round(demand, digits=2)
            end
        end
    end

    # Ensure supply can meet demand for each product
    for p in 1:n_products
        total_supply_p = sum(get(inbound_volumes, (s, p), 0.0) for s in 1:n_suppliers)
        total_demand_p = sum(get(outbound_demands, (c, p), 0.0) for c in 1:n_customers)

        if total_supply_p < total_demand_p
            # Scale up supplies for this product
            scale = (total_demand_p * 1.1) / max(total_supply_p, 1e-6)
            for s in 1:n_suppliers
                if haskey(inbound_volumes, (s, p))
                    inbound_volumes[(s, p)] = round(inbound_volumes[(s, p)] * scale, digits=2)
                end
            end
        end
    end

    # Dock capacities (throughput per time period)
    total_flow = sum(values(inbound_volumes)) + sum(values(outbound_demands))
    avg_dock_capacity = (total_flow / n_docks) * rand(1.3:0.1:2.0)
    dock_capacities = [round(avg_dock_capacity * (0.8 + 0.4 * rand()), digits=2)
                       for _ in 1:n_docks]

    # Costs
    min_cost, max_cost = cost_range

    # Inbound costs (supplier to dock)
    inbound_costs = [round(rand(Uniform(min_cost, max_cost)), digits=2)
                     for _ in 1:n_suppliers, _ in 1:n_docks]

    # Outbound costs (dock to customer)
    outbound_costs = [round(rand(Uniform(min_cost, max_cost)), digits=2)
                      for _ in 1:n_docks, _ in 1:n_customers]

    # Transfer costs (dock to dock for sorting/consolidation)
    transfer_cost_range = (min_cost * 0.2, max_cost * 0.3)
    transfer_costs = zeros(n_docks, n_docks)
    for i in 1:n_docks
        for j in 1:n_docks
            if i != j
                transfer_costs[i, j] = round(rand(Uniform(transfer_cost_range...)), digits=2)
            end
        end
    end

    # Storage costs (discourage long storage)
    storage_cost_range = (min_cost * 0.5, max_cost * 0.8)
    storage_costs = [round(rand(Uniform(storage_cost_range...)), digits=2) for _ in 1:n_docks]

    # Maximum storage time (cross-docking is about speed)
    max_storage_time = rand(2.0:0.5:8.0)  # hours

    # Processing times per unit at each dock
    dock_processing_times = [round(rand(Uniform(0.5, 3.0)), digits=2) for _ in 1:n_docks]

    # Adjust for feasibility
    if feasibility_status == feasible
        # Ensure sufficient dock capacity
        total_dock_capacity = sum(dock_capacities)
        total_throughput_needed = (sum(values(inbound_volumes)) + sum(values(outbound_demands))) / 2
        if total_dock_capacity < total_throughput_needed
            scale = (total_throughput_needed * 1.3) / total_dock_capacity
            dock_capacities .*= scale
            dock_capacities = round.(dock_capacities, digits=2)
        end

    elseif feasibility_status == infeasible
        # Create infeasibility
        choice = rand()
        if choice < 0.5
            # Insufficient dock capacity
            scale = rand(0.3:0.05:0.7)
            dock_capacities .*= scale
            dock_capacities = round.(dock_capacities, digits=2)
        else
            # Demand exceeds supply for some product
            if n_products > 0
                critical_product = rand(1:n_products)
                # Reduce supply for this product
                for s in 1:n_suppliers
                    if haskey(inbound_volumes, (s, critical_product))
                        inbound_volumes[(s, critical_product)] *= rand(0.3:0.05:0.6)
                        inbound_volumes[(s, critical_product)] = round(inbound_volumes[(s, critical_product)], digits=2)
                    end
                end
            end
        end
    end

    return CrossDocking(
        n_suppliers,
        n_customers,
        n_docks,
        n_products,
        inbound_volumes,
        outbound_demands,
        dock_capacities,
        inbound_costs,
        outbound_costs,
        transfer_costs,
        storage_costs,
        max_storage_time,
        dock_processing_times
    )
end

"""
    build_model(prob::CrossDocking)

Build a JuMP model for the cross-docking problem.

# Arguments
- `prob`: CrossDocking instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::CrossDocking)
    model = Model()

    S = prob.n_suppliers
    C = prob.n_customers
    D = prob.n_docks
    P = prob.n_products

    # Decision variables

    # x_in[s,p,d] = amount of product p from supplier s received at dock d
    @variable(model, x_in[s=1:S, p=1:P, d=1:D] >= 0)

    # x_out[d,c,p] = amount of product p shipped from dock d to customer c
    @variable(model, x_out[d=1:D, c=1:C, p=1:P] >= 0)

    # x_transfer[d1,d2,p] = amount of product p transferred from dock d1 to d2
    @variable(model, x_transfer[d1=1:D, d2=1:D, p=1:P] >= 0)

    # Objective: minimize total cost
    @objective(model, Min,
        # Inbound receiving costs
        sum(prob.inbound_costs[s, d] * x_in[s, p, d]
            for s in 1:S, p in 1:P, d in 1:D if haskey(prob.inbound_volumes, (s, p))) +
        # Outbound shipping costs
        sum(prob.outbound_costs[d, c] * x_out[d, c, p]
            for d in 1:D, c in 1:C, p in 1:P if haskey(prob.outbound_demands, (c, p))) +
        # Transfer costs
        sum(prob.transfer_costs[d1, d2] * x_transfer[d1, d2, p]
            for d1 in 1:D, d2 in 1:D, p in 1:P if d1 != d2) +
        # Storage costs (proportional to volume)
        sum(prob.storage_costs[d] * sum(x_in[s, p, d] for s in 1:S, p in 1:P if haskey(prob.inbound_volumes, (s, p)))
            for d in 1:D)
    )

    # Constraints

    # Inbound volume constraints
    for s in 1:S
        for p in 1:P
            if haskey(prob.inbound_volumes, (s, p))
                @constraint(model,
                    sum(x_in[s, p, d] for d in 1:D) <= prob.inbound_volumes[(s, p)]
                )
            else
                # No supply for this product from this supplier
                for d in 1:D
                    @constraint(model, x_in[s, p, d] == 0)
                end
            end
        end
    end

    # Outbound demand constraints
    for c in 1:C
        for p in 1:P
            if haskey(prob.outbound_demands, (c, p))
                @constraint(model,
                    sum(x_out[d, c, p] for d in 1:D) >= prob.outbound_demands[(c, p)]
                )
            else
                # No demand for this product from this customer
                for d in 1:D
                    @constraint(model, x_out[d, c, p] == 0)
                end
            end
        end
    end

    # Flow balance at each dock for each product
    for d in 1:D
        for p in 1:P
            # Inflow = outflow + transfers out - transfers in
            inflow = sum(x_in[s, p, d] for s in 1:S) +
                    sum(x_transfer[d2, d, p] for d2 in 1:D if d2 != d)

            outflow = sum(x_out[d, c, p] for c in 1:C) +
                     sum(x_transfer[d, d2, p] for d2 in 1:D if d2 != d)

            @constraint(model, inflow >= outflow)
        end
    end

    # Dock capacity constraints (throughput)
    for d in 1:D
        # Total volume handled at dock d
        @constraint(model,
            sum(x_in[s, p, d] for s in 1:S, p in 1:P) +
            sum(x_out[d, c, p] for c in 1:C, p in 1:P) +
            sum(x_transfer[d, d2, p] for d2 in 1:D, p in 1:P if d2 != d) <=
            prob.dock_capacities[d]
        )
    end

    # No self-transfers
    for d in 1:D
        for p in 1:P
            @constraint(model, x_transfer[d, d, p] == 0)
        end
    end

    return model
end

# Register the problem type
register_problem(
    :cross_docking,
    CrossDocking,
    "Cross-docking problem optimizing transfer with minimal storage time"
)
