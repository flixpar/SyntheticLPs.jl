using JuMP
using Random
using StatsBase
using Distributions

"""
Supply chain problem variants.

# Variants
- `sc_standard`: Basic supply chain with multiple transport modes
- `sc_single_source`: Each customer served by exactly one facility
- `sc_lead_time`: Maximum delivery time constraints
- `sc_carbon`: Carbon emission limits on transportation
- `sc_multi_product`: Multiple product types with different requirements
- `sc_risk_diverse`: Limit sourcing from any single region
"""
@enum SupplyChainVariant begin
    sc_standard
    sc_single_source
    sc_lead_time
    sc_carbon
    sc_multi_product
    sc_risk_diverse
end

"""
    SupplyChainProblem <: ProblemGenerator

Generator for supply chain optimization problems with multiple variants.
"""
struct SupplyChainProblem <: ProblemGenerator
    n_facilities::Int
    n_customers::Int
    transport_modes::Vector{String}
    facility_locs::Vector{Tuple{Float64,Float64}}
    customer_locs::Vector{Tuple{Float64,Float64}}
    cluster_centers::Vector{Tuple{Float64,Float64}}
    cluster_weights::Vector{Float64}
    fixed_costs::Dict{Int, Float64}
    demands::Dict{Int, Float64}
    capacities::Dict{Int, Float64}
    transport_costs::Dict{Tuple{Int,Int,String}, Float64}
    mode_capacities::Dict{String, Float64}
    total_demand::Float64
    variant::SupplyChainVariant
    # Lead time variant
    lead_times::Union{Dict{Tuple{Int,Int,String}, Float64}, Nothing}
    max_lead_time::Union{Float64, Nothing}
    # Carbon variant
    carbon_emissions::Union{Dict{Tuple{Int,Int,String}, Float64}, Nothing}
    carbon_limit::Union{Float64, Nothing}
    # Multi-product variant
    n_products::Int
    product_demands::Union{Dict{Tuple{Int,Int}, Float64}, Nothing}
    product_capacities::Union{Dict{Tuple{Int,Int}, Float64}, Nothing}
    # Risk diversification variant
    n_regions::Int
    facility_regions::Union{Dict{Int, Int}, Nothing}
    max_region_fraction::Union{Float64, Nothing}
end

# Backwards compatibility
function SupplyChainProblem(n_facilities::Int, n_customers::Int, transport_modes::Vector{String},
                            facility_locs::Vector{Tuple{Float64,Float64}},
                            customer_locs::Vector{Tuple{Float64,Float64}},
                            cluster_centers::Vector{Tuple{Float64,Float64}},
                            cluster_weights::Vector{Float64},
                            fixed_costs::Dict{Int, Float64}, demands::Dict{Int, Float64},
                            capacities::Dict{Int, Float64},
                            transport_costs::Dict{Tuple{Int,Int,String}, Float64},
                            mode_capacities::Dict{String, Float64}, total_demand::Float64)
    SupplyChainProblem(
        n_facilities, n_customers, transport_modes, facility_locs, customer_locs,
        cluster_centers, cluster_weights, fixed_costs, demands, capacities,
        transport_costs, mode_capacities, total_demand, sc_standard,
        nothing, nothing, nothing, nothing,
        1, nothing, nothing, 1, nothing, nothing
    )
end

"""
    SupplyChainProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int;
                       variant::SupplyChainVariant=sc_standard)

Construct a supply chain problem instance with the specified variant.
"""
function SupplyChainProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int;
                            variant::SupplyChainVariant=sc_standard)
    Random.seed!(seed)

    # Determine problem dimensions
    avg_density = 0.65

    if target_variables <= 250
        n_facilities = rand(DiscreteUniform(3, 8))
        n_customers = rand(DiscreteUniform(15, 35))
        n_transport_modes = rand(DiscreteUniform(1, 2))
        grid_width = rand(Uniform(200.0, 800.0))
        grid_height = rand(Uniform(200.0, 800.0))
        infrastructure_density = rand(Beta(5, 2)) * 0.3 + 0.7
        clustering_factor = rand(Beta(3, 2)) * 0.6 + 0.25
        min_fixed_cost = max(100000.0, rand(LogNormal(log(300000), 0.5)))
        max_fixed_cost = min_fixed_cost * rand(Uniform(1.8, 3.5))
        base_demand = rand(Uniform(80.0, 150.0))
        min_demand, max_demand = base_demand, base_demand * rand(Uniform(3.0, 8.0))
    elseif target_variables <= 1000
        n_facilities = rand(DiscreteUniform(6, 18))
        n_customers = rand(DiscreteUniform(25, 65))
        n_transport_modes = rand(DiscreteUniform(2, 3))
        grid_width = rand(Uniform(800.0, 2000.0))
        grid_height = rand(Uniform(800.0, 2000.0))
        infrastructure_density = rand(Beta(3, 2)) * 0.4 + 0.5
        clustering_factor = rand(Beta(2, 3)) * 0.5 + 0.2
        min_fixed_cost = max(300000.0, rand(LogNormal(log(800000), 0.6)))
        max_fixed_cost = min_fixed_cost * rand(Uniform(2.0, 4.0))
        base_demand = rand(Uniform(150.0, 300.0))
        min_demand, max_demand = base_demand, base_demand * rand(Uniform(4.0, 12.0))
    else
        n_facilities = rand(DiscreteUniform(12, 40))
        n_customers = rand(DiscreteUniform(60, 200))
        n_transport_modes = rand(DiscreteUniform(3, 4))
        grid_width = rand(Uniform(2000.0, 5000.0))
        grid_height = rand(Uniform(2000.0, 5000.0))
        infrastructure_density = rand(Beta(2, 3)) * 0.4 + 0.4
        clustering_factor = rand(Beta(1, 3)) * 0.4 + 0.15
        min_fixed_cost = max(500000.0, rand(LogNormal(log(1500000), 0.7)))
        max_fixed_cost = min_fixed_cost * rand(Uniform(2.5, 5.0))
        base_demand = rand(Uniform(300.0, 600.0))
        min_demand, max_demand = base_demand, base_demand * rand(Uniform(6.0, 20.0))
    end

    capacity_factor = rand(Uniform(1.2, 2.2))
    mode_capacity_factor = rand(Uniform(0.25, 0.65))

    # Transport modes
    all_transport_modes = ["truck", "rail", "ship", "air"]
    transport_base_costs = Dict("truck" => rand(Gamma(4, 0.25)), "rail" => rand(Gamma(3, 0.2)),
                                "ship" => rand(Gamma(2, 0.15)), "air" => rand(Gamma(6, 0.5)))
    transport_modes = sample(all_transport_modes, min(n_transport_modes, 4), replace=false)

    # Generate clusters
    n_clusters = max(2, round(Int, sqrt(n_customers) * clustering_factor))
    cluster_centers = [(grid_width * rand(), grid_height * rand()) for _ in 1:n_clusters]
    cluster_weights = rand(Dirichlet(ones(n_clusters)))

    # Generate facility locations
    facility_locs = Vector{Tuple{Float64,Float64}}()
    for _ in 1:n_facilities
        if rand() < 0.4
            center = rand(cluster_centers)
            x = clamp(center[1] + rand(Normal(0, grid_width * 0.12)), 0, grid_width)
            y = clamp(center[2] + rand(Normal(0, grid_height * 0.12)), 0, grid_height)
        else
            x = grid_width * rand(Beta(1.5, 1.5))
            y = grid_height * rand(Beta(1.5, 1.5))
        end
        push!(facility_locs, (x, y))
    end

    # Generate customer locations
    customer_locs = Vector{Tuple{Float64,Float64}}()
    for _ in 1:n_customers
        cluster_idx = sample(1:n_clusters, Weights(cluster_weights))
        center = cluster_centers[cluster_idx]
        spread = rand(LogNormal(log(grid_width * (1 - clustering_factor) * 0.08), 0.3))
        x = clamp(center[1] + rand(Normal(0, spread)), 0, grid_width)
        y = clamp(center[2] + rand(Normal(0, spread)), 0, grid_height)
        push!(customer_locs, (x, y))
    end

    # Generate fixed costs
    fixed_costs = Dict{Int, Float64}()
    for f in 1:n_facilities
        distances = [sqrt((facility_locs[f][1] - c[1])^2 + (facility_locs[f][2] - c[2])^2) for c in customer_locs]
        market_potential = sum(exp.(-distances ./ (grid_width * 0.2)))
        base_cost = min_fixed_cost + (max_fixed_cost - min_fixed_cost) * (0.2 + 0.5 * market_potential / n_customers)
        fixed_costs[f] = base_cost * rand(LogNormal(log(1.0), 0.25))
    end

    # Generate demands
    demands = Dict{Int, Float64}()
    for c in 1:n_customers
        distances = [sqrt((customer_locs[c][1] - center[1])^2 + (customer_locs[c][2] - center[2])^2) for center in cluster_centers]
        _, cluster_idx = findmin(distances)
        base_demand_val = min_demand + (max_demand - min_demand) * (0.2 + 0.8 * cluster_weights[cluster_idx])
        demands[c] = base_demand_val * rand(LogNormal(log(1.0), 0.4))
    end

    total_demand = sum(values(demands))
    avg_capacity = (total_demand / n_facilities) * capacity_factor

    # Generate capacities
    capacities = Dict{Int, Float64}()
    for f in 1:n_facilities
        relative_cost = (fixed_costs[f] - minimum(values(fixed_costs))) / max(1, maximum(values(fixed_costs)) - minimum(values(fixed_costs)))
        capacities[f] = avg_capacity * (0.6 + 0.8 * relative_cost) * rand(Gamma(3, 1/3))
    end

    # Generate transport costs
    transport_costs = Dict{Tuple{Int,Int,String}, Float64}()
    infrastructure = Dict{Tuple{Int,Int,String}, Bool}()

    for f in 1:n_facilities, c in 1:n_customers
        distance = sqrt((facility_locs[f][1] - customer_locs[c][1])^2 + (facility_locs[f][2] - customer_locs[c][2])^2)
        for mode in transport_modes
            prob_available = mode == "truck" ? 0.98 :
                            mode == "rail" ? min(0.8, 0.3 + 0.5 * (distance / sqrt(grid_width^2 + grid_height^2))) :
                            mode == "ship" ? (any(loc -> abs(loc[2]) < grid_height * 0.1, [facility_locs[f], customer_locs[c]]) ? 0.8 : 0.0) :
                            (distance > sqrt(grid_width^2 + grid_height^2) * 0.3 ? 0.7 : 0.2)

            infrastructure[(f,c,mode)] = rand() < prob_available * infrastructure_density

            if infrastructure[(f,c,mode)]
                base_cost = get(transport_base_costs, mode, 1.0)
                transport_costs[(f,c,mode)] = base_cost * distance * rand(LogNormal(log(1.0), 0.15))
            end
        end
    end

    # Generate mode capacities
    mode_capacities = Dict{String, Float64}()
    for mode in transport_modes
        base_capacity = total_demand * mode_capacity_factor
        mult = mode == "truck" ? rand(Gamma(4, 0.25)) : mode == "rail" ? rand(Gamma(6, 0.33)) :
               mode == "ship" ? rand(Gamma(9, 0.33)) : rand(Gamma(2, 0.25))
        mode_capacities[mode] = base_capacity * mult
    end

    transport_costs = Dict(k => v for (k,v) in transport_costs if get(infrastructure, k, false))

    # Initialize variant-specific fields
    lead_times = nothing
    max_lead_time = nothing
    carbon_emissions = nothing
    carbon_limit = nothing
    n_products = 1
    product_demands = nothing
    product_capacities = nothing
    n_regions = 1
    facility_regions = nothing
    max_region_fraction = nothing

    # Generate variant-specific data
    if variant == sc_lead_time
        # Lead times based on distance and mode
        lead_times = Dict{Tuple{Int,Int,String}, Float64}()
        for (key, _) in transport_costs
            f, c, mode = key
            distance = sqrt((facility_locs[f][1] - customer_locs[c][1])^2 +
                           (facility_locs[f][2] - customer_locs[c][2])^2)
            speed = mode == "truck" ? 60.0 : mode == "rail" ? 40.0 :
                    mode == "ship" ? 20.0 : 500.0
            lead_times[key] = distance / speed * rand(Uniform(0.9, 1.1))
        end
        max_lead_time = maximum(values(lead_times)) * rand(Uniform(0.6, 0.9))

    elseif variant == sc_carbon
        # Carbon emissions per unit transported
        carbon_emissions = Dict{Tuple{Int,Int,String}, Float64}()
        for (key, _) in transport_costs
            f, c, mode = key
            distance = sqrt((facility_locs[f][1] - customer_locs[c][1])^2 +
                           (facility_locs[f][2] - customer_locs[c][2])^2)
            emission_rate = mode == "truck" ? 0.1 : mode == "rail" ? 0.03 :
                           mode == "ship" ? 0.02 : 0.5
            carbon_emissions[key] = emission_rate * distance * rand(Uniform(0.9, 1.1))
        end
        # Set carbon limit as fraction of worst-case emissions
        max_possible = sum(values(carbon_emissions)) / length(carbon_emissions) * total_demand
        carbon_limit = max_possible * rand(Uniform(0.4, 0.7))

    elseif variant == sc_multi_product
        n_products = rand(2:min(4, max(2, n_facilities ÷ 2)))

        # Product-specific demands
        product_demands = Dict{Tuple{Int,Int}, Float64}()
        product_split = rand(Dirichlet(ones(n_products)))
        for c in 1:n_customers, p in 1:n_products
            product_demands[(c, p)] = demands[c] * product_split[p] * rand(Uniform(0.8, 1.2))
        end

        # Product-specific capacities at facilities
        product_capacities = Dict{Tuple{Int,Int}, Float64}()
        for f in 1:n_facilities, p in 1:n_products
            product_capacities[(f, p)] = capacities[f] / n_products * rand(Uniform(0.8, 1.2))
        end

    elseif variant == sc_risk_diverse
        # Divide facilities into regions
        n_regions = rand(2:min(4, n_facilities))
        facility_regions = Dict{Int, Int}()
        for f in 1:n_facilities
            facility_regions[f] = ((f - 1) % n_regions) + 1
        end
        max_region_fraction = rand(Uniform(0.4, 0.6))
    end

    # Handle feasibility
    if feasibility_status == feasible
        # Ensure connectivity and capacity
        fallback_mode = ("truck" in transport_modes) ? "truck" : transport_modes[1]
        K = min(max(3, ceil(Int, n_facilities ÷ 3)), n_facilities)

        for c in 1:n_customers
            dvec = [sqrt((facility_locs[f][1] - customer_locs[c][1])^2 +
                        (facility_locs[f][2] - customer_locs[c][2])^2) for f in 1:n_facilities]
            nearest_idxs = sortperm(dvec)[1:K]
            for f in nearest_idxs
                if !haskey(transport_costs, (f, c, fallback_mode))
                    transport_costs[(f, c, fallback_mode)] = transport_base_costs[fallback_mode] * dvec[f] * rand(Uniform(0.9, 1.1))
                    if lead_times !== nothing
                        lead_times[(f, c, fallback_mode)] = dvec[f] / 60.0 * rand(Uniform(0.9, 1.1))
                    end
                    if carbon_emissions !== nothing
                        carbon_emissions[(f, c, fallback_mode)] = 0.1 * dvec[f] * rand(Uniform(0.9, 1.1))
                    end
                end
            end
        end

        # Ensure mode capacity
        if mode_capacities[fallback_mode] < 1.05 * total_demand
            mode_capacities[fallback_mode] = 1.05 * total_demand
        end

        # Ensure facility capacity
        if sum(values(capacities)) < total_demand
            scale = 1.1 * total_demand / sum(values(capacities))
            for f in 1:n_facilities
                capacities[f] *= scale
            end
        end

        # For lead time variant, increase max lead time if needed
        if variant == sc_lead_time && lead_times !== nothing && max_lead_time !== nothing
            min_feasible_time = minimum(minimum(lead_times[k] for (k,_) in transport_costs if k[2] == c) for c in 1:n_customers)
            if max_lead_time < min_feasible_time
                max_lead_time = min_feasible_time * 1.2
            end
        end

    elseif feasibility_status == infeasible
        scenario = rand(1:3)
        if scenario == 1
            # Mode capacity shortage
            for m in transport_modes
                mode_capacities[m] *= 0.4
            end
        elseif scenario == 2
            # Facility capacity shortage
            for f in 1:n_facilities
                capacities[f] *= 0.3
            end
        else
            if variant == sc_lead_time && max_lead_time !== nothing
                max_lead_time *= 0.1
            elseif variant == sc_carbon && carbon_limit !== nothing
                carbon_limit *= 0.1
            else
                for f in 1:n_facilities
                    capacities[f] *= 0.3
                end
            end
        end
    end

    return SupplyChainProblem(
        n_facilities, n_customers, transport_modes, facility_locs, customer_locs,
        cluster_centers, cluster_weights, fixed_costs, demands, capacities,
        transport_costs, mode_capacities, total_demand, variant,
        lead_times, max_lead_time, carbon_emissions, carbon_limit,
        n_products, product_demands, product_capacities,
        n_regions, facility_regions, max_region_fraction
    )
end

"""
    build_model(prob::SupplyChainProblem)

Build a JuMP model for the supply chain problem based on its variant.
"""
function build_model(prob::SupplyChainProblem)
    model = Model()

    valid_combinations = [(f,c,m) for f in 1:prob.n_facilities, c in 1:prob.n_customers, m in prob.transport_modes
                          if haskey(prob.transport_costs, (f,c,m))]

    if prob.variant == sc_standard
        @variable(model, y[1:prob.n_facilities], Bin)
        @variable(model, x[valid_combinations] >= 0)

        @objective(model, Min,
            sum(prob.fixed_costs[f] * y[f] for f in 1:prob.n_facilities) +
            sum(prob.transport_costs[combo] * x[combo] for combo in valid_combinations))

        for c in 1:prob.n_customers
            combos = filter(combo -> combo[2] == c, valid_combinations)
            @constraint(model, sum(x[combo] for combo in combos) >= prob.demands[c])
        end

        for f in 1:prob.n_facilities
            combos = filter(combo -> combo[1] == f, valid_combinations)
            @constraint(model, sum(x[combo] for combo in combos) <= prob.capacities[f] * y[f])
        end

        for m in prob.transport_modes
            combos = filter(combo -> combo[3] == m, valid_combinations)
            @constraint(model, sum(x[combo] for combo in combos) <= prob.mode_capacities[m])
        end

    elseif prob.variant == sc_single_source
        @variable(model, y[1:prob.n_facilities], Bin)
        @variable(model, x[valid_combinations] >= 0)
        @variable(model, z[1:prob.n_facilities, 1:prob.n_customers], Bin)  # Assignment

        @objective(model, Min,
            sum(prob.fixed_costs[f] * y[f] for f in 1:prob.n_facilities) +
            sum(prob.transport_costs[combo] * x[combo] for combo in valid_combinations))

        # Each customer assigned to exactly one facility
        for c in 1:prob.n_customers
            @constraint(model, sum(z[f, c] for f in 1:prob.n_facilities) == 1)
        end

        # Flow only from assigned facility
        for (f, c, m) in valid_combinations
            @constraint(model, x[(f,c,m)] <= prob.demands[c] * z[f, c])
        end

        for c in 1:prob.n_customers
            combos = filter(combo -> combo[2] == c, valid_combinations)
            @constraint(model, sum(x[combo] for combo in combos) >= prob.demands[c])
        end

        for f in 1:prob.n_facilities
            combos = filter(combo -> combo[1] == f, valid_combinations)
            @constraint(model, sum(x[combo] for combo in combos) <= prob.capacities[f] * y[f])
        end

    elseif prob.variant == sc_lead_time
        @variable(model, y[1:prob.n_facilities], Bin)
        @variable(model, x[valid_combinations] >= 0)

        @objective(model, Min,
            sum(prob.fixed_costs[f] * y[f] for f in 1:prob.n_facilities) +
            sum(prob.transport_costs[combo] * x[combo] for combo in valid_combinations))

        for c in 1:prob.n_customers
            combos = filter(combo -> combo[2] == c, valid_combinations)
            @constraint(model, sum(x[combo] for combo in combos) >= prob.demands[c])
        end

        for f in 1:prob.n_facilities
            combos = filter(combo -> combo[1] == f, valid_combinations)
            @constraint(model, sum(x[combo] for combo in combos) <= prob.capacities[f] * y[f])
        end

        # Lead time constraint: only use routes within max lead time
        for combo in valid_combinations
            if prob.lead_times[combo] > prob.max_lead_time
                @constraint(model, x[combo] == 0)
            end
        end

    elseif prob.variant == sc_carbon
        @variable(model, y[1:prob.n_facilities], Bin)
        @variable(model, x[valid_combinations] >= 0)

        @objective(model, Min,
            sum(prob.fixed_costs[f] * y[f] for f in 1:prob.n_facilities) +
            sum(prob.transport_costs[combo] * x[combo] for combo in valid_combinations))

        for c in 1:prob.n_customers
            combos = filter(combo -> combo[2] == c, valid_combinations)
            @constraint(model, sum(x[combo] for combo in combos) >= prob.demands[c])
        end

        for f in 1:prob.n_facilities
            combos = filter(combo -> combo[1] == f, valid_combinations)
            @constraint(model, sum(x[combo] for combo in combos) <= prob.capacities[f] * y[f])
        end

        # Carbon limit constraint
        @constraint(model, sum(prob.carbon_emissions[combo] * x[combo] for combo in valid_combinations) <= prob.carbon_limit)

    elseif prob.variant == sc_multi_product
        @variable(model, y[1:prob.n_facilities], Bin)
        @variable(model, x[valid_combinations, 1:prob.n_products] >= 0)

        @objective(model, Min,
            sum(prob.fixed_costs[f] * y[f] for f in 1:prob.n_facilities) +
            sum(prob.transport_costs[combo] * x[combo, p] for combo in valid_combinations, p in 1:prob.n_products))

        # Demand per product per customer
        for c in 1:prob.n_customers, p in 1:prob.n_products
            combos = filter(combo -> combo[2] == c, valid_combinations)
            @constraint(model, sum(x[combo, p] for combo in combos) >= prob.product_demands[(c, p)])
        end

        # Product capacity per facility
        for f in 1:prob.n_facilities, p in 1:prob.n_products
            combos = filter(combo -> combo[1] == f, valid_combinations)
            @constraint(model, sum(x[combo, p] for combo in combos) <= prob.product_capacities[(f, p)] * y[f])
        end

        # Mode capacity
        for m in prob.transport_modes
            combos = filter(combo -> combo[3] == m, valid_combinations)
            @constraint(model, sum(x[combo, p] for combo in combos, p in 1:prob.n_products) <= prob.mode_capacities[m])
        end

    elseif prob.variant == sc_risk_diverse
        @variable(model, y[1:prob.n_facilities], Bin)
        @variable(model, x[valid_combinations] >= 0)

        @objective(model, Min,
            sum(prob.fixed_costs[f] * y[f] for f in 1:prob.n_facilities) +
            sum(prob.transport_costs[combo] * x[combo] for combo in valid_combinations))

        for c in 1:prob.n_customers
            combos = filter(combo -> combo[2] == c, valid_combinations)
            @constraint(model, sum(x[combo] for combo in combos) >= prob.demands[c])
        end

        for f in 1:prob.n_facilities
            combos = filter(combo -> combo[1] == f, valid_combinations)
            @constraint(model, sum(x[combo] for combo in combos) <= prob.capacities[f] * y[f])
        end

        # Risk diversification: limit sourcing from any single region
        for r in 1:prob.n_regions
            region_facilities = [f for f in 1:prob.n_facilities if prob.facility_regions[f] == r]
            region_combos = filter(combo -> combo[1] in region_facilities, valid_combinations)
            @constraint(model, sum(x[combo] for combo in region_combos) <= prob.max_region_fraction * prob.total_demand)
        end
    end

    return model
end

# Register the problem type
register_problem(
    :supply_chain,
    SupplyChainProblem,
    "Supply chain optimization with variants including standard, single source, lead time, carbon, multi-product, and risk diversification"
)
