using JuMP
using Random
using Distributions

"""
Facility location problem variants.

# Variants
- `fl_standard`: Capacitated facility location with budget
- `fl_uncapacitated`: No capacity limits on facilities
- `fl_p_median`: Open exactly p facilities
- `fl_p_center`: Minimize maximum distance to nearest facility
- `fl_covering`: All customers within coverage radius
- `fl_single_source`: Each customer served by one facility
"""
@enum FacilityLocationVariant begin
    fl_standard
    fl_uncapacitated
    fl_p_median
    fl_p_center
    fl_covering
    fl_single_source
end

"""
    FacilityLocationProblem <: ProblemGenerator

Generator for facility location problems with multiple variants.
"""
struct FacilityLocationProblem <: ProblemGenerator
    n_facilities::Int
    n_customers::Int
    facility_locs::Vector{Tuple{Float64,Float64}}
    customer_locs::Vector{Tuple{Float64,Float64}}
    demands::Dict{Int,Float64}
    fixed_costs::Dict{Int,Float64}
    capacities::Dict{Int,Float64}
    shipping_costs::Dict{Tuple{Int,Int},Float64}
    budget::Float64
    variant::FacilityLocationVariant
    # P-median/center variant
    p_facilities::Union{Int, Nothing}
    # Covering variant
    coverage_radius::Union{Float64, Nothing}
    distances::Union{Dict{Tuple{Int,Int},Float64}, Nothing}
end

# Backwards compatibility
function FacilityLocationProblem(n_facilities::Int, n_customers::Int,
                                 facility_locs::Vector{Tuple{Float64,Float64}},
                                 customer_locs::Vector{Tuple{Float64,Float64}},
                                 demands::Dict{Int,Float64}, fixed_costs::Dict{Int,Float64},
                                 capacities::Dict{Int,Float64},
                                 shipping_costs::Dict{Tuple{Int,Int},Float64}, budget::Float64)
    FacilityLocationProblem(
        n_facilities, n_customers, facility_locs, customer_locs,
        demands, fixed_costs, capacities, shipping_costs, budget, fl_standard,
        nothing, nothing, nothing
    )
end

"""
    FacilityLocationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int;
                            variant::FacilityLocationVariant=fl_standard)

Construct a facility location problem instance with the specified variant.
"""
function FacilityLocationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int;
                                 variant::FacilityLocationVariant=fl_standard)
    Random.seed!(seed)

    # Determine scale and ranges
    if target_variables <= 100
        min_facilities, max_facilities = 2, 20
        min_customers, max_customers = 1, 40
        grid_width = rand(200.0:50.0:800.0)
        grid_height = rand(200.0:50.0:800.0)
        transport_cost_per_km = rand(0.5:0.1:1.2)
        min_demand, max_demand = rand(5.0:1.0:20.0), rand(50.0:10.0:150.0)
        fixed_cost_min, fixed_cost_max = rand(20000.0:5000.0:80000.0), rand(100000.0:20000.0:300000.0)
        capacity_factor = rand(1.1:0.05:1.6)
        budget_factor = rand(0.4:0.05:0.8)
    elseif target_variables <= 1000
        min_facilities, max_facilities = 3, 100
        min_customers, max_customers = 5, 200
        grid_width = rand(500.0:100.0:2000.0)
        grid_height = rand(500.0:100.0:2000.0)
        transport_cost_per_km = rand(0.8:0.1:1.8)
        min_demand, max_demand = rand(10.0:2.0:30.0), rand(80.0:20.0:200.0)
        fixed_cost_min, fixed_cost_max = rand(50000.0:10000.0:150000.0), rand(250000.0:50000.0:600000.0)
        capacity_factor = rand(1.2:0.05:1.8)
        budget_factor = rand(0.5:0.05:0.9)
    else
        min_facilities, max_facilities = 5, 500
        min_customers, max_customers = 10, 2000
        grid_width = rand(1000.0:200.0:5000.0)
        grid_height = rand(1000.0:200.0:5000.0)
        transport_cost_per_km = rand(1.0:0.2:3.0)
        min_demand, max_demand = rand(20.0:5.0:60.0), rand(150.0:50.0:500.0)
        fixed_cost_min, fixed_cost_max = rand(100000.0:20000.0:300000.0), rand(500000.0:100000.0:1500000.0)
        capacity_factor = rand(1.3:0.1:2.0)
        budget_factor = rand(0.6:0.05:0.95)
    end

    # Find optimal n_facilities and n_customers
    best_n_facilities = min_facilities
    best_n_customers = min_customers
    best_error = Inf

    for n_facilities in min_facilities:max_facilities
        n_customers_exact = (target_variables / n_facilities) - 1
        if n_customers_exact >= min_customers && n_customers_exact <= max_customers
            n_customers = round(Int, n_customers_exact)
            actual_vars = n_facilities * (n_customers + 1)
            error = abs(actual_vars - target_variables) / target_variables
            if error < best_error
                best_error = error
                best_n_facilities = n_facilities
                best_n_customers = n_customers
            end
        end
    end

    if best_error > 0.1
        n_facilities_approx = max(min_facilities, min(max_facilities, round(Int, sqrt(target_variables / 4))))
        n_customers_approx = max(min_customers, min(max_customers, round(Int, (target_variables / n_facilities_approx) - 1)))
        best_n_facilities = n_facilities_approx
        best_n_customers = n_customers_approx
    end

    n_facilities = best_n_facilities
    n_customers = best_n_customers

    # Generate locations
    facility_locs = [(grid_width * rand(), grid_height * rand()) for _ in 1:n_facilities]

    n_clusters = max(2, div(n_customers, 20))
    cluster_centers = [(grid_width * rand(), grid_height * rand()) for _ in 1:n_clusters]
    customer_locs = Tuple{Float64,Float64}[]
    for _ in 1:n_customers
        center = rand(cluster_centers)
        x = clamp(center[1] + randn() * (grid_width/10), 0, grid_width)
        y = clamp(center[2] + randn() * (grid_height/10), 0, grid_height)
        push!(customer_locs, (x, y))
    end

    # Generate demands
    demands = Dict{Int,Float64}()
    for c in 1:n_customers
        demands[c] = exp(rand(Normal(log((min_demand + max_demand)/2), 0.5)))
    end

    total_demand = sum(values(demands))
    avg_facility_capacity = (total_demand / n_facilities) * capacity_factor

    # Generate costs and capacities
    fixed_costs = Dict{Int,Float64}()
    capacities = Dict{Int,Float64}()

    for w in 1:n_facilities
        capacity = avg_facility_capacity * (0.8 + 0.4 * rand())
        capacities[w] = capacity
        location_factor = 1.0 + 0.2 * (facility_locs[w][1] / grid_width + facility_locs[w][2] / grid_height)
        fixed_costs[w] = clamp(location_factor * (fixed_cost_min + (capacity/avg_facility_capacity) * (fixed_cost_max - fixed_cost_min)),
                               fixed_cost_min, fixed_cost_max)
    end

    # Shipping costs and distances
    shipping_costs = Dict{Tuple{Int,Int},Float64}()
    distances = Dict{Tuple{Int,Int},Float64}()

    for w in 1:n_facilities, c in 1:n_customers
        distance = sqrt((facility_locs[w][1] - customer_locs[c][1])^2 +
                       (facility_locs[w][2] - customer_locs[c][2])^2)
        distances[(w,c)] = distance
        shipping_costs[(w,c)] = distance * transport_cost_per_km * (0.9 + 0.2 * rand())
    end

    # Initial budget
    budget = sum(values(fixed_costs)) * budget_factor
    original_budget = budget

    # Initialize variant-specific fields
    p_facilities = nothing
    coverage_radius = nothing

    # Generate variant-specific data
    if variant == fl_uncapacitated
        # No capacity limits - set to very high values
        for w in 1:n_facilities
            capacities[w] = total_demand * 10
        end

    elseif variant == fl_p_median
        # Set p as fraction of facilities
        p_facilities = max(1, min(n_facilities - 1, round(Int, n_facilities * rand(0.3:0.1:0.6))))

    elseif variant == fl_p_center
        # Same p calculation
        p_facilities = max(1, min(n_facilities - 1, round(Int, n_facilities * rand(0.3:0.1:0.6))))

    elseif variant == fl_covering
        # Set coverage radius based on grid size and facility density
        avg_distance = sqrt(grid_width * grid_height / n_facilities)
        coverage_radius = avg_distance * rand(0.8:0.1:1.5)

    elseif variant == fl_single_source
        # Same as standard but with assignment constraints
    end

    # Feasibility handling
    solution_status = feasibility_status == feasible ? :feasible :
                     feasibility_status == infeasible ? :infeasible : :all

    caps_vec = [capacities[w] for w in 1:n_facilities]
    costs_vec = [fixed_costs[w] for w in 1:n_facilities]
    ratios = [caps_vec[i] / max(costs_vec[i], eps()) for i in 1:n_facilities]
    order_desc = sortperm(ratios, rev=true)
    total_capacity = sum(caps_vec)

    function greedy_integer_subset_for(cap_target::Float64)
        selected = Int[]
        cum_cap = 0.0
        cum_cost = 0.0
        for idx in order_desc
            push!(selected, idx)
            cum_cap += caps_vec[idx]
            cum_cost += costs_vec[idx]
            if cum_cap + 1e-9 >= cap_target
                return selected, cum_cost
            end
        end
        return selected, cum_cost
    end

    if solution_status == :feasible
        if total_capacity < total_demand
            scale = (1.05 * total_demand) / max(total_capacity, eps())
            for w in 1:n_facilities
                capacities[w] *= scale
            end
        end

        # For p-median/center, ensure p facilities can cover demand
        if variant in [fl_p_median, fl_p_center] && p_facilities !== nothing
            top_p_caps = sort([capacities[w] for w in 1:n_facilities], rev=true)[1:p_facilities]
            if sum(top_p_caps) < total_demand
                scale = 1.1 * total_demand / sum(top_p_caps)
                for w in 1:n_facilities
                    capacities[w] *= scale
                end
            end
        end

        # For covering, ensure some facility can cover each customer
        if variant == fl_covering && coverage_radius !== nothing
            for c in 1:n_customers
                covered = any(distances[(w,c)] <= coverage_radius for w in 1:n_facilities)
                if !covered
                    # Increase coverage radius
                    min_dist = minimum(distances[(w,c)] for w in 1:n_facilities)
                    coverage_radius = max(coverage_radius, min_dist * 1.1)
                end
            end
        end

        selected_idxs, min_int_budget = greedy_integer_subset_for(total_demand)
        slack_factor = 1.02 + 0.23 * rand()
        budget = max(original_budget, min_int_budget * slack_factor)

    elseif solution_status == :infeasible
        scenario = rand(1:3)

        if scenario == 1
            # Budget too low
            budget = minimum(values(fixed_costs)) * 0.5
        elseif scenario == 2
            # Capacity too low
            for w in 1:n_facilities
                capacities[w] *= 0.3
            end
        else
            # Variant-specific infeasibility
            if variant == fl_covering && coverage_radius !== nothing
                coverage_radius *= 0.2
            elseif variant in [fl_p_median, fl_p_center] && p_facilities !== nothing
                p_facilities = 1
                for w in 1:n_facilities
                    capacities[w] *= 0.2
                end
            else
                budget = minimum(values(fixed_costs)) * 0.5
            end
        end
    end

    return FacilityLocationProblem(
        n_facilities, n_customers, facility_locs, customer_locs,
        demands, fixed_costs, capacities, shipping_costs, budget, variant,
        p_facilities, coverage_radius, distances
    )
end

"""
    build_model(prob::FacilityLocationProblem)

Build a JuMP model for the facility location problem based on its variant.
"""
function build_model(prob::FacilityLocationProblem)
    model = Model()

    if prob.variant == fl_standard
        @variable(model, y[1:prob.n_facilities], Bin)
        @variable(model, x[1:prob.n_facilities, 1:prob.n_customers] >= 0)

        @objective(model, Min,
            sum(prob.fixed_costs[w] * y[w] for w in 1:prob.n_facilities) +
            sum(prob.shipping_costs[(w,c)] * x[w,c] for w in 1:prob.n_facilities, c in 1:prob.n_customers))

        for c in 1:prob.n_customers
            @constraint(model, sum(x[w,c] for w in 1:prob.n_facilities) >= prob.demands[c])
        end

        for w in 1:prob.n_facilities
            @constraint(model, sum(x[w,c] for c in 1:prob.n_customers) <= prob.capacities[w] * y[w])
        end

        @constraint(model, sum(prob.fixed_costs[w] * y[w] for w in 1:prob.n_facilities) <= prob.budget)

    elseif prob.variant == fl_uncapacitated
        @variable(model, y[1:prob.n_facilities], Bin)
        @variable(model, x[1:prob.n_facilities, 1:prob.n_customers] >= 0)

        @objective(model, Min,
            sum(prob.fixed_costs[w] * y[w] for w in 1:prob.n_facilities) +
            sum(prob.shipping_costs[(w,c)] * x[w,c] for w in 1:prob.n_facilities, c in 1:prob.n_customers))

        for c in 1:prob.n_customers
            @constraint(model, sum(x[w,c] for w in 1:prob.n_facilities) >= prob.demands[c])
        end

        # Only constraint: can't ship from closed facility
        for w in 1:prob.n_facilities
            M = sum(values(prob.demands))
            @constraint(model, sum(x[w,c] for c in 1:prob.n_customers) <= M * y[w])
        end

        @constraint(model, sum(prob.fixed_costs[w] * y[w] for w in 1:prob.n_facilities) <= prob.budget)

    elseif prob.variant == fl_p_median
        @variable(model, y[1:prob.n_facilities], Bin)
        @variable(model, x[1:prob.n_facilities, 1:prob.n_customers] >= 0)

        # Minimize total weighted distance
        @objective(model, Min,
            sum(prob.demands[c] * prob.distances[(w,c)] * x[w,c]
                for w in 1:prob.n_facilities, c in 1:prob.n_customers))

        for c in 1:prob.n_customers
            @constraint(model, sum(x[w,c] for w in 1:prob.n_facilities) >= 1)  # Fraction assigned
        end

        for w in 1:prob.n_facilities, c in 1:prob.n_customers
            @constraint(model, x[w,c] <= y[w])
        end

        for w in 1:prob.n_facilities
            @constraint(model, sum(prob.demands[c] * x[w,c] for c in 1:prob.n_customers) <= prob.capacities[w] * y[w])
        end

        # Exactly p facilities
        @constraint(model, sum(y[w] for w in 1:prob.n_facilities) == prob.p_facilities)

    elseif prob.variant == fl_p_center
        @variable(model, y[1:prob.n_facilities], Bin)
        @variable(model, x[1:prob.n_facilities, 1:prob.n_customers] >= 0)
        @variable(model, max_dist >= 0)  # Maximum distance to minimize

        @objective(model, Min, max_dist)

        for c in 1:prob.n_customers
            @constraint(model, sum(x[w,c] for w in 1:prob.n_facilities) >= 1)
        end

        for w in 1:prob.n_facilities, c in 1:prob.n_customers
            @constraint(model, x[w,c] <= y[w])
            # If customer c is assigned to w, distance contributes to max
            @constraint(model, prob.distances[(w,c)] * x[w,c] <= max_dist)
        end

        for w in 1:prob.n_facilities
            @constraint(model, sum(prob.demands[c] * x[w,c] for c in 1:prob.n_customers) <= prob.capacities[w] * y[w])
        end

        @constraint(model, sum(y[w] for w in 1:prob.n_facilities) == prob.p_facilities)

    elseif prob.variant == fl_covering
        @variable(model, y[1:prob.n_facilities], Bin)

        # Minimize number of facilities to cover all customers
        @objective(model, Min, sum(prob.fixed_costs[w] * y[w] for w in 1:prob.n_facilities))

        # Each customer must be covered by at least one open facility within radius
        for c in 1:prob.n_customers
            covering_facilities = [w for w in 1:prob.n_facilities if prob.distances[(w,c)] <= prob.coverage_radius]
            if !isempty(covering_facilities)
                @constraint(model, sum(y[w] for w in covering_facilities) >= 1)
            end
        end

        @constraint(model, sum(prob.fixed_costs[w] * y[w] for w in 1:prob.n_facilities) <= prob.budget)

    elseif prob.variant == fl_single_source
        @variable(model, y[1:prob.n_facilities], Bin)
        @variable(model, z[1:prob.n_facilities, 1:prob.n_customers], Bin)  # Assignment

        @objective(model, Min,
            sum(prob.fixed_costs[w] * y[w] for w in 1:prob.n_facilities) +
            sum(prob.shipping_costs[(w,c)] * prob.demands[c] * z[w,c]
                for w in 1:prob.n_facilities, c in 1:prob.n_customers))

        # Each customer assigned to exactly one facility
        for c in 1:prob.n_customers
            @constraint(model, sum(z[w,c] for w in 1:prob.n_facilities) == 1)
        end

        # Can only assign to open facilities
        for w in 1:prob.n_facilities, c in 1:prob.n_customers
            @constraint(model, z[w,c] <= y[w])
        end

        # Capacity constraints
        for w in 1:prob.n_facilities
            @constraint(model, sum(prob.demands[c] * z[w,c] for c in 1:prob.n_customers) <= prob.capacities[w] * y[w])
        end

        @constraint(model, sum(prob.fixed_costs[w] * y[w] for w in 1:prob.n_facilities) <= prob.budget)
    end

    return model
end

# Register the problem type
register_problem(
    :facility_location,
    FacilityLocationProblem,
    "Facility location problem with variants including standard, uncapacitated, p-median, p-center, covering, and single source"
)
