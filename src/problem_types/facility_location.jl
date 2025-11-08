using JuMP
using Random
using Distributions

"""
    FacilityLocationProblem <: ProblemGenerator

Generator for facility location problems.

# Fields
- `n_facilities::Int`: Number of potential facility locations
- `n_customers::Int`: Number of customers
- `facility_locs::Vector{Tuple{Float64,Float64}}`: Facility coordinates
- `customer_locs::Vector{Tuple{Float64,Float64}}`: Customer coordinates
- `demands::Dict{Int,Float64}`: Customer demand
- `fixed_costs::Dict{Int,Float64}`: Fixed cost to open each facility
- `capacities::Dict{Int,Float64}`: Capacity of each facility
- `shipping_costs::Dict{Tuple{Int,Int},Float64}`: Shipping cost from facility to customer
- `budget::Float64`: Total budget for opening facilities
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
end

"""
    FacilityLocationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a facility location problem instance.

# Arguments
- `target_variables`: Target number of variables (n_facilities × (n_customers + 1))
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function FacilityLocationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
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
        fixed_costs[w] = clamp(
            location_factor * (fixed_cost_min + (capacity/avg_facility_capacity) * (fixed_cost_max - fixed_cost_min)),
            fixed_cost_min,
            fixed_cost_max
        )
    end

    # Shipping costs
    shipping_costs = Dict{Tuple{Int,Int},Float64}()
    for w in 1:n_facilities
        for c in 1:n_customers
            distance = sqrt(
                (facility_locs[w][1] - customer_locs[c][1])^2 +
                (facility_locs[w][2] - customer_locs[c][2])^2
            )
            shipping_costs[(w,c)] = distance * transport_cost_per_km * (0.9 + 0.2 * rand())
        end
    end

    # Initial budget
    budget = sum(values(fixed_costs)) * budget_factor
    original_budget = budget

    # Adjust for feasibility
    solution_status = feasibility_status == feasible ? :feasible :
                     feasibility_status == infeasible ? :infeasible : :all

    caps_vec = [capacities[w] for w in 1:n_facilities]
    costs_vec = [fixed_costs[w] for w in 1:n_facilities]
    ratios = [caps_vec[i] / max(costs_vec[i], eps()) for i in 1:n_facilities]
    order_desc = sortperm(ratios, rev=true)
    total_capacity = sum(caps_vec)

    function fractional_budget_to_reach(cap_target::Float64)
        cum_cap = 0.0
        cum_cost = 0.0
        for idx in order_desc
            cap_i = caps_vec[idx]
            cost_i = costs_vec[idx]
            if cum_cap + cap_i >= cap_target
                rem = cap_target - cum_cap
                frac_cost = cost_i * (rem / cap_i)
                return cum_cost + frac_cost
            else
                cum_cap += cap_i
                cum_cost += cost_i
            end
        end
        return Inf
    end

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
            total_capacity = sum(values(capacities))
            caps_vec = [capacities[w] for w in 1:n_facilities]
            ratios = [caps_vec[i] / max(costs_vec[i], eps()) for i in 1:n_facilities]
            order_desc = sortperm(ratios, rev=true)
        end
        selected_idxs, min_int_budget = greedy_integer_subset_for(total_demand)
        slack_factor = 1.02 + 0.23 * rand()
        budget = max(original_budget, min_int_budget * slack_factor)
    elseif solution_status == :infeasible
        b_thresh = fractional_budget_to_reach(total_demand)
        if isfinite(b_thresh)
            tighten = rand(0.75:0.01:0.95)
            budget = min(original_budget, b_thresh * tighten)
        else
            budget = min(original_budget, sum(values(fixed_costs)) * rand(0.4:0.01:0.8))
        end
    else
        budget = original_budget
    end

    return FacilityLocationProblem(n_facilities, n_customers, facility_locs, customer_locs,
                                   demands, fixed_costs, capacities, shipping_costs, budget)
end

"""
    build_model(prob::FacilityLocationProblem)

Build a JuMP model for the facility location problem.

# Arguments
- `prob`: FacilityLocationProblem instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::FacilityLocationProblem)
    model = Model()

    # Variables
    @variable(model, y[1:prob.n_facilities], Bin)
    @variable(model, x[1:prob.n_facilities, 1:prob.n_customers] >= 0)

    # Objective
    @objective(model, Min,
        sum(prob.fixed_costs[w] * y[w] for w in 1:prob.n_facilities) +
        sum(prob.shipping_costs[(w,c)] * x[w,c] for w in 1:prob.n_facilities, c in 1:prob.n_customers)
    )

    # Customer demand
    for c in 1:prob.n_customers
        @constraint(model, sum(x[w,c] for w in 1:prob.n_facilities) >= prob.demands[c])
    end

    # Facility capacity
    for w in 1:prob.n_facilities
        @constraint(model, sum(x[w,c] for c in 1:prob.n_customers) <= prob.capacities[w] * y[w])
    end

    # Budget
    @constraint(model, sum(prob.fixed_costs[w] * y[w] for w in 1:prob.n_facilities) <= prob.budget)

    return model
end

# Register the problem type
register_problem(
    :facility_location,
    FacilityLocationProblem,
    "Facility location problem that minimizes the cost of opening facilities and shipping to customers while meeting demand"
)
