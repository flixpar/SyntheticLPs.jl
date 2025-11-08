using JuMP
using Random
using Distributions
using Statistics

"""
    CropPlanningProblem <: ProblemGenerator

Generator for crop planning optimization problems.

Models realistic agricultural crop planning where farmers must decide how much area
to allocate to different crops while maximizing profit and satisfying resource constraints
(land, water, labor), crop diversity requirements, and market demand limitations.

# Fields
- `n_crops::Int`: Number of different crops
- `total_land::Float64`: Total available land in hectares
- `crop_types::Vector{Symbol}`: Type of each crop (:cereal, :vegetable, :legume, :industrial, :oilseed)
- `crop_names::Vector{String}`: Name of each crop
- `yields::Vector{Float64}`: Yield in tons/hectare for each crop
- `prices::Vector{Float64}`: Price in dollars/ton for each crop
- `production_costs::Vector{Float64}`: Production cost in dollars/hectare for each crop
- `water_requirements::Vector{Float64}`: Water requirement in mm/season for each crop
- `labor_requirements::Vector{Float64}`: Labor requirement in hours/hectare for each crop
- `net_profit_per_ha::Vector{Float64}`: Net profit per hectare for each crop
- `market_demand_limits::Vector{Float64}`: Market demand limit in hectares for each crop
- `min_area_per_crop::Vector{Float64}`: Minimum area requirement in hectares for each crop
- `water_capacity::Float64`: Available water capacity
- `labor_capacity::Float64`: Available labor capacity
- `diversity_constraints::Vector{Tuple{Symbol, Float64, Vector{Int}}}`: Diversity constraints (crop_type, min_area, crop_indices)
"""
struct CropPlanningProblem <: ProblemGenerator
    n_crops::Int
    total_land::Float64
    crop_types::Vector{Symbol}
    crop_names::Vector{String}
    yields::Vector{Float64}
    prices::Vector{Float64}
    production_costs::Vector{Float64}
    water_requirements::Vector{Float64}
    labor_requirements::Vector{Float64}
    net_profit_per_ha::Vector{Float64}
    market_demand_limits::Vector{Float64}
    min_area_per_crop::Vector{Float64}
    water_capacity::Float64
    labor_capacity::Float64
    diversity_constraints::Vector{Tuple{Symbol, Float64, Vector{Int}}}
end

"""
    CropPlanningProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a crop planning problem instance.

# Arguments
- `target_variables`: Target number of variables (crops)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function CropPlanningProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # For crop planning, target_variables = n_crops
    n_crops = max(2, target_variables)

    # Determine problem scale based on target variables
    if target_variables <= 250
        # Small scale: Family farm or small agricultural operation
        total_land = rand(Uniform(50.0, 500.0))
        water_availability_factor = rand(Uniform(0.6, 0.8))
        labor_availability_factor = rand(Uniform(0.7, 0.9))
        market_demand_factor = rand(Uniform(1.0, 1.3))
        diversity_constraint_prob = rand(Uniform(0.5, 0.8))
    elseif target_variables <= 1000
        # Medium scale: Commercial farm or agricultural cooperative
        total_land = rand(Uniform(500.0, 5000.0))
        water_availability_factor = rand(Uniform(0.65, 0.85))
        labor_availability_factor = rand(Uniform(0.75, 0.95))
        market_demand_factor = rand(Uniform(1.1, 1.5))
        diversity_constraint_prob = rand(Uniform(0.6, 0.9))
    else
        # Large scale: Industrial agriculture or regional planning
        total_land = rand(Uniform(5000.0, 50000.0))
        water_availability_factor = rand(Uniform(0.7, 0.9))
        labor_availability_factor = rand(Uniform(0.8, 1.0))
        market_demand_factor = rand(Uniform(1.2, 2.0))
        diversity_constraint_prob = rand(Uniform(0.7, 0.95))
    end

    # Minimum area requirements are common in agricultural planning
    min_area_requirements = rand() < 0.85

    # Convert feasibility status
    solution_status = feasibility_status == feasible ? :feasible :
                     feasibility_status == infeasible ? :infeasible : :all

    # Define crop types for realistic modeling
    crop_type_list = [:cereal, :vegetable, :legume, :industrial, :oilseed]
    crop_name_list = [
        "Wheat", "Corn", "Rice", "Barley", "Oats",
        "Tomatoes", "Peppers", "Lettuce", "Carrots", "Onions",
        "Soybeans", "Peas", "Lentils", "Beans", "Chickpeas",
        "Cotton", "Sugarcane", "Tobacco", "Hemp", "Flax",
        "Sunflower", "Canola", "Safflower", "Sesame", "Peanuts"
    ]

    # Assign crop type to each crop
    assigned_crop_types = Vector{Symbol}(undef, n_crops)
    assigned_crop_names = Vector{String}(undef, n_crops)

    for i in 1:n_crops
        if i <= length(crop_name_list)
            assigned_crop_names[i] = crop_name_list[i]
            # Assign type based on position in crop_name_list
            if i <= 5
                assigned_crop_types[i] = :cereal
            elseif i <= 10
                assigned_crop_types[i] = :vegetable
            elseif i <= 15
                assigned_crop_types[i] = :legume
            elseif i <= 20
                assigned_crop_types[i] = :industrial
            else
                assigned_crop_types[i] = :oilseed
            end
        else
            assigned_crop_names[i] = "Crop_$(i)"
            assigned_crop_types[i] = rand(crop_type_list)
        end
    end

    # Generate crop yields (tons/hectare) based on crop type
    yields = zeros(n_crops)
    for i in 1:n_crops
        if assigned_crop_types[i] == :cereal
            # Cereals: 3-10 tons/ha
            yields[i] = exp(rand(Normal(log(5.5), 0.4)))
            yields[i] = clamp(yields[i], 3.0, 10.0)
        elseif assigned_crop_types[i] == :vegetable
            # Vegetables: 15-40 tons/ha (higher yield)
            yields[i] = exp(rand(Normal(log(25.0), 0.35)))
            yields[i] = clamp(yields[i], 15.0, 40.0)
        elseif assigned_crop_types[i] == :legume
            # Legumes: 2-4 tons/ha
            yields[i] = exp(rand(Normal(log(3.0), 0.3)))
            yields[i] = clamp(yields[i], 2.0, 4.0)
        elseif assigned_crop_types[i] == :industrial
            # Industrial crops: 3-8 tons/ha
            yields[i] = exp(rand(Normal(log(5.0), 0.35)))
            yields[i] = clamp(yields[i], 3.0, 8.0)
        else  # :oilseed
            # Oilseeds: 1.5-4 tons/ha
            yields[i] = exp(rand(Normal(log(2.5), 0.35)))
            yields[i] = clamp(yields[i], 1.5, 4.0)
        end
    end

    # Generate crop prices ($/ton) - higher value for vegetables and industrial crops
    prices = zeros(n_crops)
    for i in 1:n_crops
        if assigned_crop_types[i] == :cereal
            # Cereals: $150-350/ton
            prices[i] = exp(rand(Normal(log(230.0), 0.25)))
            prices[i] = clamp(prices[i], 150.0, 350.0)
        elseif assigned_crop_types[i] == :vegetable
            # Vegetables: $400-900/ton (higher value)
            prices[i] = exp(rand(Normal(log(600.0), 0.25)))
            prices[i] = clamp(prices[i], 400.0, 900.0)
        elseif assigned_crop_types[i] == :legume
            # Legumes: $300-600/ton
            prices[i] = exp(rand(Normal(log(420.0), 0.25)))
            prices[i] = clamp(prices[i], 300.0, 600.0)
        elseif assigned_crop_types[i] == :industrial
            # Industrial: $500-1200/ton
            prices[i] = exp(rand(Normal(log(800.0), 0.3)))
            prices[i] = clamp(prices[i], 500.0, 1200.0)
        else  # :oilseed
            # Oilseeds: $400-700/ton
            prices[i] = exp(rand(Normal(log(520.0), 0.25)))
            prices[i] = clamp(prices[i], 400.0, 700.0)
        end
    end

    # Generate production costs ($/hectare)
    production_costs = zeros(n_crops)
    for i in 1:n_crops
        if assigned_crop_types[i] == :cereal
            # Cereals: $400-900/ha
            production_costs[i] = rand(Normal(600.0, 120.0))
            production_costs[i] = clamp(production_costs[i], 400.0, 900.0)
        elseif assigned_crop_types[i] == :vegetable
            # Vegetables: $1200-2500/ha (labor intensive)
            production_costs[i] = rand(Normal(1700.0, 320.0))
            production_costs[i] = clamp(production_costs[i], 1200.0, 2500.0)
        elseif assigned_crop_types[i] == :legume
            # Legumes: $350-700/ha (lower input costs)
            production_costs[i] = rand(Normal(500.0, 90.0))
            production_costs[i] = clamp(production_costs[i], 350.0, 700.0)
        elseif assigned_crop_types[i] == :industrial
            # Industrial: $800-1800/ha
            production_costs[i] = rand(Normal(1200.0, 250.0))
            production_costs[i] = clamp(production_costs[i], 800.0, 1800.0)
        else  # :oilseed
            # Oilseeds: $400-800/ha
            production_costs[i] = rand(Normal(570.0, 100.0))
            production_costs[i] = clamp(production_costs[i], 400.0, 800.0)
        end
    end

    # Generate water requirements (mm/season)
    water_requirements = zeros(n_crops)
    for i in 1:n_crops
        if assigned_crop_types[i] == :cereal
            if assigned_crop_names[i] == "Rice"
                # Rice needs more water: 1200-1800 mm
                water_requirements[i] = rand(Normal(1500.0, 150.0))
                water_requirements[i] = clamp(water_requirements[i], 1200.0, 1800.0)
            else
                # Other cereals: 400-650 mm
                water_requirements[i] = rand(Normal(520.0, 70.0))
                water_requirements[i] = clamp(water_requirements[i], 400.0, 650.0)
            end
        elseif assigned_crop_types[i] == :vegetable
            # Vegetables: 350-600 mm (frequent irrigation)
            water_requirements[i] = rand(Normal(470.0, 65.0))
            water_requirements[i] = clamp(water_requirements[i], 350.0, 600.0)
        elseif assigned_crop_types[i] == :legume
            # Legumes: 300-500 mm (drought tolerant)
            water_requirements[i] = rand(Normal(390.0, 55.0))
            water_requirements[i] = clamp(water_requirements[i], 300.0, 500.0)
        elseif assigned_crop_types[i] == :industrial
            if assigned_crop_names[i] == "Sugarcane"
                # Sugarcane: 1500-2500 mm (water intensive)
                water_requirements[i] = rand(Normal(2000.0, 250.0))
                water_requirements[i] = clamp(water_requirements[i], 1500.0, 2500.0)
            else
                # Cotton and others: 500-800 mm
                water_requirements[i] = rand(Normal(640.0, 80.0))
                water_requirements[i] = clamp(water_requirements[i], 500.0, 800.0)
            end
        else  # :oilseed
            # Oilseeds: 350-550 mm
            water_requirements[i] = rand(Normal(440.0, 60.0))
            water_requirements[i] = clamp(water_requirements[i], 350.0, 550.0)
        end
    end

    # Generate labor requirements (hours/hectare)
    labor_requirements = zeros(n_crops)
    for i in 1:n_crops
        if assigned_crop_types[i] == :cereal
            # Cereals: 30-80 hours/ha (mechanized)
            labor_requirements[i] = rand(Gamma(4, 12))
            labor_requirements[i] = clamp(labor_requirements[i], 30.0, 80.0)
        elseif assigned_crop_types[i] == :vegetable
            # Vegetables: 120-250 hours/ha (labor intensive)
            labor_requirements[i] = rand(Gamma(6, 28))
            labor_requirements[i] = clamp(labor_requirements[i], 120.0, 250.0)
        elseif assigned_crop_types[i] == :legume
            # Legumes: 25-60 hours/ha
            labor_requirements[i] = rand(Gamma(4, 10))
            labor_requirements[i] = clamp(labor_requirements[i], 25.0, 60.0)
        elseif assigned_crop_types[i] == :industrial
            # Industrial: 80-180 hours/ha
            labor_requirements[i] = rand(Gamma(5, 24))
            labor_requirements[i] = clamp(labor_requirements[i], 80.0, 180.0)
        else  # :oilseed
            # Oilseeds: 35-75 hours/ha
            labor_requirements[i] = rand(Gamma(4, 13))
            labor_requirements[i] = clamp(labor_requirements[i], 35.0, 75.0)
        end
    end

    # Calculate net profit per hectare for each crop
    net_profit_per_ha = prices .* yields .- production_costs

    # Generate market demand limits (hectares) - based on expected production scale
    market_demand_limits = zeros(n_crops)
    for i in 1:n_crops
        # Market demand typically 10-40% of total land for each crop
        base_demand = total_land * rand(Uniform(0.1, 0.4))
        market_demand_limits[i] = base_demand * market_demand_factor
    end

    # Determine minimum area requirements for essential crops
    min_area_per_crop = zeros(n_crops)
    if min_area_requirements
        # Require minimum area for 30-50% of crops (essential for food security)
        n_essential = round(Int, n_crops * rand(Uniform(0.3, 0.5)))
        # Prefer cereals and legumes as essential
        essential_candidates = [i for i in 1:n_crops if assigned_crop_types[i] in [:cereal, :legume]]
        if length(essential_candidates) < n_essential
            # Add more crops if needed
            other_crops = setdiff(1:n_crops, essential_candidates)
            essential_candidates = vcat(essential_candidates, other_crops[1:min(n_essential - length(essential_candidates), length(other_crops))])
        end
        essential_crops = essential_candidates[1:min(n_essential, length(essential_candidates))]

        for i in essential_crops
            # Minimum 2-8% of total land
            min_area_per_crop[i] = total_land * rand(Uniform(0.02, 0.08))
        end
    end

    # Ensure minimum area requirements do not violate market demand limits
    for i in 1:n_crops
        if min_area_per_crop[i] > market_demand_limits[i]
            min_area_per_crop[i] = market_demand_limits[i]
        end
    end

    # Scale minimum areas if their total exceeds available land
    min_total_area = sum(min_area_per_crop)
    if min_total_area > total_land && min_total_area > 0
        scaling_factor = total_land / min_total_area
        min_area_per_crop .*= scaling_factor
    end

    # Calculate resource requirements for a feasible baseline allocation
    # Use profit-weighted allocation as baseline
    baseline_allocation = zeros(n_crops)

    # First, allocate minimum requirements
    baseline_allocation = copy(min_area_per_crop)
    remaining_land = total_land - sum(baseline_allocation)

    if remaining_land > 0
        # Allocate remaining land based on profitability
        profit_weights = max.(net_profit_per_ha, 0.0)
        total_weight = sum(profit_weights)

        if total_weight > 0
            for i in 1:n_crops
                additional_area = remaining_land * (profit_weights[i] / total_weight)
                # Don't exceed market demand
                max_additional = max(0.0, market_demand_limits[i] - baseline_allocation[i])
                baseline_allocation[i] += min(additional_area, max_additional)
            end
        else
            # If all profits are negative, distribute evenly
            for i in 1:n_crops
                baseline_allocation[i] += remaining_land / n_crops
            end
        end
    end

    # No renormalization: baseline must respect market limits; ensure nonzero fallback
    current_total = sum(baseline_allocation)
    if current_total == 0.0
        baseline_allocation .= min.(total_land / n_crops, market_demand_limits)
    end

    # Calculate resource usage for baseline allocation
    baseline_water_usage = sum(water_requirements .* baseline_allocation)
    baseline_labor_usage = sum(labor_requirements .* baseline_allocation)

    # Set resource capacities based on solution_status
    water_capacity = 0.0
    labor_capacity = 0.0

    if solution_status == :feasible
        # Add slack to baseline usage to guarantee feasibility
        slack_factor = rand(Uniform(1.1, 1.3))
        water_capacity = baseline_water_usage * slack_factor
        labor_capacity = baseline_labor_usage * slack_factor

        # Ensure capacities are reasonable
        water_capacity = max(water_capacity, sum(water_requirements .* min_area_per_crop) * 1.2)
        labor_capacity = max(labor_capacity, sum(labor_requirements .* min_area_per_crop) * 1.2)

    elseif solution_status == :infeasible
        # Calculate provable lower bounds on resource usage
        # Lower bound: minimum water/labor needed when using minimum-requirement crops only
        min_water_bound = 0.0
        min_labor_bound = 0.0

        # Strategy: compute minimum resource usage when satisfying all minimum area requirements
        min_water_bound = sum(water_requirements .* min_area_per_crop)
        min_labor_bound = sum(labor_requirements .* min_area_per_crop)

        # For remaining land, use crops with minimum resource requirements
        remaining_land_required = max(0.0, total_land - sum(min_area_per_crop))

        if remaining_land_required > 0
            # Find crop with minimum water requirement
            min_water_crop = argmin(water_requirements)
            min_water_bound += water_requirements[min_water_crop] * remaining_land_required

            # Find crop with minimum labor requirement
            min_labor_crop = argmin(labor_requirements)
            min_labor_bound += labor_requirements[min_labor_crop] * remaining_land_required
        end

        # Set capacity below lower bound to guarantee infeasibility
        violation_factor = rand(Uniform(0.75, 0.95))

        # Randomly choose which constraint to violate
        if rand() < 0.5
            # Make water infeasible
            water_capacity = min_water_bound * violation_factor
            labor_capacity = min_labor_bound * rand(Uniform(1.1, 1.4))
        else
            # Make labor infeasible
            water_capacity = min_water_bound * rand(Uniform(1.1, 1.4))
            labor_capacity = min_labor_bound * violation_factor
        end

    else  # :all
        # Random capacities without guarantees
        estimated_water = total_land * mean(water_requirements) * water_availability_factor
        estimated_labor = total_land * mean(labor_requirements) * labor_availability_factor

        water_capacity = estimated_water * rand(Uniform(0.6, 1.4))
        labor_capacity = estimated_labor * rand(Uniform(0.6, 1.4))
    end

    # Optional: crop diversity constraints
    diversity_constraints = Tuple{Symbol, Float64, Vector{Int}}[]
    if rand() < diversity_constraint_prob
        # Ensure at least some diversity across crop types
        crop_type_groups = Dict{Symbol, Vector{Int}}()
        for i in 1:n_crops
            ctype = assigned_crop_types[i]
            if !haskey(crop_type_groups, ctype)
                crop_type_groups[ctype] = Int[]
            end
            push!(crop_type_groups[ctype], i)
        end

        # For each crop type with multiple crops, ensure minimum total area
        for (ctype, crop_indices) in crop_type_groups
            if length(crop_indices) >= 2
                min_type_area = total_land * rand(Uniform(0.05, 0.15))

                # Only add constraint if it's compatible with solution_status
                if solution_status == :feasible
                    # Check if baseline allocation satisfies this
                    current_type_area = sum(baseline_allocation[crop_indices])
                    if current_type_area >= min_type_area * 0.95
                        push!(diversity_constraints, (ctype, min_type_area, crop_indices))
                    end
                elseif solution_status == :all
                    push!(diversity_constraints, (ctype, min_type_area, crop_indices))
                end
                # For infeasible, don't add diversity constraints that might interfere
            end
        end
    end

    return CropPlanningProblem(
        n_crops,
        total_land,
        assigned_crop_types,
        assigned_crop_names,
        yields,
        prices,
        production_costs,
        water_requirements,
        labor_requirements,
        net_profit_per_ha,
        market_demand_limits,
        min_area_per_crop,
        water_capacity,
        labor_capacity,
        diversity_constraints
    )
end

"""
    build_model(prob::CropPlanningProblem)

Build a JuMP model for the crop planning problem.

# Arguments
- `prob`: CropPlanningProblem instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::CropPlanningProblem)
    model = Model()

    # Decision variables: area allocated to each crop (hectares)
    @variable(model, x[1:prob.n_crops] >= 0)

    # Objective: maximize total net profit
    @objective(model, Max,
        sum((prob.prices[i] * prob.yields[i] - prob.production_costs[i]) * x[i] for i in 1:prob.n_crops))

    # Constraint: total land area
    @constraint(model, sum(x[i] for i in 1:prob.n_crops) <= prob.total_land)

    # Constraint: water availability
    @constraint(model,
        sum(prob.water_requirements[i] * x[i] for i in 1:prob.n_crops) <= prob.water_capacity)

    # Constraint: labor availability
    @constraint(model,
        sum(prob.labor_requirements[i] * x[i] for i in 1:prob.n_crops) <= prob.labor_capacity)

    # Constraints: market demand limits
    for i in 1:prob.n_crops
        @constraint(model, x[i] <= prob.market_demand_limits[i])
    end

    # Constraints: minimum area requirements for essential crops
    for i in 1:prob.n_crops
        if prob.min_area_per_crop[i] > 0
            @constraint(model, x[i] >= prob.min_area_per_crop[i])
        end
    end

    # Optional: crop diversity constraints
    for (ctype, min_type_area, crop_indices) in prob.diversity_constraints
        @constraint(model, sum(x[i] for i in crop_indices) >= min_type_area)
    end

    return model
end

# Register the problem type
register_problem(
    :crop_planning,
    CropPlanningProblem,
    "Crop planning optimization problem that maximizes agricultural profit by allocating land to different crops while satisfying resource constraints (water, labor), crop diversity requirements, and market demand limitations"
)
