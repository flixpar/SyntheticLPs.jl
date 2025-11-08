using JuMP
using Random
using Distributions
using Statistics

"""
    generate_crop_planning_problem(params::Dict=Dict(); seed::Int=0)

Generate a crop planning optimization problem instance.

This models realistic agricultural crop planning where farmers must decide how much area
to allocate to different crops while maximizing profit and satisfying resource constraints
(land, water, labor), crop diversity requirements, and market demand limitations.

# Arguments
- `params`: Dictionary of problem parameters
  - `:n_crops`: Number of different crops to consider (default: 10)
  - `:n_seasons`: Number of growing seasons in the planning period (default: 1)
  - `:total_land`: Total available land in hectares (default: 1000.0)
  - `:water_availability_factor`: Factor controlling irrigation water capacity (default: 0.7)
  - `:labor_availability_factor`: Factor controlling available labor hours (default: 0.8)
  - `:market_demand_factor`: Factor controlling market demand limits (default: 1.2)
  - `:diversity_constraint_prob`: Probability of adding crop diversity constraints (default: 0.7)
  - `:min_area_requirements`: Whether to enforce minimum area for essential crops (default: true)
  - `:solution_status`: Desired feasibility of the generated instance. One of `:feasible`, `:infeasible`, or `:all`.
    Default: `:feasible`. When `:feasible`, a feasible allocation is constructed and capacities are
    adjusted to guarantee feasibility. When `:infeasible`, capacities are set below a
    provable lower bound to guarantee infeasibility. When `:all`, behavior follows the unconstrained
    random generation (no guarantees).
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- `model`: The JuMP model
- `params`: Dictionary of all parameters used (including defaults)
"""
function generate_crop_planning_problem(params::Dict=Dict(); seed::Int=0)
    # Set random seed
    Random.seed!(seed)

    # Extract parameters with defaults
    n_crops = get(params, :n_crops, 10)
    n_seasons = get(params, :n_seasons, 1)
    total_land = get(params, :total_land, 1000.0)
    water_availability_factor = get(params, :water_availability_factor, 0.7)
    labor_availability_factor = get(params, :labor_availability_factor, 0.8)
    market_demand_factor = get(params, :market_demand_factor, 1.2)
    diversity_constraint_prob = get(params, :diversity_constraint_prob, 0.7)
    min_area_requirements = get(params, :min_area_requirements, true)
    solution_status = get(params, :solution_status, :feasible)

    if solution_status isa String
        solution_status = Symbol(lowercase(solution_status))
    end
    if !(solution_status in (:feasible, :infeasible, :all))
        error("Unknown solution_status=$(solution_status). Use :feasible, :infeasible, or :all")
    end

    # Save actual parameters used
    actual_params = Dict{Symbol, Any}(
        :n_crops => n_crops,
        :n_seasons => n_seasons,
        :total_land => total_land,
        :water_availability_factor => water_availability_factor,
        :labor_availability_factor => labor_availability_factor,
        :market_demand_factor => market_demand_factor,
        :diversity_constraint_prob => diversity_constraint_prob,
        :min_area_requirements => min_area_requirements,
        :solution_status => solution_status
    )

    # Define crop types for realistic modeling
    crop_types = [:cereal, :vegetable, :legume, :industrial, :oilseed]
    crop_names = [
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
        if i <= length(crop_names)
            assigned_crop_names[i] = crop_names[i]
            # Assign type based on position in crop_names
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
            assigned_crop_types[i] = rand(crop_types)
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

        actual_params[:baseline_allocation] = baseline_allocation
        actual_params[:baseline_water_usage] = baseline_water_usage
        actual_params[:baseline_labor_usage] = baseline_labor_usage

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

        actual_params[:min_water_bound] = min_water_bound
        actual_params[:min_labor_bound] = min_labor_bound

    else  # :all
        # Random capacities without guarantees
        estimated_water = total_land * mean(water_requirements) * water_availability_factor
        estimated_labor = total_land * mean(labor_requirements) * labor_availability_factor

        water_capacity = estimated_water * rand(Uniform(0.6, 1.4))
        labor_capacity = estimated_labor * rand(Uniform(0.6, 1.4))
    end

    # Store generated data in params
    actual_params[:crop_types] = assigned_crop_types
    actual_params[:crop_names] = assigned_crop_names
    actual_params[:yields] = yields
    actual_params[:prices] = prices
    actual_params[:production_costs] = production_costs
    actual_params[:water_requirements] = water_requirements
    actual_params[:labor_requirements] = labor_requirements
    actual_params[:net_profit_per_ha] = net_profit_per_ha
    actual_params[:market_demand_limits] = market_demand_limits
    actual_params[:min_area_per_crop] = min_area_per_crop
    actual_params[:water_capacity] = water_capacity
    actual_params[:labor_capacity] = labor_capacity

    # Create JuMP model
    model = Model()

    # Decision variables: area allocated to each crop (hectares)
    @variable(model, x[1:n_crops] >= 0)

    # Objective: maximize total net profit
    @objective(model, Max,
        sum((prices[i] * yields[i] - production_costs[i]) * x[i] for i in 1:n_crops))

    # Constraint: total land area
    @constraint(model, sum(x[i] for i in 1:n_crops) <= total_land)

    # Constraint: water availability
    @constraint(model,
        sum(water_requirements[i] * x[i] for i in 1:n_crops) <= water_capacity)

    # Constraint: labor availability
    @constraint(model,
        sum(labor_requirements[i] * x[i] for i in 1:n_crops) <= labor_capacity)

    # Constraints: market demand limits
    for i in 1:n_crops
        @constraint(model, x[i] <= market_demand_limits[i])
    end

    # Constraints: minimum area requirements for essential crops
    for i in 1:n_crops
        if min_area_per_crop[i] > 0
            @constraint(model, x[i] >= min_area_per_crop[i])
        end
    end

    # Optional: crop diversity constraints
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
        diversity_constraints = []
        for (ctype, crop_indices) in crop_type_groups
            if length(crop_indices) >= 2
                min_type_area = total_land * rand(Uniform(0.05, 0.15))

                # Only add constraint if it's compatible with solution_status
                if solution_status == :feasible
                    # Check if baseline allocation satisfies this
                    current_type_area = sum(baseline_allocation[crop_indices])
                    if current_type_area >= min_type_area * 0.95
                        @constraint(model, sum(x[i] for i in crop_indices) >= min_type_area)
                        push!(diversity_constraints, (ctype, min_type_area, crop_indices))
                    end
                elseif solution_status == :all
                    @constraint(model, sum(x[i] for i in crop_indices) >= min_type_area)
                    push!(diversity_constraints, (ctype, min_type_area, crop_indices))
                end
                # For infeasible, don't add diversity constraints that might interfere
            end
        end
        actual_params[:diversity_constraints] = diversity_constraints
    end

    return model, actual_params
end

"""
    sample_crop_planning_parameters(target_variables::Int; seed::Int=0)

Sample realistic parameters for a crop planning problem targeting approximately the specified number of variables.

Variables = n_crops (one continuous variable per crop for area allocation)

# Arguments
- `target_variables`: Target number of variables in the LP formulation (approximately within ±10%)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_crop_planning_parameters(target_variables::Int; seed::Int=0)
    Random.seed!(seed)

    params = Dict{Symbol, Any}()

    # For crop planning, target_variables = n_crops
    params[:n_crops] = max(2, target_variables)

    # Determine problem scale based on target variables
    if target_variables <= 250
        # Small scale: Family farm or small agricultural operation
        scale = :small
        params[:total_land] = rand(Uniform(50.0, 500.0))  # 50-500 hectares
        params[:water_availability_factor] = rand(Uniform(0.6, 0.8))
        params[:labor_availability_factor] = rand(Uniform(0.7, 0.9))
        params[:market_demand_factor] = rand(Uniform(1.0, 1.3))
        params[:diversity_constraint_prob] = rand(Uniform(0.5, 0.8))
    elseif target_variables <= 1000
        # Medium scale: Commercial farm or agricultural cooperative
        scale = :medium
        params[:total_land] = rand(Uniform(500.0, 5000.0))  # 500-5000 hectares
        params[:water_availability_factor] = rand(Uniform(0.65, 0.85))
        params[:labor_availability_factor] = rand(Uniform(0.75, 0.95))
        params[:market_demand_factor] = rand(Uniform(1.1, 1.5))
        params[:diversity_constraint_prob] = rand(Uniform(0.6, 0.9))
    else
        # Large scale: Industrial agriculture or regional planning
        scale = :large
        params[:total_land] = rand(Uniform(5000.0, 50000.0))  # 5000-50000 hectares
        params[:water_availability_factor] = rand(Uniform(0.7, 0.9))
        params[:labor_availability_factor] = rand(Uniform(0.8, 1.0))
        params[:market_demand_factor] = rand(Uniform(1.2, 2.0))
        params[:diversity_constraint_prob] = rand(Uniform(0.7, 0.95))
    end

    # Single season planning (multi-season could be future extension)
    params[:n_seasons] = 1

    # Minimum area requirements are common in agricultural planning
    params[:min_area_requirements] = rand() < 0.85

    # Iteratively adjust to get within 10% tolerance
    for iteration in 1:10
        current_vars = calculate_crop_planning_variable_count(params)

        if abs(current_vars - target_variables) / target_variables < 0.1
            break  # Within 10% tolerance
        end

        # Adjust n_crops
        if current_vars < target_variables
            params[:n_crops] += 1
        elseif current_vars > target_variables
            params[:n_crops] = max(2, params[:n_crops] - 1)
        end
    end

    return params
end

"""
    sample_crop_planning_parameters(size::Symbol=:medium; seed::Int=0)

Sample realistic parameters for a crop planning problem using size categories.

# Arguments
- `size`: Symbol specifying the problem size (:small, :medium, :large)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_crop_planning_parameters(size::Symbol=:medium; seed::Int=0)
    Random.seed!(seed)

    # Map size categories to realistic target variable ranges
    target_map = Dict(
        :small => rand(50:250),     # Small farm: 50-250 crops
        :medium => rand(250:1000),  # Commercial farm: 250-1000 crops
        :large => rand(1000:10000)  # Industrial/regional: 1000-10000 crops
    )

    if !haskey(target_map, size)
        error("Unknown size: $size. Must be :small, :medium, or :large")
    end

    return sample_crop_planning_parameters(target_map[size]; seed=seed)
end

"""
    calculate_crop_planning_variable_count(params::Dict)

Calculate the number of variables in a crop planning problem.

# Arguments
- `params`: Dictionary of problem parameters containing :n_crops

# Returns
- Number of variables (n_crops continuous variables for area allocation)
"""
function calculate_crop_planning_variable_count(params::Dict)
    n_crops = get(params, :n_crops, 10)
    return n_crops
end

# Register the problem type
register_problem(
    :crop_planning,
    generate_crop_planning_problem,
    sample_crop_planning_parameters,
    "Crop planning optimization problem that maximizes agricultural profit by allocating land to different crops while satisfying resource constraints (water, labor), crop diversity requirements, and market demand limitations"
)
