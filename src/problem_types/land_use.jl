using JuMP
using Random
using Distributions
using StatsBase

"""
    generate_land_use_problem(params::Dict=Dict(); seed::Int=0)

Generate a land use optimization problem instance with parcels and zoning types.

This models realistic land use planning where parcels must be allocated to zoning types
(residential, commercial, industrial, agricultural, conservation) while satisfying
infrastructure constraints, environmental regulations, and economic objectives.

# Arguments
- `params`: Dictionary of problem parameters
  - `:n_parcels`: Number of land parcels (default: 10)
  - `:n_zoning_types`: Number of zoning types (default: 5)
  - `:n_resources`: Number of resource constraints (default: 4)
  - `:development_cost_scale`: Scale factor for development costs (default: 100000)
  - `:revenue_scale`: Scale factor for revenue generation (default: 50000)
  - `:infrastructure_capacity_factor`: Factor controlling infrastructure capacity (default: 0.7)
  - `:environmental_constraint_prob`: Probability of environmental constraints (default: 0.3)
  - `:zoning_adjacency_constraints`: Whether to include adjacency constraints (default: true)
  - `:minimum_zoning_requirements`: Whether to require minimum allocations (default: true)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- `model`: The JuMP model
- `params`: Dictionary of all parameters used (including defaults)
"""
function generate_land_use_problem(params::Dict=Dict(); seed::Int=0)
    # Set random seed
    Random.seed!(seed)
    
    # Extract parameters with defaults
    n_parcels = get(params, :n_parcels, 10)
    n_zoning_types = get(params, :n_zoning_types, 5)
    n_resources = get(params, :n_resources, 4)
    development_cost_scale = get(params, :development_cost_scale, 100000)
    revenue_scale = get(params, :revenue_scale, 50000)
    infrastructure_capacity_factor = get(params, :infrastructure_capacity_factor, 0.7)
    environmental_constraint_prob = get(params, :environmental_constraint_prob, 0.3)
    zoning_adjacency_constraints = get(params, :zoning_adjacency_constraints, true)
    minimum_zoning_requirements = get(params, :minimum_zoning_requirements, true)
    
    # Save actual parameters used
    actual_params = Dict{Symbol, Any}(
        :n_parcels => n_parcels,
        :n_zoning_types => n_zoning_types,
        :n_resources => n_resources,
        :development_cost_scale => development_cost_scale,
        :revenue_scale => revenue_scale,
        :infrastructure_capacity_factor => infrastructure_capacity_factor,
        :environmental_constraint_prob => environmental_constraint_prob,
        :zoning_adjacency_constraints => zoning_adjacency_constraints,
        :minimum_zoning_requirements => minimum_zoning_requirements
    )
    
    # Generate parcel characteristics using realistic distributions
    # Parcel sizes (in acres) - log-normal distribution
    parcel_sizes = rand(LogNormal(log(5), 0.8), n_parcels)
    parcel_sizes = max.(parcel_sizes, 0.1)  # Minimum 0.1 acre
    
    # Zoning type names and characteristics
    zoning_names = ["Residential", "Commercial", "Industrial", "Agricultural", "Conservation"]
    if n_zoning_types > 5
        append!(zoning_names, ["Mixed_Use", "Recreational", "Institutional", "Transportation", "Special"])
    end
    zoning_names = zoning_names[1:n_zoning_types]
    
    # Resource names (infrastructure and environmental)
    resource_names = ["Water", "Sewage", "Transportation", "Power"]
    if n_resources > 4
        append!(resource_names, ["Internet", "Gas", "Environmental", "Emergency"])
    end
    resource_names = resource_names[1:n_resources]
    
    # Generate development costs per acre for each zoning type (varies by complexity)
    cost_multipliers = [1.0, 2.5, 3.0, 0.5, 0.1]  # Residential, Commercial, Industrial, Agricultural, Conservation
    if n_zoning_types > 5
        append!(cost_multipliers, [2.0, 1.5, 1.8, 4.0, 3.5])
    end
    cost_multipliers = cost_multipliers[1:n_zoning_types]
    
    # Development costs with location-specific variation
    development_costs = zeros(n_parcels, n_zoning_types)
    for i in 1:n_parcels
        # Location factor (some areas more expensive to develop)
        location_factor = rand(Gamma(2, 0.5))
        for j in 1:n_zoning_types
            base_cost = development_cost_scale * cost_multipliers[j]
            development_costs[i, j] = base_cost * location_factor * rand(Normal(1.0, 0.2))
            development_costs[i, j] = max(development_costs[i, j], base_cost * 0.1)  # Minimum cost
        end
    end
    
    # Generate revenue per acre for each zoning type
    revenue_multipliers = [1.5, 4.0, 2.0, 0.8, 0.2]  # Different economic returns
    if n_zoning_types > 5
        append!(revenue_multipliers, [3.0, 1.0, 0.5, 0.1, 2.5])
    end
    revenue_multipliers = revenue_multipliers[1:n_zoning_types]
    
    revenues = zeros(n_parcels, n_zoning_types)
    for i in 1:n_parcels
        # Location affects revenue potential
        location_revenue_factor = rand(Gamma(2, 0.6))
        for j in 1:n_zoning_types
            base_revenue = revenue_scale * revenue_multipliers[j]
            revenues[i, j] = base_revenue * location_revenue_factor * rand(Normal(1.0, 0.3))
            revenues[i, j] = max(revenues[i, j], 0.0)
        end
    end
    
    # Generate resource consumption per acre for each zoning type
    resource_consumption = zeros(n_zoning_types, n_resources)
    consumption_patterns = [
        [2.0, 1.5, 1.0, 1.5],  # Residential
        [1.0, 0.8, 3.0, 2.0],  # Commercial
        [0.5, 2.0, 2.5, 4.0],  # Industrial
        [3.0, 0.5, 0.5, 0.5],  # Agricultural
        [0.1, 0.1, 0.1, 0.1]   # Conservation
    ]
    
    for j in 1:n_zoning_types
        for k in 1:n_resources
            if j <= length(consumption_patterns) && k <= length(consumption_patterns[j])
                base_consumption = consumption_patterns[j][k]
            else
                base_consumption = rand(Uniform(0.5, 3.0))
            end
            resource_consumption[j, k] = base_consumption * rand(Gamma(2, 0.5))
        end
    end
    
    # Generate resource capacities
    total_demand_estimate = sum(parcel_sizes) * mean(resource_consumption, dims=1)
    resource_capacities = vec(total_demand_estimate) .* infrastructure_capacity_factor .* rand(Uniform(0.8, 1.2), n_resources)
    
    # Environmental constraints (some parcels restricted for certain zoning)
    environmental_restrictions = zeros(Bool, n_parcels, n_zoning_types)
    for i in 1:n_parcels
        if rand() < environmental_constraint_prob
            # Restrict high-impact zoning types
            restricted_types = sample(1:n_zoning_types, rand(1:min(3, n_zoning_types)), replace=false)
            environmental_restrictions[i, restricted_types] .= true
        end
    end
    
    # Generate adjacency matrix for parcels (simplified - random adjacency)
    adjacency_matrix = zeros(Bool, n_parcels, n_parcels)
    if zoning_adjacency_constraints && n_parcels > 1
        for i in 1:n_parcels
            # Each parcel has 2-4 neighbors on average
            n_neighbors = rand(2:min(4, n_parcels-1))
            neighbors = sample(setdiff(1:n_parcels, [i]), n_neighbors, replace=false)
            adjacency_matrix[i, neighbors] .= true
            adjacency_matrix[neighbors, i] .= true
        end
    end
    
    # Store generated data in params
    actual_params[:parcel_sizes] = parcel_sizes
    actual_params[:development_costs] = development_costs
    actual_params[:revenues] = revenues
    actual_params[:resource_consumption] = resource_consumption
    actual_params[:resource_capacities] = resource_capacities
    actual_params[:environmental_restrictions] = environmental_restrictions
    actual_params[:adjacency_matrix] = adjacency_matrix
    actual_params[:zoning_names] = zoning_names
    actual_params[:resource_names] = resource_names
    
    # Create model
    model = Model()
    
    # Variables: binary variables for parcel-zoning allocation
    @variable(model, x[1:n_parcels, 1:n_zoning_types], Bin)
    
    # Objective: maximize net benefit (revenue - development costs)
    @objective(model, Max, 
        sum(parcel_sizes[i] * (revenues[i, j] - development_costs[i, j]) * x[i, j] 
            for i in 1:n_parcels, j in 1:n_zoning_types))
    
    # Constraint: each parcel must be assigned to exactly one zoning type
    for i in 1:n_parcels
        @constraint(model, sum(x[i, j] for j in 1:n_zoning_types) == 1)
    end
    
    # Constraints: resource capacity limitations
    for k in 1:n_resources
        @constraint(model, 
            sum(parcel_sizes[i] * resource_consumption[j, k] * x[i, j] 
                for i in 1:n_parcels, j in 1:n_zoning_types) <= resource_capacities[k])
    end
    
    # Constraints: environmental restrictions
    for i in 1:n_parcels
        for j in 1:n_zoning_types
            if environmental_restrictions[i, j]
                @constraint(model, x[i, j] == 0)
            end
        end
    end
    
    # Constraints: minimum zoning requirements (ensure diverse development)
    if minimum_zoning_requirements
        for j in 1:n_zoning_types
            if j <= 3  # Require some residential, commercial, industrial
                min_parcels = max(1, round(Int, n_parcels * 0.1))
                @constraint(model, sum(x[i, j] for i in 1:n_parcels) >= min_parcels)
            end
        end
    end
    
    # Constraints: adjacency constraints (limit certain zoning combinations)
    if zoning_adjacency_constraints && n_parcels > 1
        for i in 1:n_parcels
            for i2 in 1:n_parcels
                if adjacency_matrix[i, i2]
                    # Industrial cannot be adjacent to residential
                    if n_zoning_types >= 3
                        @constraint(model, x[i, 1] + x[i2, 3] <= 1)  # Residential + Industrial
                        @constraint(model, x[i, 3] + x[i2, 1] <= 1)  # Industrial + Residential
                    end
                end
            end
        end
    end
    
    return model, actual_params
end

"""
    sample_land_use_parameters(target_variables::Int; seed::Int=0)

Sample realistic parameters for a land use optimization problem targeting approximately the specified number of variables.

Variables = n_parcels × n_zoning_types (binary assignment variables)

# Arguments
- `target_variables`: Target number of variables in the LP formulation (approximately within ±10%)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_land_use_parameters(target_variables::Int; seed::Int=0)
    Random.seed!(seed)
    
    params = Dict{Symbol, Any}()
    
    # Determine problem scale based on target variables
    if target_variables <= 250
        # Small scale: local/municipal planning
        scale = :small
        params[:n_zoning_types] = rand(3:5)
        params[:n_resources] = rand(3:5)
        params[:development_cost_scale] = rand(50000:150000)
        params[:revenue_scale] = rand(20000:80000)
        params[:infrastructure_capacity_factor] = rand(Uniform(0.6, 0.8))
        params[:environmental_constraint_prob] = rand(Uniform(0.2, 0.4))
    elseif target_variables <= 1000
        # Medium scale: regional planning
        scale = :medium
        params[:n_zoning_types] = rand(4:8)
        params[:n_resources] = rand(4:6)
        params[:development_cost_scale] = rand(75000:250000)
        params[:revenue_scale] = rand(40000:120000)
        params[:infrastructure_capacity_factor] = rand(Uniform(0.65, 0.85))
        params[:environmental_constraint_prob] = rand(Uniform(0.25, 0.45))
    else
        # Large scale: state/national planning
        scale = :large
        params[:n_zoning_types] = rand(5:12)
        params[:n_resources] = rand(5:8)
        params[:development_cost_scale] = rand(100000:500000)
        params[:revenue_scale] = rand(60000:200000)
        params[:infrastructure_capacity_factor] = rand(Uniform(0.7, 0.9))
        params[:environmental_constraint_prob] = rand(Uniform(0.3, 0.5))
    end
    
    # Calculate n_parcels to achieve target variables
    target_parcels = round(Int, target_variables / params[:n_zoning_types])
    params[:n_parcels] = max(2, target_parcels)
    
    # Iteratively adjust to get within 10% tolerance
    for iteration in 1:10
        current_vars = calculate_land_use_variable_count(params)
        
        if abs(current_vars - target_variables) / target_variables < 0.1
            break  # Within 10% tolerance
        end
        
        # Adjust n_parcels or n_zoning_types
        if current_vars < target_variables
            if rand() < 0.7  # Prefer adjusting parcels
                params[:n_parcels] += 1
            else
                max_zoning = scale == :small ? 5 : scale == :medium ? 8 : 12
                params[:n_zoning_types] = min(max_zoning, params[:n_zoning_types] + 1)
            end
        elseif current_vars > target_variables
            if rand() < 0.7  # Prefer adjusting parcels
                params[:n_parcels] = max(2, params[:n_parcels] - 1)
            else
                min_zoning = scale == :small ? 3 : scale == :medium ? 4 : 5
                params[:n_zoning_types] = max(min_zoning, params[:n_zoning_types] - 1)
            end
        end
    end
    
    # Scale-appropriate constraint settings
    params[:zoning_adjacency_constraints] = rand() < 0.8  # More likely for realistic problems
    params[:minimum_zoning_requirements] = rand() < 0.9  # Almost always want diverse zoning
    
    return params
end

"""
    sample_land_use_parameters(size::Symbol=:medium; seed::Int=0)

Sample realistic parameters for a land use optimization problem using size categories.

# Arguments
- `size`: Symbol specifying the problem size (:small, :medium, :large)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_land_use_parameters(size::Symbol=:medium; seed::Int=0)
    Random.seed!(seed)
    
    # Map size categories to realistic target variable ranges
    target_map = Dict(
        :small => rand(50:250),    # Local/municipal planning
        :medium => rand(250:1000), # Regional planning
        :large => rand(1000:10000) # State/national planning
    )
    
    if !haskey(target_map, size)
        error("Unknown size: $size. Must be :small, :medium, or :large")
    end
    
    return sample_land_use_parameters(target_map[size]; seed=seed)
end

"""
    calculate_land_use_variable_count(params::Dict)

Calculate the number of variables in a land use optimization problem.

# Arguments
- `params`: Dictionary of problem parameters containing :n_parcels and :n_zoning_types

# Returns
- Number of variables (n_parcels × n_zoning_types binary assignment variables)
"""
function calculate_land_use_variable_count(params::Dict)
    n_parcels = get(params, :n_parcels, 10)
    n_zoning_types = get(params, :n_zoning_types, 5)
    return n_parcels * n_zoning_types
end

# Register the problem type
register_problem(
    :land_use,
    generate_land_use_problem,
    sample_land_use_parameters,
    "Land use optimization problem that maximizes economic benefits by allocating land parcels to zoning types (residential, commercial, industrial, agricultural, conservation) while satisfying infrastructure constraints, environmental regulations, and adjacency requirements"
)