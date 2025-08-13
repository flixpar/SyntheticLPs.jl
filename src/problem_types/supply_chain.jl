using JuMP
using Random
using StatsBase
using Distributions

"""
    generate_supply_chain_problem(params::Dict=Dict(); seed::Int=0)

Generate a supply chain optimization problem instance.

# Arguments
- `params`: Dictionary of problem parameters
  - `:n_facilities`: Number of potential facility locations (default: 8)
  - `:n_customers`: Number of customer locations (default: 25)
  - `:n_transport_modes`: Number of transportation modes (default: 3)
  - `:grid_width`: Width of the geographic area (default: 1000.0)
  - `:grid_height`: Height of the geographic area (default: 1000.0)
  - `:min_fixed_cost`: Minimum fixed cost for opening a facility (default: 500000.0)
  - `:max_fixed_cost`: Maximum fixed cost for opening a facility (default: 2000000.0)
  - `:min_demand`: Minimum customer demand (default: 100.0)
  - `:max_demand`: Maximum customer demand (default: 1000.0)
  - `:capacity_factor`: Facility capacity as multiple of average demand (default: 1.5)
  - `:mode_capacity_factor`: Transport mode capacity as fraction of total demand (default: 0.4)
  - `:clustering_factor`: Degree of geographic clustering (default: 0.3)
  - `:infrastructure_density`: Availability of transport modes between locations (default: 0.7)
  - `:transport_modes`: List of transport modes to use (default: ["truck", "rail", "ship"])
  - `:transport_base_costs`: Base cost per unit-distance for each mode (default: Dict with values)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- `model`: The JuMP model
- `params`: Dictionary of all parameters used (including defaults)
"""
function generate_supply_chain_problem(params::Dict=Dict(); seed::Int=0)
    # Set random seed
    Random.seed!(seed)
    
    # Extract parameters with defaults
    n_facilities = get(params, :n_facilities, 8)
    n_customers = get(params, :n_customers, 25)
    n_transport_modes = get(params, :n_transport_modes, 3)
    grid_width = get(params, :grid_width, 1000.0)
    grid_height = get(params, :grid_height, 1000.0)
    min_fixed_cost = get(params, :min_fixed_cost, 500000.0)
    max_fixed_cost = get(params, :max_fixed_cost, 2000000.0)
    min_demand = get(params, :min_demand, 100.0)
    max_demand = get(params, :max_demand, 1000.0)
    capacity_factor = get(params, :capacity_factor, 1.5)
    mode_capacity_factor = get(params, :mode_capacity_factor, 0.4)
    clustering_factor = get(params, :clustering_factor, 0.3)
    infrastructure_density = get(params, :infrastructure_density, 0.7)
    all_transport_modes = get(params, :transport_modes, ["truck", "rail", "ship"])
    
    default_transport_costs = Dict(
        "truck" => 1.0,
        "rail" => 0.6,
        "ship" => 0.3,
        "air" => 3.0
    )
    transport_base_costs = get(params, :transport_base_costs, default_transport_costs)
    
    # Save actual parameters used
    actual_params = Dict{Symbol, Any}(
        :n_facilities => n_facilities,
        :n_customers => n_customers,
        :n_transport_modes => n_transport_modes,
        :grid_width => grid_width,
        :grid_height => grid_height,
        :min_fixed_cost => min_fixed_cost,
        :max_fixed_cost => max_fixed_cost,
        :min_demand => min_demand,
        :max_demand => max_demand,
        :capacity_factor => capacity_factor,
        :mode_capacity_factor => mode_capacity_factor,
        :clustering_factor => clustering_factor,
        :infrastructure_density => infrastructure_density,
        :transport_modes => all_transport_modes,
        :transport_base_costs => transport_base_costs
    )
    
    # Select transport modes
    transport_modes = sample(all_transport_modes, min(n_transport_modes, length(all_transport_modes)), replace=false)
    actual_params[:selected_transport_modes] = transport_modes
    
    # Generate geographic clusters for realistic location distribution
    n_clusters = max(2, round(Int, sqrt(n_customers) * clustering_factor))
    cluster_centers = [(grid_width * rand(), grid_height * rand()) for _ in 1:n_clusters]
    
    # Generate facility locations (more dispersed than customers)
    facility_locs = Vector{Tuple{Float64,Float64}}()
    for _ in 1:n_facilities
        # Use Beta distribution to create more realistic facility placement
        # Facilities tend to be placed strategically considering market access
        if rand() < 0.4  # 40% chance to be near a cluster center (strategic placement)
            center = rand(cluster_centers)
            # Use normal distribution with realistic spread
            spread_x = grid_width * 0.12
            spread_y = grid_height * 0.12
            x = clamp(center[1] + rand(Normal(0, spread_x)), 0, grid_width)
            y = clamp(center[2] + rand(Normal(0, spread_y)), 0, grid_height)
        else
            # More dispersed location using Beta distribution for edge preference
            x = grid_width * rand(Beta(1.5, 1.5))  # Slight preference for central areas
            y = grid_height * rand(Beta(1.5, 1.5))
        end
        push!(facility_locs, (x, y))
    end
    
    # Generate customer locations (more clustered)
    customer_locs = Vector{Tuple{Float64,Float64}}()
    # Use Dirichlet distribution for more realistic cluster weights
    cluster_weights = rand(Dirichlet(ones(n_clusters)))
    
    for _ in 1:n_customers
        # Select cluster based on weights
        cluster_idx = sample(1:n_clusters, Weights(cluster_weights))
        center = cluster_centers[cluster_idx]
        
        # Generate location with realistic clustering using log-normal spread
        base_spread = grid_width * (1 - clustering_factor) * 0.08
        spread = rand(LogNormal(log(base_spread), 0.3))  # Realistic spread variation
        
        # Use bivariate normal for more realistic customer clustering
        x = clamp(center[1] + rand(Normal(0, spread)), 0, grid_width)
        y = clamp(center[2] + rand(Normal(0, spread)), 0, grid_height)
        push!(customer_locs, (x, y))
    end
    
    # Generate facility fixed costs (correlated with location and market size)
    fixed_costs = Dict{Int, Float64}()
    
    for f in 1:n_facilities
        # Calculate market potential based on proximity to customers
        distances_to_customers = [
            sqrt((facility_locs[f][1] - c[1])^2 + (facility_locs[f][2] - c[2])^2)
            for c in customer_locs
        ]
        market_potential = sum(exp.(-distances_to_customers ./ (grid_width * 0.2)))
        
        # Adjust cost based on market potential and location using realistic distributions
        location_factor = (facility_locs[f][1] / grid_width + facility_locs[f][2] / grid_height) / 2
        
        # Use log-normal distribution for cost variation (more realistic for facility costs)
        base_cost = min_fixed_cost + (max_fixed_cost - min_fixed_cost) *
                   (0.2 + 0.5 * market_potential / n_customers + 0.3 * location_factor)
        
        # Add realistic cost variation using log-normal distribution
        cost_multiplier = rand(LogNormal(log(1.0), 0.25))  # Mean=1, realistic variation
        fixed_costs[f] = base_cost * cost_multiplier
    end
    
    # Generate customer demands (correlated with cluster size)
    demands = Dict{Int, Float64}()
    
    for c in 1:n_customers
        # Find nearest cluster center
        distances_to_clusters = [
            sqrt((customer_locs[c][1] - center[1])^2 + (customer_locs[c][2] - center[2])^2)
            for center in cluster_centers
        ]
        _, cluster_idx = findmin(distances_to_clusters)
        
        # Base demand on cluster weight using realistic distribution
        cluster_influence = cluster_weights[cluster_idx]
        base_demand = min_demand + (max_demand - min_demand) *
                     (0.2 + 0.8 * cluster_influence)
        
        # Add realistic demand variation using log-normal distribution
        # This creates more realistic demand patterns with occasional high-demand customers
        demand_multiplier = rand(LogNormal(log(1.0), 0.4))  # More variation than costs
        demands[c] = base_demand * demand_multiplier
    end
    
    # Generate facility capacities
    total_demand = sum(values(demands))
    avg_capacity = (total_demand / n_facilities) * capacity_factor
    capacities = Dict{Int, Float64}()
    
    for f in 1:n_facilities
        # Capacity correlated with fixed cost using realistic distribution
        relative_cost = (fixed_costs[f] - minimum(values(fixed_costs))) / 
                       (maximum(values(fixed_costs)) - minimum(values(fixed_costs)))
        
        # Use Gamma distribution for capacity variation (realistic for facility capacity)
        base_capacity = avg_capacity * (0.6 + 0.8 * relative_cost)
        capacity_multiplier = rand(Gamma(3, 1/3))  # Mean=1, realistic variation
        capacities[f] = base_capacity * capacity_multiplier
    end
    
    # Generate transport costs and infrastructure availability
    transport_costs = Dict{Tuple{Int,Int,String}, Float64}()
    
    # First, determine infrastructure availability
    infrastructure = Dict{Tuple{Int,Int,String}, Bool}()
    
    for f in 1:n_facilities
        for c in 1:n_customers
            distance = sqrt(
                (facility_locs[f][1] - customer_locs[c][1])^2 +
                (facility_locs[f][2] - customer_locs[c][2])^2
            )
            
            for mode in transport_modes
                # Determine if this transport mode is available for this route
                prob_available = if mode == "truck"
                    0.98  # Trucks almost always available
                elseif mode == "rail"
                    # Rail more likely for longer distances and between clusters
                    min(0.8, 0.3 + 0.5 * (distance / sqrt(grid_width^2 + grid_height^2)))
                elseif mode == "ship"
                    # Ships only available if near water (simplified)
                    any(loc -> abs(loc[2]) < grid_height * 0.1, [facility_locs[f], customer_locs[c]]) ? 0.8 : 0.0
                else  # air
                    # Air more likely for long distances
                    distance > sqrt(grid_width^2 + grid_height^2) * 0.3 ? 0.7 : 0.2
                end
                
                infrastructure[(f,c,mode)] = rand() < prob_available * infrastructure_density
                
                if infrastructure[(f,c,mode)]
                    # Calculate transport cost based on distance and mode
                    base_cost = get(transport_base_costs, mode, 1.0)
                    
                    # Add route-specific factors with realistic distributions
                    terrain_factor = rand(LogNormal(log(1.0), 0.15))  # Realistic terrain variation
                    volume_factor = 1.0 - 0.25 * (demands[c] / maximum(values(demands)))  # Volume discount
                    
                    # Add congestion/efficiency factor using Beta distribution
                    efficiency_factor = rand(Beta(3, 2)) * 0.4 + 0.8  # 0.8-1.2 range
                    
                    transport_costs[(f,c,mode)] = base_cost * distance * terrain_factor * volume_factor * efficiency_factor
                end
            end
        end
    end
    
    # Generate mode capacities
    mode_capacities = Dict{String, Float64}()
    for mode in transport_modes
        # Base capacity on total demand and mode characteristics
        base_capacity = total_demand * mode_capacity_factor
        
        # Adjust by mode type with realistic capacity distributions
        capacity_multiplier = if mode == "truck"
            rand(Gamma(4, 0.25))  # Mean=1.0, moderate variation
        elseif mode == "rail"
            rand(Gamma(6, 0.33))  # Mean=2.0, higher capacity for rail
        elseif mode == "ship"
            rand(Gamma(9, 0.33))  # Mean=3.0, highest capacity for ships
        else  # air
            rand(Gamma(2, 0.25))  # Mean=0.5, lower capacity for air
        end
        
        mode_capacities[mode] = base_capacity * capacity_multiplier
    end
    
    # Clean up transport costs to only include available routes
    transport_costs = Dict(k => v for (k,v) in transport_costs if infrastructure[k])
    
    # Store generated data in params
    actual_params[:facilities] = collect(1:n_facilities)
    actual_params[:customers] = collect(1:n_customers)
    actual_params[:facility_locs] = facility_locs
    actual_params[:customer_locs] = customer_locs
    actual_params[:cluster_centers] = cluster_centers
    actual_params[:cluster_weights] = cluster_weights
    actual_params[:fixed_costs] = fixed_costs
    actual_params[:demands] = demands
    actual_params[:capacities] = capacities
    actual_params[:transport_costs] = transport_costs
    actual_params[:mode_capacities] = mode_capacities
    actual_params[:total_demand] = total_demand
    
    # Create model
    model = Model()
    
    # Decision variables
    @variable(model, y[1:n_facilities], Bin)  # 1 if facility is opened
    
    # Create a set of valid (facility, customer, mode) combinations
    valid_combinations = [(f,c,m) for f in 1:n_facilities, c in 1:n_customers, m in transport_modes 
                          if haskey(transport_costs, (f,c,m))]
    
    # Shipping quantities
    @variable(model, x[valid_combinations] >= 0)
    
    # Objective: Minimize total cost
    @objective(model, Min,
        sum(fixed_costs[f] * y[f] for f in 1:n_facilities) +
        sum(transport_costs[combo] * x[combo] for combo in valid_combinations)
    )
    
    # Customer demand satisfaction
    for c in 1:n_customers
        combos_for_customer = filter(combo -> combo[2] == c, valid_combinations)
        @constraint(model,
            sum(x[combo] for combo in combos_for_customer) >= demands[c]
        )
    end
    
    # Facility capacity constraints
    for f in 1:n_facilities
        combos_for_facility = filter(combo -> combo[1] == f, valid_combinations)
        @constraint(model,
            sum(x[combo] for combo in combos_for_facility) <= capacities[f] * y[f]
        )
    end
    
    # Transport mode capacity constraints
    for m in transport_modes
        combos_for_mode = filter(combo -> combo[3] == m, valid_combinations)
        @constraint(model,
            sum(x[combo] for combo in combos_for_mode) <= mode_capacities[m]
        )
    end
    
    return model, actual_params
end

"""
    sample_supply_chain_parameters(size::Symbol=:medium; seed::Int=0)

Sample realistic parameters for a supply chain optimization problem.

# Arguments
- `size`: Symbol specifying the problem size (:small, :medium, :large)
  - :small: Local/regional supply chain (50-250 variables)
  - :medium: Regional/national supply chain (250-1000 variables)
  - :large: National/global supply chain (1000-10000 variables)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_supply_chain_parameters(size::Symbol=:medium; seed::Int=0)
    Random.seed!(seed)
    
    # Sample target variable count from realistic ranges
    target_variables = if size == :small
        # Local/regional supply chain: 50-250 variables
        rand(50:250)
    elseif size == :medium
        # Regional/national supply chain: 250-1000 variables  
        rand(250:1000)
    elseif size == :large
        # National/global supply chain: 1000-10000 variables
        rand(1000:10000)
    else
        error("Unknown size: $size. Must be :small, :medium, or :large")
    end
    
    # Use the target variable approach to generate parameters
    params = sample_supply_chain_parameters(target_variables; seed=seed)
    
    # Override some parameters based on size category for realistic scaling
    if size == :small
        # Local/regional supply chain characteristics
        params[:grid_width] = rand(Uniform(200.0, 800.0))
        params[:grid_height] = rand(Uniform(200.0, 800.0))
        # Lower fixed costs for smaller facilities
        base_fixed = rand(LogNormal(log(300000), 0.5))
        params[:min_fixed_cost] = max(100000.0, base_fixed)
        params[:max_fixed_cost] = params[:min_fixed_cost] * rand(Uniform(1.8, 3.5))
        # Lower demand range for local markets
        base_demand = rand(Uniform(80.0, 150.0))
        params[:min_demand] = base_demand
        params[:max_demand] = base_demand * rand(Uniform(3.0, 8.0))
        # Higher clustering for local operations
        params[:clustering_factor] = rand(Beta(3, 2)) * 0.6 + 0.25  # 0.25-0.85
        # Good infrastructure density for local operations
        params[:infrastructure_density] = rand(Beta(5, 2)) * 0.3 + 0.7  # 0.7-1.0
        
    elseif size == :medium
        # Regional/national supply chain characteristics
        params[:grid_width] = rand(Uniform(800.0, 2000.0))
        params[:grid_height] = rand(Uniform(800.0, 2000.0))
        # Medium fixed costs for regional facilities
        base_fixed = rand(LogNormal(log(800000), 0.6))
        params[:min_fixed_cost] = max(300000.0, base_fixed)
        params[:max_fixed_cost] = params[:min_fixed_cost] * rand(Uniform(2.0, 4.0))
        # Medium demand range for regional markets
        base_demand = rand(Uniform(150.0, 300.0))
        params[:min_demand] = base_demand
        params[:max_demand] = base_demand * rand(Uniform(4.0, 12.0))
        # Moderate clustering for regional operations
        params[:clustering_factor] = rand(Beta(2, 3)) * 0.5 + 0.2  # 0.2-0.7
        # Moderate infrastructure density for regional operations
        params[:infrastructure_density] = rand(Beta(3, 2)) * 0.4 + 0.5  # 0.5-0.9
        
    elseif size == :large
        # National/global supply chain characteristics
        params[:grid_width] = rand(Uniform(2000.0, 5000.0))
        params[:grid_height] = rand(Uniform(2000.0, 5000.0))
        # Higher fixed costs for large facilities
        base_fixed = rand(LogNormal(log(1500000), 0.7))
        params[:min_fixed_cost] = max(500000.0, base_fixed)
        params[:max_fixed_cost] = params[:min_fixed_cost] * rand(Uniform(2.5, 5.0))
        # Higher demand range for global markets
        base_demand = rand(Uniform(300.0, 600.0))
        params[:min_demand] = base_demand
        params[:max_demand] = base_demand * rand(Uniform(6.0, 20.0))
        # Lower clustering for global operations (more dispersed)
        params[:clustering_factor] = rand(Beta(1, 3)) * 0.4 + 0.15  # 0.15-0.55
        # Lower infrastructure density for global operations
        params[:infrastructure_density] = rand(Beta(2, 3)) * 0.4 + 0.4  # 0.4-0.8
    end
    
    # Capacity and efficiency parameters with realistic distributions
    params[:capacity_factor] = rand(Uniform(1.2, 2.2))  # Facility capacity vs demand
    params[:mode_capacity_factor] = rand(Uniform(0.25, 0.65))  # Transport mode capacity
    
    # Transport modes and costs with realistic distributions
    params[:transport_modes] = ["truck", "rail", "ship", "air"]
    
    # Transport costs based on realistic cost per unit-distance
    params[:transport_base_costs] = Dict(
        "truck" => rand(Gamma(4, 0.25)),  # Mean ~1.0, realistic variation
        "rail" => rand(Gamma(3, 0.2)),    # Mean ~0.6, lower cost than truck
        "ship" => rand(Gamma(2, 0.15)),   # Mean ~0.3, lowest cost for bulk
        "air" => rand(Gamma(6, 0.5))      # Mean ~3.0, highest cost for speed
    )
    
    return params
end

"""
    sample_supply_chain_parameters(target_variables::Int; seed::Int=0)

Sample realistic parameters for a supply chain optimization problem targeting approximately the specified number of variables.

# Arguments
- `target_variables`: Target number of variables in the LP formulation (approximately within ±10%)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_supply_chain_parameters(target_variables::Int; seed::Int=0)
    Random.seed!(seed)
    
    params = Dict{Symbol, Any}()
    
    # Determine appropriate size category and initial parameters based on target
    if target_variables <= 250
        # Small size category (50-250 variables)
        size_category = :small
        params[:n_facilities] = rand(DiscreteUniform(3, 8))
        params[:n_customers] = rand(DiscreteUniform(15, 35))
        params[:n_transport_modes] = rand(DiscreteUniform(1, 2))
        params[:grid_width] = rand(Uniform(200.0, 800.0))
        params[:grid_height] = rand(Uniform(200.0, 800.0))
        # Lower fixed costs for smaller facilities
        base_fixed = rand(LogNormal(log(300000), 0.5))
        params[:min_fixed_cost] = max(100000.0, base_fixed)
        params[:max_fixed_cost] = params[:min_fixed_cost] * rand(Uniform(1.8, 3.5))
        params[:infrastructure_density] = rand(Beta(5, 2)) * 0.3 + 0.7  # 0.7-1.0
        params[:clustering_factor] = rand(Beta(3, 2)) * 0.6 + 0.25  # 0.25-0.85
    elseif target_variables <= 1000
        # Medium size category (250-1000 variables)
        size_category = :medium
        params[:n_facilities] = rand(DiscreteUniform(6, 18))
        params[:n_customers] = rand(DiscreteUniform(25, 65))
        params[:n_transport_modes] = rand(DiscreteUniform(2, 3))
        params[:grid_width] = rand(Uniform(800.0, 2000.0))
        params[:grid_height] = rand(Uniform(800.0, 2000.0))
        # Medium fixed costs for regional facilities
        base_fixed = rand(LogNormal(log(800000), 0.6))
        params[:min_fixed_cost] = max(300000.0, base_fixed)
        params[:max_fixed_cost] = params[:min_fixed_cost] * rand(Uniform(2.0, 4.0))
        params[:infrastructure_density] = rand(Beta(3, 2)) * 0.4 + 0.5  # 0.5-0.9
        params[:clustering_factor] = rand(Beta(2, 3)) * 0.5 + 0.2  # 0.2-0.7
    else
        # Large size category (1000-10000 variables)
        size_category = :large
        params[:n_facilities] = rand(DiscreteUniform(12, 40))
        params[:n_customers] = rand(DiscreteUniform(60, 200))
        params[:n_transport_modes] = rand(DiscreteUniform(3, 4))
        params[:grid_width] = rand(Uniform(2000.0, 5000.0))
        params[:grid_height] = rand(Uniform(2000.0, 5000.0))
        # Higher fixed costs for large facilities
        base_fixed = rand(LogNormal(log(1500000), 0.7))
        params[:min_fixed_cost] = max(500000.0, base_fixed)
        params[:max_fixed_cost] = params[:min_fixed_cost] * rand(Uniform(2.5, 5.0))
        params[:infrastructure_density] = rand(Beta(2, 3)) * 0.4 + 0.4  # 0.4-0.8
        params[:clustering_factor] = rand(Beta(1, 3)) * 0.4 + 0.15  # 0.15-0.55
    end
    
    # Common parameters with realistic distributions based on size category
    if size_category == :small
        base_demand = rand(Uniform(80.0, 150.0))
        params[:min_demand] = base_demand
        params[:max_demand] = base_demand * rand(Uniform(3.0, 8.0))
    elseif size_category == :medium
        base_demand = rand(Uniform(150.0, 300.0))
        params[:min_demand] = base_demand
        params[:max_demand] = base_demand * rand(Uniform(4.0, 12.0))
    else  # large
        base_demand = rand(Uniform(300.0, 600.0))
        params[:min_demand] = base_demand
        params[:max_demand] = base_demand * rand(Uniform(6.0, 20.0))
    end
    
    params[:capacity_factor] = rand(Uniform(1.2, 2.2))
    params[:mode_capacity_factor] = rand(Uniform(0.25, 0.65))
    params[:transport_modes] = ["truck", "rail", "ship", "air"]
    
    # Transport costs with realistic distributions
    params[:transport_base_costs] = Dict(
        "truck" => rand(Gamma(4, 0.25)),  # Mean ~1.0
        "rail" => rand(Gamma(3, 0.2)),    # Mean ~0.6
        "ship" => rand(Gamma(2, 0.15)),   # Mean ~0.3
        "air" => rand(Gamma(6, 0.5))      # Mean ~3.0
    )
    
    # Iteratively adjust parameters to reach target
    for iteration in 1:20
        # Test actual model generation for more accurate variable count
        if iteration <= 10
            # Use calculation for first 10 iterations for speed
            current_vars = calculate_supply_chain_variable_count(params)
        else
            # Use actual model generation for final iterations for accuracy
            try
                test_model, _ = generate_supply_chain_problem(params; seed=seed)
                current_vars = JuMP.num_variables(test_model)
            catch
                # Fall back to calculation if model generation fails
                current_vars = calculate_supply_chain_variable_count(params)
            end
        end
        
        if abs(current_vars - target_variables) / target_variables < 0.1
            break  # Within 10% tolerance
        end
        
        # Adjust parameters based on current vs target
        ratio = target_variables / current_vars
        
        # More aggressive adjustments for better convergence
        if ratio > 1.5  # Need significantly more variables
            # Increase problem size parameters more aggressively
            if size_category == :small
                params[:n_facilities] = min(8, round(Int, params[:n_facilities] * min(1.5, ratio^0.7)))
                params[:n_customers] = min(35, round(Int, params[:n_customers] * min(1.5, ratio^0.7)))
                params[:n_transport_modes] = min(2, round(Int, params[:n_transport_modes] * min(1.3, ratio^0.5)))
            elseif size_category == :medium
                params[:n_facilities] = min(18, round(Int, params[:n_facilities] * min(1.5, ratio^0.7)))
                params[:n_customers] = min(65, round(Int, params[:n_customers] * min(1.5, ratio^0.7)))
                params[:n_transport_modes] = min(3, round(Int, params[:n_transport_modes] * min(1.3, ratio^0.5)))
            else  # large
                params[:n_facilities] = min(40, round(Int, params[:n_facilities] * min(1.5, ratio^0.7)))
                params[:n_customers] = min(200, round(Int, params[:n_customers] * min(1.5, ratio^0.7)))
                params[:n_transport_modes] = min(4, round(Int, params[:n_transport_modes] * min(1.3, ratio^0.5)))
            end
            # Also increase infrastructure density to get more valid combinations
            params[:infrastructure_density] = min(0.95, params[:infrastructure_density] * 1.1)
            
        elseif ratio < 0.6  # Need significantly fewer variables
            # Decrease problem size parameters more aggressively
            if size_category == :small
                params[:n_facilities] = max(3, round(Int, params[:n_facilities] * max(0.7, ratio^0.7)))
                params[:n_customers] = max(15, round(Int, params[:n_customers] * max(0.7, ratio^0.7)))
                params[:n_transport_modes] = max(1, round(Int, params[:n_transport_modes] * max(0.8, ratio^0.5)))
            elseif size_category == :medium
                params[:n_facilities] = max(6, round(Int, params[:n_facilities] * max(0.7, ratio^0.7)))
                params[:n_customers] = max(25, round(Int, params[:n_customers] * max(0.7, ratio^0.7)))
                params[:n_transport_modes] = max(2, round(Int, params[:n_transport_modes] * max(0.8, ratio^0.5)))
            else  # large
                params[:n_facilities] = max(12, round(Int, params[:n_facilities] * max(0.7, ratio^0.7)))
                params[:n_customers] = max(60, round(Int, params[:n_customers] * max(0.7, ratio^0.7)))
                params[:n_transport_modes] = max(3, round(Int, params[:n_transport_modes] * max(0.8, ratio^0.5)))
            end
            # Also decrease infrastructure density to get fewer valid combinations
            params[:infrastructure_density] = max(0.5, params[:infrastructure_density] * 0.9)
            
        else  # Fine-tune with more precise adjustments
            # Small adjustments within size category bounds
            adjustment_factor = 1.0 + (ratio - 1.0) * 0.3  # Damped adjustment
            
            if size_category == :small
                params[:n_facilities] = max(3, min(8, round(Int, params[:n_facilities] * adjustment_factor)))
                params[:n_customers] = max(15, min(35, round(Int, params[:n_customers] * adjustment_factor)))
                params[:n_transport_modes] = max(1, min(2, round(Int, params[:n_transport_modes] * adjustment_factor)))
            elseif size_category == :medium
                params[:n_facilities] = max(6, min(18, round(Int, params[:n_facilities] * adjustment_factor)))
                params[:n_customers] = max(25, min(65, round(Int, params[:n_customers] * adjustment_factor)))
                params[:n_transport_modes] = max(2, min(3, round(Int, params[:n_transport_modes] * adjustment_factor)))
            else  # large
                params[:n_facilities] = max(12, min(40, round(Int, params[:n_facilities] * adjustment_factor)))
                params[:n_customers] = max(60, min(200, round(Int, params[:n_customers] * adjustment_factor)))
                params[:n_transport_modes] = max(3, min(4, round(Int, params[:n_transport_modes] * adjustment_factor)))
            end
            
            # Fine-tune infrastructure density
            if ratio > 1.0
                params[:infrastructure_density] = min(0.95, params[:infrastructure_density] * 1.05)
            else
                params[:infrastructure_density] = max(0.5, params[:infrastructure_density] * 0.95)
            end
        end
    end
    
    return params
end

function calculate_supply_chain_variable_count(params::Dict)
    # Extract parameters with defaults
    n_facilities = get(params, :n_facilities, 8)
    n_customers = get(params, :n_customers, 25)
    n_transport_modes = get(params, :n_transport_modes, 3)
    infrastructure_density = get(params, :infrastructure_density, 0.7)
    all_transport_modes = get(params, :transport_modes, ["truck", "rail", "ship"])
    
    # Select transport modes (same logic as in generate function)
    selected_modes = min(n_transport_modes, length(all_transport_modes))
    
    # Variables: y[1:n_facilities] - binary variables for facility opening
    facility_vars = n_facilities
    
    # Variables: x[valid_combinations] - continuous variables for shipping quantities
    # Calculate expected number of valid combinations based on infrastructure logic
    expected_valid_combinations = 0.0
    
    # Calculate expected combinations for each transport mode
    for f in 1:n_facilities
        for c in 1:n_customers
            # Expected valid modes for this facility-customer pair
            expected_valid_modes = 0.0
            
            # Check each possible transport mode
            if "truck" in all_transport_modes[1:selected_modes]
                # Truck almost always available (98% probability)
                expected_valid_modes += 0.98
            end
            
            if "rail" in all_transport_modes[1:selected_modes]
                # Rail: expected probability is average of min(0.8, 0.3 + 0.5 * distance_ratio)
                # For uniform distance distribution, average distance ratio is 0.5
                avg_rail_prob = min(0.8, 0.3 + 0.5 * 0.5)  # = 0.55
                expected_valid_modes += avg_rail_prob * infrastructure_density
            end
            
            if "ship" in all_transport_modes[1:selected_modes]
                # Ships: simplified 20% base probability
                expected_valid_modes += 0.2 * infrastructure_density
            end
            
            if "air" in all_transport_modes[1:selected_modes]
                # Air: weighted average of long-distance (70% of 70% routes) and short-distance (20% of 30% routes)
                # Assuming 70% of routes are long-distance based on threshold
                avg_air_prob = 0.7 * 0.7 + 0.3 * 0.2  # = 0.55
                expected_valid_modes += avg_air_prob * infrastructure_density
            end
            
            expected_valid_combinations += expected_valid_modes
        end
    end
    
    # Use a conservative estimate (80% of expected combinations to account for filtering)
    shipping_vars = Int(round(expected_valid_combinations * 0.8))
    
    # Ensure at least some shipping variables exist (at least one per facility)
    if shipping_vars < n_facilities
        shipping_vars = n_facilities
    end
    
    # Total variables
    return facility_vars + shipping_vars
end

# Register the problem type
register_problem(
    :supply_chain,
    generate_supply_chain_problem,
    sample_supply_chain_parameters,
    "Supply chain optimization problem that minimizes facility and transportation costs while meeting customer demands and respecting capacity constraints"
)