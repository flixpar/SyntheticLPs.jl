using JuMP
using Random
using StatsBase
using Distributions

"""
    generate_energy_problem(params::Dict=Dict(); seed::Int=0)

Generate an energy generation mix optimization problem instance using sophisticated probability distributions.

# Arguments
- `params`: Dictionary of problem parameters
  - `:n_sources`: Number of power generation sources (default: 6)
  - `:n_periods`: Number of time periods (e.g., 24 for hourly day-ahead planning) (default: 24)
  - `:renewable_fraction`: Target fraction of renewable sources (default: 0.4)
  - `:peak_demand`: Maximum demand in any period (MW) (default: 1000.0)
  - `:demand_variation`: Variation in demand between peak and off-peak (default: 0.3)
  - `:base_generation_cost`: Base cost per MWh for conventional sources (default: 50.0)
  - `:renewable_cost_factor`: Cost multiplier for renewable vs conventional sources (default: 1.2)
  - `:capacity_margin`: Required capacity margin over peak demand (default: 1.3)
  - `:emission_limit`: Maximum emissions per MWh (default: 0.5)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- `model`: The JuMP model with realistic energy optimization constraints
- `params`: Dictionary of all parameters used (including defaults)

# Features
- Uses sophisticated probability distributions reflecting real energy market dynamics:
  - Generation costs: Source-specific distributions (LogNormal for fossil fuels, Gamma for renewables)
  - Capacity allocation: Market-realistic distributions (Gamma for baseload, Beta for distributed)
  - Demand patterns: Multi-layered variability (weather, economic, random, seasonal effects)
- Supports multiple energy source types (coal, gas, nuclear, solar, wind, hydro, biomass)
- Implements load-type-specific demand patterns (residential, commercial, industrial)
- Includes capacity constraints, emission limits, and renewable targets
- Scales appropriately for different problem sizes (residential to utility-scale)
"""
function generate_energy_problem(params::Dict=Dict(); seed::Int=0)
    # Set random seed
    Random.seed!(seed)
    
    # Extract parameters with defaults
    n_sources = get(params, :n_sources, 6)
    n_periods = get(params, :n_periods, 24)
    renewable_fraction = get(params, :renewable_fraction, 0.4)
    peak_demand = get(params, :peak_demand, 1000.0)
    demand_variation = get(params, :demand_variation, 0.3)
    base_generation_cost = get(params, :base_generation_cost, 50.0)
    renewable_cost_factor = get(params, :renewable_cost_factor, 1.2)
    capacity_margin = get(params, :capacity_margin, 1.3)
    emission_limit = get(params, :emission_limit, 0.5)
    
    # Save actual parameters used
    actual_params = Dict{Symbol, Any}(
        :n_sources => n_sources,
        :n_periods => n_periods,
        :renewable_fraction => renewable_fraction,
        :peak_demand => peak_demand,
        :demand_variation => demand_variation,
        :base_generation_cost => base_generation_cost,
        :renewable_cost_factor => renewable_cost_factor,
        :capacity_margin => capacity_margin,
        :emission_limit => emission_limit
    )
    
    # Define source types and their characteristics
    # (name, renewable, availability, capacity_factor, cost_factor)
    source_types = [
        ("coal", false, 0.95, 0.9, 1.0),
        ("gas", false, 0.98, 0.85, 1.2),
        ("nuclear", false, 0.92, 0.95, 0.8),
        ("solar", true, 0.99, 0.25, 0.3),
        ("wind", true, 0.95, 0.35, 0.4),
        ("hydro", true, 0.90, 0.50, 0.6),
        ("biomass", true, 0.88, 0.75, 1.1)
    ]
    
    # Select sources ensuring minimum renewable fraction
    n_renewables = max(1, ceil(Int, n_sources * renewable_fraction))
    n_conventional = n_sources - n_renewables
    
    # Make sure we don't try to sample more sources than available
    renewable_indices = findall(s -> s[2], source_types)
    conventional_indices = findall(s -> !s[2], source_types)
    
    n_renewables = min(n_renewables, length(renewable_indices))
    n_conventional = min(n_conventional, length(conventional_indices))
    
    renewable_sources = source_types[sample(renewable_indices, n_renewables, replace=false)]
    conventional_sources = source_types[sample(conventional_indices, n_conventional, replace=false)]
    selected_sources = vcat(renewable_sources, conventional_sources)
    
    sources = [s[1] for s in selected_sources]
    time_periods = collect(1:n_periods)
    
    # Generate generation costs with realistic market-driven distributions
    generation_costs = Dict{String, Float64}()
    for (i, (name, is_renewable, _, _, cost_factor)) in enumerate(selected_sources)
        base_cost = base_generation_cost * cost_factor
        if is_renewable
            base_cost *= renewable_cost_factor
        end
        
        # Add realistic cost variation based on source type and market dynamics
        if name in ["coal", "gas"]
            # Fossil fuels: volatile due to commodity price fluctuations
            variation = rand(LogNormal(log(1.0), 0.15))  # 15% volatility
        elseif name == "nuclear"
            # Nuclear: stable long-term costs but high capital uncertainty
            variation = rand(Normal(1.0, 0.08))  # 8% variation, more stable
        elseif name in ["solar", "wind"]
            # Modern renewables: rapidly declining costs with learning curves
            variation = rand(Gamma(8, 0.12))  # Slight skew toward lower costs
        else
            # Other sources (hydro, biomass): moderate variation
            variation = rand(Normal(1.0, 0.12))  # 12% variation
        end
        
        generation_costs[name] = base_cost * max(0.3, variation)
    end
    
    # Generate capacities based on peak demand and realistic market distribution
    capacities = Dict{String, Float64}()
    total_required_capacity = peak_demand * capacity_margin
    
    # Distribute capacity among sources considering their characteristics and market dynamics
    capacity_shares = Float64[]
    for (name, is_renewable, availability, capacity_factor, _) in selected_sources
        # Different capacity distribution patterns for different source types
        if name == "coal"
            # Coal: traditionally large baseload plants, but declining
            share = rand(Gamma(2, 0.25))  # Mean ~0.5, some large plants remain
        elseif name == "gas"
            # Natural gas: flexible, often used for peaking and intermediate load
            share = rand(Gamma(3, 0.15))  # Mean ~0.45, widely distributed
        elseif name == "nuclear"
            # Nuclear: very large baseload plants, limited number
            share = rand(Gamma(1.5, 0.4))  # Mean ~0.6, few large plants
        elseif name == "solar"
            # Solar: distributed generation, many small to medium installations
            share = rand(Beta(2, 4))  # Mean ~0.33, distributed pattern
        elseif name == "wind"
            # Wind: mix of large wind farms and smaller installations
            share = rand(Beta(3, 3))  # Mean ~0.5, moderate distribution
        elseif name == "hydro"
            # Hydro: varies greatly by geography, existing infrastructure
            share = rand(LogNormal(log(0.3), 0.6))  # Wide variation, geographical constraints
        else  # biomass, other
            # Other renewables: typically smaller scale, niche applications
            share = rand(Beta(2, 5))  # Mean ~0.29, smaller scale
        end
        
        push!(capacity_shares, share)
    end
    
    # Normalize capacity shares to ensure they sum to reasonable total
    total_share = sum(capacity_shares)
    for (i, (name, _, availability, capacity_factor, _)) in enumerate(selected_sources)
        normalized_share = capacity_shares[i] / total_share
        # Account for availability and capacity factors in realistic capacity sizing
        effective_capacity = total_required_capacity * normalized_share / (availability * capacity_factor)
        capacities[name] = max(10.0, effective_capacity)  # Minimum 10 MW capacity
    end
    
    # Generate realistic demand profile with sophisticated variability patterns
    demands = Float64[]
    
    # Create multiple demand patterns based on different contexts
    # Base hourly patterns for different load profiles
    residential_pattern = [0.6, 0.55, 0.5, 0.5, 0.55, 0.7, 0.85, 1.0, 0.95, 0.9, 0.85, 0.9, 0.95, 1.0, 0.9, 0.85, 0.9, 0.95, 1.0, 0.95, 0.9, 0.8, 0.7, 0.65]
    commercial_pattern = [0.4, 0.35, 0.3, 0.3, 0.35, 0.5, 0.7, 0.9, 1.0, 1.0, 0.95, 0.9, 0.85, 0.9, 0.95, 1.0, 0.95, 0.9, 0.75, 0.6, 0.5, 0.45, 0.4, 0.35]
    industrial_pattern = [0.8, 0.75, 0.7, 0.7, 0.75, 0.85, 0.95, 1.0, 1.0, 0.95, 0.9, 0.85, 0.9, 0.95, 1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.7, 0.75, 0.8]
    
    # Choose pattern based on peak demand size (proxy for customer type)
    if peak_demand < 100
        hour_factors = residential_pattern  # Small scale, residential-like
    elseif peak_demand < 1000
        hour_factors = commercial_pattern   # Medium scale, commercial-like
    else
        hour_factors = industrial_pattern   # Large scale, industrial-like
    end
    
    # Generate demand for each period
    base_demand = peak_demand * (1 - demand_variation)
    
    if n_periods == 24
        for h in 1:24
            # Base demand from pattern
            pattern_demand = base_demand + (peak_demand - base_demand) * hour_factors[h]
            
            # Add multiple sources of realistic variability
            # 1. Weather-related variation (affects both heating/cooling and renewable output)
            weather_effect = rand(Normal(1.0, 0.03))  # 3% weather variation
            
            # 2. Economic activity variation (affects commercial/industrial demand)
            economic_effect = rand(Normal(1.0, 0.02))  # 2% economic variation
            
            # 3. Random demand fluctuations (measurement errors, unexpected events)
            random_effect = rand(Normal(1.0, 0.025))  # 2.5% random variation
            
            # 4. Seasonal/long-term trend (very mild for daily patterns)
            seasonal_effect = rand(Normal(1.0, 0.01))  # 1% seasonal variation
            
            # Combine all effects multiplicatively
            total_effect = weather_effect * economic_effect * random_effect * seasonal_effect
            demand = pattern_demand * max(0.7, min(1.4, total_effect))  # Clamp to reasonable range
            
            push!(demands, demand)
        end
    else
        # For different period lengths, interpolate and add appropriate variability
        for p in 1:n_periods
            relative_hour = (p - 1) * 24 / n_periods
            hour_idx = 1 + floor(Int, relative_hour % 24)
            if hour_idx > 24
                hour_idx = 24
            end
            
            # Base demand from pattern
            pattern_demand = base_demand + (peak_demand - base_demand) * hour_factors[hour_idx]
            
            # Scale variability with period length (longer periods = more aggregation = less variability)
            variability_scale = sqrt(24 / n_periods)  # Longer periods have less relative variability
            
            # Combined variability effects
            weather_effect = rand(Normal(1.0, 0.03 * variability_scale))
            economic_effect = rand(Normal(1.0, 0.02 * variability_scale))
            random_effect = rand(Normal(1.0, 0.025 * variability_scale))
            seasonal_effect = rand(Normal(1.0, 0.01 * variability_scale))
            
            total_effect = weather_effect * economic_effect * random_effect * seasonal_effect
            demand = pattern_demand * max(0.7, min(1.4, total_effect))
            
            push!(demands, demand)
        end
    end
    
    # Generate emission limits
    emission_limits = Dict{String, Float64}()
    for (name, is_renewable, _, _, _) in selected_sources
        if is_renewable
            emission_limits[name] = 0.0
        else
            if name == "coal"
                emission_limits[name] = emission_limit
            elseif name == "gas"
                emission_limits[name] = emission_limit * 0.5
            else  # nuclear
                emission_limits[name] = 0.0
            end
        end
    end
    
    # Store generated data in params
    actual_params[:sources] = sources
    actual_params[:time_periods] = time_periods
    actual_params[:generation_costs] = generation_costs
    actual_params[:capacities] = capacities
    actual_params[:demands] = demands
    actual_params[:emission_limits] = emission_limits
    
    # Build the model
    model = Model()
    
    # Decision variables: power generated from each source in each period
    @variable(model, 0 <= x[s in sources, t in time_periods] <= capacities[s])
    
    # Objective: Minimize total cost
    @objective(model, Min,
        sum(generation_costs[s] * x[s,t]
            for s in sources, t in time_periods)
    )
    
    # Meet demand in each period
    for t in time_periods
        @constraint(model,
            sum(x[s,t] for s in sources) >= demands[t]
        )
    end
    
    # Emissions constraints
    max_emission = maximum(values(emission_limits))
    for t in time_periods
        @constraint(model,
            sum(emission_limits[s] * x[s,t] for s in sources) <=
            sum(x[s,t] for s in sources) * max_emission
        )
    end
    
    # Minimum renewables constraint
    renewable_sources = [s for s in sources if emission_limits[s] == 0.0]
    for t in time_periods
        @constraint(model,
            sum(x[s,t] for s in renewable_sources) >=
            renewable_fraction * sum(x[s,t] for s in sources)
        )
    end
    
    return model, actual_params
end

"""
    sample_energy_parameters(size::Symbol=:medium; seed::Int=0)

Sample realistic parameters for an energy generation mix problem using probability distributions.

# Arguments
- `size`: Symbol specifying the problem size and scale:
  - `:small`: Residential/small commercial (50-250 variables, 5-50 MW)
  - `:medium`: Commercial/industrial (250-1000 variables, 50-500 MW) 
  - `:large`: Utility scale (1000-10000 variables, 500-5000 MW)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters using realistic probability distributions:
  - Renewable fraction: Beta distribution with scale-dependent parameters (higher targets for larger scales)
  - Demand variation: Beta distribution (higher volatility for smaller scales due to less aggregation)
  - Generation costs: Log-normal distribution with economies of scale (lower costs for larger scales)
  - Renewable cost factor: Gamma distribution reflecting scale-dependent cost competitiveness
  - Capacity margin: Normal distribution with scale-dependent reliability requirements
  - Emission limits: Beta distribution reflecting regulatory environment at different scales
  - Other parameters: Uniform distributions for discrete choices and realistic ranges
"""
function sample_energy_parameters(size::Symbol=:medium; seed::Int=0)
    Random.seed!(seed)
    
    params = Dict{Symbol, Any}()
    
    # Set size-dependent parameters with realistic energy scales
    if size == :small  # Residential/small commercial scale (50-250 variables)
        # Target 50-250 variables: adjust sources and periods accordingly
        target_vars = rand(50:250)
        params[:n_sources] = rand(4:8)  # Limited source diversity
        params[:n_periods] = max(6, div(target_vars, params[:n_sources]))  # Ensure target range
        # Peak demand: 5-50 MW (residential aggregation/small commercial)
        params[:peak_demand] = rand(Uniform(5.0, 50.0))
    elseif size == :medium  # Commercial/industrial scale (250-1000 variables)
        # Target 250-1000 variables: adjust sources and periods accordingly
        target_vars = rand(250:1000)
        params[:n_sources] = rand(6:12)  # More source diversity
        params[:n_periods] = max(24, div(target_vars, params[:n_sources]))  # Ensure target range
        # Peak demand: 50-500 MW (large commercial/small utility)
        params[:peak_demand] = rand(Uniform(50.0, 500.0))
    elseif size == :large  # Utility scale (1000-10000 variables)
        # Target 1000-10000 variables: adjust sources and periods accordingly
        target_vars = rand(1000:10000)
        params[:n_sources] = rand(8:20)  # Full source diversity
        params[:n_periods] = max(48, div(target_vars, params[:n_sources]))  # Ensure target range
        # Peak demand: 500-5000 MW (utility-scale grid)
        params[:peak_demand] = rand(Uniform(500.0, 5000.0))
    else
        error("Unknown size: $size. Must be :small, :medium, or :large")
    end
    
    # Parameters that scale with problem complexity using realistic distributions
    # Renewable fraction: Scale-dependent targets reflecting real energy transition patterns
    if size == :small
        # Small scale: More variable renewable adoption, early adopters vs laggards
        params[:renewable_fraction] = rand(Beta(2, 3))  # Mean ~0.4, wide range 0.1-0.8
    elseif size == :medium
        # Medium scale: Moderate renewable targets, corporate sustainability goals
        params[:renewable_fraction] = rand(Beta(3, 4))  # Mean ~0.43, moderate range
    else  # large
        # Large scale: Utility-scale renewable mandates, more stable targets
        params[:renewable_fraction] = rand(Beta(4, 3))  # Mean ~0.57, higher renewable targets
    end
    
    # Demand variation: Higher for smaller scales due to less aggregation
    if size == :small
        # Small scale: High volatility, weather-dependent, less predictable
        params[:demand_variation] = rand(Beta(2, 2))  # Mean ~0.5, more volatile
    elseif size == :medium
        # Medium scale: Moderate volatility, some aggregation benefits
        params[:demand_variation] = rand(Beta(3, 5))  # Mean ~0.375, moderate volatility
    else  # large
        # Large scale: Low volatility, law of large numbers, better forecasting
        params[:demand_variation] = rand(Beta(4, 8))  # Mean ~0.33, more stable
    end
    
    # Base generation cost: Scale-dependent costs reflecting economies of scale
    if size == :small
        # Small scale: Higher costs due to lack of economies of scale
        params[:base_generation_cost] = rand(LogNormal(log(60.0), 0.4))  # Mean ~60, higher costs
    elseif size == :medium
        # Medium scale: Moderate costs, some economies of scale
        params[:base_generation_cost] = rand(LogNormal(log(45.0), 0.3))  # Mean ~45, moderate costs
    else  # large
        # Large scale: Lower costs due to economies of scale
        params[:base_generation_cost] = rand(LogNormal(log(35.0), 0.25))  # Mean ~35, lower costs
    end
    
    # Renewable cost factor: Scale-dependent cost advantage
    if size == :small
        # Small scale: Renewables less cost-competitive
        params[:renewable_cost_factor] = rand(Gamma(3, 0.4))  # Mean ~1.2, renewables more expensive
    elseif size == :medium
        # Medium scale: Renewables approaching cost parity
        params[:renewable_cost_factor] = rand(Gamma(2.5, 0.35))  # Mean ~0.875, near parity
    else  # large
        # Large scale: Renewables cost-competitive or cheaper
        params[:renewable_cost_factor] = rand(Gamma(2, 0.3))  # Mean ~0.6, renewables cheaper
    end
    
    # Capacity margin: Scale-dependent reliability requirements
    if size == :small
        # Small scale: Higher margins for reliability, less sophisticated forecasting
        params[:capacity_margin] = max(1.15, min(1.6, rand(Normal(1.35, 0.08))))
    elseif size == :medium
        # Medium scale: Standard engineering margins
        params[:capacity_margin] = max(1.1, min(1.5, rand(Normal(1.25, 0.05))))
    else  # large
        # Large scale: Lower margins due to better forecasting and grid stability
        params[:capacity_margin] = max(1.05, min(1.3, rand(Normal(1.15, 0.04))))
    end
    
    # Emission limit: Scale-dependent regulatory environment
    if size == :small
        # Small scale: Variable local regulations, wide range
        params[:emission_limit] = rand(Beta(2, 2))  # Mean ~0.5, wide range for local regulations
    elseif size == :medium
        # Medium scale: State/regional regulations, moderate consistency
        params[:emission_limit] = rand(Beta(3, 3))  # Mean ~0.5, moderate range
    else  # large
        # Large scale: Federal/utility regulations, stricter and more consistent
        params[:emission_limit] = rand(Beta(4, 6))  # Mean ~0.4, stricter emissions limits
    end
    
    return params
end

"""
    sample_energy_parameters(target_variables::Int; seed::Int=0)

Sample realistic parameters for an energy generation mix problem targeting approximately the specified number of variables.

# Arguments
- `target_variables`: Target number of variables in the LP formulation (approximately within ±10%)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_energy_parameters(target_variables::Int; seed::Int=0)
    Random.seed!(seed)
    
    params = Dict{Symbol, Any}()
    
    # Determine problem scale based on target variables
    if target_variables < 250
        scale = :small
        min_sources, max_sources = 3, 8
        min_periods, max_periods = 12, 48
        peak_demand_range = (10.0, 100.0)
    elseif target_variables < 1000
        scale = :medium
        min_sources, max_sources = 5, 12
        min_periods, max_periods = 24, 72
        peak_demand_range = (100.0, 1000.0)
    else
        scale = :large
        min_sources, max_sources = 8, 20
        min_periods, max_periods = 48, 200
        peak_demand_range = (1000.0, 10000.0)
    end
    
    # Start with reasonable defaults based on scale
    params[:n_sources] = min_sources + 2
    params[:n_periods] = min_periods + 12
    
    # Iteratively adjust parameters to reach target
    for iteration in 1:15
        current_vars = calculate_energy_variable_count(params)
        
        if abs(current_vars - target_variables) / target_variables < 0.1
            break  # Within 10% tolerance
        end
        
        # Adjust parameters based on current vs target
        ratio = target_variables / current_vars
        
        if ratio > 1.1  # Need more variables
            # Prefer increasing periods over sources for realism
            if params[:n_periods] < max_periods
                params[:n_periods] = min(max_periods, round(Int, params[:n_periods] * sqrt(ratio)))
            elseif params[:n_sources] < max_sources
                params[:n_sources] = min(max_sources, round(Int, params[:n_sources] * sqrt(ratio)))
            end
        elseif ratio < 0.9  # Need fewer variables
            # Prefer decreasing periods over sources for realism
            if params[:n_periods] > min_periods
                params[:n_periods] = max(min_periods, round(Int, params[:n_periods] * sqrt(ratio)))
            elseif params[:n_sources] > min_sources
                params[:n_sources] = max(min_sources, round(Int, params[:n_sources] * sqrt(ratio)))
            end
        end
    end
    
    # Sample realistic parameters using distributions based on scale
    if scale == :small
        params[:renewable_fraction] = rand(Beta(2, 3))  # Mean ~0.4, more variable
        params[:demand_variation] = rand(Beta(2, 3))  # Mean ~0.4, more volatile
        params[:peak_demand] = rand(Uniform(peak_demand_range...))
    elseif scale == :medium
        params[:renewable_fraction] = rand(Beta(3, 4))  # Mean ~0.43, moderate variation
        params[:demand_variation] = rand(Beta(3, 5))  # Mean ~0.375, moderate volatility
        params[:peak_demand] = rand(Uniform(peak_demand_range...))
    else  # large
        params[:renewable_fraction] = rand(Beta(4, 5))  # Mean ~0.44, less variation
        params[:demand_variation] = rand(Beta(4, 8))  # Mean ~0.33, less volatility
        params[:peak_demand] = rand(Uniform(peak_demand_range...))
    end
    
    # Scale-dependent parameters with realistic distributions
    if scale == :small
        # Small scale: Higher costs, less renewable competitiveness, higher margins
        params[:base_generation_cost] = rand(LogNormal(log(60.0), 0.4))  # Mean ~60, higher costs
        params[:renewable_cost_factor] = rand(Gamma(3, 0.4))  # Mean ~1.2, renewables more expensive
        params[:capacity_margin] = max(1.15, min(1.6, rand(Normal(1.35, 0.08))))  # Higher margins
        params[:emission_limit] = rand(Beta(2, 2))  # Mean ~0.5, wide range for local regulations
    elseif scale == :medium
        # Medium scale: Moderate costs, renewable near parity, standard margins
        params[:base_generation_cost] = rand(LogNormal(log(45.0), 0.3))  # Mean ~45, moderate costs
        params[:renewable_cost_factor] = rand(Gamma(2.5, 0.35))  # Mean ~0.875, near parity
        params[:capacity_margin] = max(1.1, min(1.5, rand(Normal(1.25, 0.05))))  # Standard margins
        params[:emission_limit] = rand(Beta(3, 3))  # Mean ~0.5, moderate range
    else  # large
        # Large scale: Lower costs, renewables competitive, lower margins
        params[:base_generation_cost] = rand(LogNormal(log(35.0), 0.25))  # Mean ~35, lower costs
        params[:renewable_cost_factor] = rand(Gamma(2, 0.3))  # Mean ~0.6, renewables cheaper
        params[:capacity_margin] = max(1.05, min(1.3, rand(Normal(1.15, 0.04))))  # Lower margins
        params[:emission_limit] = rand(Beta(4, 6))  # Mean ~0.4, stricter emissions limits
    end
    
    return params
end

function calculate_energy_variable_count(params::Dict)
    # Extract parameters with defaults
    n_sources = get(params, :n_sources, 6)
    n_periods = get(params, :n_periods, 24)
    
    # Variables: x[s in sources, t in time_periods] - continuous variables for power generation
    return n_sources * n_periods
end

# Register the problem type
register_problem(
    :energy,
    generate_energy_problem,
    sample_energy_parameters,
    "Energy generation mix problem that optimizes the allocation of different energy sources to meet demand while minimizing costs and emissions"
)