using JuMP
using Random
using Distributions
using StatsBase

"""
    generate_cutting_stock_problem(params::Dict=Dict(); seed::Int=0)

Generate a cutting stock problem instance.

# Arguments
- `params`: Dictionary of problem parameters
  - `:n_piece_types`: Number of different piece lengths required (default: 10)
  - `:min_length`: Minimum piece length (default: 0.5)
  - `:max_length`: Maximum piece length (default: 3.0)
  - `:stock_length`: Length of stock material (default: 6.0)
  - `:demand_min`: Minimum demand for each piece type (default: 10)
  - `:demand_max`: Maximum demand for each piece type (default: 100)
  - `:max_patterns`: Maximum number of cutting patterns to consider (default: 50)
  - `:common_lengths`: List of commonly requested lengths (default: [1.0, 1.5, 2.0, 2.4])
  - `:common_length_prob`: Probability of using a common length (default: 0.4)
  - `:waste_factor`: Factor controlling waste tolerance (default: 0.1)
  - `:solution_status`: Control feasibility (:feasible, :infeasible, :all) (default: :feasible)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- `model`: The JuMP model
- `params`: Dictionary of all parameters used (including defaults)
"""
function generate_cutting_stock_problem(params::Dict=Dict(); seed::Int=0)
    # Set random seed
    Random.seed!(seed)
    
    # Extract parameters with defaults
    n_piece_types = get(params, :n_piece_types, 10)
    min_length = get(params, :min_length, 0.5)
    max_length = get(params, :max_length, 3.0)
    stock_length = get(params, :stock_length, 6.0)
    demand_min = get(params, :demand_min, 10)
    demand_max = get(params, :demand_max, 100)
    max_patterns = get(params, :max_patterns, 50)
    common_lengths = get(params, :common_lengths, [1.0, 1.5, 2.0, 2.4])
    common_length_prob = get(params, :common_length_prob, 0.4)
    waste_factor = get(params, :waste_factor, 0.1)
    solution_status = get(params, :solution_status, :feasible)
    
    # Save actual parameters used
    actual_params = Dict{Symbol, Any}(
        :n_piece_types => n_piece_types,
        :min_length => min_length,
        :max_length => max_length,
        :stock_length => stock_length,
        :demand_min => demand_min,
        :demand_max => demand_max,
        :max_patterns => max_patterns,
        :common_lengths => common_lengths,
        :common_length_prob => common_length_prob,
        :waste_factor => waste_factor,
        :solution_status => solution_status
    )
    
    # Determine actual feasibility requirement
    target_feasible = if solution_status == :feasible
        true
    elseif solution_status == :infeasible
        false
    else  # :all
        rand() < 0.5  # 50% chance of feasible
    end
    
    # Generate realistic piece lengths (all pieces should fit in stock)
    piece_lengths = Float64[]
    effective_max_length = min(max_length, stock_length * 0.95)  # All pieces fit in stock
    
    for i in 1:n_piece_types
        if rand() < common_length_prob && !isempty(common_lengths)
            # Pick a common length with slight variation
            base_length = rand(common_lengths)
            # Ensure common length fits in stock
            if base_length > effective_max_length
                base_length = rand(filter(x -> x <= effective_max_length, common_lengths))
                if isempty(filter(x -> x <= effective_max_length, common_lengths))
                    base_length = effective_max_length * 0.8
                end
            end
            # Add small variation to simulate real-world tolerances
            variation = rand(Normal(0, 0.02))
            length = clamp(base_length + variation, min_length, effective_max_length)
            push!(piece_lengths, round(length, digits=2))
        else
            # Generate length using Beta distribution (skewed toward smaller pieces)
            α, β = 2.0, 3.0
            normalized = rand(Beta(α, β))
            length = min_length + (effective_max_length - min_length) * normalized
            
            # Round to realistic precision based on stock length
            precision = stock_length > 10 ? 0.1 : 0.05
            length = round(length / precision) * precision
            push!(piece_lengths, length)
        end
    end
    
    # Remove any duplicates and ensure lengths are unique
    unique!(piece_lengths)
    
    # Generate initial demands using realistic distributions
    base_demands = Int[]
    
    for length in piece_lengths
        # Use LogNormal distribution for demands (realistic for manufacturing)
        if length in common_lengths
            # Higher demand for common lengths
            μ = log((demand_min + demand_max) / 1.3)
            σ = 0.5
        else
            # Lower demand for custom lengths
            μ = log((demand_min + demand_max) / 2.0)
            σ = 0.7
        end
        
        base_demand = rand(LogNormal(μ, σ))
        
        # Round to realistic batch sizes
        if base_demand < 50
            base_demand = round(base_demand / 5) * 5  # Round to nearest 5
        elseif base_demand < 200
            base_demand = round(base_demand / 10) * 10  # Round to nearest 10
        else
            base_demand = round(base_demand / 25) * 25  # Round to nearest 25
        end
        
        push!(base_demands, clamp(round(Int, base_demand), demand_min, demand_max))
    end
    
    # Adjust parameters based on desired feasibility (patterns generated within this function)
    demands, stock_limit, modified_waste_factor, patterns, restricted_patterns = 
        adjust_for_feasibility(base_demands, piece_lengths, stock_length, 
                              common_lengths, target_feasible, waste_factor, max_patterns)
    
    # Store generated data in params
    actual_params[:piece_lengths] = piece_lengths
    actual_params[:demands] = demands
    actual_params[:stock_limit] = stock_limit
    actual_params[:modified_waste_factor] = modified_waste_factor
    actual_params[:patterns] = patterns
    actual_params[:restricted_patterns] = restricted_patterns
    
    # Model
    model = Model()
    
    # Decision variables: how many times to use each pattern
    @variable(model, x[1:length(patterns)] >= 0)
    
    # Objective: Minimize number of standard sheets used
    @objective(model, Min, sum(x))
    
    # Meet demand for each piece size
    for i in 1:length(piece_lengths)
        @constraint(model, sum(patterns[j][i] * x[j] for j in 1:length(patterns)) >= demands[i])
    end
    
    # Add stock limit constraint if specified
    if stock_limit > 0
        @constraint(model, sum(x) <= stock_limit)
    end
    
    return model, actual_params
end

"""
Calculate maximum possible production for each piece type given constraints
"""
function calculate_maximum_production_bounds(piece_lengths, stock_length, patterns, stock_limit=0)
    n_pieces = length(piece_lengths)
    
    if stock_limit <= 0
        # Unlimited stock - theoretical maximum per piece is very high
        # But still bounded by pattern efficiency
        max_bounds = zeros(Int, n_pieces)
        for i in 1:n_pieces
            # Best case: most efficient pattern for this piece type used exclusively
            best_efficiency = 0
            for pattern in patterns
                if pattern[i] > 0
                    best_efficiency = max(best_efficiency, pattern[i])
                end
            end
            # Set a very high but finite bound
            max_bounds[i] = best_efficiency * 10000  # Effectively unlimited
        end
        return max_bounds
    else
        # Limited stock - calculate exact mathematical upper bounds
        max_bounds = zeros(Int, n_pieces)
        for i in 1:n_pieces
            # Maximum possible production of piece i if all stock was used optimally for piece i
            best_pattern_for_i = 0
            for pattern in patterns
                best_pattern_for_i = max(best_pattern_for_i, pattern[i])
            end
            max_bounds[i] = best_pattern_for_i * stock_limit
        end
        return max_bounds
    end
end

"""
Calculate realistic demand scaling based on business scenarios
"""
function calculate_demand_scaling_factors(piece_lengths, common_lengths, scenario::Symbol)
    n_pieces = length(piece_lengths)
    scaling_factors = ones(Float64, n_pieces)
    
    if scenario == :rush_order
        # Rush orders typically focus on specific pieces with high multipliers
        # Common pieces get higher rush demand
        for i in 1:n_pieces
            if piece_lengths[i] in common_lengths
                scaling_factors[i] = rand(Uniform(2.0, 4.0))  # High rush demand
            else
                scaling_factors[i] = rand(Uniform(0.8, 1.5))  # Normal or slightly higher
            end
        end
        
    elseif scenario == :seasonal_spike
        # Seasonal spikes affect all pieces but with variation
        base_spike = rand(Uniform(1.8, 2.5))
        for i in 1:n_pieces
            # Add some randomness around the base spike
            scaling_factors[i] = base_spike * rand(Uniform(0.8, 1.2))
        end
        
    elseif scenario == :backlog_clearing
        # Clearing backlog - high demand across all pieces
        for i in 1:n_pieces
            scaling_factors[i] = rand(Uniform(2.2, 3.5))
        end
        
    else  # :mixed
        # Mixed scenario with varied demand patterns
        for i in 1:n_pieces
            scaling_factors[i] = rand(Uniform(1.5, 2.8))
        end
    end
    
    return scaling_factors
end

"""
Adjust demands and constraints to achieve desired feasibility with mathematical guarantees
"""
function adjust_for_feasibility(base_demands, piece_lengths, stock_length, 
                               common_lengths, target_feasible, waste_factor, max_patterns)
    
    if target_feasible
        # FEASIBLE: Create realistic, solvable manufacturing scenarios
        
        # Generate full pattern set first to understand capacity
        patterns = generate_cutting_patterns(stock_length, piece_lengths, max_patterns, waste_factor)
        
        # Use conservative demand scaling to ensure feasibility
        demands = copy(base_demands)
        
        # Add some realistic demand variation (±20%) but keep conservative
        for i in 1:length(demands)
            variation = rand(Uniform(0.8, 1.2))
            demands[i] = max(1, round(Int, demands[i] * variation))
        end
        
        # No stock limit (represents good inventory management)
        stock_limit = 0
        modified_waste_factor = waste_factor
        restricted_patterns = false
        
        return demands, stock_limit, modified_waste_factor, patterns, restricted_patterns
        
    else
        # INFEASIBLE: Create mathematically guaranteed infeasible scenarios
        
        # Choose realistic business scenario that leads to infeasibility
        scenarios = [:rush_order, :seasonal_spike, :backlog_clearing, :mixed]
        scenario = rand(scenarios)
        
        infeasibility_method = rand(1:3)  # Three guaranteed methods
        
        if infeasibility_method == 1
            # Method 1: GUARANTEED infeasibility via single piece impossibility
            
            # Generate patterns first
            patterns = generate_cutting_patterns(stock_length, piece_lengths, max_patterns, waste_factor)
            
            # Pick one piece type to make impossible (prefer smaller pieces as more realistic for rush orders)
            target_piece_idx = argmin(piece_lengths)
            
            # Calculate absolute maximum production possible for this piece with reasonable stock
            reasonable_stock_limit = round(Int, sum(base_demands) * rand(Uniform(0.8, 1.2)))
            
            # Find best possible efficiency for target piece
            best_efficiency = 0
            for pattern in patterns
                if pattern[target_piece_idx] > 0
                    best_efficiency = max(best_efficiency, pattern[target_piece_idx])
                end
            end
            if best_efficiency == 0
                best_efficiency = floor(Int, stock_length / piece_lengths[target_piece_idx])
            end
            
            # Maximum possible production of target piece = best_efficiency * stock_limit
            max_possible_target = best_efficiency * reasonable_stock_limit
            
            # Set demand for target piece ABOVE this maximum (guaranteed infeasible)
            scaling_factors = calculate_demand_scaling_factors(piece_lengths, common_lengths, scenario)
            demands = [max(1, round(Int, d * f)) for (d, f) in zip(base_demands, scaling_factors)]
            
            # Override target piece demand to exceed maximum possible
            demands[target_piece_idx] = max_possible_target + round(Int, max_possible_target * rand(Uniform(0.3, 0.8)))
            
            stock_limit = reasonable_stock_limit
            modified_waste_factor = waste_factor
            restricted_patterns = false
            
        elseif infeasibility_method == 2
            # Method 2: GUARANTEED infeasibility via no-pattern scenario
            
            # Create a scenario where one piece type has NO feasible pattern
            target_piece_idx = rand(1:length(piece_lengths))
            
            # Create patterns that deliberately exclude the target piece
            patterns = []
            for (i, piece_length) in enumerate(piece_lengths)
                if i != target_piece_idx  # Skip target piece
                    pattern = zeros(Int, length(piece_lengths))
                    pattern[i] = max(1, floor(Int, stock_length / piece_length))
                    push!(patterns, pattern)
                end
            end
            
            # If we don't have enough patterns, add some mixed patterns (but still no target piece)
            while length(patterns) < max(5, length(piece_lengths) - 1)
                pattern = zeros(Int, length(piece_lengths))
                # Add pieces randomly except target piece
                available_indices = [i for i in 1:length(piece_lengths) if i != target_piece_idx]
                for idx in available_indices
                    if rand() < 0.3  # 30% chance to include each piece
                        max_fit = floor(Int, stock_length / piece_lengths[idx])
                        pattern[idx] = rand(1:max(1, max_fit))
                    end
                end
                if sum(pattern) > 0  # Only add non-empty patterns
                    push!(patterns, pattern)
                end
            end
            
            # Set reasonable demands including positive demand for impossible piece
            scaling_factors = calculate_demand_scaling_factors(piece_lengths, common_lengths, scenario)
            demands = [max(1, round(Int, d * f)) for (d, f) in zip(base_demands, scaling_factors)]
            
            # Ensure target piece has positive demand (makes problem infeasible since no pattern produces it)
            demands[target_piece_idx] = max(1, round(Int, base_demands[target_piece_idx] * scaling_factors[target_piece_idx]))
            
            stock_limit = 0  # Unlimited stock won't help if no pattern can produce the required piece
            modified_waste_factor = waste_factor
            restricted_patterns = true
            
        else
            # Method 3: GUARANTEED infeasibility via combined stock+demand contradiction
            
            # Generate normal patterns
            patterns = generate_cutting_patterns(stock_length, piece_lengths, max_patterns, waste_factor)
            
            # Calculate the minimum possible stock needed to satisfy base demands
            min_total_stock_needed = 0
            for i in 1:length(piece_lengths)
                best_efficiency = 0
                for pattern in patterns
                    if pattern[i] > 0
                        best_efficiency = max(best_efficiency, pattern[i])
                    end
                end
                if best_efficiency == 0
                    best_efficiency = floor(Int, stock_length / piece_lengths[i])
                end
                min_stock_for_piece_i = ceil(base_demands[i] / best_efficiency)
                min_total_stock_needed = max(min_total_stock_needed, min_stock_for_piece_i)
            end
            
            # Set stock limit below this minimum AND increase demands
            stock_limit = max(1, round(Int, min_total_stock_needed * rand(Uniform(0.4, 0.7))))
            
            # Also scale up demands to make the contradiction even stronger
            scaling_factors = calculate_demand_scaling_factors(piece_lengths, common_lengths, scenario)
            enhanced_factors = [f * rand(Uniform(1.2, 1.8)) for f in scaling_factors]  # Extra boost
            demands = [max(1, round(Int, d * f)) for (d, f) in zip(base_demands, enhanced_factors)]
            
            modified_waste_factor = waste_factor
            restricted_patterns = false
        end
        
        return demands, stock_limit, modified_waste_factor, patterns, restricted_patterns
    end
end

"""
Helper function to generate feasible cutting patterns
"""
function generate_cutting_patterns(standard_length, piece_lengths, max_patterns, waste_factor=0.1)
    patterns = Vector{Vector{Int}}()
    
    # Start with single-piece patterns
    for (i, piece_length) in enumerate(piece_lengths)
        pattern = zeros(Int, length(piece_lengths))
        pattern[i] = floor(Int, standard_length / piece_length)
        push!(patterns, pattern)
    end
    
    # Add more complex patterns up to max_patterns
    attempts = 0
    max_attempts = max_patterns * 10  # Prevent infinite loops
    
    while length(patterns) < max_patterns && attempts < max_attempts
        attempts += 1
        
        # Try to generate a random pattern
        new_pattern = zeros(Int, length(piece_lengths))
        remaining_length = standard_length
        indices = collect(1:length(piece_lengths))
        
        # Use exponential distribution to favor patterns with fewer piece types
        num_types_to_use = min(length(piece_lengths), 
                              max(1, round(Int, rand(Exponential(2.0)))))
        
        # Select random subset of piece types
        selected_indices = sample(indices, num_types_to_use, replace=false)
        
        # Keep adding pieces until we can't fit anymore or waste is acceptable
        while !isempty(selected_indices)
            # Weight selection by piece efficiency (less waste)
            weights = [standard_length / piece_lengths[i] for i in selected_indices]
            idx = sample(selected_indices, Weights(weights))
            
            if piece_lengths[idx] <= remaining_length
                new_pattern[idx] += 1
                remaining_length -= piece_lengths[idx]
                
                # Stop if waste is acceptable
                if remaining_length / standard_length <= waste_factor
                    break
                end
            else
                # Remove indices that won't fit
                filter!(i -> piece_lengths[i] <= remaining_length, selected_indices)
            end
        end
        
        # Only add if it's a new pattern and not trivial
        if sum(new_pattern) > 0 && !(new_pattern in patterns)
            push!(patterns, new_pattern)
        end
    end
    
    return patterns
end

"""
    calculate_cutting_stock_variable_count(params::Dict)

Calculate the number of variables (cutting patterns) in a cutting stock problem.

# Arguments
- `params`: Dictionary of problem parameters

# Returns
- Number of variables in the problem
"""
function calculate_cutting_stock_variable_count(params::Dict)
    return get(params, :max_patterns, 50)
end

"""
    sample_cutting_stock_parameters(target_variables::Int; seed::Int=0, solution_status::Symbol=:feasible)

Sample realistic parameters for a cutting stock problem targeting a specific number of variables.

# Arguments
- `target_variables`: Target number of variables (cutting patterns)
- `seed`: Random seed for reproducibility (default: 0)
- `solution_status`: Control feasibility (:feasible, :infeasible, :all) (default: :feasible)

# Returns
- Dictionary of sampled parameters
"""
function sample_cutting_stock_parameters(target_variables::Int; seed::Int=0, solution_status::Symbol=:feasible)
    Random.seed!(seed)
    
    params = Dict{Symbol, Any}()
    
    # Scale parameters based on target variable count
    if target_variables <= 250
        # Small shop: Few piece types, smaller stock, lower demands
        params[:n_piece_types] = rand(3:min(15, max(3, target_variables ÷ 10)))
        params[:max_patterns] = target_variables
        params[:stock_length] = rand(Uniform(3.0, 8.0))  # 3-8 meters
        params[:min_length] = 0.3
        params[:max_length] = params[:stock_length] * 0.7
        params[:demand_min] = rand(5:20)
        params[:demand_max] = rand(50:200)
        params[:common_lengths] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        params[:common_length_prob] = rand(Uniform(0.3, 0.6))
        params[:waste_factor] = rand(Uniform(0.05, 0.15))
        params[:solution_status] = solution_status
        
    elseif target_variables <= 1000
        # Manufacturing plant: More piece types, standard industrial stock, higher demands
        params[:n_piece_types] = rand(8:min(50, max(8, target_variables ÷ 20)))
        params[:max_patterns] = target_variables
        params[:stock_length] = rand(Uniform(6.0, 12.0))  # 6-12 meters (standard industrial)
        params[:min_length] = 0.5
        params[:max_length] = params[:stock_length] * 0.8
        params[:demand_min] = rand(20:100)
        params[:demand_max] = rand(200:1000)
        params[:common_lengths] = [1.0, 1.2, 1.5, 2.0, 2.4, 3.0, 4.0, 6.0]
        params[:common_length_prob] = rand(Uniform(0.4, 0.7))
        params[:waste_factor] = rand(Uniform(0.03, 0.10))
        params[:solution_status] = solution_status
        
    else
        # Large industrial facility: Many piece types, very large stock, very high demands
        params[:n_piece_types] = rand(20:min(200, max(20, target_variables ÷ 50)))
        params[:max_patterns] = target_variables
        params[:stock_length] = rand(Uniform(8.0, 20.0))  # 8-20 meters (industrial/structural)
        params[:min_length] = 0.8
        params[:max_length] = params[:stock_length] * 0.9
        params[:demand_min] = rand(100:500)
        params[:demand_max] = rand(1000:10000)
        params[:common_lengths] = [1.0, 1.2, 1.5, 2.0, 2.4, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        params[:common_length_prob] = rand(Uniform(0.5, 0.8))
        params[:waste_factor] = rand(Uniform(0.02, 0.08))
        params[:solution_status] = solution_status
    end
    
    return params
end

"""
    sample_cutting_stock_parameters(size::Symbol=:medium; seed::Int=0, solution_status::Symbol=:feasible)

Sample realistic parameters for a cutting stock problem (legacy size-based interface).

# Arguments
- `size`: Symbol specifying the problem size (:small, :medium, :large)
- `seed`: Random seed for reproducibility (default: 0)
- `solution_status`: Control feasibility (:feasible, :infeasible, :all) (default: :feasible)

# Returns
- Dictionary of sampled parameters
"""
function sample_cutting_stock_parameters(size::Symbol=:medium; seed::Int=0, solution_status::Symbol=:feasible)
    Random.seed!(seed)
    
    # Map legacy sizes to target variable counts
    target_map = Dict(
        :small => rand(50:250),
        :medium => rand(250:1000),
        :large => rand(1000:10000)
    )
    
    if !haskey(target_map, size)
        error("Unknown size: $size. Must be :small, :medium, or :large")
    end
    
    return sample_cutting_stock_parameters(target_map[size]; seed=seed, solution_status=solution_status)
end

# Register the problem type
register_problem(
    :cutting_stock,
    generate_cutting_stock_problem,
    sample_cutting_stock_parameters,
    "Cutting stock optimization problem that minimizes waste by determining optimal cutting patterns for stock material to satisfy demand for pieces of various lengths. Scales from small shops (50-250 patterns) to large industrial facilities (1000-10000 patterns) with realistic material sizes, demand distributions, and waste constraints."
)