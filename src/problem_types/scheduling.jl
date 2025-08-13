using JuMP
using Random
using StatsBase
using Distributions

"""
    generate_scheduling_problem(params::Dict=Dict(); seed::Int=0)

Generate a workforce scheduling optimization problem instance.

# Arguments
- `params`: Dictionary of problem parameters
  - `:n_workers`: Number of workers (default: 8)
  - `:n_shifts`: Number of shifts (default: 7)
  - `:n_days`: Number of days in planning horizon (default: 5)
  - `:min_staffing`: Minimum staffing requirement per shift (default: 2)
  - `:max_staffing`: Maximum staffing requirement per shift (default: 5)
  - `:availability_density`: Probability of worker availability (default: 0.8)
  - `:min_worker_shifts`: Minimum shifts per worker (default: 3)
  - `:max_worker_shifts`: Maximum shifts per worker (default: 5)
  - `:max_consecutive_shifts`: Maximum consecutive shifts (default: 3)
  - `:min_cost`: Minimum cost per worker per shift (default: 50)
  - `:max_cost`: Maximum cost per worker per shift (default: 100)
  - `:skill_based`: Whether to include worker skills (default: false)
  - `:n_skills`: Number of skill types (if skill_based is true) (default: 3)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- `model`: The JuMP model
- `params`: Dictionary of all parameters used (including defaults)
"""
function generate_scheduling_problem(params::Dict=Dict(); seed::Int=0)
    # Set random seed
    Random.seed!(seed)
    
    # Extract parameters with defaults
    n_workers = get(params, :n_workers, 8)
    n_shifts = get(params, :n_shifts, 7)
    n_days = get(params, :n_days, 5)
    min_staffing = get(params, :min_staffing, 2)
    max_staffing = get(params, :max_staffing, 5)
    availability_density = get(params, :availability_density, 0.8)
    min_worker_shifts = get(params, :min_worker_shifts, 3)
    max_worker_shifts = get(params, :max_worker_shifts, 5)
    max_consecutive_shifts = get(params, :max_consecutive_shifts, 3)
    min_cost = get(params, :min_cost, 50)
    max_cost = get(params, :max_cost, 100)
    skill_based = get(params, :skill_based, false)
    n_skills = get(params, :n_skills, 3)
    
    # Save actual parameters used
    actual_params = Dict{Symbol, Any}(
        :n_workers => n_workers,
        :n_shifts => n_shifts,
        :n_days => n_days,
        :min_staffing => min_staffing,
        :max_staffing => max_staffing,
        :availability_density => availability_density,
        :min_worker_shifts => min_worker_shifts,
        :max_worker_shifts => max_worker_shifts,
        :max_consecutive_shifts => max_consecutive_shifts,
        :min_cost => min_cost,
        :max_cost => max_cost,
        :skill_based => skill_based,
        :n_skills => n_skills
    )
    
    # Calculate total number of shifts (across all days)
    total_shifts = n_shifts * n_days
    
    # Generate staffing requirements per shift using Poisson distribution
    # Higher requirements during peak shifts, lower during off-peak
    staffing_req = Vector{Int}(undef, total_shifts)
    for s in 1:total_shifts
        # Create daily patterns with peak and off-peak periods
        day_idx = div(s-1, n_shifts) + 1
        shift_in_day = ((s-1) % n_shifts) + 1
        
        # Peak shifts (middle of day) need more staff
        peak_factor = 1.0 + 0.5 * sin(π * shift_in_day / n_shifts)
        
        # Weekend effect (assuming 7-day cycles)
        weekend_factor = (day_idx % 7 in [0, 6]) ? 0.8 : 1.0
        
        # Sample from Poisson distribution
        mean_staffing = (min_staffing + max_staffing) / 2 * peak_factor * weekend_factor
        staffing_req[s] = max(min_staffing, min(max_staffing, 
                             round(Int, rand(Poisson(mean_staffing)))))
    end
    
    # Generate worker availability using realistic patterns
    availability = zeros(Int, n_workers, total_shifts)
    
    # Base availability patterns for different worker types
    for w in 1:n_workers
        # Some workers are full-time (higher availability), others part-time
        worker_type = rand() < 0.6 ? :full_time : :part_time
        
        for s in 1:total_shifts
            day_idx = div(s-1, n_shifts) + 1
            shift_in_day = ((s-1) % n_shifts) + 1
            
            # Worker preferences based on shift time
            if worker_type == :full_time
                # Full-time workers prefer regular schedules
                base_prob = availability_density
                
                # Less likely to work very early or very late shifts
                if shift_in_day <= 2 || shift_in_day >= n_shifts - 1
                    base_prob *= 0.7
                end
                
                # Weekend availability might be lower
                if day_idx % 7 in [0, 6]
                    base_prob *= 0.8
                end
            else
                # Part-time workers have more variable availability
                base_prob = availability_density * 0.8
                
                # Prefer specific shifts (e.g., evening shifts for students)
                if shift_in_day > n_shifts / 2
                    base_prob *= 1.2
                end
                
                # Weekend availability might be higher for part-time
                if day_idx % 7 in [0, 6]
                    base_prob *= 1.1
                end
            end
            
            availability[w, s] = rand() < min(1.0, base_prob) ? 1 : 0
        end
    end
    
    # Generate worker skills if skill-based scheduling is enabled
    worker_skills = nothing
    shift_skill_req = nothing
    
    if skill_based
        # Assign skills to workers (each worker has 1-3 skills)
        worker_skills = zeros(Int, n_workers, n_skills)
        for w in 1:n_workers
            num_skills = rand(1:min(3, n_skills))
            skill_indices = sample(1:n_skills, num_skills, replace=false)
            for s in skill_indices
                worker_skills[w, s] = 1
            end
        end
        
        # Assign skill requirements to shifts
        shift_skill_req = zeros(Int, total_shifts, n_skills)
        for s in 1:total_shifts
            required_skill = rand(1:n_skills)
            shift_skill_req[s, required_skill] = 1
        end
        
        # Update availability based on skills
        for w in 1:n_workers, s in 1:total_shifts
            if availability[w, s] == 1
                required_skills = findall(sk -> sk == 1, shift_skill_req[s, :])
                worker_has_skill = any(worker_skills[w, required_skills] .== 1)
                if !worker_has_skill
                    availability[w, s] = 0
                end
            end
        end
    end
    
    # Generate costs per worker per shift using realistic distributions
    costs = zeros(n_workers, total_shifts)
    
    # Create different worker cost tiers (junior, regular, senior)
    worker_tiers = rand([:junior, :regular, :senior], n_workers)
    
    for w in 1:n_workers
        # Base cost depends on worker tier
        tier = worker_tiers[w]
        if tier == :junior
            base_cost = min_cost + rand(Normal(0, (max_cost - min_cost) * 0.1))
        elseif tier == :regular
            base_cost = (min_cost + max_cost) / 2 + rand(Normal(0, (max_cost - min_cost) * 0.15))
        else  # senior
            base_cost = max_cost + rand(Normal(0, (max_cost - min_cost) * 0.1))
        end
        
        # Skill premium if skill-based scheduling
        if skill_based
            num_skills = sum(worker_skills[w, :])
            skill_premium = num_skills / n_skills * (max_cost - min_cost) * 0.3
            base_cost += skill_premium
        end
        
        # Clamp base cost to reasonable bounds
        base_cost = max(min_cost * 0.8, min(max_cost * 1.2, base_cost))
        
        # Generate shift-specific costs
        for s in 1:total_shifts
            day_idx = div(s-1, n_shifts) + 1
            shift_in_day = ((s-1) % n_shifts) + 1
            
            # Shift premiums for difficult shifts
            shift_premium = 1.0
            
            # Night shift premium (early morning or late evening)
            if shift_in_day <= 2 || shift_in_day >= n_shifts - 1
                shift_premium *= 1.15
            end
            
            # Weekend premium
            if day_idx % 7 in [0, 6]
                shift_premium *= 1.1
            end
            
            # Holiday premium (simplified: every 30th day)
            if day_idx % 30 == 0
                shift_premium *= 1.25
            end
            
            # Add small random variation using log-normal distribution
            random_factor = rand(LogNormal(0, 0.1))
            
            costs[w, s] = base_cost * shift_premium * random_factor
        end
    end
    
    # Store generated data in params
    actual_params[:staffing_req] = staffing_req
    actual_params[:availability] = availability
    actual_params[:costs] = costs
    if skill_based
        actual_params[:worker_skills] = worker_skills
        actual_params[:shift_skill_req] = shift_skill_req
    end
    
    # Create model
    model = Model()
    
    # Decision variables: worker assignment to shifts
    @variable(model, x[1:n_workers, 1:total_shifts], Bin)
    
    # Objective: minimize total cost
    @objective(model, Min, sum(costs[w, s] * x[w, s] for w in 1:n_workers, s in 1:total_shifts))
    
    # Constraints
    
    # Staffing requirements
    for s in 1:total_shifts
        @constraint(model, sum(x[w, s] for w in 1:n_workers) >= staffing_req[s])
    end
    
    # Availability constraints
    for w in 1:n_workers, s in 1:total_shifts
        if availability[w, s] == 0
            @constraint(model, x[w, s] == 0)
        end
    end
    
    # Minimum and maximum shifts per worker
    for w in 1:n_workers
        @constraint(model, sum(x[w, s] for s in 1:total_shifts) >= min_worker_shifts)
        @constraint(model, sum(x[w, s] for s in 1:total_shifts) <= max_worker_shifts)
    end
    
    # Maximum consecutive shifts
    for w in 1:n_workers, day in 1:n_days
        if day <= n_days - max_consecutive_shifts + 1
            shifts_in_window = vec([day*n_shifts-n_shifts+i for i in 1:n_shifts*max_consecutive_shifts])
            shifts_in_window = shifts_in_window[shifts_in_window .<= total_shifts]
            if !isempty(shifts_in_window)
                @constraint(model, sum(x[w, s] for s in shifts_in_window) <= max_consecutive_shifts)
            end
        end
    end
    
    return model, actual_params
end

"""
    sample_scheduling_parameters(size::Symbol=:medium; seed::Int=0)

Sample realistic parameters for a workforce scheduling problem.

# Arguments
- `size`: Symbol specifying the problem size (:small, :medium, :large)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_scheduling_parameters(size::Symbol=:medium; seed::Int=0)
    Random.seed!(seed)
    
    params = Dict{Symbol, Any}()
    
    # Set size-dependent parameters with realistic scaling
    # Target variable ranges: small=50-250, medium=250-1000, large=1000-10000
    
    # Variables = n_workers * n_shifts * n_days
    # Use discrete uniform distributions to hit target ranges more precisely
    target_variables = if size == :small
        rand(50:250)  # Small: department stores, small restaurants, clinics
    elseif size == :medium
        rand(250:1000)  # Medium: hospitals, call centers, retail chains
    else  # :large
        rand(1000:10000)  # Large: airlines, large hospitals, manufacturing
    end
    
    if size == :small
        # Small scheduling problems: department stores, small restaurants, clinics
        # Target 50-250 variables
        params[:n_workers] = round(Int, rand(Uniform(4, 12)))
        params[:n_shifts] = round(Int, rand(Uniform(2, 4)))
        params[:n_days] = round(Int, rand(Uniform(3, 7)))
        
        # Adjust to hit target variable count
        current_vars = params[:n_workers] * params[:n_shifts] * params[:n_days]
        if current_vars < target_variables * 0.8
            # Need more variables - increase workers first (most realistic)
            params[:n_workers] = min(20, round(Int, params[:n_workers] * sqrt(target_variables / current_vars)))
            current_vars = params[:n_workers] * params[:n_shifts] * params[:n_days]
            if current_vars < target_variables * 0.8
                params[:n_days] = min(10, round(Int, params[:n_days] * (target_variables / current_vars)))
            end
        elseif current_vars > target_variables * 1.2
            # Need fewer variables - reduce workers first
            params[:n_workers] = max(3, round(Int, params[:n_workers] * sqrt(target_variables / current_vars)))
        end
        
        # Smaller organizations have tighter staffing requirements
        params[:min_staffing] = round(Int, rand(Uniform(1, 2)))
        params[:max_staffing] = params[:min_staffing] + round(Int, rand(Uniform(1, 3)))
        
        # Higher availability density in smaller organizations
        params[:availability_density] = rand(Beta(8, 2))  # skewed toward higher values
        
        # Flexible shift requirements for small organizations
        params[:min_worker_shifts] = round(Int, rand(Uniform(2, 4)))
        params[:max_worker_shifts] = params[:min_worker_shifts] + round(Int, rand(Uniform(1, 3)))
        params[:max_consecutive_shifts] = round(Int, rand(Uniform(2, 3)))
        
        # Lower cost ranges for small businesses
        params[:min_cost] = rand(Uniform(15, 25))
        params[:max_cost] = params[:min_cost] + rand(Uniform(20, 40))
        
    elseif size == :medium
        # Medium scheduling problems: hospitals, call centers, retail chains
        # Target 250-1000 variables
        params[:n_workers] = round(Int, rand(Uniform(8, 25)))
        params[:n_shifts] = round(Int, rand(Uniform(3, 6)))
        params[:n_days] = round(Int, rand(Uniform(5, 14)))
        
        # Adjust to hit target variable count
        current_vars = params[:n_workers] * params[:n_shifts] * params[:n_days]
        if current_vars < target_variables * 0.8
            # Need more variables - increase workers first (most realistic)
            params[:n_workers] = min(50, round(Int, params[:n_workers] * sqrt(target_variables / current_vars)))
            current_vars = params[:n_workers] * params[:n_shifts] * params[:n_days]
            if current_vars < target_variables * 0.8
                params[:n_days] = min(21, round(Int, params[:n_days] * (target_variables / current_vars)))
            end
        elseif current_vars > target_variables * 1.2
            # Need fewer variables - reduce workers first
            params[:n_workers] = max(6, round(Int, params[:n_workers] * sqrt(target_variables / current_vars)))
        end
        
        # Medium organizations have more structured requirements
        params[:min_staffing] = round(Int, rand(Uniform(2, 4)))
        params[:max_staffing] = params[:min_staffing] + round(Int, rand(Uniform(2, 5)))
        
        # Moderate availability density
        params[:availability_density] = rand(Beta(6, 3))
        
        # More structured shift requirements
        params[:min_worker_shifts] = round(Int, rand(Uniform(3, 5)))
        params[:max_worker_shifts] = params[:min_worker_shifts] + round(Int, rand(Uniform(2, 4)))
        params[:max_consecutive_shifts] = round(Int, rand(Uniform(2, 4)))
        
        # Standard cost ranges
        params[:min_cost] = rand(Uniform(20, 35))
        params[:max_cost] = params[:min_cost] + rand(Uniform(25, 60))
        
    elseif size == :large
        # Large scheduling problems: airlines, large hospitals, manufacturing
        # Target 1000-10000 variables
        params[:n_workers] = round(Int, rand(Uniform(25, 80)))
        params[:n_shifts] = round(Int, rand(Uniform(4, 8)))
        params[:n_days] = round(Int, rand(Uniform(7, 30)))
        
        # Adjust to hit target variable count
        current_vars = params[:n_workers] * params[:n_shifts] * params[:n_days]
        if current_vars < target_variables * 0.8
            # Need more variables - increase workers first (most realistic)
            params[:n_workers] = min(200, round(Int, params[:n_workers] * sqrt(target_variables / current_vars)))
            current_vars = params[:n_workers] * params[:n_shifts] * params[:n_days]
            if current_vars < target_variables * 0.8
                params[:n_days] = min(60, round(Int, params[:n_days] * (target_variables / current_vars)))
            end
        elseif current_vars > target_variables * 1.2
            # Need fewer variables - reduce workers first
            params[:n_workers] = max(20, round(Int, params[:n_workers] * sqrt(target_variables / current_vars)))
        end
        
        # Large organizations have complex staffing requirements
        params[:min_staffing] = round(Int, rand(Uniform(3, 6)))
        params[:max_staffing] = params[:min_staffing] + round(Int, rand(Uniform(3, 8)))
        
        # Lower availability density due to complexity
        params[:availability_density] = rand(Beta(4, 3))
        
        # More complex shift requirements
        params[:min_worker_shifts] = round(Int, rand(Uniform(4, 6)))
        params[:max_worker_shifts] = params[:min_worker_shifts] + round(Int, rand(Uniform(2, 5)))
        params[:max_consecutive_shifts] = round(Int, rand(Uniform(3, 5)))
        
        # Higher cost ranges for specialized workers
        params[:min_cost] = rand(Uniform(25, 50))
        params[:max_cost] = params[:min_cost] + rand(Uniform(30, 100))
        
    else
        error("Unknown size: $size. Must be :small, :medium, or :large")
    end
    
    # Skill-based scheduling more common in larger organizations
    skill_prob = size == :small ? 0.2 : (size == :medium ? 0.4 : 0.6)
    params[:skill_based] = rand() < skill_prob
    
    if params[:skill_based]
        # Number of skills scales with organization size
        if size == :small
            params[:n_skills] = round(Int, rand(Uniform(2, 3)))
        elseif size == :medium
            params[:n_skills] = round(Int, rand(Uniform(3, 5)))
        else  # large
            params[:n_skills] = round(Int, rand(Uniform(4, 8)))
        end
    end
    
    return params
end

"""
    sample_scheduling_parameters(target_variables::Int; seed::Int=0)

Sample realistic parameters for a workforce scheduling problem targeting approximately the specified number of variables.

# Arguments
- `target_variables`: Target number of variables in the LP formulation (approximately within ±10%)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_scheduling_parameters(target_variables::Int; seed::Int=0)
    Random.seed!(seed)
    
    params = Dict{Symbol, Any}()
    
    # Determine appropriate size category based on target variables
    if target_variables <= 250
        size_category = :small
    elseif target_variables <= 1000
        size_category = :medium
    else
        size_category = :large
    end
    
    # Start with realistic defaults based on size
    if size_category == :small
        # Small scheduling problems: department stores, small restaurants, clinics
        params[:n_workers] = round(Int, rand(Uniform(4, 12)))
        params[:n_shifts] = round(Int, rand(Uniform(2, 4)))
        params[:n_days] = round(Int, rand(Uniform(3, 7)))
        params[:min_staffing] = round(Int, rand(Uniform(1, 2)))
        params[:max_staffing] = params[:min_staffing] + round(Int, rand(Uniform(1, 3)))
        params[:availability_density] = rand(Beta(8, 2))
        params[:min_worker_shifts] = round(Int, rand(Uniform(2, 4)))
        params[:max_worker_shifts] = params[:min_worker_shifts] + round(Int, rand(Uniform(1, 3)))
        params[:max_consecutive_shifts] = round(Int, rand(Uniform(2, 3)))
        params[:min_cost] = rand(Uniform(15, 25))
        params[:max_cost] = params[:min_cost] + rand(Uniform(20, 40))
        params[:skill_based] = rand() < 0.2
    elseif size_category == :medium
        # Medium scheduling problems: hospitals, call centers, retail chains
        params[:n_workers] = round(Int, rand(Uniform(8, 25)))
        params[:n_shifts] = round(Int, rand(Uniform(3, 6)))
        params[:n_days] = round(Int, rand(Uniform(5, 14)))
        params[:min_staffing] = round(Int, rand(Uniform(2, 4)))
        params[:max_staffing] = params[:min_staffing] + round(Int, rand(Uniform(2, 5)))
        params[:availability_density] = rand(Beta(6, 3))
        params[:min_worker_shifts] = round(Int, rand(Uniform(3, 5)))
        params[:max_worker_shifts] = params[:min_worker_shifts] + round(Int, rand(Uniform(2, 4)))
        params[:max_consecutive_shifts] = round(Int, rand(Uniform(2, 4)))
        params[:min_cost] = rand(Uniform(20, 35))
        params[:max_cost] = params[:min_cost] + rand(Uniform(25, 60))
        params[:skill_based] = rand() < 0.4
    else  # large
        # Large scheduling problems: airlines, large hospitals, manufacturing
        params[:n_workers] = round(Int, rand(Uniform(25, 80)))
        params[:n_shifts] = round(Int, rand(Uniform(4, 8)))
        params[:n_days] = round(Int, rand(Uniform(7, 30)))
        params[:min_staffing] = round(Int, rand(Uniform(3, 6)))
        params[:max_staffing] = params[:min_staffing] + round(Int, rand(Uniform(3, 8)))
        params[:availability_density] = rand(Beta(4, 3))
        params[:min_worker_shifts] = round(Int, rand(Uniform(4, 6)))
        params[:max_worker_shifts] = params[:min_worker_shifts] + round(Int, rand(Uniform(2, 5)))
        params[:max_consecutive_shifts] = round(Int, rand(Uniform(3, 5)))
        params[:min_cost] = rand(Uniform(25, 50))
        params[:max_cost] = params[:min_cost] + rand(Uniform(30, 100))
        params[:skill_based] = rand() < 0.6
    end
    
    # Set skills if skill-based
    if params[:skill_based]
        if size_category == :small
            params[:n_skills] = round(Int, rand(Uniform(2, 3)))
        elseif size_category == :medium
            params[:n_skills] = round(Int, rand(Uniform(3, 5)))
        else  # large
            params[:n_skills] = round(Int, rand(Uniform(4, 8)))
        end
    end
    
    # Iteratively adjust parameters to reach target with realistic constraints
    for iteration in 1:15
        current_vars = calculate_scheduling_variable_count(params)
        
        if abs(current_vars - target_variables) / target_variables < 0.1
            break  # Within 10% tolerance
        end
        
        # Adjust parameters based on current vs target
        ratio = target_variables / current_vars
        
        if ratio > 1.2  # Need significantly more variables
            # Prioritize scaling the most impactful parameters
            if rand() < 0.4  # 40% chance to increase workers
                max_workers = size_category == :small ? 20 : (size_category == :medium ? 50 : 200)
                params[:n_workers] = min(max_workers, round(Int, params[:n_workers] * rand(Uniform(1.1, 1.3))))
            end
            if rand() < 0.3  # 30% chance to increase shifts
                max_shifts = size_category == :small ? 6 : (size_category == :medium ? 8 : 12)
                params[:n_shifts] = min(max_shifts, round(Int, params[:n_shifts] * rand(Uniform(1.1, 1.2))))
            end
            if rand() < 0.3  # 30% chance to increase days
                max_days = size_category == :small ? 10 : (size_category == :medium ? 21 : 60)
                params[:n_days] = min(max_days, round(Int, params[:n_days] * rand(Uniform(1.1, 1.2))))
            end
        elseif ratio < 0.8  # Need significantly fewer variables
            # Reduce parameters proportionally
            if rand() < 0.4  # 40% chance to reduce workers
                min_workers = size_category == :small ? 3 : (size_category == :medium ? 6 : 20)
                params[:n_workers] = max(min_workers, round(Int, params[:n_workers] * rand(Uniform(0.7, 0.9))))
            end
            if rand() < 0.3  # 30% chance to reduce shifts
                params[:n_shifts] = max(2, round(Int, params[:n_shifts] * rand(Uniform(0.8, 0.9))))
            end
            if rand() < 0.3  # 30% chance to reduce days
                params[:n_days] = max(2, round(Int, params[:n_days] * rand(Uniform(0.8, 0.9))))
            end
        else  # Fine-tune
            # Make small adjustments with realistic bounds
            if rand() < 0.5
                min_workers = size_category == :small ? 3 : (size_category == :medium ? 6 : 20)
                max_workers = size_category == :small ? 20 : (size_category == :medium ? 50 : 200)
                params[:n_workers] = max(min_workers, min(max_workers, round(Int, params[:n_workers] * rand(Uniform(0.95, 1.05)))))
            end
            if rand() < 0.3
                max_shifts = size_category == :small ? 6 : (size_category == :medium ? 8 : 12)
                params[:n_shifts] = max(2, min(max_shifts, round(Int, params[:n_shifts] * rand(Uniform(0.95, 1.05)))))
            end
            if rand() < 0.2
                max_days = size_category == :small ? 10 : (size_category == :medium ? 21 : 60)
                params[:n_days] = max(2, min(max_days, round(Int, params[:n_days] * rand(Uniform(0.95, 1.05)))))
            end
        end
    end
    
    return params
end

function calculate_scheduling_variable_count(params::Dict)
    # Extract parameters with defaults
    n_workers = get(params, :n_workers, 8)
    n_shifts = get(params, :n_shifts, 7)
    n_days = get(params, :n_days, 5)
    
    # Calculate total number of shifts (across all days)
    total_shifts = n_shifts * n_days
    
    # Variables: x[1:n_workers, 1:total_shifts] - binary variables for worker-shift assignment
    return n_workers * total_shifts
end

# Register the problem type
register_problem(
    :scheduling,
    generate_scheduling_problem,
    sample_scheduling_parameters,
    "Workforce scheduling problem that minimizes staffing costs while meeting shift requirements and respecting worker constraints"
)