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
  - `:solution_status`: Desired feasibility of generated instance. One of `:feasible`, `:infeasible`, `:all` (default: `:feasible`).
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
    solution_status = get(params, :solution_status, :feasible)
    if !(solution_status in (:feasible, :infeasible, :all))
        error("Invalid :solution_status=$(solution_status). Use :feasible, :infeasible, or :all")
    end
    
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
        :n_skills => n_skills,
        :solution_status => solution_status
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
    
    # Map day -> shifts index set (for day-level constraints and reasoning)
    shifts_for_day = [collect(((d-1)*n_shifts + 1):min(d*n_shifts, total_shifts)) for d in 1:n_days]
    
    # Helper: number of days each worker is available for at least one shift
    days_available_per_worker = zeros(Int, n_workers)
    for w in 1:n_workers
        count_days = 0
        for d in 1:n_days
            if any(availability[w, s] == 1 for s in shifts_for_day[d])
                count_days += 1
            end
        end
        days_available_per_worker[w] = count_days
    end
    
    # Generate costs per worker per shift using realistic distributions (needed for feasibility construction)
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
    
    # If a feasible instance is requested, keep exogenous demand but cap with capacity-aware, randomized slack
    if solution_status == :feasible
        # Helper: consecutive-days capacity per worker given availability by day
        function max_assignable_days_for_runs(avail_bool::AbstractVector{Bool}, K::Int)
            if K <= 0
                return 0
            end
            total = 0
            i = 1
            L = length(avail_bool)
            while i <= L
                if !avail_bool[i]
                    i += 1
                    continue
                end
                start_i = i
                while i <= L && avail_bool[i]
                    i += 1
                end
                run_len = i - start_i
                if K >= run_len
                    total += run_len
                else
                    q = fld(run_len, K + 1)
                    r = run_len % (K + 1)
                    total += q * K + min(K, r)
                end
            end
            return total
        end

        # Day availability per worker
        avail_day = falses(n_workers, n_days)
        for w in 1:n_workers, d in 1:n_days
            sd = shifts_for_day[d]
            avail_day[w, d] = any(availability[w, s] == 1 for s in sd)
        end

        # Per-worker capacity considering daily and consecutive-day limits
        worker_cap = zeros(Int, n_workers)
        for w in 1:n_workers
            cap_runs = max_assignable_days_for_runs(vec(avail_day[w, :]), max_consecutive_shifts)
            worker_cap[w] = min(max_worker_shifts, cap_runs)
        end

        # Adjust global min to guarantee attainability for all workers
        feasible_global_min = minimum(worker_cap)
        min_worker_shifts = min(min_worker_shifts, feasible_global_min)
        min_worker_shifts = max(0, min_worker_shifts)
        max_worker_shifts = max(max_worker_shifts, min_worker_shifts)
        max_worker_shifts = min(max_worker_shifts, n_days)
        # Randomized slack factors
        rng = Random.default_rng()
        per_shift_alpha = [clamp(rand(rng, Beta(8, 2)), 0.6, 1.0) for _ in 1:total_shifts]
        per_day_reserve = [clamp(rand(rng, Beta(2, 12)), 0.05, 0.35) for _ in 1:n_days]
        global_reserve = clamp(rand(rng, Beta(2, 10)), 0.03, 0.25)

        # Per-shift capping
        avail_count_per_shift = [sum(availability[:, s]) for s in 1:total_shifts]
        for s in 1:total_shifts
            cap_s = floor(Int, per_shift_alpha[s] * avail_count_per_shift[s])
            staffing_req[s] = min(staffing_req[s], cap_s)
            staffing_req[s] = max(0, staffing_req[s])
        end

        # Per-day capping with reserve
        day_avail_workers = [sum(avail_day[:, d]) for d in 1:n_days]
        function scale_down_counts!(idxs::Vector{Int}, counts::Vector{Int}, target_sum::Int)
            current_sum = sum(counts[idxs])
            if current_sum <= target_sum
                return
            end
            if target_sum <= 0
                for i in idxs
                    counts[i] = 0
                end
                return
            end
            # Proportional scaling with randomized rounding
            scaled = Dict{Int, Float64}()
            for i in idxs
                scaled[i] = counts[i] * target_sum / current_sum
            end
            base = Dict{Int, Int}()
            rema = Dict{Int, Float64}()
            for i in idxs
                base[i] = floor(Int, scaled[i])
                rema[i] = scaled[i] - base[i]
            end
            need = target_sum - sum(values(base))
            order = sort(collect(idxs), by = i -> -rema[i])
            for k in 1:need
                base[order[k]] += 1
            end
            for i in idxs
                counts[i] = base[i]
            end
        end

        for d in 1:n_days
            day_shifts = shifts_for_day[d]
            cap_d = floor(Int, (1 - per_day_reserve[d]) * day_avail_workers[d])
            if cap_d < 0
                cap_d = 0
            end
            scale_down_counts!(day_shifts, staffing_req, cap_d)
        end

        # Global capping with reserve
        total_cap = sum(worker_cap)
        global_target = floor(Int, (1 - global_reserve) * total_cap)
        if global_target < 0
            global_target = 0
        end
        if sum(staffing_req) > global_target
            idxs_all = collect(1:total_shifts)
            scale_down_counts!(idxs_all, staffing_req, global_target)
        end

        # Per-day feasibility check using randomized matching that respects daily and consecutive-day caps
        assigned = zeros(Int, n_workers, total_shifts)
        worked_day = falses(n_workers, n_days)
        assigned_count_worker = zeros(Int, n_workers)
        # Helper: check window constraint before day assignment
        function can_add_day!(work_vec::AbstractVector{Bool}, d::Int, K::Int)
            if work_vec[d]
                return false
            end
            if K <= 0
                return false
            end
            nD = length(work_vec)
            start_min = max(1, d - K)
            start_max = min(nD - K, d)
            for start in start_min:start_max
                cnt = 0
                for t in start:(start + K)
                    cnt += (t == d ? 1 : (work_vec[t] ? 1 : 0))
                end
                if cnt > K
                    return false
                end
            end
            return true
        end
        # First, try to cover all per-shift demands per day
        for d in 1:n_days
            day_shifts = shifts_for_day[d]
            # Create a pool of workers available this day
            workers_d = [w for w in 1:n_workers if any(availability[w, s] == 1 for s in day_shifts)]
            shuffle!(rng, workers_d)
            demand_rem = Dict(s => staffing_req[s] for s in day_shifts)
            # Randomized passes
            progress = true
            while any(demand_rem[s] > 0 for s in day_shifts) && progress
                progress = false
                for w in workers_d
                    if worked_day[w, d]
                        continue
                    end
                    if assigned_count_worker[w] >= max_worker_shifts
                        continue
                    end
                    if !can_add_day!(vec(worked_day[w, :]), d, max_consecutive_shifts)
                        continue
                    end
                    # eligible shifts for this worker with remaining demand
                    eligible = [s for s in day_shifts if availability[w, s] == 1 && demand_rem[s] > 0]
                    if !isempty(eligible)
                        s_pick = rand(rng, eligible)
                        assigned[w, s_pick] = 1
                        worked_day[w, d] = true
                        assigned_count_worker[w] += 1
                        demand_rem[s_pick] -= 1
                        progress = true
                    end
                    if !any(demand_rem[s] > 0 for s in day_shifts)
                        break
                    end
                end
            end
            # If unmet demand remains, reduce those shift requirements locally
            for s in day_shifts
                if demand_rem[s] > 0
                    staffing_req[s] -= demand_rem[s]
                end
            end
        end
        # Align per-shift requirements exactly to covered amounts to certify feasibility
        staffing_req = vec(sum(assigned, dims=1))

        # Bring each worker up to minimum by using day capacity slack; free minimal demand if needed
        assigned_count_worker = [sum(assigned[w, :]) for w in 1:n_workers]
        day_used = [sum(WorkedDay -> WorkedDay ? 1 : 0, worked_day[:, d]) for d in 1:n_days]

        for w in 1:n_workers
            need = max(0, min_worker_shifts - assigned_count_worker[w])
            while need > 0
                # Prefer days with slack
                candidate_days = [d for d in 1:n_days if avail_day[w, d] && !worked_day[w, d] && can_add_day!(vec(worked_day[w, :]), d, max_consecutive_shifts)]
                # Sort by available slack descending
                candidate_days = sort(candidate_days, by = d -> (sum(avail_day[:, d]) - day_used[d]), rev = true)
                placed = false
                for d in candidate_days
                    cap_d = sum(avail_day[:, d])
                    if day_used[d] < cap_d
                        # Assign this worker to any shift this day they are available (overstaffing allowed)
                        day_shifts = shifts_for_day[d]
                        elig = [s for s in day_shifts if availability[w, s] == 1]
                        if isempty(elig)
                            continue
                        end
                        s_pick = rand(rng, elig)
                        assigned[w, s_pick] = 1
                        worked_day[w, d] = true
                        assigned_count_worker[w] += 1
                        day_used[d] += 1
                        need -= 1
                        placed = true
                        break
                    end
                end
                if !placed
                    # Free up one slot by reducing a staffed requirement on an available day
                    freed = false
                    for d in candidate_days
                        # Find a shift with positive requirement to reduce
                        day_shifts = shifts_for_day[d]
                        # pick shift with max current requirement
                        s_sorted = sort(day_shifts, by = s -> staffing_req[s], rev = true)
                        for s in s_sorted
                            if staffing_req[s] > 0
                                staffing_req[s] -= 1
                                cap_d = sum(avail_day[:, d])
                                if day_used[d] < cap_d
                                    elig = [ss for ss in day_shifts if availability[w, ss] == 1]
                                    if isempty(elig)
                                        continue
                                    end
                                    s_pick = rand(rng, elig)
                                    assigned[w, s_pick] = 1
                                    worked_day[w, d] = true
                                    assigned_count_worker[w] += 1
                                    day_used[d] += 1
                                    need -= 1
                                    freed = true
                                    break
                                end
                            end
                        end
                        if freed
                            break
                        end
                    end
                    if !freed
                        # Cannot satisfy this worker's minimum under structural constraints; lower the global minimum
                        min_worker_shifts = min(min_worker_shifts, assigned_count_worker[w])
                        break
                    end
                end
            end
        end
    end
    
    # If an infeasible instance is requested, inject a realistic hard conflict with diversity
    if solution_status == :infeasible
        rng = Random.default_rng()
        modes = [:shift_blackout, :day_overload, :min_over_cap]
        mode = rand(rng, modes)
        if mode == :shift_blackout
            # Pick 1-2 busy shifts and make them completely unavailable while keeping demand
            num = rand(rng, 1:2)
            shift_indices = sortperm(staffing_req, rev=true)[1:min(num, total_shifts)]
            for s in shift_indices
                staffing_req[s] = max(staffing_req[s], 1)
                for w in 1:n_workers
                    availability[w, s] = 0
                end
            end
        elseif mode == :day_overload
            # Choose a day and increase total demand beyond day capacity without exceeding per-shift availability
            d = rand(rng, 1:n_days)
            day_shifts = shifts_for_day[d]
            cap_d = sum([any(availability[w, s] == 1 for s in day_shifts) for w in 1:n_workers])
            cur = sum(staffing_req[day_shifts])
            target = max(cur, cap_d + rand(rng, 1:3))
            # Increment across random shifts, but do not exceed per-shift availability for subtle infeasibility
            while sum(staffing_req[day_shifts]) < target
                s = rand(rng, day_shifts)
                if staffing_req[s] < sum(availability[:, s])
                    staffing_req[s] += 1
                else
                    # occasionally also allow exceeding per-shift availability to ensure infeasibility
                    if rand(rng) < 0.3
                        staffing_req[s] += 1
                    end
                end
            end
        else
            # Raise the global minimum above at least one worker's capacity
            # Compute per-worker capacity as before
            avail_day = falses(n_workers, n_days)
            for w in 1:n_workers, d in 1:n_days
                sd = shifts_for_day[d]
                avail_day[w, d] = any(availability[w, s] == 1 for s in sd)
            end
            function max_assignable_days_for_runs2(av::AbstractVector{Bool}, K::Int)
                if K <= 0
                    return 0
                end
                total = 0
                i = 1
                L = length(av)
                while i <= L
                    if !av[i]
                        i += 1
                        continue
                    end
                    j = i
                    while j <= L && av[j]
                        j += 1
                    end
                    run_len = j - i
                    if K >= run_len
                        total += run_len
                    else
                        q = fld(run_len, K + 1)
                        r = run_len % (K + 1)
                        total += q * K + min(K, r)
                    end
                    i = j
                end
                return total
            end
            caps = [min(max_worker_shifts, max_assignable_days_for_runs2(vec(avail_day[w, :]), max_consecutive_shifts)) for w in 1:n_workers]
            cmin = minimum(caps)
            min_worker_shifts = max(min_worker_shifts, cmin + 1)
        end
    end
    
    # Store generated data in params (include any adjustments)
    actual_params[:staffing_req] = staffing_req
    actual_params[:availability] = availability
    actual_params[:costs] = costs
    if skill_based
        actual_params[:worker_skills] = worker_skills
        actual_params[:shift_skill_req] = shift_skill_req
    end
    actual_params[:min_worker_shifts] = min_worker_shifts
    actual_params[:max_worker_shifts] = max_worker_shifts
    
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
    
    # At most one shift per worker per day (realistic daily scheduling limit)
    for w in 1:n_workers, d in 1:n_days
        day_shifts = shifts_for_day[d]
        if !isempty(day_shifts)
            @constraint(model, sum(x[w, s] for s in day_shifts) <= 1)
        end
    end
    
    # Minimum and maximum shifts per worker
    for w in 1:n_workers
        @constraint(model, sum(x[w, s] for s in 1:total_shifts) >= min_worker_shifts)
        @constraint(model, sum(x[w, s] for s in 1:total_shifts) <= max_worker_shifts)
    end
    
    # Maximum consecutive working days: in any window of (max_consecutive_shifts + 1) days,
    # a worker must have at least one day off.
    if max_consecutive_shifts >= 1 && n_days > max_consecutive_shifts
        window_len = max_consecutive_shifts + 1
        for w in 1:n_workers
            for start_day in 1:(n_days - window_len + 1)
                window_days = start_day:(start_day + window_len - 1)
                window_shifts = reduce(vcat, [shifts_for_day[d] for d in window_days])
                @constraint(model, sum(x[w, s] for s in window_shifts) <= max_consecutive_shifts)
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