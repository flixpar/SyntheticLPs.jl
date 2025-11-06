using JuMP
using Random
using StatsBase
using Distributions

"""
    SchedulingProblem <: ProblemGenerator

Generator for workforce scheduling optimization problems that minimize staffing costs while meeting
shift requirements and respecting worker constraints.

This problem models realistic workforce scheduling with:
- Multiple shifts across multiple days
- Worker availability patterns (full-time vs part-time)
- Shift-specific staffing requirements
- Worker minimum/maximum shift constraints
- Maximum consecutive working days constraint
- At most one shift per worker per day
- Optional skill-based scheduling

# Fields
All data generated in constructor based on target_variables and feasibility_status:
- `n_workers::Int`: Number of workers
- `n_shifts::Int`: Number of shifts per day
- `n_days::Int`: Number of days in planning horizon
- `total_shifts::Int`: Total number of shifts (n_shifts × n_days)
- `staffing_req::Vector{Int}`: Staffing requirement for each shift
- `availability::Matrix{Int}`: Worker availability (1 if available, 0 otherwise)
- `costs::Matrix{Float64}`: Cost per worker per shift
- `min_worker_shifts::Int`: Minimum shifts per worker
- `max_worker_shifts::Int`: Maximum shifts per worker
- `max_consecutive_shifts::Int`: Maximum consecutive working days
- `skill_based::Bool`: Whether skill-based scheduling is enabled
- `worker_skills::Union{Matrix{Int}, Nothing}`: Worker skills matrix (if skill-based)
- `shift_skill_req::Union{Matrix{Int}, Nothing}`: Shift skill requirements (if skill-based)
"""
struct SchedulingProblem <: ProblemGenerator
    n_workers::Int
    n_shifts::Int
    n_days::Int
    total_shifts::Int
    staffing_req::Vector{Int}
    availability::Matrix{Int}
    costs::Matrix{Float64}
    min_worker_shifts::Int
    max_worker_shifts::Int
    max_consecutive_shifts::Int
    skill_based::Bool
    worker_skills::Union{Matrix{Int}, Nothing}
    shift_skill_req::Union{Matrix{Int}, Nothing}
end

"""
    SchedulingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a scheduling problem instance with sophisticated consecutive-day capacity logic.

# Sophisticated Feasibility Logic Preserved:
- **Consecutive-day capacity**: Calculates exact assignable days considering consecutive working limit
- **Randomized matching**: Uses randomized greedy matching to assign workers to shifts
- **Window constraint checking**: Helper function validates consecutive-day constraints before assignment
- **Demand capping with slack**: Uses randomized Beta-distributed slack factors per shift and per day
- **Proportional scaling**: Scales demands down proportionally with randomized rounding when needed
- **Global capacity balancing**: Ensures total demand doesn't exceed total worker capacity with reserve
- **Minimum enforcement**: Brings each worker up to minimum by using day capacity slack
- **Diverse infeasibility modes**: Shift blackout, day overload, or minimum over capacity

# Arguments
- `target_variables`: Target number of variables (n_workers × n_shifts × n_days)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function SchedulingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Determine problem dimensions
    if target_variables <= 250
        # Small: department stores, small restaurants, clinics
        n_workers = round(Int, rand(Uniform(4, 12)))
        n_shifts = round(Int, rand(Uniform(2, 4)))
        n_days = round(Int, rand(Uniform(3, 7)))
        min_staffing = round(Int, rand(Uniform(1, 2)))
        max_staffing = min_staffing + round(Int, rand(Uniform(1, 3)))
        availability_density = rand(Beta(8, 2))
        min_worker_shifts = round(Int, rand(Uniform(2, 4)))
        max_worker_shifts = min_worker_shifts + round(Int, rand(Uniform(1, 3)))
        max_consecutive_shifts = round(Int, rand(Uniform(2, 3)))
        min_cost = rand(Uniform(15, 25))
        max_cost = min_cost + rand(Uniform(20, 40))
        skill_based = rand() < 0.2
    elseif target_variables <= 1000
        # Medium: hospitals, call centers, retail chains
        n_workers = round(Int, rand(Uniform(8, 25)))
        n_shifts = round(Int, rand(Uniform(3, 6)))
        n_days = round(Int, rand(Uniform(5, 14)))
        min_staffing = round(Int, rand(Uniform(2, 4)))
        max_staffing = min_staffing + round(Int, rand(Uniform(2, 5)))
        availability_density = rand(Beta(6, 3))
        min_worker_shifts = round(Int, rand(Uniform(3, 5)))
        max_worker_shifts = min_worker_shifts + round(Int, rand(Uniform(2, 4)))
        max_consecutive_shifts = round(Int, rand(Uniform(2, 4)))
        min_cost = rand(Uniform(20, 35))
        max_cost = min_cost + rand(Uniform(25, 60))
        skill_based = rand() < 0.4
    else
        # Large: airlines, large hospitals, manufacturing
        n_workers = round(Int, rand(Uniform(25, 80)))
        n_shifts = round(Int, rand(Uniform(4, 8)))
        n_days = round(Int, rand(Uniform(7, 30)))
        min_staffing = round(Int, rand(Uniform(3, 6)))
        max_staffing = min_staffing + round(Int, rand(Uniform(3, 8)))
        availability_density = rand(Beta(4, 3))
        min_worker_shifts = round(Int, rand(Uniform(4, 6)))
        max_worker_shifts = min_worker_shifts + round(Int, rand(Uniform(2, 5)))
        max_consecutive_shifts = round(Int, rand(Uniform(3, 5)))
        min_cost = rand(Uniform(25, 50))
        max_cost = min_cost + rand(Uniform(30, 100))
        skill_based = rand() < 0.6
    end

    n_skills = skill_based ? (target_variables <= 250 ? round(Int, rand(Uniform(2, 3))) :
                              target_variables <= 1000 ? round(Int, rand(Uniform(3, 5))) :
                              round(Int, rand(Uniform(4, 8)))) : 0

    total_shifts = n_shifts * n_days

    # Generate staffing requirements
    staffing_req = Vector{Int}(undef, total_shifts)
    for s in 1:total_shifts
        day_idx = div(s-1, n_shifts) + 1
        shift_in_day = ((s-1) % n_shifts) + 1

        peak_factor = 1.0 + 0.5 * sin(π * shift_in_day / n_shifts)
        weekend_factor = (day_idx % 7 in [0, 6]) ? 0.8 : 1.0

        mean_staffing = (min_staffing + max_staffing) / 2 * peak_factor * weekend_factor
        staffing_req[s] = max(min_staffing, min(max_staffing,
                             round(Int, rand(Poisson(mean_staffing)))))
    end

    # Generate worker availability
    availability = zeros(Int, n_workers, total_shifts)

    for w in 1:n_workers
        worker_type = rand() < 0.6 ? :full_time : :part_time

        for s in 1:total_shifts
            day_idx = div(s-1, n_shifts) + 1
            shift_in_day = ((s-1) % n_shifts) + 1

            if worker_type == :full_time
                base_prob = availability_density
                if shift_in_day <= 2 || shift_in_day >= n_shifts - 1
                    base_prob *= 0.7
                end
                if day_idx % 7 in [0, 6]
                    base_prob *= 0.8
                end
            else
                base_prob = availability_density * 0.8
                if shift_in_day > n_shifts / 2
                    base_prob *= 1.2
                end
                if day_idx % 7 in [0, 6]
                    base_prob *= 1.1
                end
            end

            availability[w, s] = rand() < min(1.0, base_prob) ? 1 : 0
        end
    end

    # Generate worker skills if skill-based
    worker_skills = nothing
    shift_skill_req = nothing

    if skill_based
        worker_skills = zeros(Int, n_workers, n_skills)
        for w in 1:n_workers
            num_skills = rand(1:min(3, n_skills))
            skill_indices = sample(1:n_skills, num_skills, replace=false)
            for s in skill_indices
                worker_skills[w, s] = 1
            end
        end

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

    # Map day -> shifts
    shifts_for_day = [collect(((d-1)*n_shifts + 1):min(d*n_shifts, total_shifts)) for d in 1:n_days]

    # Generate costs
    costs = zeros(n_workers, total_shifts)
    worker_tiers = rand([:junior, :regular, :senior], n_workers)

    for w in 1:n_workers
        tier = worker_tiers[w]
        if tier == :junior
            base_cost = min_cost + rand(Normal(0, (max_cost - min_cost) * 0.1))
        elseif tier == :regular
            base_cost = (min_cost + max_cost) / 2 + rand(Normal(0, (max_cost - min_cost) * 0.15))
        else  # senior
            base_cost = max_cost + rand(Normal(0, (max_cost - min_cost) * 0.1))
        end

        if skill_based
            num_skills = sum(worker_skills[w, :])
            skill_premium = num_skills / n_skills * (max_cost - min_cost) * 0.3
            base_cost += skill_premium
        end

        base_cost = max(min_cost * 0.8, min(max_cost * 1.2, base_cost))

        for s in 1:total_shifts
            day_idx = div(s-1, n_shifts) + 1
            shift_in_day = ((s-1) % n_shifts) + 1

            shift_premium = 1.0
            if shift_in_day <= 2 || shift_in_day >= n_shifts - 1
                shift_premium *= 1.15
            end
            if day_idx % 7 in [0, 6]
                shift_premium *= 1.1
            end
            if day_idx % 30 == 0
                shift_premium *= 1.25
            end

            random_factor = rand(LogNormal(0, 0.1))
            costs[w, s] = base_cost * shift_premium * random_factor
        end
    end

    # SOPHISTICATED FEASIBILITY ENFORCEMENT
    if feasibility_status == feasible
        # Helper: consecutive-days capacity
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

        # Per-worker capacity
        worker_cap = zeros(Int, n_workers)
        for w in 1:n_workers
            cap_runs = max_assignable_days_for_runs(vec(avail_day[w, :]), max_consecutive_shifts)
            worker_cap[w] = min(max_worker_shifts, cap_runs)
        end

        # Adjust global min to guarantee attainability
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

        # Randomized matching respecting daily and consecutive-day caps
        assigned = zeros(Int, n_workers, total_shifts)
        worked_day = falses(n_workers, n_days)
        assigned_count_worker = zeros(Int, n_workers)

        # Helper: check window constraint
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

        # Cover per-shift demands per day
        for d in 1:n_days
            day_shifts = shifts_for_day[d]
            workers_d = [w for w in 1:n_workers if any(availability[w, s] == 1 for s in day_shifts)]
            shuffle!(rng, workers_d)
            demand_rem = Dict(s => staffing_req[s] for s in day_shifts)

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

            # Reduce unmet demand
            for s in day_shifts
                if demand_rem[s] > 0
                    staffing_req[s] -= demand_rem[s]
                end
            end
        end

        # Align requirements to covered amounts
        staffing_req = vec(sum(assigned, dims=1))

        # Bring each worker up to minimum
        assigned_count_worker = [sum(assigned[w, :]) for w in 1:n_workers]
        day_used = [sum(WorkedDay -> WorkedDay ? 1 : 0, worked_day[:, d]) for d in 1:n_days]

        for w in 1:n_workers
            need = max(0, min_worker_shifts - assigned_count_worker[w])
            while need > 0
                candidate_days = [d for d in 1:n_days if avail_day[w, d] && !worked_day[w, d] &&
                                 can_add_day!(vec(worked_day[w, :]), d, max_consecutive_shifts)]
                candidate_days = sort(candidate_days, by = d -> (sum(avail_day[:, d]) - day_used[d]), rev = true)
                placed = false
                for d in candidate_days
                    cap_d = sum(avail_day[:, d])
                    if day_used[d] < cap_d
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
                    # Free up one slot
                    freed = false
                    for d in candidate_days
                        day_shifts = shifts_for_day[d]
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
                        min_worker_shifts = min(min_worker_shifts, assigned_count_worker[w])
                        break
                    end
                end
            end
        end

    elseif feasibility_status == infeasible
        # Diverse infeasibility modes
        rng = Random.default_rng()
        modes = [:shift_blackout, :day_overload, :min_over_cap]
        mode = rand(rng, modes)

        if mode == :shift_blackout
            # Pick busy shifts and make them unavailable
            num = rand(rng, 1:2)
            shift_indices = sortperm(staffing_req, rev=true)[1:min(num, total_shifts)]
            for s in shift_indices
                staffing_req[s] = max(staffing_req[s], 1)
                for w in 1:n_workers
                    availability[w, s] = 0
                end
            end
        elseif mode == :day_overload
            # Increase day demand beyond capacity
            d = rand(rng, 1:n_days)
            day_shifts = shifts_for_day[d]
            cap_d = sum([any(availability[w, s] == 1 for s in day_shifts) for w in 1:n_workers])
            cur = sum(staffing_req[day_shifts])
            target = max(cur, cap_d + rand(rng, 1:3))
            while sum(staffing_req[day_shifts]) < target
                s = rand(rng, day_shifts)
                if staffing_req[s] < sum(availability[:, s])
                    staffing_req[s] += 1
                else
                    if rand(rng) < 0.3
                        staffing_req[s] += 1
                    end
                end
            end
        else
            # Raise minimum above worker capacity
            shifts_for_day_temp = [collect(((d-1)*n_shifts + 1):min(d*n_shifts, total_shifts)) for d in 1:n_days]
            avail_day = falses(n_workers, n_days)
            for w in 1:n_workers, d in 1:n_days
                sd = shifts_for_day_temp[d]
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

    return SchedulingProblem(
        n_workers,
        n_shifts,
        n_days,
        total_shifts,
        staffing_req,
        availability,
        costs,
        min_worker_shifts,
        max_worker_shifts,
        max_consecutive_shifts,
        skill_based,
        worker_skills,
        shift_skill_req
    )
end

"""
    build_model(prob::SchedulingProblem)

Build a JuMP model for the scheduling problem (deterministic).

# Arguments
- `prob`: SchedulingProblem instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::SchedulingProblem)
    model = Model()

    @variable(model, x[1:prob.n_workers, 1:prob.total_shifts], Bin)

    @objective(model, Min, sum(prob.costs[w, s] * x[w, s] for w in 1:prob.n_workers, s in 1:prob.total_shifts))

    # Staffing requirements
    for s in 1:prob.total_shifts
        @constraint(model, sum(x[w, s] for w in 1:prob.n_workers) >= prob.staffing_req[s])
    end

    # Availability constraints
    for w in 1:prob.n_workers, s in 1:prob.total_shifts
        if prob.availability[w, s] == 0
            @constraint(model, x[w, s] == 0)
        end
    end

    # At most one shift per worker per day
    shifts_for_day = [collect(((d-1)*prob.n_shifts + 1):min(d*prob.n_shifts, prob.total_shifts)) for d in 1:prob.n_days]
    for w in 1:prob.n_workers, d in 1:prob.n_days
        day_shifts = shifts_for_day[d]
        if !isempty(day_shifts)
            @constraint(model, sum(x[w, s] for s in day_shifts) <= 1)
        end
    end

    # Minimum and maximum shifts per worker
    for w in 1:prob.n_workers
        @constraint(model, sum(x[w, s] for s in 1:prob.total_shifts) >= prob.min_worker_shifts)
        @constraint(model, sum(x[w, s] for s in 1:prob.total_shifts) <= prob.max_worker_shifts)
    end

    # Maximum consecutive working days
    if prob.max_consecutive_shifts >= 1 && prob.n_days > prob.max_consecutive_shifts
        window_len = prob.max_consecutive_shifts + 1
        for w in 1:prob.n_workers
            for start_day in 1:(prob.n_days - window_len + 1)
                window_days = start_day:(start_day + window_len - 1)
                window_shifts = reduce(vcat, [shifts_for_day[d] for d in window_days])
                @constraint(model, sum(x[w, s] for s in window_shifts) <= prob.max_consecutive_shifts)
            end
        end
    end

    return model
end

# Register the problem type
register_problem(
    :scheduling,
    SchedulingProblem,
    "Workforce scheduling problem that minimizes staffing costs while meeting shift requirements and respecting worker constraints"
)
