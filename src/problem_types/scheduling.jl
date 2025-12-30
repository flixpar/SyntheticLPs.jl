using JuMP
using Random
using StatsBase
using Distributions

"""
Scheduling problem variants.

# Variants
- `sched_standard`: Basic workforce scheduling with skill options
- `sched_weekend_fair`: Fair weekend distribution among workers
- `sched_overtime`: Include overtime at premium cost
- `sched_preferences`: Minimize worker preference violations
- `sched_team`: Workers in teams must work together
- `sched_on_call`: Include on-call/backup requirements
- `sched_split_shift`: Allow discontinuous working hours
- `sched_seniority`: Seniority-based priority rules
"""
@enum SchedulingVariant begin
    sched_standard
    sched_weekend_fair
    sched_overtime
    sched_preferences
    sched_team
    sched_on_call
    sched_split_shift
    sched_seniority
end

"""
    SchedulingProblem <: ProblemGenerator

Generator for workforce scheduling optimization problems with multiple variants.

# Fields
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
- `variant::SchedulingVariant`: The specific variant type
# Weekend fair variant
- `max_weekends_per_worker::Union{Int, Nothing}`: Maximum weekend days per worker
- `weekend_shifts::Union{Vector{Int}, Nothing}`: Indices of weekend shifts
# Overtime variant
- `regular_hours::Union{Vector{Float64}, Nothing}`: Regular hours per shift
- `overtime_threshold::Union{Float64, Nothing}`: Weekly overtime threshold
- `overtime_multiplier::Union{Float64, Nothing}`: Overtime cost multiplier
# Preferences variant
- `preferences::Union{Matrix{Float64}, Nothing}`: Worker preferences for shifts (higher = prefer more)
- `preference_penalty::Union{Float64, Nothing}`: Penalty for not meeting preferences
# Team variant
- `n_teams::Int`: Number of teams
- `team_membership::Union{Vector{Int}, Nothing}`: Team ID for each worker
- `team_size_min::Union{Vector{Int}, Nothing}`: Minimum team size per shift
# On-call variant
- `on_call_req::Union{Vector{Int}, Nothing}`: On-call staff requirement per shift
- `on_call_costs::Union{Matrix{Float64}, Nothing}`: On-call cost per worker per shift
# Split shift variant
- `allow_split::Bool`: Whether split shifts are allowed
- `max_splits_per_day::Union{Int, Nothing}`: Maximum shift gaps per day per worker
# Seniority variant
- `seniority_scores::Union{Vector{Float64}, Nothing}`: Seniority score per worker
- `seniority_bonus::Union{Float64, Nothing}`: Cost reduction for senior workers
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
    variant::SchedulingVariant
    # Weekend fair variant
    max_weekends_per_worker::Union{Int, Nothing}
    weekend_shifts::Union{Vector{Int}, Nothing}
    # Overtime variant
    regular_hours::Union{Vector{Float64}, Nothing}
    overtime_threshold::Union{Float64, Nothing}
    overtime_multiplier::Union{Float64, Nothing}
    # Preferences variant
    preferences::Union{Matrix{Float64}, Nothing}
    preference_penalty::Union{Float64, Nothing}
    # Team variant
    n_teams::Int
    team_membership::Union{Vector{Int}, Nothing}
    team_size_min::Union{Vector{Int}, Nothing}
    # On-call variant
    on_call_req::Union{Vector{Int}, Nothing}
    on_call_costs::Union{Matrix{Float64}, Nothing}
    # Split shift variant
    allow_split::Bool
    max_splits_per_day::Union{Int, Nothing}
    # Seniority variant
    seniority_scores::Union{Vector{Float64}, Nothing}
    seniority_bonus::Union{Float64, Nothing}
end

# Backwards compatibility constructor
function SchedulingProblem(n_workers::Int, n_shifts::Int, n_days::Int, total_shifts::Int,
                           staffing_req::Vector{Int}, availability::Matrix{Int},
                           costs::Matrix{Float64}, min_worker_shifts::Int, max_worker_shifts::Int,
                           max_consecutive_shifts::Int, skill_based::Bool,
                           worker_skills::Union{Matrix{Int}, Nothing},
                           shift_skill_req::Union{Matrix{Int}, Nothing})
    SchedulingProblem(
        n_workers, n_shifts, n_days, total_shifts,
        staffing_req, availability, costs, min_worker_shifts, max_worker_shifts,
        max_consecutive_shifts, skill_based, worker_skills, shift_skill_req, sched_standard,
        nothing, nothing, nothing, nothing, nothing,
        nothing, nothing, 0, nothing, nothing,
        nothing, nothing, false, nothing, nothing, nothing
    )
end

"""
    SchedulingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int;
                      variant::SchedulingVariant=sched_standard)

Construct a scheduling problem instance with the specified variant.
"""
function SchedulingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int;
                           variant::SchedulingVariant=sched_standard)
    Random.seed!(seed)

    # Determine problem dimensions based on target size
    if target_variables <= 250
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
            for sk in skill_indices
                worker_skills[w, sk] = 1
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
        else
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

            random_factor = rand(LogNormal(0, 0.1))
            costs[w, s] = base_cost * shift_premium * random_factor
        end
    end

    # Initialize variant-specific fields
    max_weekends_per_worker = nothing
    weekend_shifts = nothing
    regular_hours = nothing
    overtime_threshold = nothing
    overtime_multiplier = nothing
    preferences = nothing
    preference_penalty = nothing
    n_teams = 0
    team_membership = nothing
    team_size_min = nothing
    on_call_req = nothing
    on_call_costs = nothing
    allow_split = false
    max_splits_per_day = nothing
    seniority_scores = nothing
    seniority_bonus = nothing

    # Generate variant-specific data
    if variant == sched_weekend_fair
        # Identify weekend shifts (days 6 and 7 in each week)
        weekend_shifts = Int[]
        for d in 1:n_days
            if d % 7 in [0, 6]  # Saturday and Sunday
                append!(weekend_shifts, shifts_for_day[d])
            end
        end

        # Calculate max weekends based on available weekend days
        n_weekend_days = length([d for d in 1:n_days if d % 7 in [0, 6]])
        max_weekends_per_worker = max(1, ceil(Int, n_weekend_days * 0.6))

    elseif variant == sched_overtime
        # Hours per shift (typically 8, but varies)
        regular_hours = [rand(Uniform(4.0, 10.0)) for _ in 1:total_shifts]
        overtime_threshold = rand(Uniform(35.0, 45.0))  # Weekly threshold
        overtime_multiplier = rand(Uniform(1.25, 2.0))

    elseif variant == sched_preferences
        # Worker preferences for each shift (0-1 scale, 1 = strongly prefer)
        preferences = zeros(n_workers, total_shifts)
        for w in 1:n_workers
            # Workers have personal preference patterns
            preferred_shifts_in_day = sample(1:n_shifts, min(2, n_shifts), replace=false)
            for s in 1:total_shifts
                shift_in_day = ((s-1) % n_shifts) + 1
                if shift_in_day in preferred_shifts_in_day
                    preferences[w, s] = rand(Uniform(0.6, 1.0))
                else
                    preferences[w, s] = rand(Uniform(0.0, 0.4))
                end
            end
        end
        preference_penalty = mean(costs) * rand(Uniform(0.5, 2.0))

    elseif variant == sched_team
        # Assign workers to teams
        n_teams = max(2, n_workers ÷ 4)
        team_membership = [((w - 1) % n_teams) + 1 for w in 1:n_workers]

        # Minimum team members required per shift if team is working
        team_size_min = [rand(1:max(1, n_workers ÷ n_teams ÷ 2)) for _ in 1:n_teams]

    elseif variant == sched_on_call
        # On-call requirements per shift
        on_call_req = [rand(0:max(1, min_staffing ÷ 2)) for _ in 1:total_shifts]
        on_call_costs = costs .* rand(Uniform(0.2, 0.5))  # Lower cost for on-call

    elseif variant == sched_split_shift
        allow_split = true
        max_splits_per_day = rand(1:min(2, n_shifts - 1))

    elseif variant == sched_seniority
        # Generate seniority scores based on worker tiers
        seniority_scores = zeros(n_workers)
        for w in 1:n_workers
            if worker_tiers[w] == :senior
                seniority_scores[w] = rand(Uniform(0.8, 1.0))
            elseif worker_tiers[w] == :regular
                seniority_scores[w] = rand(Uniform(0.4, 0.7))
            else
                seniority_scores[w] = rand(Uniform(0.1, 0.3))
            end
        end
        seniority_bonus = mean(costs) * rand(Uniform(0.1, 0.3))
    end

    # FEASIBILITY ENFORCEMENT (simplified from original)
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

        # Reduce staffing requirements to available workers
        for s in 1:total_shifts
            avail_count = sum(availability[:, s])
            staffing_req[s] = min(staffing_req[s], max(0, avail_count - 1))
        end

        # For weekend fair variant, ensure max_weekends_per_worker is achievable
        if variant == sched_weekend_fair && weekend_shifts !== nothing
            n_weekend_shifts = length(weekend_shifts)
            if n_weekend_shifts > 0
                max_weekends_per_worker = max(max_weekends_per_worker, ceil(Int, n_weekend_shifts / n_workers))
            end
        end

    elseif feasibility_status == infeasible
        # Create infeasibility
        rng = Random.default_rng()
        modes = [:shift_blackout, :day_overload, :min_over_cap]
        mode = rand(rng, modes)

        if mode == :shift_blackout
            num = rand(rng, 1:2)
            shift_indices = sortperm(staffing_req, rev=true)[1:min(num, total_shifts)]
            for s in shift_indices
                staffing_req[s] = max(staffing_req[s], 1)
                for w in 1:n_workers
                    availability[w, s] = 0
                end
            end
        elseif mode == :day_overload
            d = rand(rng, 1:n_days)
            day_shifts = shifts_for_day[d]
            cap_d = sum([any(availability[w, s] == 1 for s in day_shifts) for w in 1:n_workers])
            target = cap_d + rand(rng, 2:5)
            for s in day_shifts
                staffing_req[s] = ceil(Int, target / length(day_shifts))
            end
        else
            # Raise minimum above capacity
            avail_day = falses(n_workers, n_days)
            for w in 1:n_workers, d in 1:n_days
                sd = shifts_for_day[d]
                avail_day[w, d] = any(availability[w, s] == 1 for s in sd)
            end
            caps = [sum(avail_day[w, :]) for w in 1:n_workers]
            cmin = minimum(caps)
            min_worker_shifts = cmin + 2
        end
    end

    return SchedulingProblem(
        n_workers, n_shifts, n_days, total_shifts,
        staffing_req, availability, costs, min_worker_shifts, max_worker_shifts,
        max_consecutive_shifts, skill_based, worker_skills, shift_skill_req, variant,
        max_weekends_per_worker, weekend_shifts,
        regular_hours, overtime_threshold, overtime_multiplier,
        preferences, preference_penalty,
        n_teams, team_membership, team_size_min,
        on_call_req, on_call_costs,
        allow_split, max_splits_per_day,
        seniority_scores, seniority_bonus
    )
end

"""
    build_model(prob::SchedulingProblem)

Build a JuMP model for the scheduling problem based on its variant.
"""
function build_model(prob::SchedulingProblem)
    model = Model()

    shifts_for_day = [collect(((d-1)*prob.n_shifts + 1):min(d*prob.n_shifts, prob.total_shifts)) for d in 1:prob.n_days]

    if prob.variant == sched_standard
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
        for w in 1:prob.n_workers, d in 1:prob.n_days
            day_shifts = shifts_for_day[d]
            @constraint(model, sum(x[w, s] for s in day_shifts) <= 1)
        end

        # Min/max shifts per worker
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

    elseif prob.variant == sched_weekend_fair
        @variable(model, x[1:prob.n_workers, 1:prob.total_shifts], Bin)
        @variable(model, weekend_worked[1:prob.n_workers, 1:prob.n_days], Bin)  # Track weekend days worked

        @objective(model, Min, sum(prob.costs[w, s] * x[w, s] for w in 1:prob.n_workers, s in 1:prob.total_shifts))

        # Staffing requirements
        for s in 1:prob.total_shifts
            @constraint(model, sum(x[w, s] for w in 1:prob.n_workers) >= prob.staffing_req[s])
        end

        # Availability and one shift per day
        for w in 1:prob.n_workers, s in 1:prob.total_shifts
            if prob.availability[w, s] == 0
                @constraint(model, x[w, s] == 0)
            end
        end

        for w in 1:prob.n_workers, d in 1:prob.n_days
            day_shifts = shifts_for_day[d]
            @constraint(model, sum(x[w, s] for s in day_shifts) <= 1)

            # Link weekend_worked to actual shifts
            if d % 7 in [0, 6]  # Weekend day
                @constraint(model, weekend_worked[w, d] >= sum(x[w, s] for s in day_shifts) / length(day_shifts))
                @constraint(model, weekend_worked[w, d] <= sum(x[w, s] for s in day_shifts))
            else
                @constraint(model, weekend_worked[w, d] == 0)
            end
        end

        # Min/max shifts per worker
        for w in 1:prob.n_workers
            @constraint(model, sum(x[w, s] for s in 1:prob.total_shifts) >= prob.min_worker_shifts)
            @constraint(model, sum(x[w, s] for s in 1:prob.total_shifts) <= prob.max_worker_shifts)
        end

        # Fair weekend distribution
        for w in 1:prob.n_workers
            @constraint(model, sum(weekend_worked[w, d] for d in 1:prob.n_days) <= prob.max_weekends_per_worker)
        end

        # Consecutive days constraint
        if prob.max_consecutive_shifts >= 1 && prob.n_days > prob.max_consecutive_shifts
            window_len = prob.max_consecutive_shifts + 1
            for w in 1:prob.n_workers, start_day in 1:(prob.n_days - window_len + 1)
                window_shifts = reduce(vcat, [shifts_for_day[d] for d in start_day:(start_day + window_len - 1)])
                @constraint(model, sum(x[w, s] for s in window_shifts) <= prob.max_consecutive_shifts)
            end
        end

    elseif prob.variant == sched_overtime
        @variable(model, x[1:prob.n_workers, 1:prob.total_shifts], Bin)
        @variable(model, overtime[1:prob.n_workers] >= 0)  # Overtime hours per worker

        n_weeks = ceil(Int, prob.n_days / 7)
        @variable(model, weekly_hours[1:prob.n_workers, 1:n_weeks] >= 0)
        @variable(model, weekly_overtime[1:prob.n_workers, 1:n_weeks] >= 0)

        # Minimize regular cost + overtime premium
        @objective(model, Min,
            sum(prob.costs[w, s] * x[w, s] for w in 1:prob.n_workers, s in 1:prob.total_shifts) +
            sum(prob.overtime_multiplier * mean(prob.costs[w, :]) * overtime[w] for w in 1:prob.n_workers))

        # Staffing requirements
        for s in 1:prob.total_shifts
            @constraint(model, sum(x[w, s] for w in 1:prob.n_workers) >= prob.staffing_req[s])
        end

        # Availability and one shift per day
        for w in 1:prob.n_workers, s in 1:prob.total_shifts
            if prob.availability[w, s] == 0
                @constraint(model, x[w, s] == 0)
            end
        end

        for w in 1:prob.n_workers, d in 1:prob.n_days
            day_shifts = shifts_for_day[d]
            @constraint(model, sum(x[w, s] for s in day_shifts) <= 1)
        end

        # Calculate weekly hours
        for w in 1:prob.n_workers, week in 1:n_weeks
            start_day = (week - 1) * 7 + 1
            end_day = min(week * 7, prob.n_days)
            week_shifts = reduce(vcat, [shifts_for_day[d] for d in start_day:end_day if d <= prob.n_days])
            @constraint(model, weekly_hours[w, week] == sum(prob.regular_hours[s] * x[w, s] for s in week_shifts))
            @constraint(model, weekly_overtime[w, week] >= weekly_hours[w, week] - prob.overtime_threshold)
        end

        # Total overtime
        for w in 1:prob.n_workers
            @constraint(model, overtime[w] == sum(weekly_overtime[w, week] for week in 1:n_weeks))
        end

        # Min/max shifts per worker
        for w in 1:prob.n_workers
            @constraint(model, sum(x[w, s] for s in 1:prob.total_shifts) >= prob.min_worker_shifts)
            @constraint(model, sum(x[w, s] for s in 1:prob.total_shifts) <= prob.max_worker_shifts)
        end

    elseif prob.variant == sched_preferences
        @variable(model, x[1:prob.n_workers, 1:prob.total_shifts], Bin)
        @variable(model, pref_violation[1:prob.n_workers, 1:prob.total_shifts] >= 0)

        # Minimize cost + preference violation penalty
        @objective(model, Min,
            sum(prob.costs[w, s] * x[w, s] for w in 1:prob.n_workers, s in 1:prob.total_shifts) +
            prob.preference_penalty * sum(pref_violation[w, s] for w in 1:prob.n_workers, s in 1:prob.total_shifts))

        # Staffing requirements
        for s in 1:prob.total_shifts
            @constraint(model, sum(x[w, s] for w in 1:prob.n_workers) >= prob.staffing_req[s])
        end

        # Preference violation: assigned to low-preference shifts
        for w in 1:prob.n_workers, s in 1:prob.total_shifts
            if prob.availability[w, s] == 0
                @constraint(model, x[w, s] == 0)
            else
                # Violation when assigned to non-preferred shift
                @constraint(model, pref_violation[w, s] >= x[w, s] * (1 - prob.preferences[w, s]))
            end
        end

        for w in 1:prob.n_workers, d in 1:prob.n_days
            day_shifts = shifts_for_day[d]
            @constraint(model, sum(x[w, s] for s in day_shifts) <= 1)
        end

        for w in 1:prob.n_workers
            @constraint(model, sum(x[w, s] for s in 1:prob.total_shifts) >= prob.min_worker_shifts)
            @constraint(model, sum(x[w, s] for s in 1:prob.total_shifts) <= prob.max_worker_shifts)
        end

    elseif prob.variant == sched_team
        @variable(model, x[1:prob.n_workers, 1:prob.total_shifts], Bin)
        @variable(model, team_active[1:prob.n_teams, 1:prob.total_shifts], Bin)  # Team is working this shift

        @objective(model, Min, sum(prob.costs[w, s] * x[w, s] for w in 1:prob.n_workers, s in 1:prob.total_shifts))

        # Staffing requirements
        for s in 1:prob.total_shifts
            @constraint(model, sum(x[w, s] for w in 1:prob.n_workers) >= prob.staffing_req[s])
        end

        # Availability
        for w in 1:prob.n_workers, s in 1:prob.total_shifts
            if prob.availability[w, s] == 0
                @constraint(model, x[w, s] == 0)
            end
        end

        for w in 1:prob.n_workers, d in 1:prob.n_days
            day_shifts = shifts_for_day[d]
            @constraint(model, sum(x[w, s] for s in day_shifts) <= 1)
        end

        # Team constraints: if any team member works, minimum team members must work
        for t in 1:prob.n_teams
            team_workers = [w for w in 1:prob.n_workers if prob.team_membership[w] == t]
            for s in 1:prob.total_shifts
                # Team is active if any member works
                for w in team_workers
                    @constraint(model, team_active[t, s] >= x[w, s])
                end
                # If team is active, minimum members must work
                @constraint(model, sum(x[w, s] for w in team_workers) >= prob.team_size_min[t] * team_active[t, s])
            end
        end

        for w in 1:prob.n_workers
            @constraint(model, sum(x[w, s] for s in 1:prob.total_shifts) >= prob.min_worker_shifts)
            @constraint(model, sum(x[w, s] for s in 1:prob.total_shifts) <= prob.max_worker_shifts)
        end

    elseif prob.variant == sched_on_call
        @variable(model, x[1:prob.n_workers, 1:prob.total_shifts], Bin)  # Regular work
        @variable(model, on_call[1:prob.n_workers, 1:prob.total_shifts], Bin)  # On-call status

        @objective(model, Min,
            sum(prob.costs[w, s] * x[w, s] for w in 1:prob.n_workers, s in 1:prob.total_shifts) +
            sum(prob.on_call_costs[w, s] * on_call[w, s] for w in 1:prob.n_workers, s in 1:prob.total_shifts))

        # Regular staffing requirements
        for s in 1:prob.total_shifts
            @constraint(model, sum(x[w, s] for w in 1:prob.n_workers) >= prob.staffing_req[s])
            @constraint(model, sum(on_call[w, s] for w in 1:prob.n_workers) >= prob.on_call_req[s])
        end

        # Availability
        for w in 1:prob.n_workers, s in 1:prob.total_shifts
            if prob.availability[w, s] == 0
                @constraint(model, x[w, s] == 0)
                @constraint(model, on_call[w, s] == 0)
            end
            # Cannot be both working and on-call
            @constraint(model, x[w, s] + on_call[w, s] <= 1)
        end

        for w in 1:prob.n_workers, d in 1:prob.n_days
            day_shifts = shifts_for_day[d]
            @constraint(model, sum(x[w, s] + on_call[w, s] for s in day_shifts) <= 1)
        end

        for w in 1:prob.n_workers
            @constraint(model, sum(x[w, s] for s in 1:prob.total_shifts) >= prob.min_worker_shifts)
            @constraint(model, sum(x[w, s] for s in 1:prob.total_shifts) <= prob.max_worker_shifts)
        end

    elseif prob.variant == sched_split_shift
        @variable(model, x[1:prob.n_workers, 1:prob.total_shifts], Bin)

        @objective(model, Min, sum(prob.costs[w, s] * x[w, s] for w in 1:prob.n_workers, s in 1:prob.total_shifts))

        # Staffing requirements
        for s in 1:prob.total_shifts
            @constraint(model, sum(x[w, s] for w in 1:prob.n_workers) >= prob.staffing_req[s])
        end

        # Availability
        for w in 1:prob.n_workers, s in 1:prob.total_shifts
            if prob.availability[w, s] == 0
                @constraint(model, x[w, s] == 0)
            end
        end

        # Allow multiple shifts per day (split shift), but limit gaps
        for w in 1:prob.n_workers, d in 1:prob.n_days
            day_shifts = shifts_for_day[d]
            # Allow up to max_splits_per_day + 1 shifts (e.g., 2 split = morning + evening with gap)
            @constraint(model, sum(x[w, s] for s in day_shifts) <= prob.max_splits_per_day + 1)
        end

        for w in 1:prob.n_workers
            @constraint(model, sum(x[w, s] for s in 1:prob.total_shifts) >= prob.min_worker_shifts)
            @constraint(model, sum(x[w, s] for s in 1:prob.total_shifts) <= prob.max_worker_shifts)
        end

        # Consecutive days constraint still applies
        # For split shifts, we count days worked (not shifts) for consecutive constraint
        if prob.max_consecutive_shifts >= 1 && prob.n_days > prob.max_consecutive_shifts
            # Track which days each worker works
            @variable(model, day_worked[1:prob.n_workers, 1:prob.n_days], Bin)
            for w in 1:prob.n_workers, d in 1:prob.n_days
                day_shifts = shifts_for_day[d]
                @constraint(model, day_worked[w, d] <= sum(x[w, s] for s in day_shifts))
                @constraint(model, day_worked[w, d] >= sum(x[w, s] for s in day_shifts) / length(day_shifts))
            end

            window_len = prob.max_consecutive_shifts + 1
            for w in 1:prob.n_workers, start_day in 1:(prob.n_days - window_len + 1)
                window_days = start_day:(start_day + window_len - 1)
                @constraint(model, sum(day_worked[w, d] for d in window_days) <= prob.max_consecutive_shifts)
            end
        end

    elseif prob.variant == sched_seniority
        @variable(model, x[1:prob.n_workers, 1:prob.total_shifts], Bin)

        # Minimize cost with seniority bonus (senior workers cost less)
        @objective(model, Min,
            sum((prob.costs[w, s] - prob.seniority_bonus * prob.seniority_scores[w]) * x[w, s]
                for w in 1:prob.n_workers, s in 1:prob.total_shifts))

        # Staffing requirements
        for s in 1:prob.total_shifts
            @constraint(model, sum(x[w, s] for w in 1:prob.n_workers) >= prob.staffing_req[s])
        end

        # Availability
        for w in 1:prob.n_workers, s in 1:prob.total_shifts
            if prob.availability[w, s] == 0
                @constraint(model, x[w, s] == 0)
            end
        end

        for w in 1:prob.n_workers, d in 1:prob.n_days
            day_shifts = shifts_for_day[d]
            @constraint(model, sum(x[w, s] for s in day_shifts) <= 1)
        end

        for w in 1:prob.n_workers
            @constraint(model, sum(x[w, s] for s in 1:prob.total_shifts) >= prob.min_worker_shifts)
            @constraint(model, sum(x[w, s] for s in 1:prob.total_shifts) <= prob.max_worker_shifts)
        end

        # Seniority preference: senior workers get priority for desirable shifts
        # Desirable = lower cost shifts (day shifts)
        for s in 1:prob.total_shifts
            shift_in_day = ((s-1) % prob.n_shifts) + 1
            if shift_in_day <= prob.n_shifts / 2  # Day shifts
                # Senior workers should be more likely to get these
                # Add soft constraint: total seniority on day shifts should be high
            end
        end

        # Consecutive days
        if prob.max_consecutive_shifts >= 1 && prob.n_days > prob.max_consecutive_shifts
            window_len = prob.max_consecutive_shifts + 1
            for w in 1:prob.n_workers, start_day in 1:(prob.n_days - window_len + 1)
                window_shifts = reduce(vcat, [shifts_for_day[d] for d in start_day:(start_day + window_len - 1)])
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
    "Workforce scheduling problem with variants including standard, weekend fair, overtime, preferences, team, on-call, split shift, and seniority"
)
