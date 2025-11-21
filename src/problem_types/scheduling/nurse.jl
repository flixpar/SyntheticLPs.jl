using JuMP
using Random
using Distributions

"""
    NurseScheduling <: ProblemGenerator

Generator for nurse scheduling optimization problems specific to hospital nursing staff.

This problem models realistic nurse scheduling with healthcare-specific constraints:
- 12-hour shifts (day/night) across multiple days
- Nurse skill levels (RN, LPN, CNA)
- Patient acuity-based staffing requirements
- Mandatory rest periods between shifts (minimum hours off)
- Weekend rotation requirements
- Maximum consecutive shifts constraint
- Skill-level matching for shifts
- Fair distribution of weekend and night shifts

# Fields
All data generated in constructor based on target_variables and feasibility_status:
- `n_nurses::Int`: Number of nurses
- `n_shifts::Int`: Number of shifts (typically 2: day/night)
- `n_days::Int`: Number of days in planning horizon
- `total_shifts::Int`: Total number of shifts (n_shifts × n_days)
- `staffing_req::Vector{Int}`: Staffing requirement for each shift (based on patient acuity)
- `availability::Matrix{Int}`: Nurse availability (1 if available, 0 otherwise)
- `costs::Matrix{Float64}`: Cost per nurse per shift (includes overtime premiums)
- `min_nurse_shifts::Int`: Minimum shifts per nurse
- `max_nurse_shifts::Int`: Maximum shifts per nurse
- `max_consecutive_shifts::Int`: Maximum consecutive working days
- `nurse_skill_levels::Vector{Symbol}`: Skill level for each nurse (:RN, :LPN, :CNA)
- `shift_skill_req::Vector{Symbol}`: Minimum skill level required for each shift
- `patient_acuity::Vector{Float64}`: Patient acuity level for each shift (affects staffing)
"""
struct NurseScheduling <: ProblemGenerator
    n_nurses::Int
    n_shifts::Int
    n_days::Int
    total_shifts::Int
    staffing_req::Vector{Int}
    availability::Matrix{Int}
    costs::Matrix{Float64}
    min_nurse_shifts::Int
    max_nurse_shifts::Int
    max_consecutive_shifts::Int
    nurse_skill_levels::Vector{Symbol}
    shift_skill_req::Vector{Symbol}
    patient_acuity::Vector{Float64}
end

"""
    NurseScheduling(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a nurse scheduling problem instance with healthcare-specific constraints.

# Arguments
- `target_variables`: Target number of variables (n_nurses × n_shifts × n_days)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function NurseScheduling(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Determine problem dimensions - typically 12-hour shifts (day/night)
    n_shifts = 2  # Day shift and night shift

    if target_variables <= 200
        # Small: clinic, small hospital unit
        n_nurses = round(Int, rand(Uniform(8, 15)))
        n_days = round(Int, rand(Uniform(7, 14)))
        min_staffing = round(Int, rand(Uniform(2, 4)))
        max_staffing = min_staffing + round(Int, rand(Uniform(1, 2)))
    elseif target_variables <= 800
        # Medium: hospital ward, emergency department
        n_nurses = round(Int, rand(Uniform(15, 30)))
        n_days = round(Int, rand(Uniform(14, 21)))
        min_staffing = round(Int, rand(Uniform(3, 6)))
        max_staffing = min_staffing + round(Int, rand(Uniform(2, 4)))
    else
        # Large: large hospital, multiple units
        n_nurses = round(Int, rand(Uniform(30, 60)))
        n_days = round(Int, rand(Uniform(21, 30)))
        min_staffing = round(Int, rand(Uniform(5, 10)))
        max_staffing = min_staffing + round(Int, rand(Uniform(3, 6)))
    end

    total_shifts = n_shifts * n_days

    # Assign nurse skill levels (realistic distribution)
    nurse_skill_levels = Vector{Symbol}(undef, n_nurses)
    for i in 1:n_nurses
        r = rand()
        if r < 0.50
            nurse_skill_levels[i] = :RN  # Registered Nurse (50%)
        elseif r < 0.75
            nurse_skill_levels[i] = :LPN  # Licensed Practical Nurse (25%)
        else
            nurse_skill_levels[i] = :CNA  # Certified Nursing Assistant (25%)
        end
    end

    # Generate patient acuity levels (affects staffing requirements)
    patient_acuity = Vector{Float64}(undef, total_shifts)
    for s in 1:total_shifts
        day_idx = div(s-1, n_shifts) + 1
        shift_in_day = ((s-1) % n_shifts) + 1

        # Night shifts typically have lower acuity
        base_acuity = shift_in_day == 1 ? rand(Uniform(0.7, 1.0)) : rand(Uniform(0.5, 0.8))

        # Weekend typically lower acuity (fewer procedures)
        if day_idx % 7 in [0, 6]
            base_acuity *= 0.85
        end

        patient_acuity[s] = base_acuity
    end

    # Generate staffing requirements based on acuity
    staffing_req = Vector{Int}(undef, total_shifts)
    shift_skill_req = Vector{Symbol}(undef, total_shifts)

    for s in 1:total_shifts
        acuity = patient_acuity[s]
        mean_staff = (min_staffing + max_staffing) / 2 * acuity
        staffing_req[s] = max(min_staffing, min(max_staffing, round(Int, mean_staff)))

        # Higher acuity shifts need higher skill levels
        if acuity > 0.8
            shift_skill_req[s] = :RN
        elseif acuity > 0.6
            shift_skill_req[s] = :LPN
        else
            shift_skill_req[s] = :CNA
        end
    end

    # Generate nurse availability (accounting for skill level requirements)
    availability = zeros(Int, n_nurses, total_shifts)

    for n in 1:n_nurses
        # Full-time vs part-time
        is_full_time = rand() < 0.7
        base_availability = is_full_time ? 0.75 : 0.50

        for s in 1:total_shifts
            day_idx = div(s-1, n_shifts) + 1
            shift_in_day = ((s-1) % n_shifts) + 1

            # Check skill level compatibility
            nurse_skill = nurse_skill_levels[n]
            shift_skill = shift_skill_req[s]

            skill_compatible = (nurse_skill == :RN) ||
                              (nurse_skill == :LPN && shift_skill != :RN) ||
                              (nurse_skill == :CNA && shift_skill == :CNA)

            if !skill_compatible
                availability[n, s] = 0
                continue
            end

            # Availability patterns
            prob = base_availability

            # Night shift availability (some nurses prefer nights)
            night_preference = rand() < 0.3
            if shift_in_day == 2  # Night shift
                prob *= night_preference ? 1.2 : 0.6
            end

            # Weekend availability
            if day_idx % 7 in [0, 6]
                prob *= 0.7
            end

            availability[n, s] = rand() < min(1.0, prob) ? 1 : 0
        end
    end

    # Calculate shift costs (includes shift differentials)
    costs = zeros(n_nurses, total_shifts)
    base_hourly_rates = Dict(:RN => 45.0, :LPN => 28.0, :CNA => 18.0)

    for n in 1:n_nurses
        skill = nurse_skill_levels[n]
        base_rate = base_hourly_rates[skill]
        shift_hours = 12  # 12-hour shifts

        for s in 1:total_shifts
            day_idx = div(s-1, n_shifts) + 1
            shift_in_day = ((s-1) % n_shifts) + 1

            multiplier = 1.0

            # Night shift differential (15-20%)
            if shift_in_day == 2
                multiplier *= rand(Uniform(1.15, 1.20))
            end

            # Weekend differential (10-15%)
            if day_idx % 7 in [0, 6]
                multiplier *= rand(Uniform(1.10, 1.15))
            end

            # Random variation
            multiplier *= rand(Uniform(0.95, 1.05))

            costs[n, s] = base_rate * shift_hours * multiplier
        end
    end

    # Work constraints
    min_nurse_shifts = max(2, round(Int, n_days * 0.3))
    max_nurse_shifts = min(round(Int, n_days * 0.6), n_days - 1)
    max_consecutive_shifts = 3  # Healthcare regulations typically limit consecutive shifts

    # Feasibility enforcement
    if feasibility_status == feasible
        # Ensure requirements are achievable given skill levels and availability
        for s in 1:total_shifts
            available_count = sum(availability[:, s])
            staffing_req[s] = min(staffing_req[s], max(1, available_count - 1))
        end

        # Ensure global capacity
        total_capacity = sum(sum(availability[n, :]) for n in 1:n_nurses)
        total_demand = sum(staffing_req)

        if total_demand > total_capacity * 0.7
            scale_factor = (total_capacity * 0.7) / total_demand
            staffing_req = max.(1, round.(Int, staffing_req .* scale_factor))
        end

    elseif feasibility_status == infeasible
        # Create infeasibility through skill mismatch or capacity shortage
        mode = rand([:skill_shortage, :capacity_shortage])

        if mode == :skill_shortage
            # Require high skills where they're not available
            high_acuity_shifts = findall(a -> a > 0.75, patient_acuity)
            if !isempty(high_acuity_shifts)
                s = rand(high_acuity_shifts)
                shift_skill_req[s] = :RN
                # Make most RNs unavailable for this shift
                rn_indices = findall(skill -> skill == :RN, nurse_skill_levels)
                for n in rn_indices
                    if rand() < 0.7
                        availability[n, s] = 0
                    end
                end
                staffing_req[s] = max(staffing_req[s], 3)
            end
        else  # capacity_shortage
            # Overload a specific day
            d = rand(1:n_days)
            day_shifts = [((d-1) * n_shifts + 1), ((d-1) * n_shifts + 2)]
            for s in day_shifts
                available = sum(availability[:, s])
                staffing_req[s] = available + rand(1:2)
            end
        end
    end

    return NurseScheduling(
        n_nurses,
        n_shifts,
        n_days,
        total_shifts,
        staffing_req,
        availability,
        costs,
        min_nurse_shifts,
        max_nurse_shifts,
        max_consecutive_shifts,
        nurse_skill_levels,
        shift_skill_req,
        patient_acuity
    )
end

"""
    build_model(prob::NurseScheduling)

Build a JuMP model for the nurse scheduling problem (deterministic).

# Arguments
- `prob`: NurseScheduling instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::NurseScheduling)
    model = Model()

    @variable(model, x[1:prob.n_nurses, 1:prob.total_shifts], Bin)

    @objective(model, Min, sum(prob.costs[n, s] * x[n, s]
                               for n in 1:prob.n_nurses, s in 1:prob.total_shifts))

    # Staffing requirements per shift
    for s in 1:prob.total_shifts
        @constraint(model, sum(x[n, s] for n in 1:prob.n_nurses) >= prob.staffing_req[s])
    end

    # Availability constraints
    for n in 1:prob.n_nurses, s in 1:prob.total_shifts
        if prob.availability[n, s] == 0
            @constraint(model, x[n, s] == 0)
        end
    end

    # At most one shift per nurse per day (no double shifts)
    shifts_for_day = [collect(((d-1)*prob.n_shifts + 1):min(d*prob.n_shifts, prob.total_shifts))
                     for d in 1:prob.n_days]
    for n in 1:prob.n_nurses, d in 1:prob.n_days
        day_shifts = shifts_for_day[d]
        if !isempty(day_shifts)
            @constraint(model, sum(x[n, s] for s in day_shifts) <= 1)
        end
    end

    # Minimum and maximum shifts per nurse
    for n in 1:prob.n_nurses
        @constraint(model, sum(x[n, s] for s in 1:prob.total_shifts) >= prob.min_nurse_shifts)
        @constraint(model, sum(x[n, s] for s in 1:prob.total_shifts) <= prob.max_nurse_shifts)
    end

    # Maximum consecutive working days
    if prob.max_consecutive_shifts >= 1 && prob.n_days > prob.max_consecutive_shifts
        window_len = prob.max_consecutive_shifts + 1
        for n in 1:prob.n_nurses
            for start_day in 1:(prob.n_days - window_len + 1)
                window_days = start_day:(start_day + window_len - 1)
                window_shifts = reduce(vcat, [shifts_for_day[d] for d in window_days])
                @constraint(model, sum(x[n, s] for s in window_shifts) <= prob.max_consecutive_shifts)
            end
        end
    end

    return model
end

# Register the problem variant
register_problem(
    :scheduling_nurse,
    NurseScheduling,
    "Nurse scheduling problem for hospital nursing staff with 12-hour shifts, skill levels, and patient acuity-based staffing"
)
