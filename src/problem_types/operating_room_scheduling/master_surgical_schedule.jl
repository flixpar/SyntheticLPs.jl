using JuMP
using Random
using Distributions

"""
    OperatingRoomMasterScheduleProblem <: ProblemGenerator

Generator for Master Surgical Schedule (MSS) block planning problems with downstream
ward bed leveling, quota targets, and room-specialty affinities.

# Overview
Models tactical cyclical block planning (e.g. 1-week or 2-week repeating cycles) where
operating room blocks are assigned to surgical specialties/services. The formulation balances
competing objectives: satisfying departmental case demand quotas, maximizing surgeon/specialty
room-day preferences, minimizing operating room opening costs, and smoothing (leveling)
downstream inpatient/ICU ward bed occupancies.

Constraints model:
- Room exclusivity: at most one specialty per OR per day block.
- Equipment/specialty compatibility between ORs and specialties.
- Minimum and maximum block allocation bounds per specialty.
- Specialty daily room limits (preventing single-day specialty over-concentration).
- Multi-day cyclical downstream ward bed occupancy tracking and capacity bounds.
- Min-max peak bed occupancy leveling across the horizon.

# Fields
- `n_specialties::Int`: Number of surgical specialties
- `n_rooms::Int`: Number of operating rooms
- `n_days::Int`: Days in planning cycle (e.g. 5 or 10 days)
- `specialty_names::Vector{String}`: Name/label of each specialty
- `target_blocks::Vector{Int}`: Target block allocation per specialty
- `min_blocks::Vector{Int}`: Minimum guaranteed blocks per specialty
- `max_blocks::Vector{Int}`: Maximum allowed blocks per specialty
- `max_daily_rooms::Vector{Int}`: Maximum simultaneous rooms specialty can use on any day
- `room_specialty_matrix::Matrix{Int}`: Compatibility matrix (R x S)
- `specialty_preference_costs::Array{Float64,3}`: Preference cost tensor (S x R x D)
- `room_fixed_costs::Matrix{Float64}`: Fixed cost to open room r on day d
- `bed_occupancy_profiles::Matrix{Float64}`: Expected downstream bed occupancy profile (S x D)
- `ward_bed_capacity::Vector{Float64}`: Maximum inpatient bed capacity per day
- `under_allocation_penalty::Vector{Float64}`: Penalty per under-allocated block
- `over_allocation_penalty::Vector{Float64}`: Penalty per over-allocated block
- `peak_bed_weight::Float64`: Weight on peak bed leveling in objective
- `feasibility_status::FeasibilityStatus`: Resolved feasibility status
"""
struct OperatingRoomMasterScheduleProblem <: ProblemGenerator
    n_specialties::Int
    n_rooms::Int
    n_days::Int
    specialty_names::Vector{String}
    target_blocks::Vector{Int}
    min_blocks::Vector{Int}
    max_blocks::Vector{Int}
    max_daily_rooms::Vector{Int}
    room_specialty_matrix::Matrix{Int}
    specialty_preference_costs::Array{Float64,3}
    room_fixed_costs::Matrix{Float64}
    bed_occupancy_profiles::Matrix{Float64}
    ward_bed_capacity::Vector{Float64}
    under_allocation_penalty::Vector{Float64}
    over_allocation_penalty::Vector{Float64}
    peak_bed_weight::Float64
    feasibility_status::FeasibilityStatus
end

"""
    OperatingRoomMasterScheduleProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a Master Surgical Schedule (MSS) instance targeting `target_variables`.

# Variable-count formula
- `x[1:n_specialties, 1:n_rooms, 1:n_days]`: `n_specialties * n_rooms * n_days`
- `y[1:n_rooms, 1:n_days]`: `n_rooms * n_days`
- `under_blocks[1:n_specialties]`, `over_blocks[1:n_specialties]`: `2 * n_specialties`
- `daily_bed_occ[1:n_days]`: `n_days`
- `peak_bed_occ`: 1
Total variables = `(n_specialties + 1) * n_rooms * n_days + 2 * n_specialties + n_days + 1`.
"""
function OperatingRoomMasterScheduleProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    target = max(20, target_variables)

    # Scale dimensions based on target
    if target <= 100
        n_days = 5
        n_rooms = 2
    elseif target <= 400
        n_days = 5
        n_rooms = 4
    elseif target <= 1500
        n_days = 10
        n_rooms = 6
    else
        n_days = 10
        n_rooms = clamp(round(Int, sqrt(target / 40)), 8, 25)
    end

    rd = n_rooms * n_days
    # Solve (S + 1) * rd + 2S + D + 1 ≈ target => S * (rd + 2) + rd + D + 1 ≈ target
    n_specialties = max(2, round(Int, (target - rd - n_days - 1) / (rd + 2)))

    actual_status = feasibility_status

    specialty_names = ["Specialty_$s" for s in 1:n_specialties]

    # Room-specialty compatibility
    room_specialty_matrix = ones(Int, n_rooms, n_specialties)
    if n_specialties > 3 && n_rooms > 3
        for s in 2:n_specialties, r in 1:n_rooms
            if rand() < 0.35 && sum(room_specialty_matrix[:, s]) > 1
                room_specialty_matrix[r, s] = 0
            end
        end
    end
    for s in 1:n_specialties
        if sum(room_specialty_matrix[:, s]) == 0
            room_specialty_matrix[rand(1:n_rooms), s] = 1
        end
    end

    # Block targets and quotas
    total_slots = n_rooms * n_days
    target_blocks = Vector{Int}(undef, n_specialties)
    min_blocks = Vector{Int}(undef, n_specialties)
    max_blocks = Vector{Int}(undef, n_specialties)
    max_daily_rooms = Vector{Int}(undef, n_specialties)

    base_quota = max(1, div(total_slots, n_specialties))
    for s in 1:n_specialties
        t = max(1, base_quota + rand(-1:1))
        target_blocks[s] = t
        min_blocks[s] = max(1, t - rand(0:1))
        max_blocks[s] = t + rand(1:2)
        max_daily_rooms[s] = max(1, min(n_rooms, ceil(Int, n_rooms * 0.45) + rand(0:1)))
    end

    # Preference costs (0 = ideal, higher = unfavorable day/room for specialty)
    specialty_preference_costs = Array{Float64,3}(undef, n_specialties, n_rooms, n_days)
    for s in 1:n_specialties, r in 1:n_rooms, d in 1:n_days
        specialty_preference_costs[s, r, d] = rand(Uniform(0.0, 50.0))
    end

    # Room fixed opening costs
    room_fixed_costs = Matrix{Float64}(undef, n_rooms, n_days)
    for r in 1:n_rooms, d in 1:n_days
        room_fixed_costs[r, d] = rand(Uniform(500.0, 1200.0))
    end

    # Downstream bed occupancy profile: specialty s generates patients staying tau days
    bed_occupancy_profiles = Matrix{Float64}(undef, n_specialties, n_days)
    for s in 1:n_specialties
        avg_los = rand(Uniform(1.5, 4.5))
        for tau in 0:(n_days - 1)
            # Exponential decay or bell shape
            decay = exp(-tau / max(1.0, avg_los))
            bed_occupancy_profiles[s, tau + 1] = rand(Uniform(0.8, 1.2)) * decay * rand(Uniform(2.0, 5.0))
        end
    end

    # Penalties and weights
    under_allocation_penalty = [rand(Uniform(200.0, 500.0)) for _ in 1:n_specialties]
    over_allocation_penalty = [rand(Uniform(100.0, 300.0)) for _ in 1:n_specialties]
    peak_bed_weight = rand(Uniform(10.0, 30.0))

    # Ward bed capacity
    ward_bed_capacity = fill(0.0, n_days)

    if actual_status == feasible
        # Construct a feasible assignment witness
        assigned = zeros(Int, n_specialties, n_rooms, n_days)
        room_day_occ = zeros(Int, n_rooms, n_days)

        for s in 1:n_specialties
            needed = min_blocks[s]
            placed = 0
            for d in 1:n_days, r in 1:n_rooms
                placed < needed || break
                if room_specialty_matrix[r, s] == 1 && room_day_occ[r, d] == 0
                    if sum(assigned[s, :, d]) < max_daily_rooms[s]
                        assigned[s, r, d] = 1
                        room_day_occ[r, d] = 1
                        placed += 1
                    end
                end
            end
            if placed < needed
                # Ensure compatibility and place in first available
                for d in 1:n_days, r in 1:n_rooms
                    placed < needed || break
                    if room_day_occ[r, d] == 0
                        room_specialty_matrix[r, s] = 1
                        assigned[s, r, d] = 1
                        room_day_occ[r, d] = 1
                        placed += 1
                    end
                end
                min_blocks[s] = placed
                target_blocks[s] = max(placed, target_blocks[s])
                max_blocks[s] = max(target_blocks[s], max_blocks[s])
            end
        end

        # Calculate bed occupancy from witness schedule
        calc_bed_occ = zeros(Float64, n_days)
        for d in 1:n_days
            occ = 0.0
            for s in 1:n_specialties, r in 1:n_rooms, dp in 1:n_days
                if assigned[s, r, dp] == 1
                    tau = mod(d - dp, n_days)
                    occ += bed_occupancy_profiles[s, tau + 1]
                end
            end
            calc_bed_occ[d] = occ
        end

        # Set ward capacity with 50% headroom
        for d in 1:n_days
            ward_bed_capacity[d] = ceil(calc_bed_occ[d] * 1.5) + 20.0
        end

    elseif actual_status == infeasible
        # Deterministic contradiction:
        # Require specialty 1 to receive more blocks than total compatible room-days in the cycle.
        compat_room_days = sum(room_specialty_matrix[:, 1]) * n_days
        min_blocks[1] = compat_room_days + 5
        target_blocks[1] = min_blocks[1]
        max_blocks[1] = min_blocks[1] + 2
        ward_bed_capacity .= 10000.0
    else
        # Unknown status: natural capacity
        for d in 1:n_days
            ward_bed_capacity[d] = rand(Uniform(30.0, 80.0))
        end
    end

    return OperatingRoomMasterScheduleProblem(
        n_specialties,
        n_rooms,
        n_days,
        specialty_names,
        target_blocks,
        min_blocks,
        max_blocks,
        max_daily_rooms,
        room_specialty_matrix,
        specialty_preference_costs,
        room_fixed_costs,
        bed_occupancy_profiles,
        ward_bed_capacity,
        under_allocation_penalty,
        over_allocation_penalty,
        peak_bed_weight,
        actual_status,
    )
end

"""
    build_model(prob::OperatingRoomMasterScheduleProblem)

Build a JuMP model for Master Surgical Schedule block planning with bed leveling.
Deterministic implementation.
"""
function build_model(prob::OperatingRoomMasterScheduleProblem)
    model = Model()

    S = prob.n_specialties
    R = prob.n_rooms
    D = prob.n_days

    # Decision variables
    @variable(model, x[1:S, 1:R, 1:D], Bin)
    @variable(model, y[1:R, 1:D], Bin)
    @variable(model, under_blocks[1:S] >= 0)
    @variable(model, over_blocks[1:S] >= 0)
    @variable(model, daily_bed_occ[1:D] >= 0)
    @variable(model, peak_bed_occ >= 0)

    # Objective: Minimize peak bed occupancy + preference costs + quota deviation + fixed room costs
    @objective(
        model,
        Min,
        prob.peak_bed_weight * peak_bed_occ +
        sum(prob.specialty_preference_costs[s, r, d] * x[s, r, d] for s in 1:S, r in 1:R, d in 1:D) +
        sum(prob.under_allocation_penalty[s] * under_blocks[s] + prob.over_allocation_penalty[s] * over_blocks[s] for s in 1:S) +
        sum(prob.room_fixed_costs[r, d] * y[r, d] for r in 1:R, d in 1:D)
    )

    # 1. At most one specialty per room per day block
    for r in 1:R, d in 1:D
        @constraint(model, sum(x[s, r, d] for s in 1:S) == y[r, d])
    end

    # 2. Room-specialty compatibility
    for s in 1:S, r in 1:R, d in 1:D
        if prob.room_specialty_matrix[r, s] == 0
            @constraint(model, x[s, r, d] == 0)
        end
    end

    # 3. Specialty quota target balance and min/max bounds
    for s in 1:S
        @constraint(model, sum(x[s, r, d] for r in 1:R, d in 1:D) + under_blocks[s] - over_blocks[s] == prob.target_blocks[s])
        @constraint(model, sum(x[s, r, d] for r in 1:R, d in 1:D) >= prob.min_blocks[s])
        @constraint(model, sum(x[s, r, d] for r in 1:R, d in 1:D) <= prob.max_blocks[s])
    end

    # 4. Daily room ceiling per specialty
    for s in 1:S, d in 1:D
        @constraint(model, sum(x[s, r, d] for r in 1:R) <= prob.max_daily_rooms[s])
    end

    # 5. Cyclical downstream ward bed occupancy tracking
    for d in 1:D
        @constraint(
            model,
            daily_bed_occ[d] == sum(
                prob.bed_occupancy_profiles[s, mod(d - dp, D) + 1] * x[s, r, dp]
                for s in 1:S, r in 1:R, dp in 1:D
            )
        )
        @constraint(model, peak_bed_occ >= daily_bed_occ[d])
        @constraint(model, daily_bed_occ[d] <= prob.ward_bed_capacity[d])
    end

    return model
end

# Register the master surgical schedule variant
register_variant(
    :operating_room_scheduling,
    :master_surgical_schedule,
    OperatingRoomMasterScheduleProblem,
    "Master surgical schedule (MSS) cyclical block planning with downstream bed leveling, quota targets, and room-specialty affinities",
)
