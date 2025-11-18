using JuMP
using Random
using Distributions

"""
    ORScheduling <: ProblemGenerator

Generator for Operating Room (OR) scheduling optimization problems.

This problem models realistic OR scheduling with surgery-specific constraints:
- Multiple operating rooms with different capabilities
- Scheduled surgeries with estimated durations
- Surgeon and surgical team availability
- Equipment and room type requirements
- Block scheduling (dedicated OR time blocks)
- Turnover time between surgeries
- Overtime costs for extending beyond regular hours

# Fields
All data generated in constructor based on target_variables and feasibility_status:
- `n_surgeries::Int`: Number of surgeries to schedule
- `n_rooms::Int`: Number of operating rooms
- `n_time_blocks::Int`: Number of time blocks per room (e.g., 30-min blocks)
- `total_slots::Int`: Total scheduling slots (n_rooms × n_time_blocks)
- `surgery_durations::Vector{Int}`: Duration of each surgery in time blocks
- `surgery_types::Vector{Symbol}`: Type of each surgery (e.g., :cardiac, :orthopedic, :general)
- `room_capabilities::Matrix{Int}`: Room capability matrix (1 if room can handle surgery type)
- `surgeon_availability::Matrix{Int}`: Surgeon availability per time block
- `room_costs::Matrix{Float64}`: Cost per room per time block
- `turnover_time::Int`: Required time blocks between surgeries in same room
- `regular_hours_end::Int`: Time block where regular hours end (after = overtime)
- `overtime_multiplier::Float64`: Cost multiplier for overtime blocks
"""
struct ORScheduling <: ProblemGenerator
    n_surgeries::Int
    n_rooms::Int
    n_time_blocks::Int
    total_slots::Int
    surgery_durations::Vector{Int}
    surgery_types::Vector{Symbol}
    room_capabilities::Matrix{Int}
    surgeon_availability::Matrix{Int}
    room_costs::Matrix{Float64}
    turnover_time::Int
    regular_hours_end::Int
    overtime_multiplier::Float64
end

"""
    ORScheduling(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct an OR scheduling problem instance.

# Arguments
- `target_variables`: Target number of variables (n_surgeries × n_rooms × n_time_blocks)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function ORScheduling(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Determine problem dimensions
    # Typical: 5-10 ORs, 16-20 blocks (30-min each = 8-10 hours), 10-30 surgeries
    if target_variables <= 500
        # Small: outpatient surgery center
        n_rooms = round(Int, rand(Uniform(2, 4)))
        n_time_blocks = round(Int, rand(Uniform(12, 16)))  # 6-8 hours
        n_surgeries = round(Int, rand(Uniform(5, 12)))
    elseif target_variables <= 2000
        # Medium: hospital surgical department
        n_rooms = round(Int, rand(Uniform(4, 8)))
        n_time_blocks = round(Int, rand(Uniform(16, 20)))  # 8-10 hours
        n_surgeries = round(Int, rand(Uniform(12, 25)))
    else
        # Large: major medical center
        n_rooms = round(Int, rand(Uniform(8, 15)))
        n_time_blocks = round(Int, rand(Uniform(18, 24)))  # 9-12 hours
        n_surgeries = round(Int, rand(Uniform(25, 50)))
    end

    total_slots = n_rooms * n_time_blocks

    # Define surgery types (realistic distribution)
    surgery_type_options = [:general, :orthopedic, :cardiac, :neurosurgery, :vascular, :thoracic]
    surgery_types = Vector{Symbol}(undef, n_surgeries)
    for i in 1:n_surgeries
        # General and orthopedic are more common
        r = rand()
        if r < 0.35
            surgery_types[i] = :general
        elseif r < 0.60
            surgery_types[i] = :orthopedic
        elseif r < 0.75
            surgery_types[i] = :cardiac
        elseif r < 0.85
            surgery_types[i] = :neurosurgery
        elseif r < 0.93
            surgery_types[i] = :vascular
        else
            surgery_types[i] = :thoracic
        end
    end

    # Generate surgery durations (in time blocks)
    # Different surgery types have different typical durations
    duration_ranges = Dict(
        :general => (2, 6),        # 1-3 hours
        :orthopedic => (3, 8),     # 1.5-4 hours
        :cardiac => (6, 12),       # 3-6 hours
        :neurosurgery => (8, 16),  # 4-8 hours
        :vascular => (4, 10),      # 2-5 hours
        :thoracic => (6, 14)       # 3-7 hours
    )

    surgery_durations = Vector{Int}(undef, n_surgeries)
    for i in 1:n_surgeries
        min_dur, max_dur = duration_ranges[surgery_types[i]]
        surgery_durations[i] = round(Int, rand(Uniform(min_dur, max_dur)))
    end

    # Room capabilities (some ORs are specialized)
    room_capabilities = zeros(Int, n_rooms, length(surgery_type_options))
    type_to_idx = Dict(t => i for (i, t) in enumerate(surgery_type_options))

    for r in 1:n_rooms
        # Decide if this is a specialized or general OR
        is_specialized = rand() < 0.3

        if is_specialized
            # Specialized room: good at 1-2 types
            n_specialties = rand([1, 2])
            specialty_types = sample(surgery_type_options, n_specialties, replace=false)
            for st in specialty_types
                room_capabilities[r, type_to_idx[st]] = 1
            end
            # Can handle general surgeries too
            room_capabilities[r, type_to_idx[:general]] = 1
        else
            # General OR: can handle most types except highly specialized
            for (idx, st) in enumerate(surgery_type_options)
                if st in [:general, :orthopedic, :vascular]
                    room_capabilities[r, idx] = 1
                elseif rand() < 0.5  # 50% chance for other types
                    room_capabilities[r, idx] = 1
                end
            end
        end
    end

    # Surgeon/team availability (some surgeries can only happen at certain times)
    surgeon_availability = ones(Int, n_surgeries, n_time_blocks)

    for i in 1:n_surgeries
        # Some surgeons have block times or limited availability
        if rand() < 0.3
            # Restricted availability (e.g., block scheduling)
            block_start = rand(1:(n_time_blocks - surgery_durations[i]))
            block_end = min(n_time_blocks, block_start + rand(4:10))

            for t in 1:n_time_blocks
                if t < block_start || t > block_end
                    surgeon_availability[i, t] = 0
                end
            end
        end
    end

    # Room costs (varies by room type and time)
    room_costs = zeros(n_rooms, n_time_blocks)
    base_or_cost = 1000.0  # Base cost per 30-min block

    for r in 1:n_rooms
        # Specialized rooms cost more
        n_capabilities = sum(room_capabilities[r, :])
        specialization_factor = n_capabilities <= 2 ? 1.3 : 1.0

        for t in 1:n_time_blocks
            cost = base_or_cost * specialization_factor
            cost *= rand(Uniform(0.95, 1.05))  # Random variation
            room_costs[r, t] = cost
        end
    end

    # Turnover time between surgeries (cleaning, setup)
    turnover_time = round(Int, rand(Uniform(1, 3)))  # 30-90 minutes

    # Regular hours (after which overtime applies)
    regular_hours_end = round(Int, n_time_blocks * rand(Uniform(0.7, 0.85)))

    # Overtime multiplier
    overtime_multiplier = rand(Uniform(1.4, 1.6))

    # Apply overtime costs
    for r in 1:n_rooms, t in (regular_hours_end+1):n_time_blocks
        room_costs[r, t] *= overtime_multiplier
    end

    # Feasibility enforcement
    if feasibility_status == feasible
        # Ensure all surgeries can be assigned to at least one room
        for i in 1:n_surgeries
            surgery_type = surgery_types[i]
            type_idx = type_to_idx[surgery_type]

            # Find compatible rooms
            compatible_rooms = findall(r -> room_capabilities[r, type_idx] == 1, 1:n_rooms)

            if isempty(compatible_rooms)
                # Make a random room compatible
                random_room = rand(1:n_rooms)
                room_capabilities[random_room, type_idx] = 1
            end
        end

        # Ensure total capacity is sufficient
        total_surgery_duration = sum(surgery_durations) + n_surgeries * turnover_time
        total_capacity = n_rooms * n_time_blocks

        if total_surgery_duration > total_capacity * 0.8
            # Scale down surgery durations
            scale_factor = (total_capacity * 0.8) / total_surgery_duration
            surgery_durations = max.(1, round.(Int, surgery_durations .* scale_factor))
        end

    elseif feasibility_status == infeasible
        # Create infeasibility
        mode = rand([:room_conflict, :time_conflict, :capacity_shortage])

        if mode == :room_conflict
            # Make a surgery incompatible with all rooms
            surgery_idx = rand(1:n_surgeries)
            surgery_type = surgery_types[surgery_idx]
            type_idx = type_to_idx[surgery_type]

            for r in 1:n_rooms
                room_capabilities[r, type_idx] = 0
            end

        elseif mode == :time_conflict
            # Make a surgery available only when surgeon is not
            surgery_idx = rand(1:n_surgeries)
            surgeon_availability[surgery_idx, :] .= 0

        else  # capacity_shortage
            # Make total duration exceed capacity
            total_capacity = n_rooms * n_time_blocks
            current_duration = sum(surgery_durations)

            shortage = round(Int, total_capacity * 0.3)
            added_per_surgery = div(shortage, n_surgeries)

            for i in 1:n_surgeries
                surgery_durations[i] += added_per_surgery
            end
        end
    end

    return ORScheduling(
        n_surgeries,
        n_rooms,
        n_time_blocks,
        total_slots,
        surgery_durations,
        surgery_types,
        room_capabilities,
        surgeon_availability,
        room_costs,
        turnover_time,
        regular_hours_end,
        overtime_multiplier
    )
end

"""
    build_model(prob::ORScheduling)

Build a JuMP model for the OR scheduling problem (deterministic).

# Arguments
- `prob`: ORScheduling instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::ORScheduling)
    model = Model()

    # Decision variable: x[i,r,t] = 1 if surgery i starts in room r at time block t
    @variable(model, x[1:prob.n_surgeries, 1:prob.n_rooms, 1:prob.n_time_blocks], Bin)

    # Objective: minimize total OR cost
    @objective(model, Min,
        sum(prob.room_costs[r, t] * prob.surgery_durations[i] * x[i, r, t]
            for i in 1:prob.n_surgeries, r in 1:prob.n_rooms, t in 1:prob.n_time_blocks
            if t + prob.surgery_durations[i] - 1 <= prob.n_time_blocks))

    # Each surgery must be scheduled exactly once
    for i in 1:prob.n_surgeries
        @constraint(model,
            sum(x[i, r, t]
                for r in 1:prob.n_rooms, t in 1:(prob.n_time_blocks - prob.surgery_durations[i] + 1))
            == 1)
    end

    # Room capability constraints
    surgery_type_options = [:general, :orthopedic, :cardiac, :neurosurgery, :vascular, :thoracic]
    type_to_idx = Dict(t => idx for (idx, t) in enumerate(surgery_type_options))

    for i in 1:prob.n_surgeries
        surgery_type_idx = type_to_idx[prob.surgery_types[i]]
        for r in 1:prob.n_rooms
            if prob.room_capabilities[r, surgery_type_idx] == 0
                for t in 1:prob.n_time_blocks
                    @constraint(model, x[i, r, t] == 0)
                end
            end
        end
    end

    # Surgeon availability constraints
    for i in 1:prob.n_surgeries, r in 1:prob.n_rooms, t in 1:prob.n_time_blocks
        if t + prob.surgery_durations[i] - 1 <= prob.n_time_blocks
            # Check if surgeon is available for entire surgery duration
            for τ in t:(t + prob.surgery_durations[i] - 1)
                if prob.surgeon_availability[i, τ] == 0
                    @constraint(model, x[i, r, t] == 0)
                    break
                end
            end
        else
            @constraint(model, x[i, r, t] == 0)
        end
    end

    # Room conflict constraints (no overlapping surgeries in same room)
    for r in 1:prob.n_rooms, t in 1:prob.n_time_blocks
        # Find all surgeries that would occupy room r at time t
        overlapping = []
        for i in 1:prob.n_surgeries
            # Surgery i starting at time s would occupy time t if:
            # s <= t < s + duration[i] + turnover
            for s in max(1, t - prob.surgery_durations[i] - prob.turnover_time + 1):min(prob.n_time_blocks, t)
                if s >= 1 && s + prob.surgery_durations[i] + prob.turnover_time - 1 >= t &&
                   s + prob.surgery_durations[i] - 1 <= prob.n_time_blocks
                    push!(overlapping, (i, s))
                end
            end
        end

        # At most one surgery can occupy this room at this time
        if !isempty(overlapping)
            @constraint(model, sum(x[i, r, s] for (i, s) in unique(overlapping)) <= 1)
        end
    end

    return model
end

# Register the problem variant
register_problem(
    :scheduling_or,
    ORScheduling,
    "Operating room scheduling problem with surgery types, room capabilities, and block scheduling constraints"
)
