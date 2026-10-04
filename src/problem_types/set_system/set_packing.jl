using JuMP
using Random

"""
    SetPackingWitness

A planted conflict-free timetable: the columns `paths` are pairwise disjoint
(no two share a track-slot cell or a train), and they include one path of every
mandatory train.
"""
struct SetPackingWitness
    paths::Vector{Int}
end

"""
    BottleneckCertificate

Peak-hour capacity deficit at a single-track bottleneck. Every candidate path
of each mandatory train in `trains` occupies at least `occupancy[k]` of the
bottleneck cells `cells` (one per time slot of the peak window). Weighting each
mandatory train's equality row by `occupancy[k]` and summing the packing rows
of `cells` gives `sum(occupancy) <= length(cells)`; the generator makes
`sum(occupancy) > length(cells)`, so even the LP relaxation is infeasible. The
argument aggregates every train row and every slot row of the window, so
presolve does not detect it.
"""
struct BottleneckCertificate
    trains::Vector{Int}
    occupancy::Vector{Int}
    cells::Vector{Int}
end

"""
    SetPackingProblem <: ProblemGenerator

Maximum-value set packing as railway timetabling / train-path allocation on a
corridor (Caprara–Fischetti–Toth style): each train request has candidate
paths (departure shifts around its requested time); a path occupies
space-time cells — (section, slot) on single-track sections, (section, slot,
direction) on double-track sections — for its running time plus a headway, and
no two selected paths may share a cell. At most one path runs per train, and
mandatory (public-service) trains must run.

# Formulation

    max  sum_j value_j x_j
    s.t. sum_{j ∋ c} x_j <= 1          for every space-time cell c used by a path
         sum_{j ∈ paths(r)} x_j <= 1   for every optional train r
         sum_{j ∈ paths(r)} x_j == 1   for every mandatory train r
         x binary

All coefficients are 0/1: a set packing over cells and train rows with
partitioning rows for the mandatory trains. Variables: exactly
`target_variables` candidate paths.

# Feasibility

  - `feasible`: a greedy conflict-free timetable (mandatory trains first) is the
    `feasible_witness`; mandatory trains are passenger trains it scheduled.
  - `infeasible`: a group of mandatory passenger trains all must cross the
    single-track bottleneck inside one peak window whose slots cannot hold
    them ([`BottleneckCertificate`](@ref)).
  - `unknown`: a random 15–60% of the passenger trains are mandatory, with
    no adjustment; whether the corridor can carry them depends on congestion.
"""
struct SetPackingProblem <: ProblemGenerator
    n_elements::Int
    columns::Vector{Vector{Int}}
    values::Vector{Float64}
    train_of::Vector{Int}
    train_rows::Vector{Int}
    mandatory::Vector{Bool}
    feasible_witness::Union{Nothing, SetPackingWitness}
    infeasibility_certificate::Union{Nothing, BottleneckCertificate}
end

function SetPackingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    target_variables >= 2 || throw(ArgumentError("set packing needs at least 2 variables"))
    rng = MersenneTwister(seed)

    # --- Corridor ---
    n_sections = clamp(round(Int, 4 + 2 * log(target_variables)), 5, 40)
    single = [rand(rng) < 0.4 for _ in 1:n_sections]
    bottleneck = cld(n_sections, 2)
    single[bottleneck] = true

    # --- Train requests: route, speed, flexibility; candidates fill the target ---
    speed_slots = (1, 2, 3)               # passenger, regional, freight
    priority = (4.0, 2.5, 1.2)
    kinds = Int[]
    dirs = Int[]
    routes = UnitRange{Int}[]
    flex = Int[]
    n_candidates = Int[]
    total = 0
    while total < target_variables
        kind = rand(rng) < 0.45 ? 1 : (rand(rng) < 0.45 ? 2 : 3)
        len = rand(rng, 2:min(n_sections, 8))
        # Passenger trains favour long through-routes across the bottleneck.
        first_section = if kind == 1 && rand(rng) < 0.7
            clamp(bottleneck - rand(rng, 0:(len - 1)), 1, n_sections - len + 1)
        else
            rand(rng, 1:(n_sections - len + 1))
        end
        # Tiny infeasible requests use one path per train so that at least
        # two trains exist to over-subscribe the bottleneck.
        w = (feasibility_status == infeasible && target_variables < 20) ? 0 : rand(rng, 2:5)
        count = min(2w + 1, target_variables - total)
        push!(kinds, kind)
        push!(dirs, rand(rng, (-1, 1)))
        push!(routes, first_section:(first_section + len - 1))
        push!(flex, w)
        push!(n_candidates, count)
        total += count
    end
    n_trains = length(kinds)
    deltas = [sort(collect((-flex[r]):flex[r]); by=abs)[1:n_candidates[r]] for r in 1:n_trains]
    headway = 1
    occ(r) = speed_slots[kinds[r]] + headway          # cells per section crossed
    duration(r) = length(routes[r]) * speed_slots[kinds[r]] + headway
    # Offset (slots after departure) at which train r enters section s.
    entry_offset(r, s) =
        (dirs[r] == 1 ? s - first(routes[r]) : last(routes[r]) - s) * speed_slots[kinds[r]]

    # --- Horizon from a target utilisation of the track capacity ---
    utilisation = 0.55 + 0.35 * rand(rng)
    demand = sum(length(routes[r]) * occ(r) for r in 1:n_trains)
    lanes = sum(single[s] ? 1 : 2 for s in 1:n_sections)
    max_dur = maximum(duration(r) for r in 1:n_trains)
    window = 2 * maximum(flex) + maximum(occ(r) for r in 1:n_trains) + 6
    n_slots = max(round(Int, demand / (lanes * utilisation)), window + 2 * (max_dur + 6) + 2)

    # Requested departures: passenger peaks in the morning/evening shoulders.
    depart = zeros(Int, n_trains)
    for r in 1:n_trains
        lo, hi = 1 + flex[r], n_slots - duration(r) - flex[r] + 1
        if kinds[r] == 1 && rand(rng) < 0.6
            peak = rand(rng) < 0.5 ? 0.3 : 0.7
            depart[r] = clamp(round(Int, (peak + 0.08 * randn(rng)) * n_slots), lo, hi)
        else
            depart[r] = rand(rng, lo:hi)
        end
    end

    # --- Infeasible: cram mandatory trains into one bottleneck window ---
    # Group trains crossing the bottleneck until their bottleneck occupancy
    # exceeds the smallest window that contains every candidate of every group
    # member; re-route further trains across the bottleneck if needed.
    mandatory = falses(n_trains)
    certificate = nothing
    spread(r) = maximum(deltas[r]) - minimum(deltas[r]) + occ(r)
    window_start = n_slots ÷ 2 - window ÷ 2
    if feasibility_status == infeasible
        order = shuffle(rng, collect(1:n_trains))
        sort!(order; by=r -> bottleneck in routes[r] ? 0 : 1)   # crossing trains first
        group = Int[]
        load = 0
        need = 0
        for r in order
            if !(bottleneck in routes[r])
                len = length(routes[r])
                f = clamp(bottleneck - rand(rng, 0:(len - 1)), 1, n_sections - len + 1)
                routes[r] = f:(f + len - 1)
            end
            push!(group, r)
            load += occ(r)
            need = max(need, spread(r))
            load >= need + max(1, ceil(Int, 0.1 * need)) && length(group) >= 2 && break
        end
        window = need
        for r in group
            mandatory[r] = true
            # Every candidate's bottleneck interval stays inside the window.
            lo = window_start - entry_offset(r, bottleneck) - minimum(deltas[r])
            hi = window_start + window - occ(r) - entry_offset(r, bottleneck) - maximum(deltas[r])
            depart[r] = rand(rng, lo:max(lo, hi))
        end
    elseif feasibility_status == unknown
        share = 0.15 + 0.45 * rand(rng)
        for r in 1:n_trains
            kinds[r] == 1 && rand(rng) < share && (mandatory[r] = true)
        end
    end

    # --- Cells and candidate paths ---
    section_base = zeros(Int, n_sections)
    cursor = 0
    for s in 1:n_sections
        section_base[s] = cursor
        cursor += (single[s] ? 1 : 2) * n_slots
    end
    n_cells = cursor
    cell(s, t, dir) = section_base[s] + (single[s] || dir == 1 ? 0 : n_slots) + t

    columns = Vector{Vector{Int}}()
    values = Float64[]
    train_of = Int[]
    shifts = Int[]
    for r in 1:n_trains
        for delta in deltas[r]
            start = depart[r] + delta
            cells = Int[]
            for s in routes[r]
                t_in = start + entry_offset(r, s)
                for t in t_in:(t_in + occ(r) - 1)
                    1 <= t <= n_slots && push!(cells, cell(s, t, dirs[r]))
                end
            end
            push!(columns, sort!(cells))
            push!(train_of, r)
            push!(shifts, delta)
            base = priority[kinds[r]] * length(routes[r]) * 10.0
            push!(values, round(base * (1 - 0.06 * abs(delta)) * (0.9 + 0.2 * rand(rng)); digits=2))
        end
    end
    train_rows = collect((n_cells + 1):(n_cells + n_trains))
    for (j, r) in enumerate(train_of)
        push!(columns[j], train_rows[r])
    end

    if feasibility_status == infeasible
        group = findall(mandatory)
        window_cells = [cell(bottleneck, t, 1) for t in window_start:(window_start + window - 1)]
        in_window = Set(window_cells)
        occupancy = [
            minimum(
                count(in(in_window), columns[j]) for j in eachindex(train_of) if train_of[j] == r
            ) for r in group
        ]
        certificate = BottleneckCertificate(group, occupancy, window_cells)
    end

    # --- Feasible: greedy conflict-free timetable, mandatory = scheduled passengers ---
    witness = nothing
    if feasibility_status == feasible
        used = falses(n_cells + n_trains)
        chosen = Int[]
        first_column = cumsum([1; n_candidates[1:(end - 1)]])
        order = sortperm([(kinds[r] == 1 ? 0 : 1, rand(rng)) for r in 1:n_trains])
        for r in order
            for j in first_column[r]:(first_column[r] + n_candidates[r] - 1)
                if !any(used[c] for c in columns[j])
                    used[columns[j]] .= true
                    push!(chosen, j)
                    kinds[r] == 1 && (mandatory[r] = true)
                    break
                end
            end
        end
        witness = SetPackingWitness(sort!(chosen))
    end

    return SetPackingProblem(
        n_cells + n_trains,
        columns,
        values,
        train_of,
        train_rows,
        collect(mandatory),
        witness,
        certificate,
    )
end

function build_model(prob::SetPackingProblem)
    model = Model()
    n_columns = length(prob.columns)
    incidence = _set_elements_to_columns(prob.columns, prob.n_elements)
    train_row = Dict(e => r for (r, e) in enumerate(prob.train_rows))
    @variable(model, x[1:n_columns], Bin)
    @objective(model, Max, sum(prob.values[j] * x[j] for j in 1:n_columns))
    for i in 1:prob.n_elements
        isempty(incidence[i]) && continue
        r = get(train_row, i, 0)
        if r > 0 && prob.mandatory[r]
            @constraint(model, sum(x[j] for j in incidence[i]) == 1)
        elseif length(incidence[i]) >= 2
            # A cell used by a single path is just the bound x <= 1.
            @constraint(model, sum(x[j] for j in incidence[i]) <= 1)
        end
    end
    return model
end

register_variant(
    :set_system,
    :set_packing,
    SetPackingProblem,
    "Set packing as railway train-path allocation over space-time track cells with mandatory services";
    tags=[:scheduling, :packing, :time_indexed],
    min_target_variables=2,
)
