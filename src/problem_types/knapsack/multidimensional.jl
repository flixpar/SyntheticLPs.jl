using JuMP
using Random

"""
    MultidimensionalSelectionWitness

Planted selection for a `feasible` instance: `selected` (sorted) lists the
items at `x_i = 1`. Every capacity is its usage times a factor `>= 1.02` (plus a
small allowance for local rows), and every program floor is at most the
planted count, so the 0/1 point is feasible with slack.
"""
struct MultidimensionalSelectionWitness
    selected::Vector{Int}
end

"""
    ProgramFloorCertificate

Relaxation-valid infeasibility proof spanning every program-floor row and one
global resource row. Program `p` must select at least `floors[p]` of its items;
for `0 <= x <= 1` the cheapest way to do so in global resource `resource` is
its `floors[p]` lightest items, so any fractional selection uses at least

    lightest_sum = sum_p (sum of the floors[p] smallest usages in program p)

of that resource, while `capacity * 1.04 <= lightest_sum`. The Farkas multipliers
are 1 on the resource row and the `floors[p]`-th smallest usage on each floor
row (plus upper-bound multipliers), so the argument needs all programs at once.
"""
struct ProgramFloorCertificate
    resource::Int
    floors::Vector{Int}
    lightest_sum::Float64
    capacity::Float64
end

"""
    MultidimensionalKnapsackProblem <: ProblemGenerator

Sparse multi-dimensional 0/1 knapsack (MDKP) with program commitments: select
jobs that consume a window of consecutive machine-week capacities plus three
shared budgets, subject to minimum delivery counts per program.

# Overview

`n_local = clamp(round(n / 12), 2, n)` local resources (machine-weeks arranged
on a ring) and 3 global resources (capital, labour, energy). Item `i` has a
latent size, occupies a window of `2-6` consecutive local resources, and
consumes every global resource; usages are size times a resource intensity
times lognormal noise, so resource columns are correlated. Values are
correlated with total usage but noisy. Items belong to
`n_programs = clamp(round(n / 40), 1, n)` programs (product lines), each with a
minimum number of selected items — covering rows mixed into the packing
structure, so the LP is not solved by a density greedy.

```text
max  sum_i v_i x_i
s.t. sum_i a_ri x_i <= C_r          for every local resource r (window rows)
     sum_i g_si x_i <= G_s          for the 3 global resources
     sum_{i in p} x_i >= f_p        for every program with f_p > 0
     x binary  (relaxed to [0, 1] by default)
```

Exactly `n = target_variables` columns; about `n / 12 + n / 40 + 3` rows and
`~7` nonzeros per column (the old version had 3-5 dense rows at any size).

# Feasibility

  - `feasible`: 35-50% of the items are planted; local capacities are the
    planted window usage times `U(1.03, 1.25)` plus a small allowance (capped
    below the row total so no row is redundant), global capacities planted
    usage times `U(1.02, 1.10)`, floors `U(0.5, 0.95)` of the planted count.
    Witness: `MultidimensionalSelectionWitness`.
  - `infeasible`: same instance, then floors are raised (round-robin over
    programs) until the lightest items meeting them need at least
    `U(1.04, 1.12)` times one global capacity. Certificate:
    `ProgramFloorCertificate`.
  - `unknown`: capacities are fractions of the row totals (local `U(0.35, 0.65)`,
    global `U(0.30, 0.55)`) and floors a per-instance fraction `U(0.35, 0.85)`
    of each program; the LP decides.
"""
struct MultidimensionalKnapsackProblem <: ProblemGenerator
    n_items::Int
    n_local::Int
    n_programs::Int
    values::Vector{Float64}
    window_start::Vector{Int}
    local_usage::Vector{Vector{Float64}}   # usage over the item's window
    global_usage::Matrix{Float64}          # 3 x n
    program::Vector{Int}
    local_capacity::Vector{Float64}
    global_capacity::Vector{Float64}
    program_floor::Vector{Int}
    feasibility_status::FeasibilityStatus
    feasible_witness::Union{Nothing, MultidimensionalSelectionWitness}
    infeasibility_certificate::Union{Nothing, ProgramFloorCertificate}
end

const MDKP_GLOBAL = 3

# Local resource index of position k (1-based) in item i's window.
_mdkp_resource(start::Int, k::Int, n_local::Int) = mod(start + k - 2, n_local) + 1

function _mdkp_local_totals(sel::AbstractVector{Bool}, window_start, local_usage, n_local::Int)
    tot = zeros(n_local)
    for i in eachindex(sel)
        sel[i] || continue
        for (k, a) in enumerate(local_usage[i])
            tot[_mdkp_resource(window_start[i], k, n_local)] += a
        end
    end
    return tot
end

function MultidimensionalKnapsackProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be positive (got $target_variables)"))
    rng = MersenneTwister(seed)
    n = target_variables
    n_local = clamp(round(Int, n / 12), 2, max(2, n))
    n_programs = clamp(round(Int, n / 40), 1, n)

    local_intensity = [0.7 + 0.6 * rand(rng) for _ in 1:n_local]
    global_intensity = [0.6 + 1.2 * rand(rng) for _ in 1:MDKP_GLOBAL]

    size = [20.0 * exp(0.5 * randn(rng)) for _ in 1:n]
    window_start = rand(rng, 1:n_local, n)
    local_usage = Vector{Vector{Float64}}(undef, n)
    global_usage = zeros(MDKP_GLOBAL, n)
    for i in 1:n
        w = min(rand(rng, 2:6), n_local)
        local_usage[i] = [
            size[i] *
            local_intensity[_mdkp_resource(window_start[i], k, n_local)] *
            exp(0.3 * randn(rng)) for k in 1:w
        ]
        for s in 1:MDKP_GLOBAL
            global_usage[s, i] = size[i] * global_intensity[s] * exp(0.3 * randn(rng))
        end
    end
    total_use = [
        sum(local_usage[i]) / length(local_usage[i]) + sum(global_usage[:, i]) for i in 1:n
    ]
    mean_use = sum(total_use) / n
    values = [
        max(1.0, 50.0 * (0.4 + 0.6 * (total_use[i] / mean_use) * (0.6 + 0.8 * rand(rng)))) for
        i in 1:n
    ]

    program = [mod(i - 1, n_programs) + 1 for i in 1:n]
    shuffle!(rng, program)
    members = [Int[] for _ in 1:n_programs]
    for i in 1:n
        push!(members[program[i]], i)
    end

    local_total = _mdkp_local_totals(trues(n), window_start, local_usage, n_local)
    global_total = vec(sum(global_usage; dims=2))

    feasible_witness = nothing
    infeasibility_certificate = nothing
    if feasibility_status == feasible || feasibility_status == infeasible
        frac = 0.35 + 0.15 * rand(rng)
        sel = BitVector([rand(rng) < frac for _ in 1:n])
        any(sel) || (sel[1] = true)
        planted_local = _mdkp_local_totals(sel, window_start, local_usage, n_local)
        allowance = 0.5 * mean_use
        local_capacity = [
            max(
                planted_local[r],
                min(planted_local[r] * (1.03 + 0.22 * rand(rng)) + allowance, 0.9 * local_total[r]),
            ) for r in 1:n_local
        ]
        planted_global = [sum(global_usage[s, i] for i in 1:n if sel[i]) for s in 1:MDKP_GLOBAL]
        global_capacity = planted_global .* (1.02 .+ 0.08 .* rand(rng, MDKP_GLOBAL))
        planted_count = [count(i -> sel[i], members[p]) for p in 1:n_programs]
        program_floor = [
            floor(Int, planted_count[p] * (0.5 + 0.45 * rand(rng))) for p in 1:n_programs
        ]
        if feasibility_status == feasible
            feasible_witness = MultidimensionalSelectionWitness(findall(sel))
        else
            s = rand(rng, 1:MDKP_GLOBAL)
            sorted_use = [sort([global_usage[s, i] for i in members[p]]) for p in 1:n_programs]
            prefix = [cumsum(u) for u in sorted_use]
            lightest(p) = program_floor[p] == 0 ? 0.0 : prefix[p][program_floor[p]]
            target = (1.04 + 0.08 * rand(rng)) * global_capacity[s]
            current = sum(lightest(p) for p in 1:n_programs)
            order = randperm(rng, n_programs)
            progressed = true
            while current < target && progressed
                progressed = false
                for p in order
                    current >= target && break
                    program_floor[p] < length(members[p]) || continue
                    current -= lightest(p)
                    program_floor[p] += 1
                    current += lightest(p)
                    progressed = true
                end
            end
            if current < target
                # Every item is mandated and still fits: cut the budget.
                global_capacity[s] = current / (1.04 + 0.08 * rand(rng))
            end
            infeasibility_certificate = ProgramFloorCertificate(
                s, copy(program_floor), current, global_capacity[s]
            )
        end
    else
        local_capacity = local_total .* (0.35 .+ 0.30 .* rand(rng, n_local))
        global_capacity = global_total .* (0.30 .+ 0.25 .* rand(rng, MDKP_GLOBAL))
        f = 0.35 + 0.50 * rand(rng)
        program_floor = [
            clamp(
                round(Int, length(members[p]) * (f + 0.05 * (2 * rand(rng) - 1))),
                0,
                length(members[p]),
            ) for p in 1:n_programs
        ]
    end

    return MultidimensionalKnapsackProblem(
        n,
        n_local,
        n_programs,
        values,
        window_start,
        local_usage,
        global_usage,
        program,
        local_capacity,
        global_capacity,
        program_floor,
        feasibility_status,
        feasible_witness,
        infeasibility_certificate,
    )
end

function build_model(prob::MultidimensionalKnapsackProblem)
    model = Model()
    n = prob.n_items
    @variable(model, x[1:n], Bin)
    @objective(model, Max, sum(prob.values[i] * x[i] for i in 1:n))

    loc = [AffExpr() for _ in 1:prob.n_local]
    glob = [AffExpr() for _ in 1:MDKP_GLOBAL]
    prog = [AffExpr() for _ in 1:prob.n_programs]
    for i in 1:n
        for (k, a) in enumerate(prob.local_usage[i])
            add_to_expression!(loc[_mdkp_resource(prob.window_start[i], k, prob.n_local)], a, x[i])
        end
        for s in 1:MDKP_GLOBAL
            add_to_expression!(glob[s], prob.global_usage[s, i], x[i])
        end
        add_to_expression!(prog[prob.program[i]], 1.0, x[i])
    end
    for r in 1:prob.n_local
        isempty(loc[r].terms) && continue
        @constraint(model, loc[r] <= prob.local_capacity[r])
    end
    @constraint(model, global_capacity[s in 1:MDKP_GLOBAL], glob[s] <= prob.global_capacity[s])
    for p in 1:prob.n_programs
        prob.program_floor[p] > 0 || continue
        @constraint(model, prog[p] >= prob.program_floor[p])
    end
    return model
end

register_variant(
    :knapsack,
    :multidimensional,
    MultidimensionalKnapsackProblem,
    "Sparse multi-dimensional 0/1 knapsack: machine-week window resources, shared budgets, and program minimum-selection commitments";
    tags=[:production, :packing],
)
