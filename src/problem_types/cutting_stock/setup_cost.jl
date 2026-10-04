using JuMP
using Random

"""
    SetupPlanWitness

Single-item plan for a `feasible` setup-cost instance: item `i` is cut on
column pair `pair[i]` (its maximal single-item pattern on one eligible
machine) `runs[i] = cld(demand[i], yield)` times with that pair's setup
switched on. Demands, stock availabilities, machine minutes and the
`x <= M * y` links all hold in exact arithmetic.
"""
struct SetupPlanWitness
    pair::Vector{Int}
    runs::Vector{Int}
end

"""
    SetupCostCuttingStockProblem <: ProblemGenerator

Multi-machine cutting stock with pattern setups: every pattern can run on 1-3
of several parallel saws; running a (pattern, machine) pair needs a setup
(knife positioning) that costs money and machine minutes.

# Overview

Patterns come from the shared generator (`cs_generate_patterns`) over a
multi-stock bar catalogue and a large order book. Column pair `q = (j, m)` has
a continuous run count `x_q` and a binary setup `y_q`:

```text
min  sum_q cost[stock[j]] x_q + sum_q setup_cost_q y_q
s.t. sum_q a_{i,j(q)} x_q >= d_i                      for every item
     sum_{q on stock k} x_q <= S_k                     for every stock type
     sum_{q on m} (run_q x_q + setup_q y_q) <= H_m     for every machine (minutes)
     x_q <= M_q y_q                                    for every pair
     x >= 0, y binary
```

Run time grows with the number of cuts, setup time with the number of
distinct lengths in the pattern (more knife moves), and both differ by
machine. `M_q = min(max_i ceil(d_i / a_ij), floor(H_m / run_q))` is the
tightest valid link (never run a pattern beyond its largest order or beyond the
machine's shift). Because `y_q` also sits in its machine's time row, it is not
a column singleton: under relaxation a setup still consumes `setup_q / M_q`
minutes and costs `setup_cost_q / M_q` per bar, so machine capacity and setup
economics shape the LP (the old single-machine big-M variant collapsed to 50%
of its columns and 2% of its rows in presolve).

Sizing: `Q = target_variables ÷ 2` pairs (columns `2Q`), patterns
`ceil(Q / 2)`, machines `clamp(round(Q / 400), 2, 250)`, items
`clamp(round(Q / 12), 1, ...)`. Rows: `Q + n_types + n_stock + n_machines`.

# Feasibility

  - `feasible`: single-item plan (`SetupPlanWitness`); stock availability
    `U(1.05, 1.35)` and machine minutes `U(1.02, 1.15)` times the plan's usage.
  - `infeasible`: stock availabilities scaled so ordered material exceeds the
    total stock length by `U(8%, 20%)` (`MaterialShortageCertificate`).
  - `unknown`: stock length `U(0.97, 1.10)` times the ordered material, machine
    minutes as for `feasible`; the LP decides.
"""
struct SetupCostCuttingStockProblem <: ProblemGenerator
    stock_lengths::Vector{Int}
    stock_costs::Vector{Float64}
    piece_lengths::Vector{Int}
    demands::Vector{Int}
    patterns::CSPatterns
    n_machines::Int
    pair_pattern::Vector{Int}
    pair_machine::Vector{Int}
    run_minutes::Vector{Float64}
    setup_minutes::Vector{Float64}
    setup_cost::Vector{Float64}
    link_bound::Vector{Float64}
    machine_minutes::Vector{Float64}
    availability::Vector{Int}
    feasibility_status::FeasibilityStatus
    feasible_witness::Union{Nothing, SetupPlanWitness}
    infeasibility_certificate::Union{Nothing, MaterialShortageCertificate}
end

function cs_setup_dimensions(n::Int)
    Q = max(1, n ÷ 2)
    P = max(1, cld(Q, 2))
    n_machines = clamp(round(Int, Q / 400), 2, 250)
    n_stock = clamp(round(Int, log10(max(n, 1))) - 1, 1, 4)
    n_types = clamp(max(round(Int, Q / 12), min(6, P ÷ (2 * n_stock))), 1, max(1, P ÷ n_stock))
    n_stock = clamp(n_stock, 1, max(1, P ÷ n_types))
    return Q, P, n_machines, n_stock, n_types
end

function SetupCostCuttingStockProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be positive (got $target_variables)"))
    rng = MersenneTwister(seed)
    Q, P, n_machines, n_stock, n_types = cs_setup_dimensions(target_variables)

    stock_lengths, stock_costs = cs_stock_types(rng, n_stock)
    piece_lengths = cs_piece_lengths(rng, n_types, floor(Int, 0.45 * stock_lengths[end]))
    demands = cs_demands(rng, n_types)
    patterns = cs_generate_patterns(rng, stock_lengths, piece_lengths, P)
    single = cs_single_index(patterns, stock_lengths, piece_lengths)

    # Pairs: every pattern on one random machine, then extra machines for
    # random patterns until Q pairs exist (at most n_machines per pattern).
    pair_pattern = Int[]
    pair_machine = Int[]
    machines_of = [Int[] for _ in 1:P]
    for j in 1:P
        m = rand(rng, 1:n_machines)
        push!(machines_of[j], m)
    end
    extra = Q - P
    while extra > 0
        j = rand(rng, 1:P)
        length(machines_of[j]) < min(3, n_machines) || continue
        m = rand(rng, 1:n_machines)
        m in machines_of[j] && continue
        push!(machines_of[j], m)
        extra -= 1
    end
    for j in 1:P, m in sort!(machines_of[j])
        push!(pair_pattern, j)
        push!(pair_machine, m)
    end

    # Machine characteristics (minutes).
    speed = [0.7 + 0.6 * rand(rng) for _ in 1:n_machines]
    setup_base = [8.0 + 12.0 * rand(rng) for _ in 1:n_machines]
    labour_rate = 0.6 + 0.4 * rand(rng)     # cost units per setup minute
    npairs = length(pair_pattern)
    run_minutes = zeros(npairs)
    setup_minutes = zeros(npairs)
    setup_cost = zeros(npairs)
    for q in 1:npairs
        j, m = pair_pattern[q], pair_machine[q]
        pieces = sum(patterns.counts[j])
        run_minutes[q] = speed[m] * (1.5 + 0.4 * pieces)
        setup_minutes[q] = setup_base[m] * (1.0 + 0.15 * length(patterns.items[j]))
        setup_cost[q] = labour_rate * setup_minutes[q] + 0.3 * stock_costs[patterns.stock[j]]
    end

    # Single-item plan on the first pair of each single pattern.
    first_pair = zeros(Int, P)
    for q in npairs:-1:1
        first_pair[pair_pattern[q]] = q
    end
    plan_pair = zeros(Int, n_types)
    runs = zeros(Int, n_types)
    per_stock = zeros(Int, n_stock)
    plan_minutes = zeros(n_machines)
    for i in 1:n_types
        fits = [k for k in 1:n_stock if single[i, k] > 0]
        k = fits[rand(rng, 1:length(fits))]
        q = first_pair[single[i, k]]
        plan_pair[i] = q
        runs[i] = cld(demands[i], stock_lengths[k] ÷ piece_lengths[i])
        per_stock[k] += runs[i]
        plan_minutes[pair_machine[q]] += run_minutes[q] * runs[i] + setup_minutes[q]
    end
    shift = 480.0 * (1 + round(Int, sum(plan_minutes) / (480.0 * n_machines)))
    machine_minutes = [
        plan_minutes[m] > 0 ? plan_minutes[m] * (1.02 + 0.13 * rand(rng)) : shift * (0.5 + 0.5 * rand(rng)) for m in 1:n_machines
    ]

    # Tightest valid links.
    link_bound = zeros(npairs)
    for q in 1:npairs
        j, m = pair_pattern[q], pair_machine[q]
        dem = maximum(cld(demands[i], c) for (i, c) in zip(patterns.items[j], patterns.counts[j]))
        link_bound[q] = max(1.0, min(Float64(dem), floor(machine_minutes[m] / run_minutes[q])))
    end

    material = sum(Float64(piece_lengths[i]) * demands[i] for i in 1:n_types)
    feasible_witness = nothing
    infeasibility_certificate = nothing
    if feasibility_status == feasible
        availability = [ceil(Int, per_stock[k] * (1.05 + 0.30 * rand(rng))) for k in 1:n_stock]
        feasible_witness = SetupPlanWitness(plan_pair, runs)
    else
        base = [per_stock[k] * (1.05 + 0.30 * rand(rng)) + 1.0 for k in 1:n_stock]
        supply = sum(stock_lengths[k] * base[k] for k in 1:n_stock)
        ratio = feasibility_status == infeasible ? 1.0 / (1.08 + 0.12 * rand(rng)) : 0.97 + 0.13 * rand(rng)
        availability = [floor(Int, base[k] * ratio * material / supply) for k in 1:n_stock]
        if feasibility_status == infeasible
            supply = sum(Float64(stock_lengths[k]) * availability[k] for k in 1:n_stock)
            material >= 1.04 * supply || error("setup_cost: shortage certificate lost its margin")
            infeasibility_certificate = MaterialShortageCertificate(material, supply)
        end
    end

    return SetupCostCuttingStockProblem(
        stock_lengths,
        stock_costs,
        piece_lengths,
        demands,
        patterns,
        n_machines,
        pair_pattern,
        pair_machine,
        run_minutes,
        setup_minutes,
        setup_cost,
        link_bound,
        machine_minutes,
        availability,
        feasibility_status,
        feasible_witness,
        infeasibility_certificate,
    )
end

function build_model(prob::SetupCostCuttingStockProblem)
    model = Model()
    pats = prob.patterns
    Q = length(prob.pair_pattern)
    @variable(model, x[1:Q] >= 0)
    @variable(model, y[1:Q], Bin)
    @objective(
        model,
        Min,
        sum(prob.stock_costs[pats.stock[prob.pair_pattern[q]]] * x[q] for q in 1:Q) +
            sum(prob.setup_cost[q] * y[q] for q in 1:Q)
    )
    produced = [AffExpr() for _ in eachindex(prob.piece_lengths)]
    used = [AffExpr() for _ in eachindex(prob.stock_lengths)]
    minutes = [AffExpr() for _ in 1:prob.n_machines]
    for q in 1:Q
        j = prob.pair_pattern[q]
        for (i, c) in zip(pats.items[j], pats.counts[j])
            add_to_expression!(produced[i], c, x[q])
        end
        add_to_expression!(used[pats.stock[j]], 1.0, x[q])
        m = prob.pair_machine[q]
        add_to_expression!(minutes[m], prob.run_minutes[q], x[q])
        add_to_expression!(minutes[m], prob.setup_minutes[q], y[q])
    end
    @constraint(model, demand[i in eachindex(produced)], produced[i] >= prob.demands[i])
    for k in eachindex(used)
        isempty(used[k].terms) && continue
        @constraint(model, used[k] <= prob.availability[k])
    end
    for m in 1:prob.n_machines
        isempty(minutes[m].terms) && continue
        @constraint(model, minutes[m] <= prob.machine_minutes[m])
    end
    @constraint(model, link[q in 1:Q], x[q] <= prob.link_bound[q] * y[q])
    return model
end

register_variant(
    :cutting_stock,
    :setup_cost,
    SetupCostCuttingStockProblem,
    "Multi-machine cutting stock with pattern setups: run counts linked to setup binaries that cost money and machine minutes",
)
