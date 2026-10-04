using JuMP
using Random
using Distributions

"""
Planted lot plan: item `i` is set up in the periods `setups[i]`, and each
production run covers the net demand of the periods up to the next run (a
periodic-order-quantity policy that respects the carrying window). `source[i][t]`
is the production period serving item `i`'s period-`t` net demand (0 when
there is none); `load[s]` is the capacity the plan uses in period `s`
(processing plus setup time).
"""
struct LotSizingPlanWitness
    setups::Vector{Vector{Int}}
    source::Vector{Vector{Int}}
    load::Vector{Float64}
end

"""
Prefix capacity certificate. Every period-`t` demand must be produced in
`[t - window + 1, t]`, and each production period used needs a setup
(`w[i,s,t] <= y[i,s]`). For each item, `setup_lower_bounds[i]` is the number of
pairwise-disjoint production windows among its demand periods `<= horizon`; the
assignment rows over those windows force `Σ_{s <= horizon} y[i,s] >=
setup_lower_bounds[i]` even in the LP relaxation. Hence the first `horizon`
periods must absorb `required = Σ_i proc_time[i] * net demand of i in
1..horizon + Σ_i setup_time[i] * setup_lower_bounds[i]` capacity units, while
the capacity rows supply `available = Σ_{s <= horizon} capacity[s] < required`.
"""
struct LotSizingPrefixCertificate
    horizon::Int
    setup_lower_bounds::Vector{Int}
    required::Float64
    available::Float64
end

"""
    LotSizingInventoryProblem <: ProblemGenerator

Multi-item capacitated lot sizing with setup times, in the facility-location
(Krarup–Bilde) reformulation.

# Overview

`n_items` items share one production resource over `n_periods` periods. Each
period's net demand of an item (after initial stock) must be produced in the
same period or up to `window - 1` periods earlier (shelf life / carrying
limits). Columns:

  - `w[i, s, t] >= 0` — share of item `i`'s period-`t` net demand produced
    in period `s ∈ [t - window + 1, t]` (one column per positive-demand
    period and admissible production period);
  - `y[i, s] ∈ {0, 1}` — setup of item `i` in period `s` (relaxed to `[0, 1]`
    under the default `relax_integer=true`).

Rows:

  - demand assignment, per item and positive-demand period: `Σ_s w[i,s,t] = 1`;
  - disaggregated setup linking, per column: `w[i,s,t] <= y[i,s]`;
  - shared capacity with setup times, per period:
    `Σ_i proc_time[i] Σ_t d[i,t] w[i,s,t] + Σ_i setup_time[i] y[i,s] <= capacity[s]`.

Objective: unit cost plus holding cost for the periods carried
(`(unit_cost[i] + holding_cost[i] (t - s)) d[i,t]` per unit share) plus setup
costs.

This is the strong formulation of capacitated lot sizing: its LP relaxation
does not collapse — fractional setups `y[i,s] >= max_t w[i,s,t]` still consume
setup *time* in the capacity rows, so the relaxation must batch demand into
fewer production periods exactly as an integer plan does. (The previous
big-M `x = lot · n_lots` model relaxed to `y = x / (2 cap)` and a plain
single-item inventory LP.)

# Feasibility control

  - `feasible`: a periodic-order-quantity plan is planted
    ([`LotSizingPlanWitness`](@ref)); capacities are flat around its load and
    never below it.
  - `infeasible`: capacities are cut so that some prefix of periods must absorb
    10–35% more processing-plus-setup time than it has
    ([`LotSizingPrefixCertificate`](@ref)). Every single row stays satisfiable,
    so the contradiction is found by simplex, not presolve.
  - `unknown`: the same prefix ratio drawn as `1 ± U(0.03, 0.30)`.

# Fields

  - `n_items`, `n_periods`, `window::Int`
  - `demand::Matrix{Float64}`: net demand (after initial stock), `n_items × n_periods`
  - `initial_inventory::Vector{Float64}`, `gross_demand::Matrix{Float64}`: raw data
  - `proc_time`, `setup_time`, `unit_cost`, `holding_cost`, `setup_cost::Vector{Float64}`
  - `capacity::Vector{Float64}`: per period
  - `feasible_witness`, `infeasibility_certificate`, `feasibility_status`
"""
struct LotSizingInventoryProblem <: ProblemGenerator
    n_items::Int
    n_periods::Int
    window::Int
    demand::Matrix{Float64}
    initial_inventory::Vector{Float64}
    gross_demand::Matrix{Float64}
    proc_time::Vector{Float64}
    setup_time::Vector{Float64}
    unit_cost::Vector{Float64}
    holding_cost::Vector{Float64}
    setup_cost::Vector{Float64}
    capacity::Vector{Float64}
    feasible_witness::Union{Nothing, LotSizingPlanWitness}
    infeasibility_certificate::Union{Nothing, LotSizingPrefixCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    _lot_sizing_item_columns(demand_row, window) -> Int

Columns of one item: one `w` per positive-demand period and admissible
production period, plus one `y` per production period that serves some demand.
"""
function _lot_sizing_item_columns(d::AbstractVector{Float64}, window::Int)
    T = length(d)
    ys = falses(T)
    nw = 0
    for t in 1:T
        d[t] > 0 || continue
        for s in max(1, t - window + 1):t
            nw += 1
            ys[s] = true
        end
    end
    return nw + count(ys)
end

"""
    _lot_sizing_disjoint_windows(demand_row, window, horizon) -> Int

Maximum number of pairwise-disjoint production windows `[t - window + 1, t]`
over positive-demand periods `t <= horizon` (greedy by right end — optimal for
intervals). Each forces a distinct setup in the LP relaxation.
"""
function _lot_sizing_disjoint_windows(d::AbstractVector{Float64}, window::Int, horizon::Int)
    count_ = 0
    last_end = 0
    for t in 1:horizon
        d[t] > 0 || continue
        if t - window + 1 > last_end
            count_ += 1
            last_end = t
        end
    end
    return count_
end

"""
    _lot_sizing_prefix(prob_fields...) -> (required, setup_lbs) for a horizon
"""
function _lot_sizing_prefix_requirement(
    demand::Matrix{Float64}, proc_time, setup_time, window::Int, horizon::Int
)
    n = size(demand, 1)
    lbs = [_lot_sizing_disjoint_windows(view(demand, i, :), window, horizon) for i in 1:n]
    req = sum(proc_time[i] * sum(view(demand, i, 1:horizon)) + setup_time[i] * lbs[i] for i in 1:n)
    return req, lbs
end

"""
    LotSizingInventoryProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a multi-item capacitated lot-sizing instance with about
`target_variables` columns (items are added until the budget is reached).
"""
function LotSizingInventoryProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be >= 1 (got $target_variables)."))
    rng = MersenneTwister(seed)
    target = max(target_variables, 12)

    T = target <= 300 ? rand(rng, 4:6) : (target <= 5_000 ? rand(rng, 8:13) : rand(rng, 12:20))
    window = min(T, target <= 300 ? rand(rng, 2:3) : rand(rng, 3:6))
    phase = 2π * rand(rng)
    amp = 0.3 * rand(rng)

    # Items until the column budget is used (the last item may overshoot by
    # at most its own column count).
    gross = Vector{Vector{Float64}}()
    init = Float64[]
    net = Vector{Vector{Float64}}()
    cols = 0
    while cols < target
        base = rand(rng, LogNormal(log(80.0), 0.9))
        d = _inventory_demand(
            rng, T, base; amp=amp, phase=phase + 0.4 * randn(rng), trend=0.01 * randn(rng),
            cv=0.2 + 0.3 * rand(rng), intermittent=rand(rng) < 0.15,
        )
        # Initial stock covers part of the opening demand.
        i0 = round(sum(d[1:min(2, T)]) * rand(rng, Uniform(0.0, 0.8)); digits=1)
        n = copy(d)
        left = i0
        for t in 1:T
            used = min(left, n[t])
            n[t] = round(n[t] - used; digits=6)
            left -= used
        end
        c = _lot_sizing_item_columns(n, window)
        c == 0 && continue
        push!(gross, d)
        push!(init, i0)
        push!(net, n)
        cols += c
    end
    N = length(net)
    demand = permutedims(reduce(hcat, net))
    gross_demand = permutedims(reduce(hcat, gross))

    # Item economics: processing time (capacity units per unit), setup time
    # comparable to a fraction of a typical run, setup cost and holding cost
    # with economic-order-interval tension.
    proc_time = rand(rng, LogNormal(log(0.05), 0.5), N)
    mean_run = [proc_time[i] * sum(demand[i, :]) / T for i in 1:N]
    setup_time = [max(0.05, mean_run[i] * rand(rng, Uniform(0.2, 1.0))) for i in 1:N]
    unit_cost = rand(rng, LogNormal(log(10.0), 0.6), N)
    holding_cost = unit_cost .* rand(rng, Uniform(0.005, 0.03), N)
    setup_cost = [
        holding_cost[i] * max(sum(demand[i, :]) / T, 1.0) * rand(rng, Uniform(1.0, 6.0)) for i in 1:N
    ]

    # --- Planted periodic-order-quantity plan --------------------------------
    setups = Vector{Vector{Int}}(undef, N)
    source = Vector{Vector{Int}}(undef, N)
    load = zeros(T)
    for i in 1:N
        interval = rand(rng, 1:window)
        src = zeros(Int, T)
        su = Int[]
        current = 0
        for t in 1:T
            demand[i, t] > 0 || continue
            if current == 0 || t - current >= interval
                current = t
                push!(su, t)
                load[t] += setup_time[i]
            end
            src[t] = current
            load[current] += proc_time[i] * demand[i, t]
        end
        setups[i] = su
        source[i] = src
    end
    flat = sum(load) / T * rand(rng, Uniform(1.05, 1.3))
    capacity = [max(flat * (rand(rng) < 0.1 ? rand(rng, Uniform(0.75, 0.9)) : 1.0), load[s] * 1.03, 1e-3)
                for s in 1:T]

    witness = nothing
    certificate = nothing
    if feasibility_status == feasible
        witness = LotSizingPlanWitness(setups, source, copy(load))
    else
        # The most loaded prefix, then a uniform capacity cut to the drawn ratio.
        ratio = _inventory_scale_ratio(rng, feasibility_status)
        best_h, best_r = 1, -1.0
        for h in 1:T
            req, _ = _lot_sizing_prefix_requirement(demand, proc_time, setup_time, window, h)
            r = req / sum(capacity[1:h])
            if r > best_r
                best_h, best_r = h, r
            end
        end
        capacity .*= best_r / ratio
        if feasibility_status == infeasible
            req, lbs = _lot_sizing_prefix_requirement(demand, proc_time, setup_time, window, best_h)
            certificate = LotSizingPrefixCertificate(best_h, lbs, req, sum(capacity[1:best_h]))
        end
    end

    return LotSizingInventoryProblem(
        N,
        T,
        window,
        demand,
        init,
        gross_demand,
        proc_time,
        setup_time,
        unit_cost,
        holding_cost,
        setup_cost,
        capacity,
        witness,
        certificate,
        feasibility_status,
    )
end

"""
    _lot_sizing_layout(prob) -> (w_item, w_prod, w_dem, y_item, y_period, y_index)

Column layout: `w` columns (item, production period, demand period) in item /
demand-period / production-period order, then `y` columns; `y_index[i][s]` is
the position of `y[i, s]` among the `y` columns (0 when absent).
"""
function _lot_sizing_layout(prob::LotSizingInventoryProblem)
    w_item, w_prod, w_dem = Int[], Int[], Int[]
    y_item, y_period = Int[], Int[]
    y_index = [zeros(Int, prob.n_periods) for _ in 1:prob.n_items]
    for i in 1:prob.n_items
        for t in 1:prob.n_periods
            prob.demand[i, t] > 0 || continue
            for s in max(1, t - prob.window + 1):t
                push!(w_item, i)
                push!(w_prod, s)
                push!(w_dem, t)
                if y_index[i][s] == 0
                    y_index[i][s] = -1
                end
            end
        end
        for s in 1:prob.n_periods
            if y_index[i][s] == -1
                push!(y_item, i)
                push!(y_period, s)
                y_index[i][s] = length(y_item)
            end
        end
    end
    return w_item, w_prod, w_dem, y_item, y_period, y_index
end

"""
    build_model(prob::LotSizingInventoryProblem)

Build the facility-location lot-sizing model (`w` continuous in [0, 1], `y`
binary; `w <= 1` is implied by the assignment rows). Deterministic.
"""
function build_model(prob::LotSizingInventoryProblem)
    model = Model()
    w_item, w_prod, w_dem, y_item, y_period, y_index = _lot_sizing_layout(prob)
    nw, ny = length(w_item), length(y_item)
    @variable(model, w[1:nw] >= 0)
    @variable(model, y[1:ny], Bin)

    # Assignment rows: consecutive w columns share (item, demand period).
    c = 1
    while c <= nw
        e = c
        while e < nw && w_item[e + 1] == w_item[c] && w_dem[e + 1] == w_dem[c]
            e += 1
        end
        @constraint(model, sum(w[k] for k in c:e) == 1)
        c = e + 1
    end
    # Disaggregated setup linking.
    for k in 1:nw
        @constraint(model, w[k] <= y[y_index[w_item[k]][w_prod[k]]])
    end
    # Capacity rows with setup times.
    cap_terms = [AffExpr(0.0) for _ in 1:prob.n_periods]
    for k in 1:nw
        i = w_item[k]
        add_to_expression!(cap_terms[w_prod[k]], prob.proc_time[i] * prob.demand[i, w_dem[k]], w[k])
    end
    for k in 1:ny
        add_to_expression!(cap_terms[y_period[k]], prob.setup_time[y_item[k]], y[k])
    end
    for s in 1:prob.n_periods
        @constraint(model, cap_terms[s] <= prob.capacity[s])
    end

    @objective(
        model,
        Min,
        sum(
            (prob.unit_cost[w_item[k]] + prob.holding_cost[w_item[k]] * (w_dem[k] - w_prod[k])) *
            prob.demand[w_item[k], w_dem[k]] * w[k] for k in 1:nw
        ) + sum(prob.setup_cost[y_item[k]] * y[k] for k in 1:ny)
    )
    return model
end

register_variant(
    :inventory,
    :lot_sizing,
    LotSizingInventoryProblem,
    "Multi-item capacitated lot sizing with setup times in the facility-location reformulation: demand-assignment, disaggregated setup-linking, and shared capacity rows whose LP relaxation keeps the setup trade-off, with a planted POQ plan and a prefix capacity certificate",
)
