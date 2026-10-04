using JuMP
using Random
using Distributions

"""
Planted MRP plan: lot-for-lot explosion of the demand through the bill of
materials with lead-time offsets, initial stock covering the lead-time gap and
retained safety stock, no backlog, and overtime only where a work center's
lumpy lot-for-lot load exceeds its regular hours. `production[i, t]` is the
quantity of item `i` released (manufactured or ordered) in period `t` (zero for
periods after `n_periods - lead_time[i]`, which have no column), `inventory[i,
t]` the end-of-period stock, and `overtime[w, t]` the overtime hours booked.
"""
struct ProductionPlanWitness
    production::Matrix{Float64}
    inventory::Matrix{Float64}
    overtime::Matrix{Float64}
end

"""
Echelon-load infeasibility certificate. Summing item `i`'s balance rows over
the horizon (backlog must be cleared by the last period and stock is
nonnegative) gives `Σ_t x[i,t] >= Σ_parents a[i,p] Σ_t x[p,t] + D_i - I0_i`, so
by induction down the bill of materials every item's total release is at least
`lower_bounds[i] = max(0, Σ_p a[i,p] lower_bounds[p] + D_i - I0_i)`. Work
center `work_center` must therefore absorb `required_load = Σ_{i at w}
run_time[i] * lower_bounds[i]` hours, while its capacity rows (regular hours
plus overtime bounds) supply at most `available_capacity` over the horizon.
`required_load > available_capacity` refutes the LP using balance rows of every
item upstream of the work center plus all of its capacity rows — an aggregate
argument presolve cannot see.
"""
struct EchelonCapacityCertificate
    work_center::Int
    items::Vector{Int}
    lower_bounds::Vector{Float64}
    required_load::Float64
    available_capacity::Float64
end

"""
    ProductionPlanningProblem <: ProblemGenerator

Multi-level, multi-period capacitated production planning (MRP II) LP.

# Overview

A manufacturer plans `n_periods` periods (weeks) for a product structure of
end products, subassemblies, components, and purchased raw materials linked by
a sparse bill of materials (BOM) with shared common parts. Items are indexed
level by level (end products first), so every BOM edge points from a lower to a
higher index.

Columns:

  - `x[i, t] >= 0` — release of item `i` in period `t` (production start, or a
    purchase order for raw materials), for `t <= n_periods - lead_time[i]`; it
    arrives `lead_time[i]` periods later;
  - `I[i, t] >= 0` — end-of-period inventory, every item and period;
  - `B[e, t] >= 0` — backlog of end product `e` for `t < n_periods` (all
    demand must be served by the end of the horizon);
  - `O[w, t] ∈ [0, max_overtime[w, t]]` — overtime hours at work center `w`.

Rows:

  - dependent-demand balance, every item and period:
    `I[i,t-1] + x[i,t-L_i] - Σ_parents a[i,p] x[p,t] - d[i,t] (+ B[i,t] - B[i,t-1]) = I[i,t]`
    with `I[i,0] = initial_inventory[i]`;
  - work-center capacity, every work center and period:
    `Σ_{i at w} run_time[i] x[i,t] - O[w,t] <= regular_capacity[w,t]`;
  - supplier capacity, every supplier and period (raw materials):
    `Σ_{i from s} volume[i] x[i,t] <= supplier_capacity[s,t]`.

Objective: minimize value-added and purchase cost, holding cost, backlog
penalties, and overtime premiums. The staircase structure (inventory chains
coupled across items by the BOM and across items by shared capacity rows) is
the classical hard case for simplex, unlike the single-period `product_mix`.

# Data grounding

End-product demand is seasonal with trend, Gamma noise, and some intermittent
(lumpy) items; 15% of subassemblies/components also carry service-part
demand. BOM children are drawn with a heavy-tailed popularity so common
components are shared widely. Lead times are 0–1 periods for manufactured items
and 1–3 for purchased ones. Item values roll up through the BOM; holding cost is
an annual carrying rate on value, backlog penalties a fraction of value per
period, overtime 1.5× the labor rate. Regular capacity is flat per work center
(with holiday dips) around the plan's average load; overtime absorbs peaks.

# Feasibility control

  - `feasible`: the lot-for-lot MRP plan is planted as a
    [`ProductionPlanWitness`](@ref); capacities are drawn around it.
  - `infeasible`: the bottleneck work center loses capacity (regular hours and
    overtime scaled down) until its echelon lower-bound load exceeds its
    horizon capacity by 8–30% — an [`EchelonCapacityCertificate`](@ref).
  - `unknown`: the same scaling with load/capacity ratio `1 ± U(0.03, 0.30)`;
    above 1 provably infeasible, below 1 decided by timing (lead times, early
    demand, backlog limits).

# Fields

  - `n_items`, `n_periods`, `n_work_centers`, `n_suppliers`, `n_end_items::Int`
  - `level::Vector{Int}`: BOM level (1 = end product, deepest = purchased)
  - `purchased::Vector{Bool}`
  - `bom_parent`, `bom_child::Vector{Int}`, `bom_qty::Vector{Float64}`: BOM edges
  - `lead_time::Vector{Int}`
  - `work_center::Vector{Int}`, `run_time::Vector{Float64}`: (0 / 0.0 for purchased)
  - `supplier::Vector{Int}`, `volume::Vector{Float64}`: (0 / 0.0 for manufactured)
  - `demand::Matrix{Float64}`: independent demand, `n_items × n_periods`
  - `initial_inventory::Vector{Float64}`
  - `regular_capacity`, `max_overtime::Matrix{Float64}`: `n_work_centers × n_periods`
  - `supplier_capacity::Matrix{Float64}`: `n_suppliers × n_periods`
  - `unit_cost`, `holding_cost`, `backlog_cost::Vector{Float64}`, `overtime_cost::Vector{Float64}`
  - `feasible_witness`, `infeasibility_certificate`, `feasibility_status`
"""
struct ProductionPlanningProblem <: ProblemGenerator
    n_items::Int
    n_periods::Int
    n_work_centers::Int
    n_suppliers::Int
    n_end_items::Int
    level::Vector{Int}
    purchased::Vector{Bool}
    bom_parent::Vector{Int}
    bom_child::Vector{Int}
    bom_qty::Vector{Float64}
    lead_time::Vector{Int}
    work_center::Vector{Int}
    run_time::Vector{Float64}
    supplier::Vector{Int}
    volume::Vector{Float64}
    demand::Matrix{Float64}
    initial_inventory::Vector{Float64}
    regular_capacity::Matrix{Float64}
    max_overtime::Matrix{Float64}
    supplier_capacity::Matrix{Float64}
    unit_cost::Vector{Float64}
    holding_cost::Vector{Float64}
    backlog_cost::Vector{Float64}
    overtime_cost::Vector{Float64}
    feasible_witness::Union{Nothing, ProductionPlanWitness}
    infeasibility_certificate::Union{Nothing, EchelonCapacityCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    _production_planning_columns(n_items, T, lead_time, n_end, n_wc) -> Int

Column count: releases `Σ_i (T - L_i)`, inventories `n_items·T`, backlogs
`n_end·(T - 1)`, overtime `n_wc·T`.
"""
function _production_planning_columns(lead_time::Vector{Int}, T::Int, n_end::Int, n_wc::Int)
    return sum(T - L for L in lead_time) + length(lead_time) * T + n_end * (T - 1) + n_wc * T
end

"""
    _production_planning_lower_bounds(prob_parts...) -> Vector{Float64}

Echelon lower bounds on total horizon release per item (see
[`EchelonCapacityCertificate`](@ref)); parents precede children in index order.
"""
function _production_planning_lower_bounds(
    n_items::Int,
    bom_parent::Vector{Int},
    bom_child::Vector{Int},
    bom_qty::Vector{Float64},
    demand::Matrix{Float64},
    initial_inventory::Vector{Float64},
)
    parents = [Tuple{Int, Float64}[] for _ in 1:n_items]
    for (p, c, a) in zip(bom_parent, bom_child, bom_qty)
        push!(parents[c], (p, a))
    end
    lb = zeros(n_items)
    for i in 1:n_items
        need = sum(@view demand[i, :]) - initial_inventory[i]
        for (p, a) in parents[i]
            need += a * lb[p]
        end
        lb[i] = max(0.0, need)
    end
    return lb
end

"""
    ProductionPlanningProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a multi-level capacitated MRP planning instance with about
`target_variables` columns (within a few percent from ~200 upward).
"""
function ProductionPlanningProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be >= 1 (got $target_variables)."))
    rng = MersenneTwister(seed)
    target = max(target_variables, 40)

    # --- Horizon and product structure size ----------------------------------
    # Roughly 2.25 columns per item-period (release + inventory, plus backlog
    # and overtime columns spread over the items).
    T, n_items = if target < 400
        # Tiny plans: a handful of items over a short horizon.
        n = 5 + target ÷ 60
        (max(3, round(Int, target / (2.3 * n))), n)
    else
        t = if target <= 1_500
            rand(rng, 6:10)
        elseif target <= 20_000
            rand(rng, 10:20)
        else
            rand(rng, 16:30)
        end
        (t, round(Int, target / (2.14 * t)))
    end
    n_levels = n_items >= 12 ? 4 : 3
    shares = n_levels == 4 ? [0.15, 0.25, 0.35, 0.25] : [0.3, 0.4, 0.3]
    counts = max.(1, round.(Int, shares .* n_items))
    counts[end] = max(1, n_items - sum(counts[1:(end - 1)]))
    n_items = sum(counts)
    level = reduce(vcat, [fill(l, counts[l]) for l in 1:n_levels])
    level_items = [findall(==(l), level) for l in 1:n_levels]
    purchased = level .== n_levels
    n_end = counts[1]

    # --- Product families, cells, and the bill of materials -------------------
    # Real product structures are modular: a family's end products share its
    # subassemblies and components, a few common parts (fasteners, packaging,
    # base materials) are used plant-wide, and each family is built in one
    # manufacturing cell. That block structure, weakly coupled through common
    # parts, is what real MRP LPs look like — a random plant-wide BOM would
    # couple every item to every other one and make the LP dense.
    fam_size = rand(rng, 16:40)
    n_fam = max(1, round(Int, n_items / fam_size))
    family = [rand(rng, 1:n_fam) for _ in 1:n_items]
    common = [level[i] >= 3 && rand(rng) < 0.08 for i in 1:n_items]
    fam_level = Dict{Tuple{Int, Int}, Vector{Int}}()
    for i in 1:n_items
        push!(get!(fam_level, (family[i], level[i]), Int[]), i)
    end
    common_level = [[i for i in level_items[l] if common[i]] for l in 1:n_levels]
    popularity = rand(rng, LogNormal(0.0, 0.6), n_items)
    function pick_child(f, l)
        if rand(rng) < 0.08 && !isempty(common_level[l])
            return rand(rng, common_level[l])
        end
        pool = get(fam_level, (f, l), level_items[l])   # no such family item: any
        w = popularity[pool]
        return pool[rand(rng, Categorical(w ./ sum(w)))]
    end
    function bom_qty_for(child_level)
        child_level == 2 && return rand(rng) < 0.7 ? 1.0 : 2.0            # subassembly
        child_level == n_levels && return round(rand(rng, LogNormal(0.0, 0.6)); digits=2) + 0.05  # raw (kg, m)
        return Float64(rand(rng, 1:4))                                   # component
    end
    bom_parent, bom_child, bom_qty = Int[], Int[], Float64[]
    has_parent = falses(n_items)
    for l in 1:(n_levels - 1), p in level_items[l]
        chosen = Int[]
        for _ in 1:(rand(rng, 1:3) + (l == 1 ? 1 : 0))
            c = pick_child(family[p], l + 1)
            c in chosen || push!(chosen, c)
        end
        # Occasionally skip a level (e.g. packaging material on an end item).
        if l + 2 <= n_levels && rand(rng) < 0.2
            c = pick_child(family[p], l + 2)
            c in chosen || push!(chosen, c)
        end
        for c in chosen
            push!(bom_parent, p)
            push!(bom_child, c)
            push!(bom_qty, bom_qty_for(level[c]))
            has_parent[c] = true
        end
    end
    # Every non-end item is used by something (no orphan parts): attach it to
    # a parent of its own family when there is one.
    for l in 2:n_levels, c in level_items[l]
        has_parent[c] && continue
        push!(bom_parent, rand(rng, get(fam_level, (family[c], l - 1), level_items[l - 1])))
        push!(bom_child, c)
        push!(bom_qty, bom_qty_for(l))
    end
    perm = sortperm(collect(zip(bom_parent, bom_child)))
    bom_parent, bom_child, bom_qty = bom_parent[perm], bom_child[perm], bom_qty[perm]

    # --- Cells, routing, suppliers, lead times -----------------------------------
    lead_time = [purchased[i] ? rand(rng, 1:3) : (rand(rng) < 0.5 ? 0 : 1) for i in 1:n_items]
    manufactured = findall(.!purchased)
    # Families are grouped into cells; each cell has its own work centers per
    # manufacturing level (assembly, subassembly, fabrication). Common parts
    # are made in a shared plant-wide cell.
    n_cells = max(1, round(Int, n_fam / rand(rng, 2:3)))
    fam_cell = [mod1(k, n_cells) for k in shuffle(rng, 1:n_fam)]
    cell_of(i) = common[i] ? n_cells + 1 : fam_cell[family[i]]
    work_center = zeros(Int, n_items)
    n_wc = 0
    for c in 1:(n_cells + 1), l in 1:(n_levels - 1)
        items = [i for i in level_items[l] if cell_of(i) == c]
        isempty(items) && continue
        k = max(1, round(Int, length(items) / rand(rng, 5:12)))
        for i in items
            work_center[i] = n_wc + rand(rng, 1:k)
        end
        n_wc += k
    end
    # Compact work-center ids (a center can end up empty).
    used = sort(unique(work_center[manufactured]))
    remap = Dict(w => k for (k, w) in enumerate(used))
    for i in manufactured
        work_center[i] = remap[work_center[i]]
    end
    n_wc = length(used)
    run_mu = [log(0.3), log(0.45), log(0.15)]
    run_time = [
        purchased[i] ? 0.0 : rand(rng, LogNormal(run_mu[min(level[i], 3)], 0.5)) for i in 1:n_items
    ]
    # Suppliers: one or two per cell for its families' materials, plus one
    # plant-wide supplier for common materials.
    raw = level_items[n_levels]
    sup_per_cell = [rand(rng, 1:2) for _ in 1:(n_cells + 1)]
    sup_offset = cumsum([0; sup_per_cell[1:(end - 1)]])
    supplier = zeros(Int, n_items)
    for i in raw
        c = cell_of(i)
        supplier[i] = sup_offset[c] + rand(rng, 1:sup_per_cell[c])
    end
    used = sort(unique(supplier[raw]))
    remap = Dict(s => k for (k, s) in enumerate(used))
    for i in raw
        supplier[i] = remap[supplier[i]]
    end
    n_sup = length(used)
    volume = [purchased[i] ? rand(rng, LogNormal(0.0, 0.5)) : 0.0 for i in 1:n_items]

    # --- Demand -------------------------------------------------------------------
    demand = zeros(n_items, T)
    phase = 2π * rand(rng)
    season_amp = 0.35 * rand(rng)
    for e in level_items[1]
        base = rand(rng, LogNormal(log(60.0), 0.9))
        trend = rand(rng, Uniform(-0.005, 0.012))
        lumpy = rand(rng) < 0.15
        for t in 1:T
            mean_t =
                base *
                (1 + season_amp * sin(2π * t / 26 + phase + 0.3 * randn(rng))) *
                (1 + trend * t)
            d = rand(rng, Gamma(16.0, max(mean_t, 0.1) / 16.0))
            demand[e, t] = lumpy && rand(rng) < 0.5 ? 0.0 : round(d; digits=1)
        end
    end
    # Service-part demand on some subassemblies/components.
    for l in 2:(n_levels - 1), i in level_items[l]
        rand(rng) < 0.15 || continue
        base = rand(rng, LogNormal(log(6.0), 0.7))
        for t in 1:T
            demand[i, t] = round(rand(rng, Gamma(4.0, base / 4.0)); digits=1)
        end
    end

    # --- Costs (values roll up through the BOM, children before parents) --------
    labor_rate = rand(rng, Uniform(28.0, 45.0))
    value = zeros(n_items)
    for i in raw
        value[i] = rand(rng, LogNormal(log(4.0), 0.7))
    end
    children = [Tuple{Int, Float64}[] for _ in 1:n_items]
    for (p, c, a) in zip(bom_parent, bom_child, bom_qty)
        push!(children[p], (c, a))
    end
    for i in n_items:-1:1
        purchased[i] && continue
        value[i] =
            sum(a * value[c] for (c, a) in children[i]; init=0.0) + labor_rate * run_time[i] * 1.6
    end
    unit_cost = [purchased[i] ? value[i] : labor_rate * run_time[i] for i in 1:n_items]
    carrying = rand(rng, Uniform(0.18, 0.35)) / 52          # weekly carrying rate
    holding_cost = carrying .* value .* rand(rng, Uniform(0.9, 1.1), n_items)
    backlog_cost = [
        level[i] == 1 ? value[i] * rand(rng, Uniform(0.04, 0.12)) : 0.0 for i in 1:n_items
    ]
    overtime_cost = labor_rate * 1.5 .* rand(rng, Uniform(0.95, 1.1), n_wc)

    # --- Planted lot-for-lot MRP explosion ------------------------------------------
    parents = [Tuple{Int, Float64}[] for _ in 1:n_items]
    for (p, c, a) in zip(bom_parent, bom_child, bom_qty)
        push!(parents[c], (p, a))
    end
    production = zeros(n_items, T)
    inventory = zeros(n_items, T)
    initial_inventory = zeros(n_items)
    for i in 1:n_items
        L = lead_time[i]
        gross = demand[i, :]
        for (p, a) in parents[i], t in 1:T
            gross[t] += a * production[p, t]
        end
        avg = sum(gross) / T
        safety = round(avg * rand(rng, Uniform(0.1, 0.4)); digits=1)
        initial_inventory[i] =
            sum(gross[1:L]; init=0.0) + safety + avg * rand(rng, Uniform(0.0, 0.5))
        stock = initial_inventory[i]
        for t in 1:T
            arrival = 0.0
            if t > L
                arrival = max(0.0, gross[t] + safety - stock)
                production[i, t - L] = arrival
            end
            stock += arrival - gross[t]
            inventory[i, t] = stock
        end
    end

    # Capacities around the plan's load: flat regular hours (holiday dips),
    # overtime for peaks; flat-ish supplier capacity with headroom.
    load = zeros(n_wc, T)
    for i in manufactured, t in 1:T
        load[work_center[i], t] += run_time[i] * production[i, t]
    end
    regular_capacity = zeros(n_wc, T)
    max_overtime = zeros(n_wc, T)
    overtime = zeros(n_wc, T)
    for w in 1:n_wc
        # A quarter of the work centers are bottlenecks run near their average
        # load; the rest carry the usual 12-45% headroom.
        tightness =
            rand(rng) < 0.25 ? rand(rng, Uniform(0.88, 1.02)) : rand(rng, Uniform(1.12, 1.45))
        level_hours = max(sum(load[w, :]) / T, 1.0) * tightness
        for t in 1:T
            avail = rand(rng) < 0.06 ? rand(rng, Uniform(0.6, 0.8)) : 1.0
            regular_capacity[w, t] = level_hours * avail
            overtime[w, t] = max(0.0, load[w, t] - regular_capacity[w, t])
            max_overtime[w, t] = max(0.2 * regular_capacity[w, t], 1.1 * overtime[w, t])
        end
    end
    supplier_capacity = zeros(n_sup, T)
    sup_load = zeros(n_sup, T)
    for i in raw, t in 1:T
        sup_load[supplier[i], t] += volume[i] * production[i, t]
    end
    for s in 1:n_sup
        base = max(sum(sup_load[s, :]) / T, 1.0) * rand(rng, Uniform(0.9, 1.15))
        for t in 1:T
            supplier_capacity[s, t] = max(sup_load[s, t] * rand(rng, Uniform(1.05, 1.3)), base)
        end
    end

    # --- Feasibility profile --------------------------------------------------------
    witness = nothing
    certificate = nothing
    if feasibility_status == feasible
        witness = ProductionPlanWitness(production, inventory, overtime)
    else
        lb = _production_planning_lower_bounds(
            n_items, bom_parent, bom_child, bom_qty, demand, initial_inventory
        )
        required = zeros(n_wc)
        for i in manufactured
            required[work_center[i]] += run_time[i] * lb[i]
        end
        available = [sum(regular_capacity[w, :]) + sum(max_overtime[w, :]) for w in 1:n_wc]
        wstar = argmax(required ./ available)
        ratio = if feasibility_status == infeasible
            1.08 + 0.22 * rand(rng)
        else
            m = 0.03 + 0.27 * rand(rng)
            rand(rng) < 0.5 ? 1.0 - m : 1.0 + m
        end
        scale = required[wstar] / (ratio * available[wstar])
        regular_capacity[wstar, :] .*= scale
        max_overtime[wstar, :] .*= scale
        if feasibility_status == infeasible
            items = [i for i in manufactured if work_center[i] == wstar]
            certificate = EchelonCapacityCertificate(
                wstar,
                items,
                lb,
                required[wstar],
                sum(regular_capacity[wstar, :]) + sum(max_overtime[wstar, :]),
            )
        end
    end

    return ProductionPlanningProblem(
        n_items,
        T,
        n_wc,
        n_sup,
        n_end,
        level,
        purchased,
        bom_parent,
        bom_child,
        bom_qty,
        lead_time,
        work_center,
        run_time,
        supplier,
        volume,
        demand,
        initial_inventory,
        regular_capacity,
        max_overtime,
        supplier_capacity,
        unit_cost,
        holding_cost,
        backlog_cost,
        overtime_cost,
        witness,
        certificate,
        feasibility_status,
    )
end

"""
    build_model(prob::ProductionPlanningProblem)

Build the multi-level MRP planning LP. Deterministic. Variables are registered
as `x[i]` (vector of releases for periods `1:(T - L_i)`), `I[i, t]`,
`B[e, t]` (end products, `t < T`), and `O[w, t]`.
"""
function build_model(prob::ProductionPlanningProblem)
    model = Model()
    N, T = prob.n_items, prob.n_periods
    @variable(model, x[i = 1:N, t = 1:(T - prob.lead_time[i])] >= 0)
    @variable(model, I[1:N, 1:T] >= 0)
    @variable(model, B[1:prob.n_end_items, 1:(T - 1)] >= 0)
    @variable(model, 0 <= O[w = 1:prob.n_work_centers, t = 1:T] <= prob.max_overtime[w, t])

    parents = [Tuple{Int, Float64}[] for _ in 1:N]
    for (p, c, a) in zip(prob.bom_parent, prob.bom_child, prob.bom_qty)
        push!(parents[c], (p, a))
    end

    # Balance rows.
    for i in 1:N, t in 1:T
        L = prob.lead_time[i]
        expr = AffExpr(0.0)
        t > 1 && add_to_expression!(expr, 1.0, I[i, t - 1])
        t > L && add_to_expression!(expr, 1.0, x[i, t - L])
        for (p, a) in parents[i]
            t <= T - prob.lead_time[p] && add_to_expression!(expr, -a, x[p, t])
        end
        add_to_expression!(expr, -1.0, I[i, t])
        if i <= prob.n_end_items
            t < T && add_to_expression!(expr, 1.0, B[i, t])
            t > 1 && add_to_expression!(expr, -1.0, B[i, t - 1])
        end
        rhs = prob.demand[i, t] - (t == 1 ? prob.initial_inventory[i] : 0.0)
        @constraint(model, expr == rhs)
    end

    # Work-center capacity rows.
    wc_items = [Int[] for _ in 1:prob.n_work_centers]
    for i in 1:N
        prob.work_center[i] > 0 && push!(wc_items[prob.work_center[i]], i)
    end
    for w in 1:prob.n_work_centers, t in 1:T
        items = [i for i in wc_items[w] if t <= T - prob.lead_time[i]]
        @constraint(
            model,
            sum(prob.run_time[i] * x[i, t] for i in items; init=AffExpr(0.0)) - O[w, t] <=
                prob.regular_capacity[w, t]
        )
    end

    # Supplier capacity rows.
    sup_items = [Int[] for _ in 1:prob.n_suppliers]
    for i in 1:N
        prob.supplier[i] > 0 && push!(sup_items[prob.supplier[i]], i)
    end
    for s in 1:prob.n_suppliers, t in 1:T
        items = [i for i in sup_items[s] if t <= T - prob.lead_time[i]]
        isempty(items) && continue
        @constraint(
            model, sum(prob.volume[i] * x[i, t] for i in items) <= prob.supplier_capacity[s, t]
        )
    end

    @objective(
        model,
        Min,
        sum(prob.unit_cost[i] * x[i, t] for i in 1:N for t in 1:(T - prob.lead_time[i])) +
            sum(prob.holding_cost[i] * I[i, t] for i in 1:N, t in 1:T) +
            sum(prob.backlog_cost[e] * B[e, t] for e in 1:prob.n_end_items, t in 1:(T - 1)) +
            sum(prob.overtime_cost[w] * O[w, t] for w in 1:prob.n_work_centers, t in 1:T)
    )
    return model
end

register_variant(
    :production_planning,
    :standard,
    ProductionPlanningProblem,
    "Multi-level, multi-period capacitated MRP planning LP: bill-of-materials balance rows with lead times, backlog, work-center capacity with bounded overtime, and supplier capacity, with a planted lot-for-lot MRP plan and an echelon-load infeasibility certificate";
    tags=[:production, :staircase],
)
