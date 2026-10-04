using JuMP
using Random
using Distributions

"""
Planted production plan: built by a backward (as-late-as-possible) pass that
fills each period's capacity with the pending net requirements, so it exists
exactly when every demand prefix fits the cumulative capacity. `production[i,
t]` and `inventory[i, t]` are a feasible point of the built model.
"""
struct MultiItemPlanWitness
    production::Matrix{Float64}
    inventory::Matrix{Float64}
end

"""
Prefix capacity certificate. Summing item `i`'s balance rows over periods
`1..horizon` (no backlog, nonnegative stock) gives `Σ_{t<=horizon} x[i,t] >=
D_i(1..horizon) - initial_inventory[i]`; weighting by resource usage and
summing the capacity rows of those periods shows the prefix needs `required =
Σ_i usage[i] max(0, D_i(1..horizon) - I0_i)` resource units but only has
`available = Σ_{t<=horizon} capacity[t] < required`.
"""
struct MultiItemPrefixCertificate
    horizon::Int
    required::Float64
    available::Float64
end

"""
    MultiItemInventoryProblem <: ProblemGenerator

Multi-item production/inventory planning with a shared, time-varying
production capacity.

# Overview

`n_items` items compete for one production resource over `n_periods`
periods. Columns `x[i, t] >= 0` (production) and `I[i, t] >= 0` (end-of-period
stock); rows: per-item balance `I[i,t-1] + x[i,t] - I[i,t] = d[i,t]` (with
`I[i,0] = initial_inventory[i]`, no backlog) and the shared capacity
`Σ_i usage[i] x[i,t] <= capacity[t]` per period. Capacity varies over time
(planned maintenance, holiday weeks), and demand is seasonal with a peak, so
building ahead of the peak is what the LP must decide.

For a single shared resource, the instance is feasible **iff** every demand
prefix fits: `Σ_i usage[i] max(0, D_i(1..τ) - I0_i) <= Σ_{t<=τ} capacity[t]`
for all `τ` (an as-late-as-possible schedule meets it). The profiles place the
binding prefix ratio:

  - `feasible`: ratio `0.70–0.92`, witness from the backward pass
    ([`MultiItemPlanWitness`](@ref));
  - `infeasible`: ratio `1.10–1.35` at the binding prefix
    ([`MultiItemPrefixCertificate`](@ref)); initial stock covers the first
    period's demand, so no single row is violated and presolve cannot decide it;
  - `unknown`: ratio `1 ± U(0.03, 0.30)` — feasible exactly when it is `<= 1`.

# Fields

  - `n_items::Int`, `n_periods::Int`
  - `demand::Matrix{Float64}`: `n_items × n_periods`
  - `initial_inventory::Vector{Float64}`, `usage::Vector{Float64}`
  - `production_cost::Matrix{Float64}`, `holding_cost::Vector{Float64}`
  - `capacity::Vector{Float64}`: per period
  - `binding_ratio::Float64`: max over prefixes of required / available
  - `feasible_witness`, `infeasibility_certificate`, `feasibility_status`
"""
struct MultiItemInventoryProblem <: ProblemGenerator
    n_items::Int
    n_periods::Int
    demand::Matrix{Float64}
    initial_inventory::Vector{Float64}
    usage::Vector{Float64}
    production_cost::Matrix{Float64}
    holding_cost::Vector{Float64}
    capacity::Vector{Float64}
    binding_ratio::Float64
    feasible_witness::Union{Nothing, MultiItemPlanWitness}
    infeasibility_certificate::Union{Nothing, MultiItemPrefixCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    _multi_item_prefix(demand, initial_inventory, usage, capacity) -> (ratio, horizon)

The binding demand prefix: `max_τ required(τ) / available(τ)` and its `τ`.
"""
function _multi_item_prefix(demand, initial_inventory, usage, capacity)
    N, T = size(demand)
    cum = zeros(N)
    best, best_h, avail = -Inf, 1, 0.0
    for t in 1:T
        avail += capacity[t]
        req = 0.0
        for i in 1:N
            cum[i] += demand[i, t]
            req += usage[i] * max(0.0, cum[i] - initial_inventory[i])
        end
        r = req / avail
        if r > best
            best, best_h = r, t
        end
    end
    return best, best_h
end

"""
    _multi_item_prefix_requirement(demand, initial_inventory, usage, horizon) -> Float64
"""
function _multi_item_prefix_requirement(demand, initial_inventory, usage, horizon::Int)
    return sum(
        usage[i] * max(0.0, sum(view(demand, i, 1:horizon)) - initial_inventory[i]) for i in 1:size(demand, 1)
    )
end

"""
    MultiItemInventoryProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a multi-item shared-capacity instance with `2 · n_items · n_periods`
columns close to `target_variables`.
"""
function MultiItemInventoryProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be >= 1 (got $target_variables)."))
    rng = MersenneTwister(seed)
    target = max(target_variables, 8)

    T = target <= 400 ? rand(rng, 4:8) : (target <= 10_000 ? rand(rng, 10:20) : rand(rng, 20:40))
    N = max(1, round(Int, target / (2T)))

    phase = 2π * rand(rng)
    amp = 0.2 + 0.25 * rand(rng)     # pronounced seasonal peak: pre-building matters
    demand = zeros(N, T)
    for i in 1:N
        demand[i, :] = _inventory_demand(
            rng, T, rand(rng, LogNormal(log(100.0), 0.8)); amp=amp, phase=phase + 0.3 * randn(rng),
            trend=0.01 * randn(rng), cv=0.15 + 0.25 * rand(rng), intermittent=rand(rng) < 0.1,
        )
    end
    # Initial stock covers the first period plus a little: no single capacity
    # row can ever be contradicted on its own.
    initial_inventory = [round(demand[i, 1] + sum(demand[i, :]) / T * rand(rng, Uniform(0.05, 0.4)); digits=1) for i in 1:N]
    usage = rand(rng, LogNormal(log(1.0), 0.4), N)
    base_cost = rand(rng, LogNormal(log(20.0), 0.6), N)
    production_cost = [base_cost[i] * (1 + 0.05 * randn(rng)) for i in 1:N, _ in 1:T]
    production_cost = max.(production_cost, 0.1)
    holding_cost = base_cost .* rand(rng, Uniform(0.004, 0.02), N)

    # Time-varying capacity shape (maintenance and holiday dips).
    shape = [rand(rng) < 0.12 ? rand(rng, Uniform(0.5, 0.8)) : rand(rng, Uniform(0.95, 1.05)) for _ in 1:T]
    ratio0, _ = _multi_item_prefix(demand, initial_inventory, usage, shape)
    target_ratio = if feasibility_status == feasible
        0.7 + 0.22 * rand(rng)
    else
        _inventory_scale_ratio(rng, feasibility_status)
    end
    capacity = shape .* (max(ratio0, 1e-9) / target_ratio)
    # With every requirement already met by initial stock the scale is moot:
    # give the plant a nominal capacity proportional to demand.
    if ratio0 <= 1e-9
        capacity = shape .* max(sum(usage[i] * sum(demand[i, :]) for i in 1:N) / T, 1.0)
    end
    binding_ratio, horizon = _multi_item_prefix(demand, initial_inventory, usage, capacity)

    witness = nothing
    certificate = nothing
    if feasibility_status == feasible
        # Backward (as-late-as-possible) fill of each period's capacity.
        net = copy(demand)
        for i in 1:N
            left = initial_inventory[i]
            for t in 1:T
                used = min(left, net[i, t])
                net[i, t] -= used
                left -= used
            end
        end
        production = zeros(N, T)
        pending = zeros(N)
        for t in T:-1:1
            pending .+= view(net, :, t)
            total = sum(usage .* pending)
            frac = total <= capacity[t] ? 1.0 : capacity[t] / total
            for i in 1:N
                production[i, t] = frac * pending[i]
                pending[i] -= production[i, t]
            end
        end
        inventory = zeros(N, T)
        for i in 1:N
            level = initial_inventory[i]
            for t in 1:T
                level += production[i, t] - demand[i, t]
                inventory[i, t] = max(level, 0.0)
            end
        end
        witness = MultiItemPlanWitness(production, inventory)
    elseif feasibility_status == infeasible
        req = _multi_item_prefix_requirement(demand, initial_inventory, usage, horizon)
        certificate = MultiItemPrefixCertificate(horizon, req, sum(capacity[1:horizon]))
    end

    return MultiItemInventoryProblem(
        N,
        T,
        demand,
        initial_inventory,
        usage,
        production_cost,
        holding_cost,
        capacity,
        binding_ratio,
        witness,
        certificate,
        feasibility_status,
    )
end

"""
    build_model(prob::MultiItemInventoryProblem)

Build the multi-item shared-capacity LP. Deterministic.
"""
function build_model(prob::MultiItemInventoryProblem)
    model = Model()
    N, T = prob.n_items, prob.n_periods
    @variable(model, x[1:N, 1:T] >= 0)
    @variable(model, I[1:N, 1:T] >= 0)
    for i in 1:N, t in 1:T
        if t == 1
            @constraint(model, x[i, 1] - I[i, 1] == prob.demand[i, 1] - prob.initial_inventory[i])
        else
            @constraint(model, I[i, t - 1] + x[i, t] - I[i, t] == prob.demand[i, t])
        end
    end
    for t in 1:T
        @constraint(model, sum(prob.usage[i] * x[i, t] for i in 1:N) <= prob.capacity[t])
    end
    @objective(
        model,
        Min,
        sum(prob.production_cost[i, t] * x[i, t] + prob.holding_cost[i] * I[i, t] for i in 1:N, t in 1:T)
    )
    return model
end

register_variant(
    :inventory,
    :multi_item,
    MultiItemInventoryProblem,
    "Multi-item production/inventory planning with a shared time-varying capacity and seasonal peaks: feasibility decided exactly by the binding demand prefix, with a planted as-late-as-possible plan and a prefix capacity certificate",
)
