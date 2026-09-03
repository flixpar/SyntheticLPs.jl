using JuMP
using Random
using Distributions

"""
Planted feasible schedule for a maritime inventory-routing instance: the
vessel routes actually sailed, the cargo moved on them, and the resulting
onboard-load and port-inventory trajectories.

All period-indexed trajectories that exist for periods `0..T` are stored with
period `t` in column `t + 1`.

# Fields

  - `position::Matrix{Int}`: `position[v, t + 1]` is the port index (1 = depot)
    occupied by vessel `v` at period `t`, for `t = 0..T`
  - `pickup::Matrix{Float64}`: `pickup[v, t]` loaded at the depot in period `t`
  - `delivery::Array{Float64,3}`: `delivery[v, c, t]` discharged at customer `c`
  - `load::Matrix{Float64}`: `load[v, t + 1]` onboard cargo after period `t`
  - `inventory::Matrix{Float64}`: `inventory[c, t + 1]` tank level after period `t`

The witness is a complete primal point of the (unrelaxed, binary) model: the
route indicators follow directly from `position`, so feasibility can be checked
by pure arithmetic against every constraint row.
"""
struct MaritimeScheduleWitness
    position::Matrix{Int}
    pickup::Matrix{Float64}
    delivery::Array{Float64, 3}
    load::Matrix{Float64}
    inventory::Matrix{Float64}
end

"""
Relaxation-proof aggregate material certificate over the prefix horizon
`1..horizon`.

Consumption in the prefix must be covered by the material already in the tanks
plus everything the fleet can discharge, and the discharge is bounded twice
over:

  - **supply bound** - deliveries only move material that is already afloat or
    taken from the depot, so
    `sum(deliveries) <= sum(initial_load) + sum(depot_supply[1:H])`
    (load balance with `load >= 0`, plus the depot-supply rows).
  - **throughput bound** - summing the flow-conservation rows gives
    `sum_p location[v, p, t] == 1` for every vessel and period, so the linking
    rows `pickup <= cap * location[v, 1, t]` and
    `delivery <= cap * location[v, c + 1, t]` add up to
    `pickup[v, t] + sum_c delivery[v, c, t] <= cap_v`. Over `H` periods that
    caps deliveries at `(initial_load_v + H * cap_v) / 2` per vessel.

`deliverable = initial_inventory + min(supply, throughput)` is therefore a valid
upper bound on what the customers can receive, and `deliverable < consumption`
refutes the instance. Every row used is a linear row of the model, so the
certificate refutes the LP relaxation as well as the integer model.

# Fields

  - `horizon::Int`: prefix length `H` the argument is made over
  - `consumption::Float64`: total customer consumption in periods `1..H`
  - `initial_inventory::Float64`: material already in the customer tanks
  - `initial_load::Float64`: material already onboard the fleet
  - `depot_supply::Float64`: total depot availability in periods `1..H`
  - `supply_bound::Float64`: `initial_inventory + initial_load + depot_supply`
  - `throughput_bound::Float64`:
    `initial_inventory + sum_v (initial_load_v + H * capacity_v) / 2`
  - `deliverable::Float64`: `min(supply_bound, throughput_bound)`, strictly below
    `consumption`
"""
struct MaritimeSupplyCertificate
    horizon::Int
    consumption::Float64
    initial_inventory::Float64
    initial_load::Float64
    depot_supply::Float64
    supply_bound::Float64
    throughput_bound::Float64
    deliverable::Float64
end

"""
    MaritimeInventoryRoutingProblem <: ProblemGenerator

Discrete-time maritime inventory-routing problem (MIRP) on a time-expanded
sailing network. A single depot port (index 1) loads a fleet of vessels that
sail to `n_customers` consumption ports; binary vessel positions and leg
choices decide where cargo can be discharged, onboard load evolves with depot
pickups and deliveries, and each customer tank drains under exogenous
consumption between visits.

# Sailing network

One period is `period_length` days. A leg `(i, j)` is present in `arcs` exactly
when it can be sailed inside one period, `travel_time[i, j] <= period_length`;
waiting arcs `(i, i)` (travel time 0) and the depot shuttle legs `(1, c)`,
`(c, 1)` are always present, and the remaining port-to-port legs are added in
increasing travel time. The arc count is the sizing knob that lets the realised
variable count track the requested target (see the constructor docstring).

# Fields

  - `n_ports::Int`: ports, including the depot at index 1
  - `n_customers::Int`: `n_ports - 1` consumption ports
  - `n_vessels::Int`, `n_periods::Int`: fleet size and planning horizon
  - `arcs::Vector{Tuple{Int,Int}}`: sailing/waiting legs of the time-expanded network
  - `travel_time::Matrix{Float64}`: sailing days between ports (zero diagonal)
  - `period_length::Float64`: days per period; every arc fits inside it
  - `vessel_capacity::Vector{Float64}`, `initial_load::Vector{Float64}`
  - `initial_inventory::Vector{Float64}`, `inventory_capacity::Vector{Float64}`
  - `consumption::Matrix{Float64}`: `consumption[c, t]` drawn from tank `c`
  - `depot_supply::Vector{Float64}`: material released by the depot per period
  - `travel_cost::Matrix{Float64}`: cost of a leg (zero for waiting arcs)
  - `holding_cost::Vector{Float64}`: per-unit tank holding cost
  - `feasible_witness::Union{Nothing,MaritimeScheduleWitness}`: populated only for
    a requested-`feasible` instance
  - `infeasibility_certificate::Union{Nothing,MaritimeSupplyCertificate}`:
    populated only for a requested-`infeasible` instance
  - `feasibility_status::FeasibilityStatus`

This is a genuine MIP (vessel positions and legs are binary); with the package
default `relax_integer=true` it is returned as its LP relaxation, in which a
vessel may sit fractionally in several ports at once. Both the planted witness
and the certificate are valid for the relaxation as well as for the MIP.
"""
struct MaritimeInventoryRoutingProblem <: ProblemGenerator
    n_ports::Int
    n_customers::Int
    n_vessels::Int
    n_periods::Int
    arcs::Vector{Tuple{Int, Int}}
    travel_time::Matrix{Float64}
    period_length::Float64
    vessel_capacity::Vector{Float64}
    initial_load::Vector{Float64}
    initial_inventory::Vector{Float64}
    inventory_capacity::Vector{Float64}
    consumption::Matrix{Float64}
    depot_supply::Vector{Float64}
    travel_cost::Matrix{Float64}
    holding_cost::Vector{Float64}
    feasible_witness::Union{Nothing, MaritimeScheduleWitness}
    infeasibility_certificate::Union{Nothing, MaritimeSupplyCertificate}
    feasibility_status::FeasibilityStatus
end

# Waiting arcs at every port plus both depot shuttle legs per customer are
# mandatory (they carry the planted rotation); the complete network is P^2.
_mirp_min_arcs(P::Int) = 3P - 2
_mirp_max_arcs(P::Int) = P * P

"""
    _mirp_base_variables(P, V, T)

Variables that do not depend on the arc count: vessel positions `V*P*(T+1)`,
deliveries `V*(P-1)*T`, pickups `V*T`, onboard loads `V*(T+1)` and tank levels
`(P-1)*(T+1)`. Affine in `P`: `b1 * P + b0` with `b1 = V*(2T+1) + (T+1)` and
`b0 = (V-1)*(T+1)`.
"""
function _mirp_base_variables(P::Int, V::Int, T::Int)
    C = P - 1
    return V * P * (T + 1) + V * C * T + V * T + V * (T + 1) + C * (T + 1)
end

"""
    _mirp_variable_count(P, V, T, A)

Exact number of variables of the built model: `_mirp_base_variables(P, V, T)`
plus the `V * A * T` leg indicators.
"""
_mirp_variable_count(P::Int, V::Int, T::Int, A::Int) = _mirp_base_variables(P, V, T) + V * A * T

"""
    _mirp_dimensions(rng, target) -> (P, V, T, A)

Pick ports, vessels, periods and arcs so that `_mirp_variable_count` lands on
`target`. Fleet size and horizon are sampled inside operationally sensible,
target-dependent ranges; for each `(V, T)` the port count is obtained in closed
form by solving

    V*T*rho*P^2 + b1*P + b0 = target

for the sampled arc density `rho = A / P^2` (the arc term dominates), and the
arc count is then read off exactly from

    A = (target - _mirp_base_variables(P, V, T)) / (V * T)

clamped to `[3P-2, P^2]`. A short scan around the closed-form `P` picks the best
of the candidates: smallest relative error first, then closeness to the sampled
fleet/horizon/density shape, so instances stay varied across seeds without
giving up sizing accuracy.

`P` is capped at `1 + V * (T ÷ 2)` - exactly the condition that the planted
rotation can visit every customer at least once inside the horizon - and at 300
ports, which no target below a few million variables reaches.
"""
function _mirp_dimensions(rng::AbstractRNG, target::Int)
    vmax = clamp(round(Int, 1.2 * target^0.20), 1, 8)
    tmax = clamp(round(Int, 2.2 * target^0.22), 6, 60)
    tmin = clamp(round(Int, 0.60 * tmax), 4, tmax)
    v_pref = rand(rng, 1:vmax)
    t_pref = rand(rng, tmin:tmax)
    rho_pref = rand(rng, Uniform(0.25, 0.65))

    best = (2, 1, 4, 4)
    best_score = (Inf, Inf)
    for V in 1:vmax, T in tmin:tmax
        b1 = V * (2T + 1) + (T + 1)
        b0 = (V - 1) * (T + 1)
        quad = V * T * rho_pref
        p_star = (sqrt(b1^2 + 4 * quad * max(target - b0, 0)) - b1) / (2 * quad)
        p_cap = min(300, 1 + V * (T ÷ 2))
        lo = clamp(floor(Int, p_star) - 4, 2, p_cap)
        hi = clamp(ceil(Int, p_star) + 4, 2, p_cap)
        for P in lo:hi
            base = _mirp_base_variables(P, V, T)
            A = clamp(round(Int, (target - base) / (V * T)), _mirp_min_arcs(P), _mirp_max_arcs(P))
            total = _mirp_variable_count(P, V, T, A)
            err = abs(total - target) / target
            shape =
                abs(V - v_pref) / vmax +
                abs(T - t_pref) / tmax +
                abs(A / _mirp_max_arcs(P) - rho_pref)
            score = (round(err; digits=5), shape)
            if score < best_score
                best_score = score
                best = (P, V, T, A)
            end
        end
    end
    return best
end

"""
    _mirp_arc_set(P, travel_time, n_arcs) -> (arcs, period_length)

Time-expanded leg set with exactly `n_arcs` entries: waiting arcs at every
port, both depot shuttle legs of every customer, then the shortest remaining
port-to-port legs (added in both directions). `period_length` is the longest
selected sailing time, so a leg is in the network exactly when it can be sailed
within one period.
"""
function _mirp_arc_set(P::Int, travel_time::Matrix{Float64}, n_arcs::Int)
    arcs = Tuple{Int, Int}[(i, i) for i in 1:P]
    for c in 2:P
        push!(arcs, (1, c))
        push!(arcs, (c, 1))
    end
    optional = Tuple{Int, Int}[(i, j) for i in 2:P for j in (i + 1):P]
    sort!(optional; by=a -> (travel_time[a[1], a[2]], a[1], a[2]))
    remaining = n_arcs - length(arcs)
    for (i, j) in optional
        remaining <= 0 && break
        push!(arcs, (i, j))
        remaining -= 1
        if remaining > 0
            push!(arcs, (j, i))
            remaining -= 1
        end
    end
    period_length = maximum(travel_time[i, j] for (i, j) in arcs)
    return arcs, period_length
end

"""
    _mirp_plan(consumption, V, T, order) -> NamedTuple

Build the planted depot-shuttle rotation. Customers are dealt round-robin to
the vessels in the (already shuffled) `order`; vessel `v` waits at the depot in
odd periods and discharges at its `k`-th rotation customer in period `2k`. Each
visit delivers exactly the consumption of that tank until the next visit (or
the end of the horizon), each depot period picks up exactly the next delivery,
and the initial tank level covers consumption until the first visit.

Returns the routes (`position`), the cargo flows (`pickup`, `delivery`) and the
resulting `load` / `inventory` trajectories; by construction loads return to
zero after every discharge and tank levels hit zero just before each visit, so
the plan is feasible for any resources at least as large as its own peaks.
"""
function _mirp_plan(consumption::Matrix{Float64}, V::Int, T::Int, order::Vector{Int})
    C = size(consumption, 1)
    rotation = [Int[] for _ in 1:V]
    for (k, c) in enumerate(order)
        push!(rotation[1 + (k - 1) % V], c)
    end

    position = fill(1, V, T + 1)
    visit_times = [Int[] for _ in 1:C]
    visit_vessel = zeros(Int, C)
    for v in 1:V
        stops = rotation[v]
        isempty(stops) && continue
        for k in 1:(T ÷ 2)
            c = stops[1 + (k - 1) % length(stops)]
            position[v, 2k + 1] = c + 1
            push!(visit_times[c], 2k)
            visit_vessel[c] = v
        end
    end

    delivery = zeros(Float64, V, C, T)
    inventory = zeros(Float64, C, T + 1)
    initial_inventory = zeros(Float64, C)
    for c in 1:C
        times = visit_times[c]
        initial_inventory[c] =
            isempty(times) ? sum(consumption[c, :]) : sum(consumption[c, 1:(times[1] - 1)])
        v = visit_vessel[c]
        for (idx, t) in enumerate(times)
            nxt = idx < length(times) ? times[idx + 1] : T + 1
            delivery[v, c, t] = sum(consumption[c, t:(nxt - 1)])
        end
        inventory[c, 1] = initial_inventory[c]
        for t in 1:T
            received = v == 0 ? 0.0 : delivery[v, c, t]
            inventory[c, t + 1] = inventory[c, t] + received - consumption[c, t]
        end
    end

    pickup = zeros(Float64, V, T)
    load = zeros(Float64, V, T + 1)
    for v in 1:V
        for t in 1:(T - 1)
            pickup[v, t] = sum(delivery[v, c, t + 1] for c in 1:C)
        end
        for t in 1:T
            load[v, t + 1] = load[v, t] + pickup[v, t] - sum(delivery[v, c, t] for c in 1:C)
        end
    end

    return (
        position=position,
        pickup=pickup,
        delivery=delivery,
        load=load,
        inventory=inventory,
        initial_inventory=initial_inventory,
    )
end

# Round up to cents so derived capacities never fall below the quantity they
# have to cover.
_mirp_ceil2(x::Real) = ceil(x * 100) / 100

"""
    _mirp_plan_resources!(vessel_capacity, inventory_capacity, rng,
                          peak_load, peak_inventory, unit)

Size the fleet and the customer tanks a comfortable margin above the peaks of
the planted rotation, so the plan fits inside them with room to spare.
"""
function _mirp_plan_resources!(
    vessel_capacity::Vector{Float64},
    inventory_capacity::Vector{Float64},
    rng::AbstractRNG,
    peak_load::Vector{Float64},
    peak_inventory::Vector{Float64},
    unit::Float64,
)
    for v in eachindex(vessel_capacity)
        vessel_capacity[v] = _mirp_ceil2(max(peak_load[v] * rand(rng, Uniform(1.05, 1.35)), unit))
    end
    for c in eachindex(inventory_capacity)
        inventory_capacity[c] = _mirp_ceil2(
            max(peak_inventory[c] * rand(rng, Uniform(1.05, 1.40)), unit)
        )
    end
    return nothing
end

"""
    _mirp_certificate(consumption, initial_inventory, initial_load,
                      depot_supply, vessel_capacity) -> certificate or nothing

Scan every prefix horizon and return the [`MaritimeSupplyCertificate`](@ref)
with the largest shortage, or `nothing` if no prefix is short.
"""
function _mirp_certificate(
    consumption::Matrix{Float64},
    initial_inventory::Vector{Float64},
    initial_load::Vector{Float64},
    depot_supply::Vector{Float64},
    vessel_capacity::Vector{Float64},
)
    T = size(consumption, 2)
    inv0 = sum(initial_inventory)
    load0 = sum(initial_load)
    best = nothing
    best_gap = 0.0
    cumulative_supply = 0.0
    cumulative_consumption = 0.0
    for H in 1:T
        cumulative_supply += depot_supply[H]
        cumulative_consumption += sum(consumption[:, H])
        supply_bound = inv0 + load0 + cumulative_supply
        throughput_bound =
            inv0 +
            sum((initial_load[v] + H * vessel_capacity[v]) / 2 for v in 1:length(vessel_capacity))
        deliverable = min(supply_bound, throughput_bound)
        gap = cumulative_consumption - deliverable
        if gap > best_gap
            best_gap = gap
            best = MaritimeSupplyCertificate(
                H,
                cumulative_consumption,
                inv0,
                load0,
                cumulative_supply,
                supply_bound,
                throughput_bound,
                deliverable,
            )
        end
    end
    return best
end

"""
    MaritimeInventoryRoutingProblem(target_variables, feasibility_status, seed)

Construct a maritime inventory-routing instance targeting `target_variables`
variables.

# Variable count

With `P` ports, `V` vessels, `T` periods and `A` sailing/waiting arcs the model
has exactly

    V*P*(T+1) + V*A*T + V*(P-1)*T + V*T + V*(T+1) + (P-1)*(T+1)

variables. [`_mirp_dimensions`](@ref) inverts this expression: it samples the
fleet/horizon/arc-density shape, solves the resulting quadratic for the port
count in closed form and then reads the arc count off exactly, so the realised
count usually equals the target and is otherwise within one `V*T` block of it.

# Feasibility (relaxation-aware)

  - `feasible`: a depot-shuttle rotation is planted (odd periods at the depot,
    discharge at the `k`-th rotation customer in period `2k`) and every resource -
    vessel capacity, tank capacity, initial tank level and depot availability - is
    sized from the peaks of that plan, so the schedule stored in `feasible_witness`
    is a feasible point of the integer model and of its relaxation.
  - `infeasible`: the fleet and the depot are starved - initial material plus the
    whole horizon's depot supply is only `0.55..0.88` of total consumption - which
    is refuted by the aggregate material argument in
    `infeasibility_certificate` using LP rows only.
  - `unknown`: fleet and tanks are sized at or above the plan (vessel capacity
    `1.00..1.50`, tank capacity `1.00..1.60` of the plan peaks), while the depot
    is scaled by a global scarcity factor in `0.80..1.15` with per-period noise
    in `0.90..1.10`. Whether a short period can be absorbed by loading earlier
    depends on the onboard capacity the fleet happens to have, so feasibility is
    genuinely undecided - and mixed both ways - at every scale.
"""
function MaritimeInventoryRoutingProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    target = max(target_variables, 1)

    P, V, T, A = _mirp_dimensions(rng, target)
    C = P - 1

    # Port geography on a 1200 x 1200 km sea grid; the depot is port 1.
    coords = [(rand(rng, Uniform(0.0, 1200.0)), rand(rng, Uniform(0.0, 1200.0))) for _ in 1:P]
    speed = rand(rng, Uniform(450.0, 650.0))          # km sailed per day
    port_fee = rand(rng, Uniform(400.0, 1600.0))      # USD per call
    bunker_rate = rand(rng, Uniform(0.8, 1.8))        # USD per km
    travel_time = zeros(Float64, P, P)
    travel_cost = zeros(Float64, P, P)
    for i in 1:P, j in 1:P
        i == j && continue
        d = hypot(coords[i][1] - coords[j][1], coords[i][2] - coords[j][2])
        travel_time[i, j] = round(d / speed; digits=3)
        travel_cost[i, j] = round(port_fee + bunker_rate * d; digits=2)
    end
    arcs, period_length = _mirp_arc_set(P, travel_time, A)

    base_rate = [rand(rng, Uniform(20.0, 120.0)) for _ in 1:C]
    consumption = [
        round(base_rate[c] * rand(rng, Uniform(0.8, 1.2)); digits=2) for c in 1:C, _ in 1:T
    ]
    holding_cost = [round(rand(rng, Uniform(0.1, 1.6)); digits=3) for _ in 1:C]

    plan = _mirp_plan(consumption, V, T, randperm(rng, C))
    peak_load = [maximum(plan.load[v, :]) for v in 1:V]
    peak_inventory = [maximum(plan.inventory[c, :]) for c in 1:C]
    plan_pickup = [sum(plan.pickup[:, t]) for t in 1:T]
    unit = sum(consumption) / max(C * T, 1)

    vessel_capacity = zeros(Float64, V)
    initial_load = zeros(Float64, V)
    initial_inventory = zeros(Float64, C)
    inventory_capacity = zeros(Float64, C)
    depot_supply = zeros(Float64, T)
    witness = nothing
    certificate = nothing

    if feasibility_status == feasible
        _mirp_plan_resources!(
            vessel_capacity, inventory_capacity, rng, peak_load, peak_inventory, unit
        )
        initial_inventory .= plan.initial_inventory
        for t in 1:T
            depot_supply[t] = _mirp_ceil2(
                max(plan_pickup[t] * rand(rng, Uniform(1.05, 1.40)), unit)
            )
        end
        witness = MaritimeScheduleWitness(
            plan.position, plan.pickup, plan.delivery, plan.load, plan.inventory
        )
    elseif feasibility_status == infeasible
        _mirp_plan_resources!(
            vessel_capacity, inventory_capacity, rng, peak_load, peak_inventory, unit
        )
        # Starve the system: everything the customers could ever receive is a
        # strict fraction of what they consume over the horizon.
        budget = rand(rng, Uniform(0.55, 0.88)) * sum(consumption)
        initial_inventory .= plan.initial_inventory .* rand(rng, Uniform(0.0, 0.4))
        initial_load .= vessel_capacity .* rand(rng, Uniform(0.0, 0.3))
        held = sum(initial_inventory) + sum(initial_load)
        if held > 0.5 * budget
            scale = 0.5 * budget / held
            initial_inventory .*= scale
            initial_load .*= scale
            held = sum(initial_inventory) + sum(initial_load)
        end
        weights = [rand(rng, Uniform(0.7, 1.3)) for _ in 1:T]
        depot_supply .= (budget - held) .* weights ./ sum(weights)
        certificate = _mirp_certificate(
            consumption, initial_inventory, initial_load, depot_supply, vessel_capacity
        )
    else
        # Fleet and tanks are sized like the feasible branch, but the depot is
        # scaled by a global scarcity factor straddling 1 and shaken by
        # per-period noise. Whether the fleet can shift cargo between periods
        # to absorb a short period depends on the onboard capacity it happens
        # to have, so feasibility is genuinely undecided at every scale.
        scarcity = rand(rng, Uniform(0.80, 1.15))
        for v in 1:V
            vessel_capacity[v] = _mirp_ceil2(max(peak_load[v] * rand(rng, Uniform(1.0, 1.5)), unit))
            initial_load[v] = vessel_capacity[v] * rand(rng, Uniform(0.0, 0.3))
        end
        for c in 1:C
            inventory_capacity[c] = _mirp_ceil2(
                max(peak_inventory[c] * rand(rng, Uniform(1.0, 1.6)), unit)
            )
            initial_inventory[c] = plan.initial_inventory[c]
        end
        for t in 1:T
            depot_supply[t] = _mirp_ceil2(plan_pickup[t] * scarcity * rand(rng, Uniform(0.9, 1.1)))
        end
    end

    return MaritimeInventoryRoutingProblem(
        P,
        C,
        V,
        T,
        arcs,
        travel_time,
        period_length,
        vessel_capacity,
        initial_load,
        initial_inventory,
        inventory_capacity,
        consumption,
        depot_supply,
        travel_cost,
        holding_cost,
        witness,
        certificate,
        feasibility_status,
    )
end

"""
    build_model(prob::MaritimeInventoryRoutingProblem)

Build the time-expanded MIRP model. Deterministic - uses only struct fields.

# Model

Variables:

  - `location[v, p, t] in {0,1}`: vessel `v` is at port `p` at period `t = 0..T`
  - `move[v, a, t] in {0,1}`: vessel `v` traverses leg `prob.arcs[a]` in period `t`
  - `delivery[v, c, t] >= 0`, `pickup[v, t] >= 0`: cargo discharged / loaded
  - `0 <= load[v, t] <= capacity_v`, `0 <= inventory[c, t] <= tank_c`

Constraints: vessels start at the depot with their initial load; leg flow
conservation ties `move` to `location` on both ends; pickups and deliveries are
allowed only where the vessel is (big-M on the vessel capacity); onboard load
and tank levels follow their balances; and the depot releases at most
`depot_supply[t]` per period.
"""
function build_model(prob::MaritimeInventoryRoutingProblem)
    model = Model()
    P = prob.n_ports
    C = prob.n_customers
    V = prob.n_vessels
    T = prob.n_periods
    arcs = prob.arcs
    A = length(arcs)

    out_arcs = [Int[] for _ in 1:P]
    in_arcs = [Int[] for _ in 1:P]
    for (a, (i, j)) in enumerate(arcs)
        push!(out_arcs[i], a)
        push!(in_arcs[j], a)
    end

    @variable(model, location[1:V, 1:P, 0:T], Bin)
    @variable(model, move[1:V, 1:A, 1:T], Bin)
    @variable(model, delivery[1:V, 1:C, 1:T] >= 0)
    @variable(model, pickup[1:V, 1:T] >= 0)
    @variable(model, 0 <= load[v in 1:V, 0:T] <= prob.vessel_capacity[v])
    @variable(model, 0 <= inventory[c in 1:C, 0:T] <= prob.inventory_capacity[c])

    @objective(
        model,
        Min,
        sum(
            prob.travel_cost[arcs[a][1], arcs[a][2]] * move[v, a, t] for
            v in 1:V, a in 1:A, t in 1:T
        ) + sum(prob.holding_cost[c] * inventory[c, t] for c in 1:C, t in 1:T)
    )

    for v in 1:V
        @constraint(model, location[v, 1, 0] == 1)
        for p in 2:P
            @constraint(model, location[v, p, 0] == 0)
        end
        @constraint(model, load[v, 0] == prob.initial_load[v])
    end
    for c in 1:C
        @constraint(model, inventory[c, 0] == prob.initial_inventory[c])
    end

    for v in 1:V, t in 1:T
        for i in 1:P
            @constraint(model, sum(move[v, a, t] for a in out_arcs[i]) == location[v, i, t - 1])
        end
        for j in 1:P
            @constraint(model, sum(move[v, a, t] for a in in_arcs[j]) == location[v, j, t])
        end
        @constraint(model, pickup[v, t] <= prob.vessel_capacity[v] * location[v, 1, t])
        @constraint(
            model, load[v, t] == load[v, t - 1] + pickup[v, t] - sum(delivery[v, c, t] for c in 1:C)
        )
        for c in 1:C
            @constraint(model, delivery[v, c, t] <= prob.vessel_capacity[v] * location[v, c + 1, t])
        end
    end
    for t in 1:T
        @constraint(model, sum(pickup[v, t] for v in 1:V) <= prob.depot_supply[t])
    end
    for c in 1:C, t in 1:T
        @constraint(
            model,
            inventory[c, t] ==
                inventory[c, t - 1] + sum(delivery[v, c, t] for v in 1:V) - prob.consumption[c, t]
        )
    end

    return model
end

register_variant(
    :maritime_inventory_routing,
    :standard,
    MaritimeInventoryRoutingProblem,
    "Time-expanded maritime inventory routing on a sailing-time network with binary vessel legs, onboard cargo, depot pickup, deliveries, and customer tank balances",
)
