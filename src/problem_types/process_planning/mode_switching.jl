using JuMP
using Random
using Distributions

"""
    RefineryModeSwitchingProblem <: ProblemGenerator

Multi-period refinery planning with the operating mode of every conversion unit
left to the solver: the campaign-planning problem behind
`process_planning/refinery`.

# Formulation

The flowsheet, tankage, blending and market data are those of
[`RefineryPlanningProblem`](@ref). What changes is the unit model. A unit runs at
most one of its operating modes in each period — a catalytic cracker in
maximum-gasoline, maximum-distillate or maximum-olefins mode, a reformer at mid
or high severity, a hydrocracker swung to diesel, jet or naphtha, a coker making
fuel-grade or anode-grade coke, a diesel hydrotreater at ULSD or mild severity —
and each mode has its own yield slate, capacity de-rate and operating cost. The
run indicator gates the unit's throughput from both sides, so a minimum rate
applies only while the unit is actually running (something a pure LP cannot
state), and changing mode between periods costs a changeover.

Feasibility contracts, the planted operation and both infeasibility certificates
are those of the shared flowsheet: the certificates bound production using each
feed's best mode, so they refute every mode assignment at once.

# Fields
- `flowsheet::RefineryFlowsheet`: crude assays, streams, multi-mode units, grades
- `data::ProcessPlanData`: prices, availability, capacities, tanks, demands and
  the per-unit changeover cost
- `initial_mode::Vector{Int}`: the mode each unit ran before the horizon opened
  (`0` for a unit that was down), which the first period's changeovers are
  measured against
- `feasible_witness`, `infeasibility_certificate`, `feasibility_status`

Run indicators and changeovers are binary, so this is a genuine MILP; with the
package default `relax_integer=true` it is returned as its LP relaxation, in
which a unit may run several modes at once in fractional proportions. The planted
witness is integral and both certificates hold for the relaxation.
"""
struct RefineryModeSwitchingProblem <: ProblemGenerator
    flowsheet::RefineryFlowsheet
    data::ProcessPlanData
    initial_mode::Vector{Int}
    feasible_witness::Union{Nothing,RefineryOperatingPlan}
    infeasibility_certificate::Union{Nothing,RefineryInfeasibilityCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    _pp_campaign_modes(rng, fs, n_periods) -> Matrix{Int}

Plant a seasonal campaign: each multi-mode unit runs its first mode through the
summer half of the horizon and its second through the winter half — a cracker
swinging from gasoline to distillate as the season turns — with an occasional
extra campaign of a third mode. Units with one mode simply run it.
"""
function _pp_campaign_modes(rng::AbstractRNG, fs::RefineryFlowsheet, T::Int)
    U = n_units(fs)
    modes = ones(Int, U, T)
    for u in 1:U
        count = length(fs.units[u].modes)
        count == 1 && continue
        phase = rand(rng, Uniform(0.0, 2pi))
        alternate = min(2, count)
        third = count >= 3 && rand(rng) < 0.5 ? 3 : 0
        campaign = third > 0 ? rand(rng, 1:T) : 0
        for t in 1:T
            season = sin(2pi * (t - 1) / max(T, 2) + phase)
            modes[u, t] = season >= 0 ? 1 : alternate
            t == campaign && (modes[u, t] = third)
        end
    end
    return modes
end

"""
    RefineryModeSwitchingProblem(target_variables, feasibility_status, seed)

Construct a multi-period refinery campaign-planning instance.

# Variable count

With `C` crudes, `T` periods, units `u` running `M_u` modes over feed sets `F_u`,
grades `p` with component sets `B_p`, `n_store` tanked streams, `n_buy` purchased
blendstocks and `n_spot` streams saleable as they are, the model has exactly

    T * (3C + sum_u M_u (|F_u| + 3) + sum_p |B_p| + n_store + n_buy + n_spot + 2P)

variables: the three per mode are the mode's throughput, its run indicator and
its changeover. The count is affine in `C` for a fixed flowsheet, so
[`_pp_dimensions`](@ref) inverts it exactly for the crude count at each candidate
horizon.

# Feasibility
- `feasible`: a seasonal campaign is planted (see [`_pp_campaign_modes`](@ref)),
  the flowsheet is operated through it, and every capacity, tank, availability,
  contract and blend window is placed around the result — including the
  mode-dependent capacity de-rates — so the witness is a feasible point of the
  integer model.
- `infeasible`: as in [`RefineryPlanningProblem`](@ref), and both certificates
  take each feed's best mode, so no mode assignment escapes them.
- `unknown`: assets, contracts and quality windows are drawn from design rules
  and a market view rather than reconciled with the plan, leaving feasibility
  genuinely open.
"""
function RefineryModeSwitchingProblem(target_variables::Int,
                                      feasibility_status::FeasibilityStatus,
                                      seed::Int)
    rng = MersenneTwister(seed)
    target = max(target_variables, 1)

    _, skeleton, n_periods = _pp_dimensions(rng, target; mode_vars=true)
    flowsheet = _pp_build_flowsheet(rng, skeleton)
    mode_choice = _pp_campaign_modes(rng, flowsheet, n_periods)

    data, plan, certificate = _pp_plan_instance(rng, flowsheet, n_periods,
                                                feasibility_status, mode_choice;
                                                conditional_rates=true)

    # What the units were running when the horizon opened; a unit that was down
    # is recorded as mode 0 and pays a changeover to start.
    initial_mode = [rand(rng) < 0.85 ? rand(rng, 1:length(unit.modes)) : 0
                    for unit in flowsheet.units]

    problem = RefineryModeSwitchingProblem(
        flowsheet, data, initial_mode,
        feasibility_status == feasible ? plan : nothing, certificate,
        feasibility_status)

    if feasibility_status == feasible
        @assert refinery_plan_satisfies(flowsheet, data, plan)
    elseif feasibility_status == infeasible
        @assert refinery_certificate_holds(flowsheet, data, certificate)
    end
    return problem
end

"""
    build_model(prob::RefineryModeSwitchingProblem)

Build the multi-period campaign-planning MILP. Deterministic — uses only the
stored flowsheet and data.

# Model
Variables per period: crude purchases, runs and tank levels; per unit and mode, a
feed flow for every admissible stream, the mode's throughput, a binary run
indicator and a changeover indicator; a blend flow per (grade, component);
intermediate tank levels; blendstock purchases; spot sales; and finished-product
sales and tank levels.

Constraints: the crude and stream balances, crude-unit capacity, turndown and
charge sulfur of the LP model, plus — per unit, mode and period — the throughput
definition, the two-sided gate `min * z <= throughput <= capacity * derate * z`,
the at-most-one-mode rule and the changeover linking `switch >= z_t - z_{t-1}`.
The objective is the LP objective less the changeover cost.
"""
function build_model(prob::RefineryModeSwitchingProblem)
    fs = prob.flowsheet
    data = prob.data
    C = fs.n_crudes
    S = n_streams(fs)
    U = n_units(fs)
    P = n_products(fs)
    T = data.n_periods

    model = Model()

    @variable(model, 0 <= crude_buy[c in 1:C, t in 1:T] <=
                     data.crude_availability[c, t])
    # Implied but stated: no single crude can be charged beyond the crude unit's
    # capacity, no feed beyond its unit's, and nothing can be blended into a grade
    # beyond what that grade can sell or store. The bounds cut nothing off, and a
    # simplex given them does not have to discover them.
    @variable(model, 0 <= crude_run[c in 1:C, t in 1:T] <= data.cdu_capacity[t])
    @variable(model, 0 <= crude_inventory[c in 1:C, t in 1:T] <=
                     data.crude_tank_capacity[c])
    @variable(model, 0 <= stream_inventory[s in fs.storable, t in 1:T] <=
                     data.stream_tank_capacity[s])
    @variable(model, 0 <= purchase[s in fs.purchasable, t in 1:T] <=
                     data.stream_purchase_limit[s])
    @variable(model, 0 <= spot_sale[s in fs.spot, t in 1:T] <=
                     data.stream_spot_limit[s])
    @variable(model, data.demand_min[p, t] <= sales[p in 1:P, t in 1:T] <=
                     data.demand_max[p, t])
    @variable(model, 0 <= product_inventory[p in 1:P, t in 1:T] <=
                     data.product_tank_capacity[p])
    feed = [@variable(model, [f in 1:length(fs.units[u].feeds),
                              m in 1:length(fs.units[u].modes), t in 1:T],
                      lower_bound = 0,
                      upper_bound = data.unit_capacity[u, t] *
                                    fs.units[u].modes[m].capacity_factor,
                      base_name = "feed_$(fs.units[u].name)") for u in 1:U]
    throughput = [@variable(model, [1:length(fs.units[u].modes), 1:T],
                            lower_bound = 0,
                            base_name = "throughput_$(fs.units[u].name)")
                  for u in 1:U]
    run_mode = [@variable(model, [1:length(fs.units[u].modes), 1:T], Bin,
                          base_name = "run_$(fs.units[u].name)") for u in 1:U]
    switch = [@variable(model, [1:length(fs.units[u].modes), 1:T], lower_bound = 0,
                        base_name = "switch_$(fs.units[u].name)") for u in 1:U]
    blend = [@variable(model, [b in 1:length(fs.products[p].components), t in 1:T],
                       lower_bound = 0,
                       upper_bound = data.demand_max[p, t] +
                                     data.product_tank_capacity[p],
                       base_name = "blend_$(fs.products[p].name)") for p in 1:P]

    for t in 1:T
        for c in 1:C
            previous = t == 1 ? data.crude_initial_inventory[c] :
                       crude_inventory[c, t - 1]
            @constraint(model, previous + crude_buy[c, t] ==
                               crude_run[c, t] + crude_inventory[c, t])
        end
        @constraint(model, sum(crude_run[c, t] for c in 1:C) <= data.cdu_capacity[t])
        data.cdu_min_throughput[t] > 0 &&
            @constraint(model, sum(crude_run[c, t] for c in 1:C) >=
                               data.cdu_min_throughput[t])
        @constraint(model,
            sum((fs.crude_sulfur[c] - data.cdu_sulfur_limit) * crude_run[c, t]
                for c in 1:C) <= 0)
    end

    balance = Matrix{AffExpr}(undef, S, T)
    for s in 1:S, t in 1:T
        balance[s, t] = AffExpr(0.0)
    end
    for c in 1:C, k in eachindex(fs.cut_classes)
        s = fs.cut_stream[c, k]
        yield = fs.cut_yields[c, k]
        for t in 1:T
            add_to_expression!(balance[s, t], yield, crude_run[c, t])
        end
    end
    for u in 1:U
        unit = fs.units[u]
        modes = length(unit.modes)
        for m in 1:modes
            mode = unit.modes[m]
            for (f, s) in enumerate(unit.feeds), t in 1:T
                add_to_expression!(balance[s, t], -1.0, feed[u][f, m, t])
            end
            for f in eachindex(unit.feeds), (o, out) in enumerate(unit.outputs)
                yield = mode.yields[f, o]
                yield == 0.0 && continue
                for t in 1:T
                    add_to_expression!(balance[out, t], yield, feed[u][f, m, t])
                end
            end
            for t in 1:T
                @constraint(model, throughput[u][m, t] ==
                                   sum(feed[u][f, m, t]
                                       for f in eachindex(unit.feeds)))
                @constraint(model, throughput[u][m, t] <=
                                   data.unit_capacity[u, t] * mode.capacity_factor *
                                   run_mode[u][m, t])
                data.unit_min_throughput[u, t] > 0 &&
                    @constraint(model, throughput[u][m, t] >=
                                       data.unit_min_throughput[u, t] *
                                       run_mode[u][m, t])
                previous = t == 1 ? (prob.initial_mode[u] == m ? 1 : 0) :
                           run_mode[u][m, t - 1]
                @constraint(model, switch[u][m, t] >= run_mode[u][m, t] - previous)
            end
        end
        for t in 1:T
            @constraint(model, sum(run_mode[u][m, t] for m in 1:modes) <= 1)
        end
    end
    for p in 1:P, (b, s) in enumerate(fs.products[p].components), t in 1:T
        add_to_expression!(balance[s, t], -1.0, blend[p][b, t])
    end
    for s in fs.purchasable, t in 1:T
        add_to_expression!(balance[s, t], 1.0, purchase[s, t])
    end
    for s in fs.spot, t in 1:T
        add_to_expression!(balance[s, t], -1.0, spot_sale[s, t])
    end
    for s in fs.storable, t in 1:T
        add_to_expression!(balance[s, t], -1.0, stream_inventory[s, t])
        t > 1 && add_to_expression!(balance[s, t], 1.0, stream_inventory[s, t - 1])
    end
    for s in 1:S
        add_to_expression!(balance[s, 1], data.stream_initial_inventory[s])
    end
    @constraint(model, stream_balance[s in 1:S, t in 1:T], balance[s, t] == 0)

    for p in 1:P, t in 1:T
        previous = t == 1 ? data.product_initial_inventory[p] :
                   product_inventory[p, t - 1]
        @constraint(model,
            previous + sum(blend[p][b, t]
                           for b in eachindex(fs.products[p].components)) ==
            sales[p, t] + product_inventory[p, t])
    end
    _pp_add_blend_specifications!(model, fs, blend, T)

    @objective(model, Max,
        sum(data.product_price[p, t] * sales[p, t] for p in 1:P, t in 1:T) +
        sum(data.stream_spot_price[s] * spot_sale[s, t] for s in fs.spot, t in 1:T) -
        sum(data.crude_price[c, t] * crude_buy[c, t] for c in 1:C, t in 1:T) -
        sum(data.stream_purchase_cost[s, t] * purchase[s, t]
            for s in fs.purchasable, t in 1:T) -
        sum(fs.units[u].modes[m].operating_cost * throughput[u][m, t] +
            data.unit_switch_cost[u] * switch[u][m, t]
            for u in 1:U for m in 1:length(fs.units[u].modes) for t in 1:T) -
        sum(data.stream_holding_cost[s] * stream_inventory[s, t]
            for s in fs.storable, t in 1:T) -
        sum(data.product_holding_cost[p] * product_inventory[p, t]
            for p in 1:P, t in 1:T))

    return model
end

register_variant(
    :process_planning,
    :mode_switching,
    RefineryModeSwitchingProblem,
    "Multi-period refinery campaign planning: one operating mode per conversion " *
    "unit and period, mode-dependent yields, gated turndown, and changeover costs",
)
