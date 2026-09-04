using JuMP
using Random

"""
    RefineryPlanningProblem <: ProblemGenerator

Multi-period refinery production planning: the linear program a refinery's
planning department solves to decide what to buy, run, convert, store and blend
over the next weeks or months.

# Formulation

Crude is bought against per-period availability, held in crude tanks and charged
to the crude unit, which cuts it into fractions at that crude's assay yields.
Each cut is a stream of its own, carrying the quality of the crude it came from —
segregating cuts by crude is exactly what keeps the blend rows linear, since a
common pool of unknown composition would make them bilinear. Streams feed
conversion units (hydrotreating, reforming, isomerization, catalytic cracking,
hydrocracking, coking, alkylation), each of which converts its feed at fixed
volumetric yields, bank in intermediate tankage, blend into finished grades, or
are sold as they are (refinery fuel, LPG, petcoke, cutter stock). Finished grades
must meet a quality window written on the blend — octane, vapour pressure,
sulfur, aromatics, cetane, cold flow, density, viscosity index — and are sold
into a demand window or carried in product tankage. The objective maximizes
margin: product and spot revenue less crude, purchased blendstock, unit operating
and inventory-holding cost.

# Fields
- `flowsheet::RefineryFlowsheet`: crude assays, streams, units and grades
- `data::ProcessPlanData`: prices, availability, capacities, tanks and demands
- `feasible_witness`: the planted operation, on requested-feasible instances
- `infeasibility_certificate`: the structural refutation, on requested-infeasible
  instances
- `feasibility_status::FeasibilityStatus`

The model is a pure LP: every operating mode is fixed at generation time, so
`relax_integer` has nothing to relax. See `process_planning/mode_switching` for
the same flowsheet with the mode decision left to the solver.
"""
struct RefineryPlanningProblem <: ProblemGenerator
    flowsheet::RefineryFlowsheet
    data::ProcessPlanData
    feasible_witness::Union{Nothing, RefineryOperatingPlan}
    infeasibility_certificate::Union{Nothing, RefineryInfeasibilityCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    RefineryPlanningProblem(target_variables, feasibility_status, seed)

Construct a multi-period refinery planning instance.

# Variable count

With `C` crudes, `T` periods, units `u` with feed sets `F_u`, grades `p` with
component sets `B_p`, `n_store` tanked streams, `n_buy` purchased blendstocks and
`n_spot` streams saleable as they are, the model has exactly

    T * (3C + sum_u (|F_u| + 1) + sum_p |B_p| + n_store + n_buy + n_spot + 2P)

variables. That count is affine in `C` for a fixed flowsheet, so
[`_pp_dimensions`](@ref) solves it for the crude count at each candidate horizon
and keeps the pair that lands closest to the target with an operationally
ordinary shape. Refinery complexity itself is set by the scale of the request
rather than by that search — see [`_pp_level_floor`](@ref) — so a small request
yields a topping or hydroskimming refinery with a coarse cut slate, and a large
one a cracking or full conversion refinery with parallel trains and many grades.

# Feasibility
- `feasible`: a complete operation is simulated through the flowsheet and every
  capacity, tank, availability, purchase limit, contract and blend window is
  placed around it, so `feasible_witness` is a feasible point that
  [`refinery_plan_satisfies`](@ref) re-checks row by row.
- `infeasible`: either the contracted volumes exceed everything the crude menu,
  the crude unit and the purchased blendstocks could be converted into over the
  horizon, or one grade's specification is tightened past every component that
  may enter it. Both are recorded in `infeasibility_certificate` and are proved
  from the model's own linear rows.
- `unknown`: assets are sized from engineering design rules rather than from the
  plan, and each quality window is stated at the edge of what the configuration
  supports rather than where the plan happens to land, so whether the slate, the
  units and the specifications can serve all the contracts together is genuinely
  open.
"""
function RefineryPlanningProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    target = max(target_variables, 1)

    _, skeleton, n_periods = _pp_dimensions(rng, target; mode_vars=false)
    flowsheet = _pp_build_flowsheet(rng, skeleton)
    mode_choice = ones(Int, n_units(flowsheet), n_periods)

    data, plan, certificate = _pp_plan_instance(
        rng,
        flowsheet,
        n_periods,
        feasibility_status,
        mode_choice;
        unknown_position=_pp_seed_position(seed),
    )

    problem = RefineryPlanningProblem(
        flowsheet,
        data,
        feasibility_status == feasible ? plan : nothing,
        certificate,
        feasibility_status,
    )

    if feasibility_status == feasible
        @assert refinery_plan_satisfies(flowsheet, data, plan)
    elseif feasibility_status == infeasible
        @assert refinery_certificate_holds(flowsheet, data, certificate)
    end
    return problem
end

"""
    _pp_add_blend_specifications!(model, fs, blend, n_periods)

Add the quality rows of every finished grade. A volumetric property `q` with an
upper bound `s` becomes `sum_b (Q[b,q] - s) x_b <= 0`, which is the linear form
of "the volume-weighted average stays below `s`"; RVP uses the monotone Chevron
index `RVP^1.25`, and a weight-basis property (sulfur) carries the component
density, giving the mass-weighted average.

Each row is divided by its largest coefficient. The specification is unchanged —
the right-hand side is zero — but the properties span very different units
(sulfur in weight ppm against octane numbers and densities), and writing every
quality row on the same scale keeps the constraint matrix from spanning seven
orders of magnitude.
"""
function _pp_add_blend_specifications!(
    model::Model, fs::RefineryFlowsheet, blend::Vector{<:AbstractArray}, T::Int
)
    for (p, product) in enumerate(fs.products)
        for q in 1:PP_N_QUALITIES
            weight = if _pp_is_weight_basis(q)
                [fs.qualities[s, PP_Q_DENSITY] for s in product.components]
            else
                ones(Float64, length(product.components))
            end
            for (bound, sense) in ((product.spec_min[q], :min), (product.spec_max[q], :max))
                isfinite(bound) || continue
                indexed_bound = q == PP_Q_RVP ? bound^1.25 : bound
                coefficient = [
                    weight[b] * (
                        (q == PP_Q_RVP ? fs.qualities[s, q]^1.25 : fs.qualities[s, q]) -
                        indexed_bound
                    ) for (b, s) in enumerate(product.components)
                ]
                largest = maximum(abs, coefficient)
                largest > 0 && (coefficient ./= largest)
                for t in 1:T
                    row = sum(
                        coefficient[b] * blend[p][b, t] for b in eachindex(product.components)
                    )
                    sense == :min ? @constraint(model, row >= 0) : @constraint(model, row <= 0)
                end
            end
        end
    end
    return nothing
end

"""
    _pp_add_renewable_blending!(model, fs, data, blend, n_periods)

Add a horizon renewable-fuel obligation and a per-period gasoline blend wall
when ethanol is present in the generated flowsheet. Both constraints are linear
because ethanol remains a segregated purchased component.
"""
function _pp_add_renewable_blending!(
    model::Model,
    fs::RefineryFlowsheet,
    data::ProcessPlanData,
    blend::Vector{<:AbstractArray},
    T::Int,
)
    gasoline_products = [
        p for
        p in eachindex(fs.products) if fs.products[p].key in (:regular_gasoline, :premium_gasoline)
    ]
    isempty(gasoline_products) && return nothing

    gasoline = [
        sum(
            blend[p][b, t] for p in gasoline_products for b in eachindex(fs.products[p].components)
        ) for t in 1:T
    ]
    renewable = [
        sum(
            blend[p][b, t] for p in gasoline_products for
            (b, s) in enumerate(fs.products[p].components) if fs.stream_classes[s] == :ethanol;
            init=0.0,
        ) for t in 1:T
    ]
    if data.renewable_min_fraction > 0
        @constraint(model, sum(renewable) >= data.renewable_min_fraction * sum(gasoline))
    end
    if data.renewable_max_fraction < 1
        @constraint(model, [t in 1:T], renewable[t] <= data.renewable_max_fraction * gasoline[t])
    end
    return nothing
end

"""
    build_model(prob::RefineryPlanningProblem)

Build the multi-period refinery planning LP. Deterministic — uses only the stored
flowsheet and data.

# Model
Variables per period: crude purchases, crude runs and crude tank levels; a feed
flow for every (unit, admissible stream) pair and the unit's throughput; a blend
flow for every (grade, admissible component) pair; intermediate tank levels;
blendstock purchases; spot sales of intermediate streams; and finished-product
sales and tank levels.

Constraints: crude tank balances and availability, crude-unit capacity, turndown
and charge sulfur, a balance row for every stream and period, unit throughput
definitions and turndown, product balances, the demand window, and the blend
specification rows.
"""
function build_model(prob::RefineryPlanningProblem)
    fs = prob.flowsheet
    data = prob.data
    C = fs.n_crudes
    S = n_streams(fs)
    U = n_units(fs)
    P = n_products(fs)
    T = data.n_periods

    model = Model()

    @variable(model, 0 <= crude_buy[c in 1:C, t in 1:T] <= data.crude_availability[c, t])
    # Implied but stated: no single crude can be charged beyond the crude unit's
    # capacity, no feed beyond its unit's, and nothing can be blended into a grade
    # beyond what that grade can sell or store. The bounds cut nothing off, and a
    # simplex given them does not have to discover them.
    @variable(model, 0 <= crude_run[c in 1:C, t in 1:T] <= data.cdu_capacity[t])
    @variable(model, 0 <= crude_inventory[c in 1:C, t in 1:T] <= data.crude_tank_capacity[c])
    @variable(
        model,
        0 <=
            throughput[u in 1:U, t in 1:T] <=
            data.unit_capacity[u, t] * fs.units[u].modes[1].capacity_factor
    )
    @variable(
        model, 0 <= stream_inventory[s in fs.storable, t in 1:T] <= data.stream_tank_capacity[s]
    )
    @variable(model, 0 <= purchase[s in fs.purchasable, t in 1:T] <= data.stream_purchase_limit[s])
    @variable(model, 0 <= spot_sale[s in fs.spot, t in 1:T] <= data.stream_spot_limit[s])
    @variable(model, data.demand_min[p, t] <= sales[p in 1:P, t in 1:T] <= data.demand_max[p, t])
    @variable(model, 0 <= product_inventory[p in 1:P, t in 1:T] <= data.product_tank_capacity[p])
    feed = [
        @variable(
            model,
            [f in 1:length(fs.units[u].feeds), t in 1:T],
            lower_bound = 0,
            upper_bound = data.unit_capacity[u, t] * fs.units[u].modes[1].capacity_factor,
            base_name = "feed_$(fs.units[u].name)"
        ) for u in 1:U
    ]
    blend = [
        @variable(
            model,
            [b in 1:length(fs.products[p].components), t in 1:T],
            lower_bound = 0,
            upper_bound = data.demand_max[p, t] + data.product_tank_capacity[p],
            base_name = "blend_$(fs.products[p].name)"
        ) for p in 1:P
    ]
    # Extension variants (notably `hydrogen_network`) need the physically
    # resolved feeds and blends. Register these anonymous ragged containers in
    # the model dictionary without changing the public variable names.
    model[:unit_feed] = feed
    model[:blend_flow] = blend

    # Crude tanks, crude-unit capacity and charge quality.
    for t in 1:T
        for c in 1:C
            previous = t == 1 ? data.crude_initial_inventory[c] : crude_inventory[c, t - 1]
            @constraint(
                model, previous + crude_buy[c, t] == crude_run[c, t] + crude_inventory[c, t]
            )
        end
        @constraint(model, sum(crude_run[c, t] for c in 1:C) <= data.cdu_capacity[t])
        data.cdu_min_throughput[t] > 0 &&
            @constraint(model, sum(crude_run[c, t] for c in 1:C) >= data.cdu_min_throughput[t])
        @constraint(
            model,
            sum((fs.crude_sulfur[c] - data.cdu_sulfur_limit) * crude_run[c, t] for c in 1:C) <= 0
        )
    end

    # Stream balances, assembled term by term.
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
        mode = unit.modes[1]
        for (f, s) in enumerate(unit.feeds), t in 1:T
            add_to_expression!(balance[s, t], -1.0, feed[u][f, t])
        end
        for f in eachindex(unit.feeds), (o, out) in enumerate(unit.outputs)
            yield = mode.yields[f, o]
            yield == 0.0 && continue
            for t in 1:T
                add_to_expression!(balance[out, t], yield, feed[u][f, t])
            end
        end
        for t in 1:T
            @constraint(
                model, throughput[u, t] == sum(feed[u][f, t] for f in eachindex(unit.feeds))
            )
            data.unit_min_throughput[u, t] > 0 &&
                @constraint(model, throughput[u, t] >= data.unit_min_throughput[u, t])
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

    # Finished-product balances and blend specifications.
    for p in 1:P, t in 1:T
        previous = t == 1 ? data.product_initial_inventory[p] : product_inventory[p, t - 1]
        @constraint(
            model,
            previous + sum(blend[p][b, t] for b in eachindex(fs.products[p].components)) ==
                sales[p, t] + product_inventory[p, t]
        )
    end
    _pp_add_blend_specifications!(model, fs, blend, T)
    _pp_add_renewable_blending!(model, fs, data, blend, T)

    @objective(
        model,
        Max,
        sum(data.product_price[p, t] * sales[p, t] for p in 1:P, t in 1:T) +
        sum(data.stream_spot_price[s] * spot_sale[s, t] for s in fs.spot, t in 1:T) -
        sum(data.crude_price[c, t] * crude_buy[c, t] for c in 1:C, t in 1:T) -
        sum(data.stream_purchase_cost[s, t] * purchase[s, t] for s in fs.purchasable, t in 1:T) -
        sum(fs.units[u].modes[1].operating_cost * throughput[u, t] for u in 1:U, t in 1:T) -
        sum(data.stream_holding_cost[s] * stream_inventory[s, t] for s in fs.storable, t in 1:T) -
            sum(data.product_holding_cost[p] * product_inventory[p, t] for p in 1:P, t in 1:T)
    )

    return model
end

register_variant(
    :process_planning,
    :refinery,
    RefineryPlanningProblem,
    "Multi-period refinery production planning: crude purchase and charge, " *
    "assay-driven distillation cuts, fixed-yield conversion units, intermediate " *
    "tankage, and component blending into specification-constrained grades";
    default=true,
)
