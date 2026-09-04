using JuMP
using Random
using Distributions

"""Hydrogen, sulfur-recovery, and carbon data attached to a refinery plan."""
struct RefineryHydrogenData
    unit_h2_rate::Vector{Vector{Float64}}       # tonne H2 / kbbl feed
    reformer_h2_yield::Vector{Float64}          # tonne H2 / kbbl throughput
    sulfur_removal_rate::Vector{Vector{Float64}} # tonne sulfur / kbbl feed
    smr_capacity::Vector{Float64}
    import_capacity::Vector{Float64}
    h2_storage_capacity::Float64
    initial_h2_inventory::Float64
    sru_capacity::Vector{Float64}
    smr_emission_factor::Float64                # tonne CO2 / tonne H2
    unit_emission_rate::Vector{Float64}          # tonne CO2 / kbbl throughput
    carbon_cap::Vector{Float64}
    cumulative_carbon_cap::Float64
    smr_cost::Vector{Float64}
    import_cost::Vector{Float64}
    h2_holding_cost::Float64
    vent_penalty::Float64
    carbon_price::Float64
end

"""Complete extension of a planted refinery operation into H2/SRU/CO2 flows."""
struct RefineryHydrogenPlan
    refinery::RefineryOperatingPlan
    smr_h2::Vector{Float64}
    imported_h2::Vector{Float64}
    h2_inventory::Vector{Float64}
    vented_h2::Vector{Float64}
    sulfur_recovered::Vector{Float64}
    carbon_emissions::Vector{Float64}
end

"""
    RefineryHydrogenPlanningProblem <: ProblemGenerator

Multi-period clean-fuels refinery LP built on the same assay-origin material
balances as `process_planning/refinery`. Hydrotreater and hydrocracker feed flows
consume hydrogen, reformer throughput produces by-product hydrogen, and the
remaining demand is served by an emitting SMR, merchant/green imports, or H2
inventory. Sulfur recovered is calculated from the actual sulfur carried by
hydroprocessing feeds, and process plus SMR emissions must satisfy period and
horizon carbon caps.

Because every environmental term is linked to the refinery's physical feed or
throughput variables, products cannot bypass hydrogen, sulfur recovery, or
carbon accounting. Requested-feasible instances store and verify the full
extended operating point; requested-infeasible instances inherit the refinery's
solver-independent volume/specification certificate.
"""
struct RefineryHydrogenPlanningProblem <: ProblemGenerator
    flowsheet::RefineryFlowsheet
    data::ProcessPlanData
    hydrogen::RefineryHydrogenData
    feasible_witness::Union{Nothing, RefineryHydrogenPlan}
    infeasibility_certificate::Union{Nothing, RefineryInfeasibilityCertificate}
    feasibility_status::FeasibilityStatus
end

const _PP_HYDROPROCESSING_H2_RANGE = Dict(
    :naphtha_ht => (0.12, 0.80),
    :kero_ht => (0.30, 1.40),
    :vgo_ht => (1.20, 2.90),
    :resid_ht => (2.00, 4.50),
    :diesel_ht => (0.80, 2.90),
    :hydrocracker => (2.40, 6.00),
)

_pp_is_hydroprocessing(key::Symbol) = haskey(_PP_HYDROPROCESSING_H2_RANGE, key)

"""Sulfur removed by one kbbl of a unit feed, respecting volume and density."""
function _pp_sulfur_removal_rate(fs::RefineryFlowsheet, unit::RefineryUnit, feed_index::Int)
    _pp_is_hydroprocessing(unit.key) || return 0.0
    input = unit.feeds[feed_index]
    input_mass_ppm = fs.qualities[input, PP_Q_DENSITY] * fs.qualities[input, PP_Q_SULFUR]
    mode = unit.modes[1]
    output_mass_ppm = sum(
        mode.yields[feed_index, o] * fs.qualities[s, PP_Q_DENSITY] * fs.qualities[s, PP_Q_SULFUR]
        for (o, s) in enumerate(unit.outputs)
    )
    # One kbbl is 158.987 m^3; with specific gravity as tonne/m^3 and ppm as
    # tonne/10^6 tonne, the result is tonnes of sulfur per kbbl.
    return 158.987e-6 * max(input_mass_ppm - output_mass_ppm, 0.0)
end

function _pp_hydrogen_extensions(
    rng::AbstractRNG,
    fs::RefineryFlowsheet,
    data::ProcessPlanData,
    plan::RefineryOperatingPlan,
    status::FeasibilityStatus,
    unknown_position::Float64=0.5,
)
    U, T = n_units(fs), data.n_periods
    0.0 <= unknown_position <= 1.0 || throw(ArgumentError("unknown_position must lie in [0, 1]"))
    unit_h2_rate = Vector{Vector{Float64}}(undef, U)
    sulfur_rate = Vector{Vector{Float64}}(undef, U)
    reformer_yield = zeros(Float64, U)
    unit_emission = zeros(Float64, U)
    for u in 1:U
        unit = fs.units[u]
        unit_h2_rate[u] = zeros(Float64, length(unit.feeds))
        sulfur_rate[u] = zeros(Float64, length(unit.feeds))
        if _pp_is_hydroprocessing(unit.key)
            low, high = _PP_HYDROPROCESSING_H2_RANGE[unit.key]
            for f in eachindex(unit.feeds)
                sulfur = fs.qualities[unit.feeds[f], PP_Q_SULFUR]
                severity = clamp(log10(1 + sulfur) / 5, 0.0, 1.0)
                unit_h2_rate[u][f] = rand(rng, Uniform(low, high)) * (0.75 + 0.50 * severity)
                sulfur_rate[u][f] = _pp_sulfur_removal_rate(fs, unit, f)
            end
        elseif unit.key == :reformer
            reformer_yield[u] = rand(rng, Uniform(0.65, 1.35))
        end
        unit_emission[u] = rand(rng, Uniform(0.015, 0.080))
    end

    h2_demand = zeros(Float64, T)
    byproduct = zeros(Float64, T)
    sulfur = zeros(Float64, T)
    throughput = _pp_plan_throughput(fs, plan)
    for u in 1:U, t in 1:T
        byproduct[t] += reformer_yield[u] * throughput[u, t]
        for f in eachindex(fs.units[u].feeds)
            flow = plan.unit_feed[u][f, t]
            h2_demand[t] += unit_h2_rate[u][f] * flow
            sulfur[t] += sulfur_rate[u][f] * flow
        end
    end

    smr = zeros(Float64, T)
    imported = zeros(Float64, T)
    inventory = zeros(Float64, T)
    vented = zeros(Float64, T)
    for t in 1:T
        deficit = max(h2_demand[t] - byproduct[t], 0.0)
        smr[t] = 0.72 * deficit
        imported[t] = deficit - smr[t]
        vented[t] = max(byproduct[t] - h2_demand[t], 0.0)
    end

    # Size an ordinary hydrogen plant from installed hydroprocessing capacity,
    # its sampled feed-specific consumption, and expected reformer by-product.
    # This keeps unknown instances on the physical tonne-H2/kbbl scale instead
    # of accidentally making the H2 system orders of magnitude too small.
    gross_design_h2 =
        data.nameplate * sum(
            _pp_unit_template(fs.units[u].key).capacity_fraction *
            (isempty(unit_h2_rate[u]) ? 0.0 : sum(unit_h2_rate[u]) / length(unit_h2_rate[u])) for
            u in 1:U
        )
    reformer_design_h2 =
        data.nameplate *
        sum(_pp_unit_template(fs.units[u].key).capacity_fraction * reformer_yield[u] for u in 1:U)
    design_h2 = max(gross_design_h2 - reformer_design_h2, 0.05 * data.nameplate, 0.25)
    design_sulfur = max(
        data.nameplate * sum(
            _pp_unit_template(fs.units[u].key).capacity_fraction *
            (isempty(sulfur_rate[u]) ? 0.0 : sum(sulfur_rate[u]) / length(sulfur_rate[u])) for
            u in 1:U
        ),
        0.0005 * data.nameplate,
        0.01,
    )

    smr_capacity = zeros(Float64, T)
    import_capacity = zeros(Float64, T)
    sru_capacity = zeros(Float64, T)
    for t in 1:T
        if status == feasible
            smr_capacity[t] = max(design_h2, smr[t] * rand(rng, Uniform(1.10, 1.45)))
            import_capacity[t] = max(0.35 * design_h2, imported[t] * rand(rng, Uniform(1.10, 1.60)))
            sru_capacity[t] = max(design_sulfur, sulfur[t] * rand(rng, Uniform(1.10, 1.50)))
        else
            scenario = 0.75 + 0.50 * unknown_position
            smr_capacity[t] = design_h2 * rand(rng, Uniform(0.58, 0.88)) * scenario
            import_capacity[t] = design_h2 * rand(rng, Uniform(0.18, 0.45)) * scenario
            sru_capacity[t] = design_sulfur * rand(rng, Uniform(0.75, 1.25)) * scenario
        end
    end
    storage_capacity = max(0.15 * maximum(h2_demand; init=0.0), 0.10)
    initial_inventory = status == feasible ? 0.0 : storage_capacity * rand(rng, Uniform(0.0, 0.35))

    smr_emission_factor = rand(rng, Uniform(8.5, 11.5))
    carbon = [
        smr_emission_factor * smr[t] + sum(unit_emission[u] * throughput[u, t] for u in 1:U) for
        t in 1:T
    ]
    carbon_cap = [
        if status == feasible
            carbon[t] * rand(rng, Uniform(1.08, 1.35)) + 1e-4
        else
            max(0.06 * data.nameplate, carbon[t] * rand(rng, Uniform(0.75, 1.25)))
        end for t in 1:T
    ]
    cumulative_cap = if status == feasible
        sum(carbon) * rand(rng, Uniform(1.08, 1.30)) + 1e-4
    else
        sum(carbon_cap) * (0.76 + 0.28 * unknown_position)
    end

    hdata = RefineryHydrogenData(
        unit_h2_rate,
        reformer_yield,
        sulfur_rate,
        smr_capacity,
        import_capacity,
        storage_capacity,
        initial_inventory,
        sru_capacity,
        smr_emission_factor,
        unit_emission,
        carbon_cap,
        cumulative_cap,
        [rand(rng, Uniform(1.0, 2.2)) for _ in 1:T],
        [rand(rng, Uniform(2.0, 5.5)) for _ in 1:T],
        rand(rng, Uniform(0.02, 0.12)),
        rand(rng, Uniform(0.5, 2.0)),
        rand(rng, Uniform(0.0, 0.12)),
    )
    witness = RefineryHydrogenPlan(plan, smr, imported, inventory, vented, sulfur, carbon)
    return hdata, witness
end

"""
    RefineryHydrogenPlanningProblem(target_variables, feasibility_status, seed)

Construct the clean-fuels extension. The model has exactly the shared fixed-mode
refinery variable count plus `6T` variables for SMR production, imports, H2
inventory, venting, recovered sulfur, and carbon emissions.

A requested-feasible instance sizes every added capacity and emissions cap above
the extended planted operation, checked by
[`refinery_hydrogen_plan_satisfies`](@ref). A requested-infeasible instance uses
one of the base refinery's volume/specification certificates; because the
extension only adds constraints, that certificate remains valid. Unknown data
uses independent engineering capacities and carbon limits.
"""
function RefineryHydrogenPlanningProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    target = max(target_variables, 1)
    _, skeleton, T = _pp_dimensions(
        rng,
        target;
        mode_vars=false,
        minimum_level=2,
        extra_variables=6,
        compact_hydroprocessing=true,
    )
    flowsheet = _pp_build_flowsheet(rng, skeleton)
    mode_choice = ones(Int, n_units(flowsheet), T)
    data, refinery_plan, certificate = _pp_plan_instance(
        rng, flowsheet, T, feasibility_status, mode_choice; unknown_position=_pp_seed_position(seed)
    )
    hydrogen, hydrogen_plan = _pp_hydrogen_extensions(
        rng, flowsheet, data, refinery_plan, feasibility_status, _pp_seed_position(seed)
    )
    problem = RefineryHydrogenPlanningProblem(
        flowsheet,
        data,
        hydrogen,
        feasibility_status == feasible ? hydrogen_plan : nothing,
        certificate,
        feasibility_status,
    )
    feasibility_status == feasible && @assert refinery_hydrogen_plan_satisfies(problem)
    feasibility_status == infeasible &&
        @assert refinery_certificate_holds(flowsheet, data, certificate)
    return problem
end

"""Check the complete planted H2/SRU/CO2 extension without a solver."""
function refinery_hydrogen_plan_satisfies(prob::RefineryHydrogenPlanningProblem; atol::Float64=1e-6)
    plan = prob.feasible_witness
    plan === nothing && return false
    fs, data, hd = prob.flowsheet, prob.data, prob.hydrogen
    refinery_plan_satisfies(fs, data, plan.refinery; atol=atol) || return false
    T, U = data.n_periods, n_units(fs)
    throughput = _pp_plan_throughput(fs, plan.refinery)
    previous = hd.initial_h2_inventory
    for t in 1:T
        demand = 0.0
        sulfur = 0.0
        byproduct = 0.0
        for u in 1:U
            byproduct += hd.reformer_h2_yield[u] * throughput[u, t]
            for f in eachindex(fs.units[u].feeds)
                flow = plan.refinery.unit_feed[u][f, t]
                demand += hd.unit_h2_rate[u][f] * flow
                sulfur += hd.sulfur_removal_rate[u][f] * flow
            end
        end
        scale = max(1.0, demand, byproduct)
        abs(
            previous + plan.smr_h2[t] + plan.imported_h2[t] + byproduct - demand -
            plan.h2_inventory[t] - plan.vented_h2[t],
        ) <= atol * scale || return false
        plan.smr_h2[t] <= hd.smr_capacity[t] + atol || return false
        plan.imported_h2[t] <= hd.import_capacity[t] + atol || return false
        plan.h2_inventory[t] <= hd.h2_storage_capacity + atol || return false
        abs(plan.sulfur_recovered[t] - sulfur) <= atol * max(1.0, sulfur) || return false
        sulfur <= hd.sru_capacity[t] + atol || return false
        emissions =
            hd.smr_emission_factor * plan.smr_h2[t] +
            sum(hd.unit_emission_rate[u] * throughput[u, t] for u in 1:U)
        abs(plan.carbon_emissions[t] - emissions) <= atol * max(1.0, emissions) || return false
        emissions <= hd.carbon_cap[t] + atol || return false
        previous = plan.h2_inventory[t]
    end
    sum(plan.carbon_emissions) <= hd.cumulative_carbon_cap + atol || return false
    return true
end

function build_model(prob::RefineryHydrogenPlanningProblem)
    base = RefineryPlanningProblem(
        prob.flowsheet,
        prob.data,
        prob.feasible_witness === nothing ? nothing : prob.feasible_witness.refinery,
        prob.infeasibility_certificate,
        prob.feasibility_status,
    )
    model = build_model(base)
    fs, data, hd = prob.flowsheet, prob.data, prob.hydrogen
    U, T = n_units(fs), data.n_periods
    feed = model[:unit_feed]
    throughput = model[:throughput]

    @variable(model, 0 <= smr_h2[t in 1:T] <= hd.smr_capacity[t])
    @variable(model, 0 <= imported_h2[t in 1:T] <= hd.import_capacity[t])
    @variable(model, 0 <= h2_inventory[t in 1:T] <= hd.h2_storage_capacity)
    @variable(model, vented_h2[t in 1:T] >= 0)
    @variable(model, 0 <= sulfur_recovered[t in 1:T] <= hd.sru_capacity[t])
    @variable(model, 0 <= carbon_emissions[t in 1:T] <= hd.carbon_cap[t])

    @expression(
        model,
        h2_demand[t in 1:T],
        sum(
            hd.unit_h2_rate[u][f] * feed[u][f, t] for u in 1:U for f in eachindex(fs.units[u].feeds)
        )
    )
    @expression(
        model, reformer_h2[t in 1:T], sum(hd.reformer_h2_yield[u] * throughput[u, t] for u in 1:U)
    )
    @constraint(
        model,
        [t in 1:T],
        (t == 1 ? hd.initial_h2_inventory : h2_inventory[t - 1]) +
        smr_h2[t] +
        imported_h2[t] +
        reformer_h2[t] == h2_demand[t] + h2_inventory[t] + vented_h2[t]
    )
    @constraint(
        model,
        [t in 1:T],
        sulfur_recovered[t] == sum(
            hd.sulfur_removal_rate[u][f] * feed[u][f, t] for u in 1:U for
            f in eachindex(fs.units[u].feeds)
        )
    )
    @constraint(
        model,
        [t in 1:T],
        carbon_emissions[t] ==
            hd.smr_emission_factor * smr_h2[t] +
        sum(hd.unit_emission_rate[u] * throughput[u, t] for u in 1:U)
    )
    @constraint(model, sum(carbon_emissions) <= hd.cumulative_carbon_cap)

    base_objective = objective_function(model)
    @objective(
        model,
        Max,
        base_objective - sum(
            hd.smr_cost[t] * smr_h2[t] +
            hd.import_cost[t] * imported_h2[t] +
            hd.h2_holding_cost * h2_inventory[t] +
            hd.vent_penalty * vented_h2[t] +
            hd.carbon_price * carbon_emissions[t] for t in 1:T
        )
    )

    if prob.feasible_witness !== nothing
        witness = prob.feasible_witness
        for t in 1:T
            set_start_value(smr_h2[t], witness.smr_h2[t])
            set_start_value(imported_h2[t], witness.imported_h2[t])
            set_start_value(h2_inventory[t], witness.h2_inventory[t])
            set_start_value(vented_h2[t], witness.vented_h2[t])
            set_start_value(sulfur_recovered[t], witness.sulfur_recovered[t])
            set_start_value(carbon_emissions[t], witness.carbon_emissions[t])
        end
    end
    return model
end

register_variant(
    :process_planning,
    :hydrogen_network,
    RefineryHydrogenPlanningProblem,
    "Clean-fuels refinery LP with feed-linked hydrogen demand, reformer H2, " *
    "SMR/import/storage decisions, sulfur recovery, and carbon caps",
)
