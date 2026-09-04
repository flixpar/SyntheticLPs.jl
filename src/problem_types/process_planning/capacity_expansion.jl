using JuMP
using Random
using Distributions

# Long-range planning of a chemical process network: which processes to build or
# expand, and how hard to run them, over a multi-year horizon. The formulation
# follows the multiperiod capacity-expansion MILP of Sahinidis, Grossmann,
# Fornari & Chathrathi (Comput. Chem. Eng. 13, 1989) and Sahinidis & Grossmann
# (Oper. Res. 40, 1992): capacity carries forward and grows by discrete
# expansions, operating levels are bounded by installed capacity, chemicals
# balance across the network at fixed conversion ratios, and the objective is
# the discounted net present value of operation less investment.
#
# Scale conventions: flows and capacities are thousand tonnes per period,
# chemical prices and operating costs are dollars per tonne (so revenue and cost
# are in thousands of dollars), and investment is quoted in the same thousands.

"""A chemical in the network: where it sits, and whether it can be bought or sold."""
struct ProcessChemical
    name::Symbol
    layer::Int
    purchasable::Bool
    sellable::Bool
end

"""
A process technology.

`main_output` is the chemical the operating level is measured in;
`outputs` holds every produced chemical with its positive yield (the main output
plus any dead-end byproduct), and `inputs` the consumption per unit of operating
level. Investment is the usual fixed-plus-linear cost of an expansion.
"""
struct ProcessTechnology
    name::Symbol
    layer::Int
    main_output::Int
    outputs::Vector{Pair{Int,Float64}}
    inputs::Vector{Pair{Int,Float64}}
    operating_cost::Float64
    fixed_investment::Float64
    variable_investment::Float64
    existing_capacity::Float64
    min_expansion::Float64
    max_expansion::Float64
end

"""
    ProcessExpansionPlan

A complete primal point of the capacity-expansion model: the operating level,
installed capacity, expansion and expansion indicator of every process in every
period, plus the purchases and sales that balance the network.
"""
struct ProcessExpansionPlan
    operating_level::Matrix{Float64}
    capacity::Matrix{Float64}
    expansion::Matrix{Float64}
    expand::Matrix{Int}
    purchase::Matrix{Float64}
    sales::Matrix{Float64}
end

"""Structural reason a requested-infeasible expansion instance has no plan."""
@enum ProcessExpansionInfeasibilityKind begin
    expansion_demand_above_capacity_bound
    expansion_demand_above_feedstock_bound
end

"""
    ProcessExpansionCertificate

Solver-independent proof stored on a requested-infeasible instance.

`expansion_demand_above_capacity_bound` names one chemical whose contracted
sales exceed what every process that makes it could produce even if each were
expanded by the largest permitted amount in every period: chaining
`W <= Q`, `Q_t = Q_{t-1} + QE_t` and `QE_t <= QE_max` bounds the operating levels
period by period, and the balance row for that chemical turns that into a bound
on its sales.

`expansion_demand_above_feedstock_bound` is the network-wide version. Each
chemical is given a value `v >= 0` with `v >= 1` for anything saleable and, for
every process, output value no greater than input value. Multiplying the balance
rows by those values and summing cancels the operating levels, leaving total
sales bounded by the value of the raw material the market can supply.
"""
struct ProcessExpansionCertificate
    kind::ProcessExpansionInfeasibilityKind
    chemical::Int
    achievable::Float64
    required::Float64
end

"""
    ProcessCapacityExpansionProblem <: ProblemGenerator

Multi-period capacity expansion of a chemical process network: the long-range
investment plan behind the operating plans of `process_planning/refinery`.

# Formulation

Processes convert chemicals at fixed ratios. A process's capacity carries
forward from period to period and grows only through an expansion, which incurs
a fixed charge plus a linear cost and must fall between a minimum economic size
and a maximum permitted size when it happens; the operating level never exceeds
the installed capacity. Every chemical balances in every period: what the
processes make, plus purchases of raw material, equals what they consume plus
sales. Raw materials are limited by market availability, finished chemicals by a
demand window with a contracted floor. The objective maximizes discounted net
present value: sales revenue less feedstock, operating and investment cost.

# Fields
- `chemicals`, `technologies`: the process network
- `purchase_cost`, `availability`, `sale_price`, `demand_min`, `demand_max`,
  `discount`: the market over the horizon
- `feasible_witness`, `infeasibility_certificate`, `feasibility_status`

Expansion indicators are binary, so this is a genuine MILP; with the package
default `relax_integer=true` it is returned as its LP relaxation, in which a
fractional indicator buys a fractionally-sized expansion. The planted witness
and both certificates are valid for the relaxation as well as for the MILP.
"""
struct ProcessCapacityExpansionProblem <: ProblemGenerator
    n_periods::Int
    chemicals::Vector{ProcessChemical}
    technologies::Vector{ProcessTechnology}
    raw_chemicals::Vector{Int}
    sellable_chemicals::Vector{Int}
    purchase_cost::Matrix{Float64}
    availability::Matrix{Float64}
    sale_price::Matrix{Float64}
    demand_min::Matrix{Float64}
    demand_max::Matrix{Float64}
    discount::Vector{Float64}
    feasible_witness::Union{Nothing,ProcessExpansionPlan}
    infeasibility_certificate::Union{Nothing,ProcessExpansionCertificate}
    feasibility_status::FeasibilityStatus
end

n_chemicals(prob::ProcessCapacityExpansionProblem) = length(prob.chemicals)
n_technologies(prob::ProcessCapacityExpansionProblem) = length(prob.technologies)

"""Variables per period: operating level, capacity, expansion and its indicator, plus trade."""
_pp_expansion_variables(n_tech::Int, n_raw::Int, n_sell::Int) =
    4 * n_tech + n_raw + n_sell

"""Number of processes carrying a byproduct, and of intermediates that also trade."""
_pp_expansion_byproducts(n_tech::Int, rate::Float64) = round(Int, rate * n_tech)
_pp_expansion_traded_intermediates(n_intermediate::Int) = fld(n_intermediate, 3)

"""
    _pp_expansion_sellable(n_tech, byproduct_rate) -> Int

Saleable chemicals of a network with `n_tech` processes: the finished slate, the
intermediates that also trade, and one chemical per byproduct-bearing process.
Both counts are fixed rather than sampled per process, so the variable count is
an exact function of the sizing decision.
"""
function _pp_expansion_sellable(n_tech::Int, byproduct_rate::Float64)
    _, _, _, n_intermediate, n_final = _pp_expansion_shape(n_tech)
    return n_final + _pp_expansion_traded_intermediates(n_intermediate) +
           _pp_expansion_byproducts(n_tech, byproduct_rate)
end

"""
    _pp_expansion_shape(n_tech) -> (n_layers, per_layer, n_raw, n_intermediate, n_final)

Chemical inventory of a network with `n_tech` processes: a layered slate of raw
materials, intermediates and finished chemicals, sized so that roughly two
processes compete to make each producible chemical. A pure function of the
process count, so the variable count can be evaluated before any data is drawn.
"""
function _pp_expansion_shape(n_tech::Int)
    n_layers = clamp(2 + fld(n_tech, 6), 2, 5)
    per_layer = max(1, cld(n_tech, 2 * max(n_layers - 1, 1)))
    n_raw = per_layer
    n_intermediate = per_layer * max(n_layers - 2, 0)
    n_final = per_layer
    return n_layers, per_layer, n_raw, n_intermediate, n_final
end

"""
    _pp_expansion_dimensions(rng, target) -> (n_tech, n_periods, byproduct_rate)

Pick the number of processes and the horizon so the variable count lands on the
target. The per-period block is `4 I + n_raw + n_sell`, which grows with the
process count in a fixed pattern, so the count is evaluated exactly for a few
candidate process counts at every candidate horizon.
"""
function _pp_expansion_dimensions(rng::AbstractRNG, target::Int)
    horizon_pref = clamp(round(Int, 2.0 * log10(max(target, 10))) + 2, 5, 15)
    byproduct_rate = rand(rng, Uniform(0.15, 0.45))
    best = (2, 5)
    best_score = (Inf, Inf)
    for T in 3:25
        approximate = max(1, round(Int, target / (4.6 * T)))
        for n_tech in unique(clamp.(approximate .+ (-2:2), 2, 200_000))
            n_sell = _pp_expansion_sellable(n_tech, byproduct_rate)
            _, _, n_raw, _, _ = _pp_expansion_shape(n_tech)
            total = _pp_expansion_variables(n_tech, n_raw, n_sell) * T
            err = abs(total - target) / target
            shape = abs(T - horizon_pref) / 15
            score = (round(err, digits=3), shape)
            if score < best_score
                best_score = score
                best = (n_tech, T)
            end
        end
    end
    return best[1], best[2], byproduct_rate
end

"""
    _pp_expansion_network(rng, n_tech, byproduct_rate) -> (chemicals, technologies)

Build a layered process network: raw materials at the bottom, then intermediates,
then finished chemicals, with every process consuming one to three chemicals from
strictly lower layers and producing one chemical at its own layer (plus, some of
the time, a dead-end byproduct that is only ever sold). Conversion is set by a
mass yield in the 60-95% range typical of continuous chemical processes, and
investment cost carries the usual fixed charge plus linear term.
"""
function _pp_expansion_network(rng::AbstractRNG, n_tech::Int, byproduct_rate::Float64)
    n_layers, per_layer, n_raw, n_intermediate, n_final = _pp_expansion_shape(n_tech)

    chemicals = ProcessChemical[]
    layer_members = [Int[] for _ in 1:n_layers]
    for _ in 1:n_raw
        push!(chemicals, ProcessChemical(Symbol(:raw_, length(chemicals) + 1), 1,
                                         true, false))
        push!(layer_members[1], length(chemicals))
    end
    intermediate_index = 0
    # A fixed share of the intermediates also trade on the open market; which
    # ones is random, how many is not, so the variable count stays exact.
    traded = _pp_expansion_traded_intermediates(n_intermediate)
    traded_set = Set(shuffle(rng, collect(1:max(n_intermediate, 1)))[1:traded])
    for layer in 2:(n_layers - 1), _ in 1:per_layer
        intermediate_index += 1
        push!(chemicals, ProcessChemical(Symbol(:intermediate_, length(chemicals) + 1),
                                         layer, false, intermediate_index in traded_set))
        push!(layer_members[layer], length(chemicals))
    end
    for _ in 1:n_final
        push!(chemicals, ProcessChemical(Symbol(:product_, length(chemicals) + 1),
                                         n_layers, false, true))
        push!(layer_members[n_layers], length(chemicals))
    end

    technologies = ProcessTechnology[]
    byproduct_set = Set(shuffle(rng, collect(1:n_tech))[
        1:_pp_expansion_byproducts(n_tech, byproduct_rate)])
    for i in 1:n_tech
        # Deal processes round-robin over the producing layers so every chemical
        # has a maker before any gets a second one.
        layer = 2 + (i - 1) % (n_layers - 1)
        members = layer_members[layer]
        main = members[1 + (i - 1) ÷ (n_layers - 1) % length(members)]
        lower = vcat(layer_members[1:(layer - 1)]...)
        n_inputs = min(length(lower), rand(rng, 1:3))
        chosen = shuffle(rng, lower)[1:n_inputs]
        yield = rand(rng, Uniform(0.60, 0.95))
        weights = rand(rng, Dirichlet(fill(2.0, n_inputs)))
        inputs = [chosen[k] => round(weights[k] / yield, digits=4)
                  for k in 1:n_inputs]
        outputs = [main => 1.0]
        if i in byproduct_set
            byproduct = ProcessChemical(Symbol(:byproduct_, length(chemicals) + 1),
                                        layer, false, true)
            push!(chemicals, byproduct)
            push!(outputs, length(chemicals) => round(rand(rng, Uniform(0.05, 0.35)),
                                                      digits=3))
        end
        push!(technologies, ProcessTechnology(
            Symbol(:process_, i), layer, main, outputs, inputs,
            round(rand(rng, Uniform(20.0, 120.0)), digits=2),
            round(rand(rng, Uniform(8_000.0, 60_000.0)), digits=1),
            round(rand(rng, Uniform(300.0, 1_500.0)), digits=2),
            0.0, 0.0, 0.0))
    end
    return chemicals, technologies
end

"""
    _pp_expansion_potential(chemicals, technologies) -> Vector{Float64}

Value `v[j] >= 0` for every chemical with `v >= 1` on anything saleable and, for
every process, output value no greater than input value. Computed in one pass
from the top layer down, which is exact here because a byproduct is never an
input, so no chemical's value can rise after the processes that consume it have
been visited.
"""
function _pp_expansion_potential(chemicals::Vector{ProcessChemical},
                                 technologies::Vector{ProcessTechnology})
    value = [chemical.sellable ? 1.0 : 0.0 for chemical in chemicals]
    isempty(technologies) && return value
    for layer in maximum(t.layer for t in technologies):-1:2
        for technology in technologies
            technology.layer == layer || continue
            produced = sum(coefficient * value[j] for (j, coefficient) in technology.outputs)
            for (j, coefficient) in technology.inputs
                coefficient <= 0 && continue
                value[j] = max(value[j], produced / coefficient)
            end
        end
    end
    return value
end

"""
    _pp_expansion_capacity_bound(prob, chemical) -> Float64

Largest volume of `chemical` the network could ever sell over the horizon: every
process that makes it running at the capacity it would have after expanding by
the maximum permitted amount in every period.
"""
function _pp_expansion_capacity_bound(prob::ProcessCapacityExpansionProblem,
                                      chemical::Int)
    bound = 0.0
    for t in 1:prob.n_periods
        for technology in prob.technologies
            index = findfirst(pair -> pair.first == chemical, technology.outputs)
            index === nothing && continue
            capacity = technology.existing_capacity + t * technology.max_expansion
            bound += technology.outputs[index].second * capacity
        end
    end
    return bound
end

"""Largest total saleable volume the raw-material market can support (see the certificate)."""
function _pp_expansion_feedstock_bound(prob::ProcessCapacityExpansionProblem)
    value = _pp_expansion_potential(prob.chemicals, prob.technologies)
    return sum(value[j] * prob.availability[j, t]
               for j in prob.raw_chemicals, t in 1:prob.n_periods)
end

"""
    process_expansion_plan_satisfies(prob, plan=prob.feasible_witness; atol=1e-6)

Re-check a planted expansion plan against every row: the capacity recursion, the
expansion window and its indicator, the operating-level bound, every chemical
balance, raw-material availability and the demand window. Solver-independent.
"""
function process_expansion_plan_satisfies(
    prob::ProcessCapacityExpansionProblem,
    plan::Union{Nothing,ProcessExpansionPlan}=prob.feasible_witness;
    atol::Float64=1e-6,
)
    plan === nothing && return false
    I = n_technologies(prob)
    J = n_chemicals(prob)
    T = prob.n_periods
    size(plan.operating_level) == (I, T) || return false
    scale = max(1.0, maximum(prob.demand_max; init=1.0))
    tol = atol * scale

    all(>=(-tol), plan.operating_level) || return false
    all(>=(-tol), plan.capacity) || return false
    all(>=(-tol), plan.expansion) || return false
    all(>=(-tol), plan.purchase) || return false
    all(>=(-tol), plan.sales) || return false
    all(x -> x == 0 || x == 1, plan.expand) || return false

    for i in 1:I
        technology = prob.technologies[i]
        for t in 1:T
            previous = t == 1 ? technology.existing_capacity : plan.capacity[i, t - 1]
            abs(previous + plan.expansion[i, t] - plan.capacity[i, t]) <= tol ||
                return false
            plan.expansion[i, t] <= technology.max_expansion * plan.expand[i, t] + tol ||
                return false
            plan.expansion[i, t] + tol >=
                technology.min_expansion * plan.expand[i, t] || return false
            plan.operating_level[i, t] <= plan.capacity[i, t] + tol || return false
        end
    end

    for t in 1:T
        balance = zeros(Float64, J)
        for i in 1:I
            level = plan.operating_level[i, t]
            for (j, coefficient) in prob.technologies[i].outputs
                balance[j] += coefficient * level
            end
            for (j, coefficient) in prob.technologies[i].inputs
                balance[j] -= coefficient * level
            end
        end
        for j in 1:J
            balance[j] += plan.purchase[j, t] - plan.sales[j, t]
            abs(balance[j]) <= tol || return false
            prob.chemicals[j].purchasable ||
                (plan.purchase[j, t] <= tol || return false)
            prob.chemicals[j].sellable || (plan.sales[j, t] <= tol || return false)
            plan.purchase[j, t] <= prob.availability[j, t] + tol || return false
            plan.sales[j, t] + tol >= prob.demand_min[j, t] || return false
            plan.sales[j, t] <= prob.demand_max[j, t] + tol || return false
        end
    end
    return true
end

"""
    process_expansion_certificate_holds(prob; atol=1e-6)

Recompute the stored infeasibility certificate from the instance data and check
that it still refutes the instance. No optimization solver is used.
"""
function process_expansion_certificate_holds(prob::ProcessCapacityExpansionProblem;
                                             atol::Float64=1e-6)
    certificate = prob.infeasibility_certificate
    certificate === nothing && return false
    if certificate.kind == expansion_demand_above_capacity_bound
        j = certificate.chemical
        1 <= j <= n_chemicals(prob) || return false
        achievable = _pp_expansion_capacity_bound(prob, j)
        required = sum(view(prob.demand_min, j, :))
    else
        certificate.chemical == 0 || return false
        achievable = _pp_expansion_feedstock_bound(prob)
        required = sum(prob.demand_min)
    end
    scale = max(1.0, abs(achievable), abs(required))
    isapprox(certificate.achievable, achievable; rtol=1e-9, atol=atol * scale) ||
        return false
    isapprox(certificate.required, required; rtol=1e-9, atol=atol * scale) ||
        return false
    return achievable + atol * scale < required
end

"""
    _pp_expansion_operate(rng, chemicals, technologies, sale_target)
        -> (operating_level, purchase, sales)

Run the network backwards from a sales target for one period: finished chemicals
pull on the processes that make them, those processes pull on their inputs, and
whatever reaches the bottom layer is bought. Byproducts are credited against the
sales of their own chemical, so the balances close exactly.
"""
function _pp_expansion_operate(rng::AbstractRNG, chemicals::Vector{ProcessChemical},
                               technologies::Vector{ProcessTechnology},
                               sale_target::Vector{Float64})
    J = length(chemicals)
    I = length(technologies)
    required = copy(sale_target)
    level = zeros(Float64, I)
    makers = [Int[] for _ in 1:J]
    for (i, technology) in enumerate(technologies)
        push!(makers[technology.main_output], i)
    end

    layers = isempty(technologies) ? Int[] : maximum(t.layer for t in technologies):-1:2
    for layer in layers
        for j in 1:J
            chemicals[j].layer == layer || continue
            needed = max(required[j], 0.0)
            needed <= 0.0 && continue
            options = makers[j]
            isempty(options) && continue
            weights = rand(rng, Dirichlet(fill(3.0, length(options))))
            for (k, i) in enumerate(options)
                technology = technologies[i]
                share = k == length(options) ?
                        needed - sum(weights[1:(k - 1)]) * needed :
                        weights[k] * needed
                level[i] += share
                for (input, coefficient) in technology.inputs
                    required[input] += coefficient * share
                end
            end
            required[j] = 0.0
        end
    end

    balance = zeros(Float64, J)
    for (i, technology) in enumerate(technologies)
        for (j, coefficient) in technology.outputs
            balance[j] += coefficient * level[i]
        end
        for (j, coefficient) in technology.inputs
            balance[j] -= coefficient * level[i]
        end
    end
    purchase = zeros(Float64, J)
    sales = zeros(Float64, J)
    for j in 1:J
        if balance[j] < 0.0
            purchase[j] = -balance[j]
        else
            sales[j] = balance[j]
        end
    end
    return level, purchase, sales
end

"""
    ProcessCapacityExpansionProblem(target_variables, feasibility_status, seed)

Construct a multi-period process-network capacity-expansion instance.

# Variable count

With `I` processes, `T` periods, `n_raw` purchasable raw materials and `n_sell`
saleable chemicals the model has exactly `T * (4I + n_raw + n_sell)` variables:
an operating level, an installed capacity, an expansion and its indicator per
process and period, plus one purchase and one sale variable per tradable
chemical and period. The chemical slate follows the process count, so the count
is evaluated exactly while searching for the horizon and process count closest to
the target.

# Feasibility
- `feasible`: a sales plan is run backwards through the network into operating
  levels and purchases, capacity is expanded to cover it, and availability and
  the demand window are placed around it, so `feasible_witness` is a feasible
  point of the integer model (its indicators are 0/1).
- `infeasible`: either one chemical's contracted sales exceed everything the
  processes that make it could produce under the largest permitted expansions, or
  the contracted sales of the whole network exceed what the raw-material market
  can support. Both are recorded in `infeasibility_certificate` and use only
  linear rows, so they refute the relaxation too.
- `unknown`: capacity, availability and contracts are drawn from a market view
  around the reference operation without being reconciled with it.
"""
function ProcessCapacityExpansionProblem(target_variables::Int,
                                         feasibility_status::FeasibilityStatus,
                                         seed::Int)
    rng = MersenneTwister(seed)
    target = max(target_variables, 1)

    n_tech, T, byproduct_rate = _pp_expansion_dimensions(rng, target)
    chemicals, technologies = _pp_expansion_network(rng, n_tech, byproduct_rate)
    J = length(chemicals)
    I = length(technologies)
    raw_chemicals = [j for j in 1:J if chemicals[j].purchasable]
    sellable_chemicals = [j for j in 1:J if chemicals[j].sellable]

    # A reference sales path: the market the network is designed around, growing
    # over the horizon the way long-range plans assume.
    scale = rand(rng, Uniform(80.0, 900.0))
    growth = rand(rng, Uniform(0.0, 0.09))
    reference_sales = zeros(Float64, J, T)
    for j in sellable_chemicals
        base = scale * rand(rng, Uniform(0.25, 1.6)) *
               (chemicals[j].layer == 1 ? 0.2 : 1.0)
        path = _pp_market_path(rng, T, base; volatility=0.05, seasonality=0.0)
        for t in 1:T
            reference_sales[j, t] = path[t] * (1.0 + growth)^(t - 1)
        end
    end

    level = zeros(Float64, I, T)
    purchase = zeros(Float64, J, T)
    sales = zeros(Float64, J, T)
    top_layer = maximum(c.layer for c in chemicals)
    for t in 1:T
        target_sales = zeros(Float64, J)
        for j in sellable_chemicals
            # Only the finished chemicals are pulled on directly; intermediates
            # and byproducts are sold out of whatever the network leaves over.
            chemicals[j].layer == top_layer &&
                (target_sales[j] = reference_sales[j, t])
        end
        level[:, t], purchase[:, t], sales[:, t] =
            _pp_expansion_operate(rng, chemicals, technologies, target_sales)
    end

    # Capacity: part of the network already stands, the rest is expanded into
    # place in the period it is first needed.
    planned = feasibility_status == feasible
    capacity = zeros(Float64, I, T)
    expansion = zeros(Float64, I, T)
    expand = zeros(Int, I, T)
    for i in 1:I
        peak = maximum(view(level, i, :))
        existing = peak * rand(rng, Uniform(0.0, 0.75))
        step = max(peak * rand(rng, Uniform(0.25, 0.60)), 1e-3)
        # The reference operation is always buildable: one expansion covers the
        # whole level, so the plan's balances close on the levels it really runs.
        # Both window ends are rounded here rather than at storage time, so the
        # planted expansions are clamped by exactly the bounds the model
        # publishes; rounding afterwards can move a bound below the expansion it
        # was supposed to admit.
        covering = round(max(peak * rand(rng, Uniform(1.05, 1.80)), step), digits=4)
        min_expansion = round(step * rand(rng, Uniform(0.10, 0.45)), digits=4)
        # A requested-feasible instance publishes that window. Otherwise the
        # window is an engineering rule — a plant is debottlenecked in steps of a
        # given size — which may or may not stretch to the market being asked for.
        stated_max = planned ? covering :
                     round(max(peak * rand(rng, Uniform(0.30, 1.40)), step),
                           digits=4)
        technologies[i] = ProcessTechnology(
            technologies[i].name, technologies[i].layer, technologies[i].main_output,
            technologies[i].outputs, technologies[i].inputs,
            technologies[i].operating_cost, technologies[i].fixed_investment,
            technologies[i].variable_investment,
            round(existing, digits=4), min_expansion, stated_max)
        installed = technologies[i].existing_capacity
        for t in 1:T
            shortfall = level[i, t] - installed
            if shortfall > 0.0
                amount = clamp(shortfall * rand(rng, Uniform(1.02, 1.30)),
                               technologies[i].min_expansion, covering)
                expansion[i, t] = amount
                expand[i, t] = 1
                installed += amount
            end
            capacity[i, t] = installed
        end
    end
    # Close each chemical's balance at the levels the plan runs: what the network
    # is short of is bought, what it has left over is sold.
    for t in 1:T
        residual = zeros(Float64, J)
        for i in 1:I
            for (j, coefficient) in technologies[i].outputs
                residual[j] += coefficient * level[i, t]
            end
            for (j, coefficient) in technologies[i].inputs
                residual[j] -= coefficient * level[i, t]
            end
        end
        for j in 1:J
            purchase[j, t] = chemicals[j].purchasable ? max(-residual[j], 0.0) : 0.0
            sales[j, t] = chemicals[j].sellable ? max(residual[j], 0.0) : 0.0
        end
    end

    # Markets.
    purchase_cost = zeros(Float64, J, T)
    availability = zeros(Float64, J, T)
    sale_price = zeros(Float64, J, T)
    demand_min = zeros(Float64, J, T)
    demand_max = zeros(Float64, J, T)
    for j in 1:J
        chemical = chemicals[j]
        if chemical.purchasable
            base = rand(rng, Uniform(280.0, 900.0))
            purchase_cost[j, :] .= _pp_market_path(rng, T, base; volatility=0.07)
            reference = maximum(view(purchase, j, :))
            offered = planned ?
                      reference * rand(rng, Uniform(1.2, 2.5)) +
                      scale * rand(rng, Uniform(0.05, 0.5)) :
                      reference * rand(rng, Uniform(0.80, 1.70))
            for t in 1:T
                availability[j, t] = planned ?
                    max(offered, purchase[j, t] * rand(rng, Uniform(1.05, 1.4))) :
                    offered * rand(rng, Uniform(0.90, 1.10))
            end
        end
        if chemical.sellable
            base = rand(rng, Uniform(700.0, 2_400.0)) *
                   (1.0 + 0.12 * (chemical.layer - 1))
            sale_price[j, :] .= _pp_market_path(rng, T, base; volatility=0.06)
            contract = rand(rng, Uniform(0.35, 0.85))
            # Not every chemical is sold forward: some move only on the spot
            # market, and carry no floor at all. Keeping the contracted share
            # modest also keeps a large network from becoming an implausibly long
            # conjunction of independent commitments, all of which would have to
            # hold for the plan to exist.
            contracted = planned || rand(rng) < 0.35
            for t in 1:T
                reference = sales[j, t]
                if planned
                    demand_min[j, t] = reference * contract
                    demand_max[j, t] = max(reference * rand(rng, Uniform(1.05, 1.7)),
                                           demand_min[j, t] * 1.05)
                else
                    demand_min[j, t] = contracted ?
                                       reference * rand(rng, Uniform(0.50, 1.30)) : 0.0
                    demand_max[j, t] = max(reference * rand(rng, Uniform(1.0, 1.8)),
                                           demand_min[j, t] * 1.05)
                end
            end
        end
    end
    rate = rand(rng, Uniform(0.07, 0.15))
    discount = [1.0 / (1.0 + rate)^(t - 1) for t in 1:T]

    plan = ProcessExpansionPlan(level, capacity, expansion, expand, purchase, sales)
    certificate = nothing
    if feasibility_status == infeasible
        certificate = _pp_expansion_break!(rng, chemicals, technologies,
                                           sellable_chemicals, raw_chemicals,
                                           demand_min, demand_max, availability, T)
    end

    problem = ProcessCapacityExpansionProblem(
        T, chemicals, technologies, raw_chemicals, sellable_chemicals,
        purchase_cost, availability, sale_price, demand_min, demand_max, discount,
        feasibility_status == feasible ? plan : nothing, certificate,
        feasibility_status)

    if feasibility_status == feasible
        @assert process_expansion_plan_satisfies(problem)
    elseif feasibility_status == infeasible
        @assert process_expansion_certificate_holds(problem)
    end
    return problem
end

"""
    _pp_expansion_break!(rng, chemicals, technologies, sellable, raw,
                         demand_min, demand_max, availability, T) -> certificate

Over-commit the contracts in one of the two auditable ways and return the
matching certificate: past what the processes making a single chemical could ever
be expanded to produce, or past what the raw-material market can supply the whole
network.
"""
function _pp_expansion_break!(rng::AbstractRNG, chemicals::Vector{ProcessChemical},
                              technologies::Vector{ProcessTechnology},
                              sellable::Vector{Int}, raw::Vector{Int},
                              demand_min::Matrix{Float64},
                              demand_max::Matrix{Float64},
                              availability::Matrix{Float64}, T::Int)
    # A chemical nothing makes has a zero bound, which no positive contract can
    # sit strictly above by the margin the certificate needs; only argue about
    # chemicals some process actually produces.
    made = [j for j in sellable
            if any(any(pair.first == j for pair in technology.outputs)
                   for technology in technologies)]
    if !isempty(made) && rand(rng) < 0.5
        j = made[rand(rng, 1:length(made))]
        bound = 0.0
        for t in 1:T, technology in technologies
            index = findfirst(pair -> pair.first == j, technology.outputs)
            index === nothing && continue
            bound += technology.outputs[index].second *
                     (technology.existing_capacity + t * technology.max_expansion)
        end
        required = bound * rand(rng, Uniform(1.15, 1.60))
        for t in 1:T
            demand_min[j, t] = required / T
            demand_max[j, t] = max(demand_max[j, t], demand_min[j, t] * 1.05)
        end
        return ProcessExpansionCertificate(expansion_demand_above_capacity_bound, j,
                                           bound, sum(view(demand_min, j, :)))
    end

    value = _pp_expansion_potential(chemicals, technologies)
    bound = sum(value[j] * availability[j, t] for j in raw, t in 1:T)
    wanted = bound * rand(rng, Uniform(1.15, 1.60))
    committed = sum(demand_min)
    if committed <= 0.0
        share = wanted / max(length(sellable) * T, 1)
        for j in sellable, t in 1:T
            demand_min[j, t] = share
        end
    else
        demand_min .*= wanted / committed
    end
    for j in 1:size(demand_min, 1), t in 1:T
        demand_max[j, t] = max(demand_max[j, t], demand_min[j, t] * 1.05)
    end
    return ProcessExpansionCertificate(expansion_demand_above_feedstock_bound, 0,
                                       bound, sum(demand_min))
end

"""
    build_model(prob::ProcessCapacityExpansionProblem)

Build the multi-period capacity-expansion MILP. Deterministic — uses only the
stored network and market data.

# Model
Variables per period: the operating level, installed capacity, expansion and
binary expansion indicator of every process, plus a purchase variable for every
raw material and a sale variable for every saleable chemical.

Constraints: the capacity recursion `Q_t = Q_{t-1} + QE_t`, the expansion window
`QE_min y <= QE <= QE_max y`, the operating bound `W <= Q`, one balance row per
chemical and period, raw-material availability, and the demand window.
"""
function build_model(prob::ProcessCapacityExpansionProblem)
    I = n_technologies(prob)
    J = n_chemicals(prob)
    T = prob.n_periods

    model = Model()
    @variable(model, operating_level[1:I, 1:T] >= 0)
    @variable(model, capacity[1:I, 1:T] >= 0)
    @variable(model, expansion[1:I, 1:T] >= 0)
    @variable(model, expand[1:I, 1:T], Bin)
    @variable(model, 0 <= purchase[j in prob.raw_chemicals, t in 1:T] <=
                     prob.availability[j, t])
    @variable(model, prob.demand_min[j, t] <=
                     sales[j in prob.sellable_chemicals, t in 1:T] <=
                     prob.demand_max[j, t])

    for i in 1:I, t in 1:T
        technology = prob.technologies[i]
        previous = t == 1 ? technology.existing_capacity : capacity[i, t - 1]
        @constraint(model, capacity[i, t] == previous + expansion[i, t])
        @constraint(model, expansion[i, t] <= technology.max_expansion * expand[i, t])
        @constraint(model, expansion[i, t] >= technology.min_expansion * expand[i, t])
        @constraint(model, operating_level[i, t] <= capacity[i, t])
    end

    balance = Matrix{AffExpr}(undef, J, T)
    for j in 1:J, t in 1:T
        balance[j, t] = AffExpr(0.0)
    end
    for i in 1:I
        technology = prob.technologies[i]
        for (j, coefficient) in technology.outputs, t in 1:T
            add_to_expression!(balance[j, t], coefficient, operating_level[i, t])
        end
        for (j, coefficient) in technology.inputs, t in 1:T
            add_to_expression!(balance[j, t], -coefficient, operating_level[i, t])
        end
    end
    for j in prob.raw_chemicals, t in 1:T
        add_to_expression!(balance[j, t], 1.0, purchase[j, t])
    end
    for j in prob.sellable_chemicals, t in 1:T
        add_to_expression!(balance[j, t], -1.0, sales[j, t])
    end
    @constraint(model, chemical_balance[j in 1:J, t in 1:T], balance[j, t] == 0)

    @objective(model, Max,
        sum(prob.discount[t] *
            (sum(prob.sale_price[j, t] * sales[j, t] for j in prob.sellable_chemicals;
                 init = AffExpr(0.0)) -
             sum(prob.purchase_cost[j, t] * purchase[j, t] for j in prob.raw_chemicals;
                 init = AffExpr(0.0)) -
             sum(prob.technologies[i].operating_cost * operating_level[i, t] +
                 prob.technologies[i].fixed_investment * expand[i, t] +
                 prob.technologies[i].variable_investment * expansion[i, t]
                 for i in 1:I; init = AffExpr(0.0)))
            for t in 1:T))

    return model
end

register_variant(
    :process_planning,
    :capacity_expansion,
    ProcessCapacityExpansionProblem,
    "Long-range capacity expansion of a chemical process network: discrete " *
    "capacity additions, fixed-ratio conversion, feedstock availability and " *
    "contracted demand, on a discounted net-present-value objective",
)
