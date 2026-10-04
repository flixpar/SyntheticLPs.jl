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
    outputs::Vector{Pair{Int, Float64}}
    inputs::Vector{Pair{Int, Float64}}
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
    expansion_capital_below_requirement
end

"""
    ProcessExpansionCertificate

Solver-independent proof stored on a requested-infeasible instance:
`expansion_capital_below_requirement`, the contracted sales of period `period`
need more new capacity than the capital budget can buy.

Give every process the least investment per unit of new capacity its expansions
can cost in the LP relaxation, `gamma_i = variable_investment + fixed_investment
/ max_expansion` (the window row `QE <= QE_max y` makes `y >= QE / QE_max`), and
every chemical a capital potential `potential[j]` (see
[`_pp_expansion_capital_potential`](@ref)): zero on raw materials and
byproducts, and for a main product the cheapest maker's `gamma` plus its input
potentials. Multiplying period `period`'s balance rows by the potentials and
summing gives `sum_j potential[j] * sales[j]  = sum_i d_i * level_i` with
`d_i = sum_out potential - sum_in potential <= gamma_i`; with `level <= capacity
= existing + new capacity` and the sales floors this bounds the investment
`sum_i gamma_i * new_capacity_i` - hence the budget row's left side - below by

    required = sum_j potential[j] * demand_min[j, period] - sum_i max(d_i, 0) * existing_i

which exceeds the budget `achievable`. The argument chains every process layer,
every contracted product, the capacity recursion over every earlier period and
the budget row, so no single row or bound propagation exposes it.
"""
struct ProcessExpansionCertificate
    kind::ProcessExpansionInfeasibilityKind
    period::Int
    potential::Vector{Float64}
    achievable::Float64
    required::Float64
end

"""
    ProcessExpansionBudgetScenario

Capital envelope of an unknown-status instance. Every other row is placed
around the reference plan exactly as for a requested-feasible instance; the
capital budget is then placed relative to two anchors: `requirement`, the best
certified lower bound on the investment the contracts need (the certificate's
`required`), and the reference plan's own investment at its LP-relaxed cost
(`gamma_i` per unit of new capacity). A negative `budget_share` puts the budget
`|budget_share|` below the requirement (infeasible by the certificate's
argument); a positive one moves it that fraction of the way up to the plan's
relaxed cost, which the plan itself meets at one. In between, whether the
contracts can be served within the envelope is genuinely open. `budget_share`
follows the golden-ratio position of the seed.
"""
struct ProcessExpansionBudgetScenario
    budget_share::Float64
    position::Float64
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
demand window with a contracted floor, and the whole investment programme by a
capital budget over the horizon. The objective maximizes discounted net
present value: sales revenue less feedstock, operating and investment cost.

# Fields
- `chemicals`, `technologies`: the process network
- `purchase_cost`, `availability`, `sale_price`, `demand_min`, `demand_max`,
  `discount`: the market over the horizon
- `capital_budget`: total investment (fixed charges plus linear cost) the
  programme may spend over the horizon
- `feasible_witness`, `infeasibility_certificate`, `budget_scenario`,
  `feasibility_status`

Expansion indicators are binary, so this is a genuine MILP; with the package
default `relax_integer=true` it is returned as its LP relaxation, in which a
fractional indicator buys a fractionally-sized expansion. The planted witness
and the certificate are valid for the relaxation as well as for the MILP.
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
    capital_budget::Float64
    feasible_witness::Union{Nothing, ProcessExpansionPlan}
    infeasibility_certificate::Union{Nothing, ProcessExpansionCertificate}
    budget_scenario::Union{Nothing, ProcessExpansionBudgetScenario}
    feasibility_status::FeasibilityStatus
end

n_chemicals(prob::ProcessCapacityExpansionProblem) = length(prob.chemicals)
n_technologies(prob::ProcessCapacityExpansionProblem) = length(prob.technologies)

"""Variables per period: operating level, capacity, expansion and its indicator, plus trade."""
_pp_expansion_variables(n_tech::Int, n_raw::Int, n_sell::Int) = 4 * n_tech + n_raw + n_sell

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
    return n_final +
           _pp_expansion_traded_intermediates(n_intermediate) +
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
            score = (round(err; digits=3), shape)
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
        push!(chemicals, ProcessChemical(Symbol(:raw_, length(chemicals) + 1), 1, true, false))
        push!(layer_members[1], length(chemicals))
    end
    intermediate_index = 0
    # A fixed share of the intermediates also trade on the open market; which
    # ones is random, how many is not, so the variable count stays exact.
    traded = _pp_expansion_traded_intermediates(n_intermediate)
    traded_set = Set(shuffle(rng, collect(1:max(n_intermediate, 1)))[1:traded])
    for layer in 2:(n_layers - 1), _ in 1:per_layer
        intermediate_index += 1
        push!(
            chemicals,
            ProcessChemical(
                Symbol(:intermediate_, length(chemicals) + 1),
                layer,
                false,
                intermediate_index in traded_set,
            ),
        )
        push!(layer_members[layer], length(chemicals))
    end
    for _ in 1:n_final
        push!(
            chemicals,
            ProcessChemical(Symbol(:product_, length(chemicals) + 1), n_layers, false, true),
        )
        push!(layer_members[n_layers], length(chemicals))
    end

    technologies = ProcessTechnology[]
    byproduct_set = Set(
        shuffle(rng, collect(1:n_tech))[1:_pp_expansion_byproducts(n_tech, byproduct_rate)]
    )
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
        inputs = [chosen[k] => max(round(weights[k] / yield; digits=4), 0.02) for k in 1:n_inputs]
        outputs = [main => 1.0]
        if i in byproduct_set
            byproduct = ProcessChemical(
                Symbol(:byproduct_, length(chemicals) + 1), layer, false, true
            )
            push!(chemicals, byproduct)
            push!(outputs, length(chemicals) => round(rand(rng, Uniform(0.05, 0.35)); digits=3))
        end
        push!(
            technologies,
            ProcessTechnology(
                Symbol(:process_, i),
                layer,
                main,
                outputs,
                inputs,
                round(rand(rng, Uniform(20.0, 120.0)); digits=2),
                round(rand(rng, Uniform(8_000.0, 60_000.0)); digits=1),
                round(rand(rng, Uniform(300.0, 1_500.0)); digits=2),
                0.0,
                0.0,
                0.0,
            ),
        )
    end
    return chemicals, technologies
end

"""Least LP-relaxed investment per unit of new capacity of a process: `v + f / QE_max`."""
_pp_expansion_unit_capital(technology::ProcessTechnology) =
    technology.variable_investment + technology.fixed_investment / technology.max_expansion

"""
    _pp_expansion_capital_potential(chemicals, technologies) -> Vector{Float64}

Capital potential of every chemical: the least new-capacity investment embodied
in one unit of it. Zero on raw materials (bought, not built) and on byproducts;
for a main product the minimum over the processes that make it of their unit
capital (see [`_pp_expansion_unit_capital`](@ref)) plus the potentials of their
inputs, per unit of main output. Processes only consume chemicals from strictly
lower layers, so one pass up the layers is exact, and every process satisfies
`sum_out c * potential - sum_in a * potential <= unit capital` — the dual
condition behind [`ProcessExpansionCertificate`](@ref). A chemical nothing makes
gets potential zero.
"""
function _pp_expansion_capital_potential(
    chemicals::Vector{ProcessChemical}, technologies::Vector{ProcessTechnology}
)
    J = length(chemicals)
    potential = zeros(Float64, J)
    isempty(technologies) && return potential
    best = fill(Inf, J)
    for layer in 2:maximum(t.layer for t in technologies)
        for technology in technologies
            technology.layer == layer || continue
            embodied =
                _pp_expansion_unit_capital(technology) +
                sum(coefficient * potential[j] for (j, coefficient) in technology.inputs)
            j = technology.main_output
            best[j] = min(best[j], embodied / technology.outputs[1].second)
        end
        for j in 1:J
            chemicals[j].layer == layer &&
                isfinite(best[j]) &&
                !chemicals[j].purchasable &&
                (potential[j] = best[j])
        end
    end
    return potential
end

"""Net capital potential a process creates per unit of operating level."""
_pp_expansion_net_potential(technology::ProcessTechnology, potential::Vector{Float64}) =
    sum(c * potential[j] for (j, c) in technology.outputs) -
    sum(a * potential[j] for (j, a) in technology.inputs)

"""
    _pp_expansion_capital_requirement(technologies, demand_min, potential, t) -> Float64

The certificate's lower bound on the investment needed to serve period `t`'s
contracted sales: `sum_j potential[j] * demand_min[j, t]` less the potential the
existing capacity already supplies, `sum_i max(d_i, 0) * existing_i`.
"""
function _pp_expansion_capital_requirement(
    technologies::Vector{ProcessTechnology},
    demand_min::Matrix{Float64},
    potential::Vector{Float64},
    t::Int,
)
    contracted = sum(potential[j] * demand_min[j, t] for j in axes(demand_min, 1); init=0.0)
    existing = sum(
        max(_pp_expansion_net_potential(technology, potential), 0.0) * technology.existing_capacity
        for technology in technologies;
        init=0.0,
    )
    return contracted - existing
end

"""
    _pp_expansion_best_requirement(chemicals, technologies, demand_min)
        -> (requirement, period, potential)

The largest certified capital requirement over the horizon and the period that
attains it.
"""
function _pp_expansion_best_requirement(chemicals, technologies, demand_min::Matrix{Float64})
    potential = _pp_expansion_capital_potential(chemicals, technologies)
    best, period = -Inf, 1
    for t in axes(demand_min, 2)
        value = _pp_expansion_capital_requirement(technologies, demand_min, potential, t)
        value > best && ((best, period) = (value, t))
    end
    return best, period, potential
end

"""Total investment of a plan: fixed charges on its expansions plus the linear cost."""
_pp_expansion_capital_spend(technologies, expansion::Matrix{Float64}, expand::AbstractMatrix) = sum(
    technologies[i].fixed_investment * expand[i, t] +
    technologies[i].variable_investment * expansion[i, t] for
    i in axes(expansion, 1), t in axes(expansion, 2);
    init=0.0,
)

"""
    process_expansion_plan_satisfies(prob, plan=prob.feasible_witness; atol=1e-6)

Re-check a planted expansion plan against every row: the capacity recursion, the
expansion window and its indicator, the operating-level bound, every chemical
balance, raw-material availability, the demand window and the capital budget.
Solver-independent.
"""
function process_expansion_plan_satisfies(
    prob::ProcessCapacityExpansionProblem,
    plan::Union{Nothing, ProcessExpansionPlan}=prob.feasible_witness;
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
            abs(previous + plan.expansion[i, t] - plan.capacity[i, t]) <= tol || return false
            plan.expansion[i, t] <= technology.max_expansion * plan.expand[i, t] + tol ||
                return false
            plan.expansion[i, t] + tol >= technology.min_expansion * plan.expand[i, t] ||
                return false
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
            prob.chemicals[j].purchasable || (plan.purchase[j, t] <= tol || return false)
            prob.chemicals[j].sellable || (plan.sales[j, t] <= tol || return false)
            plan.purchase[j, t] <= prob.availability[j, t] + tol || return false
            plan.sales[j, t] + tol >= prob.demand_min[j, t] || return false
            plan.sales[j, t] <= prob.demand_max[j, t] + tol || return false
        end
    end
    spend = _pp_expansion_capital_spend(prob.technologies, plan.expansion, plan.expand)
    spend <= prob.capital_budget + atol * max(1.0, prob.capital_budget) || return false
    return true
end

"""
    process_expansion_certificate_holds(prob; atol=1e-6)

Recompute the stored infeasibility certificate from the instance data and check
that it still refutes the instance. No optimization solver is used.
"""
function process_expansion_certificate_holds(
    prob::ProcessCapacityExpansionProblem; atol::Float64=1e-6
)
    certificate = prob.infeasibility_certificate
    certificate === nothing && return false
    certificate.kind == expansion_capital_below_requirement || return false
    1 <= certificate.period <= prob.n_periods || return false
    potential = certificate.potential
    length(potential) == n_chemicals(prob) || return false
    # Dual feasibility: nonnegative potentials, none on a purchasable chemical,
    # and no process creating more potential than its unit capital.
    all(>=(0.0), potential) || return false
    for (j, chemical) in enumerate(prob.chemicals)
        chemical.purchasable && potential[j] != 0.0 && return false
    end
    for technology in prob.technologies
        net = _pp_expansion_net_potential(technology, potential)
        unit = _pp_expansion_unit_capital(technology)
        net <= unit + atol * max(1.0, unit) || return false
    end
    required = _pp_expansion_capital_requirement(
        prob.technologies, prob.demand_min, potential, certificate.period
    )
    scale = max(1.0, abs(required), abs(prob.capital_budget))
    isapprox(certificate.required, required; rtol=1e-9, atol=atol * scale) || return false
    isapprox(certificate.achievable, prob.capital_budget; rtol=1e-9, atol=atol * scale) ||
        return false
    return prob.capital_budget + atol * scale < required
end

"""
    _pp_expansion_operate(rng, chemicals, technologies, sale_target)
        -> (operating_level, purchase, sales)

Run the network backwards from a sales target for one period: finished chemicals
pull on the processes that make them, those processes pull on their inputs, and
whatever reaches the bottom layer is bought. Byproducts are credited against the
sales of their own chemical, so the balances close exactly.
"""
function _pp_expansion_operate(
    rng::AbstractRNG,
    chemicals::Vector{ProcessChemical},
    technologies::Vector{ProcessTechnology},
    sale_target::Vector{Float64},
)
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
                share = if k == length(options)
                    needed - sum(weights[1:(k - 1)]) * needed
                else
                    weights[k] * needed
                end
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
  levels and purchases, capacity is expanded to cover it, availability and the
  demand window are placed around it, and the capital budget sits 10-50% above
  the plan's spend, so `feasible_witness` is a feasible point of the integer
  model (its indicators are 0/1).
- `infeasible`: the same reference-planned data, with the capital budget cut
  6-20% below the investment the contracts certifiably need (a capital squeeze;
  see [`ProcessExpansionCertificate`](@ref)). The refutation chains every process
  layer, the capacity recursion and the budget row, so presolve's bound
  propagation cannot see it.
- `unknown`: the reference-planned data with the capital budget placed between
  that certified requirement and the plan's relaxed investment (see
  [`ProcessExpansionBudgetScenario`](@ref)).

Contracts cover 75-97% of the reference plan's finished sales; in the rare case
that the existing fleet still covers them, the infeasible and unknown branches
sell the finished chemicals forward at 99% of the plan's sales first (see
[`_pp_expansion_capital_requirement_or_raise!`](@ref)).
"""
function ProcessCapacityExpansionProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
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
        base = scale * rand(rng, Uniform(0.25, 1.6)) * (chemicals[j].layer == 1 ? 0.2 : 1.0)
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
            chemicals[j].layer == top_layer && (target_sales[j] = reference_sales[j, t])
        end
        level[:, t], purchase[:, t], sales[:, t] = _pp_expansion_operate(
            rng, chemicals, technologies, target_sales
        )
    end

    # Capacity: part of the network already stands, the rest is expanded into
    # place in the period it is first needed. Every status places the local data
    # (expansion windows, demand windows) around this reference plan; the
    # statuses differ only in the raw-material market (see below).
    capacity = zeros(Float64, I, T)
    expansion = zeros(Float64, I, T)
    expand = zeros(Int, I, T)
    running_peaks = [maximum(view(level, i, :)) for i in 1:I]
    typical_peak = let busy = filter(>(0.0), running_peaks)
        isempty(busy) ? scale : sort(busy)[cld(length(busy), 2)]
    end
    for i in 1:I
        peak = running_peaks[i]
        if peak <= 0.0
            # A process the reference plan never runs is still a real option:
            # a greenfield project sized like a typical unit, with no existing
            # capacity (rather than a degenerate, unbuildable window).
            peak = typical_peak * rand(rng, Uniform(0.3, 0.8))
            existing = 0.0
        else
            existing = peak * rand(rng, Uniform(0.0, 0.75))
        end
        step = max(peak * rand(rng, Uniform(0.25, 0.60)), 1e-3)
        # The reference operation is always buildable: one expansion covers the
        # whole level, so the plan's balances close on the levels it really runs.
        # Both window ends are rounded here rather than at storage time, so the
        # planted expansions are clamped by exactly the bounds the model
        # publishes; rounding afterwards can move a bound below the expansion it
        # was supposed to admit.
        covering = round(max(peak * rand(rng, Uniform(1.05, 1.80)), step); digits=4)
        min_expansion = round(step * rand(rng, Uniform(0.10, 0.45)); digits=4)
        stated_max = covering
        technologies[i] = ProcessTechnology(
            technologies[i].name,
            technologies[i].layer,
            technologies[i].main_output,
            technologies[i].outputs,
            technologies[i].inputs,
            technologies[i].operating_cost,
            technologies[i].fixed_investment,
            technologies[i].variable_investment,
            round(existing; digits=4),
            min_expansion,
            stated_max,
        )
        installed = technologies[i].existing_capacity
        for t in 1:T
            shortfall = level[i, t] - installed
            if shortfall > 0.0
                amount = clamp(
                    shortfall * rand(rng, Uniform(1.02, 1.30)),
                    technologies[i].min_expansion,
                    covering,
                )
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
            # Round-off residue is not a trade: dropping it keeps the published
            # market bounds clear of 1e-15 noise.
            bought = max(-residual[j], 0.0)
            sold = max(residual[j], 0.0)
            purchase[j, t] = chemicals[j].purchasable && bought > 1e-9 ? bought : 0.0
            sales[j, t] = chemicals[j].sellable && sold > 1e-9 ? sold : 0.0
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
            offered =
                reference * rand(rng, Uniform(1.2, 2.5)) + scale * rand(rng, Uniform(0.05, 0.5))
            for t in 1:T
                availability[j, t] = max(offered, purchase[j, t] * rand(rng, Uniform(1.05, 1.4)))
            end
        end
        if chemical.sellable
            base = rand(rng, Uniform(700.0, 2_400.0)) * (1.0 + 0.12 * (chemical.layer - 1))
            sale_price[j, :] .= _pp_market_path(rng, T, base; volatility=0.06)
            contract = rand(rng, Uniform(0.75, 0.97))
            spot_floor = scale * rand(rng, Uniform(0.02, 0.10))
            # Finished chemicals are sold forward under term contracts;
            # intermediates and byproducts move on the spot market with no floor.
            contracted = startswith(String(chemical.name), "product_")
            for t in 1:T
                reference = sales[j, t]
                demand_min[j, t] = contracted ? reference * contract : 0.0
                # A spot outlet exists even for chemicals the plan does not
                # sell (a byproduct of an idle process, a surplus intermediate).
                demand_max[j, t] = max(
                    reference * rand(rng, Uniform(1.05, 1.7)), demand_min[j, t] * 1.05, spot_floor
                )
            end
        end
    end
    rate = rand(rng, Uniform(0.07, 0.15))
    discount = [1.0 / (1.0 + rate)^(t - 1) for t in 1:T]

    plan = ProcessExpansionPlan(level, capacity, expansion, expand, purchase, sales)
    # Capital envelope. The plan's own spend (fixed charges on its 0/1
    # expansions) bounds what a feasible request needs; its LP-relaxed cost
    # (`gamma` per unit of new capacity) and the certified requirement bracket the
    # threshold of the relaxation.
    plan_spend = _pp_expansion_capital_spend(technologies, expansion, expand)
    plan_relaxed = sum(
        _pp_expansion_unit_capital(technologies[i]) * expansion[i, t] for i in 1:I, t in 1:T;
        init=0.0,
    )
    certificate = nothing
    scenario = nothing
    if feasibility_status == feasible
        capital_budget = plan_spend * rand(rng, Uniform(1.10, 1.50))
    else
        requirement, period, potential = _pp_expansion_capital_requirement_or_raise!(
            chemicals, technologies, demand_min, demand_max, sales, plan_relaxed
        )
        requirement > 0 || error(
            "capacity_expansion: the existing fleet covers every contract; no capital " *
            "requirement to certify (seed $seed)",
        )
        if feasibility_status == unknown
            position = _pp_seed_position(seed)
            share = -0.15 + 1.10 * position
            capital_budget = if share < 0
                requirement * (1 + share)
            else
                requirement + share * max(plan_relaxed - requirement, 0.0)
            end
            scenario = ProcessExpansionBudgetScenario(share, position)
        else
            # A capital squeeze: the programme's envelope is cut 6-20% below the
            # investment the contracts certifiably need.
            capital_budget = requirement / rand(rng, Uniform(1.06, 1.20))
            certificate = ProcessExpansionCertificate(
                expansion_capital_below_requirement, period, potential, capital_budget, requirement
            )
        end
    end

    problem = ProcessCapacityExpansionProblem(
        T,
        chemicals,
        technologies,
        raw_chemicals,
        sellable_chemicals,
        purchase_cost,
        availability,
        sale_price,
        demand_min,
        demand_max,
        discount,
        capital_budget,
        feasibility_status == feasible ? plan : nothing,
        certificate,
        scenario,
        feasibility_status,
    )

    if feasibility_status == feasible
        @assert process_expansion_plan_satisfies(problem)
    elseif feasibility_status == infeasible
        @assert process_expansion_certificate_holds(problem)
    end
    return problem
end

"""
    _pp_expansion_capital_requirement_or_raise!(rng, chemicals, technologies,
        demand_min, demand_max, sales, plan_relaxed) -> (requirement, period, potential)

The certified capital requirement of the contracts (see
[`ProcessExpansionCertificate`](@ref)). Contracts cover 75-97% of the reference
plan's finished sales, so the existing fleet rarely covers them; when it does
(a requirement below 2% of the plan's relaxed investment) the finished
chemicals are sold forward at 99% of the plan's sales instead - still within
what the plan delivers, so availability and expansion windows stay consistent
with it - before the requirement is recomputed.
"""
function _pp_expansion_capital_requirement_or_raise!(
    chemicals,
    technologies,
    demand_min::Matrix{Float64},
    demand_max::Matrix{Float64},
    sales::Matrix{Float64},
    plan_relaxed::Float64,
)
    requirement, period, potential = _pp_expansion_best_requirement(
        chemicals, technologies, demand_min
    )
    if requirement < 0.02 * plan_relaxed
        for j in axes(demand_min, 1), t in axes(demand_min, 2)
            demand_min[j, t] > 0 || continue
            demand_min[j, t] = max(demand_min[j, t], 0.99 * sales[j, t])
            demand_max[j, t] = max(demand_max[j, t], 1.05 * demand_min[j, t])
        end
        requirement, period, potential = _pp_expansion_best_requirement(
            chemicals, technologies, demand_min
        )
    end
    return requirement, period, potential
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
chemical and period, raw-material availability, the demand window, and one
capital-budget row over every expansion of the horizon.
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
    @variable(model, 0 <= purchase[j in prob.raw_chemicals, t in 1:T] <= prob.availability[j, t])
    @variable(
        model,
        prob.demand_min[j, t] <=
            sales[j in prob.sellable_chemicals, t in 1:T] <=
            prob.demand_max[j, t]
    )

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

    # Capital budget over the whole programme.
    @constraint(
        model,
        capital_budget,
        sum(
            prob.technologies[i].fixed_investment * expand[i, t] +
            prob.technologies[i].variable_investment * expansion[i, t] for i in 1:I, t in 1:T
        ) <= prob.capital_budget
    )

    @objective(
        model,
        Max,
        sum(
            prob.discount[t] * (
                sum(
                    prob.sale_price[j, t] * sales[j, t] for j in prob.sellable_chemicals;
                    init=AffExpr(0.0),
                ) - sum(
                    prob.purchase_cost[j, t] * purchase[j, t] for j in prob.raw_chemicals;
                    init=AffExpr(0.0),
                ) - sum(
                    prob.technologies[i].operating_cost * operating_level[i, t] +
                    prob.technologies[i].fixed_investment * expand[i, t] +
                    prob.technologies[i].variable_investment * expansion[i, t] for i in 1:I;
                    init=AffExpr(0.0),
                )
            ) for t in 1:T
        )
    )

    return model
end

register_variant(
    :process_planning,
    :capacity_expansion,
    ProcessCapacityExpansionProblem,
    "Long-range capacity expansion of a chemical process network: discrete " *
    "capacity additions, fixed-ratio conversion, feedstock availability and " *
    "contracted demand, on a discounted net-present-value objective";
    tags=[:production, :staircase, :big_m],
)
