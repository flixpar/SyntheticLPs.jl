using JuMP
using Random
using Distributions

"""Tramp elements whose scrap assays are uncertain (Si, Fe, Cu, Zn)."""
const BLEND_UNCERTAIN_ELEMENTS = (1, 2, 3, 6)

"""
    RobustBlendPlan

Planted solution of `RobustBlendingProblem`: tonnes `charge_plan[k]` per blend
pair, protection levels `protection[r]` (one per robust row) and term slacks
`excess[t]` (one per robust term).
"""
struct RobustBlendPlan
    charge_plan::Vector{Float64}
    protection::Vector{Float64}
    excess::Vector{Float64}
end

"""
    RobustBlendingProblem <: ProblemGenerator

Alloy blending that is robust to scrap-assay uncertainty: the multi-plant
order-blending model of `blending/standard`, with every tramp-element maximum
(Si, Fe, Cu, Zn) protected against the true content of up to `gamma` charged
scrap lots deviating from their assay (Bertsimas–Sim budgeted uncertainty).

# Formulation

A scrap lot's content of tramp element `e` lies in `comp[e,i] ± deviation[i] ·
comp[e,i]`; primary metal and master alloys are certified (no deviation). For
order `o` and uncertain element `e` the nominal maximum row is replaced by its
robust counterpart (LP duality on the inner worst case):

    Σ_i comp[e,i] x[i,o] + Γ_o z[o,e] + Σ_{i∈scrap(o)} p[o,e,i] ≤ hi[e,o] · charge[o]
    z[o,e] + p[o,e,i] ≥ deviation[i] · comp[e,i] · x[i,o]      for every scrap lot i
    z, p ≥ 0,   Γ_o = min(gamma, |scrap(o)|).

Everything else (charge balance, output window, element minimums, other
maxima, material availability, maximum-margin objective) is as in
`BlendingProblem`. The protection rows are three-term rows coupling a blend
variable to two auxiliary variables — a structure absent from the nominal model
— and the budget makes the cheapest dirty lots pay for their uncertainty, so
optimal charges diversify across lots.

# Sizing

Per order: compatible pairs + 1 charge variable + Σ over its protected rows of
`1 + |scrap lots carrying e|`. Orders are added until that total reaches the
target (within one order's count).

# Feasibility

  - `feasible`: the standard planted recipes; each protected maximum is raised
    (2% over) only where the recipe's worst case `Σ comp·x + β(x)` exceeds it,
    with `β` the exact budgeted protection; the stored `RobustBlendPlan` holds
    the optimal `z`/`p` for the planted charges.
  - `infeasible`: the standard family-shortage or element-shortage mutation;
    its certificate is about the nominal model, whose feasible set contains the
    robust one, so it remains a valid proof.
  - `unknown`: registered windows and nominal availabilities as in
    `blending/standard`, plus sampled deviations and budget.
"""
struct RobustBlendingProblem <: ProblemGenerator
    n_plants::Int
    materials::BlendMaterials
    availability::Vector{Float64}
    order_grade::Vector{Int}
    order_plant::Vector{Int}
    pairs::Vector{Tuple{Int, Int}}
    order_pairs::Vector{UnitRange{Int}}
    lo::Matrix{Float64}
    hi::Matrix{Float64}
    demand_min::Vector{Float64}
    demand_max::Vector{Float64}
    price::Vector{Float64}
    deviation::Vector{Float64}
    gamma::Float64
    robust_rows::Vector{Tuple{Int, Int}}
    robust_terms::Vector{Tuple{Int, Int}}
    feasible_witness::Union{Nothing, RobustBlendPlan}
    infeasibility_certificate::Union{Nothing, BlendInfeasibilityCertificate}
    feasibility_status::FeasibilityStatus
end

# Protected elements of an order: uncertain tramp elements carried by at least
# one of its scrap candidates; terms are those carriers.
function _robust_order_rows(grade::Int, candidates::Vector{Int}, mats::BlendMaterials)
    rows = Tuple{Int, Vector{Int}}[]
    for e in BLEND_UNCERTAIN_ELEMENTS
        carriers = [i for i in candidates if mats.kind[i] == :scrap && mats.comp[e, i] > 0]
        isempty(carriers) || push!(rows, (e, carriers))
    end
    return rows
end

_robust_extra(grade, candidates, mats) =
    sum(1 + length(c) for (_, c) in _robust_order_rows(grade, candidates, mats); init=0)

"""
    _robust_protection(a, gamma) -> (beta, z, p)

Exact budgeted protection `β = max_{|S| ≤ Γ} Σ_{i∈S} a_i` (fractional Γ) of
nonnegative deviations `a`, with the optimal dual pair `z` (the `(⌊Γ⌋+1)`-th
largest deviation) and `p_i = max(a_i − z, 0)`.
"""
function _robust_protection(a::AbstractVector{<:Real}, gamma::Real)
    n = length(a)
    g = min(Float64(gamma), n)
    sorted = sort(collect(Float64, a); rev=true)
    whole = floor(Int, g)
    z = whole < n ? sorted[whole + 1] : 0.0
    p = [max(Float64(ai) - z, 0.0) for ai in a]
    return g * z + sum(p; init=0.0), z, p
end

robust_gamma(prob::RobustBlendingProblem, r::Int) =
    min(prob.gamma, count(t -> t[1] == r, prob.robust_terms))

"""
    robust_plan_satisfies(prob, plan=prob.feasible_witness; atol=1e-7)

Check a `RobustBlendPlan` against every row of the robust model.
"""
function robust_plan_satisfies(
    prob::RobustBlendingProblem, plan::Union{Nothing, RobustBlendPlan}=prob.feasible_witness; atol::Float64=1e-7
)
    plan === nothing && return false
    x, z, p = plan.charge_plan, plan.protection, plan.excess
    length(x) == length(prob.pairs) && length(z) == length(prob.robust_rows) || return false
    length(p) == length(prob.robust_terms) || return false
    all(>=(-atol), x) && all(>=(-atol), z) && all(>=(-atol), p) || return false
    mats = prob.materials
    tol(v) = atol * max(1.0, abs(v))
    pair_of = Dict(prob.pairs[k] => k for k in eachindex(prob.pairs))
    protected = Set(prob.robust_rows)
    for (o, ks) in enumerate(prob.order_pairs)
        out = sum(x[k] * mats.yield[prob.pairs[k][1]] for k in ks)
        prob.demand_min[o] - tol(out) <= out <= prob.demand_max[o] + tol(out) || return false
        mass = sum(x[k] for k in ks)
        for e in eachindex(BLEND_ELEMENTS)
            content = sum(x[k] * mats.comp[e, prob.pairs[k][1]] for k in ks)
            content + tol(mass) >= prob.lo[e, o] * mass || return false
            (o, e) in protected && continue
            content <= prob.hi[e, o] * mass + tol(mass) || return false
        end
    end
    extra = zeros(Float64, length(prob.robust_rows))
    for (t, (r, k)) in enumerate(prob.robust_terms)
        i = prob.pairs[k][1]
        e = prob.robust_rows[r][2]
        z[r] + p[t] + tol(x[k]) >= prob.deviation[i] * mats.comp[e, i] * x[k] || return false
        extra[r] += p[t]
    end
    for (r, (o, e)) in enumerate(prob.robust_rows)
        ks = prob.order_pairs[o]
        mass = sum(x[k] for k in ks)
        content = sum(x[k] * mats.comp[e, prob.pairs[k][1]] for k in ks)
        content + robust_gamma(prob, r) * z[r] + extra[r] <= prob.hi[e, o] * mass + tol(mass) || return false
    end
    usage = _blend_usage(length(prob.availability), prob.pairs, x)
    all(i -> usage[i] <= prob.availability[i] + tol(usage[i]), eachindex(usage)) || return false
    return true
end

"""
    robust_certificate_holds(prob::RobustBlendingProblem)

Recompute the stored (nominal-model) certificate and check it.
"""
robust_certificate_holds(prob::RobustBlendingProblem) = _blend_certificate_holds(
    prob.infeasibility_certificate, prob.pairs, prob.order_pairs, prob.materials, prob.availability,
    prob.lo, prob.demand_min,
)

"""
    RobustBlendingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a budgeted-robust alloy-blending instance (see the type).
"""
function RobustBlendingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    rng = MersenneTwister(seed)
    target = max(target_variables, 1)
    n_plants, mats, order_grade, order_plant, pairs, order_pairs =
        _blend_build_network(rng, target; extra=_robust_extra)
    n_orders = length(order_grade)
    n_materials = length(mats.kind)
    price = [_BLEND_GRADES[g].price * rand(rng, Uniform(0.95, 1.08)) for g in order_grade]
    # Assay reliability: older lots and mixed classes carry wider deviations.
    deviation = [mats.kind[i] == :scrap ? rand(rng, Uniform(0.08, 0.35)) : 0.0 for i in 1:n_materials]
    gamma = rand(rng, Uniform(0.5, 3.0))

    robust_rows = Tuple{Int, Int}[]
    robust_terms = Tuple{Int, Int}[]
    for o in 1:n_orders
        ks = order_pairs[o]
        candidates = [pairs[k][1] for k in ks]
        for (e, carriers) in _robust_order_rows(order_grade[o], candidates, mats)
            push!(robust_rows, (o, e))
            r = length(robust_rows)
            for k in ks
                pairs[k][1] in carriers && push!(robust_terms, (r, k))
            end
        end
    end
    terms_of = [Int[] for _ in robust_rows]
    for (t, (r, _)) in enumerate(robust_terms)
        push!(terms_of[r], t)
    end

    availability = zeros(Float64, n_materials)
    witness = nothing
    if feasibility_status == unknown
        lo, hi = _blend_registered_windows(order_grade)
        nominal = [clamp(rand(rng, LogNormal(log(40.0), 0.7)), 5.0, 400.0) for _ in 1:n_orders]
        demand_min = nominal .* rand(rng, Uniform(0.80, 0.97), n_orders)
        demand_max = nominal .* rand(rng, Uniform(1.15, 1.80), n_orders)
        plant_demand = zeros(Float64, n_plants)
        for o in 1:n_orders
            plant_demand[order_plant[o]] += demand_min[o]
        end
        lots_at = [count(==(p), mats.site) for p in 1:n_plants]
        total = sum(demand_min)
        # One market tightness per instance scales scrap and primary metal
        # against the contracted output, so the book may or may not fit.
        tightness = rand(rng, Uniform(0.6, 1.4))
        for i in 1:n_materials
            if mats.kind[i] == :scrap
                p = mats.site[i]
                availability[i] = plant_demand[p] * tightness * rand(rng, Uniform(0.3, 0.8)) / lots_at[p]
            elseif mats.kind[i] == :primary
                availability[i] = total * tightness * rand(rng, Uniform(0.15, 0.35))
            else
                e = _BLEND_HARDENERS[mats.source[i]].element
                need = sum(demand_min[o] * _BLEND_GRADES[order_grade[o]].hi[e] for o in 1:n_orders) / mats.comp[e, i]
                availability[i] = max(need, 1.0) * rand(rng, Uniform(0.6, 2.0))
            end
        end
    else
        x, lo, hi, output = _blend_planted_orders(rng, mats, order_grade, pairs, order_pairs)
        z = zeros(Float64, length(robust_rows))
        p = zeros(Float64, length(robust_terms))
        for (r, (o, e)) in enumerate(robust_rows)
            ts = terms_of[r]
            a = [deviation[pairs[robust_terms[t][2]][1]] * mats.comp[e, pairs[robust_terms[t][2]][1]] *
                 x[robust_terms[t][2]] for t in ts]
            beta, z[r], pr = _robust_protection(a, gamma)
            p[ts] .= pr
            ks = order_pairs[o]
            mass = sum(x[k] for k in ks)
            content = sum(x[k] * mats.comp[e, pairs[k][1]] for k in ks)
            hi[e, o] = max(hi[e, o], 1.02 * (content + beta) / mass)
        end
        demand_min = output .* rand(rng, Uniform(0.80, 0.97), n_orders)
        demand_max = output .* rand(rng, Uniform(1.15, 1.80), n_orders)
        usage = _blend_usage(n_materials, pairs, x)
        for i in 1:n_materials
            slack = mats.kind[i] == :scrap ? rand(rng, Uniform(1.02, 1.30)) :
                    mats.kind[i] == :primary ? rand(rng, Uniform(1.05, 1.40)) : rand(rng, Uniform(1.1, 2.0))
            availability[i] = usage[i] > 0 ? usage[i] * slack :
                              clamp(rand(rng, LogNormal(log(30.0), 0.7)), 2.0, 300.0)
        end
        witness = RobustBlendPlan(x, z, p)
    end

    certificate = nothing
    if feasibility_status == infeasible
        certificate = _blend_make_infeasible!(rng, availability, mats, order_grade, pairs, order_pairs, lo, demand_min)
        witness = nothing
    end
    prob = RobustBlendingProblem(
        n_plants, mats, availability, order_grade, order_plant, pairs, order_pairs, lo, hi, demand_min,
        demand_max, price, deviation, gamma, robust_rows, robust_terms, witness, certificate, feasibility_status,
    )
    feasibility_status == feasible && @assert robust_plan_satisfies(prob)
    feasibility_status == infeasible && @assert robust_certificate_holds(prob)
    return prob
end

"""
    build_model(prob::RobustBlendingProblem)

Build the budgeted-robust alloy-blending LP (deterministic; see the type).
"""
function build_model(prob::RobustBlendingProblem)
    model = Model()
    mats = prob.materials
    K = length(prob.pairs)
    R = length(prob.robust_rows)
    @variable(model, x[1:K] >= 0)
    @variable(model, charge[1:length(prob.order_grade)] >= 0)
    @variable(model, z[1:R] >= 0)
    @variable(model, p[1:length(prob.robust_terms)] >= 0)
    @objective(
        model,
        Max,
        sum((prob.price[o] * mats.yield[i] - mats.cost[i]) * x[k] for (k, (i, o)) in enumerate(prob.pairs))
    )
    protected = Dict(row => r for (r, row) in enumerate(prob.robust_rows))
    terms_of = [Int[] for _ in 1:R]
    for (t, (r, _)) in enumerate(prob.robust_terms)
        push!(terms_of[r], t)
    end
    for (o, ks) in enumerate(prob.order_pairs)
        maxs = _blend_add_order_rows!(
            model, x, charge[o], mats, prob.pairs, ks, view(prob.lo, :, o), view(prob.hi, :, o),
            prob.demand_min[o], prob.demand_max[o],
        )
        for e in maxs
            haskey(protected, (o, e)) && continue
            @constraint(model, _blend_max_row_expr(x, charge[o], mats, prob.pairs, ks, e, prob.hi[e, o]) <= 0)
        end
    end
    for (r, (o, e)) in enumerate(prob.robust_rows)
        ks = prob.order_pairs[o]
        @constraint(
            model,
            _blend_max_row_expr(x, charge[o], mats, prob.pairs, ks, e, prob.hi[e, o]) +
            robust_gamma(prob, r) * z[r] + sum(p[t] for t in terms_of[r]) <= 0
        )
    end
    for (t, (r, k)) in enumerate(prob.robust_terms)
        i = prob.pairs[k][1]
        e = prob.robust_rows[r][2]
        @constraint(model, z[r] + p[t] - prob.deviation[i] * mats.comp[e, i] * x[k] >= 0)
    end
    uses = [Int[] for _ in eachindex(prob.availability)]
    for (k, (i, _)) in enumerate(prob.pairs)
        push!(uses[i], k)
    end
    for i in eachindex(uses)
        isempty(uses[i]) && continue
        @constraint(model, sum(x[k] for k in uses[i]) <= prob.availability[i])
    end
    return model
end

register_variant(
    :blending,
    :robust,
    RobustBlendingProblem,
    "Alloy blending robust to scrap-assay uncertainty: Bertsimas-Sim budgeted protection of the " *
    "tramp-element limits (auxiliary protection variables and three-term rows per scrap lot)";
    tags=[:production, :blending, :robust, :block_angular],
)
