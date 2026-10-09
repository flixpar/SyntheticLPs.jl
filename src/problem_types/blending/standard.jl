using JuMP
using Random
using Distributions
using StatsBase

"""
Reason a requested-infeasible alloy-blending instance has no feasible plan.

  - `blend_family_shortage`: the contracted minimum output of every order of one
    alloy family exceeds the metal recoverable (`yield × availability`) from all
    materials those orders may use.
  - `blend_element_shortage`: one order cannot reach the minimum of one
    alloying element: even using every carrier of the element up to its full
    availability, the best attainable `Σ (comp − min) x` over charges producing
    the contracted output is negative (an exact covering-knapsack bound).
"""
@enum BlendInfeasibilityKind begin
    blend_family_shortage
    blend_element_shortage
end

"""
    BlendInfeasibilityCertificate

LP-row proof shared by the blending variants. `orders` are the starved orders,
`materials` the materials they may use, `element` the short element
(`blend_element_shortage`, else 0). For a family shortage `achievable` is the
recoverable tonnes and `required` the summed minimum output; for an element
shortage `achievable` is the covering-knapsack optimum and `required = 0`.
`achievable < required` with a clear margin.
"""
struct BlendInfeasibilityCertificate
    kind::BlendInfeasibilityKind
    orders::Vector{Int}
    materials::Vector{Int}
    element::Int
    achievable::Float64
    required::Float64
end

"""
    BlendingProblem <: ProblemGenerator

Multi-plant secondary-aluminium alloy blending: every customer order (an alloy
grade, a tonnage window and a price) is charged at its plant from that plant's
scrap yard plus shared primary metal and master alloys, so that the melt lands
inside the grade's composition window, at maximum margin.

# Formulation

Variables `x[k] ≥ 0`, tonnes of material `i` charged to order `o` for every
compatible pair `k = (i, o)` (`pairs`; a plant's scrap lots only reach its own
orders, scrap classes only enter alloy families their segregation rules allow,
master alloys only enter grades with a minimum on their element), plus the
charge mass `charge[o] ≥ 0` of every order. Maximize
`Σ_k (price[o] · yield[i] − cost[i]) x[k]`. Rows:

  - per order, the charge balance `Σ_i x[i,o] = charge[o]` and the output
    window `demand_min[o] ≤ Σ_i yield[i] x[i,o] ≤ demand_max[o]` (two rows);
  - per order and element, the composition window on the charge in
    element-mass form, `Σ_i comp[e,i] x[i,o] ≥ lo[e,o] · charge[o]` and
    `Σ_i comp[e,i] x[i,o] ≤ hi[e,o] · charge[o]` — emitted only when some
    candidate material lies outside the bound (otherwise the row is implied);
  - per material, `Σ_o x[i,o] ≤ availability[i]` (scrap lots couple the orders
    of a plant; primary metal and master alloys couple every plant).

Dirty scrap is cheap and dilution with primary metal is expensive, so the
optimum is a vertex of many active composition, output and supply rows.

# Sizing

Plants `≈ target / 8000` (1–30), each with `≈ target^0.4` scrap lots (3–150).
Orders are added round-robin over the plants until the number of variables
(compatible pairs plus one charge mass per order) reaches the target, so the
count lands within one order's pair list (a few dozen) of it.

# Feasibility

  - `feasible`: every order gets a hand-written charge recipe
    (`_blend_recipe`); windows are the registered ones, widened only where the
    recipe falls outside; output windows bracket its output; availabilities are
    1.02–1.30× its usage (lots), 1.05–1.40× (primary) and 1.1–2× (master
    alloys). The plan is stored as `feasible_witness` (tonnes per pair).
  - `infeasible`: one mutation with a `BlendInfeasibilityCertificate` — a metal
    shortage for one alloy family (default) or a master-alloy/carrier shortage
    that leaves one order short of an alloying element. Both aggregate several
    rows; neither is visible to a single-row presolve check.
  - `unknown`: registered windows, nominal order tonnages, scrap availability
    drawn against each plant's demand and a primary-metal quota drawn against
    the total; no planted point.
"""
struct BlendingProblem <: ProblemGenerator
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
    feasible_witness::Union{Nothing, Vector{Float64}}
    infeasibility_certificate::Union{Nothing, BlendInfeasibilityCertificate}
    feasibility_status::FeasibilityStatus
end

# Relative frequency of scrap classes in a casthouse yard.
const _BLEND_CLASS_WEIGHTS = [0.6, 1.0, 1.4, 1.2, 1.0, 0.7, 0.4, 0.3, 0.7, 0.6, 0.6, 0.6, 0.4]

"""
    _blend_build_network(rng, target; extra=(grade, candidates, mats) -> 0)

Sample plants, materials (shared primaries and master alloys, per-plant scrap
lots) and orders until the variable count reaches `target`. Each order counts
its compatible pairs, its charge-mass variable and `extra(grade, candidates,
mats)` further variables (the robust variant's protection variables).
"""
function _blend_build_network(rng::AbstractRNG, target::Int; extra=(grade, candidates, mats) -> 0)
    n_plants = clamp(round(Int, target / 8000 * rand(rng, Uniform(0.8, 1.25))), 1, 30)
    lots_mean = clamp(target^0.4, 4.0, 80.0)
    kind, source, comps, cost, yield = _blend_base_materials()
    site = zeros(Int, length(kind))
    class_dist = Categorical(_BLEND_CLASS_WEIGHTS ./ sum(_BLEND_CLASS_WEIGHTS))
    for p in 1:n_plants
        n_lots = clamp(round(Int, lots_mean * rand(rng, LogNormal(0.0, 0.2))), 3, 150)
        for _ in 1:n_lots
            class = rand(rng, class_dist)
            comp, c = _blend_lot(rng, class)
            push!(kind, :scrap)
            push!(source, class)
            push!(comps, comp)
            push!(cost, c)
            push!(yield, _BLEND_SCRAP_CLASSES[class].yield * rand(rng, Uniform(0.97, 1.02)))
            push!(site, p)
        end
    end
    mats = BlendMaterials(kind, source, reduce(hcat, comps), cost, min.(yield, 0.995), site)
    order_grade = Int[]
    order_plant = Int[]
    pairs = Tuple{Int, Int}[]
    order_pairs = UnitRange{Int}[]
    count = 0
    while count < target || isempty(order_grade)
        o = length(order_grade) + 1
        p = (o - 1) % n_plants + 1
        g = _blend_sample_grade(rng)
        candidates = [
            i for
            i in eachindex(kind) if (site[i] == 0 || site[i] == p) && blend_compatible(mats, i, g)
        ]
        push!(order_grade, g)
        push!(order_plant, p)
        first_pair = length(pairs) + 1
        for i in candidates
            push!(pairs, (i, o))
        end
        push!(order_pairs, first_pair:length(pairs))
        count += length(candidates) + 1 + extra(g, candidates, mats)
    end
    return n_plants, mats, order_grade, order_plant, pairs, order_pairs
end

"""
    _blend_plant_witness(rng, mats, order_grade, pairs, order_pairs)

Planted charge per pair, the windows widened around it, and its outputs.
"""
function _blend_planted_orders(
    rng::AbstractRNG, mats::BlendMaterials, order_grade, pairs, order_pairs
)
    n_orders = length(order_grade)
    x = zeros(Float64, length(pairs))
    lo = zeros(Float64, length(BLEND_ELEMENTS), n_orders)
    hi = zeros(Float64, length(BLEND_ELEMENTS), n_orders)
    output = zeros(Float64, n_orders)
    for o in 1:n_orders
        ks = order_pairs[o]
        candidates = [pairs[k][1] for k in ks]
        fractions, composition = _blend_recipe(
            rng, order_grade[o], candidates, mats.kind, mats.source, mats.comp
        )
        out = clamp(rand(rng, LogNormal(log(40.0), 0.7)), 5.0, 400.0)
        mass = out / sum(fractions[j] * mats.yield[candidates[j]] for j in eachindex(candidates))
        x[ks] .= mass .* fractions
        output[o] = sum(x[k] * mats.yield[pairs[k][1]] for k in ks)
        lo[:, o], hi[:, o] = _blend_window(order_grade[o], composition)
    end
    return x, lo, hi, output
end

function _blend_usage(n_materials::Int, pairs, x)
    usage = zeros(Float64, n_materials)
    for (k, (i, _)) in enumerate(pairs)
        usage[i] += x[k]
    end
    return usage
end

"""Order windows straight from the registry (no planted point)."""
function _blend_registered_windows(order_grade)
    lo = [Float64(_BLEND_GRADES[g].lo[e]) for e in eachindex(BLEND_ELEMENTS), g in order_grade]
    hi = [Float64(_BLEND_GRADES[g].hi[e]) for e in eachindex(BLEND_ELEMENTS), g in order_grade]
    return lo, hi
end

"""
    blend_charge_satisfies(prob, x=prob.feasible_witness; atol=1e-7)

Check a charge plan (tonnes per pair) against every row of a `BlendingProblem`.
"""
function blend_charge_satisfies(
    prob::BlendingProblem,
    x::Union{Nothing, AbstractVector{<:Real}}=prob.feasible_witness;
    atol::Float64=1e-7,
)
    x === nothing && return false
    length(x) == length(prob.pairs) || return false
    all(>=(-atol), x) || return false
    mats = prob.materials
    tol(v) = atol * max(1.0, abs(v))
    for (o, ks) in enumerate(prob.order_pairs)
        out = sum(x[k] * mats.yield[prob.pairs[k][1]] for k in ks)
        prob.demand_min[o] - tol(out) <= out <= prob.demand_max[o] + tol(out) || return false
        mass = sum(x[k] for k in ks)
        for e in eachindex(BLEND_ELEMENTS)
            content = sum(x[k] * mats.comp[e, prob.pairs[k][1]] for k in ks)
            content + tol(mass) >= prob.lo[e, o] * mass || return false
            content <= prob.hi[e, o] * mass + tol(mass) || return false
        end
    end
    usage = _blend_usage(length(prob.availability), prob.pairs, x)
    all(i -> usage[i] <= prob.availability[i] + tol(usage[i]), eachindex(usage)) || return false
    return true
end

function _blend_order_materials(pairs, order_pairs, orders)
    return sort!(unique!([pairs[k][1] for o in orders for k in order_pairs[o]]))
end

"""
    blend_certificate_value(cert, pairs, order_pairs, materials, availability, lo, demand_min)

Recompute `(achievable, required)` of a blending certificate from instance data.
"""
function blend_certificate_value(
    cert::BlendInfeasibilityCertificate,
    pairs,
    order_pairs,
    mats::BlendMaterials,
    availability,
    lo,
    demand_min,
)
    if cert.kind == blend_family_shortage
        materials = _blend_order_materials(pairs, order_pairs, cert.orders)
        materials == cert.materials || return (NaN, NaN)
        achievable = sum(mats.yield[i] * availability[i] for i in materials)
        required = sum(demand_min[o] for o in cert.orders)
    else
        length(cert.orders) == 1 || return (NaN, NaN)
        o = only(cert.orders)
        e = cert.element
        materials = [pairs[k][1] for k in order_pairs[o]]
        materials == cert.materials || return (NaN, NaN)
        a = [mats.comp[e, i] - lo[e, o] for i in materials]
        achievable = _blend_max_excess(
            a, mats.yield[materials], availability[materials], demand_min[o]
        )
        required = 0.0
    end
    return achievable, required
end

function _blend_certificate_holds(cert, pairs, order_pairs, mats, availability, lo, demand_min)
    cert === nothing && return false
    achievable, required = blend_certificate_value(
        cert, pairs, order_pairs, mats, availability, lo, demand_min
    )
    isfinite(required) || return false
    isapprox(achievable, cert.achievable; rtol=1e-8, atol=1e-9) ||
        achievable == cert.achievable ||
        return false
    isapprox(required, cert.required; rtol=1e-8, atol=1e-9) || return false
    return achievable < required - 1e-9 * max(1.0, abs(required))
end

"""
    blend_certificate_holds(prob::BlendingProblem)

Recompute the stored certificate from the data and check `achievable < required`.
"""
blend_certificate_holds(prob::BlendingProblem) = _blend_certificate_holds(
    prob.infeasibility_certificate,
    prob.pairs,
    prob.order_pairs,
    prob.materials,
    prob.availability,
    prob.lo,
    prob.demand_min,
)

"""
    _blend_make_infeasible!(rng, availability, mats, order_grade, pairs, order_pairs, lo, demand_min)

Mutate `availability` so the instance is infeasible and return the certificate:
an element shortage for one order (when one exists whose non-carrier materials
alone lose at least 5% of the element target), else a family metal shortage.
"""
function _blend_make_infeasible!(
    rng, availability, mats, order_grade, pairs, order_pairs, lo, demand_min
)
    n_orders = length(order_grade)
    if rand(rng) < 0.4
        for o in shuffle(rng, collect(1:n_orders))
            materials = [pairs[k][1] for k in order_pairs[o]]
            for e in shuffle(rng, findall(>(0.0), view(lo, :, o)))
                a = [mats.comp[e, i] - lo[e, o] for i in materials]
                margin = 0.05 * lo[e, o] * demand_min[o]
                carriers = findall(>(0.0), a)
                isempty(carriers) && continue
                upper = availability[materials]
                cut = copy(upper)
                cut[carriers] .= 0.0
                _blend_max_excess(a, mats.yield[materials], cut, demand_min[o]) < -2 * margin ||
                    continue
                # Bisection on the carriers' availability (monotone in theta).
                lo_t, hi_t = 0.0, 1.0
                for _ in 1:60
                    mid = (lo_t + hi_t) / 2
                    cut[carriers] .= mid .* upper[carriers]
                    v = _blend_max_excess(a, mats.yield[materials], cut, demand_min[o])
                    v < -margin ? (lo_t = mid) : (hi_t = mid)
                end
                for (j, i) in enumerate(materials)
                    j in carriers && (availability[i] = lo_t * upper[j])
                end
                value = _blend_max_excess(
                    a, mats.yield[materials], availability[materials], demand_min[o]
                )
                return BlendInfeasibilityCertificate(
                    blend_element_shortage, [o], materials, e, value, 0.0
                )
            end
        end
    end
    # Family metal shortage. Only families whose every order could still be
    # filled on its own after the cut qualify: then no single output row is
    # contradicted by the (implied) column bounds, and the proof needs the
    # whole family's rows — invisible to presolve.
    margin = rand(rng, Uniform(1.08, 1.25))
    best = nothing
    for family in shuffle(rng, unique([_blend_family(g) for g in order_grade]))
        orders = [o for o in 1:n_orders if _blend_family(order_grade[o]) == family]
        materials = _blend_order_materials(pairs, order_pairs, orders)
        required = sum(demand_min[o] for o in orders)
        theta = required / margin / sum(mats.yield[i] * availability[i] for i in materials)
        solo(o) =
            theta *
            sum(mats.yield[pairs[k][1]] * availability[pairs[k][1]] for k in order_pairs[o]) /
            demand_min[o]
        slack = minimum(solo(o) for o in orders)
        (best === nothing || slack > best[1]) &&
            (best = (slack, orders, materials, required, theta))
        slack >= 1.1 && break
    end
    _, orders, materials, required, theta = best
    for i in materials
        availability[i] *= theta
    end
    achievable = sum(mats.yield[i] * availability[i] for i in materials)
    return BlendInfeasibilityCertificate(
        blend_family_shortage, orders, materials, 0, achievable, required
    )
end

"""
    BlendingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a multi-plant alloy-blending instance (see `BlendingProblem`).
"""
function BlendingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    rng = MersenneTwister(seed)
    target = max(target_variables, 1)
    n_plants, mats, order_grade, order_plant, pairs, order_pairs = _blend_build_network(rng, target)
    n_orders = length(order_grade)
    n_materials = length(mats.kind)
    price = [_BLEND_GRADES[g].price * rand(rng, Uniform(0.95, 1.08)) for g in order_grade]
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
                availability[i] =
                    plant_demand[p] * tightness * rand(rng, Uniform(0.3, 0.8)) / lots_at[p]
            elseif mats.kind[i] == :primary
                availability[i] = total * tightness * rand(rng, Uniform(0.15, 0.35))
            else
                e = _BLEND_HARDENERS[mats.source[i]].element
                need =
                    sum(demand_min[o] * _BLEND_GRADES[order_grade[o]].hi[e] for o in 1:n_orders) /
                    mats.comp[e, i]
                availability[i] = max(need, 1.0) * rand(rng, Uniform(0.6, 2.0))
            end
        end
    else
        x, lo, hi, output = _blend_planted_orders(rng, mats, order_grade, pairs, order_pairs)
        demand_min = output .* rand(rng, Uniform(0.80, 0.97), n_orders)
        demand_max = output .* rand(rng, Uniform(1.15, 1.80), n_orders)
        usage = _blend_usage(n_materials, pairs, x)
        for i in 1:n_materials
            slack = if mats.kind[i] == :scrap
                rand(rng, Uniform(1.02, 1.30))
            elseif mats.kind[i] == :primary
                rand(rng, Uniform(1.05, 1.40))
            else
                rand(rng, Uniform(1.1, 2.0))
            end
            availability[i] = if usage[i] > 0
                usage[i] * slack
            else
                clamp(rand(rng, LogNormal(log(30.0), 0.7)), 2.0, 300.0)
            end
        end
        witness = x
    end

    certificate = nothing
    if feasibility_status == infeasible
        certificate = _blend_make_infeasible!(
            rng, availability, mats, order_grade, pairs, order_pairs, lo, demand_min
        )
        witness = nothing
    end
    prob = BlendingProblem(
        n_plants,
        mats,
        availability,
        order_grade,
        order_plant,
        pairs,
        order_pairs,
        lo,
        hi,
        demand_min,
        demand_max,
        price,
        witness,
        certificate,
        feasibility_status,
    )
    feasibility_status == feasible && @assert blend_charge_satisfies(prob)
    feasibility_status == infeasible && @assert blend_certificate_holds(prob)
    return prob
end

"""
    _blend_add_order_rows!(model, x, charge, mats, pairs, ks, lo, hi, dmin, dmax)

Charge-mass balance, output window and active composition-minimum rows of one
order (shared by the variants); returns the elements whose maximum row can
bind. Composition rows are written in element-mass form,
`Σ_i comp[e,i] x[i] − lo[e] · charge ≥ 0`, with the explicit charge mass
`charge = Σ_i x[i]`: the coefficients are the assay values themselves rather
than the differences `comp − lo`, which cancel to tiny numbers for materials
close to the limit and make the simplex bases badly conditioned.
"""
function _blend_add_order_rows!(
    model, x, charge, mats::BlendMaterials, pairs, ks, lo, hi, dmin, dmax
)
    materials = [pairs[k][1] for k in ks]
    @constraint(model, sum(x[k] for k in ks) - charge == 0)
    @constraint(model, sum(mats.yield[pairs[k][1]] * x[k] for k in ks) >= dmin)
    @constraint(model, sum(mats.yield[pairs[k][1]] * x[k] for k in ks) <= dmax)
    mins, maxs = _blend_active_rows(mats.comp, materials, lo, hi)
    for e in mins
        carriers = [k for k in ks if mats.comp[e, pairs[k][1]] > 0]
        @constraint(
            model, sum(mats.comp[e, pairs[k][1]] * x[k] for k in carriers) - lo[e] * charge >= 0
        )
    end
    return maxs
end

"""Element-mass maximum row `Σ_i comp[e,i] x[i] − hi · charge ≤ 0` of one order."""
function _blend_max_row_expr(x, charge, mats::BlendMaterials, pairs, ks, e, hi)
    carriers = [k for k in ks if mats.comp[e, pairs[k][1]] > 0]
    return @expression(
        owner_model(charge), sum(mats.comp[e, pairs[k][1]] * x[k] for k in carriers) - hi * charge
    )
end

"""
    build_model(prob::BlendingProblem)

Build the multi-plant alloy-blending LP (deterministic; see `BlendingProblem`).
"""
function build_model(prob::BlendingProblem)
    model = Model()
    mats = prob.materials
    K = length(prob.pairs)
    @variable(model, x[1:K] >= 0)
    @variable(model, charge[1:length(prob.order_grade)] >= 0)
    @objective(
        model,
        Max,
        sum(
            (prob.price[o] * mats.yield[i] - mats.cost[i]) * x[k] for
            (k, (i, o)) in enumerate(prob.pairs)
        )
    )
    for (o, ks) in enumerate(prob.order_pairs)
        maxs = _blend_add_order_rows!(
            model,
            x,
            charge[o],
            mats,
            prob.pairs,
            ks,
            view(prob.lo, :, o),
            view(prob.hi, :, o),
            prob.demand_min[o],
            prob.demand_max[o],
        )
        for e in maxs
            @constraint(
                model,
                _blend_max_row_expr(x, charge[o], mats, prob.pairs, ks, e, prob.hi[e, o]) <= 0
            )
        end
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
    :standard,
    BlendingProblem,
    "Multi-plant secondary-aluminium alloy blending: scrap lots, primary metal and master " *
    "alloys charged to customer orders inside registered composition windows at maximum margin";
    default=true,
    tags=[:production, :blending, :block_angular],
)
