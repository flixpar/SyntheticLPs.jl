using JuMP
using Random
using Distributions
using StatsBase

"""
Nutrients tracked by the food-aid variant (WFP NutVal planning set): energy
kcal, protein g, fat g, calcium mg, iron mg, zinc mg, vitamin A µg RAE and
vitamin C mg.
"""
const AID_NUTRIENTS = (:energy, :protein, :fat, :calcium, :iron, :zinc, :vitamin_a, :vitamin_c)

# Food-aid commodity catalog: family, nutrients per 100 g (AID_NUTRIENTS order)
# and an international reference price in USD per tonne.
const _AID_COMMODITIES = (
    (name=:maize_grain, family=:cereal, per100=(350.0, 10.0, 4.5, 7.0, 2.7, 2.2, 0.0, 0.0), price=280.0),
    (name=:maize_meal_fortified, family=:cereal, per100=(360.0, 9.0, 3.5, 7.0, 4.5, 3.0, 150.0, 0.0), price=420.0),
    (name=:wheat_flour_fortified, family=:cereal, per100=(350.0, 11.5, 1.5, 29.0, 5.0, 2.8, 150.0, 0.0), price=380.0),
    (name=:wheat_grain, family=:cereal, per100=(330.0, 12.3, 1.5, 36.0, 4.0, 3.0, 0.0, 0.0), price=290.0),
    (name=:rice, family=:cereal, per100=(360.0, 7.0, 0.5, 7.0, 0.7, 1.2, 0.0, 0.0), price=480.0),
    (name=:sorghum, family=:cereal, per100=(335.0, 11.0, 3.0, 26.0, 4.5, 2.0, 0.0, 0.0), price=260.0),
    (name=:bulgur, family=:cereal, per100=(350.0, 11.0, 1.5, 23.0, 3.5, 1.9, 0.0, 0.0), price=450.0),
    (name=:lentils, family=:pulse, per100=(340.0, 20.0, 1.2, 51.0, 9.0, 3.1, 4.0, 0.0), price=700.0),
    (name=:split_peas, family=:pulse, per100=(335.0, 22.0, 1.4, 52.0, 4.4, 3.0, 7.0, 0.0), price=520.0),
    (name=:beans, family=:pulse, per100=(335.0, 20.0, 1.2, 143.0, 8.2, 2.8, 0.0, 0.0), price=750.0),
    (name=:chickpeas, family=:pulse, per100=(335.0, 19.0, 6.0, 105.0, 6.2, 3.4, 3.0, 0.0), price=800.0),
    (name=:vegetable_oil, family=:oil, per100=(885.0, 0.0, 100.0, 0.0, 0.0, 0.0, 900.0, 0.0), price=1300.0),
    (name=:supercereal, family=:supercereal, per100=(380.0, 14.0, 6.0, 831.0, 12.5, 5.0, 1400.0, 100.0), price=650.0),
    (name=:supercereal_plus, family=:supercereal_plus, per100=(410.0, 16.4, 9.2, 1100.0, 7.0, 5.0, 1600.0, 90.0), price=1100.0),
    (name=:lns_medium_quantity, family=:lns, per100=(530.0, 13.0, 35.0, 280.0, 9.0, 9.0, 400.0, 30.0), price=2600.0),
    (name=:sugar, family=:sugar, per100=(400.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0), price=500.0),
    (name=:dried_skim_milk, family=:dairy, per100=(360.0, 36.0, 1.0, 1300.0, 0.5, 4.0, 1500.0, 0.0), price=3200.0),
    (name=:canned_fish, family=:fish, per100=(200.0, 20.0, 12.0, 300.0, 2.5, 1.2, 30.0, 0.0), price=3500.0),
    (name=:dates, family=:fruit, per100=(280.0, 2.5, 0.4, 39.0, 1.0, 0.3, 0.0, 0.4), price=1100.0),
    (name=:high_energy_biscuits, family=:biscuit, per100=(450.0, 12.0, 15.0, 500.0, 10.0, 7.0, 450.0, 50.0), price=2200.0),
)
const _AID_CORE_FAMILIES = (:cereal, :pulse, :oil, :supercereal, :supercereal_plus, :sugar)

# Beneficiary programme types: NutVal-style daily targets (AID_NUTRIENTS
# order), the commodity families a ration may contain with their maximum
# grams per person per day, the reference basket (family => grams), the
# minimum share of energy from fat, and the share of sites of this type.
const _AID_PROGRAMMES = (
    (name=:general_distribution,
        target=(2100.0, 52.5, 40.0, 989.0, 22.0, 12.4, 550.0, 41.6),
        max_g=Dict(:cereal => 500.0, :pulse => 90.0, :oil => 40.0, :supercereal => 80.0,
            :sugar => 30.0, :fish => 60.0, :fruit => 50.0, :dairy => 30.0),
        basket=(:cereal => 420.0, :pulse => 60.0, :oil => 27.0, :supercereal => 50.0, :sugar => 18.0),
        fat_share=0.17, weight=0.50),
    (name=:school_meals,
        target=(700.0, 21.0, 15.0, 300.0, 6.0, 3.5, 250.0, 15.0),
        max_g=Dict(:cereal => 180.0, :pulse => 50.0, :oil => 15.0, :supercereal => 80.0,
            :sugar => 15.0, :biscuit => 100.0, :dairy => 30.0, :fruit => 30.0),
        basket=(:cereal => 140.0, :pulse => 35.0, :oil => 10.0, :supercereal => 40.0),
        fat_share=0.15, weight=0.25),
    (name=:child_supplementary,
        target=(500.0, 12.5, 15.0, 400.0, 9.0, 4.5, 400.0, 30.0),
        max_g=Dict(:supercereal_plus => 250.0, :lns => 100.0, :oil => 25.0, :sugar => 20.0,
            :dairy => 40.0),
        basket=(:supercereal_plus => 200.0, :oil => 10.0),
        fat_share=0.20, weight=0.15),
    (name=:pregnant_lactating,
        target=(1000.0, 33.0, 25.0, 650.0, 18.0, 7.0, 500.0, 45.0),
        max_g=Dict(:supercereal => 300.0, :oil => 30.0, :sugar => 25.0, :pulse => 60.0,
            :cereal => 200.0, :lns => 60.0),
        basket=(:supercereal => 250.0, :oil => 25.0, :sugar => 15.0),
        fat_share=0.17, weight=0.10),
)

const AID_DAYS_PER_CYCLE = 30.0

"""
Reason a requested-infeasible food-aid instance has no feasible plan.

  - `aid_hub_throughput`: the sites served only by hub `hub` need, just to reach
    their energy minimums within the ration limits, more tonnes than the hub can
    handle in the cycle.
  - `aid_nutrient_shortage`: summed over all sites, the requirement for nutrient
    `nutrient` exceeds what the whole procurement pipeline (every source at
    capacity) can carry.
"""
@enum AidInfeasibilityKind begin
    aid_hub_throughput
    aid_nutrient_shortage
end

"""
    AidInfeasibilityCertificate

LP-row proof for `FoodAidDietProblem`. `hub`/`sites` identify the starved hub
and its single-hub sites (`aid_hub_throughput`), `nutrient` the short nutrient
(`aid_nutrient_shortage`); unused fields are 0/empty. Quantities are in tonnes
per cycle (hub) or nutrient-units·t/g (shortage); `achievable < required` by at
least 7%.
"""
struct AidInfeasibilityCertificate
    kind::AidInfeasibilityKind
    hub::Int
    sites::Vector{Int}
    nutrient::Int
    achievable::Float64
    required::Float64
end

"""
    AidPlan

A complete food-aid plan: `ration[k]` grams/person/day for ration pair `k`,
`delivery[k]` tonnes on delivery arc `k` (hub → site) and `procurement[k]`
tonnes on procurement arc `k` (source → hub).
"""
struct AidPlan
    ration::Vector{Float64}
    delivery::Vector{Float64}
    procurement::Vector{Float64}
end

"""
    FoodAidDietProblem <: ProblemGenerator

Humanitarian food-basket design with sourcing, after WFP's Optimus model:
choose the ration of every distribution site and the procurement and delivery
plan that supplies it, at least cost.

# Formulation

Sites `j` belong to a programme (general distribution, school meals, child
supplementary feeding, pregnant/lactating women) with NutVal nutrient targets,
allowed commodities and per-commodity ration limits. Each site is served by
one or two hubs (extended delivery points); commodities are bought from
international, regional and local sources with limited capacity.

  - `r[k] ∈ [0, ration_upper[k]]` g/person/day for ration pair `k = (c, j)`;
  - `y[k] ≥ 0` tonnes on delivery arc `k = (c, h, j)` — only for sites served
    by two hubs; a single-hub site's delivery is fixed by its ration, so it
    enters its hub's balance (and the objective) through `r` directly;
  - `q[k] ≥ 0` tonnes on procurement arc `k = (c, s, h)`.

Minimize `Σ procurement_cost·q + Σ delivery_cost·y`. Rows:

  - per site: energy band (ranged), minimums for protein, fat, calcium, iron,
    zinc, vitamin A and vitamin C, and a minimum share of energy from fat
    (homogeneous, mixed signs);
  - per ration pair of a two-hub site: `Σ_h y[c,h,j] = 30·10⁻⁶·beneficiaries[j]·r[c,j]`
    (grams per person-day to tonnes per 30-day cycle);
  - per (commodity, hub): `Σ_s q[c,s,h] ≥ Σ_j y[c,h,j]`;
  - per (commodity, source): `Σ_h q[c,s,h] ≤ source_capacity`;
  - per hub: `Σ q[·,·,h] ≤ hub_capacity[h]`.

# Sizing

Hubs (`≈ √target / 6`), sources and the commodity list are fixed first, then
sites are added until the variable count reaches the target (each site adds
`|allowed|` ration variables, plus `2|allowed|` delivery arcs if it has two hubs), so the count lands within one site
(~10–40 variables) of the target.

# Feasibility

  - `feasible`: each site gets its programme's reference basket (one commodity
    per basket family, ±15%); requirements are the NutVal targets lowered only
    where the basket falls short; deliveries split over the site's hubs and
    procurement over each commodity's sources; source and hub capacities are
    1.02–1.30× the planted flows. Stored as an `AidPlan` witness.
  - `infeasible`: one mutation with a typed `AidInfeasibilityCertificate`: a hub
    whose single-hub sites cannot reach their energy minimum through it
    (default, when such sites exist) or a pipeline-wide nutrient shortage.
  - `unknown`: NutVal targets and capacities drawn around nominal demand.
"""
struct FoodAidDietProblem <: ProblemGenerator
    commodities::Vector{Int}
    content::Matrix{Float64}
    n_sources::Int
    n_hubs::Int
    n_sites::Int
    site_programme::Vector{Int}
    beneficiaries::Vector{Float64}
    site_hubs::Vector{Vector{Int}}
    ration_pairs::Vector{Tuple{Int, Int}}
    ration_upper::Vector{Float64}
    delivery_arcs::Vector{Tuple{Int, Int, Int}}
    delivery_cost::Vector{Float64}
    site_delivery_cost::Vector{Float64}
    procurement_arcs::Vector{Tuple{Int, Int, Int}}
    procurement_cost::Vector{Float64}
    source_capacity::Dict{Tuple{Int, Int}, Float64}
    hub_capacity::Vector{Float64}
    requirement::Matrix{Float64}
    energy_upper::Vector{Float64}
    fat_share::Vector{Float64}
    feasible_witness::Union{Nothing, AidPlan}
    infeasibility_certificate::Union{Nothing, AidInfeasibilityCertificate}
    feasibility_status::FeasibilityStatus
end

# Nutrient per gram of commodity c (row k) for the instance's commodity list.
function _aid_content(commodities::Vector{Int})
    return [_AID_COMMODITIES[c].per100[k] / 100.0 for k in eachindex(AID_NUTRIENTS), c in commodities]
end

"""
    _aid_min_ration_mass(energy, upper, energy_min)

Exact minimum of `Σ r` subject to `Σ energy·r ≥ energy_min`, `0 ≤ r ≤ upper`
(fill the most energy-dense commodities first); `Inf` if unreachable.
"""
function _aid_min_ration_mass(energy::AbstractVector, upper::AbstractVector, energy_min::Real)
    remaining = Float64(energy_min)
    mass = 0.0
    for i in sortperm(energy; rev=true)
        remaining <= 0.0 && break
        energy[i] > 0.0 || break
        amount = min(Float64(upper[i]), remaining / energy[i])
        mass += amount
        remaining -= energy[i] * amount
    end
    return remaining <= 1e-9 * max(1.0, energy_min) ? mass : Inf
end

function _aid_site_pairs(prob::FoodAidDietProblem)
    pairs = [Int[] for _ in 1:prob.n_sites]
    for (k, (_, j)) in enumerate(prob.ration_pairs)
        push!(pairs[j], k)
    end
    return pairs
end

function _aid_hub_min_tonnes(prob::FoodAidDietProblem, j::Int, site_pairs)
    ks = site_pairs[j]
    energy = [prob.content[1, prob.ration_pairs[k][1]] for k in ks]
    mass = _aid_min_ration_mass(energy, prob.ration_upper[ks], prob.requirement[1, j])
    return AID_DAYS_PER_CYCLE * 1e-6 * prob.beneficiaries[j] * mass
end

function _aid_pipeline_capacity(prob::FoodAidDietProblem, nutrient::Int)
    total = 0.0
    for ((c, _), cap) in prob.source_capacity
        total += prob.content[nutrient, c] * cap
    end
    return total
end

function _aid_pipeline_requirement(prob::FoodAidDietProblem, nutrient::Int)
    return sum(
        AID_DAYS_PER_CYCLE * 1e-6 * prob.beneficiaries[j] * prob.requirement[nutrient, j] for
        j in 1:prob.n_sites
    )
end

"""
    aid_certificate_holds(prob::FoodAidDietProblem; rtol=1e-9)

Recompute the stored certificate from the data and check `achievable < required`.
"""
function aid_certificate_holds(prob::FoodAidDietProblem; rtol::Float64=1e-9)
    cert = prob.infeasibility_certificate
    cert === nothing && return false
    if cert.kind == aid_hub_throughput
        1 <= cert.hub <= prob.n_hubs || return false
        isempty(cert.sites) && return false
        all(j -> prob.site_hubs[j] == [cert.hub], cert.sites) || return false
        site_pairs = _aid_site_pairs(prob)
        required = sum(_aid_hub_min_tonnes(prob, j, site_pairs) for j in cert.sites)
        achievable = prob.hub_capacity[cert.hub]
    else
        2 <= cert.nutrient <= length(AID_NUTRIENTS) || return false
        achievable = _aid_pipeline_capacity(prob, cert.nutrient)
        required = _aid_pipeline_requirement(prob, cert.nutrient)
    end
    isapprox(achievable, cert.achievable; rtol=1e-8, atol=1e-9) || return false
    isapprox(required, cert.required; rtol=1e-8, atol=1e-9) || return false
    return achievable < required * (1 - rtol)
end

"""
    aid_plan_satisfies(prob::FoodAidDietProblem, plan=prob.feasible_witness; atol=1e-7)

Check an `AidPlan` against every bound and row of the model without a solver.
"""
function aid_plan_satisfies(
    prob::FoodAidDietProblem,
    plan::Union{Nothing, AidPlan}=prob.feasible_witness;
    atol::Float64=1e-7,
)
    plan === nothing && return false
    r, y, q = plan.ration, plan.delivery, plan.procurement
    length(r) == length(prob.ration_pairs) || return false
    length(y) == length(prob.delivery_arcs) && length(q) == length(prob.procurement_arcs) ||
        return false
    tol(v) = atol * max(1.0, abs(v))
    all(>=(-atol), r) && all(>=(-atol), y) && all(>=(-atol), q) || return false
    all(k -> r[k] <= prob.ration_upper[k] + tol(prob.ration_upper[k]), eachindex(r)) || return false
    intake = zeros(Float64, length(AID_NUTRIENTS), prob.n_sites)
    for (k, (c, j)) in enumerate(prob.ration_pairs)
        intake[:, j] .+= r[k] .* view(prob.content, :, c)
    end
    for j in 1:prob.n_sites
        e = intake[1, j]
        prob.requirement[1, j] - tol(e) <= e <= prob.energy_upper[j] + tol(e) || return false
        for n in 2:length(AID_NUTRIENTS)
            intake[n, j] + tol(intake[n, j]) >= prob.requirement[n, j] || return false
        end
        9.0 * intake[3, j] + tol(e) >= prob.fat_share[j] * e || return false
    end
    delivered = Dict{Tuple{Int, Int}, Float64}()
    hub_out = Dict{Tuple{Int, Int}, Float64}()
    for (k, (c, h, j)) in enumerate(prob.delivery_arcs)
        delivered[(c, j)] = get(delivered, (c, j), 0.0) + y[k]
        hub_out[(c, h)] = get(hub_out, (c, h), 0.0) + y[k]
    end
    for (k, (c, j)) in enumerate(prob.ration_pairs)
        need = AID_DAYS_PER_CYCLE * 1e-6 * prob.beneficiaries[j] * r[k]
        if length(prob.site_hubs[j]) == 1
            # A single-hub site draws its ration straight from its hub.
            h = only(prob.site_hubs[j])
            hub_out[(c, h)] = get(hub_out, (c, h), 0.0) + need
        else
            abs(get(delivered, (c, j), 0.0) - need) <= tol(need) || return false
        end
    end
    hub_in = Dict{Tuple{Int, Int}, Float64}()
    source_out = Dict{Tuple{Int, Int}, Float64}()
    hub_total = zeros(Float64, prob.n_hubs)
    for (k, (c, s, h)) in enumerate(prob.procurement_arcs)
        hub_in[(c, h)] = get(hub_in, (c, h), 0.0) + q[k]
        source_out[(c, s)] = get(source_out, (c, s), 0.0) + q[k]
        hub_total[h] += q[k]
    end
    for ((c, h), out) in hub_out
        get(hub_in, (c, h), 0.0) + tol(out) >= out || return false
    end
    for ((c, s), out) in source_out
        out <= prob.source_capacity[(c, s)] + tol(out) || return false
    end
    all(h -> hub_total[h] <= prob.hub_capacity[h] + tol(hub_total[h]), 1:prob.n_hubs) || return false
    return true
end

"""
    FoodAidDietProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a food-aid ration and sourcing instance (see `FoodAidDietProblem`).
"""
function FoodAidDietProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    rng = MersenneTwister(seed)
    target = max(target_variables, 1)

    # --- network skeleton -------------------------------------------------
    n_hubs = clamp(round(Int, sqrt(target) / 6 * rand(rng, Uniform(0.8, 1.25))), 1, 80)
    n_sources = clamp(round(Int, 1.0 + log(target) / 1.6), 1, 8)
    n_commodities = clamp(round(Int, 4 + target^0.28), 6, length(_AID_COMMODITIES))
    core = [first(shuffle(rng, [i for i in eachindex(_AID_COMMODITIES) if _AID_COMMODITIES[i].family == fam])) for
            fam in _AID_CORE_FAMILIES]
    others = shuffle(rng, setdiff(collect(eachindex(_AID_COMMODITIES)), core))
    commodities = sort(vcat(core, others[1:max(0, n_commodities - length(core))]))
    content = _aid_content(commodities)
    n_comm = length(commodities)
    family = [_AID_COMMODITIES[c].family for c in commodities]

    hub_xy = [rand(rng, Uniform(0.0, 1000.0), 2) for _ in 1:n_hubs]
    # Source 1 is the international port; regional and local markets follow.
    source_kind = [s == 1 ? :international : (isodd(s) ? :local : :regional) for s in 1:n_sources]
    source_xy = [
        source_kind[s] == :international ? [0.0, rand(rng, Uniform(200.0, 800.0))] :
        source_kind[s] == :regional ? [rand(rng, Uniform(800.0, 1100.0)), rand(rng, Uniform(0.0, 1000.0))] :
        rand(rng, Uniform(100.0, 900.0), 2) for s in 1:n_sources
    ]
    offered = Dict{Int, Vector{Int}}()
    price = Dict{Tuple{Int, Int}, Float64}()
    for c in 1:n_comm
        srcs = [s for s in 1:n_sources if s == 1 || rand(rng) < 0.55]
        offered[c] = srcs
        base = _AID_COMMODITIES[commodities[c]].price
        for s in srcs
            factor = source_kind[s] == :international ? 1.0 :
                     source_kind[s] == :regional ? rand(rng, Uniform(0.85, 1.15)) :
                     rand(rng, Uniform(0.75, 1.35))
            price[(c, s)] = base * factor * rand(rng, LogNormal(0.0, 0.05))
        end
    end
    procurement_arcs = Tuple{Int, Int, Int}[]
    procurement_cost = Float64[]
    for c in 1:n_comm, s in offered[c], h in 1:n_hubs
        dist = hypot((source_xy[s] .- hub_xy[h])...)
        sea = source_kind[s] == :international ? 60.0 : 0.0
        push!(procurement_arcs, (c, s, h))
        push!(procurement_cost, price[(c, s)] + sea + 15.0 + 0.09 * dist)
    end

    # --- sites, added until the variable count reaches the target -----------
    weights = [p.weight for p in _AID_PROGRAMMES]
    programme_dist = Categorical(weights ./ sum(weights))
    site_programme = Int[]
    beneficiaries = Float64[]
    site_hubs = Vector{Vector{Int}}()
    ration_pairs = Tuple{Int, Int}[]
    ration_upper = Float64[]
    delivery_arcs = Tuple{Int, Int, Int}[]
    delivery_cost = Float64[]
    single_cost = Float64[]
    n_vars = length(procurement_arcs)
    while n_vars < target || isempty(site_programme)
        p = rand(rng, programme_dist)
        allowed = [c for c in 1:n_comm if haskey(_AID_PROGRAMMES[p].max_g, family[c])]
        isempty(allowed) && continue
        j = length(site_programme) + 1
        home = rand(rng, 1:n_hubs)
        xy = hub_xy[home] .+ rand(rng, Normal(0.0, 60.0), 2)
        order = sortperm([hypot((xy .- hub_xy[h])...) for h in 1:n_hubs])
        hubs = n_hubs >= 2 && rand(rng) < 0.35 ? sort(order[1:2]) : [order[1]]
        push!(site_programme, p)
        push!(site_hubs, hubs)
        push!(beneficiaries, Float64(max(200, round(Int, rand(rng, LogNormal(log(3000.0), 0.9))))))
        for c in allowed
            push!(ration_pairs, (c, j))
            push!(ration_upper, _AID_PROGRAMMES[p].max_g[family[c]] * rand(rng, Uniform(0.85, 1.15)))
            length(hubs) == 1 && continue
            for h in hubs
                push!(delivery_arcs, (c, h, j))
                push!(delivery_cost, 8.0 + 0.2 * hypot((xy .- hub_xy[h])...))
            end
        end
        if length(hubs) == 1
            # Delivery to a single-hub site is fixed by its ration; its cost
            # per tonne is folded into the ration variable's objective.
            push!(single_cost, 8.0 + 0.2 * hypot((xy .- hub_xy[only(hubs)])...))
            n_vars += length(allowed)
        else
            push!(single_cost, 0.0)
            n_vars += length(allowed) * (1 + length(hubs))
        end
    end
    n_sites = length(site_programme)
    site_pairs = [Int[] for _ in 1:n_sites]
    for (k, (_, j)) in enumerate(ration_pairs)
        push!(site_pairs[j], k)
    end
    tonnes_per_gram(j) = AID_DAYS_PER_CYCLE * 1e-6 * beneficiaries[j]

    target_matrix = [Float64(_AID_PROGRAMMES[site_programme[j]].target[n]) for
                     n in eachindex(AID_NUTRIENTS), j in 1:n_sites]
    requirement = copy(target_matrix)
    energy_upper = [1.15 * target_matrix[1, j] for j in 1:n_sites]
    fat_share = [_AID_PROGRAMMES[site_programme[j]].fat_share for j in 1:n_sites]
    source_capacity = Dict{Tuple{Int, Int}, Float64}()
    hub_capacity = zeros(Float64, n_hubs)
    witness = nothing

    r = zeros(Float64, length(ration_pairs))
    for j in 1:n_sites
        prog = _AID_PROGRAMMES[site_programme[j]]
        for (fam, grams) in prog.basket
            ks = [k for k in site_pairs[j] if family[ration_pairs[k][1]] == fam]
            isempty(ks) && continue
            k = rand(rng, ks)
            r[k] = min(grams * rand(rng, Uniform(0.85, 1.15)), 0.95 * ration_upper[k])
        end
        # Occasional extras (canned fish, dates, milk, biscuits) where allowed.
        for k in site_pairs[j]
            r[k] == 0.0 && rand(rng) < 0.15 || continue
            r[k] = 0.3 * ration_upper[k] * rand(rng)
        end
        intake = zeros(Float64, length(AID_NUTRIENTS))
        for k in site_pairs[j]
            intake .+= r[k] .* view(content, :, ration_pairs[k][1])
        end
        if feasibility_status == unknown
            # NutVal targets, micronutrients planned at 60-95% of NutVal (the
            # usual compromise when fortified commodities are scarce); the
            # basket is only last cycle's plan, not a promise.
            for n in eachindex(AID_NUTRIENTS)
                requirement[n, j] *= n <= 3 ? rand(rng, Uniform(0.97, 1.03)) : rand(rng, Uniform(0.6, 0.95))
            end
        else
            for n in eachindex(AID_NUTRIENTS)
                requirement[n, j] = min(target_matrix[n, j], intake[n] * rand(rng, Uniform(0.92, 0.99)))
            end
            energy_upper[j] = max(energy_upper[j], 1.03 * intake[1])
            fat_share[j] = min(fat_share[j], 9.0 * intake[3] / intake[1] - 0.01)
        end
    end
    # Deliveries split over each site's hubs, procurement over sources.
    y = zeros(Float64, length(delivery_arcs))
    hub_need = zeros(Float64, n_comm, n_hubs)
    split = Dict{Tuple{Int, Int}, Vector{Float64}}()
    for (k, (c, h, j)) in enumerate(delivery_arcs)
        w = get!(split, (c, j)) do
            rand(rng, Dirichlet([2.0, 2.0]))
        end
        pos = findfirst(==(h), site_hubs[j])
        rk = site_pairs[j][findfirst(kk -> ration_pairs[kk][1] == c, site_pairs[j])]
        y[k] = tonnes_per_gram(j) * r[rk] * w[pos]
        hub_need[c, h] += y[k]
    end
    for (k, (c, j)) in enumerate(ration_pairs)
        length(site_hubs[j]) == 1 || continue
        hub_need[c, only(site_hubs[j])] += tonnes_per_gram(j) * r[k]
    end
    q = zeros(Float64, length(procurement_arcs))
    shares = Dict(c => rand(rng, Dirichlet(fill(1.5, length(offered[c])))) for c in 1:n_comm)
    hub_total = zeros(Float64, n_hubs)
    for (k, (c, s, h)) in enumerate(procurement_arcs)
        q[k] = hub_need[c, h] * shares[c][findfirst(==(s), offered[c])]
        source_capacity[(c, s)] = get(source_capacity, (c, s), 0.0) + q[k]
        hub_total[h] += q[k]
    end
    if feasibility_status == unknown
        # Capacities are this cycle's market and logistics draws around last
        # cycle's flows: one market and one logistics tightness per instance
        # (with per-source / per-hub noise) decide whether the pipeline can
        # carry the programme, so the outcome is genuinely two-sided.
        market = rand(rng, Uniform(0.9, 1.8))
        logistics = rand(rng, Uniform(0.85, 1.6))
        for key in sort!(collect(keys(source_capacity)))
            used = max(source_capacity[key], 1.0)
            source_capacity[key] = used * market * rand(rng, Uniform(0.8, 1.25))
        end
        for h in 1:n_hubs
            hub_capacity[h] = max(hub_total[h], 1.0) * logistics * rand(rng, LogNormal(0.0, 0.1))
        end
    else
        for key in sort!(collect(keys(source_capacity)))
            used = source_capacity[key]
            source_capacity[key] = used > 0.0 ? used * rand(rng, Uniform(1.02, 1.30)) :
                                   rand(rng, Uniform(1.0, 50.0))
        end
        for h in 1:n_hubs
            hub_capacity[h] = max(hub_total[h], 1.0) * rand(rng, Uniform(1.05, 1.30))
        end
        witness = AidPlan(r, y, q)
    end

    prob = FoodAidDietProblem(
        commodities, content, n_sources, n_hubs, n_sites, site_programme, beneficiaries,
        site_hubs, ration_pairs, ration_upper, delivery_arcs, delivery_cost, single_cost, procurement_arcs,
        procurement_cost, source_capacity, hub_capacity, requirement, energy_upper, fat_share,
        witness, nothing, feasibility_status,
    )

    if feasibility_status == infeasible
        single = [[j for j in 1:n_sites if site_hubs[j] == [h]] for h in 1:n_hubs]
        hubs_with_single = [h for h in 1:n_hubs if !isempty(single[h])]
        if rand(rng) < 0.65 && !isempty(hubs_with_single)
            h = rand(rng, hubs_with_single)
            sites = single[h]
            required = sum(_aid_hub_min_tonnes(prob, j, site_pairs) for j in sites)
            hub_capacity[h] = required * rand(rng, Uniform(0.78, 0.93))
            cert = AidInfeasibilityCertificate(aid_hub_throughput, h, sites, 0, hub_capacity[h], required)
        else
            # A micronutrient every site needs (fortified commodities run short).
            candidates = [n for n in 4:length(AID_NUTRIENTS) if all(>(0.0), view(requirement, n, :))]
            isempty(candidates) && (candidates = [2])
            n = rand(rng, candidates)
            required = _aid_pipeline_requirement(prob, n)
            available = _aid_pipeline_capacity(prob, n)
            theta = required / rand(rng, Uniform(1.08, 1.25)) / available
            for key in collect(keys(source_capacity))
                content[n, key[1]] > 0.0 && (source_capacity[key] *= theta)
            end
            cert = AidInfeasibilityCertificate(
                aid_nutrient_shortage, 0, Int[], n, _aid_pipeline_capacity(prob, n), required
            )
        end
        prob = FoodAidDietProblem(
            commodities, content, n_sources, n_hubs, n_sites, site_programme, beneficiaries,
            site_hubs, ration_pairs, ration_upper, delivery_arcs, delivery_cost, single_cost,
            procurement_arcs, procurement_cost, source_capacity, hub_capacity, requirement,
            energy_upper, fat_share, nothing, cert, feasibility_status,
        )
        @assert aid_certificate_holds(prob)
    elseif feasibility_status == feasible
        @assert aid_plan_satisfies(prob)
    end
    return prob
end

"""
    build_model(prob::FoodAidDietProblem)

Build the food-aid ration and sourcing LP (deterministic; see `FoodAidDietProblem`).
"""
function build_model(prob::FoodAidDietProblem)
    model = Model()
    R, Y, Q = length(prob.ration_pairs), length(prob.delivery_arcs), length(prob.procurement_arcs)
    @variable(model, 0 <= r[k=1:R] <= prob.ration_upper[k])
    @variable(model, y[1:Y] >= 0)
    @variable(model, q[1:Q] >= 0)
    @objective(
        model,
        Min,
        sum(prob.procurement_cost[k] * q[k] for k in 1:Q) + sum(prob.delivery_cost[k] * y[k] for k in 1:Y; init=0.0) +
        sum(
            prob.site_delivery_cost[j] * AID_DAYS_PER_CYCLE * 1e-6 * prob.beneficiaries[j] * r[k] for
            (k, (_, j)) in enumerate(prob.ration_pairs) if length(prob.site_hubs[j]) == 1;
            init=0.0,
        )
    )
    C = prob.content
    site_pairs = _aid_site_pairs(prob)
    for j in 1:prob.n_sites
        ks = site_pairs[j]
        @constraint(
            model,
            prob.requirement[1, j] <= sum(C[1, prob.ration_pairs[k][1]] * r[k] for k in ks) <= prob.energy_upper[j]
        )
        for n in 2:length(AID_NUTRIENTS)
            prob.requirement[n, j] > 0.0 || continue
            carriers = [k for k in ks if C[n, prob.ration_pairs[k][1]] > 0.0]
            @constraint(model, sum(C[n, prob.ration_pairs[k][1]] * r[k] for k in carriers) >= prob.requirement[n, j])
        end
        @constraint(
            model,
            sum((9.0 * C[3, prob.ration_pairs[k][1]] - prob.fat_share[j] * C[1, prob.ration_pairs[k][1]]) * r[k] for k in ks) >= 0
        )
    end
    arcs_of_pair = Dict{Tuple{Int, Int}, Vector{Int}}()
    out_of_hub = Dict{Tuple{Int, Int}, Vector{Int}}()
    for (k, (c, h, j)) in enumerate(prob.delivery_arcs)
        push!(get!(arcs_of_pair, (c, j), Int[]), k)
        push!(get!(out_of_hub, (c, h), Int[]), k)
    end
    direct = Dict{Tuple{Int, Int}, Vector{Int}}()
    for (k, (c, j)) in enumerate(prob.ration_pairs)
        if length(prob.site_hubs[j]) == 1
            push!(get!(direct, (c, only(prob.site_hubs[j])), Int[]), k)
            continue
        end
        @constraint(
            model,
            sum(y[a] for a in arcs_of_pair[(c, j)]) - AID_DAYS_PER_CYCLE * 1e-6 * prob.beneficiaries[j] * r[k] == 0
        )
    end
    into_hub = Dict{Tuple{Int, Int}, Vector{Int}}()
    from_source = Dict{Tuple{Int, Int}, Vector{Int}}()
    hub_arcs = [Int[] for _ in 1:prob.n_hubs]
    for (k, (c, s, h)) in enumerate(prob.procurement_arcs)
        push!(get!(into_hub, (c, h), Int[]), k)
        push!(get!(from_source, (c, s), Int[]), k)
        push!(hub_arcs[h], k)
    end
    for key in sort!(collect(union(keys(out_of_hub), keys(direct))))
        @constraint(
            model,
            sum(q[a] for a in into_hub[key]) - sum(y[a] for a in get(out_of_hub, key, Int[]); init=0.0) -
            sum(AID_DAYS_PER_CYCLE * 1e-6 * prob.beneficiaries[prob.ration_pairs[k][2]] * r[k] for k in get(direct, key, Int[]); init=0.0) >= 0
        )
    end
    for key in sort!(collect(keys(from_source)))
        @constraint(model, sum(q[a] for a in from_source[key]) <= prob.source_capacity[key])
    end
    for h in 1:prob.n_hubs
        @constraint(model, sum(q[a] for a in hub_arcs[h]) <= prob.hub_capacity[h])
    end
    return model
end

register_variant(
    :diet_problem,
    :food_aid,
    FoodAidDietProblem,
    "Humanitarian food-basket design with sourcing (WFP Optimus-style): NutVal ration rows " *
    "per distribution site linked to a source-hub-site procurement and delivery network";
    tags=[:agriculture, :network, :multicommodity, :covering, :blending],
)
