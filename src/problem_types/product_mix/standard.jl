using JuMP
using Random
using Distributions
using StatsBase

"""
Planted operating plan: one production quantity per routing (model column),
the machine hours, labor hours, and material units it consumes. For the
`feasible` profile it satisfies every row of the built model, with strict slack
on every machine, labor, and material row.
"""
struct ProductMixPlanWitness
    production::Vector{Float64}
    machine_hours::Vector{Float64}
    labor_hours::Vector{Float64}
    material_use::Vector{Float64}
end

"""
Material-shortage certificate at plant-area (or plant) level. Every product in
`products` has a committed sales floor `floor[p]` in its market row
(`Σ_r x[r] >= floor[p]`), and each unit it sells draws at least `min_use[k]`
of material `material` whatever routing makes it (`qty * min routing yield`).
Multiplying each market row by that minimum and adding the material's
availability row shows the commitments need `required = Σ floor[p] *
min_use[k]` units while only `available < required` are allocated. `scope` is
`:area` or `:plant` for a shared material (`:department` only in shops without
areas). The material row is protected so that no single row or product is
refuted on its own, and the floors of multi-routing products are row
constraints rather than column bounds, so HiGHS presolve's bound propagation
cannot assemble the contradiction.
"""
struct ProductMixMaterialCertificate
    material::Int
    scope::Symbol
    products::Vector{Int}
    min_use::Vector{Float64}
    required::Float64
    available::Float64
end

"""
    ProductMixProblem <: ProblemGenerator

Single-period product mix with alternative routings over a shop of many
machines, labor pools, and materials.

# Overview

A plant sells `n_products` products. Each product can be made by one to three
alternative *routings* (process plans), each visiting a sparse sequence of
2–5 machines with its own processing times and conversion cost; one column
`x[r] >= 0` per routing. Machines are grouped into departments, every product
family has a primary department all of its routings pass through, and each
department has a labor pool (operator hours per step depend on crew size and
how manual the operation is, so labor is not proportional to machine time).
Products consume
purchased materials with limited availability.

Rows:

  - machine capacity: `Σ_r time[r,m] x[r] <= machine_capacity[m]`;
  - department labor: `Σ_r Σ_{steps of r in d} labor[r,step] x[r] <= labor_capacity[d]`;
  - material availability: `Σ_r qty[p(r),k] x[r] <= material_capacity[k]`;
  - market, per product: `floor[p] <= Σ_{r of p} x[r] <= ceiling[p]` — a ranged
    row for multi-routing products (a pair of variable bounds when there is a
    single routing).

Objective: maximize contribution margin (`price − materials − conversion
cost`, positive for every routing).

The row count grows with the column count (machines ≈ 7% of routings,
materials ≈ 10% of products, plus one market row per multi-routing product),
so unlike the previous ≤30-row formulation — which HiGHS presolve dissolved
completely — the LP keeps a full-rank coupled core. Compared with
`production_planning` this is a single-period model: its structure comes from
routing choice and many shared resources, not from inventory staircases.

# Feasibility control

A nominal plan is planted first (a fraction of each product's market ceiling,
split across its routings); capacities are its consumption plus heterogeneous
headroom (a few near-saturated bottlenecks), and floors are fractions of the
plan.

  - `feasible`: the plan is stored as a [`ProductMixPlanWitness`](@ref).
  - `infeasible`: a supply shortage on a shared area- or plant-level material:
    its multi-routing users are put under contract (floors 60–95% of plan) and
    its allocation is cut until the floors need 10–35% more of it than is
    available (at the most material-efficient routing mix) — a [`ProductMixMaterialCertificate`](@ref). The allocation never
    drops below 1.3× what single-routing floors (column bounds) force through it
    plus 1.3× the largest single commitment, so presolve's bound propagation
    cannot refute it; only the material row plus all committed market rows do.
  - `unknown`: the same mechanism with ratio `1 ± U(0.03, 0.30)`, measured at
    the plan's own routing mix.

# Fields

  - `n_products`, `n_routings`, `n_machines`, `n_departments`, `n_materials::Int`
  - `routing_product::Vector{Int}`, `routing_machines::Vector{Vector{Int}}`,
    `routing_times::Vector{Vector{Float64}}`, `routing_cost::Vector{Float64}`
  - `routing_labor::Vector{Vector{Float64}}`: operator hours per unit, aligned with `routing_machines`
  - `routing_yield::Vector{Float64}`: material multiplier (scrap) of each routing
  - `machine_department::Vector{Int}`, `machine_capacity::Vector{Float64}`
  - `labor_capacity::Vector{Float64}`: per department
  - `product_materials::Vector{Vector{Int}}`, `material_qty::Vector{Vector{Float64}}`
  - `material_capacity::Vector{Float64}`, `material_cost::Vector{Float64}`
  - `material_owner::Vector{Int}`: owning department (> 0), area (negated, < 0), or plant (0)
  - `price`, `floor`, `ceiling::Vector{Float64}`: per product
  - `primary_department::Vector{Int}`: per product
  - `industry::Symbol`
  - `feasible_witness`, `infeasibility_certificate`, `feasibility_status`
"""
struct ProductMixProblem <: ProblemGenerator
    n_products::Int
    n_routings::Int
    n_machines::Int
    n_departments::Int
    n_materials::Int
    routing_product::Vector{Int}
    routing_machines::Vector{Vector{Int}}
    routing_times::Vector{Vector{Float64}}
    routing_cost::Vector{Float64}
    routing_labor::Vector{Vector{Float64}}
    routing_yield::Vector{Float64}
    machine_department::Vector{Int}
    machine_capacity::Vector{Float64}
    labor_capacity::Vector{Float64}
    product_materials::Vector{Vector{Int}}
    material_qty::Vector{Vector{Float64}}
    material_capacity::Vector{Float64}
    material_cost::Vector{Float64}
    material_owner::Vector{Int}
    price::Vector{Float64}
    floor::Vector{Float64}
    ceiling::Vector{Float64}
    primary_department::Vector{Int}
    industry::Symbol
    feasible_witness::Union{Nothing, ProductMixPlanWitness}
    infeasibility_certificate::Union{Nothing, ProductMixMaterialCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    _product_mix_usage(prob_fields..., production) -> (machine, labor, material)

Machine hours, department labor hours, and material units consumed by a
per-routing production vector.
"""
function _product_mix_usage(
    n_machines::Int,
    n_departments::Int,
    n_materials::Int,
    routing_product::Vector{Int},
    routing_machines::Vector{Vector{Int}},
    routing_times::Vector{Vector{Float64}},
    routing_labor::Vector{Vector{Float64}},
    routing_yield::Vector{Float64},
    machine_department::Vector{Int},
    product_materials::Vector{Vector{Int}},
    material_qty::Vector{Vector{Float64}},
    production::Vector{Float64},
)
    machine = zeros(n_machines)
    labor = zeros(n_departments)
    material = zeros(n_materials)
    for r in eachindex(production)
        q = production[r]
        q == 0.0 && continue
        for (m, t, h) in zip(routing_machines[r], routing_times[r], routing_labor[r])
            machine[m] += t * q
            labor[machine_department[m]] += h * q
        end
        p = routing_product[r]
        for (k, a) in zip(product_materials[p], material_qty[p])
            material[k] += a * routing_yield[r] * q
        end
    end
    return machine, labor, material
end

const _PRODUCT_MIX_INDUSTRIES = (
    :metalworking, :electronics, :furniture, :food_processing, :chemical, :automotive
)

"""
    ProductMixProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a product mix instance with exactly `max(target_variables, 2)`
routing columns.
"""
function ProductMixProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be >= 1 (got $target_variables)."))
    rng = MersenneTwister(seed)
    target = max(target_variables, 2)

    industry = sample(rng, collect(_PRODUCT_MIX_INDUSTRIES), Weights([0.25, 0.2, 0.12, 0.15, 0.13, 0.15]))
    # Industry regime: routing flexibility, routing length, processing-time
    # scale, material intensity, margin level.
    alt_routing_prob, route_len, time_mu, margin_mu = if industry == :electronics
        (0.6, (3, 5), log(0.08), log(0.45))
    elseif industry == :furniture
        (0.4, (2, 4), log(0.6), log(0.35))
    elseif industry == :food_processing
        (0.3, (2, 3), log(0.02), log(0.2))
    elseif industry == :chemical
        (0.5, (2, 4), log(0.05), log(0.3))
    elseif industry == :automotive
        (0.45, (3, 5), log(0.25), log(0.25))
    else  # :metalworking
        (0.55, (2, 5), log(0.3), log(0.3))
    end

    # --- Shop: machines in departments, departments in areas -------------------
    # Production departments (fabrication cells, lines) own their product
    # families. Departments are grouped into plant areas; each area has a
    # shared finishing/packaging department and an area material store, and a
    # few materials are bought plant-wide. This block-angular locality is how
    # real plants look, and it keeps the LP sparse to factor — plant-wide
    # random routing would turn every basis into a dense random matrix.
    n_machines = max(3, round(Int, target * rand(rng, Uniform(0.05, 0.09))))
    n_departments = max(1, round(Int, n_machines / rand(rng, 4:9)))
    n_areas = n_departments >= 4 ? max(1, round(Int, n_departments / rand(rng, 4:8))) : 0
    n_prod_depts = n_departments - n_areas
    machine_department = shuffle(rng, [mod1(m, n_departments) for m in 1:n_machines])
    dept_machines = [findall(==(d), machine_department) for d in 1:n_departments]
    dept_area = [d <= n_prod_depts ? mod1(d, max(n_areas, 1)) : d - n_prod_depts for d in 1:n_departments]
    shared_dept_of_area = [n_prod_depts + a for a in 1:n_areas]
    production_depts = collect(1:n_prod_depts)
    crew = rand(rng, LogNormal(log(1.2), 0.4), n_machines)
    machine_speed = rand(rng, LogNormal(0.0, 0.3), n_machines)   # slow / fast machines

    # Materials: department stores, area stores (owner -area), plant-wide (0).
    n_materials = max(2, round(Int, target / rand(rng, 12:25)))
    material_owner = map(1:n_materials) do _
        u = rand(rng)
        u < 0.06 ? 0 : (u < 0.25 && n_areas > 0 ? -rand(rng, 1:n_areas) : rand(rng, production_depts))
    end
    # Every production department stocks at least one material of its own.
    for (k, d) in enumerate(production_depts)
        k <= n_materials && (material_owner[k] = d)
    end
    dept_materials = [findall(==(d), material_owner) for d in 1:n_departments]
    area_materials = [findall(==(-a), material_owner) for a in 1:n_areas]
    plant_materials = findall(==(0), material_owner)

    # --- Products and routings, until the column budget is used exactly -------
    routing_product = Int[]
    routing_machines = Vector{Int}[]
    routing_times = Vector{Float64}[]
    routing_cost = Float64[]
    routing_labor = Vector{Float64}[]
    routing_yield = Float64[]
    primary_department = Int[]
    product_materials = Vector{Int}[]
    material_qty = Vector{Float64}[]
    n_products = 0
    while length(routing_product) < target
        n_products += 1
        p = n_products
        d0 = rand(rng, production_depts)
        push!(primary_department, d0)
        n_r = rand(rng) < alt_routing_prob ? rand(rng, 2:3) : 1
        n_r = min(n_r, target - length(routing_product))
        bt = rand(rng, LogNormal(time_mu, 0.5))
        sibling = [d for d in production_depts if d != d0 && dept_area[d] == dept_area[d0]]
        for k_r in 1:n_r
            len = rand(rng, route_len[1]:route_len[2])
            # Every routing starts in the family's department; later steps stay
            # there or visit a shared department.
            machines = [rand(rng, dept_machines[d0])]
            tries = 0
            while length(machines) < len && tries < 50
                tries += 1
                u = rand(rng)
                d = if n_areas > 0 && u < 0.15
                    shared_dept_of_area[dept_area[d0]]
                elseif k_r > 1 && u < 0.4 && !isempty(sibling)
                    rand(rng, sibling)          # alternative line in a sibling department
                else
                    d0
                end
                m = rand(rng, dept_machines[d])
                m in machines || push!(machines, m)
            end
            times = [bt * rand(rng, LogNormal(0.0, 0.35)) / machine_speed[m] for m in machines]
            # Operator hours: crew size times a manual-content factor drawn per
            # step (loading, inspection, rework) — independent of machine time.
            labor = [crew[m] * bt * rand(rng, LogNormal(0.0, 0.6)) for m in machines]
            push!(routing_product, p)
            push!(routing_machines, machines)
            push!(routing_times, times)
            push!(routing_labor, labor)
            # Scrap: the preferred routing is lean, alternatives waste more.
            push!(routing_yield, 1.0 + (k_r == 1 ? 0.04 : 0.18) * rand(rng))
            push!(routing_cost, 0.0)   # filled once machine rates are known
        end
        mats = Int[]
        for _ in 1:rand(rng, 1:4)
            u = rand(rng)
            pool = if u < 0.02 && !isempty(plant_materials)
                plant_materials
            elseif u < 0.25 && n_areas > 0 && !isempty(area_materials[dept_area[d0]])
                area_materials[dept_area[d0]]
            else
                dept_materials[d0]
            end
            isempty(pool) && (pool = collect(1:n_materials))
            k = rand(rng, pool)
            k in mats || push!(mats, k)
        end
        sort!(mats)
        push!(product_materials, mats)
        push!(material_qty, [rand(rng, LogNormal(0.0, 0.6)) for _ in mats])
    end
    n_routings = length(routing_product)
    material_cost = rand(rng, LogNormal(log(3.0), 0.7), n_materials)

    # Prices: materials plus conversion plus a positive margin; conversion cost
    # per routing is machine-hour cost, so the slower routing costs more.
    machine_rate = rand(rng, LogNormal(log(60.0), 0.3), n_machines)
    price = zeros(n_products)
    for r in 1:n_routings
        routing_cost[r] = sum(machine_rate[m] * t for (m, t) in zip(routing_machines[r], routing_times[r]))
    end
    prod_routings = [Int[] for _ in 1:n_products]
    for r in 1:n_routings
        push!(prod_routings[routing_product[r]], r)
    end
    for p in 1:n_products
        mat = sum(a * material_cost[k] for (k, a) in zip(product_materials[p], material_qty[p]); init=0.0)
        conv = maximum(routing_cost[r] + (routing_yield[r] - 1) * mat for r in prod_routings[p])
        price[p] = (mat + conv) * (1 + rand(rng, LogNormal(margin_mu, 0.4)))
    end

    # --- Planted plan -------------------------------------------------------------
    ceiling = [rand(rng, LogNormal(log(200.0), 0.9)) for _ in 1:n_products]
    production = zeros(n_routings)
    for p in 1:n_products
        q = ceiling[p] * rand(rng, Uniform(0.35, 0.85))
        shares = rand(rng, Dirichlet(length(prod_routings[p]), 1.5))
        for (s, r) in zip(shares, prod_routings[p])
            production[r] = q * s
        end
    end
    machine_load, labor_load, material_load = _product_mix_usage(
        n_machines,
        n_departments,
        n_materials,
        routing_product,
        routing_machines,
        routing_times,
        routing_labor,
        routing_yield,
        machine_department,
        product_materials,
        material_qty,
        production,
    )
    # Heterogeneous headroom: a few nearly saturated bottlenecks, most with slack.
    headroom(n) = clamp.(rand(rng, LogNormal(log(0.15), 0.8), n), 0.02, 1.5)
    machine_capacity = max.(machine_load .* (1 .+ headroom(n_machines)), 1.0)
    labor_capacity = max.(labor_load .* (1 .+ headroom(n_departments)), 1.0)
    material_capacity = max.(material_load .* (1 .+ headroom(n_materials)), 1.0)
    planned = [sum(production[r] for r in prod_routings[p]) for p in 1:n_products]
    floor = [rand(rng) < 0.45 ? planned[p] * rand(rng, Uniform(0.3, 0.9)) : 0.0 for p in 1:n_products]

    # --- Feasibility profile ----------------------------------------------------------
    witness = nothing
    certificate = nothing
    if feasibility_status == feasible
        witness = ProductMixPlanWitness(production, machine_load, labor_load, material_load)
    else
        # Material shortage at plant-area (or plant) level. Per unit of
        # product p, every routing uses at least `qty * min yield` of each of
        # its materials, so committed floors force a minimum draw on a shared
        # material. Only one row is involved per material, but the floors of
        # multi-routing products live in ranged market rows, not in column
        # bounds, so presolve's bound propagation cannot add them up — the
        # contradiction needs the material row plus every committed market row.
        min_use(p, k) = material_qty[p][findfirst(==(k), product_materials[p])] *
            minimum(routing_yield[r] for r in prod_routings[p])
        max_use(p, k) = material_qty[p][findfirst(==(k), product_materials[p])] *
            maximum(routing_yield[r] for r in prod_routings[p])
        users = [Int[] for _ in 1:n_materials]
        for p in 1:n_products, k in product_materials[p]
            push!(users[k], p)
        end
        multi(p) = length(prod_routings[p]) > 1
        # Protection a material row must keep so no single row or single
        # product is refuted on its own: 1.3x what single-routing floors
        # (variable bounds) force through it plus 1.3x the largest single
        # multi-routing commitment.
        function protection(k)
            forced = sum((min_use(p, k) * floor[p] for p in users[k] if !multi(p) && floor[p] > 0); init=0.0)
            big = maximum((max_use(p, k) * floor[p] for p in users[k] if multi(p)); init=0.0)
            return 1.3 * forced + 1.3 * big
        end
        # Candidate: the shared (area or plant) material whose multi-routing
        # users could carry the largest commitment relative to its protection;
        # department materials only when the shop has no areas.
        shared = [k for k in 1:n_materials if material_owner[k] <= 0 && any(multi, users[k])]
        candidates = isempty(shared) ? [k for k in 1:n_materials if any(multi, users[k])] : shared
        isempty(candidates) && (candidates = [k for k in 1:n_materials if !isempty(users[k])])
        function potential(k)
            pot = sum((0.95 * planned[p] * min_use(p, k) for p in users[k] if multi(p)); init=0.0)
            pot += sum((min_use(p, k) * floor[p] for p in users[k] if !multi(p) && floor[p] > 0); init=0.0)
            prot = 1.3 * sum((min_use(p, k) * floor[p] for p in users[k] if !multi(p) && floor[p] > 0); init=0.0) +
                1.3 * maximum((max_use(p, k) * 0.95 * planned[p] for p in users[k] if multi(p)); init=0.0)
            return pot / max(prot, 1e-9)
        end
        kstar = argmax(potential, candidates)
        # Its multi-routing users go under contract (60-95% of plan).
        for p in users[kstar]
            multi(p) && (floor[p] = max(floor[p], planned[p] * rand(rng, Uniform(0.6, 0.95))))
        end
        if !any(floor[p] > 0 for p in users[kstar])
            for p in users[kstar]   # tiny shops: commit whatever uses it
                floor[p] = planned[p] * rand(rng, Uniform(0.6, 0.95))
            end
        end
        prods = [p for p in users[kstar] if floor[p] > 0]
        uses = [min_use(p, kstar) for p in prods]
        required = sum(floor[p] * u for (p, u) in zip(prods, uses))
        ratio = if feasibility_status == infeasible
            1.1 + 0.25 * rand(rng)
        else
            m = 0.03 + 0.27 * rand(rng)
            rand(rng) < 0.5 ? 1.0 - m : 1.0 + m
        end
        # Supply allocation cut to `required / ratio`, never below the
        # protection; an infeasible request that the protection would push
        # under a 1.1 ratio falls back to the plain cut (still certified,
        # only less hidden from presolve).
        # `unknown` measures the ratio against the plan's own routing mix
        # (the minimum-yield mix is usually blocked by machine capacity, which
        # would make nearly every draw infeasible).
        basis = if feasibility_status == infeasible
            required
        else
            sum(
                floor[p] * material_qty[p][findfirst(==(kstar), product_materials[p])] *
                sum(routing_yield[r] * production[r] for r in prod_routings[p]) / planned[p] for p in prods
            )
        end
        cap = max(basis / ratio, protection(kstar))
        if feasibility_status == infeasible && required / cap < 1.1
            cap = required / ratio
        end
        material_capacity[kstar] = cap
        if feasibility_status == infeasible
            scope = material_owner[kstar] == 0 ? :plant : (material_owner[kstar] < 0 ? :area : :department)
            certificate = ProductMixMaterialCertificate(kstar, scope, prods, uses, required, cap)
        end
    end

    return ProductMixProblem(
        n_products,
        n_routings,
        n_machines,
        n_departments,
        n_materials,
        routing_product,
        routing_machines,
        routing_times,
        routing_cost,
        routing_labor,
        routing_yield,
        machine_department,
        machine_capacity,
        labor_capacity,
        product_materials,
        material_qty,
        material_capacity,
        material_cost,
        material_owner,
        price,
        floor,
        ceiling,
        primary_department,
        industry,
        witness,
        certificate,
        feasibility_status,
    )
end

"""
    build_model(prob::ProductMixProblem)

Build the product mix LP (one column per routing). Deterministic.
"""
function build_model(prob::ProductMixProblem)
    model = Model()
    R = prob.n_routings
    @variable(model, x[1:R] >= 0)

    prod_routings = [Int[] for _ in 1:prob.n_products]
    for r in 1:R
        push!(prod_routings[prob.routing_product[r]], r)
    end

    # Machine, labor, and material rows (sparse accumulation).
    machine_terms = [Tuple{Int, Float64}[] for _ in 1:prob.n_machines]
    labor_terms = [Dict{Int, Float64}() for _ in 1:prob.n_departments]
    material_terms = [Tuple{Int, Float64}[] for _ in 1:prob.n_materials]
    for r in 1:R
        for (m, t, h) in zip(prob.routing_machines[r], prob.routing_times[r], prob.routing_labor[r])
            push!(machine_terms[m], (r, t))
            d = prob.machine_department[m]
            labor_terms[d][r] = get(labor_terms[d], r, 0.0) + h
        end
        p = prob.routing_product[r]
        for (k, a) in zip(prob.product_materials[p], prob.material_qty[p])
            push!(material_terms[k], (r, a * prob.routing_yield[r]))
        end
    end
    for m in 1:prob.n_machines
        isempty(machine_terms[m]) && continue
        @constraint(model, sum(t * x[r] for (r, t) in machine_terms[m]) <= prob.machine_capacity[m])
    end
    for d in 1:prob.n_departments
        isempty(labor_terms[d]) && continue
        @constraint(
            model, sum(h * x[r] for (r, h) in sort!(collect(labor_terms[d]))) <= prob.labor_capacity[d]
        )
    end
    for k in 1:prob.n_materials
        isempty(material_terms[k]) && continue
        @constraint(model, sum(a * x[r] for (r, a) in material_terms[k]) <= prob.material_capacity[k])
    end

    # Market rows: ranged for multi-routing products, bounds otherwise.
    for p in 1:prob.n_products
        rs = prod_routings[p]
        if length(rs) == 1
            set_upper_bound(x[rs[1]], prob.ceiling[p])
            prob.floor[p] > 0 && set_lower_bound(x[rs[1]], prob.floor[p])
        elseif prob.floor[p] > 0
            @constraint(model, prob.floor[p] <= sum(x[r] for r in rs) <= prob.ceiling[p])
        else
            @constraint(model, sum(x[r] for r in rs) <= prob.ceiling[p])
        end
    end

    margin = zeros(R)
    for r in 1:R
        p = prob.routing_product[r]
        mat = sum(a * prob.material_cost[k] for (k, a) in zip(prob.product_materials[p], prob.material_qty[p]); init=0.0)
        margin[r] = prob.price[p] - mat * prob.routing_yield[r] - prob.routing_cost[r]
    end
    @objective(model, Max, sum(margin[r] * x[r] for r in 1:R))
    return model
end

register_variant(
    :product_mix,
    :standard,
    ProductMixProblem,
    "Single-period product mix with alternative routings over many machines, department labor pools, and materials: ranged market rows, planted operating plan, and an area/plant material-shortage infeasibility certificate",
)
