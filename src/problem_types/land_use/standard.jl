using JuMP
using Random
using Distributions
using StatsBase

# The large size regime can request as many as twelve zoning types. Keeping all
# zoning metadata in one catalog prevents names, economic parameters, and
# resource profiles from drifting to different lengths.
const _LAND_USE_ZONING_CATALOG = (
    (
        name="Residential",
        cost=1.00,
        revenue=1.50,
        resources=(2.00, 1.50, 1.00, 1.50, 1.20, 0.80, 0.40, 1.00),
    ),
    (
        name="Commercial",
        cost=2.50,
        revenue=4.00,
        resources=(1.00, 0.80, 3.00, 2.00, 1.80, 1.20, 0.50, 1.50),
    ),
    (
        name="Industrial",
        cost=3.00,
        revenue=2.00,
        resources=(1.50, 2.50, 2.50, 4.00, 1.10, 3.00, 2.50, 1.80),
    ),
    (
        name="Agricultural",
        cost=0.50,
        revenue=0.80,
        resources=(3.00, 0.50, 0.60, 0.60, 0.30, 0.40, 1.40, 0.50),
    ),
    (
        name="Conservation",
        cost=0.10,
        revenue=0.20,
        resources=(0.08, 0.05, 0.08, 0.05, 0.02, 0.02, 0.05, 0.10),
    ),
    (
        name="Mixed Use",
        cost=2.00,
        revenue=3.00,
        resources=(1.50, 1.20, 2.00, 1.80, 1.70, 1.00, 0.45, 1.30),
    ),
    (
        name="Recreational",
        cost=1.50,
        revenue=1.00,
        resources=(1.20, 0.70, 1.80, 0.80, 0.70, 0.30, 0.35, 1.20),
    ),
    (
        name="Institutional",
        cost=1.80,
        revenue=0.50,
        resources=(1.50, 1.30, 2.20, 1.80, 1.40, 0.80, 0.50, 2.00),
    ),
    (
        name="Transportation",
        cost=4.00,
        revenue=0.10,
        resources=(0.30, 0.20, 4.00, 2.00, 0.80, 1.50, 1.00, 2.20),
    ),
    (
        name="Special",
        cost=3.50,
        revenue=2.50,
        resources=(1.40, 1.10, 1.80, 2.20, 1.30, 1.20, 0.80, 1.80),
    ),
    (
        name="Utilities",
        cost=2.80,
        revenue=1.20,
        resources=(0.60, 0.80, 1.50, 4.00, 1.80, 3.00, 1.50, 2.00),
    ),
    (
        name="Open Space",
        cost=0.15,
        revenue=0.15,
        resources=(0.15, 0.08, 0.20, 0.08, 0.05, 0.03, 0.10, 0.20),
    ),
)

const _LAND_USE_RESOURCE_NAMES = (
    "Water", "Sewage", "Transportation", "Power", "Internet", "Gas", "Environmental", "Emergency"
)

# Zone roles (catalog indices): 1 Residential, 2 Commercial, 3 Industrial,
# 5 Conservation, 6 Mixed Use, 7 Recreational, 8 Institutional, 12 Open Space.
const _LAND_USE_HOUSING_DENSITY = Dict(1 => 20.0, 6 => 15.0)              # dwellings / ha
const _LAND_USE_JOB_DENSITY = Dict(2 => 40.0, 3 => 25.0, 6 => 20.0, 8 => 15.0)  # jobs / ha
const _LAND_USE_GREEN_ZONES = (5, 7, 12)
const _LAND_USE_RESIDENTIAL = 1
const _LAND_USE_INDUSTRIAL = 3

"""
    LandUseInfeasibilityCertificate

Solver-independent proof that a land-use instance is infeasible. Every parcel
of service district `district` must take one allowed zone, and each allowed zone
consumes at least `per_parcel_minimum[i]` of resource `resource_index` on parcel
`i` (`parcels[i]`), so the district consumes at least `lower_bound`, while its
capacity row only permits `capacity < lower_bound`. The bound uses the
assignment equalities and the capacity row only, so it holds for the LP
relaxation too.
"""
struct LandUseInfeasibilityCertificate
    district::Int
    resource_index::Int
    parcels::Vector{Int}
    per_parcel_minimum::Vector{Float64}
    lower_bound::Float64
    capacity::Float64
end

"""
    LandUseProblem <: ProblemGenerator

Spatial zoning plan for a growing city: every parcel on a planar parcel graph
takes exactly one land-use zone, at maximum net development value, subject to
district infrastructure capacities, district housing and employment targets,
green-space accessibility around homes, and residential–industrial buffer rules.

# Formulation

Variables `x[k] ∈ {0, 1}` for every allowed parcel–zone pair `k = (i, z)`
(`pairs`); environmentally excluded pairs simply do not exist. Maximize
`Σ_k size_i (revenue[i,z] − cost[i,z]) x[k]`. Rows:

  - assignment `Σ_z x[i,z] = 1` per parcel;
  - infrastructure, per service district `d` and resource `r`:
    `Σ_{i∈d} size_i consumption[z,r] x[i,z] ≤ capacity[d,r]`;
  - housing `Σ_{i∈d} size_i density_z x[i,z] ≥ housing_target[d]` (residential
    and mixed use) and jobs `≥ jobs_target[d]` (commercial, industrial, mixed
    use, institutional) per district;
  - green-space accessibility, for parcels that may become residential:
    `Σ_{j∈N[i]} size_j x[j,green] ≥ green_ratio[i] · size_i · x[i,residential]`
    (`N[i]` is the parcel and its neighbours; green = conservation,
    recreational, open space);
  - buffer rules `x[i,residential] + x[j,industrial] ≤ 1` for each neighbouring
    pair (both orientations) where both variables exist.

Under the default `relax_integer=true` this is a fractional land-allocation LP
(each parcel split across uses), whose district, accessibility and buffer rows
all stay meaningful.

# Sizing

Parcels `≈ target / E[allowed zones per parcel]`, so the number of pairs is
within a few percent of the target; districts hold `≈ 80` parcels.

# Feasibility

  - `feasible`: a greedy reference plan (best net value, never residential next
    to industrial, homes given a green neighbour where possible) is drawn before
    the environmental exclusions, which never remove its zones; capacities are
    1.03–1.20× its use, targets 85–97% of what it provides, and accessibility
    ratios at most what it achieves. Stored as `feasible_witness` (zone per
    parcel).
  - `infeasible`: one district's capacity for one resource is cut below the
    least that any allowed zoning of its parcels consumes
    (`LandUseInfeasibilityCertificate`).
  - `unknown`: nominal capacities and targets with no planted plan.
"""
struct LandUseProblem <: ProblemGenerator
    n_parcels::Int
    n_zoning_types::Int
    n_resources::Int
    n_districts::Int
    parcel_sizes::Vector{Float64}
    parcel_district::Vector{Int}
    parcel_coordinates::Matrix{Float64}
    adjacency_edges::Vector{Tuple{Int, Int}}
    pairs::Vector{Tuple{Int, Int}}
    development_costs::Matrix{Float64}
    revenues::Matrix{Float64}
    resource_consumption::Matrix{Float64}
    resource_capacities::Matrix{Float64}
    housing_target::Vector{Float64}
    jobs_target::Vector{Float64}
    green_ratio::Vector{Float64}
    zoning_names::Vector{String}
    resource_names::Vector{String}
    feasible_witness::Union{Nothing, Vector{Int}}
    infeasibility_certificate::Union{Nothing, LandUseInfeasibilityCertificate}
    feasibility_status::FeasibilityStatus
end

# A connected planar-like four-neighbour graph on a jittered grid, parcel ids
# shuffled over the cells; also returns each parcel's grid cell.
function _land_use_spatial_graph(rng::AbstractRNG, n_parcels::Int)
    n_columns = ceil(Int, sqrt(n_parcels))
    n_rows = ceil(Int, n_parcels / n_columns)
    cells = Tuple{Int, Int}[]
    for row in 1:n_rows, column in 1:n_columns
        length(cells) == n_parcels && break
        push!(cells, (row, column))
    end
    shuffle!(rng, cells)
    cell_to_parcel = zeros(Int, n_rows, n_columns)
    coordinates = zeros(Float64, n_parcels, 2)
    for parcel in 1:n_parcels
        row, column = cells[parcel]
        cell_to_parcel[row, column] = parcel
        coordinates[parcel, 1] = (column - 0.5 + 0.18 * (2.0 * rand(rng) - 1.0)) / n_columns
        coordinates[parcel, 2] = (row - 0.5 + 0.18 * (2.0 * rand(rng) - 1.0)) / n_rows
    end
    edges = Tuple{Int, Int}[]
    for row in 1:n_rows, column in 1:n_columns
        parcel = cell_to_parcel[row, column]
        parcel == 0 && continue
        for (next_row, next_column) in ((row, column + 1), (row + 1, column))
            (next_row <= n_rows && next_column <= n_columns) || continue
            neighbor = cell_to_parcel[next_row, next_column]
            neighbor == 0 && continue
            push!(edges, minmax(parcel, neighbor))
        end
    end
    sort!(edges)
    return coordinates, edges, cells, n_columns
end

function _land_use_neighbors(n_parcels::Int, edges)
    neighbors = [Int[] for _ in 1:n_parcels]
    for (i, j) in edges
        push!(neighbors[i], j)
        push!(neighbors[j], i)
    end
    return neighbors
end

"""Index of the variable for each allowed (parcel, zone), 0 when excluded."""
function land_use_pair_index(prob::LandUseProblem)
    index = zeros(Int, prob.n_parcels, prob.n_zoning_types)
    for (k, (i, z)) in enumerate(prob.pairs)
        index[i, z] = k
    end
    return index
end

_land_use_green(n_zones::Int) = [g for g in _LAND_USE_GREEN_ZONES if g <= n_zones]

function _land_use_district_value(sizes, district, assignment, density::Dict, n_districts)
    value = zeros(Float64, n_districts)
    for i in eachindex(assignment)
        value[district[i]] += sizes[i] * get(density, assignment[i], 0.0)
    end
    return value
end

"""
    land_use_plan_satisfies(prob, plan=prob.feasible_witness; atol=1e-8)

Check an integer zoning plan (zone per parcel) against every row of the model.
"""
function land_use_plan_satisfies(
    prob::LandUseProblem, plan::Union{Nothing, AbstractVector{<:Integer}}=prob.feasible_witness; atol::Float64=1e-8
)
    plan === nothing && return false
    length(plan) == prob.n_parcels || return false
    index = land_use_pair_index(prob)
    all(i -> 1 <= plan[i] <= prob.n_zoning_types && index[i, plan[i]] > 0, 1:prob.n_parcels) || return false
    usage = zeros(Float64, prob.n_districts, prob.n_resources)
    for i in 1:prob.n_parcels, r in 1:prob.n_resources
        usage[prob.parcel_district[i], r] += prob.parcel_sizes[i] * prob.resource_consumption[plan[i], r]
    end
    all(usage .<= prob.resource_capacities .+ atol .* max.(1.0, prob.resource_capacities)) || return false
    housing = _land_use_district_value(prob.parcel_sizes, prob.parcel_district, plan, _LAND_USE_HOUSING_DENSITY, prob.n_districts)
    jobs = _land_use_district_value(prob.parcel_sizes, prob.parcel_district, plan, _LAND_USE_JOB_DENSITY, prob.n_districts)
    all(housing .+ atol .>= prob.housing_target) && all(jobs .+ atol .>= prob.jobs_target) || return false
    green = _land_use_green(prob.n_zoning_types)
    neighbors = _land_use_neighbors(prob.n_parcels, prob.adjacency_edges)
    for i in 1:prob.n_parcels
        prob.green_ratio[i] > 0 && plan[i] == _LAND_USE_RESIDENTIAL || continue
        area = sum(prob.parcel_sizes[j] for j in vcat(i, neighbors[i]) if plan[j] in green; init=0.0)
        area + atol >= prob.green_ratio[i] * prob.parcel_sizes[i] || return false
    end
    for (i, j) in prob.adjacency_edges
        (plan[i], plan[j]) in ((_LAND_USE_RESIDENTIAL, _LAND_USE_INDUSTRIAL), (_LAND_USE_INDUSTRIAL, _LAND_USE_RESIDENTIAL)) &&
            return false
    end
    return true
end

function _land_use_district_lower_bound(prob, district::Int, resource::Int)
    parcels = findall(==(district), prob.parcel_district)
    allowed = [Int[] for _ in 1:prob.n_parcels]
    for (i, z) in prob.pairs
        push!(allowed[i], z)
    end
    minimum_use = [
        prob.parcel_sizes[i] * minimum(prob.resource_consumption[z, resource] for z in allowed[i]) for i in parcels
    ]
    return parcels, minimum_use
end

"""
    land_use_certificate_holds(prob::LandUseProblem)

Recompute the district resource lower bound and check it exceeds the capacity.
"""
function land_use_certificate_holds(prob::LandUseProblem)
    cert = prob.infeasibility_certificate
    cert === nothing && return false
    1 <= cert.district <= prob.n_districts && 1 <= cert.resource_index <= prob.n_resources || return false
    parcels, minimum_use = _land_use_district_lower_bound(prob, cert.district, cert.resource_index)
    parcels == cert.parcels && minimum_use ≈ cert.per_parcel_minimum || return false
    cert.lower_bound ≈ sum(minimum_use) || return false
    cert.capacity == prob.resource_capacities[cert.district, cert.resource_index] || return false
    return cert.capacity < cert.lower_bound * (1 - 1e-9)
end

"""
    LandUseProblem(target_variables, feasibility_status, seed)

Construct a reproducible zoning-plan instance (see `LandUseProblem`).
"""
function LandUseProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    rng = MersenneTwister(seed)
    target = max(target_variables, 1)
    if target <= 250
        n_zoning_types = rand(rng, 3:5)
        n_resources = rand(rng, 3:5)
        development_cost_scale = rand(rng, 50_000:150_000)
        revenue_scale = rand(rng, 20_000:80_000)
        environmental_probability = rand(rng, Uniform(0.20, 0.40))
    elseif target <= 1000
        n_zoning_types = rand(rng, 4:8)
        n_resources = rand(rng, 4:6)
        development_cost_scale = rand(rng, 75_000:250_000)
        revenue_scale = rand(rng, 40_000:120_000)
        environmental_probability = rand(rng, Uniform(0.25, 0.45))
    else
        n_zoning_types = rand(rng, 5:length(_LAND_USE_ZONING_CATALOG))
        n_resources = rand(rng, 5:length(_LAND_USE_RESOURCE_NAMES))
        development_cost_scale = rand(rng, 100_000:500_000)
        revenue_scale = rand(rng, 60_000:200_000)
        environmental_probability = rand(rng, Uniform(0.30, 0.50))
    end
    # Expected number of excluded zones on a restricted parcel: uniform 1..m.
    m = min(3, n_zoning_types - 1)
    expected_allowed = n_zoning_types - environmental_probability * (m + 1) / 2
    n_parcels = max(2, round(Int, target / expected_allowed))
    zoning_names = [String(_LAND_USE_ZONING_CATALOG[j].name) for j in 1:n_zoning_types]
    resource_names = [String(_LAND_USE_RESOURCE_NAMES[k]) for k in 1:n_resources]

    coordinates, edges, cells, n_columns = _land_use_spatial_graph(rng, n_parcels)
    neighbors = _land_use_neighbors(n_parcels, edges)
    # Service districts: square blocks of the grid holding about 80 parcels.
    side = max(1, round(Int, sqrt(80.0)))
    block_of = Dict{Tuple{Int, Int}, Int}()
    parcel_district = zeros(Int, n_parcels)
    for i in 1:n_parcels
        row, column = cells[i]
        key = ((row - 1) ÷ side, (column - 1) ÷ side)
        parcel_district[i] = get!(block_of, key, length(block_of) + 1)
    end
    n_districts = length(block_of)

    parcel_sizes = max.(rand(rng, LogNormal(log(5.0), 0.75), n_parcels), 0.1)
    development_costs = zeros(Float64, n_parcels, n_zoning_types)
    revenues = zeros(Float64, n_parcels, n_zoning_types)
    for i in 1:n_parcels
        accessibility = exp(-2.5 * hypot(coordinates[i, 1] - 0.5, coordinates[i, 2] - 0.5))
        for z in 1:n_zoning_types
            profile = _LAND_USE_ZONING_CATALOG[z]
            urban = z in (1, 2, 3, 6, 8, 9, 10, 11) ? accessibility : 1.0 - accessibility
            development_costs[i, z] =
                development_cost_scale * profile.cost * (0.70 + 0.65 * urban) * rand(rng, LogNormal(0.0, 0.18))
            revenues[i, z] = revenue_scale * profile.revenue * (0.55 + 1.05 * urban) * rand(rng, LogNormal(0.0, 0.22))
        end
    end
    resource_consumption = [
        _LAND_USE_ZONING_CATALOG[z].resources[r] * rand(rng, LogNormal(0.0, 0.16)) for z in 1:n_zoning_types,
        r in 1:n_resources
    ]
    green = _land_use_green(n_zoning_types)

    # Reference plan: parcels in random order take their best-value zone that
    # does not put homes next to industry; homes without a green neighbour
    # then get one where a neighbour can be converted.
    net = revenues .- development_costs
    plan = zeros(Int, n_parcels)
    for i in shuffle(rng, collect(1:n_parcels))
        for z in sortperm(view(net, i, :); rev=true)
            z == _LAND_USE_RESIDENTIAL && any(plan[j] == _LAND_USE_INDUSTRIAL for j in neighbors[i]) && continue
            z == _LAND_USE_INDUSTRIAL && any(plan[j] == _LAND_USE_RESIDENTIAL for j in neighbors[i]) && continue
            plan[i] = z
            break
        end
        plan[i] == 0 && (plan[i] = 2)  # commercial is neutral under the buffer rule
    end
    # Make sure the plan houses people and employs them somewhere.
    if !any(==(_LAND_USE_RESIDENTIAL), plan)
        i = rand(rng, [i for i in 1:n_parcels if all(plan[j] != _LAND_USE_INDUSTRIAL for j in neighbors[i])])
        plan[i] = _LAND_USE_RESIDENTIAL
    end
    if !isempty(green)
        for i in shuffle(rng, findall(==(_LAND_USE_RESIDENTIAL), plan))
            any(plan[j] in green for j in neighbors[i]) && continue
            candidates = [j for j in neighbors[i] if plan[j] != _LAND_USE_RESIDENTIAL]
            (!isempty(candidates) && rand(rng) < 0.7) || continue
            plan[rand(rng, candidates)] = rand(rng, green)
        end
    end

    # Environmental exclusions never remove the reference zone.
    pairs = Tuple{Int, Int}[]
    for i in 1:n_parcels
        excluded = Int[]
        if rand(rng) < environmental_probability
            candidates = [z for z in 1:n_zoning_types if z != plan[i]]
            excluded = sample(rng, candidates, rand(rng, 1:min(3, length(candidates))); replace=false)
        end
        for z in 1:n_zoning_types
            z in excluded || push!(pairs, (i, z))
        end
    end

    usage = zeros(Float64, n_districts, n_resources)
    for i in 1:n_parcels, r in 1:n_resources
        usage[parcel_district[i], r] += parcel_sizes[i] * resource_consumption[plan[i], r]
    end
    housing = _land_use_district_value(parcel_sizes, parcel_district, plan, _LAND_USE_HOUSING_DENSITY, n_districts)
    jobs = _land_use_district_value(parcel_sizes, parcel_district, plan, _LAND_USE_JOB_DENSITY, n_districts)
    district_area = zeros(Float64, n_districts)
    for i in 1:n_parcels
        district_area[parcel_district[i]] += parcel_sizes[i]
    end
    rho = rand(rng, Uniform(0.15, 0.40))
    allowed_residential = falses(n_parcels)
    for (i, z) in pairs
        z == _LAND_USE_RESIDENTIAL && (allowed_residential[i] = true)
    end

    green_ratio = zeros(Float64, n_parcels)
    witness = nothing
    if feasibility_status == unknown
        tightness = rand(rng, Uniform(0.75, 1.25))
        average = vec(sum(resource_consumption; dims=1)) ./ n_zoning_types
        capacities = [district_area[d] * average[r] * tightness * rand(rng, Uniform(0.85, 1.15)) for
                      d in 1:n_districts, r in 1:n_resources]
        housing_target = [district_area[d] * rand(rng, Uniform(0.15, 0.35)) * 20.0 for d in 1:n_districts]
        jobs_target = [district_area[d] * rand(rng, Uniform(0.10, 0.25)) * 30.0 for d in 1:n_districts]
        isempty(green) || (green_ratio[allowed_residential] .= rho)
    else
        capacities = usage .* rand(rng, Uniform(1.03, 1.20), n_districts, n_resources)
        housing_target = housing .* rand(rng, Uniform(0.85, 0.97), n_districts)
        jobs_target = jobs .* rand(rng, Uniform(0.85, 0.97), n_districts)
        if !isempty(green)
            for i in 1:n_parcels
                allowed_residential[i] || continue
                if plan[i] == _LAND_USE_RESIDENTIAL
                    area = sum(parcel_sizes[j] for j in vcat(i, neighbors[i]) if plan[j] in green; init=0.0)
                    green_ratio[i] = min(rho, 0.95 * area / parcel_sizes[i])
                else
                    green_ratio[i] = rho
                end
            end
        end
        witness = plan
    end

    prob = LandUseProblem(
        n_parcels, n_zoning_types, n_resources, n_districts, parcel_sizes, parcel_district, coordinates, edges,
        pairs, development_costs, revenues, resource_consumption, capacities, housing_target, jobs_target,
        green_ratio, zoning_names, resource_names, witness, nothing, feasibility_status,
    )
    if feasibility_status == infeasible
        d = rand(rng, 1:n_districts)
        r = rand(rng, 1:n_resources)
        parcels, minimum_use = _land_use_district_lower_bound(prob, d, r)
        lower_bound = sum(minimum_use)
        capacities[d, r] = lower_bound * rand(rng, Uniform(0.75, 0.93))
        certificate = LandUseInfeasibilityCertificate(d, r, parcels, minimum_use, lower_bound, capacities[d, r])
        prob = LandUseProblem(
            n_parcels, n_zoning_types, n_resources, n_districts, parcel_sizes, parcel_district, coordinates, edges,
            pairs, development_costs, revenues, resource_consumption, capacities, housing_target, jobs_target,
            green_ratio, zoning_names, resource_names, nothing, certificate, feasibility_status,
        )
        @assert land_use_certificate_holds(prob)
    elseif feasibility_status == feasible
        @assert land_use_plan_satisfies(prob)
    end
    return prob
end

"""
    build_model(prob::LandUseProblem)

Build the binary zoning-plan model (deterministic; see `LandUseProblem`).
"""
function build_model(prob::LandUseProblem)
    model = Model()
    K = length(prob.pairs)
    @variable(model, x[1:K], Bin)
    @objective(
        model,
        Max,
        sum(
            prob.parcel_sizes[i] * (prob.revenues[i, z] - prob.development_costs[i, z]) * x[k] for
            (k, (i, z)) in enumerate(prob.pairs)
        )
    )
    index = land_use_pair_index(prob)
    of_parcel = [Int[] for _ in 1:prob.n_parcels]
    of_district = [Int[] for _ in 1:prob.n_districts]
    for (k, (i, _)) in enumerate(prob.pairs)
        push!(of_parcel[i], k)
        push!(of_district[prob.parcel_district[i]], k)
    end
    @constraint(model, parcel_assignment[i in 1:prob.n_parcels], sum(x[k] for k in of_parcel[i]) == 1)
    size(i) = prob.parcel_sizes[i]
    for d in 1:prob.n_districts
        ks = of_district[d]
        for r in 1:prob.n_resources
            @constraint(
                model,
                sum(size(prob.pairs[k][1]) * prob.resource_consumption[prob.pairs[k][2], r] * x[k] for k in ks) <=
                prob.resource_capacities[d, r]
            )
        end
        housing = [k for k in ks if haskey(_LAND_USE_HOUSING_DENSITY, prob.pairs[k][2])]
        if prob.housing_target[d] > 0 && !isempty(housing)
            @constraint(
                model,
                sum(size(prob.pairs[k][1]) * _LAND_USE_HOUSING_DENSITY[prob.pairs[k][2]] * x[k] for k in housing) >=
                prob.housing_target[d]
            )
        end
        jobs = [k for k in ks if haskey(_LAND_USE_JOB_DENSITY, prob.pairs[k][2])]
        if prob.jobs_target[d] > 0 && !isempty(jobs)
            @constraint(
                model,
                sum(size(prob.pairs[k][1]) * _LAND_USE_JOB_DENSITY[prob.pairs[k][2]] * x[k] for k in jobs) >=
                prob.jobs_target[d]
            )
        end
    end
    green = _land_use_green(prob.n_zoning_types)
    neighbors = _land_use_neighbors(prob.n_parcels, prob.adjacency_edges)
    for i in 1:prob.n_parcels
        prob.green_ratio[i] > 0 && index[i, _LAND_USE_RESIDENTIAL] > 0 || continue
        greens = [index[j, g] for j in vcat(i, neighbors[i]) for g in green if index[j, g] > 0]
        @constraint(
            model,
            sum(size(prob.pairs[k][1]) * x[k] for k in greens; init=0.0) -
            prob.green_ratio[i] * size(i) * x[index[i, _LAND_USE_RESIDENTIAL]] >= 0
        )
    end
    for (i, j) in prob.adjacency_edges, (a, b) in ((i, j), (j, i))
        ka, kb = index[a, _LAND_USE_RESIDENTIAL], index[b, _LAND_USE_INDUSTRIAL]
        (ka > 0 && kb > 0) || continue
        @constraint(model, x[ka] + x[kb] <= 1)
    end
    return model
end

register_variant(
    :land_use,
    :standard,
    LandUseProblem,
    "Spatial zoning plan: parcel-zone assignment with district infrastructure capacities, housing and " *
    "jobs targets, green-space accessibility, and residential-industrial buffer rules";
    tags=[:agriculture, :partitioning, :packing],
)
