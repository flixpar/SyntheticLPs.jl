using JuMP
using Random
using Distributions
using StatsBase
using Statistics

"""
    LandUseInfeasibilityCertificate

Solver-independent proof that a land-use instance is infeasible. For the
identified resource, `per_parcel_minimum[i]` is the least consumption possible
for parcel `i` over every environmentally allowed zoning type. Every assignment
therefore consumes at least `lower_bound`, while the model only permits
`capacity < lower_bound`.
"""
struct LandUseInfeasibilityCertificate
    resource_index::Int
    per_parcel_minimum::Vector{Float64}
    lower_bound::Float64
    capacity::Float64
end

Base.:(==)(a::LandUseInfeasibilityCertificate, b::LandUseInfeasibilityCertificate) =
    a.resource_index == b.resource_index &&
    a.per_parcel_minimum == b.per_parcel_minimum &&
    a.lower_bound == b.lower_bound &&
    a.capacity == b.capacity
Base.isequal(a::LandUseInfeasibilityCertificate, b::LandUseInfeasibilityCertificate) = a == b
Base.hash(a::LandUseInfeasibilityCertificate, h::UInt) =
    hash((a.resource_index, a.per_parcel_minimum, a.lower_bound, a.capacity), h)

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

"""
    LandUseProblem <: ProblemGenerator

Binary parcel-zoning assignment with infrastructure capacities, environmental
exclusions, minimum zoning counts, and residential-industrial incompatibility
on a spatial parcel graph.

`feasible_witness` is populated only for requested-feasible instances.
`infeasibility_certificate` is populated only for requested-infeasible
instances. Unknown instances intentionally expose neither claim.
"""
struct LandUseProblem <: ProblemGenerator
    n_parcels::Int
    n_zoning_types::Int
    n_resources::Int
    parcel_sizes::Vector{Float64}
    development_costs::Matrix{Float64}
    revenues::Matrix{Float64}
    resource_consumption::Matrix{Float64}
    resource_capacities::Vector{Float64}
    environmental_restrictions::Matrix{Bool}
    adjacency_matrix::Matrix{Bool}
    zoning_names::Vector{String}
    resource_names::Vector{String}
    min_counts_by_type::Vector{Int}
    zoning_adjacency_constraints::Bool
    minimum_zoning_requirements::Bool
    parcel_coordinates::Matrix{Float64}
    adjacency_edges::Vector{Tuple{Int, Int}}
    feasible_witness::Union{Nothing, Vector{Int}}
    infeasibility_certificate::Union{Nothing, LandUseInfeasibilityCertificate}
end

# Generate a connected, planar-like four-neighbor graph on a jittered grid.
# Parcel identifiers are shuffled over grid cells, so graph structure is not an
# artifact of consecutive variable indices. Only right/down cell pairs are
# emitted, making every edge undirected and unique by construction.
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
    parity = falses(n_parcels)
    for parcel in 1:n_parcels
        row, column = cells[parcel]
        cell_to_parcel[row, column] = parcel
        x_jitter = 0.18 * (2.0 * rand(rng) - 1.0)
        y_jitter = 0.18 * (2.0 * rand(rng) - 1.0)
        coordinates[parcel, 1] = (column - 0.5 + x_jitter) / n_columns
        coordinates[parcel, 2] = (row - 0.5 + y_jitter) / n_rows
        parity[parcel] = isodd(row + column)
    end

    adjacency = falses(n_parcels, n_parcels)
    edges = Tuple{Int, Int}[]
    for row in 1:n_rows, column in 1:n_columns
        parcel = cell_to_parcel[row, column]
        parcel == 0 && continue
        for (next_row, next_column) in ((row, column + 1), (row + 1, column))
            next_row > n_rows && continue
            next_column > n_columns && continue
            neighbor = cell_to_parcel[next_row, next_column]
            neighbor == 0 && continue
            first, second = parcel < neighbor ? (parcel, neighbor) : (neighbor, parcel)
            push!(edges, (first, second))
            adjacency[first, second] = true
            adjacency[second, first] = true
        end
    end
    sort!(edges)
    return coordinates, adjacency, edges, parity
end

function _land_use_minimum_counts(n_parcels::Int, n_zoning_types::Int, enabled::Bool)
    enabled || return Int[]
    n_required = min(3, n_zoning_types, n_parcels)
    base_minimum = max(1, round(Int, 0.10 * n_parcels))
    while base_minimum * n_required > n_parcels
        base_minimum -= 1
    end
    return fill(base_minimum, n_required)
end

# Build a concrete assignment before adding environmental exclusions. The grid
# graph is bipartite, so putting the required residential and industrial parcels
# on the same parity class guarantees that those two required sets are mutually
# nonadjacent without deleting geography to fit the requested label.
function _land_use_reference_assignment(
    net_benefit::Matrix{Float64},
    coordinates::Matrix{Float64},
    neighbors::Vector{Vector{Int}},
    parity::BitVector,
    minimum_counts::Vector{Int},
)
    n_parcels, n_zoning_types = size(net_benefit)
    assignment = zeros(Int, n_parcels)

    residential_minimum = length(minimum_counts) >= 1 ? minimum_counts[1] : 0
    commercial_minimum = length(minimum_counts) >= 2 ? minimum_counts[2] : 0
    industrial_minimum = length(minimum_counts) >= 3 ? minimum_counts[3] : 0

    if residential_minimum + industrial_minimum > 0
        odd_parcels = findall(parity)
        even_parcels = findall(.!parity)
        compatible_pool = length(odd_parcels) >= length(even_parcels) ? odd_parcels : even_parcels
        length(compatible_pool) >= residential_minimum + industrial_minimum ||
            error("Spatial graph partition is too small for zoning minimums")

        residential_order = sort(compatible_pool; by=i -> (coordinates[i, 1], -coordinates[i, 2]))
        residential = residential_order[1:residential_minimum]
        assignment[residential] .= 1

        remaining_compatible = [i for i in compatible_pool if assignment[i] == 0]
        industrial_order = sort(
            remaining_compatible; by=i -> (-coordinates[i, 1], coordinates[i, 2])
        )
        industrial = industrial_order[1:industrial_minimum]
        assignment[industrial] .= 3
    end

    if commercial_minimum > 0
        available = [i for i in 1:n_parcels if assignment[i] == 0]
        # Commercial parcels favor central, accessible locations.
        sort!(available; by=i -> (coordinates[i, 1] - 0.5)^2 + (coordinates[i, 2] - 0.5)^2)
        assignment[available[1:commercial_minimum]] .= 2
    end

    for parcel in 1:n_parcels
        assignment[parcel] != 0 && continue
        zoning_order = sortperm(view(net_benefit, parcel, :); rev=true)
        selected = 0
        for zoning in zoning_order
            if zoning == 1 && any(assignment[neighbor] == 3 for neighbor in neighbors[parcel])
                continue
            elseif zoning == 3 && any(assignment[neighbor] == 1 for neighbor in neighbors[parcel])
                continue
            end
            selected = zoning
            break
        end
        # Commercial is neutral under the only modeled incompatibility and is
        # always present because all size regimes use at least three zones.
        assignment[parcel] = selected == 0 ? 2 : selected
    end
    return assignment
end

"""
    LandUseProblem(target_variables, feasibility_status, seed)

Construct a reproducible land-use instance. All randomness is drawn from a
constructor-local `MersenneTwister`; generation does not reset or consume the
process-wide random stream.
"""
function LandUseProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    rng = MersenneTwister(seed)

    if target_variables <= 250
        n_zoning_types = rand(rng, 3:5)
        n_resources = rand(rng, 3:5)
        development_cost_scale = rand(rng, 50_000:150_000)
        revenue_scale = rand(rng, 20_000:80_000)
        infrastructure_factor = rand(rng, Uniform(0.60, 0.80))
        environmental_probability = rand(rng, Uniform(0.20, 0.40))
    elseif target_variables <= 1000
        n_zoning_types = rand(rng, 4:8)
        n_resources = rand(rng, 4:6)
        development_cost_scale = rand(rng, 75_000:250_000)
        revenue_scale = rand(rng, 40_000:120_000)
        infrastructure_factor = rand(rng, Uniform(0.65, 0.85))
        environmental_probability = rand(rng, Uniform(0.25, 0.45))
    else
        n_zoning_types = rand(rng, 5:length(_LAND_USE_ZONING_CATALOG))
        n_resources = rand(rng, 5:length(_LAND_USE_RESOURCE_NAMES))
        development_cost_scale = rand(rng, 100_000:500_000)
        revenue_scale = rand(rng, 60_000:200_000)
        infrastructure_factor = rand(rng, Uniform(0.70, 0.90))
        environmental_probability = rand(rng, Uniform(0.30, 0.50))
    end

    n_parcels = max(2, round(Int, target_variables / n_zoning_types))
    zoning_names = [String(_LAND_USE_ZONING_CATALOG[j].name) for j in 1:n_zoning_types]
    resource_names = [String(_LAND_USE_RESOURCE_NAMES[k]) for k in 1:n_resources]

    parcel_coordinates, adjacency_matrix, adjacency_edges, parity = _land_use_spatial_graph(
        rng, n_parcels
    )
    neighbors = [Int[] for _ in 1:n_parcels]
    for (first_parcel, second_parcel) in adjacency_edges
        push!(neighbors[first_parcel], second_parcel)
        push!(neighbors[second_parcel], first_parcel)
    end

    zoning_adjacency_constraints = rand(rng) < 0.80
    minimum_zoning_requirements = rand(rng) < 0.90
    min_counts_by_type = _land_use_minimum_counts(
        n_parcels, n_zoning_types, minimum_zoning_requirements
    )

    # Nearby parcels have related land values through their common distance to
    # the urban center, while idiosyncratic log-normal noise preserves variety.
    parcel_sizes = rand(rng, LogNormal(log(5.0), 0.75), n_parcels)
    parcel_sizes = max.(parcel_sizes, 0.1)
    development_costs = zeros(Float64, n_parcels, n_zoning_types)
    revenues = zeros(Float64, n_parcels, n_zoning_types)
    for parcel in 1:n_parcels
        x_coord = parcel_coordinates[parcel, 1]
        y_coord = parcel_coordinates[parcel, 2]
        center_distance = hypot(x_coord - 0.5, y_coord - 0.5)
        accessibility = exp(-2.5 * center_distance)
        rurality = 1.0 - accessibility
        for zoning in 1:n_zoning_types
            profile = _LAND_USE_ZONING_CATALOG[zoning]
            urban_weight = zoning in (1, 2, 3, 6, 8, 9, 10, 11) ? accessibility : rurality
            development_costs[parcel, zoning] =
                development_cost_scale *
                profile.cost *
                (0.70 + 0.65 * urban_weight) *
                rand(rng, LogNormal(0.0, 0.18))
            revenues[parcel, zoning] =
                revenue_scale *
                profile.revenue *
                (0.55 + 1.05 * urban_weight) *
                rand(rng, LogNormal(0.0, 0.22))
        end
    end

    resource_consumption = zeros(Float64, n_zoning_types, n_resources)
    for zoning in 1:n_zoning_types, resource in 1:n_resources
        base = _LAND_USE_ZONING_CATALOG[zoning].resources[resource]
        resource_consumption[zoning, resource] = base * rand(rng, LogNormal(0.0, 0.16))
    end

    net_benefit = revenues .- development_costs
    reference_assignment = _land_use_reference_assignment(
        net_benefit, parcel_coordinates, neighbors, parity, min_counts_by_type
    )

    # Restrictions are status-independent and never erase the planted reference
    # assignment. Thus requested status is controlled by capacity construction,
    # not by silently deleting geography or environmental rules.
    environmental_restrictions = falses(n_parcels, n_zoning_types)
    for parcel in 1:n_parcels
        rand(rng) < environmental_probability || continue
        candidates = [
            zoning for zoning in 1:n_zoning_types if zoning != reference_assignment[parcel]
        ]
        isempty(candidates) && continue
        n_restricted = rand(rng, 1:min(3, length(candidates)))
        restricted = sample(rng, candidates, n_restricted; replace=false)
        environmental_restrictions[parcel, restricted] .= true
    end

    reference_usage = [
        sum(
            parcel_sizes[parcel] * resource_consumption[reference_assignment[parcel], resource] for
            parcel in 1:n_parcels
        ) for resource in 1:n_resources
    ]
    total_area = sum(parcel_sizes)
    average_consumption = vec(mean(resource_consumption; dims=1))
    nominal_capacity =
        total_area .* average_consumption .* infrastructure_factor .*
        rand(rng, Uniform(0.85, 1.15), n_resources)

    feasible_witness = nothing
    infeasibility_certificate = nothing
    if feasibility_status == feasible
        slack = rand(rng, Uniform(1.05, 1.20), n_resources)
        resource_capacities = max.(nominal_capacity, reference_usage .* slack)
        feasible_witness = copy(reference_assignment)
    elseif feasibility_status == infeasible
        # All noncritical resources admit the reference assignment. One random
        # critical resource is then placed strictly below a relaxation-valid
        # lower bound, making both the MILP and its LP relaxation infeasible.
        resource_capacities = reference_usage .* rand(rng, Uniform(1.05, 1.20), n_resources)
        critical_resource = rand(rng, 1:n_resources)
        per_parcel_minimum = zeros(Float64, n_parcels)
        for parcel in 1:n_parcels
            allowed = [
                zoning for zoning in 1:n_zoning_types if !environmental_restrictions[parcel, zoning]
            ]
            minimum_rate = minimum(
                resource_consumption[zoning, critical_resource] for zoning in allowed
            )
            per_parcel_minimum[parcel] = parcel_sizes[parcel] * minimum_rate
        end
        lower_bound = sum(per_parcel_minimum)
        capacity = lower_bound * rand(rng, Uniform(0.72, 0.92))
        resource_capacities[critical_resource] = capacity
        infeasibility_certificate = LandUseInfeasibilityCertificate(
            critical_resource, per_parcel_minimum, lower_bound, capacity
        )
    else
        # Unknown is a nominal scenario distribution, not a hidden label. The
        # planted reference is deliberately not exposed as a feasibility claim.
        resource_capacities = nominal_capacity .* rand(rng, Uniform(0.80, 1.20), n_resources)
    end

    return LandUseProblem(
        n_parcels,
        n_zoning_types,
        n_resources,
        parcel_sizes,
        development_costs,
        revenues,
        resource_consumption,
        resource_capacities,
        environmental_restrictions,
        adjacency_matrix,
        zoning_names,
        resource_names,
        min_counts_by_type,
        zoning_adjacency_constraints,
        minimum_zoning_requirements,
        parcel_coordinates,
        adjacency_edges,
        feasible_witness,
        infeasibility_certificate,
    )
end

"""
    build_model(prob::LandUseProblem)

Build the binary parcel-zoning assignment model. Each undirected spatial edge
is visited exactly once; its two distinct residential-industrial orientations
produce exactly two incompatibility inequalities.
"""
function build_model(prob::LandUseProblem)
    model = Model()

    @variable(model, x[1:prob.n_parcels, 1:prob.n_zoning_types], Bin)

    @objective(
        model,
        Max,
        sum(
            prob.parcel_sizes[parcel] *
            (prob.revenues[parcel, zoning] - prob.development_costs[parcel, zoning]) *
            x[parcel, zoning] for parcel in 1:prob.n_parcels, zoning in 1:prob.n_zoning_types
        )
    )

    @constraint(
        model,
        parcel_assignment[parcel in 1:prob.n_parcels],
        sum(x[parcel, zoning] for zoning in 1:prob.n_zoning_types) == 1
    )

    @constraint(
        model,
        resource_capacity[resource in 1:prob.n_resources],
        sum(
            prob.parcel_sizes[parcel] *
            prob.resource_consumption[zoning, resource] *
            x[parcel, zoning] for parcel in 1:prob.n_parcels, zoning in 1:prob.n_zoning_types
        ) <= prob.resource_capacities[resource]
    )

    for parcel in 1:prob.n_parcels, zoning in 1:prob.n_zoning_types
        if prob.environmental_restrictions[parcel, zoning]
            @constraint(model, x[parcel, zoning] == 0)
        end
    end

    if prob.minimum_zoning_requirements
        @constraint(
            model,
            minimum_zoning[zoning in eachindex(prob.min_counts_by_type)],
            sum(x[parcel, zoning] for parcel in 1:prob.n_parcels) >=
                prob.min_counts_by_type[zoning]
        )
    end

    if prob.zoning_adjacency_constraints && prob.n_zoning_types >= 3
        @constraint(
            model,
            residential_industrial_forward[edge in eachindex(prob.adjacency_edges)],
            x[prob.adjacency_edges[edge][1], 1] + x[prob.adjacency_edges[edge][2], 3] <= 1
        )
        @constraint(
            model,
            residential_industrial_reverse[edge in eachindex(prob.adjacency_edges)],
            x[prob.adjacency_edges[edge][1], 3] + x[prob.adjacency_edges[edge][2], 1] <= 1
        )
    end

    return model
end

register_variant(
    :land_use,
    :standard,
    LandUseProblem,
    "Spatial parcel-zoning assignment with infrastructure, environmental, minimum-mix, and residential-industrial adjacency constraints",
)
