using JuMP
using Random
using Distributions

"""
    RevenueManagementProduct

Typed itinerary metadata for the deterministic network revenue-management LP.
`resources` contains the capacity legs consumed by one accepted booking.
"""
struct RevenueManagementProduct
    id::Int
    origin::Int
    destination::Int
    fare_class::Symbol
    resources::Vector{Int}
end

"""
Feasible acceptance vector stored for solver-independent status auditing.
"""
struct RevenueManagementWitness
    acceptance::Vector{Float64}
end

"""
Proof that one resource's mandatory contractual load exceeds its capacity.
`excess` equals `committed_load - capacity` and is strictly positive.
"""
struct RevenueManagementCapacityCertificate
    resource::Int
    committed_load::Float64
    capacity::Float64
    excess::Float64
end

"""
    RevenueManagementProblem <: ProblemGenerator

Deterministic network revenue management (the classic deterministic LP, or DLP).
Products are coherent one-leg or hub-connecting itineraries with differentiated
fare classes. Acceptance is bounded by forecast demand and contractual group
commitments, while each physical leg has a shared, perishable capacity.

Requested-feasible instances store the commitment vector as a primal witness.
Requested-infeasible instances store a leg whose mandatory committed load exceeds
capacity. `unknown` resolves to one of the two profiles and records that status.
"""
struct RevenueManagementProblem <: ProblemGenerator
    n_products::Int
    n_resources::Int
    n_nodes::Int
    products::Vector{RevenueManagementProduct}
    product_resources::Vector{Vector{Int}}
    resource_products::Vector{Vector{Int}}
    resource_names::Vector{String}
    resource_origin::Vector{Int}
    resource_destination::Vector{Int}
    fare::Vector{Float64}
    demand::Vector{Float64}
    commitment::Vector{Float64}
    capacity::Vector{Float64}
    market_profile::Symbol
    resolved_status::FeasibilityStatus
    feasible_witness::Union{Nothing, RevenueManagementWitness}
    infeasibility_certificate::Union{Nothing, RevenueManagementCapacityCertificate}
end

function _sample_revenue_market_profile(rng::AbstractRNG)
    profiles = (
        (
            name=:regional_airline,
            base_fare=(55.0, 145.0),
            base_demand=(18.0, 48.0),
            seat_capacity=(55.0, 110.0),
            connection_share=0.32,
        ),
        (
            name=:network_airline,
            base_fare=(90.0, 240.0),
            base_demand=(12.0, 38.0),
            seat_capacity=(120.0, 260.0),
            connection_share=0.62,
        ),
        (
            name=:intercity_rail,
            base_fare=(28.0, 105.0),
            base_demand=(30.0, 75.0),
            seat_capacity=(180.0, 430.0),
            connection_share=0.24,
        ),
    )
    return profiles[rand(rng, eachindex(profiles))]
end

"""
    _generate_revenue_network(n_resources)

Create directed hub-and-spoke capacity legs. Odd-numbered resources leave the hub;
even-numbered resources return from the same spoke. This supplies coherent
two-leg spoke-hub-spoke itineraries without materializing a dense graph.
"""
function _generate_revenue_network(n_resources::Int)
    n_spokes = max(1, cld(n_resources, 2))
    n_nodes = n_spokes + 1
    origin = Int[]
    destination = Int[]
    for spoke in 2:n_nodes
        length(origin) < n_resources || break
        push!(origin, 1)
        push!(destination, spoke)
        length(origin) < n_resources || break
        push!(origin, spoke)
        push!(destination, 1)
    end
    names = ["LEG$(r):$(origin[r])-$(destination[r])" for r in 1:n_resources]
    return n_nodes, names, origin, destination
end

@inline function _sample_revenue_fare_class(rng::AbstractRNG)
    draw = rand(rng)
    return if draw < 0.62
        :economy
    elseif draw < 0.86
        :premium
    else
        :business
    end
end

function _generate_revenue_products(
    rng::AbstractRNG,
    n_products::Int,
    resource_origin::Vector{Int},
    resource_destination::Vector{Int},
    profile,
)
    n_resources = length(resource_origin)
    inbound = [r for r in 1:n_resources if resource_destination[r] == 1]
    outbound = [r for r in 1:n_resources if resource_origin[r] == 1]

    products = Vector{RevenueManagementProduct}(undef, n_products)
    fare = zeros(Float64, n_products)
    demand = zeros(Float64, n_products)
    for j in 1:n_products
        # Give every resource a local product before sampling the remaining mix.
        resources = if j <= n_resources
            [j]
        elseif rand(rng) < profile.connection_share && !isempty(inbound) && !isempty(outbound)
            first_leg = rand(rng, inbound)
            candidates = [
                r for r in outbound if resource_destination[r] != resource_origin[first_leg]
            ]
            if isempty(candidates)
                [rand(rng, 1:n_resources)]
            else
                [first_leg, rand(rng, candidates)]
            end
        else
            [rand(rng, 1:n_resources)]
        end

        origin = resource_origin[first(resources)]
        destination = resource_destination[last(resources)]
        fare_class = _sample_revenue_fare_class(rng)
        products[j] = RevenueManagementProduct(j, origin, destination, fare_class, resources)

        class_fare = if fare_class == :economy
            1.0
        elseif fare_class == :premium
            1.65
        else
            2.7
        end
        class_demand = if fare_class == :economy
            1.0
        elseif fare_class == :premium
            0.58
        else
            0.32
        end
        base_fare = rand(rng, Uniform(profile.base_fare...))
        route_factor = length(resources) == 1 ? 1.0 : rand(rng, Uniform(1.55, 1.9))
        fare[j] = round(
            base_fare * class_fare * route_factor * rand(rng, Uniform(0.88, 1.12)); digits=2
        )

        mean_demand = rand(rng, Uniform(profile.base_demand...)) * class_demand
        demand[j] = round(clamp(rand(rng, LogNormal(log(mean_demand), 0.32)), 2.0, 140.0); digits=2)
    end
    return products, fare, demand
end

"""
    RevenueManagementProblem(target_variables, feasibility_status, seed)

Construct a deterministic network DLP. The model has exactly one acceptance
variable per product, so `n_products = max(2, target_variables)`.
"""
function RevenueManagementProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    resolved_status = if feasibility_status == unknown
        (rand(rng) < 0.72 ? feasible : infeasible)
    else
        feasibility_status
    end

    n_products = max(2, target_variables)
    resource_ratio = rand(rng, Uniform(3.0, 5.5))
    n_resources = clamp(round(Int, n_products / resource_ratio), 2, min(n_products, 80))
    profile = _sample_revenue_market_profile(rng)
    n_nodes, resource_names, resource_origin, resource_destination = _generate_revenue_network(
        n_resources
    )
    products, fare, demand = _generate_revenue_products(
        rng, n_products, resource_origin, resource_destination, profile
    )
    product_resources = [copy(product.resources) for product in products]
    resource_products = [Int[] for _ in 1:n_resources]
    for product in products, resource in product.resources
        push!(resource_products[resource], product.id)
    end

    # Contractual floors represent group blocks and protected allotments. Most
    # products have no floor; positive blocks stay well below forecast demand.
    commitment = zeros(Float64, n_products)
    commitment_probability = profile.name == :intercity_rail ? 0.30 : 0.22
    for j in 1:n_products
        if rand(rng) < commitment_probability
            commitment[j] = round(demand[j] * rand(rng, Uniform(0.04, 0.18)); digits=3)
        end
    end
    if all(iszero, commitment)
        j = rand(rng, 1:n_products)
        commitment[j] = round(0.1 * demand[j]; digits=3)
    end

    capacity = zeros(Float64, n_resources)
    for r in 1:n_resources
        through_demand = sum(demand[j] for j in resource_products[r])
        committed_load = sum(commitment[j] for j in resource_products[r])
        schedule_capacity = rand(rng, Uniform(profile.seat_capacity...))
        demand_capacity = through_demand * rand(rng, Uniform(0.48, 0.78))
        natural_capacity = min(schedule_capacity, demand_capacity)
        capacity[r] = max(natural_capacity, 1.08 * committed_load + 0.5)
    end

    feasible_witness = nothing
    infeasibility_certificate = nothing
    if resolved_status == feasible
        feasible_witness = RevenueManagementWitness(copy(commitment))
    else
        critical_resource = rand(rng, 1:n_resources)
        affected_products = resource_products[critical_resource]
        for j in affected_products
            commitment[j] = max(
                commitment[j], round(demand[j] * rand(rng, Uniform(0.30, 0.62)); digits=3)
            )
        end
        committed_load = sum(commitment[j] for j in affected_products)
        capacity[critical_resource] = committed_load * rand(rng, Uniform(0.68, 0.88))
        excess = committed_load - capacity[critical_resource]
        infeasibility_certificate = RevenueManagementCapacityCertificate(
            critical_resource, committed_load, capacity[critical_resource], excess
        )
    end

    problem = RevenueManagementProblem(
        n_products,
        n_resources,
        n_nodes,
        products,
        product_resources,
        resource_products,
        resource_names,
        resource_origin,
        resource_destination,
        fare,
        demand,
        commitment,
        capacity,
        profile.name,
        resolved_status,
        feasible_witness,
        infeasibility_certificate,
    )
    if resolved_status == feasible
        @assert _revenue_management_witness_is_valid(problem)
    else
        @assert _revenue_management_certificate_is_valid(problem)
    end
    return problem
end

function _revenue_management_witness_is_valid(problem::RevenueManagementProblem; atol::Float64=1e-8)
    problem.resolved_status == feasible || return false
    problem.infeasibility_certificate === nothing || return false
    witness = problem.feasible_witness
    witness === nothing && return false
    length(witness.acceptance) == problem.n_products || return false
    for j in 1:problem.n_products
        witness.acceptance[j] + atol >= problem.commitment[j] || return false
        witness.acceptance[j] <= problem.demand[j] + atol || return false
    end
    for r in 1:problem.n_resources
        load = sum(witness.acceptance[j] for j in problem.resource_products[r])
        load <= problem.capacity[r] + atol || return false
    end
    return true
end

function _revenue_management_certificate_is_valid(
    problem::RevenueManagementProblem; atol::Float64=1e-8
)
    problem.resolved_status == infeasible || return false
    problem.feasible_witness === nothing || return false
    certificate = problem.infeasibility_certificate
    certificate === nothing && return false
    1 <= certificate.resource <= problem.n_resources || return false
    committed_load = sum(
        problem.commitment[j] for j in problem.resource_products[certificate.resource]
    )
    isapprox(certificate.committed_load, committed_load; atol=atol, rtol=1e-10) || return false
    isapprox(certificate.capacity, problem.capacity[certificate.resource]; atol=atol, rtol=1e-10) ||
        return false
    isapprox(
        certificate.excess,
        committed_load - problem.capacity[certificate.resource];
        atol=atol,
        rtol=1e-10,
    ) || return false
    return certificate.excess > atol
end

"""
    build_model(problem::RevenueManagementProblem)

Build the deterministic network revenue-management LP. This function is fully
deterministic and consumes only stored problem data.
"""
function build_model(problem::RevenueManagementProblem)
    model = Model()
    @variable(
        model, problem.commitment[j] <= acceptance[j in 1:problem.n_products] <= problem.demand[j],
    )
    # Preserve the legacy model lookup while giving the variable its clearer
    # domain name. Both keys reference the same JuMP container.
    model[:x] = acceptance
    if problem.feasible_witness !== nothing
        witness = problem.feasible_witness
        for j in 1:problem.n_products
            set_start_value(acceptance[j], witness.acceptance[j])
        end
    end
    @objective(model, Max, sum(problem.fare[j] * acceptance[j] for j in 1:problem.n_products),)
    @constraint(
        model,
        resource_capacity[r in 1:problem.n_resources],
        sum(acceptance[j] for j in problem.resource_products[r]) <= problem.capacity[r],
    )
    return model
end

register_variant(
    :revenue_management,
    :standard,
    RevenueManagementProblem,
    "Deterministic network revenue management with typed hub itineraries, fare classes, group commitments, and shared leg capacity";
    default=true,
)
