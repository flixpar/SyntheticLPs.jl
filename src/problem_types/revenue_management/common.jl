using JuMP
using Random
using Distributions

# Shared hub-and-spoke itinerary data used by the `stochastic_overbooking`
# variant (the `standard` variant builds its own multi-hub flight schedule).

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
