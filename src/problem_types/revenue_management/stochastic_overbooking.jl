using JuMP
using Random
using Distributions

"""
Stored feasible point for stochastic overbooking: first-stage bookings and the
served/denied recourse quantities for every product-scenario pair.
"""
struct StochasticOverbookingWitness
    bookings::Vector{Float64}
    served::Matrix{Float64}
    denied::Matrix{Float64}
end

"""
Proof that mandatory service on one resource in one scenario exceeds capacity.
The mandatory load follows from booking floors, show-up balance, and each
product's maximum denied fraction.
"""
struct StochasticOverbookingCertificate
    resource::Int
    scenario::Int
    mandatory_service_load::Float64
    capacity::Float64
    excess::Float64
end

"""
    StochasticOverbookingRevenueProblem <: ProblemGenerator

Two-stage stochastic network overbooking LP. First-stage bookings are chosen before
show-up rates are known. In every scenario, show-ups are split into served and
denied customers. Served itineraries consume capacity on every leg; product-level
and scenario-wide service standards cap denied service.
"""
struct StochasticOverbookingRevenueProblem <: ProblemGenerator
    n_products::Int
    n_resources::Int
    n_scenarios::Int
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
    scenario_probability::Vector{Float64}
    show_rate::Matrix{Float64}
    denied_service_cost::Vector{Float64}
    max_denied_fraction::Vector{Float64}
    scenario_denied_cap::Vector{Float64}
    market_profile::Symbol
    show_profile::Symbol
    resolved_status::FeasibilityStatus
    feasible_witness::Union{Nothing, StochasticOverbookingWitness}
    infeasibility_certificate::Union{Nothing, StochasticOverbookingCertificate}
end

function _plan_overbooking_dimensions(target_variables::Int)
    target = max(target_variables, 14) # two products and three scenarios
    scenario_range = if target < 150
        (3:5)
    elseif target < 1_200
        (4:8)
    else
        (6:12)
    end
    preferred = (first(scenario_range) + last(scenario_range)) / 2
    best = (typemax(Int), Inf, 2, first(scenario_range))
    for n_scenarios in scenario_range
        block = 1 + 2 * n_scenarios
        n_products = max(2, round(Int, target / block))
        actual = n_products * block
        candidate = (abs(actual - target), abs(n_scenarios - preferred), n_products, n_scenarios)
        candidate < best && (best = candidate)
    end
    return best[3], best[4]
end

@inline function _base_show_rate(rng::AbstractRNG, fare_class::Symbol)
    if fare_class == :economy
        return rand(rng, Uniform(0.92, 0.985))
    elseif fare_class == :premium
        return rand(rng, Uniform(0.88, 0.97))
    end
    return rand(rng, Uniform(0.80, 0.94))
end

function _generate_show_rates(
    rng::AbstractRNG, products::Vector{RevenueManagementProduct}, n_scenarios::Int
)
    profile_draw = rand(rng)
    show_profile = if profile_draw < 0.4
        :stable_business
    elseif profile_draw < 0.78
        :mixed_leisure
    else
        :disruption_prone
    end
    scenario_factor = if show_profile == :stable_business
        rand(rng, Uniform(0.96, 1.025), n_scenarios)
    elseif show_profile == :mixed_leisure
        rand(rng, Uniform(0.88, 1.04), n_scenarios)
    else
        factors = rand(rng, Uniform(0.9, 1.04), n_scenarios)
        factors[rand(rng, 1:n_scenarios)] = rand(rng, Uniform(0.68, 0.84))
        factors
    end

    show_rate = zeros(Float64, length(products), n_scenarios)
    for product in products
        base = _base_show_rate(rng, product.fare_class)
        product_effect = rand(rng, Normal(0.0, 0.012))
        for s in 1:n_scenarios
            show_rate[product.id, s] = clamp(
                base * scenario_factor[s] + product_effect + rand(rng, Normal(0.0, 0.008)),
                0.55,
                0.995,
            )
        end
    end
    return show_profile, show_rate
end

"""
    StochasticOverbookingRevenueProblem(target_variables, feasibility_status, seed)

Construct a stochastic overbooking LP. Variables are bookings `x[p]`, served
show-ups `served[p,s]`, and denied show-ups `denied[p,s]`, for an exact total of
`n_products * (1 + 2*n_scenarios)`.
"""
function StochasticOverbookingRevenueProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    resolved_status = if feasibility_status == unknown
        (rand(rng) < 0.70 ? feasible : infeasible)
    else
        feasibility_status
    end
    n_products, n_scenarios = _plan_overbooking_dimensions(target_variables)
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

    raw_probability = rand(rng, Uniform(0.5, 1.5), n_scenarios)
    scenario_probability = raw_probability ./ sum(raw_probability)
    show_profile, show_rate = _generate_show_rates(rng, products, n_scenarios)

    denied_service_cost = zeros(Float64, n_products)
    max_denied_fraction = zeros(Float64, n_products)
    for product in products
        j = product.id
        if product.fare_class == :economy
            denied_service_cost[j] = fare[j] * rand(rng, Uniform(1.15, 1.65))
            max_denied_fraction[j] = rand(rng, Uniform(0.035, 0.075))
        elseif product.fare_class == :premium
            denied_service_cost[j] = fare[j] * rand(rng, Uniform(1.35, 1.95))
            max_denied_fraction[j] = rand(rng, Uniform(0.018, 0.05))
        else
            denied_service_cost[j] = fare[j] * rand(rng, Uniform(1.7, 2.5))
            max_denied_fraction[j] = rand(rng, Uniform(0.006, 0.025))
        end
    end

    commitment = zeros(Float64, n_products)
    for j in 1:n_products
        if rand(rng) < 0.25
            commitment[j] = round(demand[j] * rand(rng, Uniform(0.04, 0.16)); digits=3)
        end
    end
    if all(iszero, commitment)
        j = rand(rng, 1:n_products)
        commitment[j] = round(0.1 * demand[j]; digits=3)
    end

    scenario_denied_cap = zeros(Float64, n_scenarios)
    for s in 1:n_scenarios
        total_forecast_show = sum(show_rate[j, s] * demand[j] for j in 1:n_products)
        scenario_denied_cap[s] = total_forecast_show * rand(rng, Uniform(0.012, 0.04))
    end

    capacity = zeros(Float64, n_resources)
    for r in 1:n_resources
        products_on_leg = resource_products[r]
        forecast_load = sum(
            scenario_probability[s] * show_rate[j, s] * demand[j] for
            j in products_on_leg, s in 1:n_scenarios
        )
        witness_load = maximum(
            sum(show_rate[j, s] * commitment[j] for j in products_on_leg) for s in 1:n_scenarios
        )
        schedule_capacity = rand(rng, Uniform(profile.seat_capacity...))
        natural_capacity = min(schedule_capacity, forecast_load * rand(rng, Uniform(0.52, 0.78)))
        capacity[r] = max(natural_capacity, 1.07 * witness_load + 0.5)
    end

    feasible_witness = nothing
    infeasibility_certificate = nothing
    if resolved_status == feasible
        served = show_rate .* reshape(commitment, n_products, 1)
        denied = zeros(Float64, n_products, n_scenarios)
        feasible_witness = StochasticOverbookingWitness(copy(commitment), served, denied)
    else
        critical_resource = rand(rng, 1:n_resources)
        affected_products = resource_products[critical_resource]
        for j in affected_products
            commitment[j] = max(
                commitment[j], round(demand[j] * rand(rng, Uniform(0.30, 0.58)); digits=3)
            )
        end
        mandatory_by_scenario = [
            sum(
                (1.0 - max_denied_fraction[j]) * show_rate[j, s] * commitment[j] for
                j in affected_products
            ) for s in 1:n_scenarios
        ]
        mandatory_service_load, critical_scenario = findmax(mandatory_by_scenario)
        capacity[critical_resource] = mandatory_service_load * rand(rng, Uniform(0.70, 0.88))
        excess = mandatory_service_load - capacity[critical_resource]
        infeasibility_certificate = StochasticOverbookingCertificate(
            critical_resource,
            critical_scenario,
            mandatory_service_load,
            capacity[critical_resource],
            excess,
        )
    end

    problem = StochasticOverbookingRevenueProblem(
        n_products,
        n_resources,
        n_scenarios,
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
        scenario_probability,
        show_rate,
        denied_service_cost,
        max_denied_fraction,
        scenario_denied_cap,
        profile.name,
        show_profile,
        resolved_status,
        feasible_witness,
        infeasibility_certificate,
    )
    if resolved_status == feasible
        @assert _stochastic_overbooking_witness_is_valid(problem)
    else
        @assert _stochastic_overbooking_certificate_is_valid(problem)
    end
    return problem
end

function _stochastic_overbooking_witness_is_valid(
    problem::StochasticOverbookingRevenueProblem; atol::Float64=1e-8
)
    problem.resolved_status == feasible || return false
    problem.infeasibility_certificate === nothing || return false
    witness = problem.feasible_witness
    witness === nothing && return false
    length(witness.bookings) == problem.n_products || return false
    size(witness.served) == (problem.n_products, problem.n_scenarios) || return false
    size(witness.denied) == (problem.n_products, problem.n_scenarios) || return false

    for j in 1:problem.n_products
        witness.bookings[j] + atol >= problem.commitment[j] || return false
        witness.bookings[j] <= problem.demand[j] + atol || return false
        for s in 1:problem.n_scenarios
            served = witness.served[j, s]
            denied = witness.denied[j, s]
            served >= -atol || return false
            denied >= -atol || return false
            abs(served + denied - problem.show_rate[j, s] * witness.bookings[j]) <= atol ||
                return false
            denied <=
            problem.max_denied_fraction[j] * problem.show_rate[j, s] * witness.bookings[j] + atol ||
                return false
        end
    end
    for s in 1:problem.n_scenarios
        sum(witness.denied[:, s]) <= problem.scenario_denied_cap[s] + atol || return false
        for r in 1:problem.n_resources
            load = sum(witness.served[j, s] for j in problem.resource_products[r])
            load <= problem.capacity[r] + atol || return false
        end
    end
    return true
end

function _stochastic_overbooking_certificate_is_valid(
    problem::StochasticOverbookingRevenueProblem; atol::Float64=1e-8
)
    problem.resolved_status == infeasible || return false
    problem.feasible_witness === nothing || return false
    certificate = problem.infeasibility_certificate
    certificate === nothing && return false
    1 <= certificate.resource <= problem.n_resources || return false
    1 <= certificate.scenario <= problem.n_scenarios || return false

    mandatory = sum(
        (1.0 - problem.max_denied_fraction[j]) *
        problem.show_rate[j, certificate.scenario] *
        problem.commitment[j] for j in problem.resource_products[certificate.resource]
    )
    isapprox(certificate.mandatory_service_load, mandatory; atol=atol, rtol=1e-10) || return false
    isapprox(certificate.capacity, problem.capacity[certificate.resource]; atol=atol, rtol=1e-10) ||
        return false
    isapprox(
        certificate.excess,
        mandatory - problem.capacity[certificate.resource];
        atol=atol,
        rtol=1e-10,
    ) || return false
    return certificate.excess > atol
end

function build_model(problem::StochasticOverbookingRevenueProblem)
    model = Model()
    P = 1:problem.n_products
    S = 1:problem.n_scenarios
    R = 1:problem.n_resources

    @variable(model, problem.commitment[j] <= bookings[j in P] <= problem.demand[j],)
    @variable(model, served[P, S] >= 0)
    @variable(model, denied[P, S] >= 0)

    if problem.feasible_witness !== nothing
        witness = problem.feasible_witness
        for j in P
            set_start_value(bookings[j], witness.bookings[j])
            for s in S
                set_start_value(served[j, s], witness.served[j, s])
                set_start_value(denied[j, s], witness.denied[j, s])
            end
        end
    end

    @objective(
        model,
        Max,
        sum(
            problem.scenario_probability[s] *
            (problem.fare[j] * served[j, s] - problem.denied_service_cost[j] * denied[j, s]) for
            j in P, s in S
        ),
    )
    @constraint(
        model,
        show_balance[j in P, s in S],
        served[j, s] + denied[j, s] == problem.show_rate[j, s] * bookings[j],
    )
    @constraint(
        model,
        product_denial_cap[j in P, s in S],
        denied[j, s] <= problem.max_denied_fraction[j] * problem.show_rate[j, s] * bookings[j],
    )
    @constraint(
        model,
        scenario_denial_cap[s in S],
        sum(denied[j, s] for j in P) <= problem.scenario_denied_cap[s],
    )
    @constraint(
        model,
        scenario_capacity[r in R, s in S],
        sum(served[j, s] for j in problem.resource_products[r]) <= problem.capacity[r],
    )
    return model
end

register_variant(
    :revenue_management,
    :stochastic_overbooking,
    StochasticOverbookingRevenueProblem,
    "Two-stage stochastic network overbooking with scenario show-ups, denied-service recourse, compensation, and service guarantees",
)
