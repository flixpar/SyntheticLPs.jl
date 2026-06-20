using JuMP
using Random
using Distributions

"""
    RevenueManagementProblem <: ProblemGenerator

Generator for network revenue management problems (the deterministic LP, "DLP").

# Overview
Models the capacity-allocation problem faced by airlines, hotels, and car-rental
firms. A set of fixed, perishable **resources** (e.g. flight legs) with limited
capacity is consumed by **products** (e.g. origin–destination itineraries at a
fare class), each of which uses one or more resources. The decisions are how many
of each product to accept, `x[j]`, bounded above by forecast demand and below by
contractual commitments (e.g. group allotments). The objective maximizes total
fare revenue subject to per-resource capacity. This deterministic LP is the
classic relaxation whose optimal dual prices give the bid prices used in
revenue-management control.

Structurally it is a packing LP over a resource–product incidence matrix (each
product touches only a few resources, so the matrix is sparse). Feasibility is
controlled through the commitments: the problem is infeasible exactly when the
committed acceptances on some resource exceed its capacity.

# Fields
- `n_products::Int`: Number of products (decision variables)
- `n_resources::Int`: Number of capacity resources (legs)
- `product_resources::Vector{Vector{Int}}`: Resources consumed by each product
- `fare::Vector{Float64}`: Revenue per unit of each product
- `demand::Vector{Float64}`: Forecast demand (upper bound) per product
- `commitment::Vector{Float64}`: Contractually committed minimum acceptance per product
- `capacity::Vector{Float64}`: Capacity of each resource
"""
struct RevenueManagementProblem <: ProblemGenerator
    n_products::Int
    n_resources::Int
    product_resources::Vector{Vector{Int}}
    fare::Vector{Float64}
    demand::Vector{Float64}
    commitment::Vector{Float64}
    capacity::Vector{Float64}
end

"""
    RevenueManagementProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a network revenue management (DLP) instance.

Variables: `x[j]` per product, for a total of `n_products`.

# Arguments
- `target_variables`: Target number of variables
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function RevenueManagementProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Variables = n_products = target. Resources are fewer (products share legs).
    n_products = max(2, target_variables)
    n_resources = max(2, round(Int, n_products / rand(Uniform(2.5, 5.0))))

    # Incidence: each product uses 1–3 resources (mostly 1–2, as for itineraries).
    product_resources = Vector{Vector{Int}}(undef, n_products)
    for j in 1:n_products
        r = rand()
        k = r < 0.5 ? 1 : (r < 0.9 ? 2 : 3)
        k = min(k, n_resources)
        product_resources[j] = sort(randperm(n_resources)[1:k])
    end

    # Fares: longer itineraries (more legs) earn more, with noise.
    base_fare = rand(Uniform(40.0, 120.0))
    fare = [base_fare * length(product_resources[j]) * rand(Uniform(0.7, 1.6))
            for j in 1:n_products]

    # Forecast demand per product.
    demand = rand(Uniform(2.0, 40.0), n_products)

    # --- Feasibility handling (governed by commitments vs. capacity) ---
    actual_status = feasibility_status
    if feasibility_status == unknown
        actual_status = rand() < 0.7 ? feasible : infeasible
    end

    if actual_status == feasible
        commit_frac = rand(Uniform(0.0, 0.25), n_products)
        cap_frac = rand(Uniform(0.4, 0.8), n_resources)
    else
        commit_frac = rand(Uniform(0.3, 0.7), n_products)
        cap_frac = rand(Uniform(0.4, 0.8), n_resources)
    end

    commitment = demand .* commit_frac

    # Resource membership and scarce capacity sized off the demand passing through.
    products_on_resource = [Int[] for _ in 1:n_resources]
    for j in 1:n_products
        for l in product_resources[j]
            push!(products_on_resource[l], j)
        end
    end
    capacity = zeros(Float64, n_resources)
    for l in 1:n_resources
        through_demand = sum(demand[j] for j in products_on_resource[l]; init = 0.0)
        capacity[l] = through_demand * cap_frac[l]
    end

    if actual_status == infeasible
        # Force a guaranteed violation on one non-empty resource: capacity strictly
        # below the committed acceptances routed through it.
        candidates = [l for l in 1:n_resources if !isempty(products_on_resource[l])]
        if !isempty(candidates)
            l_star = rand(candidates)
            committed_through = sum(commitment[j] for j in products_on_resource[l_star])
            capacity[l_star] = committed_through * rand(Uniform(0.5, 0.9))
        end
    end

    return RevenueManagementProblem(n_products, n_resources, product_resources,
                                    fare, demand, commitment, capacity)
end

"""
    build_model(prob::RevenueManagementProblem)

Build a JuMP model for the network revenue management DLP. Deterministic — uses
only data from the struct fields.

# Returns
- `model`: The JuMP model
"""
function build_model(prob::RevenueManagementProblem)
    model = Model()

    P = prob.n_products
    R = prob.n_resources

    # Acceptances bounded below by commitments and above by forecast demand.
    @variable(model, prob.commitment[j] <= x[j in 1:P] <= prob.demand[j])

    # Objective: maximize total fare revenue.
    @objective(model, Max, sum(prob.fare[j] * x[j] for j in 1:P))

    # Resource capacity constraints.
    products_on_resource = [Int[] for _ in 1:R]
    for j in 1:P
        for l in prob.product_resources[j]
            push!(products_on_resource[l], j)
        end
    end
    for l in 1:R
        if !isempty(products_on_resource[l])
            @constraint(model, sum(x[j] for j in products_on_resource[l]) <= prob.capacity[l])
        end
    end

    return model
end

# Register the variant
register_variant(
    :revenue_management,
    :standard,
    RevenueManagementProblem,
    "Network revenue management deterministic LP (DLP): allocate perishable resource capacity to fare-bearing products to maximize revenue",
)
