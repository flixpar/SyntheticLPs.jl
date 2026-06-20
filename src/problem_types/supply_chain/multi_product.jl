using JuMP
using Random
using StatsBase
using Distributions

"""
    MultiProductSupplyChainProblem <: ProblemGenerator

Generator for multi-commodity capacitated facility-location / supply-chain
problems with several product types shipped over a shared transportation network.

This is a multi-commodity extension of the standard supply-chain network design
problem. Each (facility, customer, mode) arc carries a separate flow variable per
product, demand is specified per (customer, product), facility capacity is
specified per (facility, product) and linked to the facility-open decision, and
transportation modes have a single shared capacity across all products.

# Overview
Models strategic multi-product supply-chain network design. The decisions open
facilities (binary `y`) and ship per-product demand from open facilities to
customers over available transportation modes (continuous `x[arc, product]`).
The objective minimizes fixed facility cost plus transportation cost summed over
all products. Constraints satisfy per-product customer demand, gate per-product
shipments by per-(facility, product) capacity (only when the facility is open),
limit total throughput at each facility by a shared per-facility capacity, and
limit aggregate (cross-product) shipment volume on each transportation mode.

# Fields
- `n_facilities::Int`: Number of potential facility locations
- `n_customers::Int`: Number of customer locations
- `n_products::Int`: Number of distinct product types (commodities)
- `transport_modes::Vector{String}`: Selected transport modes
- `facility_locs::Vector{Tuple{Float64,Float64}}`: Geographic facility locations
- `customer_locs::Vector{Tuple{Float64,Float64}}`: Geographic customer locations
- `cluster_centers::Vector{Tuple{Float64,Float64}}`: Cluster centers for customer distribution
- `cluster_weights::Vector{Float64}`: Weights for cluster importance
- `fixed_costs::Dict{Int, Float64}`: Fixed cost to open each facility
- `demands::Dict{Int, Float64}`: Aggregate demand at each customer (sum over products)
- `capacities::Dict{Int, Float64}`: Shared total capacity at each facility
- `transport_costs::Dict{Tuple{Int,Int,String}, Float64}`: Transport cost per (facility, customer, mode)
- `mode_capacities::Dict{String, Float64}`: Total (cross-product) capacity per transport mode
- `total_demand::Float64`: Total demand across all customers and products
- `product_demands::Dict{Tuple{Int,Int}, Float64}`: Demand per (customer, product)
- `product_capacities::Dict{Tuple{Int,Int}, Float64}`: Capacity per (facility, product)
"""
struct MultiProductSupplyChainProblem <: ProblemGenerator
    n_facilities::Int
    n_customers::Int
    n_products::Int
    transport_modes::Vector{String}
    facility_locs::Vector{Tuple{Float64,Float64}}
    customer_locs::Vector{Tuple{Float64,Float64}}
    cluster_centers::Vector{Tuple{Float64,Float64}}
    cluster_weights::Vector{Float64}
    fixed_costs::Dict{Int, Float64}
    demands::Dict{Int, Float64}
    capacities::Dict{Int, Float64}
    transport_costs::Dict{Tuple{Int,Int,String}, Float64}
    mode_capacities::Dict{String, Float64}
    total_demand::Float64
    product_demands::Dict{Tuple{Int,Int}, Float64}
    product_capacities::Dict{Tuple{Int,Int}, Float64}
end

"""
    MultiProductSupplyChainProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a multi-product (multi-commodity) supply-chain problem instance.

# Variable count
The model has `y[1:n_facilities]` (binary) plus `x[arc, product]` (continuous),
where `arc` ranges over the available (facility, customer, mode) routes:

    n_vars = n_facilities + n_arcs * n_products

Because the per-product flow set multiplies the arc count by `n_products`, the
dimensions are sized with `n_products` factored in so the instance hits the
requested `target_variables`. We approximate
`n_arcs ≈ n_facilities * n_customers * n_modes * density`, pick `n_products` and
`n_modes` up front, then scale `n_facilities` / `n_customers` accordingly.

# Sophisticated feasibility logic
- **Geographic clustering**: Customers clustered with Dirichlet-weighted clusters
- **Facility placement**: Beta-distributed strategic placement near markets
- **K-nearest connectivity**: feasible instances connect each customer to K nearest facilities via a fallback mode
- **Capacity smoothing**: feasible instances widen per-product, shared per-facility, and per-mode capacities to admit a flow

# Arguments
- `target_variables`: Target number of variables (≈ n_facilities + n_arcs × n_products)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function MultiProductSupplyChainProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # --- Pick products / modes / density first, then size the network ---
    # n_vars = n_facilities + n_arcs * n_products,  n_arcs ≈ n_fac * n_cust * n_modes * density
    if target_variables <= 250
        n_products = rand(2:3)
        n_transport_modes = rand(DiscreteUniform(1, 2))
        grid_width = rand(Uniform(200.0, 800.0))
        grid_height = rand(Uniform(200.0, 800.0))
        infrastructure_density = rand(Beta(5, 2)) * 0.3 + 0.7  # 0.7-1.0
        clustering_factor = rand(Beta(3, 2)) * 0.6 + 0.25
        min_fixed_cost = max(100000.0, rand(LogNormal(log(300000), 0.5)))
        max_fixed_cost = min_fixed_cost * rand(Uniform(1.8, 3.5))
        base_demand = rand(Uniform(80.0, 150.0))
        min_demand = base_demand
        max_demand = base_demand * rand(Uniform(3.0, 8.0))
    elseif target_variables <= 1000
        n_products = rand(2:4)
        n_transport_modes = rand(DiscreteUniform(2, 3))
        grid_width = rand(Uniform(800.0, 2000.0))
        grid_height = rand(Uniform(800.0, 2000.0))
        infrastructure_density = rand(Beta(3, 2)) * 0.4 + 0.5  # 0.5-0.9
        clustering_factor = rand(Beta(2, 3)) * 0.5 + 0.2
        min_fixed_cost = max(300000.0, rand(LogNormal(log(800000), 0.6)))
        max_fixed_cost = min_fixed_cost * rand(Uniform(2.0, 4.0))
        base_demand = rand(Uniform(150.0, 300.0))
        min_demand = base_demand
        max_demand = base_demand * rand(Uniform(4.0, 12.0))
    else
        n_products = rand(3:4)
        n_transport_modes = rand(DiscreteUniform(3, 4))
        grid_width = rand(Uniform(2000.0, 5000.0))
        grid_height = rand(Uniform(2000.0, 5000.0))
        infrastructure_density = rand(Beta(2, 3)) * 0.4 + 0.4  # 0.4-0.8
        clustering_factor = rand(Beta(1, 3)) * 0.4 + 0.15
        min_fixed_cost = max(500000.0, rand(LogNormal(log(1500000), 0.7)))
        max_fixed_cost = min_fixed_cost * rand(Uniform(2.5, 5.0))
        base_demand = rand(Uniform(300.0, 600.0))
        min_demand = base_demand
        max_demand = base_demand * rand(Uniform(6.0, 20.0))
    end

    # Effective per-arc density: only available routes get a flow variable.
    # Transport modes and base costs (selected before sizing so the realized
    # arc density can be calibrated for the actual chosen modes).
    all_transport_modes = ["truck", "rail", "ship", "air"]
    transport_base_costs = Dict(
        "truck" => rand(Gamma(4, 0.25)),
        "rail" => rand(Gamma(3, 0.2)),
        "ship" => rand(Gamma(2, 0.15)),
        "air" => rand(Gamma(6, 0.5)),
    )
    transport_modes = sample(all_transport_modes, min(n_transport_modes, length(all_transport_modes)), replace=false)

    capacity_factor = rand(Uniform(1.2, 2.2))
    mode_capacity_factor = rand(Uniform(0.25, 0.65))

    # --- Calibrate realized arc density ---
    # An arc (f,c,m) gets a flow variable only if its infrastructure roll
    # succeeds, and ship/air availability depends on geography, so the realized
    # density is well below the raw infrastructure_density for non-truck modes.
    # Estimate the expected number of available arcs PER (facility, customer)
    # pair by Monte-Carlo over random locations with a throwaway RNG (the main
    # RNG stream is untouched, preserving reproducibility).
    function expected_arcs_per_pair()
        rng = MersenneTwister(seed + 100003)
        diag = sqrt(grid_width^2 + grid_height^2)
        trials = 400
        acc = 0.0
        for _ in 1:trials
            fx = grid_width * rand(rng); fy = grid_height * rand(rng)
            cx = grid_width * rand(rng); cy = grid_height * rand(rng)
            distance = sqrt((fx - cx)^2 + (fy - cy)^2)
            for mode in transport_modes
                prob_available = if mode == "truck"
                    0.98
                elseif mode == "rail"
                    min(0.8, 0.3 + 0.5 * (distance / diag))
                elseif mode == "ship"
                    (abs(fy) < grid_height * 0.1 || abs(cy) < grid_height * 0.1) ? 0.8 : 0.0
                else  # air
                    distance > diag * 0.3 ? 0.7 : 0.2
                end
                acc += prob_available * infrastructure_density
            end
        end
        return max(acc / trials, 0.05)
    end

    density_per_pair = expected_arcs_per_pair()

    # Target arcs so that n_arcs * n_products ≈ target_variables (y vars are few).
    target_arcs = max(n_products + 1, (target_variables - 1) / n_products)

    # n_arcs ≈ (n_fac * n_cust) * density_per_pair.  Pick a facility:customer
    # shape similar to the standard variant (more customers than facilities),
    # solving n_fac * n_cust = target_arcs / density_per_pair.
    fac_cust_product = target_arcs / density_per_pair
    fac_cust_product = max(fac_cust_product, 9.0)

    # Use roughly n_customers ≈ shape * n_facilities.
    shape = 4.0
    n_facilities = max(3, round(Int, sqrt(fac_cust_product / shape)))
    n_customers = max(n_products, round(Int, fac_cust_product / n_facilities))

    # Geographic clusters
    n_clusters = max(2, round(Int, sqrt(n_customers) * clustering_factor))
    cluster_centers = [(grid_width * rand(), grid_height * rand()) for _ in 1:n_clusters]

    # Facility locations
    facility_locs = Vector{Tuple{Float64,Float64}}()
    for _ in 1:n_facilities
        if rand() < 0.4
            center = rand(cluster_centers)
            x = clamp(center[1] + rand(Normal(0, grid_width * 0.12)), 0, grid_width)
            y = clamp(center[2] + rand(Normal(0, grid_height * 0.12)), 0, grid_height)
        else
            x = grid_width * rand(Beta(1.5, 1.5))
            y = grid_height * rand(Beta(1.5, 1.5))
        end
        push!(facility_locs, (x, y))
    end

    # Customer locations
    cluster_weights = rand(Dirichlet(ones(n_clusters)))
    customer_locs = Vector{Tuple{Float64,Float64}}()
    for _ in 1:n_customers
        cluster_idx = sample(1:n_clusters, Weights(cluster_weights))
        center = cluster_centers[cluster_idx]
        base_spread = grid_width * (1 - clustering_factor) * 0.08
        spread = rand(LogNormal(log(base_spread), 0.3))
        x = clamp(center[1] + rand(Normal(0, spread)), 0, grid_width)
        y = clamp(center[2] + rand(Normal(0, spread)), 0, grid_height)
        push!(customer_locs, (x, y))
    end

    # Fixed costs (market-potential correlated)
    fixed_costs = Dict{Int, Float64}()
    for f in 1:n_facilities
        distances = [sqrt((facility_locs[f][1] - c[1])^2 + (facility_locs[f][2] - c[2])^2) for c in customer_locs]
        market_potential = sum(exp.(-distances ./ (grid_width * 0.2)))
        base_cost = min_fixed_cost + (max_fixed_cost - min_fixed_cost) * (0.2 + 0.5 * market_potential / n_customers)
        fixed_costs[f] = base_cost * rand(LogNormal(log(1.0), 0.25))
    end

    # Aggregate customer demands (cluster-size correlated)
    demands = Dict{Int, Float64}()
    for c in 1:n_customers
        distances = [sqrt((customer_locs[c][1] - center[1])^2 + (customer_locs[c][2] - center[2])^2) for center in cluster_centers]
        _, cluster_idx = findmin(distances)
        base_demand_val = min_demand + (max_demand - min_demand) * (0.2 + 0.8 * cluster_weights[cluster_idx])
        demands[c] = base_demand_val * rand(LogNormal(log(1.0), 0.4))
    end

    total_demand = sum(values(demands))
    avg_capacity = (total_demand / n_facilities) * capacity_factor

    # Shared per-facility capacities
    capacities = Dict{Int, Float64}()
    fc_min = minimum(values(fixed_costs))
    fc_max = maximum(values(fixed_costs))
    for f in 1:n_facilities
        relative_cost = (fixed_costs[f] - fc_min) / max(1.0, fc_max - fc_min)
        capacities[f] = avg_capacity * (0.6 + 0.8 * relative_cost) * rand(Gamma(3, 1/3))
    end

    # Transport costs and infrastructure availability
    transport_costs = Dict{Tuple{Int,Int,String}, Float64}()
    infrastructure = Dict{Tuple{Int,Int,String}, Bool}()
    max_demand_val = maximum(values(demands))
    for f in 1:n_facilities
        for c in 1:n_customers
            distance = sqrt((facility_locs[f][1] - customer_locs[c][1])^2 + (facility_locs[f][2] - customer_locs[c][2])^2)
            for mode in transport_modes
                prob_available = if mode == "truck"
                    0.98
                elseif mode == "rail"
                    min(0.8, 0.3 + 0.5 * (distance / sqrt(grid_width^2 + grid_height^2)))
                elseif mode == "ship"
                    any(loc -> abs(loc[2]) < grid_height * 0.1, [facility_locs[f], customer_locs[c]]) ? 0.8 : 0.0
                else  # air
                    distance > sqrt(grid_width^2 + grid_height^2) * 0.3 ? 0.7 : 0.2
                end
                infrastructure[(f, c, mode)] = rand() < prob_available * infrastructure_density
                if infrastructure[(f, c, mode)]
                    base_cost = get(transport_base_costs, mode, 1.0)
                    terrain_factor = rand(LogNormal(log(1.0), 0.15))
                    volume_factor = 1.0 - 0.25 * (demands[c] / max_demand_val)
                    efficiency_factor = rand(Beta(3, 2)) * 0.4 + 0.8
                    transport_costs[(f, c, mode)] = base_cost * distance * terrain_factor * volume_factor * efficiency_factor
                end
            end
        end
    end

    # Mode capacities (shared across products)
    mode_capacities = Dict{String, Float64}()
    for mode in transport_modes
        base_capacity = total_demand * mode_capacity_factor
        mult = if mode == "truck"
            rand(Gamma(4, 0.25))
        elseif mode == "rail"
            rand(Gamma(6, 0.33))
        elseif mode == "ship"
            rand(Gamma(9, 0.33))
        else
            rand(Gamma(2, 0.25))
        end
        mode_capacities[mode] = base_capacity * mult
    end

    transport_costs = Dict(k => v for (k, v) in transport_costs if infrastructure[k])

    # Per-(customer, product) demands via a product split
    product_split = rand(Dirichlet(ones(n_products)))
    product_demands = Dict{Tuple{Int,Int}, Float64}()
    for c in 1:n_customers, p in 1:n_products
        product_demands[(c, p)] = demands[c] * product_split[p] * rand(Uniform(0.8, 1.2))
    end

    # Per-(facility, product) capacities
    product_capacities = Dict{Tuple{Int,Int}, Float64}()
    for f in 1:n_facilities, p in 1:n_products
        product_capacities[(f, p)] = capacities[f] / n_products * rand(Uniform(0.8, 1.2))
    end

    # --- Feasibility handling ---
    if feasibility_status == feasible
        # K-NEAREST CONNECTIVITY via fallback mode
        fallback_mode = ("truck" in transport_modes) ? "truck" : transport_modes[1]
        K = min(max(3, ceil(Int, n_facilities ÷ 3)), n_facilities)
        customers_linked_to_facility = [Int[] for _ in 1:n_facilities]
        for c in 1:n_customers
            dvec = [sqrt((facility_locs[f][1] - customer_locs[c][1])^2 + (facility_locs[f][2] - customer_locs[c][2])^2) for f in 1:n_facilities]
            nearest_idxs = sortperm(dvec)[1:K]
            for f in nearest_idxs
                if !haskey(transport_costs, (f, c, fallback_mode))
                    base_cost = get(transport_base_costs, fallback_mode, 1.0)
                    terrain_factor = rand(LogNormal(log(1.0), 0.15))
                    volume_factor = 1.0 - 0.25 * (demands[c] / max_demand_val)
                    efficiency_factor = rand(Beta(3, 2)) * 0.4 + 0.8
                    transport_costs[(f, c, fallback_mode)] = base_cost * dvec[f] * terrain_factor * volume_factor * efficiency_factor
                end
                push!(customers_linked_to_facility[f], c)
            end
        end

        # Approximate demand share routed to each facility
        approx_share = zeros(Float64, n_facilities)
        for f in 1:n_facilities
            for c in customers_linked_to_facility[f]
                nlinks = length([ff for ff in 1:n_facilities if c in customers_linked_to_facility[ff]])
                approx_share[f] += demands[c] / max(1, nlinks)
            end
        end

        # Widen shared facility capacity to cover its routed share with margin
        for f in 1:n_facilities
            if capacities[f] < 1.1 * approx_share[f]
                capacities[f] = 1.1 * approx_share[f]
            end
        end

        # The model enforces the realized per-(customer, product) demands, whose
        # total can drift above `total_demand` because of the U(0.8, 1.2) jitter
        # on each product demand. Size all capacities against the ACTUAL realized
        # demand, not `total_demand`, otherwise the requested-feasible instance
        # can be genuinely infeasible.
        effective_demand = sum(values(product_demands))

        # Connectivity-aware capacity guarantee: a customer can only be served by
        # the facilities it is linked to (via the fallback mode), so an aggregate
        # per-product total is not enough — capacity could sit at facilities the
        # customer cannot reach. Size each facility's per-product capacity to
        # absorb ALL of its linked customers' demand for that product; then
        # routing every customer to a single linked facility is always feasible.
        for f in 1:n_facilities
            for p in 1:n_products
                linked_demand = isempty(customers_linked_to_facility[f]) ? 0.0 :
                    sum(product_demands[(c, p)] for c in customers_linked_to_facility[f])
                needed = 1.1 * linked_demand
                if product_capacities[(f, p)] < needed
                    product_capacities[(f, p)] = needed
                end
            end
        end

        # Keep each shared facility capacity at least the sum of its per-product
        # caps (so the per-facility constraint never undercuts the per-product ones).
        for f in 1:n_facilities
            sum_pp = sum(product_capacities[(f, p)] for p in 1:n_products)
            if capacities[f] < sum_pp
                capacities[f] = sum_pp
            end
        end

        # Ensure total facility capacity covers the realized demand.
        if sum(values(capacities)) < 1.05 * effective_demand
            scale = 1.05 * effective_demand / max(sum(values(capacities)), eps())
            for f in 1:n_facilities
                capacities[f] *= scale
                for p in 1:n_products
                    product_capacities[(f, p)] *= scale
                end
            end
        end

        # Ensure mode capacities can move all realized demand (fallback alone is enough)
        if mode_capacities[fallback_mode] < 1.1 * effective_demand
            mode_capacities[fallback_mode] = 1.1 * effective_demand
        end
        if sum(mode_capacities[m] for m in transport_modes) < 1.05 * effective_demand
            scale = 1.05 * effective_demand / max(sum(mode_capacities[m] for m in transport_modes), eps())
            for m in transport_modes
                mode_capacities[m] *= scale
            end
        end

    elseif feasibility_status == infeasible
        # Deterministic contradiction: total facility throughput cannot meet
        # total demand. Shrink shared per-facility capacities (and per-product
        # caps) so their sum is strictly below total demand with a margin.
        desired_ratio = rand(Uniform(0.6, 0.8))
        cur_total = sum(values(capacities))
        scale = desired_ratio * total_demand / max(cur_total, eps())
        for f in 1:n_facilities
            capacities[f] *= scale
            for p in 1:n_products
                # Per-product caps also shrink and cannot exceed shared cap
                product_capacities[(f, p)] = min(product_capacities[(f, p)] * scale, capacities[f])
            end
        end
        # demand satisfaction requires sum_f throughput_f >= total_demand, but
        # sum_f capacities[f] = desired_ratio * total_demand < total_demand.
    end
    # For unknown, leave as-is (natural instance, no forced infeasibility)

    return MultiProductSupplyChainProblem(
        n_facilities,
        n_customers,
        n_products,
        transport_modes,
        facility_locs,
        customer_locs,
        cluster_centers,
        cluster_weights,
        fixed_costs,
        demands,
        capacities,
        transport_costs,
        mode_capacities,
        total_demand,
        product_demands,
        product_capacities,
    )
end

"""
    build_model(prob::MultiProductSupplyChainProblem)

Build a JuMP model for the multi-product supply-chain problem (deterministic).

# Returns
- `model`: The JuMP model
"""
function build_model(prob::MultiProductSupplyChainProblem)
    model = Model()

    # Open-facility decisions
    @variable(model, y[1:prob.n_facilities], Bin)

    # Available (facility, customer, mode) arcs
    valid_combinations = [(f, c, m) for f in 1:prob.n_facilities, c in 1:prob.n_customers, m in prob.transport_modes
                          if haskey(prob.transport_costs, (f, c, m))]

    # Per-arc, per-product flow.  n_vars = n_facilities + n_arcs * n_products
    @variable(model, x[valid_combinations, 1:prob.n_products] >= 0)

    # Objective: fixed facility cost + transport cost summed over products
    @objective(model, Min,
        sum(prob.fixed_costs[f] * y[f] for f in 1:prob.n_facilities) +
        sum(prob.transport_costs[combo] * x[combo, p] for combo in valid_combinations, p in 1:prob.n_products)
    )

    # Per-(customer, product) demand satisfaction
    for c in 1:prob.n_customers, p in 1:prob.n_products
        combos = filter(combo -> combo[2] == c, valid_combinations)
        @constraint(model, sum(x[combo, p] for combo in combos) >= prob.product_demands[(c, p)])
    end

    # Per-(facility, product) capacity, gated by facility-open decision
    for f in 1:prob.n_facilities, p in 1:prob.n_products
        combos = filter(combo -> combo[1] == f, valid_combinations)
        @constraint(model, sum(x[combo, p] for combo in combos) <= prob.product_capacities[(f, p)] * y[f])
    end

    # Shared per-facility total-capacity constraint (across all products)
    for f in 1:prob.n_facilities
        combos = filter(combo -> combo[1] == f, valid_combinations)
        @constraint(model, sum(x[combo, p] for combo in combos, p in 1:prob.n_products) <= prob.capacities[f] * y[f])
    end

    # Shared per-mode capacity (across all products)
    for m in prob.transport_modes
        combos = filter(combo -> combo[3] == m, valid_combinations)
        @constraint(model, sum(x[combo, p] for combo in combos, p in 1:prob.n_products) <= prob.mode_capacities[m])
    end

    return model
end

# Register the variant
register_variant(
    :supply_chain,
    :multi_product,
    MultiProductSupplyChainProblem,
    "Multi-commodity capacitated supply-chain network design with per-product flows, demands, and facility capacities over a shared transport network",
)
