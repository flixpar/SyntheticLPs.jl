using JuMP
using Random
using Distributions

"""
    CVRPWitness

Planted feasible routing for [`CVRPProblem`](@ref): exactly `K` non-empty routes,
each a depot-to-depot sequence of customer indices (`1..N`, i.e. node `c + 1`)
whose total demand is at most `Q`. Setting `x = 1` on consecutive arcs and the
remaining load on each arc as `f` satisfies every row, so the MIP and its LP
relaxation are feasible.
"""
struct CVRPWitness
    routes::Vector{Vector{Int}}
end

"""
    CVRPFleetCapacityCertificate

LP-row infeasibility certificate for [`CVRPProblem`](@ref): the depot load row
gives `total_demand = Σ_j f[depot,j] - Σ_i f[i,depot] ≤ Σ_j f[depot,j]`, the
coupling rows bound that by `Q · Σ_j x[depot,j]`, and the depot degree row
fixes `Σ_j x[depot,j] = K`; so `total_demand ≤ Q·K = fleet_capacity`. The
generator keeps `total_demand ≥ 1.1 · fleet_capacity`.
"""
struct CVRPFleetCapacityCertificate
    total_demand::Float64
    fleet_capacity::Float64
end

"""
    CVRPProblem <: ProblemGenerator

Generator for the Capacitated Vehicle Routing Problem (CVRP), formulated as a
mixed-integer program with a meaningful continuous (LP) relaxation.

# Overview

A homogeneous fleet of `K` vehicles, each of capacity `Q`, is based at a single
depot and must serve `N` customers, each with a positive demand `d_c`. The
network is a complete directed graph over `{depot} ∪ customers` (no self-loops).
The objective minimizes total Euclidean travel cost over the arcs that are used.

The formulation uses **single-commodity flow** (Gavish–Graves) subtour
elimination, which is what makes the LP relaxation a genuine routing relaxation
rather than a collection of fractional inter-customer cycles:

  - Binary arc variables `x[i,j] ∈ {0,1}` select which arcs are traversed.
  - Continuous load variables `f[i,j] ≥ 0` carry the (single-commodity) vehicle
    load along each arc. Load is sourced at the depot and consumed at customers.

Key structural couplings:

  - **Degree constraints** force every customer to have exactly one incoming and
    one outgoing arc, and the depot to have exactly `K` outgoing and `K` incoming
    arcs (i.e. `K` routes leave and return).
  - **Flow (load) conservation**: at each customer the inbound load minus outbound
    load equals that customer's demand; at the depot the net outflow of load equals
    total demand. This *anchors* all load to the depot.
  - **Capacity coupling** `d_j · x[i,j] ≤ f[i,j] ≤ (Q - d_i) · x[i,j]` (with
    `d_depot = 0`) simultaneously (a) forbids load on unused arcs, (b) limits the
    load on any depot-leaving arc to `Q`, the per-route capacity bound, and
    (c) uses the strengthened Gavish–Graves bounds — an arc entering `j` carries
    at least `j`'s own demand, and an arc leaving customer `i` at most what is
    left after serving `i` (Letchford & Salazar-González 2006).

Because load must originate at the depot and flow only along used arcs, the
continuous relaxation cannot manufacture free inter-customer cycles: the depot
net-outflow constraint `Σ_j f[depot,j] - Σ_i f[i,depot] = total_demand` combined
with `f[depot,j] ≤ Q · x[depot,j]` ties feasibility directly to fleet capacity.
This is what the brief calls a *non-degenerate* relaxation, in contrast to the
per-vehicle flow formulation in the original source branch.

This is a MIP whose continuous relaxation is a meaningful routing relaxation
(cf. the CLAUDE.md "Model classes" section): the relaxed model is a useful LP
test instance, but a fractional `x` is not a directly implementable set of tours.

# Fields

  - `n_customers::Int`: Number of customers `N`
  - `n_vehicles::Int`: Fleet size `K`
  - `vehicle_capacity::Float64`: Per-vehicle capacity `Q`
  - `depot_location::Tuple{Float64,Float64}`: Depot coordinates
  - `customer_locations::Vector{Tuple{Float64,Float64}}`: Customer coordinates
  - `demands::Vector{Float64}`: Demand at each customer (length `N`, all `> 0`)
  - `dist::Matrix{Float64}`: Arc cost matrix over nodes `1..N+1` (node 1 = depot,
    nodes `2..N+1` = customers); `dist[i,i] = 0`
  - `feasible_witness::Union{Nothing,CVRPWitness}`: planted routes (`feasible` only)
  - `infeasibility_certificate::Union{Nothing,CVRPFleetCapacityCertificate}`:
    fleet-capacity shortfall (`infeasible` only)
"""
struct CVRPProblem <: ProblemGenerator
    n_customers::Int
    n_vehicles::Int
    vehicle_capacity::Float64
    depot_location::Tuple{Float64, Float64}
    customer_locations::Vector{Tuple{Float64, Float64}}
    demands::Vector{Float64}
    dist::Matrix{Float64}
    feasible_witness::Union{Nothing, CVRPWitness}
    infeasibility_certificate::Union{Nothing, CVRPFleetCapacityCertificate}
end

# First-fit-decreasing packing of customer indices into capacity-`Q` bins.
function _cvrp_ffd_bins(demands::Vector{Float64}, Q::Float64)
    bins = Vector{Int}[]
    remaining = Float64[]
    for c in sortperm(demands; rev=true)
        b = findfirst(>=(demands[c]), remaining)
        if b === nothing
            push!(bins, [c])
            push!(remaining, Q - demands[c])
        else
            push!(bins[b], c)
            remaining[b] -= demands[c]
        end
    end
    return bins
end

# Turn `bins` (<= K of them, N >= K customers) into exactly K routes by peeling
# single customers off multi-customer bins, then order each route by a
# nearest-neighbour walk from the depot (node 1; customer c is node c + 1).
function _cvrp_routes_from_bins(bins::Vector{Vector{Int}}, K::Int, dist::Matrix{Float64})
    routes = [copy(b) for b in bins]
    while length(routes) < K
        r = findfirst(r -> length(r) >= 2, routes)
        push!(routes, [pop!(routes[r])])
    end
    for r in eachindex(routes)
        left = Set(routes[r])
        ordered = Int[]
        at = 1
        while !isempty(left)
            nxt = argmin(c -> (dist[at, c + 1], c), collect(left))
            push!(ordered, nxt)
            delete!(left, nxt)
            at = nxt + 1
        end
        routes[r] = ordered
    end
    return routes
end

# First-fit-decreasing bin count: the number of capacity-`Q` bins a
# first-fit-decreasing packing of `demands` uses. With `Q >= maximum(demands)`
# every item fits in some bin, so this certifies the demands can be split into
# that many routes of capacity `Q` (used to guarantee integer CVRP feasibility).
function _ffd_bin_count(demands::Vector{Float64}, Q::Float64)
    remaining = Float64[]            # remaining capacity of each open bin
    for d in sort(demands; rev=true)
        b = findfirst(>=(d), remaining)
        if b === nothing
            push!(remaining, Q - d)
        else
            remaining[b] -= d
        end
    end
    return length(remaining)
end

"""
    CVRPProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a Capacitated Vehicle Routing Problem instance.

# Variable-count formula

On a complete directed graph over `N+1` nodes (depot + `N` customers) with no
self-loops there are `(N+1)*N` arcs. The model creates one binary `x` and one
continuous `f` per arc:

    total = 2 * (N + 1) * N

So `N = round((sqrt(1 + 2·target) - 1) / 2)`, the root of `2N(N+1) = target`
(clamped to `N ≥ 3`): the count is within `2N` of the target. For
`target = 100` this gives `N = 7` (112 vars); for `target = 500`, `N = 15`
(480 vars).

# Arguments

  - `target_variables`: Target number of decision variables across both arc blocks
  - `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
  - `seed`: Random seed for reproducibility

# Feasibility

  - `feasible`: `K*Q ≥ total_demand` with margin (≈ 1.15×), `K ≤ N`, every single
    demand `≤ Q`, and the demands are certified to pack into `≤ K` routes of capacity
    `Q` (Q is raised until a first-fit-decreasing packing fits). A concrete integer
    routing therefore exists, so both the MIP and its LP relaxation are feasible.
    The routes are stored as a [`CVRPWitness`](@ref).
  - `infeasible`: vehicles are out of service: the fleet shrinks to
    `K = floor(total_demand / (Q · overload))` with `overload ∈ [1.1, 1.3]` and
    `Q` is then raised to `total_demand / (K · overload)`, so every demand still
    fits a vehicle but `total_demand = overload · K·Q`. Since the depot
    net-outflow must equal `total_demand` yet is bounded by
    `Q · Σ_j x[depot,j] = Q·K`, the model is infeasible even relaxed
    ([`CVRPFleetCapacityCertificate`](@ref)). The argument chains the depot load
    row, `K` coupling rows and the depot degree row, so presolve does not see
    it. (Tiny instances whose demand cannot fill even one overloaded vehicle
    inflate demands instead.)
  - `unknown`: a natural instance, biased toward feasible but not forced.
"""
function CVRPProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    rng = MersenneTwister(seed)

    # --- Dimension sizing ---
    # total = 2 * (N + 1) * N  =>  N is the rounded positive root.
    N = max(3, round(Int, (sqrt(1 + 2 * target_variables) - 1) / 2))

    # --- Scale-tiered parameter ranges ---
    total_vars = 2 * (N + 1) * N
    if total_vars <= 250
        grid_size = rand(rng, 40.0:5.0:120.0)
        demand_lo, demand_hi = 5.0, 45.0
        cost_per_km = rand(rng, 0.8:0.1:1.8)
        n_clusters = rand(rng, 2:3)
    elseif total_vars <= 1000
        grid_size = rand(rng, 100.0:20.0:300.0)
        demand_lo, demand_hi = 10.0, 90.0
        cost_per_km = rand(rng, 1.0:0.1:2.5)
        n_clusters = rand(rng, 3:5)
    else
        grid_size = rand(rng, 250.0:50.0:700.0)
        demand_lo, demand_hi = 20.0, 200.0
        cost_per_km = rand(rng, 1.5:0.2:3.5)
        n_clusters = rand(rng, 5:8)
    end

    # --- Depot near grid center ---
    depot_location = (grid_size * (0.4 + 0.2 * rand(rng)), grid_size * (0.4 + 0.2 * rand(rng)))

    # --- Customers clustered into a few neighborhoods ---
    cluster_centers = [(grid_size * rand(rng), grid_size * rand(rng)) for _ in 1:n_clusters]
    cluster_spread = grid_size / (2.5 * n_clusters)
    customer_locations = Tuple{Float64, Float64}[]
    for _ in 1:N
        center = rand(rng, cluster_centers)
        x = clamp(center[1] + randn(rng) * cluster_spread, 0.0, grid_size)
        y = clamp(center[2] + randn(rng) * cluster_spread, 0.0, grid_size)
        push!(customer_locations, (x, y))
    end

    # --- Log-normal demands (few large shipments, many small) ---
    log_mean = log(sqrt(demand_lo * demand_hi))
    log_std = log(demand_hi / demand_lo) / 4
    demands = [clamp(exp(rand(rng, Normal(log_mean, log_std))), demand_lo, demand_hi) for _ in 1:N]
    demands = round.(demands; digits=2)

    total_demand = sum(demands)
    avg_demand = total_demand / N
    max_demand = maximum(demands)

    # --- Vehicle capacity: a vehicle serves ~3-6 customers on average ---
    serve_count = 3.0 + 3.0 * rand(rng)           # 3..6 customers per vehicle
    vehicle_capacity = avg_demand * serve_count
    # Every single customer must fit in a vehicle (for feasible/unknown).
    vehicle_capacity = max(vehicle_capacity, max_demand * 1.1)
    vehicle_capacity = round(vehicle_capacity; digits=2)

    # --- Fleet size: enough vehicles to cover demand, clamped to N ---
    slack = 1.15 + 0.15 * rand(rng)               # 1.15 .. 1.30
    n_vehicles = max(2, ceil(Int, total_demand / vehicle_capacity * slack))
    n_vehicles = min(n_vehicles, N)            # require K <= N

    # --- Distance / cost matrix over nodes 1..N+1 (node 1 = depot) ---
    all_locs = [depot_location; customer_locations]   # length N+1
    n_nodes = N + 1
    dist = zeros(n_nodes, n_nodes)
    for i in 1:n_nodes, j in 1:n_nodes
        if i == j
            dist[i, j] = 0.0
        else
            a = all_locs[i]
            b = all_locs[j]
            d = sqrt((a[1] - b[1])^2 + (a[2] - b[2])^2)
            # small asymmetric per-arc variation for realism
            dist[i, j] = round(d * cost_per_km * (0.95 + 0.1 * rand(rng)); digits=2)
        end
    end

    # --- Resolve feasibility intent ---
    witness = nothing
    certificate = nothing
    if feasibility_status == feasible
        # Guarantee K*Q >= total_demand with margin, K <= N, max demand <= Q.
        # K is already <= N and sized from demand; widen Q if the margin is thin.
        if n_vehicles * vehicle_capacity < total_demand * 1.15
            vehicle_capacity = round(total_demand * 1.15 / n_vehicles; digits=2)
        end
        # Re-assert per-customer fit (Q may have grown, never shrink below it).
        if vehicle_capacity < max_demand * 1.1
            vehicle_capacity = round(max_demand * 1.1; digits=2)
        end
        # Aggregate capacity (K*Q >= total_demand) is necessary but NOT sufficient
        # for the *integer* CVRP: the demands must also partition into K routes of
        # capacity Q. Raise Q until a first-fit-decreasing packing fits in <= K
        # bins, certifying a concrete integer routing exists (with N >= K this
        # extends to exactly K non-empty routes). Raising Q only loosens the LP
        # relaxation, so its feasibility is preserved too. Terminates because Q
        # grows monotonically and Q >= total_demand needs a single bin.
        while _ffd_bin_count(demands, vehicle_capacity) > n_vehicles
            vehicle_capacity = round(vehicle_capacity * 1.1; digits=2)
        end
        witness = CVRPWitness(
            _cvrp_routes_from_bins(_cvrp_ffd_bins(demands, vehicle_capacity), n_vehicles, dist)
        )

    elseif feasibility_status == infeasible
        # Aggregate fleet capacity strictly insufficient: total_demand > K*Q.
        overload = 1.1 + 0.2 * rand(rng)          # 1.10 .. 1.30
        k_out = floor(Int, total_demand / (vehicle_capacity * overload))
        if k_out >= 1
            # Vehicles out of service; Q only grows, so every demand still fits.
            n_vehicles = min(k_out, n_vehicles)
            vehicle_capacity = floor(total_demand / (n_vehicles * overload); digits=2)
        else
            # Tiny instance: inflate demands instead (some may exceed Q).
            scale = n_vehicles * vehicle_capacity * overload / total_demand
            demands = round.(demands .* scale; digits=2)
            total_demand = sum(demands)
        end
        certificate = CVRPFleetCapacityCertificate(total_demand, n_vehicles * vehicle_capacity)
    end
    # unknown: leave as sampled (biased feasible via the slack-based K sizing).

    return CVRPProblem(
        N,
        n_vehicles,
        vehicle_capacity,
        depot_location,
        customer_locations,
        demands,
        dist,
        witness,
        certificate,
    )
end

"""
    build_model(prob::CVRPProblem)

Build a JuMP model for the CVRP using the single-commodity flow (Gavish–Graves)
formulation. Deterministic — uses only data from the struct fields.

Node indexing: node `1` is the depot; nodes `2..N+1` are customers.

Decision variables (over all directed arcs `(i,j)`, `i ≠ j`):

  - `x[i,j] ∈ {0,1}`: arc `(i,j)` is traversed
  - `f[i,j] ≥ 0`: single-commodity load carried on arc `(i,j)`

# Returns

  - `model`: The JuMP model
"""
function build_model(prob::CVRPProblem)
    model = Model()

    N = prob.n_customers
    K = prob.n_vehicles
    Q = prob.vehicle_capacity
    n_nodes = N + 1                  # node 1 = depot, 2..N+1 = customers
    depot = 1
    customers = 2:n_nodes
    nodes = 1:n_nodes

    # demand indexed by node (depot has zero demand)
    dem(j) = prob.demands[j - 1]     # j in customers -> demand index j-1
    total_demand = sum(prob.demands)

    # --- Variables: one binary x and one continuous f per directed arc (no self-loops) ---
    # Count = 2 * (N+1) * N
    @variable(model, x[i in nodes, j in nodes; i != j], Bin)
    @variable(model, f[i in nodes, j in nodes; i != j] >= 0)

    # --- Objective: minimize total travel cost ---
    @objective(model, Min, sum(prob.dist[i, j] * x[i, j] for i in nodes, j in nodes if i != j))

    # --- Degree constraints for customers: exactly one in-arc and one out-arc ---
    for j in customers
        @constraint(model, sum(x[i, j] for i in nodes if i != j) == 1)   # in
        @constraint(model, sum(x[j, k] for k in nodes if k != j) == 1)   # out
    end

    # --- Depot degree: K routes leave and K return ---
    @constraint(model, sum(x[depot, j] for j in customers) == K)
    @constraint(model, sum(x[i, depot] for i in customers) == K)

    # --- Flow (load) conservation ---
    # At each customer: inbound load - outbound load = demand.
    for j in customers
        @constraint(
            model,
            sum(f[i, j] for i in nodes if i != j) - sum(f[j, k] for k in nodes if k != j) == dem(j)
        )
    end
    # At the depot: net outflow of load = total demand.
    @constraint(
        model,
        sum(f[depot, j] for j in customers) - sum(f[i, depot] for i in customers) == total_demand
    )

    # --- Capacity coupling (strengthened Gavish–Graves bounds) ---
    # Load on (i,j) is at most what remains after serving i, and at least j's
    # own demand. A demand above Q (tiny infeasible fallback) keeps the plain
    # Q bound so the upper coefficient never turns negative.
    node_demand(i) = i == depot ? 0.0 : dem(i)
    for i in nodes, j in nodes
        i == j && continue
        upper = node_demand(i) < Q ? Q - node_demand(i) : Q
        @constraint(model, f[i, j] <= upper * x[i, j])
        j == depot || @constraint(model, f[i, j] >= dem(j) * x[i, j])
    end

    return model
end

# Register the variant (lazily creates the :vehicle_routing category).
register_variant(
    :vehicle_routing,
    :cvrp,
    CVRPProblem,
    "Capacitated vehicle routing problem (CVRP) with single-commodity-flow subtour elimination; a MIP whose continuous relaxation is a genuine depot-anchored routing relaxation",
)
