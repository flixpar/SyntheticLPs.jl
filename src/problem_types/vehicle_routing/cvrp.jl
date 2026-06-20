using JuMP
using Random
using Distributions

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
- **Capacity coupling** `f[i,j] ≤ Q · x[i,j]` simultaneously (a) forbids load on
  unused arcs and (b) limits the load on any depot-leaving arc to `Q`, which is
  the per-route capacity bound.

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
"""
struct CVRPProblem <: ProblemGenerator
    n_customers::Int
    n_vehicles::Int
    vehicle_capacity::Float64
    depot_location::Tuple{Float64,Float64}
    customer_locations::Vector{Tuple{Float64,Float64}}
    demands::Vector{Float64}
    dist::Matrix{Float64}
end

"""
    CVRPProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a Capacitated Vehicle Routing Problem instance.

# Variable-count formula
On a complete directed graph over `N+1` nodes (depot + `N` customers) with no
self-loops there are `(N+1)*N` arcs. The model creates one binary `x` and one
continuous `f` per arc:

    total = 2 * (N + 1) * N

So `N ≈ round(sqrt(target_variables / 2))` (clamped to `N ≥ 3`). For
`target = 100` this gives `N = 7` (112 vars); for `target = 500`, `N = 16`
(544 vars).

# Arguments
- `target_variables`: Target number of decision variables across both arc blocks
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility

# Feasibility (must hold for the LP relaxation)
- `feasible`: `K*Q ≥ total_demand` with margin (≈ 1.15×), `K ≤ N`, and every
  single demand `≤ Q`. A feasible routing then exists and the relaxation is feasible.
- `infeasible`: keep the structure but inflate demands so aggregate fleet
  capacity is strictly insufficient: `total_demand = K*Q * (1.1..1.3)`. Since the
  depot net-outflow `Σ_j f[depot,j]` must equal `total_demand` yet is bounded by
  `Q * Σ_j x[depot,j] = Q*K` in the relaxation, `total_demand > Q*K` is infeasible
  even relaxed.
- `unknown`: a natural instance, biased toward feasible but not forced.
"""
function CVRPProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # --- Dimension sizing ---
    # total = 2 * (N + 1) * N  ≈ 2 * N^2  for large N  =>  N ≈ sqrt(target / 2).
    N = max(3, round(Int, sqrt(target_variables / 2)))

    # --- Scale-tiered parameter ranges ---
    total_vars = 2 * (N + 1) * N
    if total_vars <= 250
        grid_size = rand(40.0:5.0:120.0)
        demand_lo, demand_hi = 5.0, 45.0
        cost_per_km = rand(0.8:0.1:1.8)
        n_clusters = rand(2:3)
    elseif total_vars <= 1000
        grid_size = rand(100.0:20.0:300.0)
        demand_lo, demand_hi = 10.0, 90.0
        cost_per_km = rand(1.0:0.1:2.5)
        n_clusters = rand(3:5)
    else
        grid_size = rand(250.0:50.0:700.0)
        demand_lo, demand_hi = 20.0, 200.0
        cost_per_km = rand(1.5:0.2:3.5)
        n_clusters = rand(5:8)
    end

    # --- Depot near grid center ---
    depot_location = (grid_size * (0.4 + 0.2 * rand()), grid_size * (0.4 + 0.2 * rand()))

    # --- Customers clustered into a few neighborhoods ---
    cluster_centers = [(grid_size * rand(), grid_size * rand()) for _ in 1:n_clusters]
    cluster_spread = min(grid_size, grid_size) / (2.5 * n_clusters)
    customer_locations = Tuple{Float64,Float64}[]
    for _ in 1:N
        center = rand(cluster_centers)
        x = clamp(center[1] + randn() * cluster_spread, 0.0, grid_size)
        y = clamp(center[2] + randn() * cluster_spread, 0.0, grid_size)
        push!(customer_locations, (x, y))
    end

    # --- Log-normal demands (few large shipments, many small) ---
    log_mean = log(sqrt(demand_lo * demand_hi))
    log_std = log(demand_hi / demand_lo) / 4
    demands = [clamp(exp(rand(Normal(log_mean, log_std))), demand_lo, demand_hi) for _ in 1:N]
    demands = round.(demands, digits=2)

    total_demand = sum(demands)
    avg_demand = total_demand / N
    max_demand = maximum(demands)

    # --- Vehicle capacity: a vehicle serves ~3-6 customers on average ---
    serve_count = 3.0 + 3.0 * rand()           # 3..6 customers per vehicle
    vehicle_capacity = avg_demand * serve_count
    # Every single customer must fit in a vehicle (for feasible/unknown).
    vehicle_capacity = max(vehicle_capacity, max_demand * 1.1)
    vehicle_capacity = round(vehicle_capacity, digits=2)

    # --- Fleet size: enough vehicles to cover demand, clamped to N ---
    slack = 1.15 + 0.15 * rand()               # 1.15 .. 1.30
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
            dist[i, j] = round(d * cost_per_km * (0.95 + 0.1 * rand()), digits=2)
        end
    end

    # --- Resolve feasibility intent ---
    if feasibility_status == feasible
        # Guarantee K*Q >= total_demand with margin, K <= N, max demand <= Q.
        # K is already <= N and sized from demand; widen Q if the margin is thin.
        if n_vehicles * vehicle_capacity < total_demand * 1.15
            vehicle_capacity = round(total_demand * 1.15 / n_vehicles, digits=2)
        end
        # Re-assert per-customer fit (Q may have grown, never shrink below it).
        if vehicle_capacity < max_demand * 1.1
            vehicle_capacity = round(max_demand * 1.1, digits=2)
        end

    elseif feasibility_status == infeasible
        # Aggregate fleet capacity strictly insufficient: total_demand > K*Q.
        # Inflate demands so total_demand = K*Q * (1.1..1.3). This survives the
        # LP relaxation because depot net-outflow = total_demand but is bounded
        # by Q * K. (Some individual demands may exceed Q, which only reinforces
        # infeasibility.)
        overload = 1.1 + 0.2 * rand()          # 1.10 .. 1.30
        target_total = n_vehicles * vehicle_capacity * overload
        scale = target_total / total_demand
        demands = round.(demands .* scale, digits=2)
        total_demand = sum(demands)
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
    @objective(model, Min,
        sum(prob.dist[i, j] * x[i, j] for i in nodes, j in nodes if i != j))

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
        @constraint(model,
            sum(f[i, j] for i in nodes if i != j) -
            sum(f[j, k] for k in nodes if k != j) == dem(j))
    end
    # At the depot: net outflow of load = total demand.
    @constraint(model,
        sum(f[depot, j] for j in customers) -
        sum(f[i, depot] for i in customers) == total_demand)

    # --- Capacity coupling: load only on used arcs, and bounded by Q ---
    for i in nodes, j in nodes
        i == j && continue
        @constraint(model, f[i, j] <= Q * x[i, j])
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
