using JuMP
using Random
using Distributions

"""
    PMedianFacilityLocationProblem <: ProblemGenerator

Generator for classic *p-median* facility location problems.

# Overview
Models the textbook p-median problem: from a set of candidate facility sites,
open **exactly `p`** of them and assign every customer to one open facility so
as to minimize the total demand-weighted travel distance. There are no fixed
opening costs, no budget, and no continuous shipping flows — this is the pure
median objective, which makes it structurally distinct from the capacitated
`standard` variant (fixed costs + budget + continuous shipments) and from the
`two_echelon` variant (supplier→warehouse→customer flows with discrete sizing).

A *service capacity in customer count* (`count_cap`) caps how many customers any
single facility may serve. This count-based capacity is what enables a clean,
relaxation-aware infeasibility mode: if the `p` open facilities cannot
collectively absorb all `C` customers (`p * count_cap < C`), the assignment
constraints are unsatisfiable even in the LP relaxation (pigeonhole on the
aggregate assignment count).

The model is a proper MIP (`z`, `y` binary); its LP relaxation is the standard
p-median LP relaxation, with the disaggregated linking `y[w,c] <= z[w]` that
keeps the relaxation tight (as opposed to the loose aggregate cover form, which
degenerates).

# Fields
- `n_facilities::Int`: Number of candidate facility sites
- `n_customers::Int`: Number of customers
- `p::Int`: Number of facilities to open (exactly)
- `count_cap::Int`: Maximum number of customers a single facility may serve
- `facility_locs::Vector{Tuple{Float64,Float64}}`: Facility coordinates
- `customer_locs::Vector{Tuple{Float64,Float64}}`: Customer coordinates
- `demands::Vector{Float64}`: Customer demand (assignment weight)
- `distances::Matrix{Float64}`: Euclidean distance facility→customer (F×C)
"""
struct PMedianFacilityLocationProblem <: ProblemGenerator
    n_facilities::Int
    n_customers::Int
    p::Int
    count_cap::Int
    facility_locs::Vector{Tuple{Float64,Float64}}
    customer_locs::Vector{Tuple{Float64,Float64}}
    demands::Vector{Float64}
    distances::Matrix{Float64}
end

"""
    PMedianFacilityLocationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a p-median facility location instance.

# Arguments
- `target_variables`: Target number of decision variables
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility

# Variable count
The model has assignment variables `y[w,c]` (F×C) plus opening variables
`z[w]` (F), so:

    total_variables = n_facilities * n_customers + n_facilities
                    = n_facilities * (n_customers + 1)

Dimensions `n_facilities` and `n_customers` are sized in the constructor to hit
`target_variables`. The number of facilities to open is `p ∈ [2, max(2, F/3)]`.

# Feasibility (relaxation-aware)
- `feasible`/`unknown`: choose `count_cap` generously so that
  `p * count_cap >= C` with a margin (the `p` open facilities can collectively
  serve every customer). Assigning each customer to its nearest open facility is
  then admissible; this point is also feasible for the LP relaxation. `unknown`
  uses the same generous sizing (biased feasible without forcing).
- `infeasible`: set `count_cap = floor(C * f / p)` with `f ∈ [0.6, 0.9]`, so
  `p * count_cap < C`. At most `p` facilities open, each serving at most
  `count_cap` customers, so at most `p * count_cap < C` customers can be
  assigned — yet every customer must be assigned exactly once. The aggregate
  assignment count `sum_{w,c} y[w,c] = C` cannot be covered by the aggregate
  service capacity `count_cap * sum_w z[w] = count_cap * p < C`, so the
  relaxation is infeasible.
"""
function PMedianFacilityLocationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Scale-tiered ranges
    if target_variables <= 100
        min_facilities, max_facilities = 2, 20
        min_customers, max_customers = 2, 40
        grid_width = rand(200.0:50.0:800.0)
        grid_height = rand(200.0:50.0:800.0)
        min_demand, max_demand = rand(5.0:1.0:20.0), rand(50.0:10.0:150.0)
    elseif target_variables <= 1000
        min_facilities, max_facilities = 3, 100
        min_customers, max_customers = 5, 200
        grid_width = rand(500.0:100.0:2000.0)
        grid_height = rand(500.0:100.0:2000.0)
        min_demand, max_demand = rand(10.0:2.0:30.0), rand(80.0:20.0:200.0)
    else
        min_facilities, max_facilities = 5, 500
        min_customers, max_customers = 10, 2000
        grid_width = rand(1000.0:200.0:5000.0)
        grid_height = rand(1000.0:200.0:5000.0)
        min_demand, max_demand = rand(20.0:5.0:60.0), rand(150.0:50.0:500.0)
    end

    # Size n_facilities, n_customers so F*(C+1) ~ target_variables.
    best_n_facilities = min_facilities
    best_n_customers = min_customers
    best_error = Inf

    for n_facilities in min_facilities:max_facilities
        n_customers_exact = (target_variables / n_facilities) - 1
        if n_customers_exact >= min_customers && n_customers_exact <= max_customers
            n_customers = round(Int, n_customers_exact)
            actual_vars = n_facilities * (n_customers + 1)
            error = abs(actual_vars - target_variables) / target_variables
            if error < best_error
                best_error = error
                best_n_facilities = n_facilities
                best_n_customers = n_customers
            end
        end
    end

    if best_error > 0.1
        n_facilities_approx = max(min_facilities, min(max_facilities, round(Int, sqrt(target_variables / 4))))
        n_customers_approx = max(min_customers, min(max_customers, round(Int, (target_variables / n_facilities_approx) - 1)))
        best_n_facilities = n_facilities_approx
        best_n_customers = n_customers_approx
    end

    n_facilities = best_n_facilities
    n_customers = best_n_customers

    # Number of facilities to open: p in [2, max(2, F/3)], capped below F.
    p_hi = max(2, fld(n_facilities, 3))
    p_hi = min(p_hi, n_facilities)
    p_lo = min(2, p_hi)
    p = p_lo == p_hi ? p_lo : rand(p_lo:p_hi)

    # For the infeasible mode the pigeonhole needs p < C (so that even
    # count_cap=1 yields p*count_cap < C). Shrink p if necessary — allow p down to
    # 1 (the 1-median), so the shrink still works when C is as small as 2. Flooring
    # at p_lo (=2) would leave p == C == 2, and the later count_cap >= 1 clamp would
    # then permit a fully feasible assignment despite an `infeasible` request.
    if feasibility_status == infeasible && p >= n_customers
        p = max(1, n_customers - 1)
    end

    # Facility candidate sites: uniform over the grid.
    facility_locs = [(grid_width * rand(), grid_height * rand()) for _ in 1:n_facilities]

    # Customers clustered into "cities".
    n_clusters = max(2, div(n_customers, 15))
    cluster_centers = [(grid_width * rand(), grid_height * rand()) for _ in 1:n_clusters]
    customer_locs = Tuple{Float64,Float64}[]
    for _ in 1:n_customers
        center = rand(cluster_centers)
        x = clamp(center[1] + randn() * (grid_width / 10), 0.0, grid_width)
        y = clamp(center[2] + randn() * (grid_height / 10), 0.0, grid_height)
        push!(customer_locs, (x, y))
    end

    # Log-normal demands.
    log_mean = log(sqrt(min_demand * max_demand))
    log_std = log(max_demand / min_demand) / 4
    demands = [clamp(exp(rand(Normal(log_mean, log_std))), min_demand, max_demand) for _ in 1:n_customers]
    demands = round.(demands, digits=2)

    # Euclidean distances facility→customer.
    distances = zeros(n_facilities, n_customers)
    for w in 1:n_facilities, c in 1:n_customers
        distances[w, c] = sqrt(
            (facility_locs[w][1] - customer_locs[c][1])^2 +
            (facility_locs[w][2] - customer_locs[c][2])^2
        )
    end

    # --- Feasibility handling via the count-based service capacity ---
    actual_status = feasibility_status == infeasible ? infeasible : feasible

    if actual_status == feasible
        # Generous capacity: the p open facilities can collectively serve all C
        # customers with a margin. count_cap >= ceil(C / p) * slack, capped at C.
        base = ceil(Int, n_customers / p)
        slack = rand(1.2:0.1:1.8)
        count_cap = min(n_customers, max(base, ceil(Int, base * slack)))
        # Guarantee p * count_cap >= C with margin (defensive).
        while p * count_cap < n_customers
            count_cap += 1
        end
    else
        # Pigeonhole infeasibility: p * count_cap < C.
        f = rand(0.6:0.05:0.9)
        count_cap = max(1, fld(round(Int, n_customers * f), p))
        # Defensive: ensure strict shortfall p * count_cap < C.
        while p * count_cap >= n_customers && count_cap > 1
            count_cap -= 1
        end
        if p * count_cap >= n_customers
            count_cap = max(1, fld(n_customers - 1, p))
        end
    end

    return PMedianFacilityLocationProblem(
        n_facilities,
        n_customers,
        p,
        count_cap,
        facility_locs,
        customer_locs,
        demands,
        distances,
    )
end

"""
    build_model(prob::PMedianFacilityLocationProblem)

Build a JuMP model for the p-median facility location problem. Deterministic —
uses only data from the struct fields.

# Returns
- `model`: The JuMP model
"""
function build_model(prob::PMedianFacilityLocationProblem)
    model = Model()

    F = prob.n_facilities
    C = prob.n_customers

    # Decision variables
    @variable(model, z[1:F], Bin)            # open facility w
    @variable(model, y[1:F, 1:C], Bin)       # assign customer c to facility w

    # Objective: minimize total demand-weighted travel distance.
    @objective(model, Min,
        sum(prob.distances[w, c] * prob.demands[c] * y[w, c] for w in 1:F, c in 1:C)
    )

    # Each customer assigned to exactly one facility.
    for c in 1:C
        @constraint(model, sum(y[w, c] for w in 1:F) == 1)
    end

    # Disaggregated (tight) linking: can only assign to an open facility.
    for w in 1:F, c in 1:C
        @constraint(model, y[w, c] <= z[w])
    end

    # Open exactly p facilities.
    @constraint(model, sum(z[w] for w in 1:F) == prob.p)

    # Service capacity in customer count, active only at open facilities.
    for w in 1:F
        @constraint(model, sum(y[w, c] for c in 1:C) <= prob.count_cap * z[w])
    end

    return model
end

# Register the variant
register_variant(
    :facility_location,
    :p_median,
    PMedianFacilityLocationProblem,
    "Classic p-median: open exactly p facilities and assign every customer to minimize demand-weighted distance, with count-based service capacity",
)
