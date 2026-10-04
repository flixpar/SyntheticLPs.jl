using JuMP
using Random
using Distributions

"""
    PMedianWitness

Planted feasible plan for [`PMedianFacilityLocationProblem`](@ref): the `p`
open facilities and the facility each customer is assigned to. Integral, so it
is feasible for the MIP and its relaxation.
"""
struct PMedianWitness
    open::Vector{Int}
    assignment::Vector{Int}
end

"""
    PMedianCapacityCertificate

LP-row infeasibility certificate for [`PMedianFacilityLocationProblem`](@ref).
Weighting each assignment row by `d_c` and summing gives
`total_demand = Σ_w Σ_c d_c y[w,c] ≤ Σ_w Q_w z_w` (capacity rows), and with
`Σ_w z_w = p`, `0 ≤ z ≤ 1` the right-hand side is at most the sum of the `p`
largest capacities, `top_p_capacity`. The generator keeps
`top_p_capacity ≤ total_demand / 1.05`.
"""
struct PMedianCapacityCertificate
    top_p_capacity::Float64
    total_demand::Float64
end

"""
    PMedianFacilityLocationProblem <: ProblemGenerator

Capacitated p-median (CPMP): open exactly `p` facilities and assign every
customer to one open facility, minimizing demand-weighted distance, subject to
demand-weighted facility capacities (the Osman–Christofides / Lorena–Senne
benchmark family).

# Formulation

  - `z[w] ∈ {0,1}` opens facility `w`; `y[w,c] ∈ {0,1}` assigns customer `c`;
  - assignment `Σ_w y[w,c] = 1`;
  - strong linking `y[w,c] ≤ z[w]` (disaggregated; keeps the relaxation tight);
  - cardinality `Σ_w z[w] = p`;
  - capacity `Σ_c d_c y[w,c] ≤ Q_w z[w]` with heterogeneous site capacities.

Distinct from `standard` (fixed costs + budget + continuous shipments) and
`two_echelon` (sparse two-level flows with discrete sizing).

# Fields

  - `n_facilities::Int`, `n_customers::Int`, `p::Int`
  - `capacities::Vector{Float64}`: demand capacity `Q_w` of each site
  - `facility_locs`, `customer_locs`: coordinates
  - `demands::Vector{Float64}`
  - `distances::Matrix{Float64}`: `F × C` Euclidean distances
  - `feasible_witness::Union{Nothing,PMedianWitness}`
  - `infeasibility_certificate::Union{Nothing,PMedianCapacityCertificate}`
"""
struct PMedianFacilityLocationProblem <: ProblemGenerator
    n_facilities::Int
    n_customers::Int
    p::Int
    capacities::Vector{Float64}
    facility_locs::Vector{Tuple{Float64, Float64}}
    customer_locs::Vector{Tuple{Float64, Float64}}
    demands::Vector{Float64}
    distances::Matrix{Float64}
    feasible_witness::Union{Nothing, PMedianWitness}
    infeasibility_certificate::Union{Nothing, PMedianCapacityCertificate}
end

"""
    PMedianFacilityLocationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a capacitated p-median instance.

# Variable-count formula

    total = F · (C + 1)

(`F` opening plus `F·C` assignment variables), with a sampled customer/site
ratio `r ∈ 1..3` (CPMP benchmarks use about as many sites as customers), `F = max(2, round(sqrt(target / r)))` and
`C = max(2, round(target / F) - 1)`. `p ∈ [F/10, F/4]` (at least 1, at most `C - 1`).
No size cap.

# Feasibility

Capacities are `Q_w = (total_demand / p) · ρ · U(0.8, 1.25)` with a sampled
tightness `ρ ∈ [0.8, 1.3]`.

  - `feasible`: `p` facilities are planted (greedy weighted p-median seeds),
    every customer is assigned to its nearest planted site, and a planted site
    whose load exceeds its drawn capacity is expanded to 1.02–1.12× that load.
    Capacities therefore stay tight (many bind in the LP). Stored as a
    [`PMedianWitness`](@ref).
  - `infeasible`: capacities are scaled so the `p` largest sum to
    `total_demand / (1.05..1.25)` ([`PMedianCapacityCertificate`](@ref)). The
    proof aggregates every assignment and capacity row with the cardinality
    row, so presolve does not see it.
  - `unknown`: `ρ ∈ [0.8, 1.3]` as drawn. The LP is feasible whenever the `p`
    largest capacities cover demand (usually, not always); the integer model
    additionally faces a bin-packing question, so the instance is genuinely
    undetermined.
"""
function PMedianFacilityLocationProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)

    ratio = rand(rng, 1:3)
    F = max(2, round(Int, sqrt(target_variables / ratio)))
    C = max(2, round(Int, target_variables / F) - 1)
    p_lo = clamp(fld(F, 10), 1, F)
    p_hi = clamp(fld(F, 4), max(p_lo, 2), F)
    p = rand(rng, p_lo:p_hi)
    p = min(p, C - 1, F)
    p = max(p, 1)

    span = rand(rng, 500.0:100.0:3000.0)
    min_demand, max_demand = rand(rng, 5.0:5.0:30.0), rand(rng, 80.0:20.0:300.0)
    facility_locs = [(span * rand(rng), span * rand(rng)) for _ in 1:F]
    n_clusters = max(2, div(C, 15))
    centers = [(span * rand(rng), span * rand(rng)) for _ in 1:n_clusters]
    customer_locs = _fl_clustered_points(rng, C, centers, span / 10, span; rural_fraction=0.1)

    log_mean = log(sqrt(min_demand * max_demand))
    log_std = log(max_demand / min_demand) / 4
    demands = [
        round(clamp(exp(rand(rng, Normal(log_mean, log_std))), min_demand, max_demand); digits=2)
        for _ in 1:C
    ]
    total_demand = sum(demands)

    distances = Matrix{Float64}(undef, F, C)
    for c in 1:C, w in 1:F
        distances[w, c] = round(_fl_dist(facility_locs[w], customer_locs[c]); digits=3)
    end

    # Capacity tightness: with F ≈ 4–10p sites drawn at U(0.8, 1.25) × total/p,
    # the p largest sum to ≈ 1.2ρ × demand, so ρ ≈ 0.83 is the LP threshold.
    rho = rand(rng, Uniform(0.8, 1.3))
    capacities = [
        round(total_demand / p * rho * rand(rng, Uniform(0.8, 1.25)); digits=2) for _ in 1:F
    ]

    witness = nothing
    certificate = nothing
    if feasibility_status == feasible
        # Greedy weighted p-median seeds: each step opens the site that most
        # reduces demand-weighted distance to the nearest open site.
        best = fill(Inf, C)
        open = Int[]
        for _ in 1:p
            gain_best, pick = -Inf, 0
            for w in 1:F
                w in open && continue
                gain = 0.0
                for c in 1:C
                    gain += demands[c] * (min(best[c], 4span) - min(best[c], distances[w, c]))
                end
                gain > gain_best && ((gain_best, pick) = (gain, w))
            end
            push!(open, pick)
            for c in 1:C
                best[c] = min(best[c], distances[pick, c])
            end
        end
        # Every customer goes to its nearest planted site; a planted site whose
        # load exceeds its drawn capacity is expanded to just cover it (2-12%
        # headroom), so capacities stay tight and binding in the LP.
        assignment = [open[argmin([distances[w, c] for w in open])] for c in 1:C]
        load = zeros(F)
        for c in 1:C
            load[assignment[c]] += demands[c]
        end
        for w in open
            need = load[w] * rand(rng, Uniform(1.02, 1.12))
            capacities[w] < need && (capacities[w] = ceil(need; digits=2))
        end
        witness = PMedianWitness(sort!(open), assignment)
    elseif feasibility_status == infeasible
        top = sum(partialsort(capacities, 1:p; rev=true))
        shrink = total_demand / rand(rng, Uniform(1.05, 1.25)) / top
        capacities .= floor.(capacities .* shrink; digits=2)
        top = sum(partialsort(capacities, 1:p; rev=true))
        certificate = PMedianCapacityCertificate(top, total_demand)
    end

    return PMedianFacilityLocationProblem(
        F,
        C,
        p,
        capacities,
        facility_locs,
        customer_locs,
        demands,
        distances,
        witness,
        certificate,
    )
end

"""
    build_model(prob::PMedianFacilityLocationProblem)

Build the capacitated p-median model. Deterministic — uses only data from the
struct fields.
"""
function build_model(prob::PMedianFacilityLocationProblem)
    model = Model()
    F, C = prob.n_facilities, prob.n_customers

    @variable(model, z[1:F], Bin)
    @variable(model, y[1:F, 1:C], Bin)
    @objective(
        model, Min, sum(prob.distances[w, c] * prob.demands[c] * y[w, c] for w in 1:F, c in 1:C)
    )
    for c in 1:C
        @constraint(model, sum(y[w, c] for w in 1:F) == 1)
    end
    for c in 1:C, w in 1:F
        @constraint(model, y[w, c] <= z[w])
    end
    @constraint(model, sum(z) == prob.p)
    for w in 1:F
        @constraint(model, sum(prob.demands[c] * y[w, c] for c in 1:C) <= prob.capacities[w] * z[w])
    end
    return model
end

register_variant(
    :facility_location,
    :p_median,
    PMedianFacilityLocationProblem,
    "Capacitated p-median: open exactly p sites and assign every customer to one, minimizing demand-weighted distance under demand-weighted site capacities";
    tags=[:location, :bipartite, :partitioning, :big_m],
)
