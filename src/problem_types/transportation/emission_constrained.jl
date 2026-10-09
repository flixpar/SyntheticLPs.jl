using JuMP
using Random
using Distributions

const _EMISSION_MODES = (:truck, :rail, :intermodal)

"""
Planted multimodal plan: per-option flows (`options[k] = (lane, mode)`) of an
exact max-flow transportation plan whose every lane is shipped entirely by its
lowest-emission available mode. Supplies, demands, rail-terminal capacities
and every emission cap hold by construction.
"""
struct EmissionTransportationWitness
    flows::Vector{Float64}
end

"""
Emission lower-bound certificate. Every unit customer `j` receives travels on
some option into `j`, which emits at least `min_rate[j]` per unit, so the
demand rows imply

    total emissions >= sum_j demand[j] * min_rate[j] = lower_bound

for every feasible plan (relaxation-proof: no integrality anywhere). The
global cap is set strictly below it (`global_cap < lower_bound` by 5%-15%): a
decarbonisation target no mode choice can reach. The emission row alone is
satisfiable (zero shipments emit nothing) — only its combination with the
demand rows refutes the model, so presolve cannot see it.
"""
struct EmissionTransportationCertificate
    min_rate::Vector{Float64}
    lower_bound::Float64
    global_cap::Float64
end

"""
    EmissionConstrainedTransportationProblem <: ProblemGenerator

Multimodal transportation under regional and corporate CO2 caps, on sparse
geographic lane networks.

# Overview

Each lane `(i, j)` (customers' nearest sources plus long-haul lanes, as in
`transportation/standard`) can be served by truck (always), rail (when the
source has a rail siding and the lane is long enough) and intermodal (longer
lanes). Decision `x[k] >= 0` per (lane, mode) option.

    minimize    sum_k cost[k] x[k]
    subject to  sum_{k from i} x[k] <= supply[i]                  every source i
                sum_{k to j}   x[k] >= demand[j]                  every customer j
                sum_{rail k from i} x[k] <= rail_capacity[i]      sources with a siding
                sum_{k from region r} emission[k] x[k] <= region_cap[r]   every region r
                sum_k emission[k] x[k] <= global_cap

The dense emission rows (one per sales region of sources, plus the corporate
cap) break total unimodularity and couple the whole network, so the cheap
truck-heavy routing conflicts with the caps.

# Data grounding

Per unit of freight: truck costs `rate * d` (+2 handling) and emits
`0.062 * d` (kg CO2 per t-km, with lognormal vehicle noise); rail costs
`0.45 * rate * d` (+8 terminal fees) and emits `0.022 * d + 0.3` (drayage);
intermodal costs `0.6 * rate * d` (+5) and emits `0.035 * d + 0.6`. Sources
are grouped into 2-12 sales regions (nearest of random region seats).
Rail-terminal capacities are sized above the planted rail volume. Supplies are
market-sized with regional under-build shocks.

# Feasibility control

Demands are `load_factor * lambda*` times the nominal profile, where `lambda*`
is the exact largest deliverable scale of the lane network (Dinkelbach
iterations over exact max flows), so the shipping part is feasible for every
profile; the caps decide. The planted plan (max-flow shipments, lowest-emission
mode per lane) sets the cap levels:

  - `feasible`: `load_factor` in [0.6, 0.9]; region caps 1.03-1.25x and the
    global cap 1.0-1.08x the plan's emissions; the plan is the witness.
  - `infeasible`: as `feasible`, but the global cap is 85%-95% of the
    demand-weighted minimum emission rate bound (certificate above).
  - `unknown`: `load_factor` in [0.7, 0.95]; region caps 0.9-1.2x the
    plan's regional emissions, and the global cap drawn uniformly across the
    lower 70% of the gap between the demand-weighted rate bound and the
    plan's emissions — the true minimum lies somewhere in that gap, so the
    target may or may not be reachable: natural, either side.

# Fields

  - `n_sources`, `n_customers`, `lanes`, `options::Vector{Tuple{Int,Int}}`
    (`(lane, mode)` with mode 1 = truck, 2 = rail, 3 = intermodal), `cost`,
    `emission` (per option)
  - `supplies`, `demands`, `rail_capacity` (per source; 0 without a siding)
  - `region_of_source`, `region_cap`, `global_cap`
  - positions, `geography`, `load_factor`
  - `feasible_witness`, `infeasibility_certificate`, `feasibility_status`
"""
struct EmissionConstrainedTransportationProblem <: ProblemGenerator
    n_sources::Int
    n_customers::Int
    lanes::Vector{Tuple{Int, Int}}
    options::Vector{Tuple{Int, Int}}
    cost::Vector{Float64}
    emission::Vector{Float64}
    supplies::Vector{Float64}
    demands::Vector{Float64}
    rail_capacity::Vector{Float64}
    region_of_source::Vector{Int}
    region_cap::Vector{Float64}
    global_cap::Float64
    source_positions::Vector{Tuple{Float64, Float64}}
    customer_positions::Vector{Tuple{Float64, Float64}}
    geography::Symbol
    load_factor::Float64
    feasible_witness::Union{Nothing, EmissionTransportationWitness}
    infeasibility_certificate::Union{Nothing, EmissionTransportationCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    EmissionConstrainedTransportationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Variables are the (lane, mode) options, exactly `max(target_variables, 2)`:
about `target / 1.7` lanes, with optional rail/intermodal options added or
removed at random until the count matches. Rows:
`n_sources + n_customers + (#sources with a rail option) + n_regions + 1`.
"""
function EmissionConstrainedTransportationProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    _tp_check_target(target_variables, "emission_constrained")
    rng = MersenneTwister(seed)
    target = max(target_variables, 2)
    n_lanes = max(2, round(Int, target / 1.7))

    mean_lanes = 3.0 + 3.0 * rand(rng)
    nS, nD = _tp_dimensions(rng, n_lanes, mean_lanes, 5.0 + 7.0 * rand(rng))
    src_pos, dst_pos, src_w, dst_w, geography = _tp_geography(rng, nS, nD)
    lanes, primary = _tp_lanes(rng, src_pos, dst_pos, n_lanes; mean_lanes=mean_lanes, weights=dst_w)
    L = length(lanes)
    dist = [hypot(src_pos[i][1] - dst_pos[j][1], src_pos[i][2] - dst_pos[j][2]) for (i, j) in lanes]
    med = sort(dist)[cld(L, 2)]

    # Mode availability, then exact sizing by adding/removing optional modes.
    siding = [rand(rng) < 0.5 for _ in 1:nS]
    avail = [falses(3) for _ in 1:L]
    for (l, (i, _)) in enumerate(lanes)
        avail[l][1] = true
        avail[l][2] = siding[i] && dist[l] > 0.5 * med
        avail[l][3] = dist[l] > 0.8 * med && rand(rng) < 0.6
    end
    total = sum(count, avail)
    while total > target
        l = rand(rng, 1:L)
        m = rand(rng, 2:3)
        avail[l][m] || continue
        avail[l][m] = false
        total -= 1
    end
    while total < target
        l = rand(rng, 1:L)
        m = siding[lanes[l][1]] ? rand(rng, 2:3) : 3
        avail[l][m] && continue
        avail[l][m] = true
        total += 1
    end
    options = [(l, m) for l in 1:L for m in 1:3 if avail[l][m]]
    K = length(options)

    d0 = 100.0 .* dst_w ./ (sum(dst_w) / nD)
    supplies = _tp_market_supply(rng, src_pos, lanes, primary, d0)

    rate = 0.8 + 0.8 * rand(rng)
    cost = Vector{Float64}(undef, K)
    emission = Vector{Float64}(undef, K)
    for (k, (l, m)) in enumerate(options)
        d = dist[l]
        if m == 1
            cost[k] = rate * d * rand(rng, LogNormal(0.0, 0.1)) + 2.0
            emission[k] = 0.062 * d * rand(rng, LogNormal(0.0, 0.15)) + 0.05
        elseif m == 2
            cost[k] = 0.45 * rate * d * rand(rng, LogNormal(0.0, 0.1)) + 8.0
            emission[k] = 0.022 * d * rand(rng, LogNormal(0.0, 0.1)) + 0.3
        else
            cost[k] = 0.6 * rate * d * rand(rng, LogNormal(0.0, 0.1)) + 5.0
            emission[k] = 0.035 * d * rand(rng, LogNormal(0.0, 0.1)) + 0.6
        end
        cost[k] = round(cost[k]; digits=3)
        emission[k] = round(emission[k]; digits=4)
    end

    # Exact transport boundary (uncapacitated lanes; supplies bind).
    big = 4.0 * (sum(supplies) + 2.0 * sum(d0))
    arcs = [(i, nS + j) for (i, j) in lanes]
    caps = fill(big, L)
    src_nodes = collect(1:nS)
    dst_nodes = collect((nS + 1):(nS + nD))
    lambda_star = _network_flow_max_scale(nS + nD, arcs, caps, src_nodes, supplies, dst_nodes, d0)
    load_factor = feasibility_status == unknown ? 0.7 + 0.25 * rand(rng) : 0.6 + 0.3 * rand(rng)
    demands = max.(round.(load_factor * lambda_star .* d0; digits=2), 0.01)
    value, ext_flows, _, _, _ = _network_flow_extended(
        nS + nD, arcs, caps, src_nodes, supplies, dst_nodes, demands
    )
    value >= sum(demands) * (1 - 1e-9) ||
        error("transportation/emission_constrained: planted load not deliverable (seed $seed)")

    # Planted plan: each lane's flow on its lowest-emission mode.
    opts_of_lane = [Int[] for _ in 1:L]
    for (k, (l, _)) in enumerate(options)
        push!(opts_of_lane[l], k)
    end
    plan = zeros(K)
    for l in 1:L
        ext_flows[l] > 0 || continue
        plan[argmin(k -> (emission[k], k), opts_of_lane[l])] = ext_flows[l]
    end

    # Rail terminal capacities (sources with a siding).
    rail_volume = zeros(nS)
    for (k, (l, m)) in enumerate(options)
        m == 2 && (rail_volume[lanes[l][1]] += plan[k])
    end
    rail_capacity = [
        if siding[i]
            round(rail_volume[i] * (1.1 + 0.4 * rand(rng)) + 0.05 * supplies[i]; digits=2)
        else
            0.0
        end for i in 1:nS
    ]

    # Sales regions: nearest of n_regions random source seats.
    n_regions = clamp(round(Int, nS / 25), min(2, nS), 12)
    seats = randperm(rng, nS)[1:n_regions]
    region_of_source = [
        argmin(
            r -> (
                hypot(src_pos[i][1] - src_pos[seats[r]][1], src_pos[i][2] - src_pos[seats[r]][2]),
                r,
            ),
            1:n_regions,
        ) for i in 1:nS
    ]
    planted_region = zeros(n_regions)
    for (k, (l, _)) in enumerate(options)
        planted_region[region_of_source[lanes[l][1]]] += emission[k] * plan[k]
    end
    planted_total = sum(planted_region)

    feasible_witness = nothing
    infeasibility_certificate = nothing
    in_opts = [Int[] for _ in 1:nD]
    for (k, (l, _)) in enumerate(options)
        push!(in_opts[lanes[l][2]], k)
    end
    min_rate = [minimum(emission[k] for k in in_opts[j]) for j in 1:nD]
    lower_bound = sum(demands[j] * min_rate[j] for j in 1:nD)
    if feasibility_status == unknown
        # The true minimum-emission level lies between the demand-weighted
        # rate bound and the planted plan; the corporate target is drawn
        # across that whole gap.
        region_cap = [ceil(e * (0.9 + 0.3 * rand(rng)); digits=2) for e in planted_region]
        global_cap = round(lower_bound + rand(rng) * 0.7 * (planted_total - lower_bound); digits=2)
    else
        region_cap = [ceil(e * (1.03 + 0.22 * rand(rng)) + 1e-6; digits=2) for e in planted_region]
        global_cap = ceil(planted_total * (1.0 + 0.08 * rand(rng)) + 1e-6; digits=2)
        if feasibility_status == feasible
            feasible_witness = EmissionTransportationWitness(plan)
        else
            global_cap = floor(lower_bound * (0.85 + 0.1 * rand(rng)); digits=2)
            infeasibility_certificate = EmissionTransportationCertificate(
                min_rate, lower_bound, global_cap
            )
        end
    end

    return EmissionConstrainedTransportationProblem(
        nS,
        nD,
        lanes,
        options,
        cost,
        emission,
        supplies,
        demands,
        rail_capacity,
        region_of_source,
        region_cap,
        global_cap,
        src_pos,
        dst_pos,
        geography,
        load_factor,
        feasible_witness,
        infeasibility_certificate,
        feasibility_status,
    )
end

"""
    build_model(prob::EmissionConstrainedTransportationProblem)

Build the multimodal emission-capped transportation LP. Deterministic — uses
only the struct fields.
"""
function build_model(prob::EmissionConstrainedTransportationProblem)
    model = Model()
    K = length(prob.options)
    @variable(model, x[1:K] >= 0)
    @objective(model, Min, sum(prob.cost[k] * x[k] for k in 1:K))
    nS, nD = prob.n_sources, prob.n_customers
    from_src = [Int[] for _ in 1:nS]
    rail_from = [Int[] for _ in 1:nS]
    to_cust = [Int[] for _ in 1:nD]
    in_region = [Int[] for _ in 1:length(prob.region_cap)]
    for (k, (l, m)) in enumerate(prob.options)
        i, j = prob.lanes[l]
        push!(from_src[i], k)
        push!(to_cust[j], k)
        m == 2 && push!(rail_from[i], k)
        push!(in_region[prob.region_of_source[i]], k)
    end
    for i in 1:nS
        @constraint(model, sum(x[k] for k in from_src[i]; init=AffExpr(0.0)) <= prob.supplies[i])
    end
    for j in 1:nD
        @constraint(model, sum(x[k] for k in to_cust[j]) >= prob.demands[j])
    end
    for i in 1:nS
        isempty(rail_from[i]) && continue
        @constraint(model, sum(x[k] for k in rail_from[i]) <= prob.rail_capacity[i])
    end
    for r in eachindex(prob.region_cap)
        @constraint(
            model,
            sum(prob.emission[k] * x[k] for k in in_region[r]; init=AffExpr(0.0)) <=
                prob.region_cap[r]
        )
    end
    @constraint(model, sum(prob.emission[k] * x[k] for k in 1:K) <= prob.global_cap)
    return model
end

register_variant(
    :transportation,
    :emission_constrained,
    EmissionConstrainedTransportationProblem,
    "Multimodal (truck/rail/intermodal) transportation on sparse geographic lanes under regional and corporate CO2 caps and rail-terminal capacities; planted lowest-emission plan and an emission lower-bound certificate";
    tags=[:logistics, :bipartite],
    max_target_variables=1_000_000,
)
