using JuMP
using Random
using Distributions

"""
Integral hub set and node assignment planted for the compact formulation.
"""
struct CompactHubWitness
    hubs::Vector{Int}
    assignment::Vector{Int}
end

"""
    CompactSingleAllocationHubProblem <: ProblemGenerator

Origin-indexed `O(n^3)` formulation of the uncapacitated single-allocation
`p`-hub median problem.  Unlike the four-index SKO variant, it retains directed
OD flows and uses `n^2(n-1)` continuous inter-hub-flow variables together with
`n^2` assignment/opening variables, for exactly `n^3` variables.

A feeder **reach window** limits allocation: node `i` may only be allocated to a
hub within `reach` of it (`z_ik` is fixed at 0 beyond it; the node itself is
always admissible). The window is a service promise — a city's collection leg
may not exceed a maximum drive — and it is what makes the hub count `p`
genuinely binding.

# Feasibility

  - `feasible`: greedy `p` hubs with nearest-hub allocation (`CompactHubWitness`);
    the reach is 1.05-1.3x the largest witness allocation distance.
  - `infeasible`: `p + 1` mutually unreachable island regions of at least three
    cities each (`DisjointRegionCertificate`): every node's allocation row
    `sum_{k in A_i} z_ik = 1` with `z_ik <= z_kk` forces an open hub inside its
    region, so `sum_k z_kk >= p + 1`, contradicting the hub-count row in the LP
    relaxation. No single row or bound is violated - the refutation aggregates
    allocation, linking and hub-count rows across all regions.
  - `unknown`: the reach is 0.8-1.2x the greedy hubs' covering radius, so the
    `p` hubs may or may not be able to serve every city within reach.
"""
struct CompactSingleAllocationHubProblem <: ProblemGenerator
    n_nodes::Int
    p::Int
    profile::Symbol
    chi::Float64
    alpha::Float64
    delta::Float64
    locations::Vector{Tuple{Float64, Float64}}
    dist::Matrix{Float64}
    flow::Matrix{Float64}
    outvolume::Vector{Float64}
    involume::Vector{Float64}
    reach::Float64
    feasible_witness::Union{Nothing, CompactHubWitness}
    infeasibility_certificate::Union{Nothing, DisjointRegionCertificate}
    feasibility_status::FeasibilityStatus
end

function _hub_nearest_cube_dimension(target_variables::Int)
    target = max(target_variables, 1)
    candidates = collect(3:max(4, ceil(Int, cbrt(target)) + 2))
    admissible = [n for n in candidates if abs(n^3 - target) <= 0.25 * target || n^3 <= 50]
    pool = isempty(admissible) ? candidates : admissible
    return pool[argmin([abs(n^3 - target) for n in pool])]
end

function CompactSingleAllocationHubProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    n = _hub_nearest_cube_dimension(target_variables)
    profile = rand(rng, (:passenger, :freight, :telecom))
    shape = rand(rng, (:clustered, :corridor, :archipelago))
    nominal_p = clamp(round(Int, rand(rng, Uniform(0.18, 0.38)) * n), 2, n - 1)
    groups = Vector{Int}[]
    if feasibility_status == infeasible
        # p + 1 island regions of >= 3 cities each must fit in n nodes.
        p = clamp(nominal_p, 1, max(1, min(n ÷ 3 - 1, 10)))
        locations, groups, min_sep = _hub_island_geography(rng, n, p + 1; min_members=3)
    else
        p = nominal_p
        locations = _hub_city_locations(rng, n, shape)
    end
    dist = _hub_distance_matrix(locations)
    populations = _hub_populations(rng, n)

    if profile == :passenger
        chi, delta = 1.0, 1.0
        alpha = rand(rng, (0.2, 0.4, 0.6, 0.8))
        symmetric = true
        scale = rand(rng, Uniform(20.0, 90.0))
        decay = rand(rng, Uniform(0.4, 1.0))
        noise = rand(rng, Uniform(0.5, 0.9))
    elseif profile == :freight
        chi, delta = rand(rng, Uniform(2.7, 3.3)), rand(rng, Uniform(1.8, 2.2))
        alpha = rand(rng, Uniform(0.7, 0.8))
        symmetric = false
        scale = rand(rng, Uniform(0.5, 2.0))
        decay = rand(rng, Uniform(0.5, 1.1))
        noise = rand(rng, Uniform(0.7, 1.2))
    else
        chi = delta = rand(rng, Uniform(1.0, 2.5))
        alpha = rand(rng, Uniform(0.05, 0.4))
        symmetric = false
        scale = rand(rng, Uniform(0.8, 3.0))
        decay = rand(rng, Uniform(0.5, 1.0))
        noise = rand(rng, Uniform(0.6, 1.0))
    end
    flow = _hub_gravity_flows(
        rng, n, populations, dist, decay, noise; symmetric=symmetric, scale=scale
    )
    outvolume = vec(sum(flow; dims=2))
    involume = vec(sum(flow; dims=1))

    witness = nothing
    certificate = nothing
    if feasibility_status == infeasible
        # Regions are > 0.6 min_sep apart and < 0.4 min_sep across, so the
        # windows never leave a region (see `_hub_island_geography`).
        reach = 0.40 * min_sep
        certificate = DisjointRegionCertificate(groups, p)
    else
        hubs = sort(_hub_greedy_hubs(dist, outvolume .+ involume, p))
        assignment = _hub_nearest_assignment(dist, hubs)
        radius = maximum(dist[i, assignment[i]] for i in 1:n)
        if feasibility_status == feasible
            reach = radius * rand(rng, Uniform(1.05, 1.3)) + 1e-6
            witness = CompactHubWitness(hubs, assignment)
        else
            reach = radius * rand(rng, Uniform(0.8, 1.2))
        end
    end

    return CompactSingleAllocationHubProblem(
        n,
        p,
        profile,
        chi,
        alpha,
        delta,
        locations,
        dist,
        flow,
        outvolume,
        involume,
        reach,
        witness,
        certificate,
        feasibility_status,
    )
end

function build_model(prob::CompactSingleAllocationHubProblem)
    model = Model()
    n = prob.n_nodes
    @variable(model, z[1:n, 1:n], Bin)
    @variable(model, q[i = 1:n, k = 1:n, m = 1:n; k != m] >= 0)
    # Reach window: allocations beyond `reach` are fixed shut (a node is always
    # admissible to itself, distance 0).
    for i in 1:n, k in 1:n
        prob.dist[i, k] > prob.reach && fix(z[i, k], 0.0; force=true)
    end

    @constraint(model, single_allocation[i = 1:n], sum(z[i, k] for k in 1:n) == 1)
    @constraint(model, allocation_open[i = 1:n, k = 1:n], z[i, k] <= z[k, k])
    @constraint(model, hub_count, sum(z[k, k] for k in 1:n) == prob.p)
    @constraint(
        model,
        flow_balance[i = 1:n, k = 1:n],
        sum(q[i, k, m] for m in 1:n if m != k) - sum(q[i, m, k] for m in 1:n if m != k) ==
            prob.outvolume[i] * z[i, k] - sum(prob.flow[i, j] * z[j, k] for j in 1:n)
    )
    @constraint(
        model,
        flow_tail_open[i = 1:n, k = 1:n, m = 1:n; k != m],
        q[i, k, m] <= prob.outvolume[i] * z[k, k]
    )
    @constraint(
        model,
        flow_head_open[i = 1:n, k = 1:n, m = 1:n; k != m],
        q[i, k, m] <= prob.outvolume[i] * z[m, m]
    )

    @objective(
        model,
        Min,
        sum(prob.chi * prob.dist[i, k] * prob.outvolume[i] * z[i, k] for i in 1:n, k in 1:n) +
            sum(prob.delta * prob.dist[k, j] * prob.involume[j] * z[j, k] for j in 1:n, k in 1:n) +
            sum(
                prob.alpha * prob.dist[k, m] * q[i, k, m] for i in 1:n, k in 1:n, m in 1:n if k != m
            )
    )
    return model
end

register_variant(
    :hub_location,
    :compact_single_allocation,
    CompactSingleAllocationHubProblem,
    "Compact origin-indexed O(n^3) single-allocation p-hub median with directed OD traffic and passenger/freight/telecom profiles";
    tags=[:location, :network, :multicommodity, :big_m],
)
