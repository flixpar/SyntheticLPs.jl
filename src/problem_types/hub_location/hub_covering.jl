using JuMP
using Random
using Distributions

"""
Feasible hub set-covering witness: a minimal open hub set (greedy pruning of
the all-open solution, most expensive hubs first) under which every ordered OD
pair keeps at least one admissible path with both hubs open, its opening cost,
and the service threshold.
"""
struct HubCoveringWitness
    open_hubs::Vector{Int}
    opening_cost::Float64
    threshold::Float64
end

"""
Fallback certificate: an OD pair whose cheapest two-hub path exceeds the
service threshold (its covering row is empty). Used only when the island
geography cannot certify the budget argument below.
"""
struct HubCoveringCertificate
    origin::Int
    destination::Int
    minimum_route_cost::Float64
    threshold::Float64
end

"""
Relaxation-proof budget certificate. For each listed region, `pairs[r]` is an
OD pair inside it whose admissible paths only use hubs of `hub_sets[r]` (a
subset of the region), and the hub sets are pairwise disjoint. The pair's
covering row `sum x >= 1` and its per-hub linking rows
`sum_{paths through k} x <= y_k` give `sum_{k in hub_sets[r]} y_k >= 1`, so the
opening cost is at least `minimum_cost = sum_r min_{k in hub_sets[r]} f_k` -
above `budget`. Only covering, linking and budget rows are used, so the
argument holds in the LP relaxation.
"""
struct HubCoveringBudgetCertificate
    pairs::Vector{Tuple{Int, Int}}
    hub_sets::Vector{Vector{Int}}
    minimum_cost::Float64
    budget::Float64
end

"""
    HubSetCoveringProblem <: ProblemGenerator

Budgeted multiple-allocation hub set-covering location problem (Campbell 1994;
Kara & Tansel 2003). Every ordered OD pair must select a
collection/transfer/distribution path `i -> k -> m -> j` whose generalized cost
`chi d_ik + alpha d_km + delta d_mj` is within `service_threshold` (the express
delivery promise), using only open hubs. The objective is hub opening cost
plus the operating cost of the chosen service paths (OD volume times path
cost, scaled so operations are roughly 10-40% of the opening bill), subject to
an opening `budget`. Without the operating term every path variable would be
cost-free, and reverse-orientation paths (`(k,m)` vs `(m,k)`) would be
duplicate columns that presolve merges.

Path-to-hub linking uses the tight per-hub form of Hamacher et al. (2004): for
every OD pair and hub `k`, the total flow on admissible paths that visit `k`
(as collection or distribution hub, counted once for `k = m`) is at most `y_k`.
It implies the weaker per-path rows `x_ijkm <= y_k`, `x_ijkm <= y_m` and makes
every covering row a genuine fractional hub-covering requirement.

# Fields

  - `n_nodes`, `profile` (`:passenger`, `:freight`, `:express`), `chi`, `alpha`, `delta`
  - `locations`, `dist`, `fixed_cost`
  - `service_threshold::Float64`
  - `covering_sets::Dict{Tuple{Int,Int},Vector{Tuple{Int,Int}}}`: admissible
    hub pairs `(k, m)` per ordered OD pair
  - `flow::Matrix{Float64}`: OD volumes (gravity model, zero diagonal)
  - `unit_cost::Float64`: operating cost per unit volume per unit path cost
  - `budget::Float64`: opening budget
  - `feasible_witness`, `infeasibility_certificate`, `feasibility_status`
"""
struct HubSetCoveringProblem <: ProblemGenerator
    n_nodes::Int
    profile::Symbol
    chi::Float64
    alpha::Float64
    delta::Float64
    locations::Vector{Tuple{Float64, Float64}}
    dist::Matrix{Float64}
    fixed_cost::Vector{Float64}
    service_threshold::Float64
    covering_sets::Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}
    flow::Matrix{Float64}
    unit_cost::Float64
    budget::Float64
    feasible_witness::Union{Nothing, HubCoveringWitness}
    infeasibility_certificate::Union{Nothing, HubCoveringCertificate, HubCoveringBudgetCertificate}
    feasibility_status::FeasibilityStatus
end

_hub_covering_variable_count(prob::HubSetCoveringProblem) =
    prob.n_nodes + sum(length, values(prob.covering_sets); init=0)

# Hubs visited by the admissible paths of one OD pair.
_hub_covering_hubs(paths::Vector{Tuple{Int, Int}}) =
    sort!(unique!(vcat(first.(paths), last.(paths))))

"""
    _hub_covering_prune(covering_sets, fixed_cost, n) -> Vector{Int}

Greedy minimal cover: start with every hub open and close hubs in decreasing
fixed-cost order whenever every OD pair still keeps an admissible path with
both hubs open. Returns the sorted open set (all hubs if some pair is
uncoverable).
"""
function _hub_covering_prune(
    covering_sets::Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}},
    fixed_cost::Vector{Float64},
    n::Int,
)
    ods = sort!(collect(keys(covering_sets)))
    open_paths = [length(covering_sets[od]) for od in ods]
    any(iszero, open_paths) && return collect(1:n)
    # paths_through[h] = (od index, path) pairs that visit hub h
    through = [Tuple{Int, Tuple{Int, Int}}[] for _ in 1:n]
    for (o, od) in enumerate(ods), path in covering_sets[od]
        push!(through[path[1]], (o, path))
        path[2] != path[1] && push!(through[path[2]], (o, path))
    end
    is_open = trues(n)
    for h in sortperm(fixed_cost; rev=true)
        # Paths through h that are currently fully open lose their status.
        lost = Dict{Int, Int}()
        for (o, (k, m)) in through[h]
            (is_open[k] && is_open[m]) || continue
            lost[o] = get(lost, o, 0) + 1
        end
        all(open_paths[o] > c for (o, c) in lost) || continue
        for (o, c) in lost
            open_paths[o] -= c
        end
        is_open[h] = false
    end
    return findall(is_open)
end

function _build_hub_covering(
    n::Int, target_variables::Int, feasibility_status::FeasibilityStatus, rng::AbstractRNG
)
    profile = rand(rng, (:passenger, :freight, :express))
    shape = rand(rng, (:clustered, :corridor, :archipelago))
    regions = Vector{Int}[]
    if feasibility_status == infeasible
        q = min(rand(rng, 2:3), max(2, n ÷ 3))
        locations, regions, _ = _hub_island_geography(rng, n, q; min_members=3)
    else
        locations = _hub_city_locations(rng, n, shape)
    end
    dist = _hub_distance_matrix(locations)
    if profile == :freight
        chi, alpha, delta = 3.0, 0.75, 2.0
    elseif profile == :passenger
        chi, alpha, delta = 1.0, rand(rng, (0.2, 0.4, 0.6, 0.8)), 1.0
    else
        chi = rand(rng, Uniform(1.2, 2.0))
        alpha = rand(rng, Uniform(0.35, 0.65))
        delta = rand(rng, Uniform(1.2, 2.0))
    end

    mean_dist = sum(dist) / max(n^2 - n, 1)
    base_fixed = mean_dist * n * rand(rng, Uniform(1.5, 4.0))
    fixed_cost = [base_fixed * rand(rng, Uniform(0.7, 1.3)) for _ in 1:n]

    min_route = fill(Inf, n, n)
    route_costs = Float64[]
    sizehint!(route_costs, n^3 * (n - 1))
    route_cost(i, j, k, m) = chi * dist[i, k] + alpha * dist[k, m] + delta * dist[m, j]
    for i in 1:n, j in 1:n
        i == j && continue
        for k in 1:n, m in 1:n
            c = route_cost(i, j, k, m)
            push!(route_costs, c)
            min_route[i, j] = min(min_route[i, j], c)
        end
    end
    sort!(route_costs)
    desired = max(target_variables - n, 0)
    worst_minimum = maximum(min_route[i, j] for i in 1:n for j in 1:n if i != j)
    minimum_rank = searchsortedlast(route_costs, worst_minimum)

    # Island mode: for each region, the intra-region OD pair whose cheapest
    # path through any hub outside the region is most expensive; the threshold
    # must stay below that cost for the pair's paths to stay in-region.
    region_pairs = Tuple{Int, Int}[]
    cap = Inf
    if feasibility_status == infeasible
        region_of = zeros(Int, n)
        for (g, members) in enumerate(regions), v in members
            region_of[v] = g
        end
        for (g, members) in enumerate(regions)
            best_pair, best_escape = (0, 0), -Inf
            for i in members, j in members
                i == j && continue
                escape = minimum(
                    route_cost(i, j, k, m) for
                    k in 1:n, m in 1:n if region_of[k] != g || region_of[m] != g
                )
                escape > best_escape && ((best_pair, best_escape) = ((i, j), escape))
            end
            if best_escape > worst_minimum * 1.02
                push!(region_pairs, best_pair)
                cap = min(cap, best_escape)
            end
        end
    end

    if feasibility_status == infeasible && !isempty(region_pairs)
        # Cover every pair (threshold >= worst minimum) yet keep the
        # certifying pairs in-region (threshold below their escape cost).
        maximum_rank = searchsortedfirst(route_costs, cap) - 1
        rank = clamp(desired, minimum_rank, max(minimum_rank, maximum_rank))
        threshold = route_costs[rank]
    elseif feasibility_status == infeasible
        maximum_rank = searchsortedfirst(route_costs, worst_minimum) - 1
        rank = clamp(desired, 0, maximum_rank)
        threshold = rank == 0 ? prevfloat(first(route_costs)) : route_costs[rank]
        threshold < worst_minimum || (threshold = prevfloat(worst_minimum))
    else
        rank = clamp(desired, minimum_rank, length(route_costs))
        threshold = route_costs[rank]
    end

    covering_sets = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}()
    for i in 1:n, j in 1:n
        i == j && continue
        paths = Tuple{Int, Int}[]
        for k in 1:n, m in 1:n
            route_cost(i, j, k, m) <= threshold && push!(paths, (k, m))
        end
        covering_sets[(i, j)] = paths
    end

    # OD volumes and an operating-cost scale: the cheapest-path operating bill
    # is rho (10-40%) of the opening cost of about sqrt(n) average hubs.
    populations = _hub_populations(rng, n)
    flow = _hub_gravity_flows(
        rng,
        n,
        populations,
        dist,
        rand(rng, Uniform(0.5, 1.0)),
        rand(rng, Uniform(0.6, 1.0));
        symmetric=profile == :passenger,
    )
    cheapest_bill = sum(flow[i, j] * min_route[i, j] for i in 1:n, j in 1:n if i != j)
    rho = rand(rng, Uniform(0.1, 0.4))
    unit_cost = rho * sqrt(n) * base_fixed / max(cheapest_bill, eps())

    witness = nothing
    certificate = nothing
    if feasibility_status == infeasible && !isempty(region_pairs)
        hub_sets = [_hub_covering_hubs(covering_sets[od]) for od in region_pairs]
        minimum_cost = sum(minimum(fixed_cost[k] for k in hs) for hs in hub_sets)
        budget = minimum_cost * rand(rng, Uniform(0.75, 0.95))
        certificate = HubCoveringBudgetCertificate(region_pairs, hub_sets, minimum_cost, budget)
    elseif feasibility_status == infeasible
        uncovered = first(
            od for od in sort!(collect(keys(covering_sets))) if isempty(covering_sets[od])
        )
        i, j = uncovered
        certificate = HubCoveringCertificate(i, j, min_route[i, j], threshold)
        budget = sum(fixed_cost)
    else
        open_hubs = _hub_covering_prune(covering_sets, fixed_cost, n)
        cover_cost = sum(fixed_cost[open_hubs])
        if feasibility_status == feasible
            budget = cover_cost * rand(rng, Uniform(1.05, 1.3))
            witness = HubCoveringWitness(open_hubs, cover_cost, threshold)
        else
            # Fractional covers (and other integer covers) may be cheaper than
            # the greedy one, so a budget below its cost may still suffice.
            budget = cover_cost * rand(rng, Uniform(0.7, 1.05))
        end
    end
    return HubSetCoveringProblem(
        n,
        profile,
        chi,
        alpha,
        delta,
        locations,
        dist,
        fixed_cost,
        threshold,
        covering_sets,
        flow,
        unit_cost,
        budget,
        witness,
        certificate,
        feasibility_status,
    )
end

"""
    HubSetCoveringProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a budgeted hub set-covering instance.

# Variable count

`vars = n + sum_{i != j} |S_ij|` (one opening variable per node, one path
variable per admissible path); the service threshold is chosen by rank among
all `n^3 (n-1)` path costs to hit the target, and a re-sizing loop adjusts `n`.

# Feasibility (relaxation-aware)

  - `feasible`: the threshold covers every OD pair and a greedy minimal hub
    cover fits within the budget (1.05-1.3x its cost; `HubCoveringWitness`).
  - `infeasible`: two or three island regions (at least three cities each). The
    threshold covers every OD pair, but for one OD pair inside each region all
    admissible paths stay on the region's own hubs, so each such region needs
    open hub capacity summing to one; the budget is 0.75-0.95x the sum of the
    regions' cheapest hubs (`HubCoveringBudgetCertificate`). Refuting it
    aggregates covering, linking and budget rows - no single row or bound is
    contradictory. If no region can be certified (never observed in practice),
    the threshold is set below one pair's cheapest path instead
    (`HubCoveringCertificate`, an empty covering row).
  - `unknown`: the budget is 0.7-1.05x the greedy cover's cost; cheaper integer
    or fractional covers may or may not exist.
"""
function HubSetCoveringProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target = max(target_variables, 1)
    hint = clamp(round(Int, 1.25 * target^0.25), 4, 45)
    best = nothing
    best_score = (1, 1, Inf)
    for attempt in 1:16
        rng = MersenneTwister(seed + 67867967 * attempt)
        candidate = _build_hub_covering(hint, target, feasibility_status, rng)
        total = _hub_covering_variable_count(candidate)
        gap = abs(total - target) / target
        # Prefer the budget certificate over the empty-row fallback.
        fallback = candidate.infeasibility_certificate isa HubCoveringCertificate ? 1 : 0
        score = (fallback, gap <= 0.25 || total <= 50 ? 0 : 1, gap)
        if score < best_score
            best, best_score = candidate, score
        end
        gap <= 0.03 && fallback == 0 && break
        ratio = clamp((target / max(total, 1))^0.25, 0.65, 1.5)
        next_hint = round(Int, hint * ratio)
        next_hint == hint && (next_hint += total < target ? 1 : -1)
        hint = clamp(next_hint, 4, 45)
    end
    return best::HubSetCoveringProblem
end

function build_model(prob::HubSetCoveringProblem)
    model = Model()
    n = prob.n_nodes
    ods = sort!(collect(keys(prob.covering_sets)))
    routes = NTuple{4, Int}[]
    for (i, j) in ods, (k, m) in prob.covering_sets[(i, j)]
        push!(routes, (i, j, k, m))
    end
    @variable(model, y[1:n], Bin)
    @variable(model, 0 <= x[routes] <= 1)
    path_cost(i, j, k, m) =
        prob.chi * prob.dist[i, k] + prob.alpha * prob.dist[k, m] + prob.delta * prob.dist[m, j]
    @objective(
        model,
        Min,
        sum(prob.fixed_cost[k] * y[k] for k in 1:n) + sum(
            prob.unit_cost * prob.flow[r[1], r[2]] * path_cost(r...) * x[r] for r in routes;
            init=0.0,
        )
    )

    for (i, j) in ods
        paths = prob.covering_sets[(i, j)]
        if isempty(paths)
            @constraint(model, 0 >= 1)
            continue
        end
        @constraint(model, sum(x[(i, j, k, m)] for (k, m) in paths) >= 1)
        # Per-hub linking: flow on the pair's paths visiting hub h <= y_h.
        visiting = Dict{Int, Vector{Tuple{Int, Int}}}()
        for (k, m) in paths
            push!(get!(visiting, k, Tuple{Int, Int}[]), (k, m))
            m != k && push!(get!(visiting, m, Tuple{Int, Int}[]), (k, m))
        end
        for h in sort!(collect(keys(visiting)))
            @constraint(model, sum(x[(i, j, k, m)] for (k, m) in visiting[h]) <= y[h])
        end
    end
    @constraint(model, sum(prob.fixed_cost[k] * y[k] for k in 1:n) <= prob.budget)
    return model
end

register_variant(
    :hub_location,
    :hub_covering,
    HubSetCoveringProblem,
    "Budgeted multiple-allocation hub set covering: every ordered OD pair needs an open two-hub path within the service threshold, with tight per-hub path linking";
    tags=[:location, :covering],
)
