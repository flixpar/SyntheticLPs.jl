using JuMP
using Random

"""
    SetCoverWitness

A planted cover: every demand point lies in at least one of the columns
`sites`, and `length(sites) <= maximum_selected`.
"""
struct SetCoverWitness
    sites::Vector{Int}
end

"""
    SetCoverPackingCertificate

LP lower bound on the number of selected sites: no column contains two of the
demand points `points`, so summing their covering rows (each `>= 1`) counts
every `x_j` at most once and gives `sum(x) >= length(points)`. A budget
`maximum_selected < length(points)` is therefore infeasible even in the LP
relaxation; the argument aggregates hundreds of rows, so presolve does not see
it.
"""
struct SetCoverPackingCertificate
    points::Vector{Int}
end

"""
    SetCoverProblem <: ProblemGenerator

Location set covering (LSCP; Toregas et al. 1971) with heterogeneous station
types: choose emergency-service or cell-tower sites so that every demand point
lies within the coverage radius of some chosen site, at minimum cost, subject to
a budget on the number of sites.

# Data

Demand points are clustered (towns over rural scatter). Each candidate site is
anchored at a demand point (every point anchors at least one site, so a cover
always exists) and has one of three station types with coverage radius
`r ∈ {0.75, 1, 1.5}·r₀`; its column is the set of demand points within its
radius, so column sizes are heavy-tailed (urban sites cover many points). Site
cost grows with the station type and with local land price (crowding).

# Formulation

    min  sum_j c_j x_j
    s.t. sum_{j : i in S_j} x_j >= 1   for every demand point i
         sum_j x_j <= maximum_selected
         x binary

Variables: exactly `target_variables` candidate sites.

# Feasibility

  - `feasible`: a greedy cover (largest covering site per uncovered point) is the
    `feasible_witness`; the budget is its size plus 0–10%.
  - `infeasible`: the budget is at least 3% below a co-coverage packing bound
    ([`SetCoverPackingCertificate`](@ref)).
  - `unknown`: the budget lies between the packing bound and the greedy cover
    size; the LP optimum of `sum(x)` lies in between.
"""
struct SetCoverProblem <: ProblemGenerator
    n_elements::Int
    columns::Vector{Vector{Int}}
    costs::Vector{Float64}
    maximum_selected::Int
    feasible_witness::Union{Nothing, SetCoverWitness}
    infeasibility_certificate::Union{Nothing, SetCoverPackingCertificate}
end

function SetCoverProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    target_variables >= 2 || throw(ArgumentError("set cover needs at least 2 variables"))
    rng = MersenneTwister(seed)
    n_columns = target_variables
    n_elements = max(1, round(Int, (0.3 + 0.2 * rand(rng)) * n_columns))

    # Demand points: unit-radius-scaled geography with ~`coverage` points per
    # base-radius disk on average (town hotspots are denser).
    coverage = 5.0 + 7.0 * rand(rng)
    px, py, _ = _graph_geometric_points(rng, n_elements, coverage; hotspot_share=0.6)
    radii = (0.75, 1.0, 1.5)
    type_cost = (40.0, 60.0, 110.0)
    buckets = _graph_buckets(px, py, maximum(radii))

    columns = Vector{Vector{Int}}(undef, n_columns)
    costs = Vector{Float64}(undef, n_columns)
    anchors = vcat(randperm(rng, n_elements), rand(rng, 1:n_elements, n_columns - n_elements))
    for j in 1:n_columns
        a = anchors[j]
        kind = rand(rng) < 0.5 ? 1 : (rand(rng) < 0.7 ? 2 : 3)
        r = radii[kind]
        # The site sits near its anchor, which it always covers.
        angle, offset = 2pi * rand(rng), 0.3 * r * rand(rng)
        sx, sy = px[a] + offset * cos(angle), py[a] + offset * sin(angle)
        gx, gy = floor(Int, sx / maximum(radii)), floor(Int, sy / maximum(radii))
        covered = Int[]
        for dx in -1:1, dy in -1:1
            bucket = get(buckets, (gx + dx, gy + dy), nothing)
            bucket === nothing && continue
            for i in bucket
                (px[i] - sx)^2 + (py[i] - sy)^2 <= r^2 && push!(covered, i)
            end
        end
        a in covered || push!(covered, a)
        columns[j] = sort!(covered)
        # Land price rises with crowding (points covered per unit area).
        crowding = length(covered) / (coverage * r^2)
        costs[j] = round(type_cost[kind] * (0.8 + 0.3 * min(crowding, 3.0)) * exp(0.15 * randn(rng)); digits=2)
    end

    incidence = _set_elements_to_columns(columns, n_elements)

    # Greedy cover: for each still-uncovered point (random order) take the
    # covering site that covers the most points.
    covered = falses(n_elements)
    cover = Int[]
    for i in randperm(rng, n_elements)
        covered[i] && continue
        j = incidence[i][argmax(length.(columns[incidence[i]]))]
        push!(cover, j)
        covered[columns[j]] .= true
    end
    sort!(cover)

    # Co-coverage packing: points no two of which share a covering site.
    blocked = falses(n_elements)
    packing = Int[]
    for i in randperm(rng, n_elements)
        blocked[i] && continue
        push!(packing, i)
        for j in incidence[i]
            blocked[columns[j]] .= true
        end
    end
    sort!(packing)

    witness = nothing
    certificate = nothing
    if feasibility_status == feasible
        maximum_selected = length(cover) + floor(Int, 0.1 * rand(rng) * length(cover))
        witness = SetCoverWitness(cover)
    elseif feasibility_status == infeasible
        maximum_selected = length(packing) - max(1, ceil(Int, 0.03 * length(packing)))
        certificate = SetCoverPackingCertificate(packing)
    else
        lo, hi = length(packing), length(cover)
        maximum_selected = lo + round(Int, rand(rng) * max(0, hi - lo))
    end

    return SetCoverProblem(n_elements, columns, costs, maximum_selected, witness, certificate)
end

function build_model(prob::SetCoverProblem)
    model = Model()
    n_columns = length(prob.columns)
    incidence = _set_elements_to_columns(prob.columns, prob.n_elements)
    @variable(model, x[1:n_columns], Bin)
    @objective(model, Min, sum(prob.costs[j] * x[j] for j in 1:n_columns))
    for i in 1:prob.n_elements
        @constraint(model, sum(x[j] for j in incidence[i]) >= 1)
    end
    @constraint(model, sum(x) <= prob.maximum_selected)
    return model
end

register_variant(
    :set_system,
    :set_cover,
    SetCoverProblem,
    "Location set covering with heterogeneous station radii over clustered demand points and a site budget";
    default=true,
)
