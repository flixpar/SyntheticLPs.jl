using JuMP
using Random

"""
    AuctionWitness

A planted allocation: the bids `accepted` come from distinct bidders, jointly
request at most `supply[i]` units of every item, and raise at least the reserve
revenue.
"""
struct AuctionWitness
    accepted::Vector{Int}
end

"""
    AuctionDualCertificate

LP-dual bound on the auction's revenue. With item prices `prices[i] >= 0` and
bidder surpluses `surplus[k] >= 0` such that every bid `b` of bidder `k`
satisfies `sum_i quantity[b,i] * prices[i] + surplus[k] >= value[b]`, weak
duality gives `sum_b value[b] x[b] <= sum_i supply[i] prices[i] + sum_k
surplus[k] == bound` for every LP-feasible `x`. A reserve above `bound` is
infeasible even in the LP relaxation, and presolve cannot see it (the
argument combines every item and bidder row with the dense revenue row).
"""
struct AuctionDualCertificate
    prices::Vector{Float64}
    surplus::Vector{Float64}
    bound::Float64
end

"""
    CombinatorialAuctionProblem <: ProblemGenerator

Multi-unit combinatorial-auction winner determination with XOR bidders
(spectrum- or slot-auction style, after the CATS "regions" distribution of
Leyton-Brown et al.). Items are licences placed on a map, each with several
identical units; every bidder wants a geographically compact bundle around its
home region and submits a few XOR alternatives (substitute bundles) with
per-item unit quantities. Values combine a common per-item value, a private
bidder deviation, and bundle complementarity.

# Formulation

    max  sum_b v_b x_b
    s.t. sum_b q[b,i] x_b <= u_i       for every item i (multi-unit supply)
         sum_{b in bids(k)} x_b <= 1   for every bidder k (XOR)
         sum_b v_b x_b >= reserve      (seller's reserve revenue)
         x binary

Unlike `set_packing` (0/1 space-time cells), the item rows carry general integer
quantities and supplies, bidders add XOR rows, and the reserve row couples all
columns. Variables: exactly `target_variables` bids.

# Feasibility

  - `feasible`: a greedy allocation by value per requested unit is the
    `feasible_witness`; the reserve is 85–100% of its revenue.
  - `infeasible`: the reserve exceeds an LP-dual revenue bound by 3%
    ([`AuctionDualCertificate`](@ref)).
  - `unknown`: the reserve lies between the greedy revenue and the dual bound.
"""
struct CombinatorialAuctionProblem <: ProblemGenerator
    n_items::Int
    supply::Vector{Int}
    bundles::Vector{Vector{Int}}
    quantities::Vector{Vector{Int}}
    bidder_of::Vector{Int}
    bid_values::Vector{Float64}
    reserve::Float64
    feasible_witness::Union{Nothing, AuctionWitness}
    infeasibility_certificate::Union{Nothing, AuctionDualCertificate}
end

function CombinatorialAuctionProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target_variables >= 2 || throw(ArgumentError("combinatorial auction needs at least 2 variables"))
    rng = MersenneTwister(seed)
    n_bids = target_variables
    n_items = max(3, round(Int, (0.15 + 0.1 * rand(rng)) * n_bids))

    # Items on a map; neighbourhoods from unit-disk proximity (~6 neighbours).
    ix, iy, _ = _graph_geometric_points(rng, n_items, 6.0; hotspot_share=0.4)
    item_adj = _graph_adjacency(n_items, _graph_pairs_within(ix, iy, 1.0))
    supply = [rand(rng) < 0.4 ? 1 : rand(rng, 2:6) for _ in 1:n_items]
    common_value = round.(_graph_lognormal_weights(rng, n_items; median=30.0, sigma=0.5); digits=2)

    # Bundle grown by a random walk over neighbouring items.
    function grow_bundle(home::Int, size::Int)
        bundle = [home]
        frontier = copy(item_adj[home])
        while length(bundle) < size
            isempty(frontier) && (frontier = [rand(rng, 1:n_items)])
            next = frontier[rand(rng, 1:length(frontier))]
            if !(next in bundle)
                push!(bundle, next)
                append!(frontier, item_adj[next])
            end
            filter!(i -> !(i in bundle), frontier)
        end
        return sort!(bundle)
    end

    bundles = Vector{Vector{Int}}()
    quantities = Vector{Vector{Int}}()
    bidder_of = Int[]
    bid_values = Float64[]
    bidder = 0
    while length(bundles) < n_bids
        bidder += 1
        home = rand(rng, 1:n_items)
        core_size = clamp(round(Int, 2 + 2 * abs(randn(rng))), 1, min(8, n_items))
        core = grow_bundle(home, core_size)
        deviation = 1 + 0.25 * randn(rng)
        n_alternatives = min(rand(rng, 1:6), n_bids - length(bundles))
        for a in 1:n_alternatives
            # Substitutes: swap one or two items of the core for neighbours.
            bundle = copy(core)
            if a > 1
                for _ in 1:rand(rng, 1:2)
                    pos = rand(rng, 1:length(bundle))
                    options = [i for i in item_adj[bundle[pos]] if !(i in bundle)]
                    isempty(options) || (bundle[pos] = options[rand(rng, 1:length(options))])
                end
                sort!(unique!(bundle))
            end
            q = [rand(rng, 1:min(supply[i], 3)) for i in bundle]
            base = sum(q[t] * common_value[bundle[t]] * (1 + 0.2 * randn(rng)) for t in eachindex(bundle))
            complementarity = 1 + 0.15 * (length(bundle) - 1)
            value = max(1.0, base * deviation * complementarity)
            push!(bundles, bundle)
            push!(quantities, q)
            push!(bidder_of, bidder)
            push!(bid_values, round(value; digits=2))
        end
    end
    n_bidders = bidder

    # Greedy allocation by value per requested unit.
    remaining = copy(supply)
    served = falses(n_bidders)
    accepted = Int[]
    for b in sortperm([bid_values[b] / sum(quantities[b]) for b in 1:n_bids]; rev=true)
        served[bidder_of[b]] && continue
        all(remaining[bundles[b][t]] >= quantities[b][t] for t in eachindex(bundles[b])) || continue
        for t in eachindex(bundles[b])
            remaining[bundles[b][t]] -= quantities[b][t]
        end
        served[bidder_of[b]] = true
        push!(accepted, b)
    end
    sort!(accepted)
    greedy_revenue = sum(bid_values[accepted]; init=0.0)

    # Dual bound: price every item at its best value per unit among the bids
    # that request it, then give each bidder the surplus its bids still need.
    prices = zeros(n_items)
    for b in 1:n_bids, i in bundles[b]
        prices[i] = max(prices[i], bid_values[b] / sum(quantities[b]))
    end
    # Scale prices down so bidder surpluses carry part of the bound (tighter).
    prices .*= 0.6
    surplus = zeros(n_bidders)
    for b in 1:n_bids
        priced = sum(quantities[b][t] * prices[bundles[b][t]] for t in eachindex(bundles[b]))
        surplus[bidder_of[b]] = max(surplus[bidder_of[b]], bid_values[b] - priced)
    end
    bound = sum(supply .* prices) + sum(surplus)

    witness = nothing
    certificate = nothing
    if feasibility_status == feasible
        reserve = round((0.85 + 0.15 * rand(rng)) * greedy_revenue; digits=2)
        witness = AuctionWitness(accepted)
    elseif feasibility_status == infeasible
        reserve = round(1.03 * bound; digits=2)
        certificate = AuctionDualCertificate(prices, surplus, bound)
    else
        reserve = round(greedy_revenue + 0.5 * rand(rng) * (bound - greedy_revenue); digits=2)
    end

    return CombinatorialAuctionProblem(
        n_items,
        supply,
        bundles,
        quantities,
        bidder_of,
        bid_values,
        reserve,
        witness,
        certificate,
    )
end

function build_model(prob::CombinatorialAuctionProblem)
    model = Model()
    n_bids = length(prob.bundles)
    @variable(model, accept[1:n_bids], Bin)
    @objective(model, Max, sum(prob.bid_values[b] * accept[b] for b in 1:n_bids))
    requests = [Tuple{Int, Int}[] for _ in 1:prob.n_items]
    for b in 1:n_bids, (t, i) in enumerate(prob.bundles[b])
        push!(requests[i], (b, prob.quantities[b][t]))
    end
    for i in 1:prob.n_items
        isempty(requests[i]) && continue
        @constraint(model, sum(q * accept[b] for (b, q) in requests[i]) <= prob.supply[i])
    end
    bids_of = Dict{Int, Vector{Int}}()
    for (b, k) in enumerate(prob.bidder_of)
        push!(get!(bids_of, k, Int[]), b)
    end
    for k in sort!(collect(keys(bids_of)))
        length(bids_of[k]) >= 2 && @constraint(model, sum(accept[b] for b in bids_of[k]) <= 1)
    end
    @constraint(model, sum(prob.bid_values[b] * accept[b] for b in 1:n_bids) >= prob.reserve)
    return model
end

register_variant(
    :set_system,
    :combinatorial_auction,
    CombinatorialAuctionProblem,
    "Multi-unit combinatorial-auction winner determination with XOR bidders, regional bundles, and a reserve revenue";
    tags=[:economics, :packing],
    min_target_variables=2,
)
