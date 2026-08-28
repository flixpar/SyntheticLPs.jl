using JuMP
using Random

"""
    CombinatorialAuctionProblem <: ProblemGenerator

Winner-determination set packing in which each binary bid requests a bundle of
items and receives a value correlated with item prices and bundle synergy.
"""
struct CombinatorialAuctionProblem <: ProblemGenerator
    n_items::Int
    bundles::Vector{Vector{Int}}
    bid_values::Vector{Float64}
    minimum_winners::Int
end

function CombinatorialAuctionProblem(
    target_variables::Int,
    feasibility_status::FeasibilityStatus,
    seed::Int,
)
    target_variables >= 4 ||
        throw(ArgumentError("combinatorial auction needs at least 4 variables"))
    rng = MersenneTwister(seed)
    n_bids = target_variables
    n_items = max(4, round(Int, 0.35 * n_bids))
    bundles, n_planted = _set_columns_with_partition(
        rng, n_items, n_bids; max_size=max(2, min(8, n_items)),
    )

    if feasibility_status == infeasible
        for bundle in bundles
            push!(bundle, 1)
            sort!(unique!(bundle))
        end
        minimum_winners = 2
    elseif feasibility_status == feasible
        minimum_winners = n_planted
    else
        minimum_winners = 0
    end

    item_values = _set_positive_coefficients(rng, n_items; low=10, high=80)
    bid_values = Float64[]
    for bundle in bundles
        base = sum(item_values[i] for i in bundle)
        synergy = 0.85 + 0.5 * rand(rng) + 0.04 * max(0, length(bundle) - 1)
        push!(bid_values, round(base * synergy; digits=2))
    end

    return CombinatorialAuctionProblem(n_items, bundles, bid_values, minimum_winners)
end

function build_model(prob::CombinatorialAuctionProblem)
    model = Model()
    n_bids = length(prob.bundles)
    incidence = _set_elements_to_columns(prob.bundles, prob.n_items)
    @variable(model, accept[1:n_bids], Bin)
    @objective(model, Max, sum(prob.bid_values[b] * accept[b] for b in 1:n_bids))
    for i in 1:prob.n_items
        @constraint(model, sum(accept[b] for b in incidence[i]) <= 1)
    end
    if prob.minimum_winners > 0
        @constraint(model, sum(accept) >= prob.minimum_winners)
    end
    return model
end

register_variant(
    :set_system,
    :combinatorial_auction,
    CombinatorialAuctionProblem,
    "Combinatorial-auction winner determination with bundle-correlated bid values",
)
