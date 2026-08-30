using JuMP
using Random
using Distributions

"""
    BinPackingCapacityCertificate

Relaxation-valid proof that total item size exceeds the aggregate capacity of
every available bin. It applies to both identical- and heterogeneous-bin
variants.
"""
struct BinPackingCapacityCertificate
    total_item_size::Float64
    total_available_capacity::Float64
    excess::Float64
end

const _PACKING_CATEGORY_PROFILES = (
    (name = "Food_Grade", mean_fraction = 0.24, deviation = 0.055),
    (name = "Hazardous",  mean_fraction = 0.30, deviation = 0.060),
    (name = "Fragile",    mean_fraction = 0.19, deviation = 0.050),
    (name = "Odorous",    mean_fraction = 0.23, deviation = 0.055),
    (name = "Chilled",    mean_fraction = 0.27, deviation = 0.055),
    (name = "Frozen",     mean_fraction = 0.31, deviation = 0.060),
    (name = "High_Value", mean_fraction = 0.14, deviation = 0.040),
    (name = "Ambient",    mean_fraction = 0.25, deviation = 0.060),
)

# Operationally motivated separation rules. Sampling a subset changes the
# conflict graph by seed while retaining interpretable handling restrictions.
const _PACKING_CONFLICT_CANDIDATES = (
    (1, 2), # food-grade / hazardous
    (1, 4), # food-grade / odorous
    (2, 3), # hazardous / fragile
    (2, 5), # hazardous / chilled
    (2, 6), # hazardous / frozen
    (4, 5), # odorous / chilled
    (4, 6), # odorous / frozen
)

"""
    BinPackingProblem <: ProblemGenerator

Identical-bin packing with handling-category conflicts. `feasible_witness`
stores a bin index per item for requested-feasible instances. Requested-
infeasible instances instead carry an aggregate-capacity certificate that is
valid before or after integrality relaxation.
"""
struct BinPackingProblem <: ProblemGenerator
    n_items::Int
    n_bins::Int
    n_categories::Int
    item_sizes::Vector{Float64}
    bin_capacity::Float64
    item_categories::Vector{Int}
    incompatible_pairs::Vector{Tuple{Int,Int}}
    category_names::Vector{String}
    load_profile::Symbol
    target_variables::Int
    actual_variables::Int
    feasibility_status::FeasibilityStatus
    feasible_witness::Union{Nothing,Vector{Int}}
    infeasibility_certificate::Union{Nothing,BinPackingCapacityCertificate}
end

# Unknown instances intentionally span operating regimes instead of inheriting
# an accidental size-dependent feasibility bias. The arithmetic assignment is
# stable across Julia versions and gives every ten consecutive seeds a 30/40/30
# split, with the target offset preventing one seed from meaning the same regime
# at every scale.
function _packing_unknown_load_profile(target_variables::Int, seed::Int)
    residue = mod(mod(seed, 10) + 3 * mod(target_variables, 10), 10)
    return residue <= 2 ? :light : residue <= 6 ? :nominal : :surge
end

_bin_packing_variable_count(n_items::Int, n_bins::Int, n_categories::Int) =
    n_bins * (n_items + n_categories + 1)

# Search dimensions against the variables that are actually emitted. Candidate
# solutions keep two to six items per bin and at most eight nonempty categories,
# avoiding the discontinuities caused by the old target-size bands.
function _bin_packing_dimensions(target_variables::Int)
    target = max(target_variables, 12)
    maximum_items = max(12, ceil(Int, sqrt(8 * target)) + 12)
    best_key = (typemax(Int), Inf, Inf)
    best_dimensions = (3, 2, 2)

    for n_items in 3:maximum_items
        minimum_bins = max(2, ceil(Int, n_items / 6))
        maximum_bins = max(minimum_bins, min(n_items, ceil(Int, n_items / 2)))
        desired_categories = clamp(round(Int, sqrt(n_items) / 1.8), 2,
                                   length(_PACKING_CATEGORY_PROFILES))
        for n_bins in minimum_bins:maximum_bins
            maximum_categories = min(
                length(_PACKING_CATEGORY_PROFILES), n_items,
                max(2, 2 * n_bins - 2),
            )
            for n_categories in 2:maximum_categories
                actual = _bin_packing_variable_count(
                    n_items, n_bins, n_categories,
                )
                realism_penalty = abs(n_items / n_bins - 3.5) +
                                  0.2 * abs(n_categories - desired_categories)
                shape_penalty = abs(n_bins - n_items / 3.5)
                key = (abs(actual - target), realism_penalty, shape_penalty)
                if key < best_key
                    best_key = key
                    best_dimensions = (n_items, n_bins, n_categories)
                end
            end
        end
    end
    return best_dimensions
end

function _packing_category_data(rng::AbstractRNG, n_items::Int,
                                n_categories::Int)
    categories = [mod(item - 1, n_categories) + 1 for item in 1:n_items]
    shuffle!(rng, categories)
    names = [String(_PACKING_CATEGORY_PROFILES[category].name)
             for category in 1:n_categories]

    candidates = [pair for pair in _PACKING_CONFLICT_CANDIDATES
                  if pair[1] <= n_categories && pair[2] <= n_categories]
    incompatible_pairs = Tuple{Int,Int}[]
    if !isempty(candidates)
        push!(incompatible_pairs, first(candidates))
        for pair in Iterators.drop(candidates, 1)
            rand(rng) < 0.62 && push!(incompatible_pairs, pair)
        end
    end
    sort!(unique!(incompatible_pairs))
    return categories, names, incompatible_pairs
end

function _sample_packing_sizes(rng::AbstractRNG, item_categories::Vector{Int},
                               capacity::Float64)
    sizes = zeros(Float64, length(item_categories))
    for item in eachindex(item_categories)
        profile = _PACKING_CATEGORY_PROFILES[item_categories[item]]
        fraction = rand(rng, Normal(profile.mean_fraction, profile.deviation))
        if rand(rng) < 0.55
            # Common carton/pallet modules with small measurement variation.
            fraction = round(fraction / 0.05) * 0.05 +
                       rand(rng, Normal(0.0, 0.0075))
        end
        sizes[item] = capacity * clamp(fraction, 0.055, 0.62)
    end
    return sizes
end

function _packing_conflict_set(incompatible_pairs::Vector{Tuple{Int,Int}})
    return Set((min(first(pair), last(pair)), max(first(pair), last(pair)))
               for pair in incompatible_pairs)
end

function _first_fit_conflict_packing(item_sizes::Vector{Float64},
                                     capacity::Float64,
                                     item_categories::Vector{Int},
                                     incompatible_pairs::Vector{Tuple{Int,Int}})
    conflicts = _packing_conflict_set(incompatible_pairs)
    order = sortperm(eachindex(item_sizes);
                     by = item -> (-item_sizes[item], item))
    bins = Vector{Vector{Int}}()
    loads = Float64[]
    bin_categories = Vector{Set{Int}}()

    for item in order
        category = item_categories[item]
        selected = 0
        for bin in eachindex(bins)
            loads[bin] + item_sizes[item] <= capacity + 1e-10 || continue
            conflicts_with_bin = any(
                (min(category, other), max(category, other)) in conflicts
                for other in bin_categories[bin] if other != category
            )
            conflicts_with_bin && continue
            selected = bin
            break
        end
        if selected == 0
            push!(bins, [item])
            push!(loads, item_sizes[item])
            push!(bin_categories, Set([category]))
        else
            push!(bins[selected], item)
            loads[selected] += item_sizes[item]
            push!(bin_categories[selected], category)
        end
    end
    return bins, loads
end

function _fit_standard_packing!(item_sizes::Vector{Float64}, capacity::Float64,
                                item_categories::Vector{Int},
                                incompatible_pairs::Vector{Tuple{Int,Int}},
                                n_bins::Int)
    bins = Vector{Vector{Int}}()
    loads = Float64[]
    for _ in 1:60
        bins, loads = _first_fit_conflict_packing(
            item_sizes, capacity, item_categories, incompatible_pairs,
        )
        length(bins) <= n_bins && break
        item_sizes .*= 0.92
    end
    length(bins) <= n_bins || error("Could not construct a packing witness")

    # Avoid exceptionally loose planted instances while preserving the exact
    # conflict-respecting bin partition.
    maximum_load = maximum(loads)
    if maximum_load < 0.78 * capacity
        scale = 0.82 * capacity / maximum_load
        item_sizes .*= scale
    end
    return bins
end

function _canonical_bin_assignment(bins::Vector{Vector{Int}}, n_items::Int)
    ordered_bins = sort(bins; by = minimum)
    assignment = zeros(Int, n_items)
    for (bin, items) in enumerate(ordered_bins), item in items
        assignment[item] = bin
    end
    return assignment
end

"""
    validate_bin_packing_witness(prob::BinPackingProblem) -> Bool

Validate the stored assignment without a solver, including capacity, handling
conflicts, used-bin prefix order, and the triangular canonical-label rule.
"""
function validate_bin_packing_witness(prob::BinPackingProblem)
    witness = prob.feasible_witness
    witness === nothing && return false
    length(witness) == prob.n_items || return false
    all(bin -> 1 <= bin <= prob.n_bins, witness) || return false
    all(witness[item] <= item for item in 1:prob.n_items) || return false

    used_bins = sort(unique(witness))
    used_bins == collect(1:maximum(used_bins)) || return false
    conflicts = _packing_conflict_set(prob.incompatible_pairs)
    for bin in used_bins
        items = findall(==(bin), witness)
        load = sum(prob.item_sizes[item] for item in items)
        load <= prob.bin_capacity + 1e-8 || return false
        categories = unique(prob.item_categories[items])
        for first_index in 1:length(categories)
            for second_index in (first_index + 1):length(categories)
                first_category = categories[first_index]
                second_category = categories[second_index]
                pair = (min(first_category, second_category),
                        max(first_category, second_category))
                pair in conflicts && return false
            end
        end
    end
    return true
end

"""
    validate_bin_packing_certificate(prob::BinPackingProblem) -> Bool

Recompute and validate the aggregate-capacity contradiction.
"""
function validate_bin_packing_certificate(prob::BinPackingProblem)
    certificate = prob.infeasibility_certificate
    certificate === nothing && return false
    demand = sum(prob.item_sizes)
    capacity = prob.n_bins * prob.bin_capacity
    return isapprox(certificate.total_item_size, demand; atol = 1e-8) &&
           isapprox(certificate.total_available_capacity, capacity; atol = 1e-8) &&
           isapprox(certificate.excess, demand - capacity; atol = 1e-8) &&
           certificate.excess > 1e-8
end

# Scale positive samples to an exact target while respecting item-specific
# upper bounds. This preserves profile heterogeneity better than replacing all
# sizes with a common contradiction value.
function _bounded_sizes_with_total(base_sizes::Vector{Float64}, target::Float64,
                                   upper_bounds::Vector{Float64})
    target <= sum(upper_bounds) + 1e-9 ||
        error("Requested aggregate size exceeds item upper bounds")
    sizes = min.(base_sizes .* (target / sum(base_sizes)), upper_bounds)
    remaining = target - sum(sizes)
    if remaining > 1e-10
        headroom = upper_bounds .- sizes
        total_headroom = sum(headroom)
        total_headroom > 0 || error("No headroom remains for aggregate size")
        sizes .+= remaining .* headroom ./ total_headroom
    end
    return sizes
end


function _force_standard_overload!(rng::AbstractRNG,
                                   item_sizes::Vector{Float64},
                                   n_bins::Int, capacity::Float64;
                                   factor_bounds = (1.08, 1.22))
    aggregate_capacity = n_bins * capacity
    factor = rand(rng, Uniform(factor_bounds...))
    total_size = aggregate_capacity * factor
    item_sizes .= _bounded_sizes_with_total(
        item_sizes, total_size, fill(0.82 * capacity, length(item_sizes)),
    )
    return aggregate_capacity
end


function _set_bin_packing_starts!(model::Model, prob)
    witness = prob.feasible_witness
    witness === nothing && return model

    used = falses(prob.n_bins)
    present = falses(prob.n_categories, prob.n_bins)
    for item in 1:prob.n_items
        bin = witness[item]
        used[bin] = true
        present[prob.item_categories[item], bin] = true
        for candidate_bin in 1:prob.n_bins
            set_start_value(
                model[:x][item, candidate_bin],
                candidate_bin == bin ? 1.0 : 0.0,
            )
        end
    end
    for bin in 1:prob.n_bins
        set_start_value(model[:y][bin], used[bin] ? 1.0 : 0.0)
        for category in 1:prob.n_categories
            set_start_value(
                model[:category_present][category, bin],
                present[category, bin] ? 1.0 : 0.0,
            )
        end
    end
    return model
end

"""
    BinPackingProblem(target_variables, feasibility_status, seed)

Construct an identical-bin instance using only a local RNG. The dimensions are
fixed before status-specific data generation, so requested status never causes
variable-count drift.
"""
function BinPackingProblem(target_variables::Int,
                           feasibility_status::FeasibilityStatus, seed::Int)
    rng = MersenneTwister(seed)
    n_items, n_bins, n_categories = _bin_packing_dimensions(target_variables)
    actual_variables = _bin_packing_variable_count(
        n_items, n_bins, n_categories,
    )

    bin_capacity = n_items <= 25 ? rand(rng, Uniform(80.0, 140.0)) :
                   n_items <= 100 ? rand(rng, Uniform(300.0, 650.0)) :
                   rand(rng, Uniform(900.0, 2_000.0))
    item_categories, category_names, incompatible_pairs =
        _packing_category_data(rng, n_items, n_categories)
    item_sizes = _sample_packing_sizes(rng, item_categories, bin_capacity)
    load_profile = feasibility_status == feasible ? :guaranteed_feasible :
                   feasibility_status == infeasible ? :aggregate_overload :
                   _packing_unknown_load_profile(target_variables, seed)

    feasible_witness = nothing
    infeasibility_certificate = nothing
    if feasibility_status == feasible
        bins = _fit_standard_packing!(
            item_sizes, bin_capacity, item_categories, incompatible_pairs,
            n_bins,
        )
        feasible_witness = _canonical_bin_assignment(bins, n_items)
    elseif feasibility_status == infeasible
        aggregate_capacity = _force_standard_overload!(
            rng, item_sizes, n_bins, bin_capacity,
        )
        infeasibility_certificate = BinPackingCapacityCertificate(
            sum(item_sizes), aggregate_capacity,
            sum(item_sizes) - aggregate_capacity,
        )
    elseif load_profile in (:light, :nominal)
        _fit_standard_packing!(
            item_sizes, bin_capacity, item_categories, incompatible_pairs,
            n_bins,
        )
        if load_profile == :light
            aggregate_capacity = n_bins * bin_capacity
            utilization = sum(item_sizes) / aggregate_capacity
            utilization > 0.68 && (item_sizes .*= 0.68 / utilization)
        end
    else
        _force_standard_overload!(
            rng, item_sizes, n_bins, bin_capacity;
            factor_bounds = (1.03, 1.09),
        )
    end

    problem = BinPackingProblem(
        n_items,
        n_bins,
        n_categories,
        item_sizes,
        bin_capacity,
        item_categories,
        incompatible_pairs,
        category_names,
        load_profile,
        target_variables,
        actual_variables,
        feasibility_status,
        feasible_witness,
        infeasibility_certificate,
    )
    if feasibility_status == feasible
        @assert validate_bin_packing_witness(problem)
    elseif feasibility_status == infeasible
        @assert validate_bin_packing_certificate(problem)
    end
    return problem
end

"""
    build_model(prob::BinPackingProblem)

Build the identical-bin MILP. Category-presence indicators have a two-sided LP
envelope: every assigned item implies presence, and presence cannot exceed the
category's total assignment. The chain `x[i,b] <= category_present[c(i),b] <=
y[b]` makes separate item-to-used-bin rows unnecessary even in the relaxation.
Canonical bin ordering removes label symmetries without excluding any unlabeled
packing.
"""
function build_model(prob::BinPackingProblem)
    model = Model()
    I = 1:prob.n_items
    B = 1:prob.n_bins
    C = 1:prob.n_categories
    category_items = [findall(==(category), prob.item_categories)
                      for category in C]

    @variable(model, x[I, B], Bin)
    @variable(model, y[B], Bin)
    @variable(model, category_present[C, B], Bin)

    @objective(model, Min, sum(y[bin] for bin in B))

    @constraint(model, item_assignment[item in I],
        sum(x[item, bin] for bin in B) == 1)
    @constraint(model, bin_capacity[bin in B],
        sum(prob.item_sizes[item] * x[item, bin] for item in I) <=
        prob.bin_capacity * y[bin])

    @constraint(model, presence_lower[item in I, bin in B],
        x[item, bin] <= category_present[prob.item_categories[item], bin])
    @constraint(model, presence_upper[category in C, bin in B],
        category_present[category, bin] <=
        sum(x[item, bin] for item in category_items[category]))
    @constraint(model, presence_used[category in C, bin in B],
        category_present[category, bin] <= y[bin])

    @constraint(model,
        category_conflict[pair in eachindex(prob.incompatible_pairs), bin in B],
        category_present[prob.incompatible_pairs[pair][1], bin] +
        category_present[prob.incompatible_pairs[pair][2], bin] <= 1)

    # Identical-bin symmetry: open a prefix of bins, and label that prefix by
    # increasing smallest item index. Any unlabeled packing has this canonical
    # representation.
    if prob.n_bins >= 2
        @constraint(model, used_prefix[bin in 1:(prob.n_bins - 1)],
            y[bin] >= y[bin + 1])
    end
    @constraint(model, canonical_label[item in I, bin in B; bin > item],
        x[item, bin] == 0)

    _set_bin_packing_starts!(model, prob)
    return model
end

register_variant(
    :bin_packing,
    :standard,
    BinPackingProblem,
    "Identical-bin packing with handling conflicts, two-sided category links, constructive witnesses, and aggregate capacity certificates",
    default = true,
)
