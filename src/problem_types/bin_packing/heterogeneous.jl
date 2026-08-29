using JuMP
using Random
using Distributions

"""
    HeterogeneousBinPackingProblem <: ProblemGenerator

Pack scalar-size consignments into an available fleet of typed bins. Bin types
have distinct capacities, fixed activation costs, availability counts, and
handling-category eligibility. This is a fleet-selection packing model, rather
than the multidimensional or geometric loading formulations in
`container_loading`.
"""
struct HeterogeneousBinPackingProblem <: ProblemGenerator
    n_items::Int
    n_bins::Int
    n_categories::Int
    n_bin_types::Int
    item_sizes::Vector{Float64}
    item_categories::Vector{Int}
    category_names::Vector{String}
    incompatible_pairs::Vector{Tuple{Int,Int}}
    bin_types::Vector{Int}
    bin_type_names::Vector{String}
    type_capacities::Vector{Float64}
    type_costs::Vector{Float64}
    type_availability::Vector{Int}
    type_category_compatibility::BitMatrix
    load_profile::Symbol
    target_variables::Int
    actual_variables::Int
    feasibility_status::FeasibilityStatus
    feasible_witness::Union{Nothing,Vector{Int}}
    infeasibility_certificate::Union{Nothing,BinPackingCapacityCertificate}
end

function _heterogeneous_type_data(rng::AbstractRNG, n_bins::Int,
                                  n_categories::Int, base_capacity::Float64)
    n_bin_types = min(4, n_bins)
    if n_bin_types == 2
        names = ["Standard", "Controlled_Specialty"]
        capacity_factors = [1.00, 0.86]
        cost_factors = [1.00, 1.52]
    elseif n_bin_types == 3
        names = ["Standard", "Refrigerated", "Hazmat"]
        capacity_factors = [1.00, 0.86, 0.74]
        cost_factors = [1.00, 1.42, 1.78]
    else
        names = ["Standard", "Refrigerated", "Hazmat", "High_Cube"]
        capacity_factors = [1.00, 0.86, 0.74, 1.38]
        cost_factors = [1.00, 1.42, 1.78, 1.24]
    end
    capacities = base_capacity .* capacity_factors
    base_cost = rand(rng, Uniform(90.0, 210.0))
    costs = base_cost .* cost_factors

    # Balanced slot counts make every advertised type genuinely available.
    availability = fill(div(n_bins, n_bin_types), n_bin_types)
    extra_order = randperm(rng, n_bin_types)
    for index in 1:rem(n_bins, n_bin_types)
        availability[extra_order[index]] += 1
    end
    bin_types = reduce(vcat, [fill(bin_type, availability[bin_type])
                              for bin_type in 1:n_bin_types])

    compatibility = falses(n_bin_types, n_categories)
    if n_bin_types == 2
        # General freight versus temperature-controlled / regulated freight.
        for category in 1:n_categories
            compatibility[1, category] = category in (1, 3, 4, 7, 8)
            compatibility[2, category] = true
        end
    else
        for category in 1:n_categories
            compatibility[1, category] = category in (1, 3, 4, 7, 8)
            compatibility[2, category] = category in (1, 3, 5, 6, 7, 8)
            compatibility[3, category] = category in (2, 4, 7, 8)
            if n_bin_types >= 4
                compatibility[4, category] = category in (1, 3, 4, 7, 8)
            end
        end
    end
    all(any(view(compatibility, :, category)) for category in 1:n_categories) ||
        error("Every item category must have an eligible bin type")
    return names, capacities, costs, availability, bin_types, compatibility
end

function _heterogeneous_greedy_packing(
    item_sizes::Vector{Float64},
    item_categories::Vector{Int},
    incompatible_pairs::Vector{Tuple{Int,Int}},
    bin_types::Vector{Int},
    type_capacities::Vector{Float64},
    compatibility::BitMatrix,
)
    n_bins = length(bin_types)
    loads = zeros(Float64, n_bins)
    bin_categories = [Set{Int}() for _ in 1:n_bins]
    conflicts = _packing_conflict_set(incompatible_pairs)
    assignment = zeros(Int, length(item_sizes))
    # Place restricted categories before flexible freight; otherwise a general
    # item can consume the only specialist slot needed by a later item.
    eligible_slot_count(item) = count(
        compatibility[bin_types[bin], item_categories[item]]
        for bin in 1:n_bins
    )
    order = sortperm(eachindex(item_sizes);
        by = item -> (eligible_slot_count(item), -item_sizes[item], item))

    for item in order
        category = item_categories[item]
        best_bin = 0
        best_remaining = Inf
        for bin in 1:n_bins
            bin_type = bin_types[bin]
            compatibility[bin_type, category] || continue
            capacity = type_capacities[bin_type]
            new_load = loads[bin] + item_sizes[item]
            new_load <= capacity + 1e-10 || continue
            conflict = any(
                (min(category, other), max(category, other)) in conflicts
                for other in bin_categories[bin] if other != category
            )
            conflict && continue
            remaining = capacity - new_load
            if remaining < best_remaining
                best_remaining = remaining
                best_bin = bin
            end
        end
        best_bin == 0 && return nothing, loads
        assignment[item] = best_bin
        loads[best_bin] += item_sizes[item]
        push!(bin_categories[best_bin], category)
    end
    return assignment, loads
end

function _fit_heterogeneous_packing!(
    item_sizes::Vector{Float64},
    item_categories::Vector{Int},
    incompatible_pairs::Vector{Tuple{Int,Int}},
    bin_types::Vector{Int},
    type_capacities::Vector{Float64},
    compatibility::BitMatrix,
)
    assignment = nothing
    loads = Float64[]
    for _ in 1:70
        assignment, loads = _heterogeneous_greedy_packing(
            item_sizes, item_categories, incompatible_pairs, bin_types,
            type_capacities, compatibility,
        )
        assignment !== nothing && break
        item_sizes .*= 0.92
    end
    assignment === nothing && error("Could not construct heterogeneous witness")

    maximum_utilization = maximum(
        loads[bin] / type_capacities[bin_types[bin]]
        for bin in eachindex(bin_types)
    )
    if maximum_utilization < 0.74
        item_sizes .*= 0.80 / maximum_utilization
        assignment, loads = _heterogeneous_greedy_packing(
            item_sizes, item_categories, incompatible_pairs, bin_types,
            type_capacities, compatibility,
        )
        assignment === nothing && error("Tightened heterogeneous witness failed")
    end
    return assignment
end


_heterogeneous_aggregate_capacity(bin_types::Vector{Int},
                                  type_capacities::Vector{Float64}) =
    sum(type_capacities[bin_type] for bin_type in bin_types)


function _heterogeneous_size_upper_bounds(
    item_categories::Vector{Int},
    type_capacities::Vector{Float64},
    compatibility::BitMatrix,
)
    return [
        0.86 * maximum(type_capacities[bin_type]
                       for bin_type in eachindex(type_capacities)
                       if compatibility[bin_type, item_categories[item]])
        for item in eachindex(item_categories)
    ]
end


function _force_heterogeneous_overload!(
    rng::AbstractRNG,
    item_sizes::Vector{Float64},
    item_categories::Vector{Int},
    bin_types::Vector{Int},
    type_capacities::Vector{Float64},
    compatibility::BitMatrix;
    factor_bounds = (1.07, 1.18),
)
    aggregate_capacity = _heterogeneous_aggregate_capacity(
        bin_types, type_capacities,
    )
    upper_bounds = _heterogeneous_size_upper_bounds(
        item_categories, type_capacities, compatibility,
    )
    desired_total = aggregate_capacity * rand(rng, Uniform(factor_bounds...))
    total_size = min(desired_total, 0.97 * sum(upper_bounds))
    total_size > aggregate_capacity ||
        error("Fleet dimensions leave no room for an aggregate overload")
    item_sizes .= _bounded_sizes_with_total(
        item_sizes, total_size, upper_bounds,
    )
    return aggregate_capacity
end

# Candidate bins of the same type are interchangeable. Remap used slots to a
# prefix within each type so the stored witness also satisfies model symmetry.
function _canonicalize_heterogeneous_assignment(assignment::Vector{Int},
                                                bin_types::Vector{Int},
                                                n_bin_types::Int)
    canonical = copy(assignment)
    for bin_type in 1:n_bin_types
        type_slots = findall(==(bin_type), bin_types)
        used_slots = sort(unique(bin for bin in assignment
                                 if bin_types[bin] == bin_type))
        mapping = Dict(used_slots[index] => type_slots[index]
                       for index in eachindex(used_slots))
        for item in eachindex(canonical)
            bin_types[assignment[item]] == bin_type || continue
            canonical[item] = mapping[assignment[item]]
        end
    end
    return canonical
end

"""
    validate_bin_packing_witness(prob::HeterogeneousBinPackingProblem) -> Bool

Validate typed-bin eligibility, capacity, conflicts, availability, and the
within-type prefix symmetry of a stored assignment.
"""
function validate_bin_packing_witness(prob::HeterogeneousBinPackingProblem)
    witness = prob.feasible_witness
    witness === nothing && return false
    length(witness) == prob.n_items || return false
    all(bin -> 1 <= bin <= prob.n_bins, witness) || return false
    length(prob.bin_types) == prob.n_bins || return false
    count_by_type = [count(==(bin_type), prob.bin_types)
                     for bin_type in 1:prob.n_bin_types]
    count_by_type == prob.type_availability || return false

    conflicts = _packing_conflict_set(prob.incompatible_pairs)
    for bin in sort(unique(witness))
        items = findall(==(bin), witness)
        bin_type = prob.bin_types[bin]
        all(prob.type_category_compatibility[
                bin_type, prob.item_categories[item],
            ] for item in items) || return false
        load = sum(prob.item_sizes[item] for item in items)
        load <= prob.type_capacities[bin_type] + 1e-8 || return false
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

    for bin_type in 1:prob.n_bin_types
        slots = findall(==(bin_type), prob.bin_types)
        used_slots = [slot for slot in slots if slot in witness]
        used_slots == slots[1:length(used_slots)] || return false
    end
    return true
end

"""
    validate_bin_packing_certificate(prob::HeterogeneousBinPackingProblem) -> Bool

Recompute the total capacity of all available typed slots and validate the
stored aggregate contradiction.
"""
function validate_bin_packing_certificate(prob::HeterogeneousBinPackingProblem)
    certificate = prob.infeasibility_certificate
    certificate === nothing && return false
    length(prob.bin_types) == prob.n_bins || return false
    length(prob.type_availability) == prob.n_bin_types || return false
    length(prob.type_capacities) == prob.n_bin_types || return false
    all(bin_type -> 1 <= bin_type <= prob.n_bin_types, prob.bin_types) ||
        return false
    observed_availability = [count(==(bin_type), prob.bin_types)
                             for bin_type in 1:prob.n_bin_types]
    observed_availability == prob.type_availability || return false
    demand = sum(prob.item_sizes)
    capacity = _heterogeneous_aggregate_capacity(
        prob.bin_types, prob.type_capacities,
    )
    return isapprox(certificate.total_item_size, demand; atol = 1e-8) &&
           isapprox(certificate.total_available_capacity, capacity; atol = 1e-8) &&
           isapprox(certificate.excess, demand - capacity; atol = 1e-8) &&
           certificate.excess > 1e-8
end

"""
    HeterogeneousBinPackingProblem(target_variables, feasibility_status, seed)

Construct a typed-fleet instance with exact size metadata and local RNG use.
"""
function HeterogeneousBinPackingProblem(
    target_variables::Int,
    feasibility_status::FeasibilityStatus,
    seed::Int,
)
    rng = MersenneTwister(seed)
    n_items, n_bins, n_categories = _bin_packing_dimensions(target_variables)
    actual_variables = _bin_packing_variable_count(
        n_items, n_bins, n_categories,
    )
    base_capacity = n_items <= 25 ? rand(rng, Uniform(85.0, 145.0)) :
                    n_items <= 100 ? rand(rng, Uniform(320.0, 700.0)) :
                    rand(rng, Uniform(950.0, 2_100.0))

    item_categories, category_names, incompatible_pairs =
        _packing_category_data(rng, n_items, n_categories)
    item_sizes = _sample_packing_sizes(rng, item_categories, base_capacity)
    bin_type_names, type_capacities, type_costs, type_availability,
        bin_types, compatibility = _heterogeneous_type_data(
            rng, n_bins, n_categories, base_capacity,
        )
    n_bin_types = length(bin_type_names)
    load_profile = feasibility_status == feasible ? :guaranteed_feasible :
                   feasibility_status == infeasible ? :aggregate_overload :
                   _packing_unknown_load_profile(target_variables, seed)

    feasible_witness = nothing
    infeasibility_certificate = nothing
    if feasibility_status == feasible
        assignment = _fit_heterogeneous_packing!(
            item_sizes, item_categories, incompatible_pairs, bin_types,
            type_capacities, compatibility,
        )
        feasible_witness = _canonicalize_heterogeneous_assignment(
            assignment, bin_types, n_bin_types,
        )
    elseif feasibility_status == infeasible
        aggregate_capacity = _force_heterogeneous_overload!(
            rng, item_sizes, item_categories, bin_types, type_capacities,
            compatibility,
        )
        infeasibility_certificate = BinPackingCapacityCertificate(
            sum(item_sizes), aggregate_capacity,
            sum(item_sizes) - aggregate_capacity,
        )
    elseif load_profile in (:light, :nominal)
        _fit_heterogeneous_packing!(
            item_sizes, item_categories, incompatible_pairs, bin_types,
            type_capacities, compatibility,
        )
        if load_profile == :light
            aggregate_capacity = _heterogeneous_aggregate_capacity(
                bin_types, type_capacities,
            )
            utilization = sum(item_sizes) / aggregate_capacity
            utilization > 0.68 && (item_sizes .*= 0.68 / utilization)
        end
    else
        _force_heterogeneous_overload!(
            rng, item_sizes, item_categories, bin_types, type_capacities,
            compatibility;
            factor_bounds = (1.03, 1.09),
        )
    end

    problem = HeterogeneousBinPackingProblem(
        n_items,
        n_bins,
        n_categories,
        n_bin_types,
        item_sizes,
        item_categories,
        category_names,
        incompatible_pairs,
        bin_types,
        bin_type_names,
        type_capacities,
        type_costs,
        type_availability,
        compatibility,
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

function build_model(prob::HeterogeneousBinPackingProblem)
    model = Model()
    I = 1:prob.n_items
    B = 1:prob.n_bins
    C = 1:prob.n_categories
    category_items = [findall(==(category), prob.item_categories)
                      for category in C]

    @variable(model, x[I, B], Bin)
    @variable(model, y[B], Bin)
    @variable(model, category_present[C, B], Bin)

    @objective(model, Min,
        sum(prob.type_costs[prob.bin_types[bin]] * y[bin] for bin in B))

    @constraint(model, item_assignment[item in I],
        sum(x[item, bin] for bin in B) == 1)
    @constraint(model, bin_capacity[bin in B],
        sum(prob.item_sizes[item] * x[item, bin] for item in I) <=
        prob.type_capacities[prob.bin_types[bin]] * y[bin])

    @constraint(model,
        category_eligibility[item in I, bin in B;
            !prob.type_category_compatibility[
                prob.bin_types[bin], prob.item_categories[item],
            ]],
        x[item, bin] == 0)

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

    # Only bins of the same type are interchangeable. Prefix ordering within
    # each type removes that symmetry without imposing an invalid global order
    # across differently priced and sized bins.
    type_prefix_pairs = Tuple{Int,Int}[]
    for bin_type in 1:prob.n_bin_types
        slots = findall(==(bin_type), prob.bin_types)
        append!(type_prefix_pairs,
                [(slots[index], slots[index + 1])
                 for index in 1:(length(slots) - 1)])
    end
    @constraint(model, used_type_prefix[index in eachindex(type_prefix_pairs)],
        y[type_prefix_pairs[index][1]] >= y[type_prefix_pairs[index][2]])

    _set_bin_packing_starts!(model, prob)
    return model
end

register_variant(
    :bin_packing,
    :heterogeneous,
    HeterogeneousBinPackingProblem,
    "Typed-fleet bin packing with type-specific capacity, fixed cost, availability, and handling eligibility",
)
