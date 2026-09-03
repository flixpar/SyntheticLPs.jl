using JuMP
using Random
using Distributions
using LinearAlgebra
using SparseArrays

"""
Certificate that a positive packing decision is strictly interior.
"""
struct PackingInteriorCertificate
    slacks::Vector{Float64}
end

"""
Certificate that every admissible normalized objective incurs an average
duality gap above `lower_bound`, contradicting `tolerance`.
"""
struct GapToleranceCertificate
    lower_bound::Float64
    tolerance::Float64
end

"""
Shared sparse packing-system data used by the exact and panel variants.
"""
struct InversePackingData
    consumption::SparseMatrixCSC{Float64, Int}
    true_cost::Vector{Float64}
    true_dual::Vector{Float64}
    prior_cost::Vector{Float64}
    cost_lower::Vector{Float64}
    cost_upper::Vector{Float64}
    deviation_weight::Vector{Float64}
end

function _inverse_sparse_consumption(rng::AbstractRNG, n_resources::Int, n_activities::Int)
    rows = Int[]
    columns = Int[]
    values = Float64[]
    for j in 1:n_activities
        # Every resource appears before the remaining sparse supports are
        # sampled. Activities use a small bundle of inputs, as in product-mix
        # and production-planning matrices.
        mandatory = j <= n_resources ? j : 0
        width = rand(rng, 1:min(n_resources, 4))
        support = mandatory == 0 ? Int[] : [mandatory]
        candidates = randperm(rng, n_resources)
        for i in candidates
            length(support) >= width && break
            i in support || push!(support, i)
        end
        sort!(support)
        for i in support
            push!(rows, i)
            push!(columns, j)
            # Positive, right-skewed technological coefficients with moderate
            # dispersion; the median is close to one input unit.
            push!(values, rand(rng, LogNormal(0.0, 0.42)))
        end
    end
    return sparse(rows, columns, values, n_resources, n_activities)
end

function _inverse_packing_data(rng::AbstractRNG, n_resources::Int, n_activities::Int)
    A = _inverse_sparse_consumption(rng, n_resources, n_activities)
    raw_dual = rand(rng, LogNormal(0.0, 0.55), n_resources)
    raw_cost = transpose(A) * raw_dual
    scale = sum(raw_cost)
    true_cost = Vector(raw_cost ./ scale)
    true_dual = raw_dual ./ scale

    prior_cost = true_cost .* rand(rng, LogNormal(0.0, 0.34), n_activities)
    prior_cost ./= sum(prior_cost)
    cost_lower = 0.20 .* min.(true_cost, prior_cost)
    cost_upper = 2.80 .* max.(true_cost, prior_cost)
    deviation_weight = 1.0 ./ max.(prior_cost, 1.0e-4)
    return InversePackingData(
        A, true_cost, true_dual, prior_cost, cost_lower, cost_upper, deviation_weight
    )
end

function _inverse_column_expression(A, dual, j::Int)
    rows, coefficients = findnz(@view A[:, j])
    return sum(coefficients[k] * dual[rows[k]] for k in eachindex(rows))
end

function _inverse_row_expression(A, decision, i::Int)
    columns, coefficients = findnz(@view A[i, :])
    return sum(coefficients[k] * decision[columns[k]] for k in eachindex(columns))
end

function _packing_interior_certificate_is_valid(certificate)
    return certificate isa PackingInteriorCertificate && all(>(0.0), certificate.slacks)
end

function _gap_tolerance_certificate_is_valid(certificate)
    return certificate isa GapToleranceCertificate &&
           certificate.lower_bound > certificate.tolerance >= 0.0
end
