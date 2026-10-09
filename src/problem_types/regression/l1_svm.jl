using JuMP
using Random
using Distributions
using SparseArrays
using Statistics

"""
    L1SVMWitness

Planted feasible point of an [`L1SVMProblem`](@ref): the data-generating linear
classifier. Its hinge losses `ξ_i = max(0, 1 − y_i (x_i·w + bias))` sum to
`hinge`, and `hinge < hinge_budget` with a planted margin.
"""
struct L1SVMWitness
    w::Vector{Float64}
    bias::Float64
    hinge::Float64
end

"""
    L1SVMCertificate

Infeasibility proof for an [`L1SVMProblem`](@ref). Every `(a, b) ∈ pairs` is a
conflicting duplicate: identical feature rows with labels `+1` and `−1`. Adding
their margin rows `ξ_a + (x·w + bias) ≥ 1` and `ξ_b − (x·w + bias) ≥ 1` cancels
the classifier, so `ξ_a + ξ_b ≥ 2` for every feasible point. Summing over the
disjoint pairs gives `Σ ξ ≥ lower_bound = 2·|pairs|`, contradicting the budget
row `Σ ξ ≤ budget < lower_bound`.
"""
struct L1SVMCertificate
    pairs::Vector{Tuple{Int, Int}}
    lower_bound::Float64
    budget::Float64
end

"""
    L1SVMProblem <: ProblemGenerator

1-norm (L1-regularized) linear support vector machine for sparse text
classification, with a total hinge-loss budget.

# Data profile

Bag-of-words documents: each document has a log-normal number of distinct terms
drawn from a Zipf vocabulary (every term appears in at least three documents —
minimum document-frequency pruning), weighted by `(1 + log tf)·idf` and
L2-normalized per document (tf-idf). Labels come from a sparse planted linear
classifier on mid-frequency terms with logistic label noise and class
imbalance. A small fraction of documents are near-duplicates re-posted with the
opposite label (conflicting duplicates — common in crawled corpora).

# Formulation

```math
\\min λ \\sum_j (w^+_j + w^-_j) + \\sum_i c_i ξ_i \\quad\\text{s.t.}\\quad
y_i\\big(x_i·(w^+ - w^-) + b\\big) + ξ_i \\ge 1,\\quad \\sum_i ξ_i \\le B,
```

with `w^±, ξ ≥ 0`, free bias `b`, and class-balancing weights `c_i`.

# Feasibility

  - `feasible`: `B` is 1.05–1.2× the planted classifier's hinge loss
    (`feasible_witness`).
  - `infeasible`: `B` is 70–90% of the conflicting-duplicate bound `2·|pairs|`
    (`infeasibility_certificate`).
  - `unknown`: `B` lies between the duplicate bound and the planted hinge loss;
    whether the hinge-minimizing classifier fits under it depends on the data.

# Sizing

Variables = `2·n_terms + 1 + n_documents`, exact for targets ≥ 20. Rows =
`n_documents + 1`; nonzeros ≈ `2·n_documents·terms per document`
(≈ 2–4M at 100k variables).
"""
struct L1SVMProblem <: ProblemGenerator
    n_documents::Int
    n_terms::Int
    Xt::SparseMatrixCSC{Float64, Int}
    labels::Vector{Float64}
    class_weights::Vector{Float64}
    penalty::Float64
    hinge_budget::Float64
    conflict_pairs::Vector{Tuple{Int, Int}}
    feasible_witness::Union{Nothing, L1SVMWitness}
    infeasibility_certificate::Union{Nothing, L1SVMCertificate}
end

"""
    L1SVMProblem(target_variables, feasibility_status, seed)

Construct a 1-norm SVM text-classification instance with a constructor-local
RNG. See the type docstring for the data profile, sizing, and contracts.
"""
function L1SVMProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    rng = MersenneTwister(seed)
    V = max(target_variables, 20)

    # --- Dimensions: V = 2p + 1 + m exactly, with more documents than terms. ---
    ratio = rand(rng, Uniform(3.0, 6.0))
    n_terms = max(3, round(Int, (V - 1) / (2 + ratio)))
    n_documents = V - 1 - 2 * n_terms

    # --- Documents: Zipf vocabulary, log-normal lengths, tf-idf rows. ---
    n_conflicts = clamp(
        round(Int, n_documents * rand(rng, Uniform(0.01, 0.03))), 1, n_documents ÷ 4
    )
    n_base = n_documents - n_conflicts
    mean_len = rand(rng, Uniform(15.0, 45.0))
    lengths = [
        clamp(round(Int, mean_len * exp(0.5 * randn(rng) - 0.125)), 3, n_terms) for _ in 1:n_base
    ]
    terms = _regression_sparse_incidence(
        rng, n_base, n_terms, lengths, rand(rng, Uniform(0.9, 1.15)), min(3, n_base)
    )
    df = zeros(Int, n_terms)
    for doc in terms, j in doc
        df[j] += 1
    end
    idf = log.((n_base + 1) ./ (df .+ 1)) .+ 1.0

    I = Int[]
    J = Int[]
    Vals = Float64[]
    starts = Vector{Int}(undef, n_base + 1)
    for (i, doc) in enumerate(terms)
        starts[i] = length(I) + 1
        tf = 1.0 .+ log.(rand(rng, Geometric(0.5), length(doc)) .+ 1.0)
        row = tf .* idf[doc]
        row ./= norm(row)
        append!(I, doc)
        append!(J, fill(i, length(doc)))
        append!(Vals, row)
    end
    starts[n_base + 1] = length(I) + 1

    # Conflicting duplicates: exact copies of base documents (labels set below).
    originals = _regression_distinct(rng, n_base, n_conflicts)
    for (k, i) in enumerate(originals)
        span = starts[i]:(starts[i + 1] - 1)
        append!(I, I[span])
        append!(J, fill(n_base + k, length(span)))
        append!(Vals, Vals[span])
    end
    Xt = sparse(I, J, Vals, n_terms, n_documents)

    # --- Planted sparse classifier on mid-frequency terms, noisy labels. ---
    w = zeros(Float64, n_terms)
    order = sortperm(df; rev=true)
    lo = max(1, round(Int, 0.02 * n_terms))
    candidates = order[lo:end]
    n_informative = clamp(
        round(Int, n_terms * rand(rng, Uniform(0.05, 0.12))), 1, length(candidates)
    )
    informative = candidates[_regression_distinct(rng, length(candidates), n_informative)]
    w[informative] .= randn(rng, n_informative)
    scores = transpose(Xt) * w
    spread = max(std(scores), 1e-8)
    w .*= rand(rng, Uniform(2.0, 4.0)) / spread
    scores = transpose(Xt) * w
    imbalance = rand(rng, Uniform(0.15, 0.5))              # share of the minority class
    bias = -quantile(scores, 1 - imbalance)
    noise_scale = rand(rng, Uniform(0.3, 1.0))
    labels = Vector{Float64}(undef, n_documents)
    for i in 1:n_base
        labels[i] = scores[i] + bias + noise_scale * rand(rng, Logistic()) >= 0 ? 1.0 : -1.0
    end
    conflict_pairs = Tuple{Int, Int}[]
    for (k, i) in enumerate(originals)
        labels[n_base + k] = -labels[i]
        push!(conflict_pairs, labels[i] > 0 ? (i, n_base + k) : (n_base + k, i))
    end

    n_pos = count(>(0), labels)
    n_neg = n_documents - n_pos
    class_weights = [
        labels[i] > 0 ? n_documents / (2 * max(n_pos, 1)) : n_documents / (2 * max(n_neg, 1)) for
        i in 1:n_documents
    ]
    mean_entry = sum(Vals) / length(Vals)
    penalty = rand(rng, Uniform(0.1, 0.6)) * 3 * mean_entry

    hinge = sum(max(0.0, 1 - labels[i] * (scores[i] + bias)) for i in 1:n_documents)
    lower_bound = 2.0 * n_conflicts

    witness = nothing
    certificate = nothing
    if feasibility_status == feasible
        hinge_budget = hinge * rand(rng, Uniform(1.05, 1.2))
        witness = L1SVMWitness(w, bias, hinge)
    elseif feasibility_status == infeasible
        hinge_budget = lower_bound * rand(rng, Uniform(0.7, 0.9))
        certificate = L1SVMCertificate(conflict_pairs, lower_bound, hinge_budget)
    else
        hinge_budget = lower_bound + rand(rng, Uniform(0.25, 0.75)) * (hinge - lower_bound)
    end

    return L1SVMProblem(
        n_documents,
        n_terms,
        Xt,
        labels,
        class_weights,
        penalty,
        hinge_budget,
        conflict_pairs,
        witness,
        certificate,
    )
end

"""
    build_model(prob::L1SVMProblem)

Build the 1-norm SVM LP. Deterministic and linear in the number of nonzeros.
"""
function build_model(prob::L1SVMProblem)
    model = Model()
    p = prob.n_terms
    m = prob.n_documents

    @variable(model, w_pos[1:p] >= 0)
    @variable(model, w_neg[1:p] >= 0)
    @variable(model, bias)
    @variable(model, xi[1:m] >= 0)

    objective = AffExpr(0.0)
    sizehint!(objective.terms, 2p + m)
    for j in 1:p
        add_to_expression!(objective, prob.penalty, w_pos[j])
        add_to_expression!(objective, prob.penalty, w_neg[j])
    end
    for i in 1:m
        add_to_expression!(objective, prob.class_weights[i], xi[i])
    end
    @objective(model, Min, objective)

    rows = rowvals(prob.Xt)
    vals = nonzeros(prob.Xt)
    for i in 1:m
        yi = prob.labels[i]
        range = nzrange(prob.Xt, i)
        expr = AffExpr(0.0)
        sizehint!(expr.terms, 2 * length(range) + 2)
        for ptr in range
            add_to_expression!(expr, yi * vals[ptr], w_pos[rows[ptr]])
            add_to_expression!(expr, -yi * vals[ptr], w_neg[rows[ptr]])
        end
        add_to_expression!(expr, yi, bias)
        add_to_expression!(expr, 1.0, xi[i])
        @constraint(model, expr >= 1.0)
    end

    budget = AffExpr(0.0)
    sizehint!(budget.terms, m)
    for i in 1:m
        add_to_expression!(budget, 1.0, xi[i])
    end
    @constraint(model, hinge_budget, budget <= prob.hinge_budget)

    return model
end

register_variant(
    :regression,
    :l1_svm,
    L1SVMProblem,
    "1-norm linear SVM text classification on sparse tf-idf features with a hinge-loss budget";
    tags=[:machine_learning],
)
