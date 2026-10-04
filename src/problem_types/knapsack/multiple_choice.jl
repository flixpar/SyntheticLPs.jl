using JuMP
using Random

"""
    MultipleChoiceWitness

Planted configuration for a `feasible` multiple-choice instance: `choice[g]`
is the global column index of the option selected for class `g` (exactly one
per class). Every cluster and shared capacity was drawn as this choice's usage
times a factor `>= 1.02`, so the 0/1 point is feasible with positive slack.
"""
struct MultipleChoiceWitness
    choice::Vector{Int}
end

"""
    MinimumLevelCertificate

Relaxation-valid infeasibility proof: every class must select exactly one
option (equality rows), and even its least-consuming option of shared resource
`resource` uses `min_usage[g]`, so any fractional selection consumes at least
`min_usage_total = sum(min_usage)` of that resource, while its capacity is
`capacity <= min_usage_total / 1.04`. The Farkas combination is the shared
row plus `min_usage[g]` times each class row — one row per class, so the
contradiction is spread over the whole model rather than a single row.
"""
struct MinimumLevelCertificate
    resource::Int
    min_usage::Vector{Float64}
    min_usage_total::Float64
    capacity::Float64
end

"""
    MultipleChoiceKnapsackProblem <: ProblemGenerator

Multiple-choice multi-dimensional knapsack (MMKP): workloads choose exactly one
service configuration each, under per-cluster and shared resource limits.

# Overview

Each class `g` is a workload homed on one cluster with `3-8` candidate
configurations (service levels). A configuration consumes the cluster's three
local resources (CPU, memory, storage I/O) and two shared site resources
(power, WAN bandwidth). Higher service levels consume more and are worth more,
with diminishing returns (`value ~ level^0.6`); within a level the
configurations trade resources against each other (a lognormal tilt
normalised per option), so the least-consuming option differs by resource and
no option is uniformly cheapest. This is the classic MMKP benchmark structure
(Khan et al.; Hifi et al.) with sparse cluster-local resources so the model
scales: every column has 6 nonzeros (one class row, three cluster rows, two
shared rows).

```text
max  sum_{g,k} v_gk x_gk
s.t. sum_k x_gk = 1                         for every class g
     sum_{g on c, k} a_gkr x_gk <= C_cr      for every cluster c, local resource r
     sum_{g,k} a_gks x_gk <= S_s             for every shared resource s
     x binary  (relaxed to [0, 1] by default)
```

Sizing: classes take `3-8` options until the target is met, so the variable
count equals `target_variables` exactly; `n_clusters = clamp(round(G / 25), 1, G)`.
Rows: `G + 3 * n_clusters + 2`, about 20% of the columns.

# Feasibility

  - `feasible`: a configuration is planted per class; cluster capacities are
    its usage times `U(1.03, 1.20)`, shared capacities times `U(1.02, 1.10)`.
    The LP optimum wants richer configurations, so capacity rows bind.
    Witness: `MultipleChoiceWitness`.
  - `infeasible`: same instance, then one shared resource's capacity is cut to
    `min_usage_total / U(1.04, 1.12)` — a site power curtailment below what even
    the leanest configurations need. Certificate: `MinimumLevelCertificate`.
  - `unknown`: every capacity is placed between the total of per-class minima
    and the total at the median service level (cluster rows `U(0.25, 0.85)`,
    shared rows `U(0.0, 0.45)` of that span). Each row alone is satisfiable;
    whether one selection satisfies all of them jointly is decided by the LP.
"""
struct MultipleChoiceKnapsackProblem <: ProblemGenerator
    n_classes::Int
    n_clusters::Int
    class_cluster::Vector{Int}
    class_options::Vector{UnitRange{Int}}
    values::Vector{Float64}
    local_usage::Matrix{Float64}    # 3 x n
    shared_usage::Matrix{Float64}   # 2 x n
    local_capacity::Matrix{Float64} # 3 x n_clusters
    shared_capacity::Vector{Float64}
    feasibility_status::FeasibilityStatus
    feasible_witness::Union{Nothing, MultipleChoiceWitness}
    infeasibility_certificate::Union{Nothing, MinimumLevelCertificate}
end

const MCK_LOCAL = 3
const MCK_SHARED = 2

# Split n columns into classes of 3-8 options (exact total).
function _mck_class_sizes(rng::AbstractRNG, n::Int)
    sizes = Int[]
    remaining = n
    while remaining >= 11
        k = rand(rng, 3:8)
        push!(sizes, k)
        remaining -= k
    end
    if remaining >= 2
        if remaining <= 8
            push!(sizes, remaining)
        else
            push!(sizes, remaining ÷ 2, remaining - remaining ÷ 2)
        end
    elseif remaining == 1
        isempty(sizes) ? push!(sizes, 1) : (sizes[end] += 1)
    end
    return sizes
end

function _mck_totals(usage::Matrix{Float64}, pick::Vector{Int}, group_of::Vector{Int}, n_groups::Int)
    tot = zeros(size(usage, 1), n_groups)
    for (g, j) in enumerate(pick)
        for r in 1:size(usage, 1)
            tot[r, group_of[g]] += usage[r, j]
        end
    end
    return tot
end

function MultipleChoiceKnapsackProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be positive (got $target_variables)"))
    rng = MersenneTwister(seed)
    n = target_variables

    sizes = _mck_class_sizes(rng, n)
    G = length(sizes)
    class_options = Vector{UnitRange{Int}}(undef, G)
    start = 1
    for g in 1:G
        class_options[g] = start:(start + sizes[g] - 1)
        start += sizes[g]
    end
    n_clusters = clamp(round(Int, G / 25), 1, G)
    class_cluster = [mod(g - 1, n_clusters) + 1 for g in 1:G]
    shuffle!(rng, class_cluster)

    # Resource profiles: cluster hardware differs, workloads differ in size.
    local_profile = [0.6 + 0.8 * rand(rng) for _ in 1:MCK_LOCAL]
    shared_profile = [0.6 + 0.8 * rand(rng) for _ in 1:MCK_SHARED]
    values = Vector{Float64}(undef, n)
    local_usage = zeros(MCK_LOCAL, n)
    shared_usage = zeros(MCK_SHARED, n)
    for g in 1:G
        size_g = exp(0.6 * randn(rng))                    # workload scale
        mix_l = [exp(0.3 * randn(rng)) for _ in 1:MCK_LOCAL]  # workload resource mix
        mix_s = [exp(0.3 * randn(rng)) for _ in 1:MCK_SHARED]
        worth = 100.0 * size_g * exp(0.4 * randn(rng))
        level = 1.0
        for (k, j) in enumerate(class_options[g])
            k > 1 && (level += 0.3 + 0.5 * rand(rng))
            tilt = [exp(0.35 * randn(rng)) for _ in 1:(MCK_LOCAL + MCK_SHARED)]
            tilt ./= exp(sum(log, tilt) / length(tilt))
            for r in 1:MCK_LOCAL
                local_usage[r, j] = 10.0 * size_g * local_profile[r] * mix_l[r] * level * tilt[r]
            end
            for s in 1:MCK_SHARED
                shared_usage[s, j] =
                    10.0 * size_g * shared_profile[s] * mix_s[s] * level * tilt[MCK_LOCAL + s]
            end
            values[j] = worth * level^0.6 * exp(0.08 * randn(rng))
        end
    end

    all_cluster = ones(Int, G)
    feasible_witness = nothing
    infeasibility_certificate = nothing
    if feasibility_status == feasible || feasibility_status == infeasible
        pick = [rand(rng, class_options[g]) for g in 1:G]
        loc = _mck_totals(local_usage, pick, class_cluster, n_clusters)
        sh = vec(_mck_totals(shared_usage, pick, all_cluster, 1))
        local_capacity = loc .* (1.03 .+ 0.17 .* rand(rng, MCK_LOCAL, n_clusters))
        shared_capacity = sh .* (1.02 .+ 0.08 .* rand(rng, MCK_SHARED))
        if feasibility_status == feasible
            feasible_witness = MultipleChoiceWitness(pick)
        else
            s = rand(rng, 1:MCK_SHARED)
            min_usage = [minimum(shared_usage[s, j] for j in class_options[g]) for g in 1:G]
            total = sum(min_usage)
            shared_capacity[s] = total / (1.04 + 0.08 * rand(rng))
            infeasibility_certificate = MinimumLevelCertificate(
                s, min_usage, total, shared_capacity[s]
            )
        end
    else
        # Capacity review around the all-minimal-service plan (level 1 for
        # every class): cluster rows get 0-40% headroom over it, shared rows
        # are set 10% below to 12% above it. Below-plan shared capacity can
        # only be met by switching classes to configurations that trade the
        # shared resource for local ones, which may or may not suffice.
        base_pick = [first(class_options[g]) for g in 1:G]
        base_loc = _mck_totals(local_usage, base_pick, class_cluster, n_clusters)
        base_sh = vec(_mck_totals(shared_usage, base_pick, all_cluster, 1))
        local_capacity = base_loc .* (1.0 .+ 0.4 .* rand(rng, MCK_LOCAL, n_clusters))
        shared_capacity = base_sh .* (0.90 .+ 0.22 .* rand(rng, MCK_SHARED))
    end

    return MultipleChoiceKnapsackProblem(
        G,
        n_clusters,
        class_cluster,
        class_options,
        values,
        local_usage,
        shared_usage,
        local_capacity,
        shared_capacity,
        feasibility_status,
        feasible_witness,
        infeasibility_certificate,
    )
end

function build_model(prob::MultipleChoiceKnapsackProblem)
    model = Model()
    n = length(prob.values)
    @variable(model, x[1:n], Bin)
    @objective(model, Max, sum(prob.values[j] * x[j] for j in 1:n))

    @constraint(model, choose[g in 1:prob.n_classes], sum(x[j] for j in prob.class_options[g]) == 1)

    loc = [AffExpr() for _ in 1:MCK_LOCAL, _ in 1:prob.n_clusters]
    sh = [AffExpr() for _ in 1:MCK_SHARED]
    for g in 1:prob.n_classes
        c = prob.class_cluster[g]
        for j in prob.class_options[g]
            for r in 1:MCK_LOCAL
                add_to_expression!(loc[r, c], prob.local_usage[r, j], x[j])
            end
            for s in 1:MCK_SHARED
                add_to_expression!(sh[s], prob.shared_usage[s, j], x[j])
            end
        end
    end
    @constraint(
        model,
        cluster_capacity[r in 1:MCK_LOCAL, c in 1:prob.n_clusters],
        loc[r, c] <= prob.local_capacity[r, c]
    )
    @constraint(model, shared_capacity[s in 1:MCK_SHARED], sh[s] <= prob.shared_capacity[s])
    return model
end

register_variant(
    :knapsack,
    :multiple_choice,
    MultipleChoiceKnapsackProblem,
    "Multiple-choice multi-dimensional knapsack: one configuration per workload under per-cluster and shared resource limits";
    default=true,
    tags=[:combinatorial, :packing, :partitioning],
)
