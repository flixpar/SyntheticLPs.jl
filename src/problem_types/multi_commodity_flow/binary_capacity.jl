using JuMP
using Random
using Distributions
using Statistics

"""
Planted design: every origin-destination commodity routed along a noisy
shortest path, with exactly one module — the smallest whose capacity covers
the aggregate load — installed on every used arc (`install[a, m]` in {0, 1}).
Feasible for the MIP itself: bundle, module-choice and strong-linking rows all
hold.
"""
struct BinaryCapacityMCFWitness
    flows::Matrix{Float64}
    install::Matrix{Float64}
end

"""
    BinaryCapacityMultiCommodityFlowProblem <: ProblemGenerator

Multicommodity capacitated network design with capacity modules, on sparse
geographic networks (the strong MCND formulation).

# Overview

Commodities are origin-destination pairs with demand `demand[k]`. On every arc
at most one of three capacity modules (e.g. 1x, 2.5x and 6x a base unit) can be
installed at a fixed cost with economies of scale; routing is continuous.

    minimize    sum_{a,k} routing_cost[a,k] x[a,k] + sum_{a,m} module_cost[a,m] y[a,m]
    subject to  flow conservation per (node, commodity)
                sum_k x[a,k] <= sum_m module_capacity[a,m] y[a,m]      every arc
                sum_m y[a,m] <= 1                                      every arc
                x[a,k] <= min(demand[k], max_m module_capacity[a,m]) * sum_m y[a,m]
                                                                       every arc, commodity
                x >= 0, y in {0,1}

The last family is the classic STRONG linking inequality: without it the LP
relaxation installs a sliver of a module per unit of flow (the weak
relaxation the previous version had); with it every commodity using an arc
must pay for a proportional share of a module, so the relaxation keeps genuine
design structure.

# Data grounding

Network, origins and gravity destinations as in `multi_commodity_flow/standard`
(one destination per commodity). Module capacities are a lognormal base unit
(trunk arcs 1.6x) times (1, 2.5, 6); module cost = arc length x capacity^0.7
(economies of scale) plus a fixed installation charge. Routing cost =
arc length x commodity value factor.

# Feasibility control

Planted routing along commodity-specific noisy shortest paths; on every used
arc the base unit is raised if needed so the largest module covers
`rho * load`:

  - `feasible`: `rho` in [1.05, 1.5]; the planted design is the witness.
  - `infeasible`: a metric certificate (`MultiCommodityFlowMetricCertificate`)
    on the LARGEST module capacities — valid in the relaxation because
    `sum_m y <= 1` caps every arc at its largest module. Module capacities are
    scaled down (all arcs in `:length` mode, the region's exit arcs in
    `:regional_cut` mode) until it separates.
  - `unknown`: modules as `feasible`, then every demand grows by a common
    factor in [1.0, 1.8] (`_mcf_unknown_growth!`; the largest modules leave
    more headroom than continuous capacities); natural, either side.

# Sizing

Variables `n_arcs * (n_commodities + 3)` with `n_commodities =
clamp(round(0.5 * target^0.3), 3, 40)` and `n_arcs ~ target / (n_commodities +
3)`. Rows `n_nodes * n_commodities + 2 n_arcs + n_arcs * n_commodities`.
"""
struct BinaryCapacityMultiCommodityFlowProblem <: ProblemGenerator
    n_nodes::Int
    arcs::Vector{Tuple{Int, Int}}
    trunk::Vector{Bool}
    routing_cost::Matrix{Float64}
    module_capacity::Matrix{Float64}
    module_cost::Matrix{Float64}
    origins::Vector{Int}
    destinations::Vector{Int}
    demands::Vector{Float64}
    positions::Vector{Tuple{Float64, Float64}}
    geography::Symbol
    feasible_witness::Union{Nothing, BinaryCapacityMCFWitness}
    infeasibility_certificate::Union{Nothing, MultiCommodityFlowMetricCertificate}
    feasibility_status::FeasibilityStatus
end

const _MCND_MODULE_SIZES = (1.0, 2.5, 6.0)

function BinaryCapacityMultiCommodityFlowProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be >= 1 (got $target_variables)."))
    target_variables <= MCF_MAX_VARIABLES || throw(
        ArgumentError(
            "multi_commodity_flow/binary_capacity supports at most $MCF_MAX_VARIABLES variables; " *
            "requested $target_variables.",
        ),
    )
    rng = MersenneTwister(seed)
    M = length(_MCND_MODULE_SIZES)
    K = clamp(round(Int, 0.5 * target_variables^0.3), 3, 40)
    # OD commodities: one destination each; origins may repeat across pairs,
    # so draw them with replacement after the network exists.
    inst = _mcf_instance(rng, max(round(Int, target_variables / (K + M)), 2), 1; dest_share=(0.0, 0.0))
    n, arcs, trunk, positions, dist = inst.n, inst.arcs, inst.trunk, inst.positions, inst.dist
    A = length(arcs)
    out_adj, _ = _geo_adjacency(n, arcs)
    length_scale = 5.0 * median(dist)
    act = inst.weights
    origins = sample(rng, 1:n, Weights(act), K; replace=true)
    destinations = Vector{Int}(undef, K)
    raw = Vector{Float64}(undef, K)
    for k in 1:K
        d, r = _mcf_gravity_destinations(rng, n, origins[k], positions, act, 1, length_scale)
        destinations[k], raw[k] = d[1], r[1]
    end
    demands = max.(round.(50.0 .* raw ./ (sum(raw) / K); digits=2), 0.01)

    value_factor = [rand(rng, LogNormal(0.0, 0.35)) for _ in 1:K]
    arc_noise = [rand(rng, LogNormal(0.0, 0.2)) for _ in 1:A]
    routing_cost = [round((dist[a] * arc_noise[a] + 0.1) * value_factor[k]; digits=3) for a in 1:A, k in 1:K]

    dests = [[destinations[k]] for k in 1:K]
    dems = [[demands[k]] for k in 1:K]
    flows, load = _mcf_planted_loads(rng, n, arcs, out_adj, dist, origins, dests, dems)
    used = filter(>(0.0), load)
    unit_scale = (0.3 + 0.4 * rand(rng)) * (isempty(used) ? 50.0 : median(used))
    rho_range = (1.05, 1.5)
    base = Vector{Float64}(undef, A)
    for a in 1:A
        b = unit_scale * rand(rng, LogNormal(0.0, 0.5)) * (trunk[a] ? 1.6 : 1.0)
        if load[a] > 0
            need = (rho_range[1] + (rho_range[2] - rho_range[1]) * rand(rng)) * load[a]
            b = max(b, need / _MCND_MODULE_SIZES[end])
        end
        base[a] = b
    end
    module_capacity = [ceil(base[a] * _MCND_MODULE_SIZES[m]; digits=2) for a in 1:A, m in 1:M]
    install_charge = 20.0 * rand(rng, LogNormal(0.0, 0.3))
    module_cost = [
        round((dist[a] + 1.0) * module_capacity[a, m]^0.7 * 0.5 + install_charge; digits=2) for
        a in 1:A, m in 1:M
    ]

    feasible_witness = nothing
    infeasibility_certificate = nothing
    if feasibility_status == feasible
        install = zeros(A, M)
        for a in 1:A
            load[a] > 0 || continue
            m = findfirst(m -> module_capacity[a, m] >= load[a], 1:M)
            install[a, m] = 1.0
        end
        feasible_witness = BinaryCapacityMCFWitness(flows, install)
    else
        maxcap = module_capacity[:, M]
        squeezed = copy(maxcap)
        if feasibility_status == infeasible
            infeasibility_certificate = _mcf_enforce_metric!(
                rng, squeezed, n, arcs, out_adj, positions, origins, dests, dems
            )
        else
            _mcf_unknown_growth!(rng, squeezed, n, arcs, origins, dests, dems; max_growth=1.8)
            demands .= [d[1] for d in dems]
        end
        # Rescale every module of an arc with its largest one (the squeeze
        # may also enlarge an arc to keep a node open).
        for a in 1:A
            squeezed[a] == maxcap[a] && continue
            f = squeezed[a] / maxcap[a]
            for m in 1:(M - 1)
                module_capacity[a, m] = max(round(module_capacity[a, m] * f; digits=2), 0.01)
            end
            module_capacity[a, M] = squeezed[a]
        end
    end

    return BinaryCapacityMultiCommodityFlowProblem(
        n,
        arcs,
        trunk,
        routing_cost,
        module_capacity,
        module_cost,
        origins,
        destinations,
        demands,
        positions,
        inst.geography,
        feasible_witness,
        infeasibility_certificate,
        feasibility_status,
    )
end

"""
    build_model(prob::BinaryCapacityMultiCommodityFlowProblem)

Build the strong MCND formulation. Deterministic — uses only the struct fields.
"""
function build_model(prob::BinaryCapacityMultiCommodityFlowProblem)
    model = Model()
    A = length(prob.arcs)
    K = length(prob.origins)
    M = size(prob.module_capacity, 2)
    n = prob.n_nodes
    @variable(model, x[1:A, 1:K] >= 0)
    @variable(model, y[1:A, 1:M], Bin)
    @objective(
        model,
        Min,
        sum(prob.routing_cost[a, k] * x[a, k] for a in 1:A, k in 1:K) +
            sum(prob.module_cost[a, m] * y[a, m] for a in 1:A, m in 1:M)
    )
    for a in 1:A
        @constraint(model, sum(x[a, k] for k in 1:K) <= sum(prob.module_capacity[a, m] * y[a, m] for m in 1:M))
        @constraint(model, sum(y[a, m] for m in 1:M) <= 1)
        cap = prob.module_capacity[a, M]
        for k in 1:K
            @constraint(model, x[a, k] <= min(prob.demands[k], cap) * sum(y[a, m] for m in 1:M))
        end
    end
    out_adj, in_adj = _geo_adjacency(n, prob.arcs)
    for k in 1:K, v in 1:n
        rhs = (v == prob.origins[k] ? prob.demands[k] : 0.0) - (v == prob.destinations[k] ? prob.demands[k] : 0.0)
        @constraint(
            model,
            sum(x[a, k] for a in out_adj[v]; init=AffExpr(0.0)) -
            sum(x[a, k] for a in in_adj[v]; init=AffExpr(0.0)) == rhs
        )
    end
    return model
end

register_variant(
    :multi_commodity_flow,
    :binary_capacity,
    BinaryCapacityMultiCommodityFlowProblem,
    "Multicommodity capacitated network design on a sparse geographic network: origin-destination commodities, three capacity modules per arc with economies of scale, and the strong linking inequalities that keep the LP relaxation meaningful; planted design witness and metric-inequality certificate";
    tags=[:logistics, :multicommodity, :network, :big_m],
    max_target_variables=1_000_000,
)
