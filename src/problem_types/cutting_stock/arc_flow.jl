using JuMP
using Random
using Distributions

"""
    ArcFlowWitness

Integer flow for a `feasible` arc-flow instance: `flow[a]` units on arc `a`.
It is the superposition of `rolls[i]` copies of item `i`'s single-item path
(`L ÷ len_i` item arcs from node 0, then loss arcs to `L`), so it conserves flow
at every internal node, produces `rolls[i] * (L ÷ len_i) >= demand[i]` pieces
of every item, and uses `sum(rolls) <= stock_limit` rolls.
"""
struct ArcFlowWitness
    flow::Vector{Int}
    rolls::Vector{Int}
end

"""
    ArcFlowMaterialCertificate

Relaxation-valid infeasibility proof with node potentials `pi_u = u`: every
item arc `(u, u + len_i)` raises the potential by exactly `len_i` and every
loss arc by a positive amount, so each unit of flow from `0` to `L` carries at
most `L` millimetres of pieces. Hence the demanded material
`demand_length = sum_i len_i * d_i` needs at least `demand_length / L` rolls,
but `demand_length >= 1.04 * L * stock_limit`. The Farkas combination uses
every conservation row, every demand row and the stock-limit row.
"""
struct ArcFlowMaterialCertificate
    demand_length::Float64
    supply_length::Float64
end

"""
    ArcFlowCuttingStockProblem <: ProblemGenerator

Arc-flow formulation of one-dimensional cutting stock (Valério de Carvalho
1999; the VPSolver graph): patterns are paths from node `0` to node `L` in a
graph whose nodes are the reachable cut positions.

# Overview

Items have integer lengths (millimetres on a bar of length `L`), sorted
longest first. Item `i` has an arc `(u, u + len_i)` for every node `u`
reachable using items `1..i` (the standard decreasing-length symmetry
reduction, so each pattern has one canonical path), and every reachable node
`u > 0` has a loss arc to the next reachable node (or `L`). Flows are general
integers in the MIP (relaxed by default).

```text
min  sum_{a out of 0} f_a
s.t. sum_{a into u} f_a - sum_{a out of u} f_a = 0     for every internal node u
     sum_{a of item i} f_a >= d_i                       for every item i
     sum_{a out of 0} f_a <= S                          (rolls in stock)
     f >= 0, integer
```

The LP bound equals the Gilmore-Gomory bound, but the matrix is a
pseudo-polynomial network with side constraints — structurally unlike the
pattern-based `standard`. (This variant replaces `integer_patterns`, whose
relaxation was the same pattern LP as `standard` with 31 rows.)

# Sizing

`n_types = clamp(round(sqrt(n) / 8), 6, 40)` relative lengths are drawn once
(`0.01 + 0.29 * Beta(1.2, 2.5)` times `L`, skewed short, so the reachable
node set is dense); `L` is then chosen by bisection plus a
scan of +-max(40, L/50) (the count is not monotone in `L`: rounded lengths change their
common divisors) so the exact arc count is as close to `target_variables` as
the integer graph allows (within a few percent above ~400 arcs). Rows: one per internal node plus `n_types + 1`, about
`1 / (n_types + 1)` of the columns or more.

# Feasibility

  - `feasible`: single-item plan, `S = U(1.05, 1.30)` times its rolls
    (`ArcFlowWitness`).
  - `infeasible`: `S = floor(material / (L * U(1.04, 1.12)))`
    (`ArcFlowMaterialCertificate`).
  - `unknown`: `S = round(U(0.97, 1.10) * material / L)`, around the trim-loss
    threshold of the best patterns; the LP decides.
"""
struct ArcFlowCuttingStockProblem <: ProblemGenerator
    stock_length::Int
    piece_lengths::Vector{Int}     # sorted, longest first
    demands::Vector{Int}
    arc_tail::Vector{Int}
    arc_head::Vector{Int}
    arc_item::Vector{Int}          # 0 for loss arcs
    stock_limit::Int
    feasibility_status::FeasibilityStatus
    feasible_witness::Union{Nothing, ArcFlowWitness}
    infeasibility_certificate::Union{Nothing, ArcFlowMaterialCertificate}
end

"""
    cs_arc_flow_graph(L, lengths) -> (tail, head, item)

Build the reduced arc-flow graph for bar length `L` and item lengths sorted
longest first. O(n_types * L).
"""
function cs_arc_flow_graph(L::Int, lengths::Vector{Int})
    reach = falses(L + 1)
    reach[1] = true
    tail = Int[]
    head = Int[]
    item = Int[]
    for (i, len) in enumerate(lengths)
        len <= L || continue
        for u in 0:(L - len)
            reach[u + 1] && (reach[u + len + 1] = true)
        end
        for u in 0:(L - len)
            if reach[u + 1]
                push!(tail, u)
                push!(head, u + len)
                push!(item, i)
            end
        end
    end
    prev = -1
    for u in 1:L
        reach[u + 1] || u == L || continue
        if prev > 0
            push!(tail, prev)
            push!(head, u)
            push!(item, 0)
        end
        prev = u
    end
    return tail, head, item
end

# Arc count only (no allocation of arc lists), for the sizing bisection.
function _cs_arc_flow_count(L::Int, lengths::Vector{Int})
    reach = falses(L + 1)
    reach[1] = true
    narcs = 0
    for len in lengths
        len <= L || continue
        for u in 0:(L - len)
            reach[u + 1] && (reach[u + len + 1] = true)
        end
        for u in 0:(L - len)
            reach[u + 1] && (narcs += 1)
        end
    end
    nodes = count(view(reach, 2:(L + 1))) + (reach[L + 1] ? 0 : 1)   # reachable u > 0, plus L
    return narcs + max(nodes - 1, 0)
end

_cs_af_lengths(rel::Vector{Float64}, L::Int) =
    sort!(unique!(max.(1, round.(Int, rel .* L))); rev=true)

function ArcFlowCuttingStockProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be positive (got $target_variables)"))
    target_variables <= 1_000_000 || throw(
        ArgumentError("arc_flow supports at most 1,000,000 arcs (got $target_variables)")
    )
    rng = MersenneTwister(seed)
    n = target_variables
    n_types = clamp(round(Int, sqrt(n) / 8), 6, 40)
    rel = [0.01 + 0.29 * rand(rng, Beta(1.2, 2.5)) for _ in 1:n_types]

    # Bisection on L (arc count grows ~linearly in L).
    lo, hi = 8, 64
    while _cs_arc_flow_count(hi, _cs_af_lengths(rel, hi)) < n && hi < 4_000_000
        lo = hi
        hi *= 2
    end
    while hi - lo > 1
        mid = (lo + hi) ÷ 2
        if _cs_arc_flow_count(mid, _cs_af_lengths(rel, mid)) < n
            lo = mid
        else
            hi = mid
        end
    end
    # The count is not monotone in L (rounded lengths change their common
    # divisors, which thins the reachable set), so scan a window around the
    # bisection point and keep the closest count.
    L = hi
    best = abs(_cs_arc_flow_count(hi, _cs_af_lengths(rel, hi)) - n)
    w = max(40, hi ÷ 50)
    for cand in max(8, hi - w):max(1, w ÷ 40):(hi + w)
        err = abs(_cs_arc_flow_count(cand, _cs_af_lengths(rel, cand)) - n)
        if err < best
            best, L = err, cand
        end
    end
    lengths = _cs_af_lengths(rel, L)
    m = length(lengths)
    tail, head, item = cs_arc_flow_graph(L, lengths)

    demands = cs_demands(rng, m)
    material = sum(Float64(lengths[i]) * demands[i] for i in 1:m)

    # Single-item plan rolls.
    copies = [L ÷ lengths[i] for i in 1:m]
    rolls = [cld(demands[i], copies[i]) for i in 1:m]

    feasible_witness = nothing
    infeasibility_certificate = nothing
    if feasibility_status == feasible
        stock_limit = ceil(Int, sum(rolls) * (1.05 + 0.25 * rand(rng)))
        # Superpose the single-item paths.
        arc_index = Dict{Tuple{Int, Int, Int}, Int}()
        for a in eachindex(tail)
            arc_index[(tail[a], head[a], item[a])] = a
        end
        loss_out = Dict{Int, Int}()
        for a in eachindex(tail)
            item[a] == 0 && (loss_out[tail[a]] = a)
        end
        flow = zeros(Int, length(tail))
        for i in 1:m
            rolls[i] == 0 && continue
            u = 0
            for _ in 1:copies[i]
                flow[arc_index[(u, u + lengths[i], i)]] += rolls[i]
                u += lengths[i]
            end
            while u < L
                a = loss_out[u]
                flow[a] += rolls[i]
                u = head[a]
            end
        end
        feasible_witness = ArcFlowWitness(flow, rolls)
    elseif feasibility_status == infeasible
        stock_limit = floor(Int, material / (L * (1.04 + 0.08 * rand(rng))))
        infeasibility_certificate = ArcFlowMaterialCertificate(material, Float64(L) * stock_limit)
    else
        stock_limit = round(Int, (0.97 + 0.13 * rand(rng)) * material / L)
    end

    return ArcFlowCuttingStockProblem(
        L,
        lengths,
        demands,
        tail,
        head,
        item,
        stock_limit,
        feasibility_status,
        feasible_witness,
        infeasibility_certificate,
    )
end

function build_model(prob::ArcFlowCuttingStockProblem)
    model = Model()
    A = length(prob.arc_tail)
    L = prob.stock_length
    @variable(model, flow[1:A] >= 0, Int)

    balance = Dict{Int, AffExpr}()
    produced = [AffExpr() for _ in eachindex(prob.piece_lengths)]
    rolls = AffExpr()
    for a in 1:A
        u, v = prob.arc_tail[a], prob.arc_head[a]
        if u == 0
            add_to_expression!(rolls, 1.0, flow[a])
        else
            add_to_expression!(get!(AffExpr, balance, u), -1.0, flow[a])
        end
        v == L || add_to_expression!(get!(AffExpr, balance, v), 1.0, flow[a])
        prob.arc_item[a] > 0 && add_to_expression!(produced[prob.arc_item[a]], 1.0, flow[a])
    end
    @objective(model, Min, rolls)
    for u in sort!(collect(keys(balance)))
        @constraint(model, balance[u] == 0)
    end
    @constraint(model, demand[i in eachindex(produced)], produced[i] >= prob.demands[i])
    @constraint(model, stock_limit, rolls <= prob.stock_limit)
    return model
end

register_variant(
    :cutting_stock,
    :arc_flow,
    ArcFlowCuttingStockProblem,
    "Arc-flow (Valério de Carvalho) cutting stock: integer flows on a reduced cut-position graph with demand and stock-limit side constraints",
)
