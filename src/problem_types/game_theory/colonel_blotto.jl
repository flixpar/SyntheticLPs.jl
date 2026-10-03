using JuMP
using Random
using Distributions
using LinearAlgebra

"""
Planted feasible point of the compact Blotto LP: the seat player's
fictitious-play average allocation flow `allocation_flow` (a convex
combination of pure allocations, so a valid unit flow), its per-battlefield
marginals `marginals[k, a + 1]`, the induced expected payoffs
`expected_payoffs[k, b + 1]` of each opponent placement, and the opponent's
exact best-response (shortest-path) potentials `potentials` on its allocation
DAG — which satisfy every opponent-edge row. `guaranteed_value =
potentials[1]` is at least the model's `required_value`.
"""
struct ColonelBlottoWitness
    allocation_flow::Vector{Float64}
    marginals::Matrix{Float64}
    expected_payoffs::Matrix{Float64}
    potentials::Vector{Float64}
    guaranteed_value::Float64
end

"""
Certificate that no allocation strategy guarantees `required_value`: the
opponent's fictitious-play average allocation flow `opponent_flow` (a unit
flow on its DAG) and the seat player's exact best-response (longest-path)
potentials `potentials` against it, with `value_bound = potentials[1]`.
Weighting the opponent-edge rows by the flow telescopes the potentials to
`π[1] <= Σ_k Σ_b q[k, b] g[k, b]`; the payoff-definition rows turn that into
`Σ_k Σ_a h[k, a] p[k, a]` with `h[k, a] = Σ_b u_k(a, b) q[k, b]`; the marginal
rows and the potential inequalities `λ[tail] >= h[k, a] + λ[head]` bound it by
`λ[1] = value_bound` on every unit allocation flow — LP rows only — while the
requirement is `π[1] >= required_value > value_bound`.
"""
struct ColonelBlottoCertificate
    opponent_flow::Vector{Float64}
    potentials::Vector{Float64}
    value_bound::Float64
end

"""
    ColonelBlottoProblem <: ProblemGenerator

Compact equilibrium LP of a Colonel Blotto game (Borel 1921; the polynomial
LP of Ahmadinejad, Dehghani, Hajiaghayi, Lucier, Mahini & Seddighin 2016):
two players simultaneously split integer budgets over `K` battlefields —
campaign spending across electoral districts, advertising across markets,
security resources across sites.

# Strategy polytopes

A pure allocation of `S` units over `K` battlefields is a path through a
layered DAG: node `(k, s)` means `s` units were spent on battlefields
`1..k-1`, edge `(k, s) -> (k + 1, s + a)` spends `a` on battlefield `k`, and the
last layer spends the remainder (budgets are used in full). Mixed strategies
are unit flows on that DAG — exponentially many allocations, polynomially many
variables. Because the payoff is separable across battlefields, only the
per-battlefield marginals `p[k, a]` matter.

# LP (seat player = maximizer, budget `budget`)

    maximize    π[1]
    subject to  unit flow x on the seat DAG (source row + conservation rows)
                p[k, a] - Σ_{edges e of layer k spending a} x[e] = 0
                g[k, b] - Σ_a u_k(a, b) p[k, a] = 0
                π[tail(e)] - π[head(e)] - g[k, b] <= 0   (each opponent DAG edge e = (k, ·, b))
                π[1] >= required_value,  x, p >= 0,  g, π free

The last block is the dual of the opponent's shortest-path best response
(sink potential fixed at zero). The matrix therefore mixes a primal layered
flow block, a dense per-battlefield payoff block `u_k`, and a dual
potential block with three nonzeros per opponent edge.

# Data

Battlefield weights are lognormal integers (electoral votes / market sizes);
some battlefields give the opponent a small incumbency advantage `d_k` (the
seat player must exceed `b + d_k`). The contest is either `:majority`
(`u_k = w_k sign(a - b - d_k)`) or a Tullock `:lottery` with exponent `ρ`
(`u_k = w_k (2 a^ρ / (a^ρ + (b + d_k)^ρ) - 1)`). Budgets are asymmetric
(`budget / opponent_budget` in roughly `[0.67, 1.67]`).

# Feasibility control

Fictitious play with exact DP best responses yields average flows for both
players; their exact best-response values bracket the game value
(`lower_bound <= value <= upper_bound`). The guaranteed-value requirement
`π[1] >= required_value` is placed by `_game_value_requirement` (below the
lower bound / above the upper bound / bracketing both), with a typed witness
or certificate as above.
"""
struct ColonelBlottoProblem <: ProblemGenerator
    n_battlefields::Int
    budget::Int
    opponent_budget::Int
    weights::Vector{Float64}
    advantage::Vector{Int}
    contest::Symbol
    lottery_exponent::Float64
    lower_bound::Float64
    upper_bound::Float64
    required_value::Float64
    fp_iterations::Int
    feasible_witness::Union{Nothing, ColonelBlottoWitness}
    infeasibility_certificate::Union{Nothing, ColonelBlottoCertificate}
    feasibility_status::FeasibilityStatus
end

# Allocation DAG for K battlefields and budget S: node (1, 0) is index 1;
# node (k, s), k = 2..K, s = 0..S, is 2 + (k - 2)(S + 1) + s. The sink (K + 1, S)
# is implicit (its potential is fixed at zero).
_blotto_num_nodes(K::Int, S::Int) = 1 + (K - 1) * (S + 1)
_blotto_node(k::Int, s::Int, S::Int) = k == 1 ? 1 : 2 + (k - 2) * (S + 1) + s
_blotto_num_edges(K::Int, S::Int) = 2 * (S + 1) + (K - 2) * (S + 1) * (S + 2) ÷ 2

"""
    _blotto_edges(K, S) -> (layer, from, step, tail, head)

Every edge of the allocation DAG, layer by layer: `from` is the units spent
before battlefield `layer`, `step` the units placed on it, `tail`/`head` node
indices (`head = 0` for the implicit sink). Layer 1 leaves `(1, 0)`; middle
layers allow any `step <= S - from`; the last layer spends exactly the rest.
"""
function _blotto_edges(K::Int, S::Int)
    E = _blotto_num_edges(K, S)
    layer, from, step = Vector{Int}(undef, E), Vector{Int}(undef, E), Vector{Int}(undef, E)
    tail, head = Vector{Int}(undef, E), Vector{Int}(undef, E)
    e = 0
    for k in 1:K
        for s in (k == 1 ? (0:0) : (0:S))
            for a in (k == K ? ((S - s):(S - s)) : (0:(S - s)))
                e += 1
                layer[e], from[e], step[e] = k, s, a
                tail[e] = _blotto_node(k, s, S)
                head[e] = k == K ? 0 : _blotto_node(k + 1, s + a, S)
            end
        end
    end
    return layer, from, step, tail, head
end

"""
    _blotto_size_formula(K, S, Sopp) -> (variables, rows)

    variables = E(K, S) + K (S + 1) + K (Sopp + 1) + 1 + (K - 1)(Sopp + 1)
    rows      = 1 + (K - 1)(S + 1) + K (S + 1) + K (Sopp + 1) + E(K, Sopp)

with `E(K, S) = 2 (S + 1) + (K - 2)(S + 1)(S + 2) / 2` DAG edges.
"""
function _blotto_size_formula(K::Int, S::Int, So::Int)
    vars = _blotto_num_edges(K, S) + K * (S + 1) + K * (So + 1) + _blotto_num_nodes(K, So)
    rows = _blotto_num_nodes(K, S) + K * (S + 1) + K * (So + 1) + _blotto_num_edges(K, So)
    return vars, rows
end

"""
    _blotto_choose_size(rng, target) -> (K, S, Sopp)

Sample a battlefield count (log-uniform on `3..Kmax`, `Kmax` growing like the
cube root of the target so budgets stay meaningful) and pick budgets whose
exact variable count is within 2% of `target` (uniformly among qualifying
`(S, Sopp)` with `Sopp / S` in `[0.6, 1.5]`), falling back to every `K` and
then to the closest size. Targets below the smallest game round up to it.
"""
function _blotto_choose_size(rng::AbstractRNG, target::Int)
    Kmax = clamp(floor(Int, 0.8 * cbrt(8 * target)), 3, 40)
    K0 = clamp(round(Int, exp(log(3) + (log(Kmax) - log(3)) * rand(rng))), 3, Kmax)
    function scan(Ks)
        cands = NTuple{3, Int}[]
        best, best_err = (3, 2, 2), Inf
        for K in Ks
            for S in 2:5000
                _blotto_size_formula(K, S, max(2, ceil(Int, 0.6 * S)))[1] > 1.05 * target + 60 && break
                for So in max(2, ceil(Int, 0.6 * S)):max(2, floor(Int, 1.5 * S))
                    v = _blotto_size_formula(K, S, So)[1]
                    err = abs(v - target)
                    err <= 0.02 * target && push!(cands, (K, S, So))
                    if err < best_err
                        best, best_err = (K, S, So), err
                    end
                end
            end
        end
        return cands, best
    end
    cands, _ = scan(K0:K0)
    isempty(cands) || return cands[rand(rng, 1:length(cands))]
    cands, best = scan(3:Kmax)
    return isempty(cands) ? best : cands[rand(rng, 1:length(cands))]
end

"""
    _blotto_payoffs(weights, advantage, contest, ρ, S, Sopp) -> Array{Float64,3}

Seat-player payoff `u[k, a + 1, b + 1]` for placing `a` against `b` units on
battlefield `k`.
"""
function _blotto_payoffs(weights, advantage, contest, ρ, S, So)
    K = length(weights)
    u = zeros(K, S + 1, So + 1)
    for k in 1:K, a in 0:S, b in 0:So
        d = advantage[k]
        u[k, a + 1, b + 1] = if contest == :majority
            weights[k] * sign(a - b - d)
        else
            num = float(a)^ρ
            den = num + float(b + d)^ρ
            den == 0 ? 0.0 : weights[k] * (2 * num / den - 1)
        end
    end
    return u
end

"""
    _blotto_best_path(K, S, val, maximize) -> (potentials, path_edges_steps)

Exact best response on the allocation DAG against per-battlefield values
`val[k, a + 1]`: backward DP over layers (longest path when `maximize`,
shortest otherwise). Returns node potentials (value-to-go, sink = 0) and the
optimal steps `a_1..a_K` from the root.
"""
function _blotto_best_path(K::Int, S::Int, val::Matrix{Float64}, maximize::Bool)
    pot = zeros(_blotto_num_nodes(K, S))
    choice = zeros(Int, _blotto_num_nodes(K, S))
    better(x, y) = maximize ? x > y : x < y
    for k in K:-1:1
        for s in (k == 1 ? (0:0) : (0:S))
            n = _blotto_node(k, s, S)
            if k == K
                pot[n] = val[K, S - s + 1]
                choice[n] = S - s
            else
                bestv, besta = maximize ? -Inf : Inf, 0
                for a in 0:(S - s)
                    v = val[k, a + 1] + pot[_blotto_node(k + 1, s + a, S)]
                    if better(v, bestv)
                        bestv, besta = v, a
                    end
                end
                pot[n] = bestv
                choice[n] = besta
            end
        end
    end
    steps = zeros(Int, K)
    s = 0
    for k in 1:K
        steps[k] = choice[_blotto_node(k, s, S)]
        s += steps[k]
    end
    return pot, steps
end

# Edge index of the path edge on layer k leaving `s` with `a` units, matching
# `_blotto_edges` ordering.
function _blotto_edge_index(K::Int, S::Int, k::Int, s::Int, a::Int)
    k == 1 && return a + 1
    base = (S + 1) + (k - 2) * ((S + 1) * (S + 2) ÷ 2)
    if k == K
        return base + s + 1
    end
    # Nodes s' < s contribute (S - s' + 1) edges each.
    return base + s * (S + 1) - s * (s - 1) ÷ 2 + a + 1
end

"""
    _blotto_fictitious_play(u, K, S, So, iterations) -> (xbar, zbar)

Fictitious play: each round, both players play exact DP best responses to the
opponent's running average marginals; returns the average edge flows (valid
unit flows by convexity).
"""
function _blotto_fictitious_play(u::Array{Float64, 3}, K::Int, S::Int, So::Int, iterations::Int)
    xsum = zeros(_blotto_num_edges(K, S))
    zsum = zeros(_blotto_num_edges(K, So))
    psum = fill(1.0 / (S + 1), K, S + 1)     # uniform prior marginals (one pseudo-round)
    qsum = fill(1.0 / (So + 1), K, So + 1)
    h = zeros(K, S + 1)
    g = zeros(K, So + 1)
    for t in 1:iterations
        for k in 1:K
            @views mul!(h[k, :], u[k, :, :], qsum[k, :])
            @views mul!(g[k, :], transpose(u[k, :, :]), psum[k, :])
        end
        _, pa = _blotto_best_path(K, S, h, true)
        _, pb = _blotto_best_path(K, So, g, false)
        s = 0
        for k in 1:K
            xsum[_blotto_edge_index(K, S, k, s, pa[k])] += 1
            psum[k, pa[k] + 1] += 1
            s += pa[k]
        end
        s = 0
        for k in 1:K
            zsum[_blotto_edge_index(K, So, k, s, pb[k])] += 1
            qsum[k, pb[k] + 1] += 1
            s += pb[k]
        end
    end
    return xsum ./ iterations, zsum ./ iterations
end

# Per-battlefield marginals of an edge flow.
function _blotto_marginals(flow::Vector{Float64}, K::Int, S::Int)
    layer, _, step, _, _ = _blotto_edges(K, S)
    p = zeros(K, S + 1)
    for e in eachindex(flow)
        p[layer[e], step[e] + 1] += flow[e]
    end
    return p
end

"""
    ColonelBlottoProblem(target_variables, feasibility_status, seed)

Build a compact Blotto equilibrium LP within about 2% of `target_variables`
(exact formula in `_blotto_size_formula`); targets above
`GAME_THEORY_MAX_VARIABLES` raise an `ArgumentError`.
"""
function ColonelBlottoProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    _game_theory_check_target("colonel_blotto", target_variables)
    rng = MersenneTwister(seed)
    K, S, So = _blotto_choose_size(rng, target_variables)

    weights = [max(1.0, round(rand(rng, LogNormal(log(8.0), 0.8)))) for _ in 1:K]
    advantage = [rand(rng) < 0.25 ? rand(rng, 1:max(1, min(3, So ÷ 4))) : 0 for _ in 1:K]
    contest = rand(rng) < 0.6 ? :majority : :lottery
    ρ = contest == :lottery ? round(0.6 + 0.9 * rand(rng); digits=3) : 1.0
    u = _blotto_payoffs(weights, advantage, contest, ρ, S, So)

    work = K * (S + 1) * (So + 1) + K * ((S + 1)^2 + (So + 1)^2) ÷ 2
    iterations = clamp(round(Int, 3.0e8 / work), 30, 400)
    xbar, zbar = _blotto_fictitious_play(u, K, S, So, iterations)

    p = _blotto_marginals(xbar, K, S)
    q = _blotto_marginals(zbar, K, So)
    g = zeros(K, So + 1)
    h = zeros(K, S + 1)
    for k in 1:K
        @views mul!(g[k, :], transpose(u[k, :, :]), p[k, :])
        @views mul!(h[k, :], u[k, :, :], q[k, :])
    end
    opp_pot, _ = _blotto_best_path(K, So, g, false)
    λ, _ = _blotto_best_path(K, S, h, true)
    lower, upper = opp_pot[1], λ[1]

    required_value = _game_value_requirement(rng, feasibility_status, lower, upper, sum(weights))
    witness = nothing
    certificate = nothing
    if feasibility_status == feasible
        witness = ColonelBlottoWitness(xbar, p, g, opp_pot, lower)
    elseif feasibility_status == infeasible
        certificate = ColonelBlottoCertificate(zbar, λ, upper)
    end

    return ColonelBlottoProblem(
        K,
        S,
        So,
        weights,
        advantage,
        contest,
        ρ,
        lower,
        upper,
        required_value,
        iterations,
        witness,
        certificate,
        feasibility_status,
    )
end

"""
    build_model(prob::ColonelBlottoProblem)

Deterministic compact Blotto LP (see the type docstring): `x` seat allocation
flow, `p` marginals, `g` expected payoff per opponent placement, `π` opponent
DAG potentials; maximize `π[1]` with `π[1] >= required_value`.
"""
function build_model(prob::ColonelBlottoProblem)
    K, S, So = prob.n_battlefields, prob.budget, prob.opponent_budget
    u = _blotto_payoffs(prob.weights, prob.advantage, prob.contest, prob.lottery_exponent, S, So)
    layer, _, step, tail, head = _blotto_edges(K, S)
    olayer, _, ostep, otail, ohead = _blotto_edges(K, So)

    model = Model()
    @variable(model, x[1:length(layer)] >= 0)
    @variable(model, p[1:K, 0:S] >= 0)
    @variable(model, g[1:K, 0:So])
    @variable(model, pot[1:_blotto_num_nodes(K, So)])
    set_lower_bound(pot[1], prob.required_value)
    @objective(model, Max, pot[1])

    # Unit allocation flow on the seat DAG.
    nn = _blotto_num_nodes(K, S)
    net = [AffExpr(0.0) for _ in 1:nn]
    marg = [AffExpr(0.0) for _ in 1:K, _ in 0:S]
    for e in eachindex(layer)
        add_to_expression!(net[tail[e]], 1.0, x[e])
        head[e] > 0 && add_to_expression!(net[head[e]], -1.0, x[e])
        add_to_expression!(marg[layer[e], step[e] + 1], 1.0, x[e])
    end
    @constraint(model, net[1] == 1)
    for n in 2:nn
        @constraint(model, net[n] == 0)
    end

    # Marginal and expected-payoff definitions.
    for k in 1:K, a in 0:S
        @constraint(model, p[k, a] - marg[k, a + 1] == 0)
    end
    for k in 1:K, b in 0:So
        expr = AffExpr(0.0)
        add_to_expression!(expr, 1.0, g[k, b])
        for a in 0:S
            c = u[k, a + 1, b + 1]
            c == 0 || add_to_expression!(expr, -c, p[k, a])
        end
        @constraint(model, expr == 0)
    end

    # Dual of the opponent's shortest-path best response.
    for e in eachindex(olayer)
        if ohead[e] > 0
            @constraint(model, pot[otail[e]] - pot[ohead[e]] - g[olayer[e], ostep[e]] <= 0)
        else
            @constraint(model, pot[otail[e]] - g[olayer[e], ostep[e]] <= 0)
        end
    end
    return model
end

register_variant(
    :game_theory,
    :colonel_blotto,
    ColonelBlottoProblem,
    "Compact Colonel Blotto equilibrium LP: a primal unit flow over the seat player's layered allocation DAG, per-battlefield marginal and dense payoff-definition rows (majority or Tullock lottery contests with incumbency advantages), and the dualized shortest-path best response of the opponent, with a guaranteed-value requirement certified by fictitious-play strategies and exact DP best responses",
)
