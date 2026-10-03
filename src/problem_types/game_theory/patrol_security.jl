using JuMP
using Random
using Distributions
using Statistics: quantile

"""
Planted feasible point of the patrol LP: the defender's average patrol plan
`arc_flows` (the mean of the greedy best-response patrols computed while
learning the attacker mix — a convex combination of `n_units` unit paths), the
coverage it implements `coverage = min.(1, inflow)`, each attacker type's
best-response value `type_values[k] = max_o (gain_o - loss_o * coverage_o)`,
and the resulting expected loss `expected_loss = Σ_k prior_k type_values[k]`,
which is at most the model's `loss_requirement`.
"""
struct PatrolSecurityWitness
    arc_flows::Vector{Float64}
    coverage::Vector{Float64}
    type_values::Vector{Float64}
    expected_loss::Float64
end

"""
Certificate that no patrol plan holds the expected loss to `loss_requirement`:
an attacker mix `attack_mix` (a distribution over each type's options), the
induced protection weights `w_j = Σ_{o -> j} prior_k α_o loss_o`, a threshold
`θ`, and continuation potentials `potentials` (zero at the last period) with a
source potential `source_potential = λ >= 0` such that every arc satisfies
`min(w, θ)[head] + Φ[head] <= Φ[tail]` (`λ` for source arcs). Then for every
feasible point

    Σ_k prior_k v_k >= Σ_o prior_k α_o gain_o - Σ_j (w_j - θ)^+ - n_units λ = loss_bound

using only the attack rows (weights `prior_k α_o`), the bounds `c <= 1`, the
coverage rows `c <= inflow` (weights `min(w, θ)`), flow conservation
(potentials), and the unit-availability row (`λ`); `loss_requirement <
loss_bound` contradicts the requirement row.
"""
struct PatrolSecurityCertificate
    attack_mix::Vector{Float64}
    threshold::Float64
    potentials::Vector{Float64}
    source_potential::Float64
    loss_bound::Float64
end

"""
    PatrolSecurityProblem <: ProblemGenerator

Zero-sum Bayesian security game with randomized patrols on a time-expanded
network — the compact marginal-coverage LP behind deployed patrol planners
(TRUSTS fare inspection, Yin et al. 2012; PROTECT port patrols, Shieh et al.
2012; moving-target protection, Fang, Jiang & Tambe 2013).

# Setting

`n_stations` stations on a street-grid network (a random spanning tree of a
jittered grid plus a share of the remaining grid links). A shift has
`horizon` periods; `n_units` patrol units start anywhere in period 1 and each
period stay or move to an adjacent station. Station value combines downtown
hot spots, lognormal noise and a hub bonus, and varies over the shift with a
two-peak (rush-hour) profile. Attacker type 1 is opportunistic (every station,
every period); types `2..K` are focused groups with a home district and an
active time window. Each type's gain at an uncovered target is
`value * profile * interest`, and being caught costs it `κ_k` times that gain.

# LP (defender minimizes the expected loss)

    minimize    Σ_k prior_k v_k
    subject to  Σ_{source arcs} f <= n_units                       (unit availability)
                inflow(i, t) - outflow(i, t) = 0                    (t < horizon)
                c[i, t] - inflow(i, t) <= 0                         (coverage needs presence)
                v_k + loss_o c[target(o)] >= gain_o                  (each option o of type k)
                Σ_k prior_k v_k <= loss_requirement                 (risk budget)
                f >= 0,  0 <= c <= 1,  v free

a time-expanded flow block coupled through bounded coverage variables to
many epigraph (attacker best-response) rows that share a handful of dense
type columns.

# Feasibility control

The constructor learns an approximate equilibrium (attacker Hedge against
greedy saturating patrol best responses). The average patrol plan gives an
upper bound on the minimax loss (`upper_bound`); the average attacker mix,
with a threshold-Lagrangian longest-path bound on the protection any patrol
plan can buy, gives a rigorous lower bound (`lower_bound`, never below the
trivial all-covered bound). The requirement `loss_requirement` is
`upper_bound + δ` (`feasible`, witness stored), `lower_bound - δ`
(`infeasible`, certificate stored), or drawn across `[lower_bound,
upper_bound]` widened by half the gap (`unknown`).
"""
struct PatrolSecurityProblem <: ProblemGenerator
    n_stations::Int
    horizon::Int
    n_units::Int
    positions::Vector{Tuple{Float64, Float64}}
    edges::Vector{Tuple{Int, Int}}
    station_value::Vector{Float64}
    time_profile::Vector{Float64}
    type_prior::Vector{Float64}
    type_penalty::Vector{Float64}
    option_type::Vector{Int}
    option_target::Vector{Int}
    option_gain::Vector{Float64}
    option_loss::Vector{Float64}
    lower_bound::Float64
    upper_bound::Float64
    loss_requirement::Float64
    iterations::Int
    feasible_witness::Union{Nothing, PatrolSecurityWitness}
    infeasibility_certificate::Union{Nothing, PatrolSecurityCertificate}
    feasibility_status::FeasibilityStatus
end

# Node-time (station i, period t) has index (t - 1) * n + i.
_patrol_nt(i::Int, t::Int, n::Int) = (t - 1) * n + i

"""
    _patrol_moves(n, edges) -> (ptr, nbr)

CSR list of each station's moves: itself (stay) first, then its neighbours in
increasing order. Arc `k` of the time-expanded graph from `(i, t)` is
`n + (t - 1) * length(nbr) + ptr[i] - 1 + slot`.
"""
function _patrol_moves(n::Int, edges::Vector{Tuple{Int, Int}})
    adj = [Int[i] for i in 1:n]
    nb = [Int[] for _ in 1:n]
    for (a, b) in edges
        push!(nb[a], b)
        push!(nb[b], a)
    end
    ptr = Vector{Int}(undef, n + 1)
    nbr = Int[]
    ptr[1] = 1
    for i in 1:n
        append!(nbr, adj[i])
        append!(nbr, sort!(nb[i]))
        ptr[i + 1] = length(nbr) + 1
    end
    return ptr, nbr
end

"""
    _patrol_size_formula(n, n_edges, horizon, n_types) -> variables

    variables = n (start arcs) + (horizon - 1)(n + 2 n_edges) (stay/move arcs)
              + n horizon (coverage) + n_types
"""
_patrol_size_formula(n::Int, E::Int, H::Int, K::Int) = n + (H - 1) * (n + 2E) + n * H + K

"""
    _patrol_network(rng, n, n_edges) -> (positions, edges)

Street-grid network: `n` stations on a jittered grid `W` columns wide, a
uniformly shuffled Kruskal spanning tree of the grid links (connectivity), then
further shuffled grid links up to `n_edges` (clamped to what the grid holds).
"""
function _patrol_network(rng::AbstractRNG, n::Int, n_edges::Int)
    W = ceil(Int, sqrt(n))
    positions = [
        (100.0 * (((i - 1) % W) + 0.5 + 0.3 * randn(rng)) / W, 100.0 * (((i - 1) ÷ W) + 0.5 + 0.3 * randn(rng)) / W)
        for i in 1:n
    ]
    links = Tuple{Int, Int}[]
    for i in 1:n
        (i % W != 0 && i + 1 <= n) && push!(links, (i, i + 1))
        i + W <= n && push!(links, (i, i + W))
    end
    shuffle!(rng, links)
    parent = collect(1:n)
    function findroot(x)
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        return x
    end
    tree = Tuple{Int, Int}[]
    rest = Tuple{Int, Int}[]
    for (a, b) in links
        ra, rb = findroot(a), findroot(b)
        if ra != rb
            parent[ra] = rb
            push!(tree, (a, b))
        else
            push!(rest, (a, b))
        end
    end
    extra = clamp(n_edges - length(tree), 0, length(rest))
    return positions, sort!(vcat(tree, rest[1:extra]))
end

"""
    _patrol_best_path(w, n, H, ptr, nbr) -> stations

One unit's best patrol against node-time weights `w`: backward DP of the
best continuation value, then a forward argmax walk. Returns the station
visited in each period.
"""
function _patrol_best_path(w::Vector{Float64}, n::Int, H::Int, ptr::Vector{Int}, nbr::Vector{Int})
    ψ = similar(w)
    @inbounds for i in 1:n
        ψ[_patrol_nt(i, H, n)] = w[_patrol_nt(i, H, n)]
    end
    @inbounds for t in (H - 1):-1:1, i in 1:n
        m = -Inf
        for k in ptr[i]:(ptr[i + 1] - 1)
            m = max(m, ψ[_patrol_nt(nbr[k], t + 1, n)])
        end
        ψ[_patrol_nt(i, t, n)] = w[_patrol_nt(i, t, n)] + m
    end
    path = zeros(Int, H)
    best = -Inf
    for i in 1:n
        if ψ[i] > best
            best, path[1] = ψ[i], i
        end
    end
    @inbounds for t in 1:(H - 1)
        i = path[t]
        best, nxt = -Inf, i
        for k in ptr[i]:(ptr[i + 1] - 1)
            v = ψ[_patrol_nt(nbr[k], t + 1, n)]
            if v > best
                best, nxt = v, nbr[k]
            end
        end
        path[t + 1] = nxt
    end
    return path
end

"""
    _patrol_potentials(wcap, n, H, ptr, nbr) -> (Φ, λ)

Continuation potentials of the longest-path DP for capped weights `wcap`:
`Φ[(i, H)] = 0` and `Φ[(i, t)] = max_{j in moves(i)} (wcap[(j, t + 1)] +
Φ[(j, t + 1)])`, `λ = max_i (wcap[(i, 1)] + Φ[(i, 1)])`.
"""
function _patrol_potentials(wcap::Vector{Float64}, n::Int, H::Int, ptr::Vector{Int}, nbr::Vector{Int})
    Φ = zeros(n * H)
    @inbounds for t in (H - 1):-1:1, i in 1:n
        m = -Inf
        for k in ptr[i]:(ptr[i + 1] - 1)
            j = _patrol_nt(nbr[k], t + 1, n)
            m = max(m, wcap[j] + Φ[j])
        end
        Φ[_patrol_nt(i, t, n)] = m
    end
    λ = maximum(wcap[i] + Φ[i] for i in 1:n)
    return Φ, λ
end

"""
    PatrolSecurityProblem(target_variables, feasibility_status, seed)

Build a patrol security LP whose variable count is within a few percent of
`target_variables` (exact formula in `_patrol_size_formula`; the edge count is
tuned to close the gap); targets above `GAME_THEORY_MAX_VARIABLES` raise an
`ArgumentError`, tiny targets round up to a 4-station, 3-period game.
"""
function PatrolSecurityProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    _game_theory_check_target("patrol_security", target_variables)
    rng = MersenneTwister(seed)

    # --- Sizing: periods, types, stations, then links to close the gap.
    K = rand(rng, 2:6)
    H_max = clamp(target_variables ÷ 30, 3, 36)
    H = rand(rng, min(8, H_max):H_max)
    β = 1.15 + 0.6 * rand(rng)  # links per station (average degree 2.3-3.5)
    per_station = 1 + (H - 1) * (1 + 2β) + H
    n = max(4, round(Int, (target_variables - K) / per_station))
    W = ceil(Int, sqrt(n))
    grid_links = sum((i % W != 0 && i + 1 <= n) + (i + W <= n) for i in 1:n)
    E_star = round(Int, (target_variables - K - n - (H - 1) * n - n * H) / (2 * (H - 1)))
    n_edges = clamp(E_star, n - 1, grid_links)
    positions, edges = _patrol_network(rng, n, n_edges)
    ptr, nbr = _patrol_moves(n, edges)
    deg = [ptr[i + 1] - ptr[i] - 1 for i in 1:n]

    # --- Values: downtown hot spots x lognormal noise x hub bonus; rush hours.
    hubs = [(100 * rand(rng), 100 * rand(rng), 10 + 20 * rand(rng)) for _ in 1:rand(rng, 1:4)]
    station_value = [
        round(
            (0.3 + sum(exp(-((p[1] - h[1])^2 + (p[2] - h[2])^2) / (2 * h[3]^2)) for h in hubs)) *
            rand(rng, LogNormal(0.0, 0.5)) *
            (1 + 0.25 * (deg[i] - 2)) *
            10;
            digits=3,
        ) for (i, p) in enumerate(positions)
    ]
    station_value .= max.(station_value, 0.1)
    peak1, peak2 = 0.2 + 0.15 * rand(rng), 0.6 + 0.2 * rand(rng)
    time_profile = [
        round(
            0.4 + exp(-(((t - 0.5) / H - peak1) / 0.1)^2) + 0.8 * exp(-(((t - 0.5) / H - peak2) / 0.12)^2);
            digits=4,
        ) for t in 1:H
    ]

    # --- Attacker types: opportunistic type 1 plus focused groups.
    raw = [rand(rng, Gamma(2.0, 1.0)) for _ in 1:K]
    raw[1] *= 1.5
    type_prior = raw ./ sum(raw)
    type_penalty = [round(0.2 + 0.8 * rand(rng); digits=3) for _ in 1:K]
    option_type, option_target, option_gain, option_loss = Int[], Int[], Float64[], Float64[]
    for k in 1:K
        if k == 1
            stations, interest, window = 1:n, fill(0.6, n), 1:H
        else
            c = positions[rand(rng, 1:n)]
            r = 15 + 25 * rand(rng)
            interest = [1.5 * exp(-((p[1] - c[1])^2 + (p[2] - c[2])^2) / (2r^2)) for p in positions]
            stations = [i for i in 1:n if interest[i] >= 0.2]
            isempty(stations) && (stations = [argmax(interest)])
            len = max(1, ceil(Int, H * (0.3 + 0.5 * rand(rng))))
            t0 = rand(rng, 1:(H - len + 1))
            window = t0:(t0 + len - 1)
        end
        for t in window, i in stations
            gain = round(station_value[i] * time_profile[t] * interest[i]; digits=4)
            gain > 0 || continue
            push!(option_type, k)
            push!(option_target, _patrol_nt(i, t, n))
            push!(option_gain, gain)
            push!(option_loss, round(gain * (1 + type_penalty[k]); digits=4))
        end
    end
    n_units = clamp(round(Int, n * (0.04 + 0.08 * rand(rng))), 2, 16)

    # --- Approximate equilibrium: attacker Hedge vs greedy saturating patrols.
    NT = n * H
    n_arcs = n + (H - 1) * length(nbr)
    iterations = clamp(round(Int, 2.0e8 / (n_units * n_arcs + length(option_type))), 20, 200)
    by_type = [findall(==(k), option_type) for k in 1:K]
    η = [sqrt(8 * log(max(length(o), 2)) / iterations) / maximum(option_loss[o]) for o in by_type]
    cum = zeros(length(option_type))
    α = zeros(length(option_type))
    αsum = zeros(length(option_type))
    fsum = zeros(n_arcs)
    w = zeros(NT)
    count = zeros(Int, NT)
    for _ in 1:iterations
        for (k, o) in enumerate(by_type)
            m = maximum(cum[o])
            s = 0.0
            for j in o
                α[j] = exp(η[k] * (cum[j] - m))
                s += α[j]
            end
            for j in o
                α[j] /= s
            end
        end
        αsum .+= α
        fill!(w, 0.0)
        for j in eachindex(α)
            w[option_target[j]] += type_prior[option_type[j]] * α[j] * option_loss[j]
        end
        fill!(count, 0)
        for _ in 1:n_units
            path = _patrol_best_path(w, n, H, ptr, nbr)
            fsum[path[1]] += 1
            for t in 1:H
                nt = _patrol_nt(path[t], t, n)
                count[nt] += 1
                w[nt] = 0.0  # saturated: a second unit adds no coverage
                if t < H
                    slot = findfirst(==(path[t + 1]), view(nbr, ptr[path[t]]:(ptr[path[t] + 1] - 1)))
                    fsum[n + (t - 1) * length(nbr) + ptr[path[t]] - 1 + slot] += 1
                end
            end
        end
        for j in eachindex(cum)
            cum[j] += option_gain[j] - option_loss[j] * min(1, count[option_target[j]])
        end
    end
    fbar = fsum ./ iterations
    αbar = αsum ./ iterations

    # Upper bound: the average patrol plan and each type's best response.
    inflow = zeros(NT)
    for i in 1:n
        inflow[i] = fbar[i]
    end
    for t in 1:(H - 1), i in 1:n, k in ptr[i]:(ptr[i + 1] - 1)
        inflow[_patrol_nt(nbr[k], t + 1, n)] += fbar[n + (t - 1) * length(nbr) + k]
    end
    coverage = min.(1.0, inflow)
    type_values = [maximum(option_gain[j] - option_loss[j] * coverage[option_target[j]] for j in o) for o in by_type]
    upper = sum(type_prior .* type_values)

    # Lower bound: average attacker mix, best threshold-Lagrangian bound,
    # compared with the trivial everything-covered certificate.
    function lagrangian_bound(mix)
        wb = zeros(NT)
        base = 0.0
        for j in eachindex(mix)
            pk = type_prior[option_type[j]]
            wb[option_target[j]] += pk * mix[j] * option_loss[j]
            base += pk * mix[j] * option_gain[j]
        end
        levels = unique!(sort!(vcat(0.0, maximum(wb), [quantile(wb, q) for q in 0.5:0.05:1.0])))
        best = (-Inf, 0.0, zeros(NT), 0.0)
        for θ in levels
            wcap = min.(wb, θ)
            Φ, λ = _patrol_potentials(wcap, n, H, ptr, nbr)
            val = base - sum(max(x - θ, 0.0) for x in wb) - n_units * λ
            val > best[1] && (best = (val, θ, Φ, λ))
        end
        return best
    end
    lb1 = lagrangian_bound(αbar)
    trivial = zeros(length(option_type))
    for o in by_type
        trivial[o[argmax(option_gain[o] .- option_loss[o])]] = 1.0
    end
    lb0 = lagrangian_bound(trivial)
    lower, θ, Φ, λ = lb1[1] >= lb0[1] ? lb1 : lb0
    mix = lb1[1] >= lb0[1] ? αbar : trivial

    unit = sum(type_prior[k] * maximum(option_gain[o]) for (k, o) in enumerate(by_type))
    loss_requirement = -_game_value_requirement(rng, feasibility_status, -upper, -lower, unit)
    if feasibility_status == infeasible && lb1[1] > lb0[1]
        # Keep the requirement above the trivial everything-covered bound
        # (what bound propagation through single attack rows can derive), so
        # the refutation needs the whole model rather than presolve.
        loss_requirement = min(
            max(loss_requirement, lb0[1] + 0.5 * (lower - lb0[1])), lower - 0.005 * unit
        )
    end
    witness = nothing
    certificate = nothing
    if feasibility_status == feasible
        witness = PatrolSecurityWitness(fbar, coverage, type_values, upper)
    elseif feasibility_status == infeasible
        certificate = PatrolSecurityCertificate(mix, θ, Φ, λ, lower)
    end

    return PatrolSecurityProblem(
        n,
        H,
        n_units,
        positions,
        edges,
        station_value,
        time_profile,
        type_prior,
        type_penalty,
        option_type,
        option_target,
        option_gain,
        option_loss,
        lower,
        upper,
        loss_requirement,
        iterations,
        witness,
        certificate,
        feasibility_status,
    )
end

"""
    build_model(prob::PatrolSecurityProblem)

Deterministic patrol LP (see the type docstring): `f` time-expanded arc flows
(start arcs first, then per period the stay/move arcs in `_patrol_moves`
order), `c` coverage per node-time, `v` per attacker type.
"""
function build_model(prob::PatrolSecurityProblem)
    n, H, K = prob.n_stations, prob.horizon, length(prob.type_prior)
    ptr, nbr = _patrol_moves(n, prob.edges)
    L = length(nbr)
    n_arcs = n + (H - 1) * L
    NT = n * H

    model = Model()
    @variable(model, f[1:n_arcs] >= 0)
    @variable(model, 0 <= c[1:NT] <= 1)
    @variable(model, v[1:K])
    loss = AffExpr(0.0)
    for k in 1:K
        add_to_expression!(loss, prob.type_prior[k], v[k])
    end
    @objective(model, Min, loss)

    inflow = [AffExpr(0.0) for _ in 1:NT]
    outflow = [AffExpr(0.0) for _ in 1:NT]
    for i in 1:n
        add_to_expression!(inflow[i], 1.0, f[i])
    end
    for t in 1:(H - 1), i in 1:n, k in ptr[i]:(ptr[i + 1] - 1)
        a = n + (t - 1) * L + k
        add_to_expression!(outflow[_patrol_nt(i, t, n)], 1.0, f[a])
        add_to_expression!(inflow[_patrol_nt(nbr[k], t + 1, n)], 1.0, f[a])
    end

    @constraint(model, sum(f[i] for i in 1:n) <= prob.n_units)
    for t in 1:(H - 1), i in 1:n
        nt = _patrol_nt(i, t, n)
        @constraint(model, inflow[nt] - outflow[nt] == 0)
    end
    for nt in 1:NT
        @constraint(model, c[nt] - inflow[nt] <= 0)
    end
    for j in eachindex(prob.option_type)
        @constraint(
            model, v[prob.option_type[j]] + prob.option_loss[j] * c[prob.option_target[j]] >= prob.option_gain[j]
        )
    end
    @constraint(model, loss <= prob.loss_requirement)
    return model
end

register_variant(
    :game_theory,
    :patrol_security,
    PatrolSecurityProblem,
    "Bayesian zero-sum patrol security game (TRUSTS/PROTECT-style compact LP): a time-expanded patrol flow over a street-grid network feeds bounded coverage variables that couple into attacker-type epigraph rows, minimizing expected loss under a risk-budget requirement certified by a Hedge/greedy-patrol equilibrium approximation and a threshold-Lagrangian longest-path lower bound",
)
