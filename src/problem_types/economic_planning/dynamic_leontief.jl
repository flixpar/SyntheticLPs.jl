using JuMP
using Random
using Distributions
using LinearAlgebra
using SparseArrays

"""
Largest `target_variables` accepted by `DynamicLeontiefProblem`. The
input–output and capital matrices are stored sparse and every period block is
materialised in the JuMP model, so larger targets are rejected with an
`ArgumentError` instead of being silently undersized (same convention as
`telecom_network_design/standard` and `supply_chain/network_planning`).
"""
const LEONTIEF_MAX_VARIABLES = 1_000_000

"""Largest sector count the dimension search will use (≈ an MRIO-scale table)."""
const LEONTIEF_MAX_SECTORS = 10_000

"""
    LeontiefWitness

A planted feasible plan in the model's own layout (hybrid units): gross output
`output[s, t]`, aggregate consumption `consumption[t]`, annual new capacity
`new_capacity[s, t]` (entering service at the start of period `t + 1`), capacity
stock `capacity[s, t]` at the start of period `t + 1`, `imports[i, t]` /
`exports[i, t]` for the `i`-th tradable sector, and net foreign debt `debt[t]`
at the start of period `t + 1`.

It is the balanced-growth path of the sampled economy: the stationary output
vector solves `(I - A - BΓ + diag(ηρ)) x̄ = d C̄ + Ḡ + ē` (all terms
nonnegative, the matrix an M-matrix), every period scales it by the growth
factor, capacity sits at a planted utilisation below one, and the final period
is re-solved because investment for post-horizon capacity is not in the model.
"""
struct LeontiefWitness
    output::Matrix{Float64}
    consumption::Vector{Float64}
    new_capacity::Matrix{Float64}
    capacity::Matrix{Float64}
    imports::Matrix{Float64}
    exports::Matrix{Float64}
    debt::Vector{Float64}
end

"""
    LeontiefCertificate

Farkas certificate for an infeasible plan target, built from LP rows of every
period (it survives every transform). With `M = I + diag(ρ̃) - A` (ρ̃ the import
ceilings, zero on non-tradables), for each period `t`:

  - weights `π_t ≥ 0` on the commodity-balance equalities and on the
    import-ceiling rows `m ≤ ρ x` (tradable entries) give
    `π_t' M x_t ≥ π_t' (G_t + d C_t)`, because investment, exports and imports
    below the ceiling only add demand;
  - labor weights `μ_t ≥ 0` (per skill) and, in period 1 only, capacity weights
    `ν ≥ 0` (capacity is data there) satisfy `M' π_t ≤ Σ_k μ_{k,t} ℓ_k + ν`
    elementwise, so `π_t' M x_t ≤ μ_t' L_t + ν' K0`.

Hence `C_t ≤ consumption_bounds[t] = (μ_t'L_t + ν'K0 - π_t'G_t) / (π_t'd)`.
`π_t` is the labor content of the import-augmented Leontief inverse,
`M⁻ᵀ ℓ_k`, for the period's binding skill, or — in period 1 of a
`:labor_capital` certificate — `M⁻ᵀ e_s` for a bottleneck sector `s` whose
installed capacity binds first. The stored matrices are already scaled by
`w_t / (π_t'd)` (`w` the target's period weights), so adding them to the
plan-target row (multiplier 1) cancels every consumption column and refutes
`Σ_t w_t C_t ≥ consumption_target`: the target exceeds
`Σ_t w_t consumption_bounds[t]` by a planted 10–30% margin.
"""
struct LeontiefCertificate
    mode::Symbol
    balance_multipliers::Matrix{Float64}
    import_multipliers::Matrix{Float64}
    labor_multipliers::Matrix{Float64}
    capacity_multipliers::Vector{Float64}
    consumption_bounds::Vector{Float64}
    consumption_target::Float64
end

"""
    DynamicLeontiefProblem <: ProblemGenerator

Generator for multi-sector, multi-period dynamic input–output planning LPs in
the tradition of Dantzig's dynamic Leontief model, the Stanford PILOT
energy–economy model (Netlib `PILOT*`), and development-planning LPs.

# Model (per period `t = 1..T`, period length `Δ` years, flows per year)

  - commodity balance (equality, one per sector):
    `x - A x + m - e - B((1-φ)∘N_t + φ∘N_{t+1}) - d C = G`
  - capacity: `x_t ≤ K_{t-1}` (initial capacity `K0` is data)
  - capital accumulation: `K_t = σ∘K_{t-1} + Δ N_t` (`σ = (1-δ)^Δ`)
  - import ceilings on tradables: `m ≤ ρ∘x`
  - labor by skill class: `ℓ_k' x ≤ L_{k,t}`
  - external debt: `F_t = R F_{t-1} + Δ (p_m' m - p_e' e)`, `F_t ≤ F̄_t`
  - consumption floor `C_t ≥ C̲_t` and no decline `C_{t+1} ≥ C_t`
  - plan target on cumulative discounted consumption `Σ_t w_t C_t ≥ W`
  - terminal capacity `K_T ≥ K_term`, exports `e ≤ ē`

maximizing discounted consumption plus the discounted value of terminal capital
net of terminal debt. Each period couples to the next only through capital
stocks, in-progress investment (gestation share `φ`) and debt — the staircase
structure of the classical models. Gross investment is sector-specific capacity
`N` bought from the capital-goods sectors (construction and equipment
manufacturing) through the sparse capital-coefficient matrix `B`.

# Data grounding

Sectors are ordered primary / manufacturing / construction / services (production
stage), grouped into sub-industry clusters and value chains, with at most eight
"hub" suppliers (energy, trade, transport, finance) tying the chains together;
flows run mostly downstream within a chain, so the table is close to
block-triangular as disaggregated real tables are. Intermediate-input shares per
column follow the block (manufacturing 55–75% of gross output, services
25–45%), so the value table has column sums below one (productive,
Hawkins–Simon). The table is dense for small economies and sparse
(≈ 6 + 2·ln n suppliers per column) for large ones. Capital–output ratios,
structures/equipment splits, gestation shares, depreciation, labor intensities
(with an economy-wide productivity level), import ceilings, and
consumption/government bundles are block-specific; capital goods are sourced
mostly within the buyer's value chain. As in PILOT-style hybrid tables, most
primary and some manufacturing sectors are measured in physical units (PJ, Mt)
with a sampled price, so the stored matrices are a diagonal similarity of the
value table: coefficients span many orders of magnitude while productivity is
preserved.

# Solver profile

Like PILOT, these LPs have a dense basis inverse: dual prices are value
contents, which propagate through the whole economy and across periods via the
capital stocks, so BTRAN rows of a typical basis are ~70–80% dense and LU fill
is ~20x. Per-iteration simplex cost is therefore far higher than on network-like
LPs of the same size (milliseconds at 10k columns); full solves beyond ~20k
columns take minutes.

# Feasibility control

  - `feasible`: a balanced-growth reference trajectory is planted
    (`feasible_witness`); labor supply, debt ceilings, export ceilings, terminal
    capacity, consumption floors and the plan target are set with margins below
    it.
  - `infeasible`: the plan target is set 10–30% above `Σ_t w_t b_t`, where `b_t`
    is a provable upper bound on period-`t` consumption — the labor force through
    the labor content of the import-augmented Leontief inverse (`:labor`), and
    in period 1 possibly the installed capacity of a bottleneck sector
    (`:labor_capital`) — with a `LeontiefCertificate` aggregating every balance,
    import-ceiling and labor row of every period. Nothing is visible in a single
    row, and the per-period floors stay comfortable, so presolve bound
    propagation does not reach the contradiction.
  - `unknown`: the target sits 10–80% of the way from the reference path's
    `Σ_t w_t C_t` to `Σ_t w_t b_t`; the true frontier lies in between (measured
    at 15–75%, typically ~40%), so instances land on both sides; no claim is
    stored.

# Sizing

Variables are exactly `T (3n + 2 n_tr + 2)` for `n` sectors (`n_tr` tradable)
over `T ≤ 25` periods (`n ≤ LEONTIEF_MAX_SECTORS`, an MRIO-scale table); the
dimension search lands within a couple of `T` of the target (≤ 4% above 100
variables, < 0.1% above 10k; the minimum instance has 45 variables). Rows are
`T (3n + n_tr + n_skills + 1) + T`. Targets above
`LEONTIEF_MAX_VARIABLES` raise an `ArgumentError`.
"""
struct DynamicLeontiefProblem <: ProblemGenerator
    n_sectors::Int
    n_periods::Int
    period_length::Int
    sector_block::Vector{Symbol}
    tradable::Vector{Int}
    unit_price::Vector{Float64}
    A::SparseMatrixCSC{Float64, Int}
    B::SparseMatrixCSC{Float64, Int}
    gestation_share::Vector{Float64}
    survival::Vector{Float64}
    consumption_bundle::Vector{Float64}
    government_demand::Matrix{Float64}
    labor_coef::Matrix{Float64}
    labor_supply::Matrix{Float64}
    import_ceiling::Vector{Float64}
    import_price::Vector{Float64}
    export_price::Vector{Float64}
    export_ceiling::Matrix{Float64}
    initial_capacity::Vector{Float64}
    terminal_capacity::Vector{Float64}
    initial_debt::Float64
    interest_factor::Float64
    debt_ceiling::Vector{Float64}
    consumption_floor::Vector{Float64}
    consumption_target::Float64
    consumption_weight::Vector{Float64}
    terminal_capital_value::Vector{Float64}
    terminal_debt_weight::Float64
    feasible_witness::Union{Nothing, LeontiefWitness}
    infeasibility_certificate::Union{Nothing, LeontiefCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    _leontief_dimensions(rng, target) -> (T, n, n_tr)

Choose periods `T`, sectors `n` and tradable sectors `n_tr` so that
`T (3n + 2 n_tr + 2)` is as close as possible to `target`, with a sampled
preferred horizon and tradable share as tie-breakers (both matter only when
many combinations hit the target almost exactly, i.e. at larger sizes).
"""
function _leontief_dimensions(rng::AbstractRNG, target::Int)
    T_hi = clamp(target ÷ 15, 3, 25)
    T_lo = clamp(cld(target, 4 * LEONTIEF_MAX_SECTORS), 3, T_hi)
    T_top = max(T_lo, min(T_hi, 20))
    T_pref = round(Int, exp(rand(rng, Uniform(log(T_lo), log(T_top) + 1e-9))))
    f_pref = rand(rng, Uniform(0.45, 0.62))
    best = (3, 3, 2)
    best_score = Inf
    for T in T_lo:T_hi
        for P in max(13, fld(target, T) - 1):(cld(target, T) + 1)
            center = clamp((P - 2) / (3 + 2 * f_pref), 3.0, Float64(LEONTIEF_MAX_SECTORS))
            for n in max(3, floor(Int, center) - 2):min(LEONTIEF_MAX_SECTORS, ceil(Int, center) + 2)
                rem = P - 2 - 3n
                (rem < 0 || isodd(rem)) && continue
                n_tr = rem ÷ 2
                (2 <= n_tr <= n - 1) || continue
                f = n_tr / n
                (0.3 <= f <= 0.75) || continue
                score =
                    10 * abs(T * P - target) / target + abs(log(T / T_pref)) + abs(f - f_pref)
                if score < best_score
                    best_score = score
                    best = (T, n, n_tr)
                end
            end
        end
    end
    # Only the 45-variable minimum instance has no candidate near the target.
    isfinite(best_score) || target <= 60 ||
        error("dynamic_leontief: no (T, n, n_tr) combination found for target $target")
    return best
end

"""
    _leontief_solve(S, q, b; transposed=false) -> x

Solve `(I + diag(q) - S) x = b` (or the transposed system) for a nonnegative
sparse `S` whose column sums are at most ~0.9 and `q ≥ 0`, by Jacobi iteration.
The iteration matrix has 1-norm (∞-norm for the transposed system) below one,
so it converges geometrically; for `b ≥ 0` every iterate, and the limit, is
nonnegative (an M-matrix inverse). Avoids sparse LU, whose fill-in on a random
sparse input–output pattern is near dense at thousands of sectors.
"""
function _leontief_solve(
    S::SparseMatrixCSC{Float64, Int},
    q::Vector{Float64},
    b::Vector{Float64};
    transposed::Bool=false,
    tol::Float64=1e-15,
    maxiter::Int=20_000,
)
    d = 1.0 .+ q
    x = b ./ d
    y = similar(x)
    for _ in 1:maxiter
        if transposed
            mul!(y, transpose(S), x)
        else
            mul!(y, S, x)
        end
        y .= (b .+ y) ./ d
        diff = maximum(abs.(y .- x); init=0.0)
        scale = maximum(abs, y; init=0.0)
        x, y = y, x
        diff <= tol * max(scale, 1e-300) && break
    end
    return x
end

"""
    _leontief_cumweights(weights) -> Vector{Float64}

Cumulative weights for inverse-CDF sampling with `searchsortedfirst`.
"""
_leontief_cumweights(w::Vector{Float64}) = cumsum(w)

"""
    _leontief_io_matrix(rng, blocks, colshare, hubs, cluster_members, cluster_of)
        -> (A::SparseMatrixCSC, chain_of_sector::Vector{Int})

Sample the value-unit technical-coefficient matrix. Sectors are indexed in
production-stage order (primary → manufacturing → construction → services, and
by processing stage within a block). Disaggregated input–output tables are close
to block-triangular in such an order (Simpson & Tsukui's triangularization
finding) and organised in value chains (agri-food, metals–machinery,
chemicals–plastics, ...): most intermediate flows run downstream within a chain,
feedback is concentrated in tightly knit sub-industry clusters, and the chains
are tied together by a few ubiquitous "hub" suppliers (energy, transport, trade,
finance).

Sub-industry clusters are dealt round-robin (per block, from a random offset)
into `max(1, round(n / 40))` value chains. Column `j` (buyer) draws its supplier
set as: itself (own-industry purchases), then — with probability 0.3 — a member
of its own cluster, with probability 0.15 a hub, otherwise an upstream member of
its own chain (index below its cluster) from a block-affinity-weighted
distribution. It splits its intermediate input share `colshare[j]` across them
with lognormal weights (diagonal boosted). Column sums equal `colshare` exactly,
so the table is productive.

The chain locality keeps the Leontief inverse — and hence every simplex basis
containing a period's `I - A` block — from being fully dense: with suppliers
drawn from the whole upstream range the transitive closure covers almost every
sector, and per-iteration cost explodes at a few hundred sectors.
"""
function _leontief_io_matrix(
    rng::AbstractRNG,
    blocks::Vector{Int},
    colshare::Vector{Float64},
    hubs::Vector{Int},
    cluster_members::Vector{Vector{Int}},
    cluster_of::Vector{Int},
)
    n = length(blocks)
    # Supplier-block (row) x buyer-block (column) affinities: P, M, C, S.
    affinity = [
        3.0 2.0 1.5 0.7
        1.5 4.0 3.0 1.5
        0.3 0.2 1.0 0.5
        1.5 2.0 1.5 4.0
    ]
    n_chains = max(1, round(Int, n / 40))
    chain_of_cluster = zeros(Int, length(cluster_members))
    for b in 1:4
        offset = rand(rng, 0:(n_chains - 1))
        k = 0
        for (c, members) in enumerate(cluster_members)
            blocks[first(members)] == b || continue
            chain_of_cluster[c] = mod(k + offset, n_chains) + 1
            k += 1
        end
    end
    chain_members = [Int[] for _ in 1:n_chains]
    for i in 1:n
        push!(chain_members[chain_of_cluster[cluster_of[i]]], i)
    end
    chain_cums = [
        [_leontief_cumweights([affinity[blocks[i], b] for i in members]) for b in 1:4] for
        members in chain_members
    ]

    k_typ = min(0.75 * n, 6.0 + 2.0 * log(n))
    I = Int[]
    J = Int[]
    V = Float64[]
    sizehint!(I, ceil(Int, 1.3 * k_typ * n))
    sizehint!(J, ceil(Int, 1.3 * k_typ * n))
    sizehint!(V, ceil(Int, 1.3 * k_typ * n))
    chosen = Int[]
    for j in 1:n
        k = clamp(round(Int, k_typ * exp(0.3 * randn(rng))), min(2, n), n)
        empty!(chosen)
        push!(chosen, j)
        members = cluster_members[cluster_of[j]]
        ch = chain_of_cluster[cluster_of[j]]
        # Chain members strictly upstream of j's cluster are a prefix of the chain list.
        n_up = searchsortedlast(chain_members[ch], first(members) - 1)
        cum = chain_cums[ch][blocks[j]]
        attempts = 0
        while length(chosen) < k && attempts < 50k
            attempts += 1
            u = rand(rng)
            i = if u < 0.3 || (n_up == 0 && u >= 0.45)
                members[rand(rng, 1:length(members))]
            elseif u < 0.45
                hubs[rand(rng, 1:length(hubs))]
            else
                chain_members[ch][min(n_up, searchsortedfirst(cum, rand(rng) * cum[n_up]))]
            end
            i in chosen || push!(chosen, i)
        end
        w = [rand(rng, LogNormal(0.0, 1.1)) for _ in chosen]
        w[1] *= 2.5
        total = sum(w)
        for (idx, i) in enumerate(chosen)
            push!(I, i)
            push!(J, j)
            push!(V, colshare[j] * w[idx] / total)
        end
    end
    chain_of_sector = [chain_of_cluster[cluster_of[i]] for i in 1:n]
    return sparse(I, J, V, n, n), chain_of_sector
end

"""
    DynamicLeontiefProblem(target_variables, feasibility_status, seed)

Construct a dynamic Leontief planning instance with about `target_variables`
columns (see the type docstring for the exact count). Targets above
`LEONTIEF_MAX_VARIABLES` raise an `ArgumentError`; the smallest instance has 45
variables (3 sectors, 2 tradable, 3 periods).
"""
function DynamicLeontiefProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be >= 1 (got $target_variables)."))
    target_variables <= LEONTIEF_MAX_VARIABLES || throw(
        ArgumentError(
            "economic_planning/dynamic_leontief supports at most $LEONTIEF_MAX_VARIABLES " *
            "variables; requested $target_variables.",
        ),
    )
    rng = MersenneTwister(seed)

    T, n, n_tr = _leontief_dimensions(rng, target_variables)
    Δ = rand(rng, (1, 1, 2, 5))

    # ---- Sector blocks: tradables are primary + manufacturing ----------------
    n_P = clamp(round(Int, 0.3 * n_tr), 1, n_tr - 1)
    n_M = n_tr - n_P
    n_nt = n - n_tr
    n_C = n_nt >= 3 ? max(1, round(Int, 0.1 * n_nt)) : (n_nt == 2 ? 1 : 0)
    n_S = n_nt - n_C
    blocks = vcat(fill(1, n_P), fill(2, n_M), fill(3, n_C), fill(4, n_S))
    block_names = (:primary, :manufacturing, :construction, :services)
    sector_block = [block_names[b] for b in blocks]
    tradable = collect(1:n_tr)

    # Sub-industry clusters of consecutive sectors within each block.
    cluster_of = zeros(Int, n)
    cluster_members = Vector{Int}[]
    s = 1
    while s <= n
        len = rand(rng, 4:12)
        stop = s
        while stop < n && stop - s + 1 < len && blocks[stop + 1] == blocks[s]
            stop += 1
        end
        push!(cluster_members, collect(s:stop))
        cluster_of[s:stop] .= length(cluster_members)
        s = stop + 1
    end

    # Hub suppliers: energy-type primary sectors and trade/transport/finance services.
    # A handful (≤ 8) regardless of size: real tables have few truly ubiquitous suppliers.
    hub_pool = [i for i in 1:n if blocks[i] == 1 || blocks[i] == 4]
    n_hub = clamp(round(Int, 0.03 * n), 1, min(8, length(hub_pool)))
    hubs = sort(shuffle(rng, hub_pool)[1:n_hub])

    # Capital-goods suppliers: construction plus an equipment subset of manufacturing.
    manuf = findall(==(2), blocks)
    n_eq = max(1, round(Int, 0.25 * n_M))
    equipment = sort(shuffle(rng, manuf)[1:n_eq])
    construction = findall(==(3), blocks)

    # ---- Macro parameters ----------------------------------------------------
    g = rand(rng, Uniform(0.01, 0.06))                # annual growth
    Gp = (1 + g)^Δ
    β = rand(rng, Uniform(0.95, 0.985))               # annual discount factor
    r_int = rand(rng, Uniform(0.02, 0.06))
    Rp = (1 + r_int)^Δ
    Y = exp(rand(rng, Uniform(log(20.0), log(3000.0))))  # economy size, $bn value added
    productivity = exp(rand(rng, Uniform(log(0.25), log(3.0))))

    # ---- Sector parameters (value units) --------------------------------------
    colshare = Vector{Float64}(undef, n)
    kappa = Vector{Float64}(undef, n)      # capital per unit of capacity
    struct_share = Vector{Float64}(undef, n)
    labor_base = Vector{Float64}(undef, n) # thousand workers per $bn gross output
    agri = falses(n)
    for j in 1:n
        b = blocks[j]
        if b == 1
            agri[j] = rand(rng) < 0.4
            colshare[j] = rand(rng, Uniform(0.30, 0.55))
            kappa[j] = exp(rand(rng, Uniform(log(1.0), log(4.0))))
            struct_share[j] = rand(rng, Uniform(0.4, 0.8))
            labor_base[j] = agri[j] ? rand(rng, Uniform(15.0, 45.0)) : rand(rng, Uniform(1.0, 4.0))
        elseif b == 2
            colshare[j] = rand(rng, Uniform(0.55, 0.75))
            kappa[j] = exp(rand(rng, Uniform(log(0.6), log(1.8))))
            struct_share[j] = rand(rng, Uniform(0.25, 0.5))
            labor_base[j] = rand(rng, Uniform(2.0, 8.0))
        elseif b == 3
            colshare[j] = rand(rng, Uniform(0.50, 0.65))
            kappa[j] = exp(rand(rng, Uniform(log(0.3), log(0.8))))
            struct_share[j] = rand(rng, Uniform(0.15, 0.35))
            labor_base[j] = rand(rng, Uniform(6.0, 14.0))
        else
            colshare[j] = rand(rng, Uniform(0.25, 0.45))
            kappa[j] = exp(rand(rng, Uniform(log(0.5), log(3.0))))
            struct_share[j] = rand(rng, Uniform(0.4, 0.85))
            labor_base[j] = rand(rng, Uniform(4.0, 20.0))
        end
    end
    isempty(construction) && (struct_share .= 0.0)
    gestation = [
        (blocks[j] == 1 || struct_share[j] > 0.6) ? rand(rng, Uniform(0.3, 0.6)) :
        rand(rng, Uniform(0.0, 0.2)) for j in 1:n
    ]
    delta = [
        clamp(
            struct_share[j] * 0.03 + (1 - struct_share[j]) * 0.12 + 0.01 * randn(rng),
            0.02,
            0.15,
        ) for j in 1:n
    ]
    survival = (1 .- delta) .^ Δ
    util = rand(rng, Uniform(0.78, 0.95), n)

    # Investment per unit of output on the balanced-growth path, and the
    # productivity cap a_j + κ_j Γ_j ≤ 0.9 that keeps the growth system an M-matrix.
    gam = [
        ((1 - gestation[j]) + gestation[j] * Gp) * (Gp - survival[j]) / (Δ * util[j]) for j in 1:n
    ]
    for j in 1:n
        if kappa[j] * gam[j] > 0.75
            kappa[j] = 0.75 / gam[j]
        end
        colshare[j] = min(colshare[j], max(0.1, 0.9 - kappa[j] * gam[j]))
    end

    A_v, chain_of = _leontief_io_matrix(rng, blocks, colshare, hubs, cluster_members, cluster_of)

    # Capital coefficient matrix: structures from a construction sector, equipment
    # from 1–4 equipment-manufacturing suppliers. Capital goods are mostly
    # sourced within the buyer's value chain (specialised machinery: tractors
    # for agriculture, rolling mills for steel), so each pick comes from the
    # chain's own suppliers with probability 0.75 when it has any.
    n_chains = maximum(chain_of)
    eq_by_chain = [Int[] for _ in 1:n_chains]
    foreach(i -> push!(eq_by_chain[chain_of[i]], i), equipment)
    con_by_chain = [Int[] for _ in 1:n_chains]
    foreach(i -> push!(con_by_chain[chain_of[i]], i), construction)
    function pick(own::Vector{Int}, all::Vector{Int})
        pool = (!isempty(own) && rand(rng) < 0.75) ? own : all
        return pool[rand(rng, 1:length(pool))]
    end
    BI = Int[]
    BJ = Int[]
    BV = Float64[]
    suppliers = Int[]
    for j in 1:n
        if !isempty(construction) && struct_share[j] > 0
            push!(BI, pick(con_by_chain[chain_of[j]], construction))
            push!(BJ, j)
            push!(BV, kappa[j] * struct_share[j])
        end
        k_eq = min(length(equipment), rand(rng, 1:4))
        empty!(suppliers)
        while length(suppliers) < k_eq
            i = pick(eq_by_chain[chain_of[j]], equipment)
            i in suppliers || push!(suppliers, i)
        end
        w = [rand(rng, LogNormal(0.0, 0.7)) for _ in suppliers]
        eq_total = kappa[j] * (1 - (isempty(construction) ? 0.0 : struct_share[j]))
        for (idx, i) in enumerate(suppliers)
            push!(BI, i)
            push!(BJ, j)
            push!(BV, eq_total * w[idx] / sum(w))
        end
    end
    B_v = sparse(BI, BJ, BV, n, n)

    # Final-demand bundles (value shares).
    function bundle(block_share::NTuple{4, Float64}, zero_prob::NTuple{4, Float64})
        w = zeros(n)
        for b in 1:4
            members = findall(==(b), blocks)
            (isempty(members) || block_share[b] == 0) && continue
            raw = [rand(rng) < zero_prob[b] ? 0.0 : rand(rng, LogNormal(0.0, 1.0)) for _ in members]
            all(iszero, raw) && (raw[rand(rng, 1:length(raw))] = 1.0)
            w[members] .= block_share[b] .* raw ./ sum(raw)
        end
        return w ./ sum(w)
    end
    d_v = bundle((0.10, 0.30, 0.0, 0.60), (0.4, 0.4, 1.0, 0.15))
    gov_v = bundle((0.05, 0.10, 0.0, 0.85), (0.7, 0.7, 1.0, 0.5))

    # Trade: import ceilings ρ, witness import fraction η, export intensity, prices.
    rho = rand(rng, Uniform(0.1, 0.7), n_tr)
    eta = rand(rng, Uniform(0.3, 0.8), n_tr)
    export_intensity = [rand(rng, LogNormal(log(0.15), 0.6)) for _ in 1:n_tr]
    export_intensity .= clamp.(export_intensity, 0.02, 0.6)
    pm_v = rand(rng, Uniform(1.0, 1.25), n_tr)
    pe_v = rand(rng, Uniform(0.8, 1.0), n_tr)
    q_imp = zeros(n)
    q_imp[tradable] .= eta .* rho
    q_ceiling = zeros(n)
    q_ceiling[tradable] .= rho

    # Labor by skill class (thousand workers per $bn of gross output).
    n_skills = n >= 6 ? rand(rng, 2:3) : 1
    labor_v = zeros(n_skills, n)
    for j in 1:n
        shares = [rand(rng, Gamma(2.0, 1.0)) for _ in 1:n_skills]
        if n_skills > 1
            # Services and equipment lean skilled; agriculture and construction unskilled.
            tilt = blocks[j] == 4 ? 1.6 : (agri[j] || blocks[j] == 3 ? 0.6 : 1.0)
            shares[end] *= tilt
            shares[1] /= tilt
        end
        labor_v[:, j] .= productivity * labor_base[j] .* shares ./ sum(shares)
    end

    # ---- Balanced-growth reference path (value units) --------------------------
    BGamma = B_v * Diagonal(gam)
    S_growth = A_v + BGamma
    C_bar = 0.62 * Y * rand(rng, Uniform(0.9, 1.1))
    G_bar = 0.18 * Y * rand(rng, Uniform(0.8, 1.2)) .* gov_v
    x_closed = _leontief_solve(S_growth, q_imp, d_v .* C_bar .+ G_bar)
    e_bar = zeros(n_tr)
    e_bar .= export_intensity .* x_closed[tradable]
    e_full = zeros(n)
    e_full[tradable] .= e_bar
    x_bar = _leontief_solve(S_growth, q_imp, d_v .* C_bar .+ G_bar .+ e_full)
    x_bar .= max.(x_bar, 0.0)
    K_start1 = x_bar ./ util                           # capacity at the start of period 1

    grow(t) = Gp^(t - 1)
    xw = zeros(n, T)
    Nw = zeros(n, T)
    Kw = zeros(n, T)
    for t in 1:T
        Kw[:, t] .= K_start1 .* grow(t + 1)          # stock at the start of t+1
        Nw[:, t] .= K_start1 .* grow(t) .* (Gp .- survival) ./ Δ
        xw[:, t] .= x_bar .* grow(t)
    end
    Cw = [C_bar * grow(t) for t in 1:T]
    Gmat = hcat([G_bar .* grow(t) for t in 1:T]...)
    ew = hcat([e_bar .* grow(t) for t in 1:T]...)
    # Last period: no investment for post-horizon capacity, so solve it directly.
    inv_T = B_v * ((1 .- gestation) .* Nw[:, T])
    xw[:, T] .= _leontief_solve(A_v, q_imp, d_v .* Cw[T] .+ Gmat[:, T] .+ e_full .* grow(T) .+ inv_T)
    xw[:, T] .= max.(xw[:, T], 0.0)
    mw = zeros(n_tr, T)
    for t in 1:T
        mw[:, t] .= eta .* rho .* xw[tradable, t]
    end
    F0 = Y * rand(rng, Uniform(-0.1, 0.5))
    Fw = zeros(T)
    prevF = F0
    for t in 1:T
        Fw[t] = Rp * prevF + Δ * (sum(pm_v .* mw[:, t]) - sum(pe_v .* ew[:, t]))
        prevF = Fw[t]
    end

    # ---- Resource levels around the reference path ---------------------------
    labor_ref = labor_v * x_bar                        # per skill, period 1 (steady path)
    labor_slack = rand(rng, Uniform(0.03, 0.12), n_skills)
    floor_factor = rand(rng, Uniform(0.80, 0.97))
    debt_margin = rand(rng, Uniform(0.05, 0.30))
    term_factor = rand(rng, Uniform(0.85, 0.97))
    L_v = hcat([labor_ref .* (1 .+ labor_slack) .* grow(t) for t in 1:T]...)
    debt_ceiling = [Fw[t] + debt_margin * Y * grow(t) for t in 1:T]
    export_ceiling_v = ew .* rand(rng, Uniform(1.1, 1.6), n_tr)
    terminal_v = Kw[:, T] .* term_factor
    cfloor = [floor_factor * Cw[t] for t in 1:T]
    consumption_weight = [Δ * β^(Δ * (t - 1)) for t in 1:T]

    # Plan target on cumulative discounted consumption, Σ_t w_t C_t ≥ W. Feasible
    # plans aim below the reference path. Infeasible plans aim 10–30% above
    # Σ_t w_t b_t, where b_t is a provable per-period upper bound on consumption
    # from the Leontief inverse (certified). Unknown plans aim between the
    # reference path and that bound, where the true frontier lies.
    certificate_v = nothing
    if feasibility_status == feasible
        target = sum(consumption_weight .* Cw) * rand(rng, Uniform(0.85, 0.97))
    else
        margin = rand(rng, Uniform(1.1, 1.3))
        # Labor content of the import-augmented Leontief inverse, per skill.
        pis = [_leontief_solve(A_v, q_ceiling, labor_v[k, :]; transposed=true) for k in 1:n_skills]
        bounds = zeros(T)
        best_k = zeros(Int, T)
        for t in 1:T
            bounds[t] = Inf
            for k in 1:n_skills
                b = (L_v[k, t] - dot(pis[k], Gmat[:, t])) / dot(pis[k], d_v)
                if b < bounds[t]
                    bounds[t], best_k[t] = b, k
                end
            end
        end
        # Optionally a capacity bottleneck in period 1, where capacity is data.
        cap_sector = 0
        cap_pi = Float64[]
        if rand(rng) < 0.5
            cand_pool = [i for i in 1:n if d_v[i] > 0]
            for sct in shuffle(rng, cand_pool)[1:min(8, length(cand_pool))]
                unit = zeros(n)
                unit[sct] = 1.0
                pis_s = _leontief_solve(A_v, q_ceiling, unit; transposed=true)
                pd = dot(pis_s, d_v)
                pd > 0 || continue
                b = (K_start1[sct] - dot(pis_s, Gmat[:, 1])) / pd
                if b < bounds[1]
                    bounds[1], cap_sector, cap_pi = b, sct, pis_s
                end
            end
        end
        bound_total = sum(consumption_weight .* bounds)
        if feasibility_status == infeasible
            target = margin * bound_total
        else
            # The frontier sits 15–75% of the way from the reference path to the
            # bound (it is below the bound because the bound ignores investment).
            ref_total = sum(consumption_weight .* Cw)
            target = ref_total + rand(rng, Uniform(0.1, 0.8)) * (bound_total - ref_total)
        end
        if feasibility_status == infeasible
            bal = zeros(n, T)
            lab = zeros(n_skills, T)
            capm = zeros(n)
            for t in 1:T
                π_t = (t == 1 && cap_sector > 0) ? cap_pi : pis[best_k[t]]
                scale = consumption_weight[t] / dot(π_t, d_v)
                bal[:, t] .= scale .* π_t
                if t == 1 && cap_sector > 0
                    capm[cap_sector] = scale
                else
                    lab[best_k[t], t] = scale
                end
            end
            mode = cap_sector > 0 ? :labor_capital : :labor
            certificate_v = (mode, bal, lab, capm, bounds, target)
        end
    end

    # ---- Convert to hybrid (physical / value) units ----------------------------
    price = ones(n)
    for j in 1:n
        physical = (blocks[j] == 1 && rand(rng) < 0.75) || (blocks[j] == 2 && !(j in equipment) && rand(rng) < 0.15)
        physical && (price[j] = exp(rand(rng, Uniform(log(0.005), log(0.8)))))
    end
    # Coefficients a[i,j] (units of i per unit of j) scale by price_j / price_i.
    function to_hybrid(Mv::SparseMatrixCSC{Float64, Int})
        Ii, Jj, Vv = findnz(Mv)
        return sparse(Ii, Jj, Vv .* price[Jj] ./ price[Ii], n, n)
    end
    A_h = to_hybrid(A_v)
    B_h = to_hybrid(B_v)
    ptr = price[tradable]
    terminal_value = rand(rng, Uniform(0.3, 0.6)) .* vec(sum(B_v; dims=1)) .* price
    terminal_weight = β^(Δ * T)

    witness = nothing
    if feasibility_status == feasible
        witness = LeontiefWitness(
            xw ./ price,
            copy(Cw),
            Nw ./ price,
            Kw ./ price,
            mw ./ ptr,
            ew ./ ptr,
            copy(Fw),
        )
    end
    certificate = nothing
    if certificate_v !== nothing
        mode, bal_v, lab, capm_v, bounds, tgt = certificate_v
        bal_h = bal_v .* price
        certificate = LeontiefCertificate(
            mode, bal_h, bal_h[tradable, :], lab, capm_v .* price, bounds, tgt
        )
    end

    return DynamicLeontiefProblem(
        n,
        T,
        Δ,
        sector_block,
        tradable,
        price,
        A_h,
        B_h,
        gestation,
        survival,
        d_v ./ price,
        Gmat ./ price,
        labor_v .* price',
        L_v,
        rho,
        pm_v .* ptr,
        pe_v .* ptr,
        export_ceiling_v ./ ptr,
        K_start1 ./ price,
        terminal_v ./ price,
        F0,
        Rp,
        debt_ceiling,
        cfloor,
        target,
        consumption_weight,
        terminal_value .* terminal_weight,
        terminal_weight,
        witness,
        certificate,
        feasibility_status,
    )
end

"""
    build_model(prob::DynamicLeontiefProblem)

Build the dynamic Leontief planning LP. Deterministic — uses only struct data.
Registered containers: variables `x`, `C`, `N`, `K`, `imp`, `ex`, `F`;
constraints `balance`, `capacity`, `accumulation`, `import_limit`, `labor`,
`debt`, `no_decline`, `consumption_target`.
"""
function build_model(prob::DynamicLeontiefProblem)
    model = Model()
    n, T, Δ = prob.n_sectors, prob.n_periods, prob.period_length
    tr = prob.tradable
    ntr = length(tr)
    nk = size(prob.labor_coef, 1)

    @variable(model, x[1:n, 1:T] >= 0)
    @variable(model, C[t=1:T] >= prob.consumption_floor[t])
    @variable(model, N[1:n, 1:T] >= 0)
    @variable(model, K[s=1:n, t=1:T] >= (t == T ? prob.terminal_capacity[s] : 0.0))
    @variable(model, imp[1:ntr, 1:T] >= 0)
    @variable(model, 0 <= ex[i=1:ntr, t=1:T] <= prob.export_ceiling[i, t])
    @variable(model, F[t=1:T] <= prob.debt_ceiling[t])

    # Commodity balances, assembled column-wise from the sparse A and B.
    bal = [AffExpr(0.0) for _ in 1:n, _ in 1:T]
    A, B = prob.A, prob.B
    Arows, Avals = rowvals(A), nonzeros(A)
    Brows, Bvals = rowvals(B), nonzeros(B)
    phi = prob.gestation_share
    for t in 1:T
        for j in 1:n
            add_to_expression!(bal[j, t], 1.0, x[j, t])
            for p in nzrange(A, j)
                add_to_expression!(bal[Arows[p], t], -Avals[p], x[j, t])
            end
            for p in nzrange(B, j)
                i, b = Brows[p], Bvals[p]
                add_to_expression!(bal[i, t], -(1 - phi[j]) * b, N[j, t])
                t < T && add_to_expression!(bal[i, t], -phi[j] * b, N[j, t + 1])
            end
            dj = prob.consumption_bundle[j]
            dj != 0 && add_to_expression!(bal[j, t], -dj, C[t])
        end
        for (i, s) in enumerate(tr)
            add_to_expression!(bal[s, t], 1.0, imp[i, t])
            add_to_expression!(bal[s, t], -1.0, ex[i, t])
        end
    end
    @constraint(model, balance[s=1:n, t=1:T], bal[s, t] == prob.government_demand[s, t])

    @constraint(
        model,
        capacity[s=1:n, t=1:T],
        x[s, t] <= (t == 1 ? prob.initial_capacity[s] : 1.0 * K[s, t - 1])
    )
    @constraint(
        model,
        accumulation[s=1:n, t=1:T],
        K[s, t] - Δ * N[s, t] ==
        prob.survival[s] * (t == 1 ? prob.initial_capacity[s] : 1.0 * K[s, t - 1])
    )
    @constraint(
        model, import_limit[i=1:ntr, t=1:T], imp[i, t] - prob.import_ceiling[i] * x[tr[i], t] <= 0
    )
    labor_cols = [findall(!iszero, prob.labor_coef[k, :]) for k in 1:nk]
    @constraint(
        model,
        labor[k=1:nk, t=1:T],
        sum(prob.labor_coef[k, s] * x[s, t] for s in labor_cols[k]) <= prob.labor_supply[k, t]
    )
    @constraint(
        model,
        debt[t=1:T],
        F[t] - Δ * sum(prob.import_price[i] * imp[i, t] - prob.export_price[i] * ex[i, t] for i in 1:ntr) ==
        prob.interest_factor * (t == 1 ? prob.initial_debt : 1.0 * F[t - 1])
    )
    @constraint(model, no_decline[t=1:(T - 1)], C[t + 1] - C[t] >= 0)
    @constraint(
        model,
        consumption_target,
        sum(prob.consumption_weight[t] * C[t] for t in 1:T) >= prob.consumption_target
    )

    @objective(
        model,
        Max,
        sum(prob.consumption_weight[t] * C[t] for t in 1:T) +
        sum(prob.terminal_capital_value[s] * K[s, T] for s in 1:n) - prob.terminal_debt_weight * F[T]
    )
    return model
end

register_variant(
    :economic_planning,
    :dynamic_leontief,
    DynamicLeontiefProblem,
    "Dynamic multi-sector Leontief planning LP (PILOT/Dantzig staircase): sparse hybrid-unit input-output and capital matrices, capacity accumulation with gestation lags, labor, import ceilings and external debt, maximizing discounted consumption; planted balanced-growth witness and Leontief-inverse labor/capacity certificates",
    default=true,
)
