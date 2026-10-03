# Shared data generation for the cutting_stock family: integer-millimetre
# stock catalogues and order books, and a near-linear sparse pattern
# enumerator for Gilmore-Gomory master LPs.

using Random
using Distributions

"""
    CSPatterns

Sparse cutting patterns. Pattern `j` is cut from stock type `stock[j]` and
yields `counts[j][t]` pieces of item type `items[j][t]` (`items[j]` sorted,
counts positive). By construction `sum(counts[j] .* piece_lengths[items[j]])
<= stock_lengths[stock[j]]`.
"""
struct CSPatterns
    stock::Vector{Int}
    items::Vector{Vector{Int}}
    counts::Vector{Vector{Int}}
end

Base.length(p::CSPatterns) = length(p.stock)

"""Stock-bar catalogue in millimetres (6 m - 13.5 m bars and rolls)."""
const CS_STOCK_CATALOGUE = [6000, 7500, 9000, 10500, 12000, 13500]

"""
    cs_stock_types(rng, n_stock)

Draw `n_stock` distinct catalogue lengths (sorted) and a per-bar cost: steel
priced per millimetre with a small long-bar discount, `cost = L * p * (L /
6000)^-0.06`, plus +-3% supplier noise.
"""
function cs_stock_types(rng::AbstractRNG, n_stock::Int)
    lengths = sort(shuffle(rng, CS_STOCK_CATALOGUE)[1:n_stock])
    price = 0.002 * (0.8 + 0.4 * rand(rng))   # cost units per mm
    costs = [L * price * (L / 6000)^-0.06 * (0.97 + 0.06 * rand(rng)) for L in lengths]
    return lengths, costs
end

"""
    cs_piece_lengths(rng, n_types, max_length)

Integer-millimetre order lengths in `[120, max_length]`: 55% on a 50 mm
catalogue grid, the rest bespoke at 1 mm resolution, skewed short
(`Beta(1.6, 3.2)`), as in bar and rebar cutting order books.
"""
function cs_piece_lengths(rng::AbstractRNG, n_types::Int, max_length::Int)
    lo = 120
    hi = max(lo + 10, max_length)
    lengths = Vector{Int}(undef, n_types)
    for i in 1:n_types
        u = rand(rng, Beta(1.6, 3.2))
        len = lo + u * (hi - lo)
        lengths[i] = rand(rng) < 0.55 ? clamp(round(Int, len / 50) * 50, lo, hi) : round(Int, len)
    end
    return lengths
end

"""
    cs_demands(rng, n_types)

Order quantities: lognormal around 60 pieces with heavy right tail, rounded to
packs of 5 above 50 pieces, at least 1.
"""
function cs_demands(rng::AbstractRNG, n_types::Int)
    d = Vector{Int}(undef, n_types)
    for i in 1:n_types
        raw = exp(log(60.0) + 0.9 * randn(rng))
        d[i] = raw < 50 ? max(1, round(Int, raw)) : 5 * round(Int, raw / 5)
    end
    return d
end

"""
    cs_generate_patterns(rng, stock_lengths, piece_lengths, n_patterns; singles=true)

Generate exactly `n_patterns` distinct sparse patterns in near-linear time.

1. If `singles`, the maximal single-item pattern of every (item, stock type)
   pair that fits comes first, ordered by stock type then item (so pattern
   `(k - 1) * n_types + i` is item `i` on stock `k` when every item fits every
   stock; callers use `cs_single_index` instead of relying on that).
2. Random greedy fills imitate knapsack-generated columns: a stock type, 2-6
   random candidate items, a random count of each that fits, then a top-up
   with the longest fitting candidate so patterns are maximal for their
   candidates (low trim loss).
3. Deterministic fallback for tiny catalogues: sub-maximal single-item and
   two-item patterns.

Duplicates are rejected through a hash set of `(stock, items, counts)` keys.
"""
function cs_generate_patterns(
    rng::AbstractRNG,
    stock_lengths::Vector{Int},
    piece_lengths::Vector{Int},
    n_patterns::Int;
    singles::Bool=true,
)
    n_types = length(piece_lengths)
    n_stock = length(stock_lengths)
    stock = Int[]
    items = Vector{Vector{Int}}()
    counts = Vector{Vector{Int}}()
    seen = Set{UInt64}()
    sizehint!(stock, n_patterns)
    sizehint!(seen, n_patterns)

    function try_add!(k, its, cs)
        length(stock) >= n_patterns && return false
        isempty(its) && return false
        key = hash((k, its, cs))
        key in seen && return false
        push!(seen, key)
        push!(stock, k)
        push!(items, its)
        push!(counts, cs)
        return true
    end

    if singles
        for k in 1:n_stock, i in 1:n_types
            c = stock_lengths[k] ÷ piece_lengths[i]
            c >= 1 && try_add!(k, [i], [c])
        end
    end

    attempts = 0
    max_attempts = 30 * n_patterns + 1000
    cand = Int[]
    cnt = Dict{Int, Int}()
    while length(stock) < n_patterns && attempts < max_attempts
        attempts += 1
        k = rand(rng, 1:n_stock)
        L = stock_lengths[k]
        m = min(n_types, 2 + rand(rng, Binomial(4, 0.45)))
        empty!(cand)
        for _ in 1:m
            push!(cand, rand(rng, 1:n_types))
        end
        unique!(cand)
        filter!(i -> piece_lengths[i] <= L, cand)
        isempty(cand) && continue
        empty!(cnt)
        rem = L
        for i in shuffle!(rng, cand)
            cmax = rem ÷ piece_lengths[i]
            cmax == 0 && continue
            c = rand(rng, 1:cmax)
            cnt[i] = c
            rem -= c * piece_lengths[i]
        end
        # Top-up: longest fitting candidate first, until nothing fits.
        sort!(cand; by=i -> -piece_lengths[i])
        for i in cand
            c = rem ÷ piece_lengths[i]
            if c > 0
                cnt[i] = get(cnt, i, 0) + c
                rem -= c * piece_lengths[i]
            end
        end
        its = sort!(collect(keys(cnt)))
        try_add!(k, its, [cnt[i] for i in its])
    end

    if length(stock) < n_patterns
        for k in 1:n_stock, i in 1:n_types
            for c in 1:((stock_lengths[k] ÷ piece_lengths[i]) - 1)
                try_add!(k, [i], [c])
            end
        end
        for k in 1:n_stock, i in 1:n_types, j in (i + 1):n_types
            length(stock) >= n_patterns && break
            for ci in 1:(stock_lengths[k] ÷ piece_lengths[i])
                rest = stock_lengths[k] - ci * piece_lengths[i]
                for cj in 1:(rest ÷ piece_lengths[j])
                    try_add!(k, [i, j], [ci, cj])
                end
            end
        end
    end
    length(stock) == n_patterns ||
        error("cutting_stock: pattern space exhausted at $(length(stock)) of $n_patterns patterns")
    return CSPatterns(stock, items, counts)
end

"""
    cs_single_index(pats, n_types, n_stock)

Map `(item, stock type) -> pattern index` of the maximal single-item patterns
(0 where the item does not fit that stock).
"""
function cs_single_index(pats::CSPatterns, stock_lengths::Vector{Int}, piece_lengths::Vector{Int})
    idx = zeros(Int, length(piece_lengths), length(stock_lengths))
    for j in 1:length(pats)
        length(pats.items[j]) == 1 || continue
        i = pats.items[j][1]
        k = pats.stock[j]
        pats.counts[j][1] == stock_lengths[k] ÷ piece_lengths[i] || continue
        idx[i, k] == 0 && (idx[i, k] = j)
    end
    return idx
end

"""
    cs_pattern_length(pats, j, piece_lengths)

Total length cut by pattern `j`.
"""
cs_pattern_length(pats::CSPatterns, j::Int, piece_lengths) =
    sum(c * piece_lengths[i] for (i, c) in zip(pats.items[j], pats.counts[j]))

"""
    MaterialShortageCertificate

Relaxation-valid infeasibility proof shared by the pattern-based variants:
every pattern fits its stock, so with multipliers `piece_length[i]` on the
demand rows and `stock_length[k]` on the availability rows, the demanded
material `demand_length = sum_i piece_length[i] * demand[i]` must be cut from
at most `supply_length = sum_k stock_length[k] * available[k]` millimetres of
stock — but `demand_length >= 1.04 * supply_length`. The combination touches
every demand and availability row, so presolve cannot detect it.
"""
struct MaterialShortageCertificate
    demand_length::Float64
    supply_length::Float64
end
