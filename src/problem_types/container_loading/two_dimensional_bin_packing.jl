using JuMP
using Random
using Distributions

"""
    TwoStagePlanWitness

Trivial integer plan for a `feasible` instance: item `i` is cut with strip
pattern `strip_pattern[i]` (its maximal single-item strip in its own height
class on sheet type `sheet_type[i]`) `strip_runs[i]` times; the strips of each
(class, sheet) pair are cut from the maximal single-class sheet pattern
`sheet_pattern[c, s]` run `sheet_runs[c, s]` times. Demands, strip balances and
sheet availabilities hold in exact integer arithmetic.
"""
struct TwoStagePlanWitness
    strip_pattern::Vector{Int}
    strip_runs::Vector{Int}
    sheet_type::Vector{Int}
    sheet_pattern::Matrix{Int}
    sheet_runs::Matrix{Int}
end

"""
    AreaShortageCertificate

Relaxation-valid infeasibility proof with multipliers `w_i h_i` on the demand
rows, `W_s H_c` on the strip-balance rows and `W_s H_s` on the sheet rows:
a strip of class `c` holds at most `W_s H_c` of item area and a sheet at most
`H_s` of strip height, so the available sheet area `supply_area` bounds the
cut item area, yet `demand_area >= 1.04 * supply_area`.
"""
struct AreaShortageCertificate
    demand_area::Float64
    supply_area::Float64
end

"""
    TwoDimensionalBinPackingProblem <: ProblemGenerator

Two-dimensional bin packing / cutting with two-stage guillotine patterns
(Gilmore & Gomory 1965): rectangular items are cut from sheets (bins) first
into horizontal strips, then across each strip.

# Overview

Item types have integer-mm widths and heights from a catalogue of standard
heights (panel, glass and board cutting), no rotation (grain), and demands.
Height classes are the distinct item heights; a strip of class `c` on sheet
type `s` is `W_s` wide and `H_c` high and holds items with `h_i <= H_c`.

```text
min  sum_q cost_s(q) z_q
s.t. sum_p a_ip y_p >= d_i                                for every item type
     sum_{p in (c,s)} y_p - sum_{q of s} b_qc z_q <= 0     for every (class, sheet type)
     sum_{q of s} z_q <= A_s                               for every sheet type
     y, z >= 0 (general integers; relaxed by default)
```

`y_p` runs strip pattern `p` (items across the strip), `z_q` cuts sheet pattern
`q` (strips stacked in height). The strip-balance rows couple the two pattern
levels, so the LP is a genuine two-level pattern LP — not the pairwise big-M
disjunction model this variant used to be, whose relaxation collapsed to
`max(1, area ratio)` with every non-overlap row slack.

Sizing: exactly `target_variables` columns, 85% strip patterns and 15% sheet
patterns; `n_types = clamp(round(n / 20), 2, ...)`, 1-3 sheet types, height
classes `clamp(round(1.5 sqrt(n_types)), 2, 80)` — rows
`n_types + n_classes * n_sheets + n_sheets`, about 5% of the columns.
Patterns come from a hash-deduplicated greedy enumerator (near-linear).

# Feasibility

  - `feasible`: single-item strips and single-class sheets
    (`TwoStagePlanWitness`); availabilities `U(1.05, 1.35)` x plan usage.
  - `infeasible`: availabilities scaled so the item area exceeds the sheet
    area by `U(4%, 12%)` (`AreaShortageCertificate`).
  - `unknown`: sheet area `U(0.98, 1.18)` x item area — around the two-stage
    trim-loss threshold; the LP decides.
"""
struct TwoDimensionalBinPackingProblem <: ProblemGenerator
    sheet_widths::Vector{Int}
    sheet_heights::Vector{Int}
    sheet_costs::Vector{Float64}
    item_widths::Vector{Int}
    item_heights::Vector{Int}
    item_class::Vector{Int}
    class_heights::Vector{Int}
    demands::Vector{Int}
    strip_class::Vector{Int}
    strip_sheet::Vector{Int}
    strip_items::Vector{Vector{Int}}
    strip_counts::Vector{Vector{Int}}
    sheet_type::Vector{Int}
    sheet_classes::Vector{Vector{Int}}
    sheet_counts::Vector{Vector{Int}}
    availability::Vector{Int}
    feasibility_status::FeasibilityStatus
    feasible_witness::Union{Nothing, TwoStagePlanWitness}
    infeasibility_certificate::Union{Nothing, AreaShortageCertificate}
end

const TWO_D_SHEETS = ((3210, 2250, 1.00), (2800, 2070, 0.80), (2440, 1220, 0.42))

function two_d_dimensions(n::Int)
    n = max(n, 4)
    n_sheets = n < 200 ? 1 : (n < 5000 ? 2 : 3)
    n_types = clamp(max(round(Int, n / 20), min(10, n ÷ 2)), 2, n)
    n_classes = clamp(round(Int, 1.5 * sqrt(n_types)), 2, 80)
    # Every (class, sheet) needs its single-class sheet pattern and every
    # (item, sheet) its single-item strip.
    n_sheet_patterns = max(round(Int, 0.15 * n), n_classes * n_sheets)
    n_strip_patterns = n - n_sheet_patterns
    n_types = clamp(n_types, 1, max(1, n_strip_patterns ÷ n_sheets))
    return n_strip_patterns, n_sheet_patterns, n_sheets, n_types, n_classes
end

# Fill `cap` greedily from candidate `(index, size)` pairs: random counts, then
# a largest-first top-up. Returns sorted indices and counts.
function _two_d_fill(rng::AbstractRNG, cap::Int, cand::Vector{Int}, size_of)
    cnt = Dict{Int, Int}()
    rem = cap
    for i in shuffle(rng, cand)
        cmax = rem ÷ size_of(i)
        cmax == 0 && continue
        c = rand(rng, 1:cmax)
        cnt[i] = c
        rem -= c * size_of(i)
    end
    for i in sort(cand; by=i -> -size_of(i))
        c = rem ÷ size_of(i)
        c > 0 || continue
        cnt[i] = get(cnt, i, 0) + c
        rem -= c * size_of(i)
    end
    its = sort!(collect(keys(cnt)))
    return its, [cnt[i] for i in its]
end

function TwoDimensionalBinPackingProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    n = max(target_variables, 4)
    n_strip, n_sheetp, S, m_target, K_target = two_d_dimensions(n)

    sheet_widths = [TWO_D_SHEETS[s][1] for s in 1:S]
    sheet_heights = [TWO_D_SHEETS[s][2] for s in 1:S]
    unit = 20.0 + 10.0 * rand(rng)
    sheet_costs = [unit * TWO_D_SHEETS[s][3] * (0.95 + 0.1 * rand(rng)) for s in 1:S]
    Wmin, Hmin = minimum(sheet_widths), minimum(sheet_heights)

    # Standard heights catalogue (mm), then item types on it.
    heights = sort!(unique!([10 * round(Int, (150 + (0.55 * Hmin - 150) * rand(rng, Beta(1.5, 2.5))) / 10) for _ in 1:K_target]))
    m = m_target
    item_heights = [heights[rand(rng, 1:length(heights))] for _ in 1:m]
    item_widths = [round(Int, 150 + (0.6 * Wmin - 150) * rand(rng, Beta(1.4, 2.6))) for _ in 1:m]
    class_heights = sort!(unique(item_heights))
    K = length(class_heights)
    item_class = [searchsortedfirst(class_heights, h) for h in item_heights]
    items_by_class = [findall(==(c), item_class) for c in 1:K]
    demands = [max(1, round(Int, exp(log(25.0) + 0.9 * randn(rng)))) for _ in 1:m]

    # --- Strip patterns ---
    strip_class = Int[]
    strip_sheet = Int[]
    strip_items = Vector{Vector{Int}}()
    strip_counts = Vector{Vector{Int}}()
    seen = Set{UInt64}()
    function add_strip!(c, s, its, cs)
        length(strip_class) >= n_strip && return false
        isempty(its) && return false
        key = hash((c, s, its, cs))
        key in seen && return false
        push!(seen, key)
        push!(strip_class, c)
        push!(strip_sheet, s)
        push!(strip_items, its)
        push!(strip_counts, cs)
        return true
    end
    single_strip = zeros(Int, m, S)
    for s in 1:S, i in 1:m
        if add_strip!(item_class[i], s, [i], [sheet_widths[s] ÷ item_widths[i]])
            single_strip[i, s] = length(strip_class)
        end
    end
    attempts = 0
    while length(strip_class) < n_strip && attempts < 40 * n_strip + 1000
        attempts += 1
        s = rand(rng, 1:S)
        c = rand(rng, 1:K)
        cand = Int[]
        for _ in 1:rand(rng, 1:4)
            cc = rand(rng) < 0.7 ? c : rand(rng, 1:c)
            pool = items_by_class[cc]
            push!(cand, pool[rand(rng, 1:length(pool))])
        end
        unique!(cand)
        its, cs = _two_d_fill(rng, sheet_widths[s], cand, i -> item_widths[i])
        add_strip!(c, s, its, cs)
    end
    # Fallback for tiny catalogues: sub-maximal single-item strips.
    for s in 1:S, i in 1:m, k in 1:((sheet_widths[s] ÷ item_widths[i]) - 1), c in item_class[i]:K
        length(strip_class) >= n_strip && break
        add_strip!(c, s, [i], [k])
    end
    length(strip_class) == n_strip || error("two_dimensional_bin_packing: strip pattern space exhausted")

    # --- Sheet patterns ---
    sheet_type = Int[]
    sheet_classes = Vector{Vector{Int}}()
    sheet_counts = Vector{Vector{Int}}()
    seen_q = Set{UInt64}()
    function add_sheet!(s, cl, cs)
        length(sheet_type) >= n_sheetp && return false
        isempty(cl) && return false
        key = hash((s, cl, cs))
        key in seen_q && return false
        push!(seen_q, key)
        push!(sheet_type, s)
        push!(sheet_classes, cl)
        push!(sheet_counts, cs)
        return true
    end
    single_sheet = zeros(Int, K, S)
    for s in 1:S, c in 1:K
        if add_sheet!(s, [c], [sheet_heights[s] ÷ class_heights[c]])
            single_sheet[c, s] = length(sheet_type)
        end
    end
    attempts = 0
    while length(sheet_type) < n_sheetp && attempts < 40 * n_sheetp + 1000
        attempts += 1
        s = rand(rng, 1:S)
        cand = unique!([rand(rng, 1:K) for _ in 1:rand(rng, 2:4)])
        cl, cs = _two_d_fill(rng, sheet_heights[s], cand, c -> class_heights[c])
        add_sheet!(s, cl, cs)
    end
    for s in 1:S, c in 1:K, k in 1:((sheet_heights[s] ÷ class_heights[c]) - 1)
        length(sheet_type) >= n_sheetp && break
        add_sheet!(s, [c], [k])
    end
    for s in 1:S, c1 in 1:K, c2 in (c1 + 1):K
        length(sheet_type) >= n_sheetp && break
        for k1 in 1:(sheet_heights[s] ÷ class_heights[c1])
            rest = sheet_heights[s] - k1 * class_heights[c1]
            for k2 in 1:(rest ÷ class_heights[c2])
                add_sheet!(s, [c1, c2], [k1, k2])
            end
        end
    end
    length(sheet_type) == n_sheetp || error("two_dimensional_bin_packing: sheet pattern space exhausted ($(length(sheet_type)) of $n_sheetp, K=$K, heights=$class_heights)")

    # --- Trivial plan ---
    plan_sheet = [rand(rng, 1:S) for _ in 1:m]
    strip_pattern = [single_strip[i, plan_sheet[i]] for i in 1:m]
    strip_runs = [cld(demands[i], strip_counts[strip_pattern[i]][1]) for i in 1:m]
    strips_needed = zeros(Int, K, S)
    for i in 1:m
        strips_needed[item_class[i], plan_sheet[i]] += strip_runs[i]
    end
    sheet_runs = zeros(Int, K, S)
    usage = zeros(Int, S)
    for c in 1:K, s in 1:S
        strips_needed[c, s] == 0 && continue
        sheet_runs[c, s] = cld(strips_needed[c, s], sheet_counts[single_sheet[c, s]][1])
        usage[s] += sheet_runs[c, s]
    end
    demand_area = sum(Float64(item_widths[i]) * item_heights[i] * demands[i] for i in 1:m)

    feasible_witness = nothing
    infeasibility_certificate = nothing
    if feasibility_status == feasible
        availability = [ceil(Int, usage[s] * (1.05 + 0.30 * rand(rng))) for s in 1:S]
        feasible_witness = TwoStagePlanWitness(strip_pattern, strip_runs, plan_sheet, single_sheet, sheet_runs)
    else
        base = [usage[s] * (1.05 + 0.30 * rand(rng)) + 1.0 for s in 1:S]
        area = sum(Float64(sheet_widths[s]) * sheet_heights[s] * base[s] for s in 1:S)
        ratio = feasibility_status == infeasible ? 1.0 / (1.04 + 0.08 * rand(rng)) : 0.98 + 0.20 * rand(rng)
        availability = [floor(Int, base[s] * ratio * demand_area / area) for s in 1:S]
        if feasibility_status == infeasible
            supply = sum(Float64(sheet_widths[s]) * sheet_heights[s] * availability[s] for s in 1:S)
            demand_area >= 1.04 * supply || error("two_dimensional_bin_packing: certificate margin lost")
            infeasibility_certificate = AreaShortageCertificate(demand_area, supply)
        end
    end

    return TwoDimensionalBinPackingProblem(
        sheet_widths,
        sheet_heights,
        sheet_costs,
        item_widths,
        item_heights,
        item_class,
        class_heights,
        demands,
        strip_class,
        strip_sheet,
        strip_items,
        strip_counts,
        sheet_type,
        sheet_classes,
        sheet_counts,
        availability,
        feasibility_status,
        feasible_witness,
        infeasibility_certificate,
    )
end

function build_model(prob::TwoDimensionalBinPackingProblem)
    model = Model()
    P = length(prob.strip_class)
    Q = length(prob.sheet_type)
    S = length(prob.sheet_widths)
    K = length(prob.class_heights)
    @variable(model, strip_runs[1:P] >= 0, Int)
    @variable(model, sheet_runs[1:Q] >= 0, Int)
    @objective(model, Min, sum(prob.sheet_costs[prob.sheet_type[q]] * sheet_runs[q] for q in 1:Q))

    produced = [AffExpr() for _ in eachindex(prob.demands)]
    balance = [AffExpr() for _ in 1:K, _ in 1:S]
    sheets = [AffExpr() for _ in 1:S]
    for p in 1:P
        for (i, c) in zip(prob.strip_items[p], prob.strip_counts[p])
            add_to_expression!(produced[i], c, strip_runs[p])
        end
        add_to_expression!(balance[prob.strip_class[p], prob.strip_sheet[p]], 1.0, strip_runs[p])
    end
    for q in 1:Q
        s = prob.sheet_type[q]
        for (c, b) in zip(prob.sheet_classes[q], prob.sheet_counts[q])
            add_to_expression!(balance[c, s], -b, sheet_runs[q])
        end
        add_to_expression!(sheets[s], 1.0, sheet_runs[q])
    end
    @constraint(model, demand[i in eachindex(produced)], produced[i] >= prob.demands[i])
    for c in 1:K, s in 1:S
        isempty(balance[c, s].terms) && continue
        @constraint(model, balance[c, s] <= 0)
    end
    for s in 1:S
        isempty(sheets[s].terms) && continue
        @constraint(model, sheets[s] <= prob.availability[s])
    end
    return model
end

register_variant(
    :container_loading,
    :two_dimensional_bin_packing,
    TwoDimensionalBinPackingProblem,
    "Two-dimensional bin packing with two-stage guillotine strip and sheet patterns (Gilmore-Gomory)",
)
