using JuMP
using Random

"""
    TSPPrizeCollectingProblem <: ProblemGenerator

Prize-collecting / quota TSP for optional sales or service visits. Each stop has
a prize for visiting it and a penalty for omitting it. One depot-rooted tour
must collect at least a specified quota. Single-commodity flow connects every
selected stop to the depot and gives the relaxed model meaningful network
structure.

# Variable count

`x` and `f` on every allowed arc plus one `y` per stop: `2n(n-1) + (n-1)` with
complete support; a district-mode infeasible instance deletes `x` and `f` on
the `k(n-k)` arcs of its Hall block and sizes `n` against the delivered count
`2n^2 - n - 1 - 2k(n-k)`.

# Feasibility

  - `feasible`: quota at 45–70% of the total prize on complete support; the
    tour through every stop (`y ≡ 1`) collects everything.
  - `infeasible`: one of two LP-valid certificates (`infeasibility_mode`):
      - `:district` (≈75%, the default): a gated district (`_tsp_hall_block`):
        `S` (`k` stops) can be entered only from `k-1` gateway stops `T`, so
        `Σ_{j∈S} y_j = Σ_{j∈S} indeg(j) ≤ Σ_{i∈T} outdeg(i) = Σ_{i∈T} y_i ≤ k-1`.
        The LP then collects at most `total − min_{j∈S} prize_j`, and the quota
        is set to `total − θ·min_{j∈S} prize_j` with `θ ∈ [0.3, 0.7]` — at or
        below the total prize, so no single row is violated and refuting it
        needs the degree rows of `S ∪ T` together with the quota row.
      - `:over_total`: quota 1.1–1.3× the total prize, refuted by `y ≤ 1` alone.
  - `unknown`: a demanding commercial quota (60–95% of the total).
"""
struct TSPPrizeCollectingProblem <: ProblemGenerator
    n_stops::Int
    locations::Vector{Tuple{Float64, Float64}}
    dist::Matrix{Float64}
    prizes::Vector{Float64}
    penalties::Vector{Float64}
    prize_quota::Float64
    arc_ok::Matrix{Bool}
    blocked_set::Vector{Int}
    gate_set::Vector{Int}
    infeasibility_mode::Symbol
end

function TSPPrizeCollectingProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)

    # x and f on every directed arc, plus one y per non-depot stop:
    # 2n(n-1) + (n-1) = 2n^2 - n - 1.
    n0 = max(5, round(Int, (1 + sqrt(8 * target_variables + 9)) / 4))
    # Mode and district size are drawn unconditionally (RNG alignment).
    mode = rand(rng) < 0.75 ? :district : :over_total
    k = _tsp_hall_size(rng, n0)
    district = feasibility_status == infeasible && mode == :district
    n = if district
        _tsp_pick_n(n0, target_variables, k, m -> 2m^2 - m - 1 - 2k * (m - k))
    else
        n0
    end
    locations = _tsp_stops(rng, n)
    dist = _tsp_distance(rng, locations)

    prizes = zeros(n)
    penalties = zeros(n)
    for j in 2:n
        prizes[j] = round(clamp(exp(log(35.0) + 0.6 * randn(rng)), 10.0, 150.0); digits=2)
        penalties[j] = round(prizes[j] * (0.35 + 0.55 * rand(rng)); digits=2)
    end
    total_prize = sum(prizes)
    arc_ok, S, T = _tsp_full_support(n), Int[], Int[]
    prize_quota = if feasibility_status == feasible
        round(total_prize * (0.45 + 0.25 * rand(rng)); digits=2)
    elseif district
        arc_ok, S, T = _tsp_hall_block(rng, n, k, locations)
        # The LP collects at most total - min_{j in S} prize_j (see docstring);
        # rounding down keeps the quota at or below the total prize.
        min_blocked = minimum(prizes[j] for j in S)
        floor(total_prize - (0.3 + 0.4 * rand(rng)) * min_blocked; digits=2)
    elseif feasibility_status == infeasible
        # y <= 1 implies collected prize <= total_prize, even after relaxation.
        round(total_prize * (1.10 + 0.20 * rand(rng)); digits=2)
    else
        # A naturally demanding but attainable commercial target.
        round(total_prize * (0.60 + 0.35 * rand(rng)); digits=2)
    end
    resolved_mode = feasibility_status == infeasible ? mode : :none

    return TSPPrizeCollectingProblem(
        n, locations, dist, prizes, penalties, prize_quota, arc_ok, S, T, resolved_mode
    )
end

function build_model(prob::TSPPrizeCollectingProblem)
    model = Model()
    n = prob.n_stops
    nodes = 1:n
    stops = 2:n
    ok(i, j) = prob.arc_ok[i, j]

    @variable(model, x[i in nodes, j in nodes; ok(i, j)], Bin)
    @variable(model, y[j in stops], Bin)
    @variable(model, f[i in nodes, j in nodes; ok(i, j)] >= 0)

    @objective(
        model,
        Min,
        sum(prob.dist[i, j] * x[i, j] for i in nodes, j in nodes if ok(i, j)) +
            sum(prob.penalties[j] * (1 - y[j]) for j in stops)
    )

    @constraint(model, sum(x[1, j] for j in stops if ok(1, j)) == 1)
    @constraint(model, sum(x[j, 1] for j in stops if ok(j, 1)) == 1)
    for j in stops
        @constraint(model, sum(x[i, j] for i in nodes if ok(i, j)) == y[j])
        @constraint(model, sum(x[j, k] for k in nodes if ok(j, k)) == y[j])
        @constraint(
            model,
            sum(f[i, j] for i in nodes if ok(i, j)) - sum(f[j, k] for k in nodes if ok(j, k)) ==
                y[j]
        )
    end
    @constraint(model, sum(prob.prizes[j] * y[j] for j in stops) >= prob.prize_quota)
    @constraint(
        model,
        sum(f[1, j] for j in stops if ok(1, j)) - sum(f[j, 1] for j in stops if ok(j, 1)) ==
            sum(y[j] for j in stops)
    )
    for i in nodes, j in nodes
        ok(i, j) || continue
        @constraint(model, f[i, j] <= (n - 1) * x[i, j])
    end

    return model
end

register_variant(
    :tsp,
    :prize_collecting,
    TSPPrizeCollectingProblem,
    "Prize-collecting quota TSP with optional visits, omission penalties, and depot-anchored single-commodity flow";
    tags=[:routing, :network, :big_m],
)
