using JuMP
using Random

"""
    TSPPrizeCollectingProblem <: ProblemGenerator

Prize-collecting / quota TSP for optional sales or service visits. Each stop has
a prize for visiting it and a penalty for omitting it. One depot-rooted tour
must collect at least a specified quota. Single-commodity flow connects every
selected stop to the depot and gives the relaxed model meaningful network
structure.
"""
struct TSPPrizeCollectingProblem <: ProblemGenerator
    n_stops::Int
    locations::Vector{Tuple{Float64, Float64}}
    dist::Matrix{Float64}
    prizes::Vector{Float64}
    penalties::Vector{Float64}
    prize_quota::Float64
end

function TSPPrizeCollectingProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)

    # x and f on every directed arc, plus one y per non-depot stop:
    # 2n(n-1) + (n-1) = 2n^2 - n - 1.
    n = max(5, round(Int, (1 + sqrt(8 * target_variables + 9)) / 4))
    locations = _tsp_stops(rng, n)
    dist = _tsp_distance(rng, locations)

    prizes = zeros(n)
    penalties = zeros(n)
    for j in 2:n
        prizes[j] = round(clamp(exp(log(35.0) + 0.6 * randn(rng)), 10.0, 150.0); digits=2)
        penalties[j] = round(prizes[j] * (0.35 + 0.55 * rand(rng)); digits=2)
    end
    total_prize = sum(prizes)
    prize_quota = if feasibility_status == feasible
        round(total_prize * (0.45 + 0.25 * rand(rng)); digits=2)
    elseif feasibility_status == infeasible
        # y <= 1 implies collected prize <= total_prize, even after relaxation.
        round(total_prize * (1.10 + 0.20 * rand(rng)); digits=2)
    else
        # A naturally demanding but attainable commercial target.
        round(total_prize * (0.60 + 0.35 * rand(rng)); digits=2)
    end

    return TSPPrizeCollectingProblem(n, locations, dist, prizes, penalties, prize_quota)
end

function build_model(prob::TSPPrizeCollectingProblem)
    model = Model()
    n = prob.n_stops
    nodes = 1:n
    stops = 2:n

    @variable(model, x[i in nodes, j in nodes; i != j], Bin)
    @variable(model, y[j in stops], Bin)
    @variable(model, f[i in nodes, j in nodes; i != j] >= 0)

    @objective(
        model,
        Min,
        sum(prob.dist[i, j] * x[i, j] for i in nodes, j in nodes if i != j) +
            sum(prob.penalties[j] * (1 - y[j]) for j in stops)
    )

    @constraint(model, sum(x[1, j] for j in stops) == 1)
    @constraint(model, sum(x[j, 1] for j in stops) == 1)
    for j in stops
        @constraint(model, sum(x[i, j] for i in nodes if i != j) == y[j])
        @constraint(model, sum(x[j, k] for k in nodes if k != j) == y[j])
        @constraint(
            model,
            sum(f[i, j] for i in nodes if i != j) - sum(f[j, k] for k in nodes if k != j) == y[j]
        )
    end
    @constraint(model, sum(prob.prizes[j] * y[j] for j in stops) >= prob.prize_quota)
    @constraint(
        model, sum(f[1, j] for j in stops) - sum(f[j, 1] for j in stops) == sum(y[j] for j in stops)
    )
    for i in nodes, j in nodes
        i == j && continue
        @constraint(model, f[i, j] <= (n - 1) * x[i, j])
    end

    return model
end

register_variant(
    :tsp,
    :prize_collecting,
    TSPPrizeCollectingProblem,
    "Prize-collecting quota TSP with optional visits, omission penalties, and depot-anchored single-commodity flow",
)
