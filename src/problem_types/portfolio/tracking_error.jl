using JuMP
using Random
using Distributions
using Statistics

"""
    TrackingErrorPortfolioProblem <: ProblemGenerator

Generator for index-tracking / enhanced-indexing portfolio optimization problems.

Models an institutional enhanced-indexing mandate: the manager seeks to *beat* a
benchmark index in expected return while staying close to it in a risk sense. The
risk measure is the benchmark-relative tracking error, quantified via the mean
absolute deviation (MAD) of the active return across scenarios. This linearizes
exactly into a pure LP, so the model is a continuous LP (no integer variables).

# Overview
The decisions are long-only portfolio weights `x[i]` plus per-scenario absolute
active-return auxiliaries `u[s]`. The objective maximizes expected return. The
portfolio must be fully invested (`sum x_i = 1`), respect per-asset position
limits, keep each sector's *deviation* from the benchmark within a two-sided band,
and keep the average absolute active return (the tracking error) below a budget.

Structurally this differs from the CVaR sibling: risk is measured relative to a
benchmark via MAD tracking error (not absolute tail CVaR), and sector limits are
two-sided *deviation* bands around benchmark weights (not absolute upper caps).

# Fields
- `n_assets::Int`: Number of investable assets
- `n_scenarios::Int`: Number of return scenarios for the MAD linearization
- `n_sectors::Int`: Number of industry sectors
- `n_factors::Int`: Number of common risk factors driving scenario returns
- `expected_returns::Vector{Float64}`: Expected (mean + alpha) return per asset
- `scenario_returns::Matrix{Float64}`: Return matrix (n_scenarios × n_assets)
- `benchmark::Vector{Float64}`: Benchmark index weights (sum to 1)
- `te_budget::Float64`: Maximum allowable average absolute active return (tracking error)
- `sector_assignments::Vector{Int}`: Sector index per asset
- `sector_band::Vector{Float64}`: Two-sided deviation band per sector
- `max_position::Vector{Float64}`: Maximum weight per asset
"""
struct TrackingErrorPortfolioProblem <: ProblemGenerator
    n_assets::Int
    n_scenarios::Int
    n_sectors::Int
    n_factors::Int
    expected_returns::Vector{Float64}
    scenario_returns::Matrix{Float64}
    benchmark::Vector{Float64}
    te_budget::Float64
    sector_assignments::Vector{Int}
    sector_band::Vector{Float64}
    max_position::Vector{Float64}
end

"""
    TrackingErrorPortfolioProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct an index-tracking portfolio optimization problem instance.

Variables: `x[i]` (weights, n_assets) and `u[s]` (absolute active return per
scenario, n_scenarios).

    Total variables = n_assets + n_scenarios

Dimensions are sized as `n_assets = max(5, round(target / 5))` and
`n_scenarios = max(n_assets, target - n_assets)`, so the total equals roughly
`target_variables`.

# Arguments
- `target_variables`: Target number of decision variables
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility

# Feasibility
- `feasible`/`unknown` (biased feasible): the benchmark portfolio `x = b` is
  admissible by construction. It is fully invested (`sum b_i = 1`), it produces
  zero active return in every scenario (so `TE = 0 <= te_budget`), it has zero
  sector deviation (within any `sector_band >= 0`), and position limits are set
  to `max_position_i >= b_i`. Hence the LP has a finite optimum.
- `infeasible`: the per-asset position limits are scaled so that
  `sum_i max_position_i < 1`. Then `sum_i x_i <= sum_i max_position_i < 1`
  contradicts the full-investment constraint `sum_i x_i = 1`. This is a pure-LP
  aggregate contradiction that survives the relaxation (the model is already an
  LP).
"""
function TrackingErrorPortfolioProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # --- Dimension sizing: total = n_assets + n_scenarios ---
    n_assets = max(5, round(Int, target_variables / 5))
    n_scenarios = max(n_assets, target_variables - n_assets)

    # --- Scale-tiered group sizes ---
    if target_variables <= 100
        n_sectors = round(Int, rand(Uniform(3, 6)))
        n_factors = round(Int, rand(Uniform(2, 4)))
    elseif target_variables <= 500
        n_sectors = round(Int, rand(Uniform(5, 9)))
        n_factors = round(Int, rand(Uniform(3, 6)))
    else
        n_sectors = round(Int, rand(Uniform(8, 11)))
        n_factors = round(Int, rand(Uniform(5, 8)))
    end

    n_sectors = max(1, min(n_sectors, n_assets))
    n_factors = max(1, min(n_factors, n_assets))

    # --- Balanced sector assignment (every sector gets at least one asset) ---
    function balanced_assign(n_items, n_groups)
        assignment = zeros(Int, n_items)
        perm = randperm(n_items)
        for g in 1:n_groups
            assignment[perm[g]] = g
        end
        for i in (n_groups + 1):n_items
            assignment[perm[i]] = rand(1:n_groups)
        end
        return assignment
    end

    sector_assignments = balanced_assign(n_assets, n_sectors)

    # --- Factor loadings (sector-correlated) ---
    factor_loadings = rand(Normal(0.0, 0.3), n_assets, n_factors)
    for i in 1:n_assets
        primary_factor = (sector_assignments[i] - 1) % n_factors + 1
        factor_loadings[i, primary_factor] += rand(Uniform(0.3, 0.7))
    end

    # --- Scenario returns via factor model (common factors + idiosyncratic) ---
    factor_returns = rand(Normal(0.0, 0.05), n_scenarios, n_factors)
    idiosyncratic = rand(Normal(0.0, 0.02), n_scenarios, n_assets)
    scenario_returns = factor_returns * factor_loadings' + idiosyncratic

    # --- Expected returns: mean scenario return plus small alpha noise ---
    expected_returns = vec(mean(scenario_returns, dims=1))
    expected_returns .+= rand(Normal(0.0, 0.01), n_assets)

    # --- Benchmark weights (log-normal market-cap style, normalized) ---
    raw_weights = rand(LogNormal(0.0, 0.8), n_assets)
    benchmark = raw_weights ./ sum(raw_weights)

    # --- Position limits: at least the benchmark weight, with headroom ---
    # max_position_i = max(b_i, draw) * slack  (>= b_i guarantees benchmark admissible).
    # Express the per-asset cap as a multiple of equal-weight (1/n_assets), clamped
    # to sensible absolute bounds. The range is built so the lower bound is ALWAYS
    # strictly below the upper bound across every supported n_assets — a naive
    # min/max clamp inverts for very large (n_assets > 300) or very small
    # (n_assets == 5) portfolios and would make Uniform(lo, hi) throw.
    ew = 1.0 / n_assets
    pos_lo = clamp(2.0 * ew, 0.005, 0.30)
    pos_hi = max(clamp(6.0 * ew, 0.02, 0.40), pos_lo * 1.5)
    base_position = [rand(Uniform(pos_lo, pos_hi)) for _ in 1:n_assets]
    max_position = [max(benchmark[i], base_position[i]) * rand(Uniform(1.1, 1.6)) for i in 1:n_assets]

    # --- Sector deviation bands (two-sided, around benchmark) ---
    sector_band = [rand(Uniform(0.03, 0.12)) for _ in 1:n_sectors]

    # --- Tracking-error budget: a fraction of the benchmark's own scenario MAD ---
    # Reference MAD: average absolute deviation of the benchmark scenario return
    # from its own mean (a natural scale for an active-return budget).
    bench_scen = scenario_returns * benchmark
    bench_mad = mean(abs.(bench_scen .- mean(bench_scen)))
    te_budget = bench_mad * rand(Uniform(0.15, 0.5))

    # --- Feasibility handling ---
    actual_status = feasibility_status
    if feasibility_status == unknown
        actual_status = rand() < 0.75 ? feasible : infeasible
    end

    if actual_status == feasible
        # The benchmark x = b is already admissible:
        #   sum b_i = 1, A_s = 0 for all s => TE = 0 <= te_budget,
        #   sector deviation = 0 within any band >= 0, and max_position_i >= b_i.
        # Add a touch of slack to keep the model numerically comfortable.
        for i in 1:n_assets
            max_position[i] = max(max_position[i], benchmark[i] * 1.05)
        end
        te_budget = max(te_budget, 1e-6)
        sector_band .= max.(sector_band, 0.02)

    elseif actual_status == infeasible
        # Make full investment impossible: scale position limits so their sum < 1.
        target_sum = rand(Uniform(0.7, 0.9))
        max_position .*= (target_sum / sum(max_position))
        # (Aggregate contradiction: sum x_i <= sum max_position_i < 1 = required.)
    end

    return TrackingErrorPortfolioProblem(
        n_assets, n_scenarios, n_sectors, n_factors,
        expected_returns, scenario_returns, benchmark, te_budget,
        sector_assignments, sector_band, max_position,
    )
end

"""
    build_model(prob::TrackingErrorPortfolioProblem)

Build a JuMP model for the index-tracking portfolio problem. Deterministic — uses
only data from the struct fields.

# Returns
- `model`: The JuMP model
"""
function build_model(prob::TrackingErrorPortfolioProblem)
    model = Model()

    n = prob.n_assets
    S = prob.n_scenarios

    # Variables
    @variable(model, x[1:n] >= 0)        # portfolio weights (long-only)
    @variable(model, u[1:S] >= 0)        # absolute active return per scenario

    # Objective: maximize expected return
    @objective(model, Max, sum(prob.expected_returns[i] * x[i] for i in 1:n))

    # Full investment (fully invested, long-only)
    @constraint(model, sum(x[i] for i in 1:n) == 1.0)

    # MAD tracking-error linearization:
    #   A_s = sum_i r[s,i] * (x_i - b_i);  u[s] >= A_s and u[s] >= -A_s
    for s in 1:S
        active_return = sum(prob.scenario_returns[s, i] * (x[i] - prob.benchmark[i]) for i in 1:n)
        @constraint(model, u[s] >= active_return)
        @constraint(model, u[s] >= -active_return)
    end

    # Tracking-error budget: average absolute active return <= te_budget
    @constraint(model, (1.0 / S) * sum(u[s] for s in 1:S) <= prob.te_budget)

    # Position limits
    for i in 1:n
        @constraint(model, x[i] <= prob.max_position[i])
    end

    # Sector deviation bands (two-sided): within +/- band of benchmark sector weight
    for sct in 1:prob.n_sectors
        assets_in_sector = [i for i in 1:n if prob.sector_assignments[i] == sct]
        if !isempty(assets_in_sector)
            dev = sum(x[i] - prob.benchmark[i] for i in assets_in_sector)
            @constraint(model, dev <= prob.sector_band[sct])
            @constraint(model, dev >= -prob.sector_band[sct])
        end
    end

    return model
end

# Register the variant (cvar remains the default; do NOT pass default=true)
register_variant(
    :portfolio,
    :tracking_error,
    TrackingErrorPortfolioProblem,
    "Index-tracking portfolio: maximize return under a MAD tracking-error budget with sector deviation bands and position limits (pure LP)",
)
