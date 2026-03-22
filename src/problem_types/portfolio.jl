using JuMP
using Random
using Distributions
using Statistics

"""
    PortfolioProblem <: ProblemGenerator

Generator for CVaR portfolio optimization problems with institutional-grade constraints.

Uses Conditional Value-at-Risk (CVaR) as the risk measure, which naturally linearizes into
a pure LP. Includes sector/region exposure limits, asset class allocation bounds, factor
exposure constraints, position size limits, and turnover constraints.

# Fields
- `n_assets::Int`: Number of investable assets
- `n_scenarios::Int`: Number of return scenarios for CVaR linearization
- `n_sectors::Int`: Number of industry sectors
- `n_regions::Int`: Number of geographic regions
- `n_asset_classes::Int`: Number of asset classes (e.g., equities, bonds, alternatives)
- `n_factors::Int`: Number of risk factors
- `expected_returns::Vector{Float64}`: Expected return per asset
- `scenario_returns::Matrix{Float64}`: Return matrix (n_scenarios × n_assets)
- `beta::Float64`: CVaR confidence level (e.g., 0.95)
- `cvar_limit::Float64`: Maximum allowable CVaR
- `sector_assignments::Vector{Int}`: Sector index per asset
- `region_assignments::Vector{Int}`: Region index per asset
- `asset_class_assignments::Vector{Int}`: Asset class index per asset
- `sector_upper::Vector{Float64}`: Max allocation per sector
- `region_upper::Vector{Float64}`: Max allocation per region
- `asset_class_lower::Vector{Float64}`: Min allocation per asset class
- `asset_class_upper::Vector{Float64}`: Max allocation per asset class
- `factor_loadings::Matrix{Float64}`: Factor loading matrix (n_assets × n_factors)
- `factor_lower::Vector{Float64}`: Lower bound on factor exposure
- `factor_upper::Vector{Float64}`: Upper bound on factor exposure
- `max_position::Vector{Float64}`: Max weight per asset
- `benchmark::Vector{Float64}`: Benchmark portfolio weights
- `turnover_limit::Float64`: Max total turnover from benchmark
"""
struct PortfolioProblem <: ProblemGenerator
    n_assets::Int
    n_scenarios::Int
    n_sectors::Int
    n_regions::Int
    n_asset_classes::Int
    n_factors::Int
    expected_returns::Vector{Float64}
    scenario_returns::Matrix{Float64}
    beta::Float64
    cvar_limit::Float64
    sector_assignments::Vector{Int}
    region_assignments::Vector{Int}
    asset_class_assignments::Vector{Int}
    sector_upper::Vector{Float64}
    region_upper::Vector{Float64}
    asset_class_lower::Vector{Float64}
    asset_class_upper::Vector{Float64}
    factor_loadings::Matrix{Float64}
    factor_lower::Vector{Float64}
    factor_upper::Vector{Float64}
    max_position::Vector{Float64}
    benchmark::Vector{Float64}
    turnover_limit::Float64
end

"""
    PortfolioProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a CVaR portfolio optimization problem instance.

Variables: x[i] (weights), z[s] (CVaR shortfall), alpha (VaR), d_plus/d_minus (turnover).
Total = 3 * n_assets + n_scenarios + 1.

# Arguments
- `target_variables`: Target number of variables
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function PortfolioProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # --- Dimension sizing ---
    # Total variables = 3*n_assets + n_scenarios + 1
    n_assets = max(3, floor(Int, target_variables / 5))
    n_scenarios = max(n_assets, target_variables - 3 * n_assets - 1)

    # --- Scale-dependent group sizes ---
    if target_variables <= 100
        n_sectors = round(Int, rand(Uniform(3, 6)))
        n_regions = round(Int, rand(Uniform(3, 4)))
        n_asset_classes = round(Int, rand(Uniform(3, 4)))
        n_factors = round(Int, rand(Uniform(3, 5)))
    elseif target_variables <= 500
        n_sectors = round(Int, rand(Uniform(5, 10)))
        n_regions = round(Int, rand(Uniform(3, 5)))
        n_asset_classes = round(Int, rand(Uniform(3, 5)))
        n_factors = round(Int, rand(Uniform(4, 7)))
    else
        n_sectors = round(Int, rand(Uniform(8, 12)))
        n_regions = round(Int, rand(Uniform(4, 6)))
        n_asset_classes = round(Int, rand(Uniform(4, 5)))
        n_factors = round(Int, rand(Uniform(5, 8)))
    end

    n_sectors = min(n_sectors, n_assets)
    n_regions = min(n_regions, n_assets)
    n_asset_classes = min(n_asset_classes, n_assets)
    n_factors = min(n_factors, n_assets)

    # --- Balanced group assignments ---
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
    region_assignments = balanced_assign(n_assets, n_regions)
    asset_class_assignments = balanced_assign(n_assets, n_asset_classes)

    # --- Factor loadings (sector-correlated) ---
    factor_loadings = rand(Normal(0.0, 0.3), n_assets, n_factors)
    for i in 1:n_assets
        primary_factor = (sector_assignments[i] - 1) % n_factors + 1
        factor_loadings[i, primary_factor] += rand(Uniform(0.3, 0.7))
    end

    # --- Scenario returns via factor model ---
    factor_returns = rand(Normal(0.0, 0.05), n_scenarios, n_factors)
    idiosyncratic = rand(Normal(0.0, 0.02), n_scenarios, n_assets)
    scenario_returns = factor_returns * factor_loadings' + idiosyncratic

    # --- Expected returns ---
    expected_returns = vec(mean(scenario_returns, dims=1))
    expected_returns .+= rand(Normal(0.0, 0.01), n_assets)

    # --- CVaR parameters ---
    beta = rand(Uniform(0.90, 0.99))

    equal_weight = fill(1.0 / n_assets, n_assets)
    eq_scenario_returns = scenario_returns * equal_weight
    sorted_eq = sort(eq_scenario_returns)
    tail_count = max(1, floor(Int, (1 - beta) * n_scenarios))
    ref_cvar = -mean(sorted_eq[1:tail_count])
    cvar_limit = ref_cvar * rand(Uniform(0.8, 1.5))

    # --- Sector upper limits ---
    sector_counts = [count(==(s), sector_assignments) for s in 1:n_sectors]
    sector_upper = [min(1.0, (sector_counts[s] / n_assets) * rand(Uniform(1.2, 2.5))) for s in 1:n_sectors]

    # --- Region upper limits ---
    region_counts = [count(==(r), region_assignments) for r in 1:n_regions]
    region_upper = [min(1.0, (region_counts[r] / n_assets) * rand(Uniform(1.2, 2.5))) for r in 1:n_regions]

    # --- Asset class bounds ---
    class_counts = [count(==(c), asset_class_assignments) for c in 1:n_asset_classes]
    asset_class_lower = [max(0.0, (class_counts[c] / n_assets) * rand(Uniform(0.1, 0.5))) for c in 1:n_asset_classes]
    asset_class_upper = [min(1.0, (class_counts[c] / n_assets) * rand(Uniform(1.2, 2.5))) for c in 1:n_asset_classes]
    for c in 1:n_asset_classes
        asset_class_upper[c] = max(asset_class_upper[c], asset_class_lower[c] + 0.05)
    end
    if sum(asset_class_lower) > 0.95
        asset_class_lower .*= (0.9 / sum(asset_class_lower))
    end

    # --- Factor exposure bounds ---
    ref_exposure = factor_loadings' * equal_weight
    factor_lower = ref_exposure .- rand(Uniform(0.1, 0.3), n_factors)
    factor_upper = ref_exposure .+ rand(Uniform(0.1, 0.3), n_factors)

    # --- Position limits ---
    max_position = [rand(Uniform(max(2.0 / n_assets, 0.02), min(0.3, 5.0 / n_assets))) for _ in 1:n_assets]
    if sum(max_position) < 1.05
        max_position .*= (1.1 / sum(max_position))
    end

    # --- Benchmark portfolio (log-normal market-cap style weights) ---
    raw_weights = rand(LogNormal(0.0, 0.8), n_assets)
    benchmark = raw_weights ./ sum(raw_weights)

    # --- Turnover limit ---
    turnover_limit = rand(Uniform(0.3, 1.5))

    # --- Feasibility handling ---
    actual_status = feasibility_status
    if feasibility_status == unknown
        actual_status = rand() < 0.7 ? feasible : infeasible
    end

    if actual_status == feasible
        # Construct a reference portfolio and widen constraints to accommodate it
        ref_portfolio = copy(benchmark)

        # Respect position limits
        for i in 1:n_assets
            ref_portfolio[i] = min(ref_portfolio[i], max_position[i] * 0.9)
        end
        ref_portfolio ./= sum(ref_portfolio)

        # Widen sector limits
        for s in 1:n_sectors
            sw = sum(ref_portfolio[i] for i in 1:n_assets if sector_assignments[i] == s)
            sector_upper[s] = max(sector_upper[s], sw * (1.1 + 0.2 * rand()))
            sector_upper[s] = min(sector_upper[s], 1.0)
        end

        # Widen region limits
        for r in 1:n_regions
            rw = sum(ref_portfolio[i] for i in 1:n_assets if region_assignments[i] == r)
            region_upper[r] = max(region_upper[r], rw * (1.1 + 0.2 * rand()))
            region_upper[r] = min(region_upper[r], 1.0)
        end

        # Adjust asset class bounds
        for c in 1:n_asset_classes
            cw = sum(ref_portfolio[i] for i in 1:n_assets if asset_class_assignments[i] == c)
            asset_class_lower[c] = min(asset_class_lower[c], cw * (0.8 - 0.1 * rand()))
            asset_class_upper[c] = max(asset_class_upper[c], cw * (1.2 + 0.2 * rand()))
            asset_class_upper[c] = min(asset_class_upper[c], 1.0)
        end
        if sum(asset_class_lower) > 0.95
            asset_class_lower .*= (0.9 / sum(asset_class_lower))
        end

        # Adjust factor bounds
        ref_fexp = factor_loadings' * ref_portfolio
        for f in 1:n_factors
            factor_lower[f] = min(factor_lower[f], ref_fexp[f] - rand(Uniform(0.05, 0.15)))
            factor_upper[f] = max(factor_upper[f], ref_fexp[f] + rand(Uniform(0.05, 0.15)))
        end

        # Ensure CVaR limit accommodates reference
        ref_scen = scenario_returns * ref_portfolio
        ref_sorted = sort(ref_scen)
        ref_tail = max(1, floor(Int, (1 - beta) * n_scenarios))
        ref_cvar_val = -mean(ref_sorted[1:ref_tail])
        cvar_limit = max(cvar_limit, ref_cvar_val * (1.05 + 0.15 * rand()))

        # Ensure turnover limit accommodates reference
        ref_turnover = sum(abs.(ref_portfolio .- benchmark))
        turnover_limit = max(turnover_limit, ref_turnover * (1.1 + 0.2 * rand()))

        # Ensure position limits sum is sufficient
        if sum(max_position) < 1.05
            max_position .*= (1.1 / sum(max_position))
        end

    elseif actual_status == infeasible
        mode = rand(1:4)

        if mode == 1
            # Impossibly tight CVaR limit
            cvar_limit *= rand(Uniform(0.01, 0.15))
        elseif mode == 2
            # Asset class lowers sum > 1 (impossible with budget = 1)
            total_lower = sum(asset_class_lower)
            scale = (1.05 + rand() * 0.2) / max(total_lower, 0.01)
            asset_class_lower .*= scale
        elseif mode == 3
            # Position limits sum < 1 (budget constraint infeasible)
            tight_factor = rand(Uniform(0.3, 0.7))
            max_position .*= (tight_factor / sum(max_position))
        else
            # Near-zero turnover with conflicting sector constraints
            turnover_limit = rand(Uniform(0.001, 0.05))
            for s in 1:n_sectors
                sw = sum(benchmark[i] for i in 1:n_assets if sector_assignments[i] == s)
                if sw > 0
                    sector_upper[s] = min(sector_upper[s], sw * rand(Uniform(0.3, 0.6)))
                end
            end
        end
    end

    return PortfolioProblem(
        n_assets, n_scenarios, n_sectors, n_regions, n_asset_classes, n_factors,
        expected_returns, scenario_returns, beta, cvar_limit,
        sector_assignments, region_assignments, asset_class_assignments,
        sector_upper, region_upper, asset_class_lower, asset_class_upper,
        factor_loadings, factor_lower, factor_upper,
        max_position, benchmark, turnover_limit
    )
end

"""
    build_model(prob::PortfolioProblem)

Build a JuMP model for the CVaR portfolio optimization problem. Deterministic — uses
only data from the struct fields.

# Returns
- `model`: The JuMP model
"""
function build_model(prob::PortfolioProblem)
    model = Model()

    n = prob.n_assets
    S = prob.n_scenarios

    # Variables
    @variable(model, x[1:n] >= 0)          # portfolio weights
    @variable(model, z[1:S] >= 0)          # CVaR shortfall auxiliaries
    @variable(model, alpha)                 # VaR threshold (free)
    @variable(model, d_plus[1:n] >= 0)     # buy amounts (turnover)
    @variable(model, d_minus[1:n] >= 0)    # sell amounts (turnover)

    # Objective: maximize expected return
    @objective(model, Max, sum(prob.expected_returns[i] * x[i] for i in 1:n))

    # CVaR scenario constraints: z[s] >= -(portfolio return in scenario s) - alpha
    for s in 1:S
        @constraint(model, z[s] >= -sum(prob.scenario_returns[s, i] * x[i] for i in 1:n) - alpha)
    end

    # CVaR risk limit
    @constraint(model, alpha + (1.0 / ((1.0 - prob.beta) * S)) * sum(z[s] for s in 1:S) <= prob.cvar_limit)

    # Budget constraint (fully invested)
    @constraint(model, sum(x[i] for i in 1:n) == 1.0)

    # Sector upper limits
    for s in 1:prob.n_sectors
        assets_in_sector = [i for i in 1:n if prob.sector_assignments[i] == s]
        if !isempty(assets_in_sector)
            @constraint(model, sum(x[i] for i in assets_in_sector) <= prob.sector_upper[s])
        end
    end

    # Region upper limits
    for r in 1:prob.n_regions
        assets_in_region = [i for i in 1:n if prob.region_assignments[i] == r]
        if !isempty(assets_in_region)
            @constraint(model, sum(x[i] for i in assets_in_region) <= prob.region_upper[r])
        end
    end

    # Asset class min/max bounds
    for c in 1:prob.n_asset_classes
        assets_in_class = [i for i in 1:n if prob.asset_class_assignments[i] == c]
        if !isempty(assets_in_class)
            @constraint(model, sum(x[i] for i in assets_in_class) >= prob.asset_class_lower[c])
            @constraint(model, sum(x[i] for i in assets_in_class) <= prob.asset_class_upper[c])
        end
    end

    # Factor exposure bounds
    for f in 1:prob.n_factors
        @constraint(model, sum(prob.factor_loadings[i, f] * x[i] for i in 1:n) >= prob.factor_lower[f])
        @constraint(model, sum(prob.factor_loadings[i, f] * x[i] for i in 1:n) <= prob.factor_upper[f])
    end

    # Position size limits
    for i in 1:n
        @constraint(model, x[i] <= prob.max_position[i])
    end

    # Turnover decomposition: d_plus[i] - d_minus[i] = x[i] - benchmark[i]
    for i in 1:n
        @constraint(model, d_plus[i] - d_minus[i] == x[i] - prob.benchmark[i])
    end

    # Turnover limit
    @constraint(model, sum(d_plus[i] + d_minus[i] for i in 1:n) <= prob.turnover_limit)

    return model
end

# Register the problem type
register_problem(
    :portfolio,
    PortfolioProblem,
    "CVaR portfolio optimization with sector, region, asset class, factor, position, and turnover constraints"
)
