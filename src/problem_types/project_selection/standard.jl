using JuMP
using Random
using Distributions

"""
    ProjectPortfolioWitness

Planted portfolio for a `feasible` instance: `selected` lists (sorted) the
projects funded at `x_p = 1`; every other project is at `x_p = 0`. The
portfolio is closed under prerequisites, takes at most one project from every
mutually exclusive alternative group, and the budget, headcount, risk,
high-risk-count and division-mandate rows were all drawn around it with
positive slack, so it is a genuine 0/1 point of the model (and therefore also
feasible for the default LP relaxation).
"""
struct ProjectPortfolioWitness
    selected::Vector{Int}
end

"""
    DivisionMandateCertificate

Relaxation-valid infeasibility proof for an `infeasible` instance. Division
`division` must deliver at least `mandate` of its projects, but summing its
yearly budget rows gives `sum_p C_p x_p <= division_budget_total`, where
`C_p = sum_t spend[p][t]` is project `p`'s lifetime cost. For `0 <= x <= 1`
the cheapest way to fund `mandate` projects of the division costs exactly the
sum of its `mandate` cheapest lifetime costs (`cheapest_projects`,
`cheapest_cost`), and

    cheapest_cost >= (1 + margin) * division_budget_total,  margin >= 5%,

so the mandate row, the division's `n_years` budget rows and the `x <= 1`
bounds are jointly infeasible even fractionally. The argument aggregates
several rows, so bound propagation alone does not expose it.
"""
struct DivisionMandateCertificate
    division::Int
    mandate::Int
    cheapest_projects::Vector{Int}
    cheapest_cost::Float64
    division_budget_total::Float64
end

"""
    ProjectSelectionProblem <: ProblemGenerator

Multi-year, multi-division capital-portfolio selection (R&D / capital program
planning) with prerequisite chains, mutually exclusive alternatives, budget and
headcount envelopes, and division delivery mandates.

# Overview

Each project `p` belongs to a division and a strategic theme, starts in year
`start_year[p]`, runs `duration[p]` years, and has a year-by-year spending
profile `spend[p]` (front-loaded hump) and engineering headcount profile
`fte[p]` (spend times a project-specific labor intensity). Its value is a
lognormal NPV correlated with lifetime cost through an ROI class (platform
projects have low stand-alone ROI: they exist to enable others). Binary
`x_p` funds the project; the default `relax_integer=true` solves the
`0 <= x_p <= 1` relaxation.

Rows (all sparse; a project touches `3 * duration + O(1)` rows):

  - corporate capital budget per year (`n_years` rows),
  - division capital budget per (division, year) with spending,
  - division headcount capacity per (division, year) with spending,
  - prerequisites `x_p <= x_q` (dependents need their platform projects;
    platforms form a DAG; in-degree <= 2),
  - mutually exclusive alternatives `sum_{p in G} x_p <= 1` (2-3 scopes of the
    same initiative),
  - division delivery mandates `sum_{p in d} x_p >= m_d` (covering rows),
  - one portfolio-risk row and one high-risk-count row.

Divisions scale with size (about 60 projects each), so the row count grows
linearly (~1.1-1.4 rows per column) and nnz stays ~10-14 per column. The
former generator drew an O(n^2) dependency matrix (10M rows at 10k projects).

# Feasibility

  - `feasible`: a portfolio closed under prerequisites is planted (about
    35-45% of the projects); every envelope is its usage times
    `U(1.04, 1.25)` (plus a small floor) and every mandate is at most the
    planted count, so the planted 0/1 point is stored as a
    `ProjectPortfolioWitness`.
  - `infeasible`: the same instance construction, then the largest division's
    mandate is raised to the smallest count whose cheapest lifetime costs
    exceed its summed yearly budgets by `U(6%, 20%)`; stored as a
    `DivisionMandateCertificate`.
  - `unknown`: a natural capital review — division envelopes are fractions of
    division demand, mandates are fractions of division size, and the
    corporate envelope is drawn around (just below to just above) the
    estimated cost of meeting every mandate together with prerequisites. No
    witness or certificate; the outcome is decided by the LP.
"""
struct ProjectSelectionProblem <: ProblemGenerator
    n_projects::Int
    n_years::Int
    n_divisions::Int
    n_themes::Int
    division::Vector{Int}
    theme::Vector{Int}
    start_year::Vector{Int}
    duration::Vector{Int}
    spend::Vector{Vector{Float64}}
    fte::Vector{Vector{Float64}}
    returns::Vector{Float64}
    risk_scores::Vector{Float64}
    is_platform::BitVector
    prerequisites::Vector{Tuple{Int, Int}}
    exclusive_groups::Vector{Vector{Int}}
    corporate_budget::Vector{Float64}
    division_budget::Matrix{Float64}
    division_fte::Matrix{Float64}
    division_mandate::Vector{Int}
    risk_budget::Float64
    high_risk_threshold::Float64
    max_high_risk::Int
    feasibility_status::FeasibilityStatus
    feasible_witness::Union{Nothing, ProjectPortfolioWitness}
    infeasibility_certificate::Union{Nothing, DivisionMandateCertificate}
end

ps_lifetime_cost(prob::ProjectSelectionProblem, p::Int) = sum(prob.spend[p])

# Year-indexed usage of a 0/1 portfolio: corporate spend, division spend and
# division headcount, plus the per-division selected count.
function _ps_usage(
    selected::AbstractVector{Bool},
    division::Vector{Int},
    start_year::Vector{Int},
    spend::Vector{Vector{Float64}},
    fte::Vector{Vector{Float64}},
    n_years::Int,
    n_divisions::Int,
)
    corp = zeros(n_years)
    div_spend = zeros(n_divisions, n_years)
    div_fte = zeros(n_divisions, n_years)
    count = zeros(Int, n_divisions)
    for p in eachindex(selected)
        selected[p] || continue
        d = division[p]
        count[d] += 1
        for (k, c) in enumerate(spend[p])
            t = start_year[p] + k - 1
            corp[t] += c
            div_spend[d, t] += c
            div_fte[d, t] += fte[p][k]
        end
    end
    return corp, div_spend, div_fte, count
end

# Demand (usage if every project were funded) per division-year.
function _ps_demand(division, start_year, spend, fte, n_years, n_divisions)
    return _ps_usage(
        trues(length(division)), division, start_year, spend, fte, n_years, n_divisions
    )
end

# Add prerequisite closure to a portfolio (iterate to a fixed point; the
# prerequisite graph is a DAG on platform projects, so this terminates fast).
function _ps_close!(selected::BitVector, prereqs_of::Vector{Vector{Int}})
    stack = findall(selected)
    while !isempty(stack)
        p = pop!(stack)
        for q in prereqs_of[p]
            if !selected[q]
                selected[q] = true
                push!(stack, q)
            end
        end
    end
    return selected
end

"""
    ProjectSelectionProblem(target_variables, feasibility_status, seed)

Construct a portfolio with exactly `target_variables` projects (one binary per
project). All randomness comes from a local `MersenneTwister(seed)`.
"""
function ProjectSelectionProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be positive (got $target_variables)"))
    rng = MersenneTwister(seed)
    n = target_variables

    n_years = n <= 200 ? rand(rng, 3:5) : rand(rng, 5:8)
    n_divisions = clamp(round(Int, n / 60), 1, n)
    n_themes = clamp(round(Int, sqrt(n) / 3), 2, 12)

    # --- Organisation: divisions of uneven size, themes per project ---
    div_weight = rand(rng, Gamma(4.0, 1.0), n_divisions)
    division = Vector{Int}(undef, n)
    # Every division gets at least one project; the rest follow the weights.
    perm = randperm(rng, n)
    cdf = cumsum(div_weight ./ sum(div_weight))
    for (k, p) in enumerate(perm)
        division[p] = k <= n_divisions ? k : min(searchsortedfirst(cdf, rand(rng)), n_divisions)
    end
    theme = rand(rng, 1:n_themes, n)
    div_members = [Int[] for _ in 1:n_divisions]
    for p in 1:n
        push!(div_members[division[p]], p)
    end

    # --- Timing and spending profiles ---
    start_year = Vector{Int}(undef, n)
    duration = Vector{Int}(undef, n)
    spend = Vector{Vector{Float64}}(undef, n)
    fte = Vector{Vector{Float64}}(undef, n)
    lifetime = Vector{Float64}(undef, n)
    # Division cost scale: some divisions run bigger programmes.
    div_scale = exp.(0.4 .* randn(rng, n_divisions))
    for p in 1:n
        dmax = min(4, n_years)
        duration[p] = clamp(rand(rng, Poisson(1.3)) + 1, 1, dmax)
        start_year[p] = rand(rng, 1:(n_years - duration[p] + 1))
        total = 2.0e6 * div_scale[division[p]] * exp(0.9 * randn(rng))   # lifetime cost, $
        total = clamp(total, 5.0e4, 2.0e8)
        lifetime[p] = total
        # Front-loaded hump profile with noise.
        shape = [exp(-0.35 * (k - 1.6)^2) * (0.75 + 0.5 * rand(rng)) for k in 1:duration[p]]
        spend[p] = total .* shape ./ sum(shape)
        labor_share = 0.25 + 0.5 * rand(rng)
        fte_cost = 1.6e5 + 0.8e5 * rand(rng)
        fte[p] = spend[p] .* (labor_share / fte_cost)
    end

    # --- Platforms, prerequisites, exclusive alternatives ---
    is_platform = falses(n)
    for p in 1:n
        is_platform[p] = rand(rng) < 0.2
    end
    platforms_by_div = [Int[] for _ in 1:n_divisions]
    platforms_all = Int[]
    for p in 1:n
        if is_platform[p]
            push!(platforms_by_div[division[p]], p)
            push!(platforms_all, p)
        end
    end
    prereqs_of = [Int[] for _ in 1:n]
    function pick_platform(p, allow)
        # Prefer a platform of the same division that does not start later;
        # fall back to any platform. `allow(q)` enforces the DAG direction.
        pool = rand(rng) < 0.75 ? platforms_by_div[division[p]] : platforms_all
        isempty(pool) && (pool = platforms_all)
        isempty(pool) && return 0
        for _ in 1:4
            q = pool[rand(rng, 1:length(pool))]
            if q != p && allow(q) && start_year[q] <= start_year[p] && !(q in prereqs_of[p])
                return q
            end
        end
        return 0
    end
    for p in 1:n
        if is_platform[p]
            # Platforms build on earlier platforms only (index order: a DAG).
            rand(rng) < 0.3 || continue
            q = pick_platform(p, q -> q < p)
            q > 0 && push!(prereqs_of[p], q)
        else
            r = rand(rng)
            k = r < 0.35 ? 0 : (r < 0.8 ? 1 : 2)
            for _ in 1:k
                q = pick_platform(p, q -> true)
                q > 0 && push!(prereqs_of[p], q)
            end
        end
    end
    prerequisites = Tuple{Int, Int}[(p, q) for p in 1:n for q in prereqs_of[p]]

    # Alternatives: 2-3 scopes of the same initiative (same division and
    # theme, not platforms) — at most one may be funded.
    exclusive_groups = Vector{Int}[]
    in_group = falses(n)
    by_div_theme = Dict{Tuple{Int, Int}, Vector{Int}}()
    for p in 1:n
        is_platform[p] && continue
        push!(get!(by_div_theme, (division[p], theme[p]), Int[]), p)
    end
    for key in sort!(collect(keys(by_div_theme)))
        members = by_div_theme[key]
        shuffle!(rng, members)
        i = 1
        while i < length(members)
            if rand(rng) < 0.3
                g = min(length(members) - i + 1, rand(rng, 2:3))
                group = sort(members[i:(i + g - 1)])
                push!(exclusive_groups, group)
                in_group[group] .= true
                i += g
            else
                i += 1
            end
        end
    end

    # --- Value and risk ---
    returns = Vector{Float64}(undef, n)
    risk_scores = Vector{Float64}(undef, n)
    for p in 1:n
        if is_platform[p]
            roi = 0.6 + 0.6 * rand(rng)
            risk = 2.0 + 3.0 * rand(rng)
        else
            u = rand(rng)
            if u < 0.4
                roi, risk = 1.1 + 0.5 * rand(rng), 1.0 + 3.0 * rand(rng)
            elseif u < 0.8
                roi, risk = 1.4 + 1.0 * rand(rng), 3.0 + 3.0 * rand(rng)
            else
                roi, risk = 1.8 + 2.2 * rand(rng), 5.5 + 4.0 * rand(rng)
            end
        end
        returns[p] = lifetime[p] * roi * exp(0.25 * randn(rng))
        risk_scores[p] = clamp(risk * exp(0.1 * randn(rng)), 1.0, 10.0)
    end
    high_risk_threshold = 7.0

    corp_dem, div_dem, fte_dem, div_size = _ps_demand(
        division, start_year, spend, fte, n_years, n_divisions
    )

    # --- Plant a portfolio (used by feasible and infeasible) ---
    plant_fraction = 0.35 + 0.10 * rand(rng)
    planted = falses(n)
    for p in 1:n
        planted[p] = !in_group[p] && rand(rng) < plant_fraction
    end
    for group in exclusive_groups
        rand(rng) < 0.6 && (planted[group[rand(rng, 1:length(group))]] = true)
    end
    _ps_close!(planted, prereqs_of)
    any(planted) || (planted[findfirst(p -> isempty(prereqs_of[p]), 1:n)] = true)

    feasible_witness = nothing
    infeasibility_certificate = nothing
    n_high = count(>(high_risk_threshold), risk_scores)

    if feasibility_status == feasible || feasibility_status == infeasible
        corp_use, div_use, fte_use, plant_count = _ps_usage(
            planted, division, start_year, spend, fte, n_years, n_divisions
        )
        # Envelopes: planted usage x (1 + margin), plus a small floor so a
        # division-year without planted spend still has a usable allowance
        # (never above the division-year demand, so rows stay binding).
        floor_spend = 0.15 * 2.0e6
        division_budget = similar(div_use)
        division_fte = similar(fte_use)
        for d in 1:n_divisions, t in 1:n_years
            division_budget[d, t] = min(
                div_dem[d, t], div_use[d, t] * (1.04 + 0.21 * rand(rng)) + floor_spend
            )
            division_fte[d, t] = min(
                fte_dem[d, t], fte_use[d, t] * (1.04 + 0.21 * rand(rng)) + floor_spend / 2.0e5
            )
            # Keep the planted point strictly inside (demand cap can bind only
            # when the whole division-year is planted, which is then exact).
            division_budget[d, t] = max(division_budget[d, t], div_use[d, t])
            division_fte[d, t] = max(division_fte[d, t], fte_use[d, t])
        end
        # The corporate envelope binds below the sum of division envelopes.
        corporate_budget = [
            max(
                corp_use[t] * (1.01 + 0.04 * rand(rng)),
                sum(division_budget[:, t]) * (0.82 + 0.10 * rand(rng)),
            ) for t in 1:n_years
        ]
        division_mandate = [
            floor(Int, plant_count[d] * (0.55 + 0.4 * rand(rng))) for d in 1:n_divisions
        ]
        planted_risk = sum(risk_scores[p] for p in 1:n if planted[p]; init=0.0)
        risk_budget = planted_risk * (1.05 + 0.15 * rand(rng)) + 1.0
        planted_high = count(p -> planted[p] && risk_scores[p] > high_risk_threshold, 1:n)
        max_high_risk = max(planted_high, round(Int, planted_high * (1.05 + 0.2 * rand(rng))))

        if feasibility_status == feasible
            feasible_witness = ProjectPortfolioWitness(findall(planted))
        else
            # Division mandate beyond what its summed budgets can buy, even
            # taking the cheapest projects fractionally.
            dstar = argmax(div_size)
            members = sort(div_members[dstar]; by=p -> (lifetime[p], p))
            total_budget = sum(division_budget[dstar, :])
            margin = 1.06 + 0.14 * rand(rng)
            if sum(lifetime[members]) < margin * total_budget
                # Too generous an envelope for a certificate: shrink this
                # division's yearly budgets (still >= 0) to 60% of its cost.
                scale = 0.6 * sum(lifetime[members]) / total_budget
                division_budget[dstar, :] .*= scale
                total_budget = sum(division_budget[dstar, :])
            end
            cumulative = 0.0
            m = 0
            for p in members
                cumulative += lifetime[p]
                m += 1
                cumulative >= margin * total_budget && break
            end
            division_mandate[dstar] = m
            cheapest = sort(members[1:m])
            infeasibility_certificate = DivisionMandateCertificate(
                dstar, m, cheapest, sum(lifetime[cheapest]), total_budget
            )
            infeasibility_certificate.cheapest_cost >= 1.05 * total_budget ||
                error("project_selection: mandate certificate lost its margin")
        end
    else
        # Natural capital review: envelopes are fractions of demand, mandates
        # fractions of division size, and the corporate envelope is drawn
        # around the estimated cost of meeting every mandate.
        division_budget = div_dem .* (0.30 .+ 0.30 .* rand(rng, n_divisions, n_years))
        division_fte = fte_dem .* (0.35 .+ 0.30 .* rand(rng, n_divisions, n_years))
        division_mandate = [
            floor(Int, div_size[d] * (0.15 + 0.25 * rand(rng))) for d in 1:n_divisions
        ]
        # Estimated mandate portfolio: cheapest m_d per division + closure.
        estimate = falses(n)
        for d in 1:n_divisions
            members = sort(div_members[d]; by=p -> (lifetime[p], p))
            for p in members[1:division_mandate[d]]
                estimate[p] = true
            end
        end
        _ps_close!(estimate, prereqs_of)
        est_corp, _, _, _ = _ps_usage(
            estimate, division, start_year, spend, fte, n_years, n_divisions
        )
        kappa = exp(0.35 * randn(rng) - 0.40)
        corporate_budget = [
            min(sum(division_budget[:, t]), kappa * est_corp[t] * (0.9 + 0.2 * rand(rng))) for
            t in 1:n_years
        ]
        risk_budget = sum(risk_scores) * (0.25 + 0.2 * rand(rng))
        max_high_risk = max(1, round(Int, n_high * (0.2 + 0.3 * rand(rng))))
    end

    return ProjectSelectionProblem(
        n,
        n_years,
        n_divisions,
        n_themes,
        division,
        theme,
        start_year,
        duration,
        spend,
        fte,
        returns,
        risk_scores,
        is_platform,
        prerequisites,
        exclusive_groups,
        corporate_budget,
        division_budget,
        division_fte,
        division_mandate,
        risk_budget,
        high_risk_threshold,
        max_high_risk,
        feasibility_status,
        feasible_witness,
        infeasibility_certificate,
    )
end

"""
    build_model(prob::ProjectSelectionProblem)

Build the portfolio MILP (binary `x`, relaxed by default). Deterministic and
O(nnz): every row expression is accumulated in one pass over the projects.
"""
function build_model(prob::ProjectSelectionProblem)
    model = Model()
    n, T, D = prob.n_projects, prob.n_years, prob.n_divisions

    @variable(model, x[1:n], Bin)
    @objective(model, Max, sum(prob.returns[p] * x[p] for p in 1:n))

    corp = [AffExpr() for _ in 1:T]
    div_spend = [AffExpr() for _ in 1:D, _ in 1:T]
    div_fte = [AffExpr() for _ in 1:D, _ in 1:T]
    members = [AffExpr() for _ in 1:D]
    risk = AffExpr()
    high = AffExpr()
    for p in 1:n
        d = prob.division[p]
        for (k, c) in enumerate(prob.spend[p])
            t = prob.start_year[p] + k - 1
            add_to_expression!(corp[t], c, x[p])
            add_to_expression!(div_spend[d, t], c, x[p])
            add_to_expression!(div_fte[d, t], prob.fte[p][k], x[p])
        end
        add_to_expression!(members[d], 1.0, x[p])
        add_to_expression!(risk, prob.risk_scores[p], x[p])
        prob.risk_scores[p] > prob.high_risk_threshold && add_to_expression!(high, 1.0, x[p])
    end

    @constraint(model, corporate_budget[t in 1:T], corp[t] <= prob.corporate_budget[t])
    for d in 1:D, t in 1:T
        isempty(div_spend[d, t].terms) && continue
        @constraint(model, div_spend[d, t] <= prob.division_budget[d, t])
        @constraint(model, div_fte[d, t] <= prob.division_fte[d, t])
    end
    for (p, q) in prob.prerequisites
        @constraint(model, x[p] <= x[q])
    end
    for group in prob.exclusive_groups
        @constraint(model, sum(x[p] for p in group) <= 1)
    end
    for d in 1:D
        prob.division_mandate[d] > 0 || continue
        @constraint(model, members[d] >= prob.division_mandate[d])
    end
    @constraint(model, risk <= prob.risk_budget)
    isempty(high.terms) || @constraint(model, high <= prob.max_high_risk)

    return model
end

register_variant(
    :project_selection,
    :standard,
    ProjectSelectionProblem,
    "Multi-year, multi-division capital portfolio selection with yearly budget and headcount envelopes, platform prerequisites, exclusive alternatives, and division delivery mandates";
    tags=[:finance, :packing],
)
