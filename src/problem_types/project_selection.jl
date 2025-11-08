using JuMP
using Random
using Distributions
using StatsBase

"""
    ProjectSelectionProblem <: ProblemGenerator

Generator for project selection problems that maximize return by selecting a portfolio of projects.

# Fields
- `n_projects::Int`: Number of potential projects
- `projects::Vector{Int}`: List of project IDs
- `costs::Dict{Int,Float64}`: Cost of each project
- `returns::Dict{Int,Float64}`: Expected return of each project
- `risk_scores::Dict{Int,Float64}`: Risk score of each project
- `dependencies::Vector{Tuple{Int,Int}}`: Project dependencies (p1, p2) means p1 depends on p2
- `budget::Float64`: Total budget constraint
- `risk_budget::Float64`: Maximum total risk score allowed
- `max_high_risk_projects::Int`: Maximum number of high-risk projects
- `high_risk_threshold::Float64`: Threshold for defining high-risk projects
"""
struct ProjectSelectionProblem <: ProblemGenerator
    n_projects::Int
    projects::Vector{Int}
    costs::Dict{Int,Float64}
    returns::Dict{Int,Float64}
    risk_scores::Dict{Int,Float64}
    dependencies::Vector{Tuple{Int,Int}}
    budget::Float64
    risk_budget::Float64
    max_high_risk_projects::Int
    high_risk_threshold::Float64
end

"""
    ProjectSelectionProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a project selection problem instance.

# Arguments
- `target_variables`: Target number of variables (projects)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function ProjectSelectionProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # For project selection, variables = n_projects
    n_projects = target_variables

    # Determine scale and set parameters
    if target_variables <= 250
        # Small scale
        cost_mean = log(100_000)
        cost_std = 1.5
        min_cost = max(5_000, round(rand(LogNormal(cost_mean - cost_std, 0.5)), digits=0))
        max_cost = min(500_000, round(rand(LogNormal(cost_mean + cost_std, 0.5)), digits=0))

        return_multiplier = rand(Uniform(1.5, 4.0))
        min_return = max(10_000, min_cost * rand(Uniform(0.8, 1.2)))
        max_return = min(1_000_000, max_cost * return_multiplier)

        budget_factor = rand(Beta(2, 3)) * 0.4 + 0.2
        max_risk_score = rand(Uniform(8.0, 12.0))
        dependency_density = rand(Beta(2, 8)) * 0.1 + 0.05
    elseif target_variables <= 1000
        # Medium scale
        cost_mean = log(500_000)
        cost_std = 1.8
        min_cost = max(10_000, round(rand(LogNormal(cost_mean - cost_std, 0.6)), digits=0))
        max_cost = min(5_000_000, round(rand(LogNormal(cost_mean + cost_std, 0.6)), digits=0))

        return_multiplier = rand(Gamma(2, 2)) + 2.0
        min_return = max(50_000, min_cost * rand(Uniform(0.7, 1.3)))
        max_return = min(10_000_000, max_cost * return_multiplier)

        budget_factor = rand(Beta(3, 4)) * 0.35 + 0.15
        max_risk_score = rand(Uniform(12.0, 18.0))
        dependency_density = rand(Beta(3, 7)) * 0.15 + 0.1
    else
        # Large scale
        cost_mean = log(2_000_000)
        cost_std = 2.0
        min_cost = max(50_000, round(rand(LogNormal(cost_mean - cost_std, 0.7)), digits=0))
        max_cost = min(50_000_000, round(rand(LogNormal(cost_mean + cost_std, 0.7)), digits=0))

        return_multiplier = rand(Gamma(1.5, 2.5)) + 1.5
        min_return = max(100_000, min_cost * rand(Uniform(0.6, 1.4)))
        max_return = min(100_000_000, max_cost * return_multiplier)

        budget_factor = rand(Beta(4, 6)) * 0.3 + 0.1
        max_risk_score = rand(Uniform(15.0, 25.0))
        dependency_density = rand(Beta(2, 5)) * 0.15 + 0.15
    end

    # Ensure min < max
    if min_cost >= max_cost
        min_cost = max_cost * 0.3
    end
    if min_return >= max_return
        min_return = max_return * 0.4
    end

    # Risk budget
    risk_per_project = rand(Uniform(0.8, 2.5))
    risk_budget = n_projects * risk_per_project

    # Maximum high-risk projects
    high_risk_fraction = rand(Beta(2, 5)) * 0.2 + 0.1
    max_high_risk_projects = max(1, ceil(Int, n_projects * high_risk_fraction))

    # High risk threshold
    high_risk_threshold = max_risk_score * rand(Uniform(0.6, 0.8))

    projects = collect(1:n_projects)

    # Generate correlated costs and returns
    costs = Dict{Int, Float64}()
    returns = Dict{Int, Float64}()

    # Project categories: (min_roi, max_roi, risk_factor, probability_weight)
    project_categories = [
        (0.8, 1.5, 0.3, 0.4),   # Low risk, low return
        (1.2, 2.5, 0.6, 0.4),   # Medium risk, medium return
        (1.8, 4.0, 0.9, 0.2)    # High risk, high return
    ]

    category_weights = [cat[4] for cat in project_categories]

    for p in projects
        # Assign project to a category
        cat_idx = sample(1:length(project_categories), Weights(category_weights))
        cat = project_categories[cat_idx]

        # Generate cost using log-normal distribution
        cost_range = max_cost - min_cost
        cost_mean_ln = log(min_cost + cost_range * 0.3)
        cost_std_ln = 0.8
        base_cost = min_cost + (max_cost - min_cost) * min(1.0, max(0.0, rand(LogNormal(cost_mean_ln, cost_std_ln)) / exp(cost_mean_ln + cost_std_ln^2)))
        costs[p] = base_cost

        # Generate correlated return
        target_roi = rand(Uniform(cat[1], cat[2]))
        noise_factor = rand(Normal(1.0, cat[3] * 0.2))
        returns[p] = base_cost * target_roi * max(0.5, noise_factor)
        returns[p] = max(min_return, min(max_return, returns[p]))
    end

    # Generate risk scores
    risk_scores = Dict{Int, Float64}()
    for p in projects
        return_percentile = (returns[p] - min_return) / (max_return - min_return)

        alpha = 1.5 + return_percentile * 2.0
        beta = 3.0 - return_percentile * 1.5
        base_risk = rand(Beta(alpha, beta))

        noise = rand(Normal(1.0, 0.15))
        risk_scores[p] = max(1.0, min(max_risk_score, base_risk * max_risk_score * noise))
    end

    # Generate dependencies
    dependencies = Tuple{Int,Int}[]
    sorted_projects = sort(projects, by=p -> costs[p])

    for i in 1:n_projects
        for j in 1:(i-1)
            proj_i = sorted_projects[i]
            proj_j = sorted_projects[j]

            cost_factor = (costs[proj_i] - min_cost) / (max_cost - min_cost)
            risk_similarity = 1.0 - abs(risk_scores[proj_i] - risk_scores[proj_j]) / max_risk_score

            dependency_prob = dependency_density * (0.5 + 0.3 * cost_factor + 0.2 * risk_similarity)

            if rand() < dependency_prob
                push!(dependencies, (proj_i, proj_j))
            end
        end
    end

    # Calculate budget
    total_cost = sum(values(costs))
    budget = total_cost * budget_factor

    return ProjectSelectionProblem(
        n_projects, projects, costs, returns, risk_scores, dependencies,
        budget, risk_budget, max_high_risk_projects, high_risk_threshold
    )
end

"""
    build_model(prob::ProjectSelectionProblem)

Build a JuMP model for the project selection problem.

# Arguments
- `prob`: ProjectSelectionProblem instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::ProjectSelectionProblem)
    model = Model()

    # Decision variables: binary selection for each project
    @variable(model, x[prob.projects], Bin)

    # Objective: Maximize total return
    @objective(model, Max, sum(prob.returns[p] * x[p] for p in prob.projects))

    # Budget constraint
    @constraint(model, sum(prob.costs[p] * x[p] for p in prob.projects) <= prob.budget)

    # Risk constraint
    @constraint(model, sum(prob.risk_scores[p] * x[p] for p in prob.projects) <= prob.risk_budget)

    # Project dependencies
    for (p1, p2) in prob.dependencies
        @constraint(model, x[p1] <= x[p2])
    end

    # Maximum number of high-risk projects
    high_risk_projects = filter(p -> prob.risk_scores[p] > prob.high_risk_threshold, prob.projects)
    if !isempty(high_risk_projects)
        @constraint(model, sum(x[p] for p in high_risk_projects) <= prob.max_high_risk_projects)
    end

    return model
end

# Register the problem type
register_problem(
    :project_selection,
    ProjectSelectionProblem,
    "Project selection problem that maximizes return by selecting a portfolio of projects subject to budget, risk, and dependency constraints"
)
