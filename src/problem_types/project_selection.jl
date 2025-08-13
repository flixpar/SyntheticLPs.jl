using JuMP
using Random
using Distributions
using StatsBase

"""
    generate_project_selection_problem(params::Dict=Dict(); seed::Int=0)

Generate a project selection optimization problem instance using realistic distributions.

# Arguments
- `params`: Dictionary of problem parameters
  - `:n_projects`: Number of potential projects (default: 15)
  - `:min_return`: Minimum expected return per project (default: 50000.0)
  - `:max_return`: Maximum expected return per project (default: 1000000.0)
  - `:min_cost`: Minimum project cost (default: 20000.0)
  - `:max_cost`: Maximum project cost (default: 500000.0)
  - `:budget_factor`: Total budget as fraction of total cost (default: 0.3)
  - `:dependency_density`: Base probability of dependency between projects (default: 0.1)
  - `:max_risk_score`: Maximum risk score for any project (default: 10.0)
  - `:risk_budget`: Maximum total risk score allowed (default: 25.0)
  - `:max_high_risk_projects`: Maximum number of high-risk projects allowed (default: 2)
  - `:high_risk_threshold`: Threshold for defining high-risk projects (default: 7.0)
- `seed`: Random seed for reproducibility (default: 0)

# Problem Characteristics
- Project costs follow log-normal distributions, skewed toward lower costs
- Returns are correlated with costs through realistic ROI categories:
  - Low risk/low return (40% of projects): 0.8-1.5x ROI
  - Medium risk/medium return (40% of projects): 1.2-2.5x ROI  
  - High risk/high return (20% of projects): 1.8-4.0x ROI
- Risk scores use Beta distributions correlated with returns
- Dependencies consider project cost and risk similarity
- Scales appropriately for small (50-250), medium (250-1000), and large (1000-10000) problem sizes

# Returns
- `model`: The JuMP model
- `params`: Dictionary of all parameters used (including defaults)
"""
function generate_project_selection_problem(params::Dict=Dict(); seed::Int=0)
    # Set random seed
    Random.seed!(seed)
    
    # Extract parameters with defaults
    n_projects = get(params, :n_projects, 15)
    min_return = get(params, :min_return, 50000.0)
    max_return = get(params, :max_return, 1000000.0)
    min_cost = get(params, :min_cost, 20000.0)
    max_cost = get(params, :max_cost, 500000.0)
    budget_factor = get(params, :budget_factor, 0.3)
    dependency_density = get(params, :dependency_density, 0.1)
    max_risk_score = get(params, :max_risk_score, 10.0)
    risk_budget = get(params, :risk_budget, 25.0)
    max_high_risk_projects = get(params, :max_high_risk_projects, 2)
    high_risk_threshold = get(params, :high_risk_threshold, 7.0)
    
    # Save actual parameters used
    actual_params = Dict{Symbol, Any}(
        :n_projects => n_projects,
        :min_return => min_return,
        :max_return => max_return,
        :min_cost => min_cost,
        :max_cost => max_cost,
        :budget_factor => budget_factor,
        :dependency_density => dependency_density,
        :max_risk_score => max_risk_score,
        :risk_budget => risk_budget,
        :max_high_risk_projects => max_high_risk_projects,
        :high_risk_threshold => high_risk_threshold
    )
    
    projects = collect(1:n_projects)
    
    # Generate correlated costs and returns using realistic distributions
    costs = Dict{Int, Float64}()
    returns = Dict{Int, Float64}()
    
    # Create project categories with different risk-return profiles
    # Each category: (min_roi, max_roi, risk_factor, probability_weight)
    project_categories = [
        (0.8, 1.5, 0.3, 0.4),   # Low risk, low return - 40% of projects
        (1.2, 2.5, 0.6, 0.4),   # Medium risk, medium return - 40% of projects
        (1.8, 4.0, 0.9, 0.2)    # High risk, high return - 20% of projects
    ]
    
    # Use weighted sampling for categories
    category_weights = [cat[4] for cat in project_categories]
    
    for p in projects
        # Assign project to a category using weighted sampling
        cat_idx = sample(1:length(project_categories), Weights(category_weights))
        cat = project_categories[cat_idx]
        
        # Generate base cost using log-normal distribution (more realistic for project costs)
        cost_range = max_cost - min_cost
        cost_mean = log(min_cost + cost_range * 0.3)  # Skewed toward lower costs
        cost_std = 0.8
        base_cost = min_cost + (max_cost - min_cost) * min(1.0, max(0.0, rand(LogNormal(cost_mean, cost_std)) / exp(cost_mean + cost_std^2)))
        costs[p] = base_cost
        
        # Generate correlated return with category-specific ROI and noise
        target_roi = rand(Uniform(cat[1], cat[2]))
        
        # Add realistic noise based on project risk
        noise_factor = rand(Normal(1.0, cat[3] * 0.2))  # Higher risk = more noise
        returns[p] = base_cost * target_roi * max(0.5, noise_factor)
        
        # Ensure bounds
        returns[p] = max(min_return, min(max_return, returns[p]))
    end
    
    # Generate risk scores (correlated with returns and project categories)
    risk_scores = Dict{Int, Float64}()
    for p in projects
        # Higher returns tend to mean higher risk, but with realistic distributions
        return_percentile = (returns[p] - min_return) / (max_return - min_return)
        
        # Base risk follows a Beta distribution shifted and scaled
        # Higher return percentile increases the Beta parameters for higher risk
        alpha = 1.5 + return_percentile * 2.0
        beta = 3.0 - return_percentile * 1.5
        base_risk = rand(Beta(alpha, beta))
        
        # Scale to risk score range with some additional noise
        noise = rand(Normal(1.0, 0.15))
        risk_scores[p] = max(1.0, min(max_risk_score, base_risk * max_risk_score * noise))
    end
    
    # Generate dependencies (ensuring no cycles)
    dependencies = Tuple{Int,Int}[]
    
    # Sort projects by cost (larger projects tend to depend on smaller ones)
    sorted_projects = sort(projects, by=p -> costs[p])
    
    for i in 1:n_projects
        for j in 1:(i-1)  # Only allow dependencies on earlier projects
            # More realistic dependency probability based on project characteristics
            proj_i = sorted_projects[i]
            proj_j = sorted_projects[j]
            
            # Higher cost projects are more likely to depend on others
            cost_factor = (costs[proj_i] - min_cost) / (max_cost - min_cost)
            
            # Projects with similar risk profiles are more likely to be related
            risk_similarity = 1.0 - abs(risk_scores[proj_i] - risk_scores[proj_j]) / max_risk_score
            
            # Combine factors for dependency probability
            dependency_prob = dependency_density * (0.5 + 0.3 * cost_factor + 0.2 * risk_similarity)
            
            if rand() < dependency_prob
                # Project i depends on project j
                push!(dependencies, (proj_i, proj_j))
            end
        end
    end
    
    # Calculate total cost and set budget
    total_cost = sum(values(costs))
    budget = total_cost * budget_factor
    
    # Store generated data in params
    actual_params[:projects] = projects
    actual_params[:costs] = costs
    actual_params[:returns] = returns
    actual_params[:dependencies] = dependencies
    actual_params[:risk_scores] = risk_scores
    actual_params[:budget] = budget
    actual_params[:total_cost] = total_cost
    
    # Create model
    model = Model()
    
    # Decision variables: binary selection for each project
    @variable(model, x[projects], Bin)
    
    # Objective: Maximize total return
    @objective(model, Max, sum(returns[p] * x[p] for p in projects))
    
    # Budget constraint
    @constraint(model, sum(costs[p] * x[p] for p in projects) <= budget)
    
    # Risk constraint
    @constraint(model, sum(risk_scores[p] * x[p] for p in projects) <= risk_budget)
    
    # Project dependencies
    for (p1, p2) in dependencies
        @constraint(model, x[p1] <= x[p2])
    end
    
    # Maximum number of high-risk projects
    high_risk_projects = filter(p -> risk_scores[p] > high_risk_threshold, projects)
    if !isempty(high_risk_projects)
        @constraint(model, sum(x[p] for p in high_risk_projects) <= max_high_risk_projects)
    end
    
    return model, actual_params
end

"""
    sample_project_selection_parameters(size::Symbol=:medium; seed::Int=0)

Sample realistic parameters for a project selection problem using probability distributions.

# Arguments
- `size`: Symbol specifying the problem size (:small, :medium, :large)
  - `:small`: 50-250 projects (small company/startup scale)
  - `:medium`: 250-1000 projects (medium corporation/R&D department scale)  
  - `:large`: 1000-10000 projects (large corporation/government scale)
- `seed`: Random seed for reproducibility (default: 0)

# Parameter Distributions by Size
- **Small**: Project costs 5K-500K, returns 10K-1M, budget factor 0.2-0.6
- **Medium**: Project costs 10K-5M, returns 50K-10M, budget factor 0.15-0.5
- **Large**: Project costs 50K-50M, returns 100K-100M, budget factor 0.1-0.4

All distributions use realistic probability models (LogNormal, Beta, Gamma, Uniform) 
rather than simple uniform sampling to better reflect real-world project characteristics.

# Returns
- Dictionary of sampled parameters
"""
function sample_project_selection_parameters(size::Symbol=:medium; seed::Int=0)
    Random.seed!(seed)
    
    params = Dict{Symbol, Any}()
    
    # Set size-dependent parameters using realistic distributions
    if size == :small
        # Small company/startup: 50-250 projects (software features, marketing campaigns, small initiatives)
        params[:n_projects] = rand(50:250)
        
        # Project costs: $5K-$500K, using log-normal distribution for realistic cost spread
        cost_mean = log(100_000)  # Geometric mean around $100K
        cost_std = 1.5
        params[:min_cost] = max(5_000, round(rand(LogNormal(cost_mean - cost_std, 0.5)), digits=0))
        params[:max_cost] = min(500_000, round(rand(LogNormal(cost_mean + cost_std, 0.5)), digits=0))
        
        # Returns: $10K-$1M, typically 1.5-4x cost with variation
        return_multiplier = rand(Uniform(1.5, 4.0))
        params[:min_return] = max(10_000, params[:min_cost] * rand(Uniform(0.8, 1.2)))
        params[:max_return] = min(1_000_000, params[:max_cost] * return_multiplier)
        
        # Budget factor: 0.2-0.6 (smaller companies often have more flexibility)
        params[:budget_factor] = rand(Beta(2, 3)) * 0.4 + 0.2
        
        # Risk parameters scaled for small operations
        params[:max_risk_score] = rand(Uniform(8.0, 12.0))
        params[:dependency_density] = rand(Beta(2, 8)) * 0.1 + 0.05  # Lower density for simpler projects
        
    elseif size == :medium
        # Medium corporation/R&D department: 250-1000 projects (enterprise initiatives, system implementations)
        params[:n_projects] = rand(250:1000)
        
        # Project costs: $10K-$5M, broader range with heavy tail
        cost_mean = log(500_000)  # Geometric mean around $500K
        cost_std = 1.8
        params[:min_cost] = max(10_000, round(rand(LogNormal(cost_mean - cost_std, 0.6)), digits=0))
        params[:max_cost] = min(5_000_000, round(rand(LogNormal(cost_mean + cost_std, 0.6)), digits=0))
        
        # Returns: $50K-$10M, typically 2-6x cost with more variation
        return_multiplier = rand(Gamma(2, 2)) + 2.0  # Mean around 4x, with tail
        params[:min_return] = max(50_000, params[:min_cost] * rand(Uniform(0.7, 1.3)))
        params[:max_return] = min(10_000_000, params[:max_cost] * return_multiplier)
        
        # Budget factor: 0.15-0.5 (more constrained)
        params[:budget_factor] = rand(Beta(3, 4)) * 0.35 + 0.15
        
        # Risk parameters scaled for medium operations
        params[:max_risk_score] = rand(Uniform(12.0, 18.0))
        params[:dependency_density] = rand(Beta(3, 7)) * 0.15 + 0.1  # Medium density
        
    elseif size == :large
        # Large corporation/government: 1000-10000 projects (major enterprise initiatives, infrastructure)
        params[:n_projects] = rand(1000:10000)
        
        # Project costs: $50K-$50M, very broad range with heavy tail
        cost_mean = log(2_000_000)  # Geometric mean around $2M
        cost_std = 2.0
        params[:min_cost] = max(50_000, round(rand(LogNormal(cost_mean - cost_std, 0.7)), digits=0))
        params[:max_cost] = min(50_000_000, round(rand(LogNormal(cost_mean + cost_std, 0.7)), digits=0))
        
        # Returns: $100K-$100M, typically 1.5-8x cost with high variation
        return_multiplier = rand(Gamma(1.5, 2.5)) + 1.5  # Mean around 5.25x, heavy tail
        params[:min_return] = max(100_000, params[:min_cost] * rand(Uniform(0.6, 1.4)))
        params[:max_return] = min(100_000_000, params[:max_cost] * return_multiplier)
        
        # Budget factor: 0.1-0.4 (very constrained due to scale)
        params[:budget_factor] = rand(Beta(4, 6)) * 0.3 + 0.1
        
        # Risk parameters scaled for large operations
        params[:max_risk_score] = rand(Uniform(15.0, 25.0))
        params[:dependency_density] = rand(Beta(2, 5)) * 0.15 + 0.15  # Higher density for complex interdependencies
        
    else
        error("Unknown size: $size. Must be :small, :medium, or :large")
    end
    
    # Ensure min < max for costs and returns
    if params[:min_cost] >= params[:max_cost]
        params[:min_cost] = params[:max_cost] * 0.3
    end
    if params[:min_return] >= params[:max_return]
        params[:min_return] = params[:max_return] * 0.4
    end
    
    # Risk budget: Scale with number of projects and complexity
    risk_per_project = rand(Uniform(0.8, 2.5))
    params[:risk_budget] = params[:n_projects] * risk_per_project
    
    # Maximum high-risk projects: 10-30% of total projects
    high_risk_fraction = rand(Beta(2, 5)) * 0.2 + 0.1
    params[:max_high_risk_projects] = max(1, ceil(Int, params[:n_projects] * high_risk_fraction))
    
    # High risk threshold: 60-80% of maximum risk score
    params[:high_risk_threshold] = params[:max_risk_score] * rand(Uniform(0.6, 0.8))
    
    return params
end

"""
    sample_project_selection_parameters(target_variables::Int; seed::Int=0)

Sample realistic parameters for a project selection problem targeting exactly the specified number of variables.

# Arguments
- `target_variables`: Target number of variables in the LP formulation (exact, since variables = n_projects)
- `seed`: Random seed for reproducibility (default: 0)

# Parameter Scaling
Automatically determines appropriate scale based on target_variables:
- ≤250: Small scale (startup/small company parameters)
- 251-1000: Medium scale (corporation/R&D department parameters)
- >1000: Large scale (enterprise/government parameters)

Uses the same realistic probability distributions as size-based sampling, but with 
the exact number of projects specified by target_variables.

# Returns
- Dictionary of sampled parameters
"""
function sample_project_selection_parameters(target_variables::Int; seed::Int=0)
    Random.seed!(seed)
    
    params = Dict{Symbol, Any}()
    
    # For project selection, variables = n_projects, so set directly
    params[:n_projects] = target_variables
    
    # Determine scale based on target variable count and use appropriate distributions
    if target_variables <= 250
        # Small scale: 50-250 projects
        scale = :small
        
        # Project costs: $5K-$500K, using log-normal distribution
        cost_mean = log(100_000)  # Geometric mean around $100K
        cost_std = 1.5
        params[:min_cost] = max(5_000, round(rand(LogNormal(cost_mean - cost_std, 0.5)), digits=0))
        params[:max_cost] = min(500_000, round(rand(LogNormal(cost_mean + cost_std, 0.5)), digits=0))
        
        # Returns: $10K-$1M, typically 1.5-4x cost with variation
        return_multiplier = rand(Uniform(1.5, 4.0))
        params[:min_return] = max(10_000, params[:min_cost] * rand(Uniform(0.8, 1.2)))
        params[:max_return] = min(1_000_000, params[:max_cost] * return_multiplier)
        
        # Budget factor: 0.2-0.6 (smaller companies often have more flexibility)
        params[:budget_factor] = rand(Beta(2, 3)) * 0.4 + 0.2
        
        # Risk parameters scaled for small operations
        params[:max_risk_score] = rand(Uniform(8.0, 12.0))
        params[:dependency_density] = rand(Beta(2, 8)) * 0.1 + 0.05  # Lower density for simpler projects
        
    elseif target_variables <= 1000
        # Medium scale: 250-1000 projects
        scale = :medium
        
        # Project costs: $10K-$5M, broader range with heavy tail
        cost_mean = log(500_000)  # Geometric mean around $500K
        cost_std = 1.8
        params[:min_cost] = max(10_000, round(rand(LogNormal(cost_mean - cost_std, 0.6)), digits=0))
        params[:max_cost] = min(5_000_000, round(rand(LogNormal(cost_mean + cost_std, 0.6)), digits=0))
        
        # Returns: $50K-$10M, typically 2-6x cost with more variation
        return_multiplier = rand(Gamma(2, 2)) + 2.0  # Mean around 4x, with tail
        params[:min_return] = max(50_000, params[:min_cost] * rand(Uniform(0.7, 1.3)))
        params[:max_return] = min(10_000_000, params[:max_cost] * return_multiplier)
        
        # Budget factor: 0.15-0.5 (more constrained)
        params[:budget_factor] = rand(Beta(3, 4)) * 0.35 + 0.15
        
        # Risk parameters scaled for medium operations
        params[:max_risk_score] = rand(Uniform(12.0, 18.0))
        params[:dependency_density] = rand(Beta(3, 7)) * 0.15 + 0.1  # Medium density
        
    else
        # Large scale: 1000-10000 projects
        scale = :large
        
        # Project costs: $50K-$50M, very broad range with heavy tail
        cost_mean = log(2_000_000)  # Geometric mean around $2M
        cost_std = 2.0
        params[:min_cost] = max(50_000, round(rand(LogNormal(cost_mean - cost_std, 0.7)), digits=0))
        params[:max_cost] = min(50_000_000, round(rand(LogNormal(cost_mean + cost_std, 0.7)), digits=0))
        
        # Returns: $100K-$100M, typically 1.5-8x cost with high variation
        return_multiplier = rand(Gamma(1.5, 2.5)) + 1.5  # Mean around 5.25x, heavy tail
        params[:min_return] = max(100_000, params[:min_cost] * rand(Uniform(0.6, 1.4)))
        params[:max_return] = min(100_000_000, params[:max_cost] * return_multiplier)
        
        # Budget factor: 0.1-0.4 (very constrained due to scale)
        params[:budget_factor] = rand(Beta(4, 6)) * 0.3 + 0.1
        
        # Risk parameters scaled for large operations
        params[:max_risk_score] = rand(Uniform(15.0, 25.0))
        params[:dependency_density] = rand(Beta(2, 5)) * 0.15 + 0.15  # Higher density for complex interdependencies
    end
    
    # Ensure min < max for costs and returns
    if params[:min_cost] >= params[:max_cost]
        params[:min_cost] = params[:max_cost] * 0.3
    end
    if params[:min_return] >= params[:max_return]
        params[:min_return] = params[:max_return] * 0.4
    end
    
    # Risk budget: Scale with number of projects and complexity
    risk_per_project = rand(Uniform(0.8, 2.5))
    params[:risk_budget] = params[:n_projects] * risk_per_project
    
    # Maximum high-risk projects: 10-30% of total projects
    high_risk_fraction = rand(Beta(2, 5)) * 0.2 + 0.1
    params[:max_high_risk_projects] = max(1, ceil(Int, params[:n_projects] * high_risk_fraction))
    
    # High risk threshold: 60-80% of maximum risk score
    params[:high_risk_threshold] = params[:max_risk_score] * rand(Uniform(0.6, 0.8))
    
    return params
end

function calculate_project_selection_variable_count(params::Dict)
    # Extract parameters with defaults
    n_projects = get(params, :n_projects, 15)
    
    # Variables: x[projects] - binary variables for project selection
    return n_projects
end

# Register the problem type
register_problem(
    :project_selection,
    generate_project_selection_problem,
    sample_project_selection_parameters,
    "Project selection problem that maximizes return by selecting a portfolio of projects subject to budget, risk, and dependency constraints. Uses realistic probability distributions for costs, returns, and risk scores that scale appropriately from small company (50-250 projects) to large enterprise (1000-10000 projects) scenarios."
)