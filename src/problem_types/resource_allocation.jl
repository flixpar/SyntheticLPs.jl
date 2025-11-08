using JuMP
using Random
using Statistics

"""
    ResourceAllocationProblem <: ProblemGenerator

Generator for resource allocation problems.

# Fields
- `n_activities::Int`: Number of activities
- `n_resources::Int`: Number of resource types
- `profits::Vector{Float64}`: Profit per unit of activity
- `usage::Matrix{Float64}`: Resource usage per unit of activity (n_activities × n_resources)
- `resources::Vector{Float64}`: Available amount of each resource
- `min_levels::Vector{Float64}`: Minimum level for each activity
"""
struct ResourceAllocationProblem <: ProblemGenerator
    n_activities::Int
    n_resources::Int
    profits::Vector{Float64}
    usage::Matrix{Float64}
    resources::Vector{Float64}
    min_levels::Vector{Float64}
end

"""
    ResourceAllocationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a resource allocation problem instance.

# Arguments
- `target_variables`: Target number of variables (activities)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function ResourceAllocationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # For resource allocation, variables = n_activities
    n_activities = max(3, min(2000, target_variables))
    n_resources = rand(2:50)
    min_resource = rand(50:1000)
    max_resource = rand(200:10000)
    min_profit = rand(0.1:0.1:2.0)
    max_profit = rand(5.0:5.0:100.0)
    min_usage = rand(0.01:0.01:0.5)
    max_usage = rand(1.0:1.0:20.0)
    correlation_strength = rand(0.5:0.1:0.9)
    add_min_constraints = rand() < 0.7
    min_level_prob = rand(0.2:0.1:0.5)

    # Override for infeasible case
    solution_status = feasibility_status == feasible ? :feasible :
                     feasibility_status == infeasible ? :infeasible : :all
    if solution_status == :infeasible
        add_min_constraints = true
    end

    # Generate quality factors
    quality_factors = rand(n_activities)

    # Generate profits correlated with quality
    base_profit = rand(n_activities) .* (max_profit - min_profit) .+ min_profit
    quality_profit = quality_factors .* (max_profit - min_profit)
    profits = base_profit + correlation_strength * quality_profit

    # Generate usage matrix
    usage = zeros(n_activities, n_resources)

    for i in 1:n_activities
        for j in 1:n_resources
            base_usage = rand() * (max_usage - min_usage) + min_usage
            quality_usage = quality_factors[i] * (max_usage - min_usage)
            usage[i, j] = base_usage + correlation_strength * quality_usage
        end
    end

    # Helper functions
    function compute_demand_plan(profits_vec::Vector{Float64}, usage_mat::Array{Float64,2}, quality::Vector{Float64})
        demand = zeros(n_activities)
        for i in 1:n_activities
            avg_u = mean(view(usage_mat, i, :)) + 1e-9
            hetero = 0.6 + 0.6 * quality[i]
            demand[i] = hetero * profits_vec[i] / avg_u
        end
        s = 1.0 / (mean(demand) + 1e-9)
        return demand .* s
    end

    function compute_single_activity_caps(usage_mat::Array{Float64,2}, resource_caps::Vector{Float64})
        caps = zeros(n_activities)
        for i in 1:n_activities
            local_caps = Float64[]
            for j in 1:n_resources
                push!(local_caps, resource_caps[j] / usage_mat[i, j])
            end
            caps[i] = minimum(local_caps)
        end
        return caps
    end

    # Initialize outputs
    resources = zeros(n_resources)
    min_levels = zeros(n_activities)
    baseline_x = zeros(n_activities)

    if solution_status == :all
        # Original stochastic construction
        expected_usage = sum(usage, dims=1) / n_activities
        resources = vec(expected_usage) .* rand(n_resources) .* n_activities ./ 2
        resources = max.(resources, min_resource)
        resources = min.(resources, max_resource)

        if add_min_constraints
            for i in 1:n_activities
                if rand() < min_level_prob
                    max_possible = minimum([resources[j] / usage[i, j] for j in 1:n_resources])
                    min_levels[i] = rand(0.1:0.05:0.3) * max_possible
                end
            end
        end
    else
        # Constructive generation with plan-driven floors
        demand_x = compute_demand_plan(profits, usage, quality_factors)
        cons = vec(sum(usage .* demand_x, dims=1))

        for j in 1:n_resources
            if solution_status == :feasible
                capacity_anchor_theta = 1.10 + 0.40 * rand()
            else
                capacity_anchor_theta = 0.95 + 0.15 * rand()
            end
            resources[j] = cons[j] * capacity_anchor_theta
            resources[j] = max(resources[j], min_resource)
            resources[j] = min(resources[j], max_resource)
        end

        cons_post = vec(sum(usage .* demand_x, dims=1))
        scale_to_fit = minimum([resources[j] / max(cons_post[j], 1e-12) for j in 1:n_resources])
        fill_fraction = solution_status == :feasible ? (0.75 + 0.2 * rand()) : (0.7 + 0.2 * rand())
        baseline_x = demand_x .* max(scale_to_fit * fill_fraction, 0.0)

        if add_min_constraints
            selected = falses(n_activities)
            for i in 1:n_activities
                if rand() < min_level_prob && baseline_x[i] > 0
                    selected[i] = true
                    min_levels[i] = rand(0.1:0.05:0.3) * baseline_x[i]
                end
            end

            if count(selected) == 0
                idxs = findall(x -> x > 0, baseline_x)
                if !isempty(idxs)
                    pick = rand(idxs)
                    selected[pick] = true
                    min_levels[pick] = rand(0.1:0.05:0.3) * baseline_x[pick]
                end
            end

            if solution_status == :infeasible && count(selected) < 2
                idxs = setdiff(collect(1:n_activities), findall(selected))
                idxs = [i for i in idxs if baseline_x[i] > 0]
                if !isempty(idxs)
                    pick2 = rand(idxs)
                    selected[pick2] = true
                    min_levels[pick2] = rand(0.1:0.05:0.3) * baseline_x[pick2]
                end
            end
        end

        if solution_status == :feasible
            # Feasibility guaranteed
        else
            # Infeasible: Create structured violations
            xi_cap = compute_single_activity_caps(usage, resources)

            for i in 1:n_activities
                if min_levels[i] > 0
                    min_levels[i] = min(min_levels[i], 0.8 * xi_cap[i])
                end
            end

            k_viol = n_resources == 1 ? 1 : (rand() < 0.5 ? 1 : (rand() < 0.6 ? min(2, n_resources) : min(3, n_resources)))
            min_cons = [sum(usage[i, j] * min_levels[i] for i in 1:n_activities) for j in 1:n_resources]
            ratios = [min_cons[j] / max(resources[j], eps()) for j in 1:n_resources]
            order_r = sortperm(ratios; rev=true)
            violated_resources = order_r[1:k_viol]
            deltas = [0.05 + 0.20 * rand() for _ in 1:k_viol]
            targets = [resources[r] * (1.0 + deltas[idx]) for (idx, r) in enumerate(violated_resources)]

            S = findall(i -> min_levels[i] > 0, 1:n_activities)
            if length(S) < 3
                scores = [sum(usage[i, r] for r in violated_resources) for i in 1:n_activities]
                add_order = sortperm(scores; rev=true)
                for t in 1:min(5, n_activities)
                    if !(add_order[t] in S)
                        push!(S, add_order[t])
                    end
                end
            end
            S = unique(S)
            if isempty(S)
                order = sortperm(xi_cap; rev=true)
                take = min(2, length(order))
                for k in 1:take
                    i = order[k]
                    min_levels[i] = 0.2 * xi_cap[i]
                    push!(S, i)
                end
            end

            residual = [targets[idx] - sum(usage[i, r] * min_levels[i] for i in 1:n_activities) for (idx, r) in enumerate(violated_resources)]
            iter_guard = 0
            while any(residual .> 1e-9) && iter_guard < 3 * n_activities
                best_i = 0
                best_gain = 0.0
                best_inc = 0.0
                for i in S
                    cap_inc = max(0.0, 0.95 * xi_cap[i] - min_levels[i])
                    if cap_inc <= 0
                        continue
                    end
                    inc_bound = Inf
                    weighted_gain = 0.0
                    for (idx, r) in enumerate(violated_resources)
                        if residual[idx] > 0 && usage[i, r] > 0
                            inc_bound = min(inc_bound, residual[idx] / usage[i, r])
                            weighted_gain += usage[i, r] * residual[idx]
                        end
                    end
                    inc = min(cap_inc, isfinite(inc_bound) ? inc_bound : 0.0)
                    if inc > 0 && weighted_gain * inc > best_gain
                        best_gain = weighted_gain * inc
                        best_i = i
                        best_inc = inc
                    end
                end
                if best_i == 0 || best_inc <= 0
                    candidates = setdiff(collect(1:n_activities), S)
                    candidates = [i for i in candidates if 0.95 * xi_cap[i] > min_levels[i]]
                    if isempty(candidates)
                        break
                    end
                    i_new = candidates[argmax(xi_cap[candidates])]
                    push!(S, i_new)
                    min_levels[i_new] = max(min_levels[i_new], 0.1 * xi_cap[i_new])
                else
                    min_levels[best_i] += best_inc
                    for (idx, r) in enumerate(violated_resources)
                        residual[idx] -= best_inc * usage[best_i, r]
                    end
                end
                iter_guard += 1
            end

            # Final safety check
            for (idx, r) in enumerate(violated_resources)
                final_total = sum(usage[i, r] * min_levels[i] for i in 1:n_activities)
                if !(final_total > resources[r])
                    new_cap = max(min_resource, final_total * 0.95)
                    if new_cap < resources[r]
                        resources[r] = new_cap
                    else
                        bump = 0.05
                        for i in 1:n_activities
                            if 0.95 * xi_cap[i] > min_levels[i]
                                min_levels[i] = min(min_levels[i] * (1.0 + bump), 0.95 * xi_cap[i])
                            end
                        end
                    end
                end
            end

            # Recheck and force if needed
            for (idx, r) in enumerate(violated_resources)
                final_total = sum(usage[i, r] * min_levels[i] for i in 1:n_activities)
                if !(final_total > resources[r])
                    resources[r] = max(min_resource, min(resources[r], final_total * 0.98))
                end
            end
        end
    end

    return ResourceAllocationProblem(n_activities, n_resources, profits, usage, resources, min_levels)
end

"""
    build_model(prob::ResourceAllocationProblem)

Build a JuMP model for the resource allocation problem.

# Arguments
- `prob`: ResourceAllocationProblem instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::ResourceAllocationProblem)
    model = Model()

    # Decision variables
    @variable(model, x[1:prob.n_activities] >= 0)

    # Resource constraints
    for j in 1:prob.n_resources
        @constraint(model, sum(prob.usage[i,j] * x[i] for i in 1:prob.n_activities) <= prob.resources[j])
    end

    # Minimum level constraints
    for i in 1:prob.n_activities
        if prob.min_levels[i] > 0
            @constraint(model, x[i] >= prob.min_levels[i])
        end
    end

    # Objective
    @objective(model, Max, sum(prob.profits[i] * x[i] for i in 1:prob.n_activities))

    return model
end

# Register the problem type
register_problem(
    :resource_allocation,
    ResourceAllocationProblem,
    "Resource allocation problem that maximizes profit by allocating limited resources to competing activities"
)
