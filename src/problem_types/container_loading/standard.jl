using JuMP
using Random

"""
    ContainerLoadingProblem <: ProblemGenerator

Multi-container loading with binary item assignments, container-use decisions,
and three independent capacity dimensions.
"""
struct ContainerLoadingProblem <: ProblemGenerator
    n_items::Int
    n_containers::Int
    item_requirements::Matrix{Float64}
    capacities::Matrix{Float64}
end

function ContainerLoadingProblem(
    target_variables::Int,
    feasibility_status::FeasibilityStatus,
    seed::Int,
)
    rng = MersenneTwister(seed)
    # Variable count is n_items * n_containers + n_containers. The smallest
    # formulation keeps at least two items and two containers (6 variables);
    # public APIs accept sizes down to 2, so clamp rather than reject.
    target = max(target_variables, 6)

    best = (typemax(Int), 0, 0)
    for b in 2:max(2, min(20, target ÷ 3))
        n = max(b, round(Int, (target - b) / b))
        error = abs(n * b + b - target)
        error < best[1] && (best = (error, n, b))
    end
    _, n_items, n_containers = best
    requirements = Float64[rand(rng, 3:20) for _ in 1:3, _ in 1:n_items]

    planted_bin = [mod(i - 1, n_containers) + 1 for i in randperm(rng, n_items)]
    capacities = zeros(Float64, 3, n_containers)
    for d in 1:3, b in 1:n_containers
        load = sum(requirements[d, i] for i in 1:n_items if planted_bin[i] == b)
        capacities[d, b] = max(maximum(requirements[d, :]), 1.15 * load + 1.0)
    end
    if feasibility_status == infeasible
        # Summing one capacity dimension over containers contradicts the total
        # demand implied by exact item assignment, even after relaxation.
        total = sum(requirements[1, :])
        scale = 0.8 * total / sum(capacities[1, :])
        capacities[1, :] .*= scale
    end
    return ContainerLoadingProblem(n_items, n_containers, requirements, capacities)
end

function build_model(prob::ContainerLoadingProblem)
    model = Model()
    @variable(model, assign[1:prob.n_items, 1:prob.n_containers], Bin)
    @variable(model, used[1:prob.n_containers], Bin)
    @objective(model, Min, sum(used))
    for i in 1:prob.n_items
        @constraint(model, sum(assign[i, b] for b in 1:prob.n_containers) == 1)
        for b in 1:prob.n_containers
            @constraint(model, assign[i, b] <= used[b])
        end
    end
    for d in 1:3, b in 1:prob.n_containers
        @constraint(
            model,
            sum(prob.item_requirements[d, i] * assign[i, b] for i in 1:prob.n_items) <=
            prob.capacities[d, b] * used[b],
        )
    end
    return model
end

register_variant(
    :container_loading,
    :standard,
    ContainerLoadingProblem,
    "Three-resource binary item-to-container loading with container activation",
    default=true,
)
