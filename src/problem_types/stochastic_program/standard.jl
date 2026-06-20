using JuMP
using Random
using Distributions

"""
    StochasticProgramProblem <: ProblemGenerator

Generator for two-stage stochastic linear programs with recourse (a stochastic
capacity / production–distribution problem).

# Overview
Models a two-stage decision under demand uncertainty. In the **first stage**, a
capacity `x[i]` is committed at each facility `i` before demand is known, subject
to per-facility bounds and a shared first-stage resource budget. In the **second
stage**, after a demand scenario `s` is realized, goods are shipped from
facilities to customers (`y[i, j, s]`) and any unmet demand is absorbed by a
penalized shortfall (`z[j, s]`) — i.e. the model has complete recourse, so the
second stage is always feasible for any first-stage decision. The objective
minimizes first-stage cost plus the expected second-stage shipping and shortfall
cost over the scenarios.

This yields the canonical **dual block-angular** constraint structure of
two-stage stochastic programming: one first-stage block coupled to `S`
independent scenario blocks through the capacity-linking constraints. It is the
prototypical instance class for decomposition methods (L-shaped / Benders).

Feasibility is controlled entirely through the first stage: the problem is
infeasible exactly when the minimum committed capacity violates the first-stage
resource budget.

# Fields
- `n_facilities::Int`: Number of facilities (first-stage decisions)
- `n_customers::Int`: Number of customers (second-stage demand points)
- `n_scenarios::Int`: Number of demand scenarios
- `build_cost::Vector{Float64}`: First-stage cost per unit capacity at each facility
- `resource_use::Vector{Float64}`: First-stage resource consumed per unit capacity
- `resource_budget::Float64`: Total first-stage resource available
- `capacity_min::Vector{Float64}`: Minimum capacity that must be committed per facility
- `capacity_max::Vector{Float64}`: Maximum capacity per facility
- `ship_cost::Matrix{Float64}`: Second-stage shipping cost (n_facilities × n_customers)
- `shortfall_cost::Vector{Float64}`: Penalty per unit of unmet demand per customer
- `scenario_prob::Vector{Float64}`: Probability of each scenario (sums to 1)
- `demand::Matrix{Float64}`: Demand per customer per scenario (n_customers × n_scenarios)
"""
struct StochasticProgramProblem <: ProblemGenerator
    n_facilities::Int
    n_customers::Int
    n_scenarios::Int
    build_cost::Vector{Float64}
    resource_use::Vector{Float64}
    resource_budget::Float64
    capacity_min::Vector{Float64}
    capacity_max::Vector{Float64}
    ship_cost::Matrix{Float64}
    shortfall_cost::Vector{Float64}
    scenario_prob::Vector{Float64}
    demand::Matrix{Float64}
end

"""
    StochasticProgramProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a two-stage stochastic program instance.

Variables: `x[i]` (first stage) plus `y[i, j, s]` and `z[j, s]` (second stage),
for a total of `n_facilities + n_scenarios * (n_facilities * n_customers + n_customers)`.

# Arguments
- `target_variables`: Target number of variables
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function StochasticProgramProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # --- Dimension sizing ---
    # Variable count is dominated by the scenario blocks: S * (I*J + J). Choose
    # small facility/customer counts first, then set the scenario count last to
    # hit the target (so rounding error is not amplified by S).
    if target_variables < 200
        n_facilities = rand(2:4)
        n_customers = rand(2:5)
    elseif target_variables < 1000
        n_facilities = rand(3:6)
        n_customers = rand(4:8)
    else
        n_facilities = rand(5:10)
        n_customers = rand(6:12)
    end

    I = n_facilities
    J = n_customers
    block = I * J + J                                       # variables per scenario
    n_scenarios = max(2, round(Int, target_variables / block))
    S = n_scenarios

    # --- First-stage data ---
    build_cost = rand(Uniform(5.0, 20.0), I)
    resource_use = rand(Uniform(0.5, 2.0), I)
    capacity_min = rand(Uniform(1.0, 5.0), I)
    capacity_max = capacity_min .+ rand(Uniform(20.0, 60.0), I)

    required_resource = sum(resource_use .* capacity_min)

    # --- Second-stage data ---
    ship_cost = rand(Uniform(1.0, 10.0), I, J)
    # Shortfall must be more expensive than shipping so recourse prefers serving demand.
    shortfall_cost = [maximum(ship_cost[:, j]) * rand(Uniform(2.0, 5.0)) for j in 1:J]

    # Scenario probabilities (random, normalized).
    raw_p = rand(Uniform(0.5, 1.5), S)
    scenario_prob = raw_p ./ sum(raw_p)

    # Per-customer base demand, perturbed per scenario (log-normal multiplier).
    base_demand = rand(Uniform(5.0, 25.0), J)
    demand = zeros(Float64, J, S)
    for s in 1:S, j in 1:J
        demand[j, s] = base_demand[j] * rand(LogNormal(0.0, 0.35))
    end

    # --- Feasibility handling (controlled by the first stage only) ---
    actual_status = feasibility_status
    if feasibility_status == unknown
        actual_status = rand() < 0.7 ? feasible : infeasible
    end

    if actual_status == feasible
        resource_budget = required_resource * rand(Uniform(1.2, 2.0))
    elseif actual_status == infeasible
        # Budget strictly below the minimum required => no feasible first stage.
        resource_budget = required_resource * rand(Uniform(0.6, 0.9))
    else
        resource_budget = required_resource * rand(Uniform(0.7, 2.0))
    end

    return StochasticProgramProblem(
        I, J, S, build_cost, resource_use, resource_budget,
        capacity_min, capacity_max, ship_cost, shortfall_cost,
        scenario_prob, demand,
    )
end

"""
    build_model(prob::StochasticProgramProblem)

Build a JuMP model for the two-stage stochastic program. Deterministic — uses
only data from the struct fields.

# Returns
- `model`: The JuMP model
"""
function build_model(prob::StochasticProgramProblem)
    model = Model()

    I = prob.n_facilities
    J = prob.n_customers
    S = prob.n_scenarios

    # First-stage variables: committed capacity, with per-facility bounds.
    @variable(model, prob.capacity_min[i] <= x[i in 1:I] <= prob.capacity_max[i])

    # Second-stage variables: shipments and unmet-demand shortfalls per scenario.
    @variable(model, y[1:I, 1:J, 1:S] >= 0)
    @variable(model, z[1:J, 1:S] >= 0)

    # Objective: first-stage cost + expected second-stage cost.
    @objective(model, Min,
        sum(prob.build_cost[i] * x[i] for i in 1:I) +
        sum(prob.scenario_prob[s] * (
                sum(prob.ship_cost[i, j] * y[i, j, s] for i in 1:I, j in 1:J) +
                sum(prob.shortfall_cost[j] * z[j, s] for j in 1:J)
            ) for s in 1:S)
    )

    # First-stage resource budget (couples the first-stage decisions).
    @constraint(model, sum(prob.resource_use[i] * x[i] for i in 1:I) <= prob.resource_budget)

    # Scenario blocks.
    for s in 1:S
        # Capacity linking: shipments out of a facility cannot exceed its capacity.
        for i in 1:I
            @constraint(model, sum(y[i, j, s] for j in 1:J) <= x[i])
        end
        # Demand satisfaction (with penalized shortfall).
        for j in 1:J
            @constraint(model, sum(y[i, j, s] for i in 1:I) + z[j, s] >= prob.demand[j, s])
        end
    end

    return model
end

# Register the variant
register_variant(
    :stochastic_program,
    :standard,
    StochasticProgramProblem,
    "Two-stage stochastic program with recourse (stochastic capacity/distribution planning) with the canonical dual block-angular structure",
)
