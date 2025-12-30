using JuMP
using Random
using Distributions

"""
Production planning problem variants.

# Variants
- `prod_standard`: Basic production planning - maximize profit subject to resources
- `prod_multi_period`: Multi-period planning with inventory
- `prod_setup_costs`: Include fixed setup costs for production
- `prod_demand_constraints`: Must meet minimum demand levels
- `prod_overtime`: Allow overtime at higher cost
- `prod_quality_levels`: Products have quality grades with different yields
- `prod_machine_assignment`: Products require specific machines
- `prod_batch_size`: Production in fixed batch sizes
"""
@enum ProductionVariant begin
    prod_standard
    prod_multi_period
    prod_setup_costs
    prod_demand_constraints
    prod_overtime
    prod_quality_levels
    prod_machine_assignment
    prod_batch_size
end

"""
    ProductionPlanningProblem <: ProblemGenerator

Generator for production planning problems with multiple variants.
"""
struct ProductionPlanningProblem <: ProblemGenerator
    n_products::Int
    n_resources::Int
    profits::Vector{Float64}
    usage::Matrix{Float64}
    resources::Vector{Float64}
    variant::ProductionVariant
    # Multi-period variant
    n_periods::Int
    demands::Union{Matrix{Float64}, Nothing}
    holding_costs::Union{Vector{Float64}, Nothing}
    initial_inventory::Union{Vector{Float64}, Nothing}
    # Setup costs variant
    setup_costs::Union{Vector{Float64}, Nothing}
    min_production::Union{Vector{Float64}, Nothing}
    # Overtime variant
    regular_capacity::Union{Vector{Float64}, Nothing}
    overtime_capacity::Union{Vector{Float64}, Nothing}
    overtime_cost_multiplier::Float64
    # Quality levels variant
    n_quality_levels::Int
    quality_yields::Union{Matrix{Float64}, Nothing}
    quality_premiums::Union{Matrix{Float64}, Nothing}
    # Machine assignment variant
    n_machines::Int
    machine_product_compat::Union{Matrix{Bool}, Nothing}
    machine_capacities::Union{Vector{Float64}, Nothing}
    # Batch size variant
    batch_sizes::Union{Vector{Int}, Nothing}
end

# Backwards compatibility
function ProductionPlanningProblem(n_products::Int, n_resources::Int,
                                   profits::Vector{Int}, usage::Matrix{Float64},
                                   resources::Vector{Float64})
    ProductionPlanningProblem(
        n_products, n_resources, Float64.(profits), usage, resources, prod_standard,
        1, nothing, nothing, nothing, nothing, nothing,
        nothing, nothing, 1.0, 1, nothing, nothing,
        0, nothing, nothing, nothing
    )
end

"""
    ProductionPlanningProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int;
                              variant::ProductionVariant=prod_standard)

Construct a production planning problem instance with the specified variant.
"""
function ProductionPlanningProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int;
                                   variant::ProductionVariant=prod_standard)
    Random.seed!(seed)

    # Calculate dimensions
    n_products = max(2, min(2000, target_variables))
    n_resources = rand(max(1, n_products ÷ 10):max(2, min(50, n_products ÷ 2)))

    # Scale parameters
    if target_variables <= 100
        profit_range = (10.0, 200.0)
        usage_range = (0.1, 20.0)
    elseif target_variables <= 500
        profit_range = (5.0, 500.0)
        usage_range = (0.05, 30.0)
    else
        profit_range = (1.0, 1000.0)
        usage_range = (0.01, 50.0)
    end

    # Generate profits and usage
    profits = [rand(Uniform(profit_range...)) for _ in 1:n_products]
    usage = [rand(Uniform(usage_range...)) for _ in 1:n_products, _ in 1:n_resources]

    # Resource availability
    resource_factor = rand(Uniform(0.4, 0.8))
    resources = [sum(usage[:, j]) * resource_factor for j in 1:n_resources]

    # Initialize variant fields
    n_periods = 1
    demands = nothing
    holding_costs = nothing
    initial_inventory = nothing
    setup_costs = nothing
    min_production = nothing
    regular_capacity = nothing
    overtime_capacity = nothing
    overtime_cost_multiplier = 1.0
    n_quality_levels = 1
    quality_yields = nothing
    quality_premiums = nothing
    n_machines = 0
    machine_product_compat = nothing
    machine_capacities = nothing
    batch_sizes = nothing

    # Generate variant-specific data
    if variant == prod_multi_period
        n_periods = rand(4:min(12, max(4, target_variables ÷ n_products)))

        # Demand patterns with seasonality
        base_demand = [rand(Uniform(10, 100)) for _ in 1:n_products]
        demands = zeros(n_products, n_periods)
        for i in 1:n_products, t in 1:n_periods
            seasonal = 1.0 + 0.3 * sin(2π * t / n_periods)
            demands[i, t] = base_demand[i] * seasonal * rand(Uniform(0.8, 1.2))
        end

        holding_costs = profits .* rand(Uniform(0.05, 0.15), n_products)
        initial_inventory = base_demand .* rand(Uniform(0.0, 0.3), n_products)

    elseif variant == prod_setup_costs
        setup_costs = profits .* rand(Uniform(5.0, 20.0), n_products)
        min_production = [rand(Uniform(5, 20)) for _ in 1:n_products]

    elseif variant == prod_demand_constraints
        # Minimum demand that must be met
        max_possible = resources ./ maximum(usage, dims=1)[:]
        avg_possible = sum(max_possible) / n_products
        demands = Matrix{Float64}(undef, n_products, 1)
        for i in 1:n_products
            demands[i, 1] = rand(Uniform(0.1, 0.5)) * avg_possible
        end

    elseif variant == prod_overtime
        # Split resources into regular and overtime
        regular_capacity = resources .* rand(Uniform(0.6, 0.8))
        overtime_capacity = resources .* rand(Uniform(0.3, 0.5))
        overtime_cost_multiplier = rand(Uniform(1.3, 2.0))

    elseif variant == prod_quality_levels
        n_quality_levels = rand(2:min(4, max(2, n_products ÷ 5)))

        # Yield matrix: product × quality level
        quality_yields = zeros(n_products, n_quality_levels)
        for i in 1:n_products
            remaining = 1.0
            for q in 1:n_quality_levels-1
                quality_yields[i, q] = remaining * rand(Uniform(0.3, 0.5))
                remaining -= quality_yields[i, q]
            end
            quality_yields[i, n_quality_levels] = remaining
        end

        # Premium/discount by quality level
        quality_premiums = zeros(n_products, n_quality_levels)
        for i in 1:n_products
            for q in 1:n_quality_levels
                # Higher quality = higher price
                quality_premiums[i, q] = profits[i] * (0.5 + 0.5 * (n_quality_levels - q + 1) / n_quality_levels)
            end
        end

    elseif variant == prod_machine_assignment
        n_machines = rand(max(2, n_resources ÷ 2):max(3, n_resources))

        # Product-machine compatibility
        machine_product_compat = zeros(Bool, n_products, n_machines)
        for i in 1:n_products
            # Each product compatible with 1-3 machines
            n_compat = rand(1:min(3, n_machines))
            compat_machines = sample(1:n_machines, n_compat, replace=false)
            for m in compat_machines
                machine_product_compat[i, m] = true
            end
        end

        # Machine capacities
        machine_capacities = [rand(Uniform(50, 200)) for _ in 1:n_machines]

    elseif variant == prod_batch_size
        batch_sizes = [rand([5, 10, 20, 25, 50, 100]) for _ in 1:n_products]
    end

    # Handle feasibility
    if feasibility_status == infeasible
        if variant == prod_demand_constraints && demands !== nothing
            # Make demands too high
            demands .*= 3.0
        elseif variant == prod_machine_assignment
            # Remove some machine compatibilities
            for i in 1:n_products
                machine_product_compat[i, :] .= false
            end
            machine_product_compat[1, 1] = true  # Keep one to avoid empty problem
        else
            # Reduce resources
            resources .*= 0.1
        end
    elseif feasibility_status == feasible
        if variant == prod_demand_constraints && demands !== nothing
            # Ensure demands are achievable
            demands .*= 0.5
        end
        # Increase resource availability
        resources .*= 1.2
    end

    return ProductionPlanningProblem(
        n_products, n_resources, profits, usage, resources, variant,
        n_periods, demands, holding_costs, initial_inventory,
        setup_costs, min_production,
        regular_capacity, overtime_capacity, overtime_cost_multiplier,
        n_quality_levels, quality_yields, quality_premiums,
        n_machines, machine_product_compat, machine_capacities,
        batch_sizes
    )
end

"""
    build_model(prob::ProductionPlanningProblem)

Build a JuMP model for the production planning problem based on its variant.
"""
function build_model(prob::ProductionPlanningProblem)
    model = Model()

    if prob.variant == prod_standard
        @variable(model, x[1:prob.n_products] >= 0)
        @objective(model, Max, sum(prob.profits[i] * x[i] for i in 1:prob.n_products))

        for j in 1:prob.n_resources
            @constraint(model, sum(prob.usage[i, j] * x[i] for i in 1:prob.n_products) <= prob.resources[j])
        end

    elseif prob.variant == prod_multi_period
        @variable(model, x[1:prob.n_products, 1:prob.n_periods] >= 0)
        @variable(model, inventory[1:prob.n_products, 0:prob.n_periods] >= 0)

        # Maximize profit - holding costs
        @objective(model, Max,
            sum(prob.profits[i] * x[i, t] for i in 1:prob.n_products, t in 1:prob.n_periods) -
            sum(prob.holding_costs[i] * inventory[i, t] for i in 1:prob.n_products, t in 1:prob.n_periods))

        # Initial inventory
        for i in 1:prob.n_products
            @constraint(model, inventory[i, 0] == prob.initial_inventory[i])
        end

        # Inventory balance
        for i in 1:prob.n_products, t in 1:prob.n_periods
            @constraint(model, inventory[i, t-1] + x[i, t] - prob.demands[i, t] == inventory[i, t])
        end

        # Resource constraints per period
        for j in 1:prob.n_resources, t in 1:prob.n_periods
            @constraint(model, sum(prob.usage[i, j] * x[i, t] for i in 1:prob.n_products) <=
                              prob.resources[j])
        end

    elseif prob.variant == prod_setup_costs
        @variable(model, x[1:prob.n_products] >= 0)
        @variable(model, y[1:prob.n_products], Bin)  # Setup indicator

        # Profit - setup costs
        @objective(model, Max,
            sum(prob.profits[i] * x[i] for i in 1:prob.n_products) -
            sum(prob.setup_costs[i] * y[i] for i in 1:prob.n_products))

        for j in 1:prob.n_resources
            @constraint(model, sum(prob.usage[i, j] * x[i] for i in 1:prob.n_products) <= prob.resources[j])
        end

        # Link production to setup
        M = sum(prob.resources) / minimum(prob.usage)
        for i in 1:prob.n_products
            @constraint(model, x[i] <= M * y[i])
            @constraint(model, x[i] >= prob.min_production[i] * y[i])
        end

    elseif prob.variant == prod_demand_constraints
        @variable(model, x[1:prob.n_products] >= 0)

        @objective(model, Max, sum(prob.profits[i] * x[i] for i in 1:prob.n_products))

        for j in 1:prob.n_resources
            @constraint(model, sum(prob.usage[i, j] * x[i] for i in 1:prob.n_products) <= prob.resources[j])
        end

        # Demand constraints
        for i in 1:prob.n_products
            @constraint(model, x[i] >= prob.demands[i, 1])
        end

    elseif prob.variant == prod_overtime
        @variable(model, x[1:prob.n_products] >= 0)
        @variable(model, regular_usage[1:prob.n_resources] >= 0)
        @variable(model, overtime_usage[1:prob.n_resources] >= 0)

        # Profit minus overtime penalty
        avg_profit = mean(prob.profits)
        overtime_penalty = avg_profit * (prob.overtime_cost_multiplier - 1.0)

        @objective(model, Max,
            sum(prob.profits[i] * x[i] for i in 1:prob.n_products) -
            overtime_penalty * sum(overtime_usage[j] for j in 1:prob.n_resources))

        # Resource usage split
        for j in 1:prob.n_resources
            @constraint(model, sum(prob.usage[i, j] * x[i] for i in 1:prob.n_products) ==
                              regular_usage[j] + overtime_usage[j])
            @constraint(model, regular_usage[j] <= prob.regular_capacity[j])
            @constraint(model, overtime_usage[j] <= prob.overtime_capacity[j])
        end

    elseif prob.variant == prod_quality_levels
        @variable(model, x[1:prob.n_products] >= 0)  # Total production
        @variable(model, q[1:prob.n_products, 1:prob.n_quality_levels] >= 0)  # Quality output

        # Revenue from quality outputs
        @objective(model, Max,
            sum(prob.quality_premiums[i, lev] * q[i, lev]
                for i in 1:prob.n_products, lev in 1:prob.n_quality_levels))

        # Resource constraints on total production
        for j in 1:prob.n_resources
            @constraint(model, sum(prob.usage[i, j] * x[i] for i in 1:prob.n_products) <= prob.resources[j])
        end

        # Quality output from production
        for i in 1:prob.n_products, lev in 1:prob.n_quality_levels
            @constraint(model, q[i, lev] == prob.quality_yields[i, lev] * x[i])
        end

    elseif prob.variant == prod_machine_assignment
        @variable(model, x[1:prob.n_products, 1:prob.n_machines] >= 0)

        @objective(model, Max,
            sum(prob.profits[i] * x[i, m]
                for i in 1:prob.n_products, m in 1:prob.n_machines
                if prob.machine_product_compat[i, m]))

        # Resource constraints
        for j in 1:prob.n_resources
            @constraint(model,
                sum(prob.usage[i, j] * sum(x[i, m] for m in 1:prob.n_machines if prob.machine_product_compat[i, m])
                    for i in 1:prob.n_products) <= prob.resources[j])
        end

        # Machine capacity
        for m in 1:prob.n_machines
            @constraint(model,
                sum(x[i, m] for i in 1:prob.n_products if prob.machine_product_compat[i, m]) <=
                prob.machine_capacities[m])
        end

        # Only produce on compatible machines
        for i in 1:prob.n_products, m in 1:prob.n_machines
            if !prob.machine_product_compat[i, m]
                @constraint(model, x[i, m] == 0)
            end
        end

    elseif prob.variant == prod_batch_size
        # Number of batches (integer variable)
        @variable(model, n_batches[1:prob.n_products] >= 0, Int)
        @variable(model, x[1:prob.n_products] >= 0)  # Continuous production

        @objective(model, Max, sum(prob.profits[i] * x[i] for i in 1:prob.n_products))

        for j in 1:prob.n_resources
            @constraint(model, sum(prob.usage[i, j] * x[i] for i in 1:prob.n_products) <= prob.resources[j])
        end

        # Production = batch_size × n_batches
        for i in 1:prob.n_products
            @constraint(model, x[i] == prob.batch_sizes[i] * n_batches[i])
        end
    end

    return model
end

# Register the problem type
register_problem(
    :production_planning,
    ProductionPlanningProblem,
    "Production planning with variants including standard, multi-period, setup costs, demand constraints, overtime, quality levels, machine assignment, and batch sizing"
)
