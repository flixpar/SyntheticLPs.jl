using JuMP
using Random
using Distributions
using Statistics

"""
    InventoryProblem <: ProblemGenerator

Generator for inventory control problems with realistic and diverse patterns, combining richer scenario generation with precise feasibility control.

# Fields
- `n_periods::Int`: Number of time periods
- `prod_capacity::Int`: Production capacity per period
- `initial_inventory::Int`: Starting inventory level
- `backlog_allowed::Bool`: Whether backorders are permitted
- `demands::Vector{Int}`: Demand for each period
- `production_costs::Vector{Float64}`: Production cost per period
- `holding_costs::Vector{Float64}`: Holding cost per period
- `backlog_costs::Vector{Float64}`: Backlog/shortage cost per period
"""
struct InventoryProblem <: ProblemGenerator
    n_periods::Int
    prod_capacity::Int
    initial_inventory::Int
    backlog_allowed::Bool
    demands::Vector{Int}
    production_costs::Vector{Float64}
    holding_costs::Vector{Float64}
    backlog_costs::Vector{Float64}
end

"""
    InventoryProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct an inventory control problem instance with sophisticated feasibility control.

# Arguments
- `target_variables`: Target number of variables
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function InventoryProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Determine business scale by target size
    scale = target_variables <= 250 ? :small :
            target_variables <= 1000 ? :medium : :large

    # Backlog incidence increases with scale
    backlog_prob = scale == :small ? 0.2 : (scale == :medium ? 0.4 : 0.6)
    backlog_allowed = rand(Bernoulli(backlog_prob))

    # Choose n_periods from variable-budget formula
    if backlog_allowed
        # x[1:T], I_plus[0:T], I_minus[0:T] => 3T + 2
        n_periods = max(2, min(5000, round(Int, (target_variables - 2) / 3)))
    else
        # x[1:T], I[0:T] => 2T + 1
        n_periods = max(2, min(5000, round(Int, (target_variables - 1) / 2)))
    end

    # Quick iterative refinement for variable count
    for _ in 1:10
        current = backlog_allowed ? (3*n_periods + 2) : (2*n_periods + 1)
        if abs(current - target_variables) / target_variables < 0.1
            break
        end
        if current < target_variables
            n_periods = min(5000, n_periods + 1)
        else
            n_periods = max(2, n_periods - 1)
        end
    end

    # Scale-specific ranges
    if scale == :small
        prod_capacity = round(Int, rand(Uniform(50, 500)))
        demand_base = round(Int, rand(Uniform(10, 100)))
        demand_vol = rand(Uniform(0.2, 0.5))
        demand_min = max(1, round(Int, demand_base * (1 - demand_vol)))
        demand_max = round(Int, demand_base * (1 + demand_vol))

        avgd = (demand_min + demand_max) / 2
        initial_inventory = round(Int, avgd * rand(Uniform(0.1, 0.5)))

        prod_cost_base = rand(Uniform(10, 100))
        prod_cost_spread = rand(Uniform(0.1, 0.3))
        prod_cost_min = round(Int, prod_cost_base * (1 - prod_cost_spread))
        prod_cost_max = round(Int, prod_cost_base * (1 + prod_cost_spread))

        holding_rate = rand(Uniform(0.05, 0.25)) / 12
        holding_cost_min = max(0.01, round(prod_cost_base * holding_rate * 0.8, digits=2))
        holding_cost_max = round(prod_cost_base * holding_rate * 1.2, digits=2)

    elseif scale == :medium
        prod_capacity = round(Int, rand(Uniform(200, 2000)))
        demand_base = round(Int, rand(Uniform(50, 1000)))
        demand_vol = rand(Uniform(0.15, 0.4))
        demand_min = max(1, round(Int, demand_base * (1 - demand_vol)))
        demand_max = round(Int, demand_base * (1 + demand_vol))

        avgd = (demand_min + demand_max) / 2
        initial_inventory = round(Int, avgd * rand(Uniform(0.05, 0.4)))

        prod_cost_base = rand(Uniform(5, 200))
        prod_cost_spread = rand(Uniform(0.05, 0.25))
        prod_cost_min = round(Int, prod_cost_base * (1 - prod_cost_spread))
        prod_cost_max = round(Int, prod_cost_base * (1 + prod_cost_spread))

        holding_rate = rand(Uniform(0.03, 0.20)) / 12
        holding_cost_min = max(0.01, round(prod_cost_base * holding_rate * 0.8, digits=2))
        holding_cost_max = round(prod_cost_base * holding_rate * 1.2, digits=2)

    else # :large
        prod_capacity = round(Int, rand(Uniform(1000, 50000)))
        demand_base = round(Int, rand(Uniform(100, 10000)))
        demand_vol = rand(Uniform(0.1, 0.3))
        demand_min = max(1, round(Int, demand_base * (1 - demand_vol)))
        demand_max = round(Int, demand_base * (1 + demand_vol))

        avgd = (demand_min + demand_max) / 2
        initial_inventory = round(Int, avgd * rand(Uniform(0.02, 0.3)))

        prod_cost_base = rand(Uniform(1, 500))
        prod_cost_spread = rand(Uniform(0.02, 0.20))
        prod_cost_min = round(Int, prod_cost_base * (1 - prod_cost_spread))
        prod_cost_max = round(Int, prod_cost_base * (1 + prod_cost_spread))

        holding_rate = rand(Uniform(0.01, 0.15)) / 12
        holding_cost_min = max(0.01, round(prod_cost_base * holding_rate * 0.8, digits=2))
        holding_cost_max = round(prod_cost_base * holding_rate * 1.2, digits=2)
    end

    backlog_cost_factor = rand(Uniform(1.5, 5.0))

    # Base stochastic series with seasonality & trends
    demand_mean = (demand_min + demand_max) / 2
    demand_std = (demand_max - demand_min) / 4
    base_demands = rand(Normal(demand_mean, demand_std), n_periods)
    demands = round.(Int, clamp.(base_demands, demand_min, demand_max))

    # Production & holding costs with mild dispersion and optional trends
    prod_cost_mean = (prod_cost_min + prod_cost_max) / 2
    prod_cost_std = (prod_cost_max - prod_cost_min) / 4
    production_costs = clamp.(rand(Normal(prod_cost_mean, prod_cost_std), n_periods),
                              prod_cost_min, prod_cost_max)

    holding_cost_mean = (holding_cost_min + holding_cost_max) / 2
    holding_cost_std = (holding_cost_max - holding_cost_min) / 4
    holding_costs = clamp.(rand(Normal(holding_cost_mean, holding_cost_std), n_periods),
                           holding_cost_min, holding_cost_max)

    # Seasonality patterns
    if rand() < 0.6
        if n_periods >= 12
            annual = 1.0 .+ 0.2 * sin.(2π .* (1:n_periods) ./ 12)
            demands = round.(Int, demands .* annual)
        end
        if n_periods >= 52
            weekly = 1.0 .+ 0.1 * sin.(2π .* (1:n_periods) ./ 7)
            demands = round.(Int, demands .* weekly)
        end
        if n_periods >= 24
            quarterly = 1.0 .+ 0.15 * sin.(2π .* (1:n_periods) ./ (n_periods/4))
            demands = round.(Int, demands .* quarterly)
        end
    end

    # Cost trends
    if rand() < 0.4
        dir = rand() < 0.7 ? 1 : -1
        strength = rand(Uniform(0.001, 0.01))
        trend = [exp(dir * strength * t) for t in 1:n_periods]
        production_costs = production_costs .* trend
    end
    if rand() < 0.3
        dir = rand() < 0.6 ? 1 : -1
        strength = rand(Uniform(0.0005, 0.005))
        trend = [exp(dir * strength * t) for t in 1:n_periods]
        holding_costs = holding_costs .* trend
    end

    # Occasional demand disruptions
    if rand() < 0.2
        n_disruptions = rand(Poisson(max(1, n_periods ÷ 20)))
        for _ in 1:n_disruptions
            t = rand(1:n_periods)
            f = rand() < 0.5 ? rand(Uniform(0.3, 0.7)) : rand(Uniform(1.4, 2.0))
            demands[t] = round(Int, demands[t] * f)
        end
    end

    # Keep demands in reasonable bounds
    demands = max.(demands, max(1, demand_min ÷ 2))
    demands = min.(demands, demand_max * 2)

    # Backlog costs
    backlog_costs = production_costs .* backlog_cost_factor

    # Cumulative demand helper
    function cum(x)
        c = similar(x)
        s = zero(eltype(x))
        @inbounds for i in eachindex(x)
            s += x[i]
            c[i] = s
        end
        return c
    end
    cum_demands = cum(demands)

    # Max prefix shortfall helper
    function max_shortfall(cap::Int)
        max_sf = 0
        @inbounds for t in 1:n_periods
            sf = cum_demands[t] - (initial_inventory + t * cap)
            if sf > max_sf
                max_sf = sf
            end
        end
        return max_sf
    end

    # ENFORCE FEASIBILITY/INFEASIBILITY with sophisticated logic
    solution_status = feasibility_status == feasible ? :feasible :
                     feasibility_status == infeasible ? :infeasible : :all

    if solution_status == :feasible
        # Realistic operator-side actions
        if !backlog_allowed
            r = rand()
            if r < 0.30
                prod_capacity = round(Int, prod_capacity * rand(Uniform(1.10, 1.25)))
            elseif r < 0.60
                avgd = mean(demands)
                ss = round(Int, avgd * rand(Uniform(0.20, 0.40)))
                initial_inventory = max(initial_inventory, ss)
            elseif r < 0.80
                backlog_allowed = true
            else
                thr = quantile(demands, 0.90)
                for t in 1:n_periods
                    if demands[t] > thr
                        excess = demands[t] - round(Int, thr)
                        demands[t] = round(Int, thr)
                        if t > 1
                            demands[t-1] += round(Int, excess * 0.3)
                        end
                        if t < n_periods
                            demands[t+1] += round(Int, excess * 0.3)
                        end
                        if rand() < 0.5
                            prod_capacity = round(Int, prod_capacity * 1.05)
                        end
                    end
                end
                demands = max.(demands, 1)
                demands = min.(demands, demand_max * 2)
                cum_demands = cum(demands)
            end
        end

        # Surgical feasibility pass
        if !backlog_allowed
            sf = max_shortfall(prod_capacity)
            if sf > 0
                required_caps = [ceil(Int, max(0, cum_demands[t] - initial_inventory) / t) for t in 1:n_periods]
                min_cap_needed = maximum(required_caps)
                extra_inv_needed = sf

                uplift_ratio = min_cap_needed > 0 ? min_cap_needed / max(1, prod_capacity) : 1.0
                if uplift_ratio <= 1.5 && rand() < 0.6
                    prod_capacity = max(prod_capacity, min_cap_needed)
                elseif extra_inv_needed <= round(Int, max(10, 2 * (demand_min + demand_max) / 2)) && rand() < 0.5
                    initial_inventory += extra_inv_needed
                else
                    backlog_allowed = true
                end
            end
        end

    elseif solution_status == :infeasible
        # Disallow backlog
        if backlog_allowed
            backlog_allowed = false
        end

        # Create diverse causes of trouble
        scenario = rand()
        if scenario < 0.25
            # Sustained high demand
            start = rand(1:max(1, n_periods - 3))
            dur = min(rand(2:4), n_periods - start + 1)
            surge = rand(Uniform(1.5, 2.0))
            for t in start:start+dur-1
                demands[t] = round(Int, demands[t] * surge)
            end
        elseif scenario < 0.50
            # Capacity cut + lower starting stock
            prod_capacity = round(Int, prod_capacity * rand(Uniform(0.6, 0.8)))
            initial_inventory = round(Int, max(0, initial_inventory * 0.5))
        elseif scenario < 0.75
            # Supplier disruptions
            ndis = rand(1:min(3, n_periods ÷ 4))
            for _ in 1:ndis
                tp = rand(1:n_periods)
                sev = rand(Uniform(0.3, 0.6))
                if tp <= n_periods ÷ 2
                    for t in tp:n_periods
                        demands[t] = round(Int, demands[t] * rand(Uniform(1.1, 1.3)))
                    end
                else
                    demands[tp] = round(Int, demands[tp] / sev)
                end
            end
            initial_inventory = round(Int, max(0, initial_inventory * 0.3))
        else
            # Very low starting stock + high variability
            initial_inventory = max(1, round(Int, initial_inventory * 0.1))
            for t in 1:n_periods
                demands[t] = round(Int, demands[t] * rand(Uniform(0.7, 1.4)))
            end
            crisis = rand(ceil(Int, n_periods/3):n_periods)
            demands[crisis] = max(demands[crisis], round(Int, prod_capacity * rand(Uniform(1.2, 1.5))))
        end

        demands = max.(demands, 1)
        demands = min.(demands, demand_max * 2)
        cum_demands = cum(demands)

        # Guarantee prefix infeasibility
        sf = max_shortfall(prod_capacity)
        if sf <= 0
            required_caps = [ceil(Int, max(0, cum_demands[t] - initial_inventory) / t) for t in 1:n_periods]
            min_cap_needed = maximum(required_caps)
            margin = rand(1:max(1, round(Int, 0.1 * max(1, min_cap_needed))))
            new_cap = max(0, min_cap_needed - margin)
            if max_shortfall(new_cap) <= 0
                new_cap = max(0, min_cap_needed - 1)
            end
            prod_capacity = new_cap
        end
    end

    return InventoryProblem(n_periods, prod_capacity, initial_inventory, backlog_allowed,
                            demands, production_costs, holding_costs, backlog_costs)
end

"""
    build_model(prob::InventoryProblem)

Build a JuMP model for the inventory control problem.

# Arguments
- `prob`: InventoryProblem instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::InventoryProblem)
    model = Model()

    if prob.backlog_allowed
        # With backlogging
        @variable(model, x[1:prob.n_periods] >= 0)
        @variable(model, I_plus[0:prob.n_periods] >= 0)
        @variable(model, I_minus[0:prob.n_periods] >= 0)

        @objective(model, Min,
            sum(prob.production_costs[t]*x[t] + prob.holding_costs[t]*I_plus[t] + prob.backlog_costs[t]*I_minus[t]
                for t in 1:prob.n_periods))

        @constraint(model, I_plus[0] == prob.initial_inventory)
        @constraint(model, I_minus[0] == 0)

        for t in 1:prob.n_periods
            @constraint(model, I_plus[t-1] - I_minus[t-1] + x[t] - prob.demands[t] == I_plus[t] - I_minus[t])
            @constraint(model, x[t] <= prob.prod_capacity)
        end
    else
        # No backlogging
        @variable(model, x[1:prob.n_periods] >= 0)
        @variable(model, I[0:prob.n_periods] >= 0)

        @objective(model, Min,
            sum(prob.production_costs[t]*x[t] + prob.holding_costs[t]*I[t] for t in 1:prob.n_periods))

        @constraint(model, I[0] == prob.initial_inventory)

        for t in 1:prob.n_periods
            @constraint(model, I[t-1] + x[t] - prob.demands[t] == I[t])
            @constraint(model, x[t] <= prob.prod_capacity)
        end
    end

    return model
end

# Register the problem type
register_problem(
    :inventory,
    InventoryProblem,
    "Inventory control problem that minimizes production and holding costs while meeting demand over multiple periods"
)
