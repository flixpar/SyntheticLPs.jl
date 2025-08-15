using JuMP
using Random
using Distributions
using Statistics

"""
    generate_inventory_problem(params::Dict=Dict(); seed::Int=0)

Generate an inventory control problem instance with realistic and diverse patterns.
This version combines richer scenario generation with precise feasibility control.

# Arguments
- `params`:
  - `:n_periods`::Int = 6
  - `:prod_capacity`::Int = 100
  - `:initial_inventory`::Int = 20
  - `:demand_min`::Int = 50
  - `:demand_max`::Int = 100
  - `:prod_cost_min`::Real = 10
  - `:prod_cost_max`::Real = 20
  - `:holding_cost_min`::Real = 1
  - `:holding_cost_max`::Real = 5
  - `:backlog_allowed`::Bool = false
  - `:backlog_cost_factor`::Real = 2.0
  - `:solution_status`::Symbol ∈ (:feasible, :infeasible, :all) = :feasible
- `seed`::Int = 0

# Returns
- `model`::JuMP.Model
- `params`::Dict{Symbol,Any} (all parameters used)
"""
function generate_inventory_problem(params::Dict=Dict(); seed::Int=0)
    # RNG
    Random.seed!(seed)

    # --- Extract parameters with defaults ---
    n_periods        = get(params, :n_periods, 6)
    prod_capacity    = get(params, :prod_capacity, 100)
    initial_inventory= get(params, :initial_inventory, 20)
    demand_min       = get(params, :demand_min, 50)
    demand_max       = get(params, :demand_max, 100)
    prod_cost_min    = get(params, :prod_cost_min, 10)
    prod_cost_max    = get(params, :prod_cost_max, 20)
    holding_cost_min = get(params, :holding_cost_min, 1)
    holding_cost_max = get(params, :holding_cost_max, 5)
    backlog_allowed  = get(params, :backlog_allowed, false)
    backlog_cost_factor = get(params, :backlog_cost_factor, 2.0)
    solution_status  = get(params, :solution_status, :feasible)
    if !(solution_status in (:feasible, :infeasible, :all))
        error("Invalid :solution_status=$(solution_status). Use :feasible, :infeasible, or :all")
    end

    # Track actual parameters used
    actual_params = Dict{Symbol,Any}(
        :n_periods => n_periods,
        :prod_capacity => prod_capacity,
        :initial_inventory => initial_inventory,
        :demand_min => demand_min,
        :demand_max => demand_max,
        :prod_cost_min => prod_cost_min,
        :prod_cost_max => prod_cost_max,
        :holding_cost_min => holding_cost_min,
        :holding_cost_max => holding_cost_max,
        :backlog_allowed => backlog_allowed,
        :backlog_cost_factor => backlog_cost_factor,
        :solution_status => solution_status
    )

    # --- Base stochastic series (demands/costs) with seasonality & trends ---
    demand_mean = (demand_min + demand_max) / 2
    demand_std  = (demand_max - demand_min) / 4          # ~95% within [min,max]
    base_demands = rand(Normal(demand_mean, demand_std), n_periods)
    demands = round.(Int, clamp.(base_demands, demand_min, demand_max))

    # production & holding costs with mild dispersion and optional trends
    prod_cost_mean = (prod_cost_min + prod_cost_max) / 2
    prod_cost_std  = (prod_cost_max - prod_cost_min) / 4
    production_costs = clamp.(rand(Normal(prod_cost_mean, prod_cost_std), n_periods),
                              prod_cost_min, prod_cost_max)

    holding_cost_mean = (holding_cost_min + holding_cost_max) / 2
    holding_cost_std  = (holding_cost_max - holding_cost_min) / 4
    holding_costs = clamp.(rand(Normal(holding_cost_mean, holding_cost_std), n_periods),
                           holding_cost_min, holding_cost_max)

    # Seasonality (annual/weekly/quarterly)
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
            t  = rand(1:n_periods)
            f  = rand() < 0.5 ? rand(Uniform(0.3, 0.7)) : rand(Uniform(1.4, 2.0))
            demands[t] = round(Int, demands[t] * f)
        end
    end

    # Keep demands in reasonable bounds
    demands = max.(demands, max(1, demand_min ÷ 2))
    demands = min.(demands, demand_max * 2)

    # Backlog costs (if used)
    backlog_costs = production_costs .* backlog_cost_factor

    # Store generated series
    actual_params[:demands] = demands
    actual_params[:production_costs] = production_costs
    actual_params[:holding_costs] = holding_costs
    actual_params[:backlog_costs] = backlog_costs

    # --- Helpers for precise feasibility logic ---
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

    # max prefix shortfall (no backlog) at capacity "cap"
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

    # --- Enforce requested feasibility while preserving realism ---
    if solution_status == :feasible
        # 1) Add realistic operator-side actions (small buffer, safety stock, smoothing)
        if !backlog_allowed
            r = rand()
            if r < 0.30
                # modest capacity buffer (typ. 10–25%)
                prod_capacity = round(Int, prod_capacity * rand(Uniform(1.10, 1.25)))
                actual_params[:prod_capacity] = prod_capacity
            elseif r < 0.60
                # safety stock (20–40% of avg demand)
                avgd = mean(demands)
                ss = round(Int, avgd * rand(Uniform(0.20, 0.40)))
                initial_inventory = max(initial_inventory, ss)
                actual_params[:initial_inventory] = initial_inventory
            elseif r < 0.80
                # allow backorders sometimes (realistic)
                backlog_allowed = true
                actual_params[:backlog_allowed] = true
            else
                # smooth extreme peaks (redistribute tail to neighbors)
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
                        # tiny buffer bump sometimes
                        if rand() < 0.5
                            prod_capacity = round(Int, prod_capacity * 1.05)
                        end
                    end
                end
                actual_params[:prod_capacity] = prod_capacity
                # re-bound after smoothing
                demands = max.(demands, 1)
                demands = min.(demands, demand_max * 2)
                cum_demands = cum(demands)  # refresh cumulative after edits
            end
        end

        # 2) Surgical feasibility pass (no heavy-handed rescaling)
        if !backlog_allowed
            sf = max_shortfall(prod_capacity)
            if sf > 0
                required_caps = [ceil(Int, max(0, cum_demands[t] - initial_inventory) / t) for t in 1:n_periods]
                min_cap_needed = maximum(required_caps)
                extra_inv_needed = sf

                # Prefer small capacity uplift; else small inventory nudge; else allow backlog
                uplift_ratio = min_cap_needed > 0 ? min_cap_needed / max(1, prod_capacity) : 1.0
                if uplift_ratio <= 1.5 && rand() < 0.6
                    prod_capacity = max(prod_capacity, min_cap_needed)
                    actual_params[:prod_capacity] = prod_capacity
                elseif extra_inv_needed <= round(Int, max(10, 2 * (demand_min + demand_max) / 2)) && rand() < 0.5
                    initial_inventory += extra_inv_needed
                    actual_params[:initial_inventory] = initial_inventory
                else
                    backlog_allowed = true
                    actual_params[:backlog_allowed] = backlog_allowed
                end
            end
        end

    elseif solution_status == :infeasible
        # Disallow backlog (else nearly always feasible)
        if backlog_allowed
            backlog_allowed = false
            actual_params[:backlog_allowed] = false
        end

        # 1) First create diverse *causes* of trouble
        scenario = rand()
        if scenario < 0.25
            # sustained high demand (2–4 periods)
            start = rand(1:max(1, n_periods - 3))
            dur   = min(rand(2:4), n_periods - start + 1)
            surge = rand(Uniform(1.5, 2.0))
            for t in start:start+dur-1
                demands[t] = round(Int, demands[t] * surge)
            end
        elseif scenario < 0.50
            # capacity cut + lower starting stock
            prod_capacity = round(Int, prod_capacity * rand(Uniform(0.6, 0.8)))
            actual_params[:prod_capacity] = prod_capacity
            initial_inventory = round(Int, max(0, initial_inventory * 0.5))
            actual_params[:initial_inventory] = initial_inventory
        elseif scenario < 0.75
            # supplier disruptions: early → lift tail slightly; late → spike
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
            actual_params[:initial_inventory] = initial_inventory
        else
            # very low starting stock + high variability; one crisis period
            initial_inventory = max(1, round(Int, initial_inventory * 0.1))
            actual_params[:initial_inventory] = initial_inventory
            for t in 1:n_periods
                demands[t] = round(Int, demands[t] * rand(Uniform(0.7, 1.4)))
            end
            crisis = rand(ceil(Int, n_periods/3):n_periods)
            demands[crisis] = max(demands[crisis], round(Int, prod_capacity * rand(Uniform(1.2, 1.5))))
        end

        # Keep demands sane and recompute cumulative
        demands = max.(demands, 1)
        demands = min.(demands, demand_max * 2)
        cum_demands = cum(demands)

        # 2) Guarantee prefix infeasibility by setting cap just-below-required if needed
        sf = max_shortfall(prod_capacity)
        if sf <= 0
            required_caps = [ceil(Int, max(0, cum_demands[t] - initial_inventory) / t) for t in 1:n_periods]
            min_cap_needed = maximum(required_caps)
            margin = rand(1:max(1, round(Int, 0.1 * max(1, min_cap_needed))))  # ~5–10% below
            new_cap = max(0, min_cap_needed - margin)
            if max_shortfall(new_cap) <= 0
                new_cap = max(0, min_cap_needed - 1)
            end
            prod_capacity = new_cap
            actual_params[:prod_capacity] = prod_capacity
        end
    else
        # :all — preserve natural variability (no enforcement)
    end

    # --- Model ---
    model = Model()

    if backlog_allowed
        # with backlogging
        @variable(model, x[1:n_periods] >= 0)
        @variable(model, I_plus[0:n_periods] >= 0)
        @variable(model, I_minus[0:n_periods] >= 0)

        @objective(model, Min,
            sum(production_costs[t]*x[t] + holding_costs[t]*I_plus[t] + backlog_costs[t]*I_minus[t]
                for t in 1:n_periods))

        @constraint(model, I_plus[0] == initial_inventory)
        @constraint(model, I_minus[0] == 0)

        for t in 1:n_periods
            @constraint(model, I_plus[t-1] - I_minus[t-1] + x[t] - demands[t] == I_plus[t] - I_minus[t])
            @constraint(model, x[t] <= prod_capacity)
        end
    else
        # no backlogging
        @variable(model, x[1:n_periods] >= 0)
        @variable(model, I[0:n_periods] >= 0)

        @objective(model, Min,
            sum(production_costs[t]*x[t] + holding_costs[t]*I[t] for t in 1:n_periods))

        @constraint(model, I[0] == initial_inventory)

        for t in 1:n_periods
            @constraint(model, I[t-1] + x[t] - demands[t] == I[t])
            @constraint(model, x[t] <= prod_capacity)
        end
    end

    return model, actual_params
end

"""
    sample_inventory_parameters(target_variables::Int; seed::Int=0)

Sample realistic parameters to target approximately `target_variables` LP variables.
"""
function sample_inventory_parameters(target_variables::Int; seed::Int=0)
    Random.seed!(seed)

    params = Dict{Symbol,Any}()

    # business scale by target size
    scale = target_variables <= 250 ? :small :
            target_variables <= 1000 ? :medium : :large

    # backlog incidence increases with scale
    backlog_prob = scale == :small ? 0.2 : (scale == :medium ? 0.4 : 0.6)
    params[:backlog_allowed] = rand(Bernoulli(backlog_prob))

    # choose n_periods from variable-budget formula
    if params[:backlog_allowed]
        # x[1:T], I_plus[0:T], I_minus[0:T]  => 3T + 2
        params[:n_periods] = max(2, min(5000, round(Int, (target_variables - 2) / 3)))
    else
        # x[1:T], I[0:T] => 2T + 1
        params[:n_periods] = max(2, min(5000, round(Int, (target_variables - 1) / 2)))
    end

    # quick iterative refinement
    for _ in 1:10
        current = calculate_inventory_variable_count(params)
        if abs(current - target_variables) / target_variables < 0.1
            break
        end
        if current < target_variables
            params[:n_periods] = min(5000, params[:n_periods] + 1)
        else
            params[:n_periods] = max(2, params[:n_periods] - 1)
        end
    end

    # scale-specific ranges
    if scale == :small
        params[:prod_capacity] = round(Int, rand(Uniform(50, 500)))
        demand_base = round(Int, rand(Uniform(10, 100)))
        demand_vol  = rand(Uniform(0.2, 0.5))
        params[:demand_min] = max(1, round(Int, demand_base * (1 - demand_vol)))
        params[:demand_max] = round(Int, demand_base * (1 + demand_vol))

        avgd = (params[:demand_min] + params[:demand_max]) / 2
        params[:initial_inventory] = round(Int, avgd * rand(Uniform(0.1, 0.5)))

        prod_cost_base   = rand(Uniform(10, 100))
        prod_cost_spread = rand(Uniform(0.1, 0.3))
        params[:prod_cost_min] = round(Int, prod_cost_base * (1 - prod_cost_spread))
        params[:prod_cost_max] = round(Int, prod_cost_base * (1 + prod_cost_spread))

        holding_rate = rand(Uniform(0.05, 0.25)) / 12
        params[:holding_cost_min] = max(0.01, round(prod_cost_base * holding_rate * 0.8, digits=2))
        params[:holding_cost_max] = round(prod_cost_base * holding_rate * 1.2, digits=2)

    elseif scale == :medium
        params[:prod_capacity] = round(Int, rand(Uniform(200, 2000)))
        demand_base = round(Int, rand(Uniform(50, 1000)))
        demand_vol  = rand(Uniform(0.15, 0.4))
        params[:demand_min] = max(1, round(Int, demand_base * (1 - demand_vol)))
        params[:demand_max] = round(Int, demand_base * (1 + demand_vol))

        avgd = (params[:demand_min] + params[:demand_max]) / 2
        params[:initial_inventory] = round(Int, avgd * rand(Uniform(0.05, 0.4)))

        prod_cost_base   = rand(Uniform(5, 200))
        prod_cost_spread = rand(Uniform(0.05, 0.25))
        params[:prod_cost_min] = round(Int, prod_cost_base * (1 - prod_cost_spread))
        params[:prod_cost_max] = round(Int, prod_cost_base * (1 + prod_cost_spread))

        holding_rate = rand(Uniform(0.03, 0.20)) / 12
        params[:holding_cost_min] = max(0.01, round(prod_cost_base * holding_rate * 0.8, digits=2))
        params[:holding_cost_max] = round(prod_cost_base * holding_rate * 1.2, digits=2)

    else # :large
        params[:prod_capacity] = round(Int, rand(Uniform(1000, 50000)))
        demand_base = round(Int, rand(Uniform(100, 10000)))
        demand_vol  = rand(Uniform(0.1, 0.3))
        params[:demand_min] = max(1, round(Int, demand_base * (1 - demand_vol)))
        params[:demand_max] = round(Int, demand_base * (1 + demand_vol))

        avgd = (params[:demand_min] + params[:demand_max]) / 2
        params[:initial_inventory] = round(Int, avgd * rand(Uniform(0.02, 0.3)))

        prod_cost_base   = rand(Uniform(1, 500))
        prod_cost_spread = rand(Uniform(0.02, 0.20))
        params[:prod_cost_min] = round(Int, prod_cost_base * (1 - prod_cost_spread))
        params[:prod_cost_max] = round(Int, prod_cost_base * (1 + prod_cost_spread))

        holding_rate = rand(Uniform(0.01, 0.15)) / 12
        params[:holding_cost_min] = max(0.01, round(prod_cost_base * holding_rate * 0.8, digits=2))
        params[:holding_cost_max] = round(prod_cost_base * holding_rate * 1.2, digits=2)
    end

    # backlog cost factor (1.5–5x prod cost)
    params[:backlog_cost_factor] = rand(Uniform(1.5, 5.0))

    return params
end

"""
    sample_inventory_parameters(size::Symbol=:medium; seed::Int=0)

Convenience sampler by size bucket; internally maps to a target variable count.
"""
function sample_inventory_parameters(size::Symbol=:medium; seed::Int=0)
    target_map = Dict(
        :small  => rand(50:250),
        :medium => rand(250:1000),
        :large  => rand(1000:10000)
    )
    haskey(target_map, size) || error("Unknown size: $size. Must be :small, :medium, or :large")
    return sample_inventory_parameters(target_map[size]; seed=seed)
end

"""
    calculate_inventory_variable_count(params::Dict)

Return total LP variables implied by params (depends on backlog flag).
"""
function calculate_inventory_variable_count(params::Dict)
    n_periods = get(params, :n_periods, 6)
    backlog_allowed = get(params, :backlog_allowed, false)
    if backlog_allowed
        # x[1:T], I_plus[0:T], I_minus[0:T] => 3T + 2
        return 3*n_periods + 2
    else
        # x[1:T], I[0:T] => 2T + 1
        return 2*n_periods + 1
    end
end

# Optional: register with your problem factory (same API as originals)
register_problem(
    :inventory,
    generate_inventory_problem,
    sample_inventory_parameters,
    "Inventory control problem that minimizes production and holding costs while meeting demand over multiple periods",
)
