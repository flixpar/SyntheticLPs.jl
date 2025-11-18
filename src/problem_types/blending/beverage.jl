using JuMP
using Random
using Distributions

"""
    BeverageBlending <: ProblemGenerator

Generator for beverage formulation optimization problems that minimize cost while meeting
flavor, nutritional, and quality requirements.

This problem models realistic beverage formulation with:
- Multiple ingredient types (juices, sweeteners, acids, flavors, water)
- Flavor profile requirements (sweetness, acidity, bitterness, fruitiness)
- Nutritional constraints (calories, sugar, vitamins)
- pH and Brix (sugar content) specifications
- Ingredient compatibility and balance
- Batch size and cost optimization

# Fields
All data generated in constructor based on target_variables and feasibility_status:
- `n_ingredients::Int`: Number of ingredients
- `ingredient_names::Vector{Symbol}`: Names of ingredients (e.g., :apple_juice, :sugar, :citric_acid)
- `ingredient_types::Vector{Symbol}`: Types (:juice, :sweetener, :acid, :flavor, :water, :additive)
- `costs::Vector{Float64}`: Cost per unit of each ingredient
- `flavor_profiles::Matrix{Float64}`: Flavor attributes (n_ingredients × n_flavors)
- `flavor_names::Vector{Symbol}`: Flavor attribute names (:sweetness, :acidity, :bitterness, :fruitiness)
- `flavor_targets::Vector{Tuple{Float64,Float64}}`: Target ranges for each flavor attribute
- `sugar_content::Vector{Float64}`: Sugar content (Brix) for each ingredient
- `caloric_content::Vector{Float64}`: Calories per unit for each ingredient
- `ph_values::Vector{Float64}`: pH contribution of each ingredient
- `target_brix::Tuple{Float64,Float64}`: Target Brix range (sugar content)
- `target_ph::Tuple{Float64,Float64}`: Target pH range
- `target_calories::Tuple{Float64,Float64}`: Target calorie range per serving
- `batch_size::Float64`: Target batch size in liters
- `supply_limits::Vector{Float64}`: Maximum available amount of each ingredient
- `min_water_fraction::Float64`: Minimum fraction of batch that must be water
- `max_sweetener_fraction::Float64`: Maximum fraction of batch that can be sweeteners
"""
struct BeverageBlending <: ProblemGenerator
    n_ingredients::Int
    ingredient_names::Vector{Symbol}
    ingredient_types::Vector{Symbol}
    costs::Vector{Float64}
    flavor_profiles::Matrix{Float64}
    flavor_names::Vector{Symbol}
    flavor_targets::Vector{Tuple{Float64,Float64}}
    sugar_content::Vector{Float64}
    caloric_content::Vector{Float64}
    ph_values::Vector{Float64}
    target_brix::Tuple{Float64,Float64}
    target_ph::Tuple{Float64,Float64}
    target_calories::Tuple{Float64,Float64}
    batch_size::Float64
    supply_limits::Vector{Float64}
    min_water_fraction::Float64
    max_sweetener_fraction::Float64
end

"""
    BeverageBlending(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a beverage blending problem instance.

# Arguments
- `target_variables`: Target number of variables (ingredients)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function BeverageBlending(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Number of ingredients
    n_ingredients = max(5, min(50, target_variables))

    # Define possible ingredients with realistic properties
    juice_options = [:apple_juice, :orange_juice, :grape_juice, :cranberry_juice,
                     :pineapple_juice, :mango_juice, :pomegranate_juice, :lemon_juice]
    sweetener_options = [:sugar, :corn_syrup, :stevia, :sucralose, :honey]
    acid_options = [:citric_acid, :malic_acid, :ascorbic_acid]
    flavor_options = [:vanilla_extract, :mint_extract, :ginger_extract, :berry_flavor]
    additive_options = [:preservative, :vitamin_c, :vitamin_b, :calcium]

    # Allocate ingredients
    n_juices = max(2, round(Int, n_ingredients * rand(Uniform(0.3, 0.5))))
    n_sweeteners = max(1, round(Int, n_ingredients * rand(Uniform(0.15, 0.25))))
    n_acids = max(1, round(Int, n_ingredients * rand(Uniform(0.05, 0.10))))
    n_flavors = max(1, round(Int, n_ingredients * rand(Uniform(0.10, 0.15))))
    n_additives = max(1, round(Int, n_ingredients * rand(Uniform(0.05, 0.10))))
    n_water = 1

    # Adjust to match target
    total = n_juices + n_sweeteners + n_acids + n_flavors + n_additives + n_water
    while total < n_ingredients
        n_juices += 1
        total += 1
    end
    while total > n_ingredients
        if n_juices > 2
            n_juices -= 1
            total -= 1
        else
            break
        end
    end

    # Select specific ingredients
    ingredient_names = Symbol[]
    ingredient_types = Symbol[]

    append!(ingredient_names, sample(juice_options, min(n_juices, length(juice_options)), replace=false))
    append!(ingredient_types, fill(:juice, n_juices))

    append!(ingredient_names, sample(sweetener_options, min(n_sweeteners, length(sweetener_options)), replace=false))
    append!(ingredient_types, fill(:sweetener, n_sweeteners))

    append!(ingredient_names, sample(acid_options, min(n_acids, length(acid_options)), replace=false))
    append!(ingredient_types, fill(:acid, n_acids))

    append!(ingredient_names, sample(flavor_options, min(n_flavors, length(flavor_options)), replace=false))
    append!(ingredient_types, fill(:flavor, n_flavors))

    append!(ingredient_names, sample(additive_options, min(n_additives, length(additive_options)), replace=false))
    append!(ingredient_types, fill(:additive, n_additives))

    push!(ingredient_names, :water)
    push!(ingredient_types, :water)

    n_ingredients = length(ingredient_names)

    # Generate costs ($/liter or $/kg)
    costs = zeros(n_ingredients)
    for i in 1:n_ingredients
        itype = ingredient_types[i]
        if itype == :juice
            costs[i] = rand(Uniform(2.0, 8.0))
        elseif itype == :sweetener
            costs[i] = rand(Uniform(1.0, 5.0))
        elseif itype == :acid
            costs[i] = rand(Uniform(5.0, 15.0))
        elseif itype == :flavor
            costs[i] = rand(Uniform(10.0, 30.0))
        elseif itype == :additive
            costs[i] = rand(Uniform(8.0, 25.0))
        else  # water
            costs[i] = 0.1
        end
    end

    # Flavor profiles (4 dimensions: sweetness, acidity, bitterness, fruitiness)
    flavor_names = [:sweetness, :acidity, :bitterness, :fruitiness]
    n_flavors = length(flavor_names)
    flavor_profiles = zeros(n_ingredients, n_flavors)

    for i in 1:n_ingredients
        itype = ingredient_types[i]
        iname = ingredient_names[i]

        if itype == :juice
            # Juices have moderate sweetness, variable acidity, low bitterness, high fruitiness
            flavor_profiles[i, 1] = rand(Uniform(0.4, 0.7))  # sweetness
            flavor_profiles[i, 2] = rand(Uniform(0.3, 0.8))  # acidity
            flavor_profiles[i, 3] = rand(Uniform(0.0, 0.2))  # bitterness
            flavor_profiles[i, 4] = rand(Uniform(0.7, 0.95)) # fruitiness
        elseif itype == :sweetener
            flavor_profiles[i, 1] = rand(Uniform(0.85, 0.99))
            flavor_profiles[i, 2] = rand(Uniform(0.0, 0.1))
            flavor_profiles[i, 3] = rand(Uniform(0.0, 0.05))
            flavor_profiles[i, 4] = rand(Uniform(0.0, 0.1))
        elseif itype == :acid
            flavor_profiles[i, 1] = rand(Uniform(0.0, 0.1))
            flavor_profiles[i, 2] = rand(Uniform(0.90, 0.99))
            flavor_profiles[i, 3] = rand(Uniform(0.0, 0.1))
            flavor_profiles[i, 4] = rand(Uniform(0.0, 0.1))
        elseif itype == :flavor
            # Variable depending on flavor type
            flavor_profiles[i, 1] = rand(Uniform(0.1, 0.5))
            flavor_profiles[i, 2] = rand(Uniform(0.0, 0.3))
            flavor_profiles[i, 3] = rand(Uniform(0.0, 0.4))
            flavor_profiles[i, 4] = rand(Uniform(0.2, 0.6))
        elseif itype == :water
            flavor_profiles[i, :] .= 0.0
        else  # additive
            flavor_profiles[i, :] .= rand(Uniform(0.0, 0.1), n_flavors)
        end
    end

    # Sugar content (Brix degrees)
    sugar_content = zeros(n_ingredients)
    for i in 1:n_ingredients
        itype = ingredient_types[i]
        if itype == :juice
            sugar_content[i] = rand(Uniform(8.0, 15.0))
        elseif itype == :sweetener
            if ingredient_names[i] in [:sugar, :corn_syrup, :honey]
                sugar_content[i] = rand(Uniform(60.0, 80.0))
            else  # artificial sweeteners
                sugar_content[i] = 0.0
            end
        else
            sugar_content[i] = 0.0
        end
    end

    # Caloric content (kcal/100ml or kcal/100g)
    caloric_content = zeros(n_ingredients)
    for i in 1:n_ingredients
        itype = ingredient_types[i]
        if itype == :juice
            caloric_content[i] = rand(Uniform(40.0, 60.0))
        elseif itype == :sweetener
            if ingredient_names[i] in [:sugar, :corn_syrup, :honey]
                caloric_content[i] = rand(Uniform(350.0, 400.0))
            else
                caloric_content[i] = rand(Uniform(0.0, 5.0))
            end
        else
            caloric_content[i] = 0.0
        end
    end

    # pH values
    ph_values = zeros(n_ingredients)
    for i in 1:n_ingredients
        itype = ingredient_types[i]
        if itype == :juice
            ph_values[i] = rand(Uniform(3.0, 4.5))
        elseif itype == :acid
            ph_values[i] = rand(Uniform(2.0, 3.0))
        elseif itype == :water
            ph_values[i] = 7.0
        else
            ph_values[i] = rand(Uniform(5.0, 7.0))
        end
    end

    # Batch size
    batch_size = Float64(rand(Uniform(100.0, 1000.0)))  # liters

    # Target specifications
    target_brix = (rand(Uniform(10.0, 12.0)), rand(Uniform(12.0, 14.0)))
    target_ph = (rand(Uniform(3.2, 3.5)), rand(Uniform(3.8, 4.2)))
    target_calories = (rand(Uniform(40.0, 50.0)), rand(Uniform(55.0, 65.0)))

    # Flavor targets
    flavor_targets = Vector{Tuple{Float64,Float64}}(undef, n_flavors)
    flavor_targets[1] = (rand(Uniform(0.50, 0.55)), rand(Uniform(0.60, 0.65)))  # sweetness
    flavor_targets[2] = (rand(Uniform(0.25, 0.30)), rand(Uniform(0.35, 0.40)))  # acidity
    flavor_targets[3] = (rand(Uniform(0.00, 0.05)), rand(Uniform(0.08, 0.12)))  # bitterness
    flavor_targets[4] = (rand(Uniform(0.60, 0.65)), rand(Uniform(0.70, 0.75)))  # fruitiness

    # Supply limits
    supply_limits = zeros(n_ingredients)
    for i in 1:n_ingredients
        itype = ingredient_types[i]
        if itype == :juice
            supply_limits[i] = batch_size * rand(Uniform(0.4, 0.8))
        elseif itype == :sweetener
            supply_limits[i] = batch_size * rand(Uniform(0.1, 0.3))
        elseif itype == :water
            supply_limits[i] = Inf
        else
            supply_limits[i] = batch_size * rand(Uniform(0.05, 0.15))
        end
    end

    # Constraints on fractions
    min_water_fraction = rand(Uniform(0.10, 0.25))
    max_sweetener_fraction = rand(Uniform(0.10, 0.20))

    # Feasibility enforcement
    if feasibility_status == feasible
        # Build a baseline feasible solution
        blend = zeros(n_ingredients)

        # Allocate water
        water_idx = findfirst(t -> t == :water, ingredient_types)
        blend[water_idx] = batch_size * min_water_fraction

        # Allocate juices (majority)
        juice_indices = findall(t -> t == :juice, ingredient_types)
        juice_total = batch_size * rand(Uniform(0.50, 0.65))
        for idx in juice_indices
            blend[idx] = juice_total / length(juice_indices) * (0.8 + 0.4 * rand())
        end

        # Normalize to batch size
        current_total = sum(blend)
        remaining = batch_size - current_total

        # Add sweeteners
        sweetener_indices = findall(t -> t == :sweetener, ingredient_types)
        if !isempty(sweetener_indices)
            sweetener_amount = min(remaining * 0.3, batch_size * max_sweetener_fraction)
            for idx in sweetener_indices
                blend[idx] = sweetener_amount / length(sweetener_indices)
            end
        end

        # Normalize final
        blend .*= batch_size / sum(blend)

        # Calculate achieved properties
        achieved_brix = sum(sugar_content[i] * blend[i] for i in 1:n_ingredients) / batch_size
        achieved_calories = sum(caloric_content[i] * blend[i] for i in 1:n_ingredients) / batch_size
        achieved_flavors = [sum(flavor_profiles[i, f] * blend[i] for i in 1:n_ingredients) / batch_size
                           for f in 1:n_flavors]

        # Set targets around achieved values with tolerance
        tolerance = 0.08
        target_brix = (achieved_brix * (1 - tolerance), achieved_brix * (1 + tolerance))
        target_calories = (achieved_calories * (1 - tolerance), achieved_calories * (1 + tolerance))

        for f in 1:n_flavors
            target_brix_val = achieved_flavors[f]
            flavor_targets[f] = (target_brix_val * (1 - tolerance), target_brix_val * (1 + tolerance))
        end

        # Ensure supply limits are feasible
        for i in 1:n_ingredients
            if supply_limits[i] < Inf
                supply_limits[i] = max(supply_limits[i], blend[i] * 1.15)
            end
        end

    elseif feasibility_status == infeasible
        # Create infeasibility
        mode = rand([:flavor_conflict, :supply_shortage, :constraint_conflict])

        if mode == :flavor_conflict
            # Set conflicting flavor requirements
            flavor_targets[1] = (0.85, 0.90)  # High sweetness
            flavor_targets[2] = (0.75, 0.80)  # High acidity (conflicts with high sweetness)

        elseif mode == :supply_shortage
            # Restrict supply of key ingredients
            juice_indices = findall(t -> t == :juice, ingredient_types)
            for idx in juice_indices
                supply_limits[idx] = batch_size * 0.05
            end

        else  # constraint_conflict
            # Set impossible Brix with limited sweeteners
            target_brix = (18.0, 20.0)  # Very high
            sweetener_indices = findall(t -> t == :sweetener, ingredient_types)
            for idx in sweetener_indices
                supply_limits[idx] = batch_size * 0.03
            end
            max_sweetener_fraction = 0.05
        end
    end

    return BeverageBlending(
        n_ingredients,
        ingredient_names,
        ingredient_types,
        costs,
        flavor_profiles,
        flavor_names,
        flavor_targets,
        sugar_content,
        caloric_content,
        ph_values,
        target_brix,
        target_ph,
        target_calories,
        batch_size,
        supply_limits,
        min_water_fraction,
        max_sweetener_fraction
    )
end

"""
    build_model(prob::BeverageBlending)

Build a JuMP model for the beverage blending problem (deterministic).

# Arguments
- `prob`: BeverageBlending instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::BeverageBlending)
    model = Model()

    @variable(model, x[1:prob.n_ingredients] >= 0)

    # Objective: minimize cost
    @objective(model, Min, sum(prob.costs[i] * x[i] for i in 1:prob.n_ingredients))

    # Batch size constraint
    @constraint(model, sum(x[i] for i in 1:prob.n_ingredients) == prob.batch_size)

    # Supply limits
    for i in 1:prob.n_ingredients
        if prob.supply_limits[i] < Inf
            @constraint(model, x[i] <= prob.supply_limits[i])
        end
    end

    # Flavor profile constraints
    for f in 1:length(prob.flavor_names)
        lower, upper = prob.flavor_targets[f]
        @constraint(model,
            sum(prob.flavor_profiles[i, f] * x[i] for i in 1:prob.n_ingredients) >=
            lower * prob.batch_size)
        @constraint(model,
            sum(prob.flavor_profiles[i, f] * x[i] for i in 1:prob.n_ingredients) <=
            upper * prob.batch_size)
    end

    # Brix constraint
    lower_brix, upper_brix = prob.target_brix
    @constraint(model,
        sum(prob.sugar_content[i] * x[i] for i in 1:prob.n_ingredients) >=
        lower_brix * prob.batch_size)
    @constraint(model,
        sum(prob.sugar_content[i] * x[i] for i in 1:prob.n_ingredients) <=
        upper_brix * prob.batch_size)

    # Calorie constraint
    lower_cal, upper_cal = prob.target_calories
    @constraint(model,
        sum(prob.caloric_content[i] * x[i] for i in 1:prob.n_ingredients) >=
        lower_cal * prob.batch_size)
    @constraint(model,
        sum(prob.caloric_content[i] * x[i] for i in 1:prob.n_ingredients) <=
        upper_cal * prob.batch_size)

    # Water fraction constraint
    water_indices = findall(t -> t == :water, prob.ingredient_types)
    if !isempty(water_indices)
        @constraint(model,
            sum(x[i] for i in water_indices) >= prob.min_water_fraction * prob.batch_size)
    end

    # Sweetener fraction constraint
    sweetener_indices = findall(t -> t == :sweetener, prob.ingredient_types)
    if !isempty(sweetener_indices)
        @constraint(model,
            sum(x[i] for i in sweetener_indices) <= prob.max_sweetener_fraction * prob.batch_size)
    end

    return model
end

# Register the problem variant
register_problem(
    :blending_beverage,
    BeverageBlending,
    "Beverage formulation problem optimizing flavor profiles, nutritional content, and cost for drinks"
)
