using JuMP
using Random
using Distributions

"""
    PharmaceuticalBlending <: ProblemGenerator

Generator for pharmaceutical formulation optimization problems that minimize cost while
meeting strict regulatory, purity, and efficacy requirements.

This problem models realistic pharmaceutical blending with:
- Active Pharmaceutical Ingredients (APIs) with strict potency requirements
- Excipients (fillers, binders, disintegrants, lubricants, coating agents)
- Purity and contamination limits
- Dissolution and bioavailability requirements
- Stability and compatibility constraints
- Regulatory compliance (USP/FDA standards)
- Very tight tolerances (often < 5%)

# Fields
All data generated in constructor based on target_variables and feasibility_status:
- `n_ingredients::Int`: Number of ingredients
- `ingredient_names::Vector{Symbol}`: Names of ingredients
- `ingredient_types::Vector{Symbol}`: Types (:api, :filler, :binder, :disintegrant, :lubricant, :coating)
- `costs::Vector{Float64}`: Cost per unit ($/kg) - APIs are expensive
- `api_potency::Vector{Float64}`: Potency/concentration of APIs (% active)
- `purity::Vector{Float64}`: Purity level of each ingredient (0-1)
- `particle_size::Vector{Float64}`: Average particle size (μm) - affects dissolution
- `compatibility_matrix::Matrix{Int}`: Ingredient compatibility (1 if compatible, 0 if incompatible)
- `target_api_content::Tuple{Float64,Float64}`: Target API content range (very tight, ±2-5%)
- `min_purity::Float64`: Minimum average purity required
- `dissolution_coeffs::Vector{Float64}`: Dissolution rate coefficients
- `target_dissolution::Tuple{Float64,Float64}`: Target dissolution rate range
- `batch_size::Float64`: Batch size in kg
- `max_api_fraction::Float64`: Maximum fraction that can be API (typically low)
- `min_filler_fraction::Float64`: Minimum filler content for tableting
- `max_lubricant_fraction::Float64`: Maximum lubricant (too much reduces dissolution)
- `supply_limits::Vector{Float64}`: Supply limits (APIs often limited by sourcing)
"""
struct PharmaceuticalBlending <: ProblemGenerator
    n_ingredients::Int
    ingredient_names::Vector{Symbol}
    ingredient_types::Vector{Symbol}
    costs::Vector{Float64}
    api_potency::Vector{Float64}
    purity::Vector{Float64}
    particle_size::Vector{Float64}
    compatibility_matrix::Matrix{Int}
    target_api_content::Tuple{Float64,Float64}
    min_purity::Float64
    dissolution_coeffs::Vector{Float64}
    target_dissolution::Tuple{Float64,Float64}
    batch_size::Float64
    max_api_fraction::Float64
    min_filler_fraction::Float64
    max_lubricant_fraction::Float64
    supply_limits::Vector{Float64}
end

"""
    PharmaceuticalBlending(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a pharmaceutical blending problem instance.

# Arguments
- `target_variables`: Target number of variables (ingredients)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function PharmaceuticalBlending(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Number of ingredients (pharmaceutical formulations typically use 5-20 ingredients)
    n_ingredients = max(5, min(25, target_variables))

    # Define ingredient pools
    api_options = [:ibuprofen, :acetaminophen, :aspirin, :loratadine, :metformin,
                   :atorvastatin, :lisinopril, :amlodipine]
    filler_options = [:microcrystalline_cellulose, :lactose, :starch, :calcium_phosphate,
                     :mannitol, :sorbitol]
    binder_options = [:povidone, :hydroxypropyl_cellulose, :starch_paste, :gelatin]
    disintegrant_options = [:crospovidone, :sodium_starch_glycolate, :croscarmellose_sodium]
    lubricant_options = [:magnesium_stearate, :stearic_acid, :talc]
    coating_options = [:opadry, :hydroxypropyl_methylcellulose, :ethylcellulose]

    # Allocate ingredient counts
    n_apis = max(1, min(2, round(Int, n_ingredients * 0.10)))  # Usually 1-2 APIs
    n_fillers = max(1, round(Int, n_ingredients * 0.35))
    n_binders = max(1, round(Int, n_ingredients * 0.20))
    n_disintegrants = max(1, round(Int, n_ingredients * 0.15))
    n_lubricants = max(1, round(Int, n_ingredients * 0.10))
    n_coating = max(0, round(Int, n_ingredients * 0.10))

    # Adjust to match target
    total = n_apis + n_fillers + n_binders + n_disintegrants + n_lubricants + n_coating
    while total < n_ingredients
        n_fillers += 1
        total += 1
    end
    while total > n_ingredients && n_coating > 0
        n_coating -= 1
        total -= 1
    end

    # Select specific ingredients
    ingredient_names = Symbol[]
    ingredient_types = Symbol[]

    append!(ingredient_names, sample(api_options, min(n_apis, length(api_options)), replace=false))
    append!(ingredient_types, fill(:api, n_apis))

    append!(ingredient_names, sample(filler_options, min(n_fillers, length(filler_options)), replace=true))
    append!(ingredient_types, fill(:filler, n_fillers))

    append!(ingredient_names, sample(binder_options, min(n_binders, length(binder_options)), replace=true))
    append!(ingredient_types, fill(:binder, n_binders))

    append!(ingredient_names, sample(disintegrant_options, min(n_disintegrants, length(disintegrant_options)), replace=true))
    append!(ingredient_types, fill(:disintegrant, n_disintegrants))

    append!(ingredient_names, sample(lubricant_options, min(n_lubricants, length(lubricant_options)), replace=true))
    append!(ingredient_types, fill(:lubricant, n_lubricants))

    if n_coating > 0
        append!(ingredient_names, sample(coating_options, min(n_coating, length(coating_options)), replace=true))
        append!(ingredient_types, fill(:coating, n_coating))
    end

    n_ingredients = length(ingredient_names)

    # Generate costs (pharmaceutical grade is expensive, APIs are very expensive)
    costs = zeros(n_ingredients)
    for i in 1:n_ingredients
        itype = ingredient_types[i]
        if itype == :api
            costs[i] = rand(Uniform(500.0, 5000.0))  # Very expensive
        elseif itype == :filler
            costs[i] = rand(Uniform(5.0, 30.0))
        elseif itype == :binder
            costs[i] = rand(Uniform(20.0, 100.0))
        elseif itype == :disintegrant
            costs[i] = rand(Uniform(40.0, 150.0))
        elseif itype == :lubricant
            costs[i] = rand(Uniform(15.0, 80.0))
        else  # coating
            costs[i] = rand(Uniform(30.0, 120.0))
        end
    end

    # API potency (purity/concentration of active ingredient)
    api_potency = zeros(n_ingredients)
    for i in 1:n_ingredients
        if ingredient_types[i] == :api
            api_potency[i] = rand(Uniform(0.95, 0.999))  # Very high purity required
        else
            api_potency[i] = 0.0
        end
    end

    # Purity levels (pharmaceutical grade)
    purity = zeros(n_ingredients)
    for i in 1:n_ingredients
        itype = ingredient_types[i]
        if itype == :api
            purity[i] = rand(Uniform(0.98, 0.999))
        else
            purity[i] = rand(Uniform(0.95, 0.995))
        end
    end

    # Particle size (affects dissolution and mixing)
    particle_size = zeros(n_ingredients)
    for i in 1:n_ingredients
        itype = ingredient_types[i]
        if itype == :api
            particle_size[i] = rand(Uniform(10.0, 100.0))  # Smaller for better dissolution
        elseif itype == :filler
            particle_size[i] = rand(Uniform(50.0, 200.0))
        else
            particle_size[i] = rand(Uniform(20.0, 150.0))
        end
    end

    # Compatibility matrix (some ingredients are incompatible)
    compatibility_matrix = ones(Int, n_ingredients, n_ingredients)
    for i in 1:n_ingredients
        compatibility_matrix[i, i] = 1  # Compatible with self
        for j in (i+1):n_ingredients
            # Small chance of incompatibility
            if rand() < 0.05
                compatibility_matrix[i, j] = 0
                compatibility_matrix[j, i] = 0
            end
        end
    end

    # Dissolution coefficients (higher = faster dissolution)
    dissolution_coeffs = zeros(n_ingredients)
    for i in 1:n_ingredients
        itype = ingredient_types[i]
        if itype == :api
            # Inversely related to particle size
            dissolution_coeffs[i] = rand(Uniform(0.6, 0.9)) * (100.0 / particle_size[i])
        elseif itype == :disintegrant
            dissolution_coeffs[i] = rand(Uniform(0.8, 1.2))
        elseif itype == :lubricant
            dissolution_coeffs[i] = -rand(Uniform(0.3, 0.6))  # Lubricants slow dissolution
        else
            dissolution_coeffs[i] = rand(Uniform(0.0, 0.3))
        end
    end

    # Batch size
    batch_size = Float64(rand(Uniform(10.0, 100.0)))  # kg

    # Target API content (very tight tolerance, e.g., 200mg ± 5%)
    target_api_mg = rand(Uniform(50.0, 500.0))  # mg per tablet/dose
    # Convert to fraction of batch assuming ~1000 doses per batch
    doses_per_batch = rand(Uniform(500, 2000))
    target_api_fraction = (target_api_mg * doses_per_batch) / (batch_size * 1000000)  # mg to kg

    tolerance = rand(Uniform(0.02, 0.05))  # 2-5% tolerance (pharmaceutical standard)
    target_api_content = (target_api_fraction * (1 - tolerance),
                         target_api_fraction * (1 + tolerance))

    # Purity requirement
    min_purity = rand(Uniform(0.96, 0.98))

    # Dissolution target
    target_dissolution = (rand(Uniform(0.70, 0.75)), rand(Uniform(0.80, 0.90)))

    # Fraction constraints
    max_api_fraction = rand(Uniform(0.05, 0.15))  # APIs are typically small fraction
    min_filler_fraction = rand(Uniform(0.40, 0.60))  # Need bulk for tableting
    max_lubricant_fraction = rand(Uniform(0.01, 0.03))  # Too much reduces dissolution

    # Supply limits
    supply_limits = zeros(n_ingredients)
    for i in 1:n_ingredients
        itype = ingredient_types[i]
        if itype == :api
            # APIs often have limited supply
            supply_limits[i] = batch_size * rand(Uniform(0.08, 0.12))
        else
            supply_limits[i] = batch_size * rand(Uniform(0.5, 2.0))
        end
    end

    # Feasibility enforcement
    if feasibility_status == feasible
        # Build a baseline feasible formulation
        blend = zeros(n_ingredients)

        # Allocate API (small amount, within limits)
        api_indices = findall(t -> t == :api, ingredient_types)
        target_api_mid = (target_api_content[1] + target_api_content[2]) / 2
        total_api = batch_size * target_api_mid

        for idx in api_indices
            # Account for potency
            blend[idx] = (total_api / length(api_indices)) / max(0.01, api_potency[idx])
        end

        # Allocate filler (majority)
        filler_indices = findall(t -> t == :filler, ingredient_types)
        filler_amount = batch_size * min_filler_fraction
        for idx in filler_indices
            blend[idx] = filler_amount / length(filler_indices)
        end

        # Allocate other excipients
        remaining = batch_size - sum(blend)

        binder_indices = findall(t -> t == :binder, ingredient_types)
        disintegrant_indices = findall(t -> t == :disintegrant, ingredient_types)
        lubricant_indices = findall(t -> t == :lubricant, ingredient_types)
        coating_indices = findall(t -> t == :coating, ingredient_types)

        for idx in binder_indices
            blend[idx] = remaining * 0.25 / max(1, length(binder_indices))
        end
        for idx in disintegrant_indices
            blend[idx] = remaining * 0.25 / max(1, length(disintegrant_indices))
        end
        for idx in lubricant_indices
            blend[idx] = min(batch_size * max_lubricant_fraction / max(1, length(lubricant_indices)),
                           remaining * 0.15 / max(1, length(lubricant_indices)))
        end
        for idx in coating_indices
            blend[idx] = remaining * 0.15 / max(1, length(coating_indices))
        end

        # Normalize
        blend .*= batch_size / sum(blend)

        # Update supply limits to be feasible
        for i in 1:n_ingredients
            supply_limits[i] = max(supply_limits[i], blend[i] * 1.20)
        end

    elseif feasibility_status == infeasible
        # Create infeasibility
        mode = rand([:api_shortage, :purity_conflict, :incompatibility])

        if mode == :api_shortage
            # Restrict API supply below what's needed
            api_indices = findall(t -> t == :api, ingredient_types)
            target_api_mid = (target_api_content[1] + target_api_content[2]) / 2
            for idx in api_indices
                supply_limits[idx] = batch_size * target_api_mid * 0.3
            end

        elseif mode == :purity_conflict
            # Raise purity requirement impossibly high
            min_purity = 0.995

        else  # incompatibility
            # Make essential ingredients incompatible
            api_indices = findall(t -> t == :api, ingredient_types)
            filler_indices = findall(t -> t == :filler, ingredient_types)

            if !isempty(api_indices) && !isempty(filler_indices)
                api_idx = api_indices[1]
                filler_idx = filler_indices[1]
                compatibility_matrix[api_idx, filler_idx] = 0
                compatibility_matrix[filler_idx, api_idx] = 0
            end
        end
    end

    return PharmaceuticalBlending(
        n_ingredients,
        ingredient_names,
        ingredient_types,
        costs,
        api_potency,
        purity,
        particle_size,
        compatibility_matrix,
        target_api_content,
        min_purity,
        dissolution_coeffs,
        target_dissolution,
        batch_size,
        max_api_fraction,
        min_filler_fraction,
        max_lubricant_fraction,
        supply_limits
    )
end

"""
    build_model(prob::PharmaceuticalBlending)

Build a JuMP model for the pharmaceutical blending problem (deterministic).

# Arguments
- `prob`: PharmaceuticalBlending instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::PharmaceuticalBlending)
    model = Model()

    @variable(model, x[1:prob.n_ingredients] >= 0)

    # Objective: minimize cost
    @objective(model, Min, sum(prob.costs[i] * x[i] for i in 1:prob.n_ingredients))

    # Batch size constraint
    @constraint(model, sum(x[i] for i in 1:prob.n_ingredients) == prob.batch_size)

    # Supply limits
    for i in 1:prob.n_ingredients
        @constraint(model, x[i] <= prob.supply_limits[i])
    end

    # API content constraint (very tight tolerance)
    lower_api, upper_api = prob.target_api_content
    @constraint(model,
        sum(prob.api_potency[i] * x[i] for i in 1:prob.n_ingredients) >=
        lower_api * prob.batch_size)
    @constraint(model,
        sum(prob.api_potency[i] * x[i] for i in 1:prob.n_ingredients) <=
        upper_api * prob.batch_size)

    # Purity constraint
    @constraint(model,
        sum(prob.purity[i] * x[i] for i in 1:prob.n_ingredients) >=
        prob.min_purity * prob.batch_size)

    # Dissolution constraint
    lower_diss, upper_diss = prob.target_dissolution
    @constraint(model,
        sum(prob.dissolution_coeffs[i] * x[i] for i in 1:prob.n_ingredients) >=
        lower_diss * prob.batch_size)
    @constraint(model,
        sum(prob.dissolution_coeffs[i] * x[i] for i in 1:prob.n_ingredients) <=
        upper_diss * prob.batch_size)

    # API fraction limit
    api_indices = findall(t -> t == :api, prob.ingredient_types)
    if !isempty(api_indices)
        @constraint(model,
            sum(x[i] for i in api_indices) <= prob.max_api_fraction * prob.batch_size)
    end

    # Filler fraction requirement
    filler_indices = findall(t -> t == :filler, prob.ingredient_types)
    if !isempty(filler_indices)
        @constraint(model,
            sum(x[i] for i in filler_indices) >= prob.min_filler_fraction * prob.batch_size)
    end

    # Lubricant fraction limit
    lubricant_indices = findall(t -> t == :lubricant, prob.ingredient_types)
    if !isempty(lubricant_indices)
        @constraint(model,
            sum(x[i] for i in lubricant_indices) <= prob.max_lubricant_fraction * prob.batch_size)
    end

    # Compatibility constraints (incompatible ingredients cannot both be used)
    for i in 1:prob.n_ingredients
        for j in (i+1):prob.n_ingredients
            if prob.compatibility_matrix[i, j] == 0
                # If incompatible, at least one must be zero (big-M formulation)
                # For LP relaxation, we'll use a softer constraint
                @constraint(model, x[i] + x[j] <= prob.batch_size * 0.5)
            end
        end
    end

    return model
end

# Register the problem variant
register_problem(
    :blending_pharmaceutical,
    PharmaceuticalBlending,
    "Pharmaceutical formulation problem with strict API content, purity, and dissolution requirements"
)
