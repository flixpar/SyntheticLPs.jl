using JuMP
using Random
using Distributions
using Statistics

"""
    CropPlanningProblem <: ProblemGenerator

Generator for crop planning optimization problems.

# Overview
Models agricultural land allocation. The decisions are continuous planted areas
for each crop. The objective maximizes net profit from crop revenue minus
production cost. Constraints limit total land, water, labor, and market demand,
and may require minimum area for selected crops or minimum acreage in crop-type
groups for diversity.

# Fields
- `n_crops::Int`: Number of different crops
- `total_land::Float64`: Total available land in hectares
- `crop_types::Vector{Symbol}`: Type of each crop (:cereal, :vegetable, :legume, :industrial, :oilseed)
- `crop_names::Vector{String}`: Name of each crop option
- `management_systems::Vector{Symbol}`: Agronomic system used by each option
- `yields::Vector{Float64}`: Yield in tons/hectare for each crop
- `prices::Vector{Float64}`: Price in dollars/ton for each crop
- `production_costs::Vector{Float64}`: Production cost in dollars/hectare for each crop
- `water_requirements::Vector{Float64}`: Water requirement in mm/season for each crop
- `labor_requirements::Vector{Float64}`: Labor requirement in hours/hectare for each crop
- `net_profit_per_ha::Vector{Float64}`: Net profit per hectare for each crop
- `market_demand_tonnes::Vector{Float64}`: Saleable production limit in tonnes for each crop
- `min_area_per_crop::Vector{Float64}`: Minimum area requirement in hectares for each crop
- `water_capacity::Float64`: Available water capacity
- `labor_capacity::Float64`: Available labor capacity
- `diversity_requirements::Vector{CropDiversityRequirement}`: Crop-group acreage floors
- `feasible_witness::Union{Nothing,Vector{Float64}}`: Planted area plan for feasible requests
- `infeasibility_certificate::Union{Nothing,CropResourceCertificate}`: Mandatory-resource cut for infeasible requests
"""
struct CropDiversityRequirement
    crop_type::Symbol
    minimum_area::Float64
    crop_indices::Vector{Int}
end

Base.:(==)(a::CropDiversityRequirement, b::CropDiversityRequirement) =
    a.crop_type == b.crop_type && a.minimum_area == b.minimum_area &&
    a.crop_indices == b.crop_indices
Base.isequal(a::CropDiversityRequirement, b::CropDiversityRequirement) = a == b
Base.hash(a::CropDiversityRequirement, h::UInt) =
    hash((a.crop_type, a.minimum_area, a.crop_indices), h)

struct CropResourceCertificate
    resource::Symbol
    forced_usage::Float64
    capacity::Float64
end

Base.:(==)(a::CropResourceCertificate, b::CropResourceCertificate) =
    a.resource == b.resource && a.forced_usage == b.forced_usage &&
    a.capacity == b.capacity
Base.isequal(a::CropResourceCertificate, b::CropResourceCertificate) = a == b
Base.hash(a::CropResourceCertificate, h::UInt) =
    hash((a.resource, a.forced_usage, a.capacity), h)

struct CropPlanningProblem <: ProblemGenerator
    n_crops::Int
    total_land::Float64
    crop_types::Vector{Symbol}
    crop_names::Vector{String}
    management_systems::Vector{Symbol}
    yields::Vector{Float64}
    prices::Vector{Float64}
    production_costs::Vector{Float64}
    water_requirements::Vector{Float64}
    labor_requirements::Vector{Float64}
    net_profit_per_ha::Vector{Float64}
    market_demand_tonnes::Vector{Float64}
    min_area_per_crop::Vector{Float64}
    water_capacity::Float64
    labor_capacity::Float64
    diversity_requirements::Vector{CropDiversityRequirement}
    feasible_witness::Union{Nothing,Vector{Float64}}
    infeasibility_certificate::Union{Nothing,CropResourceCertificate}
end

const _CROP_CATALOG = [
    ("Wheat", :cereal), ("Maize", :cereal), ("Rice", :cereal),
    ("Barley", :cereal), ("Oats", :cereal),
    ("Tomatoes", :vegetable), ("Peppers", :vegetable),
    ("Lettuce", :vegetable), ("Carrots", :vegetable),
    ("Onions", :vegetable),
    ("Soybeans", :legume), ("Field peas", :legume),
    ("Lentils", :legume), ("Dry beans", :legume),
    ("Chickpeas", :legume),
    ("Cotton", :industrial), ("Sugarcane", :industrial),
    ("Tobacco", :industrial), ("Hemp", :industrial),
    ("Fiber flax", :industrial),
    ("Sunflower", :oilseed), ("Canola", :oilseed),
    ("Safflower", :oilseed), ("Sesame", :oilseed),
    ("Peanuts", :oilseed),
]

const _CROP_MANAGEMENT_SYSTEMS = (:rainfed, :irrigated, :low_input, :intensive)

"""
    CropPlanningProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a crop planning problem instance.

# Arguments
- `target_variables`: Target number of variables (crops)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function CropPlanningProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    rng = Random.MersenneTwister(seed)

    # For crop planning, target_variables = n_crops
    n_crops = max(2, target_variables)

    # Determine problem scale based on target variables
    if target_variables <= 250
        # Small scale: Family farm or small agricultural operation
        total_land = rand(rng, Uniform(50.0, 500.0))
        water_availability_factor = rand(rng, Uniform(0.6, 0.8))
        labor_availability_factor = rand(rng, Uniform(0.7, 0.9))
        market_demand_factor = rand(rng, Uniform(1.0, 1.3))
        diversity_constraint_prob = rand(rng, Uniform(0.5, 0.8))
    elseif target_variables <= 1000
        # Medium scale: Commercial farm or agricultural cooperative
        total_land = rand(rng, Uniform(500.0, 5000.0))
        water_availability_factor = rand(rng, Uniform(0.65, 0.85))
        labor_availability_factor = rand(rng, Uniform(0.75, 0.95))
        market_demand_factor = rand(rng, Uniform(1.1, 1.5))
        diversity_constraint_prob = rand(rng, Uniform(0.6, 0.9))
    else
        # Large scale: Industrial agriculture or regional planning
        total_land = rand(rng, Uniform(5000.0, 50000.0))
        water_availability_factor = rand(rng, Uniform(0.7, 0.9))
        labor_availability_factor = rand(rng, Uniform(0.8, 1.0))
        market_demand_factor = rand(rng, Uniform(1.2, 2.0))
        diversity_constraint_prob = rand(rng, Uniform(0.7, 0.95))
    end

    # Minimum area requirements are common in agricultural planning
    min_area_requirements = rand(rng) < 0.85

    # Convert feasibility status
    solution_status = feasibility_status == feasible ? :feasible :
                     feasibility_status == infeasible ? :infeasible : :all

    # A variable is a crop-management option, not an anonymous synthetic crop.
    # Each catalog block is independently shuffled; large instances therefore
    # repeat recognizable crops under distinct systems/cultivar identifiers.
    assigned_crop_types = Vector{Symbol}(undef, n_crops)
    assigned_crop_names = Vector{String}(undef, n_crops)
    management_systems = Vector{Symbol}(undef, n_crops)
    catalog_size = length(_CROP_CATALOG)
    option = 1
    block = 1
    while option <= n_crops
        order = randperm(rng, catalog_size)
        for catalog_index in order
            option > n_crops && break
            crop_name, crop_type = _CROP_CATALOG[catalog_index]
            system = rand(rng, _CROP_MANAGEMENT_SYSTEMS)
            suffix = block == 1 ? "" : " (cultivar $(block))"
            assigned_crop_names[option] = "$(crop_name) / $(system)$(suffix)"
            assigned_crop_types[option] = crop_type
            management_systems[option] = system
            option += 1
        end
        block += 1
    end

    # Generate crop yields (tons/hectare) based on crop type
    yields = zeros(n_crops)
    for i in 1:n_crops
        if assigned_crop_types[i] == :cereal
            # Cereals: 3-10 tons/ha
            yields[i] = exp(rand(rng, Normal(log(5.5), 0.4)))
            yields[i] = clamp(yields[i], 3.0, 10.0)
        elseif assigned_crop_types[i] == :vegetable
            # Vegetables: 15-40 tons/ha (higher yield)
            yields[i] = exp(rand(rng, Normal(log(25.0), 0.35)))
            yields[i] = clamp(yields[i], 15.0, 40.0)
        elseif assigned_crop_types[i] == :legume
            # Legumes: 2-4 tons/ha
            yields[i] = exp(rand(rng, Normal(log(3.0), 0.3)))
            yields[i] = clamp(yields[i], 2.0, 4.0)
        elseif assigned_crop_types[i] == :industrial
            # Industrial crops: 3-8 tons/ha
            yields[i] = exp(rand(rng, Normal(log(5.0), 0.35)))
            yields[i] = clamp(yields[i], 3.0, 8.0)
        else  # :oilseed
            # Oilseeds: 1.5-4 tons/ha
            yields[i] = exp(rand(rng, Normal(log(2.5), 0.35)))
            yields[i] = clamp(yields[i], 1.5, 4.0)
        end
    end

    # Generate crop prices ($/ton) - higher value for vegetables and industrial crops
    prices = zeros(n_crops)
    for i in 1:n_crops
        if assigned_crop_types[i] == :cereal
            # Cereals: $150-350/ton
            prices[i] = exp(rand(rng, Normal(log(230.0), 0.25)))
            prices[i] = clamp(prices[i], 150.0, 350.0)
        elseif assigned_crop_types[i] == :vegetable
            # Vegetables: $400-900/ton (higher value)
            prices[i] = exp(rand(rng, Normal(log(600.0), 0.25)))
            prices[i] = clamp(prices[i], 400.0, 900.0)
        elseif assigned_crop_types[i] == :legume
            # Legumes: $300-600/ton
            prices[i] = exp(rand(rng, Normal(log(420.0), 0.25)))
            prices[i] = clamp(prices[i], 300.0, 600.0)
        elseif assigned_crop_types[i] == :industrial
            # Industrial: $500-1200/ton
            prices[i] = exp(rand(rng, Normal(log(800.0), 0.3)))
            prices[i] = clamp(prices[i], 500.0, 1200.0)
        else  # :oilseed
            # Oilseeds: $400-700/ton
            prices[i] = exp(rand(rng, Normal(log(520.0), 0.25)))
            prices[i] = clamp(prices[i], 400.0, 700.0)
        end
    end

    # Generate production costs ($/hectare)
    production_costs = zeros(n_crops)
    for i in 1:n_crops
        if assigned_crop_types[i] == :cereal
            # Cereals: $400-900/ha
            production_costs[i] = rand(rng, Normal(600.0, 120.0))
            production_costs[i] = clamp(production_costs[i], 400.0, 900.0)
        elseif assigned_crop_types[i] == :vegetable
            # Vegetables: $1200-2500/ha (labor intensive)
            production_costs[i] = rand(rng, Normal(1700.0, 320.0))
            production_costs[i] = clamp(production_costs[i], 1200.0, 2500.0)
        elseif assigned_crop_types[i] == :legume
            # Legumes: $350-700/ha (lower input costs)
            production_costs[i] = rand(rng, Normal(500.0, 90.0))
            production_costs[i] = clamp(production_costs[i], 350.0, 700.0)
        elseif assigned_crop_types[i] == :industrial
            # Industrial: $800-1800/ha
            production_costs[i] = rand(rng, Normal(1200.0, 250.0))
            production_costs[i] = clamp(production_costs[i], 800.0, 1800.0)
        else  # :oilseed
            # Oilseeds: $400-800/ha
            production_costs[i] = rand(rng, Normal(570.0, 100.0))
            production_costs[i] = clamp(production_costs[i], 400.0, 800.0)
        end
    end

    # Generate water requirements (mm/season)
    water_requirements = zeros(n_crops)
    for i in 1:n_crops
        if assigned_crop_types[i] == :cereal
            if startswith(assigned_crop_names[i], "Rice /")
                # Rice needs more water: 1200-1800 mm
                water_requirements[i] = rand(rng, Normal(1500.0, 150.0))
                water_requirements[i] = clamp(water_requirements[i], 1200.0, 1800.0)
            else
                # Other cereals: 400-650 mm
                water_requirements[i] = rand(rng, Normal(520.0, 70.0))
                water_requirements[i] = clamp(water_requirements[i], 400.0, 650.0)
            end
        elseif assigned_crop_types[i] == :vegetable
            # Vegetables: 350-600 mm (frequent irrigation)
            water_requirements[i] = rand(rng, Normal(470.0, 65.0))
            water_requirements[i] = clamp(water_requirements[i], 350.0, 600.0)
        elseif assigned_crop_types[i] == :legume
            # Legumes: 300-500 mm (drought tolerant)
            water_requirements[i] = rand(rng, Normal(390.0, 55.0))
            water_requirements[i] = clamp(water_requirements[i], 300.0, 500.0)
        elseif assigned_crop_types[i] == :industrial
            if startswith(assigned_crop_names[i], "Sugarcane /")
                # Sugarcane: 1500-2500 mm (water intensive)
                water_requirements[i] = rand(rng, Normal(2000.0, 250.0))
                water_requirements[i] = clamp(water_requirements[i], 1500.0, 2500.0)
            else
                # Cotton and others: 500-800 mm
                water_requirements[i] = rand(rng, Normal(640.0, 80.0))
                water_requirements[i] = clamp(water_requirements[i], 500.0, 800.0)
            end
        else  # :oilseed
            # Oilseeds: 350-550 mm
            water_requirements[i] = rand(rng, Normal(440.0, 60.0))
            water_requirements[i] = clamp(water_requirements[i], 350.0, 550.0)
        end
    end

    # Generate labor requirements (hours/hectare)
    labor_requirements = zeros(n_crops)
    for i in 1:n_crops
        if assigned_crop_types[i] == :cereal
            # Cereals: 30-80 hours/ha (mechanized)
            labor_requirements[i] = rand(rng, Gamma(4, 12))
            labor_requirements[i] = clamp(labor_requirements[i], 30.0, 80.0)
        elseif assigned_crop_types[i] == :vegetable
            # Vegetables: 120-250 hours/ha (labor intensive)
            labor_requirements[i] = rand(rng, Gamma(6, 28))
            labor_requirements[i] = clamp(labor_requirements[i], 120.0, 250.0)
        elseif assigned_crop_types[i] == :legume
            # Legumes: 25-60 hours/ha
            labor_requirements[i] = rand(rng, Gamma(4, 10))
            labor_requirements[i] = clamp(labor_requirements[i], 25.0, 60.0)
        elseif assigned_crop_types[i] == :industrial
            # Industrial: 80-180 hours/ha
            labor_requirements[i] = rand(rng, Gamma(5, 24))
            labor_requirements[i] = clamp(labor_requirements[i], 80.0, 180.0)
        else  # :oilseed
            # Oilseeds: 35-75 hours/ha
            labor_requirements[i] = rand(rng, Gamma(4, 13))
            labor_requirements[i] = clamp(labor_requirements[i], 35.0, 75.0)
        end
    end

    # Management systems create economically coherent versions of the same
    # crop: intensive and irrigated options yield more but use more inputs;
    # rainfed and low-input options trade yield for lower cost and irrigation.
    for i in 1:n_crops
        system = management_systems[i]
        if system == :rainfed
            yields[i] *= rand(rng, Uniform(0.80, 0.95))
            production_costs[i] *= rand(rng, Uniform(0.72, 0.90))
            water_requirements[i] *= rand(rng, Uniform(0.45, 0.70))
            labor_requirements[i] *= rand(rng, Uniform(0.85, 1.00))
        elseif system == :irrigated
            yields[i] *= rand(rng, Uniform(1.04, 1.18))
            production_costs[i] *= rand(rng, Uniform(1.05, 1.18))
            water_requirements[i] *= rand(rng, Uniform(1.00, 1.15))
        elseif system == :low_input
            yields[i] *= rand(rng, Uniform(0.72, 0.90))
            production_costs[i] *= rand(rng, Uniform(0.55, 0.75))
            water_requirements[i] *= rand(rng, Uniform(0.75, 0.92))
            labor_requirements[i] *= rand(rng, Uniform(0.80, 0.95))
        else # :intensive
            yields[i] *= rand(rng, Uniform(1.10, 1.28))
            production_costs[i] *= rand(rng, Uniform(1.20, 1.45))
            water_requirements[i] *= rand(rng, Uniform(1.00, 1.18))
            labor_requirements[i] *= rand(rng, Uniform(1.12, 1.35))
        end
    end

    # Calculate net profit per hectare for each crop
    net_profit_per_ha = prices .* yields .- production_costs

    # Sale limits are expressed in tonnes, so yield actually participates in
    # the market rows. `market_area_caps` is the equivalent acreage used only
    # while constructing witnesses and mandatory-area bounds.
    market_demand_tonnes = zeros(n_crops)
    market_area_caps = zeros(n_crops)
    for i in 1:n_crops
        market_area_caps[i] = total_land * rand(rng, Uniform(0.1, 0.4)) *
                              market_demand_factor
        market_demand_tonnes[i] = yields[i] * market_area_caps[i]
    end

    # Determine minimum area requirements for essential crops
    min_area_per_crop = zeros(n_crops)
    if min_area_requirements
        # Require minimum area for 30-50% of crops (essential for food security)
        n_essential = clamp(round(Int, n_crops * rand(rng, Uniform(0.3, 0.5))),
                            1, n_crops)
        # Prefer cereals and legumes as essential
        essential_candidates = [i for i in 1:n_crops if assigned_crop_types[i] in [:cereal, :legume]]
        if length(essential_candidates) < n_essential
            # Add more crops if needed
            other_crops = collect(setdiff(1:n_crops, essential_candidates))
            shuffle!(rng, other_crops)
            essential_candidates = vcat(essential_candidates, other_crops[1:min(n_essential - length(essential_candidates), length(other_crops))])
        end
        shuffle!(rng, essential_candidates)
        essential_crops = essential_candidates[1:min(n_essential, length(essential_candidates))]

        for i in essential_crops
            # Minimum 2-8% of total land
            min_area_per_crop[i] = total_land * rand(rng, Uniform(0.02, 0.08))
        end
    end

    # Ensure minimum area requirements do not violate market demand limits
    for i in 1:n_crops
        if min_area_per_crop[i] > market_area_caps[i]
            min_area_per_crop[i] = market_area_caps[i]
        end
    end

    # Scale minimum areas if their total exceeds available land
    min_total_area = sum(min_area_per_crop)
    if min_total_area > total_land && min_total_area > 0
        scaling_factor = total_land / min_total_area
        min_area_per_crop .*= scaling_factor
    end

    # Calculate resource requirements for a feasible baseline allocation
    # Use profit-weighted allocation as baseline
    baseline_allocation = zeros(n_crops)

    # First, allocate minimum requirements
    baseline_allocation = copy(min_area_per_crop)
    remaining_land = total_land - sum(baseline_allocation)

    if remaining_land > 0
        # Allocate remaining land based on profitability
        profit_weights = max.(net_profit_per_ha, 0.0)
        total_weight = sum(profit_weights)

        if total_weight > 0
            for i in 1:n_crops
                additional_area = remaining_land * (profit_weights[i] / total_weight)
                # Don't exceed market demand
                max_additional = max(0.0, market_area_caps[i] - baseline_allocation[i])
                baseline_allocation[i] += min(additional_area, max_additional)
            end
        else
            # If all profits are negative, distribute evenly, still respecting
            # the per-crop market headroom so the witness satisfies the
            # `yield[i] * x[i] <= market_demand_tonnes[i]` rows.
            for i in 1:n_crops
                max_additional = max(0.0, market_area_caps[i] - baseline_allocation[i])
                baseline_allocation[i] += min(remaining_land / n_crops, max_additional)
            end
        end
    end

    # No renormalization: baseline must respect market limits; ensure nonzero fallback
    current_total = sum(baseline_allocation)
    if current_total == 0.0
        baseline_allocation .= min.(total_land / n_crops, market_area_caps)
    end

    # Calculate resource usage for baseline allocation
    baseline_water_usage = sum(water_requirements .* baseline_allocation)
    baseline_labor_usage = sum(labor_requirements .* baseline_allocation)

    # Set resource capacities and status metadata.
    water_capacity = 0.0
    labor_capacity = 0.0
    infeasibility_certificate = nothing

    if solution_status == :feasible
        slack_factor = rand(rng, Uniform(1.1, 1.3))
        water_capacity = baseline_water_usage * slack_factor
        labor_capacity = baseline_labor_usage * slack_factor

        # Ensure capacities are reasonable
        water_capacity = max(water_capacity, sum(water_requirements .* min_area_per_crop) * 1.2)
        labor_capacity = max(labor_capacity, sum(labor_requirements .* min_area_per_crop) * 1.2)

    elseif solution_status == :infeasible
        # A certificate is based only on mandatory crop minima. Since land is
        # an upper bound, no assumed use of fallow land enters this proof.
        if sum(min_area_per_crop) <= 0.0
            n_essential = clamp(round(Int, n_crops * rand(rng, Uniform(0.25, 0.5))),
                                1, n_crops)
            essential = randperm(rng, n_crops)[1:n_essential]
            for i in essential
                floor_i = market_area_caps[i] * rand(rng, Uniform(0.3, 0.6))
                min_area_per_crop[i] = max(min_area_per_crop[i], floor_i)
            end
            min_area_per_crop .= min.(min_area_per_crop, market_area_caps)
            min_total_area = sum(min_area_per_crop)
            if min_total_area > total_land && min_total_area > 0
                min_area_per_crop .*= total_land / min_total_area
            end
        end

        # True minimum resource usage = usage at the mandatory lower bounds.
        min_water_bound = sum(water_requirements .* min_area_per_crop)
        min_labor_bound = sum(labor_requirements .* min_area_per_crop)

        @assert min_water_bound > 0.0 && min_labor_bound > 0.0
        violation_factor = rand(rng, Uniform(0.75, 0.95))

        if rand(rng) < 0.5
            water_capacity = min_water_bound * violation_factor
            labor_capacity = min_labor_bound * rand(rng, Uniform(1.1, 1.4))
            infeasibility_certificate = CropResourceCertificate(
                :water, min_water_bound, water_capacity)
        else
            water_capacity = min_water_bound * rand(rng, Uniform(1.1, 1.4))
            labor_capacity = min_labor_bound * violation_factor
            infeasibility_certificate = CropResourceCertificate(
                :labor, min_labor_bound, labor_capacity)
        end

    else  # :all
        # Random capacities without guarantees
        estimated_water = total_land * mean(water_requirements) * water_availability_factor
        estimated_labor = total_land * mean(labor_requirements) * labor_availability_factor

        water_capacity = estimated_water * rand(rng, Uniform(0.6, 1.4))
        labor_capacity = estimated_labor * rand(rng, Uniform(0.6, 1.4))
    end

    # Crop-group floors are typed records. For feasible requests every floor is
    # derived from, and therefore exactly satisfied by, the stored witness.
    diversity_requirements = CropDiversityRequirement[]
    if solution_status != :infeasible && rand(rng) < diversity_constraint_prob
        crop_type_groups = Dict{Symbol, Vector{Int}}()
        for i in 1:n_crops
            ctype = assigned_crop_types[i]
            push!(get!(crop_type_groups, ctype, Int[]), i)
        end

        for ctype in sort!(collect(keys(crop_type_groups)))
            crop_indices = crop_type_groups[ctype]
            if length(crop_indices) >= 2
                if solution_status == :feasible
                    current_type_area = sum(baseline_allocation[crop_indices])
                    if current_type_area > 0.0
                        desired = total_land * rand(rng, Uniform(0.05, 0.15))
                        minimum_area = min(desired,
                                           current_type_area * rand(rng, Uniform(0.75, 0.95)))
                        push!(diversity_requirements,
                              CropDiversityRequirement(ctype, minimum_area,
                                                       copy(crop_indices)))
                    end
                else
                    minimum_area = total_land * rand(rng, Uniform(0.05, 0.15))
                    push!(diversity_requirements,
                          CropDiversityRequirement(ctype, minimum_area,
                                                   copy(crop_indices)))
                end
            end
        end
    end

    feasible_witness = solution_status == :feasible ? copy(baseline_allocation) : nothing

    return CropPlanningProblem(
        n_crops,
        total_land,
        assigned_crop_types,
        assigned_crop_names,
        management_systems,
        yields,
        prices,
        production_costs,
        water_requirements,
        labor_requirements,
        net_profit_per_ha,
        market_demand_tonnes,
        min_area_per_crop,
        water_capacity,
        labor_capacity,
        diversity_requirements,
        feasible_witness,
        infeasibility_certificate,
    )
end

"""
    build_model(prob::CropPlanningProblem)

Build a JuMP model for the crop planning problem.

# Arguments
- `prob`: CropPlanningProblem instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::CropPlanningProblem)
    model = Model()

    # Decision variables: area allocated to each crop (hectares)
    @variable(model, x[1:prob.n_crops] >= 0)

    # Objective: maximize total net profit
    @objective(model, Max,
        sum((prob.prices[i] * prob.yields[i] - prob.production_costs[i]) * x[i] for i in 1:prob.n_crops))

    # Constraint: total land area
    @constraint(model, land_capacity,
                sum(x[i] for i in 1:prob.n_crops) <= prob.total_land)

    # Constraint: water availability
    @constraint(model, water_capacity,
        sum(prob.water_requirements[i] * x[i] for i in 1:prob.n_crops) <= prob.water_capacity)

    # Constraint: labor availability
    @constraint(model, labor_capacity,
        sum(prob.labor_requirements[i] * x[i] for i in 1:prob.n_crops) <= prob.labor_capacity)

    # Market limits apply to harvested tonnes, not planted hectares.
    @constraint(model, market_demand[i in 1:prob.n_crops],
                prob.yields[i] * x[i] <= prob.market_demand_tonnes[i])

    # Constraints: minimum area requirements for essential crops
    mandatory = findall(>(0.0), prob.min_area_per_crop)
    @constraint(model, minimum_area[i in mandatory],
                x[i] >= prob.min_area_per_crop[i])

    # Optional: crop diversity constraints
    @constraint(model, diversity[k in eachindex(prob.diversity_requirements)],
        sum(x[i] for i in prob.diversity_requirements[k].crop_indices) >=
        prob.diversity_requirements[k].minimum_area)

    return model
end

# Register the variant
register_variant(
    :crop_planning,
    :standard,
    CropPlanningProblem,
    "Farm land allocation across named crop-management options with water, labor, tonne-demand, minimum-acreage, and diversity constraints",
)
