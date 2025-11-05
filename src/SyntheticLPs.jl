module SyntheticLPs

using JuMP
using Random
using Distributions

export generate_problem
export sample_parameters
export list_problem_types
export problem_info
export generate_random_problem
export register_problem

# Registration system
# Dictionary to store problem type metadata
const LP_REGISTRY = Dict{Symbol, Dict{Symbol, Any}}()

"""
    register_problem(type_sym::Symbol, generator_fn::Function, sampler_fn::Function, description::String)

Register a problem generator with the system.
"""
function register_problem(type_sym::Symbol, generator_fn::Function, sampler_fn::Function, description::String)
    LP_REGISTRY[type_sym] = Dict(
        :generator => generator_fn,
        :sampler => sampler_fn,
        :description => description
    )
end

"""
    get_generator(problem_type::Symbol)

Get the generator function for a problem type.
"""
function get_generator(problem_type::Symbol)
    if !haskey(LP_REGISTRY, problem_type)
        error("Unknown problem type: $problem_type")
    end
    return LP_REGISTRY[problem_type][:generator]
end

"""
    get_sampler(problem_type::Symbol)

Get the sampler function for a problem type.
"""
function get_sampler(problem_type::Symbol)
    if !haskey(LP_REGISTRY, problem_type)
        error("Unknown problem type: $problem_type")
    end
    return LP_REGISTRY[problem_type][:sampler]
end

"""
    generate_problem(problem_type::Symbol, params::Dict=Dict(); seed::Int=0)

Generate a linear programming problem of the specified type with given parameters.

# Arguments
- `problem_type`: Symbol specifying the problem type (e.g., :transportation, :diet)
- `params`: Dictionary of problem-specific parameters (optional)
- `relax_integer`: Whether to relax integer constraints (default: true)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- `model`: The JuMP model
- `params`: Dictionary of all parameters used
"""
function generate_problem(problem_type::Symbol, params::Dict=Dict(); relax_integer::Bool=true, seed::Int=0)
    if !haskey(LP_REGISTRY, problem_type)
        error("Unknown problem type: $problem_type. Use list_problem_types() to see available types.")
    end
    
    generator = get_generator(problem_type)
    model, params = generator(params; seed=seed)

    if relax_integer
        relax_integrality(model)
    end

    return model, params
end

"""
    sample_parameters(problem_type::Symbol, target_variables::Int; seed::Int=0)

Sample realistic parameters for the specified problem type targeting approximately the specified number of variables.

# Arguments
- `problem_type`: Symbol specifying the problem type (e.g., :transportation, :diet)
- `target_variables`: Target number of variables in the LP formulation (approximately within ±10%)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_parameters(problem_type::Symbol, target_variables::Int; seed::Int=0)
    if !haskey(LP_REGISTRY, problem_type)
        error("Unknown problem type: $problem_type. Use list_problem_types() to see available types.")
    end
    
    sampler = get_sampler(problem_type)
    return sampler(target_variables; seed=seed)
end

"""
    sample_parameters(problem_type::Symbol, size::Symbol; seed::Int=0)

Legacy function for backward compatibility. Sample realistic parameters for the specified problem type using size categories.

# Arguments
- `problem_type`: Symbol specifying the problem type (e.g., :transportation, :diet)
- `size`: Symbol specifying the problem size (:small, :medium, :large)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_parameters(problem_type::Symbol, size::Symbol; seed::Int=0)
    if !haskey(LP_REGISTRY, problem_type)
        error("Unknown problem type: $problem_type. Use list_problem_types() to see available types.")
    end
    
    # Map size categories to approximate target variable counts
    target_map = Dict(
        :small => 150,    # 50-250 variables
        :medium => 500,   # 250-1000 variables
        :large => 2000    # 1000-10000 variables
    )
    
    if !haskey(target_map, size)
        error("Unknown size: $size. Must be :small, :medium, or :large")
    end
    
    return sample_parameters(problem_type, target_map[size]; seed=seed)
end

"""
    list_problem_types()

List all available problem types.

# Returns
- Vector of symbols representing available problem types
"""
function list_problem_types()
    return collect(keys(LP_REGISTRY))
end

"""
    problem_info(problem_type::Symbol)

Get information about a specific problem type.

# Arguments
- `problem_type`: Symbol specifying the problem type

# Returns
- Dictionary with problem information
"""
function problem_info(problem_type::Symbol)
    if !haskey(LP_REGISTRY, problem_type)
        error("Unknown problem type: $problem_type. Use list_problem_types() to see available types.")
    end
    
    return Dict(
        :type => problem_type,
        :description => LP_REGISTRY[problem_type][:description]
    )
end

"""
    generate_random_problem(target_variables::Int; relax_integer::Bool=true, seed::Int=0)

Generate a random LP problem of a randomly selected type targeting approximately the specified number of variables.

# Arguments
- `target_variables`: Target number of variables in the LP formulation (approximately within ±10%)
- `relax_integer`: Whether to relax integer constraints (default: true)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- `model`: The JuMP model
- `problem_type`: Symbol indicating which problem type was selected
- `params`: Dictionary of all parameters used
"""
function generate_random_problem(target_variables::Int; relax_integer::Bool=true, seed::Int=0)
    Random.seed!(seed)
    
    problem_types = list_problem_types()
    if isempty(problem_types)
        error("No problem types registered. Include problem type files first.")
    end
    
    problem_type = rand(problem_types)
    
    params = sample_parameters(problem_type, target_variables; seed=seed)
    model, actual_params = generate_problem(problem_type, params; relax_integer=relax_integer, seed=seed)
    
    return model, problem_type, actual_params
end

"""
    generate_random_problem(size::Symbol; relax_integer::Bool=true, seed::Int=0)

Legacy function for backward compatibility. Generate a random LP problem of a randomly selected type using size categories.

# Arguments
- `size`: Symbol specifying the problem size (:small, :medium, :large)
- `relax_integer`: Whether to relax integer constraints (default: true)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- `model`: The JuMP model
- `problem_type`: Symbol indicating which problem type was selected
- `params`: Dictionary of all parameters used
"""
function generate_random_problem(size::Symbol; relax_integer::Bool=true, seed::Int=0)
    # Map size categories to approximate target variable counts
    target_map = Dict(
        :small => 150,    # 50-250 variables
        :medium => 500,   # 250-1000 variables
        :large => 2000    # 1000-10000 variables
    )
    
    if !haskey(target_map, size)
        error("Unknown size: $size. Must be :small, :medium, or :large")
    end
    
    return generate_random_problem(target_map[size]; relax_integer=relax_integer, seed=seed)
end

# Include all problem generators
# The problem type files will use the register_problem function from this module
include("problem_types/airline_crew.jl")
include("problem_types/assignment.jl")
include("problem_types/blending.jl")
include("problem_types/cutting_stock.jl")
include("problem_types/diet_problem.jl")
include("problem_types/energy.jl")
include("problem_types/facility_location.jl")
include("problem_types/feed_blending.jl")
include("problem_types/inventory.jl")
include("problem_types/knapsack.jl")
include("problem_types/land_use.jl")
include("problem_types/load_balancing.jl")
include("problem_types/network_flow.jl")
include("problem_types/portfolio.jl")
include("problem_types/product_mix.jl")
include("problem_types/production_planning.jl")
include("problem_types/project_selection.jl")
include("problem_types/resource_allocation.jl")
include("problem_types/scheduling.jl")
include("problem_types/supply_chain.jl")
include("problem_types/telecom_network_design.jl")
include("problem_types/transportation.jl")

end # module