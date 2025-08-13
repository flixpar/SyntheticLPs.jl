# Template for implementing a problem type
# Replace PROBLEM_TYPE with the actual problem type name (e.g., transportation)

using JuMP
using Random
using Distributions

"""
    generate_PROBLEM_TYPE_problem(params::Dict=Dict(); seed::Int=0)

Generate a PROBLEM_TYPE problem instance.

# Arguments
- `params`: Dictionary of problem parameters
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- `model`: The JuMP model
- `params`: Dictionary of all parameters used (including defaults)
"""
function generate_PROBLEM_TYPE_problem(params::Dict=Dict(); seed::Int=0)
    # Set random seed
    Random.seed!(seed)
    
    # Extract parameters with defaults
    # Example: n_sources = get(params, :n_sources, 4)
    
    # Generate model
    model = Model()
    
    # Create variables, constraints, objective
    
    # Return model and full parameter set
    return model, params
end

"""
    sample_PROBLEM_TYPE_parameters(size::Symbol=:medium; seed::Int=0)

Sample realistic parameters for a PROBLEM_TYPE problem.

# Arguments
- `size`: Symbol specifying the problem size (:small, :medium, :large)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_PROBLEM_TYPE_parameters(size::Symbol=:medium; seed::Int=0)
    Random.seed!(seed)
    
    params = Dict{Symbol, Any}()
    
    # Set size-dependent parameters
    if size == :small
        # Example: params[:n_sources] = rand(2:4)
    elseif size == :medium
        # Example: params[:n_sources] = rand(5:10)
    elseif size == :large
        # Example: params[:n_sources] = rand(11:20)
    else
        error("Unknown size: $size. Must be :small, :medium, or :large")
    end
    
    return params
end

# Register the problem type
register_problem(
    :PROBLEM_TYPE,
    generate_PROBLEM_TYPE_problem,
    sample_PROBLEM_TYPE_parameters,
    "Description of the PROBLEM_TYPE problem"
)