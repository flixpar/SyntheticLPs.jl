# Script to generate LP problems using SyntheticLPs module
# Usage: 
#   julia generate_problem.jl [problem_type] [target_variables] [output_file]
#
# Example:
#   julia --project=@. generate_problem.jl transportation 100 problem.mps

using SyntheticLPs

using JuMP
using HiGHS

# Parse command line arguments
problem_type = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :random
param_arg = length(ARGS) >= 2 ? ARGS[2] : "50"
output_file = length(ARGS) >= 3 ? ARGS[3] : nothing

# Parse the parameter - could be either integer target variables or legacy size
function parse_parameter(param_str::String)
    # Try to parse as integer first
    try
        target_vars = parse(Int, param_str)
        return target_vars, :target_variables
    catch
        # If parsing as integer fails, treat as legacy size symbol
        size_sym = Symbol(param_str)
        if size_sym in [:small, :medium, :large]
            return size_sym, :size
        else
            error("Invalid parameter '$param_str'. Must be an integer (target variables) or :small, :medium, :large")
        end
    end
end

param_value, param_type = parse_parameter(param_arg)

# Get the list of available problem types
available_types = list_problem_types()

if problem_type == :list
    println("Available problem types:")
    for (i, type) in enumerate(available_types)
        info = problem_info(type)
        println("  $i. $type - $(info[:description])")
    end
    exit(0)
elseif problem_type == :random
    if param_type == :target_variables
        println("Generating a random problem targeting ~$param_value variables")
        model, selected_type, params = generate_random_problem(param_value)
    else
        println("Generating a random problem of size: $param_value")
        model, selected_type, params = generate_random_problem(param_value)
    end
    println("Problem type selected: $selected_type")
else
    if !(problem_type in available_types)
        println("Error: Unknown problem type '$problem_type'")
        println("Available types: $available_types")
        println("Use 'list' to see details of all problem types")
        exit(1)
    end
    
    if param_type == :target_variables
        println("Generating $problem_type problem targeting ~$param_value variables")
        params = sample_parameters(problem_type, param_value)
    else
        println("Generating $problem_type problem of size: $param_value")
        params = sample_parameters(problem_type, param_value)
    end
    model, params = generate_problem(problem_type, params)
end

# Print summary
println("\nProblem summary:")
println("  Variables: $(num_variables(model))")
println("  Constraints: $(num_constraints(model; count_variable_in_set_constraints=false))")

# Optimize if requested
if "--solve" in ARGS
    println("\nSolving problem...")
    set_optimizer(model, HiGHS.Optimizer)
    optimize!(model)
    
    println("Solution status: $(termination_status(model))")
    println("Objective value: $(objective_value(model))")
end

# Export model if output file provided
if output_file !== nothing
    println("\nWriting problem to: $output_file")
    write_to_file(model, output_file)
end