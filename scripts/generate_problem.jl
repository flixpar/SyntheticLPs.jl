# Script to generate LP problems using SyntheticLPs module
# Usage:
#   julia --project=@. scripts/generate_problem.jl [problem_type] [target_variables] [output_file]
#
# Example:
#   julia --project=@. scripts/generate_problem.jl transportation 100 problem.mps
#   julia --project=@. scripts/generate_problem.jl knapsack 50 --feasible --solve
#   julia --project=@. scripts/generate_problem.jl diet_problem 100 --infeasible

using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = dirname(@__DIR__))
Pkg.instantiate()

using SyntheticLPs

using JuMP
using HiGHS

# Parse command line arguments
problem_type = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :random
target_variables = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 50
output_file = nothing
feasibility_status = unknown
seed = 0

# Parse optional arguments
for arg in ARGS[3:end]
    if arg == "--solve"
        # Handled later
    elseif arg == "--feasible"
        feasibility_status = feasible
    elseif arg == "--infeasible"
        feasibility_status = infeasible
    elseif arg == "--unknown"
        feasibility_status = unknown
    elseif startswith(arg, "--seed=")
        seed = parse(Int, split(arg, "=")[2])
    elseif !startswith(arg, "--")
        output_file = arg
    end
end

# Get the list of available problem types
available_types = list_problem_types()

if problem_type == :list
    println("Available problem types:")
    for (i, type) in enumerate(sort(available_types))
        info = problem_info(type)
        println("  $i. $type - $(info[:description])")
    end
    exit(0)
elseif problem_type == :random
    println("Generating a random problem targeting ~$target_variables variables")
    println("Feasibility status: $feasibility_status")
    model, selected_type, problem = generate_random_problem(target_variables; feasibility_status=feasibility_status, seed=seed)
    println("Problem type selected: $selected_type")
else
    if !(problem_type in available_types)
        println("Error: Unknown problem type '$problem_type'")
        println("Available types: $(sort(available_types))")
        println("Use 'list' to see details of all problem types")
        exit(1)
    end

    println("Generating $problem_type problem targeting ~$target_variables variables")
    println("Feasibility status: $feasibility_status")
    model, problem = generate_problem(problem_type, target_variables, feasibility_status, seed)
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