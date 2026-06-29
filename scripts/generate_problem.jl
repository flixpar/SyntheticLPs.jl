# Script to generate LP problems using SyntheticLPs module
# Usage:
#   julia --project=@. scripts/generate_problem.jl [problem] [target_variables] [output_file]
#
# `problem` is a category (e.g. `transportation`, uses its default variant), a
# `category/variant` reference (e.g. `portfolio/cvar`), `random`, or `list`.
#
# Example:
#   julia --project=@. scripts/generate_problem.jl transportation 100 problem.mps
#   julia --project=@. scripts/generate_problem.jl portfolio/cvar 100 problem.mps
#   julia --project=@. scripts/generate_problem.jl knapsack 50 --feasible --solve
#   julia --project=@. scripts/generate_problem.jl diet_problem 100 --infeasible
#   julia --project=@. scripts/generate_problem.jl knapsack 50 --bounds-to-constraints
#   julia --project=@. scripts/generate_problem.jl list

using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = dirname(@__DIR__))
Pkg.instantiate()

using SyntheticLPs

using JuMP
using HiGHS

# Parse command line arguments
problem_arg = length(ARGS) >= 1 ? ARGS[1] : "random"
target_variables = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 50
output_file = nothing
feasibility_status = unknown
seed = 0
bounds_to_constraints = false

# Parse optional arguments. `global` is required because this loop runs at the
# script's top level (soft scope), so assignments would otherwise create locals
# and silently leave these globals at their defaults.
for arg in ARGS[3:end]
    global feasibility_status, seed, output_file, bounds_to_constraints
    if arg == "--solve"
        # Handled later
    elseif arg == "--feasible"
        feasibility_status = feasible
    elseif arg == "--infeasible"
        feasibility_status = infeasible
    elseif arg == "--unknown"
        feasibility_status = unknown
    elseif arg == "--bounds-to-constraints"
        bounds_to_constraints = true
    elseif startswith(arg, "--seed=")
        seed = parse(Int, split(arg, "=")[2])
    elseif !startswith(arg, "--")
        output_file = arg
    end
end

if problem_arg == "list"
    println("Available problem categories (and variants):")
    for (i, category) in enumerate(sort(list_categories()))
        info = problem_info(category)
        println("  $i. $category - $(info[:description])")
        for v in info[:variants]
            marker = v == info[:default_variant] ? " (default)" : ""
            println("       • $category/$v$marker")
        end
    end
    exit(0)
elseif problem_arg == "random"
    println("Generating a random problem targeting ~$target_variables variables")
    println("Feasibility status: $feasibility_status")
    model, selected_ref, problem = generate_random_problem(target_variables; feasibility_status=feasibility_status, bounds_to_constraints=bounds_to_constraints, seed=seed)
    println("Problem selected: $selected_ref")
else
    # Accept a category (default variant) or an explicit `category/variant`.
    parts = split(problem_arg, '/')
    if length(parts) > 2
        println("Error: Invalid problem reference '$problem_arg'; expected 'category' or 'category/variant'")
        exit(1)
    end
    category = Symbol(parts[1])
    if !(category in list_categories())
        println("Error: Unknown problem category '$category'")
        println("Available categories: $(sort(list_categories()))")
        println("Use 'list' to see all categories and their variants.")
        exit(1)
    end
    if length(parts) == 2 && !(Symbol(parts[2]) in list_variants(category))
        println("Error: Unknown variant '$(parts[2])' for category '$category'")
        println("Available variants: $(list_variants(category))")
        exit(1)
    end
    ref = length(parts) == 2 ? ProblemVariant(category, Symbol(parts[2])) :
                               ProblemVariant(category)

    println("Generating $ref problem targeting ~$target_variables variables")
    println("Feasibility status: $feasibility_status")
    model, problem = generate_problem(ref, target_variables, feasibility_status, seed; bounds_to_constraints=bounds_to_constraints)
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