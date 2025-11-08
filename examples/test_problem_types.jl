using SyntheticLPs
using JuMP


# List all problem types
problem_types = list_problem_types()
println("Available problem types ($(length(problem_types))):")
for (i, type) in enumerate(sort(problem_types))
    info = problem_info(type)
    println("$i. $type - $(info[:description])")
end

# Test problem generation for each problem type with different target variable counts
println("\nTesting problem generation for all problem types...")
for type in problem_types
    try
        model, problem = generate_problem(type, 50, unknown, 42)
        println("✓ $type - Successfully generated problem with $(num_variables(model)) variables and $(num_constraints(model, count_variable_in_set_constraints=true)) constraints")
    catch e
        println("✗ $type - Error generating problem: $(e)")
    end
end

# Test feasible problem generation
println("\nTesting feasible problem generation...")
for type in problem_types
    try
        model, problem = generate_problem(type, 50, feasible, 42)
        println("✓ $type - Successfully generated feasible problem")
    catch e
        println("✗ $type - Error generating feasible problem: $(e)")
    end
end

# Test infeasible problem generation
println("\nTesting infeasible problem generation...")
for type in problem_types
    try
        model, problem = generate_problem(type, 50, infeasible, 42)
        println("✓ $type - Successfully generated infeasible problem")
    catch e
        println("✗ $type - Error generating infeasible problem: $(e)")
    end
end

# Test random problem generation
println("\nTesting random problem generation...")
try
    model, problem_sym, problem = generate_random_problem(50; feasibility_status=unknown, seed=42)
    println("✓ Successfully generated random problem of type $problem_sym with $(num_variables(model)) variables and $(num_constraints(model, count_variable_in_set_constraints=true)) constraints")
catch e
    println("✗ Error generating random problem: $(e)")
end