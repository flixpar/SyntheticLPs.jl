using SyntheticLPs
using JuMP


# List all problem types
problem_types = list_problem_types()
println("Available problem types ($(length(problem_types))):")
for (i, type) in enumerate(sort(problem_types))
    info = problem_info(type)
    println("$i. $type - $(info[:description])")
end

# Try to sample parameters for each problem type
println("\nTesting parameter sampling for all problem types...")
for type in problem_types
    try
        params = sample_parameters(type, :small; seed=42)
        println("✓ $type - Successfully sampled parameters")
    catch e
        println("✗ $type - Error sampling parameters: $(e)")
    end
end

# Try to generate a problem for each problem type
println("\nTesting problem generation for all problem types...")
for type in problem_types
    try
        params = sample_parameters(type, :small; seed=42)
        model, actual_params = generate_problem(type, params; seed=42)
        println("✓ $type - Successfully generated problem with $(num_variables(model)) variables and $(num_constraints(model, count_variable_in_set_constraints=true)) constraints")
    catch e
        println("✗ $type - Error generating problem: $(e)")
    end
end

# Test random problem generation
println("\nTesting random problem generation...")
try
    model, type, params = generate_random_problem(:small; seed=42)
    println("✓ Successfully generated random problem of type $type with $(num_variables(model)) variables and $(num_constraints(model, count_variable_in_set_constraints=true)) constraints")
catch e
    println("✗ Error generating random problem: $(e)")
end