# Example usage of SyntheticLPs

using JuMP
using HiGHS

using SyntheticLPs

# List available problem types
problem_types = list_problem_types()
println("Available problem types: ", problem_types)

# Example 1: Generate a transportation problem with default target size
model, problem = generate_problem(:transportation, 100)
println("\nTransportation problem with ~100 variables:")
println("  - ", problem.n_sources, " sources")
println("  - ", problem.n_destinations, " destinations")

# Example 2: Generate a diet problem with specific target size
model, problem = generate_problem(:diet_problem, 50)
println("\nDiet problem with ~50 variables:")
println("  - ", problem.n_foods, " foods")
println("  - ", problem.n_nutrients, " nutrients")

# Example 3: Generate a large transportation problem
model, problem = generate_problem(:transportation, 500)
println("\nLarge transportation problem with ~500 variables:")
println("  - ", problem.n_sources, " sources")
println("  - ", problem.n_destinations, " destinations")

# Example 4: Generate a random problem of any type
model, problem_type, problem = generate_random_problem(200)
println("\nRandom problem of type: ", problem_type, " with ~200 variables")

# Example 5: Generate a feasible knapsack problem and solve it
model, problem = generate_problem(:knapsack, 75, feasible, 42)
set_optimizer(model, HiGHS.Optimizer)
optimize!(model)
println("\nSolved feasible knapsack problem:")
println("  - Objective value: ", objective_value(model))
println("  - Solution status: ", termination_status(model))

# Example 6: Generate an infeasible problem for testing
model, problem = generate_problem(:diet_problem, 100, infeasible, 123)
println("\nGenerated infeasible diet problem with ~100 variables")
