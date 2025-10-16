# Example usage of SyntheticLPs

using JuMP
using Clp

using SyntheticLPs

# List available problem types
problem_types = list_problem_types()
println("Available problem types: ", problem_types)

# Example 1: Generate a transportation problem with default parameters
model, params = generate_problem(:transportation)
println("\nTransportation problem with defaults:")
println("  - ", params[:n_sources], " sources")
println("  - ", params[:n_destinations], " destinations")

# Example 2: Generate a diet problem with specific parameters
custom_params = Dict(
    :n_foods => 8,
    :n_nutrients => 4
)
model, params = generate_problem(:diet_problem, custom_params)
println("\nDiet problem with custom parameters:")
println("  - ", params[:n_foods], " foods")
println("  - ", params[:n_nutrients], " nutrients")

# Example 3: Sample parameters based on problem size
tpParams = sample_parameters(:transportation, :large)
model, params = generate_problem(:transportation, tpParams)
println("\nLarge transportation problem:")
println("  - ", params[:n_sources], " sources")
println("  - ", params[:n_destinations], " destinations")

# Example 4: Generate a random problem of any type
model, problem_type, params = generate_random_problem(:medium)
println("\nRandom problem of type: ", problem_type)

# Example 5: Solve a generated problem
model, params = generate_problem(:knapsack, sample_parameters(:knapsack, :small))
set_optimizer(model, Clp.Optimizer)
optimize!(model)
println("\nSolved knapsack problem:")
println("  - Objective value: ", objective_value(model))
println("  - Solution status: ", termination_status(model))
