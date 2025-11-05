#!/usr/bin/env julia

# Manual test script for multi-commodity flow generator
# This can be run manually when Julia is available

using Pkg
Pkg.activate(".")

using SyntheticLPs
using JuMP

println("=" ^ 80)
println("Testing Multi-Commodity Flow Generator")
println("=" ^ 80)

println("\n1. Testing registration...")
problem_types = list_problem_types()
if :multi_commodity_flow in problem_types
    println("✓ multi_commodity_flow is registered")
else
    println("✗ multi_commodity_flow is NOT registered")
    exit(1)
end

println("\n2. Testing problem info...")
info = problem_info(:multi_commodity_flow)
println("Description: ", info[:description])

println("\n3. Testing parameter sampling with target variables...")
for target in [50, 100, 200, 500, 1000]
    params = sample_parameters(:multi_commodity_flow, target; seed=42)
    model, actual_params = generate_problem(:multi_commodity_flow, params; seed=42)

    actual_vars = num_variables(model)
    error_pct = abs(actual_vars - target) / target * 100

    println("Target: $target, Actual: $actual_vars, Error: $(round(error_pct, digits=2))%")
    println("  - Commodities: $(actual_params[:n_commodities])")
    println("  - Arcs: $(actual_params[:n_arcs])")
    println("  - Nodes: $(actual_params[:n_nodes])")
    println("  - Constraints: $(num_constraints(model, count_variable_in_set_constraints=true))")

    if error_pct > 10.0
        println("  ⚠ Warning: Error exceeds 10%")
    else
        println("  ✓ Within 10% tolerance")
    end
end

println("\n4. Testing size-based parameter sampling...")
for size in [:small, :medium, :large]
    params = sample_parameters(:multi_commodity_flow, size; seed=42)
    model, actual_params = generate_problem(:multi_commodity_flow, params; seed=42)

    actual_vars = num_variables(model)
    println("Size: $size, Variables: $actual_vars")
    println("  - Commodities: $(actual_params[:n_commodities])")
    println("  - Arcs: $(actual_params[:n_arcs])")
    println("  - Nodes: $(actual_params[:n_nodes])")
end

println("\n5. Testing solution status options...")
for status in [:feasible, :infeasible, :all]
    params = sample_parameters(:multi_commodity_flow, 100; seed=42)
    params[:solution_status] = status
    model, actual_params = generate_problem(:multi_commodity_flow, params; seed=42)

    println("Status: $status")
    println("  - Total demand: $(actual_params[:total_demand])")
    println("  - Total capacity: $(actual_params[:total_capacity])")
    println("  - Ratio (capacity/demand): $(round(actual_params[:total_capacity] / actual_params[:total_demand], digits=2))")
end

println("\n6. Testing reproducibility...")
params1 = sample_parameters(:multi_commodity_flow, 100; seed=123)
model1, _ = generate_problem(:multi_commodity_flow, params1; seed=123)

params2 = sample_parameters(:multi_commodity_flow, 100; seed=123)
model2, _ = generate_problem(:multi_commodity_flow, params2; seed=123)

if num_variables(model1) == num_variables(model2) &&
   num_constraints(model1, count_variable_in_set_constraints=true) ==
   num_constraints(model2, count_variable_in_set_constraints=true)
    println("✓ Reproducibility test passed")
else
    println("✗ Reproducibility test failed")
end

println("\n7. Testing model structure...")
params = sample_parameters(:multi_commodity_flow, 100; seed=42)
model, actual_params = generate_problem(:multi_commodity_flow, params; seed=42)

println("Model has:")
println("  - Variables: $(num_variables(model))")
println("  - Constraints: $(num_constraints(model, count_variable_in_set_constraints=true))")
println("  - Objective: $(objective_sense(model))")

# Check if objective is set
if objective_function(model) !== nothing
    println("  ✓ Objective function is set")
else
    println("  ✗ Objective function is NOT set")
end

println("\n" * "=" ^ 80)
println("All tests completed!")
println("=" ^ 80)
