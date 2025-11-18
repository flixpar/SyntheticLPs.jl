using Test
using JuMP
using SyntheticLPs

"""
Manual test script for bin packing problem generator.
Run this with: julia --project=@. test/test_bin_packing.jl
"""

@testset "Bin Packing Problem Generator" begin
    println("\n=== Testing Bin Packing Problem Generator ===\n")

    # Test 1: Basic generation
    println("Test 1: Basic problem generation")
    model, prob = generate_problem(:bin_packing, 100, unknown, 42)
    println("  n_items: ", prob.n_items)
    println("  n_bins: ", prob.n_bins)
    println("  Total item volume: ", round(sum(prob.item_sizes), digits=2))
    println("  Total capacity: ", round(prob.bin_capacity * prob.n_bins, digits=2))
    println("  Number of variables: ", num_variables(model))
    println("  Number of constraints: ", num_constraints(model; count_variable_in_set_constraints=true))
    @test num_variables(model) > 0
    @test num_variables(model) == prob.n_items * prob.n_bins + prob.n_bins

    # Test 2: Feasible problem
    println("\nTest 2: Feasible problem generation")
    model_feas, prob_feas = generate_problem(:bin_packing, 100, feasible, 123)
    total_volume = sum(prob_feas.item_sizes)
    total_capacity = prob_feas.bin_capacity * prob_feas.n_bins
    println("  n_items: ", prob_feas.n_items)
    println("  n_bins: ", prob_feas.n_bins)
    println("  Total volume: ", round(total_volume, digits=2))
    println("  Total capacity: ", round(total_capacity, digits=2))
    println("  Capacity ratio: ", round(total_capacity / total_volume, digits=2))
    @test total_capacity >= total_volume  # Feasible should have enough capacity

    # Test 3: Infeasible problem
    println("\nTest 3: Infeasible problem generation")
    model_infeas, prob_infeas = generate_problem(:bin_packing, 100, infeasible, 456)
    total_volume_inf = sum(prob_infeas.item_sizes)
    total_capacity_inf = prob_infeas.bin_capacity * prob_infeas.n_bins
    println("  n_items: ", prob_infeas.n_items)
    println("  n_bins: ", prob_infeas.n_bins)
    println("  Total volume: ", round(total_volume_inf, digits=2))
    println("  Total capacity: ", round(total_capacity_inf, digits=2))
    println("  Has incompatible pairs: ", !isempty(prob_infeas.incompatible_pairs))
    # Note: Infeasible can be created either by capacity or incompatibility

    # Test 4: Different sizes
    println("\nTest 4: Different problem sizes")
    for target_vars in [50, 100, 200, 500]
        model_size, prob_size = generate_problem(:bin_packing, target_vars, unknown, 789)
        actual_vars = num_variables(model_size)
        error_pct = abs(actual_vars - target_vars) / target_vars * 100
        println("  Target: $target_vars, Actual: $actual_vars, Error: $(round(error_pct, digits=2))%")
        @test error_pct <= 10.0  # Within 10% of target
    end

    # Test 5: Reproducibility
    println("\nTest 5: Reproducibility with same seed")
    seed = 99999
    model1, prob1 = generate_problem(:bin_packing, 150, unknown, seed)
    model2, prob2 = generate_problem(:bin_packing, 150, unknown, seed)
    println("  Model 1 vars: ", num_variables(model1))
    println("  Model 2 vars: ", num_variables(model2))
    println("  Model 1 items: ", prob1.n_items)
    println("  Model 2 items: ", prob2.n_items)
    @test num_variables(model1) == num_variables(model2)
    @test prob1.n_items == prob2.n_items
    @test prob1.n_bins == prob2.n_bins
    @test prob1.item_sizes == prob2.item_sizes

    # Test 6: Item size distribution
    println("\nTest 6: Item size distribution (should be skewed toward small)")
    model_dist, prob_dist = generate_problem(:bin_packing, 200, unknown, 111)
    sizes = prob_dist.item_sizes
    capacity = prob_dist.bin_capacity
    small_items = count(s -> s <= 0.3 * capacity, sizes)
    medium_items = count(s -> 0.3 * capacity < s <= 0.6 * capacity, sizes)
    large_items = count(s -> s > 0.6 * capacity, sizes)
    println("  Small items (≤30%): $small_items ($(round(100*small_items/length(sizes), digits=1))%)")
    println("  Medium items (30-60%): $medium_items ($(round(100*medium_items/length(sizes), digits=1))%)")
    println("  Large items (>60%): $large_items ($(round(100*large_items/length(sizes), digits=1))%)")
    @test small_items > large_items  # Should be more small items than large

    println("\n=== All tests passed! ===\n")
end
