"""
Comprehensive test suite for problem instance generation.

Tests:
1. Successful generation (no errors)
2. Variable count within 15% of target
3. solution_status is obeyed (feasible/infeasible) when specified

Usage:
    julia --project=@. test/test_problem_instances.jl [problem_type]

Examples:
    julia --project=@. test/test_problem_instances.jl                # Test all problem types
    julia --project=@. test/test_problem_instances.jl transportation  # Test only transportation
"""

using Test
using JuMP
using Random
using HiGHS

using SyntheticLPs

# Number of instances to test per problem type
const NUM_INSTANCES = 50

# Target variable counts to test
const TARGET_SIZES = [50, 100, 250, 500, 1000]

"""
    check_feasibility(model::Model)

Check if a model is feasible using HiGHS solver.
Returns :Optimal if feasible, :Infeasible if infeasible, or the termination status.
"""
function check_feasibility(model::Model)
    # Set HiGHS as the solver
    set_optimizer(model, HiGHS.Optimizer)
    set_optimizer_attribute(model, "output_flag", false)
    set_optimizer_attribute(model, "time_limit", 60.0)  # 60 second timeout

    # Optimize the model
    optimize!(model)

    # Return termination status
    return termination_status(model)
end

"""
    test_problem_instance(problem_type::Symbol, target_vars::Int, seed::Int;
                         test_feasibility::Bool=true, solution_status::Union{Symbol,Nothing}=nothing)

Test a single problem instance.

Returns a Dict with test results:
- :success => true if generation succeeded, false otherwise
- :error => error message if generation failed
- :num_vars => actual number of variables
- :within_tolerance => true if within 15% of target
- :feasibility_correct => true if solution_status was obeyed (if tested)
"""
function test_problem_instance(problem_type::Symbol, target_vars::Int, seed::Int;
                               test_feasibility::Bool=true, solution_status::Union{Symbol,Nothing}=nothing)
    result = Dict{Symbol,Any}(
        :success => false,
        :num_vars => 0,
        :within_tolerance => false,
        :feasibility_correct => nothing
    )

    try
        # Sample parameters
        params = sample_parameters(problem_type, target_vars; seed=seed)

        # Add solution_status if specified
        if !isnothing(solution_status)
            params[:solution_status] = solution_status
        end

        # Generate problem
        model, actual_params = generate_problem(problem_type, params; seed=seed, relax_integer=true)

        # Check that model is valid
        @assert model isa JuMP.Model
        @assert num_variables(model) > 0
        @assert num_constraints(model, count_variable_in_set_constraints=true) > 0
        @assert objective_function(model) !== nothing

        # Record number of variables
        actual_vars = num_variables(model)
        result[:num_vars] = actual_vars

        # Check if within 15% tolerance
        error_percentage = abs(actual_vars - target_vars) / target_vars * 100
        result[:within_tolerance] = (error_percentage <= 15.0)

        # Check feasibility if requested
        if test_feasibility && !isnothing(solution_status) && solution_status != :all
            status = check_feasibility(model)
            expected_status = (solution_status == :feasible) ? :Optimal : :Infeasible

            # Also accept OPTIMAL for feasible and INFEASIBLE for infeasible
            if expected_status == :Optimal
                result[:feasibility_correct] = (status in [:Optimal, :OPTIMAL])
            else
                result[:feasibility_correct] = (status in [:Infeasible, :INFEASIBLE])
            end

            result[:actual_status] = status
            result[:expected_status] = expected_status
        end

        result[:success] = true

    catch e
        result[:success] = false
        result[:error] = sprint(showerror, e)
        result[:error_trace] = sprint(Base.show_backtrace, catch_backtrace())
    end

    return result
end

"""
    test_problem_type(problem_type::Symbol; verbose::Bool=false)

Test a single problem type with multiple instances.
"""
function test_problem_type(problem_type::Symbol; verbose::Bool=false)
    println("\n" * "="^80)
    println("Testing: $problem_type")
    println("="^80)

    # Statistics
    stats = Dict{Symbol,Any}(
        :total => 0,
        :successful => 0,
        :failed => 0,
        :within_tolerance => 0,
        :outside_tolerance => 0,
        :feasibility_tested => 0,
        :feasibility_correct => 0,
        :feasibility_incorrect => 0,
        :errors => []
    )

    # Test without solution_status constraint (faster tests)
    println("\nPhase 1: Basic generation tests (no feasibility checks)")
    println("-" * "^" * 79)

    instance_count = 0
    for target_vars in TARGET_SIZES
        num_tests_per_size = div(NUM_INSTANCES, length(TARGET_SIZES))
        for i in 1:num_tests_per_size
            instance_count += 1
            seed = 1000 * instance_count + target_vars

            result = test_problem_instance(problem_type, target_vars, seed;
                                          test_feasibility=false, solution_status=nothing)

            stats[:total] += 1

            if result[:success]
                stats[:successful] += 1
                if result[:within_tolerance]
                    stats[:within_tolerance] += 1
                else
                    stats[:outside_tolerance] += 1
                    if verbose
                        println("  ⚠ Instance $instance_count: $(result[:num_vars]) vars (target: $target_vars, $(round(abs(result[:num_vars] - target_vars) / target_vars * 100, digits=1))% error)")
                    end
                end
            else
                stats[:failed] += 1
                push!(stats[:errors], (instance=instance_count, error=result[:error]))
                if verbose
                    println("  ✗ Instance $instance_count failed: $(result[:error])")
                end
            end
        end
    end

    # Test with solution_status (smaller sample, includes feasibility checks)
    println("\nPhase 2: Feasibility tests (with solution_status)")
    println("-" * "^" * 79)

    # Only test if the problem type supports solution_status
    # We'll test a smaller number (10 instances of each status)
    num_feasibility_tests = 10

    for status in [:feasible, :infeasible]
        for i in 1:num_feasibility_tests
            instance_count += 1
            target_vars = rand([100, 250, 500])  # Random target size
            seed = 2000 * instance_count + Int(status == :feasible ? 1 : 2)

            result = test_problem_instance(problem_type, target_vars, seed;
                                          test_feasibility=true, solution_status=status)

            stats[:total] += 1

            if result[:success]
                stats[:successful] += 1
                if result[:within_tolerance]
                    stats[:within_tolerance] += 1
                else
                    stats[:outside_tolerance] += 1
                end

                # Check feasibility correctness
                if !isnothing(result[:feasibility_correct])
                    stats[:feasibility_tested] += 1
                    if result[:feasibility_correct]
                        stats[:feasibility_correct] += 1
                    else
                        stats[:feasibility_incorrect] += 1
                        if verbose
                            println("  ⚠ Instance $instance_count: Expected $(result[:expected_status]) but got $(result[:actual_status])")
                        end
                    end
                end
            else
                stats[:failed] += 1
                # Check if error is due to unsupported solution_status parameter
                if occursin("solution_status", result[:error]) ||
                   occursin("KeyError(:solution_status)", result[:error])
                    # This problem type doesn't support solution_status, skip remaining tests
                    if verbose
                        println("  ℹ Problem type doesn't support solution_status parameter, skipping feasibility tests")
                    end
                    break
                end
                push!(stats[:errors], (instance=instance_count, status=status, error=result[:error]))
                if verbose
                    println("  ✗ Instance $instance_count (status=$status) failed: $(result[:error])")
                end
            end
        end
    end

    # Print summary
    println("\n" * "="^80)
    println("Summary for $problem_type")
    println("="^80)
    println("Total instances:          $(stats[:total])")
    println("Successful:               $(stats[:successful]) ($(round(stats[:successful]/stats[:total]*100, digits=1))%)")
    println("Failed:                   $(stats[:failed]) ($(round(stats[:failed]/stats[:total]*100, digits=1))%)")
    println("Within 15% tolerance:     $(stats[:within_tolerance]) / $(stats[:successful]) ($(stats[:successful] > 0 ? round(stats[:within_tolerance]/stats[:successful]*100, digits=1) : 0.0)%)")

    if stats[:feasibility_tested] > 0
        println("Feasibility tests:        $(stats[:feasibility_tested])")
        println("  Correct:                $(stats[:feasibility_correct]) ($(round(stats[:feasibility_correct]/stats[:feasibility_tested]*100, digits=1))%)")
        println("  Incorrect:              $(stats[:feasibility_incorrect]) ($(round(stats[:feasibility_incorrect]/stats[:feasibility_tested]*100, digits=1))%)")
    else
        println("Feasibility tests:        Not tested (solution_status not supported)")
    end

    if !isempty(stats[:errors])
        println("\nErrors ($(length(stats[:errors])) total):")
        for (idx, err) in enumerate(stats[:errors][1:min(3, length(stats[:errors]))])
            println("  $(idx). Instance $(err.instance): $(first(split(err.error, '\n')))")
        end
        if length(stats[:errors]) > 3
            println("  ... and $(length(stats[:errors]) - 3) more errors")
        end
    end

    return stats
end

"""
    run_tests(problem_types::Vector{Symbol}=list_problem_types(); verbose::Bool=false)

Run comprehensive tests on specified problem types.
"""
function run_tests(problem_types::Vector{Symbol}=list_problem_types(); verbose::Bool=false)
    println("\n" * "="^80)
    println("SyntheticLPs Comprehensive Problem Instance Tests")
    println("="^80)
    println("Testing $(length(problem_types)) problem type(s)")
    println("Instances per type: ~$NUM_INSTANCES")
    println("HiGHS solver: $(HiGHS.Lib.Highs_version())")
    println()

    all_stats = Dict{Symbol,Any}()

    for problem_type in problem_types
        stats = test_problem_type(problem_type; verbose=verbose)
        all_stats[problem_type] = stats
    end

    # Overall summary
    println("\n" * "="^80)
    println("OVERALL SUMMARY")
    println("="^80)

    total_instances = sum(stats[:total] for stats in values(all_stats))
    total_successful = sum(stats[:successful] for stats in values(all_stats))
    total_failed = sum(stats[:failed] for stats in values(all_stats))
    total_within_tol = sum(stats[:within_tolerance] for stats in values(all_stats))
    total_feas_tested = sum(stats[:feasibility_tested] for stats in values(all_stats))
    total_feas_correct = sum(stats[:feasibility_correct] for stats in values(all_stats))

    println("Total instances:          $total_instances")
    println("Successful:               $total_successful ($(round(total_successful/total_instances*100, digits=1))%)")
    println("Failed:                   $total_failed ($(round(total_failed/total_instances*100, digits=1))%)")
    println("Within 15% tolerance:     $total_within_tol / $total_successful ($(total_successful > 0 ? round(total_within_tol/total_successful*100, digits=1) : 0.0)%)")

    if total_feas_tested > 0
        println("Feasibility tests:        $total_feas_tested")
        println("  Correct:                $total_feas_correct ($(round(total_feas_correct/total_feas_tested*100, digits=1))%)")
    end

    # List problem types with issues
    println("\nProblem types with failures:")
    for (ptype, stats) in all_stats
        if stats[:failed] > 0
            println("  • $ptype: $(stats[:failed]) failures")
        end
    end

    println("\nProblem types with tolerance issues (>15%):")
    for (ptype, stats) in all_stats
        if stats[:outside_tolerance] > 0 && stats[:successful] > 0
            pct = round(stats[:outside_tolerance] / stats[:successful] * 100, digits=1)
            println("  • $ptype: $(stats[:outside_tolerance]) / $(stats[:successful]) instances ($pct%)")
        end
    end

    println("\nProblem types with feasibility issues:")
    for (ptype, stats) in all_stats
        if stats[:feasibility_incorrect] > 0
            pct = round(stats[:feasibility_incorrect] / stats[:feasibility_tested] * 100, digits=1)
            println("  • $ptype: $(stats[:feasibility_incorrect]) / $(stats[:feasibility_tested]) tests ($pct%)")
        end
    end

    return all_stats
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    # Check for command-line arguments
    if length(ARGS) > 0
        # Test specific problem type(s)
        problem_types = [Symbol(arg) for arg in ARGS]

        # Validate problem types
        available = list_problem_types()
        invalid = filter(pt -> !(pt in available), problem_types)

        if !isempty(invalid)
            println("Error: Unknown problem type(s): $invalid")
            println("\nAvailable problem types:")
            for pt in sort(available)
                println("  • $pt")
            end
            exit(1)
        end

        verbose = "--verbose" in ARGS || "-v" in ARGS
        stats = run_tests(problem_types; verbose=verbose)
    else
        # Test all problem types
        verbose = "--verbose" in ARGS || "-v" in ARGS
        stats = run_tests(; verbose=verbose)
    end

    # Exit with appropriate code
    total_failed = sum(s[:failed] for s in values(stats))
    total_feas_incorrect = sum(s[:feasibility_incorrect] for s in values(stats))

    if total_failed > 0 || total_feas_incorrect > 0
        exit(1)
    end
end
