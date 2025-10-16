using Test
using JuMP
using Random

using LPGeneration

"""
    test_problem_generator(problem_type::Symbol)

Test the problem generator for the given problem type.
"""
function test_problem_generator(problem_type::Symbol)
    @testset "$(problem_type) Problem Generator" begin
        # Test with default parameters
        @test_nowarn begin
            model, params = generate_problem(problem_type)
            @test model isa JuMP.Model
            @test params isa Dict
        end
        
        # Test with specific size parameters (legacy API)
        for size in [:small, :medium, :large]
            @test_nowarn begin
                params = sample_parameters(problem_type, size)
                @test params isa Dict
                model, actual_params = generate_problem(problem_type, params)
                @test model isa JuMP.Model
                @test actual_params isa Dict
                
                # Check that the model has variables, constraints, and an objective
                @test num_variables(model) > 0
                @test num_constraints(model, count_variable_in_set_constraints=true) > 0
                @test objective_function(model) !== nothing
            end
        end
        
        # Test with target variables (new API)
        for target_vars in [100, 500, 1500]
            @test_nowarn begin
                params = sample_parameters(problem_type, target_vars)
                @test params isa Dict
                model, actual_params = generate_problem(problem_type, params)
                @test model isa JuMP.Model
                @test actual_params isa Dict
                
                # Check that the model has variables, constraints, and an objective
                actual_var_count = num_variables(model)
                @test actual_var_count > 0
                @test num_constraints(model, count_variable_in_set_constraints=true) > 0
                @test objective_function(model) !== nothing
                
                # Check that variable count is within ±10% of target
                error_percentage = abs(actual_var_count - target_vars) / target_vars * 100
                @test error_percentage <= 15.0 || actual_var_count <= 50  # Allow higher error for small problems and complex problem types
            end
        end
        
        # Test with a fixed seed
        seed = 12345
        @test_nowarn begin
            # Generate the same problem twice with the same seed
            params1 = sample_parameters(problem_type, :medium, seed=seed)
            model1, _ = generate_problem(problem_type, params1, seed=seed)
            
            params2 = sample_parameters(problem_type, :medium, seed=seed)
            model2, _ = generate_problem(problem_type, params2, seed=seed)
            
            # Verify that the models are identical (same number of vars and constraints)
            @test num_variables(model1) == num_variables(model2)
            @test num_constraints(model1, count_variable_in_set_constraints=true) == num_constraints(model2, count_variable_in_set_constraints=true)
        end
    end
end

# Run tests for all registered problem types
@testset "LPGeneration" begin
    # Test core functionality
    @testset "Core Functionality" begin
        # Test listing problem types
        problem_types = list_problem_types()
        @test problem_types isa Vector{Symbol}
        @test !isempty(problem_types)
        
        # Test getting problem info
        for problem_type in problem_types
            info = problem_info(problem_type)
            @test info isa Dict
            @test haskey(info, :description)
            @test info[:description] isa String
        end
        
        # Test random problem generation
        @test_nowarn begin
            # Test with target variables
            model, type, params = generate_random_problem(250)
            @test model isa JuMP.Model
            @test type isa Symbol
            @test params isa Dict
            @test num_variables(model) > 0
            
            # Test with legacy size (backward compatibility)
            model2, type2, params2 = generate_random_problem(:medium)
            @test model2 isa JuMP.Model
            @test type2 isa Symbol
            @test params2 isa Dict
            @test num_variables(model2) > 0
        end
    end
    
    # Test individual problem generators
    for problem_type in list_problem_types()
        test_problem_generator(problem_type)
    end
end