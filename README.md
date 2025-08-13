# LP Generation

A standardized framework for generating synthetic linear programming (LP) problem instances.

## Overview

This package provides:

- A unified interface for generating various types of LP problems
- Parameter sampling capabilities to generate realistic problem instances
- **Target variable count generation** - specify approximate number of variables
- Size-based generation (small, medium, large) for backward compatibility
- Easy extensibility for new problem types

## Problem Types

The package includes generators for 21 common LP problem types, all fully migrated to the standardized interface:

- Transportation
- Diet Problem
- Knapsack
- Portfolio Optimization
- Network Flow
- Production Planning
- Assignment
- Blending
- Airline Crew
- Cutting Stock
- Energy
- Facility Location
- Feed Blending
- Inventory
- Land Use
- Load Balancing
- Product Mix
- Project Selection
- Resource Allocation
- Scheduling
- Supply Chain

## Usage

### Basic Usage

```julia
using LPGeneration
using JuMP
using Clp  # or any other LP solver

# List available problem types
problem_types = list_problem_types()

# Get information about a problem type
info = problem_info(:transportation)

# Generate a problem with default parameters
model, params = generate_problem(:transportation)

# Generate a problem with specific parameters
params = Dict(:n_sources => 10, :n_destinations => 15)
model, params = generate_problem(:transportation, params)

# Solve the model
set_optimizer(model, Clp.Optimizer)
optimize!(model)
solution_summary(model)
```

### Target Variable Count Generation (Recommended)

```julia
# Sample parameters targeting approximately 100 variables
params = sample_parameters(:transportation, 100)  # Target ~100 variables
model, params = generate_problem(:transportation, params)

# Check actual variable count
println("Target: 100, Actual: $(num_variables(model))")

# Generate problems with different variable counts
for target_vars in [10, 50, 200, 500]
    params = sample_parameters(:knapsack, target_vars)
    model, params = generate_problem(:knapsack, params)
    println("Target: $target_vars, Actual: $(num_variables(model))")
end
```

### Legacy Size-based Generation

```julia
# Sample realistic parameters for a problem type (backward compatibility)
params = sample_parameters(:transportation, :medium)  # :small, :medium, or :large

# Generate a problem with sampled parameters
model, params = generate_problem(:transportation, params)
```

### Random Problem Generation

```julia
# Generate a random problem of any type targeting ~100 variables
model, problem_type, params = generate_random_problem(100)

# Check what problem type was selected and its variable count
println("Problem type: $problem_type")
println("Variables: $(num_variables(model))")

# Legacy: Generate a random problem by size
model, problem_type, params = generate_random_problem(:medium)  # :small, :medium, or :large

# Solve the model
set_optimizer(model, Clp.Optimizer)
optimize!(model)
solution_summary(model)
```

## Extending with New Problem Types

To add a new problem type:

1. Create a new file in `src/problem_types/your_problem.jl`
2. Implement the generator and parameter sampling functions
3. Register the problem type with the system

Example template:

```julia
using JuMP
using Random

function generate_your_problem(params::Dict=Dict(); seed::Int=0)
    # Implementation...
    return model, params
end

function calculate_your_problem_variable_count(params::Dict)
    # Calculate and return the number of variables this problem will have
    # based on the parameters
    return variable_count
end

function sample_your_problem_parameters(target_variables::Int; seed::Int=0)
    # Sample parameters to target approximately target_variables variables
    return params
end

function sample_your_problem_parameters(size::Symbol=:medium; seed::Int=0)
    # Legacy size-based parameter sampling (backward compatibility)
    target_map = Dict(:small => 150, :medium => 500, :large => 2000)
    return sample_your_problem_parameters(target_map[size]; seed=seed)
end

# Register the problem type
register_problem(
    :your_problem,
    generate_your_problem,
    sample_your_problem_parameters,
    "Description of your problem"
)
```

## Command Line Interface

The package includes a command-line script for generating problems:

```bash
# Generate a transportation problem with ~100 variables
julia generate_problem.jl transportation 100 problem.mps

# Generate a knapsack problem with ~50 variables and solve it
julia generate_problem.jl knapsack 50 --solve

# Generate a random problem with ~200 variables
julia generate_problem.jl random 200

# Legacy: Generate using size categories
julia generate_problem.jl transportation medium problem.mps

# List all available problem types
julia generate_problem.jl list
```

## Testing

Run the test suite to verify all problem generators:

```julia
using Pkg; Pkg.activate(".")
Pkg.test()
```

The test suite validates:
- All problem generators work correctly
- Target variable counts are achieved within ±10% tolerance
- Both new target-based and legacy size-based APIs work
- Generated problems are valid and can be solved