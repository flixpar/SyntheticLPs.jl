# SyntheticLPs.jl

A standardized framework for generating synthetic linear programming (LP) problem instances. The goal is to generate problems that are highly realistic and can be used to test and develop LP solvers.

## Overview

This package provides:

- A unified interface for generating various types of LP problems using multiple dispatch
- Problem generators implemented as concrete types inheriting from `ProblemGenerator`
- Controllable problem feasibility (feasible, infeasible, or unknown)
- Target variable count generation - specify approximate number of variables
- Deterministic problem generation with reproducible seeds
- Easy extensibility for new problem types

## Problem Types

The package includes generators for many common LP problem types, all unified with a standardized interface:

### General Problems
- Diet Problem
- Knapsack
- Portfolio Optimization
- Network Flow
- Multi-Commodity Flow
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
- Crop Planning
- Telecom Network Design

### Transportation & Logistics
- **Basic Transportation** - Classic source-to-destination shipping optimization
- **Vehicle Routing** - Capacitated vehicle routing with delivery optimization
- **Warehouse Location & Sizing** - Combined location and capacity decisions with multi-echelon flows
- **Hub Location** - Hub-and-spoke network design with economies of scale
- **Transshipment** - Intermediate storage and routing optimization
- **Last Mile Delivery** - Urban delivery with time windows and congestion
- **Cross-Docking** - Transfer optimization with minimal storage time
- **Cargo Loading** - Container/truck loading with weight and volume constraints

## Usage

### Basic Usage

```julia
using SyntheticLPs
using JuMP
using Clp  # or any other LP solver

# List available problem types
problem_types = list_problem_types()

# Get information about a problem type
info = problem_info(:transportation)

# Generate a problem with target variable count
model, problem = generate_problem(:transportation, 100, unknown, 0)

# The problem instance contains all the generated data
println("Number of sources: ", problem.n_sources)
println("Number of destinations: ", problem.n_destinations)

# Solve the model
set_optimizer(model, Clp.Optimizer)
optimize!(model)
solution_summary(model)
```

### Feasibility Control

```julia
# Generate a guaranteed feasible problem
model, problem = generate_problem(:transportation, 100, feasible, 0)

# Generate a guaranteed infeasible problem
model, problem = generate_problem(:diet_problem, 100, infeasible, 0)

# Generate a problem with unknown feasibility (randomized)
model, problem = generate_problem(:portfolio, 100, unknown, 0)
```

### Reproducible Generation with Seeds

```julia
# Generate the same problem twice with the same seed
seed = 12345
model1, problem1 = generate_problem(:knapsack, 50, unknown, seed)
model2, problem2 = generate_problem(:knapsack, 50, unknown, seed)

# These will be identical
@assert num_variables(model1) == num_variables(model2)
@assert problem1.n_items == problem2.n_items
```

### Random Problem Generation

```julia
# Generate a random problem of any type targeting ~100 variables
model, problem_type, problem = generate_random_problem(100)

# Check what problem type was selected and its variable count
println("Problem type: $problem_type")
println("Variables: $(num_variables(model))")

# Generate with feasibility control
model, problem_type, problem = generate_random_problem(200; feasibility_status=feasible)

# Solve the model
set_optimizer(model, Clp.Optimizer)
optimize!(model)
solution_summary(model)
```

## Extending with New Problem Types

To add a new problem type:

1. Create a new file in `src/problem_types/your_problem.jl`
2. Define a struct inheriting from `ProblemGenerator`
3. Implement the constructor and `build_model` function
4. Register the problem type

Example template:

```julia
using JuMP
using Random

"""
    YourProblem <: ProblemGenerator

Generator for your custom problem type.

# Fields
- `field1::Type1`: Description
- `field2::Type2`: Description
"""
struct YourProblem <: ProblemGenerator
    field1::Type1
    field2::Type2
    # ... store all generated data needed to build the model
end

"""
    YourProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a problem instance.

# Arguments
- `target_variables`: Target number of variables
- `feasibility_status`: Desired feasibility (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function YourProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Sample all parameters based on target_variables
    # Generate all deterministic data
    # Handle feasibility status
    # ...

    return YourProblem(field1_value, field2_value, ...)
end

"""
    build_model(prob::YourProblem)

Build a JuMP model from the problem instance. This function must be deterministic.
"""
function build_model(prob::YourProblem)
    model = Model()

    # Define variables
    # Define constraints
    # Define objective

    return model
end

# Register the problem type
register_problem(
    :your_problem,
    YourProblem,
    "Description of your problem"
)
```

Key principles:
- The struct stores ALL data needed to deterministically build the model
- ALL randomness goes in the constructor
- `build_model` must be completely deterministic (no RNG calls)
- Handle `feasible`, `infeasible`, and `unknown` feasibility statuses

## Command Line Interface

The package includes a command-line script for generating problems:

```bash
# Generate a transportation problem with ~100 variables
julia --project=@. scripts/generate_problem.jl transportation 100 problem.mps

# Generate a feasible knapsack problem with ~50 variables and solve it
julia --project=@. scripts/generate_problem.jl knapsack 50 --feasible --solve

# Generate an infeasible diet problem with ~100 variables
julia --project=@. scripts/generate_problem.jl diet_problem 100 --infeasible output.mps

# Generate a random problem with ~200 variables
julia --project=@. scripts/generate_problem.jl random 200

# Use a specific seed for reproducibility
julia --project=@. scripts/generate_problem.jl portfolio 150 --seed=12345

# List all available problem types
julia --project=@. scripts/generate_problem.jl list
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
- Generated problems are valid and can be solved


## License

SyntheticLPs.jl  
Copyright (C) 2025  Felix Parker

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details.

The full text of the license is available in the [LICENSE](LICENSE) file.
