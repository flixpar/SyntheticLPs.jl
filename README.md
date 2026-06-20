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

- Transportation
- Diet Problem
- Knapsack
- Portfolio Optimization (CVaR with sector, region, factor, and turnover constraints)
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

## Usage

### Basic Usage

```julia
using SyntheticLPs
using JuMP
using Clp  # or any other LP solver

# List available problem categories
categories = list_problem_types()   # alias for list_categories()

# Get information about a category (its description, variants, default variant)
info = problem_info(:transportation)

# Generate a problem with target variable count (uses the category's default variant)
model, problem = generate_problem(:transportation, 100, unknown, 0)

# The problem instance contains all the generated data
println("Number of sources: ", problem.n_sources)
println("Number of destinations: ", problem.n_destinations)

# Solve the model
set_optimizer(model, Clp.Optimizer)
optimize!(model)
solution_summary(model)
```

### Categories and Variants

Each problem domain is a **category** (e.g. `:transportation`) that groups one or
more **variants** — concrete formulations with their own data generation and
model. A `ProblemVariant` names one variant of one category and prints as
`category/variant`.

```julia
# Every registered category and variant
list_categories()                       # [:airline_crew, :assignment, ...]
list_variants(:portfolio)               # [:cvar]
list_problems()                         # [ProblemVariant(:airline_crew, :standard), ...]

# Select a specific variant — three equivalent forms
model, problem = generate_problem(:portfolio, 100, unknown, 0; variant=:cvar)
model, problem = generate_problem(ProblemVariant(:portfolio, :cvar), 100, unknown, 0)
model, problem = generate_problem(ProblemVariant("portfolio/cvar"), 100, unknown, 0)

# Variant-level metadata
problem_info(:portfolio, :cvar)         # Dict with :description, :type, ...
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
# Generate a random problem of any variant targeting ~100 variables
model, ref, problem = generate_random_problem(100)

# `ref` is a ProblemVariant (prints as `category/variant`)
println("Problem: $ref")           # e.g. "transportation/standard"
println("Variables: $(num_variables(model))")

# Generate with feasibility control
model, ref, problem = generate_random_problem(200; feasibility_status=feasible)

# Solve the model
set_optimizer(model, Clp.Optimizer)
optimize!(model)
solution_summary(model)
```

### Batch Dataset Generation

To build a whole dataset of LP instances (e.g. for training an ML model), use
`generate_dataset`. It repeatedly samples a problem type and a target variable
count, builds each model, and optionally solves it to filter out low-quality
instances. When `output_dir` is set, each kept instance is written to disk
along with a `manifest.json` describing the run; metadata is always returned.

```julia
using SyntheticLPs, Distributions

# Generate 100 instances with variable counts drawn from a truncated normal,
# writing .mps files plus a manifest to ./dataset. By default, the generator
# keeps an accepted candidate pool and selects instances whose actual model
# sizes match the target distribution closely.
instances = generate_dataset(
    num_problems = 100,
    size_distribution = truncated(Normal(500, 200), 50, 2000),
    output_dir = "dataset",
    seed = 1234,            # 0 = non-deterministic; any other value is reproducible
)

for inst in instances[1:3]
    println("$(inst.problem_type): $(inst.num_variables) vars, " *
            "$(inst.num_constraints) constraints → $(inst.filename)")
end
```

Uniform targets are supported as well. To make each selected problem type match
the same size distribution independently, enable per-type matching:

```julia
instances = generate_dataset(
    num_problems = 120,
    size_distribution = Uniform(50, 2000),
    problem_types = [:transportation, :knapsack, :portfolio],
    match_size_by_type = true,
    candidate_multiplier = 2,
    output_dir = "dataset_by_type",
    seed = 1234,
)
```

The package itself is solver-agnostic. To enable **quality filtering** — solving
each instance and rejecting trivial, degenerate, unbounded, timed-out, or
ill-conditioned ones — pass an `optimizer`:

```julia
using SyntheticLPs, HiGHS

instances = generate_dataset(
    num_problems = 100,
    output_dir = "dataset",
    quality_filter = true,
    optimizer = HiGHS.Optimizer,
    optimizer_attributes = ("solver" => "simplex",),
    quality_criteria = QualityCriteria(
        solve_timeout = 30.0,
        min_constraints = 5,
        min_iterations = 3,
        max_iteration_ratio = 100.0,
    ),
    max_retries = 10,       # raw retry budget for failures / filtered candidates
    feasible_only = true,
)
```

A single instance can also be evaluated directly with
`check_quality(model, HiGHS.Optimizer)`.

## Extending with New Categories and Variants

Each category lives in its own folder under `src/problem_types/<category>/`:

```
src/problem_types/transportation/
    transportation.jl   # category entry point: includes the variant file(s)
    standard.jl         # a variant: struct + constructor + build_model + register_variant
```

**To add a new variant to an existing category**, create a file in that
category's folder, implement it (see template below), and `include` it from the
category's `<category>.jl` entry point.

**To add a new category**, create `src/problem_types/<category>/<category>.jl`
(the entry point), add at least one variant file, and add a single
`include("problem_types/<category>/<category>.jl")` line to `src/SyntheticLPs.jl`.

A category is created automatically by its first variant's `register_variant`
call (which supplies the category description). Call `register_category(:cat,
"…")` explicitly in the entry point only when you want a category-level
description distinct from its variants (typical once a category has several).

Variant file template:

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

# Register this variant under its category (lazily creates the category)
register_variant(
    :your_category,
    :your_variant,
    YourProblem,
    "Description of this variant"
)
```

The category entry point (`src/problem_types/your_category/your_category.jl`)
then just includes the variant file(s):

```julia
include("your_variant.jl")
```

Key principles:
- The struct stores ALL data needed to deterministically build the model
- ALL randomness goes in the constructor
- `build_model` must be completely deterministic (no RNG calls)
- Handle `feasible`, `infeasible`, and `unknown` feasibility statuses

## Command Line Interface

The package includes a command-line script for generating problems:

```bash
# Generate a transportation problem with ~100 variables (category default variant)
julia --project=@. scripts/generate_problem.jl transportation 100 problem.mps

# Select a specific variant with category/variant
julia --project=@. scripts/generate_problem.jl portfolio/cvar 100 problem.mps

# List all categories and their variants
julia --project=@. scripts/generate_problem.jl list

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

For generating whole datasets, `scripts/generate_lps.jl` is a thin command-line
wrapper around `generate_dataset` (it supplies HiGHS for quality filtering):

```bash
# Generate 100 .mps instances into ./output
julia --project=scripts scripts/generate_lps.jl -o output -n 100

# Generate 50 feasible, quality-filtered instances with progress output
julia --project=scripts scripts/generate_lps.jl -o output -n 50 --feasible-only -q -v

# Restrict to specific problem types and a fixed seed
julia --project=scripts scripts/generate_lps.jl --problem-types transportation,knapsack -n 20 --seed 42

# Uniform actual-size matching for each selected problem type
julia --project=scripts scripts/generate_lps.jl -o output -n 60 \
  --problem-types transportation,knapsack,portfolio \
  --size-distribution uniform --var-min 50 --var-max 2000 \
  --match-size-by-type --candidate-multiplier 2 --seed 42
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
