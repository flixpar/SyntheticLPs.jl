# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context

This package is a standardized framework for generating synthetic linear programming (LP) problem instances. The goal is to generate problems that are highly realistic and can be used to test and develop LP solvers.

## General Instructions

- Make sure to explore the relevant code carefully before making any plans or changes.
- Update the changelog file after making any significant changes. Organize the changelog with sections per date, and keep track of the commit hash and the current datetime for each set of changes. Include high level summaries of the changes as well as specific details with more granular information than commit messages.
- This project is under active development and is not yet stable, so never worry about making breaking changes or backwards compatibility.
- When making major changes, always update the README.md and CLAUDE.md files to reflect the changes.
- This package is intended for research use only, so it does not need to be extremely robust and handle all edge cases.

## Commands

### Testing

Run the comprehensive test suite:
```bash
julia --project=@. test/runtests.jl
```

### Problem Generation

Generate a specific problem type targeting ~100 variables:
```bash
julia --project=@. scripts/generate_problem.jl transportation 100 output.mps
```

List all available problem types:
```bash
julia --project=@. scripts/generate_problem.jl list
```

Generate and solve a feasible problem with ~50 variables:
```bash
julia --project=@. scripts/generate_problem.jl knapsack 50 --feasible --solve
```

Generate an infeasible problem with ~100 variables:
```bash
julia --project=@. scripts/generate_problem.jl diet_problem 100 --infeasible
```

Generate a random problem with ~200 variables:
```bash
julia --project=@. scripts/generate_problem.jl random 200
```

### Development

Start Julia REPL with project loaded:
```bash
julia --project=@.
```

## Architecture

SyntheticLPs uses a type-based dispatch system for generating 21 types of realistic linear programming problems. All problem generators follow a consistent pattern using Julia's multiple dispatch.

### Core Components

**Main Module** (`src/SyntheticLPs.jl`):
- `ProblemGenerator`: Abstract base type for all problem generators
- `FeasibilityStatus`: Enum with values `feasible`, `infeasible`, `unknown`
- Registration system for problem types using `LP_REGISTRY`
- Unified interface functions: `generate_problem()`, `list_problem_types()`, `problem_info()`
- Random problem generation with `generate_random_problem()`
- Base function `build_model(problem::ProblemGenerator)` that each generator implements

**Problem Generators** (`src/problem_types/`):
- Each problem type has:
  - A struct inheriting from `ProblemGenerator` containing all generated data
  - A constructor `ProblemType(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)`
  - An implementation of `build_model(prob::ProblemType)` that deterministically builds the JuMP model
- All generators call `register_problem(:symbol, ProblemType, "description")` to register
- Structs store ALL data needed to build the model (costs, capacities, demands, etc.)
- Constructors contain ALL randomness; `build_model` is completely deterministic

**Utility Scripts**:
- `scripts/generate_problem.jl`: Command-line interface for problem generation
- `scripts/analyze_problem_statuses.jl`: Analysis of problem feasibility

### Problem Generator Pattern

Each problem generator follows this structure:

```julia
struct ProblemType <: ProblemGenerator
    # Store all generated data needed to build the model
    field1::Type1
    field2::Type2
    # ...
end

function ProblemType(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Sample parameters based on target_variables
    # Generate all deterministic data (costs, capacities, etc.)
    # Handle feasibility status (feasible, infeasible, unknown)

    return ProblemType(field1_value, field2_value, ...)
end

function build_model(prob::ProblemType)
    model = Model()

    # Build JuMP model using only prob's fields
    # This must be completely deterministic (no RNG calls)

    return model
end

register_problem(:type, ProblemType, "Description")
```

### Key Design Principles

1. **Separation of Concerns**: Randomness (constructor) vs. determinism (build_model)
2. **Reproducibility**: Same seed → identical problem instance → identical model
3. **Feasibility Control**: Generators can produce guaranteed feasible/infeasible problems
4. **Type Safety**: Each problem is a distinct type with its own data structure
5. **Dispatch**: Use Julia's multiple dispatch for clean, extensible interface

### Available Problem Types

The system includes 21 problem types covering major LP problem classes:
- Transportation, Diet Problem, Knapsack, Portfolio, Network Flow
- Production Planning, Assignment, Blending, Facility Location
- Airline Crew, Cutting Stock, Energy, Feed Blending, Inventory
- Land Use, Load Balancing, Product Mix, Project Selection
- Resource Allocation, Scheduling, Supply Chain

### Testing Strategy

- `test/runtests.jl`: Comprehensive test suite verifying all problem generators
- Tests problem generation with different target variable counts
- Tests all three feasibility statuses (feasible, infeasible, unknown)
- Verifies models have proper structure (variables, constraints, objective)
- Validates reproducibility with fixed seeds
- Each generator tested with multiple configurations

## Adding New Problem Types

1. Create new file in `src/problem_types/your_problem.jl`
2. Define a struct inheriting from `ProblemGenerator` with all necessary data fields
3. Implement constructor `YourProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)`
4. Implement `build_model(prob::YourProblem)` function (must be deterministic)
5. Call `register_problem(:your_problem, YourProblem, "Description")`
6. Add include statement to `src/SyntheticLPs.jl`
7. Run tests to verify implementation

Key principles:
- Struct stores ALL generated data needed to build the model
- Constructor contains ALL randomness and parameter sampling
- `build_model` must be completely deterministic (no RNG calls)
- Handle all three feasibility statuses appropriately
