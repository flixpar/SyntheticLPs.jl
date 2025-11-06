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

Run the standard test suite (quick smoke tests):
```bash
julia --project=@. test/runtests.jl
```

Run comprehensive tests with feasibility verification (~50 instances per problem type):
```bash
julia --project=@. test_instances.jl                    # All problem types
julia --project=@. test_instances.jl transportation     # Specific type
julia --project=@. test_instances.jl --verbose          # Verbose output
COMPREHENSIVE_TESTS=true julia --project=@. test/runtests.jl  # Alternative
```

### Problem Generation

Generate a specific problem type targeting ~100 variables:
```bash
julia --project=@. generate_problem.jl transportation 100 output.mps
```

List all available problem types:
```bash
julia --project=@. generate_problem.jl list
```

Generate and solve a problem with ~50 variables:
```bash
julia --project=@. generate_problem.jl knapsack 50 --solve
```

Generate a random problem with ~200 variables:
```bash
julia --project=@. generate_problem.jl random 200
```

### Development

Start Julia REPL with project loaded:
```bash
julia --project=@.
```

## Architecture

SyntheticLPs is a standardized framework for generating 20+ types of realistic linear programming problems. All problem generators follow a consistent interface pattern.

### Core Components

**Main Module** (`src/SyntheticLPs.jl`):
- Registration system for problem types using `LP_REGISTRY`
- Unified interface functions: `generate_problem()`, `sample_parameters()`, `list_problem_types()`
- Random problem generation with `generate_random_problem()`
- Support for both target variable count and legacy size-based generation

**Problem Generators** (`src/problem_types/`):
- Each problem type has three required functions:
  - `generate_[type]_problem(params; seed)`: Creates JuMP model from parameters
  - `sample_[type]_parameters(target_variables; seed)`: Samples parameters for target variable count
  - `sample_[type]_parameters(size; seed)`: Legacy size-based parameter sampling
- Additional helper function: `calculate_[type]_variable_count(params)`: Calculates variable count from parameters
- All generators call `register_problem()` to register with the system
- Template provided in `src/problem_types/template.jl`

**Utility Scripts**:
- `generate_problem.jl`: Command-line interface for problem generation
- `test_problem_types.jl`: Quick verification of all generators
- `verify_interface.jl`: Checks interface compliance across all problem files

### Problem Generator Pattern

Each problem generator follows this structure:

```julia
function generate_[type]_problem(params::Dict=Dict(); seed::Int=0)
    Random.seed!(seed)
    # Extract parameters with defaults
    # Generate JuMP model with variables, constraints, objective
    return model, actual_params
end

function calculate_[type]_variable_count(params::Dict)
    # Calculate and return the number of variables based on parameters
    return variable_count
end

function sample_[type]_parameters(target_variables::Int; seed::Int=0)
    # Sample parameters to target approximately target_variables variables
    # Use calculate_[type]_variable_count to achieve target within ±10%
    return params
end

function sample_[type]_parameters(size::Symbol=:medium; seed::Int=0)
    # Legacy size-based parameter sampling (backward compatibility)
    target_map = Dict(:small => 150, :medium => 500, :large => 2000)
    return sample_[type]_parameters(target_map[size]; seed=seed)
end

register_problem(:type, generate_fn, sample_fn, "Description")
```

### Available Problem Types

The system includes 20+ problem types covering major LP problem classes:
- Transportation, Diet, Knapsack, Portfolio, Network Flow
- Production Planning, Assignment, Blending, Facility Location
- Energy, Inventory, Scheduling, Supply Chain, and others

### Testing Strategy

**Standard Tests** (`test/runtests.jl`):
- Quick smoke tests for all problem generators
- Tests parameter sampling, model generation, and reproducibility
- Verifies models have proper structure (variables, constraints, objective)
- Tests both target variable count and legacy size-based parameter sampling
- Validates variable count accuracy (within ±15% tolerance)
- Each generator tested with multiple target variable counts and fixed seeds

**Comprehensive Tests** (`test/test_problem_instances.jl`, accessed via `test_instances.jl`):
- ~50 problem instances per problem type across varied target sizes
- Phase 1: Basic generation tests (~40 instances without feasibility checks)
  - Successful generation (no errors)
  - Variable count within 15% of target
  - Tests with target sizes: 50, 100, 250, 500, 1000 variables
- Phase 2: Feasibility tests (~20 instances with HiGHS solver verification)
  - Tests `solution_status` parameter when supported by problem type
  - Verifies `:feasible` problems are actually feasible
  - Verifies `:infeasible` problems are actually infeasible
  - Uses HiGHS solver to check feasibility status
- Supports testing specific problem types via command line
- Detailed statistics and error reporting

## Adding New Problem Types

1. Create new file in `src/problem_types/your_problem.jl`
2. Implement generator and parameter sampling functions following the pattern
3. Call `register_problem()` to register with the system
4. Add include statement to `src/SyntheticLPs.jl`
5. Run tests to verify implementation
