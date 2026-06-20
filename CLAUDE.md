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

### Dataset Generation

Generate a whole dataset of LP instances via the library API (`generate_dataset`)
or its CLI wrapper. The wrapper supplies HiGHS, so use the `scripts` environment:

```bash
# 100 .mps instances into ./output
julia --project=scripts scripts/generate_lps.jl -o output -n 100

# Quality-filtered, feasible-only, with progress
julia --project=scripts scripts/generate_lps.jl -o output -n 50 --feasible-only -q -v
```

### Development

Start Julia REPL with project loaded:
```bash
julia --project=@.
```

## Architecture

SyntheticLPs uses a type-based dispatch system for generating realistic linear programming problems. Problems are organized as a two-level hierarchy: a **category** (a problem domain, e.g. `:transportation`) groups one or more **variants** (concrete generators with their own data generation and model formulation, e.g. `:standard`). There are 24 categories, each currently with a single variant. All generators follow a consistent pattern using Julia's multiple dispatch.

### Core Components

**Main Module** (`src/SyntheticLPs.jl`):
- `ProblemGenerator`: Abstract base type for all problem generators
- `FeasibilityStatus`: Enum with values `feasible`, `infeasible`, `unknown`
- `ProblemVariant`: identifier for a `category/variant` pair (the canonical reference used throughout); constructible from `(category, variant)` symbols, a bare category symbol (→ default variant), or a `"category"`/`"category/variant"` string; prints as `category/variant`
- Two-level registry `LP_REGISTRY::Dict{Symbol,CategorySpec}` populated by `register_category()` and `register_variant()` (a single variant lazily creates its category)
- Unified interface functions: `generate_problem()` (accepts a category symbol with optional `variant=` keyword, a `ProblemVariant`, or a generator type), `list_categories()`/`list_problem_types()` (alias), `list_variants()`, `list_problems()`, `problem_info()`
- Random problem generation with `generate_random_problem()` (returns the selected `ProblemVariant`)
- Base function `build_model(problem::ProblemGenerator)` that each variant implements

**Dataset Generation** (`src/dataset.jl`):
- `generate_dataset(; kwargs...)`: builds a whole dataset of LP instances by sampling problem types and target variable counts; optionally writes instance files + a `manifest.json` and returns `Vector{GeneratedInstance}` metadata. Fully reproducible from a non-zero `seed`.
- `check_quality(model, optimizer; ...)` + `QualityCriteria`/`QualityResult`: solve-based filtering of trivial/degenerate/unbounded/ill-conditioned instances.
- The package stays solver-agnostic: quality filtering requires the caller to pass an `optimizer` (e.g. `HiGHS.Optimizer`). `scripts/generate_lps.jl` is a thin CLI wrapper that supplies HiGHS.

**Problem Generators** (`src/problem_types/<category>/`):
- Each category is a folder containing:
  - A `<category>.jl` entry point that `include`s the category's variant file(s) (and optionally calls `register_category(:category, "description")` for a category-level description)
  - One file per variant (or closely related group of variants), e.g. `standard.jl`
- Each variant file has:
  - A struct inheriting from `ProblemGenerator` containing all generated data
  - A constructor `VariantStruct(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)`
  - An implementation of `build_model(prob::VariantStruct)` that deterministically builds the JuMP model
  - A `register_variant(:category, :variant, VariantStruct, "description")` call (which lazily creates the category if needed)
- Structs store ALL data needed to build the model (costs, capacities, demands, etc.)
- Constructors contain ALL randomness; `build_model` is completely deterministic

**Utility Scripts**:
- `scripts/generate_problem.jl`: Command-line interface for problem generation
- `scripts/analyze_problem_statuses.jl`: Analysis of problem feasibility

### Problem Generator Pattern

A category folder `src/problem_types/<category>/` contains an entry point and one
file per variant:

```julia
# src/problem_types/<category>/<category>.jl  (entry point)
# Optionally: register_category(:category, "Category-level description")
include("standard.jl")   # include each variant file
```

```julia
# src/problem_types/<category>/standard.jl  (a variant)
struct VariantStruct <: ProblemGenerator
    # Store all generated data needed to build the model
    field1::Type1
    field2::Type2
    # ...
end

function VariantStruct(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Sample parameters based on target_variables
    # Generate all deterministic data (costs, capacities, etc.)
    # Handle feasibility status (feasible, infeasible, unknown)

    return VariantStruct(field1_value, field2_value, ...)
end

function build_model(prob::VariantStruct)
    model = Model()

    # Build JuMP model using only prob's fields
    # This must be completely deterministic (no RNG calls)

    return model
end

# Registers the variant; lazily creates the :category if it doesn't exist yet.
register_variant(:category, :standard, VariantStruct, "Description")
```

### Key Design Principles

1. **Separation of Concerns**: Randomness (constructor) vs. determinism (build_model)
2. **Reproducibility**: Same seed → identical problem instance → identical model
3. **Feasibility Control**: Generators can produce guaranteed feasible/infeasible problems
4. **Type Safety**: Each problem is a distinct type with its own data structure
5. **Dispatch**: Use Julia's multiple dispatch for clean, extensible interface

### Available Problem Categories

The system includes 24 categories covering major LP problem classes (each currently with a single variant; `portfolio`'s is `:cvar`, the rest are `:standard`):
- Transportation, Diet Problem, Knapsack, Portfolio (CVaR with institutional constraints), Network Flow, Multi-Commodity Flow
- Production Planning, Assignment, Blending, Facility Location, Crop Planning
- Airline Crew, Cutting Stock, Energy, Feed Blending, Inventory, Telecom Network Design
- Land Use, Load Balancing, Product Mix, Project Selection
- Resource Allocation, Scheduling, Supply Chain

### Testing Strategy

- `test/runtests.jl`: Comprehensive test suite verifying all problem generators
- Tests problem generation with different target variable counts
- Tests all three feasibility statuses (feasible, infeasible, unknown)
- Verifies models have proper structure (variables, constraints, objective)
- Validates reproducibility with fixed seeds
- Each generator tested with multiple configurations

## Adding New Categories and Variants

### Add a variant to an existing category

1. Create a new file in the category folder, e.g. `src/problem_types/<category>/<variant>.jl`
2. Define a struct inheriting from `ProblemGenerator` with all necessary data fields
3. Implement constructor `VariantStruct(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)`
4. Implement `build_model(prob::VariantStruct)` (must be deterministic)
5. Call `register_variant(:category, :variant, VariantStruct, "Description")` (pass `default=true` to make it the category default)
6. `include("<variant>.jl")` from the category's `<category>.jl` entry point
7. Run tests to verify implementation

### Add a new category

1. Create `src/problem_types/<category>/<category>.jl` (the entry point) and at least one variant file (steps 1–5 above)
2. The entry point `include`s the variant file(s); add `register_category(:category, "Description")` there only if you want a category-level description distinct from its variants
3. Add `include("problem_types/<category>/<category>.jl")` to `src/SyntheticLPs.jl`
4. Run tests to verify implementation

Key principles:
- Struct stores ALL generated data needed to build the model
- Constructor contains ALL randomness and parameter sampling
- `build_model` must be completely deterministic (no RNG calls)
- Handle all three feasibility statuses appropriately
