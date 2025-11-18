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

The system includes 24+ problem types covering major LP problem classes:
- Transportation, Diet Problem, Knapsack, Portfolio, Network Flow
- Production Planning, Assignment, Blending, Facility Location
- Airline Crew, Cutting Stock, Energy, Feed Blending, Inventory
- Land Use, Load Balancing, Product Mix, Project Selection
- Resource Allocation, Scheduling, Supply Chain
- Plus multiple variants of scheduling and blending problems (see Variant System below)

### Variant System

**Overview:**
The variant system allows organizing related problem types into families while maintaining clean file organization and code separation. Variants are specialized versions of base problem types that share core structure but differ in domain-specific constraints and characteristics.

**File Organization:**
```
src/problem_types/
├── scheduling.jl                 # Base/generic scheduling
├── scheduling/                   # Variants subfolder
│   ├── nurse.jl                 # Nurse scheduling variant
│   ├── or.jl                    # Operating room scheduling variant
│   └── workforce.jl             # Workforce scheduling variant (future)
├── blending.jl                  # Base/generic blending
└── blending/                    # Variants subfolder
    ├── beverage.jl              # Beverage formulation variant
    └── pharmaceutical.jl        # Pharmaceutical blending variant
```

**Naming Convention:**
- Base types: `:scheduling`, `:blending`
- Variants: `:scheduling_nurse`, `:scheduling_or`, `:blending_beverage`
- Pattern: `:{base}_{variant}` (underscore separator)

**Helper Functions:**

```julia
# Check if a problem type is a variant
is_variant(:scheduling)              # false (base type)
is_variant(:scheduling_nurse)        # true (variant)

# Get base type from any problem symbol
get_base_type(:scheduling_nurse)     # :scheduling
get_base_type(:blending_beverage)    # :blending

# List all variants of a base type
list_problem_variants(:scheduling)
# [:scheduling, :scheduling_nurse, :scheduling_or]

# List all problem types grouped by base type
list_problem_types(group_variants=true)
# Dict(:scheduling => [:scheduling, :scheduling_nurse, :scheduling_or],
#      :blending => [:blending, :blending_beverage, :blending_pharmaceutical], ...)
```

**Current Variants:**

1. **Scheduling Variants:**
   - `:scheduling` - Generic workforce scheduling (base)
   - `:scheduling_nurse` - Hospital nurse scheduling with 12-hour shifts, skill levels (RN/LPN/CNA), and patient acuity
   - `:scheduling_or` - Operating room scheduling with surgery types, room capabilities, and block scheduling

2. **Blending Variants:**
   - `:blending` - Generic blending problem (base)
   - `:blending_beverage` - Beverage formulation with flavor profiles, nutritional content, pH, and Brix
   - `:blending_pharmaceutical` - Drug formulation with APIs, excipients, purity requirements, and tight tolerances

**When to Use Variants:**

Use variants when:
- Multiple problem instances share core structure but have domain-specific constraints
- Different applications have different terminology, constraints, or typical scales
- Keeping variants separate improves code clarity and maintainability

Examples of good variant candidates:
- Scheduling: nurse, OR, classroom, course, shift, appointment
- Blending: beverage, pharmaceutical, fuel, concrete, fertilizer
- Network flow: supply chain, transportation, communication, electricity grid

### Testing Strategy

- `test/runtests.jl`: Comprehensive test suite verifying all problem generators
- Tests problem generation with different target variable counts
- Tests all three feasibility statuses (feasible, infeasible, unknown)
- Verifies models have proper structure (variables, constraints, objective)
- Validates reproducibility with fixed seeds
- Each generator tested with multiple configurations

## Adding New Problem Types

### Adding a Base Problem Type

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

### Adding a Problem Variant

1. Create subdirectory if needed: `src/problem_types/base_type/`
2. Create variant file: `src/problem_types/base_type/variant_name.jl`
3. Define struct: `struct VariantName <: ProblemGenerator`
4. Implement constructor and `build_model` following the same pattern as base types
5. Register with naming convention: `register_problem(:base_type_variant, VariantName, "Description")`
   - Example: `register_problem(:scheduling_nurse, NurseScheduling, "Description")`
6. Add include statement to `src/SyntheticLPs.jl` in the variants section
7. The variant will automatically be:
   - Included in `list_problem_types()`
   - Grouped with `list_problem_variants(:base_type)`
   - Identified by `is_variant(:base_type_variant)`

Example variant structure:
```julia
# File: src/problem_types/scheduling/nurse.jl
struct NurseScheduling <: ProblemGenerator
    # Domain-specific fields for nurse scheduling
    n_nurses::Int
    skill_levels::Vector{Symbol}
    # ...
end

function NurseScheduling(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    # Implementation with domain-specific logic
end

function build_model(prob::NurseScheduling)
    # Build JuMP model
end

register_problem(:scheduling_nurse, NurseScheduling, "Nurse scheduling with skill levels and patient acuity")
```
