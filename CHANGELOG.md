# Changelog

All notable changes to SyntheticLPs.jl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## 2025-11-18

### New Feature: Problem Variant System

**Previous Commit**: `b679ada`
**Current Date/Time**: 2025-11-18

**Summary**: Implemented a comprehensive variant system for organizing related problem types into families while maintaining clean file organization and code separation. This allows multiple specialized versions of base problem types (e.g., nurse scheduling, OR scheduling) without cluttering individual files.

### Added

- **Variant System Architecture**:
  - Folder-based organization: variants stored in subdirectories (e.g., `src/problem_types/scheduling/`)
  - Naming convention: `:{base}_{variant}` (e.g., `:scheduling_nurse`, `:blending_beverage`)
  - Automatic discovery and grouping of related variants

- **Helper Functions**:
  - `is_variant(problem_sym::Symbol)` - Check if a symbol represents a variant
  - `get_base_type(problem_sym::Symbol)` - Extract base type from variant symbol
  - `list_problem_variants(base_type::Symbol)` - List all variants of a base type
  - Enhanced `list_problem_types(; group_variants::Bool=false)` - Optionally group variants by base type

- **Scheduling Variants** (2 new problem types):
  - `:scheduling_nurse` (`NurseScheduling`) - Hospital nurse scheduling with:
    - 12-hour shifts (day/night)
    - Nurse skill levels (RN, LPN, CNA)
    - Patient acuity-based staffing requirements
    - Mandatory rest periods and weekend rotation
    - Skill-level matching for shifts
  - `:scheduling_or` (`ORScheduling`) - Operating room scheduling with:
    - Multiple operating rooms with different capabilities
    - Surgery types with estimated durations
    - Surgeon and surgical team availability
    - Equipment and room type requirements
    - Block scheduling and turnover time constraints
    - Overtime costs

- **Blending Variants** (2 new problem types):
  - `:blending_beverage` (`BeverageBlending`) - Beverage formulation with:
    - Flavor profile requirements (sweetness, acidity, bitterness, fruitiness)
    - Nutritional constraints (calories, sugar)
    - pH and Brix (sugar content) specifications
    - Ingredient types (juices, sweeteners, acids, flavors, water, additives)
    - Batch size optimization
  - `:blending_pharmaceutical` (`PharmaceuticalBlending`) - Drug formulation with:
    - Active Pharmaceutical Ingredients (APIs) with strict potency requirements
    - Excipients (fillers, binders, disintegrants, lubricants, coating agents)
    - Purity and contamination limits (pharmaceutical grade)
    - Dissolution and bioavailability requirements
    - Very tight tolerances (±2-5%)
    - Ingredient compatibility matrix

### Changed

- **File Organization**:
  - Created `src/problem_types/scheduling/` subdirectory
  - Created `src/problem_types/blending/` subdirectory
  - Added variant include statements in `src/SyntheticLPs.jl`

- **Module Exports**:
  - Exported new variant helper functions

- **Test Suite** (`test/runtests.jl`):
  - Added comprehensive tests for variant system
  - Tests for variant helper functions
  - Tests for grouped listing functionality

- **Documentation**:
  - Updated `CLAUDE.md` with comprehensive variant system documentation:
    - Overview and rationale
    - File organization structure
    - Naming conventions
    - Helper function usage examples
    - Current variants listing
    - Guidelines for when to use variants
    - Instructions for adding new variants
  - Updated problem type count (24+ types including variants)

### Technical Details

#### Design Decisions

**Why folder-based organization?**
- Keeps related variants together
- Prevents `src/problem_types/` from becoming cluttered
- Makes it clear which problems are related
- Scales well as more variants are added

**Why underscore naming convention?**
- Simple and consistent (`:base_variant`)
- Works seamlessly with existing symbol-based registration
- Easy to parse and extract base type
- Backward compatible (doesn't affect existing problem types)

**Why not use Julia parametric types?**
- Parametric types (e.g., `Scheduling{NurseVariant}`) would be over-engineered
- Current approach is simpler and more flexible
- Easier for users to understand and extend
- Maintains consistency with existing architecture

#### Variant Pattern

Each variant follows the same pattern as base types:
1. Struct inheriting from `ProblemGenerator`
2. Constructor with `(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)`
3. Deterministic `build_model(prob)` function
4. Registration with `register_problem(:base_variant, Type, "description")`

#### File Structure

```
src/problem_types/
├── scheduling.jl                 # Base type
├── scheduling/                   # Variants subfolder
│   ├── nurse.jl
│   └── or.jl
├── blending.jl                  # Base type
└── blending/                    # Variants subfolder
    ├── beverage.jl
    └── pharmaceutical.jl
```

### Examples

```julia
# List all scheduling variants
list_problem_variants(:scheduling)
# [:scheduling, :scheduling_nurse, :scheduling_or]

# Check if a type is a variant
is_variant(:scheduling_nurse)  # true

# Get base type
get_base_type(:scheduling_nurse)  # :scheduling

# Group all types by base
grouped = list_problem_types(group_variants=true)
# Dict(:scheduling => [:scheduling, :scheduling_nurse, :scheduling_or],
#      :blending => [:blending, :blending_beverage, :blending_pharmaceutical], ...)

# Generate a variant problem
model, problem = generate_problem(:scheduling_nurse, 200, feasible, 42)
```

### Future Extensions

Potential future variants to add:
- **Scheduling**: classroom, course, shift, appointment, crew
- **Blending**: fuel, concrete, fertilizer, cosmetics
- **Network flow**: supply chain, transportation grid, communication network, electricity grid
- **Location**: warehouse, retail, emergency service, base station

### Files Modified

- **Core module**: `src/SyntheticLPs.jl`
  - Added variant helper functions
  - Updated exports
  - Added variant includes

- **New variant files**:
  - `src/problem_types/scheduling/nurse.jl`
  - `src/problem_types/scheduling/or.jl`
  - `src/problem_types/blending/beverage.jl`
  - `src/problem_types/blending/pharmaceutical.jl`

- **Tests**: `test/runtests.jl`
  - Added variant system tests

- **Documentation**:
  - `CLAUDE.md` - Comprehensive variant system documentation
  - This `CHANGELOG.md`

### Backward Compatibility

✅ **Fully backward compatible**: All existing problem types continue to work exactly as before. The variant system is purely additive.

## 2025-01-07

### Major Refactoring: Type-Based Dispatch Architecture

**Previous Commit**: `6c1270f`

**Summary**: Complete refactoring of the problem generator system from function-based to type-based dispatch architecture. This is a **breaking change** that improves code organization, type safety, and extensibility.

### Added

- **`ProblemGenerator` abstract type**: Base type for all problem generators
- **`FeasibilityStatus` enum**: Enum with values `feasible`, `infeasible`, `unknown` for explicit feasibility control
- **`build_model` function**: Generic function that each problem type implements for deterministic model building
- **Struct-based problem generators**: Each of the 21 problem types now has a dedicated struct storing all generated data:
  - `TransportationProblem`
  - `KnapsackProblem`
  - `PortfolioProblem`
  - `DietProblem`
  - `NetworkFlowProblem`
  - `ProductionPlanningProblem`
  - `AssignmentProblem`
  - `BlendingProblem`
  - `AirlineCrewProblem`
  - `CuttingStockProblem`
  - `EnergyProblem`
  - `FacilityLocationProblem`
  - `FeedBlendingProblem`
  - `InventoryProblem`
  - `LandUseProblem`
  - `LoadBalancingProblem`
  - `ProductMixProblem`
  - `ProjectSelectionProblem`
  - `ResourceAllocationProblem`
  - `SchedulingProblem`
  - `SupplyChainProblem`

### Changed

- **`generate_problem` function signature**:
  - Old: `generate_problem(problem_type::Symbol, params::Dict; seed::Int=0)` → returns `(model, params::Dict)`
  - New: `generate_problem(problem_type::Symbol, target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)` → returns `(model, problem::ProblemGenerator)`

- **`register_problem` function signature**:
  - Old: `register_problem(type_sym::Symbol, generator_fn::Function, sampler_fn::Function, description::String)`
  - New: `register_problem(type_sym::Symbol, problem_type::Type{<:ProblemGenerator}, description::String)`

- **`generate_random_problem` function signature**:
  - Old: Returns `(model, problem_type::Symbol, params::Dict)`
  - New: Returns `(model, problem_type::Symbol, problem::ProblemGenerator)`

- **Problem generators**: All 21 problem type implementations refactored from functions to structs
  - Constructors now handle ALL randomness and parameter sampling
  - `build_model` methods are completely deterministic
  - All sophisticated feasibility logic preserved and improved

- **Utility script** (`scripts/generate_problem.jl`):
  - Updated to use new API
  - Added `--feasible`, `--infeasible`, `--unknown` flags for feasibility control
  - Added `--seed=N` flag for explicit seed specification
  - Simplified argument parsing

### Removed

- **Removed functions**:
  - `sample_parameters(problem_type::Symbol, target_variables::Int)` - functionality integrated into constructors
  - `sample_parameters(problem_type::Symbol, size::Symbol)` - legacy size-based API removed
  - `get_generator(problem_type::Symbol)` - replaced by `get_problem_type`
  - `get_sampler(problem_type::Symbol)` - no longer needed
  - All individual `generate_[type]_problem` functions - replaced by constructors
  - All individual `sample_[type]_parameters` functions - integrated into constructors
  - All `calculate_[type]_variable_count` functions - no longer needed

### Technical Details

#### Architecture Changes

1. **Separation of Concerns**:
   - Problem data generation (constructors) is now cleanly separated from model building (`build_model`)
   - All randomness confined to constructors; `build_model` is deterministic

2. **Type Safety**:
   - Each problem type is now a distinct Julia type with compile-time type checking
   - Problem data stored in strongly-typed struct fields instead of `Dict`

3. **Multiple Dispatch**:
   - Uses Julia's multiple dispatch for clean, extensible interface
   - `build_model(::ProblemType)` dispatches to type-specific implementations

4. **Improved Reproducibility**:
   - Same seed guarantees identical problem instance with identical data
   - Deterministic `build_model` ensures same problem always produces same model

5. **Feasibility Control**:
   - Explicit `FeasibilityStatus` enum replaces symbol-based `:solution_status`
   - All generators properly handle `feasible`, `infeasible`, and `unknown` statuses
   - Sophisticated feasibility logic preserved from original implementations:
     - Diet problem: 4 verified impossibility scenarios with final verification
     - Scheduling: Consecutive-day capacity, randomized matching, 3 infeasibility modes
     - Land use: Witness construction, adjacency-aware assignment
     - Supply chain: Geographic clustering, K-nearest connectivity
     - And many more...

#### Code Quality Improvements

- **Reduced code duplication**: Pattern consistency across all 21 generators
- **Better documentation**: Comprehensive docstrings for all structs and functions
- **Cleaner interfaces**: No more `Dict` parameter passing
- **Easier testing**: Structs can be inspected and compared directly

#### Backward Compatibility

**Breaking**: This refactoring intentionally breaks backward compatibility to improve the architecture. The old function-based API is completely removed. Users must update their code to use the new type-based API.

### Migration Guide

#### Old API:
```julia
# Old way
params = sample_parameters(:transportation, 100)
model, actual_params = generate_problem(:transportation, params)
```

#### New API:
```julia
# New way
model, problem = generate_problem(:transportation, 100, unknown, 0)
# Access problem data through struct fields
println(problem.n_sources, problem.n_destinations)
```

### Testing

- Updated test suite to use new API
- All 21 problem types tested with multiple target variable counts
- All three feasibility statuses tested for each problem type
- Reproducibility tests with fixed seeds

### Documentation

- Updated `README.md` with new API examples
- Updated `CLAUDE.md` with new architecture description
- Added comprehensive docstrings to all new types and functions

### Files Modified

- **Core module**: `src/SyntheticLPs.jl`
- **All problem types** (21 files in `src/problem_types/`):
  - airline_crew.jl
  - assignment.jl
  - blending.jl
  - cutting_stock.jl
  - diet_problem.jl
  - energy.jl
  - facility_location.jl
  - feed_blending.jl
  - inventory.jl
  - knapsack.jl
  - land_use.jl
  - load_balancing.jl
  - network_flow.jl
  - portfolio.jl
  - product_mix.jl
  - production_planning.jl
  - project_selection.jl
  - resource_allocation.jl
  - scheduling.jl
  - supply_chain.jl
  - transportation.jl
- **Utility scripts**: `scripts/generate_problem.jl`
- **Tests**: `test/runtests.jl`
- **Documentation**: `README.md`, `CLAUDE.md`
