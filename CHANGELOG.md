# Changelog

All notable changes to SyntheticLPs.jl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## 2025-11-18

### Added Vehicle Routing Problem Generator

**Base Commit**: `b679ada`
**Datetime**: 2025-11-18 13:11:27 UTC

**Summary**: Implemented a new problem generator for Capacitated Vehicle Routing Problems (CVRP) with LP relaxation. The generator creates realistic vehicle routing scenarios for testing LP solvers.

### Added

- **`VehicleRoutingProblem` struct**: New problem generator for vehicle routing problems
  - Fields: `n_customers`, `n_vehicles`, `depot_location`, `customer_locations`, `demands`, `vehicle_capacities`, `distances`
  - Implements the standard `ProblemGenerator` interface

- **Realistic problem characteristics**:
  - Geographic clustering of customers (simulates urban delivery patterns)
  - Log-normal distribution for customer demands (realistic mix of large and small orders)
  - Heterogeneous vehicle fleet with varying capacities
  - Euclidean distance-based costs with minor random variation
  - Central or edge depot location

- **LP relaxation formulation**:
  - Flow variables `x[i,j,k]`: continuous [0,1] flow from location i to j using vehicle k
  - Service variables `y[j,k]`: continuous [0,1] indicating if vehicle k serves customer j
  - Constraints: customer service, flow conservation, vehicle capacity, depot flow
  - Objective: minimize total routing cost

- **Feasibility control**:
  - **Feasible**: Ensures total vehicle capacity ≥ 1.15-1.35× total demand
  - **Infeasible**: Reduces total capacity to 0.65-0.92× total demand, or increases demands
  - **Unknown**: Natural randomness without adjustment

- **Accurate variable count targeting**:
  - Formula: `target_variables = m × ((n+1)² + n)` where n=customers, m=vehicles
  - Optimizes (n_customers, n_vehicles) combination to achieve target within ±10%
  - Scales appropriately for small (≤100), medium (≤500), large (≤2000), and very large (>2000) problems

### Implementation Details

- **Constructor** (`VehicleRoutingProblem`):
  - Samples all parameters based on target_variables
  - Generates geographic locations with clustering
  - Creates heterogeneous vehicle fleet
  - Handles all three feasibility statuses
  - All randomness confined to constructor for reproducibility

- **Model builder** (`build_model`):
  - Completely deterministic (no RNG calls)
  - Creates flow-based LP formulation
  - Implements all standard VRP constraints

- **Helper functions**:
  - `sample_vrp_parameters`: Determines optimal problem dimensions
  - `generate_clustered_customers`: Creates realistic geographic customer distribution
  - `generate_lognormal_demands`: Samples realistic demand patterns
  - `generate_vehicle_fleet`: Creates heterogeneous fleet with varying capacities
  - `calculate_distances`: Computes Euclidean distances with variation

### Files Added/Modified

- **New file**: `src/problem_types/vehicle_routing.jl` (469 lines)
- **Modified**: `src/SyntheticLPs.jl` - added include statement for vehicle_routing.jl
- **Modified**: `README.md` - added Vehicle Routing to problem types list
- **Modified**: `CHANGELOG.md` - this entry

### Problem Scaling

The generator adapts to different problem sizes:

- **Small (target ≤ 100 vars)**: 3-10 customers, 2-4 vehicles, local delivery scenarios
- **Medium (target ≤ 500 vars)**: 8-25 customers, 3-8 vehicles, regional distribution
- **Large (target ≤ 2000 vars)**: 20-50 customers, 5-15 vehicles, city-wide delivery
- **Very large (target > 2000 vars)**: 40-100 customers, 8-25 vehicles, multi-city logistics

### Registration

Problem type registered as `:vehicle_routing` with description:
"Capacitated vehicle routing problem (CVRP) with LP relaxation that routes vehicles from a depot to serve customers while minimizing travel cost"

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
