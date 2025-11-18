# Changelog

All notable changes to SyntheticLPs.jl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## 2025-11-18

### Added: Transportation Problem Generator Suite

**Base Commit**: `b679ada`
**Date**: 2025-11-18

**Summary**: Expanded the transportation problem family from a single basic generator to a comprehensive suite of 8 specialized transportation and logistics problem generators, organized in a dedicated `src/problem_types/transportation/` directory.

### New Problem Generators

1. **Basic Transportation** (`basic_transportation.jl`)
   - Classic source-to-destination shipping optimization
   - Renamed and moved from `transportation.jl` to new subfolder
   - Symbol: `:basic_transportation` (changed from `:transportation`)

2. **Vehicle Routing** (`vehicle_routing.jl`)
   - Capacitated vehicle routing problem (CVRP)
   - Multiple vehicles with capacity constraints
   - Clustered customer locations for realism
   - Flow-based LP formulation
   - Symbol: `:vehicle_routing`

3. **Warehouse Location & Sizing** (`warehouse_location.jl`)
   - Combined facility location and capacity sizing decisions
   - Multi-echelon flows: suppliers → warehouses → customers
   - Multiple warehouse size options with economies of scale
   - Geographic location-based costs
   - Symbol: `:warehouse_location`

4. **Hub Location** (`hub_location.jl`)
   - Hub-and-spoke network design
   - Economies of scale on inter-hub connections
   - Discount factors for consolidated flows
   - Sparse origin-destination demand matrix
   - Hub capacity constraints
   - Symbol: `:hub_location`

5. **Transshipment** (`transshipment.jl`)
   - Multi-echelon network with intermediate storage nodes
   - Direct and indirect routing options
   - Storage capacity constraints at transshipment nodes
   - Holding costs and transfer costs
   - Symbol: `:transshipment`

6. **Last Mile Delivery** (`last_mile_delivery.jl`)
   - Urban delivery optimization
   - Time windows for customer deliveries
   - Zone-based clustering (urban areas)
   - Traffic congestion factors
   - Maximum route time constraints
   - Service times at each location
   - Symbol: `:last_mile_delivery`

7. **Cross-Docking** (`cross_docking.jl`)
   - Transfer optimization with minimal storage
   - Inbound and outbound flow synchronization
   - Dock capacity constraints
   - Multi-product flows
   - Storage time limits
   - Inter-dock transfer costs
   - Symbol: `:cross_docking`

8. **Cargo Loading** (`cargo_loading.jl`)
   - Container/truck loading optimization
   - Multi-dimensional knapsack formulation
   - Weight and volume constraints
   - Item fragility classes
   - Multiple container types
   - Value maximization objective
   - Symbol: `:cargo_loading`

### Technical Features

All new generators implement:
- **Realistic data patterns**: Log-normal distributions for demands/capacities, geographic clustering, distance-based costs
- **Feasibility control**: Sophisticated mechanisms for guaranteed feasible/infeasible instances
- **Variable sizing**: Problems within 10% of target variable count
- **LP-friendly formulations**: Flow-based and assignment formulations suitable for continuous relaxation
- **Practical constraints**: Realistic business rules (capacity limits, time windows, network structure, etc.)

### Files Added
- `src/problem_types/transportation/basic_transportation.jl`
- `src/problem_types/transportation/vehicle_routing.jl`
- `src/problem_types/transportation/warehouse_location.jl`
- `src/problem_types/transportation/hub_location.jl`
- `src/problem_types/transportation/transshipment.jl`
- `src/problem_types/transportation/last_mile_delivery.jl`
- `src/problem_types/transportation/cross_docking.jl`
- `src/problem_types/transportation/cargo_loading.jl`

### Files Modified
- `src/SyntheticLPs.jl` - Updated includes to reference transportation subfolder
- `README.md` - Added Transportation & Logistics section with all 8 problem types
- `CLAUDE.md` - Updated architecture documentation

### Files Removed
- `src/problem_types/transportation.jl` - Moved to `transportation/basic_transportation.jl`

### Breaking Changes
- The `:transportation` symbol now refers to `:basic_transportation`
- Old code using `:transportation` will need to be updated to `:basic_transportation`

### Architecture Changes
- Created `src/problem_types/transportation/` subdirectory
- Organized all transportation-related problems in dedicated folder
- Maintained consistent interface across all generators

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
