# Changelog

All notable changes to SyntheticLPs.jl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## 2026-03-23

### Feature: Quality Filter for Batch LP Generation

**Previous Commit**: `28f882c`

**Summary**: Added a `--quality-filter` (`-q`) flag to `scripts/generate_lps.jl` that solves each generated LP instance with HiGHS simplex and filters out poor-quality test instances. The script retries generation (up to `--max-retries` × n attempts) to reach the requested problem count.

### Added

- **`--quality-filter` / `-q`**: Enables solve-and-filter pipeline. Each instance is solved with HiGHS simplex before being written to disk.
- **Filter criteria** (rejects instances that are):
  - Too few constraints (`--min-constraints`, default 5)
  - Infeasible (only when `--feasible-only` is also set)
  - Unbounded
  - Timed out (`--solve-timeout`, default 30s) or hit numerical errors
  - Nearly optimal (ALMOST_OPTIMAL status — indicates numerical conditioning issues)
  - Trivially solved / solved in phase 1 only (simplex iterations ≤ `--min-iterations`, default 3)
  - Degenerate (simplex iterations > `--max-iteration-ratio` × constraint count, default 100×)
- **`--max-retries`**: Controls total attempt budget as a multiplier of requested count (default 10)
- **Filter statistics**: Summary output shows counts of rejected instances broken down by reason

---

### Fix: Land Use Problem Generator Feasibility Guarantee

**Previous Commit**: `dfba903`

**Summary**: Fixed a bug where ~17.3% of land use problems generated with `feasible` status were actually infeasible. The root cause was that the witness assignment constructed during feasibility enforcement could violate adjacency constraints and minimum zoning requirements, but resource capacities were tightened around this invalid witness without verification.

### Fixed

- **Adjacency violations in remainder assignment**: When assigning unassigned parcels, the fallback path (when all allowed types conflict with adjacency) ignored adjacency constraints entirely, assigning residential next to industrial. The adjacency edges remained in the model, making it infeasible. Fix: after witness construction, scan for residential-industrial adjacency violations and prune offending edges from the adjacency matrix.
- **Incomplete minimum zoning fulfillment**: The type-2 (Commercial) assignment could fail when all parcels were consumed by types 1 and 3, with swap logic unable to find replacements (it only searched unassigned parcels). Fix: after witness construction, verify minimum counts are met; attempt swaps from over-represented types first, then reduce minimums to actual counts as a last resort.

### Validation

- 0/500 feasible-requested problems are infeasible (MIP), down from ~17.3%
- 0/300 feasible-requested problems are infeasible (LP relaxation)
- 0/300 infeasible-requested problems are accidentally feasible

## 2026-03-22

### Redesign: Portfolio Problem Generator (CVaR with Institutional Constraints)

**Previous Commit**: `d91324d`

**Summary**: Complete rewrite of the portfolio problem generator. The old generator was degenerate — only 2-3 constraints regardless of variable count, with 39.2% of problems solving in ≤2 simplex iterations. Replaced with a CVaR (Conditional Value-at-Risk) portfolio optimization model with rich institutional-grade constraints.

### Changed

- **`PortfolioProblem`**: Completely redesigned from a simple risk-budget model to a CVaR portfolio optimization with:
  - **CVaR risk measure**: Scenario-based linearization (Rockafellar-Uryasev) creating n_scenarios constraints that scale with problem size
  - **Sector exposure limits**: Maximum allocation per industry sector
  - **Region exposure limits**: Maximum allocation per geographic region
  - **Asset class bounds**: Min/max allocation per asset class (equities, bonds, alternatives)
  - **Factor exposure constraints**: Upper/lower bounds on risk factor exposures (beta, size, value, etc.)
  - **Position size limits**: Per-asset concentration caps
  - **Turnover constraints**: L1-norm turnover limit from benchmark portfolio via buy/sell decomposition
  - **Factor model for returns**: Correlated scenario returns via multi-factor model with sector-linked loadings

### Performance Comparison

| Metric | Old Generator | New Generator |
|---|---|---|
| Constraints (100 vars) | 2-3 | ~204 |
| Constraints (500 vars) | 2-3 | ~931 |
| Trivial solves (≤2 iters) | 39.2% | 0% |
| Median iterations (100 vars) | ~2 | ~38 |
| Median iterations (500 vars) | ~2 | ~177 |

### Feasibility Handling

- **Feasible**: Constructs a reference portfolio from benchmark weights and widens all constraints to accommodate it with randomized slack
- **Infeasible** (4 modes): (1) impossibly tight CVaR limit, (2) asset class lower bounds summing > 1, (3) position limits summing < 1, (4) near-zero turnover with conflicting sector caps
- **Unknown**: 70/30 feasible/infeasible split

### Files Modified

- `src/problem_types/portfolio.jl` — complete rewrite

---

### Bug Fixes: Feasibility Handling and Batch Generation Script

**Previous Commit**: `b679ada`

**Summary**: Fixed feasibility handling in 6 problem generators that previously ignored the `feasibility_status` parameter entirely, and fixed bugs in `scripts/generate_lps.jl` batch generation script.

### Fixed

- **`generate_lps.jl` seed handling**: When no `--seed` was provided (default `seed=0`), all problems received `problem_seed=0`, causing every instance of the same type with the same target variables to produce identical LPs. Now each problem gets a unique seed from the script's RNG regardless of whether `--seed` is specified.
- **`generate_lps.jl` error reporting**: Generation failures were silently swallowed unless `--verbose` was used. Now always emits warnings for failures and prints a warning line in the summary when any problems fail.
- **`ProductionPlanningProblem`**: Added `min_production` field and feasibility handling. For `infeasible` status, sets minimum production levels that exceed resource capacity.
- **`PortfolioProblem`**: Added `min_total_return` field and feasibility handling. For `infeasible` status, sets a minimum return constraint above what's achievable under risk constraints.
- **`ProjectSelectionProblem`**: Added `min_selected` field and feasibility handling. For `infeasible` status, requires selecting more projects than the budget allows.
- **`LoadBalancingProblem`**: Added `max_utilization` field and feasibility handling. For `infeasible` status, caps maximum utilization below what's required to satisfy demands.
- **`KnapsackProblem`**: Added `min_value` field and proper feasibility handling. For `infeasible` status, requires more total value than achievable under capacity constraint. Previously had a TODO comment about infeasibility.
- **`NetworkFlowProblem`**: Added feasibility handling for both `feasible` and `infeasible` statuses. For `feasible`, ensures target flow is within achievable range. For `infeasible`, sets target flow above max flow capacity.

### Details

Each of the 6 fixed generators previously accepted the `feasibility_status` parameter but ignored it, always producing problems with the same random feasibility regardless of what was requested. With these fixes:
- `feasible` status guarantees a feasible LP
- `infeasible` status guarantees an infeasible LP
- `unknown` status randomly selects between feasible (70%) and infeasible (30%)

All changes follow the existing architecture: new constraint data is stored in struct fields (set in the constructor with all randomness), and `build_model` remains completely deterministic.

---

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
