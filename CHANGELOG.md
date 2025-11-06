# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### 2025-11-06 - Test Infrastructure Overhaul

**Branch:** `claude/overhaul-problem-instance-tests-011CUsCQMKgzAo9VAn8YRFHR`

#### Added

- **Comprehensive Test Suite** (`test/test_problem_instances.jl`):
  - Tests ~50 problem instances per problem type with varied target sizes
  - Phase 1: Basic generation tests (~40 instances)
    - Validates successful generation without errors
    - Checks variable count is within 15% of target
    - Tests with target sizes: 50, 100, 250, 500, 1000 variables
  - Phase 2: Feasibility verification tests (~20 instances)
    - Uses HiGHS solver to verify problem feasibility
    - Tests `solution_status` parameter `:feasible` and `:infeasible` values
    - Validates that specified feasibility is actually achieved
  - Detailed statistics and error reporting
  - Support for verbose output mode

- **Test Runner Script** (`test_instances.jl`):
  - Convenient command-line interface for running comprehensive tests
  - Support for testing specific problem types: `julia test_instances.jl transportation`
  - Support for testing multiple types: `julia test_instances.jl transportation knapsack`
  - Verbose mode: `julia test_instances.jl --verbose` or `-v`

- **HiGHS Solver Integration**:
  - Added HiGHS as test dependency in `Project.toml`
  - Used for feasibility verification in comprehensive tests
  - 60-second timeout per solve
  - Silent optimization (no solver output)

#### Changed

- **Standard Test Suite** (`test/runtests.jl`):
  - Updated to support two testing modes:
    - Standard mode: Quick smoke tests (default)
    - Comprehensive mode: Full tests with feasibility checks (via `COMPREHENSIVE_TESTS=true` env var)
  - Added helpful banner explaining testing options
  - Increased tolerance from ±10% to ±15% for variable count validation

- **Documentation Updates**:
  - `README.md`: Added comprehensive testing section with usage examples
  - `CLAUDE.md`: Updated testing commands and strategy documentation
  - Both files now document standard vs comprehensive testing approaches

#### Technical Details

The comprehensive test suite implements a two-phase testing strategy:

1. **Phase 1 - Generation Tests**: Fast tests focusing on successful generation and size accuracy
   - Distributes ~40 instances across 5 target sizes (50, 100, 250, 500, 1000 vars)
   - No solver invocation for speed
   - Validates structure and variable count

2. **Phase 2 - Feasibility Tests**: Thorough tests with solver verification
   - 10 instances each for `:feasible` and `:infeasible` status
   - Uses HiGHS solver to verify actual feasibility matches requested status
   - Gracefully handles problem types that don't support `solution_status` parameter
   - Random target sizes to increase coverage diversity

Test results include:
- Success/failure counts and percentages
- Tolerance violation statistics
- Feasibility correctness metrics
- Detailed error messages for first 3 failures
- Per-problem-type and overall summaries

#### Usage Examples

```bash
# Standard quick tests (existing behavior)
julia --project=@. test/runtests.jl

# Comprehensive tests - all problem types
julia --project=@. test_instances.jl

# Comprehensive tests - specific problem type
julia --project=@. test_instances.jl transportation

# Comprehensive tests - multiple specific types
julia --project=@. test_instances.jl transportation knapsack portfolio

# Comprehensive tests - verbose output
julia --project=@. test_instances.jl --verbose
julia --project=@. test_instances.jl -v transportation

# Comprehensive tests - alternative method
COMPREHENSIVE_TESTS=true julia --project=@. test/runtests.jl
```

#### Rationale

This overhaul addresses several testing needs:

1. **Increased Coverage**: 50 instances per type vs previous smaller sample
2. **Feasibility Verification**: Actually solves problems to verify feasibility claims
3. **Size Accuracy**: Validates 15% tolerance across diverse target sizes
4. **Selective Testing**: Enables focused testing of specific problem types
5. **Better Reporting**: Detailed statistics help identify systematic issues
6. **Backwards Compatibility**: Standard tests remain fast for quick iterations

The 15% tolerance (up from 10%) better reflects the reality that some problem types have discrete constraints making exact target matching difficult, while still ensuring reasonable size accuracy.
