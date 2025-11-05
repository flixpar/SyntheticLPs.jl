# Changelog

All notable changes to this project will be documented in this file.

## 2025-11-05

### Added - Multi-Commodity Flow Problem Generator

**Commit Hash:** e391e7c
**Date/Time:** 2025-11-05

#### High-Level Summary

Implemented a new problem generator for multi-commodity flow problems, bringing the total number of problem types from 21 to 22. This generator creates realistic LP instances that model scenarios where multiple commodities (different goods, message types, or freight classes) must be routed through a shared network with capacity constraints.

#### Detailed Changes

**New Files:**
- `src/problem_types/multi_commodity_flow.jl` - Complete implementation of multi-commodity flow generator
- `test_multi_commodity_flow.jl` - Manual test script for verification

**Modified Files:**
- `src/SyntheticLPs.jl` - Added include statement for multi_commodity_flow.jl
- `README.md` - Updated problem type count from 21 to 22 and added Multi-Commodity Flow to the list

**Key Features:**
1. **Problem Formulation:**
   - Variables: flow[k, arc] for each commodity k on each arc
   - Total variables = n_commodities × n_arcs
   - Constraints: capacity limits, flow conservation, demand satisfaction

2. **Realistic Data Patterns:**
   - Connected network topology with strong connectivity (cycle-based)
   - Log-normal distribution for arc capacities (realistic infrastructure variation)
   - Log-normal distribution for commodity demands (some high-volume, many low-volume)
   - Distance-based costs with congestion factors
   - Diverse source-sink pairs mixing short-haul and long-haul commodities

3. **Feasibility Control:**
   - `:feasible` - Ensures sufficient capacity and connectivity for all commodities
   - `:infeasible` - Creates bottlenecks or excessive demand
   - `:all` - Natural randomness without forced feasibility

4. **Size Ranges:**
   - Small: 2-5 commodities, 10-50 arcs, 5-15 nodes (~50-250 variables)
   - Medium: 5-15 commodities, 30-100 arcs, 10-30 nodes (~250-1000 variables)
   - Large: 10-50 commodities, 50-500 arcs, 15-100 nodes (~1000-10000 variables)

5. **Applications Modeled:**
   - Telecommunications: routing different message types
   - Supply chain: shipping multiple product types
   - Transportation: moving different freight classes

**Implementation Details:**
- Follows the standard interface pattern with all required functions:
  - `generate_multi_commodity_flow_problem(params; seed)`
  - `sample_multi_commodity_flow_parameters(target_variables; seed)`
  - `sample_multi_commodity_flow_parameters(size; seed)` (legacy)
  - `calculate_multi_commodity_flow_variable_count(params)`
- Guarantees variable count within ±10% of target
- Ensures reproducibility with seed parameter
- Properly registered with the problem registry

**Testing:**
- Automatically included in comprehensive test suite (test/runtests.jl)
- Manual test script provided for detailed verification
- Tests parameter sampling, model generation, feasibility control, and reproducibility
