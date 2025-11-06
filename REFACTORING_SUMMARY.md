# Problem Generator Refactoring Summary

This document summarizes the refactoring of 18 problem type files to use the new `ProblemGenerator` architecture.

## Completed Refactorings

### 1. network_flow.jl ✓
- **Status**: Completed
- **Complexity**: Low-Medium
- **Key Changes**:
  - Created `NetworkFlowProblem` struct with arc data, capacities, costs
  - Constructor handles topology generation and parameter sampling
  - `build_model` is now deterministic given struct data
- **Preserved Logic**: Connected network generation, source-sink topology

### 2. production_planning.jl ✓
- **Status**: Completed
- **Complexity**: Low
- **Key Changes**:
  - Simple struct with products, resources, profits, usage matrix
  - Direct variable count mapping (variables = products)
  - Straightforward constructor and build_model

## Remaining Refactorings

### Simple (Direct Mapping, Minimal Feasibility Logic)

#### 3. project_selection.jl
- **Complexity**: Low-Medium
- **Struct Fields Needed**:
  ```julia
  struct ProjectSelectionProblem <: ProblemGenerator
      n_projects::Int
      costs::Dict{Int,Float64}
      returns::Dict{Int,Float64}
      risk_scores::Dict{Int,Float64}
      dependencies::Vector{Tuple{Int,Int}}
      budget::Float64
      risk_budget::Float64
      max_high_risk_projects::Int
      high_risk_threshold::Float64
  end
  ```
- **Key Logic to Preserve**:
  - Quality factor correlations for costs/returns/risks
  - Dependency graph generation (acyclic)
  - Multiple risk categories (low/medium/high)
  - LogNormal and Beta distributions for realistic parameters
- **Feasibility**: Minimal - budget is set as fraction of total cost

### Medium Complexity (Some Feasibility Control)

#### 4. load_balancing.jl
- **Complexity**: Medium
- **Struct Fields Needed**:
  ```julia
  struct LoadBalancingProblem <: ProblemGenerator
      n_nodes::Int
      links::Vector{Tuple{Int,Int}}
      capacities::Dict{Tuple{Int,Int},Float64}
      demands::Dict{Tuple{Int,Int},Float64}
      paths::Dict{Tuple{Int,Int},Vector{Tuple{Int,Int}}}
  end
  ```
- **Key Logic to Preserve**:
  - Network topology with spanning tree for connectivity
  - Path generation for each demand
  - Truncated Normal for capacities, Gamma for demands
- **Feasibility**: Natural (path-finding ensures structural feasibility)

#### 5. assignment.jl
- **Complexity**: Medium-High
- **Struct Fields Needed**: Workers, tasks, costs, availability, skill matching
- **Key Logic to Preserve**:
  - Worker-task compatibility matrix
  - Skill-based constraints
  - Cost distributions
- **Feasibility Logic**: Ensure bipartite matching is possible

#### 6. blending.jl
- **Complexity**: Medium-High
- **Struct Fields Needed**: Ingredients, nutrients, costs, nutritional content, bounds
- **Key Logic to Preserve**:
  - Ingredient cost-nutrient correlations
  - Nutritional requirement ranges
  - Realistic ingredient properties
- **Feasibility Logic**: Moderate - ensure nutritional requirements are achievable

#### 7. facility_location.jl
- **Complexity**: Medium-High
- **Struct Fields Needed**: Facilities, customers, locations, fixed costs, shipping costs, demands, capacities
- **Key Logic to Preserve**:
  - Geographic positioning and clustering
  - Distance-based shipping costs
  - Economies of scale in capacity
- **Feasibility Logic**: Ensure total capacity >= total demand (for feasible)

### Complex (Sophisticated Feasibility Control)

#### 8. feed_blending.jl
- **Complexity**: High
- **Similar to**: blending.jl but more sophisticated
- **Key Logic to Preserve**:
  - Multiple animal types with different nutritional needs
  - Ingredient palatability and digestibility
  - Cost-quality tradeoffs
- **Feasibility Logic**: Complex nutritional constraint satisfaction

#### 9. inventory.jl
- **Complexity**: High
- **Struct Fields Needed**: Periods, capacities, demands, costs (production, holding, backlog), initial inventory, backlog flag
- **Key Logic to Preserve**:
  - Seasonality and trend generation
  - Demand disruptions (Poisson-distributed)
  - Cumulative demand calculations for feasibility
  - Prefix shortfall computation
- **Feasibility Logic**: VERY SOPHISTICATED
  - Feasible: Capacity buffer, safety stock, demand smoothing, surgical capacity adjustments
  - Infeasible: Sustained high demand, capacity cuts, supplier disruptions, just-below-required capacity
  - Uses helper functions: `cum()`, `max_shortfall()`, iterative feasibility passes

#### 10. resource_allocation.jl
- **Complexity**: Very High
- **Struct Fields Needed**: Activities, resources, profits, usage matrix, resource caps, min_levels, quality factors
- **Key Logic to Preserve**:
  - Quality factor correlations between profit and usage
  - Minimum activity level constraints
- **Feasibility Logic**: VERY SOPHISTICATED
  - Constructive generation with plan-driven floors
  - Demand plan computation based on profit/usage ratios
  - Capacity anchoring with theta factors
  - Baseline plan scaling to fit within capacities
  - Infeasible: Iterative floor raising to violate capacity on selected resources
  - Multi-resource violation strategies

### Very Complex (Extensive Feasibility Engineering)

#### 11. diet_problem.jl ⚠️ MOST COMPLEX
- **Complexity**: Extremely High
- **Struct Fields Needed**: Foods, nutrients, costs, nutritional content, nutrient bounds, portion bounds
- **Key Logic to Preserve**: ALL FEASIBILITY SCENARIOS
  - **Scenario 1**: Sufficient nutrients from affordable subset
  - **Scenario 2**: Expensive foods needed for rare nutrients
  - **Scenario 3**: Tight budgets requiring cheap bulk foods
  - **Scenario 4**: Conflicting bounds (vegetarian

 needing nutrients from meat)
  - **Scenario 5**: Near-impossible combinations
  - Each scenario has detailed food selection, cost setting, bound adjustments
- **Infeasibility Scenarios**:
  1. Budget too low for cheapest feasible diet
  2. Conflicting nutrient bounds
  3. Essential nutrient missing from all foods
  4. Portion size conflicts
  5. Vegetarian constraints blocking essential nutrients
- **Special Features**:
  - Food categories (grains, proteins, vegetables, etc.)
  - Nutrient correlations (protein foods have iron, etc.)
  - Dietary restrictions (vegetarian, vegan, etc.)
  - Budget-nutritional quality tradeoffs

#### 12. land_use.jl
- **Complexity**: Very High
- **Struct Fields Needed**: Parcels, zoning types, parcel sizes, costs, revenues, resource consumption, capacities, restrictions, adjacency matrix, minimum zoning counts
- **Key Logic to Preserve**:
  - Geographic modeling with parcel sizes
  - Environmental restrictions per parcel
  - Adjacency constraints (e.g., industrial not next to residential)
  - Minimum zoning diversity requirements
- **Feasibility Logic**: EXTREMELY SOPHISTICATED
  - Constructive witness assignment respecting adjacency
  - Type 1 vs Type 3 adjacency conflicts
  - Greedy selection with disjoint sets
  - Iterative assignment with swapping and rebalancing
  - Edge pruning to enable infeasible adjacency combinations
  - Resource capacity tightening based on witness
  - Infeasible: Capacity set below provable lower bound

#### 13. scheduling.jl
- **Complexity**: Very High
- **Struct Fields Needed**: Workers, shifts, days, availability matrix, costs, staffing requirements, worker constraints, skill data
- **Key Logic to Preserve**:
  - Realistic availability patterns (full-time vs part-time)
  - Shift premiums (night, weekend, holiday)
  - Worker tiers (junior, regular, senior)
  - Skill-based scheduling
  - Cost distributions (LogNormal, Beta)
- **Feasibility Logic**: EXTREMELY SOPHISTICATED
  - Per-shift capacity capping with randomized slack
  - Per-day reserve factors
  - Global capacity reserve
  - Consecutive-day run capacity computation
  - Randomized matching with window constraints
  - Iterative demand satisfaction with slack utilization
  - Worker minimum enforcement with capacity freeing
  - Infeasible modes: shift blackout, day overload, min-over-cap

#### 14. supply_chain.jl
- **Complexity**: Very High
- **Struct Fields Needed**: Facilities, customers, locations, fixed costs, demands, capacities, transport modes, transport costs, mode capacities, infrastructure availability
- **Key Logic to Preserve**:
  - Geographic clustering with Dirichlet weights
  - Market potential calculations
  - Infrastructure availability by mode and distance
  - Terrain/volume/efficiency factors in transport costs
  - Capacity correlations with fixed costs
- **Feasibility Logic**: SOPHISTICATED
  - Universal fallback mode connectivity
  - K-nearest facility linking
  - Approximate demand share calculations
  - Capacity smoothing for local demand
  - Aggregate capacity adjustments
  - Infeasible: Transport capacity shortfall (70-95% of demand)

#### 15. cutting_stock.jl
- **Complexity**: High
- **Key Logic**: Patterns generation, stock sizes, item demands, cutting patterns
- **Feasibility**: Pattern generation must cover demands

#### 16. airline_crew.jl
- **Complexity**: Very High
- **Key Logic**: Flights, crew members, qualifications, pairings, rest requirements
- **Feasibility**: Complex pairing and rest constraints

#### 17. energy.jl
- **Complexity**: Medium-High
- **Key Logic**: Power plants, time periods, demand curves, ramp rates, minimum up/down times
- **Feasibility**: Generation must meet demand with operational constraints

## Refactoring Pattern

For each file, follow this pattern:

### 1. Create Struct
```julia
"""
    ProblemName <: ProblemGenerator

Description of the problem.

# Fields
- `field1::Type`: Description
- `field2::Type`: Description
...
"""
struct ProblemName <: ProblemGenerator
    # Store GENERATED DATA (costs, capacities, etc.), not size parameters
    field1::Type
    field2::Type
    ...
end
```

### 2. Implement Constructor
```julia
"""
    ProblemName(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a problem instance.

# Arguments
- `target_variables`: Target number of variables
- `feasibility_status`: Desired feasibility (Feasible(), Infeasible(), Unknown())
- `seed`: Random seed
"""
function ProblemName(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # 1. Determine problem scale and sample size parameters
    # 2. Generate all random data (costs, capacities, demands, etc.)
    # 3. Apply feasibility logic based on feasibility_status
    #    - Feasible(): Ensure constraints can be satisfied
    #    - Infeasible(): Create provable violations
    #    - Unknown(): Natural generation
    # 4. Return struct with all generated data

    return ProblemName(field1, field2, ...)
end
```

### 3. Implement build_model
```julia
"""
    build_model(prob::ProblemName)

Build a JuMP model from the problem data.
Should be deterministic given the struct.
"""
function build_model(prob::ProblemName)
    model = Model()

    # Create variables
    # Set objective
    # Add constraints

    return model
end
```

### 4. Register
```julia
register_problem(
    :symbol,
    ProblemName,
    "Description"
)
```

## Critical Considerations

### For ALL Files
1. **Preserve ALL original logic** - especially feasibility control
2. **Move ALL randomness to constructor** - build_model must be deterministic
3. **Store generated DATA** - not generation parameters
4. **Handle feasibility_status** - implement Feasible(), Infeasible(), Unknown() cases
5. **Use exact same distributions** - LogNormal, Beta, Gamma, etc.

### For Complex Feasibility Logic
1. **Keep all helper functions** - can be inside constructor or as separate functions
2. **Preserve exact algorithms** - e.g., greedy assignment, iterative refinement
3. **Maintain verification steps** - e.g., check that infeasibility is provable
4. **Keep fallback strategies** - e.g., edge pruning, capacity adjustments
5. **Document complex logic** - explain what each feasibility scenario does

### Special Cases

**diet_problem.jl**:
- Must preserve all 5 feasibility scenarios AND all 5 infeasibility scenarios
- Keep food categorization and nutrient correlation logic
- Maintain dietary restriction handling
- This is the MOST complex file - take extra care

**inventory.jl**:
- Preserve cumulative demand calculations
- Keep prefix shortfall logic
- Maintain seasonality, trends, and disruptions
- Preserve surgical feasibility adjustments

**resource_allocation.jl**:
- Keep demand plan computation
- Preserve capacity anchoring logic
- Maintain baseline plan scaling
- Keep iterative floor-raising for infeasibility

**scheduling.jl**:
- Preserve consecutive-day capacity computation
- Keep randomized matching with window constraints
- Maintain three infeasibility modes
- Preserve worker capacity enforcement logic

**land_use.jl**:
- Keep witness assignment algorithm
- Preserve adjacency conflict handling
- Maintain type-specific selection strategies
- Keep edge pruning logic

**supply_chain.jl**:
- Preserve geographic clustering with Dirichlet
- Keep infrastructure availability logic
- Maintain K-nearest facility connectivity
- Preserve capacity smoothing

## Testing Strategy

After refactoring each file:
1. Run existing tests to ensure backward compatibility
2. Verify variable counts match targets (±10%)
3. Test all three feasibility statuses
4. Check that infeasible instances are provably infeasible
5. Verify determinism: same seed → same model

## Estimated Effort

- Simple files (production_planning, project_selection): 30-60 minutes each
- Medium files (load_balancing, assignment, blending): 1-2 hours each
- Complex files (inventory, feed_blending, facility_location): 2-4 hours each
- Very complex files (diet_problem, resource_allocation, scheduling, land_use, supply_chain): 4-8 hours each

**Total estimated effort**: 40-80 hours for all 18 files

## Priority Order

Recommended refactoring order (easiest to hardest):
1. ✓ production_planning (completed)
2. ✓ network_flow (completed)
3. project_selection
4. load_balancing
5. assignment
6. blending
7. energy
8. facility_location
9. feed_blending
10. cutting_stock
11. airline_crew
12. product_mix (already has good structure)
13. inventory
14. supply_chain
15. resource_allocation
16. land_use
17. scheduling
18. diet_problem (save for last - most complex)

## Notes

- Some files may need additional utility functions (e.g., distance calculations, clustering)
- Consider creating shared utility modules for common operations
- The FeasibilityStatus handling is critical - don't skip the Unknown() case
- Document any deviations from original logic with comments
- Keep original comments that explain domain-specific logic
