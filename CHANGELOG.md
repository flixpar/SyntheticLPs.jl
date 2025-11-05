# Changelog

All notable changes to this project will be documented in this file.

## 2025-11-05

### Added - Telecommunication Network Design Problem Generator

**Commit**: TBD
**Date**: 2025-11-05

#### High-Level Summary
Added a new problem generator for telecommunication network design problems. This generator creates realistic multicommodity network design instances where the goal is to minimize installation and routing costs while satisfying capacity constraints and traffic demands.

#### Detailed Changes

**New Files**:
- `src/problem_types/telecom_network_design.jl`: Complete implementation of telecommunication network design problem generator

**Modified Files**:
- `src/SyntheticLPs.jl`: Added include statement for telecom_network_design.jl

**Problem Type**: `:telecom_network_design`

**Features**:
- Multicommodity network flow formulation with discrete capacity installation
- Geographic node placement with realistic clustering patterns
- Proximity-based network topology generation ensuring connectivity
- Heterogeneous traffic demands with log-normal distribution
- Hub-to-hub traffic prioritization for realistic demand patterns
- Distance-based installation and flow costs
- Support for different capacity modules (OC-3, OC-12, OC-48, OC-192, OC-768)
- Feasibility control via `:solution_status` parameter:
  - `:feasible`: Ensures budget and capacity allow routing all demands
  - `:infeasible`: Sets budget below minimum spanning tree cost
  - `:all`: Natural random generation without guarantees

**Mathematical Formulation**:
- Variables:
  - Binary variables `y[arc]`: 1 if link is installed (n_arcs variables)
  - Continuous variables `f[k,arc]`: Flow of commodity k on arc (n_arcs × n_commodities variables)
  - Total: n_arcs × (n_commodities + 1) variables
- Objective: Minimize installation costs + routing costs
- Constraints:
  - Flow conservation at each node for each commodity
  - Capacity constraints: total flow ≤ installed capacity
  - Budget constraint on total installation cost

**Variable Count Calculation**:
```
total_variables = n_arcs × (n_commodities + 1)
```

**Realistic Parameters** (scale with problem size):
- Small (≤100 vars): 4-12 nodes, 5-25 arcs, 3-15 commodities, metro networks
- Medium (≤1000 vars): 8-35 nodes, 15-120 arcs, 10-80 commodities, regional networks
- Large (>1000 vars): 20-120 nodes, 50-600 arcs, 20-300 commodities, national/international networks

**Parameter Sampling**:
- Targets specified variable count within ±10% tolerance
- Automatically adjusts network density, geographic area, costs, and capacity modules based on problem size
- Realistic ratios between nodes, arcs, and commodities

**Testing**:
- Static verification of implementation structure
- Manual verification of variable count formula
- Parameter sampling logic validated for various target sizes
- Follows standard interface pattern used by all other problem generators

**Interface Compliance**:
- `generate_telecom_network_design_problem(params; seed)`: Main generator function
- `sample_telecom_network_design_parameters(target_variables; seed)`: Target-based parameter sampling
- `sample_telecom_network_design_parameters(size; seed)`: Legacy size-based sampling
- `calculate_telecom_network_design_variable_count(params)`: Variable count calculation
- Registered with problem registry using `register_problem()`

**Research Basis**:
Based on established literature on multicommodity network design, capacity planning in telecommunications, and realistic network optimization problems. Incorporates features from real-world telecom network planning including:
- Discrete capacity installation with step functions
- Geographic distribution and distance-based costs
- Heterogeneous demand patterns
- Survivability through network topology design
