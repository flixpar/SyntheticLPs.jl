# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added - 2025-11-05

**Commit**: Starting from 6c1270f
**Date**: 2025-11-05

#### Crop Planning Problem Generator

Added a new problem generator for crop planning optimization problems. This expands the SyntheticLPs framework from 21 to 22 problem types.

**High-level summary**:
- Implemented realistic crop planning problem generator that models agricultural land allocation decisions
- Supports maximizing profit while satisfying resource constraints (land, water, labor)
- Includes crop diversity requirements and market demand limitations
- Guarantees feasibility or infeasibility based on `solution_status` parameter

**Detailed changes**:

1. **New file**: `src/problem_types/crop_planning.jl`
   - `generate_crop_planning_problem()`: Main generator function
   - `sample_crop_planning_parameters(target_variables)`: Parameter sampling targeting specific variable counts (±10% tolerance)
   - `sample_crop_planning_parameters(size)`: Legacy size-based parameter sampling
   - `calculate_crop_planning_variable_count()`: Variable count calculation helper

2. **Modified**: `src/SyntheticLPs.jl`
   - Added include statement for crop_planning.jl (line 239)
   - Maintains alphabetical ordering of problem type includes

**Problem characteristics**:
- **Decision variables**: Area (hectares) allocated to each crop
- **Objective**: Maximize net profit (revenue - production costs)
- **Constraints**:
  - Total land availability
  - Water capacity (irrigation resources)
  - Labor availability
  - Market demand limits per crop
  - Minimum area requirements for essential crops (food security)
  - Optional crop diversity constraints
- **Realistic data patterns**:
  - Crop yields: 2-40 tons/ha (LogNormal, varies by crop type)
  - Water requirements: 300-2500 mm/season (Normal, crop-specific)
  - Labor requirements: 25-250 hours/ha (Gamma distribution)
  - Prices: $150-1200/ton (LogNormal, market-based)
  - Production costs: $350-2500/ha (Normal, input-intensive)
- **Crop types modeled**: cereals (wheat, corn, rice), vegetables, legumes, industrial crops, oilseeds

**Feasibility guarantees**:
- **`:feasible`**: Constructs witness allocation, calculates resource needs, adds 10-30% slack
- **`:infeasible`**: Computes provable lower bounds on resource usage, sets capacity 5-30% below minimum
- **`:all`**: Random generation without guarantees (legacy behavior)

**Parameter scales**:
- Small (50-250 crops): Family farm level, 50-500 hectares
- Medium (250-1000 crops): Commercial farm, 500-5000 hectares
- Large (1000-10000 crops): Industrial agriculture, 5000-50000 hectares

**Research basis**:
Implementation based on literature review of agricultural linear programming:
- Multi-objective crop planning models (ScienceDirect)
- FAO crop water requirement guidelines
- Agricultural optimization case studies showing 60% diversity improvement and 143% profit enhancement
- Realistic yield, water, and labor data from agricultural research
