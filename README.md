# SyntheticLPs.jl

A standardized framework for generating synthetic linear programming (LP) problem instances. The goal is to generate problems that are highly realistic and can be used to test and develop LP solvers.

Requires Julia 1.11 or later.

## Overview

This package provides:

- A unified interface for generating various types of LP problems using multiple dispatch
- Problem generators implemented as concrete types inheriting from `ProblemGenerator`
- Controllable problem feasibility (feasible, infeasible, or unknown)
- Target variable count generation - specify approximate number of variables
- Deterministic problem generation with reproducible seeds
- Easy extensibility for new problem types

## Problem Types

The package includes generators for 45 common LP/MIP problem categories, all
unified with a standardized interface. Each category groups one or more
**variants** — concrete formulations with their own data generation and model
structure (see [Categories and Variants](#categories-and-variants)). Categories
with more than one variant are annotated below.

- Transportation — variants: `standard`, `balanced`, `capacitated`, `transshipment`, `emission_constrained`, `fixed_charge`
- Diet Problem — variants: `standard`, `nutrient_bounds`, `food_groups`
- Knapsack — variants: `standard`, `multidimensional`, `bounded`, `mixed_integer_set`
- Portfolio Optimization — variants: `cvar` (institutional CVaR), `tracking_error` (index tracking under a tracking-error budget)
- Network Flow — variants: `standard`, `generalized_flow`
- Multi-Commodity Flow — variants: `standard`, `binary_capacity`, `integer_flow`
- Production Planning
- Assignment — variants: `standard`, `workload_balance`
- Blending — variants: `standard`, `equipment_batches`, `multi_product`
- Airline Crew
- Bin Packing — variants: `standard` (identical bins with handling conflicts), `heterogeneous` (typed fleet with capacity, cost, availability, and eligibility differences)
- Container Loading — variants: `standard`, `two_dimensional_bin_packing`
- Cutting Stock — variants: `standard`, `setup_cost`, `due_dates`, `integer_patterns`
- Energy — variants: `standard`, `ramping`, `reserves`, `storage`, `transmission`, `dc_opf`, `optimal_transmission_switching`
- Facility Location — variants: `standard`, `two_echelon`, `p_median`
- Feed Blending
- Generic MILP
- Graph Optimization — variants: `independent_set`, `generalized_independent_set`, `vertex_cover`, `vertex_coloring`, `map_labeling`, `quasi_clique`
- Hub Location — variants: `p_hub_median` (tight four-index single allocation), `compact_single_allocation` (origin-indexed `O(n^3)` formulation), `r_allocation` (primary/backup hubs), `multiple_allocation` (fixed-charge AP-style routing), `capacitated` (loose/tight AP profiles), `hub_covering` (OD service thresholds), `hub_network` (modular regional links), `budgeted_backbone` (exact-p capacitated link investment); see [generator notes](docs/hub_location.md)
- Inventory — variants: `standard`, `lot_sizing`, `multi_item`, `multi_echelon`
- Inverse Optimization — variants: `classical` (exact weighted-L1 objective recovery), `noisy_observations` (multi-context absolute-suboptimality fitting), `shortest_path` (spatial inverse routing); see [generator notes](docs/inverse_optimization.md)
- Job Shop Scheduling
- Land Use
- Load Balancing — variants: `standard`, `discrete_placement`
- Maritime Inventory Routing
- Neural Network Verification
- Nurse Scheduling
- Operating Room Scheduling — variants: `elective_assignment` (advance scheduling to MSS blocks), `case_sequencing` (daily allocation + sequencing), `weekly_planning` (multi-day planning with sequential ICU-to-ward beds), `master_surgical_schedule` (tactical cyclic block allocation), `robust_elective` (sparse Bertsimas--Sim duration uncertainty), `benchmark_loading` (Leeftink--Hans empirical case types and load factors); see [generator notes](docs/operating_room_scheduling.md)
- Product Mix
- Project Selection
- Regression — variants: `lad`, `quantile`, `chebyshev`, `basis_pursuit` (weighted sparse recovery)
- Radiotherapy — variants: `weighted_deviation` (voxelwise piecewise-linear IMRT fluence-map planning), `mean_tail_dose` (volume-weighted CVaR/DVH surrogate), `minmax_deviation` (worst-voxel epigraph), `robust_fluence` (coherent setup-shift scenarios), `beam_angle_selection` (joint field selection and FMO MILP); see [generator notes](docs/radiotherapy.md)
- Resilient Network Design
- Resource Allocation
- Revenue Management — variants: `standard` (deterministic network LP / bid-price), `stochastic_overbooking` (scenario show-ups with denied-service recourse)
- Scheduling
- Set System — variants: `set_cover`, `set_packing`, `set_partitioning`, `combinatorial_auction`
- Stochastic Program (two-stage with recourse; dual block-angular structure)
- Supply Chain — variants: `standard`, `single_source`, `carbon`, `multi_product`, `network_planning` (multi-period, multi-product LP with sparse lanes, specialized production, shared capacity, inventory carryover, and regional/seasonal/disruption profiles)
- Crop Planning
- Telecom Network Design
- TSP — variants: `standard` (symmetric lifted MTZ), `asymmetric` (one-way-street ATSP), `flow` (single-commodity flow), `time_windows` (appointment delivery), `assignment_relaxation` (strengthened degree LP), `prize_collecting` (quota tour), `multiple_salespersons` (balanced fleet), `precedence` (ordered tasks)
- Unit Commitment
- Vehicle Routing — variants: `cvrp` (capacitated vehicle routing, single-commodity-flow formulation)
- Workforce Shift Scheduling — variant: `covering` (multi-skill, profile-driven shift-pattern staffing LP)

Several categories ship multiple variants — for example `energy` has `standard`
(generation mix) and `dc_opf` (DC optimal power flow), and `regression` has
`lad`, `quantile`, `chebyshev`, and `basis_pursuit` — selectable via the
`variant=` keyword or a `"category/variant"` reference (see below).

`supply_chain/network_planning` accepts targets through 1,000,000 variables;
larger requests raise `ArgumentError` before allocating the sparse arc data.
Its status metadata is explicit: only feasible requests store a
`feasible_witness`, only infeasible requests store an
`infeasibility_certificate`, and unknown requests store a correlated
`nominal_scenario`. Unknown scenarios preserve local lane service while varying
network-wide production/resource conditions, so they naturally include both
feasible and infeasible instances without acting as a hidden infeasible mode.

`telecom_network_design/standard` follows the same conventions: it accepts
targets through 1,000,000 variables and raises `ArgumentError` above that, and
its metadata is status-specific — a `feasible_witness` (the planted routing and
the links it installs) for feasible requests, and an
`infeasibility_certificate` for infeasible ones, either a capacity cut or a
budget shortfall. Both certificate modes are proved from LP rows alone, so the
instances stay infeasible under the default `relax_integer=true`. Demand is
calibrated against the planted routing rather than sampled independently, so
unknown requests are a genuine mix at every scale rather than drifting
infeasible as the network grows.

The `hub_location` family follows the same status discipline, grounded in the
classical benchmark datasets (CAB airline passengers; AP postal volumes with
`chi = 3`, `alpha = 0.75`, `delta = 2` leg costs). Feasible requests plant a
witness (hub set plus allocations, or a sized backbone) and infeasible requests
store a relaxation-proof certificate — disjoint reach regions that need more
than `p` hubs, an impossible exact hub count, an uncovered service-threshold
OD pair, an opening/link budget below a valid lower bound, total hub capacity
below total flow, or a regional gateway cut whose crossing capacity cannot
carry the inter-regional traffic. Every certificate refutes the LP relaxation,
not just the MIP.

## Usage

### Basic Usage

```julia
using SyntheticLPs
using JuMP
using Clp  # or any other LP solver

# List available problem categories
categories = list_problem_types()   # alias for list_categories()

# Get information about a category (its description, variants, default variant)
info = problem_info(:transportation)

# Generate a problem with target variable count (uses the category's default variant)
model, problem = generate_problem(:transportation, 100, unknown, 0)

# The problem instance contains all the generated data
println("Number of sources: ", problem.n_sources)
println("Number of destinations: ", problem.n_destinations)

# Solve the model
set_optimizer(model, Clp.Optimizer)
optimize!(model)
solution_summary(model)
```

### Categories and Variants

Each problem domain is a **category** (e.g. `:transportation`) that groups one or
more **variants** — concrete formulations with their own data generation and
model. A `ProblemVariant` names one variant of one category and prints as
`category/variant`.

```julia
# Every registered category and variant
list_categories()                       # [:airline_crew, :assignment, ...]
list_variants(:portfolio)               # [:cvar, :tracking_error]
list_problems()                         # [ProblemVariant(:airline_crew, :standard), ...]

# Select a specific variant — three equivalent forms
model, problem = generate_problem(:portfolio, 100, unknown, 0; variant=:cvar)
model, problem = generate_problem(ProblemVariant(:portfolio, :cvar), 100, unknown, 0)
model, problem = generate_problem(ProblemVariant("portfolio/cvar"), 100, unknown, 0)

# Variant-level metadata
problem_info(:portfolio, :cvar)         # Dict with :description, :type, ...
```

`regression/basis_pursuit` builds the weighted LP
`min Σⱼ wⱼ|xⱼ|` subject to exact measurements `Ax = b`, using nonnegative
positive/negative splits. Its stored `profile` selects a whitened Gaussian,
coherent-column, or sparse measurement matrix. The stored `source_signal`
generates the RHS before status handling and is an explicit witness only when
`resolved_status == feasible`. Infeasible instances instead carry a
`BasisPursuitCertificate` describing their contradictory proportional
measurement rows and inconsistent RHS. Because each feature requires two split
variables, even targets of at least two are exact, odd targets round up by one,
and smaller targets produce two variables.

The `inverse_optimization` family learns forward objective coefficients from
observed decisions. Its three pure-LP variants cover classical exact inverse
linear optimization, regularized absolute-suboptimality fitting over noisy
multi-context decision panels, and inverse shortest paths on sparse spatial road
networks. Cost normalization prevents the all-zero degeneracy; sparse
right-skewed production data, bounded behavioral-noise profiles, and
road-class-calibrated travel times provide diverse grounded structure. Feasible
instances carry exact primal/dual or shortest-path-potential witnesses, and
infeasible instances carry a contradictory admissible-cost-set certificate.

The `radiotherapy` family builds reduced but spatially grounded IMRT fluence-map
LPs and MILPs. Six TG-119/CORT-style anatomy profiles generate contoured 3-D
voxel samples with volume weights, 6/10 MV field geometries, balanced 2-D
beamlet grids, and sparse depth-attenuated pencil-beam influence matrices.
Five variants cover summed and minimax voxel deviation, volume-weighted
hot/cold mean-tail constraints, coherent rigid-setup scenarios, and joint
beam-angle selection. All include adjacent-beamlet total variation. Requested
feasible/infeasible instances carry auditable witnesses/certificates; unknown
instances retain genuine status uncertainty. These are synthetic optimization
benchmarks, not clinical dose calculations or patient treatment plans.

### Feasibility Control

```julia
# Generate a guaranteed feasible problem
model, problem = generate_problem(:transportation, 100, feasible, 0)

# Generate a guaranteed infeasible problem
model, problem = generate_problem(:diet_problem, 100, infeasible, 0)

# Generate a problem with unknown feasibility (randomized)
model, problem = generate_problem(:portfolio, 100, unknown, 0)
```

The modernized crop-planning, feed-blending, land-use, unit-commitment,
bin-packing, and revenue-management generators also store auditable status
artifacts. Feasible instances carry a complete primal witness; infeasible
instances carry a structural certificate that can be checked without a solver.
Their category documentation names the fields and validation helpers. Unknown
instances either make no claim or record the profile to which they resolved.

Generators honor the requested status by construction, but a few use heuristic
feasibility logic that occasionally misses. Pass an `optimizer` to **verify and
guarantee** the contract — the model is solved on a copy and rebuilt with a new
seed on mismatch (up to `max_feasibility_retries=10` times):

```julia
using HiGHS
# Guaranteed to solve INFEASIBLE (regenerated with a different seed if not)
model, problem = generate_problem(:energy, 300, infeasible, 1; optimizer=HiGHS.Optimizer)
```

With `optimizer` unset (the default) no solving is performed. Verification is itself
deterministic — retries walk `seed, seed+1, …` — so a given `(seed, optimizer)` pair
always resolves to the same model. `generate_dataset` records the resolved seed per
instance so verified datasets can be rebuilt without re-solving.

Each verification solve is bounded by `feasibility_timeout` (default 10s). A solve
that certifies nothing — it exceeds that limit, or returns a status that separates
neither case — raises rather than being counted as a contract violation, so a slow
solve is never misreported as a bad instance. Unrelaxed MIPs are the usual cause:

```julia
model, problem = generate_problem(:job_shop_scheduling, 2000, feasible, 2;
                                  relax_integer=false, optimizer=HiGHS.Optimizer,
                                  feasibility_timeout=120.0)
```

### Bound Reformulation

By default, variable bounds are emitted as JuMP/MOI variable bounds. Pass
`bounds_to_constraints=true` to instead reformulate them as explicit affine
constraints — useful for generating LPs in a more standard-form-like shape.
A plain `x ≥ 0` nonnegativity bound is left as a variable bound; every other
bound (upper bounds, fixed values, and nonzero lower bounds) becomes a row.

```julia
# Bounds (other than x ≥ 0) become explicit constraint rows
model, problem = generate_problem(:knapsack, 100, unknown, 0; bounds_to_constraints=true)

# Or apply it to an already-built JuMP model in place
bounds_to_constraints!(model)
```

The reformulation runs *after* integrality relaxation, so bounds introduced by
relaxing integer/binary variables (e.g. `0 ≤ x ≤ 1`) are converted too. Because
the converted bounds become genuine rows, they are now counted by
`num_constraints(model; count_variable_in_set_constraints=false)`.

### Dual Reformulation

Dualization is off by default. Its main use is adding reproducible formulation
diversity to random generation: pass `dualize_probability` to independently
choose whether each returned model is primal or dual. Integrality relaxation
runs first by default, followed by the optional bounds-to-constraints transform,
and dualization runs last when selected.

```julia
# Randomly return a primal or dual formulation with equal probability
model, ref, problem = generate_random_problem(100; seed=7,
                                              dualize_probability=0.5)
is_dual_reformulation(model)  # reports the sampled choice

# Apply the same per-instance sampling to a dataset
instances = generate_dataset(num_problems=100, seed=7,
                             dualize_probability=0.5)

# Force the dual for a specific generated LP
dual_model, problem = generate_problem(:transportation, 100, unknown, 0;
                                       dualize=true)

# Or dualize an existing continuous JuMP model (the primal is unchanged)
dual_model = dualize_model(model)
```

Dual variables and constraints use `dual_var_` and `dual_con_` name prefixes.
Unrelaxed integer or binary variables are rejected because mixed-integer models
do not have an LP/conic dual. When an optimizer is supplied for feasibility
contract verification, SyntheticLPs verifies the generated primal before
dualizing it: an infeasible primal can have either an infeasible or an unbounded
dual. Dataset generation also accepts `dualize=true`, records it in the manifest,
and computes size and quality metadata from the models actually returned. Each
`GeneratedInstance.dualized` value and manifest instance entry records the sampled
choice. For individual models, use `is_dual_reformulation(model)`. Use
`dualize=true` to force every random or dataset instance to be dualized.

### Reproducible Generation with Seeds

```julia
# Generate the same problem twice with the same seed
seed = 12345
model1, problem1 = generate_problem(:knapsack, 50, unknown, seed)
model2, problem2 = generate_problem(:knapsack, 50, unknown, seed)

# These will be identical
@assert num_variables(model1) == num_variables(model2)
@assert problem1.n_items == problem2.n_items
```

Every generator draws from a constructor-local `MersenneTwister(seed)`, so
generation neither reads nor advances the caller's global RNG. The seed is the
only thing that determines an instance: surrounding `Random.seed!` calls, other
generation happening on the same task, and concurrent generation on other
threads cannot perturb it.

### Random Problem Generation

```julia
# Generate a random problem of any variant targeting ~100 variables
model, ref, problem = generate_random_problem(100)

# `ref` is a ProblemVariant (prints as `category/variant`)
println("Problem: $ref")           # e.g. "transportation/standard"
println("Variables: $(num_variables(model))")

# Generate with feasibility control
model, ref, problem = generate_random_problem(200; feasibility_status=feasible)

# Solve the model
set_optimizer(model, Clp.Optimizer)
optimize!(model)
solution_summary(model)
```

### Batch Dataset Generation

To build a whole dataset of LP instances (e.g. for training an ML model), use
`generate_dataset`. It repeatedly samples a problem type and a target variable
count, builds each model, and optionally solves it to filter out low-quality
instances. When `output_dir` is set, each kept instance is written to disk
along with a `manifest.json` describing the run; metadata is always returned.

```julia
using SyntheticLPs, Distributions

# Generate 100 instances with variable counts drawn from a truncated normal,
# writing .mps files plus a manifest to ./dataset. By default, the generator
# keeps an accepted candidate pool and selects instances whose actual model
# sizes match the target distribution closely.
instances = generate_dataset(
    num_problems = 100,
    size_distribution = truncated(Normal(500, 200), 50, 2000),
    output_dir = "dataset",
    seed = 1234,            # 0 = non-deterministic; any other value is reproducible
)

for inst in instances[1:3]
    println("$(inst.problem_type): $(inst.num_variables) vars, " *
            "$(inst.num_constraints) constraints → $(inst.filename)")
end
```

Uniform targets are supported as well. To make each selected problem type match
the same size distribution independently, enable per-type matching:

```julia
instances = generate_dataset(
    num_problems = 120,
    size_distribution = Uniform(50, 2000),
    problem_types = [:transportation, :knapsack, :portfolio],
    match_size_by_type = true,
    candidate_multiplier = 2,
    output_dir = "dataset_by_type",
    seed = 1234,
)
```

The package itself is solver-agnostic. To enable **quality filtering** — solving
each instance and rejecting trivial, degenerate, unbounded, timed-out, or
ill-conditioned ones — pass an `optimizer`:

```julia
using SyntheticLPs, HiGHS

instances = generate_dataset(
    num_problems = 100,
    output_dir = "dataset",
    quality_filter = true,
    optimizer = HiGHS.Optimizer,
    optimizer_attributes = ("solver" => "simplex",),
    quality_criteria = QualityCriteria(
        solve_timeout = 30.0,
        min_constraints = 5,
        min_iterations = 3,
        max_iteration_ratio = 100.0,
    ),
    max_retries = 10,       # raw retry budget for failures / filtered candidates
    feasible_only = true,
)
```

A single instance can also be evaluated directly with
`check_quality(model, HiGHS.Optimizer)`.

## Extending with New Categories and Variants

Each category lives in its own folder under `src/problem_types/<category>/`:

```
src/problem_types/transportation/
    transportation.jl   # category entry point: includes the variant file(s)
    standard.jl         # a variant: struct + constructor + build_model + register_variant
```

**To add a new variant to an existing category**, create a file in that
category's folder, implement it (see template below), and `include` it from the
category's `<category>.jl` entry point.

**To add a new category**, create `src/problem_types/<category>/<category>.jl`
(the entry point), add at least one variant file, and add a single
`include("problem_types/<category>/<category>.jl")` line to `src/SyntheticLPs.jl`.

A category is created automatically by its first variant's `register_variant`
call (which supplies the category description). Call `register_category(:cat,
"…")` explicitly in the entry point only when you want a category-level
description distinct from its variants (typical once a category has several).

Variant file template:

```julia
using JuMP
using Random

"""
    YourProblem <: ProblemGenerator

Generator for your custom problem type.

# Fields
- `field1::Type1`: Description
- `field2::Type2`: Description
"""
struct YourProblem <: ProblemGenerator
    field1::Type1
    field2::Type2
    # ... store all generated data needed to build the model
end

"""
    YourProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a problem instance.

# Arguments
- `target_variables`: Target number of variables
- `feasibility_status`: Desired feasibility (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function YourProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    rng = MersenneTwister(seed)

    # Sample all parameters based on target_variables (draw from `rng`, never
    # from the global stream: `rand(rng, ...)`, `randn(rng)`, `shuffle(rng, ...)`)
    # Generate all deterministic data
    # Handle feasibility status
    # ...

    return YourProblem(field1_value, field2_value, ...)
end

"""
    build_model(prob::YourProblem)

Build a JuMP model from the problem instance. This function must be deterministic.
"""
function build_model(prob::YourProblem)
    model = Model()

    # Define variables
    # Define constraints
    # Define objective

    return model
end

# Register this variant under its category (lazily creates the category)
register_variant(
    :your_category,
    :your_variant,
    YourProblem,
    "Description of this variant"
)
```

The category entry point (`src/problem_types/your_category/your_category.jl`)
then just includes the variant file(s):

```julia
include("your_variant.jl")
```

Key principles:
- The struct stores ALL data needed to deterministically build the model
- ALL randomness goes in the constructor
- Randomness comes from a constructor-local `MersenneTwister(seed)`, never from
  `Random.seed!` and the global stream. Thread that `rng` explicitly through any
  helper the constructor calls (`helper(rng::AbstractRNG, ...)`)
- `build_model` must be completely deterministic (no RNG calls)
- Handle `feasible`, `infeasible`, and `unknown` feasibility statuses

## Command Line Interface

The package includes a command-line script for generating problems:

```bash
# Generate a transportation problem with ~100 variables (category default variant)
julia --project=@. scripts/generate_problem.jl transportation 100 problem.mps

# Select a specific variant with category/variant
julia --project=@. scripts/generate_problem.jl portfolio/cvar 100 problem.mps

# List all categories and their variants
julia --project=@. scripts/generate_problem.jl list

# Generate a feasible knapsack problem with ~50 variables and solve it
julia --project=@. scripts/generate_problem.jl knapsack 50 --feasible --solve

# Generate an infeasible diet problem with ~100 variables
julia --project=@. scripts/generate_problem.jl diet_problem 100 --infeasible output.mps

# Reformulate variable bounds (other than x >= 0) into explicit constraints
julia --project=@. scripts/generate_problem.jl knapsack 50 --bounds-to-constraints

# Generate a random problem with ~200 variables
julia --project=@. scripts/generate_problem.jl random 200

# Use a specific seed for reproducibility
julia --project=@. scripts/generate_problem.jl portfolio 150 --seed=12345

# List all available problem types
julia --project=@. scripts/generate_problem.jl list
```

For generating whole datasets, `scripts/generate_lps.jl` is a thin command-line
wrapper around `generate_dataset` (it supplies HiGHS for quality filtering):

```bash
# Generate 100 .mps instances into ./output
julia --project=scripts scripts/generate_lps.jl -o output -n 100

# Generate 50 feasible, quality-filtered instances with progress output
julia --project=scripts scripts/generate_lps.jl -o output -n 50 --feasible-only -q -v

# Restrict to specific problem types and a fixed seed
julia --project=scripts scripts/generate_lps.jl --problem-types transportation,knapsack -n 20 --seed 42

# Reformulate variable bounds (other than x >= 0) into explicit constraints
julia --project=scripts scripts/generate_lps.jl -o output -n 20 --bounds-to-constraints

# Uniform actual-size matching for each selected problem type
julia --project=scripts scripts/generate_lps.jl -o output -n 60 \
  --problem-types transportation,knapsack,portfolio \
  --size-distribution uniform --var-min 50 --var-max 2000 \
  --match-size-by-type --candidate-multiplier 2 --seed 42
```

## Testing

Run the test suite to verify all problem generators:

```julia
using Pkg; Pkg.activate(".")
Pkg.test()
```

The test suite validates:
- All problem generators work correctly
- Target variable counts are achieved within ±10% tolerance
- Generated problems are valid and can be solved


## License

SyntheticLPs.jl  
Copyright (C) 2025  Felix Parker

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details.

The full text of the license is available in the [LICENSE](LICENSE) file.
