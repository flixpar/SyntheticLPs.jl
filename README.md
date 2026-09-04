# SyntheticLPs.jl

A standardized framework for generating synthetic linear programming (LP) problem instances. The goal is to generate problems that are highly realistic and can be used to test and develop LP solvers.

Requires Julia 1.11 or later.

## Development

The repository uses JuliaFormatter for Julia, Ruff for Python, and Aqua for
Julia package-quality checks. After installing the pinned Python development
requirements, initialize the Julia tooling environment once:

```bash
python3 -m pip install -r requirements-dev.txt
make setup
```

Run `make format` to apply formatting, `make lint` for formatting and static
quality checks, or `make check` for the complete lint-and-test suite. CI runs
the same checks on every pull request and push to `main`.

## Overview

This package provides:

- A unified, multiple-dispatch interface for generating many types of LP problems
- Controllable feasibility (feasible, infeasible, or unknown)
- Target variable count generation — specify the approximate number of variables
- Deterministic, reproducible generation from a seed
- Batch dataset generation with optional solve-based quality filtering
- Easy extensibility for new problem types

## Problem Categories

Each problem domain is a **category** (e.g. `:transportation`) grouping one or
more **variants** — concrete formulations with their own data generation and
model structure. There are 45 categories; the default variant is listed first.
Detailed notes for many categories live under [`docs/`](docs/README.md).

- `airline_crew` — `standard`
- `assignment` — `standard`, `workload_balance`
- `bin_packing` — `standard`, `heterogeneous`
- `blending` — `standard`, `equipment_batches`, `multi_product`
- `container_loading` — `standard`, `two_dimensional_bin_packing`
- `crop_planning` — `standard`
- `cutting_stock` — `standard`, `due_dates`, `integer_patterns`, `setup_cost`
- `diet_problem` — `standard`, `food_groups`, `nutrient_bounds`
- `energy` — `standard`, `dc_opf`, `optimal_transmission_switching`, `ramping`, `reserves`, `storage`, `transmission`
- `facility_location` — `standard`, `p_median`, `two_echelon`
- `feed_blending` — `standard`
- `generic_milp` — `standard`
- `graph_optimization` — `independent_set`, `generalized_independent_set`, `map_labeling`, `quasi_clique`, `vertex_coloring`, `vertex_cover`
- `hub_location` — `p_hub_median`, `budgeted_backbone`, `capacitated`, `compact_single_allocation`, `hub_covering`, `hub_network`, `multiple_allocation`, `r_allocation`
- `inventory` — `standard`, `lot_sizing`, `multi_echelon`, `multi_item`
- `inverse_optimization` — `standard`, `classical_normalized`, `linf`, `market_clearing`, `noisy_observations`, `restricted_optimal_value`, `shortest_path`, `shortest_path_layered`
- `job_shop_scheduling` — `standard`
- `knapsack` — `standard`, `bounded`, `mixed_integer_set`, `multidimensional`
- `land_use` — `standard`
- `load_balancing` — `standard`, `discrete_placement`
- `maritime_inventory_routing` — `standard`
- `multi_commodity_flow` — `standard`, `binary_capacity`, `integer_flow`
- `network_flow` — `standard`, `generalized_flow`
- `neural_network_verification` — `relu_big_m`
- `nurse_scheduling` — `standard`
- `operating_room_scheduling` — `elective_assignment`, `benchmark_loading`, `case_sequencing`, `master_surgical_schedule`, `robust_elective`, `weekly_planning`
- `portfolio` — `cvar`, `tracking_error`
- `product_mix` — `standard`
- `production_planning` — `standard`
- `project_selection` — `standard`
- `radiotherapy` — `weighted_deviation`, `beam_angle_selection`, `mean_tail_dose`, `minmax_deviation`, `robust_fluence`
- `regression` — `lad`, `basis_pursuit`, `chebyshev`, `quantile`
- `resilient_network_design` — `standard`
- `resource_allocation` — `standard`
- `revenue_management` — `standard`, `stochastic_overbooking`
- `scheduling` — `standard`
- `set_system` — `set_cover`, `combinatorial_auction`, `set_packing`, `set_partitioning`
- `stochastic_program` — `standard`
- `supply_chain` — `standard`, `carbon`, `multi_product`, `network_planning`, `single_source`
- `telecom_network_design` — `standard`
- `transportation` — `standard`, `balanced`, `capacitated`, `emission_constrained`, `fixed_charge`, `transshipment`
- `tsp` — `standard`, `assignment_relaxation`, `asymmetric`, `flow`, `multiple_salespersons`, `precedence`, `prize_collecting`, `time_windows`
- `unit_commitment` — `standard`
- `vehicle_routing` — `cvrp`
- `workforce_shift_scheduling` — `covering`

## Usage

### Basic usage

```julia
using SyntheticLPs
using JuMP
using HiGHS  # or any other LP solver

# Generate a problem with a target variable count (category default variant)
model, problem = generate_problem(:transportation, 100, unknown, 0)

# The problem instance holds all the generated data
problem.n_sources, problem.n_destinations

# Solve it
set_optimizer(model, HiGHS.Optimizer)
optimize!(model)
solution_summary(model)
```

### Categories and variants

A `ProblemVariant` names one variant of one category and prints as
`category/variant`.

```julia
list_categories()                       # [:airline_crew, :assignment, ...]
list_problem_types()                    # alias for list_categories()
list_variants(:portfolio)               # [:cvar, :tracking_error]
list_problems()                         # [ProblemVariant(:airline_crew, :standard), ...]

problem_info(:transportation)           # category description, variants, default
problem_info(:portfolio, :cvar)         # variant-level metadata

# Select a specific variant — three equivalent forms
model, problem = generate_problem(:portfolio, 100, unknown, 0; variant=:cvar)
model, problem = generate_problem(ProblemVariant(:portfolio, :cvar), 100, unknown, 0)
model, problem = generate_problem(ProblemVariant("portfolio/cvar"), 100, unknown, 0)
```

The corpus mixes pure LPs, natural MIPs, and purpose-built LP relaxations.
`generate_problem` defaults to `relax_integer=true`, so MIP variants are returned
as LP relaxations unless you opt out. A relaxation is not a valid integer
solution — `tsp/assignment_relaxation`, for instance, is a fractional degree
relaxation that may contain subtours.

### Feasibility control

```julia
model, problem = generate_problem(:transportation, 100, feasible, 0)
model, problem = generate_problem(:diet_problem, 100, infeasible, 0)
model, problem = generate_problem(:portfolio, 100, unknown, 0)
```

Generators honor the requested status by construction. Many plant an auditable
artifact on the problem struct: a `feasible_witness` (a complete primal solution)
for feasible requests and an `infeasibility_certificate` (a structural proof
checkable without a solver) for infeasible ones; the category documentation names
the fields and validation helpers.

A few generators use heuristic feasibility logic that occasionally misses. Pass an
`optimizer` to **verify and guarantee** the contract — the model is solved on a
copy and rebuilt with a new seed on mismatch (up to `max_feasibility_retries=10`
times):

```julia
using HiGHS
model, problem = generate_problem(:energy, 300, infeasible, 1; optimizer=HiGHS.Optimizer)
```

With `optimizer` unset (the default) no solving is performed. Verification is
deterministic — retries walk `seed, seed+1, …` — so a given `(seed, optimizer)`
pair always resolves to the same model, and `generate_dataset` records the
resolved seed so verified datasets can be rebuilt without re-solving.

Each verification solve is bounded by `feasibility_timeout` (default 10s). A solve
that certifies nothing — it times out, or returns a status separating neither case
— raises rather than being counted as a contract violation, so a slow solve is
never misreported as a bad instance. Unrelaxed MIPs are the usual cause:

```julia
model, problem = generate_problem(:job_shop_scheduling, 2000, feasible, 2;
                                  relax_integer=false, optimizer=HiGHS.Optimizer,
                                  feasibility_timeout=120.0)
```

### Bound reformulation

By default, variable bounds are emitted as JuMP/MOI variable bounds. Pass
`bounds_to_constraints=true` to reformulate them as explicit affine constraints,
for LPs in a more standard-form-like shape. A plain `x ≥ 0` bound is left alone;
every other bound (upper, fixed, nonzero lower) becomes a row.

```julia
model, problem = generate_problem(:knapsack, 100, unknown, 0; bounds_to_constraints=true)

# Or apply it to an already-built JuMP model in place
bounds_to_constraints!(model)
```

The reformulation runs *after* integrality relaxation, so bounds introduced by
relaxing integer/binary variables are converted too. The converted bounds are
genuine rows, so they are counted by
`num_constraints(model; count_variable_in_set_constraints=false)`.

### Dual reformulation

Dualization is off by default. Its main use is adding reproducible formulation
diversity to random generation: `dualize_probability` independently chooses
whether each returned model is primal or dual. Relaxation runs first, then the
optional bounds-to-constraints transform, and dualization last.

```julia
# Randomly return a primal or dual formulation with equal probability
model, ref, problem = generate_random_problem(100; seed=7, dualize_probability=0.5)
is_dual_reformulation(model)  # reports the sampled choice

instances = generate_dataset(num_problems=100, seed=7, dualize_probability=0.5)

# Force the dual for a specific generated LP
dual_model, problem = generate_problem(:transportation, 100, unknown, 0; dualize=true)

# Or dualize an existing continuous JuMP model (the primal is unchanged)
dual_model = dualize_model(model)
```

Dual variables and constraints use `dual_var_`/`dual_con_` name prefixes.
Unrelaxed integer or binary variables are rejected, since mixed-integer models
have no LP/conic dual. When an optimizer is supplied for feasibility verification,
the primal is verified *before* dualization (an infeasible primal may have either
an infeasible or an unbounded dual). Dataset size and quality metadata are
computed from the models actually returned, and each `GeneratedInstance.dualized`
value and manifest entry records the sampled choice.

### Reproducibility

```julia
model1, problem1 = generate_problem(:knapsack, 50, unknown, 12345)
model2, problem2 = generate_problem(:knapsack, 50, unknown, 12345)  # identical
```

Every generator draws from a constructor-local `MersenneTwister(seed)`, so
generation neither reads nor advances the caller's global RNG. The seed alone
determines an instance: surrounding `Random.seed!` calls, other generation on the
same task, and concurrent generation on other threads cannot perturb it.

### Random problem generation

```julia
# Random variant targeting ~100 variables; `ref` is a ProblemVariant
model, ref, problem = generate_random_problem(100)
println("Problem: $ref")           # e.g. "transportation/standard"

model, ref, problem = generate_random_problem(200; feasibility_status=feasible)
```

### Batch dataset generation

`generate_dataset` repeatedly samples a problem type and a target variable count,
builds each model, and optionally solves it to filter out low-quality instances.
When `output_dir` is set, each kept instance is written to disk along with a
`manifest.json`; metadata is always returned.

```julia
using SyntheticLPs, Distributions

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

By default the generator keeps an accepted candidate pool and selects instances
whose actual model sizes match the target distribution closely. Set
`match_size_by_type=true` to make each selected problem type match the size
distribution independently:

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

## Command Line Interface

```bash
# Generate a problem (category default variant, or an explicit category/variant)
julia --project=@. scripts/generate_problem.jl transportation 100 problem.mps
julia --project=@. scripts/generate_problem.jl portfolio/cvar 100 problem.mps

# List all categories and their variants
julia --project=@. scripts/generate_problem.jl list

# Feasibility control, solving, bound reformulation, seeds, random selection
julia --project=@. scripts/generate_problem.jl knapsack 50 --feasible --solve
julia --project=@. scripts/generate_problem.jl diet_problem 100 --infeasible output.mps
julia --project=@. scripts/generate_problem.jl knapsack 50 --bounds-to-constraints
julia --project=@. scripts/generate_problem.jl portfolio 150 --seed=12345
julia --project=@. scripts/generate_problem.jl random 200
```

For whole datasets, `scripts/generate_lps.jl` is a thin wrapper around
`generate_dataset` that supplies HiGHS for quality filtering:

```bash
# 100 .mps instances into ./output
julia --project=scripts scripts/generate_lps.jl -o output -n 100

# 50 feasible, quality-filtered instances with progress output
julia --project=scripts scripts/generate_lps.jl -o output -n 50 --feasible-only -q -v

# Restrict to specific problem types with a fixed seed
julia --project=scripts scripts/generate_lps.jl --problem-types transportation,knapsack -n 20 --seed 42

# Uniform actual-size matching per selected problem type
julia --project=scripts scripts/generate_lps.jl -o output -n 60 \
  --problem-types transportation,knapsack,portfolio \
  --size-distribution uniform --var-min 50 --var-max 2000 \
  --match-size-by-type --candidate-multiplier 2 --seed 42
```

## Extending with New Categories and Variants

Each category lives in its own folder under `src/problem_types/<category>/`:

```
src/problem_types/transportation/
    transportation.jl   # category entry point: includes the variant file(s)
    standard.jl         # a variant: struct + constructor + build_model + register_variant
```

**To add a variant to an existing category**, create a file in that category's
folder and `include` it from the category's `<category>.jl` entry point.

**To add a new category**, create `src/problem_types/<category>/<category>.jl`
(the entry point), add at least one variant file, and add a single
`include("problem_types/<category>/<category>.jl")` line to `src/SyntheticLPs.jl`.
A category is created automatically by its first variant's `register_variant`
call; call `register_category(:cat, "…")` explicitly in the entry point only when
you want a category-level description distinct from its variants.

Variant file template:

```julia
using JuMP
using Random

struct YourProblem <: ProblemGenerator
    # store all generated data needed to build the model
    field1::Type1
    field2::Type2
end

function YourProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    rng = MersenneTwister(seed)

    # Sample all parameters from target_variables, drawing from `rng` and never
    # from the global stream: rand(rng, ...), randn(rng), shuffle(rng, ...).
    # Generate all data and handle the feasibility status here.

    return YourProblem(field1_value, field2_value)
end

# Must be deterministic: no RNG calls.
function build_model(prob::YourProblem)
    model = Model()
    # variables, constraints, objective
    return model
end

# Registers the variant, lazily creating the category
register_variant(:your_category, :your_variant, YourProblem, "Description of this variant")
```

Key principles:
- The struct stores ALL data needed to deterministically build the model
- ALL randomness goes in the constructor, drawn from a constructor-local
  `MersenneTwister(seed)` threaded explicitly through any helper it calls
  (`helper(rng::AbstractRNG, ...)`)
- `build_model` must be completely deterministic
- Handle `feasible`, `infeasible`, and `unknown` feasibility statuses

## Testing

```julia
using Pkg; Pkg.activate(".")
Pkg.test()
```

The suite checks every registered variant: target variable counts land within
±25% of the request (relaxed for very small problems), all three feasibility
statuses work, models are structurally valid and solvable, generation is
reproducible, and no generator touches the global RNG.

## License

SyntheticLPs.jl  
Copyright (C) 2025  Felix Parker

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details.

The full text of the license is available in the [LICENSE](LICENSE) file.
