# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context

This package is a standardized framework for generating synthetic linear programming (LP) problem instances. The goal is to generate problems that are highly realistic and can be used to test and develop LP solvers.

## General Instructions

- Make sure to explore the relevant code carefully before making any plans or changes.
- Update the changelog file after making any significant changes. Organize the changelog with sections per date, and keep track of the commit hash and the current datetime for each set of changes. Include high level summaries of the changes as well as specific details with more granular information than commit messages.
- This project is under active development and is not yet stable, so never worry about making breaking changes or backwards compatibility.
- When making major changes, always update the README.md and CLAUDE.md files to reflect the changes.
- This package is intended for research use only, so it does not need to be extremely robust and handle all edge cases.

## Commands

### Testing

The suite uses HiGHS (a test-only dependency in `[extras]`). Both commands work:

```bash
# Full suite including the solver-based feasibility-contract testsets:
julia --project=@. -e 'using Pkg; Pkg.test()'

# Direct run: executes the solver-free testsets and skips the solver-based ones
# (HiGHS is not resolvable outside the Pkg.test sandbox):
julia --project=@. test/runtests.jl
```

`test/runtests.jl` loads HiGHS lazily (`HAS_HIGHS` flag) so the direct command
does not error — the two feasibility-verification testsets are simply skipped
with an `@info` notice when HiGHS is unavailable.

### Problem Generation

Generate a specific problem type targeting ~100 variables:
```bash
julia --project=@. scripts/generate_problem.jl transportation 100 output.mps
```

List all available problem types:
```bash
julia --project=@. scripts/generate_problem.jl list
```

Generate and solve a feasible problem with ~50 variables:
```bash
julia --project=@. scripts/generate_problem.jl knapsack 50 --feasible --solve
```

Generate an infeasible problem with ~100 variables:
```bash
julia --project=@. scripts/generate_problem.jl diet_problem 100 --infeasible
```

Generate a random problem with ~200 variables:
```bash
julia --project=@. scripts/generate_problem.jl random 200
```

### Dataset Generation

Generate a whole dataset of LP instances via the library API (`generate_dataset`)
or its CLI wrapper. The wrapper supplies HiGHS, so use the `scripts` environment:

```bash
# 100 .mps instances into ./output
julia --project=scripts scripts/generate_lps.jl -o output -n 100

# Quality-filtered, feasible-only, with progress
julia --project=scripts scripts/generate_lps.jl -o output -n 50 --feasible-only -q -v
```

### Development

Start Julia REPL with project loaded:
```bash
julia --project=@.
```

## Architecture

SyntheticLPs uses a type-based dispatch system for generating realistic linear programming problems. Problems are organized as a two-level hierarchy: a **category** (a problem domain, e.g. `:transportation`) groups one or more **variants** (concrete generators with their own data generation and model formulation, e.g. `:standard`). There are 43 categories; most have a single variant, while several carry multiple variants with distinct formulations. All generators follow a consistent pattern using Julia's multiple dispatch.

### Core Components

**Main Module** (`src/SyntheticLPs.jl`):
- `ProblemGenerator`: Abstract base type for all problem generators
- `FeasibilityStatus`: Enum with values `feasible`, `infeasible`, `unknown`
- `ProblemVariant`: identifier for a `category/variant` pair (the canonical reference used throughout); constructible from `(category, variant)` symbols, a bare category symbol (→ default variant), or a `"category"`/`"category/variant"` string; prints as `category/variant`
- Two-level registry `LP_REGISTRY::Dict{Symbol,CategorySpec}` populated by `register_category()` and `register_variant()` (a single variant lazily creates its category)
- Unified interface functions: `generate_problem()` (accepts a category symbol with optional `variant=` keyword, a `ProblemVariant`, or a generator type), `list_categories()`/`list_problem_types()` (alias), `list_variants()`, `list_problems()`, `problem_info()`
- **Feasibility-contract verification**: every `generate_problem()`/`generate_random_problem()` overload accepts an optional `optimizer` (plus `max_feasibility_retries=10` and `feasibility_timeout=10.0`). When supplied and the requested status is `feasible`/`infeasible`, the built model is solved on a copy and the termination status is classified by the pure `_classify_termination(ts, status)` into one of three verdicts:
  - `:holds` — proved; return the model.
  - `:violated` — disproved (e.g. `INFEASIBLE` for a `feasible` request, or `OPTIMAL`/`DUAL_INFEASIBLE` for an `infeasible` one, both of which exhibit a feasible point). Rebuild with the next seed, up to `max_feasibility_retries` times.
  - `:inconclusive` — certifies nothing: `TIME_LIMIT`, `ALMOST_OPTIMAL`, `INFEASIBLE_OR_UNBOUNDED` (separates neither case, so it must never certify infeasibility), or any other status. Raises immediately rather than spending the retry budget re-asking an unanswerable question. Unrelaxed MIPs are the common trigger — raise `feasibility_timeout`.

  This is the project-level backstop for the few generators whose heuristic feasibility logic occasionally misses (~0.1% of requests corpus-wide). Central in `generate_problem` (not per-variant); with `optimizer=nothing` (default) generation is unchanged. Verification is deterministic — retries walk `seed, seed+1, …` — so a given `(seed, optimizer)` pair always resolves to the same model. `generate_dataset` records the resolved seed per instance so materialization reproduces it without re-solving, and skips verification when `quality_filter` is on (`check_quality` already solves the model and rejects anything not matching `feasible_only`, so verifying too would solve every candidate twice).
- Random problem generation with `generate_random_problem()` (returns the selected `ProblemVariant`)
- Base function `build_model(problem::ProblemGenerator)` that each variant implements

**Model transforms** (`src/transforms.jl`):
- Post-`build_model` reformulations of the finished JuMP model, applied centrally in `generate_problem()` (not per-variant), exactly like `relax_integrality`.
- `bounds_to_constraints!(model)`: reformulates variable bounds as explicit affine constraints, keeping a plain `x ≥ 0` nonnegativity bound but converting all other bounds (upper, fixed, nonzero lower). Exposed everywhere as the keyword `bounds_to_constraints::Bool=false` on `generate_problem()`/`generate_random_problem()`/`generate_dataset()` (and `--bounds-to-constraints` on both CLI scripts). Runs *after* `relax_integer`, so relaxation-introduced bounds are converted too. Converted bounds become genuine rows, so they raise `num_constraints(...; count_variable_in_set_constraints=false)` and thus affect dataset size-matching and quality-filter thresholds.

**Dataset Generation** (`src/dataset.jl`):
- `generate_dataset(; kwargs...)`: builds a whole dataset of LP instances by sampling problem types and target variable counts; optionally writes instance files + a `manifest.json` and returns `Vector{GeneratedInstance}` metadata. Fully reproducible from a non-zero `seed`.
- `check_quality(model, optimizer; ...)` + `QualityCriteria`/`QualityResult`: solve-based filtering of trivial/degenerate/unbounded/ill-conditioned instances.
- The package stays solver-agnostic: quality filtering requires the caller to pass an `optimizer` (e.g. `HiGHS.Optimizer`). `scripts/generate_lps.jl` is a thin CLI wrapper that supplies HiGHS.

**Problem Generators** (`src/problem_types/<category>/`):
- Each category is a folder containing:
  - A `<category>.jl` entry point that `include`s the category's variant file(s) (and optionally calls `register_category(:category, "description")` for a category-level description)
  - One file per variant (or closely related group of variants), e.g. `standard.jl`
- Each variant file has:
  - A struct inheriting from `ProblemGenerator` containing all generated data
  - A constructor `VariantStruct(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)`
  - An implementation of `build_model(prob::VariantStruct)` that deterministically builds the JuMP model
  - A `register_variant(:category, :variant, VariantStruct, "description")` call (which lazily creates the category if needed)
- Structs store ALL data needed to build the model (costs, capacities, demands, etc.)
- Constructors contain ALL randomness; `build_model` is completely deterministic
- Randomness comes from a constructor-local `rng = MersenneTwister(seed)`, never
  from `Random.seed!` and the global stream (which would clobber the caller's
  RNG and is not thread-safe). Every draw passes it explicitly — `rand(rng, …)`,
  `randn(rng)`, `shuffle(rng, …)`, `sample(rng, …)` — and any helper a
  constructor calls takes it as its first parameter, `helper(rng::AbstractRNG, …)`.
  The `Global RNG Isolation` testset in `test/runtests.jl` enforces this across
  every registered variant.
- `supply_chain/network_planning` is capped at a documented 1,000,000-variable
  target because each sparse shipment coordinate is represented in multiple
  dictionaries and JuMP structures; larger targets raise `ArgumentError`
  instead of silently undersizing. Its optional metadata fields are
  status-specific (`feasible_witness`, `infeasibility_certificate`, or
  `nominal_scenario`), while disruption metadata depends only on the profile.
- `telecom_network_design/standard` follows the same convention: capped at a
  documented 1,000,000-variable target (`TELECOM_MAX_VARIABLES`) raising
  `ArgumentError` above it, with status-specific `feasible_witness` /
  `infeasibility_certificate` fields.
- Status-specific typed witnesses and certificates are the general pattern for
  generators whose feasibility is planted rather than hoped for. Besides the
  above, `hub_location` (all variants), `product_mix`, `airline_crew`,
  `nurse_scheduling`, `neural_network_verification`, and
  `maritime_inventory_routing` each store a typed witness for `feasible`
  requests and a typed certificate for `infeasible` ones, and neither for
  `unknown`. Where the category is a MIP, prefer a certificate built from LP
  rows alone so the infeasibility survives the default `relax_integer=true`.

**Utility Scripts**:
- `scripts/generate_problem.jl`: Command-line interface for problem generation
- `scripts/analyze_problem_statuses.jl`: Analysis of problem feasibility

### Problem Generator Pattern

A category folder `src/problem_types/<category>/` contains an entry point and one
file per variant:

```julia
# src/problem_types/<category>/<category>.jl  (entry point)
# Optionally: register_category(:category, "Category-level description")
include("standard.jl")   # include each variant file
```

```julia
# src/problem_types/<category>/standard.jl  (a variant)
struct VariantStruct <: ProblemGenerator
    # Store all generated data needed to build the model
    field1::Type1
    field2::Type2
    # ...
end

function VariantStruct(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    rng = MersenneTwister(seed)

    # Sample parameters based on target_variables
    # Generate all deterministic data using rand(rng, ...) calls
    # Handle feasibility status (feasible, infeasible, unknown)

    return VariantStruct(field1_value, field2_value, ...)
end

function build_model(prob::VariantStruct)
    model = Model()

    # Build JuMP model using only prob's fields
    # This must be completely deterministic (no RNG calls)

    return model
end

# Registers the variant; lazily creates the :category if it doesn't exist yet.
register_variant(:category, :standard, VariantStruct, "Description")
```

### Key Design Principles

1. **Separation of Concerns**: Randomness (constructor) vs. determinism (build_model)
2. **Reproducibility**: Same seed → identical problem instance → identical model,
   independent of the caller's global RNG state or concurrent generation
3. **Feasibility Control**: Generators can produce guaranteed feasible/infeasible problems
4. **Type Safety**: Each problem is a distinct type with its own data structure
5. **Dispatch**: Use Julia's multiple dispatch for clean, extensible interface

### Available Problem Categories

The system includes 43 categories covering major LP/MIP problem classes. Each
category's default variant is `:standard` except `graph_optimization`
(`:independent_set`), `hub_location` (`:p_hub_median`),
`neural_network_verification` (`:relu_big_m`), `portfolio`
(`:cvar`), `regression` (`:lad`), `set_system` (`:set_cover`),
`vehicle_routing` (`:cvrp`), and `workforce_shift_scheduling` (`:covering`).
Categories with multiple variants are listed with them below.
- Transportation (`standard`, `balanced`, `capacitated`, `transshipment`, `emission_constrained`, `fixed_charge`), Diet Problem (`standard`, `nutrient_bounds`, `food_groups`), Knapsack (`standard`, `multidimensional`, `bounded`, `mixed_integer_set`), Portfolio (`cvar`, `tracking_error`), Network Flow (`standard`, `generalized_flow`)
- Multi-Commodity Flow (`standard`, `binary_capacity`, `integer_flow`), Assignment (`standard`, `workload_balance`), Blending (`standard`, `equipment_batches`, `multi_product`), Container Loading (`standard`, `two_dimensional_bin_packing`), Facility Location (`standard`, `two_echelon`, `p_median`), Hub Location (`p_hub_median`, `compact_single_allocation`, `r_allocation`, `multiple_allocation`, `capacitated`, `hub_covering`, `hub_network`, `budgeted_backbone`)
- Cutting Stock (`standard`, `setup_cost`, `due_dates`, `integer_patterns`), Energy (`standard`, `ramping`, `reserves`, `storage`, `transmission`, `dc_opf`, `optimal_transmission_switching`), Inventory (`standard`, `lot_sizing`, `multi_item`, `multi_echelon`), Load Balancing (`standard`, `discrete_placement`)
- Graph Optimization (`independent_set`, `generalized_independent_set`, `vertex_cover`, `vertex_coloring`, `map_labeling`, `quasi_clique`), Set System (`set_cover`, `set_packing`, `set_partitioning`, `combinatorial_auction`), Supply Chain (`standard`, `single_source`, `carbon`, `multi_product`, `network_planning`), Operating Room Scheduling (`elective_assignment`, `case_sequencing`, `weekly_planning`, `master_surgical_schedule`, `robust_elective`, `benchmark_loading`)
- TSP (`standard`, `asymmetric`, `flow`, `time_windows`, `assignment_relaxation`, `prize_collecting`, `multiple_salespersons`, `precedence`), Vehicle Routing (`cvrp`), Regression (`lad`, `quantile`, `chebyshev`, `basis_pursuit`), Workforce Shift Scheduling (`covering`), Bin Packing (`standard`, `heterogeneous`), Revenue Management (`standard`, `stochastic_overbooking`)
- Single-variant categories: Airline Crew, Crop Planning, Feed Blending, Generic MILP, Job Shop Scheduling, Land Use, Maritime Inventory Routing, Neural Network Verification, Nurse Scheduling, Product Mix, Production Planning, Project Selection, Resilient Network Design, Resource Allocation, Scheduling, Stochastic Program, Telecom Network Design, Unit Commitment

#### Model classes (LP / MIP / LP relaxation)

The corpus deliberately mixes three model classes; treat the names accordingly:
- **Pure LPs**: continuous formulations (e.g. transportation variants, diet
  variants, blending variants, most energy variants, `network_flow/generalized_flow`,
  both portfolio variants `cvar`/`tracking_error`, `supply_chain/network_planning`
  (multi-period, multi-product planning with inventory), regression variants,
  revenue management, stochastic program, and `workforce_shift_scheduling/covering`).
- **MIPs** (binary/integer variables): e.g. `facility_location` variants
  (including `p_median`), all eight `hub_location` variants (hub opening,
  allocation, and backbone-build binaries; their LP relaxations are the
  classical tight SKO path relaxation and multicommodity design relaxations,
  grounded in the CAB/AP benchmark conventions), `cutting_stock/setup_cost`,
  `inventory/lot_sizing`,
  `bin_packing`, `job_shop_scheduling`, `supply_chain/single_source`,
  `unit_commitment` (binary commitment/startup/shutdown in its natural model),
  `knapsack/multidimensional` (binary) and `knapsack/bounded` (integer),
  `assignment/workload_balance` (binary min-makespan), and `vehicle_routing/cvrp`
  (binary arc selection with single-commodity-flow subtour elimination — its
  continuous relaxation is a genuine depot-anchored routing relaxation, not a
  degenerate one). The `tsp` MIP variants `standard`/`asymmetric` (MTZ order
  variables), `flow` and `prize_collecting` (single-commodity flow),
  `time_windows` (big-M time propagation), `multiple_salespersons` (lifted
  route-order variables), and `precedence` (lifted MTZ plus ordering rows)
  likewise deliver genuine routing relaxations. The
  `operating_room_scheduling` variants are also real mixed-integer
  formulations: `elective_assignment` (binary surgery-to-block assignment),
  `weekly_planning` (binary surgery-to-day assignment with bed occupancy), and
  `case_sequencing` (binary room/surgeon assignment plus big-M ordering
  variables), plus `master_surgical_schedule` (binary tactical block design),
  `robust_elective` (a sparse budgeted-uncertainty counterpart), and
  `benchmark_loading` (empirically calibrated OR-day assignment). These are
  real mixed-integer formulations. Also genuine MIPs: `nurse_scheduling`
  (binary nurse-to-shift assignments under coverage, skill-mix, availability,
  shift/weekend/night bounds, consecutive-day limits and post-night rest — its
  `feasible_witness` is an *integral* roster satisfying the MIP as well as its
  relaxation, and its skill-shortage `infeasibility_certificate` refutes the
  relaxation too), `airline_crew` (binary set partitioning over pairings that
  are operationally legal by construction), `neural_network_verification`
  (binary ReLU phase indicators; its LP relaxation is the triangle relaxation
  of the big-M encoding), `telecom_network_design` (binary link installation),
  and `maritime_inventory_routing` (binary vessel positions and sailing legs).
  For all four of the latter, infeasible instances are refuted by LP rows
  alone, so they stay infeasible after relaxation.
- **Purpose-built LP relaxations of MIPs**: continuous relaxations useful as
  LP-solver test instances but *not* directly implementable integer solutions —
  notably `tsp/assignment_relaxation` (a strengthened
  degree LP relaxation of the TSP — fractional arc covers that may contain
  subtours).
  Its docstring states this explicitly. In addition, the public generation API
  defaults to `relax_integer=true`, so every natural MIP above—including unit
  commitment and both bin-packing variants—is returned as an LP relaxation
  unless the caller opts out.

When generating an LP-only corpus, filter out the MIP variants (or relax them);
when characterizing instances, do not present the relaxations as real-world
integer schedules.

### Testing Strategy

Tests are split by scope. `test/runtests.jl` holds only framework-level
coverage; everything specific to one problem category lives in
`test/problem_types/<category>.jl`, so a generator's source, documentation, and
regression coverage evolve as one reviewable unit.

- `test/runtests.jl`:
  - `test_problem_generator(ref)`, applied to every registered variant: target
    variable counts, all three feasibility statuses, model structure
    (variables, constraints, objective), and reproducibility under a fixed seed
  - Registry and interface tests, global-RNG isolation, dataset generation, the
    bounds-to-constraints transform, the pure `_classify_termination` table, and
    the generic half of the feasibility-contract machinery (retry budget, seed
    walk, pristine-model guarantee)
  - An include loop that pulls in every `test/problem_types/*.jl` in sorted order
- `test/problem_types/<category>.jl`: focused quality contracts for one
  category — registry shape, exact variable-count formulas, data invariants,
  witness/certificate arithmetic, edge-size robustness, and solver-backed
  feasibility contracts. Add coverage for a new category here, not in
  `runtests.jl`.
  - Files run in ambient scope (`include` evaluates at module top level), so
    `MOI`, `HAS_HIGHS`, `Uniform`, and the `using` imports from `runtests.jl`
    are visible without re-importing.
  - The include loop sits *outside* the `if HAS_HIGHS` guard at the tail of
    `runtests.jl`, so each file must guard its own solver-dependent tests with
    `if HAS_HIGHS` (the established pattern is `@testset ... begin` wrapping an
    inner `if HAS_HIGHS`). Otherwise the direct
    `julia --project=@. test/runtests.jl` run errors instead of skipping.

## Adding New Categories and Variants

### Add a variant to an existing category

1. Create a new file in the category folder, e.g. `src/problem_types/<category>/<variant>.jl`
2. Define a struct inheriting from `ProblemGenerator` with all necessary data fields
3. Implement constructor `VariantStruct(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)`
4. Implement `build_model(prob::VariantStruct)` (must be deterministic)
5. Call `register_variant(:category, :variant, VariantStruct, "Description")` (pass `default=true` to make it the category default)
6. `include("<variant>.jl")` from the category's `<category>.jl` entry point
7. Extend `test/problem_types/<category>.jl` with the variant's quality
   contracts (create the file if the category has none yet — the include loop
   picks it up automatically)
8. Run tests to verify implementation

### Add a new category

1. Create `src/problem_types/<category>/<category>.jl` (the entry point) and at least one variant file (steps 1–5 above)
2. The entry point `include`s the variant file(s); add `register_category(:category, "Description")` there only if you want a category-level description distinct from its variants
3. Add `include("problem_types/<category>/<category>.jl")` to `src/SyntheticLPs.jl`
4. Create `test/problem_types/<category>.jl` with the category's quality
   contracts; it is included automatically, so nothing in `test/runtests.jl`
   needs to change
5. Run tests to verify implementation

Key principles:
- Struct stores ALL generated data needed to build the model
- Constructor contains ALL randomness and parameter sampling, drawn from a local
  `MersenneTwister(seed)` threaded explicitly into every helper
- `build_model` must be completely deterministic (no RNG calls)
- Handle all three feasibility statuses appropriately
