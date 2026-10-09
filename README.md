# SyntheticLPs.jl

A standardized framework for generating synthetic linear programming (LP) problem
instances. The goal is problems realistic enough to test, benchmark, and train LP
solvers — pivot rules in particular — at any size from a few dozen to a million
variables.

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

- 50 problem categories and 142 variants, from transportation and network flow to
  mine planning, Markov decision processes, and game equilibria — 70 pure LPs and
  72 natural MIPs (returned as LP relaxations by default)
- Target variable counts that hold from tiny instances to 100k+ variables, with
  near-linear build times
- Controllable feasibility (`feasible`, `infeasible`, or `unknown`), usually backed
  by a planted witness or a solver-free infeasibility certificate, plus optional
  solver-based verification
- Deterministic, reproducible generation from a seed
- A registry with structure and domain tags, model classes, and size caps for
  selecting variants
- Practitioner-style model transforms (unit scaling, redundant aggregate rows,
  elastic rows, permutation) and dual reformulation for formulation diversity
- Planned, shardable batch dataset generation with size calibration, feasibility
  mixes, optional quality filtering, and a versioned manifest
- A generator audit tool that measures size fidelity, build time, presolve
  survival, and solve behavior at scale

## Problem Categories

Each problem domain is a **category** (e.g. `:transportation`) grouping one or
more **variants** — concrete formulations with their own data generation and
model structure. There are 50 categories and 142 variants; the default variant
is listed first. "Class" is the [model class](#selecting-and-filtering-variants)
of the category's variants. Every category has a page under
[`docs/`](docs/README.md).

| Category | Variants (default first) | Class | Models |
|---|---|---|---|
| `airline_crew` | `standard` | MIP | Crew pairing over legal pairings with base availability and block-hour rows |
| `assignment` | `standard`, `workload_balance` | MIP | Sparse field-service job assignment; workload balancing on unrelated workers |
| `bin_packing` | `standard`, `heterogeneous` | MIP | Identical-bin and typed-fleet one-dimensional packing |
| `blending` | `standard`, `multi_period`, `robust` | LP | Secondary-aluminium alloy blending: multi-plant, multi-period, Bertsimas–Sim robust |
| `container_loading` | `standard`, `two_dimensional_bin_packing` | MIP | ISO container fleet loading; two-stage guillotine 2D packing |
| `crop_planning` | `standard` | LP | Regional multi-year crop rotation with water, labour, and markets |
| `cutting_stock` | `standard`, `arc_flow`, `due_dates`, `setup_cost` | LP, MIP | Gilmore–Gomory pattern LPs (multi-stock, multi-period, multi-machine) and arc flow |
| `diet_problem` | `standard`, `food_aid`, `food_groups` | LP | Population diets, humanitarian food baskets with sourcing, multi-week menus |
| `economic_planning` | `dynamic_leontief`, `energy_system` | LP | PILOT-style dynamic input–output staircases; TIMES/MESSAGE energy-system models |
| `energy` | `standard`, `dc_opf`, `hydrothermal`, `reserves`, `security_constrained_dc_opf`, `storage` | LP | Multi-area dispatch with ramping, reserves, storage, and hydro cascades; N-1 DC-OPF |
| `facility_location` | `standard`, `p_median`, `two_echelon` | MIP | Strong capacitated facility location, capacitated p-median, plant→DC→customer |
| `feed_blending` | `standard` | LP | Least-cost feed formulation across a network of mills |
| `forest_planning` | `model_i`, `model_ii` | LP | Model I / Model II harvest scheduling with even flow and green-up |
| `game_theory` | `poker_sequence_form`, `colonel_blotto`, `patrol_security` | LP | Equilibrium LPs of two-player zero-sum games |
| `graph_optimization` | `independent_set`, `generalized_independent_set`, `map_labeling`, `quasi_clique`, `vertex_coloring`, `vertex_cover` | MIP | Packing and covering on application graphs (wireless, wind farms, maps, networks) |
| `hub_location` | `p_hub_median`, `budgeted_backbone`, `capacitated`, `compact_single_allocation`, `hub_covering`, `hub_network`, `multiple_allocation`, `r_allocation` | MIP | Hub location and hub-and-spoke network design |
| `inventory` | `standard`, `lot_sizing`, `multi_echelon`, `multi_item` | LP, MIP | DC replenishment, capacitated lot sizing, multi-echelon distribution, multi-item planning |
| `inverse_optimization` | `standard`, `classical_normalized`, `linf`, `market_clearing`, `noisy_observations`, `restricted_optimal_value`, `shortest_path`, `shortest_path_layered` | LP | Recover costs from exact, noisy, routed, and market observations |
| `job_shop_scheduling` | `standard` | MIP | Time-indexed job shop with parallel machines, release dates, and deadlines |
| `knapsack` | `multiple_choice`, `bounded`, `mixed_integer_set`, `multidimensional` | MIP | Multiple-choice, bounded multiple, HEM-MIK-style, and sparse multidimensional knapsacks |
| `land_use` | `standard` | MIP | Parcel zoning with infrastructure capacities, targets, and buffers |
| `load_balancing` | `standard`, `discrete_placement` | LP, MIP | Path-based traffic engineering minimizing maximum link utilization |
| `maritime_inventory_routing` | `standard` | MIP | Multi-period vessel routing with port inventories |
| `markov_decision_process` | `inventory_control`, `constrained`, `machine_maintenance`, `queueing_control` | LP | Occupation-measure LPs of discounted and average-cost MDPs |
| `mine_planning` | `cpit`, `pcpsp`, `stockpile` | MIP | MineLib-style open-pit production scheduling over a 3D block model |
| `multi_commodity_flow` | `standard`, `binary_capacity` | LP, MIP | Multicommodity min-cost flow and network design on geographic networks |
| `network_flow` | `standard`, `generalized_flow`, `time_expanded` | LP | NETGEN-style min-cost flow, flow with gains, time-expanded evacuation |
| `neural_network_verification` | `relu_big_m` | MIP | ReLU network verification queries with propagated bounds |
| `nurse_scheduling` | `standard` | MIP | Multi-ward nurse rostering with float pools and contract rules |
| `operating_room_scheduling` | `elective_assignment`, `benchmark_loading`, `case_sequencing`, `master_surgical_schedule`, `robust_elective`, `weekly_planning` | MIP | Tactical, robust, and operational operating-room planning and scheduling |
| `portfolio` | `cvar`, `tracking_error` | LP | Scenario CVaR and tracking-error portfolios on a factor market |
| `process_planning` | `refinery`, `campaign`, `capacity_expansion`, `hydrogen_network`, `mode_switching` | LP, MIP | Refinery and chemical-process planning |
| `product_mix` | `standard` | LP | Product mix with alternative routings and ranged market rows |
| `production_planning` | `standard` | LP | Multi-level, multi-period capacitated MRP |
| `project_selection` | `standard` | MIP | Multi-year, multi-division capital portfolio selection |
| `radiotherapy` | `weighted_deviation`, `beam_angle_selection`, `mean_tail_dose`, `minmax_deviation`, `robust_fluence` | LP, MIP | IMRT fluence-map optimization |
| `regression` | `lad`, `basis_pursuit`, `chebyshev`, `l1_svm`, `quantile` | LP | LAD, sparse recovery, minimax fitting, 1-norm SVM, quantile regression |
| `resilient_network_design` | `standard` | MIP | Network build and hardening under edge-failure scenarios |
| `resource_allocation` | `standard` | LP | Multi-period allocation of skilled pools to windowed activities |
| `revenue_management` | `standard`, `stochastic_overbooking` | LP | Choice-based network revenue management and stochastic overbooking |
| `scheduling` | `standard` | MIP | Multi-department staff rostering with rest rules |
| `set_system` | `set_cover`, `combinatorial_auction`, `set_packing`, `set_partitioning` | MIP | Covering, packing, partitioning, and winner determination |
| `stochastic_program` | `standard`, `multistage_alm` | LP | Two-stage recourse and multistage asset–liability scenario trees |
| `supply_chain` | `standard`, `carbon`, `multi_product`, `network_planning`, `single_source` | LP, MIP | Multi-echelon, multi-period supply-chain network design |
| `telecom_network_design` | `standard` | MIP | Link installation and routing under capacity and budget |
| `transportation` | `standard`, `emission_constrained`, `fixed_charge`, `transshipment` | LP, MIP | Shipping on sparse geographic lane networks |
| `tsp` | `standard`, `assignment_relaxation`, `asymmetric`, `flow`, `multiple_salespersons`, `precedence`, `prize_collecting`, `time_windows` | LP, MIP | Travelling-salesman formulations and operational variants |
| `unit_commitment` | `standard` | MIP | Unit commitment with ramping, reserves, and min up/down times |
| `vehicle_routing` | `cvrp` | MIP | Capacitated vehicle routing with single-commodity flow |
| `workforce_shift_scheduling` | `covering` | LP | Multi-site, multi-skill shift-pattern covering |

The live registry is authoritative: `list_categories()`, `list_variants(:cat)`,
`list_problems()`, and `problem_info(...)` (below) reflect the installed version,
and `julia --project=scripts scripts/generate_problem.jl list` prints every variant
with its tags, size cap, and model class.

## Usage

### Basic usage

```julia
using SyntheticLPs
using JuMP
using HiGHS  # or any other LP solver

# Generate a problem with a target variable count (category default variant)
model, problem = generate_problem(:transportation, 100, unknown, 0)

# The problem instance holds all the generated data
problem.n_sources, problem.n_customers

# Solve it
set_optimizer(model, HiGHS.Optimizer)
optimize!(model)
solution_summary(model)

model_statistics(model)  # (num_variables, num_constraints, num_nonzeros, num_integer)
```

`generate_problem(category, target_variables, feasibility_status=unknown, seed=0)`
returns the JuMP model and the generator struct holding every piece of generated
data. `model_statistics` counts affine rows only (variable bounds are not rows).

### Categories and variants

A `ProblemVariant` names one variant of one category and prints as
`category/variant`.

```julia
list_categories()                       # [:airline_crew, :assignment, ...] (unordered)
list_problem_types()                    # alias for list_categories()
list_variants(:portfolio)               # [:cvar, :tracking_error]
list_problems()                         # every ProblemVariant, sorted

problem_info(:transportation)           # category description, variants, default, tags
problem_info(:portfolio, :cvar)         # variant description, tags, size range, model class

# Select a specific variant — four equivalent forms
model, problem = generate_problem(:portfolio, 100, unknown, 0; variant=:cvar)
model, problem = generate_problem(ProblemVariant(:portfolio, :cvar), 100, unknown, 0)
model, problem = generate_problem(ProblemVariant("portfolio/cvar"), 100, unknown, 0)
model, problem = generate_problem("portfolio/cvar", 100, unknown, 0)
```

### Selecting and filtering variants

Every variant is registered with metadata that `list_problems`, `problem_info`,
`generate_random_problem`, and `generate_dataset` all understand:

- **Tags** from a controlled vocabulary (`list_tags()` prints each with its
  meaning). *Structure* tags describe the matrix a solver sees: `:network`,
  `:multicommodity`, `:bipartite`, `:staircase`, `:block_angular`,
  `:dual_block_angular`, `:time_indexed`, `:covering`, `:packing`,
  `:partitioning`, `:big_m`, `:dense`, `:blending`, `:unimodular`, `:degenerate`,
  `:lp_relaxation`, `:robust`. *Domain* tags (the set `DOMAIN_TAGS`) name the
  application: `:logistics`, `:routing`, `:location`, `:scheduling`,
  `:production`, `:energy`, `:finance`, `:healthcare`, `:telecom`,
  `:agriculture`, `:statistics`, `:machine_learning`, `:combinatorial`,
  `:economics`, `:game_theory`, `:markov`, `:mining`, `:forestry`. Every variant
  carries exactly one domain tag. An unknown tag is an error, so a typo never
  silently selects nothing; `register_tag` extends the vocabulary.
- **Model class**: `model_class(ref)` is `:mip` if the variant's `build_model`
  emits integer or binary columns, else `:lp`. It is derived by building one small
  probe instance and then cached, so the first class-filtered query compiles and
  probes every generator it touches (a few minutes for the whole registry).
- **Size range**: `min_target_variables`/`max_target_variables`. Some generators
  document a hard cap and raise `ArgumentError` above it — 1,000,000 for
  `economic_planning`, `forest_planning`, `game_theory`,
  `markov_decision_process`, `mine_planning`, `cutting_stock/arc_flow`,
  `process_planning/campaign`, `supply_chain/network_planning`, and
  `telecom_network_design`; 250,000 for `inverse_optimization`.
  `supports_target(ref, n)` checks a target.

```julia
list_tags()                                          # [:agriculture => "...", ...]
variant_tags("tsp/flow")                             # [:big_m, :network, :routing]
model_class("tsp/assignment_relaxation")             # :lp
supports_target("game_theory/colonel_blotto", 2_000_000)  # false (1M cap)

# Filters compose; every argument is optional.
list_problems(; tags=:network)                       # carry every listed tag
list_problems(; any_tags=[:energy, :mining])         # carry at least one
list_problems(; problem_types=[:tsp, "energy/dc_opf"], exclude="tsp/flow")
list_problems(; exclude_tags=:big_m, target_variables=(1_000, 500_000))
list_problems(; tags=:staircase, model_class=:lp)    # probes model classes
```

`problem_types` and `exclude` accept categories (`:tsp` or `"tsp"`, expanding to
every variant), `"category/variant"` strings, or `ProblemVariant`s, singly or in a
collection. `target_variables` keeps variants that support a target (an integer)
or a whole range (a `(lo, hi)` tuple).

The corpus mixes pure LPs, natural MIPs, and purpose-built LP relaxations.
`generate_problem` defaults to `relax_integer=true`, so MIP variants are returned
as LP relaxations unless you opt out. A relaxation is not a valid integer
solution — `tsp/assignment_relaxation`, for instance, is a fractional degree
relaxation that may contain subtours. Filter with `model_class=:lp` to keep only
models that are continuous by construction.

### Feasibility control

```julia
model, problem = generate_problem(:transportation, 100, feasible, 0)
model, problem = generate_problem(:diet_problem, 100, infeasible, 0)
model, problem = generate_problem(:portfolio, 100, unknown, 0)
```

Generators honor the requested status by construction. Almost every category
plants an auditable artifact on the problem struct: a typed `feasible_witness` (a
complete primal solution) for feasible requests and a typed
`infeasibility_certificate` (a proof checkable without a solver — a Farkas
combination, a cut or Hall-type deficit, a capacity prefix, a Lagrangian bound) for
infeasible ones; the category pages under [`docs/`](docs/README.md) name the
fields. `unknown` requests sample a realistic instance with no planted outcome.

Infeasibility is planted so it survives presolve: certificates combine many rows
(across periods, regions, or a network cut) rather than contradicting a single row
or bound, so a presolver cannot dismiss the instance without simplex work. In MIP
categories the certificate uses LP rows only, so infeasibility survives the
default relaxation.

#### Solver verification

A few generators' feasibility logic is heuristic and occasionally misses. Pass an
`optimizer` to **verify and guarantee** the contract: the model is solved on a copy
(the returned model stays pristine) and rebuilt with the next seed on a mismatch,
up to `max_feasibility_retries=10` times:

```julia
using HiGHS
model, problem = generate_problem(:energy, 300, infeasible, 1; optimizer=HiGHS.Optimizer)
```

With `optimizer` unset (the default) no solving is performed. Verification is
deterministic — retries walk `seed, seed+1, …` — so a given `(seed, optimizer)`
pair always resolves to the same model, and `generate_dataset` records the
resolved seed so verified datasets can be rebuilt without re-solving.

Each verification solve is bounded by `feasibility_timeout` (default 10 s). A
solve that certifies nothing (`TIME_LIMIT`, `ALMOST_OPTIMAL`,
`INFEASIBLE_OR_UNBOUNDED`, `OTHER_ERROR`, …) raises instead of being counted as a
violation, so a slow solve is never misreported as a bad instance. Unrelaxed MIPs
and very large LPs are the usual cause; give them more time:

```julia
model, problem = generate_problem(:job_shop_scheduling, 2000, feasible, 2;
                                  relax_integer=false, optimizer=HiGHS.Optimizer,
                                  feasibility_timeout=120.0)
```

`optimizer` may also be a **vector of optimizers** — an escalation chain in which
each later entry is tried only when the previous one was inconclusive (a
`:violated` verdict is final). This matters for large infeasible LPs: HiGHS's
default dual simplex sometimes ends with `OTHER_ERROR` on infeasible
`markov_decision_process`, `forest_planning`, `process_planning` (refinery family),
and `blending` instances that its interior-point solver proves `INFEASIBLE` in
seconds. The recommended HiGHS chain is:

```julia
chain = [HiGHS.Optimizer, optimizer_with_attributes(HiGHS.Optimizer, "solver" => "ipm")]
model, problem = generate_problem(:markov_decision_process, 20_000, infeasible, 3;
                                  optimizer=chain, feasibility_timeout=60.0)
```

(Here dual simplex returns `OTHER_ERROR` and the IPM proves infeasibility in a
couple of seconds.) Each optimizer in the chain gets the full
`feasibility_timeout`, so a chain also rescues a dual simplex that runs out of
time.

### Model transforms

`generate_problem`, `generate_random_problem`, and `generate_dataset` apply an
optional pipeline of post-`build_model` reformulations, in this order:

1. `relax_integer=true` (default) relaxes integrality.
2. `bounds_to_constraints=false` — when `true`, bounds become explicit rows.
3. Feasibility verification, when an `optimizer` is given, solves this primal.
4. `transforms=ModelTransforms(...)` applies practitioner-style reformulations,
   seeded by the instance seed (the identity by default).
5. `dualize=false` (or a `dualize_probability`) returns the dual of the result.

#### Practitioner-style transforms (`ModelTransforms`)

Generators emit clean, index-ordered models in natural units. Real models are
messier, and solvers — and learned pivot rules — can overfit that cleanliness.
`ModelTransforms` adds reproducible formulation diversity:

```julia
t = ModelTransforms(;
    unit_scale_decades=2,          # scale_units!: per-family units in 10^-2 … 10^2
    scale_objective=true,
    aggregate_probability=0.5,     # aggregate_rows!: redundant block-total rows
    aggregate_max_block=8,
    elastic_probability=0.0,       # elasticize_rows!: penalized violation columns
    elastic_penalty=1e3,
    permute=true,                  # permute_model: shuffle row and column order
)
model, problem = generate_problem(:transportation, 1000, feasible, 3; transforms=t)

# A NamedTuple of the same keywords also works
model, problem = generate_problem(:energy, 1000, unknown, 3;
                                  transforms=(; unit_scale_decades=2, permute=true))
```

Within `ModelTransforms` the steps always run aggregate → elastic → permute →
scale, each from its own random stream derived from the instance seed, so the same
seed and configuration always give the same model. Each step is exported for
direct use on any JuMP model, and `apply_transforms(model, t, seed)` runs a whole
configuration:

| Transform | What it does | Semantics | Measured effect |
|---|---|---|---|
| `scale_units!(model, rng; decades)` | Rescales each variable family, row family, and the objective by a random power of ten, the way a modeler picks tonnes or kg per family; integer columns keep their units; coefficients stay within `[1e-6, 1e6]` | Exact equivalence; the returned `UnitScaling` (also in `model.ext`) maps solutions and duals back | Presolved size unchanged; matrix range widened by a median 2.5 decades; iterations changed by a mean \|log2\| ratio of 0.24 (dual) / 0.33 (primal) |
| `aggregate_rows!(model, rng; probability, max_block)` | Adds an exact-sum "total" row for consecutive blocks of 2–`max_block` rows in randomly selected row families | Equivalence (implied rows); adds primal degeneracy | Inequality totals survive presolve (up to ×1.49 rows), equality totals mostly do not; iterations 0.26 / 0.15 |
| `elasticize_rows!(model, rng; probability, penalty)` | Softens randomly selected row families with violation columns charged at `penalty × max\|c\|` | A *relaxation*: feasible stays feasible, but infeasible can become feasible, so it is refused for `infeasible` requests | Presolved columns grew by a median 56%; objective unchanged on every sampled variant; iterations 0.28 / 0.35 |
| `permute_model(model, rng)` | Returns a copy with rows and columns in random order | Exact equivalence | Iterations 0.13 / 0.16 — the noise floor for the others |

Measurements are HiGHS (presolve on) on 15 variants at 20k variables. All four
combined changed iteration counts by a mean |log2| ratio of 0.46 (dual) / 0.47
(primal), with optimal objectives matching to 1e-10. Added rows and columns count
toward recorded sizes and dataset size matching, and `generate_dataset` lists each
non-identity step in `GeneratedInstance.transforms`.

#### Bound reformulation

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
genuine rows, counted by `num_constraints(model; count_variable_in_set_constraints=false)`,
so they affect dataset size metadata and quality thresholds.

**Presolve undoes it.** Every converted bound is a singleton row, and LP
presolvers turn singleton rows straight back into bounds: measured with HiGHS on
15 variants at ~10k variables, presolved models had the same size with or without
it (within 0.3%) and iteration counts barely moved. Use it to exercise a
reader/modeling pipeline or a presolve-free solver path, not to make instances
harder — `ModelTransforms` is the tool for that.

#### Dual reformulation

Dualization is off by default. Its main use is reproducible formulation diversity
in random generation: `dualize_probability` independently chooses whether each
returned model is primal or dual.

```julia
# Randomly return a primal or dual formulation with equal probability
model, ref, problem = generate_random_problem(100; seed=7, dualize_probability=0.5)
is_dual_reformulation(model)  # reports the sampled choice

dataset = generate_dataset(num_problems=10, seed=7, dualize_probability=0.5)

# Force the dual for a specific generated LP
dual_model, problem = generate_problem(:transportation, 100, unknown, 0; dualize=true)

# Or dualize an existing continuous JuMP model (the primal is unchanged)
dual_model = dualize_model(model)
```

Dual variables and constraints use `dual_var_`/`dual_con_` name prefixes.
Unrelaxed integer or binary variables are rejected, since mixed-integer models
have no LP/conic dual. Feasibility verification checks the primal *before*
dualization (an infeasible primal may have either an infeasible or an unbounded
dual), and a dualized instance is the dual of the transformed primal. Dataset size
and quality metadata are computed from the models actually returned, and each
`GeneratedInstance.dualized` value and manifest entry records the sampled choice.

### Reproducibility

```julia
model1, problem1 = generate_problem(:knapsack, 50, unknown, 12345)
model2, problem2 = generate_problem(:knapsack, 50, unknown, 12345)  # identical
```

Every generator draws from a constructor-local `MersenneTwister(seed)`, so
generation neither reads nor advances the caller's global RNG. The seed alone
determines an instance: the caller's global RNG state, other generation on the same
task, and concurrent generation on other threads cannot perturb it.

### Random problem generation

```julia
# Random variant targeting ~100 variables; `ref` is a ProblemVariant
model, ref, problem = generate_random_problem(100; seed=1)
println("Problem: $ref")           # e.g. "transportation/standard"

model, ref, problem = generate_random_problem(200; feasibility_status=feasible, seed=2,
                                              problem_types=[:energy, :network_flow],
                                              variant_weighting=:variant)
```

The variant is drawn from every variant that supports the target (or from
`problem_types`, which accepts any `list_problems` selector) with
`variant_weighting=:category` by default — uniform over categories, then over each
category's variants, so an eight-variant category is not eight times as likely as
a one-variant one. `:variant` weights variants uniformly, and a `Dict` gives
explicit weights (see [below](#variant-selection-and-weighting)).

### Batch dataset generation

`generate_dataset` builds a whole dataset in two stages. **Planning** is cheap and
builds nothing: from the master `seed`, every index `1:num_problems` is assigned a
variant, a requested feasibility status, a target size, and a private RNG stream.
**Execution** then builds each index independently from its own stream. An index
depends only on `(seed, index)`, so a dataset is reproducible from a nonzero seed
and can be split into shards that reproduce exactly the instances of the unsharded
run. `plan_dataset(; kwargs...)` returns the plan (`Vector{PlannedInstance}`)
without building anything — use it to inspect the mix of a large run first.

```julia
using SyntheticLPs

dataset = generate_dataset(;
    num_problems = 20,
    size_distribution = :loguniform,   # LogUniform(var_min, var_max)
    var_min = 200,
    var_max = 5_000,
    exclude_tags = :big_m,
    feasibility_status = Dict(feasible => 0.5, infeasible => 0.25, unknown => 0.25),
    transforms = ModelTransforms(; unit_scale_decades=2, permute=true),
    output_dir = "dataset",            # instance files + manifest.json
    seed = 1234,                       # 0 = random (still recorded in the manifest)
    on_failure = :skip,
)

for inst in dataset[1:3]
    println("$(ProblemVariant(inst)) [$(inst.feasibility_status)]: " *
            "$(inst.num_variables) vars (target $(inst.target_variables)), " *
            "$(inst.num_constraints) rows, $(inst.num_nonzeros) nnz → $(inst.filename)")
end
dataset.failures   # Vector{DatasetFailure}: indices that on_failure=:skip dropped
dataset.manifest   # the Dict written to manifest.json
```

`generate_dataset` returns a `GeneratedDataset`, which indexes and iterates like a
`Vector{GeneratedInstance}` (sorted by index) and also carries `failures` and the
`manifest`. Each `GeneratedInstance` records the category and variant, the
requested status, the planned `target_variables` and the `requested_variables`
actually passed to the generator, the returned model's `num_variables`,
`num_constraints`, `num_nonzeros`, and `num_integer` (before relaxation), the
resolved `seed`, `dualized`, the applied `transforms`, `verified_status` and
`solve_status` when a solve ran, quality-filter `iterations` and `solve_time`,
`build_time`, `generation_time`, `attempts`, and `filename`. Calling
`generate_problem(ProblemVariant(inst), inst.requested_variables,
inst.feasibility_status, inst.seed; dualize=inst.dualized, ...)` with the run's
transform settings rebuilds an instance exactly.

#### Variant selection and weighting

`problem_types`, `exclude`, `model_class`, `tags`, `any_tags`, and `exclude_tags`
select variants exactly as in `list_problems`. Variants whose size cap cannot
cover the size distribution's support are dropped and listed in the manifest.
`variant_weighting` sets the mix:

- `:category` (default) — uniform over categories, then over each category's
  variants;
- `:variant` — uniform over variants;
- a `Dict` of nonnegative weights keyed by category (split among that category's
  variants) and/or `"category/variant"`, e.g.
  `Dict(:tsp => 2.0, "knapsack/bounded" => 0.5)`. Unlisted variants get weight 0.

The realized mix is stratified: each variant appears the floor or ceiling of its
expected count, even in small datasets.

#### Sizes and size calibration

- `size_distribution`: `:normal` (the default, a normal truncated to
  `[var_min, var_max]` from `var_mean`, `var_std`, `var_min`, `var_max`, which
  default to 500, 200, 50, 2000), `:uniform`, `:loguniform` (recommended for
  ranges spanning orders of magnitude, e.g. 1k–100k), or any
  `Distributions.UnivariateDistribution`.
- With `match_size_distribution=true` (the default), targets are stratified
  quantiles of the distribution, and each build is **calibrated**: the generator is
  re-run on the same seed with the request rescaled by target/actual until
  `|log(actual/target)| ≤ size_match_tolerance` (default 0.05) or
  `size_match_attempts` (default 3) recalibrations or the `max_retries` build budget
  are spent, keeping the closest build subject to the quality filter.
  `strict_size_match=true` turns a build still out of tolerance into a
  failed attempt. `match_size_by_category=true` stratifies targets within each
  category, so every category spans the whole distribution. The manifest's
  `size_match` block reports the achieved mean and maximum log error and the
  fraction within tolerance.
- `match_size_distribution=false` draws iid targets and builds once.

Calibration replaces the earlier candidate-pool approach (`candidate_multiplier`,
`match_size_by_type`), which built surplus instances and discarded the misfits.

#### Feasibility mixes and verification

`feasibility_status` is one status for every instance or a mix such as
`Dict(feasible => 0.7, infeasible => 0.3)` (keys may also be symbols or strings).
The mix is spread with a low-discrepancy sequence, so both the overall and each
variant's realized proportions track the weights. `feasible_only=true` is
shorthand for `feasibility_status=feasible`.

With an `optimizer`, `feasible`/`infeasible` requests are
verified as in `generate_problem` (with `feasibility_timeout` taken from
`quality_criteria.solve_timeout`) and the outcome is recorded as `verified_status`.

#### Quality filtering

The package itself is solver-agnostic. To **filter** instances — solving each one
and rejecting trivial, degenerate, unbounded, timed-out, or ill-conditioned ones —
pass `quality_filter=true` and an `optimizer`. The quality solve then doubles as
verification for `feasible` requests. `infeasible` requests are still verified on
the source primal, because no passing quality solve proves infeasibility
(`INFEASIBLE_OR_UNBOUNDED` passes, and an infeasible dual leaves the primal
infeasible or unbounded); one that the quality solve shows feasible is rejected
as `"contract_violated"`.

```julia
using SyntheticLPs, HiGHS

dataset = generate_dataset(;
    num_problems = 20,
    feasibility_status = Dict(feasible => 0.5, infeasible => 0.5),
    quality_filter = true,
    optimizer = HiGHS.Optimizer,
    optimizer_attributes = ("solver" => "simplex",),
    quality_criteria = QualityCriteria(;
        solve_timeout = 30.0,
        min_constraints = 5,
        min_iterations = 3,
        max_iteration_ratio = 100.0,
    ),
    max_retries = 10,       # builds per index, calibration builds included
    on_failure = :skip,
    seed = 42,
)
```

A single instance can also be evaluated directly with
`check_quality(model, HiGHS.Optimizer)`, which returns a `QualityResult`.

#### Sharding, failures, and the manifest

- **Sharding**: `num_shards=N, shard_index=k` builds only the indices `i` with
  `(i - 1) % N == k - 1`, and requires a fixed nonzero `seed` shared by every
  shard. Each shard writes `manifest_shard_KKKK_of_NNNN.json`, and filenames embed
  the global index, so shards can share an `output_dir`. `merge_manifests(output_dir)`
  checks that every shard is present with the same configuration and writes the
  combined `manifest.json`.
- **Failures**: an index that exhausts `max_retries` builds (generator errors,
  quality rejections, strict size mismatches) raises by default
  (`on_failure=:error`). With `on_failure=:skip` it is recorded as a
  `DatasetFailure` (index, variant, status, target, attempts, final `reason`, and
  one message per attempt) and generation continues, returning a short dataset.
- **Manifest** (`format_version` 2): `provenance` (package and Julia versions, git
  commit and dirty flag), the full `config` (including `master_seed`, the transform
  settings, and the quality criteria), the `selection` (variants, weights, and
  variants dropped for their size range), `shard`, aggregate `stats` (attempts,
  failure reasons, generation time), `size_match`, and one entry per instance and
  per failure.

```julia
for k in 1:4   # e.g. four processes or cluster jobs
    generate_dataset(; num_problems=40, seed=7, num_shards=4, shard_index=k,
                     output_dir="sharded", on_failure=:skip)
end
manifest = merge_manifests("sharded")   # writes sharded/manifest.json
```

## Command Line Interface

The scripts activate their own `scripts` environment (which provides HiGHS and
ArgParse) and develop the local package into it, so they run straight from a clone.

```bash
# Generate a problem (category default variant, or an explicit category/variant)
julia --project=scripts scripts/generate_problem.jl transportation 100 problem.mps
julia --project=scripts scripts/generate_problem.jl portfolio/cvar 100 problem.mps

# List every category and variant with its tags, size cap, and model class
julia --project=scripts scripts/generate_problem.jl list

# Feasibility control, solving, bound reformulation, dualization, seeds, random selection
julia --project=scripts scripts/generate_problem.jl knapsack 50 --feasible --solve
julia --project=scripts scripts/generate_problem.jl diet_problem 100 --infeasible output.mps
julia --project=scripts scripts/generate_problem.jl knapsack 50 --bounds-to-constraints
julia --project=scripts scripts/generate_problem.jl transportation 100 --dualize
julia --project=scripts scripts/generate_problem.jl portfolio 150 --seed=12345
julia --project=scripts scripts/generate_problem.jl random 200 --dualize-probability=0.5
```

For whole datasets, `scripts/generate_lps.jl` is a thin wrapper around
`generate_dataset` that supplies HiGHS (`--help` lists every flag):

```bash
# 100 .mps instances into ./output
julia --project=scripts scripts/generate_lps.jl -o output -n 100

# 50 feasible, quality-filtered instances with progress output
julia --project=scripts scripts/generate_lps.jl -o output -n 50 --feasible-only -q -v

# Restrict to specific categories or variants with a fixed seed
julia --project=scripts scripts/generate_lps.jl --problem-types transportation,portfolio/cvar -n 20 --seed 42

# Inspect the plan of a large LP-only run without building anything
julia --project=scripts scripts/generate_lps.jl -o big -n 1000 --seed 7 \
  --size-distribution loguniform --var-min 1000 --var-max 100000 \
  --model-class lp --exclude-tags big_m --feasibility feasible=0.7,infeasible=0.3 --dry-run

# Practitioner-style transforms and an explicit variant mix
julia --project=scripts scripts/generate_lps.jl -o output -n 200 --seed 3 \
  --scale-units 2 --aggregate-rows 0.5 --permute \
  --variant-weighting 'tsp=2,energy=1,knapsack/bounded=0.5'

# Four shards of one dataset in parallel, then one merged manifest
for k in 1 2 3 4; do
  julia --project=scripts scripts/generate_lps.jl -o big -n 1000 --seed 7 \
    --num-shards 4 --shard-index $k --on-failure skip &
done; wait
julia --project=scripts scripts/generate_lps.jl -o big --merge-manifests
```

Flags by group — selection: `--problem-types`, `--exclude`, `--model-class`,
`--tags`, `--any-tags`, `--exclude-tags`, `--variant-weighting`; sizes:
`--size-distribution` (`normal`, `uniform`, `loguniform`), `--var-mean`,
`--var-std`, `--var-min`, `--var-max`, `--no-size-matching`,
`--match-size-by-category`, `--size-match-tolerance`, `--size-match-attempts`,
`--strict-size-match`; feasibility: `--feasibility`, `--feasible-only`;
transforms: `--bounds-to-constraints`, `--scale-units`, `--aggregate-rows`,
`--elastic-rows`, `--elastic-penalty`, `--permute`, `--dualize`,
`--dualize-probability`; quality: `-q`/`--quality-filter`, `--solve-timeout`,
`--min-iterations`, `--max-iteration-ratio`, `--min-constraints`, `--max-retries`;
runs: `--shard-index`, `--num-shards`, `--merge-manifests`, `--on-failure`,
`--dry-run`, `--file-format`, `--no-manifest`, `--seed`, `-v`/`--verbose`.

The script always passes HiGHS as the optimizer, so even without `-q` every
`feasible`/`infeasible` request is verified by one HiGHS solve bounded by
`--solve-timeout`. For very large infeasible instances, raise `--solve-timeout`
or pass `--on-failure skip`.

## Auditing generators

`scripts/audit_generators.jl` answers "what does a solver actually see?" at scale.
For each (variant, target, status, seed) it appends one JSON line with the model
size (`size_ratio` = columns/target, nonzeros, integer columns), `build_time`, the
HiGHS presolved size (`presolve_col_ratio`, `presolve_row_ratio`, presolve
status), and the status, iterations, and time of a presolve-on simplex solve.
Records are flushed one at a time, so an interrupted run keeps every completed
record. Unlike the other scripts it does not set up the `scripts` environment
itself: run `generate_problem.jl` or `generate_lps.jl` once first (or
`julia --project=scripts -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'`).

```bash
# Measure (defaults: targets 1000,10000,50000; all three statuses; seed 0)
julia --project=scripts scripts/audit_generators.jl -o audit.jsonl \
  --variants tsp/standard,energy --targets 1000,10000,100000 --statuses feasible,infeasible

# Registry filters work here too
julia --project=scripts scripts/audit_generators.jl -o audit.jsonl \
  --model-class lp --exclude knapsack --tags network --solve-time-limit 60

# Report: one markdown row per variant, with red flags
julia --project=scripts scripts/audit_generators.jl --report audit.jsonl \
  --report-out AUDIT.md --flagged-only
```

The report flags size ratios outside `1 ± --size-tol` (0.1), build times above
`--max-build-time` (30 s) at the largest target, presolve ratios below
`--min-presolve-ratio` (0.6) for feasible/unknown instances, instances presolve
solves outright, feasibility-contract violations, errors, solve timeouts, and
models skipped for exceeding `--max-nnz` (20M nonzeros). Errors include unsuccessful
solver statuses such as `UNKNOWN`, solver errors, and non-time termination limits,
even without a thrown exception; `TIME_LIMIT` is counted separately as a timeout.
Report mode loads neither the package nor HiGHS.

## Scale and quality

The generators are held to a common quality bar, checked by the test suite and by
audits with `scripts/audit_generators.jl`:

- **Size fidelity from tiny to 100k+.** A request for `n` variables yields close
  to `n` at every scale. From 1k to 100k almost every variant lands within a few
  percent, most within 1%; the widest deviations are about ±13%
  (`supply_chain/single_source`, `unit_commitment/standard`, some `hub_location`
  and `operating_room_scheduling` variants). Tiny requests (≈20 variables) always
  build, some at a structural floor of a few dozen variables. A generator whose
  data cannot scale further documents a cap (`max_target_variables`, usually
  1,000,000) and raises above it rather than silently undersizing.
- **Near-linear builds.** No generator uses quadratic sampling or dense
  intermediate structures: 10k-variable instances build in well under a second,
  and in a 111-variant audit every 100k-variable instance built in at most ~8 s.
  Nonzeros stay bounded — a few million at 100k variables even for the densest
  radiotherapy and regression variants.
- **Presolve survival.** Instances should be hard for the solver, not for the
  presolver. Generators emit variable bounds rather than singleton rows and avoid
  trivially fixed or redundant structure. In the 111-variant audit, HiGHS presolve
  kept at least 60% of the rows and columns of every feasible/unknown instance
  outside three `hub_location` variants, and typically 75–100%.
- **Planted, auditable outcomes.** Feasible requests carry a typed primal witness
  and infeasible requests a typed certificate, both checked by the per-category
  tests. Infeasibility is spread over many rows, so presolve rarely detects it: the
  audit found presolve-detected infeasibility only in a handful of
  `process_planning`, `supply_chain`, `telecom_network_design`,
  `resilient_network_design`, and `resource_allocation` instances (the telecom and
  resilient-network pages document a larger detectable share in wider sweeps).
- **No contract violations.** The audit (1k/10k/100k targets, HiGHS presolve plus a
  60 s solve) found no feasibility-contract violations and no build errors. Many
  100k instances exceed a 60 s HiGHS solve, by design.

Category pages under [`docs/`](docs/README.md) record each generator's measured
sizes, presolve ratios, and known solver behavior.

## Extending with New Categories and Variants

Each category lives in its own folder under `src/problem_types/<category>/`:

```
src/problem_types/transportation/
    transportation.jl   # category entry point: includes the variant file(s)
    common.jl           # optional data/helpers shared by the category's variants
    standard.jl         # a variant: struct + constructor + build_model + register_variant
```

**To add a variant to an existing category**, create a file in that category's
folder and `include` it from the category's `<category>.jl` entry point.

**To add a new category**, create `src/problem_types/<category>/<category>.jl`
(the entry point), add at least one variant file, and add a single
`include("problem_types/<category>/<category>.jl")` line to `src/SyntheticLPs.jl`.
A category is created automatically by its first variant's `register_variant`
call; call `register_category(:cat, "…")` explicitly in the entry point only when
you want a category-level description distinct from its variants. Add a
`docs/<category>.md` page and a `test/problem_types/<category>.jl` test file.

Variant file template:

```julia
using JuMP
using Random

struct YourWitness              # a complete primal solution, for `feasible`
    x::Vector{Float64}
end

struct YourCertificate          # a solver-free infeasibility proof, for `infeasible`
    multipliers::Vector{Float64}
end

struct YourProblem <: ProblemGenerator
    # store all generated data needed to build the model
    field1::Type1
    field2::Type2
    feasible_witness::Union{Nothing, YourWitness}
    infeasibility_certificate::Union{Nothing, YourCertificate}
end

function YourProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    rng = MersenneTwister(seed)

    # Sample all parameters from target_variables, drawing from `rng` and never
    # from the global stream: rand(rng, ...), randn(rng), shuffle(rng, ...).
    # Generate all data and handle the feasibility status here.

    return YourProblem(field1_value, field2_value, witness, certificate)
end

# Must be deterministic: no RNG calls.
function build_model(prob::YourProblem)
    model = Model()
    # variables (single-variable limits as bounds, not rows), constraints, objective
    return model
end

# Registers the variant, lazily creating the category
register_variant(
    :your_category,
    :your_variant,
    YourProblem,
    "Description of this variant";
    tags=[:network, :staircase, :logistics],  # structure tags + exactly one domain tag
    max_target_variables=1_000_000,           # only if the generator documents a cap
)
```

Key principles:
- The struct stores ALL data needed to deterministically build the model
- ALL randomness goes in the constructor, drawn from a constructor-local
  `MersenneTwister(seed)` threaded explicitly through any helper it calls
  (`helper(rng::AbstractRNG, ...)`)
- `build_model` must be completely deterministic
- Handle `feasible`, `infeasible`, and `unknown`; plant a witness or certificate
  where the outcome is constructed, and make infeasibility depend on many rows,
  never on one contradictory row or bound
- Hit the target size from tiny to 100k+ variables with near-linear build time and
  bounded nonzeros; check with `scripts/audit_generators.jl`

## Testing

```bash
make test                                               # full suite at -O1
make test CATEGORIES=tsp,knapsack                       # focused on some categories
julia --project=@. -O1 test/runtests.jl transportation  # direct run (skips HiGHS testsets)
```

The suite checks every registered variant: target variable counts land within
±25% of the request (relaxed for very small problems), all three feasibility
statuses work, models are structurally valid and reproducible, no generator
touches the global RNG, and every variant carries exactly one domain tag. Each
category's `test/problem_types/<category>.jl` adds focused contracts: exact size
formulas, data invariants, witness and certificate arithmetic, edge sizes, and
solver-backed feasibility checks.

## License

SyntheticLPs.jl  
Copyright (C) 2025  Felix Parker

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details.

The full text of the license is available in the [LICENSE](LICENSE) file.
