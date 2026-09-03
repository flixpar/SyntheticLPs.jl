# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context

A standardized framework for generating synthetic linear programming (LP) problem
instances. The goal is problems realistic enough to test and develop LP solvers.

## General Instructions

- Explore the relevant code carefully before making any plans or changes.
- Update `CHANGELOG.md` after any significant change: one section per date, each
  recording the commit hash and datetime, a high-level summary, and details more
  granular than the commit messages.
- The project is under active development and not yet stable, so never worry
  about breaking changes or backwards compatibility.
- Update `README.md` and `CLAUDE.md` when making major changes.
- Research code: it does not need to be extremely robust or handle every edge case.

## Commands

### Formatting and quality checks

```bash
python3 -m pip install -r requirements-dev.txt
make setup   # instantiate the dedicated Julia tooling environment
make format  # apply JuliaFormatter and Ruff
make lint    # verify formatting and run Aqua/Ruff checks
make check   # lint and run the complete Julia test suite
```

### Testing

HiGHS is a test-only dependency in `[extras]`. Both commands work:

```bash
# Full suite, including the solver-based feasibility-contract testsets:
julia --project=@. -e 'using Pkg; Pkg.test()'

# Direct run: skips the solver-based testsets (HiGHS is not resolvable outside
# the Pkg.test sandbox):
julia --project=@. test/runtests.jl
```

`test/runtests.jl` loads HiGHS lazily behind a `HAS_HIGHS` flag, so the direct
run skips those testsets with an `@info` notice instead of erroring.

### Problem generation

```bash
julia --project=@. scripts/generate_problem.jl list
julia --project=@. scripts/generate_problem.jl transportation 100 output.mps
julia --project=@. scripts/generate_problem.jl knapsack 50 --feasible --solve
```

### Dataset generation

`generate_dataset` builds a whole dataset; `scripts/generate_lps.jl` is a thin CLI
wrapper that supplies HiGHS, so use the `scripts` environment:

```bash
julia --project=scripts scripts/generate_lps.jl -o output -n 100
julia --project=scripts scripts/generate_lps.jl -o output -n 50 --feasible-only -q -v
```

## Architecture

Problems are a two-level hierarchy: a **category** is a problem domain (e.g.
`:transportation`) grouping one or more **variants**, each a concrete generator
with its own data generation and formulation (e.g. `:standard`). There are 45
categories. Query the live registry — `list_categories()`, `list_variants(:cat)`,
`list_problems()`, `problem_info(...)` — rather than a hardcoded list; `README.md`
holds the catalog and `docs/<category>.md` the per-category notes.

**Main module** (`src/SyntheticLPs.jl`):
- `ProblemGenerator` (abstract base type for generators), `FeasibilityStatus`
  (`feasible`, `infeasible`, `unknown`), and `ProblemVariant` — the canonical
  reference to one `category/variant` pair, constructible from two symbols, a
  bare category symbol (→ default variant), or a `"category/variant"` string
- Two-level registry `LP_REGISTRY::Dict{Symbol,CategorySpec}`, populated by
  `register_category()` and `register_variant()` (a variant lazily creates its
  category)
- `generate_problem()` (accepts a category symbol with optional `variant=`, a
  `ProblemVariant`, or a generator type), `generate_random_problem()` (also
  returns the selected `ProblemVariant`), and `build_model(problem)`, which every
  variant implements

**Feasibility-contract verification**: every `generate_problem` /
`generate_random_problem` overload accepts an optional `optimizer` (plus
`max_feasibility_retries=10`, `feasibility_timeout=10.0`). When supplied for a
`feasible`/`infeasible` request, the built model is solved on a copy and the pure
`_classify_termination(ts, status)` returns one of three verdicts:
- `:holds` — proved; return the model.
- `:violated` — disproved (`INFEASIBLE` for a `feasible` request;
  `OPTIMAL`/`DUAL_INFEASIBLE` for an `infeasible` one). Rebuild with the next seed.
- `:inconclusive` — certifies nothing (`TIME_LIMIT`, `ALMOST_OPTIMAL`,
  `INFEASIBLE_OR_UNBOUNDED`, or anything else). Raises immediately rather than
  spending the retry budget re-asking an unanswerable question. Unrelaxed MIPs are
  the common trigger — raise `feasibility_timeout`.

This is the project-level backstop for the few generators whose heuristic
feasibility logic occasionally misses (~0.1% of requests corpus-wide). It lives in
`generate_problem`, not per-variant; with the default `optimizer=nothing`,
generation is unchanged. Retries walk `seed, seed+1, …`, so a given
`(seed, optimizer)` pair always resolves to the same model. `generate_dataset`
records the resolved seed per instance and skips verification when
`quality_filter` is on (`check_quality` already solves every candidate).

**Model transforms** (`src/transforms.jl`) — post-`build_model` reformulations of
the finished JuMP model, applied centrally in `generate_problem()` in this order:
1. `relax_integer=true` (the default) relaxes integrality.
2. `bounds_to_constraints=true` reformulates variable bounds as explicit affine
   rows, keeping a plain `x ≥ 0` bound but converting upper, fixed, and nonzero
   lower bounds — including those introduced by relaxation. Converted bounds are
   genuine rows, so they raise
   `num_constraints(...; count_variable_in_set_constraints=false)` and affect
   dataset size-matching and quality thresholds.
3. `dualize=true` (or a per-instance `dualize_probability`) returns a separately
   named dual model (`dual_var_`/`dual_con_` prefixes), leaving the primal
   unchanged; it rejects unrelaxed discrete variables and splits ranged rows on an
   internal copy. Feasibility verification applies to the source primal; size and
   quality metadata to the returned model.

**Dataset generation** (`src/dataset.jl`): `generate_dataset(; kwargs...)` samples
problem types and target variable counts, optionally writes instance files plus a
`manifest.json`, and returns `Vector{GeneratedInstance}` metadata; fully
reproducible from a non-zero `seed`. `check_quality(model, optimizer; ...)` with
`QualityCriteria`/`QualityResult` filters trivial, degenerate, unbounded, and
ill-conditioned instances. The package stays solver-agnostic: the caller supplies
the optimizer.

**Problem generators** (`src/problem_types/<category>/`): a `<category>.jl` entry
point that `include`s one file per variant (or per closely related group), plus an
optional `register_category` call for a category-level description.

### Generator pattern

```julia
# src/problem_types/<category>/<category>.jl  (entry point)
# Optionally: register_category(:category, "Category-level description")
include("standard.jl")
```

```julia
# src/problem_types/<category>/standard.jl  (a variant)
struct VariantStruct <: ProblemGenerator
    # every field build_model needs
end

function VariantStruct(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    rng = MersenneTwister(seed)
    # Sample dimensions from target_variables, generate all data, handle the
    # three feasibility statuses. All randomness lives here.
    return VariantStruct(...)
end

function build_model(prob::VariantStruct)
    model = Model()
    # Build from prob's fields only — no RNG calls.
    return model
end

# Registers the variant; lazily creates :category. Pass default=true to make it
# the category default.
register_variant(:category, :standard, VariantStruct, "Description")
```

### Key design principles

1. **Separation of concerns**: the struct stores ALL data needed to build the
   model, the constructor holds ALL randomness, and `build_model` is completely
   deterministic.
2. **Reproducibility**: same seed → identical instance → identical model,
   independent of the caller's global RNG and of concurrent generation.
   Randomness comes from a constructor-local `rng = MersenneTwister(seed)`, never
   from `Random.seed!` and the global stream. Every draw passes it explicitly
   (`rand(rng, …)`, `randn(rng)`, `shuffle(rng, …)`, `sample(rng, …)`) and every
   helper a constructor calls takes it first: `helper(rng::AbstractRNG, …)`. The
   `Global RNG Isolation` testset enforces this across every registered variant.
3. **Feasibility control**: handle all three statuses. Where feasibility is
   planted rather than hoped for, store a typed `feasible_witness` for `feasible`
   requests, a typed `infeasibility_certificate` for `infeasible` ones, and
   neither for `unknown` (the pattern used by `hub_location`, `product_mix`,
   `airline_crew`, `nurse_scheduling`, `neural_network_verification`,
   `maritime_inventory_routing`, `supply_chain/network_planning`,
   `telecom_network_design`, and others). In a MIP category, build the certificate
   from LP rows alone so the infeasibility survives the default
   `relax_integer=true`.
4. **Sizing limits**: generators whose sparse data is represented in several
   structures at once (`supply_chain/network_planning`,
   `telecom_network_design/standard`) cap `target_variables` at a documented
   1,000,000 and raise `ArgumentError` above it rather than silently undersizing.

### Model classes

The corpus deliberately mixes pure LPs, natural MIPs (binary/integer variables),
and purpose-built LP relaxations. The public API defaults to `relax_integer=true`,
so MIP variants are returned as relaxations unless the caller opts out. When
building an LP-only corpus, filter or relax the MIP variants; when characterizing
instances, do not present a relaxation as a real-world integer solution
(`tsp/assignment_relaxation` in particular is a fractional degree relaxation that
may contain subtours, not a tour).

### Testing strategy

`test/runtests.jl` holds only framework-level coverage; everything specific to one
category lives in `test/problem_types/<category>.jl`, so a generator's source,
documentation, and regression coverage evolve as one reviewable unit.

- `test/runtests.jl`: `test_problem_generator(ref)` applied to every registered
  variant (target variable counts, all three statuses, model structure,
  reproducibility), plus registry and interface tests, global-RNG isolation,
  dataset generation, the bounds-to-constraints transform, the pure
  `_classify_termination` table, and the generic feasibility-contract machinery
  (retry budget, seed walk, pristine-model guarantee). It ends with an include
  loop over every `test/problem_types/*.jl` in sorted order.
- `test/problem_types/<category>.jl`: focused contracts for one category —
  registry shape, exact variable-count formulas, data invariants,
  witness/certificate arithmetic, edge-size robustness, and solver-backed
  feasibility contracts. These files run in ambient scope, so `MOI`, `HAS_HIGHS`,
  `Uniform`, and the imports from `runtests.jl` are already visible. The include
  loop sits *outside* the `if HAS_HIGHS` guard, so each file must guard its own
  solver-dependent tests (the established pattern is `@testset ... begin` wrapping
  an inner `if HAS_HIGHS`); otherwise the direct `julia --project=@.
  test/runtests.jl` run errors instead of skipping.

## Adding a category or variant

**New variant in an existing category**: add
`src/problem_types/<category>/<variant>.jl` following the pattern above, `include`
it from the category entry point, extend `test/problem_types/<category>.jl` with
its quality contracts (create the file if the category has none — the include loop
finds it automatically), and run the tests.

**New category**: additionally create the entry point
`src/problem_types/<category>/<category>.jl` and add one
`include("problem_types/<category>/<category>.jl")` line to `src/SyntheticLPs.jl`.
Call `register_category(:category, "…")` there only when you want a category-level
description distinct from its variants. Consider adding a `docs/<category>.md`
page; see `docs/README.md` for the index and the explainer rebuild step.
