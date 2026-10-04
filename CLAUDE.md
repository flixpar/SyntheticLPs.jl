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
make test    # full Julia test suite at -O1 (make test CATEGORIES=tsp to focus)
make check   # lint and run the complete Julia test suite
```

### Testing

Always pass `-O1`. The suite is compilation-bound — roughly half its runtime was
JIT — so `-O1` cuts wall clock by ~23% while running every assertion. CI uses it
too. The full suite takes over ten minutes.

**Prefer a focused run.** Naming categories as command-line arguments limits the
per-variant sweeps and the per-category include loop to them. When only one
problem class has changed, run just that class — it takes seconds to a minute
rather than many minutes:

```bash
# One category. Use this by default when working on a single generator.
julia --project=@. -O1 test/runtests.jl transportation

# Several, space- or comma-separated:
julia --project=@. -O1 test/runtests.jl tsp,knapsack

# Through Pkg.test (keeps the solver-based testsets):
julia --project=@. -e 'using Pkg; Pkg.test(; test_args=["tsp"], julia_args=["-O1"])'

# Or via make:
make test CATEGORIES=tsp,knapsack
```

An unregistered category name is an error rather than a silent no-op, so a typo
cannot masquerade as a passing focused run.

Run the full suite before committing, and whenever a change touches
`src/SyntheticLPs.jl`, `src/transforms.jl`, `src/dataset.jl`, `test/runtests.jl`,
or `test/transforms.jl` — those are shared by every category, so a focused run
cannot cover them.

HiGHS is a test-only dependency in `[extras]`. Both commands work:

```bash
# Full suite, including the solver-based feasibility-contract testsets:
julia --project=@. -e 'using Pkg; Pkg.test(; julia_args=["-O1"])'

# Direct run: skips the solver-based testsets (HiGHS is not resolvable outside
# the Pkg.test sandbox):
julia --project=@. -O1 test/runtests.jl
```

`test/runtests.jl` loads HiGHS lazily behind a `HAS_HIGHS` flag, so the direct
run skips those testsets with an `@info` notice instead of erroring.

Framework-level testsets always run, focused or not; they are cheap and guard the
shared machinery. Note `--problem-types` (`scripts/generate_lps.jl`), `--types`
(`scripts/analyze_problem_statuses.jl`), and `--variants`
(`scripts/audit_generators.jl`) select what to *generate*, not what to test; the
positional arguments above are the test-side filter.

### Problem generation

The scripts activate the `scripts` environment themselves (HiGHS, ArgParse) and
`Pkg.develop` the local package into it.

```bash
julia --project=scripts scripts/generate_problem.jl list   # variants with tags, cap, class
julia --project=scripts scripts/generate_problem.jl transportation 100 output.mps
julia --project=scripts scripts/generate_problem.jl knapsack/bounded 50 --feasible --solve
```

### Dataset generation

`generate_dataset` builds a whole dataset; `scripts/generate_lps.jl` is a thin CLI
wrapper that supplies HiGHS:

```bash
julia --project=scripts scripts/generate_lps.jl -o output -n 100
julia --project=scripts scripts/generate_lps.jl -o output -n 50 --feasible-only -q -v
julia --project=scripts scripts/generate_lps.jl -o big -n 1000 --seed 7 \
    --size-distribution loguniform --var-min 1000 --var-max 100000 --dry-run
```

### Auditing at scale

`scripts/audit_generators.jl` measures size ratio, build time, HiGHS presolve
survival, and solve status per (variant, target, status, seed), one JSON line
each, and `--report` turns the JSONL into a flagged markdown table. Use it after
any generator change that could affect scaling. In a worktree, stack the main
checkout's `scripts` environment so the audit uses the worktree's source:

```bash
JULIA_LOAD_PATH="@:/path/to/SyntheticLPs.jl/scripts:@stdlib" julia --project=. \
    scripts/audit_generators.jl -o audit.jsonl --variants energy --targets 1000,10000,100000
julia --project=scripts scripts/audit_generators.jl --report audit.jsonl --flagged-only
```

## Architecture

Problems are a two-level hierarchy: a **category** is a problem domain (e.g.
`:transportation`) grouping one or more **variants**, each a concrete generator
with its own data generation and formulation (e.g. `:standard`). There are 50
categories and 142 variants. Query the live registry — `list_categories()`,
`list_variants(:cat)`, `list_problems(...)`, `problem_info(...)` — rather than a
hardcoded list; `README.md` holds the catalog and `docs/<category>.md` the
per-category notes.

**Main module** (`src/SyntheticLPs.jl`):
- `ProblemGenerator` (abstract base type for generators), `FeasibilityStatus`
  (`feasible`, `infeasible`, `unknown`), and `ProblemVariant` — the canonical
  reference to one `category/variant` pair, constructible from two symbols, a
  bare category symbol (→ default variant), or a `"category/variant"` string
- Two-level registry `LP_REGISTRY::Dict{Symbol,CategorySpec}`, populated by
  `register_category()` and `register_variant()` (a variant lazily creates its
  category). Each `VariantSpec` carries `tags`, `min_target_variables`,
  `max_target_variables` (a documented cap, or `nothing`), and an optional
  declared `model_class`
- **Registry metadata**: tags come from the controlled vocabulary `VARIANT_TAGS`
  (structure tags such as `:network`, `:staircase`, `:big_m`; `register_variant`
  rejects unknown tags, `register_tag` extends the set). `DOMAIN_TAGS` is the
  application subset, and every variant carries exactly one domain tag.
  `model_class(ref)` (`:lp`/`:mip`) is derived lazily from a 200-variable probe
  build and cached. `supports_target`, `variant_tags`, `list_tags`, and
  `model_statistics(model)` (variables, affine rows, nonzeros, integer columns)
  complete the introspection API
- `list_problems(; problem_types, exclude, model_class, tags, any_tags,
  exclude_tags, target_variables)` is the shared selector used by
  `generate_random_problem` and `generate_dataset`; `variant_weights` implements
  `variant_weighting=:category` (default; uniform over categories, then variants),
  `:variant`, or an explicit `Dict`
- `generate_problem()` (accepts a category symbol with optional `variant=`, a
  `ProblemVariant`, a string, or a generator type), `generate_random_problem()`
  (also returns the selected `ProblemVariant`), and `build_model(problem)`, which
  every variant implements

**Feasibility-contract verification**: every `generate_problem` /
`generate_random_problem` overload accepts an optional `optimizer` (plus
`max_feasibility_retries=10`, `feasibility_timeout=10.0`). When supplied for a
`feasible`/`infeasible` request, the built model is solved on a copy and the pure
`_classify_termination(ts, status)` returns one of three verdicts:
- `:holds` — proved; return the model.
- `:violated` — disproved (`INFEASIBLE` for a `feasible` request;
  `OPTIMAL`/`DUAL_INFEASIBLE` for an `infeasible` one). Rebuild with the next seed.
- `:inconclusive` — certifies nothing (`TIME_LIMIT`, `ALMOST_OPTIMAL`,
  `INFEASIBLE_OR_UNBOUNDED`, `OTHER_ERROR`, or anything else). Raises immediately
  rather than spending the retry budget re-asking an unanswerable question.
  Unrelaxed MIPs are the common trigger — raise `feasibility_timeout`.

`optimizer` may be a vector — an **escalation chain** tried in order while the
verdict is `:inconclusive` (`:violated` is final; each entry gets the full
`feasibility_timeout`). HiGHS dual simplex returns
`OTHER_ERROR` on some large infeasible MDP, forest, refinery, and blending LPs that
its IPM proves infeasible, so use `[HiGHS.Optimizer,
optimizer_with_attributes(HiGHS.Optimizer, "solver" => "ipm")]`. This is the
project-level backstop for the few generators whose heuristic feasibility logic
occasionally misses. It lives in `generate_problem`, not per-variant; with the
default `optimizer=nothing`, generation is unchanged. Retries walk
`seed, seed+1, …`, so a given `(seed, optimizer)` pair always resolves to the same
model.

**Model transforms** (`src/transforms.jl`) — post-`build_model` reformulations of
the finished JuMP model, applied centrally in `generate_problem()` in this order:
1. `relax_integer=true` (the default) relaxes integrality.
2. `bounds_to_constraints=true` reformulates variable bounds as explicit affine
   rows, keeping a plain `x ≥ 0` bound but converting upper, fixed, and nonzero
   lower bounds — including those introduced by relaxation. Converted bounds are
   genuine rows, so they raise
   `num_constraints(...; count_variable_in_set_constraints=false)` and affect
   dataset size-matching and quality thresholds. They are singleton rows, which
   every presolver turns back into bounds: this changes the file, not what a
   presolving solver sees.
3. Feasibility verification (when `optimizer` is set) solves this primal.
4. `transforms=ModelTransforms(...)` (or a `NamedTuple` of its keywords; identity
   by default) applies practitioner-style reformulations in the fixed order
   `aggregate_rows!` (redundant block-total rows; equivalence) →
   `elasticize_rows!` (penalized violation columns; a *relaxation*, refused for
   `infeasible` requests) → `permute_model` (row/column order; equivalence) →
   `scale_units!` (per-family powers of ten kept within `[1e-6, 1e6]`, integer
   columns unscaled, `UnitScaling` record in `model.ext`; equivalence), each from
   its own RNG stream seeded by the resolved instance seed (`apply_transforms`).
5. `dualize=true` (or a per-instance `dualize_probability`) returns a separately
   named dual model (`dual_var_`/`dual_con_` prefixes) of the transformed primal;
   it rejects unrelaxed discrete variables and splits ranged rows on an internal
   copy. Feasibility verification applies to the source primal; size and quality
   metadata to the returned model.

**Dataset generation** (`src/dataset.jl`): two stages. `plan_dataset(; kwargs...)`
(cheap, builds nothing) assigns every index a variant (stratified by
`variant_weighting`), a feasibility status (a single status or a `Dict` mix spread
by a low-discrepancy sequence), a target size (stratified quantiles of
`size_distribution` — `:normal`, `:uniform`, `:loguniform`, or any
`UnivariateDistribution`), and a private RNG stream. `generate_dataset(; kwargs...)`
then builds each index independently, **calibrating** size by rebuilding on the
same seed with a rescaled request until `|log(actual/target)| ≤
size_match_tolerance` (this replaced the old `candidate_multiplier` /
`match_size_by_type` candidate pool). An index depends only on `(seed, index)`, so
`shard_index`/`num_shards` shards union to the unsharded dataset and
`merge_manifests` combines their manifests. `on_failure=:error|:skip` controls
exhausted indices (`DatasetFailure`). It returns a `GeneratedDataset` (a vector of
`GeneratedInstance` plus `failures` and the format-version-2 `manifest`: provenance,
config, selection, shard, stats, size_match, instances, failures). Variants whose
size cap cannot cover the distribution are dropped and recorded.
`check_quality(model, optimizer; ...)` with `QualityCriteria`/`QualityResult`
filters trivial, degenerate, unbounded, and ill-conditioned instances; with
`quality_filter=true` its solve doubles as verification. The package stays
solver-agnostic: the caller supplies the optimizer.

**Problem generators** (`src/problem_types/<category>/`): a `<category>.jl` entry
point that `include`s one file per variant (or per closely related group), often a
`common.jl` of data and helpers shared by the category's variants, plus an
optional `register_category` call for a category-level description. Cross-category
helpers exist too: `network_flow/geo_network.jl` (geographic node placement,
near-linear kNN, sparse strongly connected networks with an exact arc count,
shortest-path trees, Dinic max flow / min cut) backs `network_flow`,
`transportation`, `multi_commodity_flow`, `load_balancing`, and `assignment`.
Reuse such helpers rather than writing new graph code.

### Generator pattern

```julia
# src/problem_types/<category>/<category>.jl  (entry point)
# Optionally: register_category(:category, "Category-level description")
include("common.jl")    # optional shared helpers
include("standard.jl")
```

```julia
# src/problem_types/<category>/standard.jl  (a variant)
struct VariantWitness            # typed planted solution
    x::Vector{Float64}
end
struct VariantCertificate        # typed solver-free infeasibility proof
    multipliers::Vector{Float64}
end

struct VariantStruct <: ProblemGenerator
    # every field build_model needs
    feasible_witness::Union{Nothing, VariantWitness}
    infeasibility_certificate::Union{Nothing, VariantCertificate}
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
register_variant(:category, :standard, VariantStruct, "Description";
                 tags=[:network, :staircase, :logistics],  # structure tags + one domain tag
                 max_target_variables=1_000_000)           # only for a documented cap
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
   planted rather than hoped for, store a typed `feasible_witness` (a complete
   primal solution) for `feasible` requests, a typed `infeasibility_certificate`
   for `infeasible` ones, and neither for `unknown`; almost every category does.
   Certificates must be checkable without a solver (Farkas multipliers, a cut or
   Hall deficit, a capacity prefix, a Lagrangian bound) and tested.
4. **Presolve-resistant infeasibility**: never plant infeasibility as a
   single-row contradiction or an impossible bound — presolve detects it without
   simplex work. Make the certificate combine many rows (across periods, regions,
   or a network cut). In a MIP category, build the certificate from LP rows alone
   so the infeasibility survives the default `relax_integer=true`.
5. **Presolve survival**: an instance should be hard for the solver, not the
   presolver. Emit single-variable limits as variable bounds, never as singleton
   rows; avoid rows that fix or eliminate variables trivially, duplicated or
   dominated rows, and structure that collapses under presolve. The audit flags
   presolved row or column ratios below 0.6.
6. **Size fidelity at every scale**: the variable count should track
   `target_variables` from tiny requests (~20) to 100k+ — within a few percent at
   1k–100k — so dataset size calibration converges. Builds must be near-linear in
   size (no O(n²) pair sampling, all-pairs distance matrices, or other dense
   intermediates) and nonzeros bounded (sparse rows; a few million nonzeros at
   100k at most). Check with `scripts/audit_generators.jl`.
7. **Sizing limits**: a generator whose data cannot scale further documents a cap,
   registers it as `max_target_variables` (usually 1,000,000), and raises
   `ArgumentError` above it rather than silently undersizing. Dataset generation
   never samples a variant above its cap.

### Model classes

The corpus deliberately mixes pure LPs (70 variants), natural MIPs (72;
binary/integer variables), and purpose-built LP relaxations. The public API
defaults to `relax_integer=true`, so MIP variants are returned as relaxations
unless the caller opts out. When building an LP-only corpus, filter with
`model_class=:lp` or relax the MIP variants; when characterizing instances, do not
present a relaxation as a real-world integer solution (`tsp/assignment_relaxation`
in particular is a fractional degree relaxation that may contain subtours, not a
tour).

### Testing strategy

`test/runtests.jl` holds only framework-level coverage; everything specific to one
category lives in `test/problem_types/<category>.jl`, so a generator's source,
documentation, and regression coverage evolve as one reviewable unit.

- `test/runtests.jl`: `test_problem_generator(ref)` applied to every registered
  variant (target variable counts, all three statuses, model structure,
  reproducibility), plus registry and interface tests, `Registry Metadata` and
  `Registry Tag Coverage` (every variant has tags and exactly one domain tag; a
  temporary `_UNTAGGED_CATEGORIES_PENDING` set exempts categories still being
  tagged), global-RNG isolation, dataset planning and generation controls, the
  bounds-to-constraints transform, dual reformulation, the pure
  `_classify_termination` table, and the generic feasibility-contract machinery
  (retry budget, seed walk, pristine-model guarantee). It includes
  `test/transforms.jl` (the `ModelTransforms` contracts) and ends with an include
  loop over every `test/problem_types/*.jl` in sorted order.
- `test/problem_types/<category>.jl`: focused contracts for one category —
  registry shape, exact variable-count formulas, data invariants,
  witness/certificate arithmetic, edge-size robustness, and solver-backed
  feasibility contracts. These files run in ambient scope, so `MOI`, `HAS_HIGHS`,
  `Uniform`, and the imports from `runtests.jl` are already visible. The include
  loop sits *outside* the `if HAS_HIGHS` guard, so each file must guard its own
  solver-dependent tests (the established pattern is `@testset ... begin` wrapping
  an inner `if HAS_HIGHS`); otherwise the direct `julia --project=@. -O1
  test/runtests.jl` run errors instead of skipping.
- The focused-run filter matches the include loop by file basename, so a
  category's test file must be named after the category for a focused run to pick
  it up.
- Scale (100k builds, presolve survival, solve behavior) is too slow for the test
  suite; it is checked with `scripts/audit_generators.jl` instead.

## Adding a category or variant

**New variant in an existing category**: add
`src/problem_types/<category>/<variant>.jl` following the pattern above (with
`tags`, and `max_target_variables` if capped), `include` it from the category
entry point, extend `test/problem_types/<category>.jl` with its quality contracts
(create the file if the category has none — the include loop finds it
automatically), update `docs/<category>.md`, run the focused tests, and audit it
at 1k/10k/100k.

**New category**: additionally create the entry point
`src/problem_types/<category>/<category>.jl` and add one
`include("problem_types/<category>/<category>.jl")` line to `src/SyntheticLPs.jl`.
Call `register_category(:category, "…")` there only when you want a category-level
description distinct from its variants. Add a `docs/<category>.md` page, list it in
`docs/README.md`, add its `META` entry to `scripts/build_explainer.py` (the build
fails on a docs/`META` mismatch), and rebuild `docs/explainer.html`. Add the
category to the `README.md` catalog.
