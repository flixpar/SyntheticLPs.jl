# CLAUDE.md

Orientation for agents working in this repository. Details live elsewhere; this
file says where, and records what a quick scan of the repo will not tell you.

## What this is

A Julia package that generates synthetic linear programs (JuMP models) realistic
enough to test and develop LP solvers. Problems are organized as **categories**
(problem domains, one folder each under `src/problem_types/`) containing one or
more **variants** (concrete generators). Its main consumer is the sibling SimplexRL
project, which presolves every instance once with HiGHS and studies simplex pivot
paths. An instance is therefore only useful if it stays hard *after presolve*.

## Where to look

- `README.md`: the public API, feature walkthroughs (feasibility verification,
  transforms, dataset generation, CLI, auditing), the quality bar ("Scale and
  quality"), and the variant template ("Extending").
- `docs/README.md` covers the shared generator contract. `docs/<category>.md`
  has each category's formulations, witness and certificate types, measured
  sizes and known solver behavior.
- `CHANGELOG.md`: why things are the way they are. Read the recent sections
  before reworking a generator.
- Ask the live registry (`list_categories()`, `list_variants(:cat)`,
  `problem_info(ref)`, `generate_problem.jl list`) instead of relying on
  hardcoded counts or lists.
- Code map:
  - `src/SyntheticLPs.jl` holds the registry, `generate_problem` and feasibility
    verification.
  - `src/transforms.jl` holds the post-build reformulations.
  - `src/dataset.jl` holds dataset planning and generation.
  - `src/problem_types/network_flow/geo_network.jl` has shared graph helpers
    (kNN, sparse networks, shortest paths, max flow). Reuse them instead of
    writing new graph code.

## Conventions every generator follows

- **Determinism.** The constructor `(target_variables, feasibility_status, seed)`
  does all the sampling, from a local `rng = MersenneTwister(seed)`. Every helper
  it calls takes that `rng` as its first argument. The global RNG is never used,
  and a test enforces this. `build_model` is deterministic and builds only from
  the struct's fields.
- **Planted outcomes.** All three statuses are handled.
  - `feasible` stores a typed primal witness.
  - `infeasible` stores a typed certificate that can be checked without a
    solver (Farkas multipliers, a cut deficit, a capacity prefix, ...).
  - `unknown` stores neither.

  The certificate is tested.
- **Presolve resistance.** Infeasibility must combine many rows, never one
  contradictory row or bound. In MIP categories, build it from LP rows only so it
  survives the default `relax_integer=true`. Emit single-variable limits as
  bounds, not singleton rows. Avoid structure that presolve fixes, eliminates or
  finds dominated.
- **Size fidelity and scale.** The variable count tracks `target_variables` from
  about 20 up to 100k+, within a few percent from 1k up. Dataset size
  calibration depends on this. Clamp tiny targets upward rather than throwing.
  Builds are near-linear in size, with sparse rows and no dense or all-pairs
  intermediates. A generator that cannot scale registers
  `max_target_variables` and raises above it.
- **Tags.** Each variant registers structure tags from `VARIANT_TAGS` plus
  exactly one domain tag.
- **MIPs are relaxed by default.** About half the variants are natural MIPs. Never
  present a relaxed solution as an integer one; `tsp/assignment_relaxation` may
  contain subtours.

Judge generator changes by measurement, not inspection.
`scripts/audit_generators.jl` reports size ratio, build time, presolve survival
(flagged below 0.6) and solve status at 1k/10k/100k. Run it after any change
that could affect scaling.

## Working conventions

- Research code under active development: breaking changes are fine and
  backwards compatibility is not a concern. Don't over-engineer edge cases.
- After any significant change, add a `CHANGELOG.md` section. Use one section per
  date, giving the commit hash and datetime, a summary, and details finer than
  the commit messages. Update `README.md`, `docs/` and this file when behavior
  they describe changes.
- Run `make format` and `make lint` before committing; setup is
  `python3 -m pip install -r requirements-dev.txt && make setup`.
- New variant: add the source file, `include` it from the category entry point,
  add contracts to `test/problem_types/<category>.jl`, update
  `docs/<category>.md`, then audit. A new category also needs:
  - an `include` line in `src/SyntheticLPs.jl`
  - a `docs/<category>.md` page, listed in `docs/README.md`
  - a `META` entry in `scripts/build_explainer.py` (the build fails on a
    mismatch with `docs/`), then rebuild `docs/explainer.html` with
    `python3 scripts/build_explainer.py`
  - an entry in the `README.md` catalog

## Testing: non-obvious points

- **Always pass `-O1`.** The suite is compilation-bound, and CI uses `-O1` too.
  The full suite takes over ten minutes.
- **Prefer focused runs.** Pass category names as arguments:
  `julia --project=@. -O1 test/runtests.jl tsp,knapsack`, or
  `make test CATEGORIES=tsp`. An unknown name is an error. Framework testsets
  always run.
- **Run the full suite** (`make test`) before committing, and whenever you
  touch `src/SyntheticLPs.jl`, `src/transforms.jl`, `src/dataset.jl`,
  `test/runtests.jl` or `test/transforms.jl`.
- **HiGHS is test-only.** The direct `test/runtests.jl` run cannot load it and
  skips the solver testsets (`HAS_HIGHS == false`). `make test` / `Pkg.test`
  runs everything.
- **Per-category test files** (`test/problem_types/<category>.jl`):
  - They run in the ambient scope of `runtests.jl`.
  - They must be named after the category, or a focused run won't find them.
  - Each one must wrap its own solver-backed tests in `if HAS_HIGHS`.
- Scale behavior (100k builds, presolve, solve times) is too slow for tests; use
  the audit script.

## Scripts and solver quirks

- The scripts in `scripts/` activate their own environment (HiGHS, ArgParse)
  and `Pkg.develop` the package into it: `julia --project=scripts
  scripts/<name>.jl`. In a git worktree, stack environments so the worktree's
  source is used:
  `JULIA_LOAD_PATH="@:/path/to/main/checkout/scripts:@stdlib" julia --project=.
  scripts/audit_generators.jl ...`.
- HiGHS dual simplex sometimes returns `OTHER_ERROR` on large infeasible LPs
  (MDP, forest, refinery, blending) that its IPM proves infeasible. When
  verifying feasibility, pass an escalation chain:
  `optimizer=[HiGHS.Optimizer, optimizer_with_attributes(HiGHS.Optimizer, "solver" => "ipm")]`.
  An inconclusive verdict raises immediately rather than retrying.
  Unrelaxed MIPs usually need a larger `feasibility_timeout`.
- `tmp/` is gitignored scratch space.
