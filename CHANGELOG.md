# Changelog

All notable changes to SyntheticLPs.jl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## 2026-08-28 18:10 UTC (generic_milp sparse support sampling)

**Previous Commit**: `8ce8ccb`

**Summary**: PR-review fix — each generic MILP row drew its sparse support as
the prefix of a full `n`-element permutation, and the constructor builds Θ(n)
rows. Support generation was therefore Θ(n²) time and allocation (about three
billion indices at `n = 100_000`). Supports are now sampled without replacement
in work proportional to the requested width.

**Details**:
- New `_generic_sample_indices` draws `width` distinct columns via a set
  (expected O(width) while `width ≪ n`, which is the advertised
  `width ~ √n` regime) and sorts them. `_generic_sparse_support` uses it in
  place of `randperm(n)[1:width]`. The one-shot `randperm` in the variable
  layout is unchanged.
- Seed-identical instances change because the support RNG stream is shorter.
- Tests: tiny-target generation plus sorted/unique support checks at target 200.

## 2026-08-28 17:41 UTC (OTS sparse extra-line sampling)

**Previous Commit**: `42b1bdf`

**Summary**: PR-review fix — `energy/optimal_transmission_switching`
materialized every undirected pair among `n_buses` buses, then shuffled and
kept only ~1.45–2.05 lines per bus. A ~100,000-variable request therefore
allocated hundreds of millions of tuples. Extra mesh lines are now sampled
by rejection, matching `energy/dc_opf`.

**Details**:
- Replaced the `candidates = [(i, j) for i in 1:n_buses for j in (i+1):n_buses];
  shuffle!` loop with random endpoint-pair attempts until `n_lines` is reached
  (capped at `50 * n_lines` attempts). The spanning-tree prefix is unchanged.
- The delivered graph remains simple and at least a tree; typical sizing is
  sparse, so the attempt cap is not binding. Seed-identical instances change
  because extra-line sampling (and thus the downstream RNG stream) changed.
- Tests: tiny-target generation plus a uniqueness / tree-size check at
  target 500.

## 2026-08-28 17:36 UTC (container_loading clamp tiny targets)

**Previous Commit**: `6078534`

**Summary**: PR-review fix — both `container_loading` constructors rejected
targets below a hard floor (`standard` at 12, `two_dimensional_bin_packing` at
30). `generate_random_problem` and `generate_dataset` accept sizes down to 2,
so those calls could throw or exhaust candidates. Both now clamp to their
smallest formulation instead.

**Details**:
- `container_loading/standard`: drop the `target_variables >= 12` throw; clamp
  to 6 (two items × two containers + two use-indicators) and keep the existing
  dimension search. Targets ≥ 12 are unchanged.
- `container_loading/two_dimensional_bin_packing`: drop the
  `target_variables >= 30` throw; clamp to 26 (`n=3`, `b=2`, the search's
  existing lower corner). Targets ≥ 30 are unchanged.
- Tests: tiny-target `@test_nowarn` coverage for both variants in the
  generator-robustness testset.

## 2026-08-27 13:03 UTC (tsp/flow exact dimension sizing)

**Previous Commit**: `aa98448`

**Summary**: PR-review fix — `tsp/flow` sized its dimension `n` with the
large-`n` approximation `n ≈ sqrt(target/2)`, dropping the linear `-2n` term of
the actual count `2n(n-1)` and so biasing `n` low by ~1/2 a unit. Replaced with
the exact positive root `n = (1 + sqrt(1 + 2·target)) / 2`, matching how the
sibling variants already invert their counts (`assignment_relaxation` and
`prize_collecting` use exact quadratic roots; `standard`'s `sqrt(target+1)` is
exact for `n²−1`).

**Details**:
- `src/problem_types/tsp/flow.jl`: `n0 = round(sqrt(target/2))` →
  `round((1 + sqrt(1 + 2·target))/2)` in the constructor, with the sizing
  comment, the constructor-docstring formula/examples (`target = 100` now
  yields `n = 8`, 112 vars, instead of `n = 7`, 84 vars — the closer match;
  `target = 500` is unchanged at `n = 16`), and the overview docstring's
  `n`-sizing comparison against `tsp/standard` updated to match.
  The infeasible branch is unaffected in mechanism — it already re-centres `n`
  against the exact delivered count via `_tsp_pick_n` — and its search window
  now starts from the unbiased `n0`.
- Only the feasible/unknown branches change behaviour for a given target
  (instances for borderline targets shift by one dimension step, e.g.
  `target = 210` now builds `n = 11` / 220 vars instead of `n = 10` / 180).
  Per-seed reproducibility is unchanged.
- `vehicle_routing/cvrp` (pre-dates this PR) uses the same
  `round(sqrt(target/2))` approximation for its `2N(N+1)` count; left as is —
  out of this PR's scope, noted as a possible follow-up.

## 2026-08-27 12:03 UTC (tsp review fixes)

**Previous Commit**: `66b690c`

**Summary**: Post-review corrections to the integrated TSP family — a wrong
witness-proof in a docstring, explainer rendering artifacts in `docs/tsp.md`,
missing test coverage for the infeasible-branch sizing lambdas and tiny-target
paths, and two contained refactors (shared Hall-block constructor helpers,
shortest-path buffer reuse) verified bit-identical on saved snapshots.

**Details**:
- `tsp/standard` constructor docstring: the relaxed-witness reduction quoted the
  *unlifted* MTZ row (`n/(n-1) ≤ n-1`); the model actually builds the lifted row
  `u[i]-u[j]+(n-1)x[i,j]+(n-3)x[j,i] ≤ n-2`, whose witness reduction is
  `1 + (n-3)/(n-1) = (2n-4)/(n-1) ≤ n-2`. Docstring-only fix.
- `docs/tsp.md`: reflowed the wrapped bullets in "Feasibility controls" to one
  line each (matching every other doc page) and fixed the `Multiple-` line-break
  hyphenation, then regenerated `docs/explainer.html`. The wrapped form made the
  explainer renderer split the bullets mid-sentence into nested `<p>` blocks and
  display "Multiple- salesperson".
- Tests (`test/runtests.jl`): the eight unrolled variable-count assertions are
  now a loop over `(variant, formula)` pairs; added infeasible-branch assertions
  tying each variant's `delivered()` sizing lambda to the built model's actual
  `num_variables` (previously only `unknown` instances were counted); added the
  three missing tiny-target (`target = 3`) `@test_nowarn` lines for
  `tsp/standard`, `tsp/asymmetric`, and `tsp/assignment_relaxation` so all eight
  variants exercise the `n → 5` clamp.
- `src/problem_types/tsp/tsp.jl`: new `_tsp_plan_dimensions` and
  `_tsp_arc_support` helpers; the four Hall-block variant constructors
  (`standard`, `asymmetric`, `flow`, `assignment_relaxation`) now share them
  instead of copy-pasting the k-draw / `_tsp_pick_n` / arc-support block.
  RNG-stream order is unchanged — verified by byte-identical snapshots of
  `dist`, `arc_ok`, `blocked_set`, and `gate_set` across 4 variants × 3 statuses
  × 10 seeds before and after.
- `tsp/asymmetric`: `_tsp_street_shortest_paths` now takes caller-owned scratch
  buffers (defaults preserve the old behavior), and the constructor reuses one
  `distances`/`buckets` pair across its per-source calls instead of allocating
  ~12n² bucket vectors per source; the connectivity check now tests only the
  city-vertex entries actually read instead of rescanning all S² entries.
  Distances verified bit-identical across 12 seeds.
- `tsp/multiple_salespersons`: hoisted the duplicated `sum(x[1, j])` affine, and
  the docstring now notes that unrelaxed MIPs at `target_variables ≥ ~300` can
  exceed the central verifier's default `feasibility_timeout` (10 s) and need a
  larger timeout.

## 2026-08-27 02:22 UTC (integrated TSP generator family)

**Previous Commit**: `99db595`

**Summary**: Expanded the `tsp` category from five to eight variants and
integrated the strongest application, data-generation, formulation, testing,
and documentation ideas from the four independently developed TSP branches.
All feasibility mechanisms remain valid for the package's default continuous
relaxation.

**Details**:
- Added `prize_collecting`, `multiple_salespersons`, and `precedence` variants,
  refactored onto the category's shared clustered geography and metric-distance
  helpers. Their infeasible modes use quota overflow, aggregate fleet-capacity
  shortfall, and cyclic precedence respectively, all of which survive
  integrality relaxation.
- Replaced independent asymmetric pair perturbations with shortest-path travel
  times on an explicit strongly connected street grid: alternating one-way
  horizontal streets, two-way vertical avenues, and sampled street congestion.
- Strengthened the `standard` and `asymmetric` MTZ formulations with lifted
  reverse-arc terms. Corrected the multiple-salesperson port so depot departures
  anchor order at one and selected customer arcs advance order exactly; return
  order therefore enforces both minimum and maximum stops per route.
- Clarified that `assignment_relaxation` is a strengthened degree LP with
  pairwise two-cycle cuts, rather than the plain assignment relaxation, and
  corrected the flow variant's sizing comparison.
- Added `docs/tsp.md`, documentation index and explainer metadata, updated the
  README/CLAUDE variant taxonomy, and expanded structural, sizing, edge-case,
  and solver-backed feasibility tests to all eight variants.

## 2026-08-27 00:07 UTC (tsp generator family)

**Previous Commit**: `73f8f54`

**Summary**: Added the `tsp` category — five travelling-salesman generators
varying along both axes the corpus cares about (real-world data realism and LP
formulation) over one shared geography, so structural differences between the
delivered LPs are attributable to formulation class rather than a different
data distribution. All feasibility claims are proven for the LP relaxation,
since `generate_problem` verifies the contract on the *delivered* (by default
relaxed) model.

### Added

- **New category `tsp`** with variants `standard` (default), `asymmetric`,
  `flow`, `time_windows`, and `assignment_relaxation` (33 categories total).
  Entry point `src/problem_types/tsp/tsp.jl` registers the category and hosts
  the shared, RNG-consuming helpers `_tsp_stops` (tiered geography: depot near
  the region centre, stops in 1–6 Gaussian town clusters plus ~20% uniform
  rural scatter), `_tsp_distance` (symmetric road metric: Euclidean × one
  per-instance circuity factor 1.15–1.45, floored at 0.5 so distinct stops are
  never at distance zero — a true metric up to rounding, unlike the CVRP's
  deliberate per-arc asymmetry), `_tsp_full_support`, `_tsp_hall_block`, and
  `_tsp_pick_n`.
- **`tsp/standard`** — symmetric courier-tour TSP with the polynomial-size
  Miller–Tucker–Zemlin formulation: binary arc variables `x[i,j]` plus
  continuous visit-order variables `u[j] ∈ [1, n-1]` coupled by
  `u[i] - u[j] + n·x[i,j] ≤ n-1`. Variables `n²−1`, so
  `n = max(5, round(Int, sqrt(target+1)))` (99 vars at target 100, 483 at 500).
- **`tsp/asymmetric`** — urban ATSP with traffic-dependent travel times: the
  shared symmetric base is perturbed by per-direction congestion factors
  (0.8–1.5) and closed with Floyd–Warshall, i.e. every entry becomes the
  shortest path through the congested network, which restores the triangle
  inequality while keeping the matrix asymmetric (178 of 182 ordered pairs
  differ in a spot check). Same MTZ model and sizing as `standard`.
- **`tsp/flow`** — the same data-generating process as `standard` but the
  single-commodity-flow (Gavish–Graves) formulation used by `vehicle_routing/cvrp`:
  `x` binary plus supply `f ≥ 0` per arc, conservation (each stop consumes one
  unit, the depot sources `n−1`), and `f ≤ (n−1)·x`. A markedly stronger LP
  relaxation than MTZ's over the same kind of data. Variables `2n(n−1)`, so
  `n = max(5, round(Int, (1 + sqrt(1 + 2·target)) / 2))` — the exact positive
  root (112 vars at target 100, 480 at 500).
- **`tsp/time_windows`** — appointment-delivery TSPTW: metric travel times,
  service times, per-stop windows, an EV-style route budget
  `Σ τ_ij x_ij ≤ F`, and a shift limit on the return time. Subtour elimination
  comes free from time propagation (positive travel times force strictly
  increasing arrivals around any depot-free cycle), so no MTZ block; per-arc
  big-M values are clamped at 0 (`M = max(0, b_i + s_i + τ_ij − a_j)`), the
  smallest non-binding choice. Variables `n²` for *all* statuses
  (`n = round(sqrt(target))`).
- **`tsp/assignment_relaxation`** — the degree (assignment) LP relaxation of
  the TSP solved at the root of classical branch-and-bound: continuous arc
  fractions `x ∈ [0,1]` (integrality never declared, so it is an LP even with
  `relax_integer=false`), degree rows, and pairwise 2-matching cuts
  `x[i,j] + x[j,i] ≤ 1`. Documented as a relaxation: optima can be fractional
  and integer optima can decompose into subtours. Variables `n(n−1)`, so
  `n = round((1 + sqrt(1+4·target))/2)`.

### Feasibility scheme (shared, relaxation-proof)

- `feasible`/`unknown`: complete arc support — any permutation is a tour, and
  each docstring exhibits an explicit relaxed-feasible witness (`x = 1/(n−1)`
  with equal `u`; star supply `f[1,j] = 1` under capacity `(n−1)·x = 1`).
- `infeasible`: a Hall-deficit arc block (road closures cutting off a
  district) — a set `S` of `k ∈ {2,3}` stops keeps only the in-arcs from `k−1`
  gate nodes `T`, so the degree rows alone give
  `k = Σ_{j∈S} indeg(j) ≤ Σ_{i∈T} outdeg(i) = k−1`: infeasible even in the LP
  relaxation, identically for all four arc-based formulations. Blocked arcs are
  simply not instantiated (filtered index sets), keeping the model tight under
  `bounds_to_constraints!`. The plain-disconnection alternative was analysed
  and rejected: depot-free components with ≥3 nodes admit fractional points
  satisfying the relaxed degree *and* MTZ rows, so the relaxed model would stay
  feasible. `tsp/time_windows` instead sets the route budget below the
  degree-row lower bound on total travel time (`F = 0.85·Σ_i min_{j≠i} τ_ij`);
  its feasible branch plants a concrete tour, centres window openings on the
  no-wait schedule, and closes windows after a forward pass *with waiting*
  (the ordering that keeps the planted schedule admissible).
- Sizing on the infeasible branch targets the *delivered* variable count
  (best of `n0−1, n0, n0+1` after the block deletes `k(n−k)` arcs), and the
  block size `k` is drawn unconditionally so the RNG stream stays aligned
  across statuses.

### Tests and docs

- `test/runtests.jl`: new solver-free `"TSP Variants"` testset (registry
  wiring, matrix symmetry/asymmetry contracts, exact variable-count formulas,
  Hall-block structure, planted-tour budget certificate); a HiGHS-gated
  contract loop in `"Feasibility Contract Verification"` (5 variants × 5 seeds
  × both statuses through the central verifier); two edge-size robustness
  lines (target 3 exercises the `n → 5` clamp and `k → 2` fallback). The
  auto-discovered `test_problem_generator` loop covers the new variants with
  no further changes (37 assertions each; suite now 3201 passing under
  `Pkg.test`).
- Out-of-suite verification: 200/200 relaxed contract checks
  (feasible→OPTIMAL, infeasible→INFEASIBLE over 5 variants × 2 statuses × 20
  seeds), 20 unrelaxed MIP solves all OPTIMAL (≤3.1 s per 5-instance batch),
  MPS write/read round-trips for all variants, and dataset generation via
  `scripts/generate_lps.jl --problem-types tsp`.
- README.md: TSP bullet with its five variants; the stale "29 categories"
  count corrected to 33. CLAUDE.md: 33 categories, `tsp` in the category list,
  and the Model-classes taxonomy (four MIP variants with genuine tour
  relaxations; `assignment_relaxation` under LP relaxations of MIPs).

## 2026-07-28 01:10 UTC (verification correctness + review cleanup)

**Previous Commit**: `9dc7660`

**Summary**: Fixed two soundness bugs in the feasibility-contract verifier, made the
verification timeout configurable, removed the double solve in dataset generation,
improved `supply_chain` network realism, closed a latent `energy` feasibility hole,
and deleted the one-off `qa/` investigation scripts.

**Details**:
- **Verifier no longer conflates "uncertifiable" with "violated"** — the classifier
  now returns a three-valued verdict (`:holds` / `:violated` / `:inconclusive`)
  instead of a `Bool`. Previously any non-matching termination status counted as a
  contract violation, so a solve that merely hit the time limit triggered a
  rebuild-and-retry loop and eventually raised "Feasibility contract not satisfied".
  Reproduced on `job_shop_scheduling/standard` at target 2000 with
  `relax_integer=false`: the instance is not infeasible, HiGHS just needs longer than
  the limit — the old code burned the retry budget (100s at the default 10 retries)
  and then misdiagnosed it. `:inconclusive` now raises immediately, naming the
  termination status and the limit that produced it (7s, one attempt).
- **`INFEASIBLE_OR_UNBOUNDED` no longer satisfies an `infeasible` request.** The
  status separates neither case, so accepting it could certify an *unbounded* — hence
  feasible — model as infeasible. It is now `:inconclusive`. `ALMOST_OPTIMAL` (a solve
  that stopped short of its tolerances) is likewise no longer accepted as proof of
  feasibility. `DUAL_INFEASIBLE`, MOI's encoding of primal-unbounded, is now an
  explicit `:violated` for both statuses: it exhibits a nonempty feasible region
  (disproving `infeasible`) but has no optimum (disproving `feasible`).
- **Classification extracted to a pure `_classify_termination(ts, status)`** so the
  full status table is covered by a solver-free testset, including the `TIME_LIMIT`
  path that no reasonably fast solver call can produce on demand.
- **`feasibility_timeout` keyword** (default `10.0`) added to `generate_problem`,
  `generate_random_problem`, and `_generate_problem_verified`; the limit was
  previously hardcoded. Unrelaxed MIPs are the case that needs a larger value.
- **`generate_dataset` no longer solves each candidate twice.** With
  `feasible_only=true` and `quality_filter=true`, verification and `check_quality`
  were both solving the same model to answer the same question. Verification is now
  skipped when the quality filter is on — the filter already rejects anything not
  matching `feasible_only`, and the candidate-pool loop supplies the retry. The
  verification timeout is also tied to `QualityCriteria.solve_timeout` rather than the
  hardcoded 10s.
- **`supply_chain/standard` network shape** — the pool-growth step inflated
  `n_customers` alone to reach the variable budget, skewing instances toward a handful
  of facilities serving hundreds of customers (6 facilities / 115 customers at target
  1000). It now grows facilities and customers together, preserving the
  facility:customer ratio sampled per size band (11 / 66 at target 1000). Exact
  variable-count targeting is unchanged.
- **`energy/standard` zero-clean-capacity hole** — the renewable-floor guard was
  `clean_capacity < required && clean_capacity > 0`, silently doing nothing in exactly
  the case where the floor is least satisfiable. Now handled explicitly: no clean
  source at all drops the renewable floor to 0 (it cannot be met by any scaling);
  clean sources with zero capacity are assigned capacity outright, since scaling by a
  ratio cannot escape zero. Latent rather than live — 0 of 160 sampled feasible
  instances hit it — but the guard no longer depends on that.
- **Removed `qa/`** — the eight investigation scripts and the point-in-time
  `QA_REPORT.md` were one-off artifacts for this branch's audit, not reusable tooling.
  Their durable findings live in this changelog and in the regression tests.

## 2026-07-26 21:10 UTC (review corrections)

**Summary**: Two corrections from a self-review of the prior changes.

- **`supply_chain/standard` feasible-path sizing**: the deterministic combo
  selection landed on target for `unknown` but the pre-existing K-nearest
  coverage step added combos *on top* of the budget for `feasible`, inflating the
  variable count up to ~2×. The K-nearest coverage combos are now reserved *out
  of* the combo budget (with `K` capped so coverage fits), and the coverage step
  no longer adds combos. Verified 1.00× across all statuses and sizes 50–1500.
- **Test loadability**: `using HiGHS` at the top of `test/runtests.jl` made the
  direct `julia --project=@. test/runtests.jl` command hard-error (HiGHS is an
  `[extras]` dep, not resolvable outside `Pkg.test`). HiGHS is now loaded lazily
  (`HAS_HIGHS` flag) so the direct command runs the solver-free testsets and
  skips the two solver-based ones with an `@info` notice. Also switched the test
  file to `JuMP.MOI` and dropped the redundant `MathOptInterface` extra.

## 2026-07-26 19:50 UTC (variable-count targeting + trivial-instance fixes)

**Previous Commit**: `2cb557a`

**Summary**: Fixed variable-count drift in six generators (the pre-existing
size-targeting test failures) and eliminated structurally-trivial instances in
two. Every registered variant now lands within the project's own ±25% size bar
across small/medium/large targets, and the full test suite is green (2523/2523,
previously 6 failures).

**Details**:
- **`network_flow/standard`** — the node-count search left `n_nodes` at a tiny
  default whenever no node count in range met a density threshold, so the arc
  count (the variable count) came back a small fraction of target. Now sizes
  `n_nodes` from `target` so the complete digraph has ≥ target arcs, then emits
  exactly `target` arcs.
- **`scheduling/standard`** — picked `n_workers`, `n_shifts`, `n_days` all from
  fixed bands, so their product (the variable count) wandered 0.4–1.8× of target.
  Now keeps the horizon/shift bands for realism and solves `n_workers` from target.
- **`airline_crew/standard`** — realized pairing count drifted at larger sizes.
  Now sets `max_pairings = target` and keeps `num_flights` in `[target/2, 0.8*target]`
  so the exact-cover partition stays below target and the fill reaches it.
- **`cutting_stock/standard`** — the distinct-pattern pool stalled far below target
  for small piece-type counts (0.02–0.32×). Now floors `n_piece_types` so the
  pattern generator reaches target.
- **`supply_chain/standard`** — the valid-combo count was a high-variance Bernoulli
  draw that undershot badly when low-availability modes (inland ship, short-haul
  air) were sampled. Now ranks candidates by realistic per-mode availability and
  selects a deterministic count, landing the variable count on target while
  preserving "truck common, ship rare" structure.
- **`load_balancing/standard`** — rewritten to a genuine min-max-utilization
  routing LP: the previous model had per-link lower bounds but **no flow
  conservation**, so each link was independent and it solved in 0 simplex
  iterations. Now balances in-/out-flow at every node against demand injections
  (coupling the links). Sizing fixed (was ~0.4×). Feasibility is structural:
  connected network ⇒ always feasible (`u` unbounded); zeroing a source's
  outgoing capacity ⇒ provably infeasible via conservation. Also replaced six
  deprecated `Truncated(…)` calls with `truncated(…)` (Distributions.jl).
- **`production_planning/standard`** — raised the `n_resources` floor from
  `rand(1:50)` so instances no longer degenerate to a single-constraint
  (trivially solved) LP.

## 2026-07-26 18:19 UTC (feasibility-contract verification + generator fixes)

**Previous Commit**: `2cb557a`

**Summary**: Added project-level feasibility-contract verification to the
generation stack, and fixed five data-generation defects (four
feasibility-contract violations plus two edge-case build crashes) surfaced by a
solver-based QA sweep over every registered variant. The corpus now honors the
requested `FeasibilityStatus` by construction in the previously-broken cases, and
— when a caller supplies an `optimizer` — guarantees it by solving.

**Details**:

### Project-level feasibility verification (new)
- **`optimizer` kwarg on `generate_problem`** (all four dispatch methods),
  `generate_random_problem`, and the dataset candidate path. When supplied and
  the requested status is `feasible`/`infeasible`, the model is solved once (on a
  structural copy, so the returned model stays pristine) to verify the contract —
  a `feasible` request must solve `OPTIMAL`, an `infeasible` request must solve
  `INFEASIBLE`. On violation the problem is rebuilt with a fresh seed and
  re-checked, up to `max_feasibility_retries` (default `10`) attempts. Central in
  `generate_problem` (not per-variant), per the design that the package stays
  solver-agnostic and the caller supplies the optimizer (e.g. `HiGHS.Optimizer`).
- **`_generate_problem_verified` / `_feasibility_contract_holds`** — internal
  helpers; the verified builder returns the resolved seed so dataset materializers
  reproduce the exact verified model.
- **`generate_dataset`** records the resolved seed per instance, so `feasible_only`
  datasets with an optimizer are guaranteed feasible and remain reproducible.
- With `optimizer=nothing` (the default) generation is byte-for-byte unchanged and
  fully seed-deterministic.

### Generator fixes
- **`crop_planning/standard`** — the `infeasible` branch computed a lower bound on
  resource usage assuming *all* land is planted, but the land constraint is
  `sum(x) <= total_land` (an upper bound), so land can be left fallow and the true
  minimum usage is far lower — occasionally yielding a feasible problem. Now uses
  the true minimum `sum(req .* min_area)`, guarantees a non-empty mandatory set,
  and sets the violated capacity strictly below that bound. (~17% → 0 violations.)
- **`energy/standard`** — two fixes. (1) The infeasibility logic targeted a reserve
  constraint that is **not in the model** (`build_model` only enforces `Σx ≥ demand`
  and `x ≤ capacity`); it now guarantees `max(demand) > Σ capacities`, the
  model-consistent infeasibility condition. (2) The per-period emissions row
  `Σ em·x ≤ max_em·Σx` was an algebraic tautology (a weighted average never exceeds
  its max weight); replaced with a fixed `emission_intensity_target < max_em` so it
  can actually bind. (Size-dependent ~17% → 0 violations; feasible path unchanged.)
- **`portfolio/cvar`** — the position-limit `Uniform(max(2/n, 0.02), min(0.3, 5/n))`
  inverted (`a > b`) for `n_assets > 250` (target > ~1250) and for very small `n`,
  crashing with `ArgumentError`. Now sampled directly in `[2/n, 5/n]`.
- **`land_use/standard`** — `rand(2:min(4, n_parcels-1))` produced an empty range
  (`collection must be non-empty`) when `n_parcels == 2`. Now clamped to
  `max(1, min(n_parcels-1, rand(2:4)))`.
- **`load_balancing/standard`** — replaced six deprecated `Truncated(d, l, u)`
  calls with `truncated(d, l, u)` (Distributions.jl deprecation that surfaced as a
  test failure under `Pkg.test`).

### Tests / dev
- `HiGHS` + `MathOptInterface` added as **test-only** dependencies (`[extras]`/`
  [targets]`); the package runtime remains solver-agnostic. New testsets:
  *Generator Robustness Fixes*, *Feasibility Contract Verification* (covers
  crop_planning/energy infeasible and unit_commitment feasible, plus blending/
  feed_blending), and *Dataset Feasibility Verification* (resolved-seed
  reproducibility + guaranteed-feasible `feasible_only` datasets).
- The canonical test command is now `julia --project=@. -e 'using Pkg; Pkg.test()'`
  (so the HiGHS extra resolves). Six **pre-existing** size-targeting test failures
  (`airline_crew`, `cutting_stock`, `network_flow`, `scheduling`, `supply_chain`)
  remain unchanged on this branch — they are P2 variable-count drift, not
  regressions from these changes.

## 2026-06-29 17:26 EDT (bounds-to-constraints reformulation)

**Previous Commit**: `c617dd7`

**Summary**: Added an opt-in reformulation that rewrites variable bounds as
explicit affine constraints, plumbed through the whole generation stack as the
keyword `bounds_to_constraints` (default `false`, so the corpus is byte-identical
when unused). Also fixed a pre-existing soft-scope bug in
`scripts/generate_problem.jl` that silently dropped its `--feasible`/`--infeasible`,
`--seed=`, and output-file arguments.

**Details**:

- **New file `src/transforms.jl`** — home for post-`build_model` model
  reformulations, applied centrally in `generate_problem` (not per-variant),
  the same way JuMP's `relax_integrality` is. Exports `bounds_to_constraints!`.
- **`bounds_to_constraints!(model)`** — walks every variable and converts its
  bounds to affine rows: a fixed value becomes an equality row, an upper bound
  and a *nonzero* lower bound become inequality rows, and the corresponding
  variable bound is removed. A plain `x ≥ 0` nonnegativity lower bound is left as
  a variable bound (standard form), so only the "interesting" bounds are moved.
- **Wiring** — `bounds_to_constraints::Bool=false` keyword added to all four
  `generate_problem` methods, `generate_random_problem`, and `generate_dataset`
  (threaded through its candidate/materialize helpers alongside `relax_integer`).
  Applied *after* integrality relaxation so bounds introduced by relaxing
  integer/binary variables (e.g. `0 ≤ x ≤ 1`) are converted too. Recorded in the
  dataset `manifest.json` config and in verbose output.
- **CLI** — `--bounds-to-constraints` flag on both `scripts/generate_problem.jl`
  and `scripts/generate_lps.jl`.
- **Side effect (intended)** — converted bounds become genuine constraint rows,
  so they are now counted by
  `num_constraints(model; count_variable_in_set_constraints=false)`. This raises
  the recorded `num_constraints` per instance and therefore feeds into dataset
  size-matching and the quality filter's constraint-based thresholds
  (`min_constraints`, `max_iteration_ratio`).
- **Bug fix** — `scripts/generate_problem.jl` parsed its optional arguments in a
  top-level `for` loop whose assignments fell into Julia soft scope and became
  new locals, so `--feasible`/`--infeasible`/`--unknown`, `--seed=`, and the
  output-file positional were silently ignored. Added the required `global`
  declaration; these flags now take effect.
- **Tests** — new `Bounds to Constraints` testset: direct structural checks on
  `bounds_to_constraints!` (nonnegativity preserved, other bounds become the
  right number of rows, variable count unchanged), plus `generate_problem` and
  `generate_dataset` integration (constraint counts increase, manifest flag set).
  Suite now 2460 passing assertions (up from 2448); the 6 pre-existing
  variable-count-tolerance failures are unchanged and unrelated.

## 2026-06-20 19:10 EDT (address PR #19 review feedback)

**Previous Commit**: `f7a657f`

**Summary**: Addressed automated code-review feedback on PR #19 (gemini-code-assist
and chatgpt-codex). Five are small cleanups of redundant/non-idiomatic code in the
newly ported variants; one strengthens the `feasible` guarantee of
`vehicle_routing/cvrp` from LP-relaxation-only to genuine integer feasibility. No
behavior change for any sampled instance (the new CVRP guard is a strict no-op
across 3600 scanned instances and consumes no RNG, so the generated corpus and all
seeds are byte-identical). Test suite unchanged at 2217 passing assertions; the 6
pre-existing variable-count-tolerance failures (airline_crew, cutting_stock,
network_flow/standard, scheduling, supply_chain×2) are unrelated to these files.

### Changed

- **`vehicle_routing/cvrp`** — the `feasible` branch now certifies that customer
  demands partition into `≤ K` routes of capacity `Q` via a first-fit-decreasing
  bin-packing check (new internal `_ffd_bin_count`), raising `Q` if needed.
  Aggregate fleet capacity (`K·Q ≥ total_demand`) is necessary but not sufficient
  for the *integer* CVRP (`relax_integer=false`); the guard makes a concrete
  integer routing provably exist, so both the MIP and its LP relaxation are
  feasible (matching the integer-solution-construction approach already used by
  `assignment/workload_balance`). Verified: 100/100 feasible CVRP MIPs solve to
  OPTIMAL under HiGHS. In practice the log-normal demand sizing already kept all
  3600 scanned instances bin-packable, so `Q` is never actually raised — the guard
  is purely defensive. Docstring updated (feasibility no longer scoped to the LP
  relaxation only). Also removed a redundant `min(grid_size, grid_size)`.
- **`assignment/workload_balance`** — removed a duplicate `n_tasks` recomputation
  (the "fine-tune pass" was identical to the first assignment and a no-op); removed
  a redundant capacity rescale in the `infeasible` branch (`scale` was always
  `1.0`; the trailing `floor()` already guarantees the strict shortfall); switched
  the greedy-LPT worker pick to the idiomatic two-argument `argmin(w -> ..., cands)`
  (no temporary array).
- **`network_flow/generalized_flow`** — simplified the empty-arc conservation
  fallbacks from `AffExpr(0.0)` to `0.0` (a node with both in- and out-arcs empty
  is skipped, so at least one side is always an affine expression with variables).

## 2026-06-20 17:50 EDT (port 7 high-value variants from the branch review)

**Previous Commit**: `7c823a9`

**Summary**: Ported and upgraded seven high-value problem variants identified in
`docs/variant_branch_review.md` that had not yet been integrated into `main`,
adding one new category (`vehicle_routing`) and diversifying four single-variant
categories (`knapsack`, `network_flow`, `portfolio`, `assignment`) plus a classic
addition to `facility_location`. Each was reimplemented to the project quality bar
(scale-tiered realistic data, careful `target_variables` scaling, provable
feasibility) rather than copied — every reviewer-noted defect (degenerate LP
relaxations, no-op infeasibility handling) was fixed. The work was driven by a
dynamic multi-agent workflow (port → wire → verify → fix loop). Every new variant
was verified with HiGHS: across targets {100, 500} × seeds {0,1,2}, all
`feasible` instances solve to OPTIMAL, all `infeasible` instances solve to
INFEASIBLE *in the LP relaxation* (the framework relaxes integrality by default),
all `unknown` instances solve without error, variable counts land within ±25% of
target, and instances are reproducible. The full test suite goes from 1955 → 2217
passing assertions (+259 from the new variants, 37 each) with no new failures.
`list_problems()` grows from 49 to 56 registered `category/variant` pairs.

### Added

- **New category `vehicle_routing`** with variant `cvrp` — Capacitated Vehicle
  Routing Problem as a MIP using the Gavish–Graves **single-commodity-flow**
  subtour-elimination formulation. Binary arc variables `x[i,j]` plus continuous
  load variables `f[i,j]` coupled by `f[i,j] ≤ Q·x[i,j]`; load is sourced at the
  depot and conserved at customers, so the continuous relaxation is a genuine
  depot-anchored routing relaxation (fixing the degenerate per-vehicle-flow
  relaxation in the source branch). Depot near grid center, clustered customers,
  log-normal demands. `total_vars = 2·(N+1)·N`, so `N ≈ √(target/2)`. Infeasible
  via aggregate fleet shortfall `total_demand > K·Q` (survives relaxation).
- **`knapsack/multidimensional`** — true multi-constraint 0/1 knapsack with D=2–5
  correlated resource dimensions (weight/volume/budget/labor). Infeasibility is a
  structural covering floor (`Σx ≥ m`, where the `m` lightest items already
  overflow the tightest resource), replacing the source branch's no-op
  capacity-shrink. `n_items = target`.
- **`knapsack/bounded`** — bounded knapsack with integer per-item multiplicities
  `0 ≤ x_i ≤ u_i`. Infeasibility via a value floor above the exact bounded
  fractional-knapsack greedy optimum (provably unmet in the relaxation).
  `n_items = target`.
- **`network_flow/generalized_flow`** — generalized (lossy) min-cost flow with
  per-arc gain multipliers `g ∈ (0.85, 1.0]`; multiplicative conservation
  (`Σ_in g·f = Σ_out f`) and a delivered-at-sink demand. Objective minimizes
  routing cost (not the degenerate source-outflow maximization of the branch).
  Feasibility guaranteed by sizing the known backbone path; infeasibility by
  capping post-gain sink inflow below demand. `vars = #arcs = target`.
- **`portfolio/tracking_error`** — index-tracking / enhanced-indexing portfolio:
  maximize expected return subject to a tracking-error budget (MAD linearization
  `u_s ≥ ±active_return`), full investment, long-only position limits, and
  two-sided sector-deviation bands around the benchmark. Pure LP. The benchmark is
  feasible by construction; infeasibility via `Σ max_position < 1`.
  `vars = n_assets + n_scenarios`. `cvar` remains the category default.
- **`assignment/workload_balance`** — minimax / makespan task-to-worker
  assignment (`min L` with per-worker load `≤ L`) under per-worker capacities and
  skill eligibility; distinct from `assignment/standard` (min-cost matching) and
  `load_balancing` (network link utilization). Feasibility guaranteed by a greedy
  LPT construction; infeasibility via capacity pigeonhole `Σ cap_w < total_load`.
  `vars = W·T + 1`.
- **`facility_location/p_median`** — classic p-median: open exactly `p`
  facilities, assign each customer to one with demand-weighted distance, using the
  tight disaggregated linking `y[w,c] ≤ z[w]` (fixing the degenerate continuous
  assignment in the branch). Count-based service capacity enables infeasibility
  via `p·count_cap < C`. `vars = F·(C+1)`.

### Changed

- Wired the new variants into their category entry points
  (`knapsack.jl`, `network_flow.jl`, `portfolio.jl`, `assignment.jl`,
  `facility_location.jl`) and added the `vehicle_routing` category include to
  `src/SyntheticLPs.jl`.
- Updated `test/runtests.jl`: the `list_variants(:portfolio)` assertion now expects
  `[:cvar, :tracking_error]`.
- Updated `README.md` and `CLAUDE.md` category listings (28 → 29 categories),
  the new multi-variant annotations, and the LP/MIP model-class notes.

### Fixed (review follow-up — edge-case sizing)

- **`portfolio/tracking_error`**: the per-asset position-limit range
  `Uniform(max(2/n_assets, 0.02), min(0.4, 6/n_assets))` inverted (lower > upper)
  for `n_assets == 5` (target ≲ 24) and `n_assets > 300` (target ≳ 1505, within
  the default `var_max = 2000`), throwing a `DomainError` before any model was
  built. The cap is now expressed as a multiple of equal-weight, clamped so the
  lower bound is always strictly below the upper bound across all supported sizes.
  Feasibility is unchanged (the cap is still raised to `>= benchmark`).
- **`facility_location/p_median`**: for the minimum customer count `C == 2` (tiny
  targets, e.g. target ≤ 9 giving `C = 2`), the infeasible-mode `p` shrink floored
  at `p_lo = 2`, leaving `p == C`; combined with the `count_cap >= 1` clamp this
  gave `p*count_cap = 2 = C` (not `< C`), so an `infeasible` request silently
  produced a feasible model. `p` may now shrink to 1 (the 1-median) so the
  `p*count_cap < C` pigeonhole holds even at `C = 2`.
- Verified across targets from 6 to 2000 (440 HiGHS solves): all `feasible` →
  OPTIMAL, all `infeasible` → INFEASIBLE, no generation errors.

### Known issues (pre-existing, not introduced here)

- Six assertions fail at `test/runtests.jl:33` (the ±25% variable-count tolerance
  with fixed seed 0) for `airline_crew/standard`, `cutting_stock/standard`,
  `network_flow/standard`, `scheduling/standard`, and `supply_chain/standard`.
  Verified identical on clean `origin/main` (7c823a9), so these are pre-existing
  sizing-tolerance issues in untouched generators, out of scope for this change.

## 2026-06-20 16:50 EDT (fix nurse_scheduling feasibility & realism)

**Previous Commit**: `4179085`

**Summary**: Addressed two PR-review findings (PR #18) on
`nurse_scheduling/standard` that could make `feasible`-requested instances
solve as INFEASIBLE, and fixed an underlying dimension-selection bug that was
masking them by making every nurse instance degenerate. After these changes a
HiGHS sweep over targets {50,80,100,200,500,800,1500} × seeds 0–40 produces
287/287 feasible instances solving to OPTIMAL and 287/287 infeasible instances
solving to INFEASIBLE.

### Fixed

- **Realistic instance dimensions** (`select_nurse_dimensions`). The old greedy
  search returned the first `(nurses, days, shifts)` within 10% of target, which
  was always `days=1, shifts=1` — so every generated instance was a single-day,
  single-shift assignment with **none** of the night/weekend/consecutive-day/
  rest structure (and the two bugs below could never surface). It now scales the
  horizon and shift count with problem size (7–28 days, 2–3 shifts), so weekends
  and a night shift are always present. Variable counts stay within ~12% of
  target (≤25% test tolerance).
- **Coverage demand never exceeds achievable staffing** (`finalize_nurse_demand`,
  PR comment 3447108887). A night slot with no available night-qualified nurse
  gets zero heuristic coverage, but demand was still forced to ≥1, making the
  coverage constraint unsatisfiable (only night-qualified nurses can staff nights
  in `build_model`). Demand for an unstaffable slot is now 0, preserving the
  invariant `demand ≤ achievable coverage`.
- **Night-qualified availability on night slots** (`build_nurse_availability`).
  The availability repair guaranteed ≥2 arbitrary nurses per slot but no
  *night-qualified* nurse on night slots; it now also guarantees at least one
  night-qualified nurse is available for each night slot, so most night demand is
  staffable (residual unstaffable slots fall back to demand 0).
- **Post-night rest matches between roster heuristic and model** (PR comment
  3447108889). The feasible-roster heuristic only blocked shift 1 after a night,
  while `build_model` blocked shifts 1 *and* 2 for `n_shifts ≥ 3`; the heuristic
  also blocked one fewer day than the model (an off-by-one in the cooldown
  counter). Both now use a shared `nurse_early_shift_indices` helper and an
  absolute block window (days `d+1..d+rest`), so the heuristic roster always
  satisfies the model's rest rule.
- **Correct rest-constraint linearization** (`build_model`). The rest rule summed
  the entire post-night window into a single `≤ 1` constraint, which (for
  `rest ≥ 2`) also forbade working two early shifts in the window *with no night
  shift at all* — far stronger than the intended "no early shift after a night."
  It is now encoded pairwise (`x[night] + x[early] ≤ 1` per slot), matching the
  documented rule and the roster heuristic.

## 2026-06-20 08:23 EDT (port high-quality variants from old branches)

**Previous Commit**: `1ebc1c2`

**Summary**: Reviewed all pre-variant-system branches (`claude/*`, `codex/*`)
that introduced problem variants, evaluated each variant independently for
formulation correctness and data quality, and ported the 25 highest-quality
ones into the new category/variant system (with fixes). The package grows from
24 to **28 categories** and from 24 to **49 registered variants**. Deferred/
rejected variants (110) are catalogued in `docs/variant_branch_review.md` for
future, higher-quality reimplementation.

Every ported variant is self-contained (struct + constructor + deterministic
`build_model` + `register_variant`), documented with full docstrings, and
verified by `test/runtests.jl` (structure, ±25% variable-count scaling,
reproducibility) **and** a HiGHS solve smoke-test (feasible→solvable,
infeasible→infeasible across seeds 0–2).

### Added

- **4 new categories**:
  - `bin_packing/standard` — minimize bins used, realistic item sizes, true
    category-conflict constraints (per-bin category-presence binaries).
  - `nurse_scheduling/standard` — nurse rostering with skill mix, shift
    coverage, and realistic labor-contract rules (ported from
    `codex/add-nurse-scheduling`, score 8).
  - `job_shop_scheduling/standard` — disjunctive job-shop with machine
    no-overlap and **weighted tardiness** (soft due dates, reworked from the
    source's hard-deadline formulation to remove false infeasibility).
  - `unit_commitment/standard` — thermal+renewable unit commitment (LP
    relaxation) with ramping, startup/shutdown, and reserves
    (ported from `codex/implement-unit-commitment`, score 8).
- **21 new variants** in existing categories:
  - transportation: `balanced`, `capacitated`, `transshipment`, `emission_constrained`
  - energy: `ramping`, `reserves`, `storage`, `transmission`
  - inventory: `lot_sizing`, `multi_item`, `multi_echelon`
  - supply_chain: `single_source`, `carbon`, `multi_product`
  - blending: `equipment_batches`, `multi_product`
  - cutting_stock: `setup_cost`, `due_dates`
  - diet_problem: `nutrient_bounds`, `food_groups`
  - facility_location: `two_echelon` (two-echelon FL with discrete warehouse
    sizing; ported from `add-transportation-generators/warehouse_location`).
- **`docs/variant_branch_review.md`**: full review record — methodology, the 25
  ported variants, overlap-resolution decisions, and the 110 deferred/rejected
  variants with scores and reasons.

### Fixed (applied to ported variants during the port)

- **Variable-count scaling**: many source variants ignored extra variable sets
  when sizing (e.g. reserve/storage/echelon/transfer/setup/lot variables),
  overshooting `target_variables`. Every ported variant now sizes its dimensions
  from the full variable-count formula and stays within the test's ±25% bound,
  including small targets (added a `:tiny` band to `unit_commitment` and lowered
  the small-target minimums for `energy/reserves` and `energy/storage`).
- **Feasibility reliability**: reworked several `infeasible` constructions that
  were only probabilistically infeasible (e.g. `blending/multi_product`,
  `facility_location/two_echelon`, `cutting_stock/due_dates`) into deterministic
  contradictions with margin; `unknown` no longer force-infeasibilizes.
- **Degenerate objectives**: added an inventory holding cost to
  `cutting_stock/due_dates`, a terminal state-of-charge floor to `energy/storage`,
  reserve provision cost to `energy/reserves`, and removed the no-op emissions
  constraint carried over into the energy variants.
- **Formulation correctness**: fixed `bin_packing` category-conflict semantics
  (was forbidding two same-category items in a bin), restricted
  `inventory/multi_echelon` transfers to a star topology, and scaled
  `cutting_stock/setup_cost` setup costs relative to material value.

### Fixed (post-port review follow-up)

Five correctness defects surfaced by code review of the ported variants, all
confirmed (three by reproducing the cited seeds, two by formulation analysis)
and fixed:

- **`supply_chain/multi_product`** (P1, feasibility mislabel): feasible instances
  could be infeasible because capacities were sized against `total_demand`, while
  the model enforces the larger jittered per-product demand total. Now sized
  against realized per-product demand with a **connectivity-aware** per-facility
  guarantee (each facility can absorb all its linked customers' product demand).
  Verified feasible across 40 seeds × 3 sizes (incl. the reported `(50, feasible, 7065)`).
- **`inventory/multi_item`** (P1, feasibility mislabel): `infeasible` set capacity
  below the single-period **peak** weighted load, which prebuild-and-carry can
  satisfy. Now based on the binding **cumulative** rate
  `max_t (cumulative_weighted_demand[t] − weighted_initial_inventory) / t`, a true
  no-backlog contradiction (incl. the reported `(20, infeasible, 424)`).
- **`energy/reserves`** (P2, double-counting): spinning and non-spinning reserve
  each independently drew on `capacity − x`, certifying up to twice the real
  headroom. Replaced with a single shared-headroom constraint
  `x + spin + nonspin ≤ capacity` (matching the constructor's own feasible sizing).
- **`diet_problem/nutrient_bounds`** (P2, crash): `primary_count` could exceed
  `n_foods` for very small targets, throwing `BoundsError` (e.g. size 2). Clamped
  to the available foods.
- **`cutting_stock/setup_cost`** (P2, cuts off valid solutions): the per-pattern
  big-M used `min_i ceil(demand_i / pattern_i)`, forbidding running a pattern
  enough times to serve a high-demand piece when it co-produces a low-demand one
  (overproduction is legitimate). Changed to `max_i` (a valid bound preserving the
  optimum).

### Fixed (second review follow-up: scaling, diversity, framing)

- **`bin_packing/standard`** (variable-count scaling): the bin count is set by the
  actual packing requirement, not a free dimension, so the old sizing overshot the
  target by ~1.5–2× (e.g. target 100 → ~175–250). Now sizes `n_items` from the
  estimated packing density so the count tracks the target across the whole range
  (medians now within a few % at 100/500/2000; previously up to ~64% off at 2000).
- **`energy/ramping`, `energy/reserves`, `energy/storage`** (fleet diversity +
  large-target scaling): the 7-type generator catalogue was sampled without
  replacement, hard-capping the fleet at ~7 units, so large instances scaled only
  by time periods and were low-diversity. The fleet now uses distinct types first,
  then repeats them as additional units with unique names and jittered
  techno-economic attributes (realistic multi-unit fleets); large-band caps raised.
  n_sources now scales with the target (e.g. ramping reaches ~18 units at large
  targets) and large targets no longer badly undershoot. Also fixed an
  `energy/storage` sizing-loop bug where an over-cap initial period estimate made
  the loop break before scaling the fleet.
- **`transportation/transshipment`** (binding constraints): hub-leg arc capacities
  were ~0.8–1.4× total demand each, so a single hub could absorb everything and the
  constraints almost never bound. Resized to ≈ total_demand / n_hubs so hub legs
  are genuinely binding; feasibility is unaffected (direct arcs remain an uncapped
  fallback).
- **`cutting_stock/due_dates`** (degenerate columns): replaced the
  duplicate-allowing pattern padding with distinct reduced-yield single-piece
  columns (valid, non-duplicate) and widened genuine-pattern search, reducing
  benchmark-inflation filler.
- **Model-class framing** (docs): `nurse_scheduling` and `unit_commitment`
  docstrings now state explicitly that they are LP relaxations of integer models
  (fractional rosters / commitment), and CLAUDE.md documents that the corpus
  intentionally mixes pure LPs, MIPs, and LP relaxations.

### Fixed (third review follow-up: infeasible sizing + repeated technologies)

- **`bin_packing/standard` infeasible instances** now preserve target-sized
  dimensions. Instead of shrinking the bin count (which badly undershot
  `target_variables`) or adding an extra item, infeasibility is created by an
  aggregate capacity contradiction: total item volume is forced above
  `n_bins * bin_capacity`. This survives LP relaxation and keeps the variable
  count unchanged.
- **`energy/ramping`, `energy/reserves`, `energy/storage` repeated units** now
  carry both a unique unit name (`coal_2`) and a base technology (`coal`) during
  construction. Technology-dependent capacity-share and ramp-rate distributions
  use the base technology, so repeated nuclear/coal units no longer accidentally
  behave like generic flexible gas units.

### Changed

- **`test/runtests.jl`**: updated the variant-interface assertions for the new
  multi-variant transportation category, and the dataset-generation assertion to
  reflect that a bare category selector now samples across all its variants.

### Notes

- Deferred (not ported): both vehicle-routing implementations and last-mile
  delivery (degenerate LP relaxations — the depot does not anchor routes),
  `production_planning/multi_period_inventory` (sizing bug + overlap), and ~100
  lower-scoring or duplicate variants. See `docs/variant_branch_review.md`.
- Pre-existing test failures unrelated to this change remain in 5 `*/standard`
  generators (airline_crew, cutting_stock, network_flow, scheduling,
  supply_chain) whose variable counts fall just outside ±25% at some targets;
  these fail identically on a clean `main` and were left untouched.

## 2026-06-20 21:12 UTC (PR #17 review feedback)

**Previous Commit**: `9ace305`

**Summary**: Addressed review feedback on the new stochastic / power-flow /
regression / revenue-management generators (PR #17): removed dead feasibility
branches that diverged from the codebase convention, and made the DC-OPF feasible
witness scale to large networks.

### Fixed

- **Dead `unknown` feasibility branches** (`energy/dc_opf.jl`,
  `regression/regression.jl`, `revenue_management/standard.jl`,
  `stochastic_program/standard.jl`): each generator resolves an `unknown` request
  to `feasible`/`infeasible` (70/30) at the top, so the trailing `else` branch
  handling `unknown` was unreachable. Removed it in all four files, matching the
  convention used by every other generator (resolve `unknown`, then branch only on
  `feasible`/`infeasible`). Also corrected the `regression` data-generator docstring,
  which documented the now-removed branch ("`side_rhs` randomized across the
  boundary") — `unknown` is resolved at random to `feasible` or `infeasible`.

### Changed

- **`energy/dc_opf.jl`**: the feasible-case DC-power-flow witness now assembles the
  reduced network Laplacian as a **sparse** matrix (`SparseArrays`) instead of a
  dense `B × B` matrix. The grid topology is sparse (`n_lines` is a small multiple
  of `n_buses`), so the dense build + factorization was cubic in `n_buses` and made
  data generation the bottleneck for large instances; the sparse solve keeps feasible
  generation cheap at scale. Verified that feasible instances (including
  `target_variables = 3000`) still solve to `OPTIMAL` and infeasible requests remain
  `INFEASIBLE`.
- **`Project.toml`** / **`Manifest.toml`**: added the `SparseArrays` standard-library
  dependency.

## 2026-06-20 16:32 UTC (real-world coverage: new LP archetypes)

**Previous Commit**: `7c823a9`

**Summary**: Broadened the realized distribution of generated LPs to fill the
biggest gaps identified in a coverage review. The existing 24 categories
over-indexed on *sparse, combinatorial* OR (network flow, allocation, blending)
and under-represented four important real-world LP families: stochastic /
decomposable, physics-based, *dense* statistical, and revenue-management LPs.
This change adds three new categories and a new variant of `energy` to address
each, taking the package to 31 categories.

### Added

- **`stochastic_program/standard`** (`src/problem_types/stochastic_program/`): a
  two-stage stochastic linear program with recourse (stochastic
  capacity/distribution planning). First-stage capacity decisions are coupled to
  `S` independent scenario blocks of shipment + penalized-shortfall variables,
  producing the canonical **dual block-angular** structure that decomposition
  methods (L-shaped / Benders) target — a structure absent from the prior set.
  Complete recourse keeps the second stage always feasible; feasibility is
  controlled entirely through a first-stage resource budget vs. minimum committed
  capacity. Variables: `n_facilities + S·(n_facilities·n_customers + n_customers)`.

- **`energy/dc_opf`** (`src/problem_types/energy/dc_opf.jl`): DC optimal power
  flow / economic dispatch over a transmission network. Couples a flow network to
  physical bus-angle variables through susceptance-weighted (non-±1, non-unimodular)
  flow-definition rows, with nodal power balance, thermal line limits, and
  generator bounds. The feasible case is guaranteed by constructing an explicit
  DC-power-flow witness (a reduced network-Laplacian solve) and sizing line limits
  to accommodate it; infeasibility is forced via generation–load imbalance.
  Variables: `n_generators + n_buses + n_lines`. Adds `:dc_opf` as a further
  `energy` variant (alongside `:standard` and the others); the entry point gained a
  category-level `register_category` description and `:standard` remains the default.

- **`regression`** category with three variants — `:lad` (default), `:quantile`,
  and `:chebyshev` (`src/problem_types/regression/`): least-absolute-deviations
  (L1), quantile (pinball-loss), and Chebyshev (L∞/minimax) regression as LPs.
  These introduce **dense data-matrix LPs** — a numerical/structural profile that
  diversifies the otherwise sparse, combinatorial test set. The three variants
  share a common data generator (`generate_regression_data`) and differ only in
  the loss being linearized; feasibility is controlled by a coefficient side
  constraint relative to per-coefficient box bounds.

- **`revenue_management/standard`** (`src/problem_types/revenue_management/`):
  network revenue management deterministic LP (the "DLP"). Allocates perishable
  resource (e.g. flight-leg) capacity to fare-bearing products over a
  resource–product incidence matrix to maximize revenue; the optimal duals are the
  bid prices used in RM control. Feasibility is controlled through contractual
  commitments (minimum acceptances) vs. resource capacity. Variables: `n_products`.

### Changed

- **`src/SyntheticLPs.jl`**: registered the three new category entry points in the
  include list.
- **`src/problem_types/energy/energy.jl`**: added an explicit `register_category`
  description and an `include` for `dc_opf.jl`.
- **README.md / CLAUDE.md**: updated the category listings and counts (28 → 31),
  and documented the multi-variant `energy` and `regression` categories.

## 2026-06-19 22:47 EDT (PR #16 review feedback)

**Previous Commit**: `fc0ba22`

**Summary**: Addressed review feedback on the hierarchical category/variant
system (PR #16): made the registration API order-independent, added a string
selector overload for `generate_problem`, and tightened input validation in the
problem-generation script.

### Changed

- **`register_category`** (`src/SyntheticLPs.jl`): now always applies the supplied
  description to the `CategorySpec`, even when the category was already created
  lazily by `register_variant`. Previously the description was only set on first
  insertion (via `get!`), so an explicit `register_category` call placed after the
  variant includes was silently ignored. Registration order no longer matters.

### Added

- **String selector for `generate_problem`** (`src/SyntheticLPs.jl`): added a
  `generate_problem(ref::AbstractString, ...)` overload that parses a `"category"`
  or `"category/variant"` string via `ProblemVariant`. This makes `generate_problem`
  consistent with the rest of the string-accepting API (`ProblemVariant`,
  `get_problem_type`); previously the string form raised a `MethodError`. Covered
  by a new assertion in the "Variant Interface" testset.

### Fixed

- **`scripts/generate_problem.jl`**: a problem argument with more than one slash
  (e.g. `"category/variant/extra"`) previously fell back silently to the category's
  default variant. The script now validates `length(parts) <= 2` and errors with a
  clear message otherwise.

## 2026-06-19 21:57 EDT (hierarchical problem variant system)

**Previous Commit**: `d350e8e`

**Summary**: Introduced a first-class two-level problem hierarchy — a **category**
(the former "problem type", e.g. `:transportation`) groups one or more
**variants** (concrete generators with their own data generation and model
formulation, e.g. `:standard`). Each of the 24 problem types was migrated from a
single flat file into a folder with a thin category entry point plus one file per
variant, so new formulations can be added as separate files rather than as
branching logic inside one large file. No new variants were added: every category
keeps its existing single formulation (named `:standard`, except `portfolio`'s
CVaR formulation which is `:cvar`). Breaking change (research package, no
back-compat shims).

### Added

- **`ProblemVariant` identifier** (`src/SyntheticLPs.jl`): a `category/variant`
  reference used throughout the package. Constructible from `(category, variant)`
  symbols, a bare category symbol (→ the category's default variant), or a
  `"category"`/`"category/variant"` string; prints as `category/variant`.
- **Two-level registry** (`src/SyntheticLPs.jl`): `LP_REGISTRY::Dict{Symbol,CategorySpec}`
  with `CategorySpec`/`VariantSpec`. New registration API `register_category(:cat,
  desc)` and `register_variant(:cat, :variant, Type, desc; default=false)`. A
  variant lazily creates its category (using the variant's description) so
  single-variant categories need no explicit `register_category` call; the first
  registered variant is the default unless `default=true` designates another.
- **Introspection**: `list_categories()`, `list_variants(category)`,
  `list_problems()` (all `category/variant` pairs), and `problem_info(category,
  variant)` for variant-level metadata.
- **Variant selection** in `generate_problem`: accepts a category symbol with an
  optional `variant=` keyword, a `ProblemVariant`, or a `"category/variant"`
  string (via the scripts). `scripts/generate_problem.jl` accepts
  `category/variant` and its `list` shows variants; `scripts/generate_lps.jl`
  `--problem-types` accepts categories (expand to all their variants) and explicit
  `category/variant` references.

### Changed

- **Problem type layout** (`src/problem_types/`): each `<name>.jl` became
  `<name>/{<name>.jl (entry point), <variant>.jl (variant)}`; `register_problem`
  was replaced by `register_variant`. Include paths in `src/SyntheticLPs.jl`
  updated accordingly.
- **`list_problem_types()`** now aliases `list_categories()` (still returns a
  `Vector{Symbol}` of categories).
- **`generate_random_problem`** now returns the selected `ProblemVariant` as its
  second value (was a category `Symbol`); sampling is uniform over all registered
  variants.
- **Dataset generation** (`src/dataset.jl`): `GeneratedInstance` gained a
  `variant::Symbol` field (`problem_type` still holds the category).
  `resolve_problem_types` returns `Vector{ProblemVariant}`, expanding a selected
  category to all its variants (sorted) and accepting explicit `category/variant`
  selectors. Sampling is over variants; `match_size_by_type` groups quotas by
  category. Instance filenames now include the variant
  (`<category>_<variant>_v<n>_<idx>.<ext>`), and `manifest.json` records a
  per-instance `variant` (with `problem_types` listed as `category/variant`).

### Removed

- **`register_problem`**: superseded by `register_category` + `register_variant`.

**Previous Commit**: `7d25612`

### Fixed

- **Size-distribution truncation** (`_resolve_size_distribution`, `src/dataset.jl`): a user-supplied `size_distribution` is now truncated to `lower = 2` whenever its support reaches below 2 (finite lower bound `< 2`, e.g. `Uniform(0, 100)`/`Exponential`, in addition to the existing unbounded-below case). Previously only `-Inf`-lower distributions were truncated, so finite-support distributions could yield sizes that round toward 0/1. Also added an explicit error when the distribution's upper bound is `< 2`.
- **Duplicate problem types** (`resolve_problem_types`, `src/dataset.jl`): the requested `problem_types` are now de-duplicated via `unique`. Duplicates previously inflated `length(types)` in `_type_quotas` while collapsing in the per-type `Dict`, corrupting the per-type quota math under `match_size_by_type`.
- **Degeneracy check with zero constraints** (`check_quality`, `src/dataset.jl`): the excessive-iterations (degenerate) check is now guarded by `n_cons > 0`. With `min_constraints = 0`, `n_cons` could be 0, making `max_iters = 0` and rejecting every nonzero iteration count as degenerate.

### Changed

- **Manifest reproducibility** (`src/dataset.jl`): `manifest.json` now records the `quality_criteria` used to filter the dataset, via a new `_jsonable(::QualityCriteria)` method. This makes a filtered dataset fully documented/reproducible from the manifest alone.

## 2026-06-19 (code review fixes)

**Previous Commit**: `e3aaec0`

### Fixed

- **Reproducibility**: `resolve_problem_types` now sorts the default "all types" selection (`src/dataset.jl`). Previously it returned `list_problem_types()` in `Dict` key order, which the seeded RNG consumes positionally — so a seeded dataset built with the default `problem_types` was only reproducible within a single process/Julia version, contradicting the documented seed-reproducibility guarantee. Explicit `problem_types` selections still preserve caller order.
- **Interrupt handling**: `_attempt_candidate` now re-throws `InterruptException` instead of swallowing it in its catch-all (`src/dataset.jl`). Ctrl-C during generation now aborts the run rather than being counted as a generation failure and retried until the attempt budget is exhausted.

## 2026-06-19 13:15 EDT

### Feature: Built-in Batch Dataset Generation API

**Previous Commit**: `75882de`

**Summary**: Promoted the standalone batch-generation script (`tmp/generate_lps.jl`) into a first-class, tested library API inside the package. Datasets of LP instances can now be generated directly via `generate_dataset`, with `scripts/generate_lps.jl` reduced to a thin command-line wrapper.

### Added

- **`src/dataset.jl`** — new in-package module providing:
  - **`generate_dataset(; kwargs...)`**: samples problem types and target variable counts (truncated normal over `[var_min, var_max]`), builds each model, optionally quality-filters it, and writes instance files. Returns a `Vector{GeneratedInstance}` of metadata. Fully reproducible from a non-zero `seed` (all randomness flows from one seeded `MersenneTwister`: type choice, size, and per-instance seed).
  - **`GeneratedInstance`**: metadata struct (index, problem type, target/actual variables, constraints, per-instance seed, feasibility status, filename, simplex iterations, solve time).
  - **`QualityCriteria`** (keyword struct: `solve_timeout`, `min_constraints`, `min_iterations`, `max_iteration_ratio`) and **`QualityResult`**.
  - **`check_quality(model, optimizer; criteria, feasible_only, optimizer_attributes)`**: solves an instance and judges it as a test/training instance (rejects too-few-constraints, infeasible-when-feasible-only, unbounded, timeout, numerical error, `ALMOST_OPTIMAL`, trivially-solved, and degenerate cases).
  - **`manifest.json`** output: records the run config plus per-instance metadata alongside the generated files (disable with `write_manifest=false`).
- New exports: `generate_dataset`, `GeneratedInstance`, `QualityCriteria`, `QualityResult`, `check_quality`.
- Added `JSON` to the module imports (already a package dependency) for manifest writing.
- New **`Dataset Generation`** testset in `test/runtests.jl` covering basic generation, reproducibility, problem-type restriction, invalid-type rejection, the `quality_filter`-without-optimizer error, file/manifest output, and manifest suppression.

### Changed

- **Solver-agnostic design**: the package no longer hard-codes HiGHS. Quality filtering requires the caller to pass an `optimizer` (and optional `optimizer_attributes`). `scripts/generate_lps.jl` supplies `HiGHS.Optimizer` with `"solver" => "simplex"`.
- **`scripts/generate_lps.jl`**: rewritten as a thin argument-parsing wrapper that delegates to `generate_dataset`. New flags: `--file-format` (output extension, e.g. `mps`/`lp`) and `--no-manifest`. Behavior of existing flags is preserved.
- README and CLAUDE.md document the new dataset API and CLI usage.

---

## 2026-06-18

### Docs: Self-Contained HTML Generator Explainer

**Previous Commit**: `75882de`
**Datetime**: 2026-06-18 09:53 -0400

**Summary**: Added a single-file, fully offline HTML explainer that builds on the per-generator markdown under `docs/`. It presents a high-level, family-grouped map of all 24 problem generators alongside the full formulation, sizing rules, and feasibility tricks for each. Generated by a reproducible build script.

### Added

- **`docs/explainer.html`**: Self-contained explainer (no external dependencies — CSS and a MathJax SVG bundle are inlined, so LaTeX renders offline with no font files). Single-page app with hash routing: a landing "Overview" with the shared-contract summary (target_variables, feasibility_status, seed, relax_integer), a feasibility-status legend, and a card grid of generators grouped into seven families (Network & Routing, Facility & Supply Chain, Blending & Diet, Production & Planning, Assignment & Scheduling, Selection & Finance, Land & Agriculture). Each generator has a detail view with the original six sections (Overview, Generator Data and Sizing, LP Formulation, Feasibility Controls, Model Characteristics, Practical Notes), color-coded family accents, objective/variable-class chips, a live sidebar filter, and mobile support.
- **`scripts/build_explainer.py`**: Reproducible build script that parses the `docs/*.md` pages, converts the tailored markdown subset (headings, paragraphs, ordered/unordered/nested lists, GFM tables, ```text formula blocks, and ```math LaTeX blocks) to HTML, and assembles the page with curated per-generator metadata (family, objective sense, variable class, tagline). Re-run with `python3 scripts/build_explainer.py`. Expects a MathJax `tex-svg` bundle at `/tmp/mathjax-tex-svg.js` to inline.

### Notes

- The detail pages render `docs/` content verbatim; family grouping and the objective/variable-class chips are curated in the build script's `META` table and may need updating when generators are added or reclassified.

## 2026-03-23

### Feature: Quality Filter for Batch LP Generation

**Previous Commit**: `28f882c`

**Summary**: Added a `--quality-filter` (`-q`) flag to `scripts/generate_lps.jl` that solves each generated LP instance with HiGHS simplex and filters out poor-quality test instances. The script retries generation (up to `--max-retries` × n attempts) to reach the requested problem count.

### Added

- **`--quality-filter` / `-q`**: Enables solve-and-filter pipeline. Each instance is solved with HiGHS simplex before being written to disk.
- **Filter criteria** (rejects instances that are):
  - Too few constraints (`--min-constraints`, default 5)
  - Infeasible (only when `--feasible-only` is also set)
  - Unbounded
  - Timed out (`--solve-timeout`, default 30s) or hit numerical errors
  - Nearly optimal (ALMOST_OPTIMAL status — indicates numerical conditioning issues)
  - Trivially solved / solved in phase 1 only (simplex iterations ≤ `--min-iterations`, default 3)
  - Degenerate (simplex iterations > `--max-iteration-ratio` × constraint count, default 100×)
- **`--max-retries`**: Controls total attempt budget as a multiplier of requested count (default 10)
- **Filter statistics**: Summary output shows counts of rejected instances broken down by reason

---

### Fix: Land Use Problem Generator Feasibility Guarantee

**Previous Commit**: `dfba903`

**Summary**: Fixed a bug where ~17.3% of land use problems generated with `feasible` status were actually infeasible. The root cause was that the witness assignment constructed during feasibility enforcement could violate adjacency constraints and minimum zoning requirements, but resource capacities were tightened around this invalid witness without verification.

### Fixed

- **Adjacency violations in remainder assignment**: When assigning unassigned parcels, the fallback path (when all allowed types conflict with adjacency) ignored adjacency constraints entirely, assigning residential next to industrial. The adjacency edges remained in the model, making it infeasible. Fix: after witness construction, scan for residential-industrial adjacency violations and prune offending edges from the adjacency matrix.
- **Incomplete minimum zoning fulfillment**: The type-2 (Commercial) assignment could fail when all parcels were consumed by types 1 and 3, with swap logic unable to find replacements (it only searched unassigned parcels). Fix: after witness construction, verify minimum counts are met; attempt swaps from over-represented types first, then reduce minimums to actual counts as a last resort.

### Validation

- 0/500 feasible-requested problems are infeasible (MIP), down from ~17.3%
- 0/300 feasible-requested problems are infeasible (LP relaxation)
- 0/300 infeasible-requested problems are accidentally feasible

## 2026-03-22

### Redesign: Portfolio Problem Generator (CVaR with Institutional Constraints)

**Previous Commit**: `d91324d`

**Summary**: Complete rewrite of the portfolio problem generator. The old generator was degenerate — only 2-3 constraints regardless of variable count, with 39.2% of problems solving in ≤2 simplex iterations. Replaced with a CVaR (Conditional Value-at-Risk) portfolio optimization model with rich institutional-grade constraints.

### Changed

- **`PortfolioProblem`**: Completely redesigned from a simple risk-budget model to a CVaR portfolio optimization with:
  - **CVaR risk measure**: Scenario-based linearization (Rockafellar-Uryasev) creating n_scenarios constraints that scale with problem size
  - **Sector exposure limits**: Maximum allocation per industry sector
  - **Region exposure limits**: Maximum allocation per geographic region
  - **Asset class bounds**: Min/max allocation per asset class (equities, bonds, alternatives)
  - **Factor exposure constraints**: Upper/lower bounds on risk factor exposures (beta, size, value, etc.)
  - **Position size limits**: Per-asset concentration caps
  - **Turnover constraints**: L1-norm turnover limit from benchmark portfolio via buy/sell decomposition
  - **Factor model for returns**: Correlated scenario returns via multi-factor model with sector-linked loadings

### Performance Comparison

| Metric | Old Generator | New Generator |
|---|---|---|
| Constraints (100 vars) | 2-3 | ~204 |
| Constraints (500 vars) | 2-3 | ~931 |
| Trivial solves (≤2 iters) | 39.2% | 0% |
| Median iterations (100 vars) | ~2 | ~38 |
| Median iterations (500 vars) | ~2 | ~177 |

### Feasibility Handling

- **Feasible**: Constructs a reference portfolio from benchmark weights and widens all constraints to accommodate it with randomized slack
- **Infeasible** (4 modes): (1) impossibly tight CVaR limit, (2) asset class lower bounds summing > 1, (3) position limits summing < 1, (4) near-zero turnover with conflicting sector caps
- **Unknown**: 70/30 feasible/infeasible split

### Files Modified

- `src/problem_types/portfolio.jl` — complete rewrite

---

### Bug Fixes: Feasibility Handling and Batch Generation Script

**Previous Commit**: `b679ada`

**Summary**: Fixed feasibility handling in 6 problem generators that previously ignored the `feasibility_status` parameter entirely, and fixed bugs in `scripts/generate_lps.jl` batch generation script.

### Fixed

- **`generate_lps.jl` seed handling**: When no `--seed` was provided (default `seed=0`), all problems received `problem_seed=0`, causing every instance of the same type with the same target variables to produce identical LPs. Now each problem gets a unique seed from the script's RNG regardless of whether `--seed` is specified.
- **`generate_lps.jl` error reporting**: Generation failures were silently swallowed unless `--verbose` was used. Now always emits warnings for failures and prints a warning line in the summary when any problems fail.
- **`ProductionPlanningProblem`**: Added `min_production` field and feasibility handling. For `infeasible` status, sets minimum production levels that exceed resource capacity.
- **`PortfolioProblem`**: Added `min_total_return` field and feasibility handling. For `infeasible` status, sets a minimum return constraint above what's achievable under risk constraints.
- **`ProjectSelectionProblem`**: Added `min_selected` field and feasibility handling. For `infeasible` status, requires selecting more projects than the budget allows.
- **`LoadBalancingProblem`**: Added `max_utilization` field and feasibility handling. For `infeasible` status, caps maximum utilization below what's required to satisfy demands.
- **`KnapsackProblem`**: Added `min_value` field and proper feasibility handling. For `infeasible` status, requires more total value than achievable under capacity constraint. Previously had a TODO comment about infeasibility.
- **`NetworkFlowProblem`**: Added feasibility handling for both `feasible` and `infeasible` statuses. For `feasible`, ensures target flow is within achievable range. For `infeasible`, sets target flow above max flow capacity.

### Details

Each of the 6 fixed generators previously accepted the `feasibility_status` parameter but ignored it, always producing problems with the same random feasibility regardless of what was requested. With these fixes:
- `feasible` status guarantees a feasible LP
- `infeasible` status guarantees an infeasible LP
- `unknown` status randomly selects between feasible (70%) and infeasible (30%)

All changes follow the existing architecture: new constraint data is stored in struct fields (set in the constructor with all randomness), and `build_model` remains completely deterministic.

---

## 2025-01-07

### Major Refactoring: Type-Based Dispatch Architecture

**Previous Commit**: `6c1270f`

**Summary**: Complete refactoring of the problem generator system from function-based to type-based dispatch architecture. This is a **breaking change** that improves code organization, type safety, and extensibility.

### Added

- **`ProblemGenerator` abstract type**: Base type for all problem generators
- **`FeasibilityStatus` enum**: Enum with values `feasible`, `infeasible`, `unknown` for explicit feasibility control
- **`build_model` function**: Generic function that each problem type implements for deterministic model building
- **Struct-based problem generators**: Each of the 21 problem types now has a dedicated struct storing all generated data:
  - `TransportationProblem`
  - `KnapsackProblem`
  - `PortfolioProblem`
  - `DietProblem`
  - `NetworkFlowProblem`
  - `ProductionPlanningProblem`
  - `AssignmentProblem`
  - `BlendingProblem`
  - `AirlineCrewProblem`
  - `CuttingStockProblem`
  - `EnergyProblem`
  - `FacilityLocationProblem`
  - `FeedBlendingProblem`
  - `InventoryProblem`
  - `LandUseProblem`
  - `LoadBalancingProblem`
  - `ProductMixProblem`
  - `ProjectSelectionProblem`
  - `ResourceAllocationProblem`
  - `SchedulingProblem`
  - `SupplyChainProblem`

### Changed

- **`generate_problem` function signature**:
  - Old: `generate_problem(problem_type::Symbol, params::Dict; seed::Int=0)` → returns `(model, params::Dict)`
  - New: `generate_problem(problem_type::Symbol, target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)` → returns `(model, problem::ProblemGenerator)`

- **`register_problem` function signature**:
  - Old: `register_problem(type_sym::Symbol, generator_fn::Function, sampler_fn::Function, description::String)`
  - New: `register_problem(type_sym::Symbol, problem_type::Type{<:ProblemGenerator}, description::String)`

- **`generate_random_problem` function signature**:
  - Old: Returns `(model, problem_type::Symbol, params::Dict)`
  - New: Returns `(model, problem_type::Symbol, problem::ProblemGenerator)`

- **Problem generators**: All 21 problem type implementations refactored from functions to structs
  - Constructors now handle ALL randomness and parameter sampling
  - `build_model` methods are completely deterministic
  - All sophisticated feasibility logic preserved and improved

- **Utility script** (`scripts/generate_problem.jl`):
  - Updated to use new API
  - Added `--feasible`, `--infeasible`, `--unknown` flags for feasibility control
  - Added `--seed=N` flag for explicit seed specification
  - Simplified argument parsing

### Removed

- **Removed functions**:
  - `sample_parameters(problem_type::Symbol, target_variables::Int)` - functionality integrated into constructors
  - `sample_parameters(problem_type::Symbol, size::Symbol)` - legacy size-based API removed
  - `get_generator(problem_type::Symbol)` - replaced by `get_problem_type`
  - `get_sampler(problem_type::Symbol)` - no longer needed
  - All individual `generate_[type]_problem` functions - replaced by constructors
  - All individual `sample_[type]_parameters` functions - integrated into constructors
  - All `calculate_[type]_variable_count` functions - no longer needed

### Technical Details

#### Architecture Changes

1. **Separation of Concerns**:
   - Problem data generation (constructors) is now cleanly separated from model building (`build_model`)
   - All randomness confined to constructors; `build_model` is deterministic

2. **Type Safety**:
   - Each problem type is now a distinct Julia type with compile-time type checking
   - Problem data stored in strongly-typed struct fields instead of `Dict`

3. **Multiple Dispatch**:
   - Uses Julia's multiple dispatch for clean, extensible interface
   - `build_model(::ProblemType)` dispatches to type-specific implementations

4. **Improved Reproducibility**:
   - Same seed guarantees identical problem instance with identical data
   - Deterministic `build_model` ensures same problem always produces same model

5. **Feasibility Control**:
   - Explicit `FeasibilityStatus` enum replaces symbol-based `:solution_status`
   - All generators properly handle `feasible`, `infeasible`, and `unknown` statuses
   - Sophisticated feasibility logic preserved from original implementations:
     - Diet problem: 4 verified impossibility scenarios with final verification
     - Scheduling: Consecutive-day capacity, randomized matching, 3 infeasibility modes
     - Land use: Witness construction, adjacency-aware assignment
     - Supply chain: Geographic clustering, K-nearest connectivity
     - And many more...

#### Code Quality Improvements

- **Reduced code duplication**: Pattern consistency across all 21 generators
- **Better documentation**: Comprehensive docstrings for all structs and functions
- **Cleaner interfaces**: No more `Dict` parameter passing
- **Easier testing**: Structs can be inspected and compared directly

#### Backward Compatibility

**Breaking**: This refactoring intentionally breaks backward compatibility to improve the architecture. The old function-based API is completely removed. Users must update their code to use the new type-based API.

### Migration Guide

#### Old API:
```julia
# Old way
params = sample_parameters(:transportation, 100)
model, actual_params = generate_problem(:transportation, params)
```

#### New API:
```julia
# New way
model, problem = generate_problem(:transportation, 100, unknown, 0)
# Access problem data through struct fields
println(problem.n_sources, problem.n_destinations)
```

### Testing

- Updated test suite to use new API
- All 21 problem types tested with multiple target variable counts
- All three feasibility statuses tested for each problem type
- Reproducibility tests with fixed seeds

### Documentation

- Updated `README.md` with new API examples
- Updated `CLAUDE.md` with new architecture description
- Added comprehensive docstrings to all new types and functions

### Files Modified

- **Core module**: `src/SyntheticLPs.jl`
- **All problem types** (21 files in `src/problem_types/`):
  - airline_crew.jl
  - assignment.jl
  - blending.jl
  - cutting_stock.jl
  - diet_problem.jl
  - energy.jl
  - facility_location.jl
  - feed_blending.jl
  - inventory.jl
  - knapsack.jl
  - land_use.jl
  - load_balancing.jl
  - network_flow.jl
  - portfolio.jl
  - product_mix.jl
  - production_planning.jl
  - project_selection.jl
  - resource_allocation.jl
  - scheduling.jl
  - supply_chain.jl
  - transportation.jl
- **Utility scripts**: `scripts/generate_problem.jl`
- **Tests**: `test/runtests.jl`
- **Documentation**: `README.md`, `CLAUDE.md`
