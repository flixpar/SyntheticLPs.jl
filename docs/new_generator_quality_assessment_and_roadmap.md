# Quality assessment and calibration roadmap for the new generators

## Purpose

This note assesses the 23 generators proposed in the collected-corpus coverage
PRs. It asks a different question from
[`collected_corpus_coverage_gaps.md`](collected_corpus_coverage_gaps.md): now that
the missing formulations exist, how closely do they resemble the collected
instances, how much meaningful diversity do they produce, and what work is
needed before they can be described as corpus-realistic or hardness-calibrated?

The assessment separates five properties that are easy to conflate:

1. **Formulation coverage:** whether the defining variables and constraints of
   a problem family are present.
2. **Generator correctness:** determinism, target sizing, domain preservation,
   feasibility contracts, export, and solver verification.
3. **Instance realism:** whether dimensions, topology, sparsity, coefficients,
   bounds, and correlations resemble real or collected instances.
4. **Distributional diversity:** whether seeds and profiles produce materially
   different structural regimes rather than variations of one template.
5. **Benchmark difficulty:** whether presolve, LP relaxation, and branch-and-
   bound behavior resemble held-out benchmark instances.

Passing the first two does not establish the last three.

## Executive assessment

The new generators are strong **v1 structural-coverage generators**. They are
not generally weaker than the existing package. In several engineering respects
they are stronger than the existing median:

- constructors use local RNGs rather than mutating global random state;
- feasible requests have explicit witnesses;
- infeasible requests have certificates that remain valid after integrality is
  relaxed;
- variable-domain mixes and target sizes are tested explicitly;
- repeated model construction and MPS export are deterministic; and
- all new formulations have been checked with HiGHS in both relaxed and
  unrelaxed form.

They are nevertheless less mature than the package's strongest families in two
important respects. Most have only one data-generation regime, and none has yet
been calibrated against the collected corpus's joint distributions or solver
behavior. They reproduce the *kind* of matrix associated with a family, but not
yet the full population of matrices in that family.

The right summary is therefore:

| Dimension | Assessment |
|---|---|
| API and engineering contracts | Strong; above the existing-package median |
| Mathematical formulation | Correct and representative |
| Seed-level numeric diversity | Good |
| Structural-regime diversity | Low to moderate |
| Empirical corpus fidelity | Not established |
| Solver-hardness fidelity | Not established |

The generators should be merged and documented as v1 coverage additions. They
should not yet be described as realistic reconstructions of the collected
families or as difficulty-matched solver benchmarks.

## Evidence from the collected corpus

The catalog already shows why family membership and family prevalence are not
enough. Several collected families occupy very different structural and size
regimes:

| Collected class | Variables | Rows | Mean nonzeros per row | Important implication |
|---|---:|---:|---:|---|
| Generic IP | 1,083 | 195 | 38.2 | A specific, mostly-binary shape rather than an arbitrary mixed MILP |
| MIS | 1,000--6,000 | 3,923--24,409 | 2.0 | Sparse edge rows, but substantially larger than ordinary smoke-test instances |
| GISP | 519--54,397 | 1,849--71,820 | 2.6 | Very broad size and topology range |
| MVC | 500--2,000 | 5,975--65,100 | 2.0 | Much denser graphs than the current average-degree regime |
| Set cover | 1,000--4,000 | 500--5,000 | 138.2 | Dense row incidence despite binary columns |
| Combinatorial auction | 1,000--4,000 | 371--2,715 | 10.0 | Bundle incidence is much sparser than set cover |
| MMCN | 982--30,234 | 379--4,715 | 21.9 | Large commodity blocks over comparatively few network rows |
| NNV | 6,920--7,535 | 6,197--7,123 | 46.5 | Narrow size band and characteristic network sparsity |
| OTS | 21,318--40,727 | 48,582--92,992 | 2.7 | Large sparse topology-disjunction models |
| Load balancing | 61,000 | 64,118--64,487 | 5.6 | Nearly fixed, continuous-heavy formulation shape |
| MIRP | 1,613--92,261 | 1,080--126,621 | 4.9 | Very broad time-expanded routing range |
| Resilient pipe network | 68--39,240 | 69--38,216 | 3.6 | Broad scenario/topology range |

These are catalog-row summaries, not a complete fidelity analysis. They do,
however, expose several immediate mismatches:

- the v1 preset uses one global target-size distribution for all families;
- its 20,000-variable cap excludes most collected load-balancing and OTS models
  and part of the MIRP, GISP, and MMCN tails;
- the generic generator's fixed binary/integer/continuous mix is not the
  dominant collected generic-IP mix;
- the current vertex-cover graphs are much sparser than the collected models;
- the current set-cover columns do not reproduce the collected row density; and
- the NNV generator's dense Gaussian layers need not reproduce the collected
  network connectivity or nonzeros per row.

For that reason, `corpus_matched_preset()` v1 should be interpreted as a
**corpus-family-weight-matched** preset. It matches prevalence for represented
families, preserves integrality, and records its policy, but it does not yet
match family-conditional size, sparsity, coefficient, or difficulty
distributions.

## What is already strong

### Reliable contracts

The new generators consistently satisfy the package interface. They construct
models deterministically from stored problem data, meet the target-variable
contract closely, preserve intended integer domains, and export cleanly. This
is a meaningful improvement over generators that rely on global RNG state or
heuristic feasibility branches.

The explicit relaxation-safe certificates are especially useful for package
testing. They verify that `relax_integer=true` does not silently invalidate the
requested status and make regressions easy to diagnose.

### Defining formulation structure

The more involved additions contain genuine family-specific coupling rather
than superficial renaming:

- neural verification has affine layers, propagated bounds, stable ReLU
  elimination, unstable-neuron binaries, and bound-derived disjunctions;
- OTS has line-status binaries, gated thermal limits, DC equations, voltage
  angles, dispatch, and nodal balance;
- binary-capacity MMCN combines module installation with multicommodity routing;
- discrete load balancing couples placement, routed work, machine load,
  capacity, replication, and makespan;
- maritime inventory routing couples vessel movement, onboard load, pickup,
  delivery, and customer inventory across time; and
- resilient network design shares first-stage build/hardening decisions across
  scenario-specific flows.

The smaller variants also fill real formulation gaps. General-integer cutting
patterns, fixed-charge transportation lanes, mixed-integer knapsack sets, and
combinatorial-auction bundles are materially different from the package's
previous relaxations or neighboring application models.

## Cross-cutting weaknesses

### One structural regime per variant

Most variants use one dimension formula, one topology sampler, and narrow
coefficient ranges. Different seeds change the numbers and edges, but do not
usually change the generative mechanism. A large sample can therefore contain
many statistically similar instances.

Examples include a fixed graph average-degree range, one planted set-partition
construction, a fixed two-module capacity design, one random spanning-tree-plus-
extra-edges topology, and fixed rules for the number of neural-network layers.

### Planting and ordering artifacts

Several constructive choices are visible in the emitted matrix:

- planted set-system columns are placed first;
- direct cutting-stock patterns are placed first;
- every fourth mixed-integer-knapsack row is dense;
- neural phases follow a repeating inactive/active/unstable pattern;
- graph feasible sets and covers use fixed fractions of the vertices; and
- some time-expanded witnesses follow regular alternating schedules.

These patterns are harmless for correctness testing but can become shortcuts
for a learned model. Rows, columns, planted structures, and regime choices
should be randomized after construction unless ordering is itself part of the
target family.

### Feasibility-label leakage

Certified infeasibility is often deliberately obvious: a singleton bound
contradiction, an impossible global cardinality row, a total-capacity shortfall,
a common auction item plus a winner floor, or a property threshold above a
propagated bound. Some feasible branches also add a cardinality or budget row
that is absent from the natural formulation.

This is excellent for unit tests, but dangerous for feasibility-classification
or representation-learning datasets. A model may learn the certificate recipe
rather than the underlying family. Feasibility mode also sometimes changes row
counts or graph structure, making the label even easier to infer.

### Lack of empirical hardness calibration

Passing a solver check establishes correctness, not difficulty. The current
validation does not compare:

- presolve row and column reduction;
- root-LP objective and integrality gap;
- degeneracy or simplex iterations;
- cut generation;
- branch-and-bound nodes and depth;
- time to first feasible solution;
- primal/dual progress curves; or
- solve-time and timeout rates.

Planted witnesses and large certificate margins may make instances much easier
than their collected counterparts. Conversely, dense synthetic neural layers
or quadratic pairwise packing formulations may be computationally expensive for
reasons unlike the collected family.

### Variable count is not enough

The current common interface targets variables. It does not target rows,
nonzeros, domain proportions, coefficient dynamic range, graph degree, set
incidence, or block dimensions. Two models with the same number of variables
can therefore be structurally unrelated.

## Family-specific assessment and improvements

### Generic MILP

**Current strength:** mixed binary, general-integer, and continuous columns;
multiple bound regimes; mixed row senses; several coefficient scales; and a
transparent planted witness.

**Current limitation:** fixed domain proportions and roughly 0.3 rows per
variable define one generic distribution. There is no block structure,
near-dependence, equality-heavy mode, degeneracy control, or dedicated match to
the dominant collected IP profile.

**Recommended improvements:** make generic MILP a family of versioned profiles:

- `d_miplib_ip`: mostly binary with the observed row and continuous-column
  proportions;
- `mixed_sparse`: the current broad mixed-domain profile;
- `block_angular`: linking rows over repeated local blocks;
- `equality_heavy`: standard-form and network-like bases;
- `numerical`: controlled dynamic range and near-dependent rows; and
- `degenerate`: redundant bounds, duplicated structure, and multiple optima.

### Graph optimization

**Current strength:** correct MIS, GISP, vertex-cover, coloring, map-labeling,
and quasi-clique formulations with exact target sizing and planted structures.

**Current limitation:** most graphs are uniform sparse random graphs with
average degree in a narrow range. The collected MVC family is far denser, and
real graph families contain communities, hubs, geometry, regularity, and broad
degree distributions. Certified branches add global cardinality rows not found
in ordinary MIS or MVC formulations.

**Recommended improvements:** add Erdős--Rényi, random-regular,
preferential-attachment, geometric, stochastic-block/community, and
degree-sequence samplers. Calibrate graph order, edge count, component count,
clustering, degree quantiles, and weight correlations by collected subclass.
Construct natural feasible instances without adding nonstandard cardinality
rows; retain certificate variants as explicitly labeled challenge profiles.

### Set systems and auctions

**Current strength:** the four important row senses and auction semantics are
distinct, and nonempty rows plus planted covers prevent malformed instances.

**Current limitation:** the same planted-partition mechanism underlies several
variants. Column sizes are short, item popularity is close to uniform, and
planted columns appear first. This particularly underrepresents the collected
set-cover row density.

**Recommended improvements:** independently control row/column ratio, column
size, row degree, overlap, dominance, duplicates, and connected components of
the incidence graph. Add Zipf or lognormal item popularity. For auctions, add
bidder identities, substitutes/complements, budget effects, geographic bundles,
and value models beyond a single item-sum synergy rule.

### Neural-network verification

**Current strength:** this is one of the strongest new formulations. It uses
valid propagated bounds, eliminates stable phases, and avoids arbitrary big-M
constants.

**Current limitation:** dense Gaussian networks with a forced one-third phase
pattern are synthetic in a recognizable way. Layer counts are restricted to
one to three, the output is scalar, and the property is a box-to-threshold
query. Interval bounds may become much looser than the bounds used in collected
instances.

**Recommended improvements:** derive phase stability naturally from sampled or
trained-like weights and perturbation radii. Add sparse, convolutional,
residual, and multi-output architectures; classification-margin and robustness
properties; alternative norm balls; and multiple bound-strength profiles.
Calibrate neurons, layers, unstable fraction, row width, and bound tightness to
the collected NNV matrices.

### Optimal transmission switching

**Current strength:** electrically meaningful DC coupling and valid line-
specific disjunction bounds.

**Current limitation:** topology is a random tree plus random chords rather
than a power-grid-like network. Electrical values, generator placement, load,
line limits, and switching economics are mostly independent. Minimum output is
zero, there are no contingencies, and only one operating period is represented.

**Recommended improvements:** use perturbed public grid templates or synthetic
spatial grid models; correlate voltage, susceptance, thermal capacity, distance,
and cost; add generator technology and cost-curve regimes; and introduce
security constraints, must-run lines, switching budgets, and multi-period load
profiles where appropriate.

### Discrete multicommodity flow and load balancing

**Current strength:** both additions reproduce the important discrete/continuous
block ratios and capacity coupling absent from the previous continuous models.

**Current limitation:** network topologies are strongly connected cycles plus
random arcs, commodities are independent uniform pairs, binary capacity has
exactly two mutually exclusive modules, and integer flow may split integral
units over multiple paths. Load balancing uses all-compatible service-machine
pairs and a nearly separable processing-time model.

**Recommended improvements:** add geometric, hierarchical, hub-and-spoke, and
community network profiles; correlated commodity clusters; existing capacity;
multiple or cumulative modules; arc eligibility; fixed and variable congestion
costs; and optional unsplittable-path variables. For load balancing, add memory
and placement resources, affinity/anti-affinity, incompatible machines, setup
costs, migration, locality, skewed service popularity, and multiple bottleneck
resources.

### Maritime inventory routing and resilient networks

**Current strength:** both models have genuine multi-block application coupling
and are much closer to their collected families than an uncoupled inventory or
network-flow substitute.

**Current limitation:** maritime travel takes one period between every pair of
ports, the network is complete, and there are no travel-time, berth, production,
inventory-capacity, vessel-compatibility, or time-window regimes. Resilient
scenarios use one source-sink flow each and restore failed edges to full
capacity when hardened; real scenarios often share a common demand system and
have graded damage and repair.

**Recommended improvements:** introduce spatial travel times, sparse shipping
lanes, port production, storage limits, vessel-specific capacity and access,
service windows, and multi-depot fleets. For resilience, keep common
multi-commodity demands across scenarios; use correlated regional failures,
partial hardening, repair stages, redundancy requirements, and scenario
probabilities.

### Packing, fixed-charge transportation, cutting stock, and mixed knapsack

**Current strength:** these are clean and useful representatives of formulation
types that were previously absent or relaxed.

**Current limitation:** container loading is primarily multidimensional bin
packing rather than physical loading; two-dimensional packing has identical
bins and no rotation; fixed-charge transportation uses a complete bipartite
network with independent costs; cutting patterns come from one stock length;
and mixed knapsack follows one deterministic sparse/dense row schedule.

**Recommended improvements:** add heterogeneous containers, rotation, stacking,
incompatibilities, weight balance, and stability; spatial and sparse
transportation lanes with correlated cost/capacity; multiple stock types and
trim-loss economics; and randomized or block-correlated knapsack row regimes.

## Recommended generator semantics

The package should explicitly separate how an instance was generated from what
feasibility label was requested.

### Generation modes

- `natural`: sample the application distribution without adding a status-
  revealing certificate; solve afterward if a label is required.
- `certified_feasible`: construct and retain a witness, while avoiding
  nonstandard constraints when possible.
- `certified_infeasible`: construct a reviewable certificate and record its
  type in metadata.
- `challenge`: construct paired or near-boundary instances intended to stress a
  solver or classifier.

The current `FeasibilityStatus` API can remain backward compatible, but the
manifest should record the generation mode and certificate type separately.

### Versioned family profiles

Each profile should define and record:

- profile name and version;
- source corpus and catalog snapshot;
- dimension and structural distributions;
- coefficient and bound regimes;
- generation mode;
- any planted structure and its hidden metadata;
- generator code revision; and
- calibration report identifier.

Existing v1 profiles must remain immutable so datasets can be reproduced after
new calibrated profiles are added.

## Roadmap

### Phase 0: Freeze and label v1

**Goal:** preserve the current work without overstating it.

Deliverables:

- merge the formulation-coverage PRs;
- retain the current v1 generators as stable profiles;
- describe the preset as family-weight-matched;
- record construction mode and certificate type in manifests; and
- document known ordering and planting artifacts.

Exit gate:

- every emitted instance can be traced to a generator version, profile, seed,
  integrality policy, and construction mode.

### Phase 1: Build the corpus profiler and comparison harness

**Goal:** turn realism into a measured contract.

For each collected and synthetic instance, compute:

- variables, rows, nonzeros, and domain proportions;
- row-sense and bound-type proportions;
- row and column degree quantiles;
- coefficient, RHS, objective, and bound magnitude quantiles;
- coefficient dynamic range and zero/one fractions;
- connected components and block-structure summaries;
- application-specific statistics such as graph degree, set size, number of
  commodities, neural layer sparsity, or scenario count; and
- solver measurements described in Phase 4.

Store immutable per-instance fingerprints plus per-family reports. Split
collected instances into calibration and held-out validation subsets before
tuning profiles.

Exit gate:

- one reproducible command generates a synthetic sample and a side-by-side
  report against both calibration and held-out collected data.

### Phase 2: Remove avoidable synthetic artifacts

**Goal:** prevent trivial source and label shortcuts.

Deliverables:

- shuffle planted columns, direct patterns, rows, and variable blocks;
- sample neural phases and dense-row positions rather than cycling them;
- make feasibility construction preserve the natural row schema where
  possible;
- add paired instances that share a base matrix but differ by a controlled
  feasibility perturbation; and
- train simple diagnostic classifiers to detect generator source and
  feasibility label from cheap matrix summaries.

Exit gates:

- no ordering-only feature identifies planted columns or certificate type;
- feasibility cannot be predicted reliably from row count, column count, or one
  obvious aggregate statistic alone; and
- paired generation reproduces the same base structure exactly.

### Phase 3: Calibrate the high-volume families

**Goal:** improve the families representing most collected instances first.

Priority order:

1. generic IP;
2. MIS, GISP, and MVC;
3. set cover and combinatorial auctions;
4. MMCN;
5. NNV;
6. OTS; and
7. load balancing.

Deliverables:

- multiple versioned profiles per family;
- family-conditional target-size distributions;
- joint calibration of variables, rows, nonzeros, domains, and family-specific
  structure; and
- profile weights based on deduplicated corpus counts.

Exit gates should be predeclared per family. At minimum:

- generated 5th--95th percentile ranges overlap the held-out corpus for
  variables, rows, and nonzeros;
- domain proportions and row-sense proportions are within an agreed tolerance;
- row/column degree and coefficient-scale reports show no unexplained
  order-of-magnitude mismatch; and
- held-out fidelity improves over v1 rather than only over the calibration set.

### Phase 4: Add solver-behavior calibration

**Goal:** distinguish realistic matrices from realistic optimization tasks.

Run a fixed, versioned solver configuration on collected and synthetic samples.
Record presolve reduction, root relaxation, gap, iterations, cuts, nodes,
solution times, and termination status. Use more than one solver when practical
so a profile does not encode one solver's idiosyncrasies.

Difficulty profiles should be defined by measured bands, not by coefficient
labels such as `easy` or `hard`. Candidate instances should be sampled on a
held-out seed set before thresholds are frozen.

Exit gates:

- each calibrated profile publishes its solver configuration and empirical
  distribution;
- trivial-at-presolve and timeout rates fall within declared bands; and
- no hardness claim is made outside the tested solver/limit configuration.

### Phase 5: Deepen the smaller application families

**Goal:** add application realism after the dominant corpus mass is calibrated.

Implement the maritime, resilience, packing, cutting-stock, fixed-charge, and
mixed-knapsack improvements listed above. Use public application templates when
licenses and provenance allow; otherwise calibrate synthetic spatial and
temporal models to the catalog fingerprints.

Exit gate:

- each family has at least two materially different structural profiles, not
  merely different coefficient ranges.

### Phase 6: Release `corpus_matched` v2

**Goal:** provide a preset whose name reflects a real multidimensional match.

The v2 preset should contain:

- deduplicated family weights;
- family-specific size and structural profiles;
- explicit natural/certified/challenge proportions;
- an integrality policy;
- profile and calibration versions;
- maximum resource guards for superlinear formulations; and
- a generated-versus-target distribution report in the manifest.

Keep v1 available unchanged. A v2 release should fail closed if a referenced
profile or calibration artifact is unavailable rather than silently falling
back to uniform variant sampling.

## Validation strategy

Three test layers are needed:

1. **Deterministic unit tests:** dimensions, domains, certificates, bounds,
   invariants, and exact same-seed reconstruction.
2. **Distributional regression tests:** frozen sample summaries with broad,
   non-flaky tolerances that catch accidental changes to profiles.
3. **Offline calibration reports:** larger held-out comparisons and solver runs;
   these should produce versioned artifacts rather than make ordinary CI slow or
   solver-dependent.

Every profile change should state whether it is a bug fix to an existing
version or a new version. Changes that alter emitted distributions should
normally create a new profile version.

## Recommended immediate next steps

1. Merge the new generators as v1 structural coverage.
2. Clarify the v1 preset's family-weight-only scope in names and manifests.
3. Implement the corpus fingerprint extractor and freeze held-out splits.
4. Audit and remove ordering and feasibility-label leakage.
5. Calibrate generic IP, graph, and set-system profiles first; they represent
   the largest collected mass and expose the clearest current mismatches.
6. Add family-specific size distributions before increasing the global size
   cap.
7. Begin solver-behavior calibration only after the matrix distributions are
   credibly aligned.

This sequence preserves the value of the current implementation while placing
future realism and hardness claims behind explicit, reproducible evidence.
