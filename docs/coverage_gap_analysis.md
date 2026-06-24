# Coverage and Gap Analysis for Real-World LP Instances

This note reviews SyntheticLPs.jl's current generator portfolio against the kinds
of LP and LP-relaxation models that appear in practice. Its goal is not to rank
applications by social importance, but to identify which additions would most
improve the package as a realistic solver benchmark and synthetic dataset source.

## Current coverage snapshot

SyntheticLPs.jl already covers a broad cross-section of classical operations
research models. As of this review, the package has 32 registered categories and
56 concrete variants.

### Strongly covered domains

- **Logistics and networks:** transportation, transshipment, capacitated
  transportation, network flow, generalized flow, multi-commodity flow,
  vehicle routing, telecom network design, facility location, and supply chain.
- **Production and materials:** product mix, production planning, blending,
  feed blending, cutting stock, bin packing, inventory, and crop/land-use
  planning.
- **Workforce and scheduling:** assignment, workload-balanced assignment,
  generic scheduling, job-shop scheduling, nurse scheduling, airline crew, load
  balancing, and unit commitment.
- **Energy and infrastructure:** generation-mix energy planning, ramping,
  reserves, storage, transmission, and DC optimal power flow.
- **Finance, statistics, and uncertainty:** portfolio CVaR, tracking-error
  portfolio models, least-absolute-deviation regression, quantile regression,
  Chebyshev regression, revenue management, project selection, and two-stage
  stochastic programs.

This is a strong base: many commercial LP/MIP models are built from exactly
these primitives, especially flows, assignment, capacity expansion, blending,
inventory, production, scheduling, and portfolio/risk blocks.

## External benchmark and application context

A few external sources are useful calibration points:

- **NETLIB LP** remains a canonical LP benchmark collection, with optimal and
  infeasible MPS instances plus older generator programs. See
  <https://www.netlib.org/lp/>.
- **MIPLIB 2017** emphasizes real submitted mixed-integer models: it selected
  1,065 collection instances from 5,721 submissions, with a 240-instance
  benchmark set. See <https://miplib.zib.de/> and the MIPLIB paper summary at
  <https://optimization-online.org/2019/07/7285/>.
- **Recent MILPBench-style datasets** explicitly categorize many thousands of
  MIP instances into dozens of classes, showing that a realistic benchmark
  corpus needs both domain variety and repeated within-class structure.
- **Solver-vendor application taxonomies** commonly highlight manufacturing,
  logistics, finance, energy, and supply chain optimization as major LP/MIP use
  cases; these are mostly represented here, but with uneven depth.

The practical lesson is that SyntheticLPs.jl is already well aligned with many
classic OR textbook classes, but it still under-represents several high-volume
industrial LP structures and some matrix/pathology features seen in solver
benchmarks.

## Highest-value gaps to add next

### 1. Multi-period supply-chain network planning

**Why it matters:** Real supply-chain LPs are rarely single-period flows. They
combine procurement, production, inventory, distribution, backlogging, capacity,
imports/exports, and sometimes emissions over time. This creates large sparse
block-angular matrices with repeated temporal structure.

**Current status:** Implemented as `supply_chain/network_planning`, combining supply chain, transportation, production, and inventory ideas into a repeated multi-period network-planning formulation.

**Recommended variants:**

- `supply_chain/network_planning`: plants, warehouses, customers, products,
  periods, production, shipment, inventory, and unmet-demand penalty variables.
- `supply_chain/procurement_contracts`: supplier tiers, minimum commitments,
  spot purchases, rebates represented by piecewise-linear segments.
- `supply_chain/resilience`: disrupted arcs/facilities, emergency capacity, and
  service-level constraints.

**Model features to include:** repeated time blocks, dense-ish demand coupling,
  sparse network arcs, optional carbon budget, soft service constraints, and
  tunable degeneracy from many alternative routes.

### 2. Workforce rostering and contact-center staffing

**Why it matters:** Workforce planning is a very common industrial optimization
use case. Even when final schedules are integer, LP relaxations and continuous
staffing models are central to planning and decomposition.

**Current status:** Implemented as `workforce_shift_scheduling/covering`, a generic multi-skill shift-covering/contact-center staffing model with demand curves, breaks embedded in patterns, labor-pool limits, and undercoverage penalties.

**Recommended variants:**

- `workforce_shift_scheduling/covering`: shifts cover time buckets and skill
  groups; objective minimizes wage, overtime, and undercoverage penalties.
- `workforce_shift_scheduling/multi_skill`: employees or labor pools have skill
  matrices and cross-training costs.
- `workforce_shift_scheduling/break_placement`: break patterns and paid/unpaid
  rules via pattern variables.

**Model features to include:** set-covering matrices, highly degenerate columns,
  many nearly interchangeable variables, soft constraints, and integer-relaxed
  pattern variables.

### 3. Marketing, advertising, and media allocation

**Why it matters:** Budget allocation LPs are ubiquitous outside traditional OR:
  marketing mix, media buying, promotions, customer acquisition, and channel
  allocation. They create resource-allocation LPs with multiple budgets,
  reach/frequency proxies, fairness constraints, and piecewise-linear response.

**Current overlap:** Resource allocation and revenue management are related, but
  they do not capture media/channel budget structures.

**Recommended variants:**

- `marketing_allocation/media_mix`: campaigns, channels, geographies, time
  periods, budget constraints, reach targets, and channel caps.
- `marketing_allocation/pwl_response`: response curves approximated with
  ordered linear segments, enabling realistic diminishing returns.
- `marketing_allocation/promotion_calendar`: periods, products, lift estimates,
  inventory/service caps, and cannibalization constraints.

**Model features to include:** bounded continuous variables, many upper bounds,
  multi-budget constraints, dense campaign-to-KPI coefficients, and tunable
  coefficient scaling.

### 4. Healthcare operations beyond nurse scheduling

**Why it matters:** Hospitals and healthcare systems use LP/MIP models for beds,
  operating rooms, appointment slots, radiation therapy planning, staff pools,
  and medical supply allocation.

**Current overlap:** Nurse scheduling is present, but healthcare capacity and
  treatment-planning LPs are absent.

**Recommended variants:**

- `healthcare_capacity/bed_allocation`: units, patient classes, periods,
  discharge/transfer flows, overflow penalties, and staffing links.
- `healthcare_capacity/or_block_planning`: surgical blocks, specialties,
  rooms, periods, elective demand, emergency reserve, and overtime.
- `radiation_therapy/dose_planning`: beamlet intensity variables, voxel dose
  constraints, tumor lower bounds, organ-at-risk upper bounds, and penalty
  variables.

**Model features to include:** many soft constraints, upper/lower clinical
  bounds, block-angular patient-class structure, and dense dose matrices for the
  radiation variant.

### 5. Service-network design and fleet/asset repositioning

**Why it matters:** Airlines, railroads, trucking firms, parcel carriers,
  bike/scooter fleets, cloud capacity providers, and rental networks solve
  time-expanded service network LPs. These are large, sparse, and highly
  structured.

**Current overlap:** Transportation, network flow, multi-commodity flow, airline
  crew, and vehicle routing exist, but not time-expanded fleet circulation or
  empty repositioning.

**Recommended variants:**

- `service_network_design/time_expanded`: nodes by location-time, service arcs,
  holding arcs, demand commodities, capacity, and operating costs.
- `fleet_repositioning/empty_balance`: loaded flows, empty vehicle movements,
  depot balance, and penalty demand spill.
- `parcel_sortation/sort_center_flow`: origin-destination flows through sort
  centers with cut-time and capacity constraints.

**Model features to include:** time-expanded networks, path/arc formulations,
  huge numbers of conservation constraints, and alternative capacity bottlenecks.

### 6. Water, wastewater, and environmental resource planning

**Why it matters:** Water allocation, reservoir operation, pollution control,
  and environmental compliance are important public-sector LP domains and have
  distinctive network-plus-storage structures.

**Current overlap:** Energy storage and land/crop planning partially overlap,
  but hydrologic networks and treatment allocation are not represented.

**Recommended variants:**

- `water_resources/reservoir_operation`: reservoirs, inflows, releases,
  storage, downstream demand, flood-control, and hydropower terms.
- `water_resources/blending_quality`: sources, treatment plants, users, quality
  limits, and contaminant mass-balance constraints.
- `emissions_abatement/cap_and_trade`: sectors, technologies, abatement levels,
  allowance trading, and compliance penalties.

**Model features to include:** temporal storage balance, quality blending,
  environmental caps, and low-rank coupling constraints.

### 7. Dense scientific, estimation, and inverse-problem LPs

**Why it matters:** Solver behavior on dense or semi-dense LPs differs greatly
  from sparse network and planning models. Regression variants are a good start,
  but many real scientific LPs involve basis pursuit, sparse recovery, robust
  fitting, and inverse planning.

**Current status:** LAD, quantile, Chebyshev, and `regression/basis_pursuit` now cover several dense statistical and sparse-recovery LPs.

**Recommended variants:**

- `compressed_sensing/basis_pursuit`: split positive/negative coefficients,
  equality fitting constraints, and L1 objective.
- `robust_optimization/box_uncertainty`: robust counterparts with protection
  variables and budgeted uncertainty.
- `inverse_planning/goal_programming`: many positive/negative deviation
  variables around target achievements.

**Model features to include:** dense coefficient matrices, many equality
  constraints, split variables, free-variable transformations, and controlled
  condition numbers.

### 8. Market-clearing, auctions, and equilibrium-style LPs

**Why it matters:** Power markets, ad auctions, spectrum auctions, logistics
  exchanges, and commodity markets often solve LPs or LP relaxations for market
  clearing and prices.

**Current overlap:** Revenue management and energy models touch pricing but not
  explicit market clearing.

**Recommended variants:**

- `market_clearing/single_period`: supply bids, demand bids, network limits,
  clearing quantities, and welfare maximization.
- `market_clearing/power_pool`: generator offers, load bids, DC network, line
  limits, reserve products, and locational marginal price structure.
- `auction_allocation/package_relaxation`: bidders, items, package variables,
  item capacity constraints, and LP relaxation behavior.

**Model features to include:** primal-dual economic interpretation, many bounds,
  sparse incidence matrices, degeneracy from tied bids, and wide objective
  coefficient ranges.

### 9. Classic hard/pathological LP benchmark structures

**Why it matters:** Domain realism is not enough for solver benchmarking. Real
  benchmark suites also include pathological or numerically challenging cases:
  infeasible models, near-degenerate optima, badly scaled matrices, cycling-prone
  structures, dense columns, equality-heavy systems, and nearly parallel rows.

**Current status:** Implemented a `benchmark_pathologies` category with `scaling_stress` and `degenerate_network` variants, while leaving additional pathological structures for future work.

**Recommended variants:**

- `benchmark_pathologies/klee_minty`: scalable distorted-cube LPs for simplex
  stress tests.
- `benchmark_pathologies/degenerate_network`: many zero-cost alternate paths and
  redundant conservation rows.
- `benchmark_pathologies/scaling_stress`: coefficients and bounds sampled across
  several orders of magnitude while preserving feasibility.
- `benchmark_pathologies/nearly_infeasible`: controlled Farkas-distance cases.

**Model features to include:** reproducible condition-number controls,
  redundancy controls, explicit optimal-face dimension controls, and known
  feasibility certificates where practical.

## Medium-priority gaps

- **Education timetabling and room assignment:** adjacent to scheduling, but
  with periods, rooms, curricula, conflict graphs, and preference penalties.
- **Cloud and data-center capacity planning:** load balancing exists, but not
  multi-period VM placement, reserved/on-demand capacity, migration cost, and
  energy/cooling constraints.
- **Maintenance and asset management:** inspection intervals, spare parts,
  downtime windows, and reliability budgets.
- **Public-sector disaster relief and humanitarian logistics:** facility
  staging, commodity flows, equity constraints, deprivation penalties, and
  uncertain demand scenarios.
- **Telecommunications traffic engineering:** telecom network design is present,
  but a traffic-engineering/path-flow variant with demand matrices and latency
  budgets would be useful.
- **Agricultural whole-farm planning:** crop planning exists, but livestock feed,
  machinery, water, labor calendars, rotations, and risk would deepen the domain.

## Cross-cutting realism improvements

These changes would improve every generator, not just add categories.

### 1. Calibrate dimensions and sparsity from benchmark corpora

Add optional metadata targets for variable/constraint ratio, nonzeros per row,
nonzeros per column, bound density, equality fraction, and coefficient scale.
The MIPLIB site notes similarity features based on variables, objectives,
bounds, constraints, and right-hand sides, which is a useful signal for what a
synthetic corpus should track.

### 2. Support formulations as variants

Many real problem classes have multiple formulations with very different LP
relaxations: arc-flow vs path-flow, assignment vs set partitioning, compact vs
extended, big-M vs indicator-style relaxations, and time-indexed vs event-based
scheduling. SyntheticLPs.jl's category/variant system is already well suited to
this.

### 3. Add known-solution and certificate metadata

Where possible, store expected feasibility, a constructed primal solution, and
simple certificates or lower/upper bounds. This would make generated datasets
more useful for solver validation, ML labels, and regression tests.

### 4. Increase numerical diversity deliberately

Real LPs include objective coefficients, right-hand sides, and constraint
coefficients across many scales. Add per-generator knobs for coefficient scale,
integer-like vs decimal coefficients, near-zero coefficients, and correlation
between costs and capacities.

### 5. Add correlated, geography-like data generation

Many current generators appear to use independent random data. Real instances
have spatial clusters, time-of-day patterns, correlated costs/capacities,
seasonality, hub structures, and sparse feasible-arc graphs. Shared utilities
for clustered geography and temporal demand profiles would improve many models.

## Recommended implementation roadmap

The first implementation tranche added `supply_chain/network_planning`,
`workforce_shift_scheduling/covering`, `service_network_design/time_expanded`,
`regression/basis_pursuit`, and the `benchmark_pathologies` category. The remaining priorities are:

1. **Healthcare capacity and radiation planning** — important domain with
   distinctive soft-constraint and dense-matrix variants.
2. **Marketing/media allocation** — broad business use, relatively easy to add,
   and gives bounded resource-allocation LPs with piecewise-linear response.
3. **Water/environmental planning** — strong public-sector relevance and useful
   storage/blending hybrids.
4. **Market clearing and auctions** — economically important and useful for
   degeneracy/pricing behavior.

## What not to prioritize first

- **More single-period textbook variants** unless they introduce a new matrix
  structure. The package already has many textbook LPs.
- **Purely nonlinear domains** unless they have standard linear approximations.
- **Highly specialized one-off industrial models** unless they can be expressed
  as reusable families with controllable dimensions and realism knobs.

## Bottom line

SyntheticLPs.jl is already broad across classic OR categories. The biggest gap
is not another isolated textbook model; it is a set of generators that combine
existing primitives into larger, repeated, multi-period, multi-commodity,
soft-constrained, numerically diverse models. The first implementation tranche added multi-period supply-chain planning,
generic workforce shift covering, benchmark pathologies, dense scientific LPs,
and service-network design; healthcare capacity, marketing/media allocation,
water/environmental planning, and market clearing remain the next largest
coverage opportunities.
