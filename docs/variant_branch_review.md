# Variant Branch Review — Ported and Deferred Variants
_Generated 2026-06-20. Reviews old branches that predate the category/variant system (`claude/*` and `codex/*`), evaluating each variant for real formulation, correctness, and realistic/diverse data, then porting the highest-quality ones into the new system._
## Method
Each variant on every old branch was reviewed independently against the quality bar set by `transportation/standard.jl` and `portfolio/cvar.jl`. Criteria: a real, mathematically sound OR formulation; substantive distinctiveness vs. the `standard` variant and siblings; realistic, diverse, size-scaled data; all randomness in the constructor with a deterministic `build_model`; correct `target_variables` scaling; and reliable feasible/infeasible/unknown handling. No variant met the bar with zero issues (`port_as_is`); the strongest were `port_with_fixes` (score 7-8).
## Ported (25 variants)
All ported variants were re-expressed as self-contained files in the new variant system, had their reviewer-identified issues fixed, and were verified by the test suite (structure, ±25% variable-count scaling, reproducibility) **and** a HiGHS solve smoke-test (feasible→solvable, infeasible→infeasible across multiple seeds).
**New categories:** `bin_packing`, `nurse_scheduling`, `job_shop_scheduling`, `unit_commitment`.
**New variants in existing categories:** transportation (`balanced`, `capacitated`, `transshipment`, `emission_constrained`), energy (`ramping`, `reserves`, `storage`, `transmission`), inventory (`lot_sizing`, `multi_item`, `multi_echelon`), supply_chain (`single_source`, `carbon`, `multi_product`), blending (`equipment_batches`, `multi_product`), cutting_stock (`setup_cost`, `due_dates`), diet_problem (`nutrient_bounds`, `food_groups`), facility_location (`two_echelon`).
### Overlap resolution
Several concepts appeared on multiple branches; the strongest implementation was chosen:
- **Nurse scheduling**: ported the `codex/add-nurse-scheduling` generator (score 8) over the thinner `scheduling/nurse` on `claude/add-problem-variants` (score 6).
- **Transshipment**: ported the E2YAf `transportation/transshipment` (richer hub flow-conservation) over the `add-transportation-generators` version.
- **Two-echelon facility location**: ported `add-transportation-generators/warehouse_location` into `facility_location/two_echelon` (more appropriate home than transportation).
- **Vehicle routing** (two branches) and **last-mile delivery**: all deferred — see below.

## Deferred / not ported (110 variants)
Recorded for future, higher-quality reimplementation. Not ported to preserve the project's quality bar. Columns: variant · reviewer score (1-10) · status · distinctiveness · source branch · primary reason.
### High-value concepts worth revisiting first
These scored well or model genuinely useful problems but need non-trivial rework (degenerate LP relaxations, broken feasibility handling, or overlap):
- **vehicle_routing/standard & transportation/last_mile_delivery** — CVRP / VRPTW — degenerate LP relaxation: the depot does not anchor routes, so the relaxation collapses to fractional inter-customer cycles. Needs depot-linking or commodity-flow load tracking (or a binary MIP) to be a real routing model.
- **production_planning/multi_period_inventory** — Multi-period production with seasonality + holding — good concept but severe sizing bug (n_periods pinned to ~4, ~9x var overshoot) and overlaps the ported inventory lot-sizing variants.
- **knapsack/multidimensional, knapsack/bounded** — Classic, clean knapsack extensions (score 6) — strong candidates for a future batch; deferred only to keep this batch focused on score-7+.
- **facility_location/p_median** — Classic p-median (score 6) — recognizable and useful; p_center sibling was rejected (degenerate min-max).
- **portfolio/{tracking_error,sector_limits,cardinality,turnover}** — Standard institutional portfolio constraints (score 6) — viable future additions; the main portfolio is already a rich CVaR model.

### Full deferred list by category

#### assignment
| variant | score | status | distinct | branch | reason |
|---|---|---|---|---|---|
| `multi_assignment` | 6 | port_with_fixes | substantive | `problem-generator-variants` | In the feasible case task_requirements is all ones, so it reduces close to standard assignment with capacities; limited data diversity. |
| `workload_balance` | 6 | port_with_fixes | substantive | `problem-generator-variants` | INFEASIBLE handling is broken: it only sets task_workloads[1]=1000 and forces task 1 onto worker 1. A huge workload does not make a minimax model infeasible — m |
| `team_assignment` | 6 | port_with_fixes | substantive | `problem-generator-variants` | Feasibility risk: with each team usable on at most one project and only n_teams teams total, sum over projects of project_team_reqs[j] can exceed n_teams, makin |
| `preference` | 5 | port_with_fixes | moderate | `problem-generator-variants` | Distinctiveness is only in the objective; constraint structure identical to standard (an objective tweak, borderline cosmetic but the preference reward is a rea |
| `shift_assignment` | 5 | port_with_fixes | substantive | `problem-generator-variants` | The shift_requirements coverage constraint is largely redundant with the per-task ==1 coverage: shift_requirements[s] is set to the number of tasks in shift s,  |
| `standard` | 4 | duplicate | cosmetic | `problem-generator-variants` | Duplicates the existing main :standard assignment variant. |
| `skill_match` | 3 | reject | cosmetic | `problem-generator-variants` | Structurally identical to standard; only the sparsity pattern of allowed differs (an all-or-nothing capability mask). No skill-aware objective or constraint, so |
| `geographic` | 3 | reject | cosmetic | `problem-generator-variants` | Structurally a re-skin of standard: the only difference is the sparsity pattern of allowed (a distance mask). No distance term in the objective or distance cons |

#### blending
| variant | score | status | distinct | branch | reason |
|---|---|---|---|---|---|
| `stability` | 6 | port_with_fixes | moderate | `problem-generator-variants` | Distinctiveness is modest: it is the standard model plus one averaged-attribute lower-bound row, conceptually identical in structure to an existing quality lowe |
| `safety` | 6 | port_with_fixes | moderate | `problem-generator-variants` | max_contaminant is an absolute amount while contaminant_levels are per-unit concentrations; the 0.05*min_blend cap implies an allowed average concentration of 0 |
| `target_match` | 6 | port_with_fixes | moderate | `problem-generator-variants` | Feasible/unknown instances are not guaranteed feasible: a random target need not lie within the convex span of ingredient attributes, so tight tolerance bands c |
| `beverage` | 6 | port_with_fixes | substantive | `add-problem-variants-01PYi` | Uses OLD flat registration: register_problem(:blending_beverage, ...). Must be converted to register_variant(:blending, :beverage, BeverageBlending, ...) for th |
| `standard` | 5 | duplicate | cosmetic | `problem-generator-variants` | Duplicates the existing blending/:standard variant in main; formulation is the canonical basic blend, not richer than main's. |
| `ratio_constraints` | 5 | port_with_fixes | moderate | `problem-generator-variants` | Infeasible mode is NOT reliably infeasible: it adds x1>=2*x2 and x2>=2*x1 which only forces x1=x2=0; all OTHER ingredients remain free, so the blend (sum>=min_b |
| `pharmaceutical` | 5 | port_with_fixes | substantive | `add-problem-variants-01PYi` | Uses OLD-system register_problem(:blending_pharmaceutical, ...) instead of register_variant(:blending, :pharmaceutical, ...); will not register in the new varia |
| `max_quality` | 2 | reject | moderate | `problem-generator-variants` | Nonlinear (linear-fractional) objective sum(attr.x)/sum(x) — not an LP; serious correctness bug. |

#### cutting_stock
| variant | score | status | distinct | branch | reason |
|---|---|---|---|---|---|
| `standard` | 6 | duplicate | cosmetic | `problem-generator-variants` | Duplicates the existing main :standard variant for cutting_stock (this IS the standard cutting stock formulation). |
| `multi_stock` | 6 | port_with_fixes | substantive | `problem-generator-variants` | build_model declares a RECTANGULAR variable matrix x[1:n_stock, 1:maximum(length.(prob.patterns_by_stock))] but patterns_by_stock is the multi-stock field, yet  |
| `trim_limit` | 6 | port_with_fixes | substantive | `problem-generator-variants` | The trim constraint is GLOBAL/aggregate, so it can be satisfied by mixing low-waste patterns even if individual patterns waste more — this is a legitimate but m |
| `min_runs` | 6 | port_with_fixes | substantive | `problem-generator-variants` | SERIOUS feasibility bug: setting min_runs = sum(demands)+100 does NOT make the model infeasible because demand constraints are >= (overproduction allowed) and s |

#### diet_problem
| variant | score | status | distinct | branch | reason |
|---|---|---|---|---|---|
| `standard` | 6 | duplicate | cosmetic | `problem-generator-variants` | Duplicates the existing main diet_problem/:standard formulation. |
| `variety` | 6 | port_with_fixes | substantive | `problem-generator-variants` | The lower linking constraint simplifies to: y=1 -> x>=0 (no real forcing), y=0 -> x>=-threshold (trivially true). So y_i can be 1 with x_i=0; the variety count  |
| `allergen_free` | 6 | port_with_fixes | moderate | `problem-generator-variants` | x_i==0 equality constraints are weak modeling vs simply not generating those columns; many solvers treat as fixed var. Cosmetic-leaning distinctiveness. |
| `meal_plan` | 6 | port_with_fixes | substantive | `problem-generator-variants` | Same calorie scale-mismatch as calorie_range: meal_calorie_targets are absolute kcal (1800-2500 split) but content column 1 magnitudes are tiny; the per-meal ca |
| `macro_ratios` | 5 | port_with_fixes | substantive | `problem-generator-variants` | Feasibility NOT guaranteed for the feasible case: the baseline diet is constructed ignoring macro ratios, and the ratio bands (e.g. protein in [0.10-0.15, 0.30- |
| `calorie_range` | 5 | port_with_fixes | moderate | `problem-generator-variants` | Unit/scale mismatch: calorie window (1800-2500) is hardcoded in absolute units unrelated to the generated nutrient_content magnitudes, so the lower bound min_ca |

#### energy
| variant | score | status | distinct | branch | reason |
|---|---|---|---|---|---|
| `unit_commitment` | 6 | port_with_fixes | substantive | `problem-generator-variants` | min_down_time is computed and stored but never used in build_model (dead constraint / unused field) — incomplete vs the documented variant |
| `min_emissions` | 5 | port_with_fixes | moderate | `problem-generator-variants` | Objective frequently hits 0 with a large degenerate optimal face (any allocation among zero-emission sources), making it a weak optimization instance |
| `standard` | 4 | duplicate | cosmetic | `problem-generator-variants` | Duplicates the existing energy/:standard variant on main |
| `curtailment` | 3 | reject | cosmetic | `problem-generator-variants` | Curtailment is pure cost with zero benefit, so optimal curtailment is always 0 — the variant adds variables that contribute nothing and is effectively identical |
| `energy/scenario_mix` | 3 | duplicate | cosmetic | `add-variants-for-energy-pr` | DEGENERATE EMISSION CONSTRAINT (serious): build_model uses max_emission = maximum(values(emission_limits)) and constrains sum_s emission_limits[s]*x[s,t] <= max |

#### facility_location
| variant | score | status | distinct | branch | reason |
|---|---|---|---|---|---|
| `standard` | 6 | duplicate | cosmetic | `problem-generator-variants` | Duplicates the existing facility_location/:standard variant already in main. |
| `uncapacitated` | 6 | port_with_fixes | moderate | `problem-generator-variants` | Per-facility big-M = sum of ALL demand is loose; weakens LP relaxation (textbook UFL uses the disaggregated x[w,c]<=d_c*y_w which is much tighter and more reali |
| `p_median` | 6 | port_with_fixes | substantive | `problem-generator-variants` | Assignment uses >=1 with continuous x and only an UPPER capacity bound; since objective minimizes positive-cost assignment, optimum sets sum_w x[w,c]=1, so >= i |
| `single_source` | 6 | port_with_fixes | substantive | `problem-generator-variants` | All-binary z[w,c] makes this a pure 0-1 program of size ~n_facilities*(n_customers+1) binaries -- can be a hard MIP at large target sizes, but valid. |
| `set_covering_location` | 5 | port_with_fixes | substantive | `problem-generator-variants` | Variable count = n_facilities ONLY (no x), so it badly UNDERSHOOTS target_variables (sizing heuristic assumes n_facilities*(n_customers+1) vars). For target=100 |
| `p_center` | 3 | reject | substantive | `problem-generator-variants` | Continuous assignment makes the p-center max-distance constraint meaningless (can be driven toward 0), so the model does not actually compute the p-center objec |

#### inventory
| variant | score | status | distinct | branch | reason |
|---|---|---|---|---|---|
| `standard` | 6 | duplicate | substantive | `problem-generator-variants` | Duplicates the existing inventory/:standard variant in main; this is the same single-item lot-sizing model. |
| `safety_stock` | 6 | port_with_fixes | moderate | `problem-generator-variants` | Distinctiveness is moderate: only adds a lower bound on inventory; otherwise identical to standard. |
| `warehouse_capacity` | 6 | port_with_fixes | moderate | `problem-generator-variants` | Adds an inventory upper bound, a genuine structural change vs standard. |
| `service_level` | 6 | port_with_fixes | substantive | `problem-generator-variants` | Modeling redundancy: backlog (I_minus) and limited sales coexist; with sales capped at demand and a per-period fill-rate floor, backlog dynamics are somewhat od |
| `perishable` | 2 | reject | substantive | `problem-generator-variants` | build_model declares I[1:n_periods, 0:n_periods] = O(T^2) variables while the constructor sizes n_periods = target/4 assuming O(T) variables, so the actual vari |

#### knapsack
| variant | score | status | distinct | branch | reason |
|---|---|---|---|---|---|
| `multidimensional` | 6 | port_with_fixes | substantive | `problem-generator-variants` | Infeasible handling multiplies one resource capacity by 0.05. With continuous x>=0 and x=0 always feasible, scaling a capacity DOWN never makes the model infeas |
| `bounded` | 6 | port_with_fixes | substantive | `problem-generator-variants` | Infeasible handling sets capacity = 1.5*sum(weights.*upper_bounds) AND min_fill_fraction=1.1, then build_model adds sum(weights*x) >= capacity*1.1. Required wei |
| `dependency` | 6 | port_with_fixes | substantive | `problem-generator-variants` | Infeasible handling pushes circular deps (1->2 and 2->1) i.e. x[1]<=x[2] and x[2]<=x[1]. This only forces x[1]==x[2]; it does NOT make the model infeasible (x[1 |
| `group-cardinality` | 6 | port_with_fixes | substantive | `problem-generator-variants` | Infeasible handling sets group_min[target] = group_counts[target]+1 (require more items than exist in the group). Since sum over the group <= count < count+1, t |
| `cargo_loading` | 6 | port_with_fixes | substantive | `add-transportation-generat` | WRONG CATEGORY: assigned to transportation but this is a bin-packing/knapsack-with-containers problem, not a source-to-destination transportation flow. Belongs  |
| `conflict` | 5 | port_with_fixes | substantive | `problem-generator-variants` | Infeasible handling is a NO-OP: the if-block body is only a comment ('Create a conflict cycle...'); nothing is generated. With x=0 always feasible (conflict con |
| `standard` | 4 | duplicate | cosmetic | `problem-generator-variants` | Duplicates the existing knapsack/:standard variant in main. |
| `min-fill` | 4 | reject | moderate | `problem-generator-variants` | Often degenerate vs standard: maximizing value tends to fill the knapsack, so the min-fill lower bound is usually slack and the optimum equals the plain standar |
| `multiple-knapsacks` | 3 | reject | substantive | `problem-generator-variants` | target_variables mismatch: n_items is set = target_variables, but the model has n_items*n_knapsacks variables, so a request for ~100 vars yields 200-500. Seriou |

#### network_flow
| variant | score | status | distinct | branch | reason |
|---|---|---|---|---|---|
| `multi_source_sink_flow` | 6 | port_with_fixes | substantive | `problem-generator-variants` | A node that is both not in sources and not in sinks but lies on a supply/demand path is fine, but nodes selected as sources can also have incoming arcs whose fl |
| `generalized_flow` | 6 | port_with_fixes | substantive | `problem-generator-variants` | Objective maximizes source OUTFLOW, not delivered flow at the sink; with gains/losses the meaningful quantity is sink inflow. Maximizing source-out under gains  |
| `time_expanded_flow` | 6 | port_with_fixes | substantive | `problem-generator-variants` | Arcs are treated as instantaneous (no transit time / no flow[arc,t] arriving at t+1); a true time-expanded network usually has travel-time lags. As written it i |
| `min_max_utilization_flow` | 6 | port_with_fixes | substantive | `problem-generator-variants` | This is the ONLY max/min variant with a real forced lower bound (source-out >= min_flow), so its infeasible case (max_util<=0.01 vs required 10% throughput) is  |
| `logistics` | 6 | port_with_fixes | substantive | `add-variants-to-network_fl` | feasibility_status is ignored entirely — feasible/infeasible are no-ops; infeasible never produces an infeasible model. |
| `urban_water` | 6 | port_with_fixes | substantive | `add-variants-to-network_fl` | feasibility_status ignored — feasible/infeasible no-ops. |
| `power_transmission` | 6 | port_with_fixes | substantive | `add-variants-to-network_fl` | feasibility_status ignored — no-op for feasible/infeasible. |
| `min_cost_flow` | 5 | duplicate | cosmetic | `problem-generator-variants` | Likely duplicates main's network_flow/:standard (whichever objective standard uses). |
| `node_capacitated_flow` | 5 | port_with_fixes | substantive | `problem-generator-variants` | infeasible is a NO-OP: like max_flow, the max objective has no forced lower bound, so flow=0 is always feasible. Scaling node capacities by 0.1 only reduces the |
| `max_flow` | 4 | duplicate | cosmetic | `problem-generator-variants` | Almost certainly duplicates main's existing network_flow/:standard. |
| `reliable_flow` | 3 | reject | moderate | `problem-generator-variants` | BROKEN vs stated concept: there is NO arc-disjointness constraint. The k flows merely share arc capacity, so the model is equivalent to a single max-flow split  |
| `standard` | 3 | duplicate | cosmetic | `add-variants-to-network_fl` | Duplicates main's existing network_flow/standard variant (same code path). |

#### portfolio
| variant | score | status | distinct | branch | reason |
|---|---|---|---|---|---|
| `tracking_error` | 6 | port_with_fixes | substantive | `problem-generator-variants` | No risk-free asset and no risk-budget constraint here, so it is purely return-max under TE budget; fine but differs structurally from siblings. |
| `sector_limits` | 6 | port_with_fixes | substantive | `problem-generator-variants` | All sectors get the SAME limit (fill of a single draw) -> less diverse than per-sector limits. |
| `cardinality_constrained` | 6 | port_with_fixes | substantive | `problem-generator-variants` | infeasible sets max_assets=0 -> no risky asset can be held (all y=0 forces x=0), but x_rf is unconstrained by y, so the model is FEASIBLE (everything in risk-fr |
| `turnover_constrained` | 6 | port_with_fixes | substantive | `problem-generator-variants` | infeasible sets max_turnover=0.0; with buy/sell continuous and x = prev*total achievable, turnover 0 is satisfiable (hold the previous portfolio) UNLESS prev*to |
| `risk_budget` | 5 | port_with_fixes | moderate | `problem-generator-variants` | Variable scaling: n_options = target_variables but x_rf adds 1 extra variable; close enough for this variant. |
| `cvar_simple` | 5 | duplicate | substantive | `problem-generator-variants` | Concept duplicates main's portfolio/:cvar, which is more sophisticated -> redundant. |
| `esg_constrained` | 5 | port_with_fixes | moderate | `problem-generator-variants` | ESG constraint divides only over invested risky x; if all risky x -> 0 (everything to x_rf), constraint becomes 0 >= 0 (trivially satisfied) -> so the ESG floor |
| `mean_variance_diagonal` | 3 | reject | cosmetic | `problem-generator-variants` | Off-diagonal covariance computed then thrown away; 'mean-variance' is a misnomer for a diagonal linear penalty. |
| `min_risk_target_return` | 3 | reject | cosmetic | `problem-generator-variants` | Diagonal-only covariance: not actual variance minimization. |
| `max_sharpe_approx` | 2 | reject | cosmetic | `problem-generator-variants` | Variable k is declared but never used (dead variable) -> wrong/abandoned formulation. |

#### production_planning
| variant | score | status | distinct | branch | reason |
|---|---|---|---|---|---|
| `multi_period_inventory` | 7 | port_with_fixes | substantive | `problem-generator-variants` | n_periods derived from target_variables ÷ n_products where n_products=min(2000,target_variables); since n_products≈target_variables, target_variables÷n_products |
| `fixed_setup_costs` | 6 | port_with_fixes | substantive | `problem-generator-variants` | Big-M: M = sum(resources)/minimum(usage). minimum(usage) can be ~0.01 for large instances, making M ~ 1e6+, a loose Big-M that causes weak LP relaxation / numer |
| `min_demand_satisfaction` | 6 | port_with_fixes | moderate | `problem-generator-variants` | max_possible = resources ./ maximum(usage,dims=1)[:] is a length-n_resources vector (per-resource max throughput), then avg_possible = sum(max_possible)/n_produ |
| `machine_eligibility_assignment` | 6 | port_with_fixes | substantive | `problem-generator-variants` | Variable matrix x[i,m] is n_products*n_machines but most entries forced to 0 (only 1-3 per product nonzero). The model still CREATES all n_products*n_machines v |
| `overtime_capacity` | 5 | port_with_fixes | substantive | `problem-generator-variants` | Overtime penalty is a SCALAR (avg_profit*(mult-1)) applied uniformly to all resources' overtime, not tied to the actual cost of each resource. Crude but accepta |
| `quality_grade_yields` | 5 | port_with_fixes | substantive | `problem-generator-variants` | q variables are fully DETERMINED by x via equality q[i,lev]=yield*x_i, so they are not real decisions — the model is mathematically equivalent to standard produ |
| `standard` | 4 | duplicate | cosmetic | `problem-generator-variants` | Duplicates the existing production_planning/:standard variant in main (same max-profit-subject-to-resources LP); also overlaps heavily with product_mix category |
| `discrete_batch_production` | 3 | reject | moderate | `problem-generator-variants` | x_i is a pure alias of batch_size_i*n_batches_i (equality), so it adds n_products redundant continuous variables that inflate the count without adding structure |

#### scheduling
| variant | score | status | distinct | branch | reason |
|---|---|---|---|---|---|
| `standard` | 7 | duplicate | cosmetic | `problem-generator-variants` | Pure duplicate of the existing scheduling/standard variant on main. |
| `weekend_fair` | 6 | port_with_fixes | substantive | `problem-generator-variants` | Weekend detection d%7 in [0,6] is offset from a real Mon-start week (flags day6, day7, day13... ) - harmless but loose. |
| `overtime` | 6 | port_with_fixes | substantive | `problem-generator-variants` | Drops the max-consecutive-days constraint that standard has. |
| `preferences` | 6 | port_with_fixes | substantive | `problem-generator-variants` | (1-preferences) is constant and pref_violation is only lower-bounded+minimized, so it collapses to x*(1-pref); the auxiliary variable is redundant and could be  |
| `team` | 6 | port_with_fixes | substantive | `problem-generator-variants` | Generic feasibility enforcement ignores team-cohesion constraints, so a 'feasible'-tagged instance can be infeasible when team_size_min exceeds available member |
| `on_call` | 6 | port_with_fixes | substantive | `problem-generator-variants` | Generic feasibility enforcement ignores on_call_req coverage and the combined one-role-per-day cap, so 'feasible'-tagged instances may be infeasible (a worker c |
| `nurse` | 6 | port_with_fixes | substantive | `add-problem-variants-01PYi` | Uses OLD flat registry: register_problem(:scheduling_nurse, ...) instead of register_variant(:scheduling, :nurse, NurseScheduling, ...). |
| `or` | 6 | port_with_fixes | substantive | `add-problem-variants-01PYi` | CONSTRUCTOR CRASH RISK: surgeon-availability block uses block_start = rand(1:(n_time_blocks - surgery_durations[i])). For long surgery types (neurosurgery dur u |
| `split_shift` | 5 | port_with_fixes | moderate | `problem-generator-variants` | Structural change is modest - mainly relaxes the per-day shift bound and re-expresses consecutive-day counting; borderline cosmetic-to-moderate. |
| `seniority` | 3 | reject | cosmetic | `problem-generator-variants` | Constraint structure is exactly standard; only a per-worker constant subtracted from cost - cosmetic objective tweak, not a structural variant. |

#### supply_chain
| variant | score | status | distinct | branch | reason |
|---|---|---|---|---|---|
| `standard` | 6 | duplicate | cosmetic | `problem-generator-variants` | This is the canonical supply_chain standard and overlaps the existing main :standard variant; as the baseline it is a duplicate of the existing standard generat |
| `risk_diverse` | 6 | port_with_fixes | substantive | `problem-generator-variants` | Region assignment is a trivial round-robin index modulo, ignoring the rich spatial coordinates already generated; regions have no geographic meaning. |
| `multi_echelon` | 6 | port_with_fixes | substantive | `supply-chain-problem-varia` | Registered via OLD register_problem as a single :supply_chain type; variant is chosen randomly INSIDE the constructor (sample over 5 variants), so the user cann |
| `global_tariff` | 6 | port_with_fixes | substantive | `supply-chain-problem-varia` | Same OLD single-type registration / internal random variant selection problem as all variants here. |
| `lead_time` | 4 | reject | cosmetic | `problem-generator-variants` | The lead-time 'constraint' only fixes variables to zero, which is degenerate modeling (equivalent to removing those variables) and adds no genuine constraint st |
| `ecommerce_fulfillment` | 4 | port_with_fixes | substantive | `supply-chain-problem-varia` | BROKEN/degenerate service-level constraint (build_ecommerce_fulfillment_model lines ~1351-1360): for each customer it enforces sum over nearby open FCs of flow  |
| `multi_period_inventory` | 4 | port_with_fixes | substantive | `supply-chain-problem-varia` | SERIOUS modeling bug in inventory balance (line ~1405): inv[f,t-1] + capacities[f]*y[f,t] == outflow + inv[f,t]. Production is NOT a decision variable; it is fo |
| `make_or_buy` | 3 | reject | substantive | `supply-chain-problem-varia` | BROKEN flow balance (line ~1480): procurement is allocated to facilities as received = total_procured / n_facilities, a hardcoded equal split baked into constra |

#### transportation
| variant | score | status | distinct | branch | reason |
|---|---|---|---|---|---|
| `warehouse_location` | 7 | port_with_fixes | substantive | `add-transportation-generat` | Uses OLD flat registration API register_problem(:warehouse_location, ...) instead of new register_variant(:transportation, :warehouse_location, WarehouseLocatio |
| `standard` | 6 | duplicate | cosmetic | `problem-generator-variants` | Pure duplicate of existing transportation/standard variant in main (same constraints/objective). |
| `service_level` | 6 | port_with_fixes | substantive | `problem-generator-variants` | unsatisfied[j] carries NO penalty in the objective, so the model simply ships the cheapest flows to just meet the global service level — fine but the 'unsatisfi |
| `cross_docking` | 6 | port_with_fixes | substantive | `add-transportation-generat` | Uses OLD flat registration: register_problem(:cross_docking, ...). Must become register_variant(:transportation, :cross_docking, CrossDocking, "...") for the ne |
| `hub_location` | 6 | port_with_fixes | substantive | `add-transportation-generat` | Uses OLD flat registration: register_problem(:hub_location, ...) instead of the new register_variant(:transportation, :hub_location, HubLocation, ...). |
| `multi_commodity` | 4 | duplicate | moderate | `problem-generator-variants` | No coupling constraint (shared arc/source capacity) -> separable into k single-commodity TPs; not a genuine multi-commodity formulation. |
| `transportation/diverse_data` | 4 | duplicate | cosmetic | `add-variants-to-transporta` | Pure formulation duplicate of existing transportation/standard — identical objective and constraints; adds zero new OR structure (no arc capacities, no transshi |
| `time_windows` | 3 | reject | moderate | `problem-generator-variants` | No binary 'route used' variable; the big-M term uses a continuous flow fraction, so the if-used logic is not correctly enforced — a fully-loaded route barely ac |
| `vehicle_routing` | 3 | reject | substantive | `add-transportation-generat` | Fundamental modeling mismatch: CVRP is intrinsically a MIP requiring binary arc/assignment vars and subtour-elimination. Here everything is continuous in [0,1]  |

#### vehicle_routing
| variant | score | status | distinct | branch | reason |
|---|---|---|---|---|---|
| `standard` | 6 | port_with_fixes | substantive | `vehicle-routing-generator-` | WEAK/DEGENERATE RELAXATION: the depot node has NO flow-conservation constraint tying customer routes to the depot. Constraints only force, per vehicle, customer |
| `last_mile_delivery` | 5 | port_with_fixes | substantive | `add-transportation-generat` | OLD registration: uses register_problem(:last_mile_delivery, ...) flat API; must become register_variant(:transportation OR :vehicle_routing, :last_mile_deliver |
