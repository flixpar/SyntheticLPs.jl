# SyntheticLPs Generator Documentation

This directory collects one page per problem category under `src/problem_types/`
(50 categories, 142 variants). Categories (problem domains, one folder each) group
one or more **variants** (concrete formulations). Every variant follows the same
package-level contract: the constructor samples all randomized data from
`target_variables`, `feasibility_status`, and `seed` using a constructor-local
`MersenneTwister(seed)`, stores that data in a concrete `ProblemGenerator` struct,
and `build_model` converts the stored data into a deterministic JuMP model.

For a browsable, high-level tour of the generators, open the self-contained
[HTML explainer](explainer.html) (no server or internet required). It is generated
from these Markdown pages by `scripts/build_explainer.py`. After adding or
changing a page, or the script's `META`/`FAMILIES` catalog, run
`python3 scripts/build_explainer.py` and commit the rebuilt HTML; the build fails
if a page has no `META` entry or vice versa.

## Shared Interface

```julia
model, problem = generate_problem(:category, target_variables, feasibility_status, seed)
model, problem = generate_problem("category/variant", target_variables, feasibility_status, seed)
```

A bare category uses its default variant (listed first below). `target_variables`
is interpreted by each generator, usually by choosing dimensions whose product or
sum approximates the requested count; the variable count tracks the request from
tiny instances to 100k+ variables (within a few percent from 1k up), and a few
generators document a hard cap (see `problem_info(ref)[:max_target_variables]`).
The same `seed` always produces the same data and model.

The supported feasibility controls are:

- `feasible`: the generator constructs a feasible solution and sizes capacities,
  demands, or budgets around it. Most generators store it as a typed
  `feasible_witness` on the problem struct.
- `infeasible`: the generator plants a contradiction that a solver must work to
  find — a deficit spread across many rows (a cut, a capacity prefix, a regional
  shortage), never a single contradictory row or bound — and most store a typed
  `infeasibility_certificate` that proves it without a solver.
- `unknown`: the generator samples a realistic instance with no planted outcome.

Each page names its witness and certificate types. Callers who need a guarantee
can pass `optimizer=...` to `generate_problem` to verify the requested status by
solving (see the top-level [README](../README.md#solver-verification)).

Many generators declare binary or integer variables because the natural problem
is a mixed-integer model. `generate_problem` defaults to `relax_integer=true`, so
those variables are relaxed unless the caller opts out; `model_class(ref)` reports
whether a variant's `build_model` is `:lp` or `:mip`. The pages describe the
intended formulation and note where relaxation changes the solved model.
Post-build transforms (`bounds_to_constraints`, `ModelTransforms`, dualization)
are documented in the top-level README and apply to every category alike.

## Problem Type Pages

Each entry lists the category's variants, default first.

- [Airline Crew](airline_crew.md) — `standard`
- [Assignment](assignment.md) — `standard`, `workload_balance`
- [Bin Packing](bin_packing.md) — `standard`, `heterogeneous`
- [Blending](blending.md) — `standard`, `multi_period`, `robust`
- [Container Loading](container_loading.md) — `standard`, `two_dimensional_bin_packing`
- [Crop Planning](crop_planning.md) — `standard`
- [Cutting Stock](cutting_stock.md) — `standard`, `arc_flow`, `due_dates`, `setup_cost`
- [Diet Problem](diet_problem.md) — `standard`, `food_aid`, `food_groups`
- [Economic Planning](economic_planning.md) — `dynamic_leontief`, `energy_system`
- [Energy](energy.md) — `standard`, `dc_opf`, `hydrothermal`, `reserves`, `security_constrained_dc_opf`, `storage`
- [Facility Location](facility_location.md) — `standard`, `p_median`, `two_echelon`
- [Feed Blending](feed_blending.md) — `standard`
- [Forest Planning](forest_planning.md) — `model_i`, `model_ii`
- [Game Theory](game_theory.md) — `poker_sequence_form`, `colonel_blotto`, `patrol_security`
- [Graph Optimization](graph_optimization.md) — `independent_set`, `generalized_independent_set`, `map_labeling`, `quasi_clique`, `vertex_coloring`, `vertex_cover`
- [Hub Location](hub_location.md) — `p_hub_median`, `budgeted_backbone`, `capacitated`, `compact_single_allocation`, `hub_covering`, `hub_network`, `multiple_allocation`, `r_allocation`
- [Inventory](inventory.md) — `standard`, `lot_sizing`, `multi_echelon`, `multi_item`
- [Inverse Optimization](inverse_optimization.md) — `standard`, `classical_normalized`, `linf`, `market_clearing`, `noisy_observations`, `restricted_optimal_value`, `shortest_path`, `shortest_path_layered`
- [Job Shop Scheduling](job_shop_scheduling.md) — `standard`
- [Knapsack](knapsack.md) — `multiple_choice`, `bounded`, `mixed_integer_set`, `multidimensional`
- [Land Use](land_use.md) — `standard`
- [Load Balancing](load_balancing.md) — `standard`, `discrete_placement`
- [Maritime Inventory Routing](maritime_inventory_routing.md) — `standard`
- [Markov Decision Process](markov_decision_process.md) — `inventory_control`, `constrained`, `machine_maintenance`, `queueing_control`
- [Mine Planning](mine_planning.md) — `cpit`, `pcpsp`, `stockpile`
- [Multi-Commodity Flow](multi_commodity_flow.md) — `standard`, `binary_capacity`
- [Network Flow](network_flow.md) — `standard`, `generalized_flow`, `time_expanded`
- [Neural Network Verification](neural_network_verification.md) — `relu_big_m`
- [Nurse Scheduling](nurse_scheduling.md) — `standard`
- [Operating Room Scheduling](operating_room_scheduling.md) — `elective_assignment`, `benchmark_loading`, `case_sequencing`, `master_surgical_schedule`, `robust_elective`, `weekly_planning`
- [Portfolio](portfolio.md) — `cvar`, `tracking_error`
- [Process Planning](process_planning.md) — `refinery`, `campaign`, `capacity_expansion`, `hydrogen_network`, `mode_switching`
- [Product Mix](product_mix.md) — `standard`
- [Production Planning](production_planning.md) — `standard`
- [Project Selection](project_selection.md) — `standard`
- [Radiotherapy Fluence-Map Planning](radiotherapy.md) — `weighted_deviation`, `beam_angle_selection`, `mean_tail_dose`, `minmax_deviation`, `robust_fluence`
- [Regression](regression.md) — `lad`, `basis_pursuit`, `chebyshev`, `l1_svm`, `quantile`
- [Resilient Network Design](resilient_network_design.md) — `standard`
- [Resource Allocation](resource_allocation.md) — `standard`
- [Revenue Management](revenue_management.md) — `standard`, `stochastic_overbooking`
- [Scheduling](scheduling.md) — `standard`
- [Set Systems](set_system.md) — `set_cover`, `combinatorial_auction`, `set_packing`, `set_partitioning`
- [Stochastic Program](stochastic_program.md) — `standard`, `multistage_alm`
- [Supply Chain](supply_chain.md) — `standard`, `carbon`, `multi_product`, `network_planning`, `single_source`
- [Telecom Network Design](telecom_network_design.md) — `standard`
- [Transportation](transportation.md) — `standard`, `emission_constrained`, `fixed_charge`, `transshipment`
- [Traveling Salesperson](tsp.md) — `standard`, `assignment_relaxation`, `asymmetric`, `flow`, `multiple_salespersons`, `precedence`, `prize_collecting`, `time_windows`
- [Unit Commitment](unit_commitment.md) — `standard`
- [Vehicle Routing](vehicle_routing.md) — `cvrp`
- [Workforce Shift Scheduling](workforce_shift_scheduling.md) — `covering`
