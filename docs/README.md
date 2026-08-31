# SyntheticLPs Generator Documentation

This directory collects detailed pages for the documented synthetic LP/MIP
generator categories under `src/problem_types/`. Categories (problem domains,
one folder each) group one or more **variants** (concrete formulations). Every
variant follows the same package-level contract: the constructor samples all
randomized data from `target_variables`, `feasibility_status`, and `seed`, stores
that data in a concrete `ProblemGenerator` struct, and `build_model` converts the
stored data into a deterministic JuMP model.

For a browsable, high-level tour of the documented generators, open the
self-contained [HTML explainer](explainer.html) (no server or internet required).
It is generated from these Markdown pages by `scripts/build_explainer.py`. After
changing a generator page or the script's `META` catalog, run
`python3 scripts/build_explainer.py` and commit the rebuilt HTML.

## Shared Interface

Use `generate_problem(problem_sym, target_variables, feasibility_status, seed)`
to build an instance. `target_variables` is interpreted separately by each
generator, usually by choosing dimensions whose product or sum approximates the
requested variable count. Passing the same `seed` to the same generator should
produce the same data dimensions and model structure.

The supported feasibility controls are:

- `feasible`: the generator adjusts capacities, demands, budgets, or other
  bounds to make at least one feasible solution likely or explicitly
  constructed.
- `infeasible`: the generator deliberately tightens a binding resource,
  requirement, or budget so the resulting model should be infeasible.
- `unknown`: the generator samples a realistic random instance, often with no
  guarantee either way.

Several generators declare binary variables because the natural problem is a
mixed-integer model. The public `generate_problem` function defaults to
`relax_integer=true`, so those binary variables are relaxed unless the caller
opts out. The documentation pages describe the intended formulation and note
where relaxation changes the solved model.

## Problem Type Pages

- [Airline Crew](airline_crew.md)
- [Assignment](assignment.md)
- [Bin Packing](bin_packing.md)
- [Blending](blending.md)
- [Crop Planning](crop_planning.md)
- [Cutting Stock](cutting_stock.md)
- [Diet Problem](diet_problem.md)
- [Energy](energy.md)
- [Facility Location](facility_location.md)
- [Feed Blending](feed_blending.md)
- [Hub Location](hub_location.md)
- [Inventory](inventory.md)
- [Knapsack](knapsack.md)
- [Land Use](land_use.md)
- [Load Balancing](load_balancing.md)
- [Multi-Commodity Flow](multi_commodity_flow.md)
- [Network Flow](network_flow.md)
- [Operating Room Scheduling](operating_room_scheduling.md)
- [Portfolio](portfolio.md)
- [Product Mix](product_mix.md)
- [Production Planning](production_planning.md)
- [Project Selection](project_selection.md)
- [Radiotherapy Fluence-Map Planning](radiotherapy.md)
- [Resource Allocation](resource_allocation.md)
- [Revenue Management](revenue_management.md)
- [Scheduling](scheduling.md)
- [Supply Chain](supply_chain.md)
- [Telecom Network Design](telecom_network_design.md)
- [Transportation](transportation.md)
- [Traveling Salesperson](tsp.md)
- [Unit Commitment](unit_commitment.md)
- [Workforce Shift Scheduling](workforce_shift_scheduling.md)
