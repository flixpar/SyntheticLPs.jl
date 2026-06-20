# SyntheticLPs Generator Documentation

This directory documents each synthetic LP generator implemented under
`src/problem_types/`. Generators are organized as **categories** (problem
domains, one folder each under `src/problem_types/<category>/`) that group one or
more **variants** (concrete formulations, one file each). Every variant follows
the same package-level contract: the constructor samples all randomized data from
`target_variables`, `feasibility_status`, and `seed`, stores that data in a
concrete `ProblemGenerator` struct, and `build_model` converts the stored data
into a deterministic JuMP model.

For a browsable, high-level tour of all generators alongside these details, open
the self-contained [HTML explainer](explainer.html) (no server or internet
required). It is generated from these markdown pages by
`scripts/build_explainer.py`.

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
- [Blending](blending.md)
- [Crop Planning](crop_planning.md)
- [Cutting Stock](cutting_stock.md)
- [Diet Problem](diet_problem.md)
- [Energy](energy.md)
- [Facility Location](facility_location.md)
- [Feed Blending](feed_blending.md)
- [Inventory](inventory.md)
- [Knapsack](knapsack.md)
- [Land Use](land_use.md)
- [Load Balancing](load_balancing.md)
- [Multi-Commodity Flow](multi_commodity_flow.md)
- [Network Flow](network_flow.md)
- [Portfolio](portfolio.md)
- [Product Mix](product_mix.md)
- [Production Planning](production_planning.md)
- [Project Selection](project_selection.md)
- [Resource Allocation](resource_allocation.md)
- [Scheduling](scheduling.md)
- [Supply Chain](supply_chain.md)
- [Telecom Network Design](telecom_network_design.md)
- [Transportation](transportation.md)
