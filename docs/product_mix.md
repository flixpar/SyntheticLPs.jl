# Product Mix

Single-period product mix with alternative routings over a shop of many
machines, department labor pools, and materials. The category has one variant,
`standard`, a pure continuous LP. It is the single-period counterpart of
`production_planning` (which adds the multi-period BOM staircase): the
structure here comes from routing choice and many shared resources.

The previous `standard` had one column per product and at most 30 resource rows
plus singleton market rows; HiGHS presolve dissolved every instance (0% of rows
kept) and capped silently at 10,000 products. It was rebuilt.

## Variants

| Variant | Model class | Key structure | Domain grounding |
|---|---|---|---|
| `standard` | continuous LP | one column per routing; machine, labor, material, and ranged market rows | discrete manufacturing: metalworking, electronics, furniture, food, chemicals, automotive |

## Formulation

Columns `x[r] >= 0`, the quantity made by routing `r` of product `p(r)`.

```text
sum_r time[r,m] x[r] <= machine_capacity[m]                          (each machine)
sum_r sum_{steps of r in d} labor[r,step] x[r] <= labor_capacity[d]  (each department)
sum_r qty[p(r),k] * yield[r] x[r] <= material_capacity[k]            (each material)
floor[p] <= sum_{r of p} x[r] <= ceiling[p]                          (market, multi-routing
                                                                      products; variable bounds
                                                                      for single-routing ones)
```

Objective (maximize): contribution margin `price[p] - materials * yield[r] -
conversion cost[r]`, positive for every routing (no dual-fixable dead columns).

The market rows are genuine ranged rows (`MOI.Interval`), which the corpus
otherwise lacks.

## Data Grounding

- Industry regime (sampled): routing flexibility (probability of 2–3
  alternative routings), routing length (2–5 machines), processing-time scale,
  margin level.
- Shop: machines ≈ 5–9% of the routing count, grouped into departments of 4–9;
  departments are grouped into plant areas of 4–8, each with a shared
  finishing/packaging department. Every routing starts in its product's primary
  department; later steps stay there, visit the area's shared department (15%),
  or — for alternative routings — a sibling department's line (25%).
- Labor: operator hours per step = crew size × base time × a lognormal
  manual-content factor (independent of machine time, so labor rows are not
  combinations of machine rows).
- Materials: department stores, area stores, and a few plant-wide materials;
  routing yields 1.00–1.04 (preferred) or up to 1.18 (alternatives) scale
  material use.
- Prices: materials plus the dearest routing's conversion cost, times a
  lognormal margin.

## Feasibility Control

A nominal plan is sampled first (35–85% of each market ceiling, split across
routings); capacities are its consumption times heterogeneous headroom (a few
near-saturated bottlenecks), and 45% of products get a floor at 30–90% of their
planned output.

- `feasible`: `ProductMixPlanWitness(production, machine_hours, labor_hours,
  material_use)` — strict slack on every capacity row.
- `infeasible`: the department whose committed products weigh most on it raises
  its multi-routing products' floors to 60–95% of plan and loses machine
  capacity until the floors need 10–35% more hours than it has.
  `ProductMixDepartmentCertificate(department, machines, products, min_hours,
  required_hours, available_hours)`: each listed product's market row times the
  minimum hours any of its routings spends in the department, summed with the
  department's machine rows. Only multi-routing products' floors are raised
  (single-routing floors are variable bounds), and the cut never takes a
  machine below 1.3× what single-routing floors force onto it plus 1.3× the
  largest single commitment through it, so no single row or single product is
  contradicted. HiGHS presolve's bound propagation (machine slack → implied
  column bounds → market rows, cascading through a small department) still
  refutes roughly half of the instances at 10k; the rest need simplex work.
- `unknown`: the same mechanism with ratio `1 ± U(0.03, 0.30)`.

## Sizing

Columns are exactly `max(target, 2)` (products are added until the routing
budget is used; the last product's routing count is truncated). Rows are the
used machines, departments, and materials plus one market row per multi-routing
product — about 40–45% of the columns. 100k columns build in about a second;
HiGHS solves most 100k feasible instances in 10–50 s (block-angular but not
trivially decomposable).

## References

- Dantzig, G.B. (1963). Linear Programming and Extensions. Princeton University
  Press. (Product-mix LP.)
- Hax, A.C., Candea, D. (1984). Production and Inventory Management.
  Prentice-Hall. (Routing choice and aggregate planning.)
