# Production Planning

Multi-level, multi-period capacitated production planning (MRP II). The
category has one variant, `standard`: a bill-of-materials (BOM) explosion over a
planning horizon with lead times, end-product backlog, work-center capacity with
bounded overtime, and supplier capacity. It is a pure continuous LP with the
classical staircase structure — inventory chains linked across periods, coupled
across items by the BOM and shared capacity rows — that makes such LPs hard for
simplex. Compare `product_mix`, which is single-period (routing choice over many
resources, no inventory dynamics).

The previous `standard` was a dense single-period profit-maximisation over at
most 2,000 products and 5–50 resource rows (a strict subset of `product_mix`);
it was rebuilt from scratch.

## Variants

| Variant | Model class | Key structure | Domain grounding |
|---|---|---|---|
| `standard` | continuous LP | BOM balance staircase with lead-time offsets, capacity + overtime, supplier rows | discrete manufacturing MRP: end products, subassemblies, components, purchased materials |

## Formulation

Items `i` are indexed level by level (end products first, purchased raw materials
last), so every BOM edge `(parent p, child c, quantity a)` has `p < c`.
`L_i` is the lead time; `I0_i` the initial stock.

Columns:

```text
x[i, t] >= 0    release of item i in period t (production start / purchase order),
                t = 1 .. T - L_i (arrives in t + L_i)
I[i, t] >= 0    end-of-period inventory, t = 1 .. T
B[e, t] >= 0    backlog of end product e, t = 1 .. T - 1 (cleared by the horizon end)
O[w, t] in [0, max_overtime[w, t]]   overtime hours at work center w
```

Rows:

```text
I[i,t-1] + x[i,t-L_i] - sum_p a[i,p] x[p,t] - d[i,t] + B[i,t] - B[i,t-1] = I[i,t]
                                      (balance, every item and period; I[i,0] = I0_i;
                                       backlog terms only for end products)
sum_{i at w} run_time[i] x[i,t] - O[w,t] <= regular_capacity[w,t]      (work centers)
sum_{i from s} volume[i] x[i,t] <= supplier_capacity[s,t]              (suppliers)
```

Objective (minimize): value-added labor cost on manufactured releases, purchase
prices on raw materials, holding cost on inventory, backlog penalties on end
products, and overtime premiums.

## Data Grounding

- Product structure: 4 levels (3 for tiny instances) with shares 15/25/35/25%.
  Items belong to product families of 16–40 items; children are drawn from the
  parent's family with a lognormal popularity (shared common parts), 8% of
  picks go to plant-wide common parts, and 20% of parents also use a material
  two levels down (e.g. packaging on an end product). Every non-end item has a
  parent.
- Quantities: end product → subassembly 1–2, → component 1–4, → raw material a
  lognormal amount (kg, metres).
- Lead times: 0–1 period for manufactured items, 1–3 for purchased ones.
- Cells and work centers: families are grouped 2–3 per manufacturing cell with
  its own work centers per level (common parts in a shared cell); run times are
  lognormal by level. Suppliers serve one cell (plus one plant-wide supplier).
- Demand: end products have seasonal (26-period cycle) demand with trend and
  Gamma noise, 15% of them intermittent; 15% of subassemblies/components carry
  service-part demand.
- Economics: item values roll up through the BOM (children plus 1.6× labor
  content); weekly carrying rate 18–35%/52; backlog penalties 4–12% of value per
  period; overtime 1.5× the labor rate.
- Capacity: regular hours are flat per work center around the planted plan's
  average load (a quarter of the centers are bottlenecks at 0.88–1.02× it, the
  rest 1.12–1.45×), with holiday dips; overtime covers peaks (at least 20% of
  regular hours).

## Feasibility Control

- `feasible`: the lot-for-lot MRP explosion (initial stock covers the lead-time
  gap; safety stock retained; overtime only where the lumpy load exceeds
  regular hours) is planted as `ProductionPlanWitness(production, inventory,
  overtime)`.
- `infeasible`: the most loaded work center loses capacity (regular hours and
  overtime scaled) until its echelon load exceeds its horizon capacity by
  8–30%. `EchelonCapacityCertificate(work_center, items, lower_bounds,
  required_load, available_capacity)`: summing each item's balance rows over
  the horizon gives `sum_t x[i,t] >= LB_i := max(0, sum_p a[i,p] LB_p + D_i -
  I0_i)` by induction down the BOM, so the center needs `sum run_time[i] LB_i`
  hours; its capacity rows plus overtime bounds supply fewer. The argument uses
  balance rows of every upstream item and all capacity rows of the center —
  presolve does not see it.
- `unknown`: the same scaling with ratio `1 ± U(0.03, 0.30)`; above 1 provably
  infeasible, below 1 decided by timing.

## Sizing

```text
columns = sum_i (T - L_i) + n_items * T + n_end * (T - 1) + n_work_centers * T
rows    = n_items * T + n_work_centers * T + (supplier, period) rows with an order column
```

`T` is 3–8 for tiny targets, 6–10 up to 1.5k, 10–20 up to 20k, 16–30 beyond;
`n_items ≈ target / (2.14 T)`. The column count lands within a few percent of
the target from ~100 to 100k+ (rows ≈ 50% of columns). Build time is well under
a second at 100k. Solve time grows quickly with capacity tightness — the
coupled staircase is the point — 100k feasible instances solve in ~7–30 s with
HiGHS; HiGHS's dual-ray recomputation after proving infeasibility can add
substantial time on large infeasible instances.

## References

- Orlicky, J. (1975). Material Requirements Planning. McGraw-Hill.
- Billington, P.J., McClain, J.O., Thomas, L.J. (1983). Mathematical
  programming approaches to capacity-constrained MRP systems. Management
  Science 29(10).
- Pochet, Y., Wolsey, L.A. (2006). Production Planning by Mixed Integer
  Programming. Springer.
