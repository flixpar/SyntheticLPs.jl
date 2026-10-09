# Inventory

Multi-period inventory planning LPs. All four variants were rebuilt so that
each has a distinct, realistic structure that scales to 100k+ columns, keeps
most of the model through presolve, plants a typed witness for `feasible`
requests and an aggregate LP certificate (not a single-row contradiction) for
`infeasible` ones, and draws `unknown` as a genuine two-sided instance. All use
a constructor-local RNG; `build_model` does no sampling. Shared helpers
(`_inventory_demand`, `_inventory_scale_ratio`) live in `common.jl`.

## Variants

| Variant | Model class | Key structure | Domain grounding |
|---|---|---|---|
| `standard` (default) | continuous LP | SKU stock chains with lead times and lost sales, coupled by vendor allocation, zone storage, and receiving rows; service-level rows | distribution-center replenishment |
| `lot_sizing` | MIP (relaxed by default) | facility-location reformulation of capacitated lot sizing with setup times | multi-item batch production |
| `multi_echelon` | continuous LP | plant → DC → store network, backup lanes, transit times, plant/DC/shelf capacities | retail distribution |
| `multi_item` | continuous LP | item chains coupled by one shared time-varying capacity | production smoothing ahead of a seasonal peak |

Demand series (`_inventory_demand`) are seasonal (26-period cycle) with trend,
Gamma noise, and optional intermittency. Infeasible ratios are 1.10–1.35 and
unknown ratios `1 ± U(0.03, 0.30)` (`_inventory_scale_ratio`).

## `standard` — DC replenishment

Columns per SKU `i` (vendor lead time `L_i`):

```text
q[i,t] >= 0                 order placed in t, arrives t + L_i   (t <= T - L_i)
I[i,t] >= 0                 end-of-period stock
u[i,t] in [0, demand[i,t]]  lost sales (periods with demand)
```

Rows: stock balance `I[i,t-1] + q[i,t-L_i] + pipeline[i,t] + u[i,t] - I[i,t] =
demand[i,t]` (open orders arrive as `pipeline`); service level `sum_t u[i,t] <=
(1 - fill_rate[i]) sum_t demand[i,t]` for A/B-class SKUs (top 20% / next 30% by
volume); vendor allocation `sum_{i from v} q[i,t] <= vendor_capacity[v,t]`; zone
storage `sum_{i in z} volume[i] I[i,t] <= zone_capacity[z]`; zone receiving
`sum_{i in z} pallets[i] q[i,t-L_i] <= receiving_capacity[z,t]`. Objective:
purchase + holding + lost-sales penalties.

Feasibility: `ReplenishmentPlanWitness(orders, stock)` is an order-up-to plan
without lost sales; capacities are drawn above it. Infeasible:
`VendorAllocationCertificate(vendor, skus, required, available)` — summing a
service-level SKU's balance rows gives `sum_t q[i,t] >= fill_rate D - I0 -
pipeline`; the vendor's allocation (summed over order periods) is cut below the
total. Lost sales make the model without service rows trivially feasible; the
service rows give the certificate teeth.

Sizing: SKUs are added until `sum_i (T - L_i) + N T + #(demand > 0)` reaches the
target (`T` = 4–8 / 8–16 / 13–30 by scale); rows ≈ 40% of columns.

## `lot_sizing` — capacitated lot sizing, facility-location formulation

```text
w[i,s,t] >= 0   share of item i's period-t net demand produced in s, s in [t - window + 1, t]
y[i,s] in {0,1} setup of item i in s

sum_s w[i,s,t] = 1                                                  (assignment)
w[i,s,t] <= y[i,s]                                                  (disaggregated linking)
sum_i proc[i] sum_t d[i,t] w[i,s,t] + sum_i setup_time[i] y[i,s] <= capacity[s]
```

Objective: `(unit cost + holding × (t - s)) d[i,t] w + setup cost y`. Net demand
is gross demand after initial stock.

This is the strong formulation: fractional setups `y >= max_t w` still consume
setup time, so the relaxation must batch demand like an integer plan. (The old
`x = lot · n_lots` model with continuous `n_lots` relaxed to `y = x / (2 cap)`,
a plain single-item inventory LP.) The test suite checks that removing setup
times and costs strictly lowers the relaxed optimum.

Feasibility: `LotSizingPlanWitness(setups, source, load)` is a
periodic-order-quantity plan (interval ≤ window); capacity is flat around its
load and never below it. Infeasible: `LotSizingPrefixCertificate(horizon,
setup_lower_bounds, required, available)` — for each item, the number of
pairwise-disjoint windows among its demand periods `<= horizon` forces that many
setups even in the LP; processing plus those setup times exceed the prefix
capacity. Capacity rows have zero minimum activity, so presolve cannot decide.

Sizing: items are added until the column budget is used (`T` 4–20, window 2–6);
rows ≈ columns (assignment + linking + capacity).

## `multi_echelon` — plant → DCs → stores

Columns: `f[p,r,t]` plant → DC (one-period transit), `g[p,l,t]` lane shipments
(DC → store, 0–1 period transit), `J[p,r,t]` DC stock, `K[p,s,t]` store stock.
About a third of stores have a 30%-costlier backup lane from the nearest other
DC, used when the primary DC is short. Rows: DC and store balances per product
and period; plant hours `sum_p hours[p] sum_r f[p,r,t] <= plant_capacity[t]`;
DC throughput; DC storage; store shelf space (a variable bound for a single
product). Geography: DCs over a 100×100 region, stores clustered around them;
transport cost is distance-based. Products: 1 below 300 columns, otherwise 2–4
(single-product store chains decouple and presolve folds them away).

Feasibility: `MultiEchelonPlanWitness` is a JIT plan on primary lanes with
safety stocks; capacities are drawn above it. Infeasible:
`MultiEchelonPrefixCertificate(horizon, required, available)` — summing every
DC and store balance row of a product over `1..horizon` shows production in
`1..horizon-1` must cover that prefix's demand net of all network stock; the
plant hours for the binding prefix exceed its capacity.

The old variant was a single-product star whose 100k columns came from a
4,000-period horizon, with `(L - 1) T` dead return arcs.

## `multi_item` — shared time-varying capacity

Columns `x[i,t]`, `I[i,t]`; rows: per-item balance (no backlog) and `sum_i
usage[i] x[i,t] <= capacity[t]`. Capacity has maintenance/holiday dips and
demand a pronounced seasonal peak. For one shared resource the instance is
feasible iff every prefix fits: `sum_i usage[i] max(0, D_i(1..τ) - I0_i) <=
sum_{t <= τ} capacity[t]` for all `τ` (stored as `binding_ratio`).

- `feasible`: ratio 0.70–0.92; `MultiItemPlanWitness` from a backward
  (as-late-as-possible) capacity fill.
- `infeasible`: ratio 1.10–1.35 at the binding prefix;
  `MultiItemPrefixCertificate(horizon, required, available)`. Initial stock
  covers the first period, so no single row is violated (the old variant was
  presolve-detected).
- `unknown`: ratio `1 ± U(0.03, 0.30)` — feasible exactly when `<= 1` (the
  old variant always shared the feasible branch).

Sizing: `2 · n_items · n_periods` columns, `n_items · n_periods + n_periods`
rows (`T` 4–8 / 10–20 / 20–40 by scale).

## References

- Krarup, J., Bilde, O. (1977). Plant location, set covering and economic lot
  size: an O(mn)-algorithm for structured problems.
- Trigeiro, W.W., Thomas, L.J., McClain, J.O. (1989). Capacitated lot sizing
  with setup times. Management Science 35(3).
- Pochet, Y., Wolsey, L.A. (2006). Production Planning by Mixed Integer
  Programming. Springer.
- Silver, E.A., Pyke, D.F., Thomas, D.J. (2016). Inventory and Production
  Management in Supply Chains. CRC Press.
