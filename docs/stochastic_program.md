# Stochastic Program

The `stochastic_program` category generates extensive-form (deterministic
equivalent) stochastic linear programs. The two variants have different
decomposition structure: `standard` is a two-stage recourse model with the dual
block-angular matrix of L-shaped/Benders methods, and `multistage_alm` is a
multistage asset–liability model on a scenario tree whose matrix is a staircase
along every root-to-leaf path. Both are pure LPs; both constructors use a local
`MersenneTwister`, and `build_model` is deterministic.

## Variants

| Variant | Model class | Key structure | Domain grounding |
|---|---|---|---|
| `standard` (default) | continuous LP | first-stage capacity + capital row, `S` scenario blocks (linking, demand, service level) | facility capacity planning with sparse distribution lanes under correlated demand |
| `multistage_alm` | continuous LP | scenario tree, nested rebalancing with transaction costs, funding floor at every node | defined-benefit pension fund (Cariño–Ziemba / Consigli–Dempster style ALM) |

## `standard`: two-stage capacity and distribution

### Data

- Customers sit in geographic regions on a 100 × 100 map; facilities are placed
  near regions (70%) or uniformly. Each customer is served over a sparse lane
  set: its two nearest facilities plus the nearest further pairs, two to six
  lanes per customer.
- Scenario demand is lognormal and correlated through a global market factor, a
  regional factor, and idiosyncratic noise, so scenarios differ in both volume
  and geography. Scenario probabilities are random and normalized.
- Lane cost is a facility handling cost plus a distance-proportional freight
  rate. Capacity costs `build_cost` (annualized) and consumes `capital_use`
  under a capital budget. The shortfall penalty is calibrated so the newsvendor
  ratio `build_cost / (penalty − freight)` is about 0.12–0.35: building to
  serve is economical, so the optimal plan serves most demand (the earlier
  generator served about 29% because its budget covered only a few units).

### Formulation

```text
min  Σ_i build_i x_i + Σ_s p_s ( Σ_l ship_l y[l,s] + Σ_j penalty_j z[j,s] )
s.t. Σ_i capital_i x_i ≤ capital_budget                       (first stage)
     existing_i ≤ x_i ≤ capacity_max_i
     Σ_{l out of i} y[l,s] ≤ x_i                    ∀ i, s     (linking)
     Σ_{l into j} y[l,s] + z[j,s] = demand[j,s]     ∀ j, s     (demand)
     Σ_j z[j,s] ≤ (1 − service_level) Σ_j demand[j,s]  ∀ s     (service level)
     y, z ≥ 0
```

Variables `n_facilities + n_scenarios (n_lanes + n_customers)`; rows
`1 + n_scenarios (n_facilities + n_customers + 1)`. Customers grow like
`target^0.42`, facilities are a quarter of the customers, and the lane count is
re-tuned after the scenario count is fixed, so the total lands within about
`n_scenarios / 2` of the target.

### Feasibility control

Only the capital budget differs between statuses.

- `feasible`: every customer's served share `service_level × demand` is split
  over its lanes by fixed weights; each facility's capacity is the largest load
  any scenario puts on it. The budget is 3–15% above that plan's capital.
  Stored as `StochasticProgramWitness` (capacity, shipments, shortfall).
- `infeasible`: in the highest-demand scenario the service row and the linking
  rows force `Σ_i x_i ≥ service_level × Σ_j demand`; the cheapest capital that
  reaches this total within the capacity bounds is a fractional knapsack
  (`_stochastic_program_min_capital`). The budget is 8–18% below it. Stored as
  `StochasticProgramCertificate`. The existing capacity always fits the budget,
  so no single row is contradictory and presolve does not detect it (the old
  generator's infeasibility was a single-row bound contradiction).
- `unknown`: budget log-uniform between `0.92 × min_capital` and
  `1.08 × witness_capital`; the lane structure decides. The earlier 70/30 coin
  flip into a planted status is gone.

## `multistage_alm`: pension-fund asset–liability management

### Scenario tree and data

- Breadth-first tree with `n_stages` stages (2–6), a uniform branching factor,
  and the final stage trimmed so the variable count hits the target (every
  pre-leaf node keeps a child). Conditional probabilities are uniform.
- Each edge draws a market shock, a short-rate change, and inflation. Cash earns
  the parent's short rate; bonds earn it plus a premium minus `duration × Δrate`;
  equities and real assets load on the market shock with asset betas and
  idiosyncratic noise; index-linked bonds earn inflation. The universe grows
  from 3 asset classes (cash, government bonds, domestic equity) to 10.
- Benefits grow with inflation; the liability value `L_n` is the annuity value of
  benefits at the node's short rate (normalized to 1 at the root), so rising
  rates lower liabilities and bond prices together. Contributions are 30–80% of
  benefits.

### Formulation

Per node `n` (root: initial holdings instead of `R h[parent]`):

```text
h[a,n] = R[a,n] h[a,parent] + buy[a,n] − sell[a,n]                          (non-cash a)
h[cash,n] = R[cash,n] h[cash,parent] + Σ_a (1−tc_a) sell[a,n] − Σ_a (1+tc_a) buy[a,n]
            + inflow_n − outflow_n
wealth[n] = Σ_a h[a,n]
0 ≤ h[a,n] ≤ max_weight_a · exposure_scale · L_n                            (exposure bounds)
Σ_{equities} h[a,n] ≤ equity_cap · exposure_scale · L_n
wealth[n] ≥ funding_ratio · L_n                         (funding floor, non-root, a bound)
shortfall_n + wealth[n] ≥ target_ratio · L_n                                 (leaves)
max  Σ_leaves p_n (wealth[n] − penalty · shortfall_n)
```

Variables `n_internal (3A − 1) + n_leaves · 3A`; rows per node: `A` balances,
the wealth definition, the equity limit, and a target row at leaves.

Exposure limits are expressed in liability units (a risk budget scaled to the
strategic funding level) and emitted as variable bounds. Wealth-relative caps
`h ≤ w Σh` were tried first: those homogeneous rows made HiGHS's dual simplex
return an unknown status on some infeasible instances. Transaction costs are
0.2–3% (very small costs made buy/sell columns nearly parallel and caused the
same failure).

### Feasibility control

Only the funding ratio differs between statuses.

- `feasible`: a conservative fixed-mix policy (rebalance to the same weights at
  every node, paying transaction costs — the post-trade wealth solves a small
  fixed point) is simulated; the funding ratio is 88–97% of its lowest
  wealth-to-liability ratio. Stored as `MultistageALMWitness`.
- `infeasible`: along every path, the best wealth any policy could reach is
  bounded by holding the best-returning assets up to their exposure limits (a
  fractional knapsack per node, `_alm_best_growth`). On the weakest path the
  funding ratio is set 4–10% above that bound's ratio. Stored as
  `MultistageALMCertificate` (path, growth bounds, wealth bounds); the proof
  chains balance rows along a whole path, so presolve does not see it.
- `unknown`: funding ratio drawn in the upper part (log scale) of the interval
  between the two thresholds; about 60–70% feasible over seed blocks.

## Measured behavior (wave-2 audit, HiGHS, 60 s)

| Variant | Target | Rows | Presolve kept (cols/rows) | Notes |
|---|---|---|---|---|
| `standard` | 10k | ≈2.5k | 1.00 / 1.00 | infeasible needs ≈2.6k–3.2k iterations |
| `standard` | 100k | ≈25k | 1.00 / 0.99 | solves in 1–9 s |
| `multistage_alm` | 10k | ≈4.4k | 1.00 / 1.00 | |
| `multistage_alm` | 100k | ≈43k | 1.00 / 1.00 | 30–60 s; some instances hit the limit |
