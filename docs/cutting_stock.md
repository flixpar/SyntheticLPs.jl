# Cutting Stock

One-dimensional cutting stock in its two classic LP forms: the Gilmore-Gomory
pattern master LP (`standard`, multi-period `due_dates`, multi-machine
`setup_cost`) and the pseudo-polynomial arc-flow formulation (`arc_flow`).
All variants scale their row count with the instance, build in O(nnz), and
store a typed witness (`feasible`) or a relaxation-valid certificate
(`infeasible`) that combines many rows, so HiGHS presolve does not detect it.

## Shared data (`common.jl`)

- **Stock catalogue.** Bar lengths (mm) from `6000, 7500, 9000, 10500, 12000,
  13500`; a bar costs `L * p * (L / 6000)^-0.06` (per-mm price with a small
  long-bar discount, ±3% noise).
- **Order book.** Integer-mm lengths in `[120, 0.45 L_max]`, `Beta(1.6, 3.2)`
  skewed short, 55% snapped to a 50 mm grid; quantities lognormal around 60
  pieces, packs of 5 above 50.
- **Pattern enumerator** (`cs_generate_patterns`), near-linear: the maximal
  single-item pattern of every (item, stock) pair that fits first, then
  knapsack-like greedy fills (a stock type, 2-6 random candidate items, random
  counts, longest-first top-up to a maximal, low-trim pattern), deduplicated
  through a hash set; a deterministic sub-maximal/two-item fallback only for
  tiny catalogues. Patterns are stored sparse (`CSPatterns`). The previous
  per-variant enumerators were dense and quadratic (121-165 s builds at 50k)
  and `standard` could not reach 50k patterns at all.
- **`MaterialShortageCertificate`**: with multipliers `len_i` on demand rows and
  `L_k` on availability rows, `sum_i len_i d_i >= 1.04 * sum_k L_k S_k`
  refutes the LP because every pattern fits its bar.

## `standard` — multi-stock Gilmore-Gomory master LP

```text
min  sum_j cost[stock j] x_j
s.t. sum_j a_ij x_j >= d_i         (every item type)
     sum_{j on k} x_j <= S_k       (every stock type)
```

Exactly `n` patterns; `n_stock = clamp(round(log10 n), 1, 5)`,
`n_types = clamp(round(n / 20), 1, n / n_stock)` — rows ≈ 5% of columns
(previously ~100 rows at any size).

- `feasible`: single-item plan (`StockPlanWitness`), availabilities
  `U(1.05, 1.35)` × its usage per stock type.
- `infeasible`: availabilities scaled so ordered material exceeds the stock
  length by `U(8%, 20%)` (`MaterialShortageCertificate`).
- `unknown`: total stock length `U(0.97, 1.10)` × the ordered material — around
  the trim-loss threshold of the best patterns.

## `due_dates` — multi-period with inventory

```text
min  sum cost x_jt + sum h_i inv_it
s.t. inv_{i,t-1} + sum_j a_ij x_jt - inv_it = d_it    (inv_{i,0} = 0)
     sum_{j on k} x_jt <= S_kt
```

Each item has orders in about half of the `T = clamp(round(2 log10 n), 4, 12)`
periods (seasonal profile); holding cost ~1.5% of the material value per
period. `n_types = round(n / (16 T))`, `n_patterns = n ÷ T - n_types`; columns
`T (n_patterns + n_types)` (within `T - 1` of the target), rows
`T (n_types + n_stock)`.

- `feasible`: just-in-time single-item plan (`DueDatePlanWitness`), deliveries
  `U(1.05, 1.35)` × its per-period usage.
- `infeasible`: the opening delivery is short — material due in period 1
  exceeds period-1 stock by `U(8%, 20%)` (`CumulativeShortageCertificate`
  with `period = 1`, combining every item's period-1 balance row with the
  period-1 availability rows). A later cut-off period is an equally valid
  proof, but at 100k columns HiGHS's dual simplex intermittently returned
  UNKNOWN on the longer multi-period ray (IPM proves infeasibility), so the
  robust first-period form is used.
- `unknown`: per-period stock length `U(0.95, 1.12) × U(0.85, 1.15)` × that
  period's due material; carryover decides.

## `setup_cost` — multi-machine with pattern setups

Every pattern runs on 1-3 of `clamp(round(Q / 400), 2, 250)` parallel saws.
Pair `q = (pattern, machine)` has a run count `x_q` and setup binary `y_q`;
run minutes grow with the number of pieces, setup minutes with the number of
distinct lengths (knife moves), both machine-dependent; setup cost = labour
minutes + 30% of a bar.

```text
min  sum cost x_q + sum setup_cost_q y_q
s.t. demand rows, stock rows,
     sum_{q on m} (run_q x_q + setup_q y_q) <= H_m
     x_q <= M_q y_q,  M_q = min(max_i ceil(d_i / a_ij), floor(H_m / run_q))
```

`y_q` sits in its machine row as well as its link, so it is not a column
singleton and survives presolve (the old single-machine big-M variant was
reduced to 50% of its columns and 2% of its rows). `Q = n ÷ 2` pairs; rows
`Q + n_types + n_stock + n_machines`.

- `feasible`: single-item plan on one machine each (`SetupPlanWitness`);
  stock `U(1.05, 1.35)`, machine minutes `U(1.02, 1.15)` × plan usage.
- `infeasible`: material shortage as in `standard`.
- `unknown`: stock length `U(0.97, 1.10)` × material.

## `arc_flow` — Valério de Carvalho arc-flow

Nodes are reachable cut positions `0..L`; item `i` (sorted longest first) has
arcs `(u, u + len_i)` from every node reachable with items `1..i` (symmetry
reduction), and every reachable node has a loss arc to the next one. Flows are
general integers (relaxed by default).

```text
min  sum_{a out of 0} f_a
s.t. flow conservation at every internal node
     sum_{a of item i} f_a >= d_i,   sum_{a out of 0} f_a <= S
```

`n_types = clamp(round(sqrt(n) / 16), 4, 30)` relative lengths
(`0.04-0.42 L`); `L` is chosen by bisection so the exact arc count is as close
to the target as the integer graph allows. Its LP bound equals the
Gilmore-Gomory bound but the matrix is a network with side constraints. This
replaces `integer_patterns`, whose relaxation was the same pattern LP as
`standard` with 31 rows.

- `feasible`: superposed single-item paths (`ArcFlowWitness`),
  `S = U(1.05, 1.30)` × its rolls.
- `infeasible`: `S = floor(material / (L U(1.08, 1.20)))`
  (`ArcFlowMaterialCertificate`, node potentials `pi_u = u`).
- `unknown`: `S = round(U(0.97, 1.10) material / L)`.
