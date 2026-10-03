# Mine Planning

Open-pit mine production scheduling over a 3D block model: decide when each
block of rock is mined and where it goes, subject to the pit-slope precedence
between blocks and per-period capacities, maximising discounted cash flow.
These problems are the source of the MineLib benchmark (Espinoza et al. 2013),
whose LP relaxations reach millions of variables and motivated the
Bienstock–Zuckerberg algorithm. The relaxation is meaningful: the precedence
and chain rows survive `relax_integer=true` intact, so the LP is a
precedence-closure polytope coupled by knapsack-like capacity rows and, in
`pcpsp`/`stockpile`, by blending ratios and inventory flows. It is not
unimodular.

All three variants use a local RNG (the caller's global RNG is never read or
advanced), and `build_model` performs no sampling.

## Variants

| Variant | Model class | Key structure | Domain grounding |
|---|---|---|---|
| `cpit` (default) | MIP, LP under the default relaxation | cumulative extraction, fixed destinations, mining/milling capacities, minimum mill feed | MineLib CPIT |
| `pcpsp` | MIP (binary extraction, continuous destinations) | + mill / heap-leach / dump split variables, head-grade and arsenic blending, mill and SX-EW metal capacities | MineLib PCPSP, copper porphyry with oxide cap |
| `stockpile` | MIP (binary extraction, continuous rest) | + grade-binned stockpiles with inventory balances, reclaim capacity, head grade | Moreno et al. (2017) linear stockpile models |

A single-period ultimate-pit (UPIT) variant was deliberately left out: its LP
is the totally unimodular max-closure problem, whose dual is a max-flow and
which HiGHS presolve largely dissolves (negative leaf blocks and positive
roots fix by dual arguments, cascading through the pit). It would add size
without simplex headroom.

## Block model (shared)

- **Grid**: about `3 * n_blocks` candidate blocks over `nz ~ n_blocks^(1/3)`
  benches and an elongated footprint; block footprints of 10-25 m and bench
  heights of 10-15 m.
- **Ore bodies**: one to four rotated, anisotropic 3D Gaussian kernels buried
  under an overburden (centres at 35-75% of the grid depth). The raw grade is
  the kernel sum over a low background times a lognormal nugget; grades are
  then calibrated so 15-35% of the pit is above the sulfide milling cutoff.
- **Pit envelope**: cones on the ore bodies,
  `f(i, j) = max_k (H_k - s_k ||R_k (i - cx_k, j - cy_k)||)` with overall wall
  slope `s_k <= 0.68`, so `f` changes by less than one bench between
  neighbouring columns. Every candidate is scored `f(i, j) - z` and the pit is
  the `n_blocks` highest-scoring candidates (ties toward shallower benches).
  Each block's predecessors one bench up score at least as high and are
  shallower, so the pit is **precedence-closed by construction**, and the score
  order is a topological order whose prefixes are nested pit shells
  (pushbacks). Blocks are indexed in that order.
- **Precedence**: the classic 1-5 pattern (the block directly above and its 4
  edge neighbours, 60%) or 1-9 pattern (the 3 x 3 above, 40%). All arcs join
  adjacent benches, so the arc set is transitively reduced.
- **Rock and economics**: an oxide cap above a smooth oxidation surface
  (lighter rock, poor flotation recovery, leachable); an arsenic (enargite)
  zone around one ore body; density 2.3-2.5 t/m3 (oxide) or 2.6-2.8 t/m3
  (sulfide); mining cost 1.4-2.6 $/t plus 2-5 cents per bench of haul depth;
  copper at 6,000-10,000 $/t, flotation recovery 85-92% (sulfide) / 45-60%
  (oxide) at 7-13 $/t, heap leach 55-75% at 2.5-5 $/t; discount rate 6-12% per
  period. Units: kt, % Cu, ppm As, k$.
- **Capacities**: the fleet moves the pit in about `T / rho` periods
  (`rho in [0.8, 1.25]`), with a start-up ramp (period 1 at 35-100%, period 2
  at 70-100%); the mill is sized 0-60% above the average ore rate and
  commissioned during period 1 (60-100%).
- **Horizon**: `T = clamp(round(2.5 log10(n) - 1 +- 1), 3, 24)` periods —
  about 3 at 50 variables, 9 at 10k, 12 at 100k, 14 at 1M.

## Formulations

Shared ("by" cumulative variables, as in MineLib): `x[b, t] in {0, 1}` is 1 if
block `b` has been mined by the end of period `t`, with `x[b, 0] = 0`.

```text
x[b, t-1] <= x[b, t]                                   chain
x[b, t]   <= x[a, t]       for each arc b -> a        precedence
sum_b w_b (x[b, t] - x[b, t-1]) <= mining_capacity[t]
```

### `cpit`

Each block's destination is fixed by its grade (ore above the breakeven milling
grade is milled). Per period:

```text
min_processing[t] <= sum_{b ore} w_b (x[b,t] - x[b,t-1]) <= processing_capacity[t]
maximize sum_t delta_t sum_b v_b (x[b,t] - x[b,t-1])      (telescoped "by" form)
```

### `pcpsp`

`y[j, t] >= 0` sends a fraction of block `pair_block[j]` to plant
`pair_dest[j]` (1 = mill, 2 = heap leach) in period `t`. Mill pairs exist for
blocks above half the milling breakeven (so marginal material can be blended),
leach pairs for oxide above half the leaching breakeven; the dump is implicit.

```text
sum_{j of b} y[j, t] <= x[b, t] - x[b, t-1]                         linking
min_mill_feed[t] <= sum_{j mill} w y[j, t] <= mill_capacity[t]
sum_{j leach} w y[j, t] <= leach_capacity[t]
sum_{j mill} w (g - head_grade_min) y[j, t] >= 0                    head grade
sum_{j mill} w (a - arsenic_max) / 1000 y[j, t] <= 0                arsenic (t As)
sum_{j mill} metal_j y[j, t] <= mill_metal_capacity[t]              concentrate
sum_{j leach} metal_j y[j, t] <= leach_metal_capacity[t]            SX-EW cathode
maximize discounted plant margins - discounted mining cost
```

### `stockpile`

Destinations are the mill (direct feed) and `S in {2, 3}` stockpile bins
covering geometric grade ranges `[lo_s, hi_s)` between a stockpiling threshold
(ore that pays for rehandling at its bin's lower edge) and a high-grade limit
(richer ore always goes direct). `r[s, t]` is reclaimed tonnage and
`0 <= inv[s, t] <= stockpile_capacity[s]` end-of-period inventory. Reclaimed
ore is valued and blended at the bin's **lower** grade edge `lo_s` — a
conservative linear approximation that never credits metal the pile does not
contain.

```text
sum_{j of b} y[j, t] <= x[b, t] - x[b, t-1]
min_mill_feed[t] <= sum_{j mill} w y[j, t] + sum_s r[s, t] <= mill_capacity[t]
sum_{j mill} w (g - head_grade_min) y[j, t] + sum_s (lo_s - head_grade_min) r[s, t] >= 0
direct metal + sum_s reclaim_metal_s r[s, t] <= mill_metal_capacity[t]
sum_s r[s, t] <= reclaim_capacity[t]
inv[s, t] = inv[s, t-1] + sum_{j in bin s} w y[j, t] - r[s, t]      (inv[s, 0] = 0)
maximize discounted mill margins + reclaim margins - placement and mining costs
```

## Feasibility control

Mining nothing satisfies every capacity row, so the instances are made
non-trivial by a **minimum mill-feed contract**. The natural contract spreads
60-125% of the estimated mill-eligible reserve over a window of periods
(starting in period 1 or 2, ending in `T-1` or `T`), capped at 50-90% of the
mill capacity.

- `feasible`: a whole-block schedule is planted by mining the nested-shell
  block order greedily into periods within capacities shrunk by 3-8% (each
  block to its most valuable destination with room; mining stops for the
  period when the mill is full; in `stockpile`, spare mill capacity is filled
  by reclaiming the richest bins). The contract, head-grade and arsenic specs
  are the natural ones clipped to the plan with 5-12% margins. The integral
  plan is stored as a `MinePlanWitness` and satisfies every row of the MIP.
- `infeasible`: a `MineClosureCertificate` built from LP rows alone (it survives
  relaxation) with a 10-35% margin. Writing `X_b = x[b, k]`, the precedence
  rows put `X` in the closure polytope, the summed capacity rows of periods
  `1..k` cap `sum_b w_b X_b`, and the summed feed (and head-grade) rows force
  `sum_b q_b X_b >= requirement`. For any `lambda >= 0`, a feasible flow in the
  max-closure network with node weights `q_b - lambda w_b` bounds every
  closure, so `bound = lambda * budget + sum_{c_b > 0} (c_b - source_flow_b)`
  caps the left side; the certificate stores `lambda` (chosen by a
  golden-section search) and the flow, and `requirement >= (1 + margin) bound`.
  Modes:
  - `:ramp_up` (`k in 1:3`): the start-up contract (80-95% of mill capacity)
    exceeds the mill-eligible ore reachable with the start-up fleet — when
    needed, the start-up fleet is reduced (never below 30% of steady state,
    the low end of the natural ramp-up) or the mill and contract are enlarged
    together (at most 1.6x).
  - `:exhaustion` (`k = T`, drawn 20% of the time and the fallback): the
    contract exceeds everything the pit can supply within its total mining
    capacity.
  - `:head_grade` (`pcpsp`/`stockpile`, 50% of the time when it qualifies):
    weights `q_b = w_b (g_b - tau)^+` for a threshold grade `tau`; the minimum
    head grade is raised to `tau + (1 + margin) bound / feed`, but kept below
    80% of the richest eligible grade so no row is trivially contradictory.
    Stockpiles cannot help: reclaim never exceeds what was stockpiled from
    mined blocks, and reclaimed ore is credited at most its true grade.

  Every per-period minimum feed stays at or below 95% of its mill capacity, so
  no single row is contradictory and presolve cannot detect the infeasibility.
- `unknown`: the natural contract, with the start-up periods drawn at 60-120%
  of the closure bound on reachable mill feed (capped at 95% of mill
  capacity). The bound is exact for the cumulative relaxation, so the draw
  straddles the true feasibility boundary; no witness or certificate is
  stored. Measured with HiGHS (20 seeds): 15-55% of instances are infeasible
  depending on variant and size.

## Sizing

| Variant | Variables | Rows |
|---|---|---|
| `cpit` | `T * B`, `B = max(4, round(n / T))` (within `T/2` of the target) | `B(T-1) + A T + 2T` |
| `pcpsp` | `T (B + P)` | `B(T-1) + A T + T (B_P + 1 + 4 [mill] + 2 [leach])` |
| `stockpile` | `T (B + P + 2S)` | `B(T-1) + A T + T (B_P + 5 + S)` |

`B` blocks, `A` precedence arcs (about 4-5 per block for 1-5, 7-9 for 1-9,
fewer near the surface), `P` destination pairs on `B_P` blocks, `S` bins. For
`pcpsp` and `stockpile` the pit is the prefix of the block order whose count is
closest to the target (within about `1.5 T`). Rows are 1.5-6 per variable. All
data structures are sparse and generation is near-linear: about 2 s at 100k
variables and 20-60 s at 1M (dominated by the JuMP build; infeasible instances
add the certificate's max-flows, Dinic's algorithm with a capped phase count).
Targets above `MINE_PLANNING_MAX_VARIABLES = 1,000,000` raise `ArgumentError`
(the model would exceed ten million rows).

## Presolve and solve behaviour

HiGHS presolve keeps 90-100% of the columns and rows (it can only drop a few
dominated destination columns), and the LPs need real simplex work: thousands
of iterations at 10k variables, and the 100k-variable LPs typically exceed a
60 s dual-simplex budget — consistent with MineLib's relaxations being hard
enough to need specialised algorithms.

## References

- Espinoza, D., Goycoolea, M., Moreno, E., Newman, A. (2013). MineLib: a library
  of open pit mining problems. Annals of Operations Research 206, 93-114.
- Bienstock, D., Zuckerberg, M. (2010). Solving LP relaxations of large-scale
  precedence constrained problems. IPCO 2010, LNCS 6080, 1-14.
- Lerchs, H., Grossmann, I.F. (1965). Optimum design of open-pit mines.
  Transactions CIM 58, 47-54. (Ultimate pit as maximum closure.)
- Picard, J.-C. (1976). Maximal closure of a graph and applications to
  combinatorial problems. Management Science 22(11), 1268-1272.
- Moreno, E., Rezakhah, M., Newman, A., Ferreira, F. (2017). Linear models for
  stockpiling in open-pit mine production scheduling problems. European Journal
  of Operational Research 260(1), 212-221.
- Chicoisne, R., Espinoza, D., Goycoolea, M., Moreno, E., Rubio, E. (2012). A new
  algorithm for the open-pit mine production scheduling problem. Operations
  Research 60(3), 517-528.
