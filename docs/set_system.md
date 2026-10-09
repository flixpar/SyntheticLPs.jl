# Set Systems

The `set_system` category generates 0/1 (and multi-unit) covering, packing,
partitioning, and winner-determination models over set systems from four
applications. The column structures differ: radius-based coverage sets,
space-time staircases of train paths, a generic sparse partition, and
regional multi-unit bundles with XOR bidders. These are no longer four
relabelings of one random incidence matrix.

## Variants

| Variant | Application | Rows | Columns |
| --- | --- | --- | --- |
| `set_cover` (default) | Location set covering (ambulance stations, cell towers) | demand points (`≥ 1`) + site budget | candidate sites |
| `set_packing` | Railway timetabling / train-path allocation | space-time track cells (`≤ 1`), optional trains (`≤ 1`), mandatory trains (`= 1`) | candidate train paths |
| `set_partitioning` | Generic exact set partitioning | elements (`= 1`) + cardinality cap | candidate subsets |
| `combinatorial_auction` | Multi-unit spectrum / slot auction | items (`Σ q·x ≤ u`), bidders (XOR `≤ 1`), reserve revenue | bids |

Every variant has exactly `target_variables` columns. Each one plants a typed
`feasible_witness` for `feasible` requests and a typed
`infeasibility_certificate` for `infeasible` requests, both built from LP rows
alone. Every certificate aggregates many rows, so HiGHS presolve does not
detect it. At 10k variables the infeasible instances take about 1,000–6,000
simplex iterations; a set_partitioning instance can also run past a 60 s limit.

## `set_cover` — location set covering

- Demand points (30–50% as many as sites) are clustered towns over a rural
  scatter.
- Each candidate site is anchored at a demand point, and every point anchors
  at least one site, so a cover always exists.
- Sites come in three station types with coverage radius `{0.75, 1, 1.5}·r₀`.
  A site's column is every demand point within its radius, so urban sites
  cover far more points than rural ones (heavy-tailed column sizes, 10–20
  nonzeros per column on average).
- Site cost = type cost × land-price factor (crowding) × lognormal noise.

```text
min Σ c_j x_j   s.t.   Σ_{j ∋ i} x_j ≥ 1  ∀ demand points i,   Σ x_j ≤ B
```

Feasibility:

- `feasible`: a greedy cover takes the largest covering site for each
  uncovered point (`SetCoverWitness`). `B` is its size plus 0–10%.
- `infeasible`: a co-coverage packing is a set of points no two of which
  share a covering site. Summing their rows gives `Σx ≥ |points|`, and
  `B ≤ 0.97·|points|` (`SetCoverPackingCertificate`).
- `unknown`: `B` is drawn between the packing bound and the greedy cover size.
  Over 8 seeds this gives both outcomes at 2k and 10k.

## `set_packing` — train-path allocation

- **Corridor:** 5–40 sections, about 40% single-track (shared by both
  directions) and the rest double-track (one cell per direction). The central
  section is a single-track bottleneck.
- **Train requests:** passenger, regional or freight, at 1, 2 or 3 slots per
  section plus 1 slot of headway. Each runs a contiguous route of 2–8 sections
  (passenger trains favour through-routes across the bottleneck). Each has a
  requested departure, with passenger trains peaking at the shoulders.
- **Candidate paths:** departure shifts of up to ±2–5 slots. A path occupies
  its space-time cells, and its value is priority × route length × a lateness
  discount.
- **Horizon:** sized for 55–90% track utilisation if every train ran.
- **Rows:** a cell row only when two or more paths use the cell (a
  single-path cell is just `x ≤ 1`), one row per train (`≤ 1`, or `= 1` if
  mandatory). About 1.9 rows and 15 nonzeros per column.

Feasibility:

- `feasible`: a greedy conflict-free timetable (`SetPackingWitness`); the
  passenger trains it schedules become mandatory.
- `infeasible`: a peak-hour bottleneck deficit (`BottleneckCertificate`).
  - A group of mandatory trains has every candidate path inside one bottleneck
    window, and together they need at least 10% more bottleneck slots than the
    window has. Small requests keep enough trains for that: one path per train
    below 20 variables and at most five (shift flexibility `w <= 2`) below 80;
    the generator raises rather than emit a group that cannot over-subscribe
    its window.
  - Weighting each train's equality row by its occupancy and summing the
    window's slot rows gives the contradiction.
- `unknown`: a random 15–60% of passenger trains are mandatory, with no
  adjustment. Congestion decides whether they fit.

## `set_partitioning`

- A shuffled exact partition of the elements is planted. Random columns follow,
  mostly of size 1–4 with a heavier tail up to `≈ √n`.
- **Costs:** uniform 5–100.
- **Cardinality cap:**
  - `feasible`: the planted partition (`SetPartitioningWitness`), with the cap
    equal to its size.
  - `infeasible`: summing every row gives `Σx ≥ n / max|S|`. The cap sits 3%
    (at least 0.5) below that bound (`SetPartitioningCardinalityCertificate`).
  - `unknown`: the cap is drawn 70–95% of the way from that bound to the
    planted size, straddling the LP minimum of `Σx`, which sits at 85–90%.
- The partitioning LP is highly degenerate: at 10k variables, feasible solves
  take 20k–60k iterations.

## `combinatorial_auction` — multi-unit, XOR bidders

- **Items:** licences placed on a map, numbering 15–25% of the bids. 40% are
  single-unit and the rest have 2–6 units.
- **Bidders:** each grows a compact bundle by a random walk over neighbouring
  items, then submits 1–6 XOR alternatives that swap one or two items for
  neighbours.
- **Quantities:** 1–3 units per item.
- **Value:** sum over items of quantity × common value with ±20% private
  noise, times a bidder deviation and a complementarity factor
  `1 + 0.15(|bundle| − 1)`.

```text
max Σ v_b x_b   s.t.   Σ_b q_bi x_b ≤ u_i ∀ items,   Σ_{b∈bids(k)} x_b ≤ 1 ∀ bidders,   Σ v_b x_b ≥ R
```

Unlike `set_packing`, the item rows carry general integer coefficients and the
reserve row couples every column.

Feasibility:

- `feasible`: a greedy allocation by value per requested unit
  (`AuctionWitness`). `R` is 85–100% of its revenue.
- `infeasible`: item prices `p` and bidder surpluses `s` dual-cover every bid
  (`Σ q p + s_k ≥ v_b`). Weak duality bounds revenue by `Σ u p + Σ s`, and
  `R = 1.03 ×` that bound (`AuctionDualCertificate`).
- `unknown`: `R` lies between the greedy revenue and the dual bound.
