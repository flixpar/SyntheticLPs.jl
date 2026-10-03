# Maritime Inventory Routing

`maritime_inventory_routing/standard` is a discrete-time maritime
inventory-routing problem (MIRP) on a time-expanded sailing network: a fleet
loads at a single depot port and discharges at consumption ports whose tanks
drain over the horizon. Vessel positions and legs are binary (a genuine MIP);
with the package default `relax_integer=true` it is returned as its LP
relaxation, in which a vessel may sit fractionally in several ports.

## Data

- **Ports** on a 1200 × 1200 km sea grid; port 1 is the depot. Sailing speed
  450–650 km/day; leg cost is a port fee plus a bunker cost per km.
- **Sailing network**: waiting arcs at every port and both depot shuttle legs of
  every customer are always present; further port-to-port legs are added in
  increasing travel time. One period is `period_length` days — the longest
  selected leg — so every leg is sailable within a period. The arc count is the
  sizing knob.
- **Fleet, tanks, depot**: vessel capacities, initial loads, tank capacities,
  initial tank levels, per-period consumption, and per-period depot supply are
  sized from a planted rotation (below).

## Formulation

Sets: vessels `V`, ports `P` (depot 1, customers `c = 2..P`), periods `0..T`,
arcs `A`.

```text
min  Σ travel_cost[a] move[v,a,t] + Σ holding_cost[c] inventory[c,t]
s.t. location[v,·,0] fixed at the depot; load[v,0], inventory[c,0] fixed (bounds)
     Σ_{a out of i} move[v,a,t] = location[v,i,t−1]                 ∀ v, i, t
     Σ_{a into j}  move[v,a,t] = location[v,j,t]                    ∀ v, j, t
     pickup[v,t]     ≤ capacity_v location[v,1,t]
     delivery[v,c,t] ≤ capacity_v location[v,c+1,t]
     load[v,t] = load[v,t−1] + pickup[v,t] − Σ_c delivery[v,c,t],  0 ≤ load ≤ capacity_v
     Σ_v pickup[v,t] ≤ depot_supply[t]
     inventory[c,t] = inventory[c,t−1] + Σ_v delivery[v,c,t] − consumption[c,t],
                      0 ≤ inventory ≤ tank_capacity_c
     location, move ∈ {0,1}; pickup, delivery ≥ 0
```

Exactly `V P (T+1) + V A T + V (P−1) T + V T + V (T+1) + (P−1)(T+1)` variables;
`_mirp_dimensions` samples the fleet/horizon/arc-density shape, solves a
quadratic for the port count, and reads the arc count off exactly, so the
count usually equals the target (otherwise within one `V T` block). The port
count is capped so the planted rotation can visit every customer.

## Feasibility control

- `feasible`: a depot-shuttle rotation is planted (odd periods at the depot,
  discharge at the `k`-th rotation customer in period `2k`), and vessel
  capacity, tank capacity, initial tank level, and depot availability are sized
  from that plan's peaks. Stored as `MaritimeScheduleWitness` (positions,
  pickups, deliveries, loads, tank levels) — a feasible point of the integer
  model and of its relaxation.
- `infeasible`: initial material plus the depot's supply is only 55–88% of
  consumption over a prefix horizon. `MaritimeSupplyCertificate` bounds what
  customers can receive twice — by the material available and by fleet
  throughput (`pickup + Σ delivery ≤ capacity` per vessel-period, from the flow
  and linking rows) — using LP rows only, so it refutes the relaxation. HiGHS
  needs simplex iterations to prove it (hundreds to thousands at 10k–100k).
- `unknown`: fleet and tanks at or above the plan (vessel capacity 1.00–1.50,
  tank capacity 1.00–1.60 of the plan peaks), depot supply scaled by a global
  scarcity factor 0.80–1.15 with per-period noise; whether a short period can
  be absorbed by loading earlier depends on the fleet's spare onboard capacity.

## Measured behavior (HiGHS, 60 s limit)

| Target | Rows | Presolve kept (cols/rows) | Feasible solve |
|---|---|---|---|
| 10k | ≈4.0k | 0.93 / 0.92 | ≈0.5 s |
| 50k | ≈10.4k | 0.96 / 0.95 | ≈2.6 s |
| 100k | ≈18k | 0.96 / 0.96 | ≈30 s |

Build time is well under a second at 100k. The period-0 state is emitted as
fixed variable bounds rather than singleton equality rows.
