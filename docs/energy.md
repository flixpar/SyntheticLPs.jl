# Energy

Power-systems operations LPs. The category has six variants in two families that
share `src/problem_types/energy/common.jl`:

- **Multi-area economic dispatch** (`standard`, `reserves`, `storage`,
  `hydrothermal`): one dispatch core — balancing zones joined by lossy
  tie-lines, a technology-grounded committed fleet with must-run floors and ramp
  limits, curtailable renewables, zonal load profiles — with each variant adding
  one real coupling.
- **Bus-level DC power flow** (`dc_opf`, `security_constrained_dc_opf`): a
  geometric meshed transmission grid in the B-θ formulation, single snapshot or
  with N-1 contingency states.

All variants are continuous LPs. Every constructor draws from a local
`MersenneTwister(seed)`; generation neither reseeds nor consumes Julia's global
RNG, and `build_model` does no sampling.

## Variants

| Variant | Adds to the core | Columns | Infeasible mode (certificate) |
|---|---|---|---|
| `standard` (default) | horizon emissions budget | `T·(G + 2L)` | `:emissions_budget` (85%), `:import_pocket` (15%) |
| `reserves` | spinning/non-spinning reserves, N-1 largest-unit rule | `T·(G + 2L + G_spin + G_ns + 1)` | `:reserve_scarcity` |
| `storage` | batteries + pumped hydro, state of charge | `T·(G + 2L + 3S)` | `:energy_limited_peak` |
| `hydrothermal` | cascaded reservoirs with travel delays | `T·(G + 2L + 3P)` | drought (`HydroDroughtCertificate`) |
| `dc_opf` | bus-level DC network, one snapshot | `G + B` | load pocket (`DCPocketCertificate`) |
| `security_constrained_dc_opf` | base case + `C` post-contingency networks | `G + (1 + C)·B` | N-1 load pocket (`DCPocketCertificate`) |

`T` periods, `G` units, `L` tie-lines, `S` storage devices, `P` hydro plants, `B`
buses, `C` screened contingencies.

Removed in the 2026 rebuild: `ramping` and `transmission` (their couplings —
ramp rows and lossy inter-zone transfers — are now part of every dispatch
variant's core, so they were duplicates), and `optimal_transmission_switching`
(its big-M switching disjunction relaxes into a "phantom transport" LP — line
statuses go fractional, the DC-physics rows go slack, and power flows on
partially open lines with no Kirchhoff coupling; no polynomial-size formulation
with a tight relaxation is known, and the package's default `relax_integer=true`
is what downstream users consume). `security_constrained_dc_opf` replaces it as
the second network variant.

## Dispatch core (`standard`, `reserves`, `storage`, `hydrothermal`)

A look-ahead security-constrained economic dispatch (SCED) run after unit
commitment.

### Data grounding

- **Technology catalogue** (`ENERGY_TECHNOLOGIES`): nuclear, coal, CCGT, gas CT,
  oil, biomass, hydro, wind and solar, each with ranges for nameplate capacity
  (log-uniform), must-run floor, hourly ramp rate (nuclear 2–5 %/h, coal 8–18 %/h,
  CCGT 30–60 %/h, peakers unconstrained), marginal cost (\$/MWh, with a regional
  fuel-price index for fossil units) and emission rate (tCO2/MWh).
- **Zones** are placed on a 100 × 100 map; tie-lines form a Euclidean minimum
  spanning tree plus short meshing ties (≈ 1.4 ties per zone). Each zone gets a
  lognormal size weight and a resource mix (wind-, solar-, hydro- or coal-rich
  regions). Tie capability is 15–45 % of the smaller endpoint's peak, raised so
  every zone — and every pair of neighbouring zones — can import its coincident
  deficit with a 25 % margin (otherwise two adjacent importers form a tiny,
  trivially infeasible pocket); losses grow with distance
  (0.5–6 %), and a small wheeling charge discourages circular flows.
- **Commitment**: nuclear is always committed; coal, CCGT, biomass and hydro are
  committed (must-run floor) with probabilities 0.6/0.5/0.7/0.4, and units are
  decommitted until no zone's floors exceed 75 % of its lightest natural hour.
- **Weather**: zone-level wind (AR(1), hourly persistence 0.9), solar (clear-sky
  bell × daily cloudiness) and hydro (seasonal level) profiles with unit noise;
  peakers carry occasional forced outages.
- **Load**: each zone mixes residential, commercial and industrial 24-hour
  shapes, with a weekday/weekend effect, an AR(1) daily weather factor and 1.5 %
  hourly noise. Zone peaks follow firm capacity with an import/export imbalance
  factor and are scaled to a system planning margin (firm capacity / coincident
  peak).

### Sizing

The horizon snaps `0.75·√target` to a natural planning length (4, 6, 8, 12, 16,
24, 36, 48, 72, 96, 120, 144 or 168 hours: 24 h at 1k variables, 72 h at 10k, a
week from ~50k). The per-period column budget is `round(target / T)`; the zone
count is about budget / 16 (at least 2), and the fleet (plus storage devices or
hydro plants) is sampled against the remaining budget so the column count is
within one unit per period of the target. Growth therefore comes from zones and
fleet, not by replicating identical blocks: 100k variables is a week-long
dispatch of ~37 zones and ~500 units.

### Formulation

Sets: zones `z`, ties `l = (a → b)`, units `g` (zone `z(g)`), periods `t`.

```text
x[g,t]           output, lb[g,t] ≤ x ≤ ub[g,t]
flow_fwd[l,t]    sent a → b, 0 ≤ · ≤ cap_l
flow_bwd[l,t]    sent b → a, 0 ≤ · ≤ cap_l

lb[g,t] = floor_g · cap_g · avail[g,t],  ub[g,t] = cap_g · avail[g,t]
(period 1 also intersects [x0_g − RD_g, x0_g + RU_g] around the pre-horizon output)

zonal balance (=):  Σ_{g∈z} x[g,t] + Σ_{l into z} (1 − loss_l)·sent_l − Σ_{l out of z} sent_l (+ variant terms) = D[z,t]
ramping (ranged):   −RD_g ≤ x[g,t] − x[g,t−1] ≤ RU_g        (ramp-limited units, t ≥ 2)
objective:          min Σ cost_g x[g,t] + Σ wheel_l (flow_fwd + flow_bwd)
```

### Feasible construction (shared)

Tie flows are planted first (10–45 % of capability from the cheaper zone toward
the dearer one, following the load shape). Each zone's controllable units then
track its natural residual load — load minus renewables (run at 90–100 % of
availability), net imports and any planted storage/hydro output — inside their
ramp windows: every unit moves to the same fraction of its range, clamped to the
window (bisection). Period 1 is dispatched without a window and becomes the
pre-horizon output. Zonal demand is then **defined** as the resulting supply plus
net imports, so the planted `EnergyDispatchWitness` satisfies every balance row
exactly; planted flows are halved until every zone keeps ≥ 30 % of its natural
load. Infeasible instances start from this planted data and add one contradiction.

### `standard`: emissions budget

One horizon-wide cap-and-trade row couples every emitting unit in every period:
`Σ_{g,t} e_g x[g,t] ≤ emission_cap`. Feasible caps sit 3–15 % above the witness's
emissions; `unknown` caps are 60–100 % of the business-as-usual (load-tracking,
no-trade) emissions, with a tighter planning margin (1.02–1.30).

Infeasible modes (`EnergyAggregateCertificate`):

- `:emissions_budget` — summing the zonal balances gives `Σ_g x[g,t] ≥ D_t`
  (losses are nonnegative), so emitting units must cover `D_t − Σ_clean ub −
  Σ_emitting lb`; at the lowest emission rate that is a lower bound on horizon
  emissions. The cap is placed strictly between the emission row's own minimum
  activity (what presolve sees) and 93 % of that bound. HiGHS presolve cannot see
  it; simplex needs hundreds to thousands of iterations.
- `:import_pocket` — a connected set of ≥ 2 zones (15–35 % of the system) has
  its peak-hour demand raised 6–15 % above local available capacity plus the
  delivered capability of the ties crossing into it. The demand is spread in
  proportion to each zone's own supply (local capacity + its share of the
  crossing ties), so every zone carries the same relative deficit, and the ties
  inside the pocket are raised to 1.2 × the pocket's total deficit: no zone and
  no proper subset of zones is short on its own (small pockets are exactly what
  HiGHS's bound propagation detects), only the whole pocket is.

### `reserves`: reserve co-optimization

Adds 10-minute spinning reserve `spin[g,t]` (capped at 1/6 of the hourly ramp
for thermal units, 1/3 for hydro, 50 % / 40 % of capacity for gas / oil
peakers) and 30-minute non-spinning reserve `nonspin[g,t]` (half the hourly ramp
for thermal units, the full hourly ramp for hydro, full capacity for peakers);
wind, solar and nuclear offer neither. Rows:

```text
headroom:        x[g,t] + spin[g,t] + nonspin[g,t] ≤ ub[g,t]
ramp with spin:  x[g,t] + spin[g,t] − x[g,t−1] ≤ RU_g,   x[g,t−1] − x[g,t] ≤ RD_g
N-1 contingency: contingency[t] ≥ x[h,t] + spin[h,t]  (largest units h),   Σ_g spin[g,t] ≥ contingency[t]
operating:       Σ_g (spin + nonspin)[g,t] ≥ R_t       (5–12 % of load)
local spinning:  Σ_{g∈z} spin[g,t] ≥ ρ[z,t]            (1–3 % of zonal load)
```

Reserve offers are priced (\$2–9/MW-h spinning, \$0.5–4 non-spinning). The
feasible witness carves reserves from the planted dispatch's headroom and ramp
slack (70 % of the room); requirements are set at or below what it carries, and
only units whose loss its spinning reserve covers with 10 % margin enter the
contingency set. `:reserve_scarcity` raises the stressed hour's operating
requirement so that load plus requirement exceeds total available capacity by
4–10 %, while load alone fits and the requirement row alone is satisfiable;
exposing it needs every zone's balance, every headroom row and the requirement
together. Hours with zero available capacity are excluded from reserve-scarcity
planting, and the certificate requires a strictly positive deficit computed from
the final demand and reserve requirement. `unknown` uses natural requirements,
a planning margin of 0.98–1.25, and as contingencies the largest units the fleet's
spinning capability could cover.

### `storage`: batteries and pumped hydro

About one device per 25 per-period columns, sited in proportion to zonal peak
load; fleet power is 10–25 % of the system peak. Batteries: 2–4 h, 85–92 %
round trip, \$2–8/MWh cycling cost. Pumped hydro: 6–12 h, 72–80 % round trip.

```text
charge[s,t] ≤ P_ch,  discharge[s,t] ≤ P_dis,  soc_min ≤ soc[s,t] ≤ soc_max
soc[s,t] − η_ch·charge[s,t] + discharge[s,t]/η_dis = soc[s,t−1]   (soc_initial at t = 1)
soc[s,T] ≥ soc_initial      (folded into the last period's bound)
storage net discharge (discharge − charge) enters its zone's balance
```

The witness tracks load net of a planted daily cycle — charge in the lowest-load
hours of each day, discharge the same energy times the round-trip efficiency in
the highest-load hours, returning exactly to the initial level.
`:energy_limited_peak` raises the load over a window of consecutive peak hours
above available generation by 6–15 % more than the storage fleet's usable energy
`Σ η_dis·(soc_max − soc_min)`, while every single hour stays within generation
plus storage *power*: summing the window's balances and state-of-charge rows
(`discharge − charge ≤ η_dis·(soc[t−1] − soc[t])`) exposes it, no single row
does. `unknown` uses a planning margin of 0.80–1.12 and natural initial levels.

### `hydrothermal`: cascaded reservoirs

The thermal and renewable fleet (no `:hydro` units) plus river basins of 2–5
reservoir plants (occasionally a tributary joins further down), about one plant
per 12 per-period columns:

```text
release[r,t] ≤ Q_r,  spill[r,t] ≥ 0,  V_min ≤ volume[r,t] ≤ V_max,  volume[r,T] ≥ target_r
volume[r,t] − volume[r,t−1] + release[r,t] + spill[r,t] − Σ_{u→r} (release + spill)[u, t − delay_u]
    = inflow[r,t] (+ volume_initial at t = 1, + pre-horizon releases of u when t − delay_u < 1)
environmental flow: release[r,t] + spill[r,t] ≥ min_release_r     (half of the plants)
productivity_r · release[r,t] enters the plant's zone balance
objective adds 0.05·spill − water_value_r · volume[r,T]
```

Delays are 0–3 h; productivity 0.2–1.5 MW per m³/s (head-dependent, non-±1);
turbine capacity 50–600 m³/s; storage 12–400 hours of turbine flow; inflows are
basin-correlated with daily persistence (headwaters 15–50 % of `Q`, intermediate
2–12 %). The water value is a \$20–45/MWh price times the productivity of every
plant the stored water can still pass. The witness simulates a peaking release
schedule down each cascade inside the reservoir band (topping up inflow if even
the environmental flow would breach the floor) and sets end-of-horizon targets at
or below the simulated final volumes. The drought certificate
(`HydroDroughtCertificate`): one basin's reservoirs start 2–8 % above their
floors and may not end below their start, inflows fall to 20–40 %, and the
outlet's minimum flow over the horizon exceeds by 8–20 % all the water that can
reach it — initial storage above the end floors, every inflow, and pre-horizon
releases arriving during the horizon. Only the basin's water-balance rows summed
over every period show it. `unknown` uses wet-to-dry inflows (0.5–1.4×), natural
end targets (80–105 % of the initial content) and a planning margin of 0.88–1.20.

## DC network family (`dc_opf`, `security_constrained_dc_opf`)

### Grid and fleet

`_energy_grid` places buses around load centres (70 %) over a sparse rural
background (30 %) on a map whose side grows like √B. Lines come from
6-nearest-neighbour candidates found by spatial hashing (O(B)): a Kruskal
spanning forest, links between forest components, an extra-high-voltage (500 kV)
backbone — a spanning tree over the load centres' hub buses plus some meshing —
then a second connection for most radial leaves and short candidates until the
line target (1.3–1.6 lines per bus) is met. Voltage classes set per-km reactance
and rating: 500 kV backbone (x ≈ 0.00011 p.u./km, 2000–3200 MW), long lines mostly
345 kV (0.00033 p.u./km, 800–1500 MW), the rest 115–230 kV (0.001 p.u./km,
150–450 MW). Susceptances are `1/x` on a 100 MVA base with angles in
centiradians, so `flow[MW] = B_l·Δθ`, clamped to [5, 1000].

Generators (0.25–0.40 per bus) come from the shared technology catalogue with
snapshot availability for a random hour (solar follows the sun, wind a
system-wide level with local noise); committed units have must-run floors. About
20 % of buses are switching stations without load; the rest draw lognormal load
shares, then each ~50-bus map region's share of load is moved 85 % of the way
toward its share of generation capacity (planned grids keep generation and load
regionally balanced, which keeps angle spreads moderate as grids grow).

### Formulation (B-θ, as in MATPOWER)

```text
p[g]             pmin_g ≤ p ≤ pmax_g
θ[b]             −angle_limit ≤ θ ≤ angle_limit,  θ[ref] = 0 (fixed)
thermal (ranged): −rating_l ≤ B_l·(θ_from − θ_to) ≤ rating_l
nodal balance (=): Σ_{g at b} p[g] − Σ_{l ∋ b} B_l·(θ_b − θ_other) = d_b
objective:        min Σ cost_g p[g]
```

The bus-angle bound (at least 60°, or 1.3× the reference dispatch's largest
angle) is physically motivated and also keeps the angles from being free
columns: with free angles HiGHS's aggregator eliminates ~80 % of them (a partial
Kron reduction) and presolve kept only ~30 % of the model; with bounded angles it
keeps ~80 %. Columns: `n_generators + n_buses`, exactly the target. Rows:
`n_buses + n_lines`.

### `dc_opf`

- `feasible`: demand is 35–75 % of the way from `Σ pmin` to `Σ pmax`; every
  generator moves the same fraction up its range, the reduced Laplacian is
  solved (sparse factorization) for the angles, and each rating is widened to at
  least 1.15 × the witness flow + 1 MW (`DCPowerFlowWitness`).
- `infeasible`: the feasible data plus a load pocket — a BFS-connected set of
  2–6 % of the buses (≥ 3) whose demand exceeds its local generation plus the
  ratings of the lines crossing into it by 6–15 %. The load is spread in
  proportion to each bus's own supply (local generation + its crossing lines),
  so every bus carries the same relative deficit, and lines inside the pocket
  are strengthened to 1.2 × the pocket's deficit: no bus and no proper subset of
  the pocket is short on its own (small pockets are what presolve's bound
  propagation finds), only the whole pocket is.
  Summing the pocket's balance rows gives `Σ_{g in S} p − Σ_{b in S} d =` net
  outflow across the cut, bounded by the cut's ratings: a contradiction from LP
  rows alone (`DCPocketCertificate`). Presolve does not detect it.
- `unknown`: natural ratings — the reference (proportional-dispatch) flows times
  a per-line slack (1 + lognormal, median 1.25) and, on the bulk (345/500 kV,
  non-radial) network, one system-wide stress factor in [0.65, 1.05]; local and
  radial lines are never rated below their reference flow, and every bus keeps
  incident capability for 110 % of its import and must-run export needs. Below 1
  the bulk grid is tighter than the reference flows and redispatch must relieve
  the congestion — which may or may not be possible.

### `security_constrained_dc_opf`

Preventive N-1 SCOPF. The dispatch is shared by the base case and `C` contingency
states; each state has its own angles, balance rows and thermal rows, with the
outaged line removed and short-term emergency ratings (1.1–1.3 × normal) after a
contingency. The contingency list is the output of a screening pass: the `C`
most heavily loaded non-bridge lines (outages that do not island the grid) under
the reference dispatch, with `C = clamp(round(√target / 8), 1, 60)` (40 at 100k).
Columns `n_generators + (1 + C)·n_buses` (exactly the target), rows
`(1 + C)·(n_buses + n_lines) − C`. The structure is block-angular — `1 + C`
network copies linked only through the dispatch columns, the classic target of
Benders and column-generation decompositions.

- `feasible`: the base and every post-contingency Laplacian are solved for the
  reference dispatch; normal ratings cover 115 % of base flows and emergency
  ratings 110 % of the worst post-contingency flows (`SCOPFWitness`).
- `infeasible`: an N-1 load pocket — a connected bus set on one side of a
  screened line `k`; the base case can still serve it (`k` is rated as its main
  feeder), but after losing `k` local generation plus the emergency ratings of the
  surviving feeders fall 6–15 % short. Pockets are only accepted when every
  pocket bus row stays satisfiable on its own in every state (an outage can
  remove a bus's only internal line). The certificate names the contingency.
- `unknown`: natural N-1 planning (ratings cover the reference base and
  post-contingency flows with the same slack and bulk stress as `dc_opf`).

## Testing

`test/problem_types/energy.jl` covers the registry, exact column and row
formulas for every variant, technology/data invariants, constructor-only sizing
at 100k, every planted witness checked row by row against the built model with
`primal_feasibility_report` (no solver), the reduced-Laplacian arithmetic of the
DC witnesses (balance residuals, flows, angle bounds, every SCOPF state), every
certificate recomputed from the struct fields together with the "no single-row
contradiction" property, reproducibility, and HiGHS-backed contracts (feasible →
optimal, infeasible → infeasible, and `unknown` producing both outcomes).
