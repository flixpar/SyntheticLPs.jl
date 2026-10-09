# Economic Planning

Dynamic economy-wide planning LPs: the family of Netlib's `PILOT*` models (the
Stanford PILOT energy–economy model), Dantzig's dynamic Leontief staircase
models, development-planning LPs such as `DFL001`, and today's
TIMES/MARKAL/MESSAGE energy-system models. These are historically central real
LPs and notoriously hard for simplex: staircase multi-period structure, many
equality balance rows, hybrid physical/monetary units with coefficients spanning
many orders of magnitude, and heavy degeneracy. Both variants build pure
continuous LPs (`relax_integer` is a no-op), use a constructor-local RNG, and
`build_model` does no sampling.

## Variants

| Variant | Model class | Key structure | Domain grounding |
|---|---|---|---|
| `dynamic_leontief` (default) | continuous LP | sparse input–output and capital matrices per period, capacity accumulation with gestation lags, labor, import ceilings, external debt | Dantzig's dynamic Leontief model, PILOT, development planning |
| `energy_system` | continuous LP | technology-rich process network (supply → conversion → end-use), timesliced electricity, vintaged capacity, growth limits, peak reserve, emission caps/budget, multi-region trade | TIMES / MARKAL / MESSAGE |

The two matrix profiles are deliberately different: the Leontief model is a
dense-inverse economy (every value content propagates through every sector and
period), the energy model a sparse, many-technology generalized-flow network.

## `dynamic_leontief` Formulation

Sets: sectors `s = 1..n` (ordered primary, manufacturing, construction,
services; the first `n_tr` — primary and manufacturing — are tradable), periods
`t = 1..T` of `Δ ∈ {1, 2, 5}` years (flows per year), labor skill classes `k`.

Columns (exactly `T (3n + 2 n_tr + 2)`):

```text
x[s,t] >= 0     gross output                N[s,t] >= 0   new capacity entering at t+1
K[s,t] >= 0     capacity at start of t+1    C[t] >= C_min[t]  aggregate consumption
imp[i,t] >= 0   imports (tradable i)        0 <= ex[i,t] <= ex_max[i,t]  exports
F[t] <= F_max[t]  net foreign debt at start of t+1 (free below)
```

Rows (`T (3n + n_tr + n_skills + 1) + T`):

```text
balance[s,t]:    x - A x + imp - ex - B((1-φ)∘N_t + φ∘N_{t+1}) - d C_t = G_t      (equality)
capacity[s,t]:   x_t <= K_{t-1}                       (K_0 = initial capacity, data)
accumulation:    K_t = σ∘K_{t-1} + Δ N_t              (σ = (1-δ)^Δ)
import_limit:    imp_t <= ρ∘x_t                       (tradables)
labor[k,t]:      ℓ_k' x_t <= L[k,t]
debt[t]:         F_t = R F_{t-1} + Δ (p_m' imp_t - p_e' ex_t)
no_decline[t]:   C_{t+1} >= C_t
consumption_target:  Σ_t w_t C_t >= W                 (plan target, w_t = Δ β^{Δ(t-1)})
terminal bound:  K[:,T] >= K_term
```

Objective: maximize `Σ_t Δ β^{Δ(t-1)} C_t + β^{ΔT} (v' K_T - F_T)` — discounted
consumption plus the value of terminal capital net of terminal debt. Each period
couples to the next only through capital stocks, in-progress investment
(gestation share `φ`: investment goods for capacity entering at `t+1` are partly
bought one period ahead) and debt: the classical staircase.

## `dynamic_leontief` Data Grounding

- **Input–output table.** Intermediate-input shares per column follow the block
  (primary 30–55% of gross output, manufacturing 55–75%, construction 50–65%,
  services 25–45%), so value-unit column sums are below one: the table is
  productive (Hawkins–Simon). Suppliers are sampled in production-stage order
  within value chains (`max(1, round(n/40))` chains), from the buyer's own
  sub-industry cluster, or from at most eight hub suppliers (energy, trade,
  transport, finance) — the near-block-triangular shape disaggregated real tables
  have after triangularization (Simpson & Tsukui). Small economies get dense
  tables (up to 75% of sectors per column); large ones about `6 + 2 ln n`
  suppliers per column.
- **Capital.** Capital per unit of capacity by block (primary 1–4, manufacturing
  0.6–1.8, construction 0.3–0.8, services 0.5–3), split into structures (from a
  construction sector) and equipment (1–4 equipment-manufacturing suppliers,
  mostly within the buyer's value chain); depreciation follows the
  structures/equipment mix; gestation shares are higher for structure-heavy and
  primary sectors. Columns are capped so `A + BΓ` (intermediates plus growth
  investment) stays productive.
- **Labor, trade, demand.** Labor intensities by block (agriculture 15–45
  thousand workers per $bn, energy/mining 1–4, ...) times an economy-wide
  productivity level, split into 1–3 skill classes; import ceilings 10–70% of
  domestic output; CIF import prices above, export prices below the domestic
  price; consumption and government bundles concentrated on services.
- **Hybrid units.** As in PILOT's hybrid tables, most primary and some
  manufacturing sectors are measured in physical units (PJ, Mt) with a sampled
  price (0.005–0.8 $bn per unit). The stored matrices are a diagonal similarity
  `D⁻¹ A D` of the value table, so productivity is preserved while coefficients
  span 4–7 orders of magnitude.

## `dynamic_leontief` Feasibility Control

The reference trajectory is a balanced-growth path: the stationary output vector
solves `(I - A - BΓ + diag(ηρ)) x̄ = d C̄ + Ḡ + ē` (an M-matrix system, solved by
Jacobi iteration — sparse LU fill is near dense on these patterns), every period
scales it by the growth factor, capacity sits at a planted utilisation of
78–95%, and the last period is re-solved because investment for post-horizon
capacity is not in the model.

- `feasible`: the trajectory is stored as a typed `LeontiefWitness` (model
  layout). Labor supply (3–12% slack), debt ceilings, export ceilings, terminal
  capacity, consumption floors (80–97% of the reference) and the plan target
  (85–97% of the reference's discounted consumption) are set around it.
- `infeasible`: the plan target is set 10–30% above `Σ_t w_t b_t`, where `b_t` is
  a provable upper bound on period-`t` consumption, with a typed
  `LeontiefCertificate` aggregating the rows of every period. With
  `M = I + diag(ρ̃) - A`, any `π ≥ 0` gives `π'Mx_t ≥ π'(G_t + dC_t)`
  (investment, exports and imports below the ceiling only add demand); if
  `M'π ≤ Σ_k μ_k ℓ_k + ν` then `π'Mx_t ≤ μ'L_t + ν'K_0`. `π` is the labor content
  of the import-augmented Leontief inverse `M⁻ᵀ ℓ_k` for the period's binding
  skill (`:labor`), and in period 1 possibly `M⁻ᵀ e_s` for a bottleneck sector
  whose installed capacity binds first (`:labor_capital`). Scaling each period's
  multipliers by `w_t / (π'd)` cancels every consumption column against the
  target row.
- `unknown`: the target sits 10–80% of the way from the reference path's
  discounted consumption to `Σ_t w_t b_t`. The true frontier (measured by
  maximizing `Σ_t w_t C_t`) lies 15–75% of the way, typically ~40%, because the
  bound ignores investment needs; instances land on both sides (about 40–50%
  infeasible at 150–1,200 columns).

Why a cumulative target rather than per-period floors: an earlier design put the
infeasibility on a single period's consumption floor. HiGHS presolve's bound
propagation through the balance rows replays the Leontief-inverse iteration and,
in the MPS row order SimplexRL uses, refuted most such instances with zero
simplex iterations. The multi-period target needs upper bounds on every period's
consumption, which propagation cannot derive; all infeasible instances now need
real simplex work (presolve keeps ~75–80% of the columns).

## `energy_system` Formulation

Commodities per region: primary fuels (`COA`, `GAS`, `OIL`, `URN`, `BIO`),
secondary fuels (`PET`, `H2`, `HET`), electricity (`ELC`, balanced per
timeslice) and service demands (`RH` residential heat, `RA` appliances, `IP`
process heat in PJ; `TP` passenger transport in Gpkm; `TF` freight in Gtkm).
Processes come from a 37-row technology database, introduced in six modules as
the target grows:

| Module | Adds |
|---|---|
| 1 | gas and coal supply, coal and gas CCGT plants, onshore wind; appliances; gas boilers and heat pumps |
| 2 | solar PV, gas turbines, nuclear (uranium); industrial heat (gas, coal, electric) |
| 3 | crude supply and refineries; ICE/BEV cars, diesel/BEV trucks; oil heating |
| 4 | biomass supply, biomass power, hydro; biomass heating and industry |
| 5 | electrolysis, SMR, hydrogen turbines; FCEV, hydrogen trucks, hydrogen industry |
| 6 | district heat (gas/biomass boilers, large heat pumps, gas CHP), offshore wind, rail, resistive heating |

Columns (flat, see `ESLayout`): `act` (per timeslice for generators, annual
otherwise), `cap` and `ncap` per capacity technology and period, interconnector
`flow` per line, direction, period and slice. Rows:

```text
bal[r,c,t]        Σ outputs - Σ inputs  = 0 (fuels)  >= 0 (heat)  >= demand (services)
bal_elc[r,t,s]    generation + η·imports - exports - Σ_users φ_s·input >= 0
capact            act[k,t,s] <= avail[k,s] · capfac[k] · dur[s] · cap[k,t]    (31.536 PJ/GW-yr)
transfer[k,t]     cap[k,t] - Σ_{v > t - life} ncap[k,v] = residual[k,t]          (vintages)
growth[k,t]       ncap[k,t] - g_k ncap[k,t-1] <= seed_k                        (market growth)
peak[r,t]         Σ credit·cap - (1+margin)·(peak-slice load)/(31.536·dur) >= 0
emission[t]       Σ e·act <= cap_t ;   budget: Σ_t Δ Σ e·act <= budget
reserve[k]        Σ_t Δ act[k,t] <= cumulative reserve      (domestic supply steps)
bounds            supply steps (step size / import capacity), cap <= potential,
                  ncap[k,1] <= first-period build, flow <= line capacity
```

Objective: discounted investment (with salvage for life beyond the horizon),
fixed O&M on installed capacity, variable O&M, fuel supply and transmission
costs, in M$.

## `energy_system` Data Grounding

Technology parameters are sampled from realistic ranges: efficiencies 0.30–0.95
(COPs 2.5–4, transport 0.55–6 Gpkm or Gtkm per PJ), dispatch availability
0.80–0.95, VRE capacity factors (wind 0.25–0.38, offshore 0.38–0.50, solar
0.12–0.22, hydro 0.35–0.50) shaped by season and daypart, investment costs from
~3 M$ per PJ/yr (boilers) to 5,000–8,000 M$/GW (nuclear), lifetimes 12–80 years,
emission factors 0.056–0.095 Mt CO2/PJ. Renewables come in up to four
resource-quality classes (best sites first, falling availability, rising cost),
devices in up to four efficiency classes (rising efficiency and cost). Supply is
cost-stepped: domestic steps (with cumulative reserves) and a capacity-limited
import step. Timeslices are season × daypart (1–24 slices) with consumption
profiles (heating, appliances, EV charging, industry, flat) and availability
profiles (solar zero at night, winter-peaking wind, spring hydro). Regions differ
in size (×0.2–5), resource endowment and parameters, and are linked by a
minimum-spanning-tree-plus-extra-edges interconnector network with 0.5–5 GW lines
and 0.5–10% losses.

## `energy_system` Feasibility Control

The reference plan is built layer by layer: devices meet demands by reference
shares that shift toward electrification over the horizon; hydrogen and heat
producers follow reference shares (CHP covers a share of district heat);
electricity comes from VRE at a rising target share (full availability, surplus
curtailed) and dispatchable plants filling the residual load slice by slice;
refineries and supply steps close the fuel balances. Capacities (with 2–15%
margin), residual stocks of incumbent technologies, greedy vintage builds, growth
seeds, potentials, step sizes, import capacities, reserves, peak reserve
(topped up with gas turbines) and emission limits are all set around it.

Certificates use dual "values" `π ≥ 0` of the commodity balances: the greatest
fixed point of `π_out · out ≤ θ e + Σ π_in · in` over every process without a
capacity potential (fossil and conversion plants, devices, interconnectors), so
every unbounded column has a nonpositive aggregated coefficient. Processes whose
value exceeds their emission charge — supply steps and potential-limited plants
(renewables, nuclear, hydro, CHP, rail) — are paid for through their bounds:
capacity–activity multipliers plus either the potential bound or, when smaller,
the build-out the growth rows allow (transfer- and growth-row multipliers
unrolled over the vintage window). The certified quantity is
`Σ π_d D_d − (bounded value)`.

- `feasible`: typed `EnergySystemWitness` (activity, capacity, new capacity,
  zero interconnector flows) satisfying every row.
- `infeasible` (typed `EnergySystemCertificate`, all margins 15–40%):
  - `:emission_cap` — `θ = 1`, fossil fuels valued 0: the certified minimum
    emissions of a later period exceed its cap;
  - `:carbon_budget` — the same, summed over all periods (weights `Δ`), below
    the cumulative budget — a multi-period certificate;
  - `:supply_shortfall` — `θ = 0`, primary energy valued 1 (π = minimum
    primary-energy content over the most efficient chains): a permitting freeze
    holds clean capacity at today's stock, and from some period on domestic
    production and import capacity are cut below the certified requirement
    (demand surges if the existing clean stock alone nearly covers it).
- `unknown`: one ambition level per instance places every emission cap between
  the certified floor and the reference emissions; about half the instances are
  feasible.

## Sizing

| Variant | Columns | Cap | Fidelity |
|---|---|---|---|
| `dynamic_leontief` | `T (3n + 2 n_tr + 2)`, `T ≤ 25`, `n ≤ 10,000` | 1,000,000 (`LEONTIEF_MAX_VARIABLES`) | ≤ 4% above 100 columns, < 0.1% above 10k; minimum 45 |
| `energy_system` | `T (R c + 2 S L)` (`c` columns per region-period, `L` lines) | 1,000,000 (`ENERGY_SYSTEM_MAX_VARIABLES`) | ≤ 1.5% above 100 columns, ≤ 0.2% above 1k; minimum 44 |

Both are roughly square (rows ≈ 0.86–0.95 × columns). Larger targets add
sectors and periods (Leontief) or regions, timeslices, modules, technology
classes and supply steps (energy). A 1,000,000-column instance builds in about
5–8 s with under 2.5 GB peak memory.

## Solver Profile

HiGHS audit (seed 0; presolve-on dual simplex, 60 s limit):

| Variant | Target | Cols | Rows | Presolved cols / rows | Build | Status (feasible / infeasible / unknown) |
|---|---|---|---|---|---|---|
| `dynamic_leontief` | 10k | 10,003 | 8,855 | 77% / 70% | 0.04 s | OPTIMAL 13 s / INFEASIBLE 2.4 s / OPTIMAL 13 s |
| `dynamic_leontief` | 100k | 100,002 | 88,438 | 78% / 71% | 0.45 s | time limit (all three) |
| `energy_system` | 10k | 10,016 | 9,274 | 84% / 81% | 0.04 s | OPTIMAL 0.6 s / INFEASIBLE 0.1 s / OPTIMAL 0.9 s |
| `energy_system` | 100k | 99,984 | 90,025 | 89% / 87% | 0.2–0.45 s | OPTIMAL 32 s / INFEASIBLE 3.9 s / time limit |

No infeasible instance of either variant is refuted by presolve alone (checked
over 8 seeds at 1k and 4k for every certificate mode).

Like PILOT, `dynamic_leontief` has a dense basis inverse: dual prices are value
contents, which propagate through the whole economy and across periods via the
capital stocks, so BTRAN rows of a typical optimal basis are 70–80% dense and LU
fill is ~20×, independent of the hub suppliers, the consumption bundle, labor or
debt rows (each ablated). Per-iteration cost is therefore milliseconds at 10k
columns and full solves beyond ~20k columns take minutes. Use it at 1k–20k
columns for complete pivot paths; larger instances are realistic stress tests.

## References

- G. B. Dantzig, "Optimal solution of a dynamic Leontief model with
  substitution", *Econometrica* 23 (1955).
- D. Hawkins and H. A. Simon, "Note: Some conditions of macroeconomic
  stability", *Econometrica* 17 (1949).
- J. Simpson and J. Tsukui, "The fundamental structure of input-output tables",
  *Review of Economics and Statistics* 47 (1965).
- G. B. Dantzig et al., the PILOT energy–economic model (Stanford Systems
  Optimization Laboratory); Netlib LP test set `PILOT`, `PILOT87`, `PILOT.JA`,
  `PILOTNOV`, `DFL001`.
- R. Loulou et al., *Documentation for the TIMES Model* (IEA-ETSAP, 2016).
- IIASA MESSAGEix documentation (market-penetration / growth constraints).
