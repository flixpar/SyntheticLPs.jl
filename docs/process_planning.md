# Process Planning

Multi-period planning of refineries and chemical process networks: what to buy,
how to run the plant, what to store and what to blend over a horizon of weeks,
months or years. Three variants share the domain: `refinery` (the default) is the
production-planning LP, `mode_switching` adds the campaign decision that makes it
a MILP, and `capacity_expansion` is the long-range investment model on a chemical
process network.

## Overview

A refinery planning model is the archetypal industrial LP. Crude is bought
against per-period availability, charged to the crude unit, and cut into
fractions at each crude's assay yields. Those fractions are hydrotreated,
reformed, cracked, hydrocracked, coked or alkylated at fixed conversion yields,
banked in intermediate tankage, and finally blended into finished grades that
must satisfy quality windows — octane, vapour pressure, sulfur, aromatics,
cetane, cold flow, density, viscosity. Finished product is sold into a demand
window or carried in product tanks; the objective is the refining margin.

The generator keeps the model linear the way industrial planning models do: every
stream carries a **fixed** quality vector, so a blend's weighted average is a
linear function of the blend volumes. That is why distillation cuts are
*segregated by crude* — the light naphtha of a sweet crude and of a sour crude are
different streams with different sulfur — rather than pooled into a single
naphtha of unknown composition, which would make the quality rows bilinear (the
pooling problem).

`capacity_expansion` covers the other classical member of the family: selecting
and expanding the processes of a chemical complex over a long horizon, following
the multiperiod MILP of Sahinidis, Grossmann, Fornari & Chathrathi (*Comput.
Chem. Eng.* 13, 1989) and Sahinidis & Grossmann (*Oper. Res.* 40, 1992).

Calibration sources: crude archetypes, cut yields and cut properties follow
published assays (Hibernia, Bakken and WTI, as reproduced in J. Jechura,
*Refinery Feedstocks & Products*, Colorado School of Mines, 2019); conversion
yields follow reported commercial ranges (FCC max-gasoline at 55-65 vol%
gasoline, 25-30 vol% LPG, 15-20 vol% LCO, with max-distillate moving roughly ten
points from gasoline to LCO; reforming at 78-85 vol% reformate for 96-102 RON;
hydrocracking at 110-115% volume swell; delayed coking at a coke yield near
1.6 × CCR); product specifications and the price scale follow the multi-period
refinery planning literature (Castillo Castillo & Mahalec; Li, Lin, Su & Xie,
arXiv:2504.08642).

## Generator Data and Sizing

### The flowsheet

Instances are built at one of four refinery configurations, chosen by size:

1. **topping** — crude unit only;
2. **hydroskimming** — naphtha and kerosene hydrotreating, reforming, diesel
   hydrotreating;
3. **cracking** — adds isomerization, VGO hydrotreating, catalytic cracking and
   alkylation;
4. **full conversion** — adds delayed coking, hydrocracking and residue
   hydrotreating.

Optional units are dropped with probability 0.25, and the busiest units (crude
naphtha hydrotreating, catalytic cracking, diesel hydrotreating) are duplicated
into parallel trains at large sizes. The cut slate is one of four, from a
three-cut assay to the full eight-cut slate (LPG, light and heavy naphtha,
kerosene, distillate, gas oil, VGO, residue); cuts the slate does not carry are
lumped into their neighbour. Finished grades are drawn from regular and premium
gasoline, jet, ULSD, heating gas oil, fuel oil, LPG and petcoke, with extra
seasonal or regional grades at large sizes.

Each unit sits at a *stage* and only accepts feed made at a strictly lower stage,
so the flowsheet is always a DAG: coking and the primary hydrotreaters at stage
1, naphtha hydrotreating (which also takes coker naphtha) at stage 2, catalytic
cracking and hydrocracking at stage 2, reforming and diesel hydrotreating at
stage 3, alkylation at stage 4. A stream produced by several units carries the
latest of their stages.

### Data

Crudes are drawn from five archetypes — condensate, light sweet, medium sour,
heavy sour and extra heavy — with API gravity, whole-crude sulfur and a
volumetric yield vector that is perturbed and renormalized per instance. Each
cut inherits the sulfur of its crude (a per-cut multiplier on whole-crude wt%,
carried in weight ppm) and is made heavier and more aromatic when its crude is
heavier. Conversion products sit at their unit's typical severity: reformate at
96 or 102 RON depending on mode, alkylate at 95 RON and 92 MON, FCC gasoline at
92 RON and post-treatment sulfur, hydrocracker diesel at 58 cetane and 3 ppm
sulfur, light cycle oil at 22 cetane and 70% aromatics.

Volumes are thousands of barrels per period, prices are dollars per barrel (so
the objective is in thousands of dollars), and a period is one, two or four
weeks. Nameplate crude capacity is 60-380 thousand barrels per day. Crude prices
track API and sulfur; product prices, demand and crude-unit utilisation follow an
AR(1) path with an annual seasonal swing whose phase depends on the product
(gasoline peaks in summer, heating oil in winter).

### Sizing

With `C` crudes, `T` periods, units `u` with feed sets `F_u` and modes `M_u`,
grades `p` with component sets `B_p`, `n_store` tanked streams, `n_buy` purchased
blendstocks and `n_spot` streams saleable as they are, the models have exactly

```text
refinery:        T * (3C + sum_u (|F_u| + 1)       + sum_p |B_p| + n_store + n_buy + n_spot + 2P)
mode_switching:  T * (3C + sum_u M_u (|F_u| + 3)   + sum_p |B_p| + n_store + n_buy + n_spot + 2P)
capacity_expansion: T * (4I + n_raw + n_sell)
```

variables. For a fixed configuration the refinery count is exactly affine in the
crude count, so the generator draws one configuration per complexity level, pins
the affine coefficients with two probes, solves for the crude count at every
candidate horizon in closed form, and keeps the triple that lands closest to the
target. Ties favour operationally ordinary shapes and the more complete refinery.
Requests land within about 20% of the target from 50 to 20,000 variables and
usually much closer; a small request settles on a topping or hydroskimming
refinery with a coarse assay, a large one on a full conversion refinery running
parallel trains and a dozen grades.

## LP Formulation

Sets: crudes `C`, periods `T`, streams `S`, units `U` (with modes `M_u` and feed
sets `F_u`), grades `P`, qualities `Q`.

Decision variables per period `t`:

- `buy[c,t]`, `run[c,t]`, `ctank[c,t]`: crude purchased, charged and in tank
- `feed[u,s,t]` for `s in F_u`, and the unit throughput `thr[u,t]`
- `blend[s,p,t]` for every admissible component `s` of grade `p`
- `inv[s,t]` for tanked streams, `pur[s,t]` for purchased blendstocks,
  `spot[s,t]` for streams sold as they are
- `sell[p,t]`, `ptank[p,t]`: finished-product sales and tank level

Crude tanks, crude-unit capacity and charge quality:

```math
ctank_{c,t-1} + buy_{c,t} = run_{c,t} + ctank_{c,t}
```

```math
\underline{V}_t \le \sum_c run_{c,t} \le \overline{V}_t,
\qquad \sum_c (\sigma_c - \bar{\sigma}) \, run_{c,t} \le 0
```

where `σ_c` is the sulfur of crude `c` and `σ̄` the metallurgical limit on the
charge. Every stream balances in every period — distillation and conversion
yields in, unit feeds, blends, spot sales and tankage out:

```math
\sum_{c,k:\,s} y_{c,k} \, run_{c,t} + \sum_{u,f} \eta^u_{f,s} feed_{u,f,t}
 + pur_{s,t} + inv_{s,t-1}
 = \sum_u feed_{u,s,t} + \sum_p blend_{s,p,t} + spot_{s,t} + inv_{s,t}
```

Unit throughput is defined and bounded, `thr[u,t] = Σ_f feed[u,f,t]` with
`min[u,t] ≤ thr[u,t] ≤ cap[u,t]`. Finished product balances,
`ptank[p,t-1] + Σ_s blend[s,p,t] = sell[p,t] + ptank[p,t]`, and sales sit inside
the demand window `d^-[p,t] ≤ sell[p,t] ≤ d^+[p,t]`.

Quality is written on the blend. For a volumetric property `q` with an upper
specification `s̄`:

```math
\sum_{b} (Q_{b,q} - \bar{s}) \, blend_{b,p,t} \le 0
```

which is the linear form of "the volume-weighted average stays below `s̄`". A
weight-basis property (sulfur) carries the component density, giving the
mass-weighted average. Each such row is divided by its largest coefficient; the
specification is unchanged — the right-hand side is zero — but the properties
span very different units, and writing every quality row on the same scale keeps
the constraint matrix from spanning seven orders of magnitude.

The objective maximizes margin:

```math
\max \sum_t \Big( \sum_p \pi_{p,t} sell_{p,t} + \sum_s \rho_s spot_{s,t}
 - \sum_c \kappa_{c,t} buy_{c,t} - \sum_s \beta_{s,t} pur_{s,t}
 - \sum_u \omega_u thr_{u,t} - \sum_s h_s inv_{s,t} - \sum_p h_p ptank_{p,t} \Big)
```

### `mode_switching`

Each unit runs at most one of its operating modes per period — a catalytic
cracker in maximum-gasoline, maximum-distillate or maximum-olefins mode, a
reformer at mid or high severity, a hydrocracker swung to diesel, jet or naphtha,
a coker making fuel-grade or anode-grade coke, a diesel hydrotreater at ULSD or
mild severity. Feeds, throughput, a binary run indicator `z[u,m,t]` and a
changeover `sw[u,m,t]` exist per mode, with

```math
\sum_m z_{u,m,t} \le 1, \qquad
\underline{r}_{u,t} z_{u,m,t} \le thr_{u,m,t} \le \overline{V}_{u,t}\,\gamma_m\, z_{u,m,t},
\qquad sw_{u,m,t} \ge z_{u,m,t} - z_{u,m,t-1}
```

and the changeover charged in the objective. Because the rate is gated by the run
indicator, a minimum rate applies only while the unit is actually running —
something the pure LP cannot state, which is why in `refinery` only the units
that run whenever the crude unit does carry one.

### `capacity_expansion`

Processes convert chemicals at fixed ratios on a layered network: raw materials
at the bottom, intermediates in the middle, finished chemicals at the top, with
each process consuming one to three chemicals from strictly lower layers at a
mass yield of 60-95% and producing one chemical at its own layer, plus (sometimes)
a byproduct that is only ever sold. Per process `i` and period `t` the model
carries the operating level `W`, the installed capacity `Q`, the expansion `QE`
and its binary indicator `y`:

```math
Q_{i,t} = Q_{i,t-1} + QE_{i,t}, \qquad
\underline{E}_i y_{i,t} \le QE_{i,t} \le \overline{E}_i y_{i,t}, \qquad
W_{i,t} \le Q_{i,t}
```

with a balance row per chemical and period,
`Σ_i μ_{i,j} W_{i,t} + pur_{j,t} - sell_{j,t} = 0`, purchases under market
availability, sales inside a demand window, and a discounted net-present-value
objective of revenue less feedstock, operating and investment cost (a fixed
charge on `y` plus a linear cost on `QE`), at 7-15% per period.

## Feasibility Controls

**`feasible`.** A complete operation is simulated forward through the flowsheet:
crude is charged in fixed proportions, each stream banks a small fraction of what
is available and splits the rest across the units, blends and spot sales that can
take it, each unit converts its feed at the yields of the mode it runs, and the
last sink of every stream absorbs the residual so the balance rows hold exactly
rather than approximately. Every capacity, tank, availability, purchase limit,
spot outlet and contract is then placed around that operation, and each blend
window is opened far enough to admit the quality the recipe achieves. The
operation is stored as `feasible_witness` and re-checked row by row —
`refinery_plan_satisfies` for the two refinery variants,
`process_expansion_plan_satisfies` for the network model — by arithmetic alone.

**`infeasible`.** One of two structural refutations, stored in
`infeasibility_certificate` and re-derivable from the instance data:

- *contract above the conversion bound*. Give each stream a potential `M` — the
  largest volume of finished product a barrel of it can ever become, computed
  backwards through the flowsheet, taking each feed's best mode. Multiplying the
  stream balances by `M` and summing telescopes into an upper bound on total
  finished production from what the crude menu, the crude unit and the purchased
  blendstocks can supply. The contracted volume is set above it.
- *specification outside the component range*. One grade's published window is
  tightened past every component that may enter it, and the grade is contracted
  out of an empty opening tank. All the coefficients of that quality row are then
  one-signed, so with nonnegative blend volumes the row pins the whole blend at
  zero and the contract cannot be met.

`capacity_expansion` has the matching pair: one chemical's contracted sales above
what the processes making it could produce even under the largest permitted
expansion in every period, or the network's total contracted sales above the
value of the raw material the market can supply (the same potential argument,
applied to chemicals).

Both refinery certificates use only linear rows and each feed's best mode, so
they refute the LP relaxation and every mode assignment at once.

**`unknown`.** Assets are sized from engineering design rules — a unit's typical
fraction of crude charge, a tank's typical days of cover — rather than from the
plan; contracts are drawn from a market view that straddles what the plan
produced; and each quality window is stated at the edge of what the configuration
supports, sometimes just inside it and sometimes just outside. No grade is ever
made unmakeable on its own (that is the requested-infeasible branch's business),
but whether the slate, the units and the specifications can serve all the
contracts together is genuinely open, and the corpus contains both outcomes at
every scale.

## Model Characteristics

- `refinery` is a pure LP: the operating mode is fixed at generation time, so
  `relax_integer` has nothing to relax.
- `mode_switching` is a MILP with one binary per unit, mode and period;
  `capacity_expansion` is a MILP with one binary per process and period. Under
  the package default `relax_integer=true` both are returned as LP relaxations,
  in which a unit may run several modes at once, or a fractional indicator buy a
  fractionally-sized expansion. The planted witnesses are integral, so they are
  feasible for the integer model as well.
- All three objectives maximize; all variables are bounded, so no instance is
  unbounded. Implied bounds (no crude charged beyond the crude unit's capacity,
  no feed beyond its unit's, nothing blended into a grade beyond what it can sell
  or store) are stated explicitly: they cut nothing off, and a simplex given them
  does not have to discover them.
- Rows are dominated by the per-period stream balances, so the constraint matrix
  is sparse and strongly time-staged: the only coupling between periods is
  through the tank levels and, in `mode_switching` and `capacity_expansion`, the
  mode and capacity recursions.

## Practical Notes

- Every intermediate stream can also be sold as it is, at a discount to the
  cheapest grade it could have gone into, against a limited spot volume. That is
  how a refinery actually places naphtha, gas oil and VGO cargoes, and it means
  the model never contains a stream with nowhere to go.
- The quality vector is `(density, sulfur, RON, MON, RVP, aromatics, cetane,
  cold flow, viscosity index)`. Sulfur is carried in weight ppm and is the only
  weight-basis property; viscosity is carried as a blending index, which is how
  viscosity is blended linearly in practice.
- A relaxation of `mode_switching` is not an operating plan: a fractional mode
  mixture has no meaning on the plant floor. Solve it unrelaxed
  (`relax_integer=false`) if you need a plan rather than a bound.
