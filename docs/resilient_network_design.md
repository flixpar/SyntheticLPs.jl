# Resilient Network Design

`resilient_network_design/standard` is a two-stage network build-and-harden
problem under spatially correlated failure scenarios: decide which candidate
links to build and which of those to harden, then route each scenario's
required flow over the links that survive. It is a MIP (binary build/harden)
returned as its LP relaxation by default; the relaxation keeps the defining
structure because link capacity is `capacity × build` (or `× harden` for failed
links) in every scenario.

## Data

- **Sites** are uniform on a 100 × 100 map. Candidate links are geometric: in
  a random order each node joins its nearest earlier node (so the first
  `n_nodes − 1` links form a spanning tree), then shortcuts are added from each
  node's nearest neighbours, shortest first, until the edge budget is used.
- **Costs**: build cost is proportional to link length (plus a fixed part),
  hardening costs 30–90% of building, routing cost per unit is proportional to
  length.
- **Scenarios** are regional hazards (earthquake, flood, storm): a center and
  a radius; a link fails with probability `0.85 exp(−(r/radius)²) + 0.03`
  where `r` is the distance from its midpoint to the center. Each scenario has
  a source, a sink, and a lognormal demand.

## Formulation

```text
min  Σ_e (build_cost_e build_e + hardening_cost_e harden_e)
     + (1/S) Σ_{e,s} routing_cost_e (forward[e,s] + reverse[e,s])
s.t. Σ_e (build_cost_e build_e + hardening_cost_e harden_e) ≤ design_budget
     harden_e ≤ build_e                                              ∀ e
     forward[e,s] + reverse[e,s] ≤ capacity_e · (failed[e,s] ? harden_e : build_e)   ∀ e, s
     flow balance at every node of every scenario (source +demand, sink −demand)
     build, harden ∈ {0,1}; flows ≥ 0
```

Sizing: `S = clamp(round(sqrt(target)/3), 2, 8)` scenarios,
`E ≈ target / (2(S+1))` links, `n_nodes ≈ E / 1.8`. Variables
`2E(1 + S)`; rows `1 + E(1 + S) + n_nodes S`.

## Feasibility control

- `feasible`: the spanning tree is built and hardened (so it survives every
  scenario), tree capacities are raised to `1.15 × max demand + 1`, and the
  budget is `1.05 ×` the tree's cost `+ 1`. Each scenario's demand is routed
  along the unique tree path. Stored as `ResilientNetworkWitness`; the test
  file checks it against every row with `primal_feasibility_report`.
- `infeasible`: a hardening-budget shortfall. For each scenario a *district*
  is planned: a breadth-first ball of about `sqrt(n_nodes)` nodes around its
  sink that avoids its source and every other scenario's endpoints (so no other
  scenario must reach into it), grown until its boundary has at least six
  links, with enclosed nodes filled in. The scenario's hazard is recentred on
  the district and takes out every access link; the access links are resized
  to total `U(1.25, 1.45) ×` the demand that must cross them, and links on any
  short minimum cut are widened until, with everything built and hardened,
  every scenario's maximum flow is at least `1.25 ×` its demand — capacity is
  never the obstruction. The lower bound on design spend then has two parts
  built from LP rows: *bridges* (a scenario split by a bridge routes its whole
  demand over it, forcing `build`/`harden ≥ demand / capacity`), and the
  district's *knapsack*: summing its balance rows, the demand must cross failed
  access links that carry `capacity × harden`, and hardening costs
  `hardening_cost` per unit plus `build_cost` beyond the forced build level
  (`harden ≤ build`). The scenario with the largest knapsack spend is used,
  and the budget covers the bridges' forced spend plus
  `1 / U(1.08, 1.20)` of the knapsack (`ResilientHardeningBudgetCertificate`).
  The budget still affords every single link, so HiGHS presolve cannot see
  the shortfall. The previous mode (shrinking the region's boundary capacity
  with an unlimited budget) was refuted by presolve's doubleton/aggregator
  substitutions in 3 of 4 instances at 1k: on a small region they sum the
  balance rows exactly as the certificate does. Every audited infeasible
  instance (200–100k, seeds 0–7 at ≤3k, 0–3 above) now needs simplex.
- `unknown`: natural capacities and a budget of `0.45–1.1 ×` the tree's
  cost; whether some design within budget routes every scenario is left to the
  instance (both outcomes occur over seed blocks).

## Model characteristics

- Each scenario is a single-commodity flow block coupled to the others only
  through the shared build/harden columns: a two-stage structure like
  `stochastic_program/standard`, but with network-design linking.
- Presolve keeps about 90% of rows and columns on feasible/unknown instances
  (1k–100k). Large instances (≥ 50k) can take tens of seconds in HiGHS.
