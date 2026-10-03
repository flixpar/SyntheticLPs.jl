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
- `infeasible`: a region — a breadth-first ball of about `sqrt(n_nodes)` nodes
  around scenario 1's sink, not containing its source — is weakly connected:
  its boundary link capacities are scaled to total `demand / U(1.08, 1.20)`.
  Summing the region's balance rows, the demand must cross the boundary, which
  carries at most its total capacity for any fractional build/harden, so the
  certificate (`ResilientNetworkCutCertificate`) survives relaxation. The budget
  is unlimited, so the cut is the only obstruction. The previous certificate
  was a single sink node, which HiGHS presolve always refuted; with regions,
  presolve still refutes roughly half the instances at 1k and a quarter at 10k,
  the rest need simplex iterations.
- `unknown`: natural capacities and a budget of `0.45–1.1 ×` the tree's
  cost; whether some design within budget routes every scenario is left to the
  instance (both outcomes occur over seed blocks).

## Model characteristics

- Each scenario is a single-commodity flow block coupled to the others only
  through the shared build/harden columns: a two-stage structure like
  `stochastic_program/standard`, but with network-design linking.
- Presolve keeps about 90% of rows and columns on feasible/unknown instances
  (1k–100k). Large instances (≥ 50k) can take tens of seconds in HiGHS.
