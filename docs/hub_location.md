# Hub Location

This category generates the classical hub-location and hub-and-spoke network
design family: origin-destination traffic is consolidated through
intermediate hubs, and inter-hub legs carry a discount factor `alpha` that
models economies of scale (denser vehicles, better-loaded sortation, wavelength
multiplexing). It contains five variants spanning the problem classes most
solved in practice:

| Variant | Allocation | Key structure | Domain grounding |
|---|---|---|---|
| `p_hub_median` | single | exact `p` hubs, reach windows, tight four-index path flows | airline (CAB) |
| `r_allocation` | r of p hubs | primary/backup allocations, four-index path flows | airline resilience |
| `multiple_allocation` | multiple | fixed costs + opening budget, per-destination flows | parcel / LTL (AP) |
| `capacitated` | single | collection-inflow capacities in loose/tight profiles | postal (AP) |
| `hub_network` | single | incomplete backbone: build modular hub-hub links | telecom backbone |

All constructors use a local RNG; calling a generator does not reseed or
consume Julia's global RNG, and `build_model` does no sampling.

## Formulations

`p_hub_median` and `r_allocation` use the tight four-index path-flow
linearisation of Skorin-Kapov, Skorin-Kapov & O'Kelly (1996): variables
`x_ikmj` carry the volume of pair `(i, j)` routed `i -> k -> m -> j`, tied to
allocation binaries by disaggregated equalities (`p_hub_median`) or
inequalities (`r_allocation`, where a pair uses at most one of its origin's r
hubs). Flows and costs are symmetrised — each *unordered* pair is one
commodity, the standard symmetric-problem convention. With
`relax_integer=true` this is the famously tight SKO LP relaxation of the
p-hub median problem.

The other three variants use per-destination multicommodity flows through a
hub layer (the efficient-flow-model family surveyed by Brimberg et al. 2021):
collection arcs `i -> k`, discounted transfer arcs, and a delivery arc into
the destination. Costs are Euclidean (metric) and the discount is uniform, so
routing through additional hubs can never pay less than the direct inter-hub
leg — optimal paths visit at most two hubs and the flow model is exact.
`capacitated` adds the single-allocation coupling (all of a node's volume
enters and leaves at its own hub) plus capacity rows corrected per Correia,
Nickel & Saldanha-da-Gama (2010). `hub_network` replaces the complete
inter-hub network with *candidate backbone links* that must be built (binary)
and then carry limited both-direction capacity — single allocation over an
incomplete hub network in the sense of Yaman (2009).

## Data grounding

Conventions were verified against the published benchmark files (OR-Library
`phub1..4`; see also the Mathprog-ORlib mirror):

- **CAB** (O'Kelly 1987): 25 US cities, fully symmetric passenger flows
  spanning 565..205,088 (median ~7,000 — a two-orders-of-magnitude right
  skew), *network* costs rather than coordinate distances, discount grid
  `alpha in [0.2, 1.0]`, `p in 2..5`, `chi = delta = 1`. The airline-flavored
  variants use symmetrised gravity flows with lognormal scatter, detour-
  perturbed symmetric costs (which only approximately obey the triangle
  inequality, as in the published matrix — safe here because four-index paths
  are fixed `i-k-m-j` routes), and those cost multipliers.
- **AP** (Ernst & Krishnamoorthy 1996): 200 postal districts, Euclidean
  distances on a ~57 km x 55 km plane, asymmetric flows (63% of ordered pairs)
  with coefficient of variation ~5 and nonzero self-flows, and the published
  parameters `chi = 3` (collection), `alpha = 0.75` (transfer),
  `delta = 2` (distribution) — "per unit distance, per unit flow volume
  divided by 1000". The postal/parcel variants sample around exactly these
  values. Self-flows are not modelled (the diagonal is zero), the common
  convention when routing commodities.
- **AP capacity files** come in loose (`CapL`) and tight (`CapT`) flavours and
  bound the total flow *into* a hub including the node's own volume;
  `capacitated` mirrors this with `:loose`/`:tight` profiles on collection
  inflow.
- **Telecom conventions** (matching this package's
  `telecom_network_design`): access multipliers `chi = delta in [1, 2.5]`, a
  deep backbone discount `alpha in [0.05, 0.4]`, link build costs with a
  distance-proportional component, and capacities snapped up to the
  SONET/SDH module ladder 155 / 622 / 2488 / 9953 / 39813 (OC-3/12/48/192/768).

Flows follow a doubly-constrained gravity model
`w_ij ~ O_i * D_j / d_ij^decay * LogNormal(0, noise)` with population-driven,
independently jittered origin/destination potentials, `decay in [0.4, 1.1]`
and `noise in [0.6, 1.2]` — reproducing the heavy right skew of both
datasets. Reach windows (feeder range / catchment restrictions) restrict a
node's admissible hubs.

## Feasibility control

Every certificate refutes the LP relaxation, not only the MIP:

- `p_hub_median` / `r_allocation` **infeasible**: `p + 1` island groups with
  pairwise disjoint admissible sets. Each group needs its own open hub
  (disaggregated linking rows force `sum_{k in A_i} y_k >= 1`), contradicting
  the exact-`p` row.
- `multiple_allocation` **infeasible**: disjoint groups plus an opening
  budget strictly below `groups * min_k f_k`, contradicting the budget row.
- `capacitated` **infeasible**: total capacity strictly below total flow;
  summing the capacity rows against the single-allocation rows gives
  `W <= sum_k Gamma_k < W`.
- `hub_network` **infeasible**: a regional gateway cut whose total crossing
  capacity (with every crossing link built) is below the inter-regional
  traffic that must cross it; reach windows keep each side's traffic on its
  own hubs.

Feasible requests plant a witness — a hub set with admissible assignments
(cover-radius based), a capacity-respecting best-fit assignment, or a sized
spanning backbone whose exact routed loads fit under the module capacities.
Unknown requests sample near the corresponding feasibility boundary (reach
window around the covering radius, budget around the cover cost, capacities
around total flow, crossing capacity around the crossing traffic), which
yields a genuine mix of outcomes rather than a hidden always-infeasible mode.

## Variable counts

With `A_i` the admissible hub list of node `i`, `h` the candidate count and
`L` the candidate backbone link count:

- `p_hub_median`, `r_allocation`:
  `sum_{i<j} |A_i| |A_j| + sum_i |A_i| + |union_i A_i|`
- `multiple_allocation`:
  `sum_j [ sum_{i != j} |A_i| + h(h-1) + |A_j| ] + h`
- `capacitated`: `n h (n + h - 1) + h (n + 1)`
- `hub_network`:
  `sum_j [ sum_{i != j} |A_i| + 2L + |A_j| ] + sum_i |A_i| + h + L`

An iterative re-sizing loop adjusts the node/candidate hints so the exact
count lands within a few percent of the requested target.

## References

- O'Kelly, M.E. (1987). A quadratic integer program for the location of
  interacting hub facilities. European Journal of Operational Research 32,
  393-404. (CAB dataset; OR-Library file phub4.)
- Campbell, J.F. (1994). Integer programming formulations of discrete hub
  location problems. European Journal of Operational Research 72.
- Skorin-Kapov, D., Skorin-Kapov, J., O'Kelly, M.E. (1996). Tight linear
  programming relaxations of uncapacitated p-hub median problems. European
  Journal of Operational Research 94.
- Ernst, A.T., Krishnamoorthy, M. (1996). Efficient algorithms for the
  uncapacitated single allocation p-hub median problem. Location Science 4(3),
  139-154. (AP dataset; OR-Library files phub1-3.)
- O'Kelly, M.E., Bryan, D.L., Skorin-Kapov, D., Skorin-Kapov, J. (1996). Hub
  network design with single and multiple allocation: a computational study.
  Location Science 4(3). (CAB usage conventions.)
- Ernst, A.T., Krishnamoorthy, M. (1999). Solution algorithms for the
  capacitated single allocation hub location problem. Annals of Operations
  Research 86, 141-159.
- Yoon, M.G., Current, J. (2008). Hub network design with single and multiple
  allocation - an application to less-than-truckload freight transportation.
  Networks and Spatial Economics 8.
- Alumur, S.A., Kara, B.Y. (2009). Network hub location problems: the state
  of the art. Networks and Spatial Economics 9.
- Yaman, H. (2009). The design of single allocation incomplete hub networks.
  Transportation Research Part B 43(10).
- Correia, I., Nickel, S., Saldanha-da-Gama, F. (2010). The capacitated
  single-allocation hub location problem revisited: a note on a classic
  formulation. European Journal of Operational Research 211.
- Peiro, J., Corberan, A., Marti, R. (2014). The uncapacitated r-allocation
  p-hub median problem. European Journal of Operational Research 232.
- Campbell, J.F., O'Kelly, M.E. (2012). Twenty-five years of hub location
  research. Transportation Science 46(2).
- Brimberg, J., et al. (2021). Efficient flow models for the uncapacitated
  multiple allocation p-hub median problem on non-triangular networks.
  Computers & Operations Research 133, 105313.
