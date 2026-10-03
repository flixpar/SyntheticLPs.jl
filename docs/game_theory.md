# Game Theory

Equilibrium computation for two-player zero-sum games. By LP duality a
maximin strategy is the solution of a single LP in which one player's strategy
polytope is coupled, through a payoff block, to the *dualized* best-response
problem of the other player. The category contains three variants with very
different strategy polytopes, so their constraint matrices look nothing alike:

| Variant | Model class | Seat player's polytope | Opponent's (dualized) polytope | Coupling |
|---|---|---|---|---|
| `poker_sequence_form` (default) | continuous LP | treeplex (sequence-form flow rows, tree-structured) | treeplex: one row per opponent sequence | sparse chance-weighted payoff over terminal sequence pairs |
| `colonel_blotto` | continuous LP | unit flow on a layered allocation DAG | shortest-path potentials on the opponent's DAG | marginal rows + dense per-battlefield payoff block |
| `patrol_security` | continuous LP | patrol flow on a time-expanded street network | attacker-type epigraph rows (one per attack option) | bounded coverage variables |

All three are pure continuous LPs: `relax_integer` is a no-op and every
certificate refutes the LP as built. All use a constructor-local RNG;
`build_model` does no sampling.

## Common feasibility control

The value of a finite zero-sum game always exists, so the bare equilibrium LP
is always feasible and bounded. Feasibility is controlled through a natural
side requirement — a *guaranteed value* (poker win rate, Blotto vote share) or
a *risk budget* (patrol expected loss) — placed relative to two rigorous bounds
computed in the constructor:

- `lower_bound`: what an explicit strategy of the LP's player guarantees,
  i.e. the opponent's exact best-response value against it;
- `upper_bound`: the LP player's exact best-response value against an
  explicit opponent strategy.

The explicit strategies come from an approximate-equilibrium algorithm (CFR+
for poker, fictitious play for Blotto, attacker Hedge vs greedy saturating
patrols for the security game), so the bracket is tight but not exact.
For a maximized value `v` (minimized loss: mirror image):

- `feasible`: requirement `lower_bound - δ`; the typed witness is the explicit
  strategy plus the opponent best-response duals, a feasible point of every row;
- `infeasible`: requirement `upper_bound + δ`; the typed certificate is the
  explicit opponent strategy plus the LP player's best-response multipliers —
  a Farkas combination of LP rows. It needs the *whole* model: no single row
  (and no bound propagation) exposes it, so presolve cannot detect it;
- `unknown`: requirement uniform on `[lower_bound - w, upper_bound + w]`, `w`
  half the bracket width (at least `2%` of the payoff unit), so it lands on
  either side of the unknown exact value.

`δ` is `2%-10%` of a payoff unit (one ante per hand, the total battlefield
weight, the undefended expected loss), far above solver tolerances.

## `poker_sequence_form`

Sequence-form LP (Koller, Megiddo & von Stengel 1996) of a generalized
Kuhn/Leduc poker game.

**Game.** `n_ranks` ranks (3-20) in `n_suits` suits; both players ante one chip
and receive a private card. One or two fixed-limit betting rounds follow; before
round two a public board card is revealed and pairing it beats any unpaired
card. Each round has 1-4 bet sizes (multiples of a unit that doubles in round
two) and a raise cap of 1-4 (at most 81 all-raise lines per round). Folds pay the folder's contribution; showdowns pay
the stronger hand, ties split. The full game tree is enumerated with
suit-isomorphic deals merged, so information sets are `(private rank, [board
rank], public betting history)`. Kuhn poker (3 ranks, 1 suit, 1 round, 1 bet,
cap 1) and Leduc hold'em (3 ranks, 2 suits, bets 2/4, cap 2) are members of the
family; the tests check their known values (`-1/18` and about `-0.0856` per hand
for the first player).

**LP.** For the `seat` player (randomly 1 or 2) with realization plan `x` and the
opponent's information-set values `q`:

```text
maximize    q[1]
subject to  x[1] = 1
            sum_{a in A(I)} x[I a] - x[parent(I)] = 0            for each seat infoset I
            q[owner(τ)] - sum_{J entered by τ} q[J] - (A' x)[τ] <= 0   for each opponent sequence τ
            x[s] - ε x[parent(s)] >= 0                              (tremble rows, half the instances)
            q[1] >= required_value
            x >= 0, q free
```

`A` is the seat player's payoff over terminal sequence pairs, weighted by deal
multiplicity (an all-distinct deal weighs one). The LP value is
`value_scale * expected chips per hand`. The optional tremble rows give the
ε-perturbed sequence form used to compute equilibrium refinements (Miltersen &
Sørensen 2010, Farina & Gatti 2017), ε in `[0.002, 0.02]`.

**Witness / certificate.** Feasible: the CFR+ average realization plan
(satisfying the flow and tremble rows) and the opponent's exact best-response
values `q` (every best-response row holds, tight along the best response).
Infeasible: the opponent's CFR+ average plan `y`, infoset multipliers `p` and
tremble multipliers `μ >= 0` with `E'p - T'μ >= A y`; then
`q[1] = y'F'q <= x'A y <= p'Ex - μ'Tx <= p[1] < required_value`.

**Sizing.** With `d_p^r`/`a_p^r` the decision nodes/actions of player `p` in a
round-`r` betting tree and `C^1` the round-one continuations:

```text
|I_p| = N d_p^1 + N^2 C^1 d_p^2       |S_p| = 1 + N a_p^1 + N^2 C^1 a_p^2
variables = |S_seat| + |I_opp| + 1
rows      = |I_seat| + 1 + |S_opp| (+ |S_seat| - 1 tremble rows)
```

The generator picks uniformly among all games within 3% of the target (falling
back to the closest; below ~200 variables the grid is coarser, within ~5%), subject to a payoff block of at most about 10 nonzeros
per variable. Minimum 20 variables (Kuhn); maximum 1,000,000 (`ArgumentError`
above). CFR+ runs `clamp(2e8 / nnz(A), 16, 300)` iterations, so 100k-variable
games build in a few seconds.

## `colonel_blotto`

Compact equilibrium LP of a Colonel Blotto game (Ahmadinejad et al. 2016):
`K` battlefields (log-uniform `3..~40`), integer budgets `S` (LP player) and
`S_opp` (ratio `0.6-1.5`), lognormal integer battlefield weights (electoral
votes, market sizes), occasional incumbency advantages `d_k`, and either a
`:majority` contest (`w_k sign(a - b - d_k)`) or a Tullock `:lottery`
(`w_k (2a^ρ / (a^ρ + (b + d_k)^ρ) - 1)`).

A pure allocation is a path through a layered DAG (node `(k, s)` = `s` units
spent before battlefield `k`; the last layer spends the remainder); mixed
strategies are unit flows.

```text
maximize    π[1]
subject to  unit flow x on the seat DAG
            p[k, a] - sum_{layer-k edges spending a} x[e] = 0
            g[k, b] - sum_a u_k(a, b) p[k, a] = 0
            π[tail(e)] - π[head(e)] - g[k, b] <= 0       each opponent DAG edge (sink potential 0)
            π[1] >= required_value,  x, p >= 0,  g, π free
```

Witness: fictitious-play average flow, its marginals/payoffs, and the
opponent's shortest-path potentials. Certificate: the opponent's average flow
and the seat player's longest-path potentials `λ` with
`λ[tail] >= h[k, a] + λ[head]` on every seat edge.

**Sizing** (`E(K, S) = 2(S + 1) + (K - 2)(S + 1)(S + 2)/2` DAG edges):

```text
variables = E(K, S) + K(S + 1) + K(S_opp + 1) + 1 + (K - 1)(S_opp + 1)
rows      = 1 + (K - 1)(S + 1) + K(S + 1) + K(S_opp + 1) + E(K, S_opp)
```

Budgets are chosen to land within 2% of the target.

## `patrol_security`

Bayesian zero-sum security game with randomized patrols, in the compact
marginal-coverage form of deployed planners (TRUSTS fare inspection, PROTECT
port patrols). `n` stations on a jittered street grid (random spanning tree
plus a share of the remaining links), a shift of `H` periods with a two-peak
(rush-hour) value profile, station values from downtown hot spots x lognormal
noise x hub bonus, `2-16` patrol units, and `K = 2..6` attacker types: an
opportunistic type that threatens every station in every period and focused
groups with a home district and an active window. A type's gain at an
uncovered target is `value * profile * interest`; being caught costs it `κ_k`
times that.

```text
minimize    sum_k prior_k v_k
subject to  sum_{start arcs} f <= n_units
            inflow(i, t) - outflow(i, t) = 0               t < H
            c[i, t] - inflow(i, t) <= 0
            v_k + loss_o c[target(o)] >= gain_o             each attack option o of type k
            sum_k prior_k v_k <= loss_requirement
            f >= 0, 0 <= c <= 1, v free
```

Witness: the average greedy patrol plan, `c = min(1, inflow)`, and each type's
best-response value. Certificate: an attacker mix `α`, protection weights
`w_j = sum prior_k α_o loss_o`, a threshold `θ` and longest-path potentials for
`min(w, θ)`, giving
`loss >= sum prior α gain - sum (w - θ)^+ - n_units λ`. For infeasible requests
the requirement is kept above the trivial everything-covered bound, the only
bound single-row propagation can derive.

**Sizing:** `variables = n + (H - 1)(n + 2|E|) + nH + K`; the link count is
tuned to land within a few percent of the target.

## Notes for LP-only corpora

All variants are LPs. Instances at 10k variables solve in seconds to tens of
seconds with HiGHS dual simplex; sequence-form and Blotto LPs are notably
harder per variable than flow-like families (dense coupling blocks, heavy
degeneracy), which is precisely their value as test instances.
