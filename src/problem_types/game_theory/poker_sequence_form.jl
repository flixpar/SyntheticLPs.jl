using JuMP
using Random
using SparseArrays
using LinearAlgebra

# ---------------------------------------------------------------------------
# Sequence-form building blocks (shared by any tree-form game)
# ---------------------------------------------------------------------------

"""
    SequenceFormTreeplex

One player's sequence-form strategy space (a *treeplex*, Koller–Megiddo–von
Stengel 1996). Sequence `1` is the empty sequence; information set `I` owns the
contiguous sequences `infoset_first[I] : infoset_first[I] + infoset_num_actions[I] - 1`
(one per action) and is entered through its parent sequence
`infoset_parent[I]` — the player's own last action before reaching `I`
(perfect recall). Information sets are numbered so every parent sequence has a
smaller index than the sequences it leads to, which lets every pass below run
in one linear top-down or bottom-up sweep.

A realization plan `x` satisfies the flow rows `x[1] = 1` and
`sum(x[seqs(I)]) = x[infoset_parent[I]]` for every `I`, with `x >= 0`.
"""
struct SequenceFormTreeplex
    num_sequences::Int
    infoset_parent::Vector{Int}
    infoset_first::Vector{Int}
    infoset_num_actions::Vector{Int}
end

_sf_num_infosets(T::SequenceFormTreeplex) = length(T.infoset_parent)
_sf_seqs(T::SequenceFormTreeplex, I::Int) =
    T.infoset_first[I]:(T.infoset_first[I] + T.infoset_num_actions[I] - 1)

"""
    _sf_realization!(x, T, behavior) -> x

Convert a behaviour strategy (per-sequence action probabilities) into its
realization plan: `x[s] = x[parent] * behavior[s]`, top-down.
"""
function _sf_realization!(x::Vector{Float64}, T::SequenceFormTreeplex, behavior::Vector{Float64})
    x[1] = 1.0
    @inbounds for I in 1:_sf_num_infosets(T)
        px = x[T.infoset_parent[I]]
        for s in _sf_seqs(T, I)
            x[s] = px * behavior[s]
        end
    end
    return x
end

"""
    _sf_regret_update!(regret, behavior, T, g, tremble)

One CFR+ step for a MAXIMIZING player on treeplex `T` with sequence gradient
`g` (the opponent- and chance-weighted payoff of each sequence, e.g. `A * y`):
bottom-up counterfactual values, regret-matching+ accumulation, and the next
behaviour strategy, mixed with a per-action floor `tremble` (the perturbed
game) — `behavior[s] = tremble + (1 - k * tremble) * rm[s]`.
"""
function _sf_regret_update!(
    regret::Vector{Float64},
    behavior::Vector{Float64},
    T::SequenceFormTreeplex,
    g::Vector{Float64},
    tremble::Float64,
)
    val = copy(g)
    @inbounds for I in _sf_num_infosets(T):-1:1
        V = 0.0
        for s in _sf_seqs(T, I)
            V += behavior[s] * val[s]
        end
        for s in _sf_seqs(T, I)
            regret[s] = max(regret[s] + val[s] - V, 0.0)
        end
        val[T.infoset_parent[I]] += V
    end
    @inbounds for I in 1:_sf_num_infosets(T)
        k = T.infoset_num_actions[I]
        tot = 0.0
        for s in _sf_seqs(T, I)
            tot += regret[s]
        end
        for s in _sf_seqs(T, I)
            rm = tot > 0 ? regret[s] / tot : 1.0 / k
            behavior[s] = tremble + (1 - k * tremble) * rm
        end
    end
    return behavior
end

"""
    _sf_best_response_max(T, g, tremble) -> (value, p, μ)

Exact best-response value of a MAXIMIZING player on treeplex `T` against the
sequence gradient `g`, restricted to behaviour strategies that play every
action with probability at least `tremble`, together with its LP dual
certificate: infoset multipliers `p` (`p[1]` for the root row `x[1] = 1`,
`p[I + 1]` for infoset `I`) and tremble-row multipliers `μ >= 0` (zero for the
empty sequence) such that, for every sequence `σ`,

    (E' p)[σ] - (T' μ)[σ] == g[σ]

where `E` is the flow-row matrix and `T` the tremble rows
`x[s] - tremble * x[parent] >= 0`. Hence `g' x <= p[1] = value` for every
realization plan `x` satisfying the tremble rows.
"""
function _sf_best_response_max(T::SequenceFormTreeplex, g::Vector{Float64}, tremble::Float64)
    W = copy(g)
    nI = _sf_num_infosets(T)
    p = zeros(nI + 1)
    μ = zeros(T.num_sequences)
    @inbounds for I in nI:-1:1
        M = -Inf
        for s in _sf_seqs(T, I)
            M = max(M, W[s])
        end
        S = 0.0
        for s in _sf_seqs(T, I)
            μ[s] = M - W[s]
            S += μ[s]
        end
        p[I + 1] = M
        W[T.infoset_parent[I]] += M - tremble * S
    end
    p[1] = W[1]
    if tremble == 0
        fill!(μ, 0.0)
    end
    return p[1], p, μ
end

"""
    _sf_best_response_min(T, c) -> (value, q)

Exact best-response value of a MINIMIZING player on treeplex `T` against the
sequence cost `c` (e.g. `A' * x`), with the dual infoset values `q` (`q[1]` for
the root, `q[J + 1]` for infoset `J`) satisfying the KMvS best-response rows
`q[owner(τ)] - sum(q[children(τ)]) <= c[τ]` for every sequence `τ` — tight on
the best-response path — and `q[1] = value`.
"""
function _sf_best_response_min(T::SequenceFormTreeplex, c::Vector{Float64})
    W = copy(c)
    nI = _sf_num_infosets(T)
    q = zeros(nI + 1)
    @inbounds for I in nI:-1:1
        m = Inf
        for s in _sf_seqs(T, I)
            m = min(m, W[s])
        end
        q[I + 1] = m
        W[T.infoset_parent[I]] += m
    end
    q[1] = W[1]
    return q[1], q
end

"""
    _sf_cfr_plus(X, Y, A, iterations, tremble) -> (xbar, ybar)

Alternating CFR+ (regret-matching+, linearly weighted averages) on the
two-player zero-sum sequence-form game `max_x min_y x' A y`, with the
maximizer restricted to the `tremble`-perturbed treeplex. Returns the average
realization plans, both valid plans by convexity. Each iteration costs two
sparse matrix-vector products plus two linear treeplex sweeps.
"""
function _sf_cfr_plus(
    X::SequenceFormTreeplex,
    Y::SequenceFormTreeplex,
    A::SparseMatrixCSC{Float64, Int},
    iterations::Int,
    tremble::Float64,
)
    nx, ny = X.num_sequences, Y.num_sequences
    Rx, Ry = zeros(nx), zeros(ny)
    bx, by = zeros(nx), zeros(ny)
    for I in 1:_sf_num_infosets(X), s in _sf_seqs(X, I)
        bx[s] = 1.0 / X.infoset_num_actions[I]
    end
    for I in 1:_sf_num_infosets(Y), s in _sf_seqs(Y, I)
        by[s] = 1.0 / Y.infoset_num_actions[I]
    end
    x, y = zeros(nx), zeros(ny)
    xbar, ybar = zeros(nx), zeros(ny)
    gx, gy = zeros(nx), zeros(ny)
    _sf_realization!(y, Y, by)
    for t in 1:iterations
        mul!(gx, A, y)
        _sf_regret_update!(Rx, bx, X, gx, tremble)
        _sf_realization!(x, X, bx)
        xbar .+= t .* x
        mul!(gy, transpose(A), x)
        gy .*= -1
        _sf_regret_update!(Ry, by, Y, gy, 0.0)
        _sf_realization!(y, Y, by)
        ybar .+= t .* y
    end
    xbar ./= xbar[1]
    ybar ./= ybar[1]
    return xbar, ybar
end

# ---------------------------------------------------------------------------
# Poker game definition
# ---------------------------------------------------------------------------

"""
    PokerBettingRound(bet_sizes, raise_cap)

Fixed-limit betting rules of one round: the allowed bet/raise increments (in
chips; the ante is one chip) and the maximum number of bets plus raises in the
round. A player facing no bet may check or bet any size; a player facing a bet
may fold, call, or (below the cap) raise by any size.
"""
struct PokerBettingRound
    bet_sizes::Vector{Int}
    raise_cap::Int
end

"""
Planted feasible point of the sequence-form LP: the seat player's CFR+ average
realization plan `realization_plan` (respecting the tremble floor) together
with the opponent's exact best-response infoset values `infoset_values`, which
satisfy every best-response row against that plan. `guaranteed_value =
infoset_values[1]` is what the plan guarantees, and it is at least the
model's `required_value`.
"""
struct PokerSequenceFormWitness
    realization_plan::Vector{Float64}
    infoset_values::Vector{Float64}
    guaranteed_value::Float64
end

"""
Farkas certificate that no strategy guarantees `required_value`: an opponent
realization plan `opponent_plan` (`F y = f`, `y >= 0`) and multipliers
`infoset_multipliers` (`p`) on the seat player's flow rows and
`tremble_multipliers` (`μ >= 0`) on its tremble rows with
`E' p - T' μ >= A * opponent_plan` componentwise. Weighting the best-response
rows by `opponent_plan`, the flow rows by `p`, and the tremble rows by `μ`
gives `q[1] <= p[1] = value_bound` for every feasible point — LP rows only —
while the requirement is `q[1] >= required_value > value_bound`.
"""
struct PokerSequenceFormCertificate
    opponent_plan::Vector{Float64}
    infoset_multipliers::Vector{Float64}
    tremble_multipliers::Vector{Float64}
    value_bound::Float64
end

"""
    PokerSequenceFormProblem <: ProblemGenerator

Sequence-form LP (Koller, Megiddo & von Stengel 1996) for an equilibrium
strategy of a two-player zero-sum poker game with imperfect information — the
LP behind classical poker solving (Kuhn, Leduc, and abstracted limit
hold'em).

# Game family

A deck of `n_ranks` ranks in `n_suits` suits. Both players ante one chip and
receive one private card. One or two betting rounds follow; before round two a
public board card is revealed (Leduc-style), and a private card pairing the
board beats every unpaired card. Each round has fixed-limit rules
(`PokerBettingRound`): a set of bet/raise increments and a raise cap. The game
ends with a fold or a showdown (higher rank wins; equal strength splits). The
generator enumerates the full game tree, merging suit-isomorphic deals (cards
matter only through their ranks), so information sets are `(private rank,
[board rank], public betting history)`.

# LP

The `seat` player (the maximizer; seat 1 acts first in every round) has
realization plan `x` over its sequences, the opponent has information-set
values `q` (`q[1]` for the root). With `A` the seat player's payoff over
terminal sequence pairs (chance-weighted by deal multiplicity), the LP is

    maximize    q[1]
    subject to  x[1] = 1,  sum(x[seqs(I)]) - x[parent(I)] = 0     (each seat infoset I)
                q[owner(τ)] - sum(q[children(τ)]) - (A' x)[τ] <= 0 (each opponent sequence τ)
                x[s] - ε x[parent(s)] >= 0                         (tremble rows, if ε > 0)
                q[1] >= required_value,  x >= 0,  q free

i.e. tree-structured realization-plan flow rows coupled to the opponent's
dualized best-response rows through the sparse payoff block. The optional
tremble rows give the ε-perturbed sequence form used for equilibrium
refinements (Miltersen & Sørensen 2010; Farina & Gatti 2017): every action of
the seat player keeps probability at least `ε`.

# Feasibility control

The constructor runs CFR+ on the generated game and computes both players'
exact best responses to the averages: a lower bound `lower_bound` (what the
seat player's average plan guarantees) and an upper bound `upper_bound` (the
seat player's best response to the opponent's average plan) bracket the game
value. The requirement `q[1] >= required_value` (a guaranteed win rate) is then
placed by `_game_value_requirement`:

  - `feasible`: below `lower_bound`; the witness is the average plan plus the
    opponent's best-response values;
  - `infeasible`: above `upper_bound`; the certificate is the opponent's
    average plan plus the seat player's best-response multipliers — a
    whole-model refutation that presolve cannot see from any single row;
  - `unknown`: uniform on a window bracketing both bounds — a genuine instance
    whose status depends on the unknown exact value.

# Fields

  - `n_ranks`, `n_suits`: deck shape
  - `rounds::Vector{PokerBettingRound}`: betting rules, one per round
  - `seat::Int`: the player whose maximin strategy the LP computes (1 or 2)
  - `tremble::Float64`: per-action probability floor `ε` (0 = unperturbed)
  - `player_tree`, `opponent_tree::SequenceFormTreeplex`: the two treeplexes
  - `payoff::SparseMatrixCSC{Float64,Int}`: seat payoff, `|S_seat| x |S_opp|`
  - `value_scale::Float64`: total deal weight; the LP value is
    `value_scale x` the expected chips won per hand
  - `lower_bound`, `upper_bound`, `required_value::Float64`: value bounds and
    the requirement (LP value units)
  - `cfr_iterations::Int`
  - `feasible_witness`, `infeasibility_certificate`, `feasibility_status`
"""
struct PokerSequenceFormProblem <: ProblemGenerator
    n_ranks::Int
    n_suits::Int
    rounds::Vector{PokerBettingRound}
    seat::Int
    tremble::Float64
    player_tree::SequenceFormTreeplex
    opponent_tree::SequenceFormTreeplex
    payoff::SparseMatrixCSC{Float64, Int}
    value_scale::Float64
    lower_bound::Float64
    upper_bound::Float64
    required_value::Float64
    cfr_iterations::Int
    feasible_witness::Union{Nothing, PokerSequenceFormWitness}
    infeasibility_certificate::Union{Nothing, PokerSequenceFormCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    _poker_round_counts(n_sizes, raise_cap) -> NamedTuple

Shape statistics of one fixed-limit betting round with `n_sizes` bet sizes and
the given raise cap, independent of chip amounts: decision nodes and total
actions per player (`decisions`, `actions`, indexed by seat), round-ending
non-fold histories (`continuations` — they lead to the next round or a
showdown), and fold histories (`folds`). Mirrors `_poker_build_round!`.
"""
function _poker_round_counts(n_sizes::Int, raise_cap::Int)
    decisions = [0, 0]
    actions = [0, 0]
    continuations = Ref(0)
    folds = Ref(0)
    function rec(actor, raises, facing_bet, prev_check)
        decisions[actor] += 1
        can_raise = raises < raise_cap
        if !facing_bet
            actions[actor] += 1 + (can_raise ? n_sizes : 0)
            prev_check ? (continuations[] += 1) : rec(3 - actor, raises, false, true)
        else
            actions[actor] += 2 + (can_raise ? n_sizes : 0)
            folds[] += 1
            continuations[] += 1
        end
        if can_raise
            for _ in 1:n_sizes
                rec(3 - actor, raises + 1, true, false)
            end
        end
    end
    rec(1, 0, false, false)
    return (
        decisions=(decisions[1], decisions[2]),
        actions=(actions[1], actions[2]),
        continuations=continuations[],
        folds=folds[],
    )
end

"""
    _poker_size_formula(n_ranks, shapes, seat) -> (variables, rows_without_trembles, nnz_estimate)

Exact LP dimensions for a game with `n_ranks` ranks and per-round shapes
`shapes = [(n_sizes, raise_cap), ...]` (one or two rounds), built for `seat`:

    |I_p| = N d_p^1 + N^2 C^1 d_p^2         (second term only with two rounds)
    |S_p| = 1 + N a_p^1 + N^2 C^1 a_p^2
    variables = |S_seat| + |I_opp| + 1
    rows      = |I_seat| + 1 + |S_opp|        (+ |S_seat| - 1 tremble rows)

where `d_p^r`, `a_p^r` are player `p`'s decision nodes and actions in a round-`r`
betting tree and `C^1` the round-one continuations. The nonzero estimate is the
number of (deal class, terminal) pairs, an upper bound on the payoff block.
"""
function _poker_size_formula(n_ranks::Int, shapes::Vector{Tuple{Int, Int}}, seat::Int)
    N = n_ranks
    c1 = _poker_round_counts(shapes[1]...)
    S = [1 + N * c1.actions[p] for p in 1:2]
    I = [N * c1.decisions[p] for p in 1:2]
    nnz_est = if length(shapes) == 1
        N^2 * (c1.continuations + c1.folds)
    else
        c2 = _poker_round_counts(shapes[2]...)
        for p in 1:2
            S[p] += N^2 * c1.continuations * c2.actions[p]
            I[p] += N^2 * c1.continuations * c2.decisions[p]
        end
        N^2 * c1.folds + N^3 * c1.continuations * (c2.continuations + c2.folds)
    end
    opp = 3 - seat
    return S[seat] + I[opp] + 1, I[seat] + 1 + S[opp], nnz_est
end

"""
    _poker_choose_shape(rng, target) -> (n_ranks, shapes)

Pick a game whose exact variable count is within 3% of `target` (or the
closest one when none is), uniformly among all qualifying games with 3-20 ranks,
one or two rounds, 1-4 bet sizes and raise caps 1-4 per round (at most
81 all-raise lines, e.g. no four sizes with a cap of four), and a payoff
block of at most about 10 nonzeros per variable. Targets below the 20-variable
Kuhn-poker minimum round up to it.
"""
function _poker_choose_shape(rng::AbstractRNG, target::Int)
    candidates = Tuple{Int, Vector{Tuple{Int, Int}}}[]
    best = nothing
    best_err = Inf
    # Up to four bet sizes and four raises, but at most 81 all-raise lines.
    shapes_1 = [(nb, cap) for nb in 1:4 for cap in 1:4 if nb^cap <= 81]
    for R in 1:2, s1 in shapes_1, s2 in (R == 1 ? [(0, 0)] : shapes_1), N in 3:20
        shapes = R == 1 ? [s1] : [s1, s2]
        v, _, nnz_est = _poker_size_formula(N, shapes, 1)
        nnz_est <= 10 * target + 2000 || continue
        err = abs(v - target)
        if err <= 0.03 * target
            push!(candidates, (N, shapes))
        end
        if err < best_err
            best_err = err
            best = (N, shapes)
        end
    end
    return isempty(candidates) ? best : candidates[rand(rng, 1:length(candidates))]
end

"""
Public betting tree in preorder. Decision nodes carry the actor and action
count; terminal nodes carry the folder (0 = showdown). `last` stores, per node,
each player's most recent decision `(node, action)` strictly before it (0 when
the player has not acted yet).
"""
struct _PokerPublicTree
    is_terminal::Vector{Bool}
    round::Vector{Int}
    actor::Vector{Int}
    num_actions::Vector{Int}
    contrib1::Vector{Int}
    contrib2::Vector{Int}
    folder::Vector{Int}
    last_node::Matrix{Int}   # 2 x nodes
    last_action::Matrix{Int} # 2 x nodes
end

function _poker_push_node!(t, terminal, round, actor, c, folder, last)
    push!(t.is_terminal, terminal)
    push!(t.round, round)
    push!(t.actor, actor)
    push!(t.num_actions, 0)
    push!(t.contrib1, c[1])
    push!(t.contrib2, c[2])
    push!(t.folder, folder)
    push!(t.last_node, (last[1], last[3])...)
    push!(t.last_action, (last[2], last[4])...)
    return length(t.is_terminal)
end

function _poker_round_end!(t, rounds, r, c, last)
    if r == length(rounds)
        _poker_push_node!(t, true, r, 0, c, 0, last)
    else
        # The public board card is dealt (a chance event not represented as a
        # public node: it only refines the information-set keys).
        _poker_build_round!(t, rounds, r + 1, 1, 0, false, c, last)
    end
    return nothing
end

function _poker_build_round!(t, rounds, r, actor, raises, prev_check, c, last)
    id = _poker_push_node!(t, false, r, actor, c, -1, last)
    to_call = c[3 - actor] - c[actor]
    rules = rounds[r]
    can_raise = raises < rules.raise_cap
    with_action(a) = actor == 1 ? (id, a, last[3], last[4]) : (last[1], last[2], id, a)
    a = 0
    if to_call == 0
        a += 1  # check
        if prev_check
            _poker_round_end!(t, rounds, r, c, with_action(a))
        else
            _poker_build_round!(t, rounds, r, 3 - actor, raises, true, c, with_action(a))
        end
    else
        a += 1  # fold
        _poker_push_node!(t, true, r, 0, c, actor, with_action(a))
        a += 1  # call
        cc = copy(c)
        cc[actor] += to_call
        _poker_round_end!(t, rounds, r, cc, with_action(a))
    end
    if can_raise
        for b in rules.bet_sizes  # bet (facing nothing) or raise (facing a bet)
            a += 1
            cc = copy(c)
            cc[actor] += to_call + b
            _poker_build_round!(t, rounds, r, 3 - actor, raises + 1, false, cc, with_action(a))
        end
    end
    t.num_actions[id] = a
    return nothing
end

function _poker_public_tree(rounds::Vector{PokerBettingRound})
    lastn, lasta = Int[], Int[]
    tv = (
        is_terminal=Bool[],
        round=Int[],
        actor=Int[],
        num_actions=Int[],
        contrib1=Int[],
        contrib2=Int[],
        folder=Int[],
        last_node=lastn,
        last_action=lasta,
    )
    _poker_build_round!(tv, rounds, 1, 1, 0, false, [1, 1], (0, 0, 0, 0))
    n = length(tv.is_terminal)
    return _PokerPublicTree(
        tv.is_terminal,
        tv.round,
        tv.actor,
        tv.num_actions,
        tv.contrib1,
        tv.contrib2,
        tv.folder,
        reshape(lastn, 2, n),
        reshape(lasta, 2, n),
    )
end

"""
    _poker_treeplex(pub, p, N) -> (treeplex, seq_offset)

Player `p`'s treeplex. Information sets are `(key, h)` for each of `p`'s public
decision nodes `h` (preorder) and private key — the rank (round one) or the
`(rank, board)` pair encoded as `(rank - 1) * N + board` (round two). The
sequence of action `a` at `(key, h)` is `seq_offset[h] + (key - 1) * k_h + a`.
"""
function _poker_treeplex(pub::_PokerPublicTree, p::Int, N::Int)
    n = length(pub.is_terminal)
    seq_offset = zeros(Int, n)
    inf_offset = zeros(Int, n)
    nseq, ninf = 1, 0
    for h in 1:n
        (!pub.is_terminal[h] && pub.actor[h] == p) || continue
        nk = pub.round[h] == 1 ? N : N * N
        seq_offset[h] = nseq
        inf_offset[h] = ninf
        nseq += nk * pub.num_actions[h]
        ninf += nk
    end
    parent = Vector{Int}(undef, ninf)
    first = Vector{Int}(undef, ninf)
    nact = Vector{Int}(undef, ninf)
    for h in 1:n
        (!pub.is_terminal[h] && pub.actor[h] == p) || continue
        nk = pub.round[h] == 1 ? N : N * N
        k = pub.num_actions[h]
        ln, la = pub.last_node[p, h], pub.last_action[p, h]
        for key in 1:nk
            I = inf_offset[h] + key
            first[I] = seq_offset[h] + (key - 1) * k + 1
            nact[I] = k
            parent[I] =
                ln == 0 ? 1 : _poker_sequence(pub, seq_offset, ln, la, _poker_key(pub, ln, h, key, N))
        end
    end
    return SequenceFormTreeplex(nseq, parent, first, nact), seq_offset
end

# Key of an earlier node `ln` on the path to `h` given `h`'s key: identical
# within a round; a round-two key `(rank - 1) * N + board` maps to its rank.
_poker_key(pub, ln, h, key, N) = pub.round[ln] == pub.round[h] ? key : (key - 1) ÷ N + 1

_poker_sequence(pub, seq_offset, h, a, key) = seq_offset[h] + (key - 1) * pub.num_actions[h] + a

"""
    _poker_payoff_matrix(pub, T1, off1, T2, off2, N, n_suits, n_rounds)
        -> (A1, value_scale)

Player 1's chance-weighted payoff over terminal sequence pairs. Deals are
merged by rank: the weight of `(r1, r2[, b])` is its number of ordered card
draws divided by `n_suits^k` (`k` cards dealt), so all-distinct deals weigh one
and paired ones `(n_suits - 1) / n_suits` etc. Folds pay the folder's
contribution to the opponent; showdowns pay the (equal) contribution to the
stronger hand and nothing on a tie. `value_scale` is the total deal weight.
"""
function _poker_payoff_matrix(pub, T1, off1, T2, off2, N, s, n_rounds)
    k = n_rounds == 1 ? 2 : 3
    W3 = zeros(N, N, N)
    W2 = zeros(N, N)
    for r1 in 1:N, r2 in 1:N
        w12 = s * (s - (r1 == r2))
        if n_rounds == 1
            W2[r1, r2] = w12 / s^2
        else
            for b in 1:N
                w = w12 * max(s - (b == r1) - (b == r2), 0)
                W3[r1, r2, b] = w / s^3
            end
            W2[r1, r2] = sum(@view W3[r1, r2, :])
        end
    end
    strength(r, b) = (b > 0 && r == b) ? N + r : r

    Is, Js, Vs = Int[], Int[], Float64[]
    for z in eachindex(pub.is_terminal)
        pub.is_terminal[z] || continue
        n1, a1 = pub.last_node[1, z], pub.last_action[1, z]
        n2, a2 = pub.last_node[2, z], pub.last_action[2, z]
        c1, c2 = pub.contrib1[z], pub.contrib2[z]
        folder = pub.folder[z]
        if pub.round[z] == 1
            for r1 in 1:N, r2 in 1:N
                w = W2[r1, r2]
                w > 0 || continue
                u = folder == 1 ? -c1 : folder == 2 ? c2 : c1 * sign(r1 - r2)
                u == 0 && continue
                push!(Is, _poker_sequence(pub, off1, n1, a1, r1))
                push!(Js, _poker_sequence(pub, off2, n2, a2, r2))
                push!(Vs, w * u)
            end
        else
            for r1 in 1:N, r2 in 1:N, b in 1:N
                w = W3[r1, r2, b]
                w > 0 || continue
                u = if folder == 1
                    -c1
                elseif folder == 2
                    c2
                else
                    c1 * sign(strength(r1, b) - strength(r2, b))
                end
                u == 0 && continue
                k1 = pub.round[n1] == 2 ? (r1 - 1) * N + b : r1
                k2 = pub.round[n2] == 2 ? (r2 - 1) * N + b : r2
                push!(Is, _poker_sequence(pub, off1, n1, a1, k1))
                push!(Js, _poker_sequence(pub, off2, n2, a2, k2))
                push!(Vs, w * u)
            end
        end
    end
    A1 = sparse(Is, Js, Vs, T1.num_sequences, T2.num_sequences)
    return A1, sum(W2)
end

"""
    PokerSequenceFormProblem(target_variables, feasibility_status, seed)

Build a poker sequence-form LP whose variable count is within about 3% of
`target_variables` (exact formula in `_poker_size_formula`); targets below the
20-variable Kuhn game round up to it and targets above
`GAME_THEORY_MAX_VARIABLES` raise an `ArgumentError`.
"""
function PokerSequenceFormProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    _game_theory_check_target("poker_sequence_form", target_variables)
    rng = MersenneTwister(seed)

    n_ranks, shapes = _poker_choose_shape(rng, target_variables)
    n_rounds = length(shapes)
    # Single-round games may use one suit (Kuhn); a board round needs pairs.
    n_suits = n_rounds == 1 ? rand(rng, 1:4) : rand(rng, 2:4)
    unit = rand(rng, 1:2)  # round-one bet unit in antes; round two doubles it
    rounds = [
        PokerBettingRound(collect(unit * r .* (1:shapes[r][1])), shapes[r][2]) for r in 1:n_rounds
    ]
    seat = rand(rng, 1:2)
    tremble = rand(rng) < 0.5 ? 0.0 : round(0.002 + 0.018 * rand(rng); digits=4)
    return _poker_assemble(rng, n_ranks, n_suits, rounds, seat, tremble, feasibility_status)
end

"""
    _poker_assemble(rng, n_ranks, n_suits, rounds, seat, tremble, feasibility_status)

Enumerate the game with the given rules, run CFR+, bound the value with exact
best responses, and place the requirement (the only further randomness).
Separated from the sampling constructor so fixed classical games (Kuhn,
Leduc) can be instantiated directly.
"""
function _poker_assemble(
    rng::AbstractRNG,
    n_ranks::Int,
    n_suits::Int,
    rounds::Vector{PokerBettingRound},
    seat::Int,
    tremble::Float64,
    feasibility_status::FeasibilityStatus,
)
    n_rounds = length(rounds)
    pub = _poker_public_tree(rounds)
    T1, off1 = _poker_treeplex(pub, 1, n_ranks)
    T2, off2 = _poker_treeplex(pub, 2, n_ranks)
    A1, value_scale = _poker_payoff_matrix(pub, T1, off1, T2, off2, n_ranks, n_suits, n_rounds)
    X, Y, A = seat == 1 ? (T1, T2, A1) : (T2, T1, -copy(transpose(A1)))

    # Approximate equilibrium (CFR+), budgeted by payoff nonzeros so large
    # games stay within seconds.
    iterations = clamp(round(Int, 2.0e8 / max(nnz(A), 1)), 16, 300)
    xbar, ybar = _sf_cfr_plus(X, Y, A, iterations, tremble)
    lower, q = _sf_best_response_min(Y, Vector(transpose(A) * xbar))
    upper, p, μ = _sf_best_response_max(X, A * ybar, tremble)

    required_value = _game_value_requirement(rng, feasibility_status, lower, upper, value_scale)
    witness = nothing
    certificate = nothing
    if feasibility_status == feasible
        witness = PokerSequenceFormWitness(xbar, q, lower)
    elseif feasibility_status == infeasible
        certificate = PokerSequenceFormCertificate(ybar, p, μ, upper)
    end

    return PokerSequenceFormProblem(
        n_ranks,
        n_suits,
        rounds,
        seat,
        tremble,
        X,
        Y,
        A,
        value_scale,
        lower,
        upper,
        required_value,
        iterations,
        witness,
        certificate,
        feasibility_status,
    )
end

"""
    _sf_child_infosets(T) -> Vector{Vector{Int}}

Information sets entered through each sequence of `T`.
"""
function _sf_child_infosets(T::SequenceFormTreeplex)
    children = [Int[] for _ in 1:T.num_sequences]
    for J in 1:_sf_num_infosets(T)
        push!(children[T.infoset_parent[J]], J)
    end
    return children
end

"""
    _sf_owner(T) -> Vector{Int}

For each sequence of `T`, the index into the value vector `q` of the row that
owns it: `1` for the empty sequence (root row), `J + 1` for a sequence of
information set `J`.
"""
function _sf_owner(T::SequenceFormTreeplex)
    owner = ones(Int, T.num_sequences)
    for J in 1:_sf_num_infosets(T), s in _sf_seqs(T, J)
        owner[s] = J + 1
    end
    return owner
end

"""
    build_model(prob::PokerSequenceFormProblem)

Deterministic sequence-form LP: `x[1:|S_seat|] >= 0` (seat realization plan),
free `q[1:|I_opp| + 1]` (opponent infoset values, `q[1]` the root), maximize
`q[1]` subject to the flow rows, one best-response row per opponent sequence,
optional tremble rows, and the requirement bound `q[1] >= required_value`.
"""
function build_model(prob::PokerSequenceFormProblem)
    X, Y, A = prob.player_tree, prob.opponent_tree, prob.payoff
    model = Model()
    @variable(model, x[1:(X.num_sequences)] >= 0)
    @variable(model, q[1:(_sf_num_infosets(Y) + 1)])
    set_lower_bound(q[1], prob.required_value)
    @objective(model, Max, q[1])

    # Realization-plan flow rows E x = e.
    @constraint(model, x[1] == 1)
    for I in 1:_sf_num_infosets(X)
        expr = AffExpr(0.0)
        for s in _sf_seqs(X, I)
            add_to_expression!(expr, 1.0, x[s])
        end
        add_to_expression!(expr, -1.0, x[X.infoset_parent[I]])
        @constraint(model, expr == 0)
    end

    # Dualized opponent best-response rows F' q - A' x <= 0.
    children = _sf_child_infosets(Y)
    owner = _sf_owner(Y)
    rows, vals = rowvals(A), nonzeros(A)
    for τ in 1:(Y.num_sequences)
        expr = AffExpr(0.0)
        add_to_expression!(expr, 1.0, q[owner[τ]])
        for J in children[τ]
            add_to_expression!(expr, -1.0, q[J + 1])
        end
        for k in nzrange(A, τ)
            add_to_expression!(expr, -vals[k], x[rows[k]])
        end
        @constraint(model, expr <= 0)
    end

    # ε-perturbation (trembling-hand) rows.
    if prob.tremble > 0
        for I in 1:_sf_num_infosets(X)
            par = X.infoset_parent[I]
            for s in _sf_seqs(X, I)
                @constraint(model, x[s] - prob.tremble * x[par] >= 0)
            end
        end
    end
    return model
end

register_variant(
    :game_theory,
    :poker_sequence_form,
    PokerSequenceFormProblem,
    "Sequence-form LP (Koller-Megiddo-von Stengel) for an equilibrium of a generalized Kuhn/Leduc poker game: tree-structured realization-plan flow rows coupled to the opponent's dualized best-response rows through a sparse chance-weighted payoff block, with an optional trembling-hand perturbation and a guaranteed-value requirement certified by CFR+ strategies and exact best responses";
    default=true,
)
