# game_theory category
#
# Entry point for the `game_theory` problem category: equilibrium computation
# for two-player zero-sum games, an LP class with structure found nowhere else
# in the corpus — tree-structured (sequence-form) or layered strategy
# polytopes of one player coupled, through a payoff block, to the dualized
# best-response rows of the other.

register_category(
    :game_theory,
    "Equilibrium LPs of two-player zero-sum games: sequence-form extensive-form games (poker), compact Colonel Blotto contests, and Bayesian patrol security games",
)

include("common.jl")
include("poker_sequence_form.jl")
include("colonel_blotto.jl")
include("patrol_security.jl")
