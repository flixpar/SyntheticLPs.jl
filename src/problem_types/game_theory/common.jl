using Random

"""
Largest `target_variables` accepted by the `game_theory` generators. The poker
variant materialises the sparse payoff matrix together with CFR+ iterates, and
the Blotto and patrol variants materialise every layered/time-expanded arc, so
larger targets are rejected with an `ArgumentError` instead of being silently
undersized (same convention as `network_flow/standard` and
`telecom_network_design/standard`).
"""
const GAME_THEORY_MAX_VARIABLES = 1_000_000

function _game_theory_check_target(variant::AbstractString, target_variables::Int)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be >= 1 (got $target_variables)."))
    target_variables <= GAME_THEORY_MAX_VARIABLES || throw(
        ArgumentError(
            "game_theory/$variant supports at most $GAME_THEORY_MAX_VARIABLES " *
            "variables; requested $target_variables.",
        ),
    )
    return nothing
end

"""
    _game_value_requirement(rng, status, lower, upper, unit) -> Float64

Draw a guaranteed-value requirement `v_req` for a player who MAXIMIZES the
game value, given rigorous bounds `lower <= value <= upper` (each certified by
an explicit strategy of one player and the other's exact best response) and a
payoff `unit` setting the margin scale:

  - `feasible`: `v_req = lower - δ` — the stored strategy already guarantees
    more than required;
  - `infeasible`: `v_req = upper + δ` — no strategy can guarantee it, as the
    opponent strategy behind `upper` holds every strategy to at most `upper`;
  - `unknown`: uniform on `[lower - w, upper + w]` with `w` half the bound gap
    (at least `2%` of `unit`), which brackets the unknown true value from both
    sides.

`δ` is `2%-10%` of `unit`, far above solver tolerances, so the planted labels
are not knife edges.
"""
function _game_value_requirement(
    rng::AbstractRNG, status::FeasibilityStatus, lower::Float64, upper::Float64, unit::Float64
)
    δ = unit * (0.02 + 0.08 * rand(rng))
    if status == feasible
        return lower - δ
    elseif status == infeasible
        return upper + δ
    else
        w = max(0.5 * (upper - lower), 0.02 * unit)
        return lower - w + (upper - lower + 2w) * rand(rng)
    end
end
