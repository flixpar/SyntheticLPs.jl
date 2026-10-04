"""
    ForestModelIProblem

Alias for `ForestPlanningProblem{:model_i}`: Johnson & Scheurman's **Model I**
harvest-scheduling LP (the FORPLAN / Spectrum / Woodstock formulation). Every
column is one whole-horizon prescription of one stratum (analysis area):

  - do nothing (grow to the horizon's end);
  - clearcut in period `h1` (any period in which the stand is merchantable),
    regenerate with option `r1` (natural regeneration, planting improved
    stock, or conversion to the regional plantation type), and optionally
    clearcut the regenerated stand again in `h2 ∈ h1 + ρ0 + {0, 1, 2, 3}`
    (`ρ0` = its minimum rotation in periods), replanting under the same regime;
  - commercially thin in the stand's thinning window, then either grow to the
    end or clearcut later in `h1` and regenerate with `r1`.

Each stratum's columns compete through one area-accounting equality
(`Σ_p area[s,p] = A_s`), and all of them couple through the per-period
harvest-volume definitions, the even-flow rows, the watershed green-up rows
and the ending-inventory floor.
"""
const ForestModelIProblem = ForestPlanningProblem{:model_i}

"Append Model I prescriptions of stratum `s` (at most `budget`); returns the number added."
function _forest_model_i_stratum!(b::_ForestBuilder, s::Int, budget::Int)
    T, L = b.T, b.L
    m0 = b.stratum_model[s]
    sm0 = b.models[m0]
    spec0 = b.types[sm0.type_index]
    a0 = b.stratum_age[s]
    age(t) = a0 + (t - 0.5) * L
    age_end = a0 + T * L
    E = [t for t in 1:T if _forest_mature(spec0, sm0, age(t))]
    nopt = length(b.regen_targets[m0])
    d = _ForestColumnDraft()
    added = 0

    # Do nothing.
    _forest_draft_reset!(d)
    _forest_push_column!(b, d, s, 0, 0, 0, 0, 0, _forest_standing(b, m0, age_end, -1.0))
    added += 1

    # Single-rotation prescriptions first, so a stratum truncated at the
    # streamed tail still offers a clearcut in every merchantable period.
    # Regeneration options that cannot pay off before the horizon ends are
    # pruned (see `_forest_terminal_regen_keep`).
    for h1 in E
        keep = _forest_terminal_regen_keep(b, m0, h1, trues(nopt))
        for r1 in 1:nopt
            keep[r1] || continue
            added >= budget && return added
            m1 = b.regen_targets[m0][r1]
            _forest_draft_reset!(d)
            d.cutvol1 = _forest_clearcut_event!(d, b, m0, h1, age(h1), -1.0)
            _forest_regen_event!(d, b, m0, r1, h1)
            ei = _forest_standing(b, m1, (T - h1 + 0.5) * L, -1.0)
            _forest_push_column!(b, d, s, 0, 0, h1, r1, 0, ei)
            added += 1
        end
    end

    # Second rotations: replant under the same regime and clearcut again.
    for h1 in E, r1 in 1:nopt
        m1 = b.regen_targets[m0][r1]
        rho0 = _forest_min_rotation_periods(b, m1)
        r2 = _forest_same_regime_option(b, m1)
        m2 = b.regen_targets[m1][r2]
        for rho in 0:3
            h2 = h1 + rho0 + rho
            h2 > T && break
            added >= budget && return added
            _forest_draft_reset!(d)
            d.cutvol1 = _forest_clearcut_event!(d, b, m0, h1, age(h1), -1.0)
            _forest_regen_event!(d, b, m0, r1, h1)
            d.cutvol2 = _forest_clearcut_event!(d, b, m1, h2, Float64((h2 - h1) * L), -1.0)
            _forest_regen_event!(d, b, m1, r2, h2)
            ei = _forest_standing(b, m2, (T - h2 + 0.5) * L, -1.0)
            _forest_push_column!(b, d, s, 0, 0, h1, r1, h2, ei)
            added += 1
        end
    end

    # Commercial thinning variants (first period inside the thinning window).
    th = _forest_thin_period(b, m0, age, 1)
    if th > 0
        thin_age = age(th)
        added >= budget && return added
        _forest_draft_reset!(d)
        _forest_thin_event!(d, b, m0, th, thin_age)
        _forest_push_column!(b, d, s, 0, th, 0, 0, 0, _forest_standing(b, m0, age_end, thin_age))
        added += 1
        for h1 in E
            h1 <= th && continue
            keep = _forest_terminal_regen_keep(b, m0, h1, trues(nopt))
            for r1 in 1:nopt
                keep[r1] || continue
                added >= budget && return added
                m1 = b.regen_targets[m0][r1]
                _forest_draft_reset!(d)
                _forest_thin_event!(d, b, m0, th, thin_age)
                d.cutvol1 = _forest_clearcut_event!(d, b, m0, h1, age(h1), thin_age)
                _forest_regen_event!(d, b, m0, r1, h1)
                ei = _forest_standing(b, m1, (T - h1 + 0.5) * L, -1.0)
                _forest_push_column!(b, d, s, 0, th, h1, r1, 0, ei)
                added += 1
            end
        end
    end
    return added
end

"""
First period `t ∈ first:T-1` whose (post-lag) stand age lies in the type's
commercial-thinning window, or 0 when the type is not thinned or the window is
missed.
"""
function _forest_thin_period(b::_ForestBuilder, m::Int, age::Function, first::Int)
    sm = b.models[m]
    spec = b.types[sm.type_index]
    lo, hi = spec.thin_window
    hi <= 0.0 && return 0
    for t in first:(b.T - 1)
        a = age(t) - sm.lag
        lo <= a <= hi && return t
    end
    return 0
end

#     ForestPlanningProblem{:model_i}(target_variables, feasibility_status, seed)
#
# Sample a Model I instance: region, horizon and prices; then watersheds of
# strata streamed until the prescription columns plus the `T*K` harvest
# accounting variables reach `target_variables` (the last stratum's
# prescription list is truncated, so the count is exact up to one column).
# Feasibility is calibrated by `_forest_finalize` (planted area-control witness,
# Lagrangian certificate).
function ForestPlanningProblem{:model_i}(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng, b, budget = _forest_start(target_variables, seed)
    profile = _forest_age_profile(rng)
    zone = 0
    while budget >= 2
        greenup, cands = _forest_sample_zone(rng, b, profile)
        zone += 1
        for cand in cands
            budget < 2 && break
            s = _forest_commit_stratum!(rng, b, zone, greenup, cand)
            budget -= _forest_model_i_stratum!(b, s, budget)
        end
    end
    return _forest_finalize(ForestPlanningProblem{:model_i}, rng, b, feasibility_status)
end

register_variant(
    :forest_planning,
    :model_i,
    ForestModelIProblem,
    "Model I forest harvest scheduling: whole-horizon stratum prescriptions (clearcut timing, regeneration, second rotation, thinning) coupled by harvest-volume accounting, even-flow, watershed green-up, mill-supply and ending-inventory rows";
    default=true,
)
