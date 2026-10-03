"""
    ForestModelIIProblem

Alias for `ForestPlanningProblem{:model_ii}`: Johnson & Scheurman's **Model
II** harvest-scheduling LP. Columns describe one rotation at a time:

  - from a stratum (existing stand): grow to the end, or clearcut in period
    `j` and regenerate with option `r`, optionally after a commercial
    thinning;
  - from a regeneration node `(stand model, watershed, period i)` — all area
    regenerated in watershed `z` in period `i` into the same stand model
    (forest type x site class x regime), whatever stratum it came from: grow
    to the end, or clearcut again in `j >= i + ρ0` and regenerate.

A clearcut column whose regenerated stand can still reach merchantability
within the horizon flows into the destination node; otherwise it ends there
(its ending inventory counts the young stand). Node balance rows
`Σ out - Σ in = 0` make the area accounting a network (a DAG ordered by
period) with side constraints: harvest-volume definitions, even flow,
watershed green-up and the ending-inventory floor. At long horizons Model II
needs far fewer columns per stratum than Model I, but adds one balance row
per node.
"""
const ForestModelIIProblem = ForestPlanningProblem{:model_ii}

"Return the node `(model, zone, period)`, creating it (and its grow-to-end column) if new."
function _forest_node!(rng::AbstractRNG, b::_ForestBuilder, m::Int, z::Int, i::Int)
    key = (m, z, i)
    nd = get(b.node_lookup, key, 0)
    nd > 0 && return nd, false
    push!(b.node_model, m)
    push!(b.node_zone, z)
    push!(b.node_period, i)
    nd = length(b.node_model)
    b.node_lookup[key] = nd
    src = FOREST_NODE_OFFSET + nd
    push!(b.node_pref, rand(rng, 1:length(b.regen_targets[m])))
    d = _ForestColumnDraft()
    ei = _forest_standing(b, m, (b.T - i + 0.5) * b.L, -1.0)
    _forest_push_column!(b, d, src, 0, 0, 0, 0, 0, ei)
    return nd, true
end

"""
Append the Model II columns of source `src` (a stratum, or node
`src - FOREST_NODE_OFFSET`), at most `budget` of them. The grow-to-end column
of a stratum is created here; a node's is created with the node. A column into
a new node is taken only when 3 columns remain (itself, the node's
grow-to-end column, and one reserved clearcut column of the node). Nodes this
source opens are expanded after all of the source's own options (so a source
truncated at the streamed tail still offers clearcuts in every merchantable
period), each with its reserved column, so every node gets a real choice: a
clearcut in period `T` always regenerates into no node, so it fits in a single
remaining column. Returns the number of columns added
(including those of nodes created on the way).
"""
function _forest_model_ii_source!(rng::AbstractRNG, b::_ForestBuilder, src::Int, budget::Int)
    T, L = b.T, b.L
    is_stratum = src < FOREST_NODE_OFFSET
    m, z, o, a0, pref = if is_stratum
        b.stratum_model[src], b.stratum_zone[src], 0, b.stratum_age[src], b.source_pref[src]
    else
        nd = src - FOREST_NODE_OFFSET
        b.node_model[nd], b.node_zone[nd], b.node_period[nd], 0.0, b.node_pref[nd]
    end
    age = is_stratum ? (t -> a0 + (t - 0.5) * L) : (t -> Float64((t - o) * L))
    age_end = is_stratum ? a0 + T * L : (T - o + 0.5) * L
    sm = b.models[m]
    spec = b.types[sm.type_index]
    E = [t for t in (o + 1):T if _forest_mature(spec, sm, age(t))]
    nopt = length(b.regen_targets[m])
    d = _ForestColumnDraft()
    added = 0
    pending = Int[]

    if is_stratum
        _forest_draft_reset!(d)
        _forest_push_column!(b, d, src, 0, 0, 0, 0, 0, _forest_standing(b, m, age_end, -1.0))
        added += 1
    end

    th = _forest_thin_period(b, m, age, o + 1)
    for thin in (th > 0 ? (0, th) : (0,))
        thin_age = thin > 0 ? age(thin) : -1.0
        if thin > 0
            added + 1 + length(pending) > budget && continue
            _forest_draft_reset!(d)
            _forest_thin_event!(d, b, m, thin, thin_age)
            _forest_push_column!(b, d, src, 0, thin, 0, 0, 0, _forest_standing(b, m, age_end, thin_age))
            added += 1
        end
        # Pass 1 offers one clearcut per period (preferred regeneration
        # first), pass 2 the remaining regeneration options, so a source
        # truncated at the streamed tail still covers every period.
        done = falses(T, nopt)
        for pass in 1:2, t in E
            t <= thin && continue
            for k in 0:(nopt - 1)
                r = mod1(pref + k, nopt)
                done[t, r] && continue
                m2 = b.regen_targets[m][r]
                has_node = t + _forest_min_rotation_periods(b, m2) <= T
                new_node = has_node && !haskey(b.node_lookup, (m2, z, t))
                added + (new_node ? 3 : 1) + length(pending) > budget && continue
                dest = 0
                if has_node
                    dest, created = _forest_node!(rng, b, m2, z, t)
                    created && (added += 1)
                end
                _forest_draft_reset!(d)
                thin > 0 && _forest_thin_event!(d, b, m, thin, thin_age)
                d.cutvol1 = _forest_clearcut_event!(d, b, m, t, age(t), thin_age)
                _forest_regen_event!(d, b, m, r, t)
                ei = has_node ? 0.0 : _forest_standing(b, m2, (T - t + 0.5) * L, -1.0)
                _forest_push_column!(b, d, src, dest, thin, t, r, 0, ei)
                added += 1
                done[t, r] = true
                new_node && push!(pending, dest)
                pass == 1 && break
            end
        end
    end
    # Expand the nodes this source opened; each later pending node keeps one
    # reserved column for its own clearcut.
    for (i, nd) in enumerate(pending)
        added += _forest_model_ii_source!(rng, b, FOREST_NODE_OFFSET + nd, budget - added - (length(pending) - i))
    end
    return added
end

#     ForestPlanningProblem{:model_ii}(target_variables, feasibility_status, seed)
#
# Sample a Model II instance: region, horizon and prices; then watersheds of
# strata streamed one stratum at a time, each followed by the depth-first
# expansion of the regeneration nodes it (transitively) reaches, until the
# columns plus the `T*K` harvest accounting variables reach `target_variables`
# (exact up to one column). Feasibility is calibrated by `_forest_finalize`.
function ForestPlanningProblem{:model_ii}(
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
            budget -= _forest_model_ii_source!(rng, b, s, budget)
        end
    end
    return _forest_finalize(ForestPlanningProblem{:model_ii}, rng, b, feasibility_status)
end

register_variant(
    :forest_planning,
    :model_ii,
    ForestModelIIProblem,
    "Model II forest harvest scheduling: rotation-by-rotation area flows through watershed regeneration nodes (a network with side constraints) with harvest-volume accounting, even-flow, green-up, mill-supply and ending-inventory rows",
)
