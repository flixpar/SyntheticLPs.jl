using JuMP
using Random
using Distributions
using StatsBase
using Statistics

"""
Planted evacuation plan on the time-expanded network: per-variable values of
an exact max flow whose value equals total supply (every evacuee reaches a
shelter by the horizon), mapped onto the model's movement (`moves`), waiting
(`waits`) and shelter-intake (`intakes`) variables.
"""
struct TimeExpandedWitness
    moves::Vector{Float64}
    waits::Vector{Float64}
    intakes::Vector{Float64}
end

"""
Trapped-region certificate on the time-expanded network. `region` lists
time-expanded node copies (indices into `node_copies`) holding
`trapped_supply` evacuees at time 0, while everything that can leave the region
— movement copies (`exit_moves`), waiting copies (`exit_waits`) and shelter
intakes inside it (`exit_intakes`) — has total capacity `exit_capacity <
trapped_supply`. Summing the conservation rows of the region's copies refutes
the model from rows and bounds alone. The region is a space-time set (a
congested district over the first periods), not one node.
"""
struct TimeExpandedCertificate
    region::Vector{Int}
    exit_moves::Vector{Int}
    exit_waits::Vector{Int}
    exit_intakes::Vector{Int}
    exit_capacity::Float64
    trapped_supply::Float64
end

"""
    TimeExpandedEvacuationProblem <: ProblemGenerator

Evacuation planning as a dynamic (time-expanded) network flow LP.

# Overview

A geographic road network (`_geo_network`) has per-period capacities and
integer travel times (1-4 periods). Evacuees start at populated zones at time
0 and must reach shelters, each with a per-period intake rate, by the horizon
`H`. Copies of the network for every period, linked by movement arcs
`(u, t) -> (v, t + tau)` and waiting arcs `(v, t) -> (v, t + 1)` (at zones and
staging areas), form the time-expanded network (Ford-Fulkerson dynamic flows):

    minimize    sum_{shelter s, t} t * intake[s,t] + travel_weight * sum cost * move
    subject to  supply[v]*[t == 0] + arrivals(v,t) + wait(v,t-1)
                    = departures(v,t) + wait(v,t) + intake(v,t)       every useful (v, t)
                0 <= move <= road capacity, 0 <= wait <= holding capacity,
                0 <= intake <= shelter intake rate

The objective is the total evacuation time (quickest transshipment). Only
USEFUL copies are generated: node copy `(v, t)` exists iff a source reaches
`v` by `t` and a shelter is reachable from `v` by `H` — the standard
time-expanded preprocessing, so no variable is dead on arrival.

# Data grounding

Shelters (4%-8% of nodes: schools, stadiums) are spread over the region by a
randomised farthest-point placement favouring low-density nodes; zones
(12%-20%) are drawn by population weight, keeping those within reach of a
shelter by the horizon; travel time = distance / speed rounded
up (1-4 periods); road capacity per period lognormal, 1.8x on trunk roads;
zones let evacuees wait at home without limit, staging areas (20% of other
nodes) can hold a lognormal amount of traffic; shelter intake rates lognormal.

# Feasibility control

The constructor computes the EXACT largest uniform population scale
`lambda*` that can be evacuated by `H` (Dinkelbach min-ratio-cut iterations
over exact Dinic max flows on the time-expanded network):

  - `feasible`: populations `load_factor` in [0.6, 0.9] of `lambda*`; the
    witness is the max-flow plan.
  - `infeasible`: `load_factor` in [1.06, 1.25]; certificate = the trapped
    space-time region read off the min cut.
  - `unknown`: `load_factor` in [0.85, 1.15]; `max_flow_value` vs
    `total_supply` decides.

# Sizing

`H = clamp(round(2.5 target^0.25), 6, 48)` periods (raised if a zone needs
longer to reach a shelter); the node count is redrawn (up to twelve times, keeping the closest draw) until
the pruned variable count is within 8% of the target. Rows = useful node
copies (about a quarter of the columns).
"""
struct TimeExpandedEvacuationProblem <: ProblemGenerator
    n_nodes::Int
    horizon::Int
    arcs::Vector{Tuple{Int, Int}}
    travel_time::Vector{Int}
    road_capacity::Vector{Float64}
    road_cost::Vector{Float64}
    hold_capacity::Vector{Float64}
    intake_rate::Vector{Float64}
    supply::Vector{Float64}
    node_copies::Vector{Tuple{Int, Int}}
    moves::Vector{Tuple{Int, Int}}
    waits::Vector{Tuple{Int, Int}}
    intakes::Vector{Tuple{Int, Int}}
    travel_weight::Float64
    positions::Vector{Tuple{Float64, Float64}}
    geography::Symbol
    load_factor::Float64
    max_flow_value::Float64
    total_supply::Float64
    feasible_witness::Union{Nothing, TimeExpandedWitness}
    infeasibility_certificate::Union{Nothing, TimeExpandedCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    _te_structure(n, H, arcs, tau, hold, intake, sources) -> NamedTuple

Useful time-expanded copies: earliest arrival from any source and latest
departure that still reaches a shelter (Dijkstra on travel times both ways),
then node copies, movement copies `(arc, t)`, waiting copies `(v, t)` and
intake copies `(v, t)` among them.
"""
function _te_structure(
    n::Int,
    H::Int,
    arcs::Vector{Tuple{Int, Int}},
    tau::Vector{Int},
    hold::Vector{Float64},
    intake::Vector{Float64},
    sources::Vector{Int},
)
    out_adj, _ = _geo_adjacency(n, arcs)
    earliest, _ = _geo_dijkstra(n, arcs, out_adj, Float64.(tau), sources)
    rev = [(v, u) for (u, v) in arcs]
    rev_adj, _ = _geo_adjacency(n, rev)
    shelters = findall(>(0.0), intake)
    to_shelter, _ = _geo_dijkstra(n, rev, rev_adj, Float64.(tau), shelters)
    useful(v, t) =
        isfinite(earliest[v]) && isfinite(to_shelter[v]) && earliest[v] <= t <= H - to_shelter[v]
    node_copies = [(v, t) for t in 0:H for v in 1:n if useful(v, t)]
    moves = [
        (a, t) for (a, (u, v)) in enumerate(arcs) for
        t in 0:(H - tau[a]) if useful(u, t) && useful(v, t + tau[a])
    ]
    waits = [
        (v, t) for v in 1:n for t in 0:(H - 1) if hold[v] > 0 && useful(v, t) && useful(v, t + 1)
    ]
    intakes = [(v, t) for v in shelters for t in 0:H if useful(v, t)]
    return (; node_copies, moves, waits, intakes)
end

"""
    _te_flow_network(n, H, arcs, tau, cap, hold, intake, st)

Max-flow form of the time-expanded network: copy `(v, t)` is node
`t * n + v`; super source `S = n(H+1) + 1` feeds zones at time 0 (arc caps
set per call), super sink `Z = S + 1` collects intakes. Returns
`(arcs, caps, n_core)` where `n_core` counts the movement + waiting + intake
arcs (in that order, aligned with `st`), followed by the supply arcs.
"""
function _te_flow_network(n::Int, H::Int, arcs, tau, cap, hold, intake, st)
    id(v, t) = t * n + v
    S = n * (H + 1) + 1
    Z = S + 1
    fa = Tuple{Int, Int}[]
    fc = Float64[]
    for (a, t) in st.moves
        u, v = arcs[a]
        push!(fa, (id(u, t), id(v, t + tau[a])))
        push!(fc, cap[a])
    end
    for (v, t) in st.waits
        push!(fa, (id(v, t), id(v, t + 1)))
        push!(fc, hold[v])
    end
    for (v, t) in st.intakes
        push!(fa, (id(v, t), Z))
        push!(fc, intake[v])
    end
    return fa, fc, S, Z
end

"""
    _te_max_scale(fa, fc, S, Z, zone_nodes, s0) -> Float64

EXACT largest population scale `lambda` such that supplies `lambda * s0` at
the zone copies `zone_nodes` can all be evacuated: by max-flow/min-cut,
`lambda* = min_X exitcap(X) / s0(X)` over source-side sets `X`, and
Dinkelbach's iteration from an upper bound finds it in a few max flows.
Returned shaved by `1e-9` so it is deliverable.
"""
function _te_max_scale(fa, fc, S::Int, Z::Int, zone_nodes::Vector{Int}, s0::Vector{Float64})
    N = Z
    sarcs = vcat(fa, [(S, x) for x in zone_nodes])
    total = sum(s0)
    lambda = sum(fc[i] for i in eachindex(fa) if fa[i][2] == Z) / total  # intake bound
    for _ in 1:100
        value, _, side, _, _ = _flow_max_flow(N, S, Z, sarcs, vcat(fc, lambda .* s0))
        value >= lambda * total * (1 - 1e-9) && break
        on = falses(N)
        on[side] .= true
        exitcap = sum(fc[i] for i in eachindex(fa) if on[fa[i][1]] && !on[fa[i][2]]; init=0.0)
        trapped = sum(s0[i] for i in eachindex(zone_nodes) if on[zone_nodes[i]]; init=0.0)
        lambda = min(exitcap / trapped, lambda * (1 - 1e-9))
    end
    return lambda * (1 - 1e-9)
end

function TimeExpandedEvacuationProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be >= 1 (got $target_variables)."))
    target_variables <= NETWORK_FLOW_MAX_ARCS || throw(
        ArgumentError(
            "network_flow/time_expanded supports at most $NETWORK_FLOW_MAX_ARCS variables; " *
            "requested $target_variables.",
        ),
    )
    rng = MersenneTwister(seed)
    H0 = clamp(round(Int, 2.5 * target_variables^0.25), 6, 48)
    per_node = 3.2 + 1.4 * rand(rng)
    # First guess ~ (arcs + waits + intakes) per node per period after
    # pruning; then redraw with a corrected node count until the pruned
    # variable count is within 8% of the target (at most 12 draws).
    n = max(4, round(Int, target_variables / ((per_node + 0.35) * H0 * 0.6)))
    local H, arcs, trunk, positions, weights, geography, tau, road_capacity, road_cost
    local zones, intake_rate, hold_capacity, s0, st
    best = nothing
    best_gap = typemax(Int)
    for attempt in 1:12
        H = H0
        n_arcs = clamp(round(Int, n * per_node), 2 * (n - 1), n * (n - 1))
        geography = let r = rand(rng)
            r < 0.5 ? :clustered : (r < 0.8 ? :uniform : :corridor)
        end
        positions, weights = _geo_positions(rng, n, geography; span=12.0 * sqrt(n))
        arcs, trunk = _geo_network(rng, positions, n_arcs)
        m = length(arcs)
        dist = [_geo_dist(positions, u, v) for (u, v) in arcs]
        speed = median(dist) / 1.6
        tau = [clamp(ceil(Int, dist[a] / speed), 1, 4) for a in 1:m]
        road_capacity = [
            round(40.0 * rand(rng, LogNormal(0.0, 0.5)) * (trunk[a] ? 1.8 : 1.0); digits=1) for
            a in 1:m
        ]
        road_cost = [round(dist[a] * rand(rng, LogNormal(0.0, 0.2)); digits=3) for a in 1:m]

        # Shelters spread over the region (randomised farthest-point
        # placement among low-density nodes); zones drawn by population.
        n_shelter = clamp(round(Int, n * (0.04 + 0.04 * rand(rng))), 1, n - 2)
        nearest = fill(Inf, n)
        shelters = Int[]
        for _ in 1:n_shelter
            score = [
                if v in shelters
                    -Inf
                else
                    min(nearest[v], 1e9) / sqrt(weights[v]) * rand(rng, LogNormal(0.0, 0.3))
                end for v in 1:n
            ]
            v = argmax(score)
            push!(shelters, v)
            for u in 1:n
                nearest[u] = min(nearest[u], _geo_dist(positions, u, v))
            end
        end
        sort!(shelters)
        rest = setdiff(1:n, shelters)
        n_zone = clamp(round(Int, n * (0.12 + 0.08 * rand(rng))), 1, length(rest))
        zones = sort(sample(rng, rest, Weights(weights[rest]), n_zone; replace=false))
        intake_rate = zeros(n)
        intake_rate[shelters] .= [
            round(60.0 * rand(rng, LogNormal(0.0, 0.5)); digits=1) for _ in shelters
        ]
        s0 = 100.0 .* weights[zones] ./ (sum(weights[zones]) / n_zone)
        hold_capacity = zeros(n)
        for v in 1:n
            if v in zones
                # Evacuees can wait at home as long as they need: the bound
                # is everything the shelters could ever take in, so it never
                # binds (a tighter one would make a crowded zone's time-0
                # row refutable on its own).
                hold_capacity[v] = round(sum(intake_rate) * (H + 1); digits=1)
            elseif rand(rng) < 0.2
                hold_capacity[v] = round(80.0 * rand(rng, LogNormal(0.0, 0.5)); digits=1)
            end
        end

        # Zones too far from every shelter to make the horizon are left to
        # a neighbouring plan (at least the closest zone is kept; the horizon
        # stretches only if even that one needs it).
        rev = [(v, u) for (u, v) in arcs]
        rev_adj, _ = _geo_adjacency(n, rev)
        to_shelter, _ = _geo_dijkstra(n, rev, rev_adj, Float64.(tau), shelters)
        keep = [i for (i, v) in enumerate(zones) if to_shelter[v] <= H - 3]
        isempty(keep) && (keep = [argmin(to_shelter[zones])])
        zones = zones[keep]
        s0 = s0[keep]
        H = max(H, round(Int, maximum(to_shelter[zones])) + 3)

        st = _te_structure(n, H, arcs, tau, hold_capacity, intake_rate, zones)
        count = length(st.moves) + length(st.waits) + length(st.intakes)
        draw = (;
            n,
            H,
            arcs,
            trunk,
            positions,
            weights,
            geography,
            tau,
            road_capacity,
            road_cost,
            zones,
            intake_rate,
            hold_capacity,
            s0,
            st,
        )
        if best === nothing || abs(count - target_variables) < best_gap
            best, best_gap = draw, abs(count - target_variables)
        end
        best_gap <= 0.08 * target_variables && break
        n = max(4, round(Int, n * (target_variables / max(count, 1))^0.8))
    end
    (;
        n,
        H,
        arcs,
        trunk,
        positions,
        weights,
        geography,
        tau,
        road_capacity,
        road_cost,
        zones,
        intake_rate,
        hold_capacity,
        s0,
        st,
    ) = best
    n_zone = length(zones)
    fa, fc, S, Z = _te_flow_network(n, H, arcs, tau, road_capacity, hold_capacity, intake_rate, st)
    zone_nodes = [v for v in zones]  # copies (v, 0) have id v
    lambda_star = _te_max_scale(fa, fc, S, Z, zone_nodes, s0)
    load_factor = if feasibility_status == feasible
        0.6 + 0.3 * rand(rng)
    elseif feasibility_status == infeasible
        1.06 + 0.19 * rand(rng)
    else
        0.85 + 0.3 * rand(rng)
    end
    supply = zeros(n)
    supply[zones] .= max.(round.(load_factor * lambda_star .* s0; digits=1), 0.1)
    total_supply = sum(supply)
    sarcs = vcat(fa, [(S, x) for x in zone_nodes])
    value, flows, side, _, _ = _flow_max_flow(Z, S, Z, sarcs, vcat(fc, supply[zones]))

    nm, nw, ni = length(st.moves), length(st.waits), length(st.intakes)
    feasible_witness = nothing
    infeasibility_certificate = nothing
    if feasibility_status == feasible
        value >= total_supply * (1 - 1e-9) ||
            error("network_flow/time_expanded: planted load not deliverable (seed $seed)")
        feasible_witness = TimeExpandedWitness(
            flows[1:nm], flows[(nm + 1):(nm + nw)], flows[(nm + nw + 1):(nm + nw + ni)]
        )
    elseif feasibility_status == infeasible
        value < total_supply * (1 - 1e-6) ||
            error("network_flow/time_expanded: infeasible load deliverable (seed $seed)")
        on = falses(Z)
        on[side] .= true
        id(v, t) = t * n + v
        region = [c for (c, (v, t)) in enumerate(st.node_copies) if on[id(v, t)]]
        exit_moves = [i for i in 1:nm if on[fa[i][1]] && !on[fa[i][2]]]
        exit_waits = [i for i in 1:nw if on[fa[nm + i][1]] && !on[fa[nm + i][2]]]
        exit_intakes = [i for i in 1:ni if on[fa[nm + nw + i][1]]]
        exit_capacity =
            sum(fc[i] for i in exit_moves; init=0.0) +
            sum(fc[nm + i] for i in exit_waits; init=0.0) +
            sum(fc[nm + nw + i] for i in exit_intakes; init=0.0)
        trapped = sum(supply[v] for v in zones if on[v]; init=0.0)
        infeasibility_certificate = TimeExpandedCertificate(
            region, exit_moves, exit_waits, exit_intakes, exit_capacity, trapped
        )
    end
    travel_weight = round(0.01 * H / max(mean(road_cost), 1e-6); sigdigits=3)

    return TimeExpandedEvacuationProblem(
        n,
        H,
        arcs,
        tau,
        road_capacity,
        road_cost,
        hold_capacity,
        intake_rate,
        supply,
        st.node_copies,
        st.moves,
        st.waits,
        st.intakes,
        travel_weight,
        positions,
        geography,
        load_factor,
        value,
        total_supply,
        feasible_witness,
        infeasibility_certificate,
        feasibility_status,
    )
end

"""
    build_model(prob::TimeExpandedEvacuationProblem)

Build the time-expanded evacuation LP. Deterministic — uses only the struct
fields.
"""
function build_model(prob::TimeExpandedEvacuationProblem)
    model = Model()
    n, H = prob.n_nodes, prob.horizon
    nm, nw, ni = length(prob.moves), length(prob.waits), length(prob.intakes)
    @variable(model, 0 <= move[i = 1:nm] <= prob.road_capacity[prob.moves[i][1]])
    @variable(model, 0 <= wait[i = 1:nw] <= prob.hold_capacity[prob.waits[i][1]])
    @variable(model, 0 <= intake[i = 1:ni] <= prob.intake_rate[prob.intakes[i][1]])
    @objective(
        model,
        Min,
        sum(prob.intakes[i][2] * intake[i] for i in 1:ni) +
            prob.travel_weight *
        sum(prob.road_cost[prob.moves[i][1]] * move[i] for i in 1:nm; init=0.0)
    )
    row_of = Dict(c => k for (k, c) in enumerate(prob.node_copies))
    expr = [AffExpr(0.0) for _ in prob.node_copies]  # inflow - outflow
    for (i, (a, t)) in enumerate(prob.moves)
        u, v = prob.arcs[a]
        add_to_expression!(expr[row_of[(u, t)]], -1.0, move[i])
        add_to_expression!(expr[row_of[(v, t + prob.travel_time[a])]], 1.0, move[i])
    end
    for (i, (v, t)) in enumerate(prob.waits)
        add_to_expression!(expr[row_of[(v, t)]], -1.0, wait[i])
        add_to_expression!(expr[row_of[(v, t + 1)]], 1.0, wait[i])
    end
    for (i, (v, t)) in enumerate(prob.intakes)
        add_to_expression!(expr[row_of[(v, t)]], -1.0, intake[i])
    end
    for (k, (v, t)) in enumerate(prob.node_copies)
        rhs = t == 0 ? -prob.supply[v] : 0.0
        @constraint(model, expr[k] == rhs)
    end
    return model
end

register_variant(
    :network_flow,
    :time_expanded,
    TimeExpandedEvacuationProblem,
    "Evacuation planning as a time-expanded (dynamic) network flow: road copies per period with travel times, waiting arcs, shelter intake rates and a deadline, minimizing total evacuation time; exact max-flow placement with a trapped space-time region certificate";
    tags=[:logistics, :network, :unimodular, :staircase],
    max_target_variables=1_000_000,
)
