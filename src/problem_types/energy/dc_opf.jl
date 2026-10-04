using JuMP
using Random
using Distributions
using LinearAlgebra
using SparseArrays

"""
Planted DC power-flow solution: generator `dispatch`, bus `angles` (reference
bus at 0) and line `flows = susceptance·(θ_from − θ_to)`, which satisfy nodal
balance exactly because the angles solve the reduced network Laplacian for the
dispatch's nodal injections.
"""
struct DCPowerFlowWitness
    dispatch::Vector{Float64}
    angles::Vector{Float64}
    flows::Vector{Float64}
end

"""
    DCOptimalPowerFlowProblem <: ProblemGenerator

Single-snapshot DC optimal power flow on a meshed transmission grid.

# Overview

Least-cost dispatch of a technology-grounded fleet over a bus-level grid under
the linear DC power-flow approximation: generator outputs `p[g]`, bus voltage
angles `θ[b]` (reference bus fixed at 0) and line flows `f[l]`, with

  - flow definition `f[l] = B_l·(θ_from − θ_to)` (susceptance-weighted,
    non-unimodular coefficients spanning two orders of magnitude),
  - nodal balance `Σ_{g at b} p[g] − Σ_{l out of b} f[l] + Σ_{l into b} f[l] = d_b`,
  - thermal limits `|f[l]| ≤ rating_l` and generator limits `pmin ≤ p ≤ pmax`.

The grid is geometric (`_energy_grid`): buses cluster around load centres on a
map whose side grows like √B, lines come from nearest-neighbour candidates
(spanning forest + meshing, radial spurs mostly given a second connection),
susceptance falls with line length and ratings follow voltage class (115–230 kV
vs 345 kV). About 20 % of buses are switching stations without load. Generators
come from the shared technology catalogue with snapshot availability for a
random hour (solar follows the sun). Variables: `n_generators + n_buses +
n_lines`, matched exactly to the target.

# Feasibility control

  - `feasible`: total demand is 35–75 % of the way from `Σ pmin` to `Σ pmax`;
    the witness dispatch moves every generator the same fraction up its range,
    the reduced Laplacian is solved for the angles, and each rating is widened to
    at least 1.15 × the witness flow. The `DCPowerFlowWitness` meets every row.
  - `infeasible`: the feasible data plus a load pocket — a connected set of
    buses (2–6 % of the grid, ≥ 3) whose demand exceeds its local generation plus
    the ratings of the lines crossing into it by 6–15 %; no bus and no proper
    subset of the pocket is short on its own. `DCPocketCertificate` records the
    cut.
  - `unknown`: natural ratings (no widening) and demand 45–90 % of the way up
    the generation range: congestion decides.

# Fields

  - `n_buses`, `n_lines`, `n_generators::Int`
  - `line_from`, `line_to::Vector{Int}`, `susceptance`, `line_limit::Vector{Float64}`
  - `gen_bus::Vector{Int}`, `gen_tech::Vector{Symbol}`, `gen_cost`, `pmin`,
    `pmax::Vector{Float64}`
  - `demand::Vector{Float64}`: load per bus (MW)
  - `ref_bus::Int`
  - `angle_limit::Float64`: bus angle bound (centiradians; at least 60°)
  - `feasibility_status`, `feasible_witness`, `infeasibility_certificate`
"""
struct DCOptimalPowerFlowProblem <: ProblemGenerator
    n_buses::Int
    n_lines::Int
    n_generators::Int
    line_from::Vector{Int}
    line_to::Vector{Int}
    susceptance::Vector{Float64}
    line_limit::Vector{Float64}
    gen_bus::Vector{Int}
    gen_tech::Vector{Symbol}
    gen_cost::Vector{Float64}
    pmin::Vector{Float64}
    pmax::Vector{Float64}
    demand::Vector{Float64}
    ref_bus::Int
    angle_limit::Float64
    feasibility_status::FeasibilityStatus
    feasible_witness::Union{Nothing, DCPowerFlowWitness}
    infeasibility_certificate::Union{Nothing, DCPocketCertificate}
end

"""
    _dc_base_network(rng, B, L_target, G) -> NamedTuple

Grid, line parameters, fleet and load shares shared by the DC variants.
"""
function _dc_base_network(rng::AbstractRNG, B::Int, L_target::Int, G::Int)
    x, y, from, to, len, ehv = _energy_grid(rng, B, L_target)
    sus, rating, voltage = _energy_line_parameters(rng, len, ehv)
    hour = rand(rng, 0:23)
    fleet = _energy_grid_fleet(rng, B, G; hour=hour)
    shares = _dc_regional_balance(x, y, _energy_bus_loads(rng, B), fleet)
    return (; x, y, from, to, len, sus, rating, voltage, hour, fleet, shares)
end

"""
    _dc_regional_balance(x, y, shares, fleet) -> Vector{Float64}

Planned grids keep generation and load regionally balanced, so bulk transfers
(and voltage-angle spreads) stay moderate as the grid grows. Buses are grouped
into map regions of about 50 buses; each region's share of total load is
moved 85 % of the way toward its share of generation capacity, keeping the
within-region load pattern.
"""
function _dc_regional_balance(x, y, shares, fleet)
    B = length(shares)
    nr = max(1, round(Int, sqrt(B / 50)))
    lo_x, hi_x = extrema(x)
    lo_y, hi_y = extrema(y)
    region(b) = 1 + min(nr - 1, floor(Int, nr * (x[b] - lo_x) / max(hi_x - lo_x, eps()))) +
        nr * min(nr - 1, floor(Int, nr * (y[b] - lo_y) / max(hi_y - lo_y, eps())))
    R = nr * nr
    reg = [region(b) for b in 1:B]
    load = zeros(R)
    gen = zeros(R)
    for b in 1:B
        load[reg[b]] += shares[b]
    end
    for g in eachindex(fleet.pmax)
        gen[reg[fleet.gen_bus[g]]] += fleet.pmax[g]
    end
    gen ./= max(sum(gen), eps())
    out = copy(shares)
    for b in 1:B
        r = reg[b]
        load[r] > 0 || continue
        out[b] *= (0.15 * load[r] + 0.85 * gen[r]) / load[r]
    end
    # Regions with generation but no load keep their generation exporting.
    return out ./ sum(out)
end

"""Nodal injections of a dispatch `p` against bus loads `d`."""
function _dc_injection(B, gen_bus, p, d)
    inj = -copy(d)
    for g in eachindex(p)
        inj[gen_bus[g]] += p[g]
    end
    return inj
end

"""Proportional dispatch: every generator at the same fraction of its range."""
function _dc_proportional_dispatch(pmin, pmax, total)
    β = (total - sum(pmin)) / max(sum(pmax) - sum(pmin), eps())
    return pmin .+ clamp(β, 0.0, 1.0) .* (pmax .- pmin)
end

"""
    _dc_plant_pocket!(rng, B, from, to, gen_bus, pmax, demand, S, states; margin,
                      certify=1) -> Union{Nothing,NamedTuple}

Turn the connected bus set `S` into a load pocket. `states` lists the network
states the model contains as `(ratings, outage)` pairs (base case: `outage =
0`); the pocket's demand is raised to `(1 + margin)` × (local capacity + the
capability of the lines crossing into it in state `states[certify]`). The
demand is spread in proportion to each bus's own supply (local generation plus
its crossing lines' capability in the certified state), so every bus carries
the same relative deficit, and the lines inside the pocket are raised (in every
rating vector) to at least 1.2 × the pocket's total deficit: no bus and no
proper subset of the pocket is short on its own (small pockets are exactly what
presolve's bound propagation detects), only the whole pocket is. Mutates
`demand` and the internal entries of the rating vectors.
"""
function _dc_plant_pocket!(rng::AbstractRNG, B, from, to, gen_bus, pmax, demand, S, states; margin, certify=1)
    length(S) >= 2 || return nothing
    cert_rating, cert_outage = states[certify]
    cut = _grid_cut(B, from, to, S)
    live_cut = [l for l in cut if l != cert_outage]
    isempty(live_cut) && return nothing
    inS = falses(B)
    inS[S] .= true
    w = zeros(B)
    for g in eachindex(pmax)
        inS[gen_bus[g]] && (w[gen_bus[g]] += pmax[g])
    end
    local_cap = sum(w[S])
    for l in live_cut
        inS[from[l]] ? (w[from[l]] += cert_rating[l]) : (w[to[l]] += cert_rating[l])
    end
    imports = sum(cert_rating[l] for l in live_cut)
    need = (1 + margin) * (local_cap + imports)
    W = sum(w[S])
    W > 0 || return nothing
    deficit = need - W
    internal(l) = inS[from[l]] && inS[to[l]]
    # Check before mutating: in every state, every pocket bus must be able to
    # serve its new load on its own (an outage can remove a bus's only
    # internal line); otherwise the caller tries another pocket.
    gen = zeros(B)
    for g in eachindex(pmax)
        gen[gen_bus[g]] += pmax[g]
    end
    pos = Dict(b => i for (i, b) in enumerate(S))
    incident = [Int[] for _ in S]
    for l in eachindex(from)
        haskey(pos, from[l]) && push!(incident[pos[from[l]]], l)
        haskey(pos, to[l]) && push!(incident[pos[to[l]]], l)
    end
    for (r, out) in states, (i, b) in enumerate(S)
        cap = gen[b] + sum(
            (internal(l) ? max(r[l], 1.2 * deficit) : r[l] for l in incident[i] if l != out); init=0.0
        )
        cap >= 1.02 * need * w[b] / W || return nothing
    end
    for b in S
        demand[b] = need * w[b] / W
    end
    for r in unique(objectid, [st[1] for st in states]), l in eachindex(from)
        internal(l) && (r[l] = max(r[l], 1.2 * deficit))
    end
    return (; S=sort(S), cut=live_cut, local_cap, imports, need)
end

"""
    _dc_planning_ratings!(rng, B, from, to, sus, rating, fleet, demand, total, ref, voltage)

Natural ratings for `unknown` instances: lines are rated for the flows of a
typical (proportional) dispatch times a per-line planning slack (1 + lognormal,
median 1.25) and, on the bulk (345/500 kV, non-radial) network, one
system-wide stress factor in [0.65, 1.05] — below 1 the bulk grid is tighter
than the reference flows and redispatch must relieve the congestion, which may
or may not be possible; local 115–230 kV and radial lines are never rated
below their reference flow — and every bus is adequately connected: its incident ratings cover
110 % of both its load beyond local generation and its must-run output beyond
local load.
"""
function _dc_planning_ratings!(rng::AbstractRNG, B, from, to, sus, rating, fl, demand, total, ref, voltage)
    p = _dc_proportional_dispatch(fl.pmin, fl.pmax, total)
    θ, flow = _dc_flows(B, from, to, sus, _dc_injection(B, fl.gen_bus, p, demand), ref)
    # One system-wide planning-stress factor (below 1: the network is tighter
    # than the reference flows and redispatch must relieve it) times per-line
    # slack ≥ 1.
    stress = _e_unif(rng, (0.65, 1.05))
    bridges = _graph_bridges(B, from, to)
    for l in eachindex(from)
        slack = 1.0 + exp(log(0.25) + 0.6 * randn(rng))
        # Radial (bridge) lines carry whatever their side of the grid needs and
        # cannot be relieved by redispatch elsewhere; planners never rate them
        # below that flow.
        bulk = voltage[l] != :lv && !bridges[l]
        rating[l] = max(0.5 * rating[l], abs(flow[l]) * (bulk ? stress : max(stress, 1.0)) * slack)
    end
    gen_cap = zeros(B)
    for g in eachindex(fl.pmax)
        gen_cap[fl.gen_bus[g]] += fl.pmax[g]
    end
    incident = [Int[] for _ in 1:B]
    for l in eachindex(from)
        push!(incident[from[l]], l)
        push!(incident[to[l]], l)
    end
    gen_min = zeros(B)
    for g in eachindex(fl.pmin)
        gen_min[fl.gen_bus[g]] += fl.pmin[g]
    end
    for b in 1:B
        isempty(incident[b]) && continue
        lines = sum(rating[l] for l in incident[b])
        # Import adequacy (load beyond local generation) and export adequacy
        # (must-run output beyond local load), both with 10 % slack.
        need = 1.1 * max(demand[b] - gen_cap[b], gen_min[b] - demand[b], 0.0)
        lines < need && (rating[incident[b]] .*= need / lines)
    end
    return θ
end

"""
Bus angle bound (centiradians): 60° or 1.3 × the largest reference-dispatch
angle (with a little noise), whichever is larger. Angles are measured in
centiradians so that `flow[MW] = B_pu·(θ_from − θ_to)` on a 100 MVA base.
"""
_dc_angle_limit(rng::AbstractRNG, θ) = max(100 * π / 3, 1.3 * maximum(abs, θ) * _e_unif(rng, (1.0, 1.1)))

"""
    DCOptimalPowerFlowProblem(target_variables, feasibility_status, seed)

Build a DC-OPF snapshot with exactly `target_variables` columns
(`n_generators + n_buses + n_lines`, for targets ≥ 10). See the type docstring.
"""
function DCOptimalPowerFlowProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    rng = MersenneTwister(seed)
    target = max(target_variables, 10)
    ef = _e_unif(rng, (1.30, 1.55))
    gf = _e_unif(rng, (0.25, 0.40))
    B = max(4, round(Int, target / (1 + gf)))
    G = max(2, target - B)
    L_target = min(round(Int, ef * B), B * (B - 1) ÷ 2)
    net = _dc_base_network(rng, B, L_target, G)
    L = length(net.from)
    fl = net.fleet
    sum_min, sum_max = sum(fl.pmin), sum(fl.pmax)
    frac = feasibility_status == unknown ? _e_unif(rng, (0.45, 0.90)) : _e_unif(rng, (0.35, 0.75))
    total = sum_min + frac * (sum_max - sum_min)
    demand = total .* net.shares
    rating = copy(net.rating)
    degree = zeros(Int, B)
    for l in 1:L
        degree[net.from[l]] += 1
        degree[net.to[l]] += 1
    end
    ref = argmax(degree)

    witness = nothing
    certificate = nothing
    if feasibility_status == unknown
        θn = _dc_planning_ratings!(rng, B, net.from, net.to, net.sus, rating, fl, demand, total, ref, net.voltage)
        angle_limit = _dc_angle_limit(rng, θn)
    else
        p = _dc_proportional_dispatch(fl.pmin, fl.pmax, total)
        θ, flow = _dc_flows(B, net.from, net.to, net.sus, _dc_injection(B, fl.gen_bus, p, demand), ref)
        for l in 1:L
            rating[l] = max(rating[l], 1.15 * abs(flow[l]) + 1.0)
        end
        witness = DCPowerFlowWitness(p, θ, flow)
        angle_limit = _dc_angle_limit(rng, θ)
        if feasibility_status == infeasible
            witness = nothing
            m = _e_unif(rng, (0.06, 0.15))
            loaded = [b for b in 1:B if demand[b] > 0]
            for _ in 1:20
                size = clamp(round(Int, B * _e_unif(rng, (0.02, 0.06))), min(3, B - 1), B - 1)
                S = _grid_bfs(B, net.from, net.to, loaded[rand(rng, eachindex(loaded))], size)
                pk = _dc_plant_pocket!(rng, B, net.from, net.to, fl.gen_bus, fl.pmax, demand, S, [(rating, 0)]; margin=m)
                pk === nothing && continue
                certificate = DCPocketCertificate(pk.S, pk.cut, 0, 0, pk.local_cap, pk.imports, sum(demand[pk.S]))
                break
            end
            certificate === nothing && error("energy/dc_opf: could not plant a load pocket")
        end
    end

    return DCOptimalPowerFlowProblem(
        B,
        L,
        G,
        net.from,
        net.to,
        net.sus,
        rating,
        fl.gen_bus,
        fl.tech,
        fl.cost,
        fl.pmin,
        fl.pmax,
        demand,
        ref,
        angle_limit,
        feasibility_status,
        witness,
        certificate,
    )
end

"""
    _dc_network_block!(model, B, from, to, sus, limit, gen_bus, demand, ref, p,
                       angle_limit; outage=0, tag="") -> θ

Add one network state in the B-θ form used by production DC-OPF codes (e.g.
MATPOWER): bus angles bounded by `±angle_limit` (reference fixed at 0), one ranged thermal-limit row
`−limit ≤ B_l·(θ_from − θ_to) ≤ limit` per live line (the `outage` line
omitted), and nodal balance rows `Σ_{g at b} p[g] − Σ_{l ∋ b} B_l·(θ_b − θ_other)
= d_b` driven by the dispatch `p`. Used once by `dc_opf` and once per state
(base case + each contingency) by `security_constrained_dc_opf`.
"""
function _dc_network_block!(
    model::Model, B, from, to, sus, limit, gen_bus, demand, ref, p, angle_limit; outage::Int=0, tag::String=""
)
    L = length(from)
    θ = @variable(model, [b=1:B], lower_bound=-angle_limit, upper_bound=angle_limit, base_name="theta$tag")
    fix(θ[ref], 0.0; force=true)
    balance = [AffExpr(0.0) for _ in 1:B]
    for g in eachindex(p)
        add_to_expression!(balance[gen_bus[g]], 1.0, p[g])
    end
    for l in 1:L
        l == outage && continue
        a, c, s = from[l], to[l], sus[l]
        @constraint(model, -limit[l] <= s * θ[a] - s * θ[c] <= limit[l])
        # Flow a → c leaves a and enters c.
        add_to_expression!(balance[a], -s, θ[a])
        add_to_expression!(balance[a], s, θ[c])
        add_to_expression!(balance[c], s, θ[a])
        add_to_expression!(balance[c], -s, θ[c])
    end
    for b in 1:B
        @constraint(model, balance[b] == demand[b])
    end
    return θ
end

"""
    build_model(prob::DCOptimalPowerFlowProblem)

Generator limits, one DC network block, and minimum generation cost.
"""
function build_model(prob::DCOptimalPowerFlowProblem)
    model = Model()
    G = prob.n_generators
    @variable(model, prob.pmin[g] <= p[g=1:G] <= prob.pmax[g])
    model[:theta] = _dc_network_block!(
        model, prob.n_buses, prob.line_from, prob.line_to, prob.susceptance, prob.line_limit, prob.gen_bus,
        prob.demand, prob.ref_bus, p, prob.angle_limit,
    )
    @objective(model, Min, sum(prob.gen_cost[g] * p[g] for g in 1:G))
    return model
end

register_variant(
    :energy,
    :dc_opf,
    DCOptimalPowerFlowProblem,
    "DC optimal power flow snapshot on a geometric meshed grid: technology-grounded fleet, susceptance-weighted flows, voltage-class thermal ratings, nodal balance",
)
