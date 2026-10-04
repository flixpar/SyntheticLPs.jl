using JuMP
using Random
using Distributions

"""
Planted preventive-SCOPF solution: one generator `dispatch` shared by every
network state and the bus angles of each state — `angles[:, 1]` for the base
case, `angles[:, 1 + c]` after contingency `c` — each solving its own reduced
Laplacian, so every nodal balance row holds exactly.
"""
struct SCOPFWitness
    dispatch::Vector{Float64}
    angles::Matrix{Float64}
end

"""
    SecurityConstrainedDCOPFProblem <: ProblemGenerator

Preventive N-1 security-constrained DC optimal power flow.

# Overview

The bus-level DC-OPF of `energy/dc_opf` (same geometric grid, fleet and B-θ
formulation) extended with a screened list of `C` single-line outages. The
dispatch `p` is shared by all states; every state — the base case and each
post-contingency network with its outaged line removed — has its own bus
angles, nodal balance rows and thermal rows:

  - base case: `|B_l·(θ⁰_from − θ⁰_to)| ≤ rating_l` for every line;
  - contingency `c` (line `k_c` out): `|B_l·(θᶜ_from − θᶜ_to)| ≤
    emergency_l` for every `l ≠ k_c`, with short-term emergency ratings 10–30 %
    above normal.

This is the block-angular structure of the LPs ISOs solve every five minutes:
`1 + C` network copies linked only through the dispatch columns (the classic
target of Benders / column-generation decompositions). The contingency list is
the output of a screening pass: the `C` most heavily loaded non-bridge lines
(outages that do not island the grid) under a reference dispatch. `C ≈ √target
/ 8` (1 at the smallest sizes, 40 at 100k, at most 60) and the grid is sized so
that `n_generators + (1 + C)·n_buses` columns hit the target exactly.

# Feasibility control

  - `feasible`: the proportional reference dispatch is the witness; the base
    and every post-contingency Laplacian are solved, normal ratings cover 115 %
    of base flows and emergency ratings 110 % of the worst post-contingency
    flows. The `SCOPFWitness` meets every row of every state.
  - `infeasible`: an N-1 load pocket — a connected bus set on one side of a
    screened line `k` whose demand fits within local generation plus the normal
    ratings of the lines feeding it (the base case is fine; `k` is rated as the
    pocket's main feeder), but exceeds local generation plus the emergency
    ratings of the feeders that survive the loss of `k` by 6–15 %. Every bus's
    own balance row stays satisfiable in both states. The `DCPocketCertificate`
    names the contingency.
  - `unknown`: natural N-1 planning — ratings cover the reference base and
    post-contingency flows times per-line slack, and the bulk (345/500 kV,
    non-radial) network carries one system-wide stress factor in [0.65, 1.05];
    whether a preventive dispatch exists is undetermined.

# Fields

  - network and fleet fields as in `DCOptimalPowerFlowProblem`, plus
    `emergency_limit::Vector{Float64}` and `contingencies::Vector{Int}` (the
    outaged line of each contingency)
  - `feasibility_status`, `feasible_witness`, `infeasibility_certificate`
"""
struct SecurityConstrainedDCOPFProblem <: ProblemGenerator
    n_buses::Int
    n_lines::Int
    n_generators::Int
    line_from::Vector{Int}
    line_to::Vector{Int}
    susceptance::Vector{Float64}
    line_limit::Vector{Float64}
    emergency_limit::Vector{Float64}
    gen_bus::Vector{Int}
    gen_tech::Vector{Symbol}
    gen_cost::Vector{Float64}
    pmin::Vector{Float64}
    pmax::Vector{Float64}
    demand::Vector{Float64}
    ref_bus::Int
    angle_limit::Float64
    contingencies::Vector{Int}
    feasibility_status::FeasibilityStatus
    feasible_witness::Union{Nothing, SCOPFWitness}
    infeasibility_certificate::Union{Nothing, DCPocketCertificate}
end

"""Number of screened contingencies for a target size."""
_scopf_contingency_count(target::Int) = clamp(round(Int, sqrt(max(target, 1)) / 8), 1, 60)

"""
    SecurityConstrainedDCOPFProblem(target_variables, feasibility_status, seed)

Build a preventive N-1 SCOPF instance with `n_generators + (1 + C)·n_buses`
columns matching `target_variables` (exactly, for targets ≥ 20 when enough
non-bridge lines exist). See the type docstring.
"""
function SecurityConstrainedDCOPFProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    target = max(target_variables, 20)
    C = _scopf_contingency_count(target)
    ef = _e_unif(rng, (1.35, 1.60))
    gf = _e_unif(rng, (0.25, 0.40))
    B = max(4, round(Int, target / (1 + C + gf)))
    G = max(2, target - (1 + C) * B)
    L_target = min(round(Int, ef * B), B * (B - 1) ÷ 2)
    net = _dc_base_network(rng, B, L_target, G)
    from, to, sus = net.from, net.to, net.sus
    L = length(from)
    fl = net.fleet

    degree = zeros(Int, B)
    for l in 1:L
        degree[from[l]] += 1
        degree[to[l]] += 1
    end
    ref = argmax(degree)
    sum_min, sum_max = sum(fl.pmin), sum(fl.pmax)
    frac = feasibility_status == unknown ? _e_unif(rng, (0.45, 0.85)) : _e_unif(rng, (0.35, 0.70))
    total = sum_min + frac * (sum_max - sum_min)
    demand = total .* net.shares

    # Screening: reference dispatch, base flows, the C most loaded non-bridges.
    p_ref = _dc_proportional_dispatch(fl.pmin, fl.pmax, total)
    inj = _dc_injection(B, fl.gen_bus, p_ref, demand)
    θ0, f0 = _dc_flows(B, from, to, sus, inj, ref)
    bridges = _graph_bridges(B, from, to)
    candidates = [l for l in 1:L if !bridges[l]]
    sort!(candidates; by=l -> -abs(f0[l]) / net.rating[l])
    contingencies = candidates[1:min(C, length(candidates))]
    if length(contingencies) < C
        # Too few non-bridges (tiny grids): absorb the missing states into G.
        G = max(2, target - (1 + length(contingencies)) * B)
        fl = _energy_grid_fleet(rng, B, G; hour=net.hour)
        sum_min, sum_max = sum(fl.pmin), sum(fl.pmax)
        total = sum_min + frac * (sum_max - sum_min)
        demand = total .* net.shares
        p_ref = _dc_proportional_dispatch(fl.pmin, fl.pmax, total)
        inj = _dc_injection(B, fl.gen_bus, p_ref, demand)
        θ0, f0 = _dc_flows(B, from, to, sus, inj, ref)
    end
    C = length(contingencies)
    angles = zeros(B, 1 + C)
    angles[:, 1] .= θ0
    worst = zeros(L)   # worst post-contingency |flow| per line
    for (c, k) in enumerate(contingencies)
        θc, fc = _dc_flows(B, from, to, sus, inj, ref; outage=k)
        angles[:, 1 + c] .= θc
        for l in 1:L
            l == k && continue
            worst[l] = max(worst[l], abs(fc[l]))
        end
    end
    kappa = [_e_unif(rng, (1.10, 1.30)) for _ in 1:L]

    rating = copy(net.rating)
    witness = nothing
    certificate = nothing
    if feasibility_status == unknown
        stress = _e_unif(rng, (0.65, 1.05))
        for l in 1:L
            slack = 1.0 + exp(log(0.25) + 0.6 * randn(rng))
            planned = max(abs(f0[l]), worst[l] / kappa[l])
            bulk = net.voltage[l] != :lv && !bridges[l]
            rating[l] = max(0.5 * rating[l], planned * (bulk ? stress : max(stress, 1.0)) * slack)
        end
    else
        for l in 1:L
            rating[l] = max(rating[l], 1.15 * abs(f0[l]) + 1.0, 1.10 * worst[l] / kappa[l] + 1.0)
        end
        witness = SCOPFWitness(p_ref, angles)
    end
    emergency = kappa .* rating
    _scopf_bus_adequacy!(B, from, to, rating, emergency, kappa, fl, demand, contingencies)
    angle_limit = _dc_angle_limit(rng, angles)

    if feasibility_status == infeasible
        witness = nothing
        m = _e_unif(rng, (0.06, 0.15))
        # Random pockets first; then (small grids, where C = 1 and the random
        # draws only ever produce the two 3-bus pockets beside the one screened
        # line) a deterministic sweep over every contingency, side and size.
        random_tries = [
            (
                c=rand(rng, 1:C),
                side=rand(rng, Bool),
                size=round(Int, B * _e_unif(rng, (0.02, 0.06))),
            ) for _ in 1:30
        ]
        sweep = (
            (c=c, side=side, size=size) for c in 1:C for size in 3:(B - 1) for side in (true, false)
        )
        for try_ in Iterators.flatten((random_tries, sweep))
            c = try_.c
            k = contingencies[c]
            a, b = try_.side ? (from[k], to[k]) : (to[k], from[k])
            size = clamp(try_.size, min(3, B - 1), B - 1)
            S = _grid_bfs(B, from, to, a, size; exclude=b)
            length(S) >= 2 || continue
            inS = falses(B)
            inS[S] .= true
            cut = _grid_cut(B, from, to, S)
            (k in cut && length(cut) >= 2) || continue
            # Every state must keep each pocket bus row satisfiable on its own:
            # the certified contingency, the base case and every other
            # screened outage.
            states = Any[(emergency, k), (rating, 0)]
            for k2 in contingencies
                k2 != k && push!(states, (emergency, k2))
            end
            pk = _dc_plant_pocket!(rng, B, from, to, fl.gen_bus, fl.pmax, demand, S, states; margin=m)
            pk === nothing && continue
            # Keep emergency ratings above normal on the strengthened lines.
            emergency .= max.(emergency, kappa .* rating)
            # The lost line is the pocket's main feeder: in the base case the
            # normal ratings of all feeders cover the pocket with 5 % to spare.
            local_cap = pk.local_cap
            base_cap = local_cap + sum(rating[l] for l in cut)
            if base_cap < 1.05 * pk.need
                rating[k] += 1.05 * pk.need - base_cap
                emergency[k] = kappa[k] * rating[k]
            end
            certificate = DCPocketCertificate(pk.S, pk.cut, c, k, local_cap, pk.imports, sum(demand[pk.S]))
            break
        end
        certificate === nothing && error("energy/security_constrained_dc_opf: could not plant an N-1 pocket")
    end

    return SecurityConstrainedDCOPFProblem(
        B,
        L,
        length(fl.pmax),
        from,
        to,
        sus,
        rating,
        emergency,
        fl.gen_bus,
        fl.tech,
        fl.cost,
        fl.pmin,
        fl.pmax,
        demand,
        ref,
        angle_limit,
        contingencies,
        feasibility_status,
        witness,
        certificate,
    )
end

"""
Every bus is adequately connected in the base case and after each screened
outage: its live incident ratings cover 110 % of both its load beyond local
generation and its must-run output beyond local load (ratings are raised
proportionally, emergency ratings follow).
"""
function _scopf_bus_adequacy!(B, from, to, rating, emergency, kappa, fl, demand, contingencies)
    gen_cap = zeros(B)
    gen_min = zeros(B)
    for g in eachindex(fl.pmax)
        gen_cap[fl.gen_bus[g]] += fl.pmax[g]
        gen_min[fl.gen_bus[g]] += fl.pmin[g]
    end
    incident = [Int[] for _ in 1:B]
    for l in eachindex(from)
        push!(incident[from[l]], l)
        push!(incident[to[l]], l)
    end
    outaged = Set(contingencies)
    for b in 1:B
        isempty(incident[b]) && continue
        need = 1.1 * max(demand[b] - gen_cap[b], gen_min[b] - demand[b], 0.0)
        need > 0 || continue
        # Worst state: the base case, or losing the strongest incident screened line.
        lost = [l for l in incident[b] if l in outaged]
        worst_lines = sum(rating[l] for l in incident[b])
        if !isempty(lost)
            k = lost[argmax([emergency[l] for l in lost])]
            worst_lines = min(worst_lines, sum((emergency[l] for l in incident[b] if l != k); init=0.0))
        end
        if worst_lines < need
            if worst_lines <= 0
                continue
            end
            scale = need / worst_lines
            for l in incident[b]
                rating[l] *= scale
                emergency[l] = kappa[l] * rating[l]
            end
        end
    end
    return nothing
end

"""
    build_model(prob::SecurityConstrainedDCOPFProblem)

Shared dispatch columns, the base-case network block with normal ratings and
one post-contingency block per screened outage with emergency ratings;
minimize generation cost.
"""
function build_model(prob::SecurityConstrainedDCOPFProblem)
    model = Model()
    G = prob.n_generators
    @variable(model, prob.pmin[g] <= p[g=1:G] <= prob.pmax[g])
    args = (prob.n_buses, prob.line_from, prob.line_to, prob.susceptance)
    # model[:theta][1] holds the base-case angles, model[:theta][1 + c] those
    # after contingency c.
    θ = [
        _dc_network_block!(
            model, args..., prob.line_limit, prob.gen_bus, prob.demand, prob.ref_bus, p, prob.angle_limit;
            tag="_base",
        ),
    ]
    for (c, k) in enumerate(prob.contingencies)
        push!(
            θ,
            _dc_network_block!(
                model, args..., prob.emergency_limit, prob.gen_bus, prob.demand, prob.ref_bus, p,
                prob.angle_limit; outage=k, tag="_c$c",
            ),
        )
    end
    model[:theta] = θ
    @objective(model, Min, sum(prob.gen_cost[g] * p[g] for g in 1:G))
    return model
end

register_variant(
    :energy,
    :security_constrained_dc_opf,
    SecurityConstrainedDCOPFProblem,
    "Preventive N-1 security-constrained DC-OPF: a shared dispatch with a base-case and one post-contingency B-θ network block per screened line outage (emergency ratings), block-angular";
    tags=[:energy, :dual_block_angular],
)
