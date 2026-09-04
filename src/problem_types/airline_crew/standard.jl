using JuMP
using Random
using Distributions

"""
    CrewPairingRules

Legality rules every generated pairing satisfies by construction. Times are in
minutes.

A *pairing* is a sequence of flight legs a crew flies, starting and ending at
the crew's home base. Legs are grouped into *duty periods* (working days):
consecutive legs belong to the same duty when the ground time between them is
at most `max_sit`; a longer gap is an overnight *rest*. Because
`max_sit < min_rest` the grouping induced by the schedule is unambiguous, so a
pairing's duty structure can be recovered from its leg times alone.

# Fields

  - `min_connect::Int`: minimum sit (connection) time between two legs of a duty
  - `max_sit::Int`: maximum sit time inside a duty (longer ground time = rest)
  - `max_legs_per_duty::Int`: maximum number of legs in one duty period
  - `max_duty_minutes::Int`: maximum elapsed duty time (first departure to last
    arrival of the duty)
  - `max_block_minutes::Int`: maximum flight (block) time flown in one duty
  - `min_rest::Int`: minimum rest between two consecutive duties of a pairing
  - `max_rest::Int`: maximum rest between two consecutive duties
  - `max_duties::Int`: maximum number of duty periods in a pairing (trip length)
"""
struct CrewPairingRules
    min_connect::Int
    max_sit::Int
    max_legs_per_duty::Int
    max_duty_minutes::Int
    max_block_minutes::Int
    min_rest::Int
    max_rest::Int
    max_duties::Int
end

"""
    CrewPairingCoverWitness

Planted exact cover: `pairings` indexes a subset of the generated columns whose
flight sets partition every flight in the schedule. Setting those `x_p` to one
and every other `x_p` to zero satisfies all covering equalities, so the
set-partitioning model (and therefore its LP relaxation) is feasible.
"""
struct CrewPairingCoverWitness
    pairings::Vector{Int}
end

"""
    UncoverableFlightCertificate

Infeasibility certificate: flight `flight` is contained in *no* legal pairing,
so its covering equality reduces to `0 == 1` in the model and in the LP
relaxation.

The argument is structural rather than enumerative. The flight departs from
`origin`, which is not a crew base, so it cannot be the first leg of a pairing;
and `predecessors == 0` flights arrive at `origin` early enough to connect to it
(within either the sit window or the rest window), so it cannot follow another
leg either. Hence it appears in no pairing at all.
"""
struct UncoverableFlightCertificate
    flight::Int
    origin::Int
    destination::Int
    predecessors::Int
end

"""
    AirlineCrewProblem <: ProblemGenerator

Generator for airline crew pairing set-partitioning instances built from
*operationally legal* pairings.

# Overview

A dated flight schedule is generated over a hub-and-spoke airport network, then
crew pairings are built as time-and-airport-respecting walks through that
schedule. Every generated pairing satisfies airport continuity, connection
times, base return, duty limits, and rest rules (see [`CrewPairingRules`]) *by
construction*: pairings are grown leg by leg under the rules and no downstream
step ever edits a pairing's leg set - filtering only ever drops whole columns.

The model is the classical crew pairing set-partitioning problem: choose a
minimum-cost subset of pairings covering every flight exactly once.

# Schedule construction

The schedule itself is produced by *planting* lines of flying: each planted line
is a legal pairing whose legs are created as it is flown (base -> ... -> base,
with sit times, duty limits and overnight rests). The planted lines therefore
partition the flight set, which both guarantees a realistic connection structure
and provides the feasible witness. Additional columns are sampled as legal walks
over the resulting schedule, so they freely mix legs from different lines.

# Cost

Pairing cost follows standard airline crew pay: a *credit* equal to the largest
of block time, a minimum-duty-guarantee fraction of elapsed duty time, and a
minimum daily credit per duty, paid at `pay_rate`, plus per-diem over the whole
time away from base and a hotel cost per overnight.

# Fields

  - `num_flights::Int`: number of flight legs in the schedule
  - `num_airports::Int`: number of airports
  - `bases::Vector{Int}`: crew base airports (a prefix of `1:num_airports`)
  - `airport_locations::Vector{Tuple{Float64,Float64}}`: airport coordinates (km)
  - `block_minutes::Matrix{Int}`: scheduled flight time between airport pairs
  - `flight_origins::Vector{Int}`: origin airport of each flight
  - `flight_destinations::Vector{Int}`: destination airport of each flight
  - `departure_times::Vector{Int}`: departure time (minutes from horizon start)
  - `arrival_times::Vector{Int}`: arrival time (minutes from horizon start)
  - `rules::CrewPairingRules`: duty/rest legality rules
  - `pairing_costs::Vector{Float64}`: cost of each pairing column
  - `flights_in_pairing::Vector{Vector{Int}}`: legs of each pairing, in order
  - `pairing_bases::Vector{Int}`: home base of each pairing
  - `pay_rate::Float64`: crew pay per credit hour
  - `duty_guarantee::Float64`: minimum-duty-guarantee fraction (credit per duty hour)
  - `min_daily_credit::Int`: minimum credited minutes per duty period
  - `per_diem_rate::Float64`: per-diem paid per hour away from base
  - `hotel_cost::Float64`: hotel cost per overnight
  - `feasible_witness::Union{Nothing,CrewPairingCoverWitness}`: planted partition
  - `infeasibility_certificate::Union{Nothing,UncoverableFlightCertificate}`
  - `feasibility_status::FeasibilityStatus`
"""
struct AirlineCrewProblem <: ProblemGenerator
    num_flights::Int
    num_airports::Int
    bases::Vector{Int}
    airport_locations::Vector{Tuple{Float64, Float64}}
    block_minutes::Matrix{Int}
    flight_origins::Vector{Int}
    flight_destinations::Vector{Int}
    departure_times::Vector{Int}
    arrival_times::Vector{Int}
    rules::CrewPairingRules
    pairing_costs::Vector{Float64}
    flights_in_pairing::Vector{Vector{Int}}
    pairing_bases::Vector{Int}
    pay_rate::Float64
    duty_guarantee::Float64
    min_daily_credit::Int
    per_diem_rate::Float64
    hotel_cost::Float64
    feasible_witness::Union{Nothing, CrewPairingCoverWitness}
    infeasibility_certificate::Union{Nothing, UncoverableFlightCertificate}
    feasibility_status::FeasibilityStatus
end

# ---------------------------------------------------------------------------
# Flight network container (construction-time only)
# ---------------------------------------------------------------------------

"""
Mutable flight schedule with a departure-time index per origin airport, used
while growing the schedule and sampling pairings. `by_origin[a]` lists the
flights leaving `a` sorted by departure time, and `by_origin_dep[a]` holds the
matching departure times so connection windows are binary searchable.
"""
mutable struct _CrewNet
    org::Vector{Int}
    dst::Vector{Int}
    dep::Vector{Int}
    arr::Vector{Int}
    by_origin::Vector{Vector{Int}}
    by_origin_dep::Vector{Vector{Int}}
    rules::CrewPairingRules
end

_crew_net(num_airports::Int, rules::CrewPairingRules) = _CrewNet(
    Int[],
    Int[],
    Int[],
    Int[],
    [Int[] for _ in 1:num_airports],
    [Int[] for _ in 1:num_airports],
    rules,
)

"""
    _crew_add_flight!(net, o, d, dep, arr) -> Int

Append a flight and keep the per-origin departure index sorted. Returns the new
flight id.
"""
function _crew_add_flight!(net::_CrewNet, o::Int, d::Int, dep::Int, arr::Int)
    push!(net.org, o)
    push!(net.dst, d)
    push!(net.dep, dep)
    push!(net.arr, arr)
    id = length(net.org)
    pos = searchsortedfirst(net.by_origin_dep[o], dep)
    insert!(net.by_origin_dep[o], pos, dep)
    insert!(net.by_origin[o], pos, id)
    return id
end

"""
    _crew_successors(net, f, lo, hi)

Flights that may follow leg `f` with a ground time in `[lo, hi]`: they depart
from `f`'s arrival airport inside the corresponding departure-time window.
"""
function _crew_successors(net::_CrewNet, f::Int, lo::Int, hi::Int)
    a = net.dst[f]
    t = net.arr[f]
    times = net.by_origin_dep[a]
    i = searchsortedfirst(times, t + lo)
    j = searchsortedlast(times, t + hi)
    return view(net.by_origin[a], i:j)
end

# ---------------------------------------------------------------------------
# Legality checking (shared by construction, cost accounting and tests)
# ---------------------------------------------------------------------------

"""
    _crew_duty_ranges(dep, arr, legs, max_sit) -> Vector{UnitRange{Int}}

Split a leg sequence into duty periods. A ground time above `max_sit` ends the
duty; anything shorter is an in-duty connection. Indices are positions inside
`legs`.
"""
function _crew_duty_ranges(dep::Vector{Int}, arr::Vector{Int}, legs::Vector{Int}, max_sit::Int)
    ranges = UnitRange{Int}[]
    start = 1
    for i in 1:(length(legs) - 1)
        if dep[legs[i + 1]] - arr[legs[i]] > max_sit
            push!(ranges, start:i)
            start = i + 1
        end
    end
    push!(ranges, start:length(legs))
    return ranges
end

"""
    _crew_violations(org, dst, dep, arr, legs, base, bases, rules) -> Vector{Symbol}

Re-derive a pairing's legality from the raw schedule data. Returns the violated
properties, so an empty result certifies the pairing is flyable:

  - `:base_return` - the pairing does not start and end at its (base) airport
  - `:continuity` - some leg does not depart where the previous leg arrived
  - `:time` - a leg arrives before it departs, or a connection is shorter than
    `min_connect`, or the legs are not in strictly increasing time order
  - `:duty` - a duty period exceeds `max_legs_per_duty`, `max_block_minutes` or
    `max_duty_minutes`, or the pairing exceeds `max_duties`
  - `:rest` - a duty break is shorter than `min_rest` or longer than `max_rest`
"""
function _crew_violations(
    org::Vector{Int},
    dst::Vector{Int},
    dep::Vector{Int},
    arr::Vector{Int},
    legs::Vector{Int},
    base::Int,
    bases::Vector{Int},
    rules::CrewPairingRules,
)
    bad = Symbol[]
    if isempty(legs)
        push!(bad, :base_return)
        return bad
    end
    if !(base in bases) || org[legs[1]] != base || dst[legs[end]] != base
        push!(bad, :base_return)
    end
    if length(unique(legs)) != length(legs)
        push!(bad, :time)
    end

    continuity_ok = true
    time_ok = all(arr[f] > dep[f] for f in legs)
    rest_ok = true
    for i in 1:(length(legs) - 1)
        f, g = legs[i], legs[i + 1]
        dst[f] == org[g] || (continuity_ok = false)
        gap = dep[g] - arr[f]
        gap < rules.min_connect && (time_ok = false)
        if gap > rules.max_sit && !(rules.min_rest <= gap <= rules.max_rest)
            rest_ok = false
        end
    end
    continuity_ok || push!(bad, :continuity)
    time_ok || (:time in bad || push!(bad, :time))
    rest_ok || push!(bad, :rest)

    duties = _crew_duty_ranges(dep, arr, legs, rules.max_sit)
    duty_ok = length(duties) <= rules.max_duties
    for r in duties
        length(r) <= rules.max_legs_per_duty || (duty_ok = false)
        block = sum(arr[legs[i]] - dep[legs[i]] for i in r)
        block <= rules.max_block_minutes || (duty_ok = false)
        elapsed = arr[legs[last(r)]] - dep[legs[first(r)]]
        elapsed <= rules.max_duty_minutes || (duty_ok = false)
    end
    duty_ok || push!(bad, :duty)
    return bad
end

"""
    _crew_violations(prob::AirlineCrewProblem, p::Int) -> Vector{Symbol}

Legality violations of pairing `p`, re-derived from the problem's own schedule
data (see the low-level method for the property list).
"""
_crew_violations(prob::AirlineCrewProblem, p::Int) = _crew_violations(
    prob.flight_origins,
    prob.flight_destinations,
    prob.departure_times,
    prob.arrival_times,
    prob.flights_in_pairing[p],
    prob.pairing_bases[p],
    prob.bases,
    prob.rules,
)

# ---------------------------------------------------------------------------
# Cost accounting
# ---------------------------------------------------------------------------

"""
    _crew_pairing_cost(dep, arr, legs, rules, pay_rate, guarantee, min_daily,
                       per_diem, hotel)

Standard airline crew pairing cost: pay the largest of block time, a
minimum-duty-guarantee fraction of elapsed duty time, and a minimum daily credit
per duty; add per-diem over the time away from base and a hotel night per
overnight rest. Deterministic given the pairing's schedule.
"""
function _crew_pairing_cost(
    dep::Vector{Int},
    arr::Vector{Int},
    legs::Vector{Int},
    rules::CrewPairingRules,
    pay_rate::Float64,
    guarantee::Float64,
    min_daily::Int,
    per_diem::Float64,
    hotel::Float64,
)
    duties = _crew_duty_ranges(dep, arr, legs, rules.max_sit)
    block = sum(arr[f] - dep[f] for f in legs)
    duty_time = sum(arr[legs[last(r)]] - dep[legs[first(r)]] for r in duties)
    tafb = arr[legs[end]] - dep[legs[1]]
    credit = max(float(block), guarantee * duty_time, float(min_daily * length(duties)))
    return pay_rate * credit / 60 + per_diem * tafb / 60 + hotel * (length(duties) - 1)
end

"""
    _crew_pairing_cost(prob::AirlineCrewProblem, legs)

Cost of an arbitrary leg sequence under the instance's pay parameters.
"""
_crew_pairing_cost(prob::AirlineCrewProblem, legs::Vector{Int}) = _crew_pairing_cost(
    prob.departure_times,
    prob.arrival_times,
    legs,
    prob.rules,
    prob.pay_rate,
    prob.duty_guarantee,
    prob.min_daily_credit,
    prob.per_diem_rate,
    prob.hotel_cost,
)

# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------

"""
Weighted choice without pulling in StatsBase; `weights` must be positive.
"""
function _crew_wsample(rng::AbstractRNG, items::Vector{Int}, weights::Vector{Float64})
    total = sum(weights)
    r = rand(rng) * total
    acc = 0.0
    for (i, w) in enumerate(weights)
        acc += w
        acc >= r && return items[i]
    end
    return items[end]
end

"""
    _crew_geography(rng, num_airports, num_bases)

Hub-and-spoke airport map: bases spread around the centre of a continental-scale
box, spokes scattered around them. Returns locations and the block-time matrix
(35 min taxi/climb overhead plus cruise at ~720 km/h, rounded to 5 minutes and
clamped to a narrowbody 45-240 minute range).
"""
function _crew_geography(rng::AbstractRNG, num_airports::Int, num_bases::Int)
    width, height = 2000.0, 1400.0
    locations = Tuple{Float64, Float64}[]
    angle0 = rand(rng) * 2pi
    for b in 1:num_bases
        theta = angle0 + 2pi * (b - 1) / num_bases
        push!(
            locations,
            (
                width / 2 + 0.30 * width * cos(theta) + 40 * (rand(rng) - 0.5),
                height / 2 + 0.30 * height * sin(theta) + 40 * (rand(rng) - 0.5),
            ),
        )
    end
    for _ in (num_bases + 1):num_airports
        push!(locations, (rand(rng) * width, rand(rng) * height))
    end
    block = zeros(Int, num_airports, num_airports)
    for i in 1:num_airports, j in 1:num_airports
        i == j && continue
        d = hypot(locations[i][1] - locations[j][1], locations[i][2] - locations[j][2])
        block[i, j] = clamp(5 * round(Int, (35 + d / 12.0) / 5), 45, 240)
    end
    return locations, block
end

"""
    _crew_rules(rng)

Sample duty and rest rules in ranges typical of a domestic narrowbody
operation. `max_sit < min_rest` always holds, so the duty structure of a pairing
is uniquely determined by its leg times; and `max_block_minutes >= 2 * 240`
guarantees a two-leg out-and-back always fits inside one duty, which the
schedule builder relies on as a fallback.
"""
function _crew_rules(rng::AbstractRNG)
    return CrewPairingRules(
        rand(rng, 30:5:45),          # min_connect
        rand(rng, 180:30:300),       # max_sit
        rand(rng, 3:5),              # max_legs_per_duty
        rand(rng, 690:30:840),       # max_duty_minutes
        rand(rng, 480:30:570),       # max_block_minutes
        rand(rng, 600:30:660),       # min_rest
        rand(rng, 960:60:1200),      # max_rest
        rand(rng, 2:4),              # max_duties
    )
end

"""
    _crew_next_airport(rng, block, cur, base, num_bases, num_airports, dep_t,
                       duty_start, duty_block, rules, reserve)

Pick the next airport for a planted leg. Hub-and-spoke bias: from a spoke the
crew usually flies back to a hub, from a hub usually out to a spoke. Candidates
must keep the duty inside its block and elapsed limits; when `reserve` is set
(the final duty of the line) they must additionally leave room to fly home to
`base` afterwards. Returns `0` when nothing fits.
"""
function _crew_next_airport(
    rng::AbstractRNG,
    block::Matrix{Int},
    cur::Int,
    base::Int,
    num_bases::Int,
    num_airports::Int,
    dep_t::Int,
    duty_start::Int,
    duty_block::Int,
    rules::CrewPairingRules,
    reserve::Bool,
)
    cands = Int[]
    weights = Float64[]
    cur_is_base = cur <= num_bases
    for a in 1:num_airports
        a == cur && continue
        ft = block[cur, a]
        duty_block + ft <= rules.max_block_minutes || continue
        (dep_t + ft) - duty_start <= rules.max_duty_minutes || continue
        if reserve && a != base
            home = block[a, base]
            duty_block + ft + home <= rules.max_block_minutes || continue
            (dep_t + ft + rules.min_connect + home) - duty_start <= rules.max_duty_minutes ||
                continue
        end
        push!(cands, a)
        a_is_base = a <= num_bases
        push!(weights, cur_is_base ? (a_is_base ? 1.0 : 4.0) : (a_is_base ? 5.0 : 1.5))
    end
    isempty(cands) && return 0
    return _crew_wsample(rng, cands, weights)
end

"""
    _crew_plant_line!(rng, net, block, bases, num_airports, n_days, waves)

Fly one new line: a legal pairing whose legs are *created* as it goes, from a
randomly chosen base back to that base, across 1..`max_duties` duty periods
separated by legal rests. The leg sequence is buffered and checked against
[`_crew_violations`] before it is committed to `net`; on repeated failure a
two-leg out-and-back (always legal under the sampled rules) is committed
instead. Returns `(base, legs)`.
"""
function _crew_plant_line!(
    rng::AbstractRNG,
    net::_CrewNet,
    block::Matrix{Int},
    bases::Vector{Int},
    num_airports::Int,
    n_days::Int,
    waves::Vector{Int},
)
    rules = net.rules
    num_bases = length(bases)
    base = rand(rng, bases)

    for attempt in 1:8
        buffer = NTuple{4, Int}[]        # (origin, destination, dep, arr)
        n_duties = rand(rng, 1:rules.max_duties)
        t = (rand(rng, 1:n_days) - 1) * 1440 + rand(rng, waves) + 5 * rand(rng, 0:5)
        cur = base
        for d in 1:n_duties
            is_final = d == n_duties
            planned = rand(rng, 1:rules.max_legs_per_duty)
            (is_final && cur == base) && (planned = max(planned, 2))
            duty_start = t
            duty_block = 0
            duty_legs = 0
            t_arr = t
            for l in 1:planned
                dep_t = if duty_legs == 0
                    duty_start
                else
                    t_arr + rand(rng, (rules.min_connect ÷ 5):(rules.max_sit ÷ 5)) * 5
                end
                forced_home = is_final && l == planned
                nxt = 0
                if !forced_home
                    nxt = _crew_next_airport(
                        rng,
                        block,
                        cur,
                        base,
                        num_bases,
                        num_airports,
                        dep_t,
                        duty_start,
                        duty_block,
                        rules,
                        is_final,
                    )
                    nxt == 0 && (forced_home = true)
                end
                if forced_home
                    cur == base && break
                    nxt = base
                    # The reserve check on the previous leg guarantees the way
                    # home fits with a minimum connection; take it exactly.
                    dep_t = duty_legs == 0 ? duty_start : t_arr + rules.min_connect
                end
                ft = block[cur, nxt]
                arr_t = dep_t + ft
                if duty_block + ft > rules.max_block_minutes ||
                    arr_t - duty_start > rules.max_duty_minutes
                    break
                end
                push!(buffer, (cur, nxt, dep_t, arr_t))
                duty_legs += 1
                duty_block += ft
                cur = nxt
                t_arr = arr_t
                forced_home && break
            end
            duty_legs == 0 && break
            is_final || (t = t_arr + 5 * rand(rng, (rules.min_rest ÷ 5):(rules.max_rest ÷ 5)))
        end

        if !isempty(buffer) && buffer[1][1] == base && buffer[end][2] == base
            legs = [_crew_add_flight!(net, o, d, dp, ar) for (o, d, dp, ar) in buffer]
            if isempty(
                _crew_violations(net.org, net.dst, net.dep, net.arr, legs, base, bases, rules)
            )
                return base, legs
            end
            # Never commit an illegal line: retract its legs and retry.
            for _ in legs
                _crew_pop_flight!(net)
            end
        end
    end

    # Guaranteed-legal fallback: nearest spoke, out and straight back.
    spoke = argmin([a in bases ? typemax(Int) : block[base, a] for a in 1:num_airports])
    day = (rand(rng, 1:n_days) - 1) * 1440 + rand(rng, waves)
    ft1 = block[base, spoke]
    ft2 = block[spoke, base]
    f1 = _crew_add_flight!(net, base, spoke, day, day + ft1)
    dep2 = day + ft1 + rules.min_connect
    f2 = _crew_add_flight!(net, spoke, base, dep2, dep2 + ft2)
    return base, [f1, f2]
end

"""
Undo the most recent `_crew_add_flight!` (used to retract a rejected line).
"""
function _crew_pop_flight!(net::_CrewNet)
    id = length(net.org)
    o = net.org[id]
    pos = findfirst(==(id), net.by_origin[o])
    deleteat!(net.by_origin[o], pos)
    deleteat!(net.by_origin_dep[o], pos)
    pop!(net.org)
    pop!(net.dst)
    pop!(net.dep)
    pop!(net.arr)
    return nothing
end

"""
    _crew_extend!(rng, net, base, legs, used, duty_start, duty_block, duty_legs,
                  n_duties, budget, stop_prob) -> Bool

Randomised depth-first search over legal continuations. Every option already
respects airport continuity (successors depart where the previous leg lands),
the connection or rest window, the per-duty leg/block/elapsed limits and the
maximum number of duties, so any accepted walk is a legal pairing. The walk is
accepted only while standing at `base`, which enforces base return. `budget`
bounds the number of expansions so sampling stays cheap.
"""
function _crew_extend!(
    rng::AbstractRNG,
    net::_CrewNet,
    base::Int,
    legs::Vector{Int},
    used::Set{Int},
    duty_start::Int,
    duty_block::Int,
    duty_legs::Int,
    n_duties::Int,
    budget::Base.RefValue{Int},
    stop_prob::Float64,
)
    r = net.rules
    max_total = r.max_legs_per_duty * r.max_duties
    at_base = net.dst[legs[end]] == base && length(legs) >= 2
    if at_base && (length(legs) >= max_total || rand(rng) < stop_prob)
        return true
    end
    (budget[] <= 0 || length(legs) >= max_total) && return at_base

    options = Tuple{Int, Bool}[]
    if duty_legs < r.max_legs_per_duty
        for g in _crew_successors(net, legs[end], r.min_connect, r.max_sit)
            g in used && continue
            b = net.arr[g] - net.dep[g]
            duty_block + b <= r.max_block_minutes || continue
            net.arr[g] - duty_start <= r.max_duty_minutes || continue
            push!(options, (g, false))
        end
    end
    if n_duties < r.max_duties
        for g in _crew_successors(net, legs[end], r.min_rest, r.max_rest)
            g in used && continue
            push!(options, (g, true))
        end
    end
    shuffle!(rng, options)

    for (g, new_duty) in options
        budget[] -= 1
        budget[] <= 0 && break
        push!(legs, g)
        push!(used, g)
        b = net.arr[g] - net.dep[g]
        ok = if new_duty
            _crew_extend!(rng, net, base, legs, used, net.dep[g], b, 1, n_duties + 1, budget, stop_prob)
        else
            _crew_extend!(
                rng,
                net,
                base,
                legs,
                used,
                duty_start,
                duty_block + b,
                duty_legs + 1,
                n_duties,
                budget,
                stop_prob,
            )
        end
        ok && return true
        pop!(legs)
        delete!(used, g)
    end
    return at_base
end

"""
    _crew_sample_pairing(rng, net, base_starts) -> (base, legs)

Sample one legal pairing from the existing schedule by walking it from a
base departure. Returns an empty leg vector when the search budget runs out
without getting the crew home.
"""
function _crew_sample_pairing(rng::AbstractRNG, net::_CrewNet, base_starts::Vector{Int})
    isempty(base_starts) && return 0, Int[]
    start = rand(rng, base_starts)
    base = net.org[start]
    legs = Int[start]
    used = Set{Int}(legs)
    budget = Ref(400)
    block = net.arr[start] - net.dep[start]
    ok = _crew_extend!(
        rng, net, base, legs, used, net.dep[start], block, 1, 1, budget, 0.15 + 0.45 * rand(rng)
    )
    return ok ? (base, legs) : (base, Int[])
end

# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

"""
    AirlineCrewProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a crew pairing instance whose columns are all operationally legal
pairings.

# Sizing

One binary variable per pairing column, and the generator emits exactly
`target_variables` columns: it keeps sampling legal pairings until the target is
met, and whenever the sampler stalls it grows the schedule with another planted
line (which is itself a new column), so the loop always makes progress. The
schedule holds roughly `0.55 * target_variables` flights, one covering equality
each.

# Feasibility

  - `feasible`: every planted line is kept as a column, so those columns partition
    the flight set - an integral exact cover recorded in `feasible_witness`.
  - `infeasible`: the same construction plus one extra flight departing from a
    non-base airport, scheduled beyond every other arrival so that nothing can
    connect into it. No legal pairing can contain it (it can neither open a
    pairing nor follow another leg), so its covering row is `0 == 1`
    (`infeasibility_certificate`). Every other flight is still covered, making the
    infeasibility minimal and structural.
  - `unknown`: a three-way mix - the planted partition is kept intact (feasible),
    only a random subset of the planted lines is kept as columns (genuinely
    undecided: the surviving columns may or may not still admit an exact cover),
    or an uncoverable flight is planted (infeasible). Metadata is status-specific,
    so `unknown` instances carry neither a witness nor a certificate.
"""
function AirlineCrewProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    rng = MersenneTwister(seed)

    target = max(target_variables, 4)
    flights_target = clamp(round(Int, 0.55 * target), 12, 200_000)
    num_airports = clamp(round(Int, 5 + flights_target / 7), 6, 60)
    num_bases = clamp(round(Int, num_airports / 6), 2, 10)
    n_days = clamp(round(Int, flights_target / (2.5 * num_airports)), 2, 12)
    waves = collect(360:120:1080)

    rules = _crew_rules(rng)
    locations, block = _crew_geography(rng, num_airports, num_bases)
    bases = collect(1:num_bases)

    pay_rate = 180.0 + 140.0 * rand(rng)
    duty_guarantee = 0.50 + 0.10 * rand(rng)
    min_daily_credit = rand(rng, 240:15:315)
    per_diem_rate = 2.0 + 1.5 * rand(rng)
    hotel_cost = 90.0 + 70.0 * rand(rng)

    # `unknown` is a genuine three-way mix: the planted partition survives
    # intact (feasible), only part of it is kept as columns (undecided), or an
    # uncoverable flight is planted (infeasible).
    roll = feasibility_status == unknown ? rand(rng) : 0.0
    drop_planted = feasibility_status == unknown && 0.4 <= roll < 0.7
    plant_orphan =
        feasibility_status == infeasible || (feasibility_status == unknown && roll >= 0.7)
    keep_fraction = drop_planted ? 0.10 + 0.50 * rand(rng) : 1.0

    net = _crew_net(num_airports, rules)
    columns = Vector{Int}[]
    column_bases = Int[]
    seen = Set{Vector{Int}}()
    planted_columns = Int[]
    all_planted_kept = true

    function push_column!(base::Int, legs::Vector{Int})
        legs in seen && return 0
        push!(columns, legs)
        push!(column_bases, base)
        push!(seen, legs)
        return length(columns)
    end

    # Phase 1: grow the schedule out of planted lines of flying.
    while length(net.org) < flights_target && length(columns) < target
        base, legs = _crew_plant_line!(rng, net, block, bases, num_airports, n_days, waves)
        if rand(rng) <= keep_fraction
            idx = push_column!(base, legs)
            idx == 0 ? (all_planted_kept = false) : push!(planted_columns, idx)
        else
            all_planted_kept = false
        end
    end

    # Phase 2: fill up with legal walks over the schedule; if the walk sampler
    # stalls (a small schedule can only be flown so many ways) plant another
    # line, which both enlarges the schedule and contributes a column.
    base_starts = [f for f in 1:length(net.org) if net.org[f] <= num_bases]
    stall = 0
    while length(columns) < target
        if stall >= 60 || isempty(base_starts)
            base, legs = _crew_plant_line!(rng, net, block, bases, num_airports, n_days, waves)
            idx = push_column!(base, legs)
            idx == 0 ? (all_planted_kept = false) : push!(planted_columns, idx)
            append!(
                base_starts,
                [
                    f for f in (length(net.org) - length(legs) + 1):length(net.org) if
                    net.org[f] <= num_bases
                ],
            )
            stall = 0
            continue
        end
        base, legs = _crew_sample_pairing(rng, net, base_starts)
        if isempty(legs) || push_column!(base, legs) == 0
            stall += 1
        else
            stall = 0
        end
    end

    # Infeasible mode: one flight no legal pairing can ever contain.
    certificate = nothing
    if plant_orphan
        origin = num_bases < num_airports ? rand(rng, (num_bases + 1):num_airports) : 1
        destination = origin == num_airports ? 1 : origin + 1
        dep = maximum(net.arr) + rules.max_rest + 60
        orphan = _crew_add_flight!(net, origin, destination, dep, dep + block[origin, destination])
        predecessors = count(
            f ->
                f != orphan &&
                net.dst[f] == origin &&
                (
                    rules.min_connect <= dep - net.arr[f] <= rules.max_sit ||
                    rules.min_rest <= dep - net.arr[f] <= rules.max_rest
                ),
            1:length(net.org),
        )
        cert = UncoverableFlightCertificate(orphan, origin, destination, predecessors)
        # Metadata is status-specific: `unknown` promises nothing, so it keeps
        # neither a witness nor a certificate even when a branch happens to
        # settle the question.
        certificate = feasibility_status == infeasible ? cert : nothing
    end

    witness = if (feasibility_status == feasible && all_planted_kept)
        CrewPairingCoverWitness(sort(planted_columns))
    else
        nothing
    end

    costs = [
        _crew_pairing_cost(
            net.dep,
            net.arr,
            legs,
            rules,
            pay_rate,
            duty_guarantee,
            min_daily_credit,
            per_diem_rate,
            hotel_cost,
        ) for legs in columns
    ]

    return AirlineCrewProblem(
        length(net.org),
        num_airports,
        bases,
        locations,
        block,
        net.org,
        net.dst,
        net.dep,
        net.arr,
        rules,
        costs,
        columns,
        column_bases,
        pay_rate,
        duty_guarantee,
        min_daily_credit,
        per_diem_rate,
        hotel_cost,
        witness,
        certificate,
        feasibility_status,
    )
end

"""
    build_model(prob::AirlineCrewProblem)

Build the crew pairing set-partitioning model. Deterministic - uses only the
struct's fields.

# Model

  - `x[p] in {0,1}`: pairing `p` is flown
  - objective: `min sum_p c_p x_p`
  - covering: `sum_{p : f in A_p} x_p == 1` for every flight `f`

A flight contained in no column yields an empty left-hand side, i.e. the
infeasible row `0 == 1` - exactly the certificate the infeasible mode plants.
"""
function build_model(prob::AirlineCrewProblem)
    model = Model()
    n_pairings = length(prob.pairing_costs)

    @variable(model, x[1:n_pairings], Bin)
    @objective(model, Min, sum(prob.pairing_costs[p] * x[p] for p in 1:n_pairings))

    covering = [Int[] for _ in 1:prob.num_flights]
    for p in 1:n_pairings, f in prob.flights_in_pairing[p]
        push!(covering[f], p)
    end
    for f in 1:prob.num_flights
        @constraint(model, sum(x[p] for p in covering[f]) == 1)
    end

    return model
end

# Register the variant
register_variant(
    :airline_crew,
    :standard,
    AirlineCrewProblem,
    "Airline crew pairing set partitioning over operationally legal pairings (airport continuity, connection and rest times, duty limits, base return) with standard credit-hour crew costs",
)
