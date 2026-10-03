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

Planted solution: `pairings` indexes a subset of the generated columns whose
flight sets partition every flight in the schedule. Setting those `x_p` to one
and every other `x_p` to zero satisfies every covering equality, every
crew-availability row (capacities are drawn at or above the witness's own
base-day usage) and every base block-hour balance row (drawn around the
witness's own block hours), so the model and its LP relaxation are feasible.
"""
struct CrewPairingCoverWitness
    pairings::Vector{Int}
end

"""
    CrewShortageCertificate

Infeasibility certificate for a crew shortage on calendar day `day`, built from
LP rows alone so it refutes the LP relaxation as well as the binary model.

Let `F_d = flights_on_day` be the flights departing on `day`, and let
`max_legs_on_day` be the largest number of those flights any single generated
pairing contains. Summing the covering equalities of the `F_d` flights gives
`sum_p n_p x_p = F_d`, where `n_p <= max_legs_on_day` counts pairing `p`'s legs
departing that day. Every pairing with `n_p > 0` is away from base on `day`, so
it appears with coefficient one in the crew-availability row of its base for
that day (`rows`, indices into `crew_rows`). Summing those rows gives
`sum_{p : n_p > 0} x_p <= crew_capacity`. Hence

    F_d = sum_p n_p x_p <= max_legs_on_day * sum_{p : n_p > 0} x_p
        <= max_legs_on_day * crew_capacity < F_d,

a contradiction for any `x >= 0`. The capacity is set with a 10% margin
(`max_legs_on_day * crew_capacity <= 0.9 * F_d`), so the refutation is robust
to solver tolerances. Unlike an empty covering row, it involves every
covering row of the day plus every crew row of the day, so presolve alone does
not find it.
"""
struct CrewShortageCertificate
    day::Int
    flights_on_day::Int
    max_legs_on_day::Int
    crew_capacity::Int
    rows::Vector{Int}
end

"""
    AirlineCrewProblem <: ProblemGenerator

Generator for airline crew pairing instances: set partitioning over
*operationally legal* pairings, with base crew-availability and base block-hour
balance side constraints.

# Overview

A dated flight schedule is generated over a hub-and-spoke airport network, then
crew pairings are built as time-and-airport-respecting walks through that
schedule. Every generated pairing satisfies airport continuity, connection
times, base return, duty limits, and rest rules (see [`CrewPairingRules`]) *by
construction*: pairings are grown leg by leg under the rules and no downstream
step ever edits a pairing's leg set.

The model is the classical crew pairing problem: choose a minimum-cost set of
pairings covering every flight exactly once, subject to the number of crews
each base has available on each calendar day and to a negotiated band on the
block hours flown out of each base.

# Schedule and pairing construction

The schedule is produced by *planting* lines of flying: each planted line is a
legal pairing whose legs are created as it is flown (base -> ... -> base, with
sit times, duty limits and overnight rests). The planted lines therefore
partition the flight set, which both guarantees a realistic connection
structure and provides the feasible witness.

The remaining columns are *through-flight* samples: pick a flight, walk
backwards through legal predecessors until a crew base, then forwards through
legal successors until the crew is home again. First every flight is given at
least `min_coverage` covering pairings, then flights are drawn uniformly. This
is how real pairing generators enumerate: each flight ends up in dozens of
pairings (as in production crew pairing LPs, which are famously degenerate),
and no flight is covered by a single column, which would otherwise let presolve
fix that column and cascade through the whole partitioning matrix.

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
  - `pairing_first_day::Vector{Int}`, `pairing_last_day::Vector{Int}`: calendar
    days (1-based, `departure ÷ 1440 + 1`) of the pairing's first departure and
    last arrival; the crew is away from base on every day in between
  - `pairing_block_hours::Vector{Float64}`: block hours flown by each pairing
  - `crew_rows::Vector{Tuple{Int,Int}}`: `(base, day)` of each crew-availability
    row (only rows that can bind are emitted: more columns than capacity)
  - `crew_capacity::Vector{Int}`: crews available for each row of `crew_rows`
  - `base_block_lower::Vector{Float64}`, `base_block_upper::Vector{Float64}`:
    block-hour band per base
  - `pay_rate::Float64`: crew pay per credit hour
  - `duty_guarantee::Float64`: minimum-duty-guarantee fraction (credit per duty hour)
  - `min_daily_credit::Int`: minimum credited minutes per duty period
  - `per_diem_rate::Float64`: per-diem paid per hour away from base
  - `hotel_cost::Float64`: hotel cost per overnight
  - `feasible_witness::Union{Nothing,CrewPairingCoverWitness}`: planted partition
  - `infeasibility_certificate::Union{Nothing,CrewShortageCertificate}`
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
    pairing_first_day::Vector{Int}
    pairing_last_day::Vector{Int}
    pairing_block_hours::Vector{Float64}
    crew_rows::Vector{Tuple{Int, Int}}
    crew_capacity::Vector{Int}
    base_block_lower::Vector{Float64}
    base_block_upper::Vector{Float64}
    pay_rate::Float64
    duty_guarantee::Float64
    min_daily_credit::Int
    per_diem_rate::Float64
    hotel_cost::Float64
    feasible_witness::Union{Nothing, CrewPairingCoverWitness}
    infeasibility_certificate::Union{Nothing, CrewShortageCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    _crew_day(t) -> Int

Calendar day (1-based) containing minute `t` of the horizon.
"""
_crew_day(t::Integer) = fld(t, 1440) + 1

# ---------------------------------------------------------------------------
# Flight network container (construction-time only)
# ---------------------------------------------------------------------------

"""
Mutable flight schedule with time indexes per airport, used while growing the
schedule and sampling pairings. `by_origin[a]` lists the flights leaving `a`
sorted by departure time (`by_origin_dep[a]` holds the matching times), and
`by_dest[a]` the flights arriving at `a` sorted by arrival time
(`by_dest_arr[a]`), so connection windows are binary searchable both ways.
"""
mutable struct _CrewNet
    org::Vector{Int}
    dst::Vector{Int}
    dep::Vector{Int}
    arr::Vector{Int}
    by_origin::Vector{Vector{Int}}
    by_origin_dep::Vector{Vector{Int}}
    by_dest::Vector{Vector{Int}}
    by_dest_arr::Vector{Vector{Int}}
    rules::CrewPairingRules
end

_crew_net(num_airports::Int, rules::CrewPairingRules) = _CrewNet(
    Int[],
    Int[],
    Int[],
    Int[],
    [Int[] for _ in 1:num_airports],
    [Int[] for _ in 1:num_airports],
    [Int[] for _ in 1:num_airports],
    [Int[] for _ in 1:num_airports],
    rules,
)

"""
    _crew_add_flight!(net, o, d, dep, arr) -> Int

Append a flight and keep both time indexes sorted. Returns the new flight id.
"""
function _crew_add_flight!(net::_CrewNet, o::Int, d::Int, dep::Int, arr::Int)
    push!(net.org, o)
    push!(net.dst, d)
    push!(net.dep, dep)
    push!(net.arr, arr)
    id = length(net.org)
    pos = searchsortedlast(net.by_origin_dep[o], dep) + 1
    insert!(net.by_origin_dep[o], pos, dep)
    insert!(net.by_origin[o], pos, id)
    pos = searchsortedlast(net.by_dest_arr[d], arr) + 1
    insert!(net.by_dest_arr[d], pos, arr)
    insert!(net.by_dest[d], pos, id)
    return id
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
    d = net.dst[id]
    pos = findfirst(==(id), net.by_dest[d])
    deleteat!(net.by_dest[d], pos)
    deleteat!(net.by_dest_arr[d], pos)
    pop!(net.org)
    pop!(net.dst)
    pop!(net.dep)
    pop!(net.arr)
    return nothing
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

"""
    _crew_predecessors(net, f, lo, hi)

Flights that leg `f` may follow with a ground time in `[lo, hi]`: they arrive
at `f`'s origin inside the corresponding arrival-time window.
"""
function _crew_predecessors(net::_CrewNet, f::Int, lo::Int, hi::Int)
    a = net.org[f]
    t = net.dep[f]
    times = net.by_dest_arr[a]
    i = searchsortedfirst(times, t - hi)
    j = searchsortedlast(times, t - lo)
    return view(net.by_dest[a], i:j)
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

Hub-and-spoke airport map: bases (the hubs) spread around the centre of a
continental-scale box, and each spoke scattered around a home hub (spokes are
dealt round-robin to hubs and placed within ~550 km of them). Returns
locations, the home hub of every airport (a hub is its own home) and the
block-time matrix (35 min taxi/climb overhead plus cruise at ~720 km/h, rounded
to 5 minutes and clamped to a narrowbody 45-240 minute range).
"""
function _crew_geography(rng::AbstractRNG, num_airports::Int, num_bases::Int)
    width, height = 2000.0, 1400.0
    locations = Tuple{Float64, Float64}[]
    home = collect(1:num_airports)
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
    for a in (num_bases + 1):num_airports
        hub = mod1(a - num_bases, num_bases)
        home[a] = hub
        radius = 150 + 400 * rand(rng)
        theta = rand(rng) * 2pi
        push!(
            locations,
            (
                clamp(locations[hub][1] + radius * cos(theta), 0.0, width),
                clamp(locations[hub][2] + radius * sin(theta), 0.0, height),
            ),
        )
    end
    block = zeros(Int, num_airports, num_airports)
    for i in 1:num_airports, j in 1:num_airports
        i == j && continue
        d = hypot(locations[i][1] - locations[j][1], locations[i][2] - locations[j][2])
        block[i, j] = clamp(5 * round(Int, (35 + d / 12.0) / 5), 45, 240)
    end
    return locations, home, block
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
    _crew_next_airport(rng, block, cur, base, home, dep_t, duty_start,
                       duty_block, rules, reserve)

Pick the next airport for a planted leg. Hub-and-spoke bias: from a hub the
crew usually flies out to one of that hub's own spokes and sometimes to another
hub; from a spoke it almost always flies back to its home hub. Candidates
must keep the duty inside its block and elapsed limits; when `reserve` is set
(the final duty of the line) they must additionally leave room to fly home to
`base` afterwards. Returns `0` when nothing fits.
"""
function _crew_next_airport(
    rng::AbstractRNG,
    block::Matrix{Int},
    cur::Int,
    base::Int,
    home::Vector{Int},
    dep_t::Int,
    duty_start::Int,
    duty_block::Int,
    rules::CrewPairingRules,
    reserve::Bool,
)
    cands = Int[]
    weights = Float64[]
    cur_is_hub = home[cur] == cur
    for a in eachindex(home)
        a == cur && continue
        a_is_hub = home[a] == a
        w = if cur_is_hub
            a_is_hub ? 1.0 : (home[a] == cur ? 4.0 : 0.0)
        else
            a == home[cur] ? 6.0 : (a_is_hub ? 0.4 : (home[a] == home[cur] ? 0.3 : 0.0))
        end
        # Off-network legs only exist to fly the crew home in the final duty.
        (w > 0 || (reserve && a == base)) || continue
        ft = block[cur, a]
        duty_block + ft <= rules.max_block_minutes || continue
        (dep_t + ft) - duty_start <= rules.max_duty_minutes || continue
        if reserve && a != base
            back = block[a, base]
            duty_block + ft + back <= rules.max_block_minutes || continue
            (dep_t + ft + rules.min_connect + back) - duty_start <= rules.max_duty_minutes ||
                continue
        end
        push!(cands, a)
        push!(weights, max(w, 0.2))
    end
    isempty(cands) && return 0
    return _crew_wsample(rng, cands, weights)
end

"""
    _crew_plant_line!(rng, net, block, bases, home, n_days, waves)

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
    home::Vector{Int},
    n_days::Int,
    waves::Vector{Int},
)
    rules = net.rules
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
                        home,
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
    spoke = argmin([a in bases ? typemax(Int) : block[base, a] for a in eachindex(home)])
    day = (rand(rng, 1:n_days) - 1) * 1440 + rand(rng, waves)
    ft1 = block[base, spoke]
    ft2 = block[spoke, base]
    f1 = _crew_add_flight!(net, base, spoke, day, day + ft1)
    dep2 = day + ft1 + rules.min_connect
    f2 = _crew_add_flight!(net, spoke, base, dep2, dep2 + ft2)
    return base, [f1, f2]
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
    _crew_sample_through(rng, net, f, is_base) -> (base, legs)

Sample one legal pairing that contains flight `f`. The walk first goes
*backwards* from `f` through legal predecessors (same-duty connections inside
the sit window, or an earlier duty across a legal rest) until it stands at a
crew base, which becomes the pairing's base; it then continues *forwards* from
`f` with [`_crew_extend!`](@ref) until the crew is back at that base. The
backward steps check the duty leg/block/elapsed limits and the duty count, and
the forward search is seeded with the state of the duty that holds `f`, so any
returned walk is a legal pairing. Returns `(0, Int[])` when a few attempts
fail.
"""
function _crew_sample_through(rng::AbstractRNG, net::_CrewNet, f::Int, is_base::BitVector)
    r = net.rules
    max_total = r.max_legs_per_duty * r.max_duties
    options = Tuple{Int, Bool}[]
    for _ in 1:4
        legs = Int[f]
        used = Set{Int}(legs)
        duty_last_arr = net.arr[f]
        duty_block = net.arr[f] - net.dep[f]
        duty_legs = 1
        n_duties = 1
        ok = false
        while true
            head = legs[1]
            at_base = is_base[net.org[head]]
            if at_base && rand(rng) < 0.55
                ok = true
                break
            end
            if length(legs) >= max_total - 1
                ok = at_base
                break
            end
            empty!(options)
            if duty_legs < r.max_legs_per_duty
                for g in _crew_predecessors(net, head, r.min_connect, r.max_sit)
                    g in used && continue
                    b = net.arr[g] - net.dep[g]
                    duty_block + b <= r.max_block_minutes || continue
                    duty_last_arr - net.dep[g] <= r.max_duty_minutes || continue
                    push!(options, (g, false))
                end
            end
            if n_duties < r.max_duties
                for g in _crew_predecessors(net, head, r.min_rest, r.max_rest)
                    g in used && continue
                    push!(options, (g, true))
                end
            end
            if isempty(options)
                ok = at_base
                break
            end
            g, new_duty = options[rand(rng, 1:length(options))]
            pushfirst!(legs, g)
            push!(used, g)
            b = net.arr[g] - net.dep[g]
            if new_duty
                duty_last_arr = net.arr[g]
                duty_block = b
                duty_legs = 1
                n_duties += 1
            else
                duty_block += b
                duty_legs += 1
            end
        end
        ok || continue

        base = net.org[legs[1]]
        duties = _crew_duty_ranges(net.dep, net.arr, legs, r.max_sit)
        last_duty = duties[end]
        duty_start = net.dep[legs[first(last_duty)]]
        block = sum(net.arr[legs[i]] - net.dep[legs[i]] for i in last_duty)
        budget = Ref(300)
        stop_prob = 0.15 + 0.45 * rand(rng)
        if _crew_extend!(
            rng, net, base, legs, used, duty_start, block, length(last_duty), length(duties), budget, stop_prob
        )
            return base, legs
        end
    end
    return 0, Int[]
end

"""
    _crew_largest_remainder(total, weights) -> Vector{Int}

Split the integer `total` across `weights` proportionally (largest remainder),
giving every entry at least one unit when `total >= length(weights)`.
"""
function _crew_largest_remainder(total::Int, weights::Vector{Float64})
    n = length(weights)
    floor_each = total >= n ? 1 : 0
    rest = total - floor_each * n
    w = max.(weights, 1e-9)
    shares = rest .* w ./ sum(w)
    alloc = floor.(Int, shares)
    order = sortperm(shares .- alloc; rev=true)
    for i in 1:(rest - sum(alloc))
        alloc[order[i]] += 1
    end
    return alloc .+ floor_each
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
`target_variables` columns (for `target_variables >= 4`). The schedule holds
about `0.35 * target_variables` flights (one covering equality each), so each
flight is covered by roughly 20 pairings on average and by at least six
whenever the column budget and network allow. On top come the crew-availability
rows - one per `(base, day)` whose column count exceeds its capacity, at most
`num_bases * horizon_days` - and one block-hour balance row per base.

Airports grow as `clamp(round(3 + sqrt(F)/1.5), 6, 80)` for `F` target
flights, bases as `clamp(round(airports/6), 2, 12)`, and the horizon as
`clamp(round(F / (10 * airports)), 2, 28)` days, so hubs see tens of
departures per day at scale.

# Feasibility

The planted lines of flying are always kept as columns, so they partition the
flight set.

  - `feasible`: crew capacities are drawn at or above the planted lines' own
    base-day usage and the block-hour bands around their block hours, so the
    planted partition (`feasible_witness`) satisfies every row.
  - `infeasible`: as `feasible`, except that on the busiest day `d` the total
    crew capacity across bases is cut to `floor(0.9 * F_d / max_legs_on_day)`,
    which a covering-plus-availability row aggregation refutes
    ([`CrewShortageCertificate`]). Each base keeps at least one crew where
    possible, so no single row is trivially contradictory: proving
    infeasibility takes simplex work, not presolve.
  - `unknown`: a natural instance - each base roster is drawn at 92-108% of
    the planted peak crew-day usage, on both sides of it, so the instance
    may or may not be feasible depending on how efficiently the generated
    pairings can use crews. No metadata is attached.
"""
function AirlineCrewProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    rng = MersenneTwister(seed)

    target = max(target_variables, 4)
    flights_target = clamp(round(Int, 0.35 * target), 12, 400_000)
    num_airports = clamp(round(Int, 3 + sqrt(flights_target) / 1.5), 6, 80)
    num_bases = clamp(round(Int, num_airports / 6), 2, 12)
    n_days = clamp(round(Int, flights_target / (10 * num_airports)), 2, 28)
    waves = collect(360:90:1170)
    min_coverage = 6

    rules = _crew_rules(rng)
    locations, home, block = _crew_geography(rng, num_airports, num_bases)
    bases = collect(1:num_bases)
    is_base = falses(num_airports)
    is_base[bases] .= true

    pay_rate = 180.0 + 140.0 * rand(rng)
    duty_guarantee = 0.50 + 0.10 * rand(rng)
    min_daily_credit = rand(rng, 240:15:315)
    per_diem_rate = 2.0 + 1.5 * rand(rng)
    hotel_cost = 90.0 + 70.0 * rand(rng)

    net = _crew_net(num_airports, rules)
    columns = Vector{Int}[]
    column_bases = Int[]
    seen = Set{Vector{Int}}()
    planted_columns = Int[]
    coverage = Int[]

    function push_column!(base::Int, legs::Vector{Int})
        legs in seen && return 0
        push!(columns, legs)
        push!(column_bases, base)
        push!(seen, legs)
        length(coverage) < length(net.org) && append!(coverage, zeros(Int, length(net.org) - length(coverage)))
        for f in legs
            coverage[f] += 1
        end
        return length(columns)
    end

    function plant!()
        base, legs = _crew_plant_line!(rng, net, block, bases, home, n_days, waves)
        idx = push_column!(base, legs)
        # Planted legs are fresh flights, so a planted line is never a duplicate.
        @assert idx > 0
        push!(planted_columns, idx)
        return nothing
    end

    # Phase 1: grow the schedule out of planted lines of flying.
    while length(net.org) < flights_target && length(columns) < target
        plant!()
    end

    # Phase 2: give every flight `min_coverage` covering pairings.
    stall = 0
    for f in shuffle(rng, collect(1:length(net.org)))
        tries = 0
        while coverage[f] < min_coverage && tries < 5 * min_coverage && length(columns) < target
            tries += 1
            base, legs = _crew_sample_through(rng, net, f, is_base)
            isempty(legs) || push_column!(base, legs)
        end
        length(columns) >= target && break
    end

    # Phase 3: fill with through-flight samples at uniformly drawn flights. A
    # small schedule can only be flown so many ways; when the sampler stalls,
    # plant another line, which enlarges the schedule and adds a column.
    while length(columns) < target
        if stall >= 60
            plant!()
            stall = 0
            continue
        end
        f = rand(rng, 1:length(net.org))
        base, legs = _crew_sample_through(rng, net, f, is_base)
        if isempty(legs) || push_column!(base, legs) == 0
            stall += 1
        else
            stall = 0
        end
    end

    num_flights = length(net.org)
    n_cols = length(columns)
    first_day = [_crew_day(net.dep[legs[1]]) for legs in columns]
    last_day = [_crew_day(net.arr[legs[end]]) for legs in columns]
    block_hours = [round(sum(net.arr[f] - net.dep[f] for f in legs) / 60; digits=2) for legs in columns]
    costs = [
        _crew_pairing_cost(
            net.dep, net.arr, legs, rules, pay_rate, duty_guarantee, min_daily_credit, per_diem_rate, hotel_cost
        ) for legs in columns
    ]

    # Crew-availability rows: columns away from base per (base, day), and the
    # planted lines' own usage.
    horizon = maximum(last_day)
    ncols = zeros(Int, num_bases, horizon)
    usage = zeros(Int, num_bases, horizon)
    is_planted = falses(n_cols)
    is_planted[planted_columns] .= true
    for p in 1:n_cols, d in first_day[p]:last_day[p]
        ncols[column_bases[p], d] += 1
        is_planted[p] && (usage[column_bases[p], d] += 1)
    end

    # Each base rosters a fixed number of crews per day. The planted lines are
    # close to crew-efficient on their peak days (measured: cutting every base
    # to 95% of its planted peak makes almost every instance infeasible), so
    # the roster is drawn relative to that peak.
    peak = vec(maximum(usage; dims=2))
    roster = if feasibility_status == unknown
        # Two-sided: rosters above or below the planted peak, so the instance
        # is feasible or not depending on how well the generated pairings can
        # absorb the peak days.
        [max(1, round(Int, peak[b] * (0.92 + 0.16 * rand(rng)))) for b in 1:num_bases]
    else
        [max(2, ceil(Int, peak[b] * (1.0 + 0.15 * rand(rng)))) for b in 1:num_bases]
    end
    capacity = repeat(roster, 1, horizon)
    forced = falses(num_bases, horizon)

    certificate = nothing
    if feasibility_status == infeasible
        # Crew shortage on the busiest day: total capacity below what any
        # combination of pairings needs to fly that day's departures.
        flights_per_day = zeros(Int, horizon)
        for f in 1:num_flights
            flights_per_day[_crew_day(net.dep[f])] += 1
        end
        shortage_day = argmax(flights_per_day)
        max_legs = maximum(count(f -> _crew_day(net.dep[f]) == shortage_day, legs) for legs in columns)
        total_cap = floor(Int, 0.9 * flights_per_day[shortage_day] / max_legs)
        active = [b for b in 1:num_bases if ncols[b, shortage_day] > 0]
        alloc = _crew_largest_remainder(total_cap, [float(usage[b, shortage_day]) for b in active])
        for (b, c) in zip(active, alloc)
            capacity[b, shortage_day] = c
            forced[b, shortage_day] = true
        end
        certificate = (shortage_day, flights_per_day[shortage_day], max_legs, total_cap)
    end

    crew_rows = Tuple{Int, Int}[]
    crew_capacity = Int[]
    for d in 1:horizon, b in 1:num_bases
        if forced[b, d] || (ncols[b, d] > 0 && capacity[b, d] < ncols[b, d])
            push!(crew_rows, (b, d))
            push!(crew_capacity, capacity[b, d])
        end
    end

    cert = nothing
    if certificate !== nothing
        day, f_d, max_legs, total_cap = certificate
        rows = [i for (i, (b, d)) in enumerate(crew_rows) if d == day]
        cert = CrewShortageCertificate(day, f_d, max_legs, total_cap, rows)
    end

    # Base block-hour balance: a negotiated band around the planted flying.
    planted_hours = zeros(num_bases)
    for p in planted_columns
        planted_hours[column_bases[p]] += block_hours[p]
    end
    mean_hours = sum(planted_hours) / num_bases
    block_lower = zeros(num_bases)
    block_upper = zeros(num_bases)
    for b in 1:num_bases
        if planted_hours[b] > 0
            block_lower[b] = floor(planted_hours[b] * (0.80 + 0.15 * rand(rng)); digits=1)
            block_upper[b] = ceil(planted_hours[b] * (1.05 + 0.20 * rand(rng)); digits=1)
        else
            block_upper[b] = ceil(mean_hours * (0.5 + 0.5 * rand(rng)); digits=1)
        end
    end

    witness = feasibility_status == feasible ? CrewPairingCoverWitness(sort(planted_columns)) : nothing

    return AirlineCrewProblem(
        num_flights,
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
        first_day,
        last_day,
        block_hours,
        crew_rows,
        crew_capacity,
        block_lower,
        block_upper,
        pay_rate,
        duty_guarantee,
        min_daily_credit,
        per_diem_rate,
        hotel_cost,
        witness,
        cert,
        feasibility_status,
    )
end

"""
    build_model(prob::AirlineCrewProblem)

Build the crew pairing model. Deterministic - uses only the struct's fields.

# Model

  - `x[p] in {0,1}`: pairing `p` is flown
  - objective: `min sum_p c_p x_p`
  - covering: `sum_{p : f in A_p} x_p == 1` for every flight `f`
  - crew availability: `sum_{p : base(p) = b, first_day(p) <= d <= last_day(p)} x_p <= cap_{b,d}`
    for every `(b, d)` in `crew_rows`
  - base balance: `lo_b <= sum_{p : base(p) = b} block_hours_p x_p <= hi_b` for
    every base that owns at least one column
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

    row_of = Dict{Tuple{Int, Int}, Int}(key => i for (i, key) in enumerate(prob.crew_rows))
    members = [Int[] for _ in prob.crew_rows]
    for p in 1:n_pairings, d in prob.pairing_first_day[p]:prob.pairing_last_day[p]
        i = get(row_of, (prob.pairing_bases[p], d), 0)
        i > 0 && push!(members[i], p)
    end
    for (i, cols) in enumerate(members)
        @constraint(model, sum(x[p] for p in cols) <= prob.crew_capacity[i])
    end

    by_base = [Int[] for _ in prob.bases]
    for p in 1:n_pairings
        push!(by_base[prob.pairing_bases[p]], p)
    end
    for b in prob.bases
        isempty(by_base[b]) && continue
        @constraint(
            model,
            prob.base_block_lower[b] <=
            sum(prob.pairing_block_hours[p] * x[p] for p in by_base[b]) <=
            prob.base_block_upper[b]
        )
    end

    return model
end

# Register the variant
register_variant(
    :airline_crew,
    :standard,
    AirlineCrewProblem,
    "Airline crew pairing over operationally legal pairings (airport continuity, connection and rest times, duty limits, base return) with credit-hour costs, dense per-flight coverage, base-day crew availability and base block-hour balance rows",
)
