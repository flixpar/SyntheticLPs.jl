using JuMP
using Random
using Distributions
using StatsBase

"""
Planted ground truth for a `feasible`
`inverse_optimization/market_clearing` instance: the true offer costs `cost`
and the market-clearing duals — `energy_duals` (the locational price `λ_t` each
period, set by the marginal unit) and `capacity_duals` (the lost-opportunity
cost `μ_{t,g}` of every generator at its ceiling).

The pair reproduces the textbook merit-order optimality conditions: the price
equals the marginal unit's offer, infra-marginal units run at their ceiling
with `μ = λ_t - cost_g > 0`, the marginal unit is partially loaded with
`μ = 0`, and supra-marginal units are offline. Every row of the built model
holds with the ramping duals at zero.
"""
struct DispatchPricingWitness
    cost::Vector{Float64}
    energy_duals::Vector{Float64}
    capacity_duals::Matrix{Float64}
end

"""
Structured infeasibility certificate for
`inverse_optimization/market_clearing`: in `period`, the observed dispatch runs
`maxed_unit` at its full capacity while `idle_unit` produces nothing, yet the
admissible box forces every cost of the maxed unit above every cost of the
idle one (`maxed_cost_lower > idle_cost_upper`) — a merit-order inversion no
offer-cost vector can explain.

For any (cost, dual) pair satisfying the built rows, complementary slackness
holds automatically: the maxed unit's positive output forces
`c_maxed = λ - μ <= λ` while the idle unit's slack capacity row forces its
capacity dual to zero and hence `λ <= c_idle` (the ramping duals vanish because
every ramping row is slack at the observed dispatch). Together
`c_maxed <= λ <= c_idle`, contradicting the box. The refutation uses LP rows
alone, so it survives `relax_integer` and `bounds_to_constraints`.
"""
struct MeritOrderInversionCertificate
    period::Int
    idle_unit::Int
    maxed_unit::Int
    idle_cost_upper::Float64
    maxed_cost_lower::Float64
end

"""
    InverseDispatchCostProblem <: ProblemGenerator

Generator for copper-plate *market offer-cost inference*: the inverse problem behind
reconstructing rival generators' offer curves from cleared market data (Ruiz,
Conejo & Bertsimas, *IEEE Trans. Power Systems* 28(3), 2013; Liang & Dvorkin,
ACM e-Energy 2023; see also Birge, Hortaçsu & Pavlin, *Operations Research*
65(4), 2017 for market-structure inference from outcomes).

# Overview
A multi-period economic dispatch is given as **data**: generator `capacities`
(MW), a diurnal `demands` profile (MWh per period), `ramp_limits` between
consecutive periods, and the `observed_dispatch` that was actually cleared.
The decision variables of the *built* model are the offer costs `c_g` inside an
admissible box and the dispatch duals — the energy price `λ_t`, the capacity
duals `μ_{t,g}`, and the ramping duals of the constrained pairs. Requiring the
observation to be optimal turns into the KKT rows of the dispatch LP:

    minimize    Σ_g w_g (p_g + q_g)
    subject to  λ_t - μ_{t,g} + a_{t,g} - a_{t-1,g} - b_{t,g} + b_{t-1,g} <= c_g
                                                        (stationarity, T·G rows)
                Σ_t D_t λ_t - Σ_{t,g} C_g μ_{t,g}
                    - Σ R_g (a + b) == Σ_{t,g} x̂_{t,g} c_g
                                                        (strong duality, 1 row)
                c - p + q == ĉ,  λ, μ, a, b >= 0,  ℓ <= c <= u

with `a`, `b` the duals of the ramp-up/ramp-down rows (only for units and
periods whose ramping is constrained). Whenever the rows hold, complementary
slackness is automatic, so the observed dispatch is optimal for the dispatch LP
under the inferred offers — the inferred costs rationalize the market outcome.

# Planted ground truth
True offers are sampled by unit type — baseload (large, cheap), intermediate,
peaker (small, expensive) — and the observation is the *exact* merit-order
dispatch against them: infra-marginal units at their ceiling, one marginal unit
partially loaded each period (setting `λ_t`), supra-marginal units off. The
prices `λ_t` therefore swing with the diurnal demand curve exactly as
market-clearing prices do. The prior is the true offer plus additive truncated
Gaussian noise (a few \$/MWh, matching the noise levels used in the market
inference literature), and the admissible box is an absolute \$/MWh interval
around it. Ramping limits are derived from the dispatch with strict slack, so
the ramping duals vanish at the observation — as Liang–Dvorkin observe for the
systems they study.

# Feasibility profiles
- `feasible`: stores a `DispatchPricingWitness` (true offers and duals).
- `infeasible`: the observation maxes an expensive-by-box unit in every period
  while a cheap-by-box unit never runs — a merit-order inversion refuted by a
  `MeritOrderInversionCertificate` from LP rows alone.
- `unknown`: a coin flip between two unguaranteed channels — prior noise and
  box sampled independently (the true offers may or may not lie inside the
  box), or a box-consistent prior whose observed dispatch is nudged out of
  merit order in one period (shifted load from the marginal unit to an offline
  dearer one), which the sampled boxes may or may not be able to rationalize.

# Fields
- `num_periods::Int`, `num_units::Int`: Dispatch horizon and fleet size
- `capacities::Vector{Float64}`: Generator ceilings `C_g` (MW)
- `demands::Vector{Float64}`: Energy demand `D_t` per period (MWh)
- `ramp_limits::Vector{Float64}`: Ramping allowance `R_g`
- `ramp_pairs::Vector{Tuple{Int,Int}}`: `(unit, period)` pairs with ramp rows
- `observed_dispatch::Matrix{Float64}`: Cleared quantities `x̂` (T × G)
- `unit_types::Vector{Symbol}`: Sampled fleet composition
- `prior_cost::Vector{Float64}`: Prior offers `ĉ`
- `cost_lower::Vector{Float64}`, `cost_upper::Vector{Float64}`: Admissible box
- `deviation_weights::Vector{Float64}`: Weighted-deviation weights
- `feasible_witness::Union{Nothing,DispatchPricingWitness}`: set for `feasible`
- `infeasibility_certificate::Union{Nothing,MeritOrderInversionCertificate}`: set for `infeasible`
- `feasibility_status::FeasibilityStatus`: Requested profile
"""
struct InverseDispatchCostProblem <: ProblemGenerator
    num_periods::Int
    num_units::Int
    capacities::Vector{Float64}
    demands::Vector{Float64}
    ramp_limits::Vector{Float64}
    ramp_pairs::Vector{Tuple{Int,Int}}
    observed_dispatch::Matrix{Float64}
    unit_types::Vector{Symbol}
    prior_cost::Vector{Float64}
    cost_lower::Vector{Float64}
    cost_upper::Vector{Float64}
    deviation_weights::Vector{Float64}
    feasible_witness::Union{Nothing,DispatchPricingWitness}
    infeasibility_certificate::Union{Nothing,MeritOrderInversionCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    _merit_order_dispatch(costs, capacities, demands)

The dispatch an ideal market produces against `costs`: generators are stacked
in ascending offer order, infra-marginal units fill their ceiling, one marginal
unit takes the remainder, and supra-marginal units stay off. Returns the `T × G`
dispatch matrix. Deterministic (ties broken by unit index; continuous offers
make ties measure-zero).
"""
function _merit_order_dispatch(costs::Vector{Float64}, capacities::Vector{Float64},
                               demands::Vector{Float64})
    T, G = length(demands), length(costs)
    order = sortperm(costs)
    dispatch = zeros(T, G)
    for t in 1:T
        remaining = demands[t]
        for g in order
            if remaining <= 0.0
                break
            end
            output = min(capacities[g], remaining)
            dispatch[t, g] = output
            remaining -= output
        end
    end
    return dispatch
end

"""
    _dispatch_dims(target)

Solve the fleet shape for a target model size. The built model carries
`3G + T + T*G + 2r` variables — offers plus deviation split (3G), prices (T),
capacity duals (T·G), and ramping duals (two per constrained `(unit, period)`
pair) — so the horizon and fleet are searched around their ideal proportions
for a pair whose remainder is an even number of ramping pairs in range.
Falls back to the smallest fleet (T=4, G=3) for very small targets.
"""
function _dispatch_dims(target::Int)
    periods0 = clamp(round(Int, sqrt(target / 6.0)), 4, 24)
    units0 = clamp(round(Int, target / (periods0 + 3.0)), 3, 300)
    for periods in _around(periods0, 4), units in _around(units0, 3)
        remainder = target - 3 * units - periods - periods * units
        0 <= remainder <= 2 * units * (periods - 1) && remainder % 2 == 0 &&
            return (periods, units, remainder ÷ 2)
    end
    periods, units = 4, 3
    ramp_pairs = clamp(target - 3 * units - periods - periods * units, 0, 2 * units * (periods - 1)) ÷ 2
    return (periods, units, ramp_pairs)
end

"""
    InverseDispatchCostProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a market offer-cost inference instance. Data ranges follow the market
inference literature: offers in the 12–85 \$/MWh band by unit type, capacities
from tens (peakers) to hundreds (baseload) of MW, a diurnal demand curve at
55–70% of fleet capacity with a 15–35% peak–valley swing, and prior noise of a
few \$/MWh.
"""
function InverseDispatchCostProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    _check_inverse_target(target_variables)
    rng = MersenneTwister(seed)

    T, G, num_ramp_pairs = _dispatch_dims(target_variables)

    # --- Fleet --------------------------------------------------------------
    # Baseload units are large and cheap, peakers small and expensive; the mix
    # sets the shape of the offer curve the inverse problem must recover.
    types = [sample(rng, [:baseload, :intermediate, :peaker],
                    Weights([0.30, 0.45, 0.25])) for _ in 1:G]
    capacities = [round(type == :baseload ? rand(rng, LogNormal(log(450.0), 0.25)) :
                          type == :intermediate ? rand(rng, LogNormal(log(180.0), 0.3)) :
                          rand(rng, LogNormal(log(90.0), 0.35)); digits=1)
                  for type in types]
    total_capacity = sum(capacities)

    # --- Demand profile -------------------------------------------------------
    # A diurnal swing around a base utilization of the fleet, capped so the
    # largest unit can fail without load shedding (a realistic reserve margin).
    base = rand(rng, Uniform(0.55, 0.70))
    amplitude = rand(rng, Uniform(0.15, 0.35))
    phase = rand(rng, Uniform(0.0, 2.0 * pi))
    demands = [base * total_capacity *
               (1.0 + amplitude * sin(2.0 * pi * (t - 1) / T + phase)) for t in 1:T]
    ceiling = 0.85 * (total_capacity - maximum(capacities))
    maximum(demands) > ceiling && (demands .*= ceiling / maximum(demands))
    demands = round.(demands; digits=1)

    # --- True offers and the observed dispatch -------------------------------
    offer = [round(type == :baseload ? rand(rng, Uniform(12.0, 32.0)) :
                    type == :intermediate ? rand(rng, Uniform(25.0, 55.0)) :
                    rand(rng, Uniform(45.0, 85.0)); digits=1) for type in types]

    prior_cost = similar(offer)
    box_radius = similar(offer)
    feasible_witness = nothing
    certificate = nothing
    perturb_observation = false

    if feasibility_status == infeasible
        # Merit-order inversion: the observation is the merit-order dispatch
        # against offers in which the fleet's largest unit is priced out of the
        # market and its smallest is cheapest — so the small unit runs flat out
        # in every period (it is first in the stack and smaller than a third of
        # the fleet, hence below even the valley demand) while the large unit
        # never runs (the reserve-capped demand never needs it). The admissible
        # box then says the opposite about their costs, and no offer vector can
        # rationalize the schedule.
        idle_unit = argmax(capacities)
        maxed_unit = argmin(capacities)
        observed_offer = copy(offer)
        observed_offer[idle_unit] = 1.10 * maximum(offer)
        observed_offer[maxed_unit] = 0.85 * minimum(offer)
        for g in 1:G
            if g == idle_unit
                prior_cost[g] = round(rand(rng, Uniform(15.0, 25.0)); digits=1)
                box_radius[g] = round(rand(rng, Uniform(3.0, 8.0)); digits=1)
            elseif g == maxed_unit
                prior_cost[g] = round(rand(rng, Uniform(58.0, 80.0)); digits=1)
                box_radius[g] = round(rand(rng, Uniform(3.0, 8.0)); digits=1)
            else
                prior_cost[g] = round(max(offer[g] + rand(rng, Normal(0.0, 2.5)),
                                          0.5 * offer[g]); digits=1)
                box_radius[g] = round(rand(rng, Uniform(4.0, 10.0)); digits=1)
            end
        end
        offer = observed_offer
    else
        # Unknown profile: a coin flip between two genuinely uncertain
        # channels. (a) The prior noise and the box are sampled independently,
        # so the true offers may or may not lie inside the box. (b) The box is
        # guaranteed to contain the true offers, but the observed dispatch is
        # nudged out of merit order in one period — whether any admissible
        # offer vector explains that period then depends on whether the two
        # affected units' boxes overlap, which the data decides.
        perturb_observation = feasibility_status == unknown && rand(rng) < 0.6
        noise_sigma = (feasibility_status == feasible || perturb_observation) ?
                      rand(rng, Uniform(1.5, 3.0)) : rand(rng, Uniform(2.0, 6.0))
        for g in 1:G
            drift = rand(rng, truncated(Normal(0.0, noise_sigma),
                                        -2.5 * noise_sigma, 2.5 * noise_sigma))
            prior_cost[g] = round(max(offer[g] + drift, 0.5 * offer[g]); digits=1)
            box_radius[g] = rand(rng, (feasibility_status == feasible ||
                                       perturb_observation) ?
                                      Uniform(4.0, 12.0) : Uniform(2.5, 8.0))
        end
        if feasibility_status == feasible || perturb_observation
            # The box must contain the true offer, so its radius stays safely
            # above the truncation bound of the prior noise.
            box_radius = max.(box_radius, abs.(prior_cost .- offer) .+ 0.5)
        end
    end

    cost_lower = max.(prior_cost .- box_radius, 0.25 .* prior_cost)
    cost_upper = prior_cost .+ box_radius

    if feasibility_status == infeasible
        certificate = MeritOrderInversionCertificate(argmax(demands),
                                                     idle_unit, maxed_unit,
                                                     cost_upper[idle_unit],
                                                     cost_lower[maxed_unit])
    end

    dispatch = _merit_order_dispatch(offer, capacities, demands)

    if perturb_observation
        # Out-of-merit adjustment in one period: shift part of the marginal
        # unit's load to an offline, more expensive unit. Demand is still met
        # exactly and every capacity stays within bounds; whether the market
        # can still be rationalized now hinges on the sampled boxes.
        t = rand(rng, 1:T)
        marginal = findfirst(g -> 0.0 < dispatch[t, g] < capacities[g], 1:G)
        offline = findall(g -> dispatch[t, g] == 0.0 && g != marginal, 1:G)
        if marginal !== nothing && !isempty(offline)
            g2 = rand(rng, offline)
            shift = min(0.5 * dispatch[t, marginal], 0.5 * capacities[g2])
            dispatch[t, marginal] -= shift
            dispatch[t, g2] += shift
        end
    end

    # --- Ramping limits ------------------------------------------------------
    # Derived from the observed dispatch with strict slack on every ramping
    # row, so the ramping duals vanish at the observation.
    ramp_limits = [let deltas = [abs(dispatch[t + 1, g] - dispatch[t, g]) for t in 1:T-1]
                       needed = isempty(deltas) ? 0.0 : maximum(deltas)
                       physical = 0.20 * capacities[g] * rand(rng, Uniform(0.8, 1.5))
                       round(max(physical, 1.15 * needed, 0.1); digits=1)
                   end for g in 1:G]
    for g in 1:G
        for t in 1:T-1
            ramp_limits[g] > abs(dispatch[t + 1, g] - dispatch[t, g]) ||
                (ramp_limits[g] = round(abs(dispatch[t + 1, g] - dispatch[t, g]) + 0.1; digits=1))
        end
    end

    ramp_pairs = Tuple{Int,Int}[]
    if T > 1 && num_ramp_pairs > 0
        all_pairs = collect((g, t) for g in 1:G for t in 1:T-1)
        ramp_pairs = sort!(sample(rng, all_pairs, min(num_ramp_pairs, length(all_pairs));
                                  replace=false))
    end

    # --- Witness -------------------------------------------------------------
    if feasibility_status == feasible
        # The clearing price of each period is the offer of its marginal unit —
        # the most expensive unit the stack actually dispatches.
        energy_duals = [maximum(offer[g] for g in 1:G if dispatch[t, g] > 0.0)
                        for t in 1:T]
        capacity_duals = [max(0.0, energy_duals[t] - offer[g]) for t in 1:T, g in 1:G]
        feasible_witness = DispatchPricingWitness(copy(offer), energy_duals, capacity_duals)
    end

    deviation_weights = 1.0 ./ (0.10 .* max.(prior_cost, 1.0))
    deviation_weights .*= G / sum(deviation_weights)

    return InverseDispatchCostProblem(T, G, capacities, demands, ramp_limits,
                                      ramp_pairs, dispatch, types,
                                      prior_cost, cost_lower, cost_upper,
                                      deviation_weights,
                                      feasible_witness, certificate,
                                      feasibility_status)
end

"""
    build_model(prob::InverseDispatchCostProblem)

Build the market offer-cost inference LP. Deterministic — uses only struct
data.
"""
function build_model(prob::InverseDispatchCostProblem)
    model = Model()
    T, G = prob.num_periods, prob.num_units

    @variable(model, prob.cost_lower[g] <= c[g=1:G] <= prob.cost_upper[g])
    @variable(model, energy_dual[1:T] >= 0)
    @variable(model, cap_dual[1:T, 1:G] >= 0)
    ramp_set = Set(prob.ramp_pairs)
    @variable(model, ramp_up[p = ramp_set] >= 0)
    @variable(model, ramp_down[p = ramp_set] >= 0)
    @variable(model, dev_plus[1:G] >= 0)
    @variable(model, dev_minus[1:G] >= 0)

    # Stationarity of every (period, unit) pair against the inferred offers.
    for t in 1:T, g in 1:G
        expr = energy_dual[t] - cap_dual[t, g] - c[g]
        (g, t) in ramp_set && (expr += ramp_up[(g, t)] - ramp_down[(g, t)])
        t > 1 && (g, t - 1) in ramp_set &&
            (expr += ramp_down[(g, t - 1)] - ramp_up[(g, t - 1)])
        @constraint(model, expr <= 0.0)
    end

    # Strong duality between the market's primal (dispatch cost) and dual
    # (price) sides pins the observed dispatch as the clearing outcome.
    @constraint(model,
                sum(prob.demands[t] * energy_dual[t] for t in 1:T) -
                sum(prob.capacities[g] * cap_dual[t, g] for t in 1:T, g in 1:G) -
                sum(prob.ramp_limits[g] * (ramp_up[(g, t)] + ramp_down[(g, t)])
                    for (g, t) in prob.ramp_pairs) ==
                sum(prob.observed_dispatch[t, g] * c[g] for t in 1:T, g in 1:G))

    for g in 1:G
        @constraint(model, c[g] - dev_plus[g] + dev_minus[g] == prob.prior_cost[g])
    end

    @objective(model, Min,
               sum(prob.deviation_weights[g] * (dev_plus[g] + dev_minus[g])
                   for g in 1:G))

    return model
end

register_variant(
    :inverse_optimization,
    :market_clearing,
    InverseDispatchCostProblem,
    "Copper-plate market offer-cost inference: recover generator costs closest to a prior that explain an observed multi-period dispatch",
)
