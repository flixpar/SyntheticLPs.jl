using JuMP
using Random
using Distributions

"""
Maximum accepted variable target for `process_planning/campaign`.

Campaign-planning models in the process industries are medium-sized even at
the complex level (a six-chain mega-site over weekly periods stays near
twenty thousand variables), and the campaign schedule, material flows, and
witness are mirrored across several parallel arrays. Targets above 20,000
would require dimension combinations the chain library cannot honestly
produce, so they are rejected rather than silently under-sized.
"""
const MAX_CAMPAIGN_PLANNING_VARIABLES = 20_000

"""
A complete primal point of the (unrelaxed, binary) campaign model: task
throughput rates, campaign selectors and starts, tiered raw-material
purchases, material inventories, and final sales for every period. The task
rates follow the planted campaign blocks exactly, so feasibility can be
re-checked by pure arithmetic against every row - material balances, unit
capacity with campaign exclusivity, minimum campaign rate and length - and
by `primal_feasibility_report` in the category tests.
"""
struct CampaignScheduleWitness
    rate::Matrix{Float64}        # [task, period]
    active::Matrix{Float64}      # [campaign task, period], y in {0,1}
    starts::Matrix{Float64}      # [campaign task, period]
    purchase::Array{Float64,3}   # [raw material, tier, period]
    inventory::Matrix{Float64}   # [material, period]
    sales::Matrix{Float64}       # [final material, period]
end

"""
Bottleneck refutation for an infeasible instance. Final material `material`
must ship at least `demand` tonnes over periods `1:horizon`, while every row
of the model caps shipments at `initial_inventory + task_bound` tonnes: the
task that produces the material cannot exceed its unit capacity in any
period (`rate <= capacity * active <= capacity`, valid for `active` relaxed
to `[0, 1]`), and the raw-material side cannot supply more than
`raw_bound` tonnes of feed. Both bounds are aggregations of linear rows and
variable bounds only, so the certificate refutes the LP relaxation of the
campaign model as well as the integer model.
"""
struct CampaignCapacityCertificate
    material::Int
    horizon::Int
    demand::Float64
    initial_inventory::Float64
    task_bound::Float64
    raw_bound::Float64
    upper_bound::Float64
    margin::Float64
end

"""
The correlated market condition applied to an unknown-status instance:
contract and spot purchase ceilings and every unit capacity scale by
`supply_factor`, term sales floors by `demand_factor`, with the factors
positioned along the band by the golden-ratio sequence of the seed so blocks
of seeds produce a genuine feasibility mix.
"""
struct CampaignMarketScenario
    supply_factor::Float64
    demand_factor::Float64
    position::Float64
end

"""
    CampaignPlanningProblem <: ProblemGenerator

Multi-period production planning for continuous chemical process plants in
the state-task-network style used for medium-term process-industry
planning: a complex of petrochemical chains (vinyls, aromatics, polyolefins,
polyester, C1 chemistry, and nitrogen fertilisers) buys raw materials on
tiered contract/spot terms, runs conversion tasks on processing units,
hosts several product grades as campaigns on shared trains, stores
intermediates and products, and sells against seasonal term and spot
demand. The objective maximises operating margin net of changeovers.

Structural pieces, drawn from published process-industry planning models:

- state-task network: tasks consume and produce materials in fixed
  stoichiometric proportions (multi-input tasks such as PET from PTA and
  MEG, multi-output tasks such as cumene oxidation yielding phenol and
  acetone), each task running on one processing unit;
- campaign operation: product grades sharing a train are run as campaigns -
  binary selectors with unit exclusivity, a minimum turndown rate, a minimum
  campaign length, and changeover penalties - while single-task units run
  as plain continuous capacity (some instances therefore contain no
  integers at all);
- tiered raw-material purchasing: contract quota at a discount, then spot
  and premium tiers at increasing marginal prices, a convex piecewise-linear
  purchase cost the LP can exploit directly;
- planned turnarounds that remove part of a unit's capacity for a window,
  seasonal demand with product-class phases (construction polymers peak in
  the paving season, fertilisers in spring), and storage tanks with holding
  costs that absorb the swings.

`feasible_witness` is populated only for a requested-feasible instance,
`infeasibility_certificate` only for a requested-infeasible one, and
`market_scenario` only for an unknown-status sample. `build_model` is
deterministic. With the default `relax_integer = true` the campaign
selectors relax to `[0, 1]` and the model is a pure LP.

The exact variable count is
`n_periods * (n_tasks + 2*n_campaign_tasks + n_raws*n_tiers +
n_materials + n_finals)` and the row count is
`n_periods * (n_materials + n_units + n_campaign_units +
4*n_campaign_tasks) - n_campaign_tasks * (campaign_length - 1)`.
Targets through `MAX_CAMPAIGN_PLANNING_VARIABLES` are supported; larger
targets raise `ArgumentError`.
"""
struct CampaignPlanningProblem <: ProblemGenerator
    n_periods::Int
    material_names::Vector{Symbol}
    material_kind::Vector{Symbol}   # :raw, :inter, or :final
    task_names::Vector{Symbol}
    task_unit::Vector{Int}
    task_inputs::Vector{Vector{Tuple{Int,Float64}}}
    task_outputs::Vector{Vector{Tuple{Int,Float64}}}
    task_cost::Vector{Float64}
    unit_names::Vector{Symbol}
    unit_capacity::Matrix{Float64}  # [unit, period], turnaround-adjusted
    campaign_unit::Vector{Bool}
    campaign_length::Int
    min_rate_fraction::Vector{Float64}  # [campaign task]
    n_tiers::Int
    tier_price::Matrix{Float64}     # [raw material, tier]
    tier_cap::Matrix{Float64}       # [raw material, tier]
    material_price::Matrix{Float64} # [final material, period]
    sales_floor::Matrix{Float64}    # [final material, period]
    sales_ceiling::Matrix{Float64}  # [final material, period]
    tank::Vector{Float64}           # [material]
    initial_inventory::Vector{Float64}
    holding_cost::Vector{Float64}
    changeover_cost::Float64
    feasible_witness::Union{Nothing,CampaignScheduleWitness}
    infeasibility_certificate::Union{Nothing,CampaignCapacityCertificate}
    market_scenario::Union{Nothing,CampaignMarketScenario}
    feasibility_status::FeasibilityStatus
end

_cp_variable_count(n_tasks, n_campaign_tasks, n_raws, n_tiers, n_materials,
                   n_finals, n_periods) =
    n_periods * (n_tasks + 2 * n_campaign_tasks + n_raws * n_tiers +
                 n_materials + n_finals)

# Chain library. Each chain: materials with kinds, tasks as
# (name, unit, inputs, outputs, variable cost per tonne of throughput),
# and which units host several grade tasks (campaign trains). Input and
# output coefficients are mass ratios per unit of task throughput with
# realistic yields; shared raw materials (ethylene, propylene, natural gas)
# let chains in one complex draw on common feed markets.
const _CP_CHAINS = Dict{Symbol,Any}()

_cp_register_chain(name::Symbol, materials, kinds, tasks) =
    (_CP_CHAINS[name] = (materials = materials, kinds = kinds, tasks = tasks))

let
    # LDPE tolling plant: the minimal complex.
    _cp_register_chain(
        :ldpe,
        [:ETH, :LDPF, :LDPC],
        [:raw, :final, :final],
        [
            (:ld_film, :LDTRAIN, [Pair(:ETH, 1.015)], [Pair(:LDPF, 0.990)], 95.0),
            (:ld_coating, :LDTRAIN, [Pair(:ETH, 1.015)], [Pair(:LDPC, 0.990)], 105.0),
        ],
    )
    # Vinyls: ethylene and chlorine to EDC, VCM, and three PVC grades.
    _cp_register_chain(
        :vinyls,
        [:ETH, :CL, :EDC, :VCM, :PVCP, :PVCF, :PVCB],
        [:raw, :raw, :inter, :inter, :final, :final, :final],
        [
            (:chlorinate, :CHLOR, [Pair(:ETH, 0.29), Pair(:CL, 0.73)],
             [Pair(:EDC, 0.985)], 42.0),
            (:edc_crack, :CRACK, [Pair(:EDC, 1.00)], [Pair(:VCM, 0.970)], 58.0),
            (:pvc_pipe, :PVCTRAIN, [Pair(:VCM, 1.005)], [Pair(:PVCP, 0.995)], 72.0),
            (:pvc_film, :PVCTRAIN, [Pair(:VCM, 1.005)], [Pair(:PVCF, 0.995)], 74.0),
            (:pvc_bottle, :PVCTRAIN, [Pair(:VCM, 1.005)], [Pair(:PVCB, 0.995)], 78.0),
        ],
    )
    # Aromatics: cumene to phenol with an acetone co-product, then BPA and
    # phenolic resin grades.
    _cp_register_chain(
        :aromatics,
        [:BNZ, :PRP, :CUM, :PHL, :ACT, :BPA, :UFR, :NOV],
        [:raw, :raw, :inter, :final, :final, :final, :final, :final],
        [
            (:cumene, :ALK, [Pair(:BNZ, 0.66), Pair(:PRP, 0.36)],
             [Pair(:CUM, 0.980)], 38.0),
            (:cumene_oxidation, :OXID, [Pair(:CUM, 1.00)],
             [Pair(:PHL, 0.930), Pair(:ACT, 0.600)], 85.0),
            (:bisphenol, :BPAU, [Pair(:PHL, 0.77), Pair(:ACT, 0.28)],
             [Pair(:BPA, 0.960)], 95.0),
            (:resole, :RESINTRAIN, [Pair(:PHL, 1.05)], [Pair(:UFR, 0.975)], 88.0),
            (:novolac, :RESINTRAIN, [Pair(:PHL, 1.00)], [Pair(:NOV, 0.970)], 92.0),
        ],
    )
    # Polyolefins: three independent trains sharing monomer markets.
    _cp_register_chain(
        :polyolefins,
        [:ETH, :PRP, :BUT, :LDPF2, :LDPC2, :LLPF, :LLPP, :PPI, :PPF],
        [:raw, :raw, :raw, :final, :final, :final, :final, :final, :final],
        [
            (:ld2_film, :LDTRAIN2, [Pair(:ETH, 1.015)], [Pair(:LDPF2, 0.990)], 95.0),
            (:ld2_coating, :LDTRAIN2, [Pair(:ETH, 1.015)], [Pair(:LDPC2, 0.990)], 105.0),
            (:lld_film, :LLDTRAIN, [Pair(:ETH, 0.96), Pair(:BUT, 0.05)],
             [Pair(:LLPF, 0.990)], 102.0),
            (:lld_pipe, :LLDTRAIN, [Pair(:ETH, 0.96), Pair(:BUT, 0.05)],
             [Pair(:LLPP, 0.990)], 104.0),
            (:pp_injection, :PPTRAIN, [Pair(:PRP, 1.010)], [Pair(:PPI, 0.990)], 88.0),
            (:pp_fiber, :PPTRAIN, [Pair(:PRP, 1.010)], [Pair(:PPF, 0.990)], 90.0),
        ],
    )
    # Polyester: PX oxidation to PTA (also sold merchant), then PET grades.
    _cp_register_chain(
        :polyester,
        [:PX, :MEG, :PTA, :PETB, :PETF],
        [:raw, :raw, :inter, :final, :final],
        [
            (:px_oxidation, :PTAU, [Pair(:PX, 0.660)], [Pair(:PTA, 0.980)], 62.0),
            (:pet_bottle, :PETTRAIN, [Pair(:PTA, 0.86), Pair(:MEG, 0.33)],
             [Pair(:PETB, 0.990)], 78.0),
            (:pet_fiber, :PETTRAIN, [Pair(:PTA, 0.86), Pair(:MEG, 0.33)],
             [Pair(:PETF, 0.988)], 80.0),
        ],
    )
    # C1 chemistry: natural gas to methanol, acetic acid, vinyl acetate, and
    # PVOH grades; methanol and acid also sold merchant.
    _cp_register_chain(
        :c1,
        [:NG, :CO, :ETH, :MEOH, :AA, :VAM, :POHF, :POHC],
        [:raw, :raw, :raw, :final, :final, :final, :final, :final],
        [
            (:methanol, :MEOHU, [Pair(:NG, 0.780)], [Pair(:MEOH, 0.950)], 55.0),
            (:carbonylation, :ACETU, [Pair(:MEOH, 0.54), Pair(:CO, 0.42)],
             [Pair(:AA, 0.950)], 48.0),
            (:vinylation, :VAMU, [Pair(:AA, 0.62), Pair(:ETH, 0.35)],
             [Pair(:VAM, 0.950)], 60.0),
            (:pvoh_fine, :POHTRAIN, [Pair(:VAM, 1.00)], [Pair(:POHF, 0.940)], 110.0),
            (:pvoh_coarse, :POHTRAIN, [Pair(:VAM, 1.00)], [Pair(:POHC, 0.945)], 105.0),
        ],
    )
    # Nitrogen fertilisers: all single-task units, no campaign structure.
    _cp_register_chain(
        :nitrogen,
        [:NG, :NH3, :UREA, :UAN, :AN],
        [:raw, :inter, :final, :final, :final],
        [
            (:ammonia, :AMMU, [Pair(:NG, 0.620)], [Pair(:NH3, 0.950)], 60.0),
            (:urea, :UREAU, [Pair(:NH3, 0.570)], [Pair(:UREA, 0.990)], 32.0),
            (:uan_blend, :UANU, [Pair(:UREA, 0.36), Pair(:NH3, 0.28)],
             [Pair(:UAN, 0.980)], 18.0),
            (:ammonium_nitrate, :ANU, [Pair(:NH3, 0.430)], [Pair(:AN, 0.960)], 45.0),
        ],
    )
end

# Base raw-material and product prices ($/t) and demand-seasonality class.
const _CP_RAW_PRICE = Dict(
    :ETH => 1050.0, :PRP => 950.0, :BUT => 1100.0, :BNZ => 1000.0,
    :PX => 1050.0, :MEG => 800.0, :NG => 420.0, :CO => 300.0, :CL => 320.0,
)
const _CP_PRODUCT_PRICE = Dict(
    :LDPF => 1250.0, :LDPC => 1280.0, :LDPF2 => 1250.0, :LDPC2 => 1280.0,
    :LLPF => 1270.0, :LLPP => 1290.0, :PPI => 1150.0, :PPF => 1170.0,
    :PVCP => 950.0, :PVCF => 970.0, :PVCB => 990.0,
    :PHL => 1250.0, :ACT => 850.0, :BPA => 1900.0, :UFR => 1450.0,
    :NOV => 1480.0, :PTA => 880.0, :PETB => 1050.0, :PETF => 1020.0,
    :MEOH => 330.0, :AA => 650.0, :VAM => 1050.0, :POHF => 2100.0,
    :POHC => 2050.0, :UREA => 360.0, :UAN => 330.0, :AN => 400.0,
)
const _CP_SEASON = Dict(
    :LDPF => :construction, :LDPC => :construction, :LDPF2 => :construction,
    :LDPC2 => :construction, :LLPF => :construction, :LLPP => :construction,
    :PPI => :construction, :PPF => :construction,
    :PVCP => :construction, :PVCF => :construction, :PVCB => :construction,
    :PHL => :flat, :ACT => :flat, :BPA => :flat, :UFR => :flat,
    :NOV => :flat, :PTA => :flat,
    :PETB => :summer, :PETF => :summer,
    :MEOH => :flat, :AA => :flat, :VAM => :flat, :POHF => :winter,
    :POHC => :winter, :UREA => :spring, :UAN => :spring, :AN => :spring,
)
const _CP_SEASON_SHAPE = Dict(
    :construction => (0.12, 0.22), :summer => (0.08, 0.15),
    :winter => (0.10, 0.20), :spring => (0.15, 0.30), :flat => (0.02, 0.05),
)

"""
Assemble the complex: merge the sampled chains' materials (shared raws by
name), flatten tasks with global indices, and mark units hosting several
tasks as campaign trains.
"""
function _cp_assemble_complex(chain_names::Vector{Symbol})
    material_names = Symbol[]
    material_kind = Symbol[]
    task_names = Symbol[]
    task_unit = Int[]
    task_inputs = Vector{Tuple{Int,Float64}}[]
    task_outputs = Vector{Tuple{Int,Float64}}[]
    task_cost = Float64[]
    unit_names = Symbol[]
    for chain in chain_names
        spec = _CP_CHAINS[chain]
        index = Dict{Symbol,Int}()
        for (m, material) in enumerate(spec.materials)
            existing = findfirst(==(material), material_names)
            if existing === nothing
                push!(material_names, material)
                push!(material_kind, spec.kinds[m])
                existing = length(material_names)
            else
                @assert material_kind[existing] == spec.kinds[m]
            end
            index[material] = existing
        end
        for (tname, uname, inputs, outputs, cost) in spec.tasks
            u = findfirst(==(uname), unit_names)
            u === nothing && (u = length(push!(unit_names, uname)))
            push!(task_names, tname)
            push!(task_unit, u)
            push!(task_inputs,
                  [(index[m], c) for (m, c) in inputs])
            push!(task_outputs,
                  [(index[m], c) for (m, c) in outputs])
            push!(task_cost, cost)
        end
    end
    campaign_unit = [count(==(u), task_unit) > 1 for u in 1:length(unit_names)]
    return material_names, material_kind, task_names, task_unit, task_inputs,
           task_outputs, task_cost, unit_names, campaign_unit
end

_cp_chain_order = [:ldpe, :vinyls, :aromatics, :polyolefins, :polyester, :c1,
                   :nitrogen]

"""
Choose chains, tier count, and horizon from a variable target. Chain
combinations are sampled from the rng and scored by the exact per-period
variable count with the horizon solved in closed form and scanned; the
minimal LDPE plant is always offered so small targets stay reachable.
"""
function _cp_choose_dimensions(rng::AbstractRNG, target_variables::Int)
    target_variables <= MAX_CAMPAIGN_PLANNING_VARIABLES ||
        throw(ArgumentError(
            "process_planning/campaign supports target_variables <= " *
            "$(MAX_CAMPAIGN_PLANNING_VARIABLES); requested $target_variables. " *
            "The six-chain complex over weekly periods tops out below this, " *
            "so a larger target cannot be produced honestly.",
        ))
    target = max(target_variables, 1)

    best = nothing
    best_score = (Inf, Inf, Inf)
    for candidate in 1:7
        minimal = candidate == 7
        if minimal
            chains = [:ldpe]
            n_tiers = 1
        else
            max_chains = target < 150 ? 2 : target < 600 ? 3 :
                         target < 2500 ? 4 : target < 8000 ? 5 : 6
            n_chains = clamp(round(Int, 1 + (max_chains - 1) *
                                   rand(rng, Uniform(0.35, 1.0))), 1, max_chains)
            pool = collect(_cp_chain_order)
            shuffle!(rng, pool)
            chains = pool[1:n_chains]
            n_tiers = target < 90 ? 1 : rand(rng) < 0.4 ? 2 : 3
        end
        material_names, material_kind, task_names, task_unit, task_inputs,
        task_outputs, task_cost, unit_names, campaign_unit =
            _cp_assemble_complex(chains)
        n_raws = count(==(:raw), material_kind)
        n_finals = count(==(:final), material_kind)
        n_campaign = count(task_unit[t] in
                           findall(campaign_unit) for t in eachindex(task_unit))
        per_period = _cp_variable_count(length(task_names), n_campaign, n_raws,
                                        n_tiers, length(material_names),
                                        n_finals, 1)
        t_star = clamp(round(Int, target / per_period), 3, 126)
        for n_periods in max(3, t_star - 2):min(126, t_star + 2)
            size = _cp_variable_count(length(task_names), n_campaign, n_raws,
                                      n_tiers, length(material_names),
                                      n_finals, n_periods)
            error = abs(size - target) / target
            shape = abs(log(n_periods / clamp(4 * sqrt(target / 200), 3, 52)))
            score = (error, shape, rand(rng))
            if score < best_score
                best_score = score
                best = (chains, n_tiers, n_periods)
            end
        end
    end
    return best
end

"""
Seasonal demand deviation per final material, normalised to mean one over
the horizon with a positive floor, following the material's demand class
(construction polymers, spring fertilisers, summer beverage packaging,
winter adhesives).
"""
function _cp_demand_deviation(rng::AbstractRNG, finals::Vector{Int},
                              material_names::Vector{Symbol}, n_periods::Int)
    peak = clamp(round(Int, n_periods * rand(rng, Uniform(0.3, 0.6))), 1, n_periods)
    delta = ones(Float64, length(finals), n_periods)
    for (i, m) in enumerate(finals)
        amp = rand(rng, Uniform(_CP_SEASON_SHAPE[_CP_SEASON[material_names[m]]]...))
        cls = _CP_SEASON[material_names[m]]
        phase = cls == :winter ? peak + n_periods ÷ 2 :
                cls == :spring ? peak - n_periods ÷ 4 :
                cls == :summer ? peak : peak + n_periods ÷ 3
        delta[i, :] .= _pp_seasonal_deviation(rng, amp, phase, n_periods)
    end
    return delta
end

function CampaignPlanningProblem(
    target_variables::Int,
    feasibility_status::FeasibilityStatus,
    seed::Int,
)
    rng = MersenneTwister(seed)
    target = max(target_variables, 1)
    chains, n_tiers, n_periods = _cp_choose_dimensions(rng, target)
    T = n_periods
    material_names, material_kind, task_names, task_unit, task_inputs,
    task_outputs, task_cost, unit_names, campaign_unit = _cp_assemble_complex(chains)
    M = length(material_names)
    NT = length(task_names)
    U = length(unit_names)
    campaign_tasks = [t for t in 1:NT if campaign_unit[task_unit[t]]]
    cT = length(campaign_tasks)
    campaign_index = Dict(t => i for (i, t) in enumerate(campaign_tasks))
    raws = [m for m in 1:M if material_kind[m] == :raw]
    finals = [m for m in 1:M if material_kind[m] == :final]
    campaign_length = rand(rng) < 0.5 ? 2 : 3

    # Campaign blocks per train: shuffled grade order, blocks of
    # campaign_length to campaign_length + 2 periods, the last block
    # extended to the horizon, grades that do not fit stay idle.
    active = zeros(Float64, cT, T)
    for u in 1:U
        campaign_unit[u] || continue
        train_tasks = shuffle!(rng, [t for t in 1:NT if task_unit[t] == u])
        slack = max(0, T ÷ max(1, length(train_tasks)) - campaign_length)
        cursor = 1
        blocks = Vector{Pair{Int,UnitRange{Int}}}()
        for task in train_tasks
            length_blk = campaign_length + rand(rng, 0:slack)
            cursor + length_blk - 1 > T && break
            push!(blocks, task => cursor:(cursor + length_blk - 1))
            cursor += length_blk
        end
        isempty(blocks) && push!(blocks, train_tasks[1] => 1:T)
        blocks[end] = blocks[end][1] => first(blocks[end][2]):T
        for (task, window) in blocks
            active[campaign_index[task], window] .= 1.0
        end
    end
    starts = zeros(Float64, cT, T)
    for i in 1:cT, t in 1:T
        starts[i, t] = t == 1 ? active[i, 1] :
                       max(0.0, active[i, t] - active[i, t - 1])
    end

    # Turnaround windows on single-task units whose rates are driven by
    # their own nominal scale - i.e. every task on the unit feeds only on
    # raw materials. Trains keep flat capacity so campaign blocks stay
    # valid, and inter-fed units track upstream production exactly.
    turnaround = ones(Float64, U, T)
    flexible = [u for u in 1:U if !campaign_unit[u] && all(
        material_kind[mm] == :raw
        for t in 1:NT if task_unit[t] == u for (mm, _) in task_inputs[t])]
    for _ in 1:(rand(rng) < 0.45 ? 0 : rand(rng) < 0.75 ? 1 : 2)
        isempty(flexible) && break
        u = rand(rng, flexible)
        start = rand(rng, 1:T)
        len = rand(rng, 2:min(4, T))
        turnaround[u, start:min(start + len - 1, T)] .= rand(rng, Uniform(0.5, 0.75))
    end

    # Feed allocation: for every intermediate material, decide what share of
    # its production each downstream UNIT may take; a campaign train draws
    # its whole share through whichever grade is active. The last internal
    # consumer absorbs the remainder unless the material is also sold
    # merchant, in which case the remainder is the sales plan.
    downstream = Dict{Int,Vector{Int}}()
    for t in 1:NT, (m, _) in task_inputs[t]
        push!(get!(downstream, m, Int[]), t)
    end
    allocated = Dict{Tuple{Int,Int},Float64}()  # (material, unit) => share
    merchant_share = Dict{Int,Float64}()
    for m in 1:M
        haskey(downstream, m) || continue
        consumer_units = unique(task_unit[t] for t in downstream[m])
        remaining = 1.0
        for (i, u) in enumerate(consumer_units)
            last = i == length(consumer_units)
            if last && material_kind[m] != :final
                share = remaining
            elseif last
                share = remaining * rand(rng, Uniform(0.5, 0.95))
            else
                share = remaining * rand(rng, Uniform(0.35, 0.75))
            end
            allocated[(m, u)] = share
            remaining -= share
        end
        merchant_share[m] = remaining
    end

    # Reference plan: nominal plant scales (kt per period) drawn per unit so
    # the grades sharing a train run at comparable levels, then propagated
    # through the network.
    nominal = zeros(Float64, NT, T)
    unit_base = [rand(rng, Uniform(25.0, 220.0)) for _ in 1:U]
    unit_phase = [rand(rng, Uniform(0, 2π)) for _ in 1:U]
    for t in 1:NT
        base = unit_base[task_unit[t]]
        for τ in 1:T
            nominal[t, τ] = base * (1 + 0.05 * sin(2π * τ / T +
                               unit_phase[task_unit[t]])) *
                            rand(rng, Uniform(0.97, 1.03))
        end
    end
    production = zeros(Float64, M, T)
    consumption = zeros(Float64, M, T)
    rate = zeros(Float64, NT, T)
    for t in 1:NT
        u = task_unit[t]
        inter_inputs = [(m, c) for (m, c) in task_inputs[t]
                        if material_kind[m] == :inter]
        for τ in 1:T
            if campaign_unit[u]
                i = campaign_index[t]
                if active[i, τ] > 0
                    if isempty(inter_inputs)
                        planned = nominal[t, τ] * rand(rng, Uniform(0.6, 0.85))
                    else
                        m, c = inter_inputs[1]
                        planned = allocated[(m, u)] * production[m, τ] / c
                    end
                    rate[t, τ] = planned
                end
            elseif !isempty(inter_inputs)
                m, c = inter_inputs[1]
                rate[t, τ] = allocated[(m, u)] * production[m, τ] / c
            else
                rate[t, τ] = nominal[t, τ] * rand(rng, Uniform(0.6, 0.85)) *
                             turnaround[u, τ]
            end
        end
        for (m, c) in task_inputs[t], τ in 1:T
            consumption[m, τ] += c * rate[t, τ]
        end
        for (m, c) in task_outputs[t], τ in 1:T
            production[m, τ] += c * rate[t, τ]
        end
    end

    unit_capacity = zeros(Float64, U, T)
    for u in 1:U
        load = zeros(Float64, T)
        for t in 1:NT
            task_unit[t] == u && (load .+= rate[t, :])
        end
        peak = maximum(load)
        headroom = rand(rng, Uniform(1.08, 1.3))
        for τ in 1:T
            # The per-period utilisation draws spread by up to 0.85/0.6 =
            # 1.42x, beyond any headroom, so the planned load itself is a
            # floor on capacity: the witness must clear every capacity row.
            unit_capacity[u, τ] = max(load[τ] * 1.02,
                                      peak * headroom * turnaround[u, τ])
        end
    end
    min_rate_fraction = [rand(rng, Uniform(0.12, 0.30)) for _ in 1:cT]
    # The turndown fraction must also clear the planned dip of its own
    # campaign: capacity is sized from the peak train load, and an upstream
    # turnaround can pull the planned rate of an inter-fed train below 30% of
    # that peak. Cap the drawn fraction by the realised minimum
    # rate-to-capacity ratio (with slack) so the witness always satisfies the
    # minimum-turndown rows.
    for (i, t) in enumerate(campaign_tasks)
        lo = Inf
        for τ in 1:T
            active[i, τ] > 0 &&
                (lo = min(lo, rate[t, τ] / unit_capacity[task_unit[t], τ]))
        end
        isfinite(lo) &&
            (min_rate_fraction[i] = min(min_rate_fraction[i], 0.98 * lo))
    end

    # Merchant sales and inventory trajectories.
    delta = _cp_demand_deviation(rng, finals, material_names, T)
    sales_plan = zeros(Float64, M, T)
    for (i, m) in enumerate(finals)
        share = get(merchant_share, m, 1.0)
        sales_plan[m, :] .= share .* production[m, :] .* delta[i, :]
    end

    initial_inventory = zeros(Float64, M)
    tank = zeros(Float64, M)
    purchase_plan = zeros(Float64, M, n_tiers, T)
    tier_price = zeros(Float64, M, n_tiers)
    tier_cap = zeros(Float64, M, n_tiers)
    for m in 1:M
        net = production[m, :] .- consumption[m, :] .- sales_plan[m, :]
        if material_kind[m] == :raw
            # Purchases exactly cover consumption; tiers split the volume.
            need = consumption[m, :]
            tier_cap[m, 1] = isempty(need) ? 0.0 :
                             n_tiers == 1 ?
                             maximum(need) * rand(rng, Uniform(1.05, 1.20)) :
                             minimum(need) * rand(rng, Uniform(0.7, 1.0))
            if n_tiers >= 2
                tier_cap[m, 2] = max(0.0, maximum(need) * 1.25 - tier_cap[m, 1]) *
                                 rand(rng, Uniform(0.6, 1.0))
            end
            if n_tiers >= 3
                tier_cap[m, 3] = max(0.0, maximum(need) * 1.45 -
                                     tier_cap[m, 1] - tier_cap[m, 2])
            end
            for τ in 1:T
                remaining = need[τ]
                for j in 1:n_tiers
                    take = min(remaining, tier_cap[m, j])
                    purchase_plan[m, j, τ] = take
                    remaining -= take
                end
            end
            base = _CP_RAW_PRICE[material_names[m]] * rand(rng, LogNormal(0, 0.12))
            tier_price[m, 1] = base * rand(rng, Uniform(0.85, 0.95))
            n_tiers >= 2 && (tier_price[m, 2] = base * rand(rng, Uniform(1.05, 1.25)))
            n_tiers >= 3 && (tier_price[m, 3] = base * rand(rng, Uniform(1.30, 1.60)))
            initial_inventory[m] = mean(need) * rand(rng, Uniform(0.05, 0.15))
            net = vec(sum(purchase_plan[m, :, :]; dims = 1)) .- need
        end
        cum = cumsum(net)
        mean_net = sum(abs.(net)) / T
        initial_inventory[m] += -min(0.0, minimum(cum)) + 0.10 * mean_net + 0.05
        tank[m] = max(initial_inventory[m] + max(0.0, maximum(cum)) +
                      0.15 * mean_net + 0.05,
                      1.1 * initial_inventory[m])
    end

    inventory_plan = zeros(Float64, M, T)
    previous = copy(initial_inventory)
    for m in 1:M, τ in 1:T
        inventory_plan[m, τ] = previous[m] +
                               (material_kind[m] == :raw ?
                                sum(purchase_plan[m, :, τ]) :
                                production[m, τ] - sales_plan[m, τ]) -
                               consumption[m, τ]
        previous[m] = inventory_plan[m, τ]
    end

    material_price = zeros(Float64, M, T)
    for (i, m) in enumerate(finals)
        base = _CP_PRODUCT_PRICE[material_names[m]] * rand(rng, LogNormal(0, 0.10))
        for τ in 1:T
            material_price[m, τ] = base * (1 + 0.35 * (delta[i, τ] - 1)) *
                                   rand(rng, Uniform(0.98, 1.02))
        end
    end
    sales_floor = zeros(Float64, M, T)
    sales_ceiling = zeros(Float64, M, T)
    for m in finals
        λ = rand(rng, Uniform(0.4, 0.9))
        sales_floor[m, :] .= λ .* sales_plan[m, :]
        sales_ceiling[m, :] .= sales_plan[m, :] .*
                               (1 .+ rand(rng, Uniform(0.10, 0.40), T))
    end
    holding_cost = rand(rng, Uniform(2.0, 8.0), M)
    changeover_cost = rand(rng, Uniform(30.0, 150.0))

    feasible_witness = nothing
    infeasibility_certificate = nothing
    market_scenario = nothing

    if feasibility_status == feasible
        feasible_witness = CampaignScheduleWitness(
            rate, active, starts, purchase_plan, inventory_plan, sales_plan)
    elseif feasibility_status == infeasible
        horizon = clamp(round(Int, T * rand(rng, Uniform(0.55, 1.0))), 2, T)
        candidates = Tuple{Int,Float64}[]
        for m in finals
            producer = findfirst(t -> any(o == m for (o, _) in task_outputs[t]),
                                 1:NT)
            producer === nothing && continue
            out_coeff = sum(c for (o, c) in task_outputs[producer] if o == m)
            task_cap = out_coeff * sum(unit_capacity[task_unit[producer], 1:horizon])
            raw_inputs = [(mm, c) for (mm, c) in task_inputs[producer]
                          if material_kind[mm] == :raw]
            raw_cap = isempty(raw_inputs) ? Inf : minimum(
                (horizon * sum(tier_cap[mm, :]) + initial_inventory[mm]) / c
                for (mm, c) in raw_inputs)
            bound = min(task_cap, raw_cap)
            demand = sum(sales_floor[m, 1:horizon])
            bound > eps() && demand > eps() && push!(candidates, (m, demand / bound))
        end
        isempty(candidates) && (candidates = [(finals[1], 1.0)])
        sort!(candidates; by = x -> -x[2])
        cut_material = candidates[rand(rng, 1:min(3, length(candidates)))][1]
        producer = findfirst(t -> any(o == cut_material
                                      for (o, _) in task_outputs[t]), 1:NT)
        out_coeff = sum(c for (o, c) in task_outputs[producer] if o == cut_material)
        raw_inputs = [(mm, c) for (mm, c) in task_inputs[producer]
                      if material_kind[mm] == :raw]
        initial_inventory[cut_material] *= rand(rng, Uniform(0.05, 0.20))
        for (mm, _) in raw_inputs
            initial_inventory[mm] *= rand(rng, Uniform(0.02, 0.15))
        end
        demand_raise = rand(rng, Uniform(1.05, 1.25))
        sales_floor[cut_material, 1:horizon] .*= demand_raise
        sales_floor[cut_material, 1:horizon] .=
            min.(sales_floor[cut_material, 1:horizon],
                 0.98 .* sales_ceiling[cut_material, 1:horizon])
        demand_cum = sum(sales_floor[cut_material, 1:horizon])
        desired_upper = demand_cum * rand(rng, Uniform(0.60, 0.85))
        # A bursty campaign product can carry planted initial stock above the
        # shrunken demand target; cap it so the supply cut stays below demand.
        # Otherwise the scale numerator goes negative, `scale` clamps at its
        # floor, and the margin below turns negative.
        initial_inventory[cut_material] =
            min(initial_inventory[cut_material], 0.4 * desired_upper)
        task_bound_raw = out_coeff * sum(unit_capacity[task_unit[producer], :])
        scale = clamp((desired_upper - initial_inventory[cut_material]) /
                      max(task_bound_raw, eps()), 0.01, 0.90)
        unit_capacity[task_unit[producer], :] .*= scale
        for (mm, _) in raw_inputs
            tier_cap[mm, :] .*= scale
        end
        task_bound = out_coeff * sum(unit_capacity[task_unit[producer], 1:horizon])
        raw_bound = isempty(raw_inputs) ? Inf : minimum(
            (horizon * sum(tier_cap[mm, :]) + initial_inventory[mm]) / c
            for (mm, c) in raw_inputs)
        upper_bound = initial_inventory[cut_material] +
                      min(task_bound, raw_bound)
        certificate_margin = demand_cum - upper_bound
        @assert certificate_margin > 1e-6
        infeasibility_certificate = CampaignCapacityCertificate(
            cut_material, horizon, demand_cum,
            initial_inventory[cut_material], task_bound, raw_bound,
            upper_bound, certificate_margin)
    else
        position = _pp_seed_position(seed)
        supply_factor = 0.55 + 0.34 * position
        demand_factor = 1 + 0.25 * (0.5 - position)
        tier_cap .*= supply_factor
        unit_capacity .*= supply_factor
        sales_floor .*= demand_factor
        sales_ceiling .= max.(sales_ceiling, sales_floor)
        market_scenario = CampaignMarketScenario(
            supply_factor, demand_factor, position)
    end

    return CampaignPlanningProblem(
        T, material_names, material_kind, task_names, task_unit, task_inputs,
        task_outputs, task_cost, unit_names, unit_capacity, campaign_unit,
        campaign_length, min_rate_fraction, n_tiers, tier_price, tier_cap,
        material_price, sales_floor, sales_ceiling, tank, initial_inventory,
        holding_cost, changeover_cost, feasible_witness,
        infeasibility_certificate, market_scenario, feasibility_status,
    )
end

function build_model(prob::CampaignPlanningProblem)
    model = Model()
    T, M, NT = prob.n_periods, length(prob.material_names), length(prob.task_names)
    U = length(prob.unit_names)
    raws = [m for m in 1:M if prob.material_kind[m] == :raw]
    finals = [m for m in 1:M if prob.material_kind[m] == :final]
    campaign_tasks = [t for t in 1:NT if prob.campaign_unit[prob.task_unit[t]]]
    cT = length(campaign_tasks)
    campaign_index = Dict(t => i for (i, t) in enumerate(campaign_tasks))

    @variable(model, rate[t = 1:NT, τ = 1:T] >= 0)
    @variable(model, active[i = 1:cT, τ = 1:T], Bin)
    @variable(model, starts[i = 1:cT, τ = 1:T] >= 0)
    @variable(model, 0 <= purchase[m = raws, j = 1:prob.n_tiers, τ = 1:T] <=
              prob.tier_cap[m, j])
    @variable(model, 0 <= inventory[m = 1:M, τ = 1:T] <= prob.tank[m])
    @variable(model, prob.sales_floor[m, τ] <=
              sales[m = finals, τ = 1:T] <= prob.sales_ceiling[m, τ])

    @objective(model, Max,
        sum(prob.material_price[m, τ] * sales[m, τ] for m in finals, τ in 1:T) -
        sum(prob.tier_price[m, j] * purchase[m, j, τ]
            for m in raws, j in 1:prob.n_tiers, τ in 1:T) -
        sum(prob.task_cost[t] * rate[t, τ] for t in 1:NT, τ in 1:T) -
        sum(prob.holding_cost[m] * inventory[m, τ] for m in 1:M, τ in 1:T) -
        prob.changeover_cost * sum(starts))

    for m in 1:M, τ in 1:T
        inflow = AffExpr(0.0)
        if prob.material_kind[m] == :raw
            for j in 1:prob.n_tiers
                add_to_expression!(inflow, 1.0, purchase[m, j, τ])
            end
        end
        for t in 1:NT, (mm, c) in prob.task_inputs[t]
            mm == m && add_to_expression!(inflow, -c, rate[t, τ])
        end
        for t in 1:NT, (mm, c) in prob.task_outputs[t]
            mm == m && add_to_expression!(inflow, c, rate[t, τ])
        end
        if prob.material_kind[m] == :final
            add_to_expression!(inflow, -1.0, sales[m, τ])
        end
        @constraint(model,
            inventory[m, τ] ==
            (τ == 1 ? prob.initial_inventory[m] : inventory[m, τ - 1]) + inflow)
    end

    for u in 1:U, τ in 1:T
        @constraint(model,
            sum(rate[t, τ] for t in 1:NT if prob.task_unit[t] == u) <=
            prob.unit_capacity[u, τ])
    end

    for u in 1:U
        prob.campaign_unit[u] || continue
        for τ in 1:T
            @constraint(model,
                sum(active[campaign_index[t], τ]
                    for t in 1:NT if prob.task_unit[t] == u) <= 1)
        end
    end

    for (i, t) in enumerate(campaign_tasks), τ in 1:T
        cap = prob.unit_capacity[prob.task_unit[t], τ]
        @constraint(model, rate[t, τ] <= cap * active[i, τ])
        @constraint(model,
            rate[t, τ] >= prob.min_rate_fraction[i] * cap * active[i, τ])
        @constraint(model,
            starts[i, τ] >= active[i, τ] - (τ == 1 ? 0 : active[i, τ - 1]))
    end

    L = prob.campaign_length
    for (i, _) in enumerate(campaign_tasks), τ in 1:(T - L + 1)
        @constraint(model,
            sum(active[i, k] for k in τ:(τ + L - 1)) >=
            L * (active[i, τ] - (τ == 1 ? 0 : active[i, τ - 1])))
    end
    return model
end

register_variant(
    :process_planning,
    :campaign,
    CampaignPlanningProblem,
    "Multi-period petrochemical campaign-planning MIP over state-task networks: tiered raw purchasing, shared trains running product grades as campaigns with minimum lengths and changeovers, storage, and seasonal term and spot demand",
)
