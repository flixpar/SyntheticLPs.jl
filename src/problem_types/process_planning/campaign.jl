using JuMP
using Random
using Distributions

"""
Maximum accepted variable target for `process_planning/campaign`.

A single chemical complex over weekly periods tops out near twenty thousand
variables, so larger requests are served by a multi-site company: several
complexes, each running its own chain portfolio, coupled through shared regional
feedstock supply and shared product markets. The company grows linearly with the
target; above one million variables the per-site data, witness and incidence
lists would need a multi-gigabyte working set, so larger targets raise
`ArgumentError` (the package's sizing-cap convention).
"""
const MAX_CAMPAIGN_PLANNING_VARIABLES = 1_000_000

"""
Targets up to this many variables are served by a single complex; larger ones by
a multi-site company (see [`MAX_CAMPAIGN_PLANNING_VARIABLES`](@ref)).
"""
const _CP_SINGLE_SITE_LIMIT = 16_000

"""
A complete primal point of the (unrelaxed, binary) campaign model: task
throughput rates, campaign selectors and starts, tiered raw-material
purchases, material inventories, and final sales for every period. The task
rates follow the planted campaign blocks exactly, so feasibility can be
re-checked by pure arithmetic against every row - material balances, unit
capacity with campaign exclusivity, minimum campaign rate and length, the
shared feedstock pools and product markets - by
[`campaign_plan_satisfies`](@ref), with no solver involved.
"""
struct CampaignScheduleWitness
    rate::Matrix{Float64}        # [task, period]
    active::Matrix{Float64}      # [campaign task, period], y in {0,1}
    starts::Matrix{Float64}      # [campaign task, period]
    purchase::Array{Float64, 3}   # [material, tier, period] (zero for non-raws)
    inventory::Matrix{Float64}   # [material, period]
    sales::Matrix{Float64}       # [material, period] (zero for non-finals)
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
campaign model as well as the integer model. Single-product bottlenecks are
easy for presolve to spot by bound propagation along the product's inventory
chain; this is the minority infeasibility mode.
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
Feedstock-pool refutation (the default infeasibility mode): the contracted
term sales of the whole company need more of one shared raw material than the
regional pool can supply over the horizon.

`potential[m]` is the least amount of the pooled raw material (`group`) that one
tonne of material `m` must contain: one on the pooled raw at every site, zero
on every other raw, and for each task its feed potential spread over its output
mass, so no task creates pooled-raw content (`sum_out c y <= sum_in c y`).
Multiplying every material balance by `potential` and summing over materials
and periods cancels the task rates and leaves

    sum_f potential[f] * sales[f, 1:T]
        <= sum_m potential[m] * initial_inventory[m] + sum_t pool supply[t]

because closing inventories are nonnegative and the pooled purchases in period
`t` cannot exceed `min(supply_cap[group, t], sum of the pool members' tier
caps)`. The sales floors make the left side at least `demand`, which exceeds
`upper_bound` by `margin`. The argument combines every product, every site and
every period that draws on the pool, so no single row or bound propagation
along one product chain exposes it; only the LP aggregation does.
"""
struct CampaignFeedstockCertificate
    group::Int
    potential::Vector{Float64}
    demand::Float64
    inventory_bound::Float64
    supply_bound::Float64
    upper_bound::Float64
    margin::Float64
end

"""
The correlated market condition applied to an unknown-status instance. Every
local row is placed around a planted schedule exactly as for a requested-feasible
instance; then every regional feedstock pool is set between its critical level
`kappa_star` (the supply at which the pool's potential bound exactly meets the
term contracts) and the planted schedule's own purchases: a pool's supply is
`kappa_star + (1 - kappa_star) * supply_factor` times the planted purchases
(with a small per-pool jitter). A negative `supply_factor` puts a pool below its
critical level, so the instance is infeasible by the feedstock-pool argument;
near one the planted schedule nearly fits; in between, whether capacity,
campaign, co-product and inventory rows still let the contracts be served is
genuinely open. `supply_factor` follows the golden-ratio position of the seed,
so blocks of seeds produce a genuine feasibility mix.
"""
struct CampaignMarketScenario
    supply_factor::Float64
    position::Float64
end

"""
    CampaignPlanningProblem <: ProblemGenerator

Multi-period production planning for continuous chemical process plants in
the state-task-network style used for medium-term process-industry
planning: one or several complexes of petrochemical chains (LDPE, vinyls,
aromatics, polyolefins, polyester, C1 chemistry, and nitrogen fertilisers) buy
raw materials on tiered contract/spot terms from shared regional feedstock
pools, run conversion tasks on processing units, host several product grades
as campaigns on shared trains, store intermediates and products, and sell
against seasonal term and spot demand into shared regional markets. The
objective maximises operating margin net of changeovers.

Structural pieces, drawn from published process-industry planning models:

- state-task network: tasks consume and produce materials in fixed
  stoichiometric proportions (multi-input tasks such as PET from PTA and
  MEG, multi-output tasks such as cumene oxidation yielding phenol and
  acetone), each task running on one processing unit;
- campaign operation: product grades sharing a train are run as campaigns -
  binary selectors with unit exclusivity, a minimum turndown rate, a minimum
  campaign length, and changeover penalties - while single-task units run
  as plain continuous capacity (a variable bound on the task rate);
- tiered raw-material purchasing: contract quota at a discount, then spot
  and premium tiers at increasing marginal prices, a convex piecewise-linear
  purchase cost the LP can exploit directly;
- multi-site coupling: every raw material is drawn from a regional pool (one
  row per raw and period summing the purchases of every site that uses it), and
  a finished grade made at several sites is sold into one regional market (one
  ceiling row per grade and period across those sites);
- planned turnarounds that remove part of a unit's capacity for a window,
  seasonal demand with product-class phases (construction polymers peak in
  the paving season, fertilisers in spring), and storage tanks with holding
  costs that absorb the swings.

Targets up to `_CP_SINGLE_SITE_LIMIT` are served by one complex; larger ones by
a company of several complexes over a longer horizon (up to three years of
weeks), so the model grows linearly to `MAX_CAMPAIGN_PLANNING_VARIABLES`.

`feasible_witness` is populated only for a requested-feasible instance,
`infeasibility_certificate` only for a requested-infeasible one, and
`market_scenario` only for an unknown-status sample. `build_model` is
deterministic. With the default `relax_integer = true` the campaign
selectors relax to `[0, 1]` and the model is a pure LP.

The exact variable count is
`n_periods * (n_tasks + 2*n_campaign_tasks + n_raws*n_tiers +
n_materials + n_finals)`; see [`campaign_row_count`](@ref) for the rows.
Late starts are fixed to zero, so the minimum campaign duration cannot be
escaped at the horizon boundary.
"""
struct CampaignPlanningProblem <: ProblemGenerator
    n_periods::Int
    period_days::Float64
    n_sites::Int
    material_names::Vector{Symbol}
    material_kind::Vector{Symbol}   # :raw, :inter, or :final
    material_site::Vector{Int}
    task_names::Vector{Symbol}
    task_unit::Vector{Int}
    task_inputs::Vector{Vector{Tuple{Int, Float64}}}
    task_outputs::Vector{Vector{Tuple{Int, Float64}}}
    task_cost::Vector{Float64}
    unit_names::Vector{Symbol}
    unit_site::Vector{Int}
    unit_capacity::Matrix{Float64}  # [unit, period], turnaround-adjusted
    campaign_unit::Vector{Bool}
    campaign_length::Int
    min_rate_fraction::Vector{Float64}  # [campaign task]
    n_tiers::Int
    tier_price::Matrix{Float64}     # [material, tier] (raws only)
    tier_cap::Matrix{Float64}       # [material, tier] (raws only)
    supply_groups::Vector{Vector{Int}}  # raw materials sharing a regional pool
    supply_cap::Matrix{Float64}     # [supply group, period]
    market_groups::Vector{Vector{Int}}  # finals sold into one regional market (>= 2 sites)
    market_cap::Matrix{Float64}     # [market group, period]
    material_price::Matrix{Float64} # [material, period]
    sales_floor::Matrix{Float64}    # [material, period]
    sales_ceiling::Matrix{Float64}  # [material, period]
    tank::Vector{Float64}           # [material]
    initial_inventory::Vector{Float64}
    holding_cost::Vector{Float64}
    changeover_cost::Float64
    feasible_witness::Union{Nothing, CampaignScheduleWitness}
    infeasibility_certificate::Union{
        Nothing, CampaignCapacityCertificate, CampaignFeedstockCertificate
    }
    market_scenario::Union{Nothing, CampaignMarketScenario}
    feasibility_status::FeasibilityStatus
end

_cp_variable_count(n_tasks, n_campaign_tasks, n_raws, n_tiers, n_materials, n_finals, n_periods) =
    n_periods * (n_tasks + 2 * n_campaign_tasks + n_raws * n_tiers + n_materials + n_finals)

"""
    campaign_row_count(prob) -> Int

Exact number of affine rows of the campaign model: a balance per material and
period; an exclusivity row per campaign train and period; per campaign task the
capacity gate and turndown rows in every period, the start definition (one row
in the first period, three after it) and a minimum-run row per admissible start
period; plus one pool row per regional feedstock pool with more than one
purchase variable (a single-variable pool is a bound) and one ceiling row per
multi-site market, per period.
"""
function campaign_row_count(prob::CampaignPlanningProblem)
    T = prob.n_periods
    L = prob.campaign_length
    cT = count(t -> prob.campaign_unit[prob.task_unit[t]], eachindex(prob.task_names))
    per_task = 2T + (1 + 3 * (T - 1)) + max(T - L + 1, 0)
    return T * (
        length(prob.material_names) +
        count(prob.campaign_unit) +
        count(g -> _cp_pool_has_row(prob, g), eachindex(prob.supply_groups)) +
        length(prob.market_groups)
    ) + cT * per_task
end

# Chain library. Each chain: materials with kinds, tasks as
# (name, unit, inputs, outputs, variable cost per tonne of throughput),
# and which units host several grade tasks (campaign trains). Input and
# output coefficients are mass ratios per unit of task throughput with
# realistic yields; shared raw materials (ethylene, propylene, natural gas)
# let chains in one complex draw on common feed markets.
const _CP_CHAINS = Dict{Symbol, Any}()

_cp_register_chain(name::Symbol, materials, kinds, tasks) =
    (_CP_CHAINS[name] = (materials=materials, kinds=kinds, tasks=tasks))

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
            (:chlorinate, :CHLOR, [Pair(:ETH, 0.29), Pair(:CL, 0.73)], [Pair(:EDC, 0.985)], 42.0),
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
            (:cumene, :ALK, [Pair(:BNZ, 0.66), Pair(:PRP, 0.36)], [Pair(:CUM, 0.980)], 38.0),
            (
                :cumene_oxidation,
                :OXID,
                [Pair(:CUM, 1.00)],
                [Pair(:PHL, 0.930), Pair(:ACT, 0.600)],
                85.0,
            ),
            (:bisphenol, :BPAU, [Pair(:PHL, 0.77), Pair(:ACT, 0.28)], [Pair(:BPA, 0.960)], 95.0),
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
            (
                :lld_film,
                :LLDTRAIN,
                [Pair(:ETH, 0.96), Pair(:BUT, 0.05)],
                [Pair(:LLPF, 0.990)],
                102.0,
            ),
            (
                :lld_pipe,
                :LLDTRAIN,
                [Pair(:ETH, 0.96), Pair(:BUT, 0.05)],
                [Pair(:LLPP, 0.990)],
                104.0,
            ),
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
            (
                :pet_bottle,
                :PETTRAIN,
                [Pair(:PTA, 0.86), Pair(:MEG, 0.33)],
                [Pair(:PETB, 0.990)],
                78.0,
            ),
            (
                :pet_fiber,
                :PETTRAIN,
                [Pair(:PTA, 0.86), Pair(:MEG, 0.33)],
                [Pair(:PETF, 0.988)],
                80.0,
            ),
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
            (
                :carbonylation,
                :ACETU,
                [Pair(:MEOH, 0.54), Pair(:CO, 0.42)],
                [Pair(:AA, 0.950)],
                48.0,
            ),
            (:vinylation, :VAMU, [Pair(:AA, 0.62), Pair(:ETH, 0.35)], [Pair(:VAM, 0.950)], 60.0),
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
            (:uan_blend, :UANU, [Pair(:UREA, 0.36), Pair(:NH3, 0.28)], [Pair(:UAN, 0.980)], 18.0),
            (:ammonium_nitrate, :ANU, [Pair(:NH3, 0.430)], [Pair(:AN, 0.960)], 45.0),
        ],
    )
end

# Base raw-material and product prices ($/t) and demand-seasonality class.
const _CP_RAW_PRICE = Dict(
    :ETH => 1050.0,
    :PRP => 950.0,
    :BUT => 1100.0,
    :BNZ => 1000.0,
    :PX => 1050.0,
    :MEG => 800.0,
    :NG => 420.0,
    :CO => 300.0,
    :CL => 320.0,
)
const _CP_PRODUCT_PRICE = Dict(
    :LDPF => 1250.0,
    :LDPC => 1280.0,
    :LDPF2 => 1250.0,
    :LDPC2 => 1280.0,
    :LLPF => 1270.0,
    :LLPP => 1290.0,
    :PPI => 1150.0,
    :PPF => 1170.0,
    :PVCP => 950.0,
    :PVCF => 970.0,
    :PVCB => 990.0,
    :PHL => 1250.0,
    :ACT => 850.0,
    :BPA => 1900.0,
    :UFR => 1450.0,
    :NOV => 1480.0,
    :PTA => 880.0,
    :PETB => 1050.0,
    :PETF => 1020.0,
    :MEOH => 330.0,
    :AA => 650.0,
    :VAM => 1050.0,
    :POHF => 2100.0,
    :POHC => 2050.0,
    :UREA => 360.0,
    :UAN => 330.0,
    :AN => 400.0,
)
const _CP_SEASON = Dict(
    :LDPF => :construction,
    :LDPC => :construction,
    :LDPF2 => :construction,
    :LDPC2 => :construction,
    :LLPF => :construction,
    :LLPP => :construction,
    :PPI => :construction,
    :PPF => :construction,
    :PVCP => :construction,
    :PVCF => :construction,
    :PVCB => :construction,
    :PHL => :flat,
    :ACT => :flat,
    :BPA => :flat,
    :UFR => :flat,
    :NOV => :flat,
    :PTA => :flat,
    :PETB => :summer,
    :PETF => :summer,
    :MEOH => :flat,
    :AA => :flat,
    :VAM => :flat,
    :POHF => :winter,
    :POHC => :winter,
    :UREA => :spring,
    :UAN => :spring,
    :AN => :spring,
)
const _CP_SEASON_SHAPE = Dict(
    :construction => (0.12, 0.22),
    :summer => (0.08, 0.15),
    :winter => (0.10, 0.20),
    :spring => (0.15, 0.30),
    :flat => (0.02, 0.05),
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
    task_inputs = Vector{Tuple{Int, Float64}}[]
    task_outputs = Vector{Tuple{Int, Float64}}[]
    task_cost = Float64[]
    unit_names = Symbol[]
    for chain in chain_names
        spec = _CP_CHAINS[chain]
        index = Dict{Symbol, Int}()
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
            push!(task_inputs, [(index[m], c) for (m, c) in inputs])
            push!(task_outputs, [(index[m], c) for (m, c) in outputs])
            push!(task_cost, cost)
        end
    end
    campaign_unit = [count(==(u), task_unit) > 1 for u in 1:length(unit_names)]
    return material_names,
    material_kind, task_names, task_unit, task_inputs, task_outputs, task_cost, unit_names,
    campaign_unit
end

_cp_chain_order = [:ldpe, :vinyls, :aromatics, :polyolefins, :polyester, :c1, :nitrogen]
"""
Seasonal demand deviation per final material, normalised to mean one over
the horizon with a positive floor, following the material's demand class
(construction polymers, spring fertilisers, summer beverage packaging,
winter adhesives).
"""
function _cp_demand_deviation(
    rng::AbstractRNG, finals::Vector{Int}, material_names::Vector{Symbol}, n_periods::Int
)
    calendar_phase = rand(rng, Uniform(0.0, 2pi))
    delta = ones(Float64, length(finals), n_periods)
    for (i, m) in enumerate(finals)
        amp = rand(rng, Uniform(_CP_SEASON_SHAPE[_CP_SEASON[material_names[m]]]...))
        cls = _CP_SEASON[material_names[m]]
        offset = if cls == :winter
            Float64(pi)
        elseif cls == :spring
            -Float64(pi) / 2
        elseif cls == :construction
            0.35
        else
            0.0
        end
        delta[i, :] .= _pp_seasonal_deviation(
            rng, amp, calendar_phase + offset, n_periods; period_days=7.0
        )
    end
    return delta
end
"""
    _cp_generate_site(rng, chains, n_tiers, T, campaign_length)

Generate one complex: assemble the chain portfolio, plant a campaign schedule and
propagate it through the state-task network into task rates, purchases, sales
and inventories, then place every local capacity, tank, tier and demand window
around that plan. Returns a named tuple of site-local data (indices are local to
the site) together with the planted schedule.
"""
function _cp_generate_site(
    rng::AbstractRNG, chains::Vector{Symbol}, n_tiers::Int, T::Int, campaign_length::Int
)
    material_names, material_kind, task_names, task_unit, task_inputs, task_outputs, task_cost, unit_names, campaign_unit = _cp_assemble_complex(
        chains
    )
    M = length(material_names)
    NT = length(task_names)
    U = length(unit_names)
    campaign_tasks = [t for t in 1:NT if campaign_unit[task_unit[t]]]
    cT = length(campaign_tasks)
    campaign_index = Dict(t => i for (i, t) in enumerate(campaign_tasks))
    finals = [m for m in 1:M if material_kind[m] == :final]

    # Campaign blocks per train: shuffled grade order, blocks of
    # campaign_length to campaign_length + 2 periods, the last block
    # extended to the horizon, grades that do not fit stay idle.
    active = zeros(Float64, cT, T)
    for u in 1:U
        campaign_unit[u] || continue
        train_tasks = shuffle!(rng, [t for t in 1:NT if task_unit[t] == u])
        slack = max(0, T ÷ max(1, length(train_tasks)) - campaign_length)
        cursor = 1
        blocks = Vector{Pair{Int, UnitRange{Int}}}()
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
        starts[i, t] = t == 1 ? active[i, 1] : max(0.0, active[i, t] - active[i, t - 1])
    end

    # Turnaround windows on single-task units whose rates are driven by
    # their own nominal scale - i.e. every task on the unit feeds only on
    # raw materials. Trains keep flat capacity so campaign blocks stay
    # valid, and inter-fed units track upstream production exactly.
    turnaround = ones(Float64, U, T)
    flexible = [
        u for u in 1:U if !campaign_unit[u] && all(
            material_kind[mm] == :raw for t in 1:NT if task_unit[t] == u for
            (mm, _) in task_inputs[t]
        )
    ]
    for _ in 1:(if rand(rng) < 0.45
            0
        elseif rand(rng) < 0.75
            1
        else
            2
        end)
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
    downstream = Dict{Int, Vector{Int}}()
    for t in 1:NT, (m, _) in task_inputs[t]
        push!(get!(downstream, m, Int[]), t)
    end
    allocated = Dict{Tuple{Int, Int}, Float64}()  # (material, unit) => share
    merchant_share = Dict{Int, Float64}()
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
            nominal[t, τ] =
                base *
                (1 + 0.05 * sin(2π * τ / T + unit_phase[task_unit[t]])) *
                rand(rng, Uniform(0.97, 1.03))
        end
    end
    production = zeros(Float64, M, T)
    consumption = zeros(Float64, M, T)
    rate = zeros(Float64, NT, T)
    for t in 1:NT
        u = task_unit[t]
        inter_inputs = [(m, c) for (m, c) in task_inputs[t] if material_kind[m] == :inter]
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
                rate[t, τ] = nominal[t, τ] * rand(rng, Uniform(0.6, 0.85)) * turnaround[u, τ]
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
            unit_capacity[u, τ] = max(load[τ] * 1.02, peak * headroom * turnaround[u, τ])
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
            active[i, τ] > 0 && (lo = min(lo, rate[t, τ] / unit_capacity[task_unit[t], τ]))
        end
        isfinite(lo) && (min_rate_fraction[i] = min(min_rate_fraction[i], 0.98 * lo))
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
            tier_cap[m, 1] = if isempty(need)
                0.0
            elseif n_tiers == 1
                maximum(need) * rand(rng, Uniform(1.05, 1.20))
            else
                minimum(need) * rand(rng, Uniform(0.7, 1.0))
            end
            if n_tiers >= 2
                tier_cap[m, 2] =
                    max(0.0, maximum(need) * 1.25 - tier_cap[m, 1]) * rand(rng, Uniform(0.6, 1.0))
            end
            if n_tiers >= 3
                tier_cap[m, 3] = max(0.0, maximum(need) * 1.45 - tier_cap[m, 1] - tier_cap[m, 2])
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
            net = vec(sum(purchase_plan[m, :, :]; dims=1)) .- need
        end
        cum = cumsum(net)
        mean_net = sum(abs.(net)) / T
        initial_inventory[m] += -min(0.0, minimum(cum)) + 0.10 * mean_net + 0.05
        tank[m] = max(
            initial_inventory[m] + max(0.0, maximum(cum)) + 0.15 * mean_net + 0.05,
            1.1 * initial_inventory[m],
        )
    end

    inventory_plan = zeros(Float64, M, T)
    previous = copy(initial_inventory)
    for m in 1:M, τ in 1:T
        inventory_plan[m, τ] =
            previous[m] + (
                if material_kind[m] == :raw
                    sum(purchase_plan[m, :, τ])
                else
                    production[m, τ] - sales_plan[m, τ]
                end
            ) - consumption[m, τ]
        previous[m] = inventory_plan[m, τ]
    end

    material_price = zeros(Float64, M, T)
    for (i, m) in enumerate(finals)
        base = _CP_PRODUCT_PRICE[material_names[m]] * rand(rng, LogNormal(0, 0.10))
        for τ in 1:T
            material_price[m, τ] =
                base * (1 + 0.35 * (delta[i, τ] - 1)) * rand(rng, Uniform(0.98, 1.02))
        end
    end
    sales_floor = zeros(Float64, M, T)
    sales_ceiling = zeros(Float64, M, T)
    for m in finals
        λ = rand(rng, Uniform(0.4, 0.9))
        sales_floor[m, :] .= λ .* sales_plan[m, :]
        sales_ceiling[m, :] .= sales_plan[m, :] .* (1 .+ rand(rng, Uniform(0.10, 0.40), T))
    end
    holding_cost = rand(rng, Uniform(2.0, 8.0), M)

    return (;
        material_names,
        material_kind,
        task_names,
        task_unit,
        task_inputs,
        task_outputs,
        task_cost,
        unit_names,
        campaign_unit,
        unit_capacity,
        min_rate_fraction,
        tier_price,
        tier_cap,
        material_price,
        sales_floor,
        sales_ceiling,
        tank,
        initial_inventory,
        holding_cost,
        rate,
        active,
        starts,
        purchase_plan,
        inventory_plan,
        sales_plan,
    )
end


"""Per-period variable count of one complex running `chains` with `n_tiers` purchase tiers."""
function _cp_site_per_period(chains::Vector{Symbol}, n_tiers::Int)
    material_names, material_kind, task_names, task_unit, _, _, _, unit_names, campaign_unit = _cp_assemble_complex(
        chains
    )
    n_raws = count(==(:raw), material_kind)
    n_finals = count(==(:final), material_kind)
    n_campaign = count(t -> campaign_unit[task_unit[t]], eachindex(task_names))
    return _cp_variable_count(
        length(task_names), n_campaign, n_raws, n_tiers, length(material_names), n_finals, 1
    )
end

_cp_canonical(chains) = sort(chains; by=c -> findfirst(==(c), _cp_chain_order))

"""
Choose the site portfolio, tier count, and horizon from a variable target.

Up to `_CP_SINGLE_SITE_LIMIT` a single complex is used: chain combinations are
sampled from the rng and scored by the exact per-period variable count with the
horizon solved in closed form and scanned; the minimal LDPE plant and the
complete seven-chain complex are always offered so both ends stay reachable.
Larger targets add complexes, each with its own sampled portfolio of two to
seven chains, until the company fills the target over a one-to-three-year weekly
horizon; the horizon is then solved for the exact count.
Returns `(site_chains, n_tiers, n_periods)`.
"""
function _cp_choose_dimensions(rng::AbstractRNG, target_variables::Int)
    target_variables <= MAX_CAMPAIGN_PLANNING_VARIABLES || throw(
        ArgumentError(
            "process_planning/campaign supports target_variables <= " *
            "$(MAX_CAMPAIGN_PLANNING_VARIABLES); requested $target_variables. " *
            "Larger multi-site companies need a multi-gigabyte working set.",
        ),
    )
    target = max(target_variables, 1)

    best = nothing
    best_score = (Inf, Inf, Inf)
    if target > _CP_SINGLE_SITE_LIMIT
        horizon_pref = clamp(round(Int, 4 * sqrt(target / 200)), 52, 156)
        for _ in 1:6
            n_tiers = rand(rng) < 0.4 ? 2 : 3
            sites = Vector{Vector{Symbol}}()
            per_period = 0
            while per_period * horizon_pref < target
                n_chains = rand(rng, 2:7)
                chains = _cp_canonical(shuffle(rng, collect(_cp_chain_order))[1:n_chains])
                push!(sites, chains)
                per_period += _cp_site_per_period(chains, n_tiers)
            end
            t_star = round(Int, target / per_period)
            for n_periods in max(3, t_star - 1):(t_star + 1)
                error = abs(n_periods * per_period - target) / target
                shape = abs(log(n_periods / horizon_pref))
                score = (error, shape, rand(rng))
                if score < best_score
                    best_score = score
                    best = (sites, n_tiers, n_periods)
                end
            end
        end
        return best
    end

    for candidate in 1:8
        minimal = candidate == 8
        maximal = candidate == 7
        if minimal
            chains = [:ldpe]
            n_tiers = 1
        elseif maximal
            # Always offer the complete complex so the upper end of the
            # single-site range stays reachable.
            chains = copy(_cp_chain_order)
            n_tiers = 3
        else
            max_chains = if target < 150
                2
            elseif target < 600
                3
            elseif target < 2500
                4
            elseif target < 8000
                5
            else
                7
            end
            n_chains = clamp(
                round(Int, 1 + (max_chains - 1) * rand(rng, Uniform(0.35, 1.0))), 1, max_chains
            )
            pool = collect(_cp_chain_order)
            shuffle!(rng, pool)
            chains = _cp_canonical(pool[1:n_chains])
            n_tiers = if target < 90
                1
            elseif rand(rng) < 0.4
                2
            else
                3
            end
        end
        per_period = _cp_site_per_period(chains, n_tiers)
        t_star = clamp(round(Int, target / per_period), 3, 126)
        for n_periods in max(3, t_star - 2):min(126, t_star + 2)
            size = per_period * n_periods
            error = abs(size - target) / target
            shape = abs(log(n_periods / clamp(4 * sqrt(target / 200), 3, 52)))
            score = (error, shape, rand(rng))
            if score < best_score
                best_score = score
                best = ([chains], n_tiers, n_periods)
            end
        end
    end
    return best
end


"""
    _cp_pool_potential(material_kind, task_inputs, task_outputs, members) -> Vector{Float64}

Least pooled-raw content per tonne of every material, for the feedstock pool
whose raw materials are `members`: one on the pooled raws, zero on every other
raw, and for each task (visited in topological order — every complex lists its
tasks upstream first and shares only raw materials across chains) its feed
potential spread over its output mass. Every non-raw material has exactly one
producing task, so each potential is assigned once, and each task satisfies
`sum_out c * y <= sum_in c * y` with equality.
"""
function _cp_pool_potential(
    material_kind::Vector{Symbol},
    task_inputs::Vector{Vector{Tuple{Int, Float64}}},
    task_outputs::Vector{Vector{Tuple{Int, Float64}}},
    members::Vector{Int},
)
    y = zeros(Float64, length(material_kind))
    y[members] .= 1.0
    for t in eachindex(task_inputs)
        feed = sum(c * y[m] for (m, c) in task_inputs[t]; init=0.0)
        mass = sum(c for (_, c) in task_outputs[t]; init=0.0)
        mass > 0 || continue
        for (m, _) in task_outputs[t]
            y[m] = feed / mass
        end
    end
    return y
end

"""Per-period supply the pool `g` can deliver: its pool row or its members' tier caps."""
_cp_pool_period_supply(prob, g::Int, τ::Int) = min(
    prob.supply_cap[g, τ], sum(prob.tier_cap[m, j] for m in prob.supply_groups[g] for j in 1:prob.n_tiers)
)

"""Whether pool `g` is written as an affine row (more than one purchase variable)."""
_cp_pool_has_row(prob, g::Int) = length(prob.supply_groups[g]) * prob.n_tiers >= 2

"""Pool feedstock-shortfall quantities: (potential, demand, inventory part, supply part)."""
function _cp_pool_bound_parts(prob, g::Int)
    y = _cp_pool_potential(
        prob.material_kind, prob.task_inputs, prob.task_outputs, prob.supply_groups[g]
    )
    T = prob.n_periods
    demand = sum(
        y[m] * prob.sales_floor[m, τ] for m in eachindex(y) if prob.material_kind[m] == :final for
        τ in 1:T;
        init=0.0,
    )
    inventory = sum(y[m] * prob.initial_inventory[m] for m in eachindex(y); init=0.0)
    supply = sum(_cp_pool_period_supply(prob, g, τ) for τ in 1:T)
    return y, demand, inventory, supply
end

function CampaignPlanningProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    target = max(target_variables, 1)
    site_chains, n_tiers, T = _cp_choose_dimensions(rng, target)
    campaign_length = rand(rng) < 0.5 ? 2 : 3
    sites = [_cp_generate_site(rng, chains, n_tiers, T, campaign_length) for chains in site_chains]
    n_sites = length(sites)

    # Flatten the sites into one company-wide network with global indices.
    material_names = Symbol[]
    material_kind = Symbol[]
    material_site = Int[]
    task_names = Symbol[]
    task_unit = Int[]
    task_inputs = Vector{Tuple{Int, Float64}}[]
    task_outputs = Vector{Tuple{Int, Float64}}[]
    task_cost = Float64[]
    unit_names = Symbol[]
    unit_site = Int[]
    campaign_unit = Bool[]
    for (s, site) in enumerate(sites)
        moff = length(material_names)
        uoff = length(unit_names)
        append!(material_names, site.material_names)
        append!(material_kind, site.material_kind)
        append!(material_site, fill(s, length(site.material_names)))
        append!(task_names, site.task_names)
        append!(task_unit, site.task_unit .+ uoff)
        append!(task_inputs, [[(m + moff, c) for (m, c) in io] for io in site.task_inputs])
        append!(task_outputs, [[(m + moff, c) for (m, c) in io] for io in site.task_outputs])
        append!(task_cost, site.task_cost)
        append!(unit_names, site.unit_names)
        append!(unit_site, fill(s, length(site.unit_names)))
        append!(campaign_unit, site.campaign_unit)
    end
    M = length(material_names)
    gather(f) = reduce(vcat, [getproperty(site, f) for site in sites])
    unit_capacity = gather(:unit_capacity)
    min_rate_fraction = gather(:min_rate_fraction)
    tier_price = gather(:tier_price)
    tier_cap = gather(:tier_cap)
    material_price = gather(:material_price)
    sales_floor = gather(:sales_floor)
    sales_ceiling = gather(:sales_ceiling)
    tank = gather(:tank)
    initial_inventory = gather(:initial_inventory)
    holding_cost = gather(:holding_cost)
    rate = gather(:rate)
    active = gather(:active)
    starts = gather(:starts)
    purchase_plan = cat([site.purchase_plan for site in sites]...; dims=1)
    inventory_plan = gather(:inventory_plan)
    sales_plan = gather(:sales_plan)
    changeover_cost = rand(rng, Uniform(30.0, 150.0))

    # Regional coupling: every raw material is drawn from one pool shared by
    # the sites that use it; a grade made at several sites is sold into one
    # regional market.
    group_of(kind) = begin
        index = Dict{Symbol, Vector{Int}}()
        order = Symbol[]
        for m in 1:M
            material_kind[m] == kind || continue
            name = material_names[m]
            haskey(index, name) || push!(order, name)
            push!(get!(index, name, Int[]), m)
        end
        [index[name] for name in order]
    end
    supply_groups = group_of(:raw)
    market_groups = filter(g -> length(g) >= 2, group_of(:final))
    G = length(supply_groups)
    pool_plan = [sum(purchase_plan[m, j, τ] for m in supply_groups[g] for j in 1:n_tiers) for
                 g in 1:G, τ in 1:T]
    supply_cap = [max(pool_plan[g, τ] * rand(rng, Uniform(1.05, 1.30)), 1e-3) for g in 1:G, τ in 1:T]
    market_cap = [
        sum(sales_plan[m, τ] for m in market_groups[g]) * rand(rng, Uniform(1.04, 1.25)) for
        g in eachindex(market_groups), τ in 1:T
    ]

    build(witness, certificate, scenario) = CampaignPlanningProblem(
        T,
        7.0,
        n_sites,
        material_names,
        material_kind,
        material_site,
        task_names,
        task_unit,
        task_inputs,
        task_outputs,
        task_cost,
        unit_names,
        unit_site,
        unit_capacity,
        campaign_unit,
        campaign_length,
        min_rate_fraction,
        n_tiers,
        tier_price,
        tier_cap,
        supply_groups,
        supply_cap,
        market_groups,
        market_cap,
        material_price,
        sales_floor,
        sales_ceiling,
        tank,
        initial_inventory,
        holding_cost,
        changeover_cost,
        witness,
        certificate,
        scenario,
        feasibility_status,
    )

    if feasibility_status == feasible
        witness = CampaignScheduleWitness(
            rate, active, starts, purchase_plan, inventory_plan, sales_plan
        )
        problem = build(witness, nothing, nothing)
        @assert campaign_plan_satisfies(problem)
        return problem
    end

    # Critical pool supply: below `kappa_star` times the planted pool purchases
    # the potential bound sits under the term contracts.
    reference = build(nothing, nothing, nothing)
    parts = [_cp_pool_bound_parts(reference, g) for g in 1:G]
    pool_total = [sum(view(pool_plan, g, :)) for g in 1:G]
    kappa_star = [
        pool_total[g] > 0 ? max(parts[g][2] - parts[g][3], 0.0) / pool_total[g] : 0.0 for g in 1:G
    ]

    if feasibility_status == unknown
        position = _pp_seed_position(seed)
        supply_factor = -0.15 + 1.10 * position
        for g in 1:G
            share = supply_factor + rand(rng, Uniform(-0.03, 0.03))
            kappa = max(kappa_star[g] + (1 - kappa_star[g]) * share, 0.30)
            supply_cap[g, :] .= max.(kappa .* view(pool_plan, g, :), 1e-3)
        end
        return build(nothing, nothing, CampaignMarketScenario(supply_factor, position))
    end

    # Requested infeasible. Default: curtail one regional feedstock pool below
    # what the company's term contracts need (an aggregate, multi-product,
    # multi-period argument presolve cannot see). Minority: a single-product
    # capacity bottleneck.
    margin_factor = 1 + rand(rng, Uniform(0.06, 0.15))
    eligible = [
        g for g in 1:G if
        pool_total[g] > 0 && parts[g][2] / margin_factor - parts[g][3] >= 0.25 * pool_total[g]
    ]
    if !isempty(eligible) && rand(rng) < 0.8
        sort!(eligible; by=g -> -kappa_star[g])
        g = eligible[rand(rng, 1:min(3, length(eligible)))]
        _, demand, inventory, _ = parts[g]
        kappa = (demand / margin_factor - inventory) / pool_total[g]
        supply_cap[g, :] .= kappa .* view(pool_plan, g, :)
        problem = build(nothing, nothing, nothing)
        y, demand, inventory, supply = _cp_pool_bound_parts(problem, g)
        upper = inventory + supply
        certificate = CampaignFeedstockCertificate(
            g, y, demand, inventory, supply, upper, demand - upper
        )
        problem = build(nothing, certificate, nothing)
        @assert campaign_certificate_holds(problem)
        return problem
    end

    certificate = _cp_capacity_bottleneck!(
        rng,
        T,
        material_kind,
        task_unit,
        task_inputs,
        task_outputs,
        unit_capacity,
        tier_cap,
        sales_floor,
        sales_ceiling,
        initial_inventory,
    )
    problem = build(nothing, certificate, nothing)
    @assert campaign_certificate_holds(problem)
    return problem
end

"""
Over-commit one final grade beyond its producing task's capacity and direct
raw-feed bound over a horizon prefix, and return the matching certificate.
"""
function _cp_capacity_bottleneck!(
    rng::AbstractRNG,
    T::Int,
    material_kind,
    task_unit,
    task_inputs,
    task_outputs,
    unit_capacity,
    tier_cap,
    sales_floor,
    sales_ceiling,
    initial_inventory,
)
    NT = length(task_inputs)
    producer_of = Dict{Int, Int}()
    for t in 1:NT, (m, _) in task_outputs[t]
        producer_of[m] = t
    end
    finals = [m for m in eachindex(material_kind) if material_kind[m] == :final]
    horizon = clamp(round(Int, T * rand(rng, Uniform(0.55, 1.0))), 2, T)
    raw_bound_of(producer, out_coeff) = begin
        raw_inputs = [(mm, c) for (mm, c) in task_inputs[producer] if material_kind[mm] == :raw]
        if isempty(raw_inputs)
            Inf
        else
            out_coeff * minimum(
                (horizon * sum(view(tier_cap, mm, :)) + initial_inventory[mm]) / c for
                (mm, c) in raw_inputs
            )
        end
    end
    candidates = Tuple{Int, Float64}[]
    for m in finals
        producer = get(producer_of, m, 0)
        producer == 0 && continue
        out_coeff = sum(c for (o, c) in task_outputs[producer] if o == m)
        task_cap = out_coeff * sum(view(unit_capacity, task_unit[producer], 1:horizon))
        bound = min(task_cap, raw_bound_of(producer, out_coeff))
        demand = sum(view(sales_floor, m, 1:horizon))
        bound > eps() && demand > eps() && push!(candidates, (m, demand / bound))
    end
    sort!(candidates; by=x -> -x[2])
    cut_material = candidates[rand(rng, 1:min(3, length(candidates)))][1]
    producer = producer_of[cut_material]
    out_coeff = sum(c for (o, c) in task_outputs[producer] if o == cut_material)
    raw_inputs = [(mm, c) for (mm, c) in task_inputs[producer] if material_kind[mm] == :raw]
    initial_inventory[cut_material] *= rand(rng, Uniform(0.05, 0.20))
    for (mm, _) in raw_inputs
        initial_inventory[mm] *= rand(rng, Uniform(0.02, 0.15))
    end
    demand_raise = rand(rng, Uniform(1.05, 1.25))
    sales_floor[cut_material, 1:horizon] .*= demand_raise
    sales_floor[cut_material, 1:horizon] .= min.(
        sales_floor[cut_material, 1:horizon], 0.98 .* sales_ceiling[cut_material, 1:horizon]
    )
    demand_cum = sum(sales_floor[cut_material, 1:horizon])
    desired_upper = demand_cum * rand(rng, Uniform(0.60, 0.85))
    # A bursty campaign product can carry planted initial stock above the
    # shrunken demand target; cap it so the supply cut stays below demand.
    initial_inventory[cut_material] = min(initial_inventory[cut_material], 0.4 * desired_upper)
    task_bound_raw = out_coeff * sum(view(unit_capacity, task_unit[producer], :))
    scale = clamp(
        (desired_upper - initial_inventory[cut_material]) / max(task_bound_raw, eps()), 0.01, 0.90
    )
    unit_capacity[task_unit[producer], :] .*= scale
    for (mm, _) in raw_inputs
        tier_cap[mm, :] .*= scale
    end
    task_bound = out_coeff * sum(view(unit_capacity, task_unit[producer], 1:horizon))
    raw_bound = raw_bound_of(producer, out_coeff)
    upper_bound = initial_inventory[cut_material] + min(task_bound, raw_bound)
    return CampaignCapacityCertificate(
        cut_material,
        horizon,
        demand_cum,
        initial_inventory[cut_material],
        task_bound,
        raw_bound,
        upper_bound,
        demand_cum - upper_bound,
    )
end

"""Per-material producing and consuming `(task, coefficient)` incidence lists."""
function _cp_incidence(prob::CampaignPlanningProblem)
    M = length(prob.material_names)
    produced = [Tuple{Int, Float64}[] for _ in 1:M]
    consumed = [Tuple{Int, Float64}[] for _ in 1:M]
    for t in eachindex(prob.task_names)
        for (m, c) in prob.task_outputs[t]
            push!(produced[m], (t, c))
        end
        for (m, c) in prob.task_inputs[t]
            push!(consumed[m], (t, c))
        end
    end
    return produced, consumed
end

"""Re-check a planted campaign schedule against every model row by arithmetic."""
function campaign_plan_satisfies(prob::CampaignPlanningProblem; atol::Float64=1e-6)
    plan = prob.feasible_witness
    plan === nothing && return false
    T = prob.n_periods
    M = length(prob.material_names)
    NT = length(prob.task_names)
    U = length(prob.unit_names)
    campaign_tasks = [t for t in 1:NT if prob.campaign_unit[prob.task_unit[t]]]
    cT = length(campaign_tasks)
    size(plan.rate) == (NT, T) || return false
    size(plan.active) == (cT, T) || return false
    size(plan.starts) == (cT, T) || return false
    size(plan.purchase) == (M, prob.n_tiers, T) || return false
    size(plan.inventory) == (M, T) || return false
    size(plan.sales) == (M, T) || return false
    all(>=(-atol), plan.rate) || return false
    all(>=(-atol), plan.purchase) || return false
    all(>=(-atol), plan.inventory) || return false
    all(>=(-atol), plan.sales) || return false
    all(x -> x in (0.0, 1.0), plan.active) || return false
    all(x -> x in (0.0, 1.0), plan.starts) || return false

    unit_tasks = [Int[] for _ in 1:U]
    for t in 1:NT
        push!(unit_tasks[prob.task_unit[t]], t)
    end
    campaign_index = Dict(t => i for (i, t) in enumerate(campaign_tasks))
    for u in 1:U, τ in 1:T
        load = sum(plan.rate[t, τ] for t in unit_tasks[u]; init=0.0)
        load <= prob.unit_capacity[u, τ] + atol * max(1.0, load) || return false
        if prob.campaign_unit[u]
            sum(plan.active[campaign_index[t], τ] for t in unit_tasks[u]; init=0.0) <=
            1.0 + atol || return false
        end
    end

    L = prob.campaign_length
    for (i, task) in enumerate(campaign_tasks), τ in 1:T
        active = plan.active[i, τ]
        previous = τ == 1 ? 0.0 : plan.active[i, τ - 1]
        expected_start = max(0.0, active - previous)
        abs(plan.starts[i, τ] - expected_start) <= atol || return false
        cap = prob.unit_capacity[prob.task_unit[task], τ]
        plan.rate[task, τ] <= cap * active + atol * max(1.0, cap) || return false
        plan.rate[task, τ] + atol * max(1.0, cap) >= prob.min_rate_fraction[i] * cap * active ||
            return false
        if expected_start > 0.5
            τ + L - 1 <= T || return false
            all(plan.active[i, k] > 0.5 for k in τ:(τ + L - 1)) || return false
        end
    end

    produced, consumed = _cp_incidence(prob)
    for m in 1:M, τ in 1:T
        made = sum(c * plan.rate[t, τ] for (t, c) in produced[m]; init=0.0)
        used = sum(c * plan.rate[t, τ] for (t, c) in consumed[m]; init=0.0)
        bought = prob.material_kind[m] == :raw ? sum(plan.purchase[m, :, τ]) : 0.0
        sold = prob.material_kind[m] == :final ? plan.sales[m, τ] : 0.0
        previous = τ == 1 ? prob.initial_inventory[m] : plan.inventory[m, τ - 1]
        scale = max(1.0, abs(previous), made, used, bought, sold)
        abs(plan.inventory[m, τ] - (previous + made + bought - used - sold)) <= atol * scale ||
            return false
        plan.inventory[m, τ] <= prob.tank[m] + atol * scale || return false
        if prob.material_kind[m] == :raw
            for j in 1:prob.n_tiers
                plan.purchase[m, j, τ] <= prob.tier_cap[m, j] + atol * scale || return false
            end
        else
            all(plan.purchase[m, j, τ] <= atol for j in 1:prob.n_tiers) || return false
        end
        if prob.material_kind[m] == :final
            plan.sales[m, τ] + atol * scale >= prob.sales_floor[m, τ] || return false
            plan.sales[m, τ] <= prob.sales_ceiling[m, τ] + atol * scale || return false
        else
            plan.sales[m, τ] <= atol || return false
        end
    end
    for (g, members) in enumerate(prob.supply_groups), τ in 1:T
        pooled = sum(plan.purchase[m, j, τ] for m in members for j in 1:prob.n_tiers)
        pooled <= prob.supply_cap[g, τ] + atol * max(1.0, pooled) || return false
    end
    for (g, members) in enumerate(prob.market_groups), τ in 1:T
        sold = sum(plan.sales[m, τ] for m in members)
        sold <= prob.market_cap[g, τ] + atol * max(1.0, sold) || return false
    end
    return true
end

"""Recompute the refutation stored on a requested-infeasible campaign instance."""
function campaign_certificate_holds(prob::CampaignPlanningProblem; atol::Float64=1e-6)
    cert = prob.infeasibility_certificate
    cert === nothing && return false
    cert isa CampaignFeedstockCertificate && return _cp_feedstock_certificate_holds(prob, cert, atol)
    M = length(prob.material_names)
    NT = length(prob.task_names)
    1 <= cert.material <= M || return false
    1 <= cert.horizon <= prob.n_periods || return false
    prob.material_kind[cert.material] == :final || return false
    producers = [t for t in 1:NT if any(m == cert.material for (m, _) in prob.task_outputs[t])]
    length(producers) == 1 || return false
    task = only(producers)
    output = sum(c for (m, c) in prob.task_outputs[task] if m == cert.material)
    task_bound = output * sum(prob.unit_capacity[prob.task_unit[task], 1:cert.horizon])
    raw_inputs = [(m, c) for (m, c) in prob.task_inputs[task] if prob.material_kind[m] == :raw]
    raw_bound = if isempty(raw_inputs)
        Inf
    else
        output * minimum(
            (cert.horizon * sum(prob.tier_cap[m, :]) + prob.initial_inventory[m]) / c for
            (m, c) in raw_inputs
        )
    end
    demand = sum(prob.sales_floor[cert.material, 1:cert.horizon])
    upper = prob.initial_inventory[cert.material] + min(task_bound, raw_bound)
    scale = max(1.0, abs(demand), abs(upper))
    isapprox(cert.demand, demand; atol=atol * scale, rtol=1e-9) || return false
    isapprox(
        cert.initial_inventory, prob.initial_inventory[cert.material]; atol=atol * scale, rtol=1e-9
    ) || return false
    isapprox(cert.task_bound, task_bound; atol=atol * scale, rtol=1e-9) || return false
    (
        if isinf(raw_bound)
            isinf(cert.raw_bound)
        else
            isapprox(cert.raw_bound, raw_bound; atol=atol * scale, rtol=1e-9)
        end
    ) || return false
    isapprox(cert.upper_bound, upper; atol=atol * scale, rtol=1e-9) || return false
    isapprox(cert.margin, demand - upper; atol=atol * scale, rtol=1e-9) || return false
    return upper + atol * scale < demand
end

function _cp_feedstock_certificate_holds(
    prob::CampaignPlanningProblem, cert::CampaignFeedstockCertificate, atol::Float64
)
    1 <= cert.group <= length(prob.supply_groups) || return false
    y, demand, inventory, supply = _cp_pool_bound_parts(prob, cert.group)
    length(cert.potential) == length(y) || return false
    all(>=(0.0), cert.potential) || return false
    isapprox(cert.potential, y; rtol=1e-9) || return false
    # Every non-pooled raw carries zero potential and no task creates content.
    for m in eachindex(y)
        if prob.material_kind[m] == :raw && !(m in prob.supply_groups[cert.group])
            cert.potential[m] == 0.0 || return false
        end
    end
    for t in eachindex(prob.task_names)
        made = sum(c * cert.potential[m] for (m, c) in prob.task_outputs[t]; init=0.0)
        fed = sum(c * cert.potential[m] for (m, c) in prob.task_inputs[t]; init=0.0)
        made <= fed * (1 + 1e-12) + atol || return false
    end
    upper = inventory + supply
    scale = max(1.0, abs(demand), abs(upper))
    isapprox(cert.demand, demand; atol=atol * scale, rtol=1e-9) || return false
    isapprox(cert.inventory_bound, inventory; atol=atol * scale, rtol=1e-9) || return false
    isapprox(cert.supply_bound, supply; atol=atol * scale, rtol=1e-9) || return false
    isapprox(cert.upper_bound, upper; atol=atol * scale, rtol=1e-9) || return false
    isapprox(cert.margin, demand - upper; atol=atol * scale, rtol=1e-9) || return false
    return upper + atol * scale < demand
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
    J = prob.n_tiers

    # A single-task unit's capacity is a bound on its task's rate; a campaign
    # train's capacity is enforced by the activity gate below.
    @variable(model, rate[t = 1:NT, τ = 1:T] >= 0)
    for t in 1:NT
        prob.campaign_unit[prob.task_unit[t]] && continue
        for τ in 1:T
            set_upper_bound(rate[t, τ], prob.unit_capacity[prob.task_unit[t], τ])
        end
    end
    @variable(model, active[i = 1:cT, τ = 1:T], Bin)
    @variable(model, starts[i = 1:cT, τ = 1:T], Bin)
    @variable(model, 0 <= purchase[m = raws, j = 1:J, τ = 1:T] <= prob.tier_cap[m, j])
    @variable(model, 0 <= inventory[m = 1:M, τ = 1:T] <= prob.tank[m])
    @variable(
        model, prob.sales_floor[m, τ] <= sales[m = finals, τ = 1:T] <= prob.sales_ceiling[m, τ]
    )

    objective = AffExpr(0.0)
    for m in finals, τ in 1:T
        add_to_expression!(objective, prob.material_price[m, τ], sales[m, τ])
    end
    for m in raws, j in 1:J, τ in 1:T
        add_to_expression!(objective, -prob.tier_price[m, j], purchase[m, j, τ])
    end
    for t in 1:NT, τ in 1:T
        add_to_expression!(objective, -prob.task_cost[t], rate[t, τ])
    end
    for m in 1:M, τ in 1:T
        add_to_expression!(objective, -prob.holding_cost[m], inventory[m, τ])
    end
    for i in 1:cT, τ in 1:T
        add_to_expression!(objective, -prob.changeover_cost, starts[i, τ])
    end
    @objective(model, Max, objective)

    produced, consumed = _cp_incidence(prob)
    for m in 1:M, τ in 1:T
        flow = AffExpr(0.0)
        if prob.material_kind[m] == :raw
            for j in 1:J
                add_to_expression!(flow, 1.0, purchase[m, j, τ])
            end
        end
        for (t, c) in consumed[m]
            add_to_expression!(flow, -c, rate[t, τ])
        end
        for (t, c) in produced[m]
            add_to_expression!(flow, c, rate[t, τ])
        end
        prob.material_kind[m] == :final && add_to_expression!(flow, -1.0, sales[m, τ])
        add_to_expression!(flow, -1.0, inventory[m, τ])
        if τ == 1
            @constraint(model, flow == -prob.initial_inventory[m])
        else
            add_to_expression!(flow, 1.0, inventory[m, τ - 1])
            @constraint(model, flow == 0)
        end
    end

    train_tasks = [Int[] for _ in 1:U]
    for t in campaign_tasks
        push!(train_tasks[prob.task_unit[t]], t)
    end
    for u in 1:U
        prob.campaign_unit[u] || continue
        for τ in 1:T
            @constraint(model, sum(active[campaign_index[t], τ] for t in train_tasks[u]) <= 1)
        end
    end

    L = prob.campaign_length
    for (i, t) in enumerate(campaign_tasks)
        for τ in 1:T
            cap = prob.unit_capacity[prob.task_unit[t], τ]
            @constraint(model, rate[t, τ] - cap * active[i, τ] <= 0)
            @constraint(model, rate[t, τ] - prob.min_rate_fraction[i] * cap * active[i, τ] >= 0)
            if τ == 1
                @constraint(model, starts[i, 1] - active[i, 1] == 0)
            else
                @constraint(model, starts[i, τ] - active[i, τ] + active[i, τ - 1] >= 0)
                @constraint(model, starts[i, τ] - active[i, τ] <= 0)
                @constraint(model, starts[i, τ] + active[i, τ - 1] <= 1)
            end
        end
        for τ in 1:(T - L + 1)
            run = sum(active[i, k] for k in τ:(τ + L - 1))
            if τ == 1
                @constraint(model, run - L * active[i, 1] >= 0)
            else
                @constraint(model, run - L * active[i, τ] + L * active[i, τ - 1] >= 0)
            end
        end
        # A campaign cannot start so late that its minimum duration would
        # extend past the modeled horizon.
        for τ in max(T - L + 2, 2):T
            fix(starts[i, τ], 0.0; force=true)
        end
    end

    for (g, members) in enumerate(prob.supply_groups)
        if _cp_pool_has_row(prob, g)
            for τ in 1:T
                @constraint(
                    model,
                    sum(purchase[m, j, τ] for m in members for j in 1:J) <= prob.supply_cap[g, τ]
                )
            end
        else
            m = only(members)
            for τ in 1:T
                set_upper_bound(purchase[m, 1, τ], min(prob.tier_cap[m, 1], prob.supply_cap[g, τ]))
            end
        end
    end
    for (g, members) in enumerate(prob.market_groups), τ in 1:T
        @constraint(model, sum(sales[m, τ] for m in members) <= prob.market_cap[g, τ])
    end
    return model
end

register_variant(
    :process_planning,
    :campaign,
    CampaignPlanningProblem,
    "Multi-period petrochemical campaign-planning MIP over state-task networks: one or several complexes with tiered raw purchasing from shared regional feedstock pools, shared trains running product grades as campaigns with minimum lengths and changeovers, storage, and seasonal term and spot demand in shared regional markets",
)
