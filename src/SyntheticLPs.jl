module SyntheticLPs

using JuMP
using Random
using Distributions
using JSON

# Base types
abstract type ProblemGenerator end

@enum FeasibilityStatus begin
    feasible
    infeasible
    unknown
end

export ProblemGenerator
export FeasibilityStatus
export feasible, infeasible, unknown
export ProblemVariant
export generate_problem
export generate_random_problem
export register_category
export register_variant
export list_categories
export list_problem_types
export list_variants
export list_problems
export problem_info
export bounds_to_constraints!
export generate_dataset
export GeneratedInstance
export QualityCriteria, QualityResult, check_quality

# ---------------------------------------------------------------------------
# Problem identity: categories and variants
# ---------------------------------------------------------------------------
#
# A *category* is a problem domain (e.g. `:transportation`). A category groups
# one or more *variants* — concrete generators with their own data generation
# and model formulation (e.g. `:standard`). A `ProblemVariant` names one
# variant of one category and is the canonical reference used throughout the
# package. Source for each category lives in `src/problem_types/<category>/`,
# with a thin `<category>.jl` that includes one file per variant.

"""
    ProblemVariant(category::Symbol, variant::Symbol)
    ProblemVariant(category::Symbol)             # the category's default variant
    ProblemVariant("category")                   # default variant, from a string
    ProblemVariant("category/variant")           # an explicit variant, from a string

A fully-qualified reference to a concrete problem generator: a `variant` of a
`category`. Prints as `category/variant`.
"""
struct ProblemVariant
    category::Symbol
    variant::Symbol
end

Base.show(io::IO, p::ProblemVariant) = print(io, p.category, '/', p.variant)

# ---------------------------------------------------------------------------
# Registration system
# ---------------------------------------------------------------------------

"""
    VariantSpec

Registry entry for a single variant: its category, variant name, generator
type, and a human-readable description.
"""
struct VariantSpec
    category::Symbol
    variant::Symbol
    type::Type{<:ProblemGenerator}
    description::String
end

"""
    CategorySpec

Registry entry for a category: its description, the variants registered under
it, and which variant is used by default when none is named.
"""
mutable struct CategorySpec
    category::Symbol
    description::String
    variants::Dict{Symbol,VariantSpec}
    default_variant::Union{Symbol,Nothing}
    explicit_default::Bool
end

# Maps category symbol -> CategorySpec.
const LP_REGISTRY = Dict{Symbol,CategorySpec}()

"""
    register_category(category::Symbol, description::AbstractString)

Register (or fetch) a category with a human-readable `description`. Returns the
`CategorySpec`.

Calling this explicitly is only necessary when a category needs a description
distinct from its variants' (typically when it has several variants). A single
variant created with [`register_variant`](@ref) will lazily create its category
using the variant's description, so single-variant categories need no explicit
`register_category` call.
"""
function register_category(category::Symbol, description::AbstractString)
    cat = get!(LP_REGISTRY, category) do
        CategorySpec(category, String(description),
                     Dict{Symbol,VariantSpec}(), nothing, false)
    end
    # Always apply the explicit description, even if the category was already
    # created lazily by `register_variant`, so registration order doesn't matter.
    cat.description = String(description)
    return cat
end

"""
    register_variant(category::Symbol, variant::Symbol,
                     problem_type::Type{<:ProblemGenerator}, description::AbstractString;
                     default::Bool=false)

Register a `variant` of `category` backed by `problem_type`. If the category is
not yet registered, it is created lazily using `description`.

The first variant registered becomes the category default; pass `default=true`
to designate a specific variant instead (only one variant may be the explicit
default).
"""
function register_variant(category::Symbol, variant::Symbol,
                          problem_type::Type{<:ProblemGenerator},
                          description::AbstractString; default::Bool=false)
    cat = get(LP_REGISTRY, category, nothing)
    if cat === nothing
        cat = register_category(category, description)
    end
    if haskey(cat.variants, variant)
        error("Variant $category/$variant is already registered.")
    end
    spec = VariantSpec(category, variant, problem_type, String(description))
    cat.variants[variant] = spec
    if default
        if cat.explicit_default
            error("Category $category already has an explicit default variant " *
                  "($(cat.default_variant)); cannot also mark $variant as default.")
        end
        cat.default_variant = variant
        cat.explicit_default = true
    elseif cat.default_variant === nothing
        cat.default_variant = variant
    end
    return spec
end

"""
    get_category(category::Symbol) -> CategorySpec

Internal: fetch a category's registry entry, erroring helpfully if unknown.
"""
function get_category(category::Symbol)
    haskey(LP_REGISTRY, category) ||
        error("Unknown problem category: $category. " *
              "Use list_categories() to see available categories.")
    return LP_REGISTRY[category]
end

"""
    get_variant(ref::ProblemVariant) -> VariantSpec

Internal: fetch a variant's registry entry, erroring helpfully if unknown.
"""
function get_variant(ref::ProblemVariant)
    cat = get_category(ref.category)
    haskey(cat.variants, ref.variant) ||
        error("Unknown variant $(ref.category)/$(ref.variant). " *
              "Available variants of $(ref.category): " *
              "$(join(sort(collect(keys(cat.variants))), ", ")).")
    return cat.variants[ref.variant]
end

"""
    default_variant(category::Symbol) -> Symbol

The default variant symbol for a category.
"""
function default_variant(category::Symbol)
    cat = get_category(category)
    cat.default_variant === nothing &&
        error("Category $category has no registered variants.")
    return cat.default_variant
end

# ProblemVariant convenience constructors (defined after the registry so they
# can resolve a category's default variant).
ProblemVariant(category::Symbol) = ProblemVariant(category, default_variant(category))

function ProblemVariant(s::AbstractString)
    parts = split(s, '/')
    if length(parts) == 1
        return ProblemVariant(Symbol(strip(parts[1])))
    elseif length(parts) == 2
        return ProblemVariant(Symbol(strip(parts[1])), Symbol(strip(parts[2])))
    end
    error("Invalid problem reference \"$s\"; expected \"category\" or " *
          "\"category/variant\".")
end

"""
    get_problem_type(ref) -> Type{<:ProblemGenerator}

Resolve a problem reference (a `ProblemVariant`, a category `Symbol`, or a
`"category"`/`"category/variant"` string) to its generator type.
"""
get_problem_type(ref::ProblemVariant) = get_variant(ref).type
get_problem_type(category::Symbol) = get_problem_type(ProblemVariant(category))
get_problem_type(s::AbstractString) = get_problem_type(ProblemVariant(s))

# ---------------------------------------------------------------------------
# Model transforms (post-build reformulations)
# ---------------------------------------------------------------------------
include("transforms.jl")

# ---------------------------------------------------------------------------
# Model building and problem generation
# ---------------------------------------------------------------------------

"""
    build_model(problem::ProblemGenerator)

Build a JuMP model from a problem generator instance.
Each variant must implement this method.

# Arguments
- `problem`: A problem generator instance containing all necessary data

# Returns
- `model`: The JuMP model
"""
function build_model end

"""
    _generate_problem_verified([ref_or_type], target_variables, feasibility_status, seed;
                               relax_integer, bounds_to_constraints, optimizer,
                               max_feasibility_retries, feasibility_timeout)

Internal builder used by [`generate_problem`](@ref). Constructs the problem and its
JuMP model and applies the `relax_integer` / `bounds_to_constraints` transforms.

When `optimizer` is supplied and `feasibility_status` is `feasible` or `infeasible`,
the model is solved once to verify the feasibility contract — a `feasible` request
must solve to `OPTIMAL`, an `infeasible` request must solve to `INFEASIBLE`. If the
solve *disproves* the requested status the problem is rebuilt with the next seed and
re-checked, up to `max_feasibility_retries` times. (Generators aim to honor the
requested status by construction, but a few have heuristic feasibility logic that
occasionally misses; this central check is the project-level backstop, so callers
receive a conforming model or an error when the retry budget is exhausted.)

If instead the solve *certifies nothing* — it hits `feasibility_timeout`, or returns a
status that separates neither case — the retry budget is not spent: verification
raises immediately, reporting the termination status. Unrelaxed MIPs are the usual
cause; give them a larger `feasibility_timeout`.

Returns `(model, problem, resolved_seed)`. With `optimizer=nothing` (or status
`unknown`) the model is built exactly once and `resolved_seed == seed`. Verification
is itself deterministic — attempts walk `seed, seed+1, …` — so a given
`(seed, optimizer)` pair always resolves to the same model.
"""
function _generate_problem_verified(::Type{T}, target_variables::Int,
                                    feasibility_status::FeasibilityStatus, seed::Int;
                                    relax_integer::Bool=true,
                                    bounds_to_constraints::Bool=false,
                                    optimizer=nothing,
                                    max_feasibility_retries::Int=10,
                                    feasibility_timeout::Float64=10.0) where T <: ProblemGenerator
    max_feasibility_retries >= 1 ||
        error("max_feasibility_retries must be >= 1 (got $max_feasibility_retries).")
    needs_check = optimizer !== nothing && feasibility_status !== unknown

    current_seed = seed
    model = nothing
    problem = nothing
    for attempt in 1:max_feasibility_retries
        problem = T(target_variables, feasibility_status, current_seed)
        model = build_model(problem)
        relax_integer && relax_integrality(model)
        bounds_to_constraints && bounds_to_constraints!(model)
        if !needs_check
            return model, problem, current_seed
        end
        verdict, ts = _check_feasibility_contract(model, optimizer, feasibility_status;
                                                  timeout=feasibility_timeout)
        if verdict === :holds
            return model, problem, current_seed
        elseif verdict === :inconclusive
            # The solve certified nothing, so we have no evidence against this
            # instance and rebuilding would just re-ask an unanswerable question.
            # Report the real cause instead of charging it to the retry budget.
            error("Feasibility contract could not be verified for $T " *
                  "(target_variables=$target_variables, status=$feasibility_status, " *
                  "seed=$current_seed): the verification solve returned $ts " *
                  "after a $(feasibility_timeout)s limit. This is not evidence of a " *
                  "contract violation. Raise `feasibility_timeout`, use a stronger " *
                  "optimizer, or drop `optimizer` to skip verification.")
        end
        # Contract disproved — rebuild with a fresh seed if another attempt remains.
        attempt < max_feasibility_retries && (current_seed += 1)
    end

    error("Feasibility contract not satisfied for $T " *
          "(target_variables=$target_variables, status=$feasibility_status) " *
          "after $max_feasibility_retries attempts " *
          "(seeds $seed through $current_seed); no model was returned.")
end

# Ref-based overload delegating to the type-based builder above.
function _generate_problem_verified(ref::ProblemVariant, target_variables::Int,
                                    feasibility_status::FeasibilityStatus, seed::Int;
                                    kwargs...)
    return _generate_problem_verified(get_problem_type(ref), target_variables,
                                      feasibility_status, seed; kwargs...)
end

# Classify a solver termination status against the requested `feasibility_status`.
# Returns one of:
# - `:holds`        — the status proves the requested feasibility.
# - `:violated`     — the status disproves it. Retrying with a different seed is
#                     meaningful, so the caller rebuilds.
# - `:inconclusive` — the status proves nothing either way (the solve hit its time
#                     limit, stopped short of its tolerances, or could not separate
#                     infeasible from unbounded). Retrying would re-ask the same
#                     unanswerable question, so the caller raises instead.
#
# Keeping `:violated` and `:inconclusive` distinct is the point of this function: a
# MIP that exceeds the verification time limit is not evidence of a contract
# violation, and treating it as one both wastes the retry budget and misreports the
# failure. Pure (no solve) so the full status table is testable without a solver.
function _classify_termination(ts, feasibility_status::FeasibilityStatus)
    # A solver that cannot separate these two cases has certified neither. Never read
    # it as proof of infeasibility: an unbounded model has a nonempty feasible region.
    ts == JuMP.MOI.INFEASIBLE_OR_UNBOUNDED && return :inconclusive
    # ALMOST_OPTIMAL means the solve stopped short of its tolerances, so it is not a
    # trustworthy certificate in either direction.
    ts == JuMP.MOI.ALMOST_OPTIMAL && return :inconclusive

    if feasibility_status == feasible
        ts == JuMP.MOI.OPTIMAL && return :holds
        # INFEASIBLE disproves the request outright. DUAL_INFEASIBLE (MOI's encoding
        # of primal-unbounded) means the model is feasible but has no optimum, which
        # the `feasible` contract also excludes — a different seed may fix either.
        (ts == JuMP.MOI.INFEASIBLE || ts == JuMP.MOI.DUAL_INFEASIBLE) && return :violated
        return :inconclusive
    elseif feasibility_status == infeasible
        ts == JuMP.MOI.INFEASIBLE && return :holds
        # Both of these exhibit a feasible point, disproving the request.
        (ts == JuMP.MOI.OPTIMAL || ts == JuMP.MOI.DUAL_INFEASIBLE) && return :violated
        return :inconclusive
    end
    return :holds
end

# Solve `model` and classify the result via `_classify_termination`, returning
# `(verdict, termination_status)`. Solves a structural copy so the caller's model is
# returned pristine (no optimizer attached, no time limit set, not pre-solved).
function _check_feasibility_contract(model::Model, optimizer,
                                     feasibility_status::FeasibilityStatus;
                                     timeout::Float64=10.0)
    check = copy(model)
    set_optimizer(check, optimizer)
    set_silent(check)
    set_time_limit_sec(check, timeout)
    optimize!(check)
    ts = termination_status(check)
    return _classify_termination(ts, feasibility_status), ts
end

"""
    generate_problem(::Type{T}, target_variables, feasibility_status, seed;
                     relax_integer=true, bounds_to_constraints=false,
                     optimizer=nothing, max_feasibility_retries=10,
                     feasibility_timeout=10.0)

Generate a linear programming problem from a generator type by constructing an
instance and building its model.

When `bounds_to_constraints=true`, variable bounds (other than plain `x ≥ 0`
nonnegativity) are reformulated as explicit affine constraints via
[`bounds_to_constraints!`](@ref). This runs *after* integrality relaxation, so
bounds introduced by relaxing integer/binary variables are converted too.

When `optimizer` is supplied (e.g. `HiGHS.Optimizer`) and `feasibility_status` is
`feasible` or `infeasible`, the model is solved to verify the feasibility contract
and rebuilt with a new seed on violation (see [`_generate_problem_verified`](@ref)).
A verification solve that certifies nothing — it exceeds `feasibility_timeout`, or
returns a status separating neither case — raises rather than counting as a violation.
With `optimizer=nothing` (the default) no solving is performed.

# Returns
- `model`: The JuMP model
- `problem`: The problem generator instance containing all parameters
"""
function generate_problem(::Type{T}, target_variables::Int,
                          feasibility_status::FeasibilityStatus=unknown, seed::Int=0;
                          relax_integer::Bool=true,
                          bounds_to_constraints::Bool=false,
                          optimizer=nothing,
                          max_feasibility_retries::Int=10,
                          feasibility_timeout::Float64=10.0) where T <: ProblemGenerator
    model, problem, _ = _generate_problem_verified(T, target_variables,
                                                   feasibility_status, seed;
                                                   relax_integer=relax_integer,
                                                   bounds_to_constraints=bounds_to_constraints,
                                                   optimizer=optimizer,
                                                   max_feasibility_retries=max_feasibility_retries,
                                                   feasibility_timeout=feasibility_timeout)
    return model, problem
end

"""
    generate_problem(ref::ProblemVariant, target_variables, feasibility_status, seed;
                     relax_integer=true, bounds_to_constraints=false,
                     optimizer=nothing, max_feasibility_retries=10,
                     feasibility_timeout=10.0)

Generate a problem from a fully-qualified `category/variant` reference.
"""
function generate_problem(ref::ProblemVariant, target_variables::Int,
                          feasibility_status::FeasibilityStatus=unknown, seed::Int=0;
                          relax_integer::Bool=true, bounds_to_constraints::Bool=false,
                          optimizer=nothing, max_feasibility_retries::Int=10,
                          feasibility_timeout::Float64=10.0)
    return generate_problem(get_problem_type(ref), target_variables,
                            feasibility_status, seed; relax_integer=relax_integer,
                            bounds_to_constraints=bounds_to_constraints,
                            optimizer=optimizer,
                            max_feasibility_retries=max_feasibility_retries,
                            feasibility_timeout=feasibility_timeout)
end

"""
    generate_problem(ref::AbstractString, target_variables, feasibility_status, seed;
                     relax_integer=true, bounds_to_constraints=false,
                     optimizer=nothing, max_feasibility_retries=10,
                     feasibility_timeout=10.0)

Generate a problem from a `"category"` or `"category/variant"` string, parsed via
[`ProblemVariant`](@ref).
"""
function generate_problem(ref::AbstractString, target_variables::Int,
                          feasibility_status::FeasibilityStatus=unknown, seed::Int=0;
                          relax_integer::Bool=true, bounds_to_constraints::Bool=false,
                          optimizer=nothing, max_feasibility_retries::Int=10,
                          feasibility_timeout::Float64=10.0)
    return generate_problem(ProblemVariant(ref), target_variables,
                            feasibility_status, seed; relax_integer=relax_integer,
                            bounds_to_constraints=bounds_to_constraints,
                            optimizer=optimizer,
                            max_feasibility_retries=max_feasibility_retries,
                            feasibility_timeout=feasibility_timeout)
end

"""
    generate_problem(category::Symbol, target_variables, feasibility_status, seed;
                     variant=nothing, relax_integer=true, bounds_to_constraints=false,
                     optimizer=nothing, max_feasibility_retries=10,
                     feasibility_timeout=10.0)

Generate a problem for a category. With `variant=nothing` the category's default
variant is used; pass `variant=:name` to select a specific variant.

# Arguments
- `category`: Problem category symbol (e.g. `:transportation`)
- `target_variables`: Target number of variables in the LP formulation
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
- `variant`: Optional variant symbol; defaults to the category default
- `relax_integer`: Relax integrality of the generated model
- `bounds_to_constraints`: Reformulate variable bounds (other than `x ≥ 0`) as
  explicit affine constraints
- `optimizer`: Optional solver used to verify the feasibility contract (see
  [`_generate_problem_verified`](@ref)). `nothing` disables verification.
- `max_feasibility_retries`: Maximum number of rebuild attempts when verification
  disproves the requested status.
- `feasibility_timeout`: Time limit (seconds) for each verification solve. Exceeding
  it raises rather than consuming a retry; unrelaxed MIPs may need more than the
  10s default.

# Returns
- `model`: The JuMP model
- `problem`: The problem generator instance
"""
function generate_problem(category::Symbol, target_variables::Int,
                          feasibility_status::FeasibilityStatus=unknown, seed::Int=0;
                          variant::Union{Symbol,Nothing}=nothing, relax_integer::Bool=true,
                          bounds_to_constraints::Bool=false,
                          optimizer=nothing, max_feasibility_retries::Int=10,
                          feasibility_timeout::Float64=10.0)
    ref = variant === nothing ? ProblemVariant(category) :
                                ProblemVariant(category, variant)
    return generate_problem(ref, target_variables, feasibility_status, seed;
                            relax_integer=relax_integer,
                            bounds_to_constraints=bounds_to_constraints,
                            optimizer=optimizer,
                            max_feasibility_retries=max_feasibility_retries,
                            feasibility_timeout=feasibility_timeout)
end

# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------

"""
    list_categories() -> Vector{Symbol}

List all registered problem categories.
"""
list_categories() = collect(keys(LP_REGISTRY))

"""
    list_problem_types() -> Vector{Symbol}

Alias for [`list_categories`](@ref).
"""
list_problem_types() = list_categories()

"""
    list_variants(category::Symbol) -> Vector{Symbol}

List the variants registered under a category, sorted.
"""
list_variants(category::Symbol) = sort!(collect(keys(get_category(category).variants)))

"""
    list_problems() -> Vector{ProblemVariant}

List every registered `category/variant` pair, sorted by category then variant.
"""
function list_problems()
    refs = ProblemVariant[]
    for category in sort(collect(keys(LP_REGISTRY)))
        for variant in list_variants(category)
            push!(refs, ProblemVariant(category, variant))
        end
    end
    return refs
end

"""
    problem_info(category::Symbol) -> Dict

Information about a category: its description, variants, and default variant.
"""
function problem_info(category::Symbol)
    cat = get_category(category)
    return Dict(
        :type => category,
        :category => category,
        :description => cat.description,
        :variants => list_variants(category),
        :default_variant => cat.default_variant,
    )
end

"""
    problem_info(category::Symbol, variant::Symbol) -> Dict

Information about a specific variant: its description and generator type.
"""
function problem_info(category::Symbol, variant::Symbol)
    spec = get_variant(ProblemVariant(category, variant))
    return Dict(
        :category => spec.category,
        :variant => spec.variant,
        :description => spec.description,
        :type => spec.type,
    )
end

"""
    generate_random_problem(target_variables; feasibility_status=unknown,
                            relax_integer=true, bounds_to_constraints=false, seed=0,
                            optimizer=nothing, max_feasibility_retries=10,
                            feasibility_timeout=10.0)

Generate a problem of a randomly selected variant targeting approximately the
specified number of variables. Sampling is uniform over all registered
`category/variant` pairs. When `optimizer` is supplied and `feasibility_status` is
`feasible`/`infeasible`, the feasibility contract is verified (see
[`generate_problem`](@ref)).

# Returns
- `model`: The JuMP model
- `ref`: The `ProblemVariant` that was selected
- `problem`: The problem generator instance
"""
function generate_random_problem(target_variables::Int;
                                 feasibility_status::FeasibilityStatus=unknown,
                                 relax_integer::Bool=true,
                                 bounds_to_constraints::Bool=false, seed::Int=0,
                                 optimizer=nothing, max_feasibility_retries::Int=10,
                                 feasibility_timeout::Float64=10.0)
    Random.seed!(seed)

    problems = list_problems()
    if isempty(problems)
        error("No problem types registered. Include problem type files first.")
    end

    ref = rand(problems)
    model, problem = generate_problem(ref, target_variables, feasibility_status, seed;
                                      relax_integer=relax_integer,
                                      bounds_to_constraints=bounds_to_constraints,
                                      optimizer=optimizer,
                                      max_feasibility_retries=max_feasibility_retries,
                                      feasibility_timeout=feasibility_timeout)

    return model, ref, problem
end

# ---------------------------------------------------------------------------
# Problem generators
# ---------------------------------------------------------------------------
# Each category lives in its own folder; the `<category>.jl` entry point
# registers the category (if needed) and includes one file per variant.
include("problem_types/airline_crew/airline_crew.jl")
include("problem_types/assignment/assignment.jl")
include("problem_types/bin_packing/bin_packing.jl")
include("problem_types/blending/blending.jl")
include("problem_types/crop_planning/crop_planning.jl")
include("problem_types/cutting_stock/cutting_stock.jl")
include("problem_types/container_loading/container_loading.jl")
include("problem_types/diet_problem/diet_problem.jl")
include("problem_types/energy/energy.jl")
include("problem_types/facility_location/facility_location.jl")
include("problem_types/feed_blending/feed_blending.jl")
include("problem_types/generic_milp/generic_milp.jl")
include("problem_types/graph_optimization/graph_optimization.jl")
include("problem_types/inventory/inventory.jl")
include("problem_types/job_shop_scheduling/job_shop_scheduling.jl")
include("problem_types/knapsack/knapsack.jl")
include("problem_types/land_use/land_use.jl")
include("problem_types/load_balancing/load_balancing.jl")
include("problem_types/maritime_inventory_routing/maritime_inventory_routing.jl")
include("problem_types/multi_commodity_flow/multi_commodity_flow.jl")
include("problem_types/network_flow/network_flow.jl")
include("problem_types/neural_network_verification/neural_network_verification.jl")
include("problem_types/nurse_scheduling/nurse_scheduling.jl")
include("problem_types/portfolio/portfolio.jl")
include("problem_types/product_mix/product_mix.jl")
include("problem_types/production_planning/production_planning.jl")
include("problem_types/project_selection/project_selection.jl")
include("problem_types/regression/regression.jl")
include("problem_types/resilient_network_design/resilient_network_design.jl")
include("problem_types/resource_allocation/resource_allocation.jl")
include("problem_types/revenue_management/revenue_management.jl")
include("problem_types/scheduling/scheduling.jl")
include("problem_types/stochastic_program/stochastic_program.jl")
include("problem_types/supply_chain/supply_chain.jl")
include("problem_types/telecom_network_design/telecom_network_design.jl")
include("problem_types/tsp/tsp.jl")
include("problem_types/transportation/transportation.jl")
include("problem_types/unit_commitment/unit_commitment.jl")
include("problem_types/vehicle_routing/vehicle_routing.jl")

# Batch dataset generation (uses the interface functions defined above)
include("dataset.jl")

end # module
