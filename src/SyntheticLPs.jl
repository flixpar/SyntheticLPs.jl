module SyntheticLPs

using JuMP
import Dualization
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
export list_tags, register_tag, variant_tags, model_class, supports_target
export model_statistics
export bounds_to_constraints!
export dualize_model, dual_reformulation, is_dual_reformulation
export ModelTransforms, apply_transforms, UnitScaling
export scale_units!, aggregate_rows!, elasticize_rows!, permute_model
export generate_dataset, plan_dataset, merge_manifests
export GeneratedInstance, GeneratedDataset, DatasetFailure, PlannedInstance
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
    VARIANT_TAGS

The controlled vocabulary of variant tags, mapping each tag to a one-line meaning.
[`register_variant`](@ref) rejects any tag not listed here (so a typo fails at load
time instead of silently never matching a filter); [`register_tag`](@ref) extends it.

Structure tags describe the constraint matrix a solver sees; domain tags describe the
application. Whether a variant is a pure LP or a natural MIP is *not* a tag: it is the
derived [`model_class`](@ref) property.
"""
const VARIANT_TAGS = Dict{Symbol, String}(
    # -- Structure -------------------------------------------------------------
    :network => "flow-conservation rows over a graph (node-arc incidence structure)",
    :multicommodity => "several commodities share arc or resource capacities",
    :bipartite => "transportation/assignment-type bipartite structure",
    :staircase => "multi-period model; consecutive periods coupled by balance rows",
    :block_angular => "independent blocks coupled by a few linking rows",
    :dual_block_angular => "independent blocks coupled by shared linking columns (e.g. two-stage stochastic)",
    :time_indexed => "time-indexed (discretized-time) scheduling formulation",
    :covering => "dominated by covering rows (sum >= demand)",
    :packing => "dominated by packing rows (sum <= capacity)",
    :partitioning => "dominated by partitioning/equality rows (sum == 1)",
    :big_m => "big-M or indicator-style linking rows (weak LP relaxation)",
    :dense => "dense rows or columns (a substantial fraction of the matrix is nonzero)",
    :blending => "ratio/proportion (quality-blend) constraints",
    :unimodular => "totally unimodular matrix: LP optimum is integral, little pivot-rule headroom",
    :degenerate => "known to be highly primal or dual degenerate",
    :lp_relaxation => "purpose-built continuous relaxation of a combinatorial problem",
    :robust => "robust or adversarial reformulation (dualized uncertainty sets)",
    # -- Domain ----------------------------------------------------------------
    :logistics => "transportation, distribution, and supply-chain planning",
    :routing => "vehicle/tour routing",
    :location => "facility, hub, or network location",
    :scheduling => "scheduling and timetabling of jobs, staff, or rooms",
    :production => "production, manufacturing, and process planning",
    :energy => "energy systems and power generation",
    :finance => "finance, portfolio, and revenue management",
    :healthcare => "healthcare and medical planning",
    :telecom => "telecommunication and network design",
    :agriculture => "agriculture, food, and land use",
    :statistics => "statistics, regression, and machine learning",
    :combinatorial => "abstract combinatorial optimization (graphs, sets, packing)",
)

"""
    register_tag(tag::Symbol, description::AbstractString)

Add `tag` to the [`VARIANT_TAGS`](@ref) vocabulary (or update its description).
"""
function register_tag(tag::Symbol, description::AbstractString)
    VARIANT_TAGS[tag] = String(description)
    return tag
end

"""
    list_tags() -> Vector{Pair{Symbol,String}}

Every known tag with its meaning, sorted by tag.
"""
list_tags() = sort!(collect(VARIANT_TAGS); by=first)

"""
    VariantSpec

Registry entry for a single variant: its category, variant name, generator type,
human-readable description, and dataset-builder metadata.

  - `tags`: structure/domain tags drawn from [`VARIANT_TAGS`](@ref).
  - `min_target_variables` / `max_target_variables`: the supported `target_variables`
    range (`max_target_variables === nothing` means no documented cap). A generator
    raises `ArgumentError` outside it; dataset generation never samples outside it.
  - `declared_model_class`: `:lp`/`:mip` when declared at registration, else
    `nothing` and [`model_class`](@ref) derives it lazily.
"""
struct VariantSpec
    category::Symbol
    variant::Symbol
    type::Type{<:ProblemGenerator}
    description::String
    tags::Vector{Symbol}
    min_target_variables::Int
    max_target_variables::Union{Int, Nothing}
    declared_model_class::Union{Symbol, Nothing}
end

"""
    CategorySpec

Registry entry for a category: its description, the variants registered under
it, and which variant is used by default when none is named.
"""
mutable struct CategorySpec
    category::Symbol
    description::String
    variants::Dict{Symbol, VariantSpec}
    default_variant::Union{Symbol, Nothing}
    explicit_default::Bool
end

# Maps category symbol -> CategorySpec.
const LP_REGISTRY = Dict{Symbol, CategorySpec}()

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
        CategorySpec(category, String(description), Dict{Symbol, VariantSpec}(), nothing, false)
    end
    # Always apply the explicit description, even if the category was already
    # created lazily by `register_variant`, so registration order doesn't matter.
    cat.description = String(description)
    return cat
end

"""
    register_variant(category::Symbol, variant::Symbol,
                     problem_type::Type{<:ProblemGenerator}, description::AbstractString;
                     default::Bool=false, tags=Symbol[], min_target_variables::Int=1,
                     max_target_variables::Union{Int,Nothing}=nothing,
                     model_class::Union{Symbol,Nothing}=nothing)

Register a `variant` of `category` backed by `problem_type`. If the category is
not yet registered, it is created lazily using `description`.

The first variant registered becomes the category default; pass `default=true`
to designate a specific variant instead (only one variant may be the explicit
default).

Optional metadata, surfaced by [`problem_info`](@ref) and used by
[`list_problems`](@ref) and [`generate_dataset`](@ref) filters:

  - `tags`: structure/domain tags; each must be in [`VARIANT_TAGS`](@ref).
  - `min_target_variables`, `max_target_variables`: the supported target range.
    Set `max_target_variables` when the generator documents a size cap (and raises
    `ArgumentError` above it).
  - `model_class`: `:lp` or `:mip`, overriding the lazily derived
    [`model_class`](@ref). Normally leave it unset.
"""
function register_variant(
    category::Symbol,
    variant::Symbol,
    problem_type::Type{<:ProblemGenerator},
    description::AbstractString;
    default::Bool=false,
    tags=Symbol[],
    min_target_variables::Int=1,
    max_target_variables::Union{Int, Nothing}=nothing,
    model_class::Union{Symbol, Nothing}=nothing,
)
    tag_vec = Symbol[Symbol(t) for t in tags]
    unknown_tags = filter(t -> !haskey(VARIANT_TAGS, t), tag_vec)
    isempty(unknown_tags) || error(
        "Unknown tags for $category/$variant: $(join(unknown_tags, ", ")). " *
        "Known tags: $(join(sort(collect(keys(VARIANT_TAGS))), ", ")). " *
        "Use register_tag to add one.",
    )
    min_target_variables >= 1 || error("min_target_variables must be >= 1 for $category/$variant.")
    max_target_variables === nothing ||
        max_target_variables >= min_target_variables ||
        error("max_target_variables must be >= min_target_variables for $category/$variant.")
    model_class in (nothing, :lp, :mip) ||
        error("model_class must be :lp, :mip, or nothing (got $model_class).")

    cat = get(LP_REGISTRY, category, nothing)
    if cat === nothing
        cat = register_category(category, description)
    end
    if haskey(cat.variants, variant)
        error("Variant $category/$variant is already registered.")
    end
    spec = VariantSpec(
        category,
        variant,
        problem_type,
        String(description),
        sort!(unique(tag_vec)),
        min_target_variables,
        max_target_variables,
        model_class,
    )
    cat.variants[variant] = spec
    if default
        if cat.explicit_default
            error(
                "Category $category already has an explicit default variant " *
                "($(cat.default_variant)); cannot also mark $variant as default.",
            )
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
    haskey(LP_REGISTRY, category) || error(
        "Unknown problem category: $category. " *
        "Use list_categories() to see available categories.",
    )
    return LP_REGISTRY[category]
end

"""
    get_variant(ref::ProblemVariant) -> VariantSpec

Internal: fetch a variant's registry entry, erroring helpfully if unknown.
"""
function get_variant(ref::ProblemVariant)
    cat = get_category(ref.category)
    haskey(cat.variants, ref.variant) || error(
        "Unknown variant $(ref.category)/$(ref.variant). " *
        "Available variants of $(ref.category): " *
        "$(join(sort(collect(keys(cat.variants))), ", ")).",
    )
    return cat.variants[ref.variant]
end

"""
    default_variant(category::Symbol) -> Symbol

The default variant symbol for a category.
"""
function default_variant(category::Symbol)
    cat = get_category(category)
    cat.default_variant === nothing && error("Category $category has no registered variants.")
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
    error("Invalid problem reference \"$s\"; expected \"category\" or " * "\"category/variant\".")
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
                               relax_integer, bounds_to_constraints, dualize, transforms,
                               optimizer,
                               max_feasibility_retries, feasibility_timeout)

Internal builder used by [`generate_problem`](@ref). Constructs the problem and its
JuMP model, applies `relax_integer` and `bounds_to_constraints`, optionally
verifies that primal model, then applies `transforms` (seeded by the resolved
seed) and finally `dualize` before returning.

When `optimizer` is supplied and `feasibility_status` is `feasible` or `infeasible`,
the model is solved once to verify the feasibility contract — a `feasible` request
must solve to `OPTIMAL`, an `infeasible` request must solve to `INFEASIBLE`. If the
solve *disproves* the requested status the problem is rebuilt with the next seed and
re-checked, up to `max_feasibility_retries` times. (Generators aim to honor the
requested status by construction, but a few have heuristic feasibility logic that
occasionally misses; this central check is the project-level backstop, so callers
receive a dual of a conforming primal or an error when the retry budget is exhausted.)

If instead the solve *certifies nothing* — it hits `feasibility_timeout`, or returns a
status that separates neither case — the retry budget is not spent: verification
raises immediately, reporting the termination status. Unrelaxed MIPs are the usual
cause; give them a larger `feasibility_timeout`.

Returns `(model, problem, resolved_seed)`. With `optimizer=nothing` (or status
`unknown`) the model is built exactly once and `resolved_seed == seed`. Verification
is itself deterministic — attempts walk `seed, seed+1, …` — so a given
`(seed, optimizer)` pair always resolves to the same model.

When `info` is a `Dict{Symbol,Any}`, it is filled with diagnostics of the returned
model's final attempt: `:num_integer` (integer/binary columns *before* relaxation),
`:build_time` (seconds spent constructing the generator, building, and applying the
pre-dualization transforms — excluding verification solves), `:attempts`, and, when a
verification solve ran, `:verification_status` (its termination status).
"""
function _generate_problem_verified(
    ::Type{T},
    target_variables::Int,
    feasibility_status::FeasibilityStatus,
    seed::Int;
    relax_integer::Bool=true,
    bounds_to_constraints::Bool=false,
    dualize::Bool=false,
    transforms=ModelTransforms(),
    optimizer=nothing,
    max_feasibility_retries::Int=10,
    feasibility_timeout::Float64=10.0,
    info::Union{Nothing, AbstractDict}=nothing,
) where {T <: ProblemGenerator}
    max_feasibility_retries >= 1 ||
        error("max_feasibility_retries must be >= 1 (got $max_feasibility_retries).")
    transforms = _as_transforms(transforms)
    _check_transforms_status(transforms, feasibility_status)
    needs_check = optimizer !== nothing && feasibility_status !== unknown

    current_seed = seed
    model = nothing
    problem = nothing
    for attempt in 1:max_feasibility_retries
        attempt_start = time()
        problem = T(target_variables, feasibility_status, current_seed)
        model = build_model(problem)
        info === nothing || (info[:num_integer] = _count_integer(model))
        relax_integer && relax_integrality(model)
        bounds_to_constraints && bounds_to_constraints!(model)
        if info !== nothing
            info[:build_time] = time() - attempt_start
            info[:attempts] = attempt
        end
        if !needs_check
            return _finalize_model(model, transforms, current_seed, dualize), problem, current_seed
        end
        verdict, ts = _check_feasibility_contract(
            model, optimizer, feasibility_status; timeout=feasibility_timeout
        )
        info === nothing || (info[:verification_status] = ts)
        if verdict === :holds
            return _finalize_model(model, transforms, current_seed, dualize), problem, current_seed
        elseif verdict === :inconclusive
            # The solve certified nothing, so we have no evidence against this
            # instance and rebuilding would just re-ask an unanswerable question.
            # Report the real cause instead of charging it to the retry budget.
            error(
                "Feasibility contract could not be verified for $T " *
                "(target_variables=$target_variables, status=$feasibility_status, " *
                "seed=$current_seed): the verification solve returned $ts " *
                "after a $(feasibility_timeout)s limit. This is not evidence of a " *
                "contract violation. Raise `feasibility_timeout`, use a stronger " *
                "optimizer, or drop `optimizer` to skip verification.",
            )
        end
        # Contract disproved — rebuild with a fresh seed if another attempt remains.
        attempt < max_feasibility_retries && (current_seed += 1)
    end

    error(
        "Feasibility contract not satisfied for $T " *
        "(target_variables=$target_variables, status=$feasibility_status) " *
        "after $max_feasibility_retries attempts " *
        "(seeds $seed through $current_seed); no model was returned.",
    )
end

# Post-verification steps: the practitioner-style transforms (seeded by the
# resolved instance seed), then optional dualization of the transformed primal.
function _finalize_model(model::Model, transforms::ModelTransforms, seed::Int, dualize::Bool)
    model = apply_transforms(model, transforms, seed)
    return dualize ? dualize_model(model) : model
end

# Ref-based overload delegating to the type-based builder above.
function _generate_problem_verified(
    ref::ProblemVariant,
    target_variables::Int,
    feasibility_status::FeasibilityStatus,
    seed::Int;
    kwargs...,
)
    return _generate_problem_verified(
        get_problem_type(ref), target_variables, feasibility_status, seed; kwargs...
    )
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
function _check_feasibility_contract(
    model::Model, optimizer, feasibility_status::FeasibilityStatus; timeout::Float64=10.0
)
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
                     relax_integer=true, bounds_to_constraints=false, dualize=false,
                     transforms=ModelTransforms(),
                     optimizer=nothing, max_feasibility_retries=10,
                     feasibility_timeout=10.0)

Generate a linear programming problem from a generator type by constructing an
instance and building its model.

When `bounds_to_constraints=true`, variable bounds (other than plain `x ≥ 0`
nonnegativity) are reformulated as explicit affine constraints via
[`bounds_to_constraints!`](@ref). This runs *after* integrality relaxation, so
bounds introduced by relaxing integer/binary variables are converted too.

When `dualize=true`, the generated continuous model is replaced by its dual via
[`dualize_model`](@ref). Dualization runs after integrality relaxation and bound
reformulation. Set `relax_integer=false` only for models that are continuous by
construction; integer and binary variables cannot be dualized. If feasibility
verification is enabled, it checks the source primal before dualization because
an infeasible primal's dual may be either infeasible or unbounded.

`transforms` (a [`ModelTransforms`](@ref), or a `NamedTuple` of its keyword
arguments) applies practitioner-style reformulations — unit scaling, redundant
aggregate rows, elastic rows, row/column permutation — after integrality
relaxation, bound reformulation and feasibility verification, and before
dualization, so a dualized instance is the dual of the transformed primal. They
are seeded by the instance seed. The default is the identity. Scaling, aggregation
and permutation preserve the feasibility label exactly; elastic rows are a
relaxation and are refused for `infeasible` requests (see
[`apply_transforms`](@ref)).

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
function generate_problem(
    ::Type{T},
    target_variables::Int,
    feasibility_status::FeasibilityStatus=unknown,
    seed::Int=0;
    relax_integer::Bool=true,
    bounds_to_constraints::Bool=false,
    dualize::Bool=false,
    transforms=ModelTransforms(),
    optimizer=nothing,
    max_feasibility_retries::Int=10,
    feasibility_timeout::Float64=10.0,
) where {T <: ProblemGenerator}
    model, problem, _ = _generate_problem_verified(
        T,
        target_variables,
        feasibility_status,
        seed;
        relax_integer=relax_integer,
        bounds_to_constraints=bounds_to_constraints,
        dualize=dualize,
        transforms=transforms,
        optimizer=optimizer,
        max_feasibility_retries=max_feasibility_retries,
        feasibility_timeout=feasibility_timeout,
    )
    return model, problem
end

"""
    generate_problem(ref::ProblemVariant, target_variables, feasibility_status, seed;
                     relax_integer=true, bounds_to_constraints=false, dualize=false,
                     transforms=ModelTransforms(),
                     optimizer=nothing, max_feasibility_retries=10,
                     feasibility_timeout=10.0)

Generate a problem from a fully-qualified `category/variant` reference.
"""
function generate_problem(
    ref::ProblemVariant,
    target_variables::Int,
    feasibility_status::FeasibilityStatus=unknown,
    seed::Int=0;
    relax_integer::Bool=true,
    bounds_to_constraints::Bool=false,
    dualize::Bool=false,
    transforms=ModelTransforms(),
    optimizer=nothing,
    max_feasibility_retries::Int=10,
    feasibility_timeout::Float64=10.0,
)
    return generate_problem(
        get_problem_type(ref),
        target_variables,
        feasibility_status,
        seed;
        relax_integer=relax_integer,
        bounds_to_constraints=bounds_to_constraints,
        dualize=dualize,
        transforms=transforms,
        optimizer=optimizer,
        max_feasibility_retries=max_feasibility_retries,
        feasibility_timeout=feasibility_timeout,
    )
end

"""
    generate_problem(ref::AbstractString, target_variables, feasibility_status, seed;
                     relax_integer=true, bounds_to_constraints=false, dualize=false,
                     transforms=ModelTransforms(),
                     optimizer=nothing, max_feasibility_retries=10,
                     feasibility_timeout=10.0)

Generate a problem from a `"category"` or `"category/variant"` string, parsed via
[`ProblemVariant`](@ref).
"""
function generate_problem(
    ref::AbstractString,
    target_variables::Int,
    feasibility_status::FeasibilityStatus=unknown,
    seed::Int=0;
    relax_integer::Bool=true,
    bounds_to_constraints::Bool=false,
    dualize::Bool=false,
    transforms=ModelTransforms(),
    optimizer=nothing,
    max_feasibility_retries::Int=10,
    feasibility_timeout::Float64=10.0,
)
    return generate_problem(
        ProblemVariant(ref),
        target_variables,
        feasibility_status,
        seed;
        relax_integer=relax_integer,
        bounds_to_constraints=bounds_to_constraints,
        dualize=dualize,
        transforms=transforms,
        optimizer=optimizer,
        max_feasibility_retries=max_feasibility_retries,
        feasibility_timeout=feasibility_timeout,
    )
end

"""
    generate_problem(category::Symbol, target_variables, feasibility_status, seed;
                     variant=nothing, relax_integer=true, bounds_to_constraints=false,
                     dualize=false, transforms=ModelTransforms(),
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
  - `dualize`: Replace the continuous generated model with its dual formulation
  - `transforms`: Practitioner-style reformulations ([`ModelTransforms`](@ref)),
    applied before dualization; the identity by default
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
function generate_problem(
    category::Symbol,
    target_variables::Int,
    feasibility_status::FeasibilityStatus=unknown,
    seed::Int=0;
    variant::Union{Symbol, Nothing}=nothing,
    relax_integer::Bool=true,
    bounds_to_constraints::Bool=false,
    dualize::Bool=false,
    transforms=ModelTransforms(),
    optimizer=nothing,
    max_feasibility_retries::Int=10,
    feasibility_timeout::Float64=10.0,
)
    ref = variant === nothing ? ProblemVariant(category) : ProblemVariant(category, variant)
    return generate_problem(
        ref,
        target_variables,
        feasibility_status,
        seed;
        relax_integer=relax_integer,
        bounds_to_constraints=bounds_to_constraints,
        dualize=dualize,
        transforms=transforms,
        optimizer=optimizer,
        max_feasibility_retries=max_feasibility_retries,
        feasibility_timeout=feasibility_timeout,
    )
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

# Every registered variant, sorted by (category, variant). A stable order matters
# wherever an RNG consumes the list positionally (dataset planning, random problem
# selection): an unsorted order would make seeded output depend on Dict layout.
function _all_problems()
    refs = ProblemVariant[]
    for category in sort(collect(keys(LP_REGISTRY)))
        for variant in list_variants(category)
            push!(refs, ProblemVariant(category, variant))
        end
    end
    return refs
end

# A single selector is accepted wherever a collection of selectors is.
_selector_list(sel::Union{Symbol, AbstractString, ProblemVariant}) = [sel]
_selector_list(sels) = collect(sels)

# Expand a single selector into its concrete variants, validating against the
# registry. A category expands to all its (sorted) variants; an explicit
# `category/variant` reference resolves to just that variant.
function _expand_selector(sel::ProblemVariant)
    get_variant(sel)  # validates category + variant; throws if unknown
    return [sel]
end
function _expand_selector(sel::AbstractString)
    return if occursin('/', sel)
        _expand_selector(ProblemVariant(sel))
    else
        _expand_selector(Symbol(strip(sel)))
    end
end
function _expand_selector(sel::Symbol)
    haskey(LP_REGISTRY, sel) || error(
        "Unknown problem category: $sel. " * "Available: $(join(sort(list_categories()), ", "))"
    )
    return [ProblemVariant(sel, v) for v in list_variants(sel)]
end

"""
    resolve_problem_types(problem_types) -> Vector{ProblemVariant}

Normalize a user-supplied selection into a validated, de-duplicated vector of
`ProblemVariant`s, in (category, variant) order.

`nothing` or an empty collection selects every registered variant. Each selector
may be:

  - a category `Symbol` (e.g. `:transportation`) or bare string (`"transportation"`),
    which expands to *all* variants of that category;
  - a `"category/variant"` string or a `ProblemVariant`, naming one specific variant.

A single selector need not be wrapped in a collection. Throws if any requested
category or variant is not registered.
"""
function resolve_problem_types(problem_types)
    if problem_types === nothing || isempty(_selector_list(problem_types))
        return _all_problems()
    end
    resolved = ProblemVariant[]
    for sel in _selector_list(problem_types)
        append!(resolved, _expand_selector(sel))
    end
    return sort!(unique(resolved); by=r -> (r.category, r.variant))
end

_tag_list(tags::Symbol) = [tags]
_tag_list(tags::AbstractString) = [Symbol(tags)]
_tag_list(tags) = Symbol[Symbol(t) for t in tags]

_as_variant(ref::ProblemVariant) = ref
_as_variant(ref::Union{Symbol, AbstractString}) = ProblemVariant(ref)

"""
    variant_tags(ref) -> Vector{Symbol}

The registered tags of a variant (`ProblemVariant`, category `Symbol`, or string).
"""
variant_tags(ref) = copy(get_variant(_as_variant(ref)).tags)

"""
    supports_target(ref, target_variables::Integer) -> Bool

Whether `target_variables` lies in the variant's registered
`min_target_variables:max_target_variables` range.
"""
function supports_target(ref, target_variables::Integer)
    spec = get_variant(_as_variant(ref))
    target_variables < spec.min_target_variables && return false
    spec.max_target_variables === nothing && return true
    return target_variables <= spec.max_target_variables
end

# Lazily derived model classes, keyed by variant. Guarded by a lock so concurrent
# callers (e.g. threaded dataset generation) cannot race on the Dict.
const _MODEL_CLASS_CACHE = Dict{ProblemVariant, Symbol}()
const _MODEL_CLASS_LOCK = ReentrantLock()

# Target size of the probe instance used to derive `model_class`.
const MODEL_CLASS_PROBE_TARGET = 200

"""
    model_class(ref) -> Symbol

`:mip` if the variant's `build_model` emits integer or binary variables, else `:lp`.
Note `generate_problem` relaxes integrality by default, so a `:mip` variant is
still returned as an LP unless `relax_integer=false`.

Unless declared at registration, the class is derived by building one probe instance
(`target_variables = 200`, clamped to the variant's supported range; status
`unknown`; seed 1) and is cached for the rest of the session. Accepts a
`ProblemVariant`, a category `Symbol` (its default variant), or a string.
"""
function model_class(ref)
    pv = _as_variant(ref)
    spec = get_variant(pv)
    spec.declared_model_class === nothing || return spec.declared_model_class
    return lock(_MODEL_CLASS_LOCK) do
        get!(() -> _probe_model_class(spec), _MODEL_CLASS_CACHE, pv)
    end
end

function _probe_model_class(spec::VariantSpec)
    hi = something(spec.max_target_variables, typemax(Int))
    target = clamp(MODEL_CLASS_PROBE_TARGET, spec.min_target_variables, hi)
    model = build_model(spec.type(target, unknown, 1))
    return _count_integer(model) > 0 ? :mip : :lp
end

_count_integer(model::Model) = count(x -> is_integer(x) || is_binary(x), all_variables(model))

function _count_nonzeros(model::Model)
    nnz = 0
    for (F, S) in list_of_constraint_types(model)
        F <: GenericAffExpr || continue
        for c in all_constraints(model, F, S)
            nnz += count(!iszero, values(constraint_object(c).func.terms))
        end
    end
    return nnz
end

"""
    model_statistics(model::Model) -> NamedTuple

Size statistics of a built JuMP model: `num_variables`, `num_constraints` (affine
rows only, i.e. excluding variable bounds), `num_nonzeros` (nonzero coefficients in
affine rows), and `num_integer` (integer or binary columns at the time of the call —
zero after the default integrality relaxation).
"""
function model_statistics(model::Model)
    return (
        num_variables=num_variables(model),
        num_constraints=num_constraints(model; count_variable_in_set_constraints=false),
        num_nonzeros=_count_nonzeros(model),
        num_integer=_count_integer(model),
    )
end

"""
    list_problems(; problem_types=nothing, exclude=nothing, model_class=nothing,
                  tags=nothing, any_tags=nothing, exclude_tags=nothing,
                  target_variables=nothing) -> Vector{ProblemVariant}

List registered `category/variant` pairs, sorted by category then variant. With no
arguments, every variant. Filters compose (a variant must pass all of them):

  - `problem_types`: selectors to restrict to (categories, `"category/variant"`
    strings, or `ProblemVariant`s; see [`resolve_problem_types`](@ref)).
  - `exclude`: selectors to remove, same forms.
  - `model_class`: `:lp` or `:mip` (see [`model_class`](@ref); deriving it builds
    one small probe instance per variant the first time).
  - `tags`: keep variants carrying *all* of these tags.
  - `any_tags`: keep variants carrying *at least one* of these tags.
  - `exclude_tags`: drop variants carrying any of these tags.
  - `target_variables`: an `Integer` (keep variants supporting that target) or a
    `(lo, hi)` tuple (keep variants supporting the whole range); see
    [`supports_target`](@ref).

Unknown selectors or tags are errors, so a typo cannot silently select nothing.
"""
function list_problems(;
    problem_types=nothing,
    exclude=nothing,
    model_class::Union{Symbol, Nothing}=nothing,
    tags=nothing,
    any_tags=nothing,
    exclude_tags=nothing,
    target_variables=nothing,
)
    for t in (tags, any_tags, exclude_tags)
        t === nothing && continue
        unknown_tags = filter(x -> !haskey(VARIANT_TAGS, x), _tag_list(t))
        isempty(unknown_tags) || error(
            "Unknown tags: $(join(unknown_tags, ", ")). " *
            "Known tags: $(join(sort(collect(keys(VARIANT_TAGS))), ", ")).",
        )
    end
    model_class in (nothing, :lp, :mip) ||
        error("model_class must be :lp or :mip (got $model_class).")

    refs = resolve_problem_types(problem_types)
    if exclude !== nothing && !isempty(_selector_list(exclude))
        excluded = Set(resolve_problem_types(exclude))
        refs = filter(r -> !(r in excluded), refs)
    end
    if tags !== nothing
        required = _tag_list(tags)
        refs = filter(r -> all(in(get_variant(r).tags), required), refs)
    end
    if any_tags !== nothing
        wanted = _tag_list(any_tags)
        refs = filter(r -> any(in(get_variant(r).tags), wanted), refs)
    end
    if exclude_tags !== nothing
        banned = _tag_list(exclude_tags)
        refs = filter(r -> !any(in(get_variant(r).tags), banned), refs)
    end
    if target_variables !== nothing
        lo, hi = if target_variables isa Integer
            (target_variables, target_variables)
        else
            (ceil(Int, target_variables[1]), floor(Int, target_variables[2]))
        end
        refs = filter(r -> supports_target(r, lo) && supports_target(r, hi), refs)
    end
    if model_class !== nothing
        # Filter last: deriving the class builds a probe instance per variant.
        refs = filter(r -> SyntheticLPs.model_class(r) === model_class, refs)
    end
    return refs
end

"""
    problem_info(category::Symbol) -> Dict

Information about a category: its description, variants, default variant, and the
union of its variants' tags.
"""
function problem_info(category::Symbol)
    cat = get_category(category)
    variants = list_variants(category)
    return Dict(
        :type => category,
        :category => category,
        :description => cat.description,
        :variants => variants,
        :num_variants => length(variants),
        :default_variant => cat.default_variant,
        :tags => sort!(unique(t for v in values(cat.variants) for t in v.tags)),
    )
end

"""
    problem_info(category::Symbol, variant::Symbol) -> Dict
    problem_info(ref::ProblemVariant) -> Dict

Information about a specific variant: `:description`, generator `:type`, `:ref`,
`:default` (whether it is the category default), `:tags`, the supported
`:min_target_variables`/`:max_target_variables` range (`nothing` = no cap), and
`:model_class` (derived lazily; see [`model_class`](@ref)).
"""
function problem_info(category::Symbol, variant::Symbol)
    ref = ProblemVariant(category, variant)
    spec = get_variant(ref)
    return Dict(
        :category => spec.category,
        :variant => spec.variant,
        :ref => ref,
        :description => spec.description,
        :type => spec.type,
        :default => get_category(category).default_variant == variant,
        :tags => copy(spec.tags),
        :min_target_variables => spec.min_target_variables,
        :max_target_variables => spec.max_target_variables,
        :model_class => model_class(ref),
    )
end

problem_info(ref::ProblemVariant) = problem_info(ref.category, ref.variant)

"""
    variant_weights(refs, weighting=:category) -> Vector{Float64}

Sampling weights (normalized to sum to 1) for the variants `refs` under a weighting
scheme:

  - `:category` — uniform over the categories present in `refs`, then uniform over
    each category's variants in `refs`. A category with 8 variants gets the same
    share as one with a single variant.
  - `:variant` — uniform over `refs`.
  - an `AbstractDict` of explicit (unnormalized, nonnegative) weights. A
    `"category/variant"` string or `ProblemVariant` key weights that variant; a
    category `Symbol`/string key's weight is split evenly among that category's
    variants in `refs` that have no key of their own. Variants matched by no key
    get weight 0.
"""
function variant_weights(refs::AbstractVector{ProblemVariant}, weighting=:category)
    isempty(refs) && error("No problem variants to weight.")
    w = if weighting === :variant
        ones(length(refs))
    elseif weighting === :category
        per_cat = Dict{Symbol, Int}()
        for r in refs
            per_cat[r.category] = get(per_cat, r.category, 0) + 1
        end
        [1.0 / per_cat[r.category] for r in refs]
    elseif weighting isa AbstractDict
        _explicit_variant_weights(refs, weighting)
    else
        error("variant_weighting must be :category, :variant, or a Dict (got $weighting).")
    end
    total = sum(w)
    total > 0 || error("variant weights select no variant (all weights are zero).")
    return w ./ total
end

function _explicit_variant_weights(refs, weights::AbstractDict)
    variant_w = Dict{ProblemVariant, Float64}()
    category_w = Dict{Symbol, Float64}()
    for (key, value) in weights
        value >= 0 || error("Variant weights must be nonnegative (got $key => $value).")
        if key isa ProblemVariant || (key isa AbstractString && occursin('/', key))
            ref = key isa ProblemVariant ? key : ProblemVariant(key)
            get_variant(ref)
            variant_w[ref] = Float64(value)
        else
            category = Symbol(key)
            get_category(category)
            category_w[category] = Float64(value)
        end
    end
    unkeyed = Dict{Symbol, Int}()
    for r in refs
        haskey(variant_w, r) || (unkeyed[r.category] = get(unkeyed, r.category, 0) + 1)
    end
    return [
        if haskey(variant_w, r)
            variant_w[r]
        else
            get(category_w, r.category, 0.0) / unkeyed[r.category]
        end for r in refs
    ]
end

# Draw one element of `refs` with probability proportional to `weights` (which sum to
# one). A single uniform draw keeps the RNG consumption fixed per call.
function _weighted_choice(rng::AbstractRNG, refs, weights)
    u = rand(rng)
    acc = 0.0
    for (r, w) in zip(refs, weights)
        acc += w
        u < acc && return r
    end
    return refs[findlast(>(0), weights)]
end

"""
    generate_random_problem(target_variables; feasibility_status=unknown,
                            relax_integer=true, bounds_to_constraints=false,
                            dualize=false, dualize_probability=0.0,
                            transforms=ModelTransforms(), seed=0,
                            problem_types=nothing, variant_weighting=:category,
                            optimizer=nothing, max_feasibility_retries=10,
                            feasibility_timeout=10.0)

Generate a problem of a randomly selected variant targeting approximately the
specified number of variables. The variant is drawn from `problem_types` (any
selector accepted by [`list_problems`](@ref); `nothing` = every registered variant
that supports `target_variables`) with weights from
[`variant_weights`](@ref)`(refs, variant_weighting)` — by default uniform over
categories, then over each category's variants. Dualization is off by default. Set
`dualize_probability` to a value in `[0, 1]` to randomly dualize the selected
model, reproducibly from `seed`; `dualize=true` forces dualization regardless of
the probability. `transforms` is applied as in [`generate_problem`](@ref). When
`optimizer` is supplied and `feasibility_status` is `feasible`/`infeasible`, the
feasibility contract is verified (see
[`generate_problem`](@ref)).

# Returns

  - `model`: The JuMP model
  - `ref`: The `ProblemVariant` that was selected
  - `problem`: The problem generator instance
"""
function generate_random_problem(
    target_variables::Int;
    feasibility_status::FeasibilityStatus=unknown,
    relax_integer::Bool=true,
    bounds_to_constraints::Bool=false,
    dualize::Bool=false,
    dualize_probability::Real=0.0,
    transforms=ModelTransforms(),
    seed::Int=0,
    problem_types=nothing,
    variant_weighting=:category,
    optimizer=nothing,
    max_feasibility_retries::Int=10,
    feasibility_timeout::Float64=10.0,
)
    probability = _validate_dualize_probability(dualize_probability)
    rng = MersenneTwister(seed)

    problems = list_problems(; problem_types=problem_types, target_variables=target_variables)
    if isempty(problems)
        error("No registered problem variant matches the selection.")
    end

    ref = _weighted_choice(rng, problems, variant_weights(problems, variant_weighting))
    apply_dualization = _should_dualize(rng, dualize, probability)
    model, problem = generate_problem(
        ref,
        target_variables,
        feasibility_status,
        seed;
        relax_integer=relax_integer,
        bounds_to_constraints=bounds_to_constraints,
        dualize=apply_dualization,
        transforms=transforms,
        optimizer=optimizer,
        max_feasibility_retries=max_feasibility_retries,
        feasibility_timeout=feasibility_timeout,
    )

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
include("problem_types/game_theory/game_theory.jl")
include("problem_types/graph_optimization/graph_optimization.jl")
include("problem_types/hub_location/hub_location.jl")
include("problem_types/inventory/inventory.jl")
include("problem_types/inverse_optimization/inverse_optimization.jl")
include("problem_types/job_shop_scheduling/job_shop_scheduling.jl")
include("problem_types/knapsack/knapsack.jl")
include("problem_types/land_use/land_use.jl")
include("problem_types/load_balancing/load_balancing.jl")
include("problem_types/maritime_inventory_routing/maritime_inventory_routing.jl")
include("problem_types/mine_planning/mine_planning.jl")
include("problem_types/multi_commodity_flow/multi_commodity_flow.jl")
include("problem_types/network_flow/network_flow.jl")
include("problem_types/neural_network_verification/neural_network_verification.jl")
include("problem_types/nurse_scheduling/nurse_scheduling.jl")
include("problem_types/operating_room_scheduling/operating_room_scheduling.jl")
include("problem_types/portfolio/portfolio.jl")
include("problem_types/product_mix/product_mix.jl")
include("problem_types/process_planning/process_planning.jl")
include("problem_types/production_planning/production_planning.jl")
include("problem_types/project_selection/project_selection.jl")
include("problem_types/regression/regression.jl")
include("problem_types/radiotherapy/radiotherapy.jl")
include("problem_types/resilient_network_design/resilient_network_design.jl")
include("problem_types/resource_allocation/resource_allocation.jl")
include("problem_types/revenue_management/revenue_management.jl")
include("problem_types/scheduling/scheduling.jl")
include("problem_types/set_system/set_system.jl")
include("problem_types/stochastic_program/stochastic_program.jl")
include("problem_types/supply_chain/supply_chain.jl")
include("problem_types/telecom_network_design/telecom_network_design.jl")
include("problem_types/tsp/tsp.jl")
include("problem_types/transportation/transportation.jl")
include("problem_types/unit_commitment/unit_commitment.jl")
include("problem_types/vehicle_routing/vehicle_routing.jl")
include("problem_types/workforce_shift_scheduling/workforce_shift_scheduling.jl")

# Batch dataset generation (uses the interface functions defined above)
include("dataset.jl")

end # module
