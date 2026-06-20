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
    return get!(LP_REGISTRY, category,
                CategorySpec(category, String(description),
                             Dict{Symbol,VariantSpec}(), nothing, false))
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
    generate_problem(::Type{T}, target_variables, feasibility_status, seed; relax_integer=true)

Generate a linear programming problem from a generator type by constructing an
instance and building its model.

# Returns
- `model`: The JuMP model
- `problem`: The problem generator instance containing all parameters
"""
function generate_problem(::Type{T}, target_variables::Int,
                          feasibility_status::FeasibilityStatus=unknown, seed::Int=0;
                          relax_integer::Bool=true) where T <: ProblemGenerator
    problem = T(target_variables, feasibility_status, seed)
    model = build_model(problem)

    if relax_integer
        relax_integrality(model)
    end

    return model, problem
end

"""
    generate_problem(ref::ProblemVariant, target_variables, feasibility_status, seed; relax_integer=true)

Generate a problem from a fully-qualified `category/variant` reference.
"""
function generate_problem(ref::ProblemVariant, target_variables::Int,
                          feasibility_status::FeasibilityStatus=unknown, seed::Int=0;
                          relax_integer::Bool=true)
    return generate_problem(get_problem_type(ref), target_variables,
                            feasibility_status, seed; relax_integer=relax_integer)
end

"""
    generate_problem(category::Symbol, target_variables, feasibility_status, seed;
                     variant=nothing, relax_integer=true)

Generate a problem for a category. With `variant=nothing` the category's default
variant is used; pass `variant=:name` to select a specific variant.

# Arguments
- `category`: Problem category symbol (e.g. `:transportation`)
- `target_variables`: Target number of variables in the LP formulation
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
- `variant`: Optional variant symbol; defaults to the category default

# Returns
- `model`: The JuMP model
- `problem`: The problem generator instance
"""
function generate_problem(category::Symbol, target_variables::Int,
                          feasibility_status::FeasibilityStatus=unknown, seed::Int=0;
                          variant::Union{Symbol,Nothing}=nothing, relax_integer::Bool=true)
    ref = variant === nothing ? ProblemVariant(category) :
                                ProblemVariant(category, variant)
    return generate_problem(ref, target_variables, feasibility_status, seed;
                            relax_integer=relax_integer)
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
    generate_random_problem(target_variables; feasibility_status=unknown, relax_integer=true, seed=0)

Generate a problem of a randomly selected variant targeting approximately the
specified number of variables. Sampling is uniform over all registered
`category/variant` pairs.

# Returns
- `model`: The JuMP model
- `ref`: The `ProblemVariant` that was selected
- `problem`: The problem generator instance
"""
function generate_random_problem(target_variables::Int;
                                 feasibility_status::FeasibilityStatus=unknown,
                                 relax_integer::Bool=true, seed::Int=0)
    Random.seed!(seed)

    problems = list_problems()
    if isempty(problems)
        error("No problem types registered. Include problem type files first.")
    end

    ref = rand(problems)
    model, problem = generate_problem(ref, target_variables, feasibility_status, seed;
                                      relax_integer=relax_integer)

    return model, ref, problem
end

# ---------------------------------------------------------------------------
# Problem generators
# ---------------------------------------------------------------------------
# Each category lives in its own folder; the `<category>.jl` entry point
# registers the category (if needed) and includes one file per variant.
include("problem_types/airline_crew/airline_crew.jl")
include("problem_types/assignment/assignment.jl")
include("problem_types/blending/blending.jl")
include("problem_types/crop_planning/crop_planning.jl")
include("problem_types/cutting_stock/cutting_stock.jl")
include("problem_types/diet_problem/diet_problem.jl")
include("problem_types/energy/energy.jl")
include("problem_types/facility_location/facility_location.jl")
include("problem_types/feed_blending/feed_blending.jl")
include("problem_types/inventory/inventory.jl")
include("problem_types/knapsack/knapsack.jl")
include("problem_types/land_use/land_use.jl")
include("problem_types/load_balancing/load_balancing.jl")
include("problem_types/multi_commodity_flow/multi_commodity_flow.jl")
include("problem_types/network_flow/network_flow.jl")
include("problem_types/portfolio/portfolio.jl")
include("problem_types/product_mix/product_mix.jl")
include("problem_types/production_planning/production_planning.jl")
include("problem_types/project_selection/project_selection.jl")
include("problem_types/resource_allocation/resource_allocation.jl")
include("problem_types/scheduling/scheduling.jl")
include("problem_types/supply_chain/supply_chain.jl")
include("problem_types/telecom_network_design/telecom_network_design.jl")
include("problem_types/transportation/transportation.jl")

# Batch dataset generation (uses the interface functions defined above)
include("dataset.jl")

end # module
