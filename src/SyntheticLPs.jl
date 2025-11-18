module SyntheticLPs

using JuMP
using Random
using Distributions

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
export generate_problem
export list_problem_types
export problem_info
export generate_random_problem
export register_problem
export list_problem_variants
export get_base_type
export is_variant

# Registration system
# Dictionary to store problem type metadata
const LP_REGISTRY = Dict{Symbol, Dict{Symbol, Any}}()

"""
    register_problem(type_sym::Symbol, problem_type::Type{<:ProblemGenerator}, description::String)

Register a problem generator type with the system.
"""
function register_problem(type_sym::Symbol, problem_type::Type{<:ProblemGenerator}, description::String)
    LP_REGISTRY[type_sym] = Dict(
        :type => problem_type,
        :description => description
    )
end

"""
    get_problem_type(problem_sym::Symbol)

Get the problem generator type for a problem symbol.
"""
function get_problem_type(problem_sym::Symbol)
    if !haskey(LP_REGISTRY, problem_sym)
        error("Unknown problem type: $problem_sym")
    end
    return LP_REGISTRY[problem_sym][:type]
end

"""
    build_model(problem::ProblemGenerator)

Build a JuMP model from a problem generator instance.
Each problem type must implement this method.

# Arguments
- `problem`: A problem generator instance containing all necessary data

# Returns
- `model`: The JuMP model
"""
function build_model end

"""
    generate_problem(::Type{T}, target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int) where T <: ProblemGenerator

Generate a linear programming problem by creating a problem generator instance and building the model.

# Arguments
- `T`: Problem generator type
- `target_variables`: Target number of variables in the LP formulation
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility

# Returns
- `model`: The JuMP model
- `problem`: The problem generator instance containing all parameters
"""
function generate_problem(::Type{T}, target_variables::Int, feasibility_status::FeasibilityStatus=unknown, seed::Int=0; relax_integer::Bool=true) where T <: ProblemGenerator
    problem = T(target_variables, feasibility_status, seed)
    model = build_model(problem)

    if relax_integer
        relax_integrality(model)
    end

    return model, problem
end

"""
    generate_problem(problem_sym::Symbol, target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Generate a linear programming problem using a problem type symbol.

# Arguments
- `problem_sym`: Symbol specifying the problem type (e.g., :transportation, :diet_problem)
- `target_variables`: Target number of variables in the LP formulation
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility

# Returns
- `model`: The JuMP model
- `problem`: The problem generator instance containing all parameters
"""
function generate_problem(problem_sym::Symbol, target_variables::Int, feasibility_status::FeasibilityStatus=unknown, seed::Int=0; relax_integer::Bool=true)
    if !haskey(LP_REGISTRY, problem_sym)
        error("Unknown problem type: $problem_sym. Use list_problem_types() to see available types.")
    end

    problem_type = get_problem_type(problem_sym)
    return generate_problem(problem_type, target_variables, feasibility_status, seed; relax_integer=relax_integer)
end

"""
    list_problem_types(; group_variants::Bool=false)

List all available problem types.

# Arguments
- `group_variants`: If true, returns a Dict grouping variants by base type. If false (default), returns flat list.

# Returns
- If `group_variants=false`: Vector of symbols representing available problem types
- If `group_variants=true`: Dict mapping base types to vectors of their variants (and base type itself)

# Examples
```julia
# Flat list
list_problem_types()  # [:scheduling, :scheduling_nurse, :scheduling_or, :blending, ...]

# Grouped by base type
list_problem_types(group_variants=true)
# Dict(:scheduling => [:scheduling, :scheduling_nurse, :scheduling_or],
#      :blending => [:blending, :blending_beverage, :blending_pharmaceutical], ...)
```
"""
function list_problem_types(; group_variants::Bool=false)
    all_types = collect(keys(LP_REGISTRY))

    if !group_variants
        return all_types
    end

    # Group variants by base type
    grouped = Dict{Symbol, Vector{Symbol}}()

    for prob_sym in all_types
        base = get_base_type(prob_sym)

        if !haskey(grouped, base)
            grouped[base] = Symbol[]
        end

        push!(grouped[base], prob_sym)
    end

    # Sort each group
    for (base, variants) in grouped
        sort!(variants)
    end

    return grouped
end

"""
    problem_info(problem_type::Symbol)

Get information about a specific problem type.

# Arguments
- `problem_type`: Symbol specifying the problem type

# Returns
- Dictionary with problem information
"""
function problem_info(problem_type::Symbol)
    if !haskey(LP_REGISTRY, problem_type)
        error("Unknown problem type: $problem_type. Use list_problem_types() to see available types.")
    end
    
    return Dict(
        :type => problem_type,
        :description => LP_REGISTRY[problem_type][:description]
    )
end

"""
    generate_random_problem(target_variables::Int; feasibility_status::FeasibilityStatus=unknown, relax_integer::Bool=true, seed::Int=0)

Generate a random LP problem of a randomly selected type targeting approximately the specified number of variables.

# Arguments
- `target_variables`: Target number of variables in the LP formulation
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `relax_integer`: Whether to relax integer constraints (default: true)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- `model`: The JuMP model
- `problem_sym`: Symbol indicating which problem type was selected
- `problem`: The problem generator instance
"""
function generate_random_problem(target_variables::Int; feasibility_status::FeasibilityStatus=unknown, relax_integer::Bool=true, seed::Int=0)
    Random.seed!(seed)

    problem_types = list_problem_types()
    if isempty(problem_types)
        error("No problem types registered. Include problem type files first.")
    end

    problem_sym = rand(problem_types)
    model, problem = generate_problem(problem_sym, target_variables, feasibility_status, seed; relax_integer=relax_integer)

    return model, problem_sym, problem
end

"""
    is_variant(problem_sym::Symbol)

Check if a problem type symbol represents a variant (as opposed to a base type).

Variants follow the naming convention `:{base}_{variant}` (e.g., `:scheduling_nurse`).

# Arguments
- `problem_sym`: Symbol representing the problem type

# Returns
- `true` if the symbol represents a variant, `false` if it's a base type

# Examples
```julia
is_variant(:scheduling)              # false (base type)
is_variant(:scheduling_nurse)        # true (variant)
is_variant(:blending_beverage)       # true (variant)
```
"""
function is_variant(problem_sym::Symbol)
    sym_str = String(problem_sym)
    return occursin("_", sym_str)
end

"""
    get_base_type(problem_sym::Symbol)

Extract the base type from a problem type symbol.

For base types, returns the symbol itself. For variants, extracts the base type
from the naming convention `:{base}_{variant}`.

# Arguments
- `problem_sym`: Symbol representing the problem type

# Returns
- Base type symbol

# Examples
```julia
get_base_type(:scheduling)              # :scheduling
get_base_type(:scheduling_nurse)        # :scheduling
get_base_type(:blending_beverage)       # :blending
```
"""
function get_base_type(problem_sym::Symbol)
    sym_str = String(problem_sym)

    # Check if this is a variant (contains underscore)
    if !occursin("_", sym_str)
        return problem_sym  # Already a base type
    end

    # Extract base type (everything before first underscore)
    parts = split(sym_str, "_", limit=2)
    return Symbol(parts[1])
end

"""
    list_problem_variants(base_type::Symbol)

List all variants of a specific base problem type.

# Arguments
- `base_type`: Symbol representing the base problem type

# Returns
- Vector of symbols representing all variants of the base type (including base type if registered)

# Examples
```julia
list_problem_variants(:scheduling)
# [:scheduling, :scheduling_nurse, :scheduling_or]

list_problem_variants(:blending)
# [:blending, :blending_beverage, :blending_pharmaceutical]
```
"""
function list_problem_variants(base_type::Symbol)
    all_types = collect(keys(LP_REGISTRY))

    # Find all types that match this base
    base_str = String(base_type)
    variants = Symbol[]

    for prob_sym in all_types
        prob_str = String(prob_sym)

        # Exact match (base type itself)
        if prob_sym == base_type
            push!(variants, prob_sym)
        # Variant match (starts with "base_")
        elseif startswith(prob_str, base_str * "_")
            push!(variants, prob_sym)
        end
    end

    sort!(variants)
    return variants
end

# Include all problem generators
# The problem type files will use the register_problem function from this module
include("problem_types/airline_crew.jl")
include("problem_types/assignment.jl")
include("problem_types/blending.jl")
include("problem_types/crop_planning.jl")
include("problem_types/cutting_stock.jl")
include("problem_types/diet_problem.jl")
include("problem_types/energy.jl")
include("problem_types/facility_location.jl")
include("problem_types/feed_blending.jl")
include("problem_types/inventory.jl")
include("problem_types/knapsack.jl")
include("problem_types/land_use.jl")
include("problem_types/load_balancing.jl")
include("problem_types/multi_commodity_flow.jl")
include("problem_types/network_flow.jl")
include("problem_types/portfolio.jl")
include("problem_types/product_mix.jl")
include("problem_types/production_planning.jl")
include("problem_types/project_selection.jl")
include("problem_types/resource_allocation.jl")
include("problem_types/scheduling.jl")
include("problem_types/supply_chain.jl")
include("problem_types/telecom_network_design.jl")
include("problem_types/transportation.jl")

# Include problem variants
# Variants are organized in subdirectories and follow the naming convention: {base}_{variant}
include("problem_types/scheduling/nurse.jl")
include("problem_types/scheduling/or.jl")
include("problem_types/blending/beverage.jl")
include("problem_types/blending/pharmaceutical.jl")

end # module