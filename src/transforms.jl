# Model-level reformulations applied to a built JuMP model.
#
# These transforms operate on the finished model produced by `build_model`,
# so they apply uniformly to every category/variant without each generator
# having to implement them: `bounds_to_constraints!` and `dualize_model` (wired
# into `generate_problem` like JuMP's `relax_integrality`), and the practitioner-
# style reformulations configured by `ModelTransforms` (unit scaling, aggregate
# rows, elastic rows, permutation) further down.

"""
    bounds_to_constraints!(model)

Reformulate variable bounds as explicit affine constraints. A plain `x ≥ 0`
nonnegativity lower bound is left as a variable bound; all other bounds
(upper bounds, fixed values, and nonzero lower bounds) become affine rows and
the corresponding variable bound is removed.

In MOI, variable bounds are stored as variable-in-set constraints, which is why
they are excluded by `num_constraints(model; count_variable_in_set_constraints=false)`.
After this transform the converted bounds are genuine affine constraints and so
*are* counted there.

!!! warning "Does not survive presolve"
    Every converted bound is a *singleton row* (one variable, one coefficient),
    and every LP presolver — HiGHS included — turns singleton rows straight back
    into variable bounds. Measured with HiGHS on 15 variants at ~10k variables,
    the presolved model had the same size with or without this transform (to
    within 0.3%) and iteration counts barely moved, so it changes the instance a
    solver actually sees only when presolve is off. Use it to exercise a
    reader/modeling pipeline or a presolve-free solver path, not to make an
    instance harder; [`ModelTransforms`](@ref) lists transforms that do survive.

Returns the (mutated) `model`.
"""
function bounds_to_constraints!(model::Model)
    for x in all_variables(model)
        if is_fixed(x)
            v = fix_value(x)
            unfix(x)
            @constraint(model, x == v)
        else
            if has_lower_bound(x)
                lb = lower_bound(x)
                if lb != 0  # keep standard nonnegativity as a variable bound
                    delete_lower_bound(x)
                    @constraint(model, x >= lb)
                end
            end
            if has_upper_bound(x)
                ub = upper_bound(x)
                delete_upper_bound(x)
                @constraint(model, x <= ub)
            end
        end
    end
    return model
end

"""
    dualize_model(model) -> Model

Return a new JuMP model containing the conic dual of the continuous `model`.
The input model is not modified. Dual variables and constraints are named from
their corresponding primal constraints and variables using the prefixes
`dual_var_` and `dual_con_`.

Integer and binary variables must be relaxed before dualization because a
mixed-integer problem has no LP/conic dual. [`generate_problem`](@ref) does this
automatically with its default `relax_integer=true`; direct callers can use
JuMP's `relax_integrality` first.
"""
function dualize_model(model::Model)
    discrete_variables = [x for x in all_variables(model) if is_binary(x) || is_integer(x)]
    if !isempty(discrete_variables)
        throw(
            ArgumentError(
                "Cannot dualize a model with integer or binary variables. " *
                "Call `relax_integrality(model)` first, or generate the model with " *
                "`relax_integer=true`.",
            ),
        )
    end

    # Dualization does not accept ranged affine rows directly. Split each
    # interval into its equivalent lower and upper inequalities on a copy so
    # the caller's primal remains untouched.
    dualization_input = _split_affine_intervals(model)
    dual = Dualization.dualize(
        dualization_input; dual_names=Dualization.DualNames("dual_var_", "dual_con_")
    )
    dual.ext[:SyntheticLPs_dual_reformulation] = true
    return dual
end

function _split_affine_intervals(model::Model)
    interval_type = MOI.Interval{Float64}
    isempty(all_constraints(model, AffExpr, interval_type)) && return model

    reformulated = copy(model)
    for constraint in all_constraints(reformulated, AffExpr, interval_type)
        object = constraint_object(constraint)
        base_name = name(constraint)
        delete(reformulated, constraint)

        lower = @constraint(reformulated, object.func >= object.set.lower)
        upper = @constraint(reformulated, object.func <= object.set.upper)
        if !isempty(base_name)
            set_name(lower, base_name * "_lower")
            set_name(upper, base_name * "_upper")
        end
    end
    return reformulated
end

"""
    dual_reformulation(model) -> Model

Alias for [`dualize_model`](@ref).
"""
dual_reformulation(model::Model) = dualize_model(model)

"""
    is_dual_reformulation(model) -> Bool

Return whether `model` was produced by [`dualize_model`](@ref), including when
dualization was selected probabilistically by [`generate_random_problem`](@ref).
"""
is_dual_reformulation(model::Model) = get(model.ext, :SyntheticLPs_dual_reformulation, false)::Bool

function _validate_dualize_probability(probability::Real)
    0 <= probability <= 1 ||
        throw(ArgumentError("dualize_probability must be between 0 and 1 (got $probability)."))
    return Float64(probability)
end

function _should_dualize(rng::AbstractRNG, force::Bool, probability::Float64)
    force && return true
    probability == 0.0 && return false
    probability == 1.0 && return true
    return rand(rng) < probability
end

# ---------------------------------------------------------------------------
# Practitioner-style reformulations
# ---------------------------------------------------------------------------
#
# Generators emit textbook formulations: every family in "natural" units, every
# row written once, every constraint hard, columns and rows in index order. Real
# models differ in ways that matter to a simplex code, and the transforms below
# reproduce the ones that (1) practitioners actually do, (2) measurably change
# what an LP solver does, and (3) survive presolve. Each was kept only after
# measuring, with HiGHS 1.13 (presolve on, one thread), presolved sizes and dual/
# primal simplex iteration counts on 15 diverse variants at 10k and 20k variables
# (3k for the two dense ones), against the untransformed model and against a pure
# row/column permutation — the noise floor for "same LP, different presentation"
# (mean |log2| iteration change 0.13 dual / 0.16 primal). Measured and rejected:
#
# - `bounds_to_constraints!`: singleton rows; presolve turns them back into bounds
#   (presolved size identical on all 15 sampled variants).
# - Objective accounting rows (`cost_g == Σ c_j x_j` per variable family with the
#   objective `Σ cost_g`): the defined variables are free column singletons, which
#   presolve substitutes out (presolved size and iteration counts identical on all
#   15).
# - Free-variable splitting (`x = x⁺ - x⁻`): survives presolve (energy/dc_opf at
#   10k: +2,333 presolved columns, +29% dual iterations), but it is a solver-
#   internal standard-form step rather than something modelers write, and it only
#   touches the handful of variants that have free variables.
# - ε-constraints for a secondary objective: a safe ε needs a solve of the source
#   model (otherwise feasibility labels can silently flip), so it does not fit a
#   solver-agnostic post-build transform.

"""
    ModelTransforms(; unit_scale_decades=0, scale_objective=true,
                    aggregate_probability=0.0, aggregate_max_block=8,
                    elastic_probability=0.0, elastic_penalty=1e3,
                    permute=false)

Configuration for the practitioner-style post-build transforms applied by
[`apply_transforms`](@ref) (and, through the `transforms` keyword, by
[`generate_problem`](@ref), [`generate_random_problem`](@ref) and
[`generate_dataset`](@ref)). The default configuration is the identity.

  - `unit_scale_decades`: when positive, [`scale_units!`](@ref) rescales each
    variable family, each row family and (with `scale_objective`) the objective by
    a power of ten drawn from `10^-d … 10^d`. Equivalence-preserving.
  - `aggregate_probability`, `aggregate_max_block`: [`aggregate_rows!`](@ref) adds
    implied "total" rows to each row family with this probability. Equivalence-
    preserving.
  - `elastic_probability`, `elastic_penalty`: [`elasticize_rows!`](@ref) softens
    each row family with this probability using penalized violation columns. A
    *relaxation*: refused for `infeasible` requests (see its docstring).
  - `permute`: [`permute_model`](@ref) shuffles column and row order.
    Equivalence-preserving.

They run in the fixed order aggregate → elastic → permute → scale, each from its
own random stream seeded by the instance seed, so the same seed and configuration
always give the same model.
"""
struct ModelTransforms
    unit_scale_decades::Int
    scale_objective::Bool
    aggregate_probability::Float64
    aggregate_max_block::Int
    elastic_probability::Float64
    elastic_penalty::Float64
    permute::Bool

    function ModelTransforms(
        unit_scale_decades::Integer,
        scale_objective::Bool,
        aggregate_probability::Real,
        aggregate_max_block::Integer,
        elastic_probability::Real,
        elastic_penalty::Real,
        permute::Bool,
    )
        0 <= unit_scale_decades <= 6 ||
            throw(ArgumentError("unit_scale_decades must be in 0:6 (got $unit_scale_decades)."))
        0 <= aggregate_probability <= 1 || throw(
            ArgumentError("aggregate_probability must be in [0, 1] (got $aggregate_probability)."),
        )
        aggregate_max_block >= 2 || throw(
            ArgumentError("aggregate_max_block must be at least 2 (got $aggregate_max_block).")
        )
        0 <= elastic_probability <= 1 || throw(
            ArgumentError("elastic_probability must be in [0, 1] (got $elastic_probability).")
        )
        elastic_penalty > 0 ||
            throw(ArgumentError("elastic_penalty must be positive (got $elastic_penalty)."))
        return new(
            unit_scale_decades,
            scale_objective,
            aggregate_probability,
            aggregate_max_block,
            elastic_probability,
            elastic_penalty,
            permute,
        )
    end
end

function ModelTransforms(;
    unit_scale_decades::Integer=0,
    scale_objective::Bool=true,
    aggregate_probability::Real=0.0,
    aggregate_max_block::Integer=8,
    elastic_probability::Real=0.0,
    elastic_penalty::Real=1e3,
    permute::Bool=false,
)
    return ModelTransforms(
        unit_scale_decades,
        scale_objective,
        aggregate_probability,
        aggregate_max_block,
        elastic_probability,
        elastic_penalty,
        permute,
    )
end

_as_transforms(t::ModelTransforms) = t
_as_transforms(::Nothing) = ModelTransforms()
_as_transforms(t::NamedTuple) = ModelTransforms(; t...)

"""
    is_identity(transforms::ModelTransforms) -> Bool

Whether `transforms` leaves a model unchanged (the default configuration).
"""
function is_identity(t::ModelTransforms)
    return t.unit_scale_decades == 0 &&
           t.aggregate_probability == 0 &&
           t.elastic_probability == 0 &&
           !t.permute
end

# Plain-data record for manifests.
function _transforms_config(t::ModelTransforms)
    return Dict{String, Any}(
        "unit_scale_decades" => t.unit_scale_decades,
        "scale_objective" => t.scale_objective,
        "aggregate_probability" => t.aggregate_probability,
        "aggregate_max_block" => t.aggregate_max_block,
        "elastic_probability" => t.elastic_probability,
        "elastic_penalty" => t.elastic_penalty,
        "permute" => t.permute,
    )
end

# Elastic rows turn an infeasible model feasible whenever the offending rows are
# softened, so an `infeasible` label could not be trusted afterwards.
function _check_transforms_status(t::ModelTransforms, status::FeasibilityStatus)
    if t.elastic_probability > 0 && status == infeasible
        throw(
            ArgumentError(
                "elastic_probability > 0 cannot be combined with an `infeasible` " *
                "request: softening rows can make the model feasible, so the label " *
                "would no longer hold.",
            ),
        )
    end
    return nothing
end

# One independent stream per transform, derived from the instance seed (never the
# global RNG), so toggling one transform does not change another's draws.
function _transform_rng(seed::Integer, stream::Integer)
    s = Int64(seed)
    return MersenneTwister(
        UInt32[s & 0xffffffff, (s >> 32) & 0xffffffff, UInt32(stream), 0x53594e54]
    )
end

"""
    apply_transforms(model, transforms::ModelTransforms, seed) -> Model

Apply the configured practitioner-style transforms to `model` in the order
aggregate → elastic → permute → scale (see [`ModelTransforms`](@ref)) and return
the result. Randomness comes from streams seeded by `seed`, so the call is
deterministic. The input is mutated, except that `permute=true` returns a new
model built from it. The identity configuration returns `model` untouched.
"""
function apply_transforms(model::Model, t::ModelTransforms, seed::Integer)
    is_identity(t) && return model
    if t.aggregate_probability > 0
        aggregate_rows!(
            model,
            _transform_rng(seed, 1);
            probability=t.aggregate_probability,
            max_block=t.aggregate_max_block,
        )
    end
    if t.elastic_probability > 0
        elasticize_rows!(
            model,
            _transform_rng(seed, 2);
            probability=t.elastic_probability,
            penalty=t.elastic_penalty,
        )
    end
    t.permute && (model = permute_model(model, _transform_rng(seed, 3)))
    if t.unit_scale_decades > 0
        scale_units!(
            model,
            _transform_rng(seed, 4);
            decades=t.unit_scale_decades,
            scale_objective=t.scale_objective,
        )
    end
    return model
end

apply_transforms(model::Model, t, seed::Integer) = apply_transforms(model, _as_transforms(t), seed)

# --- shared helpers ---------------------------------------------------------

const _LINEAR_ROW_SETS = (
    MOI.LessThan{Float64}, MOI.GreaterThan{Float64}, MOI.EqualTo{Float64}, MOI.Interval{Float64}
)

function _base_name(s::AbstractString)
    i = findfirst('[', s)
    return i === nothing ? String(s) : String(SubString(s, 1, prevind(s, i)))
end

# A variable family is the JuMP base name (`flow[1,2]` → `flow`): all members of
# one `@variable` container share a physical unit.
_variable_family(x::VariableRef) = (n=name(x); isempty(n) ? "_anonymous" : _base_name(n))

_set_kind(::MOI.LessThan) = "<="
_set_kind(::MOI.GreaterThan) = ">="
_set_kind(::MOI.EqualTo) = "=="
_set_kind(::MOI.Interval) = "in"

# Every linear row of `model`, in a deterministic order. Bounds (variable-in-set
# constraints) are not rows. Anything nonlinear is rejected.
function _linear_rows(model::Model)
    rows = ConstraintRef[]
    for (F, S) in list_of_constraint_types(model)
        F <: AbstractVariableRef && continue
        (F <: GenericAffExpr && S in _LINEAR_ROW_SETS) ||
            throw(ArgumentError("Model transforms support linear models only (found $F-in-$S)."))
        append!(rows, all_constraints(model, F, S))
    end
    return rows
end

# A row family is the base name of a named row. Most generators leave rows
# anonymous, so an anonymous row's family is the signature of its sense and the
# sorted set of variable families it touches: the rows of one algebraic family
# (`Σ_j x[i,j] ≤ s[i]` for all `i`) share it, and rows of different families
# (supply vs demand, balance vs capacity) almost always differ in it.
function _row_family(cref, f::MOI.ScalarAffineFunction, set, var_family)
    n = name(cref)
    isempty(n) || return _set_kind(set) * ":" * _base_name(n)
    families = sort!(unique!([var_family[t.variable] for t in f.terms]))
    return _set_kind(set) * "(" * join(families, ",") * ")"
end

# Index prefix that keeps generated names unique when two families share a base
# name (a named family with rows of two senses splits into two families): empty
# for the first family with that base, `"<ordinal>_"` afterwards.
function _family_tag!(seen::Set{String}, base::AbstractString, ordinal::Integer)
    isempty(base) && return ""
    tag = base in seen ? "$(ordinal)_" : ""
    push!(seen, base)
    return tag
end

# Row families in first-appearance order: `(keys, key => [(cref, f, set), ...])`.
function _row_families(model::Model)
    var_family = Dict(index(x) => _variable_family(x) for x in all_variables(model))
    moi = backend(model)
    families = Dict{String, Vector{Any}}()
    order = String[]
    for cref in _linear_rows(model)
        ci = index(cref)
        f = MOI.get(moi, MOI.ConstraintFunction(), ci)
        set = MOI.get(moi, MOI.ConstraintSet(), ci)
        key = _row_family(cref, f, set, var_family)
        haskey(families, key) || push!(order, key)
        push!(get!(families, key, Any[]), (cref, f, set))
    end
    return order, families, var_family
end

# --- unit scaling -------------------------------------------------------------

"""
    UnitScaling

Record of a [`scale_units!`](@ref) transform, also stored in
`model.ext[:SyntheticLPs_unit_scaling]`.

  - `column_scale[x]`: the scaled variable is `y = column_scale[x] * x_original`.
  - `row_scale[c]`: the scaled row is the original row times `row_scale[c]`.
  - `objective_scale`: the scaled objective is `objective_scale` times the original.
  - `column_exponents`, `row_exponents`: the power of ten applied per family
    (after the magnitude-window adjustment described in [`scale_units!`](@ref)).

An optimal `y*` of the scaled model maps back to the original optimum as
`x* = y* ./ column_scale`, with objective value `objective(y*) / objective_scale`;
a row dual maps back as `dual_original = dual_scaled * row_scale / objective_scale`.
"""
struct UnitScaling
    column_scale::Dict{VariableRef, Float64}
    row_scale::Dict{ConstraintRef, Float64}
    objective_scale::Float64
    column_exponents::Dict{String, Int}
    row_exponents::Dict{String, Int}
end

_scale_set(s::MOI.LessThan, r) = MOI.LessThan(s.upper * r)
_scale_set(s::MOI.GreaterThan, r) = MOI.GreaterThan(s.lower * r)
_scale_set(s::MOI.EqualTo, r) = MOI.EqualTo(s.value * r)
_scale_set(s::MOI.Interval, r) = MOI.Interval(s.lower * r, s.upper * r)

# Coefficients a unit choice may produce: |a| in [1e-6, 1e6]. LP solvers treat
# entries outside a window like this as noise or as infinite (HiGHS drops
# |a| < 1e-9 and fails on costs ~1e12), which would break equivalence.
const _UNIT_SCALE_WINDOW = 6.0

function _widen!(ranges::Dict{String, Tuple{Float64, Float64}}, key, v::Float64)
    lo, hi = get(ranges, key, (v, v))
    ranges[key] = (min(lo, v), max(hi, v))
    return ranges
end

# Shift the drawn exponent `k` the least amount that keeps a family whose log10
# magnitudes span `after` (after column scaling) within the window, widened to the
# family's original span `before` so a generator's own magnitudes are never
# rejected. When no exponent fits (a family spanning more than the window), centre.
function _admissible_exponent(k::Integer, before, after)
    wlo = min(-_UNIT_SCALE_WINDOW, before[1])
    whi = max(_UNIT_SCALE_WINDOW, before[2])
    kmin = ceil(Int, wlo - after[1] - 1e-9)
    kmax = floor(Int, whi - after[2] + 1e-9)
    kmin <= kmax && return clamp(k, kmin, kmax)
    return round(Int, (wlo + whi - after[1] - after[2]) / 2)
end

# One exponent per family, drawn in sorted-key order so the assignment does not
# depend on the order rows and columns happen to appear in (e.g. after `permute`).
function _draw_exponents(rng::AbstractRNG, keys, decades::Integer)
    return Dict(k => rand(rng, (-decades):decades) for k in sort!(collect(Set(keys))))
end

"""
    scale_units!(model, rng; decades=2, scale_objective=true) -> UnitScaling

Re-express `model` in randomly chosen units of measure, the way practitioners'
models come out: a modeler picks tonnes or kg, \$ or \$k, MWh or kWh per family of
quantities, not per entry, so coefficient magnitudes vary by orders of magnitude
*between* families while staying consistent *within* one.

Every variable family (JuMP base name, e.g. all of `flow[...]`) gets a factor
`s = 10^k`, every row family (base name, or for anonymous rows the signature of
sense plus touched variable families) a factor `r = 10^k`, and the objective a
factor `σ = 10^k`, with each `k` drawn uniformly from `-decades:decades`. With
`y_j = s_j x_j`, row `i` becomes `Σ_j (r_i a_ij / s_j) y_j ∈ r_i S_i`, bounds
become `s_j l_j ≤ y_j ≤ s_j u_j` and the objective `σ Σ_j (c_j / s_j) y_j`.
Integer and binary columns keep `s = 1` so integrality keeps its meaning.

Units are chosen so the model stays solvable: after the column factors are
fixed, a row family's (and the objective's) drawn exponent is shifted, only as
far as needed, so its scaled coefficients stay within `[1e-6, 1e6]` — or within
the family's own original range when the generator already exceeded that. Without
this, unlucky draws on families with large natural magnitudes produce costs near
`1e12`, on which HiGHS's dual simplex aborts, or entries below `1e-9`, which
solvers drop as zero and so silently change the problem.

**Equivalence.** Exact up to floating-point rounding of the factors: the scaled
model has the same feasibility status and its optima map one-to-one to the
original's (see [`UnitScaling`](@ref)). Feasibility labels are preserved.

**Effect on a solver.** Presolve does not undo unit choices, and although HiGHS
equilibrates the matrix internally, its power-of-two scaling only partly
compensates: the coefficient, cost and bound ranges, the meaning of absolute
feasibility and optimality tolerances, and pricing all change. Measured with
HiGHS (presolve on) on 15 variants at 20k variables, three draws each with
`decades=2`: presolved row/column counts were unchanged on every instance, the
matrix coefficient range widened by a median 2.5 decades (up to 7), and simplex
iterations changed by a mean |log2| ratio of 0.24 (dual) / 0.33 (primal) against
0.13 / 0.16 for a pure permutation — from ×0.26 (`tsp/flow`, primal, consistently
across draws) to ×2.8 (`process_planning/refinery`, primal) and ×2.9
(`hub_location/p_hub_median`, dual). `decades=3` widens the effect (primal
0.41) at the cost of harsher numerics.

Returns the [`UnitScaling`](@ref) record (also stored in
`model.ext[:SyntheticLPs_unit_scaling]`).
"""
function scale_units!(
    model::Model, rng::AbstractRNG; decades::Integer=2, scale_objective::Bool=true
)
    decades >= 0 || throw(ArgumentError("decades must be nonnegative (got $decades)."))
    vars = all_variables(model)
    rows = _linear_rows(model)
    moi = backend(model)

    var_family = Dict{MOI.VariableIndex, String}(index(x) => _variable_family(x) for x in vars)
    column_exponents = _draw_exponents(rng, values(var_family), decades)

    functions = [MOI.get(moi, MOI.ConstraintFunction(), index(c)) for c in rows]
    sets = [MOI.get(moi, MOI.ConstraintSet(), index(c)) for c in rows]
    row_keys = [_row_family(rows[i], functions[i], sets[i], var_family) for i in eachindex(rows)]
    row_exponents = _draw_exponents(rng, row_keys, decades)
    objective_exponent = scale_objective ? rand(rng, (-decades):decades) : 0

    col = Dict{MOI.VariableIndex, Float64}()
    column_scale = Dict{VariableRef, Float64}()
    for x in vars
        discrete = is_integer(x) || is_binary(x)
        s = discrete ? 1.0 : 10.0^column_exponents[var_family[index(x)]]
        col[index(x)] = s
        column_scale[x] = s
        s == 1.0 && continue
        if is_fixed(x)
            fix(x, fix_value(x) * s)
        else
            has_lower_bound(x) && set_lower_bound(x, lower_bound(x) * s)
            has_upper_bound(x) && set_upper_bound(x, upper_bound(x) * s)
        end
    end

    # Keep each row family inside the magnitude window (see docstring): track
    # the family's log10 range before and after column scaling, then shift its
    # drawn exponent into the admissible interval.
    before = Dict{String, Tuple{Float64, Float64}}()
    after = Dict{String, Tuple{Float64, Float64}}()
    for (i, f) in enumerate(functions)
        key = row_keys[i]
        for t in f.terms
            t.coefficient == 0 && continue
            _widen!(before, key, log10(abs(t.coefficient)))
            _widen!(after, key, log10(abs(t.coefficient) / col[t.variable]))
        end
    end
    for key in collect(keys(row_exponents))
        haskey(after, key) || continue
        row_exponents[key] = _admissible_exponent(row_exponents[key], before[key], after[key])
    end

    row_scale = Dict{ConstraintRef, Float64}()
    for (i, cref) in enumerate(rows)
        r = 10.0^row_exponents[row_keys[i]]
        f = functions[i]
        terms = [
            MOI.ScalarAffineTerm(t.coefficient * r / col[t.variable], t.variable) for t in f.terms
        ]
        ci = index(cref)
        MOI.set(moi, MOI.ConstraintFunction(), ci, MOI.ScalarAffineFunction(terms, f.constant * r))
        MOI.set(moi, MOI.ConstraintSet(), ci, _scale_set(sets[i], r))
        row_scale[cref] = r
    end

    sigma = 1.0
    if objective_sense(model) != MOI.FEASIBILITY_SENSE
        obj = objective_function(model, AffExpr)
        if scale_objective
            spans = Dict{String, Tuple{Float64, Float64}}()
            for (x, c) in obj.terms
                c == 0 && continue
                _widen!(spans, "before", log10(abs(c)))
                _widen!(spans, "after", log10(abs(c) / col[index(x)]))
            end
            if haskey(spans, "after")
                objective_exponent = _admissible_exponent(
                    objective_exponent, spans["before"], spans["after"]
                )
            end
            sigma = 10.0^objective_exponent
        end
        scaled = AffExpr(sigma * obj.constant)
        for (x, c) in obj.terms
            add_to_expression!(scaled, sigma * c / col[index(x)], x)
        end
        set_objective_function(model, scaled)
    end

    scaling = UnitScaling(column_scale, row_scale, sigma, column_exponents, row_exponents)
    model.ext[:SyntheticLPs_unit_scaling] = scaling
    return scaling
end

# --- redundant aggregate rows -------------------------------------------------

function _sum_sets(sets)
    first_set = first(sets)
    first_set isa MOI.LessThan && return MOI.LessThan(sum(s.upper for s in sets))
    first_set isa MOI.GreaterThan && return MOI.GreaterThan(sum(s.lower for s in sets))
    first_set isa MOI.EqualTo && return MOI.EqualTo(sum(s.value for s in sets))
    return MOI.Interval(sum(s.lower for s in sets), sum(s.upper for s in sets))
end

"""
    aggregate_rows!(model, rng; probability=0.5, max_block=8) -> Int

Add the redundant "total" rows defensive modelers write: a regional supply
total next to the per-plant supply rows, a weekly capacity total next to the
daily ones. Each row family (see [`scale_units!`](@ref) for how families are
identified) is selected with `probability`; a selected family is cut, in model
order, into consecutive blocks of `b ∈ 2:max_block` rows (one `b` per family) and
each block gets one new row equal to the exact sum of its rows — `≤` rows sum to
a `≤` row with the summed right-hand side, likewise `≥`, `==` and ranges.
Coefficients that cancel exactly are dropped, and an aggregate with no terms is
skipped. Aggregates of a named family are named `base[total<k>]` (same base, so
they share the family's units under scaling; a second family with the same base,
i.e. the same name used with another sense, gets `base[total<ordinal>_<k>]`);
aggregates of anonymous rows stay anonymous (same signature, same effect).

**Equivalence.** Every aggregate is implied by its block, so the feasible region,
optimal value and feasibility label are unchanged. Because the right-hand side is
the exact sum, an aggregate is tight whenever its block is, which adds primal
degeneracy — the central difficulty for simplex codes — rather than a slack row.

**Presolve survival.** Presolve removes rows that are redundant by variable
bounds, parallel rows and (in HiGHS) linearly dependent equalities, but not
general implied inequalities. Measured with HiGHS on 15 variants at 20k
variables, inequality aggregates survive (`probability=0.5`:
`facility_location/p_median` +10,000 presolved rows, `unit_commitment/standard`
+5,193, `tsp/flow` +5,050) while equality aggregates are mostly removed again
(`inventory/multi_item` at `probability=1`: +1,488 rows before presolve, +6
after). Simplex iterations changed by a mean |log2| ratio of 0.26 (dual) /
0.15 (primal), from ×0.63 (`network_flow/standard`, dual) to ×2.1
(`hub_location/p_hub_median`, dual).

Returns the number of rows added.
"""
function aggregate_rows!(
    model::Model, rng::AbstractRNG; probability::Real=0.5, max_block::Integer=8
)
    0 <= probability <= 1 ||
        throw(ArgumentError("probability must be in [0, 1] (got $probability)."))
    max_block >= 2 || throw(ArgumentError("max_block must be at least 2 (got $max_block)."))
    moi = backend(model)
    order, families, _ = _row_families(model)
    added = 0
    seen = Set{String}()
    for (ordinal, key) in enumerate(order)
        rows = families[key]
        # Draw for every family so the stream does not depend on family sizes.
        selected = rand(rng) < probability
        block = rand(rng, 2:max_block)
        (selected && length(rows) >= 2) || continue
        base = name(first(rows)[1])
        base = isempty(base) ? "" : _base_name(base)
        tag = _family_tag!(seen, base, ordinal)
        for (k, lo) in enumerate(1:block:length(rows))
            members = rows[lo:min(lo + block - 1, end)]
            length(members) >= 2 || continue
            acc = Dict{MOI.VariableIndex, Float64}()
            for (_, f, _) in members, t in f.terms
                acc[t.variable] = get(acc, t.variable, 0.0) + t.coefficient
            end
            largest = maximum(abs, values(acc); init=0.0)
            terms = [MOI.ScalarAffineTerm(c, v) for (v, c) in acc if abs(c) > 1e-12 * largest]
            isempty(terms) && continue
            sort!(terms; by=t -> t.variable.value)
            set = _sum_sets([s for (_, _, s) in members])
            ci = MOI.add_constraint(moi, MOI.ScalarAffineFunction(terms, 0.0), set)
            isempty(base) || MOI.set(moi, MOI.ConstraintName(), ci, "$(base)[total$(tag)$(k)]")
            added += 1
        end
    end
    return added
end

# --- elastic (soft) rows ------------------------------------------------------

"""
    elasticize_rows!(model, rng; probability=0.25, penalty=1e3) -> Int

Soften constraints the way production models do (goal programming / "elastic"
constraints): each row family is selected with `probability`, and every row of a
selected family gets nonnegative violation columns — `a·x - v ≤ b` for `≤`,
`a·x + w ≥ b` for `≥`, both for `==` and ranged rows — charged in the objective
at `penalty × max_j |c_j|` per unit (`penalty` alone when the objective has no
terms), with the sign that makes violation costly under the model's sense. A
feasibility model becomes a minimization of total violation. Violation columns
are named `<base>_over[i]` / `<base>_under[i]` (`elastic<ordinal>_…` for
anonymous families; the index gains an `<ordinal>_` prefix when two families
share a base), so [`scale_units!`](@ref) gives each family's slacks one unit.

**Semantics — a relaxation, not an equivalence.**
  - Feasible stays feasible: the original solution with zero violation remains.
  - The optimal value can only improve, and equals the original whenever the
    penalty exceeds the optimal duals of the softened rows (exact penalty). With
    the default `1e3 × max|c|` the objective was unchanged on every sampled
    variant, but this is not guaranteed; in principle a too-small penalty can even
    make an otherwise bounded model unbounded.
  - Infeasible can become feasible. [`generate_problem`](@ref) therefore refuses
    `elastic_probability > 0` for `infeasible` requests.

**Presolve survival.** A violation column is a column singleton with a positive
cost on an inequality or equality row; HiGHS cannot remove it without dual
bounds it does not have. Measured with HiGHS on 15 variants at 20k variables
with `probability=1`, presolved columns grew by a median 56% (softened singleton
capacity rows also stop collapsing into bounds: `supply_chain/network_planning`
15,879 → 36,172 presolved columns, dual iterations ×1.9, primal ×2.2), simplex
iterations changed by a mean |log2| ratio of 0.28 (dual) / 0.35 (primal), e.g.
`multi_commodity_flow/standard` dual ×0.31, and the optimal objective was
unchanged (relative difference ≤ 1e-10) on every sampled variant.

Selection is per family, so the size effect is lumpy: softening a large family
(e.g. one capacity row per arc) can double the column count, and
[`generate_dataset`](@ref) records and size-matches the transformed size.

Returns the number of violation columns added.
"""
function elasticize_rows!(model::Model, rng::AbstractRNG; probability::Real=0.25, penalty::Real=1e3)
    0 <= probability <= 1 ||
        throw(ArgumentError("probability must be in [0, 1] (got $probability)."))
    penalty > 0 || throw(ArgumentError("penalty must be positive (got $penalty)."))
    order, families, _ = _row_families(model)
    if objective_sense(model) == MOI.FEASIBILITY_SENSE
        set_objective_sense(model, MIN_SENSE)
        set_objective_function(model, AffExpr(0.0))
    end
    obj = objective_function(model, AffExpr)
    largest = maximum(abs, values(obj.terms); init=0.0)
    cost = penalty * (largest > 0 ? largest : 1.0)
    objective_sense(model) == MAX_SENSE && (cost = -cost)

    added = 0
    seen = Set{String}()
    for (ordinal, key) in enumerate(order)
        rand(rng) < probability || continue
        rows = families[key]
        base = name(first(rows)[1])
        base = isempty(base) ? "elastic$(ordinal)" : _base_name(base)
        tag = _family_tag!(seen, base, ordinal)
        for (i, (cref, _, set)) in enumerate(rows)
            if !(set isa MOI.GreaterThan)
                v = @variable(model, lower_bound = 0, base_name = "$(base)_over[$(tag)$i]")
                set_normalized_coefficient(cref, v, -1.0)
                add_to_expression!(obj, cost, v)
                added += 1
            end
            if !(set isa MOI.LessThan)
                w = @variable(model, lower_bound = 0, base_name = "$(base)_under[$(tag)$i]")
                set_normalized_coefficient(cref, w, 1.0)
                add_to_expression!(obj, cost, w)
                added += 1
            end
        end
    end
    set_objective_function(model, obj)
    return added
end

# --- presentation order -------------------------------------------------------

"""
    permute_model(model, rng) -> Model

Return a copy of `model` with its columns and rows in random order (names,
bounds, integrality, constraint sets and the objective are copied unchanged; the
input is not modified, and `model.ext` entries are carried over).

Generators create columns and rows in index order (`x[1,1], x[1,2], …`), so
structure leaks into position: a slack basis, Dantzig/Devex tie-breaking and
"first eligible" rules all behave differently on a lexicographically ordered
model than on one assembled from a database. Permuting removes that artifact —
important when the instances train or tune a pivot rule that could otherwise
learn the generator's index order.

**Equivalence.** Exact: the same LP. It survives presolve trivially, and on its
own it is a *presentation* change: measured with HiGHS on 15 variants at 20k
variables it changed simplex iteration counts by a mean |log2| ratio of 0.13
(dual) / 0.16 (primal) — typically within ±10%, occasionally up to ×2.5
(`hub_location/p_hub_median`, dual) — the noise floor the other transforms were
compared against.
"""
function permute_model(model::Model, rng::AbstractRNG)
    permuted = Model()
    vars = all_variables(model)
    vmap = Dict{VariableRef, VariableRef}()
    for j in randperm(rng, length(vars))
        x = vars[j]
        y = @variable(permuted, base_name = name(x))
        if is_fixed(x)
            fix(y, fix_value(x))
        else
            has_lower_bound(x) && set_lower_bound(y, lower_bound(x))
            has_upper_bound(x) && set_upper_bound(y, upper_bound(x))
        end
        is_binary(x) && set_binary(y)
        is_integer(x) && set_integer(y)
        vmap[x] = y
    end
    rows = _linear_rows(model)
    for i in randperm(rng, length(rows))
        c = constraint_object(rows[i])
        f = AffExpr(c.func.constant)
        for (x, a) in c.func.terms
            add_to_expression!(f, a, vmap[x])
        end
        add_constraint(permuted, ScalarConstraint(f, c.set), name(rows[i]))
    end
    sense = objective_sense(model)
    if sense != MOI.FEASIBILITY_SENSE
        obj = objective_function(model, AffExpr)
        f = AffExpr(obj.constant)
        for (x, a) in obj.terms
            add_to_expression!(f, a, vmap[x])
        end
        set_objective(permuted, sense, f)
    end
    merge!(permuted.ext, model.ext)
    return permuted
end
