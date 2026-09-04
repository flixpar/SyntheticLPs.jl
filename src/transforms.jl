# Model-level reformulations applied to a built JuMP model.
#
# These transforms operate on the finished model produced by `build_model`,
# so they apply uniformly to every category/variant without each generator
# having to implement them. They are the bounds-to-constraints counterpart of
# JuMP's `relax_integrality`, and are wired into `generate_problem` the same way.

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
