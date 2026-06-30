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
