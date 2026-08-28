using JuMP
using Random

"""One sparse affine row in a [`GenericMILPProblem`](@ref)."""
struct GenericMILPRow
    indices::Vector{Int}
    coefficients::Vector{Float64}
    sense::Symbol
    lower::Float64
    upper::Float64
end

"""
    GenericMILPProblem <: ProblemGenerator

A transparent distributional MILP with binary and bounded general-integer
variables plus continuous columns with two-sided, one-sided, fixed, or free
bounds. Sparse packing, covering, equality, and ranged rows are generated around
a planted primal witness. Infeasible instances add a bound contradiction, so
infeasibility survives integrality relaxation.

This generator is intended to cover generic matrix/domain structure. It does
not claim that its `difficulty` is calibrated to benchmark solve time.
"""
struct GenericMILPProblem <: ProblemGenerator
    n_variables::Int
    variable_domains::Vector{Symbol}
    lower_bounds::Vector{Float64}
    upper_bounds::Vector{Float64}
    objective::Vector{Float64}
    objective_sense::Symbol
    rows::Vector{GenericMILPRow}
    planted_solution::Vector{Float64}
end

function _generic_variable_layout(rng::AbstractRNG, n::Int)
    order = randperm(rng, n)
    n_binary = max(1, round(Int, 0.55 * n))
    n_integer = n >= 5 ? max(1, round(Int, 0.20 * n)) : 0
    n_binary + n_integer > n && (n_integer = n - n_binary)

    domains = fill(:continuous, n)
    domains[order[1:n_binary]] .= :binary
    if n_integer > 0
        domains[order[(n_binary + 1):(n_binary + n_integer)]] .= :integer
    end

    lower = zeros(Float64, n)
    upper = ones(Float64, n)
    witness = zeros(Float64, n)
    for i in 1:n
        if domains[i] == :binary
            witness[i] = rand(rng, Bool) ? 1.0 : 0.0
        elseif domains[i] == :integer
            upper[i] = float(rand(rng, 3:12))
            witness[i] = float(rand(rng, 0:Int(upper[i])))
        else
            bound_regime = rand(rng)
            if bound_regime < 0.45
                # Finite two-sided columns, sometimes crossing zero.
                if rand(rng) < 0.35
                    lower[i] = -rand(rng, 1.0:1.0:5.0)
                end
                upper[i] = rand(rng, 4.0:1.0:15.0)
                witness[i] = lower[i] + rand(rng) * (upper[i] - lower[i])
            elseif bound_regime < 0.65
                # Lower-bounded, no column upper bound.
                lower[i] = rand(rng, Bool) ? 0.0 : -rand(rng, 1.0:1.0:4.0)
                upper[i] = Inf
                witness[i] = lower[i] + rand(rng, 0.5:0.5:6.0)
            elseif bound_regime < 0.80
                # Upper-bounded, no column lower bound.
                lower[i] = -Inf
                upper[i] = rand(rng, 2.0:1.0:10.0)
                witness[i] = upper[i] - rand(rng, 0.5:0.5:6.0)
            elseif bound_regime < 0.90
                # Fixed columns occur in presolved and inherited benchmark data.
                fixed = rand(rng, -4.0:1.0:8.0)
                lower[i] = fixed
                upper[i] = fixed
                witness[i] = fixed
            else
                # Truly free columns. Their objective coefficient is set to zero
                # below so planted feasible models remain bounded.
                lower[i] = -Inf
                upper[i] = Inf
                witness[i] = rand(rng, -4.0:0.5:4.0)
            end
        end
    end
    return domains, lower, upper, witness
end

function _generic_sparse_support(rng::AbstractRNG, n::Int)
    # Square-root growth keeps rows sparse at large n without making small
    # instances singletons.
    width = clamp(round(Int, 2 + sqrt(n) * rand(rng, 0.5:0.1:1.2)), 2, n)
    return sort!(randperm(rng, n)[1:width])
end

function _generic_coefficients(rng::AbstractRNG, width::Int; nonnegative::Bool=false)
    coefficients = Vector{Float64}(undef, width)
    for j in 1:width
        regime = rand(rng)
        magnitude = regime < 0.75 ? float(rand(rng, 1:20)) :
                    regime < 0.93 ? rand(rng, 0.1:0.1:9.9) :
                    10.0 ^ rand(rng, 2:4)
        coefficients[j] = (nonnegative || rand(rng, Bool) ? 1.0 : -1.0) * magnitude
    end
    return coefficients
end

function _generic_row(rng::AbstractRNG, n::Int, witness::Vector{Float64}, sense::Symbol)
    indices = _generic_sparse_support(rng, n)
    # <= and >= rows are recognizable packing and covering structures; equality
    # and ranged rows retain signed heterogeneous coefficients.
    coefficients = _generic_coefficients(rng, length(indices);
                                         nonnegative = sense in (:le, :ge))
    activity = sum(coefficients[j] * witness[indices[j]] for j in eachindex(indices))
    margin = max(1.0, 0.05 * abs(activity) + rand(rng, 0.5:0.5:5.0))
    if sense == :le
        return GenericMILPRow(indices, coefficients, :le, -Inf, activity + margin)
    elseif sense == :ge
        return GenericMILPRow(indices, coefficients, :ge, activity - margin, Inf)
    elseif sense == :eq
        return GenericMILPRow(indices, coefficients, :eq, activity, activity)
    elseif sense == :range
        return GenericMILPRow(indices, coefficients, :range,
                              activity - margin, activity + margin)
    end
    error("Unsupported generic MILP row sense: $sense")
end

function GenericMILPProblem(target_variables::Int,
                            feasibility_status::FeasibilityStatus,
                            seed::Int)
    rng = MersenneTwister(seed)
    n = max(2, target_variables)
    domains, lower, upper, witness = _generic_variable_layout(rng, n)

    n_rows = max(5, round(Int, 0.30 * n))
    row_senses = (:le, :ge, :eq, :range)
    rows = GenericMILPRow[]
    for row_index in 1:n_rows
        # Guarantee that every ordinary instance contains each structural row
        # family before sampling the remaining senses.
        sense = row_index <= length(row_senses) ? row_senses[row_index] : rand(rng, row_senses)
        push!(rows, _generic_row(rng, n, witness, sense))
    end

    actual_status = feasibility_status == unknown ?
                    (rand(rng) < 0.8 ? feasible : infeasible) : feasibility_status
    if actual_status == infeasible
        # Choose a finite lower bound and contradict it with a singleton row.
        # Binary and integer bounds persist under relaxation, so the certificate
        # remains valid even when discrete domains are relaxed.
        finite_lower = findall(isfinite, lower)
        i = rand(rng, finite_lower)
        push!(rows, GenericMILPRow([i], [1.0], :le, -Inf, lower[i] - 1.0))
    end

    objective_sense = rand(rng, Bool) ? :min : :max
    objective = Vector{Float64}(undef, n)
    for i in 1:n
        magnitude = float(rand(rng, 1:50))
        if !isfinite(lower[i]) && !isfinite(upper[i])
            objective[i] = 0.0
        elseif !isfinite(upper[i])
            objective[i] = objective_sense == :min ? magnitude : -magnitude
        elseif !isfinite(lower[i])
            objective[i] = objective_sense == :min ? -magnitude : magnitude
        else
            objective[i] = (rand(rng, Bool) ? 1.0 : -1.0) * magnitude
        end
    end

    return GenericMILPProblem(n, domains, lower, upper, objective,
                              objective_sense, rows, witness)
end

function build_model(prob::GenericMILPProblem)
    model = Model()
    x = Vector{VariableRef}(undef, prob.n_variables)
    for i in 1:prob.n_variables
        x[i] = @variable(model, base_name = "x_$i")
        if prob.variable_domains[i] == :binary
            set_binary(x[i])
        elseif prob.variable_domains[i] == :integer
            set_integer(x[i])
            set_lower_bound(x[i], prob.lower_bounds[i])
            set_upper_bound(x[i], prob.upper_bounds[i])
        else
            isfinite(prob.lower_bounds[i]) && set_lower_bound(x[i], prob.lower_bounds[i])
            isfinite(prob.upper_bounds[i]) && set_upper_bound(x[i], prob.upper_bounds[i])
        end
    end

    objective = sum(prob.objective[i] * x[i] for i in 1:prob.n_variables)
    if prob.objective_sense == :min
        @objective(model, Min, objective)
    else
        @objective(model, Max, objective)
    end

    for row in prob.rows
        expression = sum(row.coefficients[j] * x[row.indices[j]]
                         for j in eachindex(row.indices))
        if row.sense == :le
            @constraint(model, expression <= row.upper)
        elseif row.sense == :ge
            @constraint(model, expression >= row.lower)
        elseif row.sense == :eq
            @constraint(model, expression == row.lower)
        elseif row.sense == :range
            @constraint(model, row.lower <= expression <= row.upper)
        else
            error("Unsupported generic MILP row sense: $(row.sense)")
        end
    end
    return model
end

register_variant(
    :generic_milp,
    :standard,
    GenericMILPProblem,
    "Sparse mixed-integer model with controlled variable domains, row senses, coefficient scales, and a planted primal witness",
    default = true,
)
