using JuMP
using LinearAlgebra
using Random

"""
    ReluNetworkWitness

Planted feasible point of a verification query.

`input` is a concrete point of the input box; `preactivations` and
`activations` are the exact vectors obtained by propagating it through the
network, so the triple satisfies every affine row and every ReLU relation of
the model by construction. `relu_binaries` gives the induced phase of each
*unstable* neuron in the order `build_model` creates its binaries (the
`findall(==(0), phases[layer])` order), which makes the witness a feasible
point of the unrelaxed MILP and not only of its continuous relaxation.
`output` is the resulting scalar network output; the feasible-mode property
threshold sits strictly below it.
"""
struct ReluNetworkWitness
    input::Vector{Float64}
    preactivations::Vector{Vector{Float64}}
    activations::Vector{Vector{Float64}}
    relu_binaries::Vector{Vector{Int8}}
    output::Float64
end

"""
    ReluOutputBoundCertificate

Relaxation-proof certificate that the network cannot reach the property
threshold, obtained by backward linear relaxation of the ReLUs ("CROWN" /
DeepPoly style bound propagation).

Each ReLU is replaced by one linear function of its preactivation,
`a_j <= slope_j * z_j + intercept_j` where the backward coefficient on `a_j` is
nonnegative and `a_j >= slope_j * z_j + intercept_j` where it is negative;
substituting layer by layer collapses the whole network into a single affine
function of the input, `input_constant + input_coefficients' * x`, whose maximum
over the input box is `attainable_upper`.

Two facts make the certificate relaxation-proof:

* Every substituted line is a valid facet of the *triangle* relaxation of its
  ReLU, and the big-M rows of `build_model` project exactly onto that triangle
  once the binary is relaxed to `[0, 1]`. Hence `attainable_upper` bounds the
  LP relaxation optimum, not merely the integer optimum, and the infeasible
  instances are infeasible as LPs as well as MILPs.
* `attainable_upper <= interval_upper - mirrored_gap < declared_upper`, so the
  threshold placed inside `(attainable_upper, declared_upper)` is consistent
  with every individual variable bound and with plain interval propagation over
  the rows. Refuting it requires the coupling between neurons, i.e. actual LP
  work rather than presolve.

# Fields
- `attainable_upper`: sound upper bound on the attainable network output.
- `interval_upper`: the interval-propagation (IBP) output bound.
- `declared_upper`: the output variable's declared upper bound in the model.
- `input_coefficients`, `input_constant`: the collapsed affine function.
- `relaxation_slopes`, `relaxation_intercepts`: the per-neuron substituted line.
- `mirrored_pair`: indices `(u, v)` of the planted opposing neuron pair in the
  last hidden layer (`w_v == -w_u`, `b_v == -b_u`, both output weights positive).
- `mirrored_gap`: `min(c_u * U_u, c_v * U_v)`, the amount by which the pair
  alone makes interval propagation provably loose.
"""
struct ReluOutputBoundCertificate
    attainable_upper::Float64
    interval_upper::Float64
    declared_upper::Float64
    input_coefficients::Vector{Float64}
    input_constant::Float64
    relaxation_slopes::Vector{Vector{Float64}}
    relaxation_intercepts::Vector{Vector{Float64}}
    mirrored_pair::Tuple{Int,Int}
    mirrored_gap::Float64
end

"""
    NeuralNetworkVerificationProblem <: ProblemGenerator

A feed-forward ReLU-network verification query over a box-bounded input. The
model asks whether an input in the box can make the scalar network output at
least `property_threshold`.

The constructor propagates interval bounds through every affine layer before
building the model. Hidden neurons are deliberately split between always
inactive (`phase == -1`), unstable (`phase == 0`), and always active
(`phase == 1`) phases. Consequently, `build_model` fixes stable ReLUs directly
and introduces a binary phase variable only for an unstable neuron. Every
big-M coefficient is the corresponding propagated preactivation bound, so the
formulation is the ideal single-neuron encoding with the tightest constants
interval propagation supports.

# Verification-grade feasibility control

The property threshold is never compared against a declared bound:

* `feasible`: a planted input is propagated through the network and the
  threshold is set strictly below the resulting output, so `feasible_witness`
  is an exactly verifiable solution of the MILP *and* of its relaxation.
* `infeasible`: a backward linear relaxation of the ReLUs yields
  `attainable_upper`, a sound upper bound on what the network can actually
  output over the box, and the threshold is placed strictly *between*
  `attainable_upper` and the (looser) interval bound declared on the output
  variable. Every variable bound taken in isolation, and plain interval
  propagation over the rows, remain consistent with the property; only
  reasoning across the ReLU layers refutes it. A planted opposing neuron pair
  in the last hidden layer guarantees that this gap is nonempty.
* `unknown`: the threshold is interpolated around the planted output without
  asserting either result.

# Fields
- `input_dim`: Number of input variables.
- `hidden_sizes`: Width of each hidden ReLU layer.
- `input_lower`, `input_upper`: Input-box bounds.
- `weights`, `biases`: Hidden-layer affine maps.
- `pre_lower`, `pre_upper`: Propagated preactivation bounds (the big-M constants).
- `activation_lower`, `activation_upper`: Propagated ReLU-output bounds.
- `phases`: ReLU phase classification (`-1`, `0`, or `1`).
- `mirrored_pair`: planted opposing neuron pair in the last hidden layer.
- `output_weights`, `output_bias`: Scalar affine output layer.
- `output_lower`, `output_upper`: Declared scalar output bounds.
- `interval_output_upper`: Raw interval-propagation output bound.
- `attainable_upper`: Sound upper bound on the attainable network output.
- `property_threshold`: Right-hand side of the verification property
  `output >= property_threshold`.
- `feasible_witness`: planted solution (`feasible` requests only).
- `infeasibility_certificate`: bound-propagation certificate (`infeasible` only).
- `feasibility_status`: the requested status.
"""
struct NeuralNetworkVerificationProblem <: ProblemGenerator
    input_dim::Int
    hidden_sizes::Vector{Int}
    input_lower::Vector{Float64}
    input_upper::Vector{Float64}
    weights::Vector{Matrix{Float64}}
    biases::Vector{Vector{Float64}}
    pre_lower::Vector{Vector{Float64}}
    pre_upper::Vector{Vector{Float64}}
    activation_lower::Vector{Vector{Float64}}
    activation_upper::Vector{Vector{Float64}}
    phases::Vector{Vector{Int8}}
    mirrored_pair::Tuple{Int,Int}
    output_weights::Vector{Float64}
    output_bias::Float64
    output_lower::Float64
    output_upper::Float64
    interval_output_upper::Float64
    attainable_upper::Float64
    property_threshold::Float64
    feasible_witness::Union{Nothing,ReluNetworkWitness}
    infeasibility_certificate::Union{Nothing,ReluOutputBoundCertificate}
    feasibility_status::FeasibilityStatus
end

# Minimum hidden width per layer. Six neurons guarantee at least two unstable
# ReLUs per layer under the repeating phase pattern, which is what the planted
# opposing pair needs.
const NNV_MIN_LAYER_WIDTH = 6

# Bounds for `W * x + b` over a box. This is exact for a single affine map.
function nnv_affine_bounds(
    weights::AbstractMatrix{<:Real},
    bias::AbstractVector{<:Real},
    lower::AbstractVector{<:Real},
    upper::AbstractVector{<:Real},
)
    n_outputs, n_inputs = size(weights)
    @assert length(bias) == n_outputs
    @assert length(lower) == n_inputs == length(upper)

    affine_lower = Float64.(bias)
    affine_upper = Float64.(bias)
    for i in 1:n_outputs, j in 1:n_inputs
        coefficient = weights[i, j]
        if coefficient >= 0.0
            affine_lower[i] += coefficient * lower[j]
            affine_upper[i] += coefficient * upper[j]
        else
            affine_lower[i] += coefficient * upper[j]
            affine_upper[i] += coefficient * lower[j]
        end
    end
    return affine_lower, affine_upper
end

# Split a total number of neurons as evenly as possible across hidden layers.
function nnv_hidden_sizes(total_neurons::Int, n_layers::Int)
    base_width, remainder = divrem(total_neurons, n_layers)
    return [base_width + (layer <= remainder ? 1 : 0) for layer in 1:n_layers]
end

# The phase pattern repeats globally, giving ceil(total_neurons / 3) unstable
# neurons and near-equal numbers of the two stable phases.
function nnv_phase_pattern(hidden_sizes::Vector{Int})
    phases = Vector{Vector{Int8}}(undef, length(hidden_sizes))
    neuron_index = 0
    for layer in eachindex(hidden_sizes)
        phases[layer] = Vector{Int8}(undef, hidden_sizes[layer])
        for neuron in 1:hidden_sizes[layer]
            neuron_index += 1
            phases[layer][neuron] = if mod(neuron_index - 1, 3) == 0
                Int8(0)   # unstable
            elseif mod(neuron_index - 1, 3) == 1
                Int8(1)   # always active
            else
                Int8(-1)  # always inactive
            end
        end
    end
    return phases
end

"""
    nnv_forward(weights, biases, input) -> (preactivations, activations)

Exact forward propagation of a concrete input through the ReLU network.
"""
function nnv_forward(
    weights::Vector{Matrix{Float64}},
    biases::Vector{Vector{Float64}},
    input::AbstractVector{<:Real},
)
    n_layers = length(weights)
    preactivations = Vector{Vector{Float64}}(undef, n_layers)
    activations = Vector{Vector{Float64}}(undef, n_layers)
    current = collect(float.(input))
    for layer in 1:n_layers
        preactivations[layer] = weights[layer] * current .+ biases[layer]
        activations[layer] = max.(0.0, preactivations[layer])
        current = activations[layer]
    end
    return preactivations, activations
end

"""
    nnv_backward_bound(...; lower_mode) -> NamedTuple

Backward linear-relaxation ("CROWN" / DeepPoly) upper bound on the network
output over the input box.

Walking backwards from the output layer, the coefficient vector `lambda` on a
layer's activations is pushed through the ReLUs by substituting one linear
function per neuron: the tightest triangle upper facet
`a <= U/(U-L) * (z - L)` where `lambda_j >= 0`, and a valid lower relaxation
(`a >= 0` or `a >= z`) where `lambda_j < 0`. Stable neurons substitute their
exact linear phase. The result collapses to an affine function of the input,
maximised exactly over the box.

`lower_mode == :zero` always picks `a >= 0` for negative coefficients. Under
that choice every substituted line, maximised on its own over `[L, U]`,
reproduces exactly the interval-propagation contribution of its neuron, so the
returned bound is provably no worse than interval propagation while capturing
the coupling between neurons. `:adaptive` uses the usual CROWN heuristic
(`a >= z` when `U >= -L`), which is often tighter; the caller takes the better
of the two.
"""
function nnv_backward_bound(
    weights::Vector{Matrix{Float64}},
    biases::Vector{Vector{Float64}},
    pre_lower::Vector{Vector{Float64}},
    pre_upper::Vector{Vector{Float64}},
    output_weights::Vector{Float64},
    output_bias::Float64,
    input_lower::Vector{Float64},
    input_upper::Vector{Float64};
    lower_mode::Symbol,
)
    n_layers = length(weights)
    slopes = [zeros(length(biases[layer])) for layer in 1:n_layers]
    intercepts = [zeros(length(biases[layer])) for layer in 1:n_layers]

    lambda = copy(output_weights)
    constant = output_bias
    for layer in n_layers:-1:1
        mu = zeros(length(lambda))
        for j in eachindex(lambda)
            lower = pre_lower[layer][j]
            upper = pre_upper[layer][j]
            slope, intercept = if lower >= 0.0
                (1.0, 0.0)                      # stable active: a == z
            elseif upper <= 0.0
                (0.0, 0.0)                      # stable inactive: a == 0
            elseif lambda[j] >= 0.0
                scale = upper / (upper - lower) # tightest triangle upper facet
                (scale, -scale * lower)
            elseif lower_mode === :adaptive && upper >= -lower
                (1.0, 0.0)                      # a >= z
            else
                (0.0, 0.0)                      # a >= 0
            end
            slopes[layer][j] = slope
            intercepts[layer][j] = intercept
            mu[j] = lambda[j] * slope
            constant += lambda[j] * intercept
        end
        constant += dot(mu, biases[layer])
        lambda = transpose(weights[layer]) * mu
    end

    bound = constant
    for i in eachindex(lambda)
        bound += lambda[i] >= 0.0 ? lambda[i] * input_upper[i] :
                 lambda[i] * input_lower[i]
    end
    return (
        bound = bound,
        input_coefficients = collect(lambda),
        input_constant = constant,
        slopes = slopes,
        intercepts = intercepts,
    )
end

"""
    NeuralNetworkVerificationProblem(target_variables, feasibility_status, seed)

Construct a deterministic bound-aware ReLU verification instance. The model has
one variable per input and output, two continuous variables per hidden neuron
(preactivation and activation), and one binary per unstable hidden neuron. The
hidden-layer total is selected by a small exhaustive search to keep that count
close to `target_variables`.

See the type docstring for how each feasibility status is realised; in
particular, an `infeasible` instance is refutable only by propagating through
the ReLU layers, never by comparing the property against a declared bound.
"""
function NeuralNetworkVerificationProblem(
    target_variables::Int,
    feasibility_status::FeasibilityStatus,
    seed::Int,
)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be positive (got $target_variables)"))

    rng = MersenneTwister(seed)

    # Keep the input meaningful without letting it dominate the formulation.
    input_dim = clamp(round(Int, sqrt(target_variables) / 1.5), 2, 24)

    # Count = input + scalar output + 2 * hidden + unstable hidden.
    # The phase pattern makes the last term ceil(hidden / 3).
    best_hidden = 1
    best_error = typemax(Int)
    for hidden in 1:max(1, target_variables)
        variable_count = input_dim + 1 + 2 * hidden + cld(hidden, 3)
        error = abs(variable_count - target_variables)
        if error < best_error
            best_hidden = hidden
            best_error = error
        end
        variable_count > target_variables + best_error && break
    end
    # Every layer needs at least two unstable neurons for the planted opposing
    # pair, hence the minimum width; tiny requests are rounded up to it.
    best_hidden = max(best_hidden, NNV_MIN_LAYER_WIDTH)

    requested_layers = target_variables < 80 ? 1 :
                       target_variables < 300 ? 2 : 3
    n_layers = min(requested_layers, max(1, best_hidden ÷ NNV_MIN_LAYER_WIDTH))
    hidden_sizes = nnv_hidden_sizes(best_hidden, n_layers)
    phases = nnv_phase_pattern(hidden_sizes)

    input_center = [rand(rng) * 2.0 - 1.0 for _ in 1:input_dim]
    input_radius = [0.5 + rand(rng) for _ in 1:input_dim]
    input_lower = input_center .- input_radius
    input_upper = input_center .+ input_radius

    weights = Matrix{Float64}[]
    biases = Vector{Float64}[]
    pre_lower = Vector{Float64}[]
    pre_upper = Vector{Float64}[]
    activation_lower = Vector{Float64}[]
    activation_upper = Vector{Float64}[]

    previous_lower = input_lower
    previous_upper = input_upper
    previous_width = input_dim

    # The two unstable neurons of the last hidden layer that carry the planted
    # opposing pair `w_v = -w_u`, `b_v = -b_u`. Because at any point at most one
    # of `relu(z)` and `relu(-z)` is positive while interval propagation adds
    # both maxima, this pair makes interval propagation provably loose - which
    # is exactly the room the infeasible threshold is placed in.
    last_unstable = findall(==(Int8(0)), phases[end])
    mirrored_pair = (last_unstable[1], last_unstable[2])

    for layer in eachindex(hidden_sizes)
        width = hidden_sizes[layer]
        weight_scale = inv(sqrt(previous_width))
        layer_weights = weight_scale .* randn(rng, width, previous_width)
        if layer == length(hidden_sizes)
            u, v = mirrored_pair
            layer_weights[v, :] .= .-layer_weights[u, :]
        end

        # First compute the no-bias interval, then select biases by phase. This
        # gives stable neurons an actual bound certificate and unstable neurons
        # finite, two-sided bounds without an arbitrary M constant. The bias
        # rule is odd-symmetric on unstable neurons, so the mirrored row also
        # receives the negated bias and satisfies `z_v == -z_u` exactly.
        raw_lower, raw_upper = nnv_affine_bounds(
            layer_weights,
            zeros(width),
            previous_lower,
            previous_upper,
        )
        layer_bias = zeros(width)
        for neuron in 1:width
            span = raw_upper[neuron] - raw_lower[neuron]
            margin = max(0.1 * span, 1.0e-6)
            phase = phases[layer][neuron]
            if phase == 0
                layer_bias[neuron] = -0.5 * (raw_lower[neuron] + raw_upper[neuron])
            elseif phase == 1
                layer_bias[neuron] = -raw_lower[neuron] + margin
            else
                layer_bias[neuron] = -raw_upper[neuron] - margin
            end
        end

        layer_pre_lower, layer_pre_upper = nnv_affine_bounds(
            layer_weights,
            layer_bias,
            previous_lower,
            previous_upper,
        )
        layer_activation_lower = max.(0.0, layer_pre_lower)
        layer_activation_upper = max.(0.0, layer_pre_upper)

        push!(weights, layer_weights)
        push!(biases, layer_bias)
        push!(pre_lower, layer_pre_lower)
        push!(pre_upper, layer_pre_upper)
        push!(activation_lower, layer_activation_lower)
        push!(activation_upper, layer_activation_upper)

        previous_lower = layer_activation_lower
        previous_upper = layer_activation_upper
        previous_width = width
    end

    output_weights = inv(sqrt(previous_width)) .* randn(rng, previous_width)
    output_bias = rand(rng) - 0.5
    # Both halves of the opposing pair must be read with a positive weight for
    # the interval bound to double-count them.
    for index in mirrored_pair
        output_weights[index] =
            max(abs(output_weights[index]), 0.25 * inv(sqrt(previous_width)))
    end

    output_lower_vec, output_upper_vec = nnv_affine_bounds(
        reshape(output_weights, 1, :),
        [output_bias],
        previous_lower,
        previous_upper,
    )
    output_lower = output_lower_vec[1]
    interval_output_upper = output_upper_vec[1]

    # The pair alone makes interval propagation loose by this much.
    mirrored_gap = minimum(
        output_weights[index] * pre_upper[end][index] for index in mirrored_pair
    )

    # Sound upper bound on what the network can actually output, and therefore
    # also on the optimum of the LP relaxation of the big-M encoding.
    relaxation_zero = nnv_backward_bound(
        weights, biases, pre_lower, pre_upper, output_weights, output_bias,
        input_lower, input_upper; lower_mode = :zero,
    )
    relaxation_adaptive = nnv_backward_bound(
        weights, biases, pre_lower, pre_upper, output_weights, output_bias,
        input_lower, input_upper; lower_mode = :adaptive,
    )
    relaxation = relaxation_adaptive.bound < relaxation_zero.bound ?
                 relaxation_adaptive : relaxation_zero
    attainable_upper = relaxation.bound

    # Deterministic search for a high-output point of the input box: the box
    # centre, the vertices suggested by the two backward relaxations, and a
    # sampled mix of vertices and interior points.
    candidates = Vector{Vector{Float64}}()
    push!(candidates, 0.5 .* (input_lower .+ input_upper))
    for coefficients in
        (relaxation_zero.input_coefficients, relaxation_adaptive.input_coefficients)
        push!(candidates, [
            coefficients[i] >= 0.0 ? input_upper[i] : input_lower[i]
            for i in 1:input_dim
        ])
    end
    for _ in 1:8
        push!(candidates, [
            rand(rng) < 0.5 ? input_lower[i] : input_upper[i] for i in 1:input_dim
        ])
    end
    for _ in 1:8
        push!(candidates, input_lower .+ rand(rng, input_dim) .* (input_upper .- input_lower))
    end

    witness_input = candidates[1]
    witness_pre, witness_act = nnv_forward(weights, biases, witness_input)
    witness_output = dot(output_weights, witness_act[end]) + output_bias
    for candidate in candidates[2:end]
        candidate_pre, candidate_act = nnv_forward(weights, biases, candidate)
        candidate_output = dot(output_weights, candidate_act[end]) + output_bias
        if candidate_output > witness_output
            witness_input = candidate
            witness_pre = candidate_pre
            witness_act = candidate_act
            witness_output = candidate_output
        end
    end

    # The declared output bound stays the (loose) interval bound; the guard only
    # matters if backward propagation failed to improve on it at all.
    epsilon = max(1.0e-9, 1.0e-9 * abs(interval_output_upper))
    output_upper = max(interval_output_upper, attainable_upper + 2.0 * epsilon)
    output_span = output_upper - output_lower
    strict_margin = max(0.05 * output_span, 1.0e-6)

    property_threshold = if feasibility_status == feasible
        # Strictly below the planted output, and strictly above the declared
        # output lower bound so the property row is never implied by a bound.
        witness_output - min(strict_margin, 0.5 * (witness_output - output_lower))
    elseif feasibility_status == infeasible
        # Strictly above what the network can attain, strictly below the
        # declared bound: no single bound and no interval propagation over the
        # rows settles the query.
        attainable_upper + 0.15 * (output_upper - attainable_upper)
    else
        # A nontrivial, reproducible query around the planted output; unlike the
        # certified branches, no conclusion about feasibility is baked in.
        scale = max(attainable_upper - witness_output, 0.02 * output_span)
        clamp(
            witness_output + (-0.25 + 1.35 * rand(rng)) * scale,
            output_lower + 0.02 * output_span,
            output_upper - 0.02 * output_span,
        )
    end

    witness = if feasibility_status == feasible
        relu_binaries = [
            Int8[witness_pre[layer][neuron] > 0.0 ? Int8(1) : Int8(0)
                 for neuron in findall(==(Int8(0)), phases[layer])]
            for layer in eachindex(hidden_sizes)
        ]
        ReluNetworkWitness(
            witness_input, witness_pre, witness_act, relu_binaries, witness_output,
        )
    else
        nothing
    end

    certificate = if feasibility_status == infeasible
        ReluOutputBoundCertificate(
            attainable_upper,
            interval_output_upper,
            output_upper,
            relaxation.input_coefficients,
            relaxation.input_constant,
            relaxation.slopes,
            relaxation.intercepts,
            mirrored_pair,
            mirrored_gap,
        )
    else
        nothing
    end

    return NeuralNetworkVerificationProblem(
        input_dim,
        hidden_sizes,
        input_lower,
        input_upper,
        weights,
        biases,
        pre_lower,
        pre_upper,
        activation_lower,
        activation_upper,
        phases,
        mirrored_pair,
        output_weights,
        output_bias,
        output_lower,
        output_upper,
        interval_output_upper,
        attainable_upper,
        property_threshold,
        witness,
        certificate,
        feasibility_status,
    )
end

"""
    build_model(prob::NeuralNetworkVerificationProblem)

Build the ReLU verification MILP. Stable neurons use their exact linear phase;
an unstable neuron with bounds `L < 0 < U` uses the ideal single-neuron big-M
formulation

```text
a >= z,  a >= 0,  a <= U*d,  a <= z - L*(1-d),  d binary.
```

The big-M constants `U` and `L` are the propagated preactivation bounds of that
neuron, which is the tightest pair interval propagation supports; relaxing `d`
to `[0, 1]` projects these rows exactly onto the triangle relaxation of the
ReLU.

All inputs to model construction are stored in `prob`, so repeated calls are
deterministic.
"""
function build_model(prob::NeuralNetworkVerificationProblem)
    model = Model()

    input = @variable(
        model,
        [i = 1:prob.input_dim],
        lower_bound = prob.input_lower[i],
        upper_bound = prob.input_upper[i],
        base_name = "input",
    )

    n_layers = length(prob.hidden_sizes)
    preactivation = Vector{Any}(undef, n_layers)
    activation = Vector{Any}(undef, n_layers)
    phase_binary = Vector{Any}(undef, n_layers)

    previous_activation = input
    for layer in 1:n_layers
        width = prob.hidden_sizes[layer]
        preactivation[layer] = @variable(
            model,
            [i = 1:width],
            lower_bound = prob.pre_lower[layer][i],
            upper_bound = prob.pre_upper[layer][i],
            base_name = "preactivation_$layer",
        )
        activation[layer] = @variable(
            model,
            [i = 1:width],
            lower_bound = prob.activation_lower[layer][i],
            upper_bound = prob.activation_upper[layer][i],
            base_name = "activation_$layer",
        )

        previous_width = layer == 1 ? prob.input_dim : prob.hidden_sizes[layer - 1]
        for neuron in 1:width
            @constraint(
                model,
                preactivation[layer][neuron] ==
                    sum(
                        prob.weights[layer][neuron, j] * previous_activation[j]
                        for j in 1:previous_width
                    ) + prob.biases[layer][neuron],
            )
        end

        unstable_neurons = findall(==(Int8(0)), prob.phases[layer])
        phase_binary[layer] = @variable(
            model,
            [k = 1:length(unstable_neurons)],
            Bin,
            base_name = "relu_phase_$layer",
        )

        unstable_index = 0
        for neuron in 1:width
            phase = prob.phases[layer][neuron]
            if phase == -1
                @constraint(model, activation[layer][neuron] == 0.0)
            elseif phase == 1
                @constraint(model, activation[layer][neuron] == preactivation[layer][neuron])
            else
                unstable_index += 1
                lower = prob.pre_lower[layer][neuron]
                upper = prob.pre_upper[layer][neuron]
                binary = phase_binary[layer][unstable_index]
                @constraint(model, activation[layer][neuron] >= preactivation[layer][neuron])
                @constraint(model, activation[layer][neuron] >= 0.0)
                @constraint(model, activation[layer][neuron] <= upper * binary)
                @constraint(
                    model,
                    activation[layer][neuron] <=
                        preactivation[layer][neuron] - lower * (1.0 - binary),
                )
            end
        end
        previous_activation = activation[layer]
    end

    output = @variable(
        model,
        lower_bound = prob.output_lower,
        upper_bound = prob.output_upper,
        base_name = "output",
    )
    @constraint(
        model,
        output == sum(
            prob.output_weights[j] * previous_activation[j]
            for j in eachindex(prob.output_weights)
        ) + prob.output_bias,
    )

    # The verification query: does any input in the box violate the property by
    # attaining an output at least this threshold?
    @constraint(model, output >= prob.property_threshold)
    @objective(model, Max, output)

    return model
end

register_variant(
    :neural_network_verification,
    :relu_big_m,
    NeuralNetworkVerificationProblem,
    "Bound-aware ReLU verification with stable-phase elimination and propagated big-M coefficients";
    default = true,
)
