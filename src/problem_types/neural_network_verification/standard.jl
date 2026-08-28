using JuMP
using LinearAlgebra
using Random

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
big-M coefficient is the corresponding propagated preactivation bound.

# Fields
- `input_dim`: Number of input variables.
- `hidden_sizes`: Width of each hidden ReLU layer.
- `input_lower`, `input_upper`: Input-box bounds.
- `weights`, `biases`: Hidden-layer affine maps.
- `pre_lower`, `pre_upper`: Propagated preactivation bounds.
- `activation_lower`, `activation_upper`: Propagated ReLU-output bounds.
- `phases`: ReLU phase classification (`-1`, `0`, or `1`).
- `output_weights`, `output_bias`: Scalar affine output layer.
- `output_lower`, `output_upper`: Propagated scalar output bounds.
- `property_threshold`: Right-hand side of the verification property
  `output >= property_threshold`.
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
    output_weights::Vector{Float64}
    output_bias::Float64
    output_lower::Float64
    output_upper::Float64
    property_threshold::Float64
end

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
    NeuralNetworkVerificationProblem(target_variables, feasibility_status, seed)

Construct a deterministic bound-aware ReLU verification instance. The model has
one variable per input and output, two continuous variables per hidden neuron
(preactivation and activation), and one binary per unstable hidden neuron. The
hidden-layer total is selected by a small exhaustive search to keep that count
close to `target_variables`.

For `feasible`, the threshold lies strictly below the output at the center of
the input box, providing a concrete witness. For `infeasible`, it lies strictly
above the propagated output upper bound, which remains a certificate after the
ReLU binaries are relaxed. For `unknown`, it is placed naturally inside the
propagated output interval without asserting a feasibility result.
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

    requested_layers = target_variables < 80 ? 1 :
                       target_variables < 300 ? 2 : 3
    n_layers = min(requested_layers, best_hidden)
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

    for layer in eachindex(hidden_sizes)
        width = hidden_sizes[layer]
        weight_scale = inv(sqrt(previous_width))
        layer_weights = weight_scale .* randn(rng, width, previous_width)

        # First compute the no-bias interval, then select biases by phase. This
        # gives stable neurons an actual bound certificate and unstable neurons
        # finite, two-sided bounds without an arbitrary M constant.
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
    output_lower_vec, output_upper_vec = nnv_affine_bounds(
        reshape(output_weights, 1, :),
        [output_bias],
        previous_lower,
        previous_upper,
    )
    output_lower = output_lower_vec[1]
    output_upper = output_upper_vec[1]

    # Evaluate a concrete input-box witness at its center.
    witness_activation = input_center
    for layer in eachindex(hidden_sizes)
        witness_activation = max.(
            0.0,
            weights[layer] * witness_activation + biases[layer],
        )
    end
    witness_output = dot(output_weights, witness_activation) + output_bias

    output_span = output_upper - output_lower
    strict_margin = max(0.05 * output_span, 1.0e-6)
    property_threshold = if feasibility_status == feasible
        witness_output - strict_margin
    elseif feasibility_status == infeasible
        output_upper + strict_margin
    else
        # A nontrivial, reproducible query inside the interval; unlike the
        # certified branches, no conclusion about exact MIP feasibility is baked in.
        interpolation = 0.35 + 0.30 * rand(rng)
        output_lower + interpolation * output_span
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
        output_weights,
        output_bias,
        output_lower,
        output_upper,
        property_threshold,
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
