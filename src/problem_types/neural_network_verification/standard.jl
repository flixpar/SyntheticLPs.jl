using JuMP
using LinearAlgebra
using Random
using SparseArrays

"""
    ReluNetworkWitness

Planted feasible point of a verification query.

`input` is a concrete point of the input box; `preactivations` and
`activations` are the exact vectors obtained by propagating it through the
network (every neuron, including the stably inactive ones that `build_model`
omits), so the triple satisfies every affine row and every ReLU relation of the
model by construction. `relu_binaries` gives the induced phase of each
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

  - Every substituted line is a valid facet of the *triangle* relaxation of its
    ReLU, and the big-M rows of `build_model` project exactly onto that triangle
    once the binary is relaxed to `[0, 1]`. Hence `attainable_upper` bounds the
    LP relaxation optimum, not merely the integer optimum, and the infeasible
    instances are infeasible as LPs as well as MILPs.
  - `attainable_upper <= interval_upper - mirrored_gap < declared_upper`, so the
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
    mirrored_pair::Tuple{Int, Int}
    mirrored_gap::Float64
end

"""
    NeuralNetworkVerificationProblem <: ProblemGenerator

A ReLU-network verification query over a box-bounded input: can an input in the
box make the scalar network output at least `property_threshold`?

# Architectures

Three network families keep the constraint matrix realistic *and* bounded as
the request grows (the old all-dense MLP had `O(width²)` nonzeros, 4.1M at 10k
variables and hundreds of millions at 100k):

  - `:dense_mlp` (targets below 2,500): a small fully connected network over a
    low-dimensional input box, in the style of ACAS Xu controllers.
  - `:pruned_mlp`: a magnitude-pruned fully connected network over an
    image-like input (pixel box `[x0 - ε, x0 + ε] ∩ [0, 1]`); every neuron keeps a
    fixed fan-in of 24–64 surviving weights.
  - `:convolutional`: 3×3, stride-1, zero-padded convolution layers with
    shared kernels over a `channels × height × width` image, followed by a
    dense fully connected head (fan-in = flattened feature map).

Targets of 2,500 variables and above pick `:pruned_mlp` or `:convolutional`
from the seed. Nonzeros grow linearly in the request (≈ 17–23 per variable).

# Formulation

Interval bounds are propagated through every affine layer before the model is
built. Biases are selected per neuron (untied biases for the convolution
layers) so that each hidden neuron has a planted phase: stably inactive
(`-1`), unstable (`0`), or stably active (`1`). Mirroring what MIP-based
verifiers do, `build_model` emits

  - nothing for a stably inactive neuron (its activation is identically zero,
    so it also disappears from the next layer's rows);
  - a single activation column `a` with its defining row `a = w'x + b` for a
    stably active neuron;
  - a preactivation `z`, an activation `a`, a phase binary `d`, the defining
    row `z = w'x + b`, and the three ideal big-M rows
    `a >= z`, `a <= U d`, `a <= z - L (1 - d)` for an unstable neuron (`a >= 0`
    is the activation's lower bound).

Roughly 35–55% of hidden neurons are unstable, 15–30% are inactive, and the
rest are active, so the relaxed LP is dominated by coupled triangle relaxations
rather than by rows presolve can substitute away.

# Verification-grade feasibility control

  - `feasible`: a planted input is propagated through the network and the
    threshold is set strictly below the resulting output, so `feasible_witness`
    is an exactly verifiable solution of the MILP *and* of its relaxation.
  - `infeasible`: a backward linear relaxation of the ReLUs yields
    `attainable_upper`, a sound upper bound on what the network (and the LP
    relaxation) can output over the box, and the threshold is placed strictly
    *between* `attainable_upper` and the (looser) interval bound declared on the
    output variable. A planted opposing neuron pair in the last hidden layer
    guarantees that this gap is nonempty.
  - `unknown`: the threshold is interpolated around the planted output without
    asserting either result.

# Fields

  - `architecture`: `:dense_mlp`, `:pruned_mlp`, or `:convolutional`.
  - `input_dim`, `input_shape`: number of inputs and their `(channels, height,
    width)` layout (`(input_dim, 1, 1)` for the MLPs).
  - `hidden_sizes`, `layer_kinds`: width and kind (`:dense`, `:pruned`,
    `:conv`) of each hidden ReLU layer.
  - `input_lower`, `input_upper`: input-box bounds.
  - `weights`, `biases`: hidden-layer affine maps (sparse weight matrices).
  - `pre_lower`, `pre_upper`: propagated preactivation bounds (big-M constants).
  - `activation_lower`, `activation_upper`: propagated ReLU-output bounds.
  - `phases`: ReLU phase classification (`-1`, `0`, or `1`).
  - `mirrored_pair`: planted opposing neuron pair in the last hidden layer.
  - `output_weights`, `output_bias`: scalar affine output layer.
  - `output_lower`, `output_upper`: declared scalar output bounds.
  - `interval_output_upper`: raw interval-propagation output bound.
  - `attainable_upper`: sound upper bound on the attainable network output.
  - `property_threshold`: right-hand side of `output >= property_threshold`.
  - `feasible_witness`: planted solution (`feasible` requests only).
  - `infeasibility_certificate`: bound-propagation certificate (`infeasible` only).
  - `feasibility_status`: the requested status.
"""
struct NeuralNetworkVerificationProblem <: ProblemGenerator
    architecture::Symbol
    input_dim::Int
    input_shape::NTuple{3, Int}
    hidden_sizes::Vector{Int}
    layer_kinds::Vector{Symbol}
    input_lower::Vector{Float64}
    input_upper::Vector{Float64}
    weights::Vector{SparseMatrixCSC{Float64, Int}}
    biases::Vector{Vector{Float64}}
    pre_lower::Vector{Vector{Float64}}
    pre_upper::Vector{Vector{Float64}}
    activation_lower::Vector{Vector{Float64}}
    activation_upper::Vector{Vector{Float64}}
    phases::Vector{Vector{Int8}}
    mirrored_pair::Tuple{Int, Int}
    output_weights::Vector{Float64}
    output_bias::Float64
    output_lower::Float64
    output_upper::Float64
    interval_output_upper::Float64
    attainable_upper::Float64
    property_threshold::Float64
    feasible_witness::Union{Nothing, ReluNetworkWitness}
    infeasibility_certificate::Union{Nothing, ReluOutputBoundCertificate}
    feasibility_status::FeasibilityStatus
end

# Minimum hidden width per layer; the last layer needs two unstable neurons for
# the planted opposing pair.
const NNV_MIN_LAYER_WIDTH = 6

# Targets from this size up use the sparse (pruned / convolutional) families.
const NNV_SPARSE_ARCHITECTURE_THRESHOLD = 2_500

"""
    nnv_affine_bounds(weights, bias, lower, upper) -> (lower, upper)

Exact bounds of `W * x + b` over the box `lower <= x <= upper`.
"""
function nnv_affine_bounds(
    weights::AbstractMatrix{<:Real},
    bias::AbstractVector{<:Real},
    lower::AbstractVector{<:Real},
    upper::AbstractVector{<:Real},
)
    size(weights, 1) == length(bias) || throw(DimensionMismatch("bias length"))
    size(weights, 2) == length(lower) == length(upper) ||
        throw(DimensionMismatch("box dimension"))
    positive = max.(weights, 0.0)
    negative = min.(weights, 0.0)
    affine_lower = Float64.(bias) .+ positive * lower .+ negative * upper
    affine_upper = Float64.(bias) .+ positive * upper .+ negative * lower
    return affine_lower, affine_upper
end

"""
    nnv_forward(weights, biases, input) -> (preactivations, activations)

Exact forward propagation of a concrete input through the ReLU network.
"""
function nnv_forward(
    weights::AbstractVector{<:AbstractMatrix{Float64}},
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

`lower_mode == :zero` always picks `a >= 0` for negative coefficients, which is
provably no worse than interval propagation; `:adaptive` uses the usual CROWN
heuristic (`a >= z` when `U >= -L`). The caller takes the better of the two.
"""
function nnv_backward_bound(
    weights::AbstractVector{<:AbstractMatrix{Float64}},
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
        bound += lambda[i] >= 0.0 ? lambda[i] * input_upper[i] : lambda[i] * input_lower[i]
    end
    return (
        bound=bound,
        input_coefficients=collect(lambda),
        input_constant=constant,
        slopes=slopes,
        intercepts=intercepts,
    )
end

# Variables contributed by one hidden layer with the given phase counts:
# one per active neuron, three (z, a, d) per unstable neuron.
nnv_layer_variables(n_active::Int, n_unstable::Int) = n_active + 3 * n_unstable

"""
    nnv_phase_counts(widths, unstable_fraction, inactive_fraction, budget)

Per-layer `(active, unstable, inactive)` counts. Nominal counts follow the two
fractions (every layer keeps at least two unstable neurons); the remaining
difference to the hidden-variable `budget` is then absorbed by converting
neurons between the stable phases (inactive ↔ active changes the count by one,
active → unstable by two), spread across layers, so the variable count is
exact whenever the widths can accommodate it.
"""
function nnv_phase_counts(
    widths::Vector{Int}, unstable_fraction::Float64, inactive_fraction::Float64, budget::Int
)
    n_layers = length(widths)
    unstable = [clamp(round(Int, unstable_fraction * w), 2, w) for w in widths]
    inactive = [
        clamp(round(Int, inactive_fraction * w), 0, w - unstable[l]) for
        (l, w) in enumerate(widths)
    ]
    active = [widths[l] - unstable[l] - inactive[l] for l in 1:n_layers]
    residual = budget - sum(nnv_layer_variables(active[l], unstable[l]) for l in 1:n_layers)

    # The first pass keeps at least 5% of each phase per layer; later passes
    # may use everything. Each layer takes an even share of what is left, and
    # passes repeat while they make progress (a narrow head layer can run out
    # of capacity before the wide layers do).
    for pass in 1:8
        residual == 0 && break
        floor_fraction = pass == 1 ? 0.05 : 0.0
        before = residual
        for l in 1:n_layers
            residual == 0 && break
            floor_count = round(Int, floor_fraction * widths[l])
            quota = cld(abs(residual), n_layers - l + 1)
            if residual > 0
                moved = clamp(min(quota, inactive[l] - floor_count), 0, quota)
                inactive[l] -= moved
                active[l] += moved
                residual -= moved
                promoted = clamp(min(fld(quota - moved, 2), active[l] - floor_count), 0, quota)
                active[l] -= promoted
                unstable[l] += promoted
                residual -= 2 * promoted
            else
                moved = clamp(min(quota, active[l] - floor_count), 0, quota)
                active[l] -= moved
                inactive[l] += moved
                residual += moved
                demoted = clamp(
                    min(fld(quota - moved, 2), unstable[l] - max(2, floor_count)), 0, quota
                )
                unstable[l] -= demoted
                active[l] += demoted
                residual += 2 * demoted
            end
        end
        pass > 1 && residual == before && break
    end
    # Parity fix-ups for an odd remainder of one.
    for l in 1:n_layers
        if residual == 1 && active[l] >= 2
            active[l] -= 2
            unstable[l] += 1
            inactive[l] += 1
            residual = 0
        elseif residual == -1 && unstable[l] > 2 && inactive[l] >= 1
            unstable[l] -= 1
            inactive[l] -= 1
            active[l] += 2
            residual = 0
        end
    end
    return active, unstable, inactive
end

# Sample `k` distinct indices from 1:n (k << n typical), sorted.
function nnv_sample_distinct(rng::AbstractRNG, n::Int, k::Int)
    k >= n && return collect(1:n)
    if 4k >= n
        return sort(randperm(rng, n)[1:k])
    end
    chosen = Set{Int}()
    while length(chosen) < k
        push!(chosen, rand(rng, 1:n))
    end
    return sort!(collect(chosen))
end

# Dense Gaussian layer, He/LeCun-style scaled.
function nnv_dense_layer(rng::AbstractRNG, width::Int, fan_in::Int)
    return sparse(randn(rng, width, fan_in) ./ sqrt(fan_in))
end

# Magnitude-pruned layer: every neuron keeps `k` surviving incoming weights.
function nnv_pruned_layer(rng::AbstractRNG, width::Int, fan_in::Int, k::Int)
    k = min(k, fan_in)
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    sizehint!(rows, width * k)
    sizehint!(cols, width * k)
    sizehint!(vals, width * k)
    scale = inv(sqrt(k))
    for i in 1:width
        for j in nnv_sample_distinct(rng, fan_in, k)
            push!(rows, i)
            push!(cols, j)
            # Pruning by magnitude removes small weights: survivors are bounded
            # away from zero.
            magnitude = (0.35 + abs(randn(rng))) * scale
            push!(vals, rand(rng, Bool) ? magnitude : -magnitude)
        end
    end
    return sparse(rows, cols, vals, width, fan_in)
end

# 3x3 stride-1 zero-padded convolution with shared kernels between
# (c_in, h, w) and (c_out, h, w) feature maps in channel-major flattening.
function nnv_conv_layer(rng::AbstractRNG, c_in::Int, c_out::Int, h::Int, w::Int)
    kernel = randn(rng, c_out, c_in, 3, 3) ./ sqrt(9 * c_in)
    index(c, y, x) = (c - 1) * h * w + (y - 1) * w + x
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    estimate = c_out * h * w * c_in * 9
    sizehint!(rows, estimate)
    sizehint!(cols, estimate)
    sizehint!(vals, estimate)
    for co in 1:c_out, y in 1:h, x in 1:w
        row = index(co, y, x)
        for ci in 1:c_in, dy in -1:1, dx in -1:1
            yy = y + dy
            xx = x + dx
            (1 <= yy <= h && 1 <= xx <= w) || continue
            push!(rows, row)
            push!(cols, index(ci, yy, xx))
            push!(vals, kernel[co, ci, dy + 2, dx + 2])
        end
    end
    return sparse(rows, cols, vals, c_out * h * w, c_in * h * w)
end

"""
    nnv_architecture(rng, target_variables, unstable_fraction, inactive_fraction)

Choose the architecture, input layout, hidden widths, and layer kinds so the
nominal variable count is close to `target_variables`; `nnv_phase_counts`
then makes it exact.
"""
function nnv_architecture(
    rng::AbstractRNG, target::Int, unstable_fraction::Float64, inactive_fraction::Float64
)
    per_neuron = (1.0 - unstable_fraction - inactive_fraction) + 3.0 * unstable_fraction
    if target < NNV_SPARSE_ARCHITECTURE_THRESHOLD
        input_dim = clamp(round(Int, sqrt(target) / 1.5), 2, 24)
        n_layers = target < 80 ? 1 : target < 300 ? 2 : rand(rng, 3:4)
        neurons = max(NNV_MIN_LAYER_WIDTH, round(Int, (target - input_dim - 1) / per_neuron))
        n_layers = min(n_layers, max(1, neurons ÷ NNV_MIN_LAYER_WIDTH))
        base, remainder = divrem(neurons, n_layers)
        widths = [max(NNV_MIN_LAYER_WIDTH, base + (l <= remainder ? 1 : 0)) for l in 1:n_layers]
        return (
            architecture=:dense_mlp,
            input_shape=(input_dim, 1, 1),
            widths=widths,
            kinds=fill(:dense, n_layers),
            fan_in=0,
        )
    end

    if rand(rng) < 0.5
        input_dim = clamp(round(Int, 0.02 * target), 64, 784)
        n_layers = rand(rng, 3:5)
        neurons = max(n_layers * NNV_MIN_LAYER_WIDTH, round(Int, (target - input_dim - 1) / per_neuron))
        base, remainder = divrem(neurons, n_layers)
        widths = [base + (l <= remainder ? 1 : 0) for l in 1:n_layers]
        return (
            architecture=:pruned_mlp,
            input_shape=(input_dim, 1, 1),
            widths=widths,
            kinds=fill(:pruned, n_layers),
            fan_in=rand(rng, 24:64),
        )
    end

    in_channels = rand(rng, (1, 3))
    channels = rand(rng) < 0.5 ? [4, 8] : [4, 8, 8]
    head_neurons = clamp(round(Int, 0.0015 * target), 12, 120)
    budget = target - 1 - per_neuron * head_neurons
    pixels = max(4, budget / (in_channels + per_neuron * sum(channels)))
    height = max(2, round(Int, sqrt(pixels)))
    width = max(2, round(Int, pixels / height))
    widths = vcat([c * height * width for c in channels], head_neurons)
    return (
        architecture=:convolutional,
        input_shape=(in_channels, height, width),
        widths=widths,
        kinds=vcat(fill(:conv, length(channels)), :dense),
        fan_in=0,
    )
end

"""
    NeuralNetworkVerificationProblem(target_variables, feasibility_status, seed)

Construct a deterministic bound-aware ReLU verification instance whose variable
count is `input_dim + 1 + Σ (active + 3 * unstable)` over the hidden layers,
matching `target_variables` exactly except at the smallest requests (where the
minimum network applies). See the type docstring for the architectures and the
feasibility mechanisms.
"""
function NeuralNetworkVerificationProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be positive (got $target_variables)"))

    rng = MersenneTwister(seed)
    unstable_fraction = 0.35 + 0.20 * rand(rng)
    inactive_fraction = 0.15 + 0.15 * rand(rng)
    arch = nnv_architecture(rng, target_variables, unstable_fraction, inactive_fraction)
    input_shape = arch.input_shape
    input_dim = prod(input_shape)
    hidden_sizes = arch.widths
    layer_kinds = arch.kinds
    n_layers = length(hidden_sizes)

    hidden_budget = target_variables - input_dim - 1
    n_active, n_unstable, n_inactive = nnv_phase_counts(
        hidden_sizes, unstable_fraction, inactive_fraction, hidden_budget
    )
    phases = Vector{Vector{Int8}}(undef, n_layers)
    for layer in 1:n_layers
        pattern = vcat(
            fill(Int8(0), n_unstable[layer]),
            fill(Int8(1), n_active[layer]),
            fill(Int8(-1), n_inactive[layer]),
        )
        phases[layer] = shuffle(rng, pattern)
    end

    # Input box. The small controller-style MLP keeps a broad box; the
    # image-like families use an L∞ ball of radius ε around a pixel image.
    if arch.architecture == :dense_mlp
        input_center = 2.0 .* rand(rng, input_dim) .- 1.0
        input_radius = 0.5 .+ rand(rng, input_dim)
        input_lower = input_center .- input_radius
        input_upper = input_center .+ input_radius
    else
        epsilon = 0.01 + 0.03 * rand(rng)
        image = 0.05 .+ 0.9 .* rand(rng, input_dim)
        input_lower = max.(0.0, image .- epsilon)
        input_upper = min.(1.0, image .+ epsilon)
    end

    weights = SparseMatrixCSC{Float64, Int}[]
    biases = Vector{Float64}[]
    pre_lower = Vector{Float64}[]
    pre_upper = Vector{Float64}[]
    activation_lower = Vector{Float64}[]
    activation_upper = Vector{Float64}[]

    previous_lower = input_lower
    previous_upper = input_upper
    previous_width = input_dim
    channels_in = input_shape[1]

    # The two unstable neurons of the last hidden layer that carry the planted
    # opposing pair `w_v = -w_u`, `b_v = -b_u`. At any point at most one of
    # `relu(z)` and `relu(-z)` is positive while interval propagation adds both
    # maxima, so this pair makes interval propagation provably loose - which is
    # exactly the room the infeasible threshold is placed in.
    last_unstable = findall(==(Int8(0)), phases[end])
    mirrored_pair = (last_unstable[1], last_unstable[2])

    for layer in 1:n_layers
        width = hidden_sizes[layer]
        kind = layer_kinds[layer]
        layer_weights = if kind == :dense
            nnv_dense_layer(rng, width, previous_width)
        elseif kind == :pruned
            nnv_pruned_layer(rng, width, previous_width, arch.fan_in)
        else
            _, h, w = input_shape
            channels_out = width ÷ (h * w)
            conv = nnv_conv_layer(rng, channels_in, channels_out, h, w)
            channels_in = channels_out
            conv
        end
        if layer == n_layers
            u, v = mirrored_pair
            # Row v becomes the exact negation of row u (same sparsity pattern).
            layer_weights[v, :] = -layer_weights[u, :]
            dropzeros!(layer_weights)
        end

        # No-bias interval first, then a per-neuron bias realising the planted
        # phase. The rule is odd-symmetric on unstable neurons, so the mirrored
        # row receives the negated bias and satisfies `z_v == -z_u` exactly.
        raw_lower, raw_upper = nnv_affine_bounds(
            layer_weights, zeros(width), previous_lower, previous_upper
        )
        layer_bias = zeros(width)
        for neuron in 1:width
            span = raw_upper[neuron] - raw_lower[neuron]
            margin = max(0.1 * span, 1.0e-6)
            phase = phases[layer][neuron]
            if phase == 0
                # Off-centre crossing points make the triangle relaxations
                # heterogeneous; the mirrored pair stays exactly symmetric.
                shift = neuron in mirrored_pair && layer == n_layers ? 0.0 : 0.3 * (rand(rng) - 0.5)
                layer_bias[neuron] = -(0.5 + shift) * raw_lower[neuron] - (0.5 - shift) * raw_upper[neuron]
            elseif phase == 1
                layer_bias[neuron] = -raw_lower[neuron] + margin
            else
                layer_bias[neuron] = -raw_upper[neuron] - margin
            end
        end
        if layer == n_layers
            u, v = mirrored_pair
            layer_bias[v] = -layer_bias[u]
        end

        layer_pre_lower, layer_pre_upper = nnv_affine_bounds(
            layer_weights, layer_bias, previous_lower, previous_upper
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
        output_weights[index] = max(abs(output_weights[index]), 0.25 * inv(sqrt(previous_width)))
    end

    output_lower_vec, output_upper_vec = nnv_affine_bounds(
        reshape(output_weights, 1, :), [output_bias], previous_lower, previous_upper
    )
    output_lower = output_lower_vec[1]
    interval_output_upper = output_upper_vec[1]

    mirrored_gap = minimum(output_weights[index] * pre_upper[end][index] for index in mirrored_pair)

    relaxation_zero = nnv_backward_bound(
        weights,
        biases,
        pre_lower,
        pre_upper,
        output_weights,
        output_bias,
        input_lower,
        input_upper;
        lower_mode=:zero,
    )
    relaxation_adaptive = nnv_backward_bound(
        weights,
        biases,
        pre_lower,
        pre_upper,
        output_weights,
        output_bias,
        input_lower,
        input_upper;
        lower_mode=:adaptive,
    )
    relaxation =
        relaxation_adaptive.bound < relaxation_zero.bound ? relaxation_adaptive : relaxation_zero
    attainable_upper = relaxation.bound

    # Deterministic search for a high-output point of the input box: the box
    # centre, the vertices suggested by the two backward relaxations, and a
    # sampled mix of vertices and interior points.
    candidates = Vector{Vector{Float64}}()
    push!(candidates, 0.5 .* (input_lower .+ input_upper))
    for coefficients in (relaxation_zero.input_coefficients, relaxation_adaptive.input_coefficients)
        push!(
            candidates,
            [coefficients[i] >= 0.0 ? input_upper[i] : input_lower[i] for i in 1:input_dim],
        )
    end
    for _ in 1:8
        push!(candidates, [rand(rng) < 0.5 ? input_lower[i] : input_upper[i] for i in 1:input_dim])
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
            witness_output + (-0.25 + 1.6 * rand(rng)) * scale,
            output_lower + 0.02 * output_span,
            output_upper - 0.02 * output_span,
        )
    end

    witness = if feasibility_status == feasible
        relu_binaries = [
            Int8[
                witness_pre[layer][neuron] > 0.0 ? Int8(1) : Int8(0) for
                neuron in findall(==(Int8(0)), phases[layer])
            ] for layer in 1:n_layers
        ]
        ReluNetworkWitness(witness_input, witness_pre, witness_act, relu_binaries, witness_output)
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
        arch.architecture,
        input_dim,
        input_shape,
        hidden_sizes,
        layer_kinds,
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
    nnv_variable_count(prob)

Exact column count of `build_model(prob)`.
"""
function nnv_variable_count(prob::NeuralNetworkVerificationProblem)
    hidden = sum(
        count(==(Int8(1)), phases) + 3 * count(==(Int8(0)), phases) for phases in prob.phases
    )
    return prob.input_dim + 1 + hidden
end

"""
    build_model(prob::NeuralNetworkVerificationProblem)

Build the ReLU verification MILP. A stably inactive neuron is omitted, a stably
active neuron is one activation column with its affine defining row, and an
unstable neuron with bounds `L < 0 < U` uses the ideal single-neuron big-M
encoding

```text
z = w'x + b,  a >= z,  a >= 0,  a <= U*d,  a <= z - L*(1-d),  d binary,
```

with `a >= 0` expressed as the activation's lower bound. The big-M constants
are the propagated preactivation bounds of that neuron; relaxing `d` to
`[0, 1]` projects these rows exactly onto the triangle relaxation of the ReLU.

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
    previous = Vector{Union{Nothing, VariableRef}}(input)
    for layer in 1:n_layers
        width = prob.hidden_sizes[layer]
        phases = prob.phases[layer]
        lower = prob.pre_lower[layer]
        upper = prob.pre_upper[layer]
        unstable = findall(==(Int8(0)), phases)
        active = findall(==(Int8(1)), phases)

        preactivation = @variable(
            model,
            [k = 1:length(unstable)],
            lower_bound = lower[unstable[k]],
            upper_bound = upper[unstable[k]],
            base_name = "preactivation_$layer",
        )
        activation = Vector{Union{Nothing, VariableRef}}(nothing, width)
        active_vars = @variable(
            model,
            [k = 1:length(active)],
            lower_bound = prob.activation_lower[layer][active[k]],
            upper_bound = prob.activation_upper[layer][active[k]],
            base_name = "active_$layer",
        )
        unstable_vars = @variable(
            model,
            [k = 1:length(unstable)],
            lower_bound = 0.0,
            upper_bound = prob.activation_upper[layer][unstable[k]],
            base_name = "activation_$layer",
        )
        phase_binary = @variable(
            model, [k = 1:length(unstable)], Bin, base_name = "relu_phase_$layer",
        )
        for (k, neuron) in enumerate(active)
            activation[neuron] = active_vars[k]
        end
        for (k, neuron) in enumerate(unstable)
            activation[neuron] = unstable_vars[k]
        end

        # Row-wise access to the sparse weights; stably inactive neurons of the
        # previous layer contribute identically zero and are skipped.
        rows = sparse(transpose(prob.weights[layer]))
        row_index = rowvals(rows)
        row_value = nonzeros(rows)
        function affine(neuron)
            expression = AffExpr(prob.biases[layer][neuron])
            for pointer in nzrange(rows, neuron)
                source = previous[row_index[pointer]]
                source === nothing && continue
                add_to_expression!(expression, row_value[pointer], source)
            end
            return expression
        end
        for (k, neuron) in enumerate(active)
            @constraint(model, active_vars[k] == affine(neuron))
        end
        for (k, neuron) in enumerate(unstable)
            z = preactivation[k]
            a = unstable_vars[k]
            d = phase_binary[k]
            @constraint(model, z == affine(neuron))
            @constraint(model, a >= z)
            @constraint(model, a <= upper[neuron] * d)
            @constraint(model, a <= z - lower[neuron] * (1.0 - d))
        end
        previous = activation
    end

    output = @variable(
        model,
        lower_bound = prob.output_lower,
        upper_bound = prob.output_upper,
        base_name = "output",
    )
    output_expression = AffExpr(prob.output_bias)
    for (j, weight) in enumerate(prob.output_weights)
        source = previous[j]
        source === nothing && continue
        add_to_expression!(output_expression, weight, source)
    end
    @constraint(model, output == output_expression)

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
    "Bound-aware ReLU verification (dense, pruned, or convolutional networks) with stable-neuron elimination and propagated big-M coefficients";
    default=true,
)
