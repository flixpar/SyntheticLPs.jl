# Neural Network Verification

The `neural_network_verification` category generates verification queries for
piecewise-linear (ReLU) neural networks in the mixed-integer encoding used by
MIP-based verifiers (MIPVerify, Tjeng et al.; the ideal single-neuron big-M
formulation of Anderson et al.). Under the package default
`relax_integer=true` the result is the **triangle LP relaxation** of the
network — the same LP that bound-propagation and LP-based verifiers solve —
so the instances are useful as LPs in their own right.

There is one variant, `relu_big_m` (the default).

## The query

Given a network `f`, an input box `[l, u]`, and a threshold `τ`, the model
asks whether some input in the box reaches `f(x) >= τ`:

```text
max  y
s.t. y = c' a_L + c0,   y >= τ
     network rows (below),   l <= x <= u
```

## Architectures

| Architecture | Used for | Input box | Hidden layers | Nonzeros per row |
| --- | --- | --- | --- | --- |
| `:dense_mlp` | targets < 2,500 | 2–24 inputs, `center ± radius` (ACAS-Xu-like controller) | 1–4 fully connected | layer width |
| `:pruned_mlp` | ≥ 2,500, by seed | 64–784 pixels, `[x0 − ε, x0 + ε] ∩ [0, 1]`, ε ∈ [0.01, 0.04] | 3–5 magnitude-pruned layers | fixed fan-in 24–64 |
| `:convolutional` | ≥ 2,500, by seed | 1- or 3-channel `H × W` image, same ε-ball | 2–3 conv layers (3×3, stride 1, zero padding, 4/8/8 channels) + dense head | ≤ 9 · channels (conv), flattened map (head) |

Convolution layers share their kernels (coefficient values repeat across
rows, as in a real CNN) but use untied per-neuron biases (see phases below).
Weights are scaled by `1/sqrt(fan_in)`; pruned survivors are bounded away from
zero, mimicking magnitude pruning.

The previous generator used only dense MLPs, whose nonzeros grow
quadratically (4.1M at 10k variables, MPS files of several GB at 50k).
Nonzeros now grow linearly: about 17–23 per column at 10k and 100k.

## Phases and formulation

Interval bounds `[L, U]` are propagated through every affine layer. Each
neuron's bias is then chosen to realise a planted phase: unstable (`L < 0 <
U`, 35–55% of neurons, with off-centre crossing points), stably active (`L >
0`), or stably inactive (`U < 0`, 15–30%). `build_model` follows what
MIP-based verifiers do with stable neurons:

| Phase | Columns | Rows |
| --- | --- | --- |
| inactive | none (the activation is identically zero and is dropped from the next layer's rows) | none |
| active | `a` with bounds `[L, U]` | `a = w'x + b` |
| unstable | `z ∈ [L, U]`, `a ∈ [0, U]`, `d ∈ {0, 1}` | `z = w'x + b`, `a >= z`, `a <= U d`, `a <= z − L(1 − d)` |

The big-M constants are the propagated bounds of that neuron, so relaxing `d`
to `[0, 1]` gives exactly the triangle relaxation. The previous encoding also
emitted stable neurons as `a == 0` / `a == z` rows with their own `z`
columns; HiGHS presolve eliminated those and kept only ~52% of the model. Now
presolve keeps 92–100% of the columns at 10k–100k.

Variable count, exact for targets above a few dozen variables:

```text
inputs + 1 + (active neurons) + 3 * (unstable neurons)
```

Widths are chosen from the request, and the remaining difference is absorbed
by converting a few neurons between phases (`nnv_phase_counts`).

## Feasibility control

- `feasible`: a high-output input is found by propagating the box centre,
  the vertices suggested by two backward relaxations, and a sampled mix of
  vertices and interior points; `τ` is set strictly below its output.
  `feasible_witness` (`ReluNetworkWitness`) stores the input, every
  pre-activation and activation (including inactive neurons), the induced
  binaries of the unstable neurons, and the output — a feasible point of the
  MILP and of its relaxation.
- `infeasible`: a backward linear relaxation (CROWN/DeepPoly style, the better
  of two lower-relaxation choices) gives `attainable_upper`, a sound upper
  bound on the LP relaxation's maximum. `τ` is placed strictly between it and
  the interval-propagation bound declared on the output variable, so no single
  bound and no interval propagation over the rows refutes the query; a planted
  opposing neuron pair in the last hidden layer (`w_v = −w_u`, `b_v = −b_u`)
  guarantees the gap is nonempty. `infeasibility_certificate`
  (`ReluOutputBoundCertificate`) stores the collapsed affine function and every
  substituted ReLU line, which the tests replay without a solver. The LP needs
  thousands of simplex iterations to prove infeasibility at 10k.
- `unknown`: `τ` is drawn around the planted output, from somewhat below it to
  beyond `attainable_upper`; a sample of seeds contains both outcomes.

## Footprint (seeds 0–1)

| Target | Rows | Nonzeros | Build | Presolve keeps (cols) |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 1.2k | 50–57k | < 0.1 s | 92–94% |
| 10,000 | 11–12k | 171–190k | < 0.5 s | 92–100% |
| 100,000 | 114–124k | 1.7–2.3M | 2–4 s | 92–100% |

The LP relaxations are hard for simplex: at 10k the feasible instances take
10k–20k iterations, and 100k instances do not finish within a 60 s limit.

## References

- V. Tjeng, K. Xiao, R. Tedrake. *Evaluating Robustness of Neural Networks with
  Mixed Integer Programming*, ICLR 2019.
- R. Anderson, J. Huchette, W. Ma, C. Tjandraatmadja, J. P. Vielma. *Strong
  mixed-integer programming formulations for trained neural networks*,
  Mathematical Programming 2020.
- H. Zhang, T.-W. Weng, P.-Y. Chen, C.-J. Hsieh, L. Daniel. *Efficient Neural
  Network Robustness Certification with General Activation Functions*
  (CROWN), NeurIPS 2018.
- G. Katz et al. *Reluplex*, CAV 2017 (ACAS Xu networks).
