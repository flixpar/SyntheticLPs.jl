# Focused quality contracts for the neural_network_verification category:
# registry shape, the exact variable-count formula, the planted opposing pair,
# exact witness arithmetic (the planted input is re-propagated and checked
# against every row of the big-M encoding), the backward-relaxation
# infeasibility certificate, the presolve-hardness sandwich
# `attainable_upper < threshold < interval_upper`, reproducibility, and
# HiGHS feasibility contracts on both the LP relaxation and the MILP.
@testset "Neural Network Verification" begin
    nnv = ProblemVariant(:neural_network_verification, :relu_big_m)

    @test :neural_network_verification in list_categories()
    @test list_variants(:neural_network_verification) == [:relu_big_m]
    info = problem_info(:neural_network_verification)
    @test info[:default_variant] == :relu_big_m
    @test occursin("network", lowercase(info[:description]))
    @test ProblemVariant("neural_network_verification") == nnv

    unstable_count(p) = sum(count(==(Int8(0)), ph) for ph in p.phases)

    # Exact variable-count formula and sizing across a committed matrix.
    for target in (50, 100, 500, 1000, 5000),
        status in (feasible, infeasible, unknown), seed in 0:3
        m, p = generate_problem(nnv, target, status, seed)
        @test num_variables(m) ==
              p.input_dim + 1 + 2 * sum(p.hidden_sizes) + unstable_count(p)
        @test abs(num_variables(m) - target) <= 0.25 * target
        @test p.feasibility_status == status
        # Every layer is wide enough to host two unstable ReLUs.
        @test minimum(p.hidden_sizes) >= SyntheticLPs.NNV_MIN_LAYER_WIDTH
        @test all(count(==(Int8(0)), ph) >= 2 for ph in p.phases)
    end
    # Tiny requests round up to the minimum network instead of degenerating.
    for seed in 0:2
        m, p = generate_problem(nnv, 1, unknown, seed)
        @test sum(p.hidden_sizes) == SyntheticLPs.NNV_MIN_LAYER_WIDTH
        @test num_variables(m) <= 50
    end

    # Bound/phase data contract: the stored intervals are the propagated ones
    # and they certify each neuron's declared phase, so the big-M constants of
    # `build_model` are exactly the propagated per-neuron bounds.
    for target in (100, 800), seed in 0:2
        _, p = generate_problem(nnv, target, unknown, seed)
        previous_lower, previous_upper = p.input_lower, p.input_upper
        for layer in eachindex(p.hidden_sizes)
            lo, up = SyntheticLPs.nnv_affine_bounds(
                p.weights[layer], p.biases[layer], previous_lower, previous_upper,
            )
            @test lo ≈ p.pre_lower[layer]
            @test up ≈ p.pre_upper[layer]
            @test p.activation_lower[layer] ≈ max.(0.0, p.pre_lower[layer])
            @test p.activation_upper[layer] ≈ max.(0.0, p.pre_upper[layer])
            for neuron in eachindex(p.phases[layer])
                phase = p.phases[layer][neuron]
                if phase == 1
                    @test p.pre_lower[layer][neuron] > 0.0
                elseif phase == -1
                    @test p.pre_upper[layer][neuron] < 0.0
                else
                    @test p.pre_lower[layer][neuron] < 0.0 <
                          p.pre_upper[layer][neuron]
                end
            end
            previous_lower, previous_upper =
                p.activation_lower[layer], p.activation_upper[layer]
        end
        out_lo, out_up = SyntheticLPs.nnv_affine_bounds(
            reshape(p.output_weights, 1, :), [p.output_bias],
            previous_lower, previous_upper,
        )
        @test p.output_lower ≈ out_lo[1]
        @test p.interval_output_upper ≈ out_up[1]
        @test p.output_upper >= p.interval_output_upper
    end

    # The planted opposing pair: mirrored weights, odd-symmetric bias, exactly
    # negated preactivation interval, and positive output weights on both.
    for target in (60, 300, 1500), status in (feasible, infeasible, unknown),
        seed in 0:2
        _, p = generate_problem(nnv, target, status, seed)
        u, v = p.mirrored_pair
        @test u != v
        @test p.phases[end][u] == Int8(0) && p.phases[end][v] == Int8(0)
        @test p.weights[end][v, :] ≈ -p.weights[end][u, :]
        @test p.biases[end][v] ≈ -p.biases[end][u]
        @test p.pre_lower[end][v] ≈ -p.pre_upper[end][u]
        @test p.pre_upper[end][v] ≈ -p.pre_lower[end][u]
        @test p.output_weights[u] > 0.0 && p.output_weights[v] > 0.0
    end

    # ---- Feasible witness: exact re-propagation and full row satisfaction ----
    for target in (50, 200, 1200), seed in 0:4
        _, p = generate_problem(nnv, target, feasible, seed)
        w = p.feasible_witness
        @test w !== nothing
        @test p.infeasibility_certificate === nothing
        @test all(p.input_lower .<= w.input .<= p.input_upper)

        current = w.input
        for layer in eachindex(p.hidden_sizes)
            pre = p.weights[layer] * current .+ p.biases[layer]
            act = max.(0.0, pre)
            @test pre ≈ w.preactivations[layer]
            @test act ≈ w.activations[layer]
            # Variable bounds of the model.
            @test all(p.pre_lower[layer] .- 1e-8 .<= pre .<=
                      p.pre_upper[layer] .+ 1e-8)
            @test all(p.activation_lower[layer] .- 1e-8 .<= act .<=
                      p.activation_upper[layer] .+ 1e-8)

            unstable = findall(==(Int8(0)), p.phases[layer])
            @test length(w.relu_binaries[layer]) == length(unstable)
            for neuron in eachindex(p.phases[layer])
                phase = p.phases[layer][neuron]
                if phase == -1
                    @test isapprox(act[neuron], 0.0; atol = 1e-9)
                elseif phase == 1
                    @test isapprox(act[neuron], pre[neuron]; atol = 1e-9)
                end
            end
            # The big-M rows, evaluated at the induced binary pattern.
            for (k, neuron) in enumerate(unstable)
                d = Float64(w.relu_binaries[layer][k])
                @test d in (0.0, 1.0)
                @test d == (pre[neuron] > 0.0 ? 1.0 : 0.0)
                lower = p.pre_lower[layer][neuron]
                upper = p.pre_upper[layer][neuron]
                @test act[neuron] >= pre[neuron] - 1e-9
                @test act[neuron] >= -1e-9
                @test act[neuron] <= upper * d + 1e-9
                @test act[neuron] <= pre[neuron] - lower * (1.0 - d) + 1e-9
            end
            current = act
        end
        output = dot(p.output_weights, current) + p.output_bias
        @test output ≈ w.output
        @test p.output_lower - 1e-8 <= output <= p.output_upper + 1e-8
        # The property row holds strictly and is not implied by a bound.
        @test output >= p.property_threshold
        @test p.property_threshold > p.output_lower
        @test p.property_threshold <= p.attainable_upper
    end

    # ---- Infeasibility certificate ----
    sample_max(p, rng, n) = maximum(
        begin
            x = [rand(rng) < 0.5 ?
                 (rand(rng) < 0.5 ? p.input_lower[i] : p.input_upper[i]) :
                 p.input_lower[i] +
                 rand(rng) * (p.input_upper[i] - p.input_lower[i])
                 for i in 1:p.input_dim]
            dot(p.output_weights,
                SyntheticLPs.nnv_forward(p.weights, p.biases, x)[2][end]) +
                p.output_bias
        end for _ in 1:n
    )

    for target in (50, 200, 1200), seed in 0:4
        _, p = generate_problem(nnv, target, infeasible, seed)
        cert = p.infeasibility_certificate
        @test cert !== nothing
        @test p.feasible_witness === nothing

        # The stored affine collapse reproduces the bound exactly.
        bound = cert.input_constant + sum(
            max(cert.input_coefficients[i] * p.input_lower[i],
                cert.input_coefficients[i] * p.input_upper[i])
            for i in 1:p.input_dim
        )
        @test bound ≈ cert.attainable_upper
        @test cert.attainable_upper ≈ p.attainable_upper

        # Replaying the backward substitution from the stored per-neuron lines
        # reproduces the collapsed affine function, and every substituted line
        # is a valid relaxation of its ReLU in the direction that matters.
        lambda = copy(p.output_weights)
        constant = p.output_bias
        for layer in length(p.hidden_sizes):-1:1
            mu = similar(lambda)
            for j in eachindex(lambda)
                slope = cert.relaxation_slopes[layer][j]
                intercept = cert.relaxation_intercepts[layer][j]
                lo = p.pre_lower[layer][j]
                up = p.pre_upper[layer][j]
                for z in (lo, up, 0.5 * (lo + up))
                    line = slope * z + intercept
                    relu = max(0.0, z)
                    if lambda[j] >= 0.0
                        @test line >= relu - 1e-7        # valid upper bound
                    else
                        @test line <= relu + 1e-7        # valid lower bound
                    end
                end
                mu[j] = lambda[j] * slope
                constant += lambda[j] * intercept
            end
            constant += dot(mu, p.biases[layer])
            lambda = transpose(p.weights[layer]) * mu
        end
        @test lambda ≈ cert.input_coefficients
        @test constant ≈ cert.input_constant

        # The planted pair makes interval propagation provably loose.
        u, v = cert.mirrored_pair
        @test cert.mirrored_pair == p.mirrored_pair
        @test cert.mirrored_gap ≈ min(p.output_weights[u] * p.pre_upper[end][u],
                                      p.output_weights[v] * p.pre_upper[end][v])
        @test cert.mirrored_gap > 0.0
        @test cert.attainable_upper <= cert.interval_upper - cert.mirrored_gap + 1e-7

        # THE presolve-hardness contract: the threshold is unreachable for the
        # network but consistent with the output variable's own bound and with
        # interval propagation over the rows.
        @test cert.interval_upper ≈ p.interval_output_upper
        @test cert.declared_upper ≈ p.output_upper
        @test p.attainable_upper < p.property_threshold < p.output_upper
        @test p.property_threshold < p.interval_output_upper
        @test p.property_threshold > p.output_lower

        # The bound really is sound: no sampled input reaches it.
        rng = MersenneTwister(1234 + seed)
        @test sample_max(p, rng, 400) < p.attainable_upper
    end

    # Reproducibility and global-RNG isolation. The witness/certificate fields
    # are composite types, so equality is compared structurally.
    deep_equal(a, b) = isequal(a, b) || (typeof(a) === typeof(b) &&
        !isempty(fieldnames(typeof(a))) &&
        all(deep_equal(getfield(a, f), getfield(b, f))
            for f in fieldnames(typeof(a))))
    for status in (feasible, infeasible, unknown)
        Random.seed!(987)
        _, p1 = generate_problem(nnv, 220, status, 42)
        Random.seed!(12345)
        _, p2 = generate_problem(nnv, 220, status, 42)
        @test all(deep_equal(getfield(p1, f), getfield(p2, f))
                  for f in fieldnames(typeof(p1)))
        m1 = SyntheticLPs.build_model(p1)
        m2 = SyntheticLPs.build_model(p2)
        @test num_variables(m1) == num_variables(m2)
        @test sprint(print, m1) == sprint(print, m2)
    end

    if HAS_HIGHS
        # The feasibility contract holds on the LP relaxation (the package
        # default) - infeasibility survives relaxing the ReLU binaries.
        for target in (60, 200, 800), status in (feasible, infeasible), seed in 0:4
            m, _ = generate_problem(nnv, target, status, seed)
            set_optimizer(m, HiGHS.Optimizer)
            set_silent(m)
            optimize!(m)
            expected = status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE
            @test termination_status(m) == expected
        end

        # ... and on the unrelaxed MILP, where the feasible witness must also
        # be integrally realisable.
        for target in (60, 200), status in (feasible, infeasible), seed in 0:2
            m, _ = generate_problem(nnv, target, status, seed; relax_integer = false)
            set_optimizer(m, HiGHS.Optimizer)
            set_silent(m)
            set_attribute(m, "time_limit", 60.0)
            optimize!(m)
            expected = status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE
            @test termination_status(m) == expected
        end

        # Regression guard for the actual defect: an infeasible instance must
        # not be refutable without real simplex work. With presolve switched
        # off the old formulation needed a handful of iterations (the property
        # clashed with the output variable's own bound); the propagated
        # threshold needs hundreds.
        for target in (500, 1000), seed in 0:2
            m, _ = generate_problem(nnv, target, infeasible, seed)
            set_optimizer(m, HiGHS.Optimizer)
            set_silent(m)
            set_attribute(m, "presolve", "off")
            optimize!(m)
            @test termination_status(m) == MOI.INFEASIBLE
            @test MOI.get(m, MOI.SimplexIterations()) >= 50
        end

        # Unknown is a genuine mix rather than an implicit one-way branch.
        optimal_count = 0
        infeasible_count = 0
        for target in (100, 400), seed in 0:14
            m, _ = generate_problem(nnv, target, unknown, seed)
            set_optimizer(m, HiGHS.Optimizer)
            set_silent(m)
            optimize!(m)
            if termination_status(m) == MOI.OPTIMAL
                optimal_count += 1
            elseif termination_status(m) == MOI.INFEASIBLE
                infeasible_count += 1
            end
        end
        @test optimal_count > 0
        @test infeasible_count > 0
    end
end
