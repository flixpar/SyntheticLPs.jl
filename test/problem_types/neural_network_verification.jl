# Focused quality contracts for the neural_network_verification category:
# registry shape, architecture selection, exact variable-count and row
# formulas, bounded sparse weights (nonzeros linear in the request), the
# planted opposing pair, exact witness arithmetic (the planted input is
# re-propagated and checked against every row of the big-M encoding), the
# backward-relaxation infeasibility certificate, the presolve-hardness sandwich
# `attainable_upper < threshold < interval_upper`, reproducibility, and HiGHS
# feasibility / presolve-survival contracts on the LP relaxation and the MILP.
using SparseArrays

function _nnv_test_constraint_nnz(model)
    total = 0
    for (F, S) in list_of_constraint_types(model)
        F <: VariableRef && continue
        for c in all_constraints(model, F, S)
            total += length(constraint_object(c).func.terms)
        end
    end
    return total
end

@testset "Neural Network Verification" begin
    nnv = ProblemVariant(:neural_network_verification, :relu_big_m)

    @test :neural_network_verification in list_categories()
    @test list_variants(:neural_network_verification) == [:relu_big_m]
    info = problem_info(:neural_network_verification)
    @test info[:default_variant] == :relu_big_m
    @test occursin("network", lowercase(info[:description]))
    @test ProblemVariant("neural_network_verification") == nnv

    phase_count(p, phase) = sum(count(==(Int8(phase)), ph) for ph in p.phases)

    @testset "exact sizing, architectures, and row formula" begin
        for target in (30, 50, 100, 500, 1000, 2499, 2500, 5000, 12_000),
            status in (feasible, infeasible, unknown),
            seed in 0:1

            m, p = generate_problem(nnv, target, status, seed)
            n_active = phase_count(p, 1)
            n_unstable = phase_count(p, 0)
            @test num_variables(m) == SyntheticLPs.nnv_variable_count(p)
            @test num_variables(m) == p.input_dim + 1 + n_active + 3 * n_unstable
            @test num_variables(m) == target
            # One defining row per modelled neuron, three big-M rows per
            # unstable neuron, plus the output definition and the property.
            @test num_constraints(m; count_variable_in_set_constraints=false) ==
                n_active + 4 * n_unstable + 2
            @test count(is_binary, all_variables(m)) == 0           # relaxed by default
            @test p.feasibility_status == status
            @test length(p.hidden_sizes) == length(p.layer_kinds) == length(p.weights)
            @test p.input_dim == prod(p.input_shape)
            @test all(count(==(Int8(0)), ph) >= 2 for ph in p.phases)
            @test minimum(p.hidden_sizes) >= SyntheticLPs.NNV_MIN_LAYER_WIDTH
            if target < SyntheticLPs.NNV_SPARSE_ARCHITECTURE_THRESHOLD
                @test p.architecture == :dense_mlp
                @test all(==(:dense), p.layer_kinds)
            else
                @test p.architecture in (:pruned_mlp, :convolutional)
            end
            # A real mix of phases, so the relaxation is not mostly substitutable.
            total = sum(p.hidden_sizes)
            @test n_unstable >= 0.25 * total
            @test phase_count(p, -1) >= 1
        end

        # Tiny requests round up to the minimum network.
        for seed in 0:2
            m, p = generate_problem(nnv, 1, unknown, seed)
            @test p.hidden_sizes == [SyntheticLPs.NNV_MIN_LAYER_WIDTH]
            @test num_variables(m) <= 15
        end

        # Both sparse families appear at large sizes.
        archs = Set(generate_problem(nnv, 3000, unknown, seed)[2].architecture for seed in 0:5)
        @test archs == Set((:pruned_mlp, :convolutional))
    end

    @testset "sparse weight structure and bounded nonzeros" begin
        for seed in 0:5
            _, p = generate_problem(nnv, 4000, feasible, seed)
            if p.architecture == :pruned_mlp
                for (layer, W) in enumerate(p.weights)
                    row_counts = vec(sum(W .!= 0.0; dims=2))
                    @test all(==(row_counts[1]), row_counts)            # fixed fan-in
                    @test 24 <= row_counts[1] <= 64
                    @test minimum(abs, nonzeros(W)) > 0.0
                end
            elseif p.architecture == :convolutional
                c0, h, w = p.input_shape
                @test c0 in (1, 3)
                for (layer, W) in enumerate(p.weights)
                    if p.layer_kinds[layer] == :conv
                        c_in = layer == 1 ? c0 : p.hidden_sizes[layer - 1] ÷ (h * w)
                        @test size(W, 2) == c_in * h * w
                        @test p.hidden_sizes[layer] % (h * w) == 0
                        row_counts = vec(sum(W .!= 0.0; dims=2))
                        @test maximum(row_counts) == 9 * c_in      # interior pixels
                        @test minimum(row_counts) == 4 * c_in      # corner pixels
                        # Shared kernels: an interior pixel's row is a
                        # translate of its neighbour's row.
                        y, x = cld(h, 2), cld(w, 2)
                        row = y * w - w + x
                        right = row + 1
                        if x < w - 1
                            @test sort(nonzeros(sparse(W[row, :]))) ≈
                                sort(nonzeros(sparse(W[right, :])))
                        end
                    else
                        @test p.layer_kinds[layer] == :dense
                        @test layer == length(p.weights)
                    end
                end
            end
            # Images in [0, 1] with a small L∞ ball.
            @test all(0.0 .<= p.input_lower .< p.input_upper .<= 1.0)
            @test maximum(p.input_upper .- p.input_lower) <= 0.08 + 1e-12
        end

        # Nonzeros stay linear in the request (the old dense MLP had 4.1M at
        # 10k variables and was quadratic), and a 100k request builds fast.
        for seed in 0:1
            m, p = generate_problem(nnv, 100_000, feasible, seed)
            @test num_variables(m) == 100_000
            @test _nnv_test_constraint_nnz(m) <= 6_000_000
            @test p.architecture in (:pruned_mlp, :convolutional)
        end
    end

    @testset "bound and phase data" begin
        for target in (100, 800, 3000), seed in 0:2
            _, p = generate_problem(nnv, target, unknown, seed)
            previous_lower, previous_upper = p.input_lower, p.input_upper
            for layer in eachindex(p.hidden_sizes)
                lo, up = SyntheticLPs.nnv_affine_bounds(
                    p.weights[layer], p.biases[layer], previous_lower, previous_upper
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
                        @test p.pre_lower[layer][neuron] < 0.0 < p.pre_upper[layer][neuron]
                    end
                end
                previous_lower, previous_upper = p.activation_lower[layer],
                p.activation_upper[layer]
            end
            out_lo, out_up = SyntheticLPs.nnv_affine_bounds(
                reshape(p.output_weights, 1, :), [p.output_bias], previous_lower, previous_upper
            )
            @test p.output_lower ≈ out_lo[1]
            @test p.interval_output_upper ≈ out_up[1]
            @test p.output_upper >= p.interval_output_upper
        end
    end

    @testset "planted opposing pair" begin
        for target in (60, 300, 1500, 3000), status in (feasible, infeasible, unknown), seed in 0:2
            _, p = generate_problem(nnv, target, status, seed)
            u, v = p.mirrored_pair
            @test u != v
            @test p.phases[end][u] == Int8(0) && p.phases[end][v] == Int8(0)
            @test Vector(p.weights[end][v, :]) == -Vector(p.weights[end][u, :])
            @test p.biases[end][v] == -p.biases[end][u]
            @test p.pre_lower[end][v] ≈ -p.pre_upper[end][u]
            @test p.pre_upper[end][v] ≈ -p.pre_lower[end][u]
            @test p.output_weights[u] > 0.0 && p.output_weights[v] > 0.0
        end
    end

    @testset "feasible witness" begin
        for target in (50, 200, 1200, 3000), seed in 0:3
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
                @test all(p.pre_lower[layer] .- 1e-8 .<= pre .<= p.pre_upper[layer] .+ 1e-8)
                @test all(
                    p.activation_lower[layer] .- 1e-8 .<= act .<= p.activation_upper[layer] .+ 1e-8
                )

                unstable = findall(==(Int8(0)), p.phases[layer])
                @test length(w.relu_binaries[layer]) == length(unstable)
                stable_ok = true
                for neuron in eachindex(p.phases[layer])
                    phase = p.phases[layer][neuron]
                    if phase == -1
                        stable_ok &= isapprox(act[neuron], 0.0; atol=1e-9)
                    elseif phase == 1
                        stable_ok &= isapprox(act[neuron], pre[neuron]; atol=1e-9)
                    end
                end
                @test stable_ok
                rows_ok = true
                for (k, neuron) in enumerate(unstable)
                    d = Float64(w.relu_binaries[layer][k])
                    lower = p.pre_lower[layer][neuron]
                    upper = p.pre_upper[layer][neuron]
                    rows_ok &= d == (pre[neuron] > 0.0 ? 1.0 : 0.0)
                    rows_ok &= act[neuron] >= pre[neuron] - 1e-9
                    rows_ok &= act[neuron] <= upper * d + 1e-9
                    rows_ok &= act[neuron] <= pre[neuron] - lower * (1.0 - d) + 1e-9
                end
                @test rows_ok
                current = act
            end
            output = dot(p.output_weights, current) + p.output_bias
            @test output ≈ w.output
            @test p.output_lower - 1e-8 <= output <= p.output_upper + 1e-8
            @test output >= p.property_threshold
            @test p.property_threshold > p.output_lower
            @test p.property_threshold <= p.attainable_upper
        end
    end

    @testset "infeasibility certificate" begin
        sample_max(p, rng, n) = maximum(
            begin
                x = [
                    if rand(rng) < 0.5
                        (rand(rng) < 0.5 ? p.input_lower[i] : p.input_upper[i])
                    else
                        p.input_lower[i] + rand(rng) * (p.input_upper[i] - p.input_lower[i])
                    end for i in 1:p.input_dim
                ]
                dot(p.output_weights, SyntheticLPs.nnv_forward(p.weights, p.biases, x)[2][end]) + p.output_bias
            end for _ in 1:n
        )

        for target in (50, 200, 1200, 3000), seed in 0:3
            _, p = generate_problem(nnv, target, infeasible, seed)
            cert = p.infeasibility_certificate
            @test cert !== nothing
            @test p.feasible_witness === nothing

            bound =
                cert.input_constant + sum(
                    max(
                        cert.input_coefficients[i] * p.input_lower[i],
                        cert.input_coefficients[i] * p.input_upper[i],
                    ) for i in 1:p.input_dim
                )
            @test bound ≈ cert.attainable_upper
            @test cert.attainable_upper ≈ p.attainable_upper

            # Replay the backward substitution from the stored per-neuron lines;
            # every substituted line is a valid relaxation of its ReLU in the
            # direction that matters.
            lambda = copy(p.output_weights)
            constant = p.output_bias
            lines_ok = true
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
                        lines_ok &= lambda[j] >= 0.0 ? line >= relu - 1e-7 : line <= relu + 1e-7
                    end
                    mu[j] = lambda[j] * slope
                    constant += lambda[j] * intercept
                end
                constant += dot(mu, p.biases[layer])
                lambda = transpose(p.weights[layer]) * mu
            end
            @test lines_ok
            @test lambda ≈ cert.input_coefficients
            @test constant ≈ cert.input_constant

            u, v = cert.mirrored_pair
            @test cert.mirrored_pair == p.mirrored_pair
            @test cert.mirrored_gap ≈ min(
                p.output_weights[u] * p.pre_upper[end][u], p.output_weights[v] * p.pre_upper[end][v]
            )
            @test cert.mirrored_gap > 0.0
            @test cert.attainable_upper <= cert.interval_upper - cert.mirrored_gap + 1e-7

            # The presolve-hardness sandwich.
            @test cert.interval_upper ≈ p.interval_output_upper
            @test cert.declared_upper ≈ p.output_upper
            @test p.attainable_upper < p.property_threshold < p.output_upper
            @test p.property_threshold < p.interval_output_upper
            @test p.property_threshold > p.output_lower

            rng = MersenneTwister(1234 + seed)
            @test sample_max(p, rng, 200) < p.attainable_upper
        end
    end

    @testset "reproducibility and RNG isolation" begin
        deep_equal(a, b) =
            isequal(a, b) || (
                typeof(a) === typeof(b) &&
                !isempty(fieldnames(typeof(a))) &&
                all(deep_equal(getfield(a, f), getfield(b, f)) for f in fieldnames(typeof(a)))
            )
        for status in (feasible, infeasible, unknown), target in (220, 3000)
            Random.seed!(987)
            _, p1 = generate_problem(nnv, target, status, 42)
            Random.seed!(12345)
            _, p2 = generate_problem(nnv, target, status, 42)
            @test all(deep_equal(getfield(p1, f), getfield(p2, f)) for f in fieldnames(typeof(p1)))
            m1 = SyntheticLPs.build_model(p1)
            m2 = SyntheticLPs.build_model(p2)
            @test num_variables(m1) == num_variables(m2)
            @test sprint(print, m1) == sprint(print, m2)
        end
    end

    @testset "solver contracts" begin
        if HAS_HIGHS
            # The contract holds on the LP relaxation (the package default):
            # infeasibility survives relaxing the ReLU binaries.
            for target in (60, 200, 800, 3000), status in (feasible, infeasible), seed in 0:3
                m, _ = generate_problem(nnv, target, status, seed)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                expected = status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE
                @test termination_status(m) == expected
            end

            # ... and on the unrelaxed MILP, where the feasible witness must
            # also be integrally realisable.
            for target in (60, 200), status in (feasible, infeasible), seed in 0:2
                m, _ = generate_problem(nnv, target, status, seed; relax_integer=false)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                set_attribute(m, "time_limit", 60.0)
                optimize!(m)
                expected = status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE
                @test termination_status(m) == expected
            end

            # An infeasible instance needs real simplex work.
            for target in (500, 3000), seed in 0:1
                m, _ = generate_problem(nnv, target, infeasible, seed)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                set_attribute(m, "presolve", "off")
                optimize!(m)
                @test termination_status(m) == MOI.INFEASIBLE
                @test MOI.get(m, MOI.SimplexIterations()) >= 50
            end

            # Presolve keeps most of the model: stable neurons are not emitted
            # as substitutable rows any more (the old encoding kept ~52%).
            for seed in 0:1
                m, _ = generate_problem(nnv, 4000, feasible, seed)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                MOI.Utilities.attach_optimizer(m)
                highs = unsafe_backend(m)
                HiGHS.Highs_presolve(highs)
                kept = HiGHS.Highs_getPresolvedNumCol(highs) / num_variables(m)
                @test kept >= 0.8
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
end
