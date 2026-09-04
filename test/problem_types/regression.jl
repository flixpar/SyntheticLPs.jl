# Focused quality contracts for the regression category: the basis_pursuit
# variant's registry wiring, sizing, conditioning profiles, planted sparse
# witness / infeasibility certificate, and its HiGHS feasibility contracts.
@testset "Regression Basis Pursuit" begin
    @test :basis_pursuit in list_variants(:regression)
    info = problem_info(:regression, :basis_pursuit)
    @test occursin("basis-pursuit", lowercase(info[:description]))
    @test ProblemVariant("regression/basis_pursuit") == ProblemVariant(:regression, :basis_pursuit)

    profiles = (:gaussian_well_conditioned, :correlated_columns, :sparse_measurements)
    profile_seeds = Dict(profile => Int[] for profile in profiles)
    for seed in 1:100
        _, prob = generate_problem("regression/basis_pursuit", 150, feasible, seed)
        length(profile_seeds[prob.profile]) < 3 && push!(profile_seeds[prob.profile], seed)
    end
    @test all(length(profile_seeds[profile]) == 3 for profile in profiles)

    check_status_data = function (prob)
        @test (prob.certificate !== nothing) == (prob.resolved_status == infeasible)
        @test all(any(!iszero, @view prob.A[i, :]) for i in 1:prob.n_measurements)
        @test all(any(!iszero, @view prob.A[:, j]) for j in 1:prob.n_features)
        if prob.resolved_status == feasible
            @test prob.certificate === nothing
            @test prob.A * prob.source_signal ≈ prob.b
        else
            certificate = prob.certificate
            @test certificate isa SyntheticLPs.BasisPursuitCertificate
            if certificate !== nothing
                r1, r2 = certificate.rows
                @test 1 <= r1 <= prob.n_measurements
                @test 1 <= r2 <= prob.n_measurements
                @test r1 != r2
                @test prob.A[r2, :] == certificate.multiplier .* prob.A[r1, :]
                @test prob.b[r2] ≈ certificate.multiplier * prob.b[r1] + certificate.rhs_gap
                @test !iszero(certificate.rhs_gap)
                @test !(prob.A * prob.source_signal ≈ prob.b)
            end
        end
    end

    # Positive/negative splitting makes the count intrinsically even: even
    # targets are exact, odd targets round up one, and two is the minimum.
    for target in (1, 2, 3, 4, 5, 50, 501, 2000)
        model, prob = generate_problem("regression/basis_pursuit", target, feasible, 17)
        expected = 2 * max(1, cld(max(target, 1), 2))
        @test num_variables(model) == expected == 2 * prob.n_features
        @test size(prob.A) == (prob.n_measurements, prob.n_features)
        @test length(prob.b) == prob.n_measurements
        @test length(prob.weights) == prob.n_features
        @test length(prob.source_signal) == prob.n_features
    end

    # Deterministic data, repeated builds, and MPS export for every profile
    # under both resolved statuses.
    mktempdir() do tmp
        for profile in profiles, status in (feasible, infeasible)
            seed = first(profile_seeds[profile])
            model1, prob1 = generate_problem("regression/basis_pursuit", 150, status, seed)
            model2, prob2 = generate_problem("regression/basis_pursuit", 150, status, seed)
            @test prob1.profile == prob2.profile == profile
            for field in fieldnames(typeof(prob1))
                @test getfield(prob1, field) == getfield(prob2, field)
            end
            rebuilt = SyntheticLPs.build_model(prob1)
            @test num_variables(model1) == num_variables(model2) == num_variables(rebuilt)
            @test num_constraints(model1; count_variable_in_set_constraints=true) ==
                num_constraints(model2; count_variable_in_set_constraints=true) ==
                num_constraints(rebuilt; count_variable_in_set_constraints=true)

            prefix = "$(profile)_$(status)"
            paths = [joinpath(tmp, "$(prefix)_$copy.mps") for copy in 1:3]
            write_to_file(model1, paths[1])
            write_to_file(model2, paths[2])
            write_to_file(rebuilt, paths[3])
            @test read(paths[1], String) == read(paths[2], String) == read(paths[3], String)
        end
    end

    # The constructor owns a local RNG and does not perturb Random.default_rng().
    Random.seed!(8801)
    expected_draws = rand(4)
    Random.seed!(8801)
    SyntheticLPs.BasisPursuitProblem(100, feasible, 9)
    @test rand(4) == expected_draws

    # A deterministic seed sample covers both natural unknown outcomes, and
    # each outcome carries exactly its matching witness/certificate data.
    unknown_statuses = Set{FeasibilityStatus}()
    varied = Any[]
    for seed in 1:60
        _, feasible_prob = generate_problem("regression/basis_pursuit", 120, feasible, seed)
        _, unknown_prob = generate_problem("regression/basis_pursuit", 120, unknown, seed)
        push!(unknown_statuses, unknown_prob.resolved_status)
        check_status_data(unknown_prob)
        seed <= 12 && push!(varied, feasible_prob)
    end
    @test unknown_statuses == Set((feasible, infeasible))
    @test length(unique(p.profile for p in varied)) > 1
    @test length(unique(Tuple(p.support) for p in varied)) > 1
    @test any(p.A != varied[1].A for p in varied[2:end])

    # Profile statistics are checked on multiple seeds, including a
    # large-instance regression against coherence decay.
    for profile in profiles, seed in profile_seeds[profile]
        _, prob = generate_problem("regression/basis_pursuit", 150, feasible, seed)
        @test prob.resolved_status == feasible
        @test issorted(prob.support)
        @test allunique(prob.support)
        @test all(1 <= j <= prob.n_features for j in prob.support)
        @test findall(!iszero, prob.source_signal) == prob.support
        @test all(>(0.0), prob.weights)
        @test norm(prob.A * prob.source_signal - prob.b, Inf) <= 1.0e-10
        @test norm(prob.b, Inf) > 1.0e-8
        @test prob.certificate === nothing

        if profile == :gaussian_well_conditioned
            identity_rows = Matrix{Float64}(I, prob.n_measurements, prob.n_measurements)
            @test norm(prob.A * transpose(prob.A) - identity_rows, Inf) <= 1.0e-10
        elseif profile == :correlated_columns
            normalized = prob.A ./ sqrt.(sum(abs2, prob.A; dims=1))
            gram = transpose(normalized) * normalized
            identity_columns = Matrix{Float64}(I, prob.n_features, prob.n_features)
            @test maximum(abs.(gram - identity_columns)) >= 0.985
        else
            density = count(!iszero, prob.A) / length(prob.A)
            @test density <= 0.2
            @test all(any(!iszero, @view prob.A[i, :]) for i in 1:prob.n_measurements)
            @test all(any(!iszero, @view prob.A[:, j]) for j in 1:prob.n_features)
        end
    end

    for seed in profile_seeds[:correlated_columns]
        _, prob = generate_problem("regression/basis_pursuit", 2000, feasible, seed)
        normalized = prob.A ./ sqrt.(sum(abs2, prob.A; dims=1))
        sample_width = min(200, prob.n_features)
        sample = @view normalized[:, 1:sample_width]
        gram = transpose(sample) * sample
        identity_columns = Matrix{Float64}(I, sample_width, sample_width)
        @test maximum(abs.(gram - identity_columns)) >= 0.985
    end

    # Feasible instances retain their source signal as an exact witness;
    # infeasible instances carry only the inspectable algebraic certificate.
    for seed in 1:12
        _, feasible_prob = generate_problem("regression/basis_pursuit", 100, feasible, seed)
        @test feasible_prob.resolved_status == feasible
        check_status_data(feasible_prob)

        _, infeasible_prob = generate_problem("regression/basis_pursuit", 100, infeasible, seed)
        @test infeasible_prob.resolved_status == infeasible
        check_status_data(infeasible_prob)
    end

    # Certificate injection must not erase sparse columns whose only
    # nonzero sat in the replaced row. Target 20 has measurement width 1.
    sparse_infeasible = 0
    for seed in 0:199
        _, prob = generate_problem("regression/basis_pursuit", 20, infeasible, seed)
        prob.profile == :sparse_measurements || continue
        sparse_infeasible += 1
        check_status_data(prob)
    end
    @test sparse_infeasible >= 20

    # Every profile also constructs correctly at the one-feature minimum,
    # under both statuses. Gaussian rows cannot both be orthonormal in this
    # 2×1 geometry, so its feasible matrix is normalized as one column.
    tiny_profile_seeds = Dict{Symbol, Int}()
    for seed in 1:60
        _, prob = generate_problem("regression/basis_pursuit", 1, feasible, seed)
        get!(tiny_profile_seeds, prob.profile, seed)
    end
    @test Set(keys(tiny_profile_seeds)) == Set(profiles)
    for profile in profiles, target in (1, 2, 3), status in (feasible, infeasible)
        model, prob = generate_problem(
            "regression/basis_pursuit", target, status, tiny_profile_seeds[profile]
        )
        @test prob.profile == profile
        @test num_variables(model) == (target <= 2 ? 2 : 4)
        @test prob.n_measurements == 2
        check_status_data(prob)
    end
    _, tiny_gaussian = generate_problem(
        "regression/basis_pursuit", 1, feasible, tiny_profile_seeds[:gaussian_well_conditioned]
    )
    @test size(tiny_gaussian.A) == (2, 1)
    @test norm(tiny_gaussian.A) ≈ 1.0

    # Coherent and sparse profiles vary numerically between same-profile
    # seeds, not merely through their profile labels.
    for profile in (:correlated_columns, :sparse_measurements)
        matrices = [
            last(generate_problem("regression/basis_pursuit", 150, feasible, seed)).A for
            seed in profile_seeds[profile]
        ]
        @test all(matrices[i] != matrices[j] for (i, j) in ((1, 2), (1, 3), (2, 3)))
    end

    # Assert the complete JuMP formulation, not only variable domains/counts.
    domain_model, domain_prob = generate_problem("regression/basis_pursuit", 80, feasible, 4)
    @test objective_sense(domain_model) == MOI.MIN_SENSE
    @test num_constraints(domain_model, AffExpr, MOI.EqualTo{Float64}) == domain_prob.n_measurements
    for variable in all_variables(domain_model)
        @test !is_binary(variable)
        @test !is_integer(variable)
        @test has_lower_bound(variable)
        @test lower_bound(variable) == 0.0
        @test !has_upper_bound(variable)
    end
    objective = objective_function(domain_model)
    for j in 1:domain_prob.n_features
        @test coefficient(objective, domain_model[:x_pos][j]) == domain_prob.weights[j]
        @test coefficient(objective, domain_model[:x_neg][j]) == domain_prob.weights[j]
    end
    for i in 1:domain_prob.n_measurements
        row = domain_model[:measurements][i]
        @test normalized_rhs(row) == domain_prob.b[i]
        for j in 1:domain_prob.n_features
            @test normalized_coefficient(row, domain_model[:x_pos][j]) == domain_prob.A[i, j]
            @test normalized_coefficient(row, domain_model[:x_neg][j]) == -domain_prob.A[i, j]
        end
    end
end

@testset "Basis Pursuit Feasibility Contracts" begin
    if HAS_HIGHS
        # Exercise three seeds per profile under both labels. Passing the
        # optimizer invokes the package-level contract check before returning
        # the pristine model.
        profiles = (:gaussian_well_conditioned, :correlated_columns, :sparse_measurements)
        profile_seeds = Dict(profile => Int[] for profile in profiles)
        for seed in 1:100
            _, prob = generate_problem("regression/basis_pursuit", 120, feasible, seed)
            length(profile_seeds[prob.profile]) < 3 && push!(profile_seeds[prob.profile], seed)
        end
        @test all(length(profile_seeds[profile]) == 3 for profile in profiles)

        for profile in profiles, seed in profile_seeds[profile]
            feasible_model, feasible_prob = generate_problem(
                "regression/basis_pursuit", 120, feasible, seed; optimizer=HiGHS.Optimizer
            )
            set_optimizer(feasible_model, HiGHS.Optimizer)
            set_silent(feasible_model)
            optimize!(feasible_model)
            @test termination_status(feasible_model) == MOI.OPTIMAL
            @test objective_value(feasible_model) > 1.0e-8
            @test feasible_prob.profile == profile
            @test feasible_prob.certificate === nothing
            @test norm(feasible_prob.A * feasible_prob.source_signal - feasible_prob.b, Inf) <=
                1.0e-10

            infeasible_model, infeasible_prob = generate_problem(
                "regression/basis_pursuit", 120, infeasible, seed; optimizer=HiGHS.Optimizer
            )
            set_optimizer(infeasible_model, HiGHS.Optimizer)
            set_silent(infeasible_model)
            optimize!(infeasible_model)
            @test termination_status(infeasible_model) in
                (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)
            @test infeasible_prob.profile == profile
            @test infeasible_prob.resolved_status == infeasible
            @test infeasible_prob.certificate !== nothing
        end

        # Unknown requests skip package-level verification, so solve both
        # resolved labels directly and require metadata and solver status to
        # agree. Two seeds per label avoid a single representative special case.
        unknown_seeds = Dict(feasible => Int[], infeasible => Int[])
        for seed in 1:100
            _, prob = generate_problem("regression/basis_pursuit", 120, unknown, seed)
            seeds = unknown_seeds[prob.resolved_status]
            length(seeds) < 2 && push!(seeds, seed)
        end
        @test all(length(unknown_seeds[status]) == 2 for status in (feasible, infeasible))
        for status in (feasible, infeasible), seed in unknown_seeds[status]
            model, prob = generate_problem("regression/basis_pursuit", 120, unknown, seed)
            @test prob.resolved_status == status
            @test (prob.certificate !== nothing) == (status == infeasible)
            if status == feasible
                @test prob.A * prob.source_signal ≈ prob.b
            else
                certificate = prob.certificate
                @test certificate !== nothing
                if certificate !== nothing
                    r1, r2 = certificate.rows
                    @test prob.A[r2, :] == certificate.multiplier .* prob.A[r1, :]
                    @test prob.b[r2] ≈ certificate.multiplier * prob.b[r1] + certificate.rhs_gap
                end
            end
            set_optimizer(model, HiGHS.Optimizer)
            set_silent(model)
            optimize!(model)
            expected = status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE
            @test termination_status(model) == expected
        end
    end
end
