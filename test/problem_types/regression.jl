# Focused quality contracts for the regression category: registry wiring, exact
# sizing, data profiles, planted witnesses and infeasibility certificates
# (checked arithmetically without a solver), reproducibility, bounded nonzeros at
# scale, and HiGHS-backed feasibility contracts for all five variants.
using SparseArrays

const REGRESSION_VARIANTS = (:lad, :quantile, :chebyshev, :basis_pursuit, :l1_svm)

"""Count affine-row nonzeros of a JuMP model (variable bounds excluded)."""
function _regression_test_nnz(model)
    total = 0
    for (F, S) in list_of_constraint_types(model)
        F <: AffExpr || continue
        for c in all_constraints(model, F, S)
            total += length(constraint_object(c).func.terms)
        end
    end
    return total
end

"""Fields-equal comparison of two generator structs (recursing into plain structs)."""
function _regression_same(a, b)
    typeof(a) == typeof(b) || return false
    isstructtype(typeof(a)) && !(a isa AbstractArray) && !(a isa Number) || return a == b
    return all(_regression_same(getfield(a, f), getfield(b, f)) for f in fieldnames(typeof(a)))
end
_regression_same(a::AbstractArray, b::AbstractArray) = a == b

@testset "Regression Registry and Sizing" begin
    @test Set(list_variants(:regression)) == Set(REGRESSION_VARIANTS)
    @test ProblemVariant(:regression) == ProblemVariant(:regression, :lad)
    for v in REGRESSION_VARIANTS
        @test !isempty(problem_info(:regression, v)[:description])
    end

    for target in (40, 200, 1000, 4000)
        # LAD: 1 + continuous + fixed effects + samples, exact.
        model, prob = generate_problem(:regression, target, feasible, 3; variant=:lad)
        n_gamma = sum(l - 1 for l in prob.levels; init=0)
        @test num_variables(model) == 1 + prob.n_continuous + n_gamma + prob.n_samples == target
        @test num_constraints(model; count_variable_in_set_constraints=false) ==
            2 * prob.n_samples + 1

        # Quantile: 1 + demographic + 2 codes + 2 samples, exact.
        model, prob = generate_problem(:regression, target, feasible, 3; variant=:quantile)
        @test num_variables(model) ==
            1 + prob.n_demographic + 2 * prob.n_codes + 2 * prob.n_samples ==
            target
        @test num_constraints(model; count_variable_in_set_constraints=false) ==
            prob.n_samples + length(prob.band_lower)

        # Chebyshev: spline coefficients + t, within 5%.
        model, prob = generate_problem(:regression, target, feasible, 3; variant=:chebyshev)
        p = (prob.nx + prob.degree) * (prob.ny + prob.degree)
        @test num_variables(model) == p + 1
        @test abs(num_variables(model) - target) <= max(3, 0.05 * target)
        @test num_constraints(model; count_variable_in_set_constraints=false) == 2 * prob.n_samples

        # Basis pursuit: 2·features, exact for even targets.
        model, prob = generate_problem(:regression, target, feasible, 3; variant=:basis_pursuit)
        @test num_variables(model) == 2 * prob.n_features == target

        # 1-norm SVM: 2·terms + bias + documents, exact.
        model, prob = generate_problem(:regression, target, feasible, 3; variant=:l1_svm)
        @test num_variables(model) == 2 * prob.n_terms + 1 + prob.n_documents == target
        @test num_constraints(model; count_variable_in_set_constraints=false) ==
            prob.n_documents + 1
    end

    # Tiny targets still build every status.
    for v in REGRESSION_VARIANTS, target in (1, 5, 12), status in (feasible, infeasible, unknown)
        model, _ = generate_problem(:regression, target, status, 1; variant=v)
        @test num_variables(model) > 0
    end

    # 100k-variable requests: exact sizing, bounded nonzeros, seconds to build.
    for v in (:lad, :chebyshev, :l1_svm, :quantile)
        elapsed = @elapsed model, _ = generate_problem(
            :regression, 100_000, infeasible, 0; variant=v
        )
        @test abs(num_variables(model) - 100_000) <= 500
        @test _regression_test_nnz(model) <= 8_000_000
        @test elapsed < 60
    end
end

@testset "Regression LAD Data and Contracts" begin
    for seed in 1:4
        _, prob = generate_problem(:regression, 600, feasible, seed; variant=:lad)
        offsets = SyntheticLPs._lad_gamma_offsets(prob.levels)
        w = prob.feasible_witness
        @test w isa SyntheticLPs.LADWitness
        @test prob.infeasibility_certificate === nothing
        fitted = [
            SyntheticLPs._lad_fitted(
                prob.X, prob.level_codes, offsets, w.intercept, w.beta, w.gamma, i
            ) for i in 1:prob.n_samples
        ]
        @test sum(abs.(prob.y .- fitted)) ≈ w.loss
        @test w.loss * 1.04 < prob.loss_budget
        @test all(>(0), prob.weights)
        # Every non-reference level is observed; replicates repeat their design.
        for f in eachindex(prob.levels)
            @test Set(prob.level_codes[f, :]) == Set(1:prob.levels[f])
        end
        for (a, b) in prob.replicate_pairs
            @test prob.X[:, a] == prob.X[:, b]
            @test prob.level_codes[:, a] == prob.level_codes[:, b]
        end
        @test !isempty(prob.outliers)

        _, bad = generate_problem(:regression, 600, infeasible, seed; variant=:lad)
        cert = bad.infeasibility_certificate
        @test cert isa SyntheticLPs.LADCertificate
        @test bad.feasible_witness === nothing
        used = Int[]
        for (a, b) in cert.pairs
            @test bad.X[:, a] == bad.X[:, b] && bad.level_codes[:, a] == bad.level_codes[:, b]
            @test bad.y[a] >= bad.y[b]
            append!(used, (a, b))
        end
        @test allunique(used)
        @test sum(bad.y[a] - bad.y[b] for (a, b) in cert.pairs) ≈ cert.lower_bound
        @test cert.budget == bad.loss_budget
        @test cert.budget <= 0.9 * cert.lower_bound + 1e-12

        _, unk = generate_problem(:regression, 600, unknown, seed; variant=:lad)
        @test unk.feasible_witness === nothing && unk.infeasibility_certificate === nothing
    end
end

@testset "Regression Quantile Data and Contracts" begin
    predict(prob, w, r) = SyntheticLPs._quantile_predict(
        w.intercept,
        w.demographic,
        w.code,
        prob.reference_codes[r],
        prob.reference_values[r],
        view(prob.reference_Z, r, :),
    )
    for seed in 1:4
        _, prob = generate_problem(:regression, 800, feasible, seed; variant=:quantile)
        @test prob.tau in (0.1, 0.25, 0.5, 0.75, 0.9)
        @test prob.penalty > 0
        # Minimum code frequency: every code appears in at least five samples.
        counts = zeros(Int, prob.n_codes)
        for codes in prob.codes, j in codes
            counts[j] += 1
        end
        @test minimum(counts) >= min(5, prob.n_samples)
        # Cohort profiles are exact averages of their members.
        for (c, members) in enumerate(prob.cohorts)
            row = prob.cohort_rows[c]
            dense = zeros(prob.n_codes)
            for r in members, j in prob.reference_codes[r]
                dense[j] += 1 / length(members)
            end
            @test dense[prob.reference_codes[row]] ≈ prob.reference_values[row]
            @test count(!iszero, dense) == length(prob.reference_codes[row])
            @test vec(sum(prob.reference_Z[members, :]; dims=1)) ./ length(members) ≈
                prob.reference_Z[row, :]
        end
        w = prob.feasible_witness
        @test w isa SyntheticLPs.QuantileWitness
        for r in eachindex(prob.band_lower)
            p = predict(prob, w, r)
            half = (prob.band_upper[r] - prob.band_lower[r]) / 2
            @test prob.band_lower[r] + 0.5 * half - 1e-9 <=
                p <=
                prob.band_upper[r] - 0.5 * half + 1e-9
        end

        _, bad = generate_problem(:regression, 800, infeasible, seed; variant=:quantile)
        cert = bad.infeasibility_certificate
        @test cert isa SyntheticLPs.QuantileCertificate
        @test cert.cohort in bad.cohort_rows
        @test cert.members == bad.cohorts[findfirst(==(cert.cohort), bad.cohort_rows)]
        mean_upper = sum(bad.band_upper[cert.members]) / length(cert.members)
        @test bad.band_lower[cert.cohort] - mean_upper ≈ cert.gap
        @test cert.gap > 0
    end
end

@testset "Regression Chebyshev Data and Contracts" begin
    for seed in 1:4
        _, prob = generate_problem(:regression, 500, feasible, seed; variant=:chebyshev)
        @test prob.degree in (1, 2, 3)
        @test size(prob.basis_cols) == ((prob.degree + 1)^2, prob.n_samples)
        @test all(0 .<= prob.points .<= 1)
        # Local partition of unity (up to the dropped tails).
        sums = vec(sum(prob.basis_vals; dims=1))
        @test all(1 - (prob.degree + 1)^2 * 1e-3 - 1e-12 .<= sums .<= 1 + 1e-12)
        w = prob.feasible_witness
        @test w isa SyntheticLPs.ChebyshevWitness
        residual = maximum(
            prob.weights[i] *
            abs(prob.y[i] - dot(prob.basis_vals[:, i], w.coefficients[prob.basis_cols[:, i]])) for
            i in 1:prob.n_samples
        )
        @test residual ≈ w.max_weighted_residual
        @test 1.09 * residual <= prob.error_cap
        @test maximum(abs, w.coefficients) <= prob.coefficient_bound

        _, bad = generate_problem(:regression, 500, infeasible, seed; variant=:chebyshev)
        cert = bad.infeasibility_certificate
        @test cert isa SyntheticLPs.ChebyshevCertificate
        @test length(cert.points) == (bad.degree + 1)^2 + 1
        # Multipliers annihilate the local basis rows.
        acc = Dict{Int, Float64}()
        for (k, i) in enumerate(cert.points), q in axes(bad.basis_cols, 1)
            acc[bad.basis_cols[q, i]] =
                get(acc, bad.basis_cols[q, i], 0.0) + cert.multipliers[k] * bad.basis_vals[q, i]
        end
        @test maximum(abs, values(acc)) <= 1e-10
        @test sum(cert.multipliers .* bad.y[cert.points]) ≈ cert.combined_residual
        @test sum(abs.(cert.multipliers) ./ bad.weights[cert.points]) ≈ cert.weighted_l1
        @test abs(cert.combined_residual) >= 1.24 * bad.error_cap * cert.weighted_l1
    end
end

@testset "Regression L1 SVM Data and Contracts" begin
    for seed in 1:4
        _, prob = generate_problem(:regression, 800, feasible, seed; variant=:l1_svm)
        @test Set(prob.labels) == Set((-1.0, 1.0))
        @test all(c -> isapprox(norm(prob.Xt[:, c]), 1.0; atol=1e-10), 1:prob.n_documents)
        @test minimum(vec(sum(prob.Xt .!= 0; dims=2))) >= 3
        w = prob.feasible_witness
        @test w isa SyntheticLPs.L1SVMWitness
        scores = transpose(prob.Xt) * w.w .+ w.bias
        @test sum(max.(0.0, 1 .- prob.labels .* scores)) ≈ w.hinge
        @test w.hinge * 1.04 < prob.hinge_budget

        _, bad = generate_problem(:regression, 800, infeasible, seed; variant=:l1_svm)
        cert = bad.infeasibility_certificate
        @test cert isa SyntheticLPs.L1SVMCertificate
        used = Int[]
        for (a, b) in cert.pairs
            @test bad.Xt[:, a] == bad.Xt[:, b]
            @test bad.labels[a] == 1.0 && bad.labels[b] == -1.0
            append!(used, (a, b))
        end
        @test allunique(used)
        @test cert.lower_bound == 2 * length(cert.pairs)
        @test bad.hinge_budget == cert.budget <= 0.9 * cert.lower_bound
    end
end

@testset "Regression Basis Pursuit" begin
    profiles = SyntheticLPs.BASIS_PURSUIT_PROFILES
    @test profiles == (:gaussian, :correlated_columns, :sparse_measurements)
    profile_seeds = Dict(profile => Int[] for profile in profiles)
    for seed in 1:100
        _, prob = generate_problem("regression/basis_pursuit", 150, feasible, seed)
        length(profile_seeds[prob.profile]) < 3 && push!(profile_seeds[prob.profile], seed)
    end
    @test all(length(profile_seeds[profile]) == 3 for profile in profiles)

    check_status_data = function (prob)
        @test (prob.certificate !== nothing) == (prob.resolved_status == infeasible)
        A = prob.A
        @test all(>(0), diff(A.colptr))                     # every feature measured
        @test length(unique(rowvals(A))) == prob.n_measurements
        if prob.resolved_status == feasible
            @test A * prob.source_signal ≈ prob.b
        else
            cert = prob.certificate
            @test cert isa SyntheticLPs.BasisPursuitCertificate
            @test allunique(cert.rows)
            @test length(cert.rows) >= min(3, prob.n_measurements)
            combo = transpose(A[cert.rows, :]) * cert.multipliers
            @test norm(combo, Inf) <= 1e-10
            @test sum(cert.multipliers .* prob.b[cert.rows]) ≈ cert.rhs_gap
            @test abs(cert.rhs_gap) >= 0.4
            @test !(A * prob.source_signal ≈ prob.b)
        end
    end

    for target in (1, 2, 3, 4, 5, 50, 501, 2000)
        model, prob = generate_problem("regression/basis_pursuit", target, feasible, 17)
        @test num_variables(model) == 2 * max(1, cld(max(target, 1), 2)) == 2 * prob.n_features
        @test size(prob.A) == (prob.n_measurements, prob.n_features)
    end

    # Column nonzeros respect the budget; large instances stay sparse.
    _, big = generate_problem("regression/basis_pursuit", 20_000, feasible, 2)
    @test nnz(big.A) <= SyntheticLPs.BASIS_PURSUIT_NNZ_BUDGET + big.n_measurements
    @test maximum(diff(big.A.colptr)) <=
        SyntheticLPs._basis_pursuit_column_nnz(big.n_measurements, big.n_features) + 2

    for profile in profiles, seed in profile_seeds[profile]
        _, prob = generate_problem("regression/basis_pursuit", 150, feasible, seed)
        @test issorted(prob.support) && allunique(prob.support)
        @test findall(!iszero, prob.source_signal) == prob.support
        @test all(>(0.0), prob.weights)
        if profile == :gaussian
            # Small instances are dense and row-whitened.
            Ad = Matrix(prob.A)
            @test norm(Ad * transpose(Ad) - I, Inf) <= 1.0e-10
        elseif profile == :correlated_columns
            Ad = Matrix(prob.A)
            normalized = Ad ./ sqrt.(sum(abs2, Ad; dims=1))
            gram = transpose(normalized) * normalized
            @test maximum(abs.(gram - I)) >= 0.985
        else
            @test nnz(prob.A) / length(prob.A) <= 0.2
        end
    end

    for seed in 1:12, target in (20, 100)
        _, f = generate_problem("regression/basis_pursuit", target, feasible, seed)
        check_status_data(f)
        _, g = generate_problem("regression/basis_pursuit", target, infeasible, seed)
        check_status_data(g)
    end
    unknown_statuses = Set{FeasibilityStatus}()
    for seed in 1:40
        _, prob = generate_problem("regression/basis_pursuit", 120, unknown, seed)
        push!(unknown_statuses, prob.resolved_status)
        check_status_data(prob)
    end
    @test unknown_statuses == Set((feasible, infeasible))

    # Complete JuMP formulation for one instance.
    model, prob = generate_problem("regression/basis_pursuit", 80, feasible, 4)
    @test objective_sense(model) == MOI.MIN_SENSE
    @test num_constraints(model, AffExpr, MOI.EqualTo{Float64}) == prob.n_measurements
    for i in 1:prob.n_measurements
        row = model[:measurements][i]
        @test normalized_rhs(row) == prob.b[i]
        for j in 1:prob.n_features
            @test normalized_coefficient(row, model[:x_pos][j]) == prob.A[i, j]
            @test normalized_coefficient(row, model[:x_neg][j]) == -prob.A[i, j]
        end
    end
end

@testset "Regression Reproducibility" begin
    for v in REGRESSION_VARIANTS, status in (feasible, infeasible, unknown)
        m1, p1 = generate_problem(:regression, 300, status, 11; variant=v)
        m2, p2 = generate_problem(:regression, 300, status, 11; variant=v)
        @test _regression_same(p1, p2)
        mktempdir() do dir
            a = joinpath(dir, "a.mps")
            b = joinpath(dir, "b.mps")
            write_to_file(m1, a)
            write_to_file(SyntheticLPs.build_model(p2), b)
            @test read(a, String) == read(b, String)
        end
    end
    Random.seed!(77)
    expected = rand(3)
    Random.seed!(77)
    for v in REGRESSION_VARIANTS
        generate_problem(:regression, 200, unknown, 5; variant=v)
    end
    @test rand(3) == expected
end

@testset "Regression Feasibility Contracts" begin
    if HAS_HIGHS
        solve_status = function (model)
            set_optimizer(model, HiGHS.Optimizer)
            set_silent(model)
            set_time_limit_sec(model, 60.0)
            optimize!(model)
            return termination_status(model)
        end
        targets = Dict(
            :lad => 400, :quantile => 400, :chebyshev => 250, :basis_pursuit => 200, :l1_svm => 400
        )
        for v in REGRESSION_VARIANTS, seed in 1:3
            # Passing the optimizer runs the package-level contract check.
            model, _ = generate_problem(
                :regression, targets[v], feasible, seed; variant=v, optimizer=HiGHS.Optimizer
            )
            @test solve_status(model) == MOI.OPTIMAL
            model, _ = generate_problem(
                :regression, targets[v], infeasible, seed; variant=v, optimizer=HiGHS.Optimizer
            )
            @test solve_status(model) in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)
        end
        # `unknown` is a natural two-sided draw for the data-driven variants.
        for v in (:lad, :quantile, :chebyshev, :l1_svm)
            outcomes = Set{MOI.TerminationStatusCode}()
            for seed in 1:12
                model, _ = generate_problem(:regression, targets[v], unknown, seed; variant=v)
                push!(outcomes, solve_status(model))
            end
            @test MOI.OPTIMAL in outcomes
            @test MOI.INFEASIBLE in outcomes
        end
    end
end
