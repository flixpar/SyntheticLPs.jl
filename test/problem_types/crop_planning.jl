@testset "Crop planning quality contracts" begin
    ref = ProblemVariant(:crop_planning, :standard)

    validate_witness = function (p)
        x = something(p.feasible_witness)
        atol = 1e-7 * max(1.0, p.total_land)
        @test length(x) == p.n_crops
        @test all(x .>= 0.0)
        @test sum(x) <= p.total_land + atol
        @test sum(p.water_requirements .* x) <= p.water_capacity + atol
        @test sum(p.labor_requirements .* x) <= p.labor_capacity + atol
        @test all(p.yields .* x .<= p.market_demand_tonnes .+ atol)
        @test all(x .+ atol .>= p.min_area_per_crop)
        for requirement in p.diversity_requirements
            @test sum(x[requirement.crop_indices]) + atol >= requirement.minimum_area
            @test all(p.crop_types[i] == requirement.crop_type for i in requirement.crop_indices)
        end
    end

    for target in (2, 25, 300, 2_000), seed in 0:5
        model, p = generate_problem(ref, target, feasible, seed)
        @test num_variables(model) == max(2, target)
        @test p.feasible_witness !== nothing
        @test p.infeasibility_certificate === nothing
        @test length(p.crop_names) == p.n_crops
        @test length(p.management_systems) == p.n_crops
        @test all(s in (:rainfed, :irrigated, :low_input, :intensive) for s in p.management_systems)
        @test all(p.market_demand_tonnes .> 0.0)
        @test !any(startswith(name, "Crop_") for name in p.crop_names)
        validate_witness(p)

        _, q = generate_problem(ref, target, infeasible, seed)
        @test q.feasible_witness === nothing
        certificate = q.infeasibility_certificate
        @test certificate isa SyntheticLPs.CropResourceCertificate
        @test certificate.forced_usage > certificate.capacity >= 0.0
        if certificate.resource == :water
            @test certificate.forced_usage ≈ sum(q.water_requirements .* q.min_area_per_crop)
            @test certificate.capacity == q.water_capacity
        else
            @test certificate.resource == :labor
            @test certificate.forced_usage ≈ sum(q.labor_requirements .* q.min_area_per_crop)
            @test certificate.capacity == q.labor_capacity
        end
    end

    # Unknown instances carry neither status claim as metadata.
    _, unknown_problem = generate_problem(ref, 120, unknown, 9)
    @test unknown_problem.feasible_witness === nothing
    @test unknown_problem.infeasibility_certificate === nothing

    # Construction is field-deterministic and isolated from the global RNG.
    _, p1 = generate_problem(ref, 180, feasible, 12345)
    _, p2 = generate_problem(ref, 180, feasible, 12345)
    @test all(isequal(getfield(p1, f), getfield(p2, f)) for f in fieldnames(typeof(p1)))
    Random.seed!(8172)
    expected = rand()
    Random.seed!(8172)
    generate_problem(ref, 180, feasible, 99)
    @test rand() == expected

    # The market rows consume harvested tonnes and all public row families are named.
    model, p = generate_problem(ref, 30, feasible, 4)
    x1 = variable_by_name(model, "x[1]")
    market1 = model[:market_demand][1]
    @test normalized_coefficient(market1, x1) == p.yields[1]
    @test normalized_rhs(market1) == p.market_demand_tonnes[1]
    @test constraint_by_name(model, "land_capacity") !== nothing
    @test constraint_by_name(model, "water_capacity") !== nothing
    @test constraint_by_name(model, "labor_capacity") !== nothing

    if HAS_HIGHS
        # Includes the former target=300, seed=4 feasible regression.
        for target in (30, 300, 1_200), seed in 0:5, status in (feasible, infeasible)
            model, _ = generate_problem(ref, target, status, seed)
            set_optimizer(model, HiGHS.Optimizer)
            set_silent(model)
            optimize!(model)
            expected_status = status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE
            @test termination_status(model) == expected_status
        end
    end
end

@testset "Crop planning feasibility contracts" begin
    if HAS_HIGHS
        # crop_planning/standard infeasible-request: previously ~17% came back
        # feasible (the "fallow-land" hole). With the optimizer guard every seed
        # must now solve INFEASIBLE.
        for s in 1:8
            m, _ = generate_problem(
                "crop_planning/standard", 120, infeasible, s; optimizer=HiGHS.Optimizer
            )
            set_optimizer(m, HiGHS.Optimizer)
            set_silent(m)
            optimize!(m)
            @test termination_status(m) in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)
        end
    end
end
