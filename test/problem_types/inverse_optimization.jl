using SparseArrays

const INVERSE_VARIANTS = (
    "inverse_optimization/classical_normalized",
    "inverse_optimization/linf",
    "inverse_optimization/market_clearing",
    "inverse_optimization/noisy_observations",
    "inverse_optimization/restricted_optimal_value",
    "inverse_optimization/shortest_path",
    "inverse_optimization/shortest_path_layered",
    "inverse_optimization/standard",
)

@testset "Inverse Optimization" begin
    @test :inverse_optimization in list_categories()
    @test list_variants(:inverse_optimization) ==
          [:classical_normalized, :linf, :market_clearing,
           :noisy_observations, :restricted_optimal_value, :shortest_path,
           :shortest_path_layered, :standard]
    @test problem_info(:inverse_optimization)[:default_variant] == :standard
    @test occursin(
        "inverse",
        lowercase(problem_info(:inverse_optimization)[:description]),
    )

    @testset "sizing and deterministic builds" begin
        for ref in INVERSE_VARIANTS, target in (12, 50, 100, 501, 2_000)
            model1, prob1 = generate_problem(ref, target, feasible, 19)
            model2, prob2 = generate_problem(ref, target, feasible, 19)
            @test num_variables(model1) == num_variables(model2)
            @test num_constraints(model1; count_variable_in_set_constraints=true) ==
                  num_constraints(model2; count_variable_in_set_constraints=true)
            @test sprint(print, model1) == sprint(print, model2)
            @test num_variables(model1) <= 50 ||
                  abs(num_variables(model1) - target) / target <= 0.05
            @test sprint(show, prob1) == sprint(show, prob2)
            rebuilt = SyntheticLPs.build_model(prob1)
            @test sprint(print, model1) == sprint(print, rebuilt)
        end

        # The resource formulations have closed-form variable counts.
        classical_model, classical = generate_problem(
            "inverse_optimization/classical_normalized", 301, feasible, 3,
        )
        @test num_variables(classical_model) ==
              3 * classical.n_activities + classical.n_resources
        panel_model, panel = generate_problem(
            "inverse_optimization/noisy_observations", 301, feasible, 3,
        )
        @test num_variables(panel_model) ==
              3 * panel.n_activities +
              panel.n_observations * panel.n_resources + panel.n_observations
        path_model, path_problem = generate_problem(
            "inverse_optimization/shortest_path", 301, feasible, 3,
        )
        @test num_variables(path_model) ==
              3 * path_problem.n_arcs +
              path_problem.n_observations * path_problem.n_nodes

        # The market planner must not silently fall back to its tiny model for
        # large requests. Every representable target from 31 through the public
        # cap has an exact analytical shape.
        for target in (31, 10_000, 50_000, 250_000)
            periods, units, ramp_pairs = SyntheticLPs._dispatch_dims(target)
            @test 3 * units + periods + periods * units + 2 * ramp_pairs == target
            @test 0 <= ramp_pairs <= units * (periods - 1)
        end
        cap_market = SyntheticLPs.InverseDispatchCostProblem(
            250_000, feasible, 3,
        )
        @test 3 * cap_market.num_units + cap_market.num_periods +
              cap_market.num_periods * cap_market.num_units +
              2 * length(cap_market.ramp_pairs) == 250_000

        for ref in INVERSE_VARIANTS
            @test_throws ArgumentError generate_problem(ref, 250_001, unknown, 1)
        end
    end

    @testset "classical inverse LP data and algebra" begin
        for seed in 0:20
            model, prob = generate_problem(
                "inverse_optimization/classical_normalized", 180, feasible, seed,
            )
            data = prob.data
            @test data.consumption isa SparseMatrixCSC{Float64,Int}
            @test size(data.consumption) ==
                  (prob.n_resources, prob.n_activities)
            @test all(diff(data.consumption.colptr) .> 0)
            @test all(vec(sum(data.consumption .> 0; dims=2)) .> 0)
            @test nnz(data.consumption) / length(data.consumption) <= 0.75
            @test all(prob.observed_decision .> 0)
            @test data.consumption * prob.observed_decision ≈ prob.capacity
            @test sum(data.true_cost) ≈ 1.0
            @test data.true_cost != data.prior_cost
            @test SyntheticLPs._classical_inverse_witness_is_valid(prob)

            j = 1
            rows, values = findnz(@view data.consumption[:, j])
            row = first(rows)
            coefficient = values[findfirst(==(row), rows)]
            @test normalized_coefficient(
                model[:stationarity][j], model[:shadow_price][row],
            ) ≈ coefficient
            @test normalized_coefficient(
                model[:stationarity][j], model[:inferred_cost][j],
            ) == -1.0
            @test objective_sense(model) == MOI.MIN_SENSE
        end
    end

    @testset "noisy observation panels" begin
        profiles = Set{Symbol}()
        for seed in 0:80
            model, prob = generate_problem(
                "inverse_optimization/noisy_observations", 240, feasible, seed,
            )
            push!(profiles, prob.profile)
            @test prob.profile in
                  (:routine, :heterogeneous, :outlier_contaminated)
            @test size(prob.observed_decisions) ==
                  (prob.n_observations, prob.n_activities)
            @test size(prob.capacities) ==
                  (prob.n_observations, prob.n_resources)
            @test all(prob.observed_decisions .> 0)
            @test all(prob.observed_decisions .< prob.optimal_decisions)
            @test SyntheticLPs._noisy_inverse_witness_is_valid(prob)
            @test all(prob.feasible_witness.gaps .> 0)
            @test prob.gap_scale >= 1.0
            @test 0.03 <= prob.regularization <= 0.18
            @test length(model[:suboptimality_gap]) == prob.n_observations
            @test length(model[:dual_feasibility]) ==
                  prob.n_observations * prob.n_activities
        end
        @test profiles == Set((:routine, :heterogeneous, :outlier_contaminated))

        # Contaminated panels contain a visibly lower-utilization observation.
        contaminated = nothing
        for seed in 0:100
            _, candidate = generate_problem(
                "inverse_optimization/noisy_observations", 240, feasible, seed,
            )
            candidate.profile == :outlier_contaminated || continue
            contaminated = candidate
            break
        end
        @test contaminated !== nothing
        if contaminated !== nothing
            utilization = contaminated.observed_decisions ./
                          contaminated.optimal_decisions
            row_means = vec(sum(utilization; dims=2)) ./
                        contaminated.n_activities
            @test minimum(row_means) <= 0.77
        end
    end

    @testset "spatial inverse shortest path" begin
        profiles = Set{Symbol}()
        for seed in 0:40
            model, prob = generate_problem(
                "inverse_optimization/shortest_path", 350, feasible, seed,
            )
            push!(profiles, prob.profile)
            @test prob.profile in
                  (:urban_grid, :regional_roads, :mixed_corridor)
            @test size(prob.coordinates_km) == (prob.n_nodes, 2)
            @test prob.n_arcs == length(prob.arcs) == 2 * (prob.n_arcs ÷ 2)
            @test all(arc.distance_km > 0 for arc in prob.arcs)
            @test all(prob.true_cost .> 0)
            @test all(prob.cost_lower .< prob.true_cost .< prob.cost_upper)
            @test SyntheticLPs._inverse_shortest_path_witness_is_valid(prob)
            @test !SyntheticLPs._inverse_paths_are_optimal(
                prob.n_nodes, prob.arcs, prob.observations, prob.prior_cost,
            )
            @test length(model[:dual_feasibility]) ==
                  prob.n_observations * prob.n_arcs

            for observation in prob.observations
                @test !isempty(observation.path_arcs)
                node = observation.source
                for edge in observation.path_arcs
                    @test prob.arcs[edge].tail == node
                    node = prob.arcs[edge].head
                end
                @test node == observation.destination
                distances, _ = SyntheticLPs._inverse_shortest_distances(
                    prob.n_nodes, prob.arcs, prob.true_cost, observation.source,
                )
                @test sum(prob.true_cost[e] for e in observation.path_arcs) ≈
                      distances[observation.destination]
            end
        end
        @test profiles == Set((:urban_grid, :regional_roads, :mixed_corridor))
    end

    @testset "generic, value, layered, and market witnesses" begin
        for variant in (:standard, :linf, :restricted_optimal_value), seed in 0:8
            model, prob = generate_problem(:inverse_optimization, 240, feasible, seed;
                                           variant=variant)
            witness = prob.feasible_witness
            @test witness !== nothing
            @test all(prob.forward_matrix * prob.reference_point .>=
                      prob.forward_rhs .- 1.0e-8)
            @test prob.forward_matrix' * witness.duals ≈ witness.cost
            @test all(prob.cost_lower .<= witness.cost .<= prob.cost_upper)
            @test dot(prob.forward_rhs, witness.duals) ≈
                  dot(prob.reference_point, witness.cost)
            if variant == :linf
                @test num_variables(model) == prob.num_cols + prob.num_rows + 1
                @test !haskey(object_dictionary(model), :dev_plus)
            elseif variant == :restricted_optimal_value
                @test dot(prob.reference_point, witness.cost) ≈ prob.target_value
            end
        end

        for seed in 0:12
            _, prob = generate_problem(:inverse_optimization, 240, feasible, seed;
                                       variant=:shortest_path_layered)
            witness = prob.feasible_witness
            @test witness !== nothing
            @test !SyntheticLPs._layered_prior_is_optimal(
                prob.num_nodes, prob.source, prob.sink, prob.tail, prob.head,
                prob.prior_cost, prob.path_arcs,
            )
            for e in 1:prob.num_arcs
                @test witness.potentials[prob.head[e]] -
                      witness.potentials[prob.tail[e]] <= witness.cost[e] + 1.0e-8
            end
            @test all(prob.cost_lower .<= witness.cost .<= prob.cost_upper)
        end

        for seed in 0:8
            _, prob = generate_problem(:inverse_optimization, 240, feasible, seed;
                                       variant=:market_clearing)
            witness = prob.feasible_witness
            @test witness !== nothing
            @test all(t -> sum(prob.observed_dispatch[t, :]) ≈ prob.demands[t],
                      1:prob.num_periods)
            dual_value = sum(prob.demands[t] * witness.energy_duals[t]
                             for t in 1:prob.num_periods) -
                         sum(prob.capacities[g] * witness.capacity_duals[t, g]
                             for t in 1:prob.num_periods, g in 1:prob.num_units)
            primal_value = sum(prob.observed_dispatch[t, g] * witness.cost[g]
                               for t in 1:prob.num_periods, g in 1:prob.num_units)
            @test dual_value ≈ primal_value
            @test SyntheticLPs._dispatch_prior_is_informative(
                prob.prior_cost, prob.capacities, prob.demands,
                prob.observed_dispatch,
            )
        end

        # Small fleets have broad merit-order cones, so unconditioned noisy
        # priors frequently yield a zero-adjustment inverse problem. Check the
        # sizes where that failure mode is most likely.
        for target in (12, 50, 100), seed in 0:20
            prob = SyntheticLPs.InverseDispatchCostProblem(
                target, feasible, seed,
            )
            @test SyntheticLPs._dispatch_prior_is_informative(
                prob.prior_cost, prob.capacities, prob.demands,
                prob.observed_dispatch,
            )
            @test all(prob.cost_lower .<= prob.feasible_witness.cost .<=
                      prob.cost_upper)
        end
    end

    @testset "native infeasibility certificates" begin
        _, classical = generate_problem(:inverse_optimization, 180, infeasible, 4;
                                        variant=:classical_normalized)
        @test SyntheticLPs._packing_interior_certificate_is_valid(
            classical.infeasibility_certificate,
        )
        @test classical.capacity ≈ classical.data.consumption *
              classical.observed_decision + classical.infeasibility_certificate.slacks

        _, panel = generate_problem(:inverse_optimization, 180, infeasible, 4;
                                    variant=:noisy_observations)
        @test SyntheticLPs._gap_tolerance_certificate_is_valid(
            panel.infeasibility_certificate,
        )
        @test panel.gap_tolerance == panel.infeasibility_certificate.tolerance

        _, spatial = generate_problem(:inverse_optimization, 180, infeasible, 4;
                                      variant=:shortest_path)
        @test SyntheticLPs._inverse_path_conflict_certificate_is_valid(spatial)

        _, layered = generate_problem(:inverse_optimization, 180, infeasible, 4;
                                      variant=:shortest_path_layered)
        @test SyntheticLPs._layered_shortcut_certificate_is_valid(layered)

        for variant in (:standard, :linf)
            _, prob = generate_problem(:inverse_optimization, 180, infeasible, 4;
                                       variant=variant)
            @test all(>(0.0), prob.infeasibility_certificate.slacks)
            @test prob.forward_matrix * prob.reference_point ≈
                  prob.forward_rhs + prob.infeasibility_certificate.slacks
        end

        _, value_problem = generate_problem(
            :inverse_optimization, 180, infeasible, 4;
            variant=:restricted_optimal_value,
        )
        value_certificate = value_problem.infeasibility_certificate
        @test value_certificate.target_value < value_certificate.value_floor ||
              value_certificate.target_value > value_certificate.value_ceiling

        _, market = generate_problem(:inverse_optimization, 180, infeasible, 4;
                                     variant=:market_clearing)
        market_certificate = market.infeasibility_certificate
        @test market_certificate.maxed_cost_lower > market_certificate.idle_cost_upper
        @test market.observed_dispatch[market_certificate.period,
                                       market_certificate.maxed_unit] ==
              market.capacities[market_certificate.maxed_unit]
        @test market.observed_dispatch[market_certificate.period,
                                       market_certificate.idle_unit] == 0.0
    end

    @testset "constructive status contracts" begin
        for ref in INVERSE_VARIANTS, seed in 0:20
            _, feasible_prob = generate_problem(ref, 140, feasible, seed)
            @test feasible_prob.feasible_witness !== nothing
            @test feasible_prob.infeasibility_certificate === nothing

            _, infeasible_prob = generate_problem(ref, 140, infeasible, seed)
            @test infeasible_prob.feasible_witness === nothing
            @test infeasible_prob.infeasibility_certificate !== nothing
        end

        for ref in INVERSE_VARIANTS
            for seed in 0:80
                _, prob = generate_problem(ref, 140, unknown, seed)
                @test prob.feasible_witness === nothing
                @test prob.infeasibility_certificate === nothing
            end
        end
    end

    @testset "solver feasibility contracts" begin
        if HAS_HIGHS
            for ref in INVERSE_VARIANTS, status in (feasible, infeasible),
                target in (50, 180), seed in 0:8
                model, _ = generate_problem(ref, target, status, seed)
                set_optimizer(model, HiGHS.Optimizer)
                set_silent(model)
                optimize!(model)
                @test termination_status(model) ==
                      (status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE)
            end

            @testset "recovered parameters explain forward behavior" begin
                for variant in (:standard, :linf, :restricted_optimal_value)
                    inverse, prob = generate_problem(
                        :inverse_optimization, 180, feasible, 9; variant=variant,
                    )
                    set_optimizer(inverse, HiGHS.Optimizer)
                    set_silent(inverse)
                    optimize!(inverse)
                    recovered = value.(inverse[:c])

                    forward = Model(HiGHS.Optimizer)
                    set_silent(forward)
                    @variable(forward, x[1:prob.num_cols] >= 0)
                    for i in 1:prob.num_rows
                        @constraint(forward,
                            sum(prob.forward_matrix[i, j] * x[j]
                                for j in 1:prob.num_cols) >= prob.forward_rhs[i])
                    end
                    @objective(forward, Min,
                               sum(recovered[j] * x[j] for j in 1:prob.num_cols))
                    optimize!(forward)
                    observed_value = dot(recovered, prob.reference_point)
                    @test objective_value(forward) ≈ observed_value rtol=1.0e-6
                    if variant == :restricted_optimal_value
                        @test objective_value(forward) ≈ prob.target_value rtol=1.0e-6
                    end
                end

                inverse, spatial = generate_problem(
                    :inverse_optimization, 240, feasible, 9; variant=:shortest_path,
                )
                set_optimizer(inverse, HiGHS.Optimizer)
                set_silent(inverse)
                optimize!(inverse)
                recovered = value.(inverse[:arc_cost])
                @test SyntheticLPs._inverse_paths_are_optimal(
                    spatial.n_nodes, spatial.arcs, spatial.observations, recovered,
                )

                inverse, layered = generate_problem(
                    :inverse_optimization, 240, feasible, 9;
                    variant=:shortest_path_layered,
                )
                set_optimizer(inverse, HiGHS.Optimizer)
                set_silent(inverse)
                optimize!(inverse)
                recovered = value.(inverse[:arc_cost])
                @test SyntheticLPs._layered_prior_is_optimal(
                    layered.num_nodes, layered.source, layered.sink,
                    layered.tail, layered.head, recovered, layered.path_arcs,
                )

                # Directly verify that informative market priors translate to
                # a nonzero inverse adjustment even with the ramp rows present.
                for target in (50, 100), seed in 0:4
                    inverse, _ = generate_problem(
                        :inverse_optimization, target, feasible, seed;
                        variant=:market_clearing,
                    )
                    set_optimizer(inverse, HiGHS.Optimizer)
                    set_silent(inverse)
                    optimize!(inverse)
                    @test termination_status(inverse) == MOI.OPTIMAL
                    @test objective_value(inverse) > 1.0e-8
                end
            end

            @testset "unknown profiles contain both outcomes" begin
                for ref in INVERSE_VARIANTS
                    outcomes = Set{MOI.TerminationStatusCode}()
                    for seed in 0:19
                        model, _ = generate_problem(ref, 140, unknown, seed)
                        set_optimizer(model, HiGHS.Optimizer)
                        set_silent(model)
                        optimize!(model)
                        @test termination_status(model) in
                              (MOI.OPTIMAL, MOI.INFEASIBLE)
                        push!(outcomes, termination_status(model))
                    end
                    @test outcomes == Set((MOI.OPTIMAL, MOI.INFEASIBLE))
                end
            end
        else
            @info "HiGHS not available; skipping inverse-optimization solver contracts"
        end
    end
end
