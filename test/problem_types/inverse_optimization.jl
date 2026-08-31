using SparseArrays

const INVERSE_VARIANTS = (
    "inverse_optimization/classical",
    "inverse_optimization/noisy_observations",
    "inverse_optimization/shortest_path",
)

@testset "Inverse Optimization" begin
    @test :inverse_optimization in list_categories()
    @test list_variants(:inverse_optimization) ==
          [:classical, :noisy_observations, :shortest_path]
    @test problem_info(:inverse_optimization)[:default_variant] == :classical
    @test occursin(
        "inverse",
        lowercase(problem_info(:inverse_optimization)[:description]),
    )

    @testset "sizing and deterministic builds" begin
        for ref in INVERSE_VARIANTS, target in (1, 50, 100, 501, 2_000)
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
            "inverse_optimization/classical", 301, feasible, 3,
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
    end

    @testset "classical inverse LP data and algebra" begin
        for seed in 0:20
            model, prob = generate_problem(
                "inverse_optimization/classical", 180, feasible, seed,
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

    @testset "constructive status contracts" begin
        for ref in INVERSE_VARIANTS, seed in 0:20
            _, feasible_prob = generate_problem(ref, 140, feasible, seed)
            @test feasible_prob.resolved_status == feasible
            @test feasible_prob.feasible_witness !== nothing
            @test feasible_prob.infeasibility_certificate === nothing

            _, infeasible_prob = generate_problem(ref, 140, infeasible, seed)
            @test infeasible_prob.resolved_status == infeasible
            @test infeasible_prob.feasible_witness === nothing
            @test SyntheticLPs._inverse_cost_certificate_is_valid(
                infeasible_prob.infeasibility_certificate,
            )
        end

        for ref in INVERSE_VARIANTS
            statuses = Set{FeasibilityStatus}()
            for seed in 0:80
                _, prob = generate_problem(ref, 140, unknown, seed)
                push!(statuses, prob.resolved_status)
                @test (prob.feasible_witness !== nothing) ==
                      (prob.resolved_status == feasible)
                @test (prob.infeasibility_certificate !== nothing) ==
                      (prob.resolved_status == infeasible)
            end
            @test statuses == Set((feasible, infeasible))
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
        else
            @info "HiGHS not available; skipping inverse-optimization solver contracts"
        end
    end
end
