# Focused quality contracts for the project_selection category: registry
# shape, exact sizing and the row-count formula, sparse (linear) structure at
# scale, data invariants (DAG prerequisites, exclusive groups, spending
# profiles), the planted-portfolio witness and division-mandate certificate
# recomputed from the struct fields, reproducibility, and HiGHS contracts.

# Expected affine row count, recomputed from the struct fields.
function ps_expected_rows(p)
    active = falses(p.n_divisions, p.n_years)
    for q in 1:p.n_projects, k in 1:length(p.spend[q])
        active[p.division[q], p.start_year[q] + k - 1] = true
    end
    high = any(>(p.high_risk_threshold), p.risk_scores) ? 1 : 0
    return p.n_years + 2 * count(active) + length(p.prerequisites) +
           length(p.exclusive_groups) + count(>(0), p.division_mandate) + 1 + high
end

@testset "Project Selection" begin
    @test :project_selection in list_categories()
    @test Set(list_variants(:project_selection)) == Set([:standard])

    # Exact sizing: one binary per project, at every scale and status; the
    # row count follows the documented formula and grows with the size.
    for target in (1, 2, 5, 40, 300, 2500), status in (feasible, infeasible, unknown), seed in 0:1
        m, p = generate_problem(:project_selection, target, status, seed)
        @test num_variables(m) == p.n_projects == target
        @test num_constraints(m; count_variable_in_set_constraints=false) == ps_expected_rows(p)
    end

    # Linear structure at scale: rows ~ columns and ~10-14 nonzeros per
    # column (the old generator emitted an O(n^2) dependency matrix).
    _, big = generate_problem(:project_selection, 20_000, unknown, 3)
    @test 0.6 * big.n_projects <= ps_expected_rows(big) <= 2.0 * big.n_projects
    @test length(big.prerequisites) <= 2 * big.n_projects
    @test big.n_divisions == round(Int, 20_000 / 60)
    nnz_est = sum(3 * length(s) for s in big.spend) + 2 * length(big.prerequisites)
    @test nnz_est <= 20 * big.n_projects

    # Data invariants shared by all profiles.
    for status in (feasible, infeasible, unknown), seed in 0:2
        _, p = generate_problem(:project_selection, 600, status, seed)
        n = p.n_projects
        @test all(1 .<= p.division .<= p.n_divisions)
        @test all(1 .<= p.theme .<= p.n_themes)
        @test all(d -> any(==(d), p.division), 1:p.n_divisions)   # no empty division
        for q in 1:n
            @test length(p.spend[q]) == length(p.fte[q]) == p.duration[q]
            @test p.start_year[q] + p.duration[q] - 1 <= p.n_years
            @test all(>(0), p.spend[q]) && all(>(0), p.fte[q])
        end
        @test all(>(0), p.returns)
        @test all(1.0 .<= p.risk_scores .<= 10.0)
        # Prerequisites point at platform projects that do not start later;
        # platform-on-platform edges go to lower indices, so the graph is a DAG.
        for (a, b) in p.prerequisites
            @test a != b
            @test p.is_platform[b]
            @test p.start_year[b] <= p.start_year[a]
            p.is_platform[a] && @test b < a
        end
        # Exclusive groups: 2-3 non-platform alternatives of one division and
        # theme, pairwise disjoint.
        seen = Set{Int}()
        for g in p.exclusive_groups
            @test 2 <= length(g) <= 3
            @test !any(p.is_platform[g])
            @test length(unique(p.division[g])) == 1 && length(unique(p.theme[g])) == 1
            @test isempty(intersect(seen, g))
            union!(seen, g)
        end
        @test all(>=(0), p.division_mandate)
        @test all(>(0), p.corporate_budget)
    end

    # Witness: the planted 0/1 portfolio satisfies every row in plain
    # arithmetic, and is primal-feasible for the built model.
    for target in (50, 700, 4000), seed in 0:2
        m, p = generate_problem(:project_selection, target, feasible, seed)
        w = p.feasible_witness
        @test w !== nothing && p.infeasibility_certificate === nothing
        sel = falses(p.n_projects)
        sel[w.selected] .= true
        @test issorted(w.selected) && !isempty(w.selected)
        @test all(sel[b] for (a, b) in p.prerequisites if sel[a])
        @test all(count(sel[g]) <= 1 for g in p.exclusive_groups)
        corp, dspend, dfte, cnt = SyntheticLPs._ps_usage(
            sel, p.division, p.start_year, p.spend, p.fte, p.n_years, p.n_divisions
        )
        @test all(corp .<= p.corporate_budget .* (1 + 1e-12))
        @test all(dspend .<= p.division_budget .* (1 + 1e-12))
        @test all(dfte .<= p.division_fte .* (1 + 1e-12))
        @test all(cnt .>= p.division_mandate)
        @test sum(p.risk_scores[sel]) <= p.risk_budget
        @test count(q -> sel[q] && p.risk_scores[q] > p.high_risk_threshold, 1:p.n_projects) <=
            p.max_high_risk
        point = Dict(m[:x][q] => (sel[q] ? 1.0 : 0.0) for q in 1:p.n_projects)
        @test isempty(primal_feasibility_report(m, point; atol=1e-6))
    end

    # Certificate: the mandated division's cheapest lifetime costs exceed its
    # summed yearly budgets by at least 5%, recomputed from raw fields.
    for target in (50, 700, 4000), seed in 0:3
        _, p = generate_problem(:project_selection, target, infeasible, seed)
        c = p.infeasibility_certificate
        @test c !== nothing && p.feasible_witness === nothing
        members = findall(==(c.division), p.division)
        costs = sort([sum(p.spend[q]) for q in members])
        @test c.mandate == p.division_mandate[c.division] <= length(members)
        @test length(c.cheapest_projects) == c.mandate
        @test all(p.division[q] == c.division for q in c.cheapest_projects)
        @test c.cheapest_cost ≈ sum(costs[1:c.mandate]) rtol = 1e-12
        @test c.division_budget_total ≈ sum(p.division_budget[c.division, :]) rtol = 1e-12
        @test c.cheapest_cost >= 1.05 * c.division_budget_total
    end

    # Unknown asserts nothing.
    for seed in 0:2
        _, p = generate_problem(:project_selection, 400, unknown, seed)
        @test p.feasible_witness === nothing && p.infeasibility_certificate === nothing
    end

    # Reproducibility, independent of the global RNG.
    for status in (feasible, infeasible, unknown)
        Random.seed!(1)
        _, p1 = generate_problem(:project_selection, 900, status, 11)
        Random.seed!(2)
        _, p2 = generate_problem(:project_selection, 900, status, 11)
        @test p1.spend == p2.spend && p1.returns == p2.returns
        @test p1.prerequisites == p2.prerequisites
        @test p1.division_budget == p2.division_budget
        @test p1.division_mandate == p2.division_mandate
    end

    @testset "Project Selection HiGHS contracts" begin
        if HAS_HIGHS
            for target in (60, 500, 3000), status in (feasible, infeasible), seed in 0:2
                m, _ = generate_problem(:project_selection, target, status, seed)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                @test termination_status(m) ==
                    (status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE)
            end
            # `unknown` is a genuine mix across seeds.
            outcomes = Set{MOI.TerminationStatusCode}()
            for target in (300, 3000), seed in 0:9
                m, _ = generate_problem(:project_selection, target, unknown, seed)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                push!(outcomes, termination_status(m))
            end
            @test MOI.OPTIMAL in outcomes
            @test MOI.INFEASIBLE in outcomes
        end
    end
end
