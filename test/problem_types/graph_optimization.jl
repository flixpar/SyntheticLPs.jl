# Focused quality contracts for the graph_optimization category: registry
# shape, exact sizing, geometric/scale-free graph invariants, clique-cover
# correctness (every edge inside a clique row, every row a clique), witness and
# certificate arithmetic recomputed from the struct fields, reproducibility, and
# HiGHS feasibility contracts on the default LP relaxation.

const GRAPH_VARIANTS = (
    :independent_set,
    :generalized_independent_set,
    :vertex_cover,
    :vertex_coloring,
    :map_labeling,
    :quasi_clique,
)

_go_ref(v) = ProblemVariant(:graph_optimization, v)

function _go_is_clique(edge_set, clique)
    return all(
        (clique[a], clique[b]) in edge_set for a in 1:(length(clique) - 1) for
        b in (a + 1):length(clique)
    )
end

function _go_cover_ok(edges, cliques)
    covered = Set{Tuple{Int, Int}}()
    for K in cliques, a in 1:(length(K) - 1), b in (a + 1):length(K)
        push!(covered, (K[a], K[b]))
    end
    return all(e -> e in covered, edges)
end

function _go_partition_ok(cert, n, cliques)
    seen = sort!(reduce(vcat, cert.parts))
    seen == collect(1:n) || return false
    for (part, row) in zip(cert.parts, cert.part_rows)
        if row == 0
            length(part) == 1 || return false
        else
            issubset(part, cliques[row]) || return false
        end
    end
    return cert.bound == length(cert.parts)
end

@testset "Graph Optimization" begin
    @test Set(list_variants(:graph_optimization)) == Set(GRAPH_VARIANTS)
    @test problem_info(:graph_optimization)[:default_variant] == :independent_set

    @testset "Exact sizing" begin
        for target in (60, 500, 3000), status in (feasible, infeasible, unknown), seed in 0:1
            for v in GRAPH_VARIANTS
                m, p = generate_problem(_go_ref(v), target, status, seed)
                @test num_variables(m) == target
            end
        end
        # Large targets build quickly and stay exact (constructor only).
        @test SyntheticLPs.IndependentSetProblem(100_000, unknown, 0).n_vertices == 100_000
        vc = SyntheticLPs.VertexCoverProblem(100_000, unknown, 0)
        @test vc.n_vertices + length(vc.edges) == 100_000
        qc = SyntheticLPs.QuasiCliqueProblem(100_000, infeasible, 0)
        @test qc.n_vertices + length(qc.edges) == 100_000
    end

    @testset "Clique formulations" begin
        for status in (feasible, infeasible, unknown), seed in 0:2
            _, p = generate_problem(_go_ref(:independent_set), 800, status, seed)
            es = Set(p.edges)
            @test issorted(p.edges) && allunique(p.edges)
            @test all(1 <= u < v <= p.n_vertices for (u, v) in p.edges)
            @test all(K -> _go_is_clique(es, K), p.cliques)
            @test _go_cover_ok(p.edges, p.cliques)
            # Unit-disk geometry: every edge within the unit radius.
            @test all(hypot(p.xs[u] - p.xs[v], p.ys[u] - p.ys[v]) <= 1.0 for (u, v) in p.edges)
            # The clique formulation is genuinely stronger than edge rows.
            @test maximum(length, p.cliques) >= 4
            m, _ = generate_problem(_go_ref(:independent_set), 800, status, seed)
            @test num_constraints(m; count_variable_in_set_constraints=false) ==
                length(p.cliques) + (p.minimum_selected > 0 ? 1 : 0)

            _, g = generate_problem(_go_ref(:generalized_independent_set), 800, status, seed)
            hs = Set(g.hard_edges)
            @test all(K -> _go_is_clique(hs, K), g.hard_cliques)
            @test _go_cover_ok(g.hard_edges, g.hard_cliques)
            @test isempty(intersect(Set(g.soft_edges), hs))
            @test all(>(0), g.edge_penalties)
            @test g.n_vertices + length(g.soft_edges) == 800

            _, ml = generate_problem(_go_ref(:map_labeling), 800, status, seed)
            pairs = Set(ml.conflicts)
            for c in ml.feature_candidates, a in 1:(length(c) - 1), b in (a + 1):length(c)
                push!(pairs, (c[a], c[b]))
            end
            @test all(K -> _go_is_clique(pairs, K), ml.cliques)
            @test _go_cover_ok(sort!(collect(pairs)), ml.cliques)
            @test sort!(reduce(vcat, ml.feature_candidates)) == collect(1:800)
            @test all(length(c) in (4, 5) for c in ml.feature_candidates)
            @test all(SyntheticLPs._map_overlap(ml.boxes[i], ml.boxes[j]) for (i, j) in ml.conflicts)

            _, vc = generate_problem(_go_ref(:vertex_coloring), 800, status, seed)
            ves = Set(vc.edges)
            @test all(K -> _go_is_clique(ves, K), vc.cliques)
            @test _go_cover_ok(vc.edges, vc.cliques)
            @test all(d -> issorted(d) && allunique(d) && all(1 .<= d .<= vc.n_channels), vc.domains)
            @test all(length(vc.costs[v]) == length(vc.domains[v]) for v in 1:vc.n_vertices)
        end
    end

    @testset "Witnesses" begin
        for seed in 0:3
            _, p = generate_problem(_go_ref(:independent_set), 1500, feasible, seed)
            w = p.feasible_witness.vertices
            chosen = falses(p.n_vertices)
            chosen[w] .= true
            @test !any(chosen[u] && chosen[v] for (u, v) in p.edges)
            @test length(w) >= p.minimum_selected
            @test p.infeasibility_certificate === nothing

            _, g = generate_problem(_go_ref(:generalized_independent_set), 1500, feasible, seed)
            chosen = falses(g.n_vertices)
            chosen[g.feasible_witness.vertices] .= true
            @test !any(chosen[u] && chosen[v] for (u, v) in g.hard_edges)
            @test length(g.feasible_witness.vertices) >= g.minimum_selected

            _, ml = generate_problem(_go_ref(:map_labeling), 1500, feasible, seed)
            placed = falses(1500)
            placed[ml.feasible_witness.vertices] .= true
            @test !any(placed[i] && placed[j] for (i, j) in ml.conflicts)
            @test all(count(placed[c]) <= 1 for c in ml.feature_candidates)
            @test count(placed) >= ml.minimum_placed

            _, vcov = generate_problem(_go_ref(:vertex_cover), 1500, feasible, seed)
            monitor = vcov.feasible_witness.monitor
            @test all(monitor[e] in vcov.edges[e] for e in eachindex(vcov.edges))
            load = zeros(Int, vcov.n_vertices)
            foreach(w -> load[w] += 1, monitor)
            @test all(load .<= vcov.capacity)
            @test all(1 .<= vcov.capacity)

            _, col = generate_problem(_go_ref(:vertex_coloring), 1500, feasible, seed)
            ch = col.feasible_witness.channel
            @test all(insorted(ch[v], col.domains[v]) for v in 1:col.n_vertices)
            @test all(ch[u] != ch[v] for (u, v) in col.edges)

            _, qc = generate_problem(_go_ref(:quasi_clique), 1500, feasible, seed)
            members = Set(qc.feasible_witness.vertices)
            @test length(members) == qc.selected_vertices
            induced = [e for (e, (u, v)) in enumerate(qc.edges) if u in members && v in members]
            @test induced == qc.feasible_witness.edges
            @test length(induced) >= qc.required_edges
            @test qc.required_edges <= qc.selected_vertices * (qc.selected_vertices - 1) ÷ 2
        end
    end

    @testset "Infeasibility certificates" begin
        for seed in 0:3
            for v in (:independent_set, :generalized_independent_set)
                _, p = generate_problem(_go_ref(v), 1500, infeasible, seed)
                cert = p.infeasibility_certificate
                cliques = v == :independent_set ? p.cliques : p.hard_cliques
                @test _go_partition_ok(cert, p.n_vertices, cliques)
                @test p.minimum_selected >= cert.bound + 1
                @test p.feasible_witness === nothing
            end
            _, ml = generate_problem(_go_ref(:map_labeling), 1500, infeasible, seed)
            @test _go_partition_ok(ml.infeasibility_certificate, 1500, ml.cliques)
            @test ml.minimum_placed > ml.infeasibility_certificate.bound
            @test ml.infeasibility_certificate.bound <= ml.n_features

            _, vcov = generate_problem(_go_ref(:vertex_cover), 1500, infeasible, seed)
            cert = vcov.infeasibility_certificate
            core = Set(cert.core)
            @test cert.core_edges ==
                [e for (e, (a, b)) in enumerate(vcov.edges) if a in core && b in core]
            @test cert.capacity_total == sum(vcov.capacity[cert.core])
            @test length(cert.core_edges) >= 1.1 * cert.capacity_total - 1
            @test length(cert.core) >= 4

            _, col = generate_problem(_go_ref(:vertex_coloring), 1500, infeasible, seed)
            cert = col.infeasibility_certificate
            venue = col.cliques[cert.clique]
            @test _go_is_clique(Set(col.edges), venue)
            @test length(cert.channels) < length(venue)
            @test all(issubset(col.domains[v], cert.channels) for v in venue)
            @test length(venue) >= 5          # not a trivially small contradiction

            _, qc = generate_problem(_go_ref(:quasi_clique), 1500, infeasible, seed)
            cert = qc.infeasibility_certificate
            @test all(cert.head[e] in qc.edges[e] for e in eachindex(qc.edges))
            load = zeros(Int, qc.n_vertices)
            foreach(h -> load[h] += 1, cert.head)
            @test load == cert.load
            @test cert.bound == sum(sort(load; rev=true)[1:qc.selected_vertices])
            @test qc.required_edges > cert.bound
            @test qc.required_edges <= qc.selected_vertices * (qc.selected_vertices - 1) ÷ 2
        end
    end

    @testset "Tiny vertex_cover deficit" begin
        # Target 6 gives a 3-vertex triangle: the rich club must clamp to the
        # whole graph (it once indexed 4 of 3 vertices).
        for seed in 0:3
            m, p = generate_problem(_go_ref(:vertex_cover), 6, infeasible, seed)
            cert = p.infeasibility_certificate
            @test num_variables(m) == 6
            @test length(cert.core) <= p.n_vertices
            @test cert.capacity_total == sum(p.capacity[cert.core])
            @test length(cert.core_edges) > cert.capacity_total
        end
    end

    @testset "Model structure" begin
        # Capacitated cover: one orientation variable per link, no doubleton
        # assignment equalities; rows = 2 linking rows per link + 1 per vertex.
        m, p = generate_problem(_go_ref(:vertex_cover), 2000, unknown, 3)
        @test num_variables(m) == p.n_vertices + length(p.edges)
        @test num_constraints(m; count_variable_in_set_constraints=false) ==
            2 * length(p.edges) + p.n_vertices
        # Quasi-clique: edge variables exist only for real edges.
        m, q = generate_problem(_go_ref(:quasi_clique), 2000, unknown, 3)
        @test num_constraints(m; count_variable_in_set_constraints=false) ==
            2 * length(q.edges) + 2
        # The default relaxation leaves no integer variables.
        @test !any(is_binary, all_variables(m))
    end

    @testset "Reproducibility" begin
        for v in GRAPH_VARIANTS, status in (feasible, infeasible, unknown)
            _, a = generate_problem(_go_ref(v), 700, status, 11)
            _, b = generate_problem(_go_ref(v), 700, status, 11)
            for f in fieldnames(typeof(a))
                x, y = getfield(a, f), getfield(b, f)
                (x === nothing || isbits(x) || x isa AbstractArray) && @test x == y
            end
        end
    end

    @testset "HiGHS feasibility contracts" begin
        if HAS_HIGHS
            for v in GRAPH_VARIANTS, target in (300, 2500), seed in 0:1
                for (status, expected) in ((feasible, MOI.OPTIMAL), (infeasible, MOI.INFEASIBLE))
                    m, _ = generate_problem(_go_ref(v), target, status, seed)
                    set_optimizer(m, HiGHS.Optimizer)
                    set_silent(m)
                    optimize!(m)
                    @test termination_status(m) == expected
                end
            end
        else
            @info "HiGHS not available; skipping graph_optimization solver contracts"
        end
    end
end
