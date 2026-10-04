# Focused quality contracts for the energy category: registry shape, exact
# variable/row formulas for all six variants, technology-grounded data
# invariants, planted witnesses checked row by row against the built models
# (no solver), certificate arithmetic recomputed from the struct fields, the
# "no single-row contradiction" property of every planted infeasibility,
# large-target sizing, and HiGHS-backed feasibility contracts.

const ENERGY_DISPATCH_VARIANTS = (:standard, :reserves, :storage, :hydrothermal)
const ENERGY_DC_VARIANTS = (:dc_opf, :security_constrained_dc_opf)

"""Map a stored witness onto the model's columns (for `primal_feasibility_report`)."""
function energy_witness_values(model, prob)
    vals = Dict{VariableRef, Float64}()
    function add_dispatch!(w)
        for g in axes(w.output, 1), t in axes(w.output, 2)
            vals[model[:x][g, t]] = w.output[g, t]
        end
        for l in axes(w.flow_fwd, 1), t in axes(w.flow_fwd, 2)
            vals[model[:flow_fwd][l, t]] = w.flow_fwd[l, t]
            vals[model[:flow_bwd][l, t]] = w.flow_bwd[l, t]
        end
    end
    w = prob.feasible_witness
    if prob isa SyntheticLPs.EconomicDispatchProblem
        add_dispatch!(w)
    elseif prob isa SyntheticLPs.ReservesDispatchProblem
        add_dispatch!(w.dispatch)
        G, T = size(w.spin)
        for g in 1:G, t in 1:T
            prob.spin_max[g] > 0 && (vals[model[:spin][g, t]] = w.spin[g, t])
            prob.nonspin_max[g] > 0 && (vals[model[:nonspin][g, t]] = w.nonspin[g, t])
        end
        for t in 1:T
            vals[model[:contingency][t]] = w.contingency[t]
        end
    elseif prob isa SyntheticLPs.StorageDispatchProblem
        add_dispatch!(w.dispatch)
        for s in axes(w.charge, 1), t in axes(w.charge, 2)
            vals[model[:charge][s, t]] = w.charge[s, t]
            vals[model[:discharge][s, t]] = w.discharge[s, t]
            vals[model[:soc][s, t]] = w.soc[s, t]
        end
    elseif prob isa SyntheticLPs.HydrothermalDispatchProblem
        add_dispatch!(w.dispatch)
        for r in axes(w.release, 1), t in axes(w.release, 2)
            vals[model[:release][r, t]] = w.release[r, t]
            vals[model[:spill][r, t]] = w.spill[r, t]
            vals[model[:volume][r, t]] = w.volume[r, t]
        end
    elseif prob isa SyntheticLPs.DCOptimalPowerFlowProblem
        for g in eachindex(w.dispatch)
            vals[model[:p][g]] = w.dispatch[g]
        end
        for b in eachindex(w.angles)
            vals[model[:theta][b]] = w.angles[b]
        end
    elseif prob isa SyntheticLPs.SecurityConstrainedDCOPFProblem
        for g in eachindex(w.dispatch)
            vals[model[:p][g]] = w.dispatch[g]
        end
        for (k, θ) in enumerate(model[:theta]), b in eachindex(θ)
            vals[θ[b]] = w.angles[b, k]
        end
    end
    return vals
end

"""Per-period column count of a dispatch-family instance (its sizing formula)."""
function energy_dispatch_columns(p)
    c = p.core
    G = length(c.unit_tech)
    L = length(c.tie_from)
    per = G + 2L
    if p isa SyntheticLPs.ReservesDispatchProblem
        per += count(>(0), p.spin_max) + count(>(0), p.nonspin_max) + 1
    elseif p isa SyntheticLPs.StorageDispatchProblem
        per += 3 * length(p.storage_zone)
    elseif p isa SyntheticLPs.HydrothermalDispatchProblem
        per += 3 * length(p.plant_zone)
    end
    return per * c.n_periods
end

"""Connectivity of an undirected edge list on `n` nodes."""
function energy_connected(n, from, to)
    adj = [Int[] for _ in 1:n]
    for (a, b) in zip(from, to)
        push!(adj[a], b)
        push!(adj[b], a)
    end
    seen = falses(n)
    seen[1] = true
    stack = [1]
    while !isempty(stack)
        v = pop!(stack)
        for u in adj[v]
            seen[u] || (seen[u] = true; push!(stack, u))
        end
    end
    return all(seen)
end

@testset "Energy" begin
    @testset "Registry" begin
        @test :energy in list_categories()
        @test Set(list_variants(:energy)) == Set([ENERGY_DISPATCH_VARIANTS..., ENERGY_DC_VARIANTS...])
        info = problem_info(:energy)
        @test info[:default_variant] == :standard
        # The deleted variants (folded into the dispatch core, or removed for a
        # meaningless relaxation) stay deleted.
        for gone in (:ramping, :transmission, :optimal_transmission_switching)
            @test !(gone in list_variants(:energy))
        end
    end

    @testset "Dispatch family sizing and structure" begin
        for v in ENERGY_DISPATCH_VARIANTS, target in (60, 400, 2_000, 12_000), seed in 0:1
            m, p = generate_problem(ProblemVariant(:energy, v), target, unknown, seed)
            c = p.core
            n = num_variables(m)
            @test n == energy_dispatch_columns(p)
            @test c.n_periods == SyntheticLPs._ed_horizon(max(target, 16))
            # Size fidelity: the fleet absorbs the per-period budget exactly, so
            # the count is within one period's rounding of the target.
            target >= 400 && @test abs(n - target) <= max(0.05 * target, c.n_periods)
            # Rows grow with the model (no wide-thin LPs).
            rows = num_constraints(m; count_variable_in_set_constraints=false)
            target >= 400 && @test rows >= 0.3 * n
            if v == :standard
                ramped = count(g -> SyntheticLPs._ed_ramped(c, g), eachindex(c.unit_tech))
                @test rows == c.n_zones * c.n_periods + ramped * (c.n_periods - 1) + 1
            end
            # Zone graph is connected; every zone hosts a unit; ties are loss-bounded.
            @test c.n_zones >= 2
            @test energy_connected(c.n_zones, c.tie_from, c.tie_to)
            @test Set(c.unit_zone) == Set(1:c.n_zones)
            @test all(0.005 .<= c.tie_loss .<= 0.06)
            @test all(c.tie_capacity .> 0)
            # Technology-grounded fleet.
            for g in eachindex(c.unit_tech)
                spec = SyntheticLPs.ENERGY_TECHNOLOGIES[c.unit_tech[g]]
                @test spec.capacity[1] - 1e-9 <= c.capacity[g] <= spec.capacity[2] + 1e-9
                @test c.min_stable[g] == 0 || spec.min_stable[1] <= c.min_stable[g] <= spec.min_stable[2]
                @test spec.emission[1] <= c.emission_rate[g] <= spec.emission[2]
                @test isfinite(c.ramp_up[g]) == isfinite(spec.ramp[1])
            end
            @test all(0.0 .<= c.availability .<= 1.0)
            @test all(c.demand .> 0)
            v == :hydrothermal && @test !(:hydro in c.unit_tech)
        end
        # Growth comes from zones and fleet, not only from periods.
        _, small = generate_problem(:energy, 2_000, feasible, 0)
        _, large = generate_problem(:energy, 40_000, feasible, 0)
        @test large.core.n_zones > 3 * small.core.n_zones
        @test length(large.core.unit_tech) > 3 * length(small.core.unit_tech)
        # Must-run floors never exceed 75 % of a zone's lightest natural hour.
        for seed in 0:3
            _, p = generate_problem(:energy, 3_000, unknown, seed)
            c = p.core
            for z in 1:c.n_zones
                floor = maximum(
                    sum((c.min_stable[g] * c.capacity[g] * c.availability[g, t] for g in eachindex(c.unit_tech) if c.unit_zone[g] == z); init=0.0)
                    for t in 1:c.n_periods
                )
                @test floor <= 0.75 * minimum(c.demand[z, :]) + 1e-6
            end
        end
    end

    @testset "DC network sizing and structure" begin
        for target in (30, 500, 5_000), status in (feasible, infeasible, unknown), seed in 0:1
            m, p = generate_problem("energy/dc_opf", target, status, seed)
            @test num_variables(m) == p.n_generators + p.n_buses == target
            @test num_constraints(m; count_variable_in_set_constraints=false) == p.n_buses + p.n_lines
            @test energy_connected(p.n_buses, p.line_from, p.line_to)
            @test allunique(minmax.(p.line_from, p.line_to))
            @test all(p.line_limit .> 0) && all(5.0 .<= p.susceptance .<= 1000.0)
            @test all(p.pmin .<= p.pmax) && all(p.demand .>= 0)
            @test p.angle_limit >= 100 * π / 3
        end
        for target in (60, 800, 6_000), status in (feasible, infeasible, unknown), seed in 0:1
            m, p = generate_problem("energy/security_constrained_dc_opf", target, status, seed)
            C = length(p.contingencies)
            @test num_variables(m) == p.n_generators + (1 + C) * p.n_buses
            C == SyntheticLPs._scopf_contingency_count(target) && @test num_variables(m) == target
            @test num_constraints(m; count_variable_in_set_constraints=false) ==
                (1 + C) * (p.n_buses + p.n_lines) - C
            @test allunique(p.contingencies)
            # Screened outages never island the grid.
            bridges = SyntheticLPs._graph_bridges(p.n_buses, p.line_from, p.line_to)
            @test !any(bridges[p.contingencies])
            @test all(p.emergency_limit .>= 1.1 .* p.line_limit .- 1e-9)
        end
        @test SyntheticLPs._scopf_contingency_count(100_000) == 40
    end

    @testset "Large-target sizing (constructor only)" begin
        for v in (:standard, :reserves, :storage, :hydrothermal)
            p = SyntheticLPs.LP_REGISTRY[:energy].variants[v].type(100_000, feasible, 0)
            @test abs(energy_dispatch_columns(p) - 100_000) <= 0.02 * 100_000
            @test p.core.n_periods == 168
            @test p.core.n_zones >= 30
        end
        p = SyntheticLPs.DCOptimalPowerFlowProblem(100_000, feasible, 0)
        @test p.n_generators + p.n_buses == 100_000
        p = SyntheticLPs.SecurityConstrainedDCOPFProblem(100_000, feasible, 0)
        @test p.n_generators + (1 + length(p.contingencies)) * p.n_buses == 100_000
    end

    @testset "Planted witnesses satisfy every row" begin
        for v in (ENERGY_DISPATCH_VARIANTS..., ENERGY_DC_VARIANTS...), target in (60, 700, 3_000), seed in 0:2
            m, p = generate_problem(ProblemVariant(:energy, v), target, feasible, seed)
            @test p.feasible_witness !== nothing
            @test p.infeasibility_certificate === nothing
            report = primal_feasibility_report(m, energy_witness_values(m, p); atol=1e-6)
            @test isempty(report)
        end
    end

    @testset "DC witness arithmetic (reduced Laplacian)" begin
        for target in (200, 2_000), seed in 0:2
            _, p = generate_problem("energy/dc_opf", target, feasible, seed)
            w = p.feasible_witness
            @test w.angles[p.ref_bus] == 0
            @test sum(w.dispatch) ≈ sum(p.demand) rtol = 1e-9
            @test all(p.pmin .- 1e-9 .<= w.dispatch .<= p.pmax .+ 1e-9)
            # Every generator sits at the same fraction of its range.
            β = [(w.dispatch[g] - p.pmin[g]) / (p.pmax[g] - p.pmin[g]) for g in eachindex(w.dispatch) if p.pmax[g] > p.pmin[g] + 1e-6]
            @test maximum(β) - minimum(β) < 1e-9
            inj = -copy(p.demand)
            for g in eachindex(w.dispatch)
                inj[p.gen_bus[g]] += w.dispatch[g]
            end
            for l in 1:p.n_lines
                @test w.flows[l] ≈ p.susceptance[l] * (w.angles[p.line_from[l]] - w.angles[p.line_to[l]]) atol = 1e-6
                @test abs(w.flows[l]) <= p.line_limit[l] / 1.15 + 1e-6 || abs(w.flows[l]) <= p.line_limit[l] - 1.0
                inj[p.line_from[l]] -= w.flows[l]
                inj[p.line_to[l]] += w.flows[l]
            end
            @test maximum(abs, inj) < 1e-6 * max(1.0, sum(p.demand))
            @test maximum(abs, w.angles) <= p.angle_limit / 1.3 + 1e-9
        end
        # SCOPF: every state's angles realize the same injections on its own network.
        _, p = generate_problem("energy/security_constrained_dc_opf", 1_500, feasible, 1)
        w = p.feasible_witness
        @test size(w.angles, 2) == 1 + length(p.contingencies)
        for (k, out) in enumerate((0, p.contingencies...))
            inj = -copy(p.demand)
            for g in eachindex(w.dispatch)
                inj[p.gen_bus[g]] += w.dispatch[g]
            end
            limit = out == 0 ? p.line_limit : p.emergency_limit
            θ = w.angles[:, k]
            @test θ[p.ref_bus] == 0
            for l in 1:p.n_lines
                l == out && continue
                f = p.susceptance[l] * (θ[p.line_from[l]] - θ[p.line_to[l]])
                @test abs(f) <= limit[l] + 1e-6
                inj[p.line_from[l]] -= f
                inj[p.line_to[l]] += f
            end
            @test maximum(abs, inj) < 1e-6 * sum(p.demand)
        end
    end

    @testset "Storage and hydro witness details" begin
        for seed in 0:2
            _, p = generate_problem("energy/storage", 2_000, feasible, seed)
            w = p.feasible_witness
            # Each planted daily cycle returns exactly to the initial level.
            @test all(isapprox.(w.soc[:, end], p.soc_initial; rtol=1e-9))
            @test all(0.7 .<= p.eta_charge .* p.eta_discharge .<= 0.93)
            @test all(p.soc_min .< p.soc_initial .< p.soc_max)
            _, h = generate_problem("energy/hydrothermal", 2_000, feasible, seed)
            @test all(h.downstream[r] == 0 || h.downstream[r] > r for r in eachindex(h.downstream))
            @test all(0 .<= h.delay .<= 3)
            @test all(h.feasible_witness.volume[:, end] .>= h.volume_target .- 1e-9)
            @test all(h.feasible_witness.release .+ h.feasible_witness.spill .>= h.min_release .- 1e-9)
            # Water value grows downstream-to-upstream (stored water passes more plants).
            for r in eachindex(h.downstream)
                d = h.downstream[r]
                d > 0 && @test h.water_value[r] > h.water_value[d]
            end
        end
    end

    @testset "Certificate arithmetic" begin
        kinds = Set{Symbol}()
        for v in ENERGY_DISPATCH_VARIANTS, target in (60, 700, 3_000), seed in 0:3
            _, p = generate_problem(ProblemVariant(:energy, v), target, infeasible, seed)
            @test p.feasible_witness === nothing
            cert = p.infeasibility_certificate
            @test cert !== nothing
            if v == :hydrothermal
                @test cert isa SyntheticLPs.HydroDroughtCertificate
                @test cert.outlet == cert.basin[end] && p.downstream[cert.outlet] == 0
                @test cert.available_water ≈ SyntheticLPs._hydro_available_water(p, cert.basin) rtol = 1e-9
                @test cert.required_release ≈ p.core.n_periods * p.min_release[cert.outlet] rtol = 1e-9
                @test cert.required_release >= 1.07 * cert.available_water
                continue
            end
            c = p.core
            push!(kinds, cert.kind)
            if cert.kind == :energy_limited_peak
                # The margin is on the storage energy, not on the window total.
                energy = sum(p.eta_discharge .* (p.soc_max .- p.soc_min))
                @test cert.requirement - cert.supply_bound >= 0.05 * energy
            else
                @test cert.requirement > 1.03 * cert.supply_bound
            end
            if cert.kind == :import_pocket
                S = cert.zones
                t = only(cert.periods)
                @test length(S) >= 2
                @test cert.supply_bound ≈ SyntheticLPs._ed_pocket_supply(c, S, t) rtol = 1e-9
                @test cert.requirement ≈ sum(c.demand[z, t] for z in S) rtol = 1e-9
                # No zone's own balance row is contradictory on its own.
                for z in S
                    @test c.demand[z, t] <= SyntheticLPs._ed_zone_row_max(c, z, t)
                end
            elseif cert.kind == :emissions_budget
                bound, row_min = SyntheticLPs._ed_emission_lower_bound(c)
                @test cert.supply_bound == p.emission_cap
                @test cert.requirement ≈ bound rtol = 1e-9
                @test p.emission_cap > row_min    # the row alone is satisfiable
            elseif cert.kind == :reserve_scarcity
                t = only(cert.periods)
                D = SyntheticLPs._ed_system_demand(c, t)
                @test cert.supply_bound ≈ SyntheticLPs._ed_system_upper(c, t) rtol = 1e-9
                @test cert.requirement ≈ D + p.operating_requirement[t] rtol = 1e-9
                @test D <= cert.supply_bound                       # load alone fits
                @test p.operating_requirement[t] <= SyntheticLPs._reserve_offer(c, p.spin_max, p.nonspin_max, t) + 1e-9
            elseif cert.kind == :energy_limited_peak
                W = cert.periods
                @test W == collect(first(W):last(W))
                energy = sum(p.eta_discharge .* (p.soc_max .- p.soc_min))
                @test cert.supply_bound ≈ sum(SyntheticLPs._ed_system_upper(c, t) for t in W) + energy rtol = 1e-9
                @test cert.requirement ≈ sum(SyntheticLPs._ed_system_demand(c, t) for t in W) rtol = 1e-9
                local_dis = zeros(c.n_zones)
                for s in eachindex(p.storage_zone)
                    local_dis[p.storage_zone[s]] += p.discharge_max[s]
                end
                for t in W, z in 1:c.n_zones
                    @test c.demand[z, t] <= SyntheticLPs._ed_zone_row_max(c, z, t; extra=local_dis[z])
                end
            end
        end
        @test :emissions_budget in kinds && :reserve_scarcity in kinds && :energy_limited_peak in kinds
        # The standard variant's pocket mode can be forced.
        p = SyntheticLPs._economic_dispatch(3_000, infeasible, 1; infeasible_mode=:import_pocket)
        @test p.infeasibility_certificate.kind == :import_pocket

        for v in ENERGY_DC_VARIANTS, target in (60, 700, 3_000), seed in 0:3
            _, p = generate_problem(ProblemVariant(:energy, v), target, infeasible, seed)
            cert = p.infeasibility_certificate
            @test cert isa SyntheticLPs.DCPocketCertificate
            S = cert.buses
            out = cert.outaged_line
            @test length(S) >= 2
            crossing = [l for l in 1:p.n_lines if (p.line_from[l] in S) != (p.line_to[l] in S)]
            @test Set(cert.cut_lines) == Set(l for l in crossing if l != out)
            ratings = cert.contingency == 0 ? p.line_limit : p.emergency_limit
            if v == :security_constrained_dc_opf
                @test cert.contingency >= 1 && p.contingencies[cert.contingency] == out
                @test out in crossing
                # The base case alone can still serve the pocket.
                @test cert.local_capacity + sum(p.line_limit[l] for l in crossing) >= cert.pocket_demand
            else
                @test cert.contingency == 0 && out == 0
            end
            @test cert.local_capacity ≈ sum((p.pmax[g] for g in eachindex(p.pmax) if p.gen_bus[g] in S); init=0.0) rtol = 1e-9
            @test cert.import_capability ≈ sum(ratings[l] for l in cert.cut_lines) rtol = 1e-9
            @test cert.pocket_demand ≈ sum(p.demand[S]) rtol = 1e-9
            @test cert.pocket_demand >= 1.05 * (cert.local_capacity + cert.import_capability)
        end
    end

    @testset "Reproducibility" begin
        for v in (ENERGY_DISPATCH_VARIANTS..., ENERGY_DC_VARIANTS...), status in (feasible, infeasible, unknown)
            ref = ProblemVariant(:energy, v)
            _, a = generate_problem(ref, 900, status, 42)
            _, b = generate_problem(ref, 900, status, 42)
            if hasproperty(a, :core)
                @test a.core.demand == b.core.demand
                @test a.core.capacity == b.core.capacity
            else
                @test a.demand == b.demand && a.line_limit == b.line_limit
            end
        end
    end

    @testset "HiGHS feasibility contracts" begin
        if HAS_HIGHS
            for v in (ENERGY_DISPATCH_VARIANTS..., ENERGY_DC_VARIANTS...), status in (feasible, infeasible),
                target in (300, 2_500), seed in 0:1

                m, _ = generate_problem(ProblemVariant(:energy, v), target, status, seed)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                @test termination_status(m) == (status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE)
            end
            # `unknown` is natural and two-sided: both outcomes occur.
            for v in (:standard, :reserves, :dc_opf)
                seen = Set{MOI.TerminationStatusCode}()
                for seed in 0:11
                    m, _ = generate_problem(ProblemVariant(:energy, v), 2_000, unknown, seed)
                    set_optimizer(m, HiGHS.Optimizer)
                    set_silent(m)
                    optimize!(m)
                    push!(seen, termination_status(m))
                end
                @test MOI.OPTIMAL in seen
                @test MOI.INFEASIBLE in seen
            end
        end
    end
end
