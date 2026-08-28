using JuMP
using Random
using Distributions
using LinearAlgebra
using SparseArrays

"""
    OptimalTransmissionSwitchingProblem <: ProblemGenerator

A single-period DC optimal-transmission-switching (OTS) model. In addition to
generation, voltage-angle, and line-flow variables, every candidate line has a
binary status variable. Thermal limits force flow to zero when a line is open,
and a pair of bound-derived big-M inequalities enforces the DC flow equation
when it is closed.

The angle bounds are part of the generated data. Consequently
`flow_big_m[l] = 2 * susceptance[l] * angle_limit` is a valid, line-specific
constant: when line `l` is open its flow is zero and the largest possible
absolute angle difference is `2 * angle_limit`.
"""
struct OptimalTransmissionSwitchingProblem <: ProblemGenerator
    n_buses::Int
    n_lines::Int
    n_generators::Int
    line_from::Vector{Int}
    line_to::Vector{Int}
    susceptance::Vector{Float64}
    line_limit::Vector{Float64}
    flow_big_m::Vector{Float64}
    gen_bus::Vector{Int}
    gen_cost::Vector{Float64}
    pmin::Vector{Float64}
    pmax::Vector{Float64}
    demand::Vector{Float64}
    switching_cost::Vector{Float64}
    angle_limit::Float64
    ref_bus::Int
end

# Add a candidate undirected line while retaining a deterministic orientation.
function _ots_add_line!(line_from::Vector{Int}, line_to::Vector{Int},
                        edge_set::Set{Tuple{Int,Int}}, a::Int, b::Int)
    a == b && return false
    key = a < b ? (a, b) : (b, a)
    key in edge_set && return false
    push!(edge_set, key)
    push!(line_from, a)
    push!(line_to, b)
    return true
end

"""
    OptimalTransmissionSwitchingProblem(target_variables, feasibility_status, seed)

Construct an OTS instance with

    n_generators + n_buses + 2 * n_lines

decision variables. Feasible requests contain an explicit connected closed-line
topology and dispatch witness. Infeasible requests set total demand strictly
above total generation capacity, a certificate that remains valid after binary
status variables are relaxed. Unknown requests retain naturally sampled thermal
limits rather than forcing either outcome.
"""
function OptimalTransmissionSwitchingProblem(
    target_variables::Int,
    feasibility_status::FeasibilityStatus,
    seed::Int,
)
    rng = MersenneTwister(seed)

    target = max(target_variables, 1)
    edge_factor = rand(rng, Uniform(1.45, 2.05))
    generator_fraction = rand(rng, Uniform(0.28, 0.45))
    n_buses = max(3, round(Int, target / (1 + 2 * edge_factor + generator_fraction)))
    n_generators = max(2, round(Int, generator_fraction * n_buses))
    desired_lines = round(Int, (target - n_buses - n_generators) / 2)
    n_lines = clamp(desired_lines, n_buses - 1, n_buses * (n_buses - 1) ÷ 2)

    # Candidate network: a random spanning tree plus additional switchable lines.
    line_from = Int[]
    line_to = Int[]
    edge_set = Set{Tuple{Int,Int}}()
    order = randperm(rng, n_buses)
    for idx in 2:n_buses
        parent_idx = rand(rng, 1:(idx - 1))
        _ots_add_line!(line_from, line_to, edge_set, order[parent_idx], order[idx])
    end
    # Sample extra mesh lines by rejection rather than materializing the
    # complete undirected edge set (Θ(n_buses²) pairs, of which only
    # ~0.45–1.05 per bus are kept). Matches `energy/dc_opf`.
    attempts = 0
    max_attempts = 50 * n_lines
    while length(line_from) < n_lines && attempts < max_attempts
        attempts += 1
        _ots_add_line!(line_from, line_to, edge_set,
                       rand(rng, 1:n_buses), rand(rng, 1:n_buses))
    end
    n_lines = length(line_from)

    susceptance = rand(rng, Uniform(6.0, 24.0), n_lines)

    # The first B-1 lines form the planted connected topology. Close a few mesh
    # lines as well, but retain open candidates whenever the graph has extras.
    planted_closed = falses(n_lines)
    planted_closed[1:(n_buses - 1)] .= true
    for l in n_buses:n_lines
        planted_closed[l] = rand(rng) < 0.30
    end
    if n_lines > n_buses - 1 && all(planted_closed)
        planted_closed[end] = false
    end

    # Generation and a balanced nominal load used to form a DC witness.
    gen_bus = rand(rng, 1:n_buses, n_generators)
    pmin = zeros(Float64, n_generators)
    pmax = rand(rng, Uniform(25.0, 90.0), n_generators)
    gen_cost = rand(rng, Uniform(12.0, 65.0), n_generators)
    total_nominal_demand = sum(pmax) * rand(rng, Uniform(0.42, 0.68))
    load_weights = rand(rng, Dirichlet(fill(1.5, n_buses)))
    nominal_demand = total_nominal_demand .* load_weights
    dispatch_witness = total_nominal_demand .* (pmax ./ sum(pmax))

    injection = -copy(nominal_demand)
    for g in 1:n_generators
        injection[gen_bus[g]] += dispatch_witness[g]
    end

    # Solve the reduced Laplacian on the planted connected topology.
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    for l in 1:n_lines
        planted_closed[l] || continue
        i = line_from[l]
        j = line_to[l]
        b = susceptance[l]
        append!(rows, (i, j, i, j))
        append!(cols, (i, j, j, i))
        append!(vals, (b, b, -b, -b))
    end
    laplacian = sparse(rows, cols, vals, n_buses, n_buses)
    ref_bus = 1
    keep = collect(2:n_buses)
    theta_witness = zeros(Float64, n_buses)
    theta_witness[keep] = laplacian[keep, keep] \ injection[keep]
    angle_limit = max(0.20, 1.25 * maximum(abs.(theta_witness)) + 1.0e-4)

    witness_flow = zeros(Float64, n_lines)
    for l in 1:n_lines
        if planted_closed[l]
            witness_flow[l] = susceptance[l] *
                              (theta_witness[line_from[l]] - theta_witness[line_to[l]])
        end
    end

    average_load = total_nominal_demand / max(n_lines, 1)
    line_limit = [max(1.0, average_load * rand(rng, Uniform(0.45, 1.35)))
                  for _ in 1:n_lines]
    if feasibility_status == feasible
        for l in 1:n_lines
            planted_closed[l] || continue
            line_limit[l] = max(line_limit[l], 1.20 * abs(witness_flow[l]) + 0.5)
        end
    end

    # Total-generation shortfall is independent of topology and integrality.
    demand = copy(nominal_demand)
    if feasibility_status == infeasible
        demand .*= sum(pmax) * rand(rng, Uniform(1.12, 1.35)) / sum(demand)
    end

    switching_cost = rand(rng, Uniform(0.05, 1.5), n_lines)
    flow_big_m = [2.0 * susceptance[l] * angle_limit for l in 1:n_lines]

    return OptimalTransmissionSwitchingProblem(
        n_buses, n_lines, n_generators,
        line_from, line_to, susceptance, line_limit, flow_big_m,
        gen_bus, gen_cost, pmin, pmax, demand, switching_cost,
        angle_limit, ref_bus,
    )
end

"""Build the deterministic JuMP formulation for an OTS instance."""
function build_model(prob::OptimalTransmissionSwitchingProblem)
    model = Model()
    B = prob.n_buses
    L = prob.n_lines
    G = prob.n_generators

    @variable(model, prob.pmin[g] <= p[g in 1:G] <= prob.pmax[g])
    @variable(model, -prob.angle_limit <= theta[1:B] <= prob.angle_limit)
    @variable(model, flow[1:L])
    @variable(model, line_on[1:L], Bin)

    @objective(model, Min,
        sum(prob.gen_cost[g] * p[g] for g in 1:G) +
        sum(prob.switching_cost[l] * line_on[l] for l in 1:L)
    )

    @constraint(model, theta[prob.ref_bus] == 0)
    for l in 1:L
        @constraint(model, flow[l] <= prob.line_limit[l] * line_on[l])
        @constraint(model, flow[l] >= -prob.line_limit[l] * line_on[l])

        dc_residual = flow[l] - prob.susceptance[l] *
                      (theta[prob.line_from[l]] - theta[prob.line_to[l]])
        @constraint(model, dc_residual <= prob.flow_big_m[l] * (1 - line_on[l]))
        @constraint(model, dc_residual >= -prob.flow_big_m[l] * (1 - line_on[l]))
    end

    generators_at = [Int[] for _ in 1:B]
    outgoing = [Int[] for _ in 1:B]
    incoming = [Int[] for _ in 1:B]
    for g in 1:G
        push!(generators_at[prob.gen_bus[g]], g)
    end
    for l in 1:L
        push!(outgoing[prob.line_from[l]], l)
        push!(incoming[prob.line_to[l]], l)
    end
    for bus in 1:B
        balance = AffExpr(-prob.demand[bus])
        for g in generators_at[bus]
            add_to_expression!(balance, 1.0, p[g])
        end
        for l in outgoing[bus]
            add_to_expression!(balance, -1.0, flow[l])
        end
        for l in incoming[bus]
            add_to_expression!(balance, 1.0, flow[l])
        end
        @constraint(model, balance == 0)
    end

    return model
end

register_variant(
    :energy,
    :optimal_transmission_switching,
    OptimalTransmissionSwitchingProblem,
    "DC optimal transmission switching with binary line status, bound-derived disjunctions, thermal limits, and nodal dispatch",
)
