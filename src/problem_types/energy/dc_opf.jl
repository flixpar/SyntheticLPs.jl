using JuMP
using Random
using Distributions
using LinearAlgebra
using SparseArrays

"""
    DCOptimalPowerFlowProblem <: ProblemGenerator

Generator for DC optimal power flow / economic dispatch problems on a
transmission network.

# Overview
Models least-cost power dispatch over a meshed transmission grid under the linear
"DC" power-flow approximation. The decisions are the generator outputs `p[g]`,
the bus voltage angles `θ[b]`, and the line flows `f[l]`. The objective minimizes
total generation cost. Constraints enforce nodal power balance (Kirchhoff's
current law) at every bus, the DC flow definition `f[l] = B_l (θ_from − θ_to)`
(a susceptance-weighted, non-unimodular network coupling), thermal line limits,
generator output limits, and a fixed reference-bus angle.

This is a genuine LP with a distinctive structure — a flow network coupled to
physical angle variables through non-±1 coefficients — that is absent from the
purely combinatorial network generators. DC-OPF / security-constrained economic
dispatch is one of the highest-volume real-world LP applications.

Feasibility is governed by the generation–load balance: because the nodal
balance rows sum to `Σp = Σdemand`, the problem is feasible only when total
demand lies within `[Σpmin, Σpmax]` and the network can route the resulting
flows within line limits. The feasible case is guaranteed by constructing an
explicit DC-power-flow witness (solving the reduced network Laplacian) and
sizing the line limits to accommodate it.

# Fields
- `n_buses::Int`: Number of buses (nodes)
- `n_lines::Int`: Number of transmission lines (edges)
- `n_generators::Int`: Number of generators
- `line_from::Vector{Int}`: "From" bus of each line
- `line_to::Vector{Int}`: "To" bus of each line
- `susceptance::Vector{Float64}`: Susceptance of each line
- `line_limit::Vector{Float64}`: Thermal flow limit of each line
- `gen_bus::Vector{Int}`: Bus at which each generator is located
- `gen_cost::Vector{Float64}`: Linear generation cost per unit output
- `pmin::Vector{Float64}`: Minimum output of each generator
- `pmax::Vector{Float64}`: Maximum output of each generator
- `demand::Vector{Float64}`: Load at each bus
- `ref_bus::Int`: Reference (slack) bus whose angle is fixed to zero
"""
struct DCOptimalPowerFlowProblem <: ProblemGenerator
    n_buses::Int
    n_lines::Int
    n_generators::Int
    line_from::Vector{Int}
    line_to::Vector{Int}
    susceptance::Vector{Float64}
    line_limit::Vector{Float64}
    gen_bus::Vector{Int}
    gen_cost::Vector{Float64}
    pmin::Vector{Float64}
    pmax::Vector{Float64}
    demand::Vector{Float64}
    ref_bus::Int
end

"""
    DCOptimalPowerFlowProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a DC optimal power flow instance.

Variables: `p[g]` (generation), `θ[b]` (bus angles), and `f[l]` (line flows), for
a total of `n_generators + n_buses + n_lines`.

# Arguments
- `target_variables`: Target number of variables
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function DCOptimalPowerFlowProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # --- Dimension sizing ---
    # Variables = n_generators + n_buses + n_lines ≈ B * (1 + edge_factor + gen_frac).
    edge_factor = rand(Uniform(1.3, 2.2))   # lines per bus (meshed but sparse)
    gen_frac = rand(Uniform(0.3, 0.5))      # generators per bus
    n_buses = max(3, round(Int, target_variables / (1 + edge_factor + gen_frac)))

    n_lines = max(n_buses - 1, round(Int, edge_factor * n_buses))
    n_generators = max(2, round(Int, gen_frac * n_buses))

    B = n_buses

    # --- Network topology: spanning tree (connectivity) + extra meshing edges ---
    line_from = Int[]
    line_to = Int[]
    edge_set = Set{Tuple{Int,Int}}()
    function add_edge!(a::Int, b::Int)
        a == b && return false
        key = a < b ? (a, b) : (b, a)
        key in edge_set && return false
        push!(edge_set, key)
        push!(line_from, a)
        push!(line_to, b)
        return true
    end

    for k in 2:B
        add_edge!(k, rand(1:(k - 1)))   # connect each new bus to an existing one
    end
    attempts = 0
    while length(line_from) < n_lines && attempts < 50 * n_lines
        add_edge!(rand(1:B), rand(1:B))
        attempts += 1
    end
    n_lines = length(line_from)

    susceptance = rand(Uniform(5.0, 30.0), n_lines)

    # --- Generators ---
    gen_bus = rand(1:B, n_generators)
    pmin = rand(Uniform(0.0, 5.0), n_generators)
    pmax = pmin .+ rand(Uniform(20.0, 80.0), n_generators)
    gen_cost = rand(Uniform(10.0, 60.0), n_generators)

    sum_pmin = sum(pmin)
    sum_pmax = sum(pmax)

    # --- Base load shape (scaled later to a target total demand) ---
    base_demand = rand(Uniform(1.0, 12.0), B)
    base_total = sum(base_demand)

    ref_bus = 1

    # --- Feasibility handling (governed by generation–load balance) ---
    actual_status = feasibility_status
    if feasibility_status == unknown
        actual_status = rand() < 0.7 ? feasible : infeasible
    end

    if actual_status == feasible
        frac = rand(Uniform(0.3, 0.7))
        total_demand = sum_pmin + frac * (sum_pmax - sum_pmin)
    else
        # Demand exceeds total generation capacity => balance is unsatisfiable.
        total_demand = sum_pmax * rand(Uniform(1.1, 1.4))
    end

    demand = base_demand .* (total_demand / base_total)

    # --- Line limits ---
    base_limit = [(total_demand / max(1, n_lines)) * rand(Uniform(2.0, 6.0)) +
                  rand(Uniform(1.0, 5.0)) for _ in 1:n_lines]
    line_limit = copy(base_limit)

    if actual_status == feasible
        # Build a DC-power-flow witness and widen line limits to accommodate it.
        # Witness dispatch: start at pmin, distribute the residual over headroom.
        p_w = copy(pmin)
        residual = total_demand - sum_pmin
        headroom = pmax .- pmin
        total_head = sum(headroom)
        if total_head > 0
            p_w .+= residual .* (headroom ./ total_head)
        end

        # Nodal net injection.
        inj = -copy(demand)
        for g in 1:n_generators
            inj[gen_bus[g]] += p_w[g]
        end

        # Reduced network Laplacian solve (reference bus removed). The topology is
        # sparse (n_lines ≈ a small multiple of n_buses), so assemble a sparse
        # Laplacian to keep the witness solve from dominating data generation on
        # large networks.
        rows = Int[]
        cols = Int[]
        vals = Float64[]
        for l in 1:n_lines
            a = line_from[l]
            b = line_to[l]
            s = susceptance[l]
            append!(rows, (a, b, a, b))
            append!(cols, (a, b, b, a))
            append!(vals, (s, s, -s, -s))   # duplicate diagonal entries are summed
        end
        L = sparse(rows, cols, vals, B, B)
        keep = setdiff(1:B, [ref_bus])
        θ = zeros(Float64, B)
        θ[keep] = L[keep, keep] \ inj[keep]

        for l in 1:n_lines
            f_w = susceptance[l] * (θ[line_from[l]] - θ[line_to[l]])
            line_limit[l] = max(base_limit[l], abs(f_w) * rand(Uniform(1.2, 2.0)))
        end
    end

    return DCOptimalPowerFlowProblem(
        B, n_lines, n_generators, line_from, line_to, susceptance, line_limit,
        gen_bus, gen_cost, pmin, pmax, demand, ref_bus,
    )
end

"""
    build_model(prob::DCOptimalPowerFlowProblem)

Build a JuMP model for the DC optimal power flow problem. Deterministic — uses
only data from the struct fields.

# Returns
- `model`: The JuMP model
"""
function build_model(prob::DCOptimalPowerFlowProblem)
    model = Model()

    B = prob.n_buses
    L = prob.n_lines
    G = prob.n_generators

    # Generation (bounded), bus angles (free), and line flows (thermally bounded).
    @variable(model, prob.pmin[g] <= p[g in 1:G] <= prob.pmax[g])
    @variable(model, θ[1:B])
    @variable(model, -prob.line_limit[l] <= f[l in 1:L] <= prob.line_limit[l])

    # Objective: minimize total generation cost.
    @objective(model, Min, sum(prob.gen_cost[g] * p[g] for g in 1:G))

    # Reference-bus angle fixed to zero.
    @constraint(model, θ[prob.ref_bus] == 0)

    # DC flow definition: f[l] = B_l (θ_from − θ_to).
    for l in 1:L
        @constraint(model, f[l] == prob.susceptance[l] * (θ[prob.line_from[l]] - θ[prob.line_to[l]]))
    end

    # Nodal power balance: generation − load = net outflow. Expressions are built
    # explicitly so buses with no generator (or only incoming/outgoing lines) are
    # handled without relying on empty-sum behavior inside the macro.
    gens_at = [Int[] for _ in 1:B]
    for g in 1:G
        push!(gens_at[prob.gen_bus[g]], g)
    end
    lines_from = [Int[] for _ in 1:B]
    lines_to = [Int[] for _ in 1:B]
    for l in 1:L
        push!(lines_from[prob.line_from[l]], l)
        push!(lines_to[prob.line_to[l]], l)
    end
    for b in 1:B
        injection = AffExpr(0.0)
        for g in gens_at[b]
            add_to_expression!(injection, 1.0, p[g])
        end
        add_to_expression!(injection, -prob.demand[b])

        net_outflow = AffExpr(0.0)
        for l in lines_from[b]
            add_to_expression!(net_outflow, 1.0, f[l])
        end
        for l in lines_to[b]
            add_to_expression!(net_outflow, -1.0, f[l])
        end

        @constraint(model, injection == net_outflow)
    end

    return model
end

# Register the variant (a second variant of the energy category).
register_variant(
    :energy,
    :dc_opf,
    DCOptimalPowerFlowProblem,
    "DC optimal power flow / economic dispatch over a transmission network (nodal balance, susceptance-weighted flows, thermal limits)",
)
