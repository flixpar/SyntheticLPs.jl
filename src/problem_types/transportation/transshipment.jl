using JuMP
using Random

"""
    TransshipmentProblem <: ProblemGenerator

Generator for transshipment transportation problems with intermediate hub nodes.

# Overview
Extends the classic transportation problem with a set of intermediate
transshipment (hub) nodes. Goods can flow from sources directly to destinations,
or be routed through hubs (source -> hub, then hub -> destination). The objective
minimizes total shipping cost across all three arc sets. Each source is limited by
its supply (counting both direct and to-hub flow), each destination must have its
demand met (from direct and from-hub flow), and each hub conserves flow (inbound
from sources equals outbound to destinations). Per-arc capacities limit the flow
that can pass through hub legs.

# Fields
- `n_sources::Int`: Number of supply sources
- `n_destinations::Int`: Number of demand destinations
- `n_hubs::Int`: Number of intermediate transshipment (hub) nodes
- `supplies::Vector{Int}`: Supply available at each source
- `demands::Vector{Int}`: Demand required at each destination
- `cost_direct::Matrix{Float64}`: Cost per unit on each source -> destination arc (n_sources × n_destinations)
- `cost_to_hub::Matrix{Float64}`: Cost per unit on each source -> hub arc (n_sources × n_hubs)
- `cost_from_hub::Matrix{Float64}`: Cost per unit on each hub -> destination arc (n_hubs × n_destinations)
- `cap_to_hub::Matrix{Float64}`: Capacity on each source -> hub arc (n_sources × n_hubs)
- `cap_from_hub::Matrix{Float64}`: Capacity on each hub -> destination arc (n_hubs × n_destinations)
"""
struct TransshipmentProblem <: ProblemGenerator
    n_sources::Int
    n_destinations::Int
    n_hubs::Int
    supplies::Vector{Int}
    demands::Vector{Int}
    cost_direct::Matrix{Float64}
    cost_to_hub::Matrix{Float64}
    cost_from_hub::Matrix{Float64}
    cap_to_hub::Matrix{Float64}
    cap_from_hub::Matrix{Float64}
end

"""
    TransshipmentProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a transshipment problem instance.

Variable-count formula (decision variables created by `build_model`):

    total = n_sources*n_destinations   (x_direct)
          + n_sources*n_hubs           (x_to_hub)
          + n_hubs*n_destinations      (x_from_hub)

The constructor sizes `n_sources`, `n_destinations`, and `n_hubs` together so this
full total lands near `target_variables` (not just the direct block).

# Arguments
- `target_variables`: Target number of decision variables across all three flow arc sets
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function TransshipmentProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # --- Dimension sizing ---
    # total = n_src*n_dst + n_src*n_hub + n_hub*n_dst
    # Choose n_hub ≈ h_frac * n_dst, and n_src ≈ n_dst (a roughly square grid).
    # With n_src = n_dst = m and n_hub = round(h_frac*m):
    #   total ≈ m^2 + 2*h_frac*m^2 = m^2 * (1 + 2*h_frac)
    # => m ≈ sqrt(target / (1 + 2*h_frac))
    h_frac = 0.3 + 0.2 * rand()  # hubs are 30%-50% of destinations
    m = max(2, round(Int, sqrt(target_variables / (1 + 2 * h_frac))))

    # Add mild asymmetry between sources and destinations while keeping the total near target.
    ratio = 0.8 + 0.4 * rand()  # 0.8 .. 1.2
    n_sources = max(2, round(Int, m * ratio))
    n_destinations = max(2, round(Int, m / ratio))
    n_hubs = max(1, round(Int, h_frac * n_destinations))

    # Fine-tune destinations so the FULL total (all three arc sets) tracks target.
    # total = n_src*n_dst + n_src*n_hub + n_hub*n_dst
    #       = n_dst*(n_src + n_hub) + n_src*n_hub
    # Solve for n_dst given current n_src, n_hub.
    denom = n_sources + n_hubs
    n_destinations = max(2, round(Int, (target_variables - n_sources * n_hubs) / denom))
    # n_hubs depends on n_destinations; recompute and re-balance once.
    n_hubs = max(1, round(Int, h_frac * n_destinations))
    denom = n_sources + n_hubs
    n_destinations = max(2, round(Int, (target_variables - n_sources * n_hubs) / denom))

    total_vars = n_sources * n_destinations + n_sources * n_hubs + n_hubs * n_destinations

    # --- Scale-dependent parameter ranges ---
    if total_vars <= 250
        supply_lo, supply_hi = rand(60:100), rand(200:400)
        demand_lo, demand_hi = rand(30:60), rand(120:220)
        cost_lo, cost_hi = 5.0, 40.0
    elseif total_vars <= 1000
        supply_lo, supply_hi = rand(150:400), rand(1200:3500)
        demand_lo, demand_hi = rand(80:200), rand(700:2000)
        cost_lo, cost_hi = 10.0, 90.0
    else
        supply_lo, supply_hi = rand(600:1500), rand(6000:30000)
        demand_lo, demand_hi = rand(300:1000), rand(3500:18000)
        cost_lo, cost_hi = 20.0, 250.0
    end

    # --- Demands and supplies ---
    demands = rand(demand_lo:demand_hi, n_destinations)
    supplies = rand(supply_lo:supply_hi, n_sources)

    # --- Arc costs (hub legs slightly cheaper per leg to make routing attractive) ---
    cost_direct = cost_lo .+ (cost_hi - cost_lo) .* rand(n_sources, n_destinations)
    cost_to_hub = (cost_lo .+ (cost_hi - cost_lo) .* rand(n_sources, n_hubs)) .* 0.6
    cost_from_hub = (cost_lo .+ (cost_hi - cost_lo) .* rand(n_hubs, n_destinations)) .* 0.6

    # Helper to distribute an integer addition across a vector (keeps reproducibility).
    function distribute_additions!(vec::Vector{Int}, amount::Int)
        amount <= 0 && return
        w = rand(length(vec))
        base = floor.(Int, (w ./ sum(w)) .* amount)
        remainder = amount - sum(base)
        if remainder > 0
            for idx in randperm(length(vec))[1:min(remainder, length(vec))]
                base[idx] += 1
            end
        end
        vec .+= base
    end

    total_supply = sum(supplies)
    total_demand = sum(demands)

    # --- Resolve feasibility intent (unknown -> natural instance, no forcing) ---
    if feasibility_status == feasible
        # Ensure aggregate supply covers demand with a clear margin.
        if total_supply < total_demand * 1.05
            shortage = ceil(Int, total_demand * 1.1) - total_supply
            distribute_additions!(supplies, shortage)
            total_supply = sum(supplies)
        end
    elseif feasibility_status == infeasible
        # Force a deterministic contradiction: aggregate demand strictly exceeds supply.
        # This makes the destination constraints unsatisfiable regardless of routing/hub caps.
        target_margin = max(1, round(Int, (0.05 + 0.05 * rand()) * max(total_supply, 1)))
        missing = (total_supply + target_margin) - total_demand
        if missing > 0
            distribute_additions!(demands, missing)
        end
        total_demand = sum(demands)
    end
    # unknown: leave supplies/demands as sampled (a natural, possibly-either instance).

    # --- Hub-leg capacities ---
    # Sized so the hub legs are a genuinely BINDING constraint, not decorative: a
    # single hub arc carries only a fraction of total demand, so routing a large
    # share through hubs requires spreading across several of them (and the per-arc
    # caps frequently bind). Feasibility is never blocked because the direct
    # source->destination arcs are uncapped and provide a complete fallback.
    # Per-arc capacity ~ (total_demand / n_hubs) so all hubs together can carry
    # roughly all demand, but no one arc can.
    hub_share = total_demand / max(n_hubs, 1)
    per_arc_cap_to = hub_share * (0.6 + 0.5 * rand())
    per_arc_cap_from = hub_share * (0.6 + 0.5 * rand())
    cap_to_hub = fill(0.0, n_sources, n_hubs)
    cap_from_hub = fill(0.0, n_hubs, n_destinations)
    for i in 1:n_sources, t in 1:n_hubs
        cap_to_hub[i, t] = per_arc_cap_to * (0.8 + 0.4 * rand())
    end
    for t in 1:n_hubs, j in 1:n_destinations
        cap_from_hub[t, j] = per_arc_cap_from * (0.8 + 0.4 * rand())
    end

    return TransshipmentProblem(
        n_sources, n_destinations, n_hubs,
        supplies, demands,
        cost_direct, cost_to_hub, cost_from_hub,
        cap_to_hub, cap_from_hub,
    )
end

"""
    build_model(prob::TransshipmentProblem)

Build a JuMP model for the transshipment problem. Deterministic — uses only data
from the struct fields.

Decision variables:
- `x_direct[i, j]`: flow shipped directly from source `i` to destination `j`
- `x_to_hub[i, t]`: flow shipped from source `i` to hub `t`
- `x_from_hub[t, j]`: flow shipped from hub `t` to destination `j`

# Returns
- `model`: The JuMP model
"""
function build_model(prob::TransshipmentProblem)
    model = Model()

    S = prob.n_sources
    D = prob.n_destinations
    H = prob.n_hubs

    # Variables (total = S*D + S*H + H*D)
    @variable(model, x_direct[1:S, 1:D] >= 0)
    @variable(model, x_to_hub[1:S, 1:H] >= 0)
    @variable(model, x_from_hub[1:H, 1:D] >= 0)

    # Objective: minimize total shipping cost over all arc sets
    @objective(model, Min,
        sum(prob.cost_direct[i, j] * x_direct[i, j] for i in 1:S, j in 1:D) +
        sum(prob.cost_to_hub[i, t] * x_to_hub[i, t] for i in 1:S, t in 1:H) +
        sum(prob.cost_from_hub[t, j] * x_from_hub[t, j] for t in 1:H, j in 1:D)
    )

    # Supply constraints: direct + to-hub flow out of each source <= supply
    for i in 1:S
        @constraint(model,
            sum(x_direct[i, j] for j in 1:D) +
            sum(x_to_hub[i, t] for t in 1:H) <= prob.supplies[i])
    end

    # Demand constraints: direct + from-hub flow into each destination >= demand
    for j in 1:D
        @constraint(model,
            sum(x_direct[i, j] for i in 1:S) +
            sum(x_from_hub[t, j] for t in 1:H) >= prob.demands[j])
    end

    # Hub flow conservation: inbound from sources == outbound to destinations
    for t in 1:H
        @constraint(model,
            sum(x_to_hub[i, t] for i in 1:S) ==
            sum(x_from_hub[t, j] for j in 1:D))
    end

    # Hub-leg capacity constraints
    for i in 1:S, t in 1:H
        @constraint(model, x_to_hub[i, t] <= prob.cap_to_hub[i, t])
    end
    for t in 1:H, j in 1:D
        @constraint(model, x_from_hub[t, j] <= prob.cap_from_hub[t, j])
    end

    return model
end

# Register the variant
register_variant(
    :transportation,
    :transshipment,
    TransshipmentProblem,
    "Transshipment problem routing goods from sources to destinations directly or through capacitated intermediate hub nodes at minimum cost",
)
