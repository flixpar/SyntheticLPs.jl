using JuMP
using Random
using Distributions
using StatsBase

"""
    LandUseProblem <: ProblemGenerator

Generator for land use optimization problems that maximize economic benefits by allocating land parcels
to zoning types while satisfying infrastructure constraints, environmental regulations, and adjacency requirements.

This problem models realistic land use planning with:
- Multiple zoning types (residential, commercial, industrial, agricultural, conservation)
- Resource capacity constraints (water, sewage, transportation, power)
- Environmental restrictions on parcel-zoning combinations
- Adjacency constraints (e.g., industrial cannot be adjacent to residential)
- Minimum zoning requirements for diverse development

# Fields
All data generated in constructor based on target_variables and feasibility_status:
- `n_parcels::Int`: Number of land parcels
- `n_zoning_types::Int`: Number of zoning types
- `n_resources::Int`: Number of resource constraints
- `parcel_sizes::Vector{Float64}`: Size of each parcel in acres
- `development_costs::Matrix{Float64}`: Development cost per acre for each (parcel, zoning) combination
- `revenues::Matrix{Float64}`: Revenue per acre for each (parcel, zoning) combination
- `resource_consumption::Matrix{Float64}`: Resource consumption per acre for each (zoning, resource) pair
- `resource_capacities::Vector{Float64}`: Total capacity for each resource
- `environmental_restrictions::Matrix{Bool}`: Whether parcel i is restricted from zoning j
- `adjacency_matrix::Matrix{Bool}`: Which parcels are adjacent
- `zoning_names::Vector{String}`: Names of zoning types
- `resource_names::Vector{String}`: Names of resources
- `min_counts_by_type::Vector{Int}`: Minimum number of parcels required for certain zoning types
- `zoning_adjacency_constraints::Bool`: Whether to include adjacency constraints
- `minimum_zoning_requirements::Bool`: Whether minimum zoning requirements are active
"""
struct LandUseProblem <: ProblemGenerator
    n_parcels::Int
    n_zoning_types::Int
    n_resources::Int
    parcel_sizes::Vector{Float64}
    development_costs::Matrix{Float64}
    revenues::Matrix{Float64}
    resource_consumption::Matrix{Float64}
    resource_capacities::Vector{Float64}
    environmental_restrictions::Matrix{Bool}
    adjacency_matrix::Matrix{Bool}
    zoning_names::Vector{String}
    resource_names::Vector{String}
    min_counts_by_type::Vector{Int}
    zoning_adjacency_constraints::Bool
    minimum_zoning_requirements::Bool
end

"""
    LandUseProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a land use problem instance with sophisticated witness-based feasibility construction.

# Sophisticated Feasibility Logic Preserved:
- **Witness construction**: For feasible instances, constructs a concrete feasible assignment respecting all constraints
- **Adjacency-aware assignment**: Carefully assigns type 1 (residential) and type 3 (industrial) to avoid conflicts
- **Edge pruning**: If adjacency conflicts occur, minimally prunes adjacency edges to allow feasible assignment
- **Environmental relaxation**: Ensures each parcel has at least one allowed zoning type
- **Minimum requirement enforcement**: Ensures minimum zoning requirements are achievable given environmental restrictions
- **Capacity tightening**: For feasible instances, adjusts resource capacities to admit the witness with small slack
- **Lower bound infeasibility**: For infeasible instances, sets capacities below provable lower bounds

# Arguments
- `target_variables`: Target number of variables (n_parcels × n_zoning_types)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function LandUseProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Determine problem scale based on target variables
    if target_variables <= 250
        # Small: local/municipal
        n_zoning_types = rand(3:5)
        n_resources = rand(3:5)
        development_cost_scale = rand(50000:150000)
        revenue_scale = rand(20000:80000)
        infrastructure_capacity_factor = rand(Uniform(0.6, 0.8))
        environmental_constraint_prob = rand(Uniform(0.2, 0.4))
    elseif target_variables <= 1000
        # Medium: regional
        n_zoning_types = rand(4:8)
        n_resources = rand(4:6)
        development_cost_scale = rand(75000:250000)
        revenue_scale = rand(40000:120000)
        infrastructure_capacity_factor = rand(Uniform(0.65, 0.85))
        environmental_constraint_prob = rand(Uniform(0.25, 0.45))
    else
        # Large: state/national
        n_zoning_types = rand(5:12)
        n_resources = rand(5:8)
        development_cost_scale = rand(100000:500000)
        revenue_scale = rand(60000:200000)
        infrastructure_capacity_factor = rand(Uniform(0.7, 0.9))
        environmental_constraint_prob = rand(Uniform(0.3, 0.5))
    end

    # Calculate n_parcels to achieve target
    n_parcels = round(Int, target_variables / n_zoning_types)
    n_parcels = max(2, n_parcels)

    zoning_adjacency_constraints = rand() < 0.8
    minimum_zoning_requirements = rand() < 0.9

    # Generate parcel characteristics
    parcel_sizes = rand(LogNormal(log(5), 0.8), n_parcels)
    parcel_sizes = max.(parcel_sizes, 0.1)

    # Zoning type names
    zoning_names = ["Residential", "Commercial", "Industrial", "Agricultural", "Conservation"]
    if n_zoning_types > 5
        append!(zoning_names, ["Mixed_Use", "Recreational", "Institutional", "Transportation", "Special"])
    end
    zoning_names = zoning_names[1:n_zoning_types]

    # Resource names
    resource_names = ["Water", "Sewage", "Transportation", "Power"]
    if n_resources > 4
        append!(resource_names, ["Internet", "Gas", "Environmental", "Emergency"])
    end
    resource_names = resource_names[1:n_resources]

    # Development costs
    cost_multipliers = [1.0, 2.5, 3.0, 0.5, 0.1]
    if n_zoning_types > 5
        append!(cost_multipliers, [2.0, 1.5, 1.8, 4.0, 3.5])
    end
    cost_multipliers = cost_multipliers[1:n_zoning_types]

    development_costs = zeros(n_parcels, n_zoning_types)
    for i in 1:n_parcels
        location_factor = rand(Gamma(2, 0.5))
        for j in 1:n_zoning_types
            base_cost = development_cost_scale * cost_multipliers[j]
            development_costs[i, j] = base_cost * location_factor * rand(Normal(1.0, 0.2))
            development_costs[i, j] = max(development_costs[i, j], base_cost * 0.1)
        end
    end

    # Revenues
    revenue_multipliers = [1.5, 4.0, 2.0, 0.8, 0.2]
    if n_zoning_types > 5
        append!(revenue_multipliers, [3.0, 1.0, 0.5, 0.1, 2.5])
    end
    revenue_multipliers = revenue_multipliers[1:n_zoning_types]

    revenues = zeros(n_parcels, n_zoning_types)
    for i in 1:n_parcels
        location_revenue_factor = rand(Gamma(2, 0.6))
        for j in 1:n_zoning_types
            base_revenue = revenue_scale * revenue_multipliers[j]
            revenues[i, j] = base_revenue * location_revenue_factor * rand(Normal(1.0, 0.3))
            revenues[i, j] = max(revenues[i, j], 0.0)
        end
    end

    # Resource consumption
    resource_consumption = zeros(n_zoning_types, n_resources)
    consumption_patterns = [
        [2.0, 1.5, 1.0, 1.5],  # Residential
        [1.0, 0.8, 3.0, 2.0],  # Commercial
        [0.5, 2.0, 2.5, 4.0],  # Industrial
        [3.0, 0.5, 0.5, 0.5],  # Agricultural
        [0.1, 0.1, 0.1, 0.1]   # Conservation
    ]

    for j in 1:n_zoning_types
        for k in 1:n_resources
            if j <= length(consumption_patterns) && k <= length(consumption_patterns[j])
                base_consumption = consumption_patterns[j][k]
            else
                base_consumption = rand(Uniform(0.5, 3.0))
            end
            resource_consumption[j, k] = base_consumption * rand(Gamma(2, 0.5))
        end
    end

    # Resource capacities (initial)
    total_demand_estimate = sum(parcel_sizes) * mean(resource_consumption, dims=1)
    resource_capacities = vec(total_demand_estimate) .* infrastructure_capacity_factor .* rand(Uniform(0.8, 1.2), n_resources)

    # Environmental restrictions
    environmental_restrictions = zeros(Bool, n_parcels, n_zoning_types)
    for i in 1:n_parcels
        if rand() < environmental_constraint_prob
            max_restrict = max(1, min(3, n_zoning_types - 1))
            num_to_restrict = rand(1:max_restrict)
            restricted_types = sample(1:n_zoning_types, num_to_restrict, replace=false)
            environmental_restrictions[i, restricted_types] .= true
        end
    end

    # Adjacency matrix
    adjacency_matrix = zeros(Bool, n_parcels, n_parcels)
    if zoning_adjacency_constraints && n_parcels > 1
        for i in 1:n_parcels
            n_neighbors = rand(2:min(4, n_parcels-1))
            neighbors = sample(setdiff(1:n_parcels, [i]), n_neighbors, replace=false)
            adjacency_matrix[i, neighbors] .= true
            adjacency_matrix[neighbors, i] .= true
        end
    end

    # Type consumption scores for tie-breaking
    type_consumption_score = [sum(resource_consumption[j, k] for k in 1:n_resources) for j in 1:n_zoning_types]

    # Ensure every parcel has at least one allowed zoning
    allowed_sets = Vector{Vector{Int}}(undef, n_parcels)
    for i in 1:n_parcels
        allowed = [j for j in 1:n_zoning_types if !environmental_restrictions[i, j]]
        if isempty(allowed)
            jbest = argmin(type_consumption_score)
            environmental_restrictions[i, jbest] = false
            allowed = [jbest]
        end
        allowed_sets[i] = allowed
    end

    # Compute minimum zoning requirements
    num_required_types = minimum([3, n_zoning_types, n_parcels])
    min_counts_by_type = Int[]
    if minimum_zoning_requirements
        base_min = max(1, round(Int, n_parcels * 0.1))
        total_base = base_min * num_required_types
        if total_base <= n_parcels
            min_counts_by_type = fill(base_min, num_required_types)
        else
            per = max(1, fld(n_parcels, num_required_types))
            min_counts_by_type = fill(per, num_required_types)
            remaining = n_parcels - per * num_required_types
            idx = 1
            while remaining > 0 && idx <= num_required_types
                min_counts_by_type[idx] += 1
                remaining -= 1
                idx += 1
            end
        end

        # Ensure environmental restrictions allow meeting minimums
        for j in 1:num_required_types
            allowed_count_j = count(i -> !environmental_restrictions[i, j], 1:n_parcels)
            deficit = max(0, min_counts_by_type[j] - allowed_count_j)
            if deficit > 0
                candidates = [i for i in 1:n_parcels if environmental_restrictions[i, j]]
                sort!(candidates, by = i -> type_consumption_score[j])
                for t in 1:min(deficit, length(candidates))
                    environmental_restrictions[candidates[t], j] = false
                end
            end
        end

        # Rebuild allowed_sets
        for i in 1:n_parcels
            allowed_sets[i] = [j for j in 1:n_zoning_types if !environmental_restrictions[i, j]]
            if isempty(allowed_sets[i])
                jbest = argmin(type_consumption_score)
                environmental_restrictions[i, jbest] = false
                allowed_sets[i] = [jbest]
            end
        end
    else
        min_counts_by_type = Int[]
    end

    # SOPHISTICATED FEASIBILITY ENFORCEMENT
    if feasibility_status == feasible
        # WITNESS CONSTRUCTION with adjacency-aware assignment
        neighbors = Vector{Vector{Int}}(undef, n_parcels)
        if zoning_adjacency_constraints && n_parcels > 1
            for i in 1:n_parcels
                neighbors[i] = [i2 for i2 in 1:n_parcels if adjacency_matrix[i, i2]]
            end
        else
            for i in 1:n_parcels
                neighbors[i] = Int[]
            end
        end

        assignment = fill(0, n_parcels)
        req_counts = copy(min_counts_by_type)

        # Helper to select disjoint set respecting adjacency
        function select_disjoint_set(candidates::Vector{Int}, restricted_neighbors::Vector{Vector{Int}}, quota::Int, forbidden_neighbors::Set{Int})
            S = Int[]
            for i in candidates
                if length(S) == quota
                    break
                end
                if i in forbidden_neighbors
                    continue
                end
                push!(S, i)
                for nb in restricted_neighbors[i]
                    push!(forbidden_neighbors, nb)
                end
            end
            return S, forbidden_neighbors
        end

        # Build candidate sets
        cand1 = [i for i in 1:n_parcels if 1 <= n_zoning_types && (1 in allowed_sets[i])]
        cand2 = [i for i in 1:n_parcels if 2 <= n_zoning_types && (2 in allowed_sets[i])]
        cand3 = [i for i in 1:n_parcels if 3 <= n_zoning_types && (3 in allowed_sets[i])]

        function preference_order(i::Int, target_type::Int)
            allowed_required = Set([t for t in 1:minimum([3, n_zoning_types]) if t in allowed_sets[i]])
            uniqueness = (length(allowed_required) == 1 && (target_type in allowed_required)) ? 0 : 1
            return (uniqueness, type_consumption_score[target_type])
        end
        sort!(cand1, by = i -> preference_order(i, 1))
        sort!(cand2, by = i -> preference_order(i, 2))
        sort!(cand3, by = i -> preference_order(i, 3))

        used = Set{Int}()
        forbidden_for_3 = Set{Int}()

        # Assign type 1 (Residential)
        if minimum_zoning_requirements && !isempty(req_counts) && length(req_counts) >= 1 && req_counts[1] > 0
            cand1_free = [i for i in cand1 if !(i in used)]
            S1, forbidden_for_3 = select_disjoint_set(cand1_free, neighbors, req_counts[1], forbidden_for_3)

            if length(S1) < req_counts[1]
                for _ in 1:5
                    shuffle!(cand1_free)
                    S1_try, ff3_try = select_disjoint_set(cand1_free, neighbors, req_counts[1], Set{Int}())
                    if length(S1_try) >= req_counts[1]
                        S1 = S1_try
                        forbidden_for_3 = ff3_try
                        break
                    end
                end
            end

            # Edge pruning if still short
            if length(S1) < req_counts[1]
                need = req_counts[1] - length(S1)
                extra = [i for i in cand1_free if !(i in S1)]
                for i in extra
                    if need == 0
                        break
                    end
                    for s in S1
                        adjacency_matrix[i, s] = false
                        adjacency_matrix[s, i] = false
                    end
                    push!(S1, i)
                    need -= 1
                end
                # Rebuild neighbors
                for i in 1:n_parcels
                    neighbors[i] = [i2 for i2 in 1:n_parcels if adjacency_matrix[i, i2]]
                end
            end

            for i in S1
                assignment[i] = 1
                push!(used, i)
            end
        end

        # Assign type 3 (Industrial), avoiding adjacency with type 1
        if minimum_zoning_requirements && !isempty(req_counts) && length(req_counts) >= 3 && req_counts[3] > 0
            cand3_free = [i for i in cand3 if !(i in used) && !(i in forbidden_for_3)]
            S3 = Int[]
            for i in cand3_free
                if length(S3) == req_counts[3]
                    break
                end
                ok = true
                for nb in neighbors[i]
                    if assignment[nb] == 1
                        ok = false
                        break
                    end
                end
                if ok
                    push!(S3, i)
                    push!(used, i)
                end
            end

            # Edge pruning if short
            if length(S3) < req_counts[3]
                for i in cand3
                    if length(S3) == req_counts[3]
                        break
                    end
                    if i in used
                        continue
                    end
                    for nb in neighbors[i]
                        if assignment[nb] == 1
                            adjacency_matrix[i, nb] = false
                            adjacency_matrix[nb, i] = false
                        end
                    end
                    neighbors[i] = [i2 for i2 in 1:n_parcels if adjacency_matrix[i, i2]]
                    conflict = any(assignment[nb] == 1 for nb in neighbors[i])
                    if !conflict
                        push!(S3, i)
                        push!(used, i)
                    end
                    if length(S3) == req_counts[3]
                        break
                    end
                end
                for ii in 1:n_parcels
                    neighbors[ii] = [i2 for i2 in 1:n_parcels if adjacency_matrix[ii, i2]]
                end
            end

            for i in S3
                assignment[i] = 3
            end
        end

        # Assign type 2 (Commercial)
        if minimum_zoning_requirements && !isempty(req_counts) && length(req_counts) >= 2 && req_counts[2] > 0
            needed2 = req_counts[2]
            count2 = 0
            for i in cand2
                if i in used
                    continue
                end
                assignment[i] == 0 || continue
                assignment[i] = 2
                push!(used, i)
                count2 += 1
                if count2 == needed2
                    break
                end
            end

            # Fallback: try any unassigned
            if count2 < needed2
                left = needed2 - count2
                for i in 1:n_parcels
                    if left == 0
                        break
                    end
                    if (assignment[i] == 0) && (2 in allowed_sets[i])
                        assignment[i] = 2
                        push!(used, i)
                        left -= 1
                    end
                end

                # Swap from type 1 or 3 if still short
                if left > 0 && minimum_zoning_requirements && length(req_counts) >= 1 && req_counts[1] > 0
                    for i in 1:n_parcels
                        if left == 0
                            break
                        end
                        if assignment[i] == 1 && (2 in allowed_sets[i])
                            repl = findfirst(ii -> (assignment[ii] == 0) && (1 in allowed_sets[ii]), cand1)
                            if repl !== nothing
                                ok = true
                                for nb in neighbors[repl]
                                    if assignment[nb] == 3
                                        ok = false
                                        break
                                    end
                                end
                                if ok
                                    assignment[repl] = 1
                                    push!(used, repl)
                                    assignment[i] = 2
                                    left -= 1
                                end
                            end
                        end
                    end
                end

                if left > 0 && minimum_zoning_requirements && length(req_counts) >= 3 && req_counts[3] > 0
                    for i in 1:n_parcels
                        if left == 0
                            break
                        end
                        if assignment[i] == 3 && (2 in allowed_sets[i])
                            repl = findfirst(ii -> (assignment[ii] == 0) && (3 in allowed_sets[ii]), cand3)
                            if repl !== nothing
                                ok = true
                                for nb in neighbors[repl]
                                    if assignment[nb] == 1
                                        ok = false
                                        break
                                    end
                                end
                                if ok
                                    assignment[repl] = 3
                                    push!(used, repl)
                                    assignment[i] = 2
                                    left -= 1
                                end
                            end
                        end
                    end
                end

                # Ultimate safeguard: relax environmental restriction
                if left > 0
                    candidates = sort([i for i in 1:n_parcels if assignment[i] == 0 && !(2 in allowed_sets[i])],
                                     by = i -> sum(resource_consumption[j, k] for j in allowed_sets[i] for k in 1:n_resources))
                    for i in candidates
                        if left == 0
                            break
                        end
                        environmental_restrictions[i, 2] = false
                        push!(allowed_sets[i], 2)
                        assignment[i] = 2
                        push!(used, i)
                        left -= 1
                    end
                end
            end
        end

        # Assign remaining parcels
        for i in 1:n_parcels
            if assignment[i] == 0
                bestj = nothing
                bestscore = -Inf
                for j in allowed_sets[i]
                    if j == 3 && any(assignment[nb] == 1 for nb in neighbors[i])
                        continue
                    end
                    if j == 1 && any(assignment[nb] == 3 for nb in neighbors[i])
                        continue
                    end
                    score = parcel_sizes[i] * (revenues[i, j] - development_costs[i, j])
                    if score > bestscore
                        bestscore = score
                        bestj = j
                    end
                end
                if bestj === nothing
                    bestj = argmin([type_consumption_score[j] for j in allowed_sets[i]])
                    bestj = allowed_sets[i][bestj]
                end
                assignment[i] = bestj
            end
        end

        # CAPACITY TIGHTENING to admit witness with slack
        usage = zeros(Float64, n_resources)
        for k in 1:n_resources
            usage[k] = sum(parcel_sizes[i] * resource_consumption[assignment[i], k] for i in 1:n_parcels)
        end
        for k in 1:n_resources
            slack_factor = 1.0 + rand(Uniform(0.05, 0.25))
            resource_capacities[k] = max(resource_capacities[k], usage[k] * slack_factor)
        end

    elseif feasibility_status == infeasible
        # LOWER BOUND INFEASIBILITY
        lb = zeros(Float64, n_resources)
        for i in 1:n_parcels
            mins = fill(Inf, n_resources)
            for j in allowed_sets[i]
                for k in 1:n_resources
                    mins[k] = min(mins[k], resource_consumption[j, k])
                end
            end
            for k in 1:n_resources
                lb[k] += parcel_sizes[i] * mins[k]
            end
        end
        for k in 1:n_resources
            violation = rand(Uniform(0.05, 0.25))
            target_cap = lb[k] * (1.0 - violation)
            resource_capacities[k] = min(resource_capacities[k], target_cap)
            resource_capacities[k] = max(resource_capacities[k], 1e-6)
        end
    end

    return LandUseProblem(
        n_parcels,
        n_zoning_types,
        n_resources,
        parcel_sizes,
        development_costs,
        revenues,
        resource_consumption,
        resource_capacities,
        environmental_restrictions,
        adjacency_matrix,
        zoning_names,
        resource_names,
        min_counts_by_type,
        zoning_adjacency_constraints,
        minimum_zoning_requirements
    )
end

"""
    build_model(prob::LandUseProblem)

Build a JuMP model for the land use problem (deterministic).

# Arguments
- `prob`: LandUseProblem instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::LandUseProblem)
    model = Model()

    @variable(model, x[1:prob.n_parcels, 1:prob.n_zoning_types], Bin)

    @objective(model, Max,
        sum(prob.parcel_sizes[i] * (prob.revenues[i, j] - prob.development_costs[i, j]) * x[i, j]
            for i in 1:prob.n_parcels, j in 1:prob.n_zoning_types))

    # Each parcel assigned to exactly one zoning type
    for i in 1:prob.n_parcels
        @constraint(model, sum(x[i, j] for j in 1:prob.n_zoning_types) == 1)
    end

    # Resource capacity constraints
    for k in 1:prob.n_resources
        @constraint(model,
            sum(prob.parcel_sizes[i] * prob.resource_consumption[j, k] * x[i, j]
                for i in 1:prob.n_parcels, j in 1:prob.n_zoning_types) <= prob.resource_capacities[k])
    end

    # Environmental restrictions
    for i in 1:prob.n_parcels
        for j in 1:prob.n_zoning_types
            if prob.environmental_restrictions[i, j]
                @constraint(model, x[i, j] == 0)
            end
        end
    end

    # Minimum zoning requirements
    if prob.minimum_zoning_requirements && !isempty(prob.min_counts_by_type)
        for j in 1:length(prob.min_counts_by_type)
            required_count = prob.min_counts_by_type[j]
            @constraint(model, sum(x[i, j] for i in 1:prob.n_parcels) >= required_count)
        end
    end

    # Adjacency constraints
    if prob.zoning_adjacency_constraints && prob.n_parcels > 1
        for i in 1:prob.n_parcels
            for i2 in 1:prob.n_parcels
                if prob.adjacency_matrix[i, i2]
                    if prob.n_zoning_types >= 3
                        @constraint(model, x[i, 1] + x[i2, 3] <= 1)
                        @constraint(model, x[i, 3] + x[i2, 1] <= 1)
                    end
                end
            end
        end
    end

    return model
end

# Register the problem type
register_problem(
    :land_use,
    LandUseProblem,
    "Land use optimization problem that maximizes economic benefits by allocating land parcels to zoning types (residential, commercial, industrial, agricultural, conservation) while satisfying infrastructure constraints, environmental regulations, and adjacency requirements"
)
