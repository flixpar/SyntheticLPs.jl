using JuMP
using Random
using Distributions
using StatsBase

"""
    generate_land_use_problem(params::Dict=Dict(); seed::Int=0)

Generate a land use optimization problem instance with parcels and zoning types.

This models realistic land use planning where parcels must be allocated to zoning types
(residential, commercial, industrial, agricultural, conservation) while satisfying
infrastructure constraints, environmental regulations, and economic objectives.

# Arguments
- `params`: Dictionary of problem parameters
  - `:n_parcels`: Number of land parcels (default: 10)
  - `:n_zoning_types`: Number of zoning types (default: 5)
  - `:n_resources`: Number of resource constraints (default: 4)
  - `:development_cost_scale`: Scale factor for development costs (default: 100000)
  - `:revenue_scale`: Scale factor for revenue generation (default: 50000)
  - `:infrastructure_capacity_factor`: Factor controlling infrastructure capacity (default: 0.7)
  - `:environmental_constraint_prob`: Probability of environmental constraints (default: 0.3)
  - `:zoning_adjacency_constraints`: Whether to include adjacency constraints (default: true)
  - `:minimum_zoning_requirements`: Whether to require minimum allocations (default: true)
  - `:solution_status`: Desired feasibility of the generated instance. One of `:feasible`, `:infeasible`, or `:all`.
    Default: `:feasible`. When `:feasible`, a feasible assignment is constructed and capacities are
    adjusted minimally to guarantee feasibility. When `:infeasible`, capacities are set below a
    provable lower bound to guarantee infeasibility. When `:all`, behavior follows the unconstrained
    random generation (no guarantees).
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- `model`: The JuMP model
- `params`: Dictionary of all parameters used (including defaults)
"""
function generate_land_use_problem(params::Dict=Dict(); seed::Int=0)
    # Set random seed
    Random.seed!(seed)
    
    # Extract parameters with defaults
    n_parcels = get(params, :n_parcels, 10)
    n_zoning_types = get(params, :n_zoning_types, 5)
    n_resources = get(params, :n_resources, 4)
    development_cost_scale = get(params, :development_cost_scale, 100000)
    revenue_scale = get(params, :revenue_scale, 50000)
    infrastructure_capacity_factor = get(params, :infrastructure_capacity_factor, 0.7)
    environmental_constraint_prob = get(params, :environmental_constraint_prob, 0.3)
    zoning_adjacency_constraints = get(params, :zoning_adjacency_constraints, true)
    minimum_zoning_requirements = get(params, :minimum_zoning_requirements, true)
    solution_status = get(params, :solution_status, :feasible)
    if solution_status isa String
        solution_status = Symbol(lowercase(solution_status))
    end
    if !(solution_status in (:feasible, :infeasible, :all))
        error("Unknown solution_status=$(solution_status). Use :feasible, :infeasible, or :all")
    end
    
    # Save actual parameters used
    actual_params = Dict{Symbol, Any}(
        :n_parcels => n_parcels,
        :n_zoning_types => n_zoning_types,
        :n_resources => n_resources,
        :development_cost_scale => development_cost_scale,
        :revenue_scale => revenue_scale,
        :infrastructure_capacity_factor => infrastructure_capacity_factor,
        :environmental_constraint_prob => environmental_constraint_prob,
        :zoning_adjacency_constraints => zoning_adjacency_constraints,
        :minimum_zoning_requirements => minimum_zoning_requirements,
        :solution_status => solution_status
    )
    
    # Generate parcel characteristics using realistic distributions
    # Parcel sizes (in acres) - log-normal distribution
    parcel_sizes = rand(LogNormal(log(5), 0.8), n_parcels)
    parcel_sizes = max.(parcel_sizes, 0.1)  # Minimum 0.1 acre
    
    # Zoning type names and characteristics
    zoning_names = ["Residential", "Commercial", "Industrial", "Agricultural", "Conservation"]
    if n_zoning_types > 5
        append!(zoning_names, ["Mixed_Use", "Recreational", "Institutional", "Transportation", "Special"])
    end
    zoning_names = zoning_names[1:n_zoning_types]
    
    # Resource names (infrastructure and environmental)
    resource_names = ["Water", "Sewage", "Transportation", "Power"]
    if n_resources > 4
        append!(resource_names, ["Internet", "Gas", "Environmental", "Emergency"])
    end
    resource_names = resource_names[1:n_resources]
    
    # Generate development costs per acre for each zoning type (varies by complexity)
    cost_multipliers = [1.0, 2.5, 3.0, 0.5, 0.1]  # Residential, Commercial, Industrial, Agricultural, Conservation
    if n_zoning_types > 5
        append!(cost_multipliers, [2.0, 1.5, 1.8, 4.0, 3.5])
    end
    cost_multipliers = cost_multipliers[1:n_zoning_types]
    
    # Development costs with location-specific variation
    development_costs = zeros(n_parcels, n_zoning_types)
    for i in 1:n_parcels
        # Location factor (some areas more expensive to develop)
        location_factor = rand(Gamma(2, 0.5))
        for j in 1:n_zoning_types
            base_cost = development_cost_scale * cost_multipliers[j]
            development_costs[i, j] = base_cost * location_factor * rand(Normal(1.0, 0.2))
            development_costs[i, j] = max(development_costs[i, j], base_cost * 0.1)  # Minimum cost
        end
    end
    
    # Generate revenue per acre for each zoning type
    revenue_multipliers = [1.5, 4.0, 2.0, 0.8, 0.2]  # Different economic returns
    if n_zoning_types > 5
        append!(revenue_multipliers, [3.0, 1.0, 0.5, 0.1, 2.5])
    end
    revenue_multipliers = revenue_multipliers[1:n_zoning_types]
    
    revenues = zeros(n_parcels, n_zoning_types)
    for i in 1:n_parcels
        # Location affects revenue potential
        location_revenue_factor = rand(Gamma(2, 0.6))
        for j in 1:n_zoning_types
            base_revenue = revenue_scale * revenue_multipliers[j]
            revenues[i, j] = base_revenue * location_revenue_factor * rand(Normal(1.0, 0.3))
            revenues[i, j] = max(revenues[i, j], 0.0)
        end
    end
    
    # Generate resource consumption per acre for each zoning type
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
    
    # Generate resource capacities
    total_demand_estimate = sum(parcel_sizes) * mean(resource_consumption, dims=1)
    resource_capacities = vec(total_demand_estimate) .* infrastructure_capacity_factor .* rand(Uniform(0.8, 1.2), n_resources)
    
    # Environmental constraints (some parcels restricted for certain zoning)
    environmental_restrictions = zeros(Bool, n_parcels, n_zoning_types)
    for i in 1:n_parcels
        if rand() < environmental_constraint_prob
            # Restrict a subset of zoning types while preserving at least one allowed option initially
            max_restrict = max(1, min(3, n_zoning_types - 1))
            num_to_restrict = rand(1:max_restrict)
            restricted_types = sample(1:n_zoning_types, num_to_restrict, replace=false)
            environmental_restrictions[i, restricted_types] .= true
        end
    end
    
    # Generate adjacency matrix for parcels (simplified - random adjacency)
    adjacency_matrix = zeros(Bool, n_parcels, n_parcels)
    if zoning_adjacency_constraints && n_parcels > 1
        for i in 1:n_parcels
            # Each parcel has 2-4 neighbors on average
            n_neighbors = rand(2:min(4, n_parcels-1))
            neighbors = sample(setdiff(1:n_parcels, [i]), n_neighbors, replace=false)
            adjacency_matrix[i, neighbors] .= true
            adjacency_matrix[neighbors, i] .= true
        end
    end
    
    # Helper to compute aggregated consumption per type (used for tie-breaking)
    type_consumption_score = [sum(resource_consumption[j, k] for k in 1:n_resources) for j in 1:n_zoning_types]
    
    # Ensure every parcel has at least one allowed zoning and that minimum requirements (if any) are achievable
    # Build allowed sets and fix empty-allowed cases by unrestricting the least resource-intensive type
    allowed_sets = Vector{Vector{Int}}(undef, n_parcels)
    for i in 1:n_parcels
        allowed = [j for j in 1:n_zoning_types if !environmental_restrictions[i, j]]
        if isempty(allowed)
            # Unrestrict the gentlest type for realism
            jbest = argmin(type_consumption_score)
            environmental_restrictions[i, jbest] = false
            allowed = [jbest]
        end
        allowed_sets[i] = allowed
    end
    
    # Compute effective minimum zoning requirement counts per type (for types 1..min(3, n_zoning_types, n_parcels))
    num_required_types = minimum([3, n_zoning_types, n_parcels])
    min_counts_by_type = Int[]
    if minimum_zoning_requirements
        base_min = max(1, round(Int, n_parcels * 0.1))
        # Adjust so that sum of minimums does not exceed number of parcels
        total_base = base_min * num_required_types
        if total_base <= n_parcels
            min_counts_by_type = fill(base_min, num_required_types)
        else
            # Spread as evenly as possible without exceeding n_parcels
            per = max(1, fld(n_parcels, num_required_types))
            min_counts_by_type = fill(per, num_required_types)
            remaining = n_parcels - per * num_required_types
            # Distribute the remainder
            idx = 1
            while remaining > 0 && idx <= num_required_types
                min_counts_by_type[idx] += 1
                remaining -= 1
                idx += 1
            end
        end
        # Ensure environmental restrictions allow meeting these mins by relaxing where needed
        for j in 1:num_required_types
            allowed_count_j = count(i -> !environmental_restrictions[i, j], 1:n_parcels)
            deficit = max(0, min_counts_by_type[j] - allowed_count_j)
            if deficit > 0
                candidates = [i for i in 1:n_parcels if environmental_restrictions[i, j]]
                # Prefer parcels where type j is comparatively gentle
                sort!(candidates, by = i -> type_consumption_score[j])
                for t in 1:min(deficit, length(candidates))
                    environmental_restrictions[candidates[t], j] = false
                end
            end
        end
        # Rebuild allowed_sets after potential relaxations
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
    
    # If requested, guarantee feasibility/infeasibility while preserving realism and diversity
    if solution_status != :all
        # Precompute neighbor lists if adjacency matters
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

        if solution_status == :feasible
            # Construct a concrete feasible assignment to serve as a witness
            assignment = fill(0, n_parcels)
            req_counts = copy(min_counts_by_type)

            # Helper to assign a set S of parcels to type jt respecting adjacency with type 3 vs 1
            function select_disjoint_set(candidates::Vector{Int}, restricted_neighbors::Vector{Vector{Int}}, quota::Int, forbidden_neighbors::Set{Int})
                S = Int[]
                for i in candidates
                    if length(S) == quota
                        break
                    end
                    # Skip if i is in forbidden neighborhood
                    if i in forbidden_neighbors
                        continue
                    end
                    push!(S, i)
                    # Add its neighbors to forbidden set
                    for nb in restricted_neighbors[i]
                        push!(forbidden_neighbors, nb)
                    end
                end
                return S, forbidden_neighbors
            end

            # Build candidate sets for types 1, 2, and 3
            cand1 = [i for i in 1:n_parcels if 1 <= n_zoning_types && (1 in allowed_sets[i])]
            cand2 = [i for i in 1:n_parcels if 2 <= n_zoning_types && (2 in allowed_sets[i])]
            cand3 = [i for i in 1:n_parcels if 3 <= n_zoning_types && (3 in allowed_sets[i])]

            # Prefer parcels that are unique to the target type among required types to reduce contention,
            # then prefer lower resource consumption
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

            # Assign type 1 (Residential) first if required
            if minimum_zoning_requirements && !isempty(req_counts) && length(req_counts) >= 1 && req_counts[1] > 0
                # Filter candidates not yet used
                cand1_free = [i for i in cand1 if !(i in used)]
                # Select S1 greedily while tracking neighbors to protect type 3 later
                S1, forbidden_for_3 = select_disjoint_set(cand1_free, neighbors, req_counts[1], forbidden_for_3)
                # If not enough, try random shuffle retries
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
                # If still short, minimally relax adjacency by pruning edges among the remaining candidates
                if length(S1) < req_counts[1]
                    need = req_counts[1] - length(S1)
                    extra = [i for i in cand1_free if !(i in S1)]
                    for i in extra
                        if need == 0
                            break
                        end
                        # Remove edges between i and S1 to allow placement
                        for s in S1
                            adjacency_matrix[i, s] = false
                            adjacency_matrix[s, i] = false
                        end
                        push!(S1, i)
                        need -= 1
                    end
                    # Rebuild neighbors after pruning
                    for i in 1:n_parcels
                        neighbors[i] = [i2 for i2 in 1:n_parcels if adjacency_matrix[i, i2]]
                    end
                end
                for i in S1
                    assignment[i] = 1
                    push!(used, i)
                end
            end

            # Assign type 3 (Industrial) if required, avoiding adjacency with type 1
            if minimum_zoning_requirements && !isempty(req_counts) && length(req_counts) >= 3 && req_counts[3] > 0
                cand3_free = [i for i in cand3 if !(i in used) && !(i in forbidden_for_3)]
                S3 = Int[]
                for i in cand3_free
                    if length(S3) == req_counts[3]
                        break
                    end
                    # Check adjacency to already assigned type-1 nodes
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
                # If still short, prune edges to avoid conflicts with already chosen type 1 parcels
                if length(S3) < req_counts[3]
                    need = req_counts[3] - length(S3)
                    for i in cand3
                        if length(S3) == req_counts[3]
                            break
                        end
                        if i in used
                            continue
                        end
                        # Remove adjacency to type-1 neighbors if any
                        for nb in neighbors[i]
                            if assignment[nb] == 1
                                adjacency_matrix[i, nb] = false
                                adjacency_matrix[nb, i] = false
                            end
                        end
                        # Rebuild neighbors after pruning
                        neighbors[i] = [i2 for i2 in 1:n_parcels if adjacency_matrix[i, i2]]
                        # Recheck and assign
                        conflict = any(assignment[nb] == 1 for nb in neighbors[i])
                        if !conflict
                            push!(S3, i)
                            push!(used, i)
                        end
                        if length(S3) == req_counts[3]
                            break
                        end
                    end
                    # Globally refresh neighbor lists
                    for ii in 1:n_parcels
                        neighbors[ii] = [i2 for i2 in 1:n_parcels if adjacency_matrix[ii, i2]]
                    end
                end
                for i in S3
                    assignment[i] = 3
                end
            end

            # Assign type 2 (Commercial) next if required
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
                # If still short, relax by converting some unassigned that allow type 2
                if count2 < needed2
                    left = needed2 - count2
                    # Try to assign any remaining unassigned first
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
                    # If still short, perform minimal rebalancing: swap from S1 or S3 where possible
                    if left > 0
                        # Prefer swapping parcels that have 2 allowed and whose original type has alternative candidates
                        # Try from type 1
                        if left > 0 && minimum_zoning_requirements && length(req_counts) >= 1 && req_counts[1] > 0
                            for i in 1:n_parcels
                                if left == 0
                                    break
                                end
                                if assignment[i] == 1 && (2 in allowed_sets[i])
                                    # Find replacement for type 1 that is currently unassigned and allowed for 1
                                    repl = findfirst(ii -> (assignment[ii] == 0) && (1 in allowed_sets[ii]), cand1)
                                    if repl !== nothing
                                        # Check adjacency with type 3 for replacement
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
                        # Then try from type 3
                        if left > 0 && minimum_zoning_requirements && length(req_counts) >= 3 && req_counts[3] > 0
                            for i in 1:n_parcels
                                if left == 0
                                    break
                                end
                                if assignment[i] == 3 && (2 in allowed_sets[i])
                                    # Find replacement for type 3 that is currently unassigned and allowed for 3
                                    repl = findfirst(ii -> (assignment[ii] == 0) && (3 in allowed_sets[ii]), cand3)
                                    if repl !== nothing
                                        # Check adjacency with type 1 for replacement
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
                    end
                    # As an ultimate safeguard, if still short, relax environmental restriction for type 2 on a few parcels
                    # and reassign, prioritizing parcels with low total consumption
                    if left > 0
                        candidates = sort([i for i in 1:n_parcels if assignment[i] == 0 && !(2 in allowed_sets[i])], by = i -> sum(resource_consumption[j, k] for j in allowed_sets[i] for k in 1:n_resources))
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

            # Assign remaining parcels to any allowed type that maximizes net benefit while respecting current adjacency
            for i in 1:n_parcels
                if assignment[i] == 0
                    bestj = nothing
                    bestscore = -Inf
                    for j in allowed_sets[i]
                        # Skip type-3 if any neighbor is type-1 due to adjacency
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
                        # Fallback to the least resource-intensive allowed type
                        bestj = argmin([type_consumption_score[j] for j in allowed_sets[i]])
                        bestj = allowed_sets[i][bestj]
                    end
                    assignment[i] = bestj
                end
            end

            # Tighten capacities just enough to admit the witness assignment
            usage = zeros(Float64, n_resources)
            for k in 1:n_resources
                usage[k] = sum(parcel_sizes[i] * resource_consumption[assignment[i], k] for i in 1:n_parcels)
            end
            for k in 1:n_resources
                slack_factor = 1.0 + rand(Uniform(0.05, 0.25))
                resource_capacities[k] = max(resource_capacities[k], usage[k] * slack_factor)
            end
            actual_params[:feasible_witness_assignment] = assignment
        elseif solution_status == :infeasible
            # Compute a provable lower bound on resource usage ignoring adjacency and min requirements
            # This bound ensures infeasibility if any capacity is set below it
            lb = zeros(Float64, n_resources)
            for i in 1:n_parcels
                # Per-parcel minima over allowed types
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
            # Set capacities below the lower bound with a realistic degradation factor
            for k in 1:n_resources
                violation = rand(Uniform(0.05, 0.25))
                target_cap = lb[k] * (1.0 - violation)
                resource_capacities[k] = min(resource_capacities[k], target_cap)
                # Ensure strictly positive but below bound
                resource_capacities[k] = max(resource_capacities[k], 1e-6)
            end
        end
    end

    # Store generated data in params (after potential adjustments)
    actual_params[:parcel_sizes] = parcel_sizes
    actual_params[:development_costs] = development_costs
    actual_params[:revenues] = revenues
    actual_params[:resource_consumption] = resource_consumption
    actual_params[:resource_capacities] = resource_capacities
    actual_params[:environmental_restrictions] = environmental_restrictions
    actual_params[:adjacency_matrix] = adjacency_matrix
    actual_params[:zoning_names] = zoning_names
    actual_params[:resource_names] = resource_names
    actual_params[:min_counts_by_type] = copy(min_counts_by_type)
    
    # Create model
    model = Model()
    
    # Variables: binary variables for parcel-zoning allocation
    @variable(model, x[1:n_parcels, 1:n_zoning_types], Bin)
    
    # Objective: maximize net benefit (revenue - development costs)
    @objective(model, Max, 
        sum(parcel_sizes[i] * (revenues[i, j] - development_costs[i, j]) * x[i, j] 
            for i in 1:n_parcels, j in 1:n_zoning_types))
    
    # Constraint: each parcel must be assigned to exactly one zoning type
    for i in 1:n_parcels
        @constraint(model, sum(x[i, j] for j in 1:n_zoning_types) == 1)
    end
    
    # Constraints: resource capacity limitations
    for k in 1:n_resources
        @constraint(model, 
            sum(parcel_sizes[i] * resource_consumption[j, k] * x[i, j] 
                for i in 1:n_parcels, j in 1:n_zoning_types) <= resource_capacities[k])
    end
    
    # Constraints: environmental restrictions
    for i in 1:n_parcels
        for j in 1:n_zoning_types
            if environmental_restrictions[i, j]
                @constraint(model, x[i, j] == 0)
            end
        end
    end
    
    # Constraints: minimum zoning requirements (ensure diverse development)
    if minimum_zoning_requirements && !isempty(min_counts_by_type)
        for j in 1:length(min_counts_by_type)
            required_count = min_counts_by_type[j]
            @constraint(model, sum(x[i, j] for i in 1:n_parcels) >= required_count)
        end
    end
    
    # Constraints: adjacency constraints (limit certain zoning combinations)
    if zoning_adjacency_constraints && n_parcels > 1
        for i in 1:n_parcels
            for i2 in 1:n_parcels
                if adjacency_matrix[i, i2]
                    # Industrial cannot be adjacent to residential
                    if n_zoning_types >= 3
                        @constraint(model, x[i, 1] + x[i2, 3] <= 1)  # Residential + Industrial
                        @constraint(model, x[i, 3] + x[i2, 1] <= 1)  # Industrial + Residential
                    end
                end
            end
        end
    end
    
    return model, actual_params
end

"""
    sample_land_use_parameters(target_variables::Int; seed::Int=0)

Sample realistic parameters for a land use optimization problem targeting approximately the specified number of variables.

Variables = n_parcels × n_zoning_types (binary assignment variables)

# Arguments
- `target_variables`: Target number of variables in the LP formulation (approximately within ±10%)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_land_use_parameters(target_variables::Int; seed::Int=0)
    Random.seed!(seed)
    
    params = Dict{Symbol, Any}()
    
    # Determine problem scale based on target variables
    if target_variables <= 250
        # Small scale: local/municipal planning
        scale = :small
        params[:n_zoning_types] = rand(3:5)
        params[:n_resources] = rand(3:5)
        params[:development_cost_scale] = rand(50000:150000)
        params[:revenue_scale] = rand(20000:80000)
        params[:infrastructure_capacity_factor] = rand(Uniform(0.6, 0.8))
        params[:environmental_constraint_prob] = rand(Uniform(0.2, 0.4))
    elseif target_variables <= 1000
        # Medium scale: regional planning
        scale = :medium
        params[:n_zoning_types] = rand(4:8)
        params[:n_resources] = rand(4:6)
        params[:development_cost_scale] = rand(75000:250000)
        params[:revenue_scale] = rand(40000:120000)
        params[:infrastructure_capacity_factor] = rand(Uniform(0.65, 0.85))
        params[:environmental_constraint_prob] = rand(Uniform(0.25, 0.45))
    else
        # Large scale: state/national planning
        scale = :large
        params[:n_zoning_types] = rand(5:12)
        params[:n_resources] = rand(5:8)
        params[:development_cost_scale] = rand(100000:500000)
        params[:revenue_scale] = rand(60000:200000)
        params[:infrastructure_capacity_factor] = rand(Uniform(0.7, 0.9))
        params[:environmental_constraint_prob] = rand(Uniform(0.3, 0.5))
    end
    
    # Calculate n_parcels to achieve target variables
    target_parcels = round(Int, target_variables / params[:n_zoning_types])
    params[:n_parcels] = max(2, target_parcels)
    
    # Iteratively adjust to get within 10% tolerance
    for iteration in 1:10
        current_vars = calculate_land_use_variable_count(params)
        
        if abs(current_vars - target_variables) / target_variables < 0.1
            break  # Within 10% tolerance
        end
        
        # Adjust n_parcels or n_zoning_types
        if current_vars < target_variables
            if rand() < 0.7  # Prefer adjusting parcels
                params[:n_parcels] += 1
            else
                max_zoning = scale == :small ? 5 : scale == :medium ? 8 : 12
                params[:n_zoning_types] = min(max_zoning, params[:n_zoning_types] + 1)
            end
        elseif current_vars > target_variables
            if rand() < 0.7  # Prefer adjusting parcels
                params[:n_parcels] = max(2, params[:n_parcels] - 1)
            else
                min_zoning = scale == :small ? 3 : scale == :medium ? 4 : 5
                params[:n_zoning_types] = max(min_zoning, params[:n_zoning_types] - 1)
            end
        end
    end
    
    # Scale-appropriate constraint settings
    params[:zoning_adjacency_constraints] = rand() < 0.8  # More likely for realistic problems
    params[:minimum_zoning_requirements] = rand() < 0.9  # Almost always want diverse zoning
    
    return params
end

"""
    sample_land_use_parameters(size::Symbol=:medium; seed::Int=0)

Sample realistic parameters for a land use optimization problem using size categories.

# Arguments
- `size`: Symbol specifying the problem size (:small, :medium, :large)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_land_use_parameters(size::Symbol=:medium; seed::Int=0)
    Random.seed!(seed)
    
    # Map size categories to realistic target variable ranges
    target_map = Dict(
        :small => rand(50:250),    # Local/municipal planning
        :medium => rand(250:1000), # Regional planning
        :large => rand(1000:10000) # State/national planning
    )
    
    if !haskey(target_map, size)
        error("Unknown size: $size. Must be :small, :medium, or :large")
    end
    
    return sample_land_use_parameters(target_map[size]; seed=seed)
end

"""
    calculate_land_use_variable_count(params::Dict)

Calculate the number of variables in a land use optimization problem.

# Arguments
- `params`: Dictionary of problem parameters containing :n_parcels and :n_zoning_types

# Returns
- Number of variables (n_parcels × n_zoning_types binary assignment variables)
"""
function calculate_land_use_variable_count(params::Dict)
    n_parcels = get(params, :n_parcels, 10)
    n_zoning_types = get(params, :n_zoning_types, 5)
    return n_parcels * n_zoning_types
end

# Register the problem type
register_problem(
    :land_use,
    generate_land_use_problem,
    sample_land_use_parameters,
    "Land use optimization problem that maximizes economic benefits by allocating land parcels to zoning types (residential, commercial, industrial, agricultural, conservation) while satisfying infrastructure constraints, environmental regulations, and adjacency requirements"
)