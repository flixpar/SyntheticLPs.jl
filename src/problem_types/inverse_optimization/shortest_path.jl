"""
A directed arc and its physical/network metadata.
"""
struct InverseNetworkArc
    tail::Int
    head::Int
    distance_km::Float64
    road_class::Symbol
end

"""
One observed origin-destination path, stored as directed arc indices.
"""
struct InversePathObservation
    source::Int
    destination::Int
    path_arcs::Vector{Int}
end

"""
Exact cost and shortest-path potentials for all planted observations.
"""
struct InverseShortestPathWitness
    cost::Vector{Float64}
    potentials::Matrix{Float64}
end

"""
Native infeasibility certificate for a route observation. After cancelling
arcs shared by the observed and alternative routes, every admissible observed
route costs at least `observed_floor`, while the alternative costs at most
`alternative_ceiling`.
"""
struct InversePathConflictCertificate
    observation::Int
    alternative_path::Vector{Int}
    observed_only::Vector{Int}
    alternative_only::Vector{Int}
    observed_floor::Float64
    alternative_ceiling::Float64
end

struct InverseShortestPathProblem <: ProblemGenerator
    n_nodes::Int
    n_arcs::Int
    n_observations::Int
    profile::Symbol
    coordinates_km::Matrix{Float64}
    arcs::Vector{InverseNetworkArc}
    observations::Vector{InversePathObservation}
    true_cost::Vector{Float64}
    prior_cost::Vector{Float64}
    cost_lower::Vector{Float64}
    cost_upper::Vector{Float64}
    deviation_weight::Vector{Float64}
    resolved_status::FeasibilityStatus
    feasible_witness::Union{Nothing, InverseShortestPathWitness}
    infeasibility_certificate::Union{Nothing, InversePathConflictCertificate}
end

function _inverse_network_dimensions(target_variables::Int, average_degree::Float64)
    target = max(target_variables, 1)
    best = (error=typemax(Int), nodes=4, edges=4, observations=2, count=32)
    max_nodes = max(4, cld(target, 8) + 4)
    preferred_k = if target < 150
        2
    elseif target < 800
        4
    else
        6
    end
    for nodes in 4:max_nodes
        edges = clamp(round(Int, average_degree * nodes / 2), nodes - 1, nodes * (nodes - 1) ÷ 2)
        for observations in max(2, preferred_k - 2):(preferred_k + 2)
            count = 6 * edges + observations * nodes
            candidate = (
                error=abs(count - target),
                nodes=nodes,
                edges=edges,
                observations=observations,
                count=count,
            )
            (candidate.error, abs(observations - preferred_k), count) <
            (best.error, abs(best.observations - preferred_k), best.count) && (best = candidate)
        end
    end
    return best.nodes, best.edges, best.observations
end

function _inverse_network_profile(rng::AbstractRNG)
    profiles = (
        (name=:urban_grid, side_km=18.0, jitter=0.16, degree=3.2),
        (name=:regional_roads, side_km=140.0, jitter=0.30, degree=2.5),
        (name=:mixed_corridor, side_km=55.0, jitter=0.22, degree=3.8),
    )
    return profiles[rand(rng, eachindex(profiles))]
end

function _inverse_grid_coordinates(
    rng::AbstractRNG, n_nodes::Int, side_km::Float64, jitter::Float64
)
    width = ceil(Int, sqrt(n_nodes))
    spacing = side_km / max(width - 1, 1)
    coordinates = Matrix{Float64}(undef, n_nodes, 2)
    for node in 1:n_nodes
        row = fld(node - 1, width)
        offset = mod(node - 1, width)
        column = iseven(row) ? offset : width - 1 - offset
        coordinates[node, 1] = spacing * column + rand(rng, Uniform(-jitter, jitter)) * spacing
        coordinates[node, 2] = spacing * row + rand(rng, Uniform(-jitter, jitter)) * spacing
    end
    return coordinates
end

function _inverse_undirected_edges(rng::AbstractRNG, coordinates::Matrix{Float64}, n_edges::Int)
    n_nodes = size(coordinates, 1)
    edges = Set{Tuple{Int, Int}}((i, i + 1) for i in 1:(n_nodes - 1))
    width = ceil(Int, sqrt(n_nodes))
    candidates = Tuple{Float64, Int, Int}[]
    for u in 1:n_nodes
        for offset in (width, width - 1, width + 1, 2)
            v = u + offset
            v <= n_nodes || continue
            pair = (min(u, v), max(u, v))
            pair in edges && continue
            distance = hypot(
                coordinates[u, 1] - coordinates[v, 1], coordinates[u, 2] - coordinates[v, 2]
            )
            push!(candidates, (distance * rand(rng, Uniform(0.90, 1.10)), pair[1], pair[2]))
        end
    end
    sort!(candidates)
    for (_, u, v) in candidates
        length(edges) >= n_edges && break
        push!(edges, (u, v))
    end
    while length(edges) < n_edges
        u, v = rand(rng, 1:n_nodes, 2)
        u == v && continue
        push!(edges, (min(u, v), max(u, v)))
    end
    return sort!(collect(edges))
end

function _inverse_road_data(
    rng::AbstractRNG,
    profile::Symbol,
    coordinates::Matrix{Float64},
    undirected_edges::Vector{Tuple{Int, Int}},
)
    arcs = InverseNetworkArc[]
    true_cost = Float64[]
    for (u, v) in undirected_edges
        distance = max(
            0.15,
            hypot(coordinates[u, 1] - coordinates[v, 1], coordinates[u, 2] - coordinates[v, 2]),
        )
        road_class = if profile == :urban_grid
            rand(rng) < 0.72 ? :local : :arterial
        elseif profile == :regional_roads
            rand(rng) < 0.66 ? :regional : :highway
        else
            draw = rand(rng)
            if draw < 0.45
                :local
            elseif draw < 0.78
                :arterial
            else
                :highway
            end
        end
        speed_range = if road_class == :local
            (24.0, 42.0)
        elseif road_class == :arterial
            (42.0, 68.0)
        elseif road_class == :regional
            (58.0, 85.0)
        else
            (88.0, 118.0)
        end
        base_time = 60.0 * distance / rand(rng, Uniform(speed_range...))
        for (tail, head) in ((u, v), (v, u))
            push!(arcs, InverseNetworkArc(tail, head, distance, road_class))
            push!(true_cost, base_time * rand(rng, LogNormal(0.0, 0.16)))
        end
    end
    return arcs, true_cost
end

function _inverse_shortest_distances(
    n_nodes::Int, arcs::Vector{InverseNetworkArc}, costs::Vector{Float64}, source::Int
)
    adjacency = [Int[] for _ in 1:n_nodes]
    for (e, arc) in enumerate(arcs)
        push!(adjacency[arc.tail], e)
    end
    distance = fill(Inf, n_nodes)
    predecessor = fill(0, n_nodes)
    distance[source] = 0.0
    # A small local binary heap keeps network generation O(E log V), which is
    # important when the requested inverse LP has tens of thousands of nodes.
    heap = Tuple{Float64, Int}[(0.0, source)]
    while !isempty(heap)
        current_distance, u = first(heap)
        last_item = pop!(heap)
        if !isempty(heap)
            heap[1] = last_item
            parent = 1
            while true
                left = 2 * parent
                left > length(heap) && break
                right = left + 1
                child = right <= length(heap) && heap[right] < heap[left] ? right : left
                heap[parent] <= heap[child] && break
                heap[parent], heap[child] = heap[child], heap[parent]
                parent = child
            end
        end
        current_distance > distance[u] && continue
        for e in adjacency[u]
            v = arcs[e].head
            candidate = distance[u] + costs[e]
            if candidate < distance[v]
                distance[v] = candidate
                predecessor[v] = e
                push!(heap, (candidate, v))
                child = length(heap)
                while child > 1
                    parent = child ÷ 2
                    heap[parent] <= heap[child] && break
                    heap[parent], heap[child] = heap[child], heap[parent]
                    child = parent
                end
            end
        end
    end
    return distance, predecessor
end

function _inverse_reconstruct_path(
    arcs::Vector{InverseNetworkArc}, predecessor::Vector{Int}, source::Int, destination::Int
)
    path = Int[]
    node = destination
    while node != source
        edge = predecessor[node]
        edge == 0 && error("Generated inverse network is disconnected")
        push!(path, edge)
        node = arcs[edge].tail
    end
    reverse!(path)
    return path
end

function _inverse_path_has_alternative(
    n_nodes::Int, arcs::Vector{InverseNetworkArc}, path::Vector{Int}, source::Int, destination::Int
)
    adjacency = [Int[] for _ in 1:n_nodes]
    for (e, arc) in enumerate(arcs)
        push!(adjacency[arc.tail], e)
    end
    for removed_edge in path
        removed = arcs[removed_edge]
        reached = falses(n_nodes)
        reached[source] = true
        queue = [source]
        cursor = 1
        while cursor <= length(queue)
            u = queue[cursor]
            cursor += 1
            for e in adjacency[u]
                arc = arcs[e]
                arc.tail == u || continue
                # Remove the physical link in both directions.
                (
                    (arc.tail == removed.tail && arc.head == removed.head) ||
                    (arc.tail == removed.head && arc.head == removed.tail)
                ) && continue
                reached[arc.head] && continue
                reached[arc.head] = true
                push!(queue, arc.head)
            end
        end
        reached[destination] && return true
    end
    return false
end

function _inverse_path_observations(
    rng::AbstractRNG,
    coordinates::Matrix{Float64},
    arcs::Vector{InverseNetworkArc},
    costs::Vector{Float64},
    n_observations::Int,
)
    n_nodes = size(coordinates, 1)
    diameter = hypot(
        maximum(coordinates[:, 1]) - minimum(coordinates[:, 1]),
        maximum(coordinates[:, 2]) - minimum(coordinates[:, 2]),
    )
    observations = InversePathObservation[]
    potentials = Matrix{Float64}(undef, n_observations, n_nodes)
    used = Set{Tuple{Int, Int}}()
    for k in 1:n_observations
        source, destination = 1, n_nodes
        path = Int[]
        distances = Float64[]
        for _ in 1:200
            candidate_source, candidate_destination = rand(rng, 1:n_nodes, 2)
            candidate_source == candidate_destination && continue
            (candidate_source, candidate_destination) in used && continue
            separation = hypot(
                coordinates[candidate_source, 1] - coordinates[candidate_destination, 1],
                coordinates[candidate_source, 2] - coordinates[candidate_destination, 2],
            )
            separation >= 0.25 * diameter || continue
            candidate_distances, predecessor = _inverse_shortest_distances(
                n_nodes, arcs, costs, candidate_source
            )
            candidate_path = _inverse_reconstruct_path(
                arcs, predecessor, candidate_source, candidate_destination
            )
            !isempty(candidate_path) || continue
            _inverse_path_has_alternative(
                n_nodes, arcs, candidate_path, candidate_source, candidate_destination
            ) || continue
            source, destination = candidate_source, candidate_destination
            path, distances = candidate_path, candidate_distances
            break
        end
        if isempty(path)
            # Exhaustive deterministic fallback for tiny networks, where the
            # random distinct-OD search can consume the few qualifying pairs.
            for allow_reuse in (false, true)
                found = false
                for candidate_source in 1:n_nodes
                    for candidate_destination in 1:n_nodes
                        candidate_source == candidate_destination && continue
                        !allow_reuse &&
                            (candidate_source, candidate_destination) in used &&
                            continue
                        candidate_distances, predecessor = _inverse_shortest_distances(
                            n_nodes, arcs, costs, candidate_source
                        )
                        candidate_path = _inverse_reconstruct_path(
                            arcs, predecessor, candidate_source, candidate_destination
                        )
                        !isempty(candidate_path) || continue
                        _inverse_path_has_alternative(
                            n_nodes, arcs, candidate_path, candidate_source, candidate_destination
                        ) || continue
                        source = candidate_source
                        destination = candidate_destination
                        path, distances = candidate_path, candidate_distances
                        found = true
                        break
                    end
                    found && break
                end
                found && break
            end
        end
        isempty(path) && error("Could not sample a nontrivial inverse path observation")
        push!(used, (source, destination))
        push!(observations, InversePathObservation(source, destination, path))
        potentials[k, :] = distances .- distances[source]
    end
    return observations, potentials
end

function _inverse_paths_are_optimal(
    n_nodes::Int,
    arcs::Vector{InverseNetworkArc},
    observations::Vector{InversePathObservation},
    costs::Vector{Float64},
)
    return all(observations) do observation
        distances, _ = _inverse_shortest_distances(n_nodes, arcs, costs, observation.source)
        path_cost = sum(costs[e] for e in observation.path_arcs)
        isapprox(path_cost, distances[observation.destination]; atol=1.0e-9)
    end
end

function _inverse_make_prior_informative!(
    n_nodes::Int,
    arcs::Vector{InverseNetworkArc},
    observations::Vector{InversePathObservation},
    prior_cost::Vector{Float64},
)
    for observation in observations, edge in observation.path_arcs
        removed = arcs[edge]
        alternative_cost = copy(prior_cost)
        for (e, arc) in enumerate(arcs)
            if (arc.tail == removed.tail && arc.head == removed.head) ||
                (arc.tail == removed.head && arc.head == removed.tail)
                alternative_cost[e] = Inf
            end
        end
        distances, _ = _inverse_shortest_distances(
            n_nodes, arcs, alternative_cost, observation.source
        )
        alternative = distances[observation.destination]
        isfinite(alternative) || continue
        other_path_cost = sum((prior_cost[e] for e in observation.path_arcs if e != edge); init=0.0)
        prior_cost[edge] = max(prior_cost[edge], alternative - other_path_cost + 0.05 * alternative)
        _inverse_paths_are_optimal(n_nodes, arcs, observations, prior_cost) || return true
    end
    return false
end

function _inverse_alternative_path(
    n_nodes::Int,
    arcs::Vector{InverseNetworkArc},
    costs::Vector{Float64},
    observation::InversePathObservation,
)
    for removed_edge in observation.path_arcs
        removed = arcs[removed_edge]
        alternative_cost = copy(costs)
        for (e, arc) in enumerate(arcs)
            if (arc.tail == removed.tail && arc.head == removed.head) ||
                (arc.tail == removed.head && arc.head == removed.tail)
                alternative_cost[e] = Inf
            end
        end
        distances, predecessor = _inverse_shortest_distances(
            n_nodes, arcs, alternative_cost, observation.source
        )
        isfinite(distances[observation.destination]) || continue
        alternative = _inverse_reconstruct_path(
            arcs, predecessor, observation.source, observation.destination
        )
        observed_only = setdiff(observation.path_arcs, alternative)
        alternative_only = setdiff(alternative, observation.path_arcs)
        !isempty(observed_only) && !isempty(alternative_only) && return alternative
    end
    error("Could not recover the alternative route guaranteed during generation")
end

function _impose_inverse_path_conflict!(
    rng::AbstractRNG,
    n_nodes::Int,
    arcs::Vector{InverseNetworkArc},
    observations::Vector{InversePathObservation},
    true_cost::Vector{Float64},
    prior_cost::Vector{Float64},
    cost_lower::Vector{Float64},
    cost_upper::Vector{Float64};
    guaranteed::Bool,
)
    observation_index = rand(rng, eachindex(observations))
    observation = observations[observation_index]
    alternative = _inverse_alternative_path(n_nodes, arcs, true_cost, observation)
    observed_only = setdiff(observation.path_arcs, alternative)
    alternative_only = setdiff(alternative, observation.path_arcs)
    observed_floor = sum(cost_lower[e] for e in observed_only)
    factor = guaranteed ? rand(rng, Uniform(0.45, 0.82)) : rand(rng, Uniform(0.78, 1.28))
    alternative_ceiling = factor * observed_floor
    weights = prior_cost[alternative_only]
    weights ./= sum(weights)
    for (position, e) in enumerate(alternative_only)
        cost_upper[e] = alternative_ceiling * weights[position]
        cost_lower[e] = 0.35 * cost_upper[e]
        prior_cost[e] = 0.70 * cost_upper[e]
    end
    return InversePathConflictCertificate(
        observation_index,
        alternative,
        observed_only,
        alternative_only,
        observed_floor,
        sum(cost_upper[e] for e in alternative_only),
    )
end

function InverseShortestPathProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    _check_inverse_target(target_variables)
    rng = MersenneTwister(seed)
    profile_data = _inverse_network_profile(rng)
    n_nodes, n_edges, K = _inverse_network_dimensions(target_variables, profile_data.degree)
    coordinates = _inverse_grid_coordinates(rng, n_nodes, profile_data.side_km, profile_data.jitter)
    undirected_edges = _inverse_undirected_edges(rng, coordinates, n_edges)
    arcs, true_cost = _inverse_road_data(rng, profile_data.name, coordinates, undirected_edges)
    observations, potentials = _inverse_path_observations(rng, coordinates, arcs, true_cost, K)

    # A cost prior that already rationalizes every observed route gives the
    # inverse model a zero-adjustment solution. Resample realistic measurement
    # error until at least one observation is informative. With continuous
    # multiplicative noise this terminates quickly; the generous cap protects
    # construction time without affecting normal draws.
    prior_cost = copy(true_cost)
    for _ in 1:100
        prior_cost = true_cost .* rand(rng, LogNormal(0.0, 0.24), length(true_cost))
        _inverse_paths_are_optimal(n_nodes, arcs, observations, prior_cost) || break
    end
    if _inverse_paths_are_optimal(n_nodes, arcs, observations, prior_cost)
        _inverse_make_prior_informative!(n_nodes, arcs, observations, prior_cost) ||
            error("Could not construct an informative inverse-shortest-path prior")
    end
    cost_lower = 0.55 .* min.(true_cost, prior_cost)
    cost_upper = 1.85 .* max.(true_cost, prior_cost)
    deviation_weight = 1.0 ./ max.(prior_cost, 0.1)

    witness = if feasibility_status == feasible
        InverseShortestPathWitness(copy(true_cost), potentials)
    else
        nothing
    end
    certificate = nothing
    if feasibility_status == infeasible
        certificate = _impose_inverse_path_conflict!(
            rng,
            n_nodes,
            arcs,
            observations,
            true_cost,
            prior_cost,
            cost_lower,
            cost_upper;
            guaranteed=true,
        )
    elseif feasibility_status == unknown && rand(rng) >= 0.30
        # A sampled near-conflict may or may not exclude all rationalizing
        # costs. No outcome metadata is stored for unknown requests.
        _impose_inverse_path_conflict!(
            rng,
            n_nodes,
            arcs,
            observations,
            true_cost,
            prior_cost,
            cost_lower,
            cost_upper;
            guaranteed=false,
        )
    end
    return InverseShortestPathProblem(
        n_nodes,
        length(arcs),
        K,
        profile_data.name,
        coordinates,
        arcs,
        observations,
        true_cost,
        prior_cost,
        cost_lower,
        cost_upper,
        deviation_weight,
        feasibility_status,
        witness,
        certificate,
    )
end

function build_model(prob::InverseShortestPathProblem)
    model = Model()
    E, N, K = prob.n_arcs, prob.n_nodes, prob.n_observations
    @variable(model, prob.cost_lower[e] <= arc_cost[e in 1:E] <= prob.cost_upper[e])
    @variable(model, potential[1:K, 1:N])
    @variable(model, deviation_positive[1:E] >= 0)
    @variable(model, deviation_negative[1:E] >= 0)
    @objective(
        model,
        Min,
        sum(
            prob.deviation_weight[e] * (deviation_positive[e] + deviation_negative[e]) for e in 1:E
        ),
    )
    @constraint(
        model,
        prior_deviation[e in 1:E],
        arc_cost[e] - prob.prior_cost[e] == deviation_positive[e] - deviation_negative[e],
    )
    @constraint(
        model,
        dual_feasibility[k in 1:K, e in 1:E],
        potential[k, prob.arcs[e].head] - potential[k, prob.arcs[e].tail] <= arc_cost[e],
    )
    @constraint(
        model, potential_anchor[k in 1:K], potential[k, prob.observations[k].source] == 0.0,
    )
    @constraint(
        model,
        observed_path_optimality[k in 1:K],
        potential[k, prob.observations[k].destination] ==
            sum(arc_cost[e] for e in prob.observations[k].path_arcs),
    )
    return model
end

function _inverse_path_conflict_certificate_is_valid(prob::InverseShortestPathProblem)
    certificate = prob.infeasibility_certificate
    certificate isa InversePathConflictCertificate || return false
    observation = prob.observations[certificate.observation]
    certificate.observed_only == setdiff(observation.path_arcs, certificate.alternative_path) ||
        return false
    certificate.alternative_only == setdiff(certificate.alternative_path, observation.path_arcs) ||
        return false
    observed_floor = sum(prob.cost_lower[e] for e in certificate.observed_only)
    alternative_ceiling = sum(prob.cost_upper[e] for e in certificate.alternative_only)
    return isapprox(certificate.observed_floor, observed_floor) &&
           isapprox(certificate.alternative_ceiling, alternative_ceiling) &&
           alternative_ceiling < observed_floor
end

function _inverse_shortest_path_witness_is_valid(prob::InverseShortestPathProblem)
    witness = prob.feasible_witness
    witness isa InverseShortestPathWitness || return false
    all(prob.cost_lower .<= witness.cost .+ 1.0e-10) || return false
    all(witness.cost .<= prob.cost_upper .+ 1.0e-10) || return false
    for k in 1:prob.n_observations
        observation = prob.observations[k]
        abs(witness.potentials[k, observation.source]) <= 1.0e-10 || return false
        for e in 1:prob.n_arcs
            arc = prob.arcs[e]
            witness.potentials[k, arc.head] - witness.potentials[k, arc.tail] <=
            witness.cost[e] + 1.0e-9 || return false
        end
        isapprox(
            witness.potentials[k, observation.destination],
            sum(witness.cost[e] for e in observation.path_arcs);
            atol=1.0e-8,
        ) || return false
    end
    return true
end

register_variant(
    :inverse_optimization,
    :shortest_path,
    InverseShortestPathProblem,
    "Weighted-L1 inverse shortest path on sparse spatial road networks with multiple observed routes",
)
