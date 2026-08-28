using JuMP
using Random

"""
    DiscretePlacementLoadBalancingProblem <: ProblemGenerator

A service-placement and workload-routing MILP. Binary `placement[s,m]`
decisions choose the machines on which service `s` is deployed. Continuous
`workload[k,s,m]` decisions route each traffic class's demand to deployed
machines, continuous machine-load variables aggregate processing time, and a
makespan variable measures the maximum load.

This formulation is structurally distinct from `load_balancing/standard`,
which is a continuous link-flow traffic-engineering LP. Multiple traffic
classes per placement decision also reproduce the continuous-variable-heavy
shape of collected discrete load-balancing models.

# Fields
- `n_services`, `n_machines`, `n_classes`: Formulation dimensions.
- `demand`: Workload demand by traffic class and service.
- `processing_time`: Machine time per unit workload by service and machine.
- `machine_capacity`: Maximum load on each machine.
- `max_replicas`: Maximum placements allowed for each service.
- `planted_machine`: A primary machine for each service, used as the feasible witness.
"""
struct DiscretePlacementLoadBalancingProblem <: ProblemGenerator
    n_services::Int
    n_machines::Int
    n_classes::Int
    demand::Matrix{Float64}
    processing_time::Matrix{Float64}
    machine_capacity::Vector{Float64}
    max_replicas::Vector{Int}
    planted_machine::Vector{Int}
end

"""
    DiscretePlacementLoadBalancingProblem(target_variables, feasibility_status, seed)

Construct a deterministic discrete placement/load-balancing instance. Its
variable count is

```text
services * machines                         binary placements
+ classes * services * machines            continuous routed workload
+ machines + 1                              loads and makespan.
```

Dimensions are selected so this total closely tracks `target_variables` while
the number of traffic classes grows faster than the placement grid.

For `feasible`, routing every class of a service to `planted_machine[s]`
satisfies the generated capacities. For `infeasible`, aggregate capacity is
strictly below

```text
sum(demand[k,s] * minimum(processing_time[s,:]), k, s),
```

a necessary workload lower bound that remains valid when placement binaries
are relaxed. `unknown` distributes capacity near that lower bound without
asserting feasibility.
"""
function DiscretePlacementLoadBalancingProblem(
    target_variables::Int,
    feasibility_status::FeasibilityStatus,
    seed::Int,
)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be positive (got $target_variables)"))

    rng = MersenneTwister(seed)

    # A balanced placement grid grows slowly; traffic classes absorb the
    # remaining target and make the continuous routing block dominant.
    grid_side = clamp(round(Int, cbrt(target_variables / 4)), 3, 20)
    n_services = grid_side
    n_machines = grid_side
    placement_variables = n_services * n_machines
    fixed_continuous = n_machines + 1
    n_classes = max(
        1,
        round(Int, (target_variables - placement_variables - fixed_continuous) /
                   placement_variables),
    )

    demand = [5.0 + 45.0 * rand(rng) for _ in 1:n_classes, _ in 1:n_services]

    # Heterogeneous service-machine affinities: lower values mean faster
    # processing. Values remain well scaled and strictly positive.
    service_scale = [0.7 + 0.6 * rand(rng) for _ in 1:n_services]
    machine_scale = [0.7 + 0.6 * rand(rng) for _ in 1:n_machines]
    processing_time = [
        service_scale[service] / machine_scale[machine] * (0.85 + 0.30 * rand(rng))
        for service in 1:n_services, machine in 1:n_machines
    ]

    # Use a permutation so the planted placement spreads services across
    # machines instead of creating an artificial single-machine bottleneck.
    machine_order = randperm(rng, n_machines)
    planted_machine = [machine_order[mod1(service, n_machines)]
                       for service in 1:n_services]
    max_replicas = [rand(rng, 2:min(4, n_machines)) for _ in 1:n_services]

    planted_load = zeros(Float64, n_machines)
    for service in 1:n_services
        machine = planted_machine[service]
        planted_load[machine] +=
            processing_time[service, machine] * sum(demand[:, service])
    end

    aggregate_lower_bound = sum(
        demand[traffic_class, service] * minimum(processing_time[service, :])
        for traffic_class in 1:n_classes, service in 1:n_services
    )

    machine_capacity = if feasibility_status == feasible
        [max(1.0, planted_load[machine] * (1.10 + 0.25 * rand(rng)))
         for machine in 1:n_machines]
    else
        capacity_weights = [0.5 + rand(rng) for _ in 1:n_machines]
        capacity_weights ./= sum(capacity_weights)
        total_capacity = if feasibility_status == infeasible
            aggregate_lower_bound * (0.65 + 0.20 * rand(rng))
        else
            aggregate_lower_bound * (0.90 + 0.35 * rand(rng))
        end
        total_capacity .* capacity_weights
    end

    return DiscretePlacementLoadBalancingProblem(
        n_services,
        n_machines,
        n_classes,
        demand,
        processing_time,
        machine_capacity,
        max_replicas,
        planted_machine,
    )
end

"""
    build_model(prob::DiscretePlacementLoadBalancingProblem)

Build the placement/routing MILP. Each routing upper bound is the corresponding
demand—not an arbitrary large constant—and therefore remains tight after
integrality relaxation.
"""
function build_model(prob::DiscretePlacementLoadBalancingProblem)
    model = Model()
    S = prob.n_services
    M = prob.n_machines
    K = prob.n_classes

    @variable(model, placement[1:S, 1:M], Bin)
    @variable(model, workload[1:K, 1:S, 1:M] >= 0)
    @variable(model, 0 <= machine_load[m = 1:M] <= prob.machine_capacity[m])
    @variable(model, 0 <= makespan <= maximum(prob.machine_capacity))

    # Each class/service demand can be split only among machines hosting the
    # service. Placement cardinality keeps deployment decisions nontrivial.
    for service in 1:S
        @constraint(model, sum(placement[service, machine] for machine in 1:M) >= 1)
        @constraint(
            model,
            sum(placement[service, machine] for machine in 1:M) <=
                prob.max_replicas[service],
        )
        for traffic_class in 1:K
            @constraint(
                model,
                sum(workload[traffic_class, service, machine] for machine in 1:M) ==
                    prob.demand[traffic_class, service],
            )
            for machine in 1:M
                @constraint(
                    model,
                    workload[traffic_class, service, machine] <=
                        prob.demand[traffic_class, service] * placement[service, machine],
                )
            end
        end
    end

    for machine in 1:M
        @constraint(
            model,
            machine_load[machine] ==
                sum(
                    prob.processing_time[service, machine] *
                    workload[traffic_class, service, machine]
                    for traffic_class in 1:K, service in 1:S
                ),
        )
        @constraint(model, machine_load[machine] <= makespan)
    end

    # Makespan is primary; the tiny placement penalty breaks ties in favor of
    # parsimonious deployments without changing the load-balancing scale.
    @objective(
        model,
        Min,
        makespan + 1.0e-4 * sum(placement[service, machine]
                                for service in 1:S, machine in 1:M),
    )

    return model
end

register_variant(
    :load_balancing,
    :discrete_placement,
    DiscretePlacementLoadBalancingProblem,
    "Discrete service placement with binary deployments, continuous workload routing, machine capacities, and makespan minimization",
)

