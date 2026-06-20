using JuMP
using Random

"""
    MultidimensionalKnapsackProblem <: ProblemGenerator

Generator for 0/1 multi-dimensional (multi-constraint) knapsack problems (MDKP).

# Overview
Models the selection of a subset of items to maximize total value subject to
*several* simultaneous resource limits — not just one weight budget. Each item
`i` consumes a correlated bundle of `D` resources (e.g. weight, volume, budget,
labor-hours): a per-item "size" factor scaled by per-resource intensities plus
multiplicative noise, so the resource columns are genuinely correlated rather
than independent uniform noise. Item values are positively correlated with total
resource use but not identical to it, so the trade-offs are non-trivial. Each
resource capacity is set to roughly 40-70% of the total usage of that resource,
making every packing constraint binding.

The decision variables are declared `Bin`. Because the framework solves the LP
*relaxation* by default (`relax_integer=true`), the solved model is the
continuous `[0, 1]` relaxation of the MDKP.

Feasibility handling (relaxation-aware). For a maximization problem with only
`<=` packing constraints, `x = 0` is always feasible, so shrinking a capacity can
never make the model infeasible. Instead:
- `feasible` / `unknown`: only the `D` packing constraints; always feasible
  (`x = 0` admissible, objective pushes selection up to the binding resources).
- `infeasible`: a structural *covering* floor is added. Let `r*` be the tightest
  resource (smallest capacity relative to total usage). Sort items by their usage
  of `r*` ascending and find the smallest count `m` whose summed smallest usages
  strictly exceed `cap[r*]`. The added constraint `sum_i x_i >= m` ("select at
  least `m` items") then contradicts the `r*` packing limit even in the
  relaxation: any `m` items (even the `m` lightest) overflow resource `r*`. This
  survives binary relaxation because the lightest-`m` argument is independent of
  integrality.

# Fields
- `n_items::Int`: Number of items (== number of decision variables)
- `n_resources::Int`: Number of resource dimensions `D`
- `values::Vector{Float64}`: Value of each item
- `usage::Matrix{Float64}`: Resource usage, `usage[r, i]` = item `i` use of resource `r` (D × n_items)
- `capacities::Vector{Float64}`: Capacity for each resource (length D)
- `required_min_items::Int`: Covering floor `m` (only enforced when `> 0`, for infeasible instances)
"""
struct MultidimensionalKnapsackProblem <: ProblemGenerator
    n_items::Int
    n_resources::Int
    values::Vector{Float64}
    usage::Matrix{Float64}
    capacities::Vector{Float64}
    required_min_items::Int
end

"""
    MultidimensionalKnapsackProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a multi-dimensional knapsack instance.

Variable-count formula (decision variables created by `build_model`):

    total = n_items

There is exactly one binary variable per item, so `n_items = target_variables`
exactly. The number of resource dimensions `D` scales with problem size
(2-3 small, 3-4 medium, 4-5 large) but adds no variables — it only adds packing
constraints.

# Arguments
- `target_variables`: Target number of decision variables (== number of items)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function MultidimensionalKnapsackProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # One binary variable per item.
    n_items = max(2, target_variables)

    # --- Scale-tiered ranges ---
    # Number of resource dimensions and parameter magnitudes grow with size.
    if n_items <= 100
        n_resources = rand(2:3)
        size_lo, size_hi = 5.0, 30.0            # per-item base "size" factor
        intensity_lo, intensity_hi = 0.5, 2.5   # per-resource intensity multiplier
        value_base_lo, value_base_hi = 10.0, 120.0
    elseif n_items <= 1000
        n_resources = rand(3:4)
        size_lo, size_hi = 10.0, 60.0
        intensity_lo, intensity_hi = 0.6, 3.0
        value_base_lo, value_base_hi = 20.0, 300.0
    else
        n_resources = rand(4:5)
        size_lo, size_hi = 20.0, 120.0
        intensity_lo, intensity_hi = 0.8, 3.5
        value_base_lo, value_base_hi = 50.0, 700.0
    end

    D = n_resources

    # Per-resource intensity: how heavily each resource is consumed in general.
    resource_intensity = intensity_lo .+ (intensity_hi - intensity_lo) .* rand(D)

    # Per-item base "size": a latent factor driving usage across ALL resources,
    # which is what makes the resource columns correlated.
    item_size = size_lo .+ (size_hi - size_lo) .* rand(n_items)

    # Resource usage: size * intensity * lognormal-ish noise (always positive).
    # usage[r, i] = item_size[i] * resource_intensity[r] * noise
    usage = zeros(Float64, D, n_items)
    for i in 1:n_items
        for r in 1:D
            noise = exp(0.30 * randn())          # multiplicative noise, mean ~1
            usage[r, i] = item_size[i] * resource_intensity[r] * noise
            usage[r, i] = max(usage[r, i], 0.1)  # strictly positive
        end
    end

    # Values: correlated with total resource consumption (sum over resources of
    # usage) but with independent noise, plus a base term, so value-density varies
    # item to item and the problem is non-trivial.
    total_use = vec(sum(usage, dims=1))          # length n_items
    mean_use = sum(total_use) / n_items
    values = zeros(Float64, n_items)
    for i in 1:n_items
        base = value_base_lo + (value_base_hi - value_base_lo) * rand()
        # density-correlated term: heavier items tend to be worth more, with noise
        density_term = (total_use[i] / mean_use) * base * (0.6 + 0.8 * rand())
        values[i] = max(1.0, 0.4 * base + 0.6 * density_term)
    end

    # Capacities: ~40-70% of the total usage of each resource (binding packing).
    capacities = zeros(Float64, D)
    for r in 1:D
        total_r = sum(usage[r, i] for i in 1:n_items)
        cap_ratio = 0.40 + 0.30 * rand()
        capacities[r] = total_r * cap_ratio
    end

    # --- Resolve feasibility intent ---
    actual_status = feasibility_status
    if feasibility_status == unknown
        # Natural instance; with only <= packing constraints this is always
        # feasible (x=0). Keep it as a plain packing MDKP.
        actual_status = feasible
    end

    required_min_items = 0
    if actual_status == infeasible
        # Pick the tightest resource: smallest capacity relative to total usage.
        tightness = [capacities[r] / sum(usage[r, i] for i in 1:n_items) for r in 1:D]
        r_star = argmin(tightness)
        cap_star = capacities[r_star]

        # Sort items by usage of r* ascending; find the smallest m whose summed
        # smallest usages strictly exceed cap_star. Any selection of m items then
        # overflows resource r* (the m lightest already do), contradicting the
        # packing limit even in the relaxation.
        order = sortperm([usage[r_star, i] for i in 1:n_items])
        cumulative = 0.0
        m = 0
        for k in 1:n_items
            cumulative += usage[r_star, order[k]]
            if cumulative > cap_star
                m = k
                break
            end
        end
        # cap_star is 40-70% of total usage, so the lightest items always overflow
        # before reaching the full count; m is well-defined and <= n_items.
        if m == 0
            m = n_items   # extreme fallback (shouldn't trigger given cap < total)
        end
        required_min_items = min(m, n_items)
    end

    return MultidimensionalKnapsackProblem(
        n_items, n_resources, values, usage, capacities, required_min_items,
    )
end

"""
    build_model(prob::MultidimensionalKnapsackProblem)

Build a JuMP model for the multi-dimensional knapsack problem. Deterministic —
uses only data from the struct fields.

Decision variables:
- `x[i]`: binary selection of item `i` (relaxed to `[0, 1]` by the framework)

# Returns
- `model`: The JuMP model
"""
function build_model(prob::MultidimensionalKnapsackProblem)
    model = Model()

    n = prob.n_items
    D = prob.n_resources

    # One binary variable per item (total = n_items).
    @variable(model, x[1:n], Bin)

    # Objective: maximize total value of selected items.
    @objective(model, Max, sum(prob.values[i] * x[i] for i in 1:n))

    # D resource (packing) constraints.
    for r in 1:D
        @constraint(model, sum(prob.usage[r, i] * x[i] for i in 1:n) <= prob.capacities[r])
    end

    # Covering floor (infeasible instances only): contradicts the tightest
    # resource even in the LP relaxation.
    if prob.required_min_items > 0
        @constraint(model, sum(x[i] for i in 1:n) >= prob.required_min_items)
    end

    return model
end

# Register the variant
register_variant(
    :knapsack,
    :multidimensional,
    MultidimensionalKnapsackProblem,
    "Multi-dimensional (multi-constraint) 0/1 knapsack maximizing value subject to several correlated resource limits",
)
