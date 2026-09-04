using JuMP
using Random

"""
    TwoDimensionalBinPackingProblem <: ProblemGenerator

Orthogonal two-dimensional bin packing with binary assignments and pairwise
left/right/above/below disjunctions plus continuous item coordinates.
"""
struct TwoDimensionalBinPackingProblem <: ProblemGenerator
    n_items::Int
    n_bins::Int
    widths::Vector{Float64}
    heights::Vector{Float64}
    bin_width::Float64
    bin_height::Float64
    maximum_used::Int
end

function TwoDimensionalBinPackingProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    # Variables are assignments, bin-use indicators, two coordinates per item,
    # and four relative-position binaries per item pair. The smallest search
    # point (n=3, b=2) already yields 26 variables; clamp the target so public
    # APIs that accept sizes down to 2 still produce that formulation.
    target = max(target_variables, 26)

    best = (typemax(Int), 0, 0)
    for n in 3:max(3, ceil(Int, sqrt(target)))
        for b in 2:min(8, n)
            total = n * b + b + 2n + 4 * (n * (n - 1) ÷ 2)
            error = abs(total - target)
            error < best[1] && (best = (error, n, b))
        end
    end
    _, n_items, n_bins = best
    bin_width = 100.0
    bin_height = 100.0

    if feasibility_status == infeasible
        widths = fill(99.0, n_items)
        heights = fill(99.0, n_items)
        maximum_used = n_bins - 1
    else
        widths = Float64[rand(rng, 8:20) for _ in 1:n_items]
        heights = Float64[rand(rng, 10:75) for _ in 1:n_items]
        planted_bin = [mod(i - 1, n_bins) + 1 for i in randperm(rng, n_items)]
        for b in 1:n_bins
            items = findall(==(b), planted_bin)
            total_width = sum(widths[i] for i in items)
            if total_width > 90.0
                scale = 90.0 / total_width
                for i in items
                    widths[i] *= scale
                end
            end
        end
        maximum_used = n_bins
    end
    return TwoDimensionalBinPackingProblem(
        n_items, n_bins, widths, heights, bin_width, bin_height, maximum_used
    )
end

function build_model(prob::TwoDimensionalBinPackingProblem)
    model = Model()
    n = prob.n_items
    b_count = prob.n_bins
    pairs = [(i, j) for i in 1:(n - 1) for j in (i + 1):n]

    @variable(model, assign[1:n, 1:b_count], Bin)
    @variable(model, used[1:b_count], Bin)
    @variable(model, 0 <= xpos[i = 1:n] <= prob.bin_width - prob.widths[i])
    @variable(model, 0 <= ypos[i = 1:n] <= prob.bin_height - prob.heights[i])
    @variable(model, relative[1:length(pairs), 1:4], Bin)
    @objective(model, Min, sum(used))

    for i in 1:n
        @constraint(model, sum(assign[i, b] for b in 1:b_count) == 1)
        for b in 1:b_count
            @constraint(model, assign[i, b] <= used[b])
        end
    end
    if prob.maximum_used < b_count
        @constraint(model, sum(used) <= prob.maximum_used)
    end
    for (p, (i, j)) in enumerate(pairs)
        @constraint(
            model, xpos[i] + prob.widths[i] <= xpos[j] + prob.bin_width * (1 - relative[p, 1])
        )
        @constraint(
            model, xpos[j] + prob.widths[j] <= xpos[i] + prob.bin_width * (1 - relative[p, 2])
        )
        @constraint(
            model, ypos[i] + prob.heights[i] <= ypos[j] + prob.bin_height * (1 - relative[p, 3])
        )
        @constraint(
            model, ypos[j] + prob.heights[j] <= ypos[i] + prob.bin_height * (1 - relative[p, 4])
        )
        for b in 1:b_count
            @constraint(model, sum(relative[p, d] for d in 1:4) >= assign[i, b] + assign[j, b] - 1,)
        end
    end
    # A valid aggregate area inequality strengthens the disjunctive relaxation
    # and supplies a clear capacity certificate for requested infeasibility.
    @constraint(
        model,
        sum(prob.widths[i] * prob.heights[i] for i in 1:n) <=
            prob.bin_width * prob.bin_height * sum(used),
    )
    return model
end

register_variant(
    :container_loading,
    :two_dimensional_bin_packing,
    TwoDimensionalBinPackingProblem,
    "Orthogonal two-dimensional bin packing with pairwise non-overlap disjunctions",
)
