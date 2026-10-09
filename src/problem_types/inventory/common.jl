using Random
using Distributions

"""
    _inventory_demand(rng, T, base; amp, phase, trend, cv, intermittent) -> Vector{Float64}

A realistic per-period demand series: `base` level with a sinusoidal seasonal
profile (`amp`, `phase`, 26-period cycle — half-yearly at weekly buckets),
a linear trend, Gamma noise with coefficient of variation `cv`, and — for
`intermittent` items — about half the periods without demand. Values are
rounded to one decimal and nonnegative.
"""
function _inventory_demand(
    rng::AbstractRNG,
    T::Int,
    base::Float64;
    amp::Float64=0.2,
    phase::Float64=0.0,
    trend::Float64=0.0,
    cv::Float64=0.25,
    intermittent::Bool=false,
)
    shape = 1 / cv^2
    d = zeros(T)
    for t in 1:T
        mean_t = base * max(0.05, 1 + amp * sin(2π * t / 26 + phase)) * max(0.2, 1 + trend * t)
        v = rand(rng, Gamma(shape, mean_t / shape))
        d[t] = intermittent && rand(rng) < 0.5 ? 0.0 : round(v; digits=1)
    end
    return d
end

"""
    _inventory_scale_ratio(rng, status) -> Float64

Load-to-capacity ratio for a planted aggregate certificate: `1.10–1.35` for
`infeasible` (a margin, not a knife edge) and `1 ± U(0.03, 0.30)` for
`unknown` (a genuine two-sided draw).
"""
function _inventory_scale_ratio(rng::AbstractRNG, status::FeasibilityStatus)
    status == infeasible && return 1.1 + 0.25 * rand(rng)
    m = 0.03 + 0.27 * rand(rng)
    return rand(rng) < 0.5 ? 1.0 - m : 1.0 + m
end
