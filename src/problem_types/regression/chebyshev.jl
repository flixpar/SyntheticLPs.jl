using JuMP
using Random
using Distributions
using LinearAlgebra

"""
    ChebyshevWitness

Planted feasible point of a [`ChebyshevRegressionProblem`](@ref): the spline
coefficients of the data-generating surface. Its maximum weighted residual is
`max_weighted_residual < error_cap` (planted margin ≥ 10%).
"""
struct ChebyshevWitness
    coefficients::Vector{Float64}
    max_weighted_residual::Float64
end

"""
    ChebyshevCertificate

Infeasibility proof for a [`ChebyshevRegressionProblem`](@ref). All `points`
lie in one knot cell, so their basis rows live in the same
`(degree+1)²`-dimensional space and have a nontrivial left null vector:
`Σ_k multipliers[k] · B(points[k]) = 0`. For any coefficients, the residuals
`r_k = y_k − B_k·c` therefore satisfy `Σ λ_k r_k = Σ λ_k y_k = combined_residual`,
while the row pairs `w_k |r_k| ≤ t ≤ error_cap` give
`|Σ λ_k r_k| ≤ error_cap · Σ |λ_k| / w_k = error_cap · weighted_l1`. The
instance is infeasible because `|combined_residual| > error_cap · weighted_l1`
(planted margin ≥ 25%). The proof combines `2·|points|` rows with the bound on
`t`.
"""
struct ChebyshevCertificate
    points::Vector{Int}
    multipliers::Vector{Float64}
    combined_residual::Float64
    weighted_l1::Float64
end

"""
    ChebyshevRegressionProblem <: ProblemGenerator

Weighted Chebyshev (minimax / L∞) fit of a tensor-product B-spline surface to
scattered measurements, with an accuracy specification on the maximum error.

# Data profile

Survey-style scattered data over the unit square: every knot cell holds at least
one measurement, the rest are a mix of uniform coverage and dense Gaussian
clusters (survey tracks / sensor sites). The underlying surface is a terrain-like
sum of Gaussian bumps, a ridge, and a trend, represented in the spline space by
its quasi-interpolant. Measurement errors are bounded (instrument tolerance) and
each point has a precision weight from three instrument classes; the weighted
error bound is common.

# Formulation

```math
\\min t \\quad\\text{s.t.}\\quad
-t \\le w_i\\,(y_i - B(p_i)·c) \\le t \\;\\; \\forall i, \\qquad 0 \\le t \\le t_{\\max},
```

with free spline coefficients `c`. Uniform B-splines of degree 1–3 on an
`nx × ny` cell grid give `(nx+degree)(ny+degree)` coefficients, and every row
has only `(degree+1)² + 1` nonzeros (local support), so the LP is sparse and
block-banded however large it grows. The accuracy cap `t ≤ t_max` is a variable
bound.

# Feasibility

  - `feasible`: `t_max` is 1.1–1.4× the planted surface's maximum weighted
    residual (`feasible_witness`).
  - `infeasible`: same specification, but the measurements in one cell carry a
    localized oscillating feature that no degree-`degree` patch can follow within
    `t_max` (`infeasibility_certificate`).
  - `unknown`: `t_max` is 0.88–1.02× the planted maximum residual (the minimax
    optimum is typically 0.90–0.99× it); whether the
    minimax optimum meets it depends on the data. Nothing is planted.

# Sizing

Variables = `(nx+degree)(ny+degree) + 1` (within a few percent of the target).
Rows = `2·n_samples` with `n_samples ≈ 1.8–3×` the coefficient count; nonzeros
≈ `2·n_samples·((degree+1)² + 1)` (≈ 3–6M at 100k variables).
"""
struct ChebyshevRegressionProblem <: ProblemGenerator
    degree::Int
    nx::Int
    ny::Int
    n_samples::Int
    points::Matrix{Float64}
    y::Vector{Float64}
    weights::Vector{Float64}
    error_cap::Float64
    coefficient_bound::Float64
    basis_cols::Matrix{Int}
    basis_vals::Matrix{Float64}
    feasible_witness::Union{Nothing, ChebyshevWitness}
    infeasibility_certificate::Union{Nothing, ChebyshevCertificate}
end

"""Basis values below this threshold are stored as exact zeros."""
const CHEBYSHEV_BASIS_DROP = 1.0e-3

"""Uniform B-spline basis values of degree `k` at local cell coordinate `u ∈ [0, 1]`."""
function _chebyshev_uniform_bspline(k::Int, u::Float64)
    if k == 1
        return (1 - u, u)
    elseif k == 2
        return ((1 - u)^2 / 2, (-2u^2 + 2u + 1) / 2, u^2 / 2)
    else
        return ((1 - u)^3 / 6, (3u^3 - 6u^2 + 4) / 6, (-3u^3 + 3u^2 + 3u + 1) / 6, u^3 / 6)
    end
end

"""Knot cell index (0-based) and local coordinate of `x ∈ [0, 1]` on `n` uniform cells."""
function _chebyshev_cell(x::Float64, n::Int)
    s = clamp(x, 0.0, 1.0) * n
    cell = min(n - 1, floor(Int, s))
    return cell, s - cell
end

"""
    _chebyshev_basis_row(px, py, k, nx, ny)

Column indices and values of the `(k+1)²` tensor B-spline basis functions active
at point `(px, py)`. Column `a·(ny+k) + b + 1` is the product of x-basis `a`
and y-basis `b` (0-based).
"""
function _chebyshev_basis_row(px::Float64, py::Float64, k::Int, nx::Int, ny::Int)
    cx, ux = _chebyshev_cell(px, nx)
    cy, uy = _chebyshev_cell(py, ny)
    bx = _chebyshev_uniform_bspline(k, ux)
    by = _chebyshev_uniform_bspline(k, uy)
    cols = Vector{Int}(undef, (k + 1)^2)
    vals = Vector{Float64}(undef, (k + 1)^2)
    q = 0
    for a in 0:k, b in 0:k
        q += 1
        cols[q] = (cx + a) * (ny + k) + (cy + b) + 1
        value = bx[a + 1] * by[b + 1]
        # Drop negligible tails (a point near a cell edge) to keep the matrix
        # coefficient range tight; the stored values define the model exactly.
        vals[q] = value < CHEBYSHEV_BASIS_DROP ? 0.0 : value
    end
    return cols, vals
end

"""Terrain-like test surface: Gaussian bumps, a ridge, and a linear trend."""
function _chebyshev_surface(rng::AbstractRNG)
    n_bumps = rand(rng, 4:12)
    centres = rand(rng, 2, n_bumps)
    widths = rand(rng, Uniform(0.05, 0.25), n_bumps)
    heights = rand(rng, Normal(0.0, 3.0), n_bumps)
    ridge_angle = rand(rng, Uniform(0, π))
    ridge_height = rand(rng, Uniform(1.0, 4.0))
    ridge_width = rand(rng, Uniform(0.03, 0.10))
    trend = rand(rng, Normal(0.0, 2.0), 2)
    nrm = (cos(ridge_angle), sin(ridge_angle))
    offset = rand(rng, Uniform(0.3, 0.7))
    return function (x, y)
        value = trend[1] * x + trend[2] * y
        for b in 1:n_bumps
            value += heights[b] * exp(-((x - centres[1, b])^2 + (y - centres[2, b])^2) / (2 * widths[b]^2))
        end
        dist = nrm[1] * x + nrm[2] * y - offset
        value += ridge_height * exp(-dist^2 / (2 * ridge_width^2))
        return value
    end
end

"""
    ChebyshevRegressionProblem(target_variables, feasibility_status, seed)

Construct a weighted minimax B-spline surface-fitting instance with a
constructor-local RNG. See the type docstring for sizing and contracts.
"""
function ChebyshevRegressionProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    V = max(target_variables, 5)

    degree = if V <= 30_000
        rand(rng, (1, 2, 2, 3))
    else
        rand(rng, (1, 2, 2))
    end
    n_coef = V - 1
    aspect = rand(rng, Uniform(1.0, 2.0))
    nx = max(1, round(Int, sqrt(n_coef * aspect)) - degree)
    ny = max(1, round(Int, n_coef / (nx + degree)) - degree)
    p = (nx + degree) * (ny + degree)
    n_local = (degree + 1)^2

    # --- Scattered measurement sites. ---
    n_cells = nx * ny
    n_spike = n_local + 1                                  # certificate cell population
    n_samples = max(n_cells + n_spike, round(Int, p * rand(rng, Uniform(1.8, 3.0))))
    points = Matrix{Float64}(undef, 2, n_samples)
    # One site per cell guarantees every basis function is measured.
    cell_order = randperm(rng, n_cells)
    for (q, c) in enumerate(cell_order)
        cx, cy = divrem(c - 1, ny)
        points[1, q] = (cx + rand(rng)) / nx
        points[2, q] = (cy + rand(rng)) / ny
    end
    # The certificate cell: `n_spike` extra sites inside one random cell.
    spike_cell = rand(rng, 1:n_cells)
    sx, sy = divrem(spike_cell - 1, ny)
    spike_points = collect((n_cells + 1):(n_cells + n_spike))
    for q in spike_points
        points[1, q] = (sx + 0.05 + 0.9 * rand(rng)) / nx
        points[2, q] = (sy + 0.05 + 0.9 * rand(rng)) / ny
    end
    # Remaining sites: uniform coverage mixed with Gaussian survey clusters.
    n_clusters = rand(rng, 3:10)
    cluster_centres = rand(rng, 2, n_clusters)
    cluster_sd = rand(rng, Uniform(0.02, 0.12), n_clusters)
    uniform_share = rand(rng, Uniform(0.3, 0.6))
    for q in (n_cells + n_spike + 1):n_samples
        if rand(rng) < uniform_share
            points[1, q] = rand(rng)
            points[2, q] = rand(rng)
        else
            c = rand(rng, 1:n_clusters)
            points[1, q] = clamp(cluster_centres[1, c] + cluster_sd[c] * randn(rng), 0.0, 1.0)
            points[2, q] = clamp(cluster_centres[2, c] + cluster_sd[c] * randn(rng), 0.0, 1.0)
        end
    end

    basis_cols = Matrix{Int}(undef, n_local, n_samples)
    basis_vals = Matrix{Float64}(undef, n_local, n_samples)
    for i in 1:n_samples
        cols, vals = _chebyshev_basis_row(points[1, i], points[2, i], degree, nx, ny)
        basis_cols[:, i] .= cols
        basis_vals[:, i] .= vals
    end

    # --- Planted surface (quasi-interpolant of the terrain) and measurements. ---
    surface = _chebyshev_surface(rng)
    coefficients = Vector{Float64}(undef, p)
    for a in 0:(nx + degree - 1), b in 0:(ny + degree - 1)
        gx = (a - (degree - 1) / 2) / nx
        gy = (b - (degree - 1) / 2) / ny
        coefficients[a * (ny + degree) + b + 1] = surface(gx, gy)
    end
    precision_classes = (1.0, 2.0, 4.0)
    weights = [precision_classes[rand(rng, 1:3)] for _ in 1:n_samples]
    eta = rand(rng, Uniform(0.05, 0.3))                     # common weighted tolerance
    y = Vector{Float64}(undef, n_samples)
    for i in 1:n_samples
        clean = dot(view(basis_vals, :, i), view(coefficients, view(basis_cols, :, i)))
        # Bounded, slightly U-shaped instrument error (Beta(0.8, 0.8) on ±η/w).
        y[i] = clean + (2 * rand(rng, Beta(0.8, 0.8)) - 1) * eta / weights[i]
    end
    max_residual(yy) = maximum(
        weights[i] * abs(yy[i] - dot(view(basis_vals, :, i), view(coefficients, view(basis_cols, :, i))))
        for i in 1:n_samples
    )
    witness_residual = max_residual(y)

    # Physical range of the surface: coefficients may not exceed twice the
    # largest planted coefficient or measurement magnitude.
    coefficient_bound = 2.0 * max(maximum(abs, coefficients), maximum(abs, y))

    witness = nothing
    certificate = nothing
    if feasibility_status == unknown
        error_cap = witness_residual * rand(rng, Uniform(0.88, 1.02))
    else
        error_cap = witness_residual * rand(rng, Uniform(1.1, 1.4))
    end
    if feasibility_status == feasible
        witness = ChebyshevWitness(coefficients, witness_residual)
    elseif feasibility_status == infeasible
        # Left null vector of the spike cell's local basis block.
        local_block = Matrix{Float64}(undef, n_spike, n_local)
        for (r, q) in enumerate(spike_points)
            local_block[r, :] .= view(basis_vals, :, q)
        end
        λ = svd(local_block; full=true).U[:, end]
        λ ./= maximum(abs, λ)
        weighted_l1 = sum(abs(λ[r]) / weights[q] for (r, q) in enumerate(spike_points))
        g0 = sum(λ[r] * y[q] for (r, q) in enumerate(spike_points))
        kick = rand(rng, Uniform(1.25, 1.5)) * error_cap + abs(g0) / weighted_l1
        for (r, q) in enumerate(spike_points)
            y[q] += kick * sign(λ[r]) / weights[q]
        end
        combined = sum(λ[r] * y[q] for (r, q) in enumerate(spike_points))
        certificate = ChebyshevCertificate(copy(spike_points), λ, combined, weighted_l1)
    end

    return ChebyshevRegressionProblem(
        degree,
        nx,
        ny,
        n_samples,
        points,
        y,
        weights,
        error_cap,
        coefficient_bound,
        basis_cols,
        basis_vals,
        witness,
        certificate,
    )
end

"""
    build_model(prob::ChebyshevRegressionProblem)

Build the weighted minimax spline-fitting LP. Deterministic and linear in the
number of nonzeros.
"""
function build_model(prob::ChebyshevRegressionProblem)
    model = Model()
    p = (prob.nx + prob.degree) * (prob.ny + prob.degree)

    @variable(model, -prob.coefficient_bound <= coef[1:p] <= prob.coefficient_bound)
    @variable(model, 0 <= t <= prob.error_cap)
    @objective(model, Min, t)

    for i in 1:prob.n_samples
        w = prob.weights[i]
        fit = _regression_affine_expr(coef, view(prob.basis_cols, :, i), w .* view(prob.basis_vals, :, i))
        @constraint(model, fit + t >= w * prob.y[i])
        @constraint(model, fit - t <= w * prob.y[i])
    end
    return model
end

register_variant(
    :regression,
    :chebyshev,
    ChebyshevRegressionProblem,
    "Weighted Chebyshev (minimax) tensor B-spline surface fit to scattered measurements under an accuracy cap",
)
