using JuMP
using LinearAlgebra
using Random
using SparseArrays
using Statistics

# Calibration sources:
# - AAPM TG-119 (Ezzell et al., Med Phys 2009, DOI 10.1118/1.3238104)
#   for anatomy, dose-goal, and 7/9-field conventions;
# - CORT (Craft et al., GigaScience 2014, DOI 10.1186/2047-217X-3-37)
#   for sparse dose-influence data, beamlet scales, and clinical dimensions.

const _RT_PROFILES = (:prostate, :head_neck, :c_shape, :liver, :lung, :breast)

"""A synthetic but spatially coherent patient, beam geometry, and dose matrix."""
struct RadiotherapyCaseData
    profile::Symbol
    prescription_gy::Float64
    n_fractions::Int
    beam_energy_mv::Int
    body_radii_cm::NTuple{3,Float64}
    structure_names::Vector{Symbol}
    structure_kinds::Vector{Symbol}
    structure_voxels::Dict{Symbol,Vector{Int}}
    voxel_structure::Vector{Symbol}
    voxel_locations_cm::Matrix{Float64}
    voxel_volume_cc::Vector{Float64}
    beam_angles_deg::Vector{Float64}
    beam_of_beamlet::Vector{Int}
    beamlet_u_cm::Vector{Float64}
    beamlet_z_cm::Vector{Float64}
    beamlet_width_cm::Vector{Float64}
    beamlet_height_cm::Vector{Float64}
    beamlet_edges::Vector{Tuple{Int,Int}}
    dose_matrix::SparseMatrixCSC{Float64,Int}
    dose_normalization::Float64
    reference_fluence::Vector{Float64}
    reference_dose::Vector{Float64}
end

"""A complete primal point used to make a requested-feasible plan auditable."""
struct RadiotherapyFluenceWitness
    fluence::Vector{Float64}
end

"""
Proof of an inconsistent pair of hard dose rows. The organ row is exactly
`multiplier` times the target row, but its upper bound is below
`multiplier * target_lower_bound`.
"""
struct RadiotherapyDoseConflictCertificate
    target_voxel::Int
    organ_voxel::Int
    organ::Symbol
    multiplier::Float64
    target_lower_bound::Float64
    organ_upper_bound::Float64
end

function _rt_profile_spec(profile::Symbol)
    if profile == :prostate
        return (
            body=(18.0, 14.0, 11.0),
            structures=[:ptv, :rectum, :bladder, :left_femoral_head,
                        :right_femoral_head, :normal_tissue],
            kinds=[:target, :parallel_oar, :parallel_oar, :serial_oar,
                   :serial_oar, :normal],
            fractions=35:44,
            prescription=(74.0, 80.0),
            energies=(6, 10),
            angles=[0.0, 50.0, 100.0, 150.0, 210.0, 260.0, 310.0],
            voxel_weights=[0.29, 0.14, 0.16, 0.09, 0.09, 0.23],
            clinical_caps=Dict(
                :rectum => 0.94, :bladder => 0.94,
                :left_femoral_head => 0.66, :right_femoral_head => 0.66,
                :normal_tissue => 0.78,
            ),
            tail_fractions=Dict(
                :ptv => 0.95, :rectum => 0.30, :bladder => 0.30,
                :left_femoral_head => 0.10, :right_femoral_head => 0.10,
                :normal_tissue => 0.20,
            ),
            volumes_cc=Dict(
                :ptv => (65.0, 130.0), :rectum => (45.0, 90.0),
                :bladder => (120.0, 320.0), :left_femoral_head => (80.0, 150.0),
                :right_femoral_head => (80.0, 150.0),
                :normal_tissue => (8_000.0, 14_000.0),
            ),
        )
    elseif profile == :head_neck
        return (
            body=(12.0, 10.0, 14.0),
            structures=[:ptv, :spinal_cord, :left_parotid, :right_parotid,
                        :oral_cavity, :normal_tissue],
            kinds=[:target, :serial_oar, :parallel_oar, :parallel_oar,
                   :parallel_oar, :normal],
            fractions=30:35,
            prescription=(50.0, 70.0),
            energies=(6,),
            angles=collect(0.0:40.0:320.0),
            voxel_weights=[0.32, 0.08, 0.12, 0.12, 0.12, 0.24],
            clinical_caps=Dict(
                :spinal_cord => 0.80, :left_parotid => 0.42,
                :right_parotid => 0.42, :oral_cavity => 0.68,
                :normal_tissue => 0.76,
            ),
            tail_fractions=Dict(
                :ptv => 0.90, :spinal_cord => 0.05,
                :left_parotid => 0.50, :right_parotid => 0.50,
                :oral_cavity => 0.30, :normal_tissue => 0.20,
            ),
            volumes_cc=Dict(
                :ptv => (90.0, 420.0), :spinal_cord => (12.0, 35.0),
                :left_parotid => (18.0, 42.0), :right_parotid => (18.0, 42.0),
                :oral_cavity => (45.0, 130.0),
                :normal_tissue => (3_500.0, 8_000.0),
            ),
        )
    elseif profile == :c_shape
        return (
            body=(15.0, 15.0, 10.0),
            structures=[:ptv, :core, :normal_tissue],
            kinds=[:target, :serial_oar, :normal],
            fractions=25:30,
            prescription=(48.0, 54.0),
            energies=(6,),
            angles=collect(0.0:40.0:320.0),
            voxel_weights=[0.40, 0.15, 0.45],
            clinical_caps=Dict(:core => 0.50, :normal_tissue => 0.76),
            tail_fractions=Dict(:ptv => 0.95, :core => 0.05,
                                :normal_tissue => 0.20),
            volumes_cc=Dict(
                :ptv => (80.0, 180.0), :core => (20.0, 60.0),
                :normal_tissue => (5_000.0, 10_000.0),
            ),
        )
    elseif profile == :liver
        return (
            body=(19.0, 14.0, 13.0),
            structures=[:ptv, :healthy_liver, :left_kidney, :right_kidney,
                        :heart, :normal_tissue],
            kinds=[:target, :parallel_oar, :parallel_oar, :parallel_oar,
                   :serial_oar, :normal],
            fractions=10:30,
            prescription=(45.0, 60.0),
            energies=(6, 10),
            angles=[5.0, 55.0, 105.0, 155.0, 205.0, 255.0, 305.0],
            voxel_weights=[0.22, 0.35, 0.10, 0.10, 0.08, 0.15],
            clinical_caps=Dict(
                :healthy_liver => 0.68, :left_kidney => 0.48,
                :right_kidney => 0.48, :heart => 0.62,
                :normal_tissue => 0.78,
            ),
            tail_fractions=Dict(
                :ptv => 0.95, :healthy_liver => 0.50,
                :left_kidney => 0.30, :right_kidney => 0.30,
                :heart => 0.10, :normal_tissue => 0.20,
            ),
            volumes_cc=Dict(
                :ptv => (35.0, 180.0), :healthy_liver => (900.0, 1_900.0),
                :left_kidney => (100.0, 190.0), :right_kidney => (100.0, 190.0),
                :heart => (450.0, 850.0),
                :normal_tissue => (8_000.0, 16_000.0),
            ),
        )
    elseif profile == :lung
        return (
            body=(17.0, 13.0, 14.0),
            structures=[:ptv, :ipsilateral_lung, :contralateral_lung, :heart,
                        :esophagus, :spinal_cord, :normal_tissue],
            kinds=[:target, :parallel_oar, :parallel_oar, :parallel_oar,
                   :serial_oar, :serial_oar, :normal],
            fractions=25:35,
            prescription=(50.0, 66.0),
            energies=(6, 10),
            angles=collect(0.0:40.0:320.0),
            voxel_weights=[0.18, 0.24, 0.20, 0.10, 0.07, 0.05, 0.16],
            clinical_caps=Dict(
                :ipsilateral_lung => 0.62, :contralateral_lung => 0.40,
                :heart => 0.58, :esophagus => 0.70, :spinal_cord => 0.72,
                :normal_tissue => 0.80,
            ),
            tail_fractions=Dict(
                :ptv => 0.95, :ipsilateral_lung => 0.35,
                :contralateral_lung => 0.35, :heart => 0.30,
                :esophagus => 0.10, :spinal_cord => 0.05,
                :normal_tissue => 0.20,
            ),
            volumes_cc=Dict(
                :ptv => (25.0, 180.0), :ipsilateral_lung => (1_100.0, 2_500.0),
                :contralateral_lung => (1_200.0, 2_700.0),
                :heart => (450.0, 850.0), :esophagus => (25.0, 70.0),
                :spinal_cord => (18.0, 45.0),
                :normal_tissue => (7_000.0, 15_000.0),
            ),
        )
    elseif profile == :breast
        return (
            body=(18.0, 13.0, 12.0),
            structures=[:ptv, :ipsilateral_lung, :contralateral_lung, :heart,
                        :contralateral_breast, :normal_tissue],
            kinds=[:target, :parallel_oar, :parallel_oar, :parallel_oar,
                   :parallel_oar, :normal],
            fractions=15:28,
            prescription=(40.0, 52.0),
            energies=(6, 10),
            # Opposed tangents plus an anterior modulation field form a compact
            # surrogate for common whole-breast IMRT arrangements.
            angles=[115.0, 145.0, 0.0, 295.0, 325.0],
            voxel_weights=[0.26, 0.20, 0.14, 0.09, 0.08, 0.23],
            clinical_caps=Dict(
                :ipsilateral_lung => 0.48, :contralateral_lung => 0.25,
                :heart => 0.38, :contralateral_breast => 0.24,
                :normal_tissue => 0.72,
            ),
            tail_fractions=Dict(
                :ptv => 0.95, :ipsilateral_lung => 0.30,
                :contralateral_lung => 0.20, :heart => 0.20,
                :contralateral_breast => 0.20, :normal_tissue => 0.20,
            ),
            volumes_cc=Dict(
                :ptv => (350.0, 1_100.0), :ipsilateral_lung => (900.0, 2_200.0),
                :contralateral_lung => (1_100.0, 2_500.0),
                :heart => (450.0, 850.0), :contralateral_breast => (300.0, 1_000.0),
                :normal_tissue => (7_000.0, 15_000.0),
            ),
        )
    end
    error("Unknown radiotherapy profile: $profile")
end

function _rt_allocate_voxels(spec, n_voxels::Int)
    n_structures = length(spec.structures)
    n_voxels >= 2 * n_structures ||
        error("Radiotherapy cases require at least two voxels per structure")
    counts = fill(2, n_structures)
    remaining = n_voxels - sum(counts)
    raw = remaining .* spec.voxel_weights ./ sum(spec.voxel_weights)
    counts .+= floor.(Int, raw)
    left = n_voxels - sum(counts)
    order = sortperm(collect(1:n_structures);
                     by=i -> (-(raw[i] - floor(raw[i])), i))
    for k in 1:left
        counts[order[k]] += 1
    end
    return counts
end

function _rt_local_layout(n::Int)
    n_rows = max(1, round(Int, sqrt(n)))
    row_counts = fill(fld(n, n_rows), n_rows)
    row_counts[1:mod(n, n_rows)] .+= 1
    u_fraction = Float64[]
    z_fraction = Float64[]
    width_fraction = Float64[]
    row_indices = Vector{Vector{Int}}(undef, n_rows)
    cursor = 1
    for row in 1:n_rows
        count = row_counts[row]
        row_indices[row] = collect(cursor:(cursor + count - 1))
        for column in 1:count
            push!(u_fraction, (column - 0.5) / count)
            push!(z_fraction, (row - 0.5) / n_rows)
            push!(width_fraction, 1.0 / count)
        end
        cursor += count
    end

    edge_set = Set{Tuple{Int,Int}}()
    for row in 1:n_rows
        indices = row_indices[row]
        for k in 1:(length(indices) - 1)
            push!(edge_set, (indices[k], indices[k + 1]))
        end
        row == n_rows && continue
        below = row_indices[row + 1]
        # Connect nearest centers in both directions. This keeps the TV graph
        # connected when adjacent balanced rows differ by one beamlet.
        for a in indices
            b = below[argmin(abs.(u_fraction[below] .- u_fraction[a]))]
            push!(edge_set, minmax(a, b))
        end
        for b in below
            a = indices[argmin(abs.(u_fraction[indices] .- u_fraction[b]))]
            push!(edge_set, minmax(a, b))
        end
    end
    return u_fraction, z_fraction, width_fraction, 1.0 / n_rows,
           sort(collect(edge_set))
end

_rt_local_edges(n::Int) = last(_rt_local_layout(n))

_rt_edge_count(n_beams::Int, beamlets_per_beam::Int) =
    n_beams * length(_rt_local_edges(beamlets_per_beam))

function _rt_variable_count(formulation::Symbol, n_beamlets::Int, n_edges::Int,
                            n_voxels::Int, counts::Vector{Int},
                            n_structures::Int, n_beams::Int, n_scenarios::Int)
    formulation == :mean_tail_dose &&
        return n_beamlets + n_edges + n_voxels + n_structures
    formulation == :minmax_deviation &&
        return n_beamlets + n_edges + n_voxels + counts[1] + 1
    formulation == :robust_fluence &&
        return n_beamlets + n_edges + n_scenarios * (n_voxels + counts[1])
    formulation == :beam_angle_selection &&
        return n_beamlets + n_edges + n_voxels + counts[1] + n_beams
    return n_beamlets + n_edges + n_voxels + counts[1]
end

function _rt_plan_dimensions(target_variables::Int, formulation::Symbol,
                             profile::Symbol; angles=nothing,
                             n_scenarios::Int=3)
    target = max(target_variables, 24)
    base_spec = _rt_profile_spec(profile)
    spec = angles === nothing ? base_spec : merge(base_spec, (angles=angles,))
    n_beams = length(spec.angles)
    minimum_voxels = 2 * length(spec.structures)
    # Clinical/downsampled FMO data contain substantially more sampled voxels
    # than beamlets. Beamlets use about 16% of the complete LP variable budget;
    # their two-dimensional TV edges use another 20-30%, leaving the majority
    # for voxel dose auxiliaries. The slightly richer aperture grid materially
    # improves target coverage for concave and multi-lobed targets.
    center = max(1, round(Int, 0.16 * target / n_beams))
    candidates = unique(vcat(1, collect(max(1, center - 100):(center + 100))))
    best_score = (typemax(Int), Inf, Inf)
    best = nothing

    for per_beam in candidates
        n_beamlets = n_beams * per_beam
        n_edges = _rt_edge_count(n_beams, per_beam)
        fixed = n_beamlets + n_edges
        fixed >= max(2 * target, 200) && continue

        voxel_candidates = Int[]
        if formulation == :mean_tail_dose
            push!(voxel_candidates,
                  max(minimum_voxels, target - fixed - length(spec.structures)))
        else
            # Every remaining formulation has an auxiliary count monotone in
            # V. A short binary search finds the nearest count without scanning
            # when million-column targets are used.
            low, high = minimum_voxels, max(minimum_voxels, target)
            while low < high
                middle = fld(low + high, 2)
                counts = _rt_allocate_voxels(spec, middle)
                actual = _rt_variable_count(
                    formulation, n_beamlets, n_edges, middle, counts,
                    length(spec.structures), n_beams, n_scenarios,
                )
                if actual < target
                    low = middle + 1
                else
                    high = middle
                end
            end
            append!(voxel_candidates,
                    max.(minimum_voxels, collect((low - 2):(low + 2))))
        end

        for n_voxels in unique(voxel_candidates)
            counts = _rt_allocate_voxels(spec, n_voxels)
            actual = _rt_variable_count(
                formulation, n_beamlets, n_edges, n_voxels, counts,
                length(spec.structures), n_beams, n_scenarios,
            )
            score = (abs(actual - target),
                     abs(n_beamlets / actual - 0.16),
                     abs(n_voxels / actual - 0.45))
            if score < best_score
                best_score = score
                best = (per_beam, n_voxels, counts, actual)
            end
        end
    end
    best === nothing && error("Could not size radiotherapy problem")
    return best
end

function _rt_rand_ellipsoid(rng::AbstractRNG, center, radii)
    while true
        p = ntuple(k -> center[k] + radii[k] * (2rand(rng) - 1), 3)
        sum(((p[k] - center[k]) / radii[k])^2 for k in 1:3) <= 1 && return p
    end
end

function _rt_sample_location(rng::AbstractRNG, profile::Symbol,
                             structure::Symbol, body)
    if profile == :prostate
        structure == :ptv && return _rt_rand_ellipsoid(rng, (0.0, 0.0, 0.0),
                                                        (3.2, 2.5, 3.8))
        structure == :rectum && return _rt_rand_ellipsoid(rng, (0.0, -2.9, 0.0),
                                                           (1.0, 1.2, 4.2))
        structure == :bladder && return _rt_rand_ellipsoid(rng, (0.0, 2.5, 3.2),
                                                            (3.0, 2.6, 2.8))
        structure == :left_femoral_head &&
            return _rt_rand_ellipsoid(rng, (-6.2, -0.8, -1.2), (1.8, 1.8, 2.0))
        structure == :right_femoral_head &&
            return _rt_rand_ellipsoid(rng, (6.2, -0.8, -1.2), (1.8, 1.8, 2.0))
    elseif profile == :head_neck
        if structure == :ptv
            center, radii = rand(rng) < 0.62 ?
                            ((0.0, 0.3, 2.5), (4.8, 3.8, 4.2)) :
                            ((0.0, -0.5, -3.5), (5.4, 3.5, 3.0))
            return _rt_rand_ellipsoid(rng, center, radii)
        end
        structure == :spinal_cord &&
            return _rt_rand_ellipsoid(rng, (0.0, -4.2, -0.5), (0.7, 0.7, 7.0))
        structure == :left_parotid &&
            return _rt_rand_ellipsoid(rng, (-4.5, -0.3, 2.5), (1.6, 1.8, 2.8))
        structure == :right_parotid &&
            return _rt_rand_ellipsoid(rng, (4.5, -0.3, 2.5), (1.6, 1.8, 2.8))
        structure == :oral_cavity &&
            return _rt_rand_ellipsoid(rng, (0.0, 3.0, 0.0), (3.0, 2.0, 2.8))
    elseif profile == :c_shape
        if structure == :ptv
            radius = sqrt(1.5^2 + rand(rng) * (3.7^2 - 1.5^2))
            angle = deg2rad(35.0 + 290.0 * rand(rng))
            return (radius * cos(angle), radius * sin(angle), 8rand(rng) - 4)
        elseif structure == :core
            radius = sqrt(rand(rng))
            angle = 2pi * rand(rng)
            return (radius * cos(angle), radius * sin(angle), 10rand(rng) - 5)
        end
    elseif profile == :liver
        structure == :ptv &&
            return _rt_rand_ellipsoid(rng, (4.8, 0.4, 0.0), (2.8, 2.5, 3.0))
        if structure == :healthy_liver
            while true
                p = _rt_rand_ellipsoid(rng, (4.0, 0.0, 0.0), (7.5, 5.0, 5.5))
                sum(((p[k] - (4.8, 0.4, 0.0)[k]) /
                     (2.8, 2.5, 3.0)[k])^2 for k in 1:3) > 1 && return p
            end
        end
        structure == :left_kidney &&
            return _rt_rand_ellipsoid(rng, (-5.0, -3.4, -1.8), (2.0, 1.7, 3.2))
        structure == :right_kidney &&
            return _rt_rand_ellipsoid(rng, (5.5, -3.6, -1.8), (2.0, 1.7, 3.2))
        structure == :heart &&
            return _rt_rand_ellipsoid(rng, (-0.8, 0.8, 6.0), (3.2, 2.8, 2.8))
    elseif profile == :lung
        structure == :ptv &&
            return _rt_rand_ellipsoid(rng, (5.2, 0.5, 1.0), (2.7, 2.4, 3.2))
        if structure == :ipsilateral_lung
            while true
                p = _rt_rand_ellipsoid(rng, (5.3, -0.2, 0.5), (5.2, 4.2, 8.5))
                sum(((p[k] - (5.2, 0.5, 1.0)[k]) /
                     (2.7, 2.4, 3.2)[k])^2 for k in 1:3) > 1 && return p
            end
        end
        structure == :contralateral_lung &&
            return _rt_rand_ellipsoid(rng, (-5.3, -0.2, 0.5), (5.2, 4.2, 8.5))
        structure == :heart &&
            return _rt_rand_ellipsoid(rng, (-1.0, 1.0, -2.0), (3.5, 3.0, 4.5))
        structure == :esophagus &&
            return _rt_rand_ellipsoid(rng, (0.0, -2.0, 0.0), (0.8, 0.8, 9.0))
        structure == :spinal_cord &&
            return _rt_rand_ellipsoid(rng, (0.0, -5.0, 0.0), (0.7, 0.7, 10.0))
    elseif profile == :breast
        structure == :ptv &&
            return _rt_rand_ellipsoid(rng, (-6.2, 5.4, 0.0), (4.8, 2.0, 6.5))
        structure == :ipsilateral_lung &&
            return _rt_rand_ellipsoid(rng, (-5.0, -1.0, 0.0), (5.0, 4.0, 8.5))
        structure == :contralateral_lung &&
            return _rt_rand_ellipsoid(rng, (5.0, -1.0, 0.0), (5.0, 4.0, 8.5))
        structure == :heart &&
            return _rt_rand_ellipsoid(rng, (-1.6, 0.8, -2.0), (3.4, 2.8, 4.2))
        structure == :contralateral_breast &&
            return _rt_rand_ellipsoid(rng, (6.2, 5.4, 0.0), (4.8, 2.0, 6.5))
    end
    return _rt_rand_ellipsoid(rng, (0.0, 0.0, 0.0), body)
end

function _rt_anatomy(rng::AbstractRNG, profile::Symbol, spec, counts)
    n_voxels = sum(counts)
    locations = zeros(Float64, n_voxels, 3)
    voxel_structure = Vector{Symbol}(undef, n_voxels)
    voxel_volume_cc = zeros(Float64, n_voxels)
    structure_voxels = Dict{Symbol,Vector{Int}}()
    cursor = 1
    for (s, structure) in enumerate(spec.structures)
        indices = collect(cursor:(cursor + counts[s] - 1))
        structure_voxels[structure] = indices
        volume_range = spec.volumes_cc[structure]
        total_volume = volume_range[1] + rand(rng) * (volume_range[2] - volume_range[1])
        # Samples represent an importance-reduced dose grid rather than equal
        # physical cubes. Mild lognormal cell weights approximate the unequal
        # represented volumes created by stratified/importance downsampling.
        relative_volumes = exp.(0.22 .* randn(rng, counts[s]))
        relative_volumes .*= total_volume / sum(relative_volumes)
        for (local_index, i) in enumerate(indices)
            location = _rt_sample_location(rng, profile, structure, spec.body)
            locations[i, :] .= location
            voxel_structure[i] = structure
            voxel_volume_cc[i] = relative_volumes[local_index]
        end
        cursor += counts[s]
    end
    return locations, voxel_structure, structure_voxels, voxel_volume_cc
end

function _rt_beamlets(profile::Symbol, spec, per_beam::Int,
                      target_locations::Matrix{Float64})
    beam_of = Int[]
    us = Float64[]
    zs = Float64[]
    widths = Float64[]
    heights = Float64[]
    edges = Tuple{Int,Int}[]
    u_fraction, z_fraction, width_fraction, height_fraction, local_edges =
        _rt_local_layout(per_beam)

    for (beam, angle_deg) in enumerate(spec.angles)
        angle = deg2rad(angle_deg)
        lateral = -sin(angle) .* target_locations[:, 1] .+
                   cos(angle) .* target_locations[:, 2]
        margin = profile == :head_neck ? 1.2 : 0.9
        u_min, u_max = minimum(lateral) - margin, maximum(lateral) + margin
        z_min = minimum(target_locations[:, 3]) - margin
        z_max = maximum(target_locations[:, 3]) + margin
        span_u = u_max - u_min
        span_z = z_max - z_min
        offset = length(beam_of)
        for k in 1:per_beam
            push!(beam_of, beam)
            push!(us, u_min + u_fraction[k] * span_u)
            push!(zs, z_min + z_fraction[k] * span_z)
            push!(widths, max(width_fraction[k] * span_u, 0.45))
            push!(heights, max(height_fraction * span_z, 0.45))
        end
        append!(edges, [(offset + a, offset + b) for (a, b) in local_edges])
    end
    return beam_of, us, zs, widths, heights, edges
end

function _rt_tissue_factor(structure::Symbol)
    structure in (:left_femoral_head, :right_femoral_head) && return 1.08
    structure in (:spinal_cord, :core) && return 1.02
    structure in (:healthy_liver, :liver) && return 0.96
    structure in (:ipsilateral_lung, :contralateral_lung) && return 0.94
    return 1.0
end

function _rt_dose_matrix(spec, locations, voxel_structure, beam_of,
                         us, zs, widths, heights, energy_mv::Int;
                         setup_shift_cm::NTuple{3,Float64}=(0.0, 0.0, 0.0))
    n_voxels = size(locations, 1)
    n_beamlets = length(beam_of)
    rows = Int[]
    columns = Int[]
    values = Float64[]
    beam_ranges = [findall(==(b), beam_of) for b in eachindex(spec.angles)]

    for i in 1:n_voxels
        x, y, z = locations[i, 1], locations[i, 2], locations[i, 3]
        world_x = x + setup_shift_cm[1]
        world_y = y + setup_shift_cm[2]
        world_z = z + setup_shift_cm[3]
        tissue = _rt_tissue_factor(voxel_structure[i])
        for (beam, angle_deg) in enumerate(spec.angles)
            angle = deg2rad(angle_deg)
            lateral = -sin(angle) * world_x + cos(angle) * world_y
            axial_depth = cos(angle) * world_x + sin(angle) * world_y
            projected_radius = sqrt((spec.body[1] * cos(angle))^2 +
                                    (spec.body[2] * sin(angle))^2)
            depth = clamp(axial_depth + projected_radius, 0.0,
                          2projected_radius)
            attenuation_coefficient = energy_mv <= 6 ? 0.027 : 0.022
            source_distance_scale = energy_mv <= 6 ? 95.0 : 105.0
            attenuation = exp(-attenuation_coefficient * depth) /
                          (1 + depth / source_distance_scale)^2
            nearest = 0
            nearest_q = Inf
            emitted = false
            for j in beam_ranges[beam]
                du = (lateral - us[j]) / widths[j]
                dz = (world_z - zs[j]) / heights[j]
                q = du^2 + dz^2
                if q < nearest_q
                    nearest_q = q
                    nearest = j
                end
                # A beamlet deposits appreciable primary/scatter dose only in
                # a local pencil-beam neighborhood. At clinically sized grids
                # this produces the sparse columns seen in CORT rather than a
                # dense random matrix with merely small far-field entries.
                if q <= 2.56
                    primary = exp(-0.5 * q / 0.62^2)
                    scatter = 0.035 * exp(-0.5 * q / 2.2^2)
                    raw_coefficient = attenuation * (primary + scatter)
                    coefficient = tissue * raw_coefficient
                    # Threshold the tissue-independent kernel so coincident
                    # voxels have identical sparsity patterns. This is both
                    # physical (support is geometric) and required by the
                    # exact proportional-row infeasibility certificate.
                    if raw_coefficient >= 1.0e-5
                        push!(rows, i); push!(columns, j); push!(values, coefficient)
                        emitted = true
                    end
                end
            end
            if !emitted
                # Out-of-field leakage/scatter: retain one small coefficient
                # per field so normal-tissue rows are physical rather than zero.
                coefficient = tissue * attenuation *
                              max(1.0e-5, 0.006 * exp(-0.12 * nearest_q))
                push!(rows, i); push!(columns, nearest); push!(values, coefficient)
            end
        end
    end
    return sparse(rows, columns, values, n_voxels, n_beamlets)
end

function _rt_reference_fluence(rng::AbstractRNG, spec, target_locations,
                               beam_of, us, zs, widths, heights)
    n_beamlets = length(beam_of)
    fluence = zeros(Float64, n_beamlets)
    for beam in eachindex(spec.angles)
        angle = deg2rad(spec.angles[beam])
        indices = findall(==(beam), beam_of)
        scores = zeros(length(indices))
        for (position, j) in enumerate(indices)
            for i in axes(target_locations, 1)
                lateral = -sin(angle) * target_locations[i, 1] +
                           cos(angle) * target_locations[i, 2]
                q = ((lateral - us[j]) / widths[j])^2 +
                    ((target_locations[i, 3] - zs[j]) / heights[j])^2
                scores[position] += exp(-0.5 * q)
            end
        end
        scores ./= max(maximum(scores), eps())
        phase = 2pi * rand(rng)
        for (position, j) in enumerate(indices)
            modulation = 0.06 * sin(phase + 2pi * position / length(indices))
            fluence[j] = clamp(0.50 + 0.72 * scores[position] + modulation,
                               0.18, 1.35)
        end
    end
    return fluence
end

function _rt_weighted_quantile(values::AbstractVector{<:Real},
                               weights::AbstractVector{<:Real},
                               probability::Float64)
    length(values) == length(weights) || throw(DimensionMismatch())
    isempty(values) && throw(ArgumentError("quantile requires nonempty values"))
    0.0 <= probability <= 1.0 || throw(ArgumentError("invalid probability"))
    order = sortperm(values)
    threshold = probability * sum(weights)
    cumulative = 0.0
    for index in order
        cumulative += weights[index]
        cumulative >= threshold && return Float64(values[index])
    end
    return Float64(values[last(order)])
end

function _rt_balance_reference_fluence!(fluence, dose_matrix, target, edges;
                                        allowed=trues(length(fluence)),
                                        target_weights=nothing)
    target_matrix = dose_matrix[target, :]
    column_mass = vec(sum(target_matrix; dims=1))
    active_indices = findall(allowed .& (column_mass .> 1.0e-12))
    isempty(active_indices) && error("No active beamlet irradiates the target")
    matrix = target_matrix[:, active_indices]
    weights = target_weights === nothing ? ones(length(target)) :
              Float64.(target_weights)
    weights .*= length(weights) / sum(weights)
    x = max.(fluence[active_indices], 0.0)
    delivered = matrix * x
    scale = _rt_weighted_quantile(delivered, weights, 0.5)
    scale > 0 || error("Reference fluence delivers zero target dose")
    x ./= scale

    # Nonnegative least squares solved with projected FISTA gives a much more
    # homogeneous planted target dose than independent beam scoring, without
    # introducing a solver dependency into data generation. A short power
    # iteration estimates the Lipschitz constant of A'A. The work cap keeps
    # very large corpus instances practical while small/medium cases converge
    # tightly enough to expose meaningful D95/D2 metrics.
    direction = fill(inv(sqrt(length(x))), length(x))
    for _ in 1:14
        direction = transpose(matrix) * (weights .* (matrix * direction))
        norm_direction = norm(direction)
        norm_direction > 0 || break
        direction ./= norm_direction
    end
    lipschitz = dot(direction,
                    transpose(matrix) * (weights .* (matrix * direction)))
    lipschitz = max(1.02 * lipschitz, 1.0e-10)
    iterations = nnz(matrix) > 5_000_000 ? 80 :
                 nnz(matrix) > 1_000_000 ? 140 : 360
    extrapolated = copy(x)
    momentum = 1.0
    previous_error = Inf
    for _ in 1:iterations
        residual = matrix * extrapolated .- 1.0
        candidate = max.(0.0, extrapolated .-
                               (transpose(matrix) * (weights .* residual)) ./
                               lipschitz)
        candidate_residual = matrix * candidate .- 1.0
        candidate_error = sum(weights .* abs2.(candidate_residual))
        if candidate_error > previous_error
            # Adaptive restart suppresses the oscillation FISTA can otherwise
            # exhibit around an active nonnegativity constraint.
            extrapolated .= x
            momentum = 1.0
            residual = matrix * extrapolated .- 1.0
            candidate = max.(0.0, extrapolated .-
                                   (transpose(matrix) * (weights .* residual)) ./
                                   lipschitz)
            candidate_residual = matrix * candidate .- 1.0
            candidate_error = sum(weights .* abs2.(candidate_residual))
        end
        next_momentum = (1 + sqrt(1 + 4momentum^2)) / 2
        extrapolated = candidate .+
                       ((momentum - 1) / next_momentum) .* (candidate .- x)
        x = candidate
        momentum = next_momentum
        previous_error = candidate_error
        candidate_error <= 1.0e-12 * length(target) && break
    end

    fluence .= 0.0
    fluence[active_indices] .= x
    final_scale = _rt_weighted_quantile(target_matrix * fluence, weights, 0.5)
    final_scale > 0 && (fluence ./= final_scale)
    return fluence
end

function _rt_build_case(target_variables::Int, formulation::Symbol,
                        feasibility_status::FeasibilityStatus, seed::Int;
                        profile_override::Union{Nothing,Symbol}=nothing,
                        angles=nothing, active_beams=nothing,
                        n_scenarios::Int=3)
    rng = MersenneTwister(seed)
    profile = profile_override === nothing ?
              _RT_PROFILES[mod(seed, length(_RT_PROFILES)) + 1] : profile_override
    base_spec = _rt_profile_spec(profile)
    spec = angles === nothing ? base_spec : merge(base_spec, (angles=angles,))
    energy_mv = rand(rng, spec.energies)
    per_beam, n_voxels, counts, _ =
        _rt_plan_dimensions(target_variables, formulation, profile;
                            angles=spec.angles, n_scenarios=n_scenarios)
    locations, voxel_structure, structure_voxels, voxel_volume_cc =
        _rt_anatomy(rng, profile, spec, counts)

    conflict_indices = nothing
    if feasibility_status == infeasible
        organ_candidates = [s for (s, kind) in zip(spec.structures, spec.kinds)
                            if kind in (:serial_oar, :parallel_oar)]
        organ = rand(rng, organ_candidates)
        target_voxel = rand(rng, structure_voxels[:ptv])
        organ_voxel = rand(rng, structure_voxels[organ])
        # Coincident target/OAR samples represent an overlap in contoured
        # structures, which public radiotherapy data formats explicitly allow.
        locations[organ_voxel, :] .= locations[target_voxel, :]
        conflict_indices = (target_voxel, organ_voxel, organ)
    end

    target_locations = locations[structure_voxels[:ptv], :]
    beam_of, us, zs, widths, heights, edges =
        _rt_beamlets(profile, spec, per_beam, target_locations)
    dose_matrix = _rt_dose_matrix(spec, locations, voxel_structure,
                                  beam_of, us, zs, widths, heights, energy_mv)
    reference_fluence = _rt_reference_fluence(
        rng, spec, target_locations, beam_of, us, zs, widths, heights,
    )
    allowed = active_beams === nothing ? trues(length(reference_fluence)) :
              [beam in active_beams for beam in beam_of]
    reference_fluence[.!allowed] .= 0.0
    _rt_balance_reference_fluence!(reference_fluence, dose_matrix,
                                   structure_voxels[:ptv], edges; allowed=allowed,
                                   target_weights=
                                       voxel_volume_cc[structure_voxels[:ptv]])
    reference_dose = dose_matrix * reference_fluence
    target_indices = structure_voxels[:ptv]
    median_target_dose = _rt_weighted_quantile(
        reference_dose[target_indices], voxel_volume_cc[target_indices], 0.5,
    )
    median_target_dose > 0 || error("Generated target receives no radiation")
    dose_normalization = inv(median_target_dose)
    dose_matrix .*= dose_normalization
    reference_dose .*= dose_normalization

    prescription_gy = round(spec.prescription[1] +
                            rand(rng) * (spec.prescription[2] - spec.prescription[1]);
                            digits=1)
    n_fractions = rand(rng, spec.fractions)
    case = RadiotherapyCaseData(
        profile, prescription_gy, n_fractions, energy_mv, spec.body,
        copy(spec.structures), copy(spec.kinds), structure_voxels,
        voxel_structure, locations, voxel_volume_cc, copy(spec.angles),
        beam_of, us, zs,
        widths, heights, edges, dose_matrix, dose_normalization,
        reference_fluence, reference_dose,
    )
    return case, spec, conflict_indices, rng
end

function _rt_hard_limits(case::RadiotherapyCaseData, spec,
                         feasibility_status::FeasibilityStatus,
                         conflict_indices, rng::AbstractRNG)
    target_dose = case.reference_dose[case.structure_voxels[:ptv]]
    # These are pointwise safety rails, not mislabeled D95/D2 prescriptions.
    # DVH-volume goals are represented separately by the mean-tail variant.
    target_floor = min(0.88, 0.99 * minimum(target_dose))
    target_ceiling = max(1.12, 1.01 * maximum(target_dose))
    structure_max = Dict{Symbol,Float64}()

    if feasibility_status == unknown
        target_floor = 0.82 + 0.14 * rand(rng)
        target_ceiling = 1.08 + 0.28 * rand(rng)
        severity = exp(log(0.86) + rand(rng) * (log(2.30) - log(0.86)))
        kind_by_structure = Dict(zip(spec.structures, spec.kinds))
        for structure in spec.structures[2:end]
            kind = kind_by_structure[structure]
            # Point maxima are natural for serial organs. Parallel-organ DVH
            # goals live in the soft/tail formulation, so their hard row is a
            # looser global safety ceiling instead of applying a mean-dose
            # target to every voxel.
            baseline = kind == :serial_oar ? spec.clinical_caps[structure] :
                       kind == :parallel_oar ? max(1.02, spec.clinical_caps[structure]) :
                       1.10
            structure_max[structure] = baseline * severity *
                                       (0.94 + 0.12 * rand(rng))
        end
        return target_floor, target_ceiling, structure_max, nothing, nothing
    end

    for structure in spec.structures[2:end]
        achieved = maximum(case.reference_dose[case.structure_voxels[structure]])
        clinical = spec.clinical_caps[structure] * (0.94 + 0.12 * rand(rng))
        structure_max[structure] = max(clinical, 1.06 * achieved)
    end

    witness = feasibility_status == feasible ?
              RadiotherapyFluenceWitness(copy(case.reference_fluence)) : nothing
    certificate = nothing
    if feasibility_status == infeasible
        target_voxel, organ_voxel, organ = conflict_indices
        multiplier = _rt_tissue_factor(organ) / _rt_tissue_factor(:ptv)
        structure_max[organ] = multiplier * target_floor *
                               (0.72 + 0.14 * rand(rng))
        certificate = RadiotherapyDoseConflictCertificate(
            target_voxel, organ_voxel, organ, multiplier, target_floor,
            structure_max[organ],
        )
    end
    return target_floor, target_ceiling, structure_max, witness, certificate
end

function _rt_dose_expressions(model::Model, fluence, dose_matrix)
    expressions = [AffExpr(0.0) for _ in axes(dose_matrix, 1)]
    for j in axes(dose_matrix, 2)
        for pointer in nzrange(dose_matrix, j)
            i = rowvals(dose_matrix)[pointer]
            add_to_expression!(expressions[i], nonzeros(dose_matrix)[pointer],
                               fluence[j])
        end
    end
    return expressions
end

function _rt_certificate_is_valid(problem)
    certificate = problem.infeasibility_certificate
    certificate === nothing && return false
    matrix = problem.case_data.dose_matrix
    target_row = Array(matrix[certificate.target_voxel, :])
    organ_row = Array(matrix[certificate.organ_voxel, :])
    return isapprox(organ_row, certificate.multiplier .* target_row;
                    rtol=1.0e-12, atol=1.0e-12) &&
           certificate.organ_upper_bound <
               certificate.multiplier * certificate.target_lower_bound &&
           problem.target_floor == certificate.target_lower_bound &&
           problem.structure_max[certificate.organ] ==
               certificate.organ_upper_bound
end

function _rt_witness_is_valid(problem; atol::Float64=1.0e-9)
    witness = problem.feasible_witness
    witness === nothing && return false
    dose = problem.case_data.dose_matrix * witness.fluence
    target = problem.case_data.structure_voxels[:ptv]
    all(dose[target] .>= problem.target_floor - atol) || return false
    all(dose[target] .<= problem.target_ceiling + atol) || return false
    for (structure, upper) in problem.structure_max
        all(dose[problem.case_data.structure_voxels[structure]] .<= upper + atol) ||
            return false
    end
    return true
end

function _rt_add_hard_constraints!(model, problem, dose)
    target = problem.case_data.structure_voxels[:ptv]
    @constraint(model, target_floor[i in target], dose[i] >= problem.target_floor)
    @constraint(model, target_ceiling[i in target],
                dose[i] <= problem.target_ceiling)
    upper_rows = Dict{Symbol,Any}()
    for structure in problem.case_data.structure_names[2:end]
        indices = problem.case_data.structure_voxels[structure]
        upper = problem.structure_max[structure]
        upper_rows[structure] = @constraint(
            model, [i in indices], dose[i] <= upper,
            base_name="$(structure)_maximum",
        )
    end
    model[:structure_maximum] = upper_rows
    return nothing
end
