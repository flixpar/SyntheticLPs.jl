# Radiotherapy Fluence-Map Planning

The `radiotherapy` category generates IMRT fluence-map optimization (FMO) LPs
and MILPs. It combines a spatial synthetic patient and sparse photon
dose-influence matrix with five complementary inverse-planning formulations. The
result is intended for optimization research and solver testing; it is not a
clinical dose engine and must not be used to plan patient treatment.

## Shared patient and beam data

Every problem stores a `RadiotherapyCaseData` in `problem.case_data`. A seed
selects one of six profiles in a reproducible rotation:

| Profile | Structures | Prescription | Fractions | Coplanar fields |
| --- | --- | ---: | ---: | ---: |
| `prostate` | PTV, rectum, bladder, femoral heads, normal tissue | 74–80 Gy | 35–44 | 7, TG-119 50-degree pattern |
| `head_neck` | PTV, cord, parotids, oral cavity, normal tissue | 50–70 Gy | 30–35 | 9 at 40-degree intervals |
| `c_shape` | annular PTV, central avoidance core, normal tissue | 48–54 Gy | 25–30 | 9 at 40-degree intervals |
| `liver` | PTV, uninvolved liver, kidneys, heart, normal tissue | 45–60 Gy | 10–30 | 7 offset fields |
| `lung` | PTV, lungs, heart, esophagus, cord, normal tissue | 50–66 Gy | 25–35 | 9 at 40-degree intervals |
| `breast` | PTV, lungs, heart, contralateral breast, normal tissue | 40–52 Gy | 15–28 | 4 tangents plus anterior field |

Voxel coordinates are stratified by contoured structure rather than sampled as
independent labels. For example, the prostate PTV is an ellipsoid abutting a
posterior rectum, the head-and-neck cord is a narrow posterior cylinder between
paired parotids, and the C-shape follows the TG-119 1.5–3.7 cm annulus around a
1 cm core. The voxel sample is an optimization-grid reduction: it preserves
structures, geometry, and a sampled physical volume in cc for every point while
scaling with `target_variables`, rather than
materializing the hundreds of thousands of voxels in a full CT.

Each field has a balanced rectangular-like 2-D beamlet grid covering the
beam's-eye-view PTV plus a margin. Neighbor pairs in each grid are stored in
`beamlet_edges`. The dose matrix `D` is generated from a photon pencil-beam
surrogate:

```text
D[i,j] = tissue_factor[i] * depth_attenuation[i,field(j)]
         * (narrow_primary_kernel[i,j] + broad_scatter_kernel[i,j])
```

Depth attenuation combines energy-dependent exponential attenuation and source
divergence; each case records a sampled 6 or 10 MV nominal beam energy.
Lateral/longitudinal Gaussian kernels make nearby beamlets affect nearby
voxels similarly; negligible far-field coefficients are omitted. One small
leakage/scatter coefficient per out-of-field voxel and field prevents empty
normal-tissue rows. Thus `D` is nonnegative, sparse, spatially correlated, and
has no empty row or column. It is not an i.i.d. random matrix.

A deterministic projected nonnegative least-squares fit balances a positive
reference fluence against the PTV without requiring an optimization solver at
generation time. Dose is normalized so the volume-median PTV dose is 1.0, the
prescription. At ordinary corpus sizes the deterministic audit requires PTV
D95 at least 0.90 and D2 at most 1.10. This reference is the feasible witness
where one is promised. Pointwise target floors/ceilings are deliberately
documented as safety rails; the generator does not mislabel them as DVH D98 or
D2 constraints.

The field counts, angles, and problem scale are grounded in TG-119 and the
public CORT benchmark. CORT uses 0.5–1 cm beamlets; this generator approaches
that resolution as the requested size grows and treats each beamlet as a
coarser aggregate at small solver-test sizes. CORT reports selected
clinical problems with about 1,166–11,489 beamlets, 6,770–22,682 target voxels,
and sparse dose matrices; this generator creates reduced problems and grows
toward that sparse regime as the requested size grows. Small exact-size
benchmarks necessarily use coarser grids and fewer sampled voxels per beamlet.

## `weighted_deviation`

The default variant uses voxelwise piecewise-linear dose penalties. Let `x_j`
be nonnegative beamlet fluence and `d_i = sum_j D[i,j] x_j`. Target underdose
`u_i` and overdose `o_i` satisfy:

```text
d_i + u_i >= desired_i             target voxels
d_i - o_i <= desired_i             all voxels
u_i, o_i >= 0
```

For every adjacent beamlet pair `(j,k)`, `v_jk` linearizes absolute fluence
variation:

```text
v_jk >= x_j - x_k
v_jk >= x_k - x_j
v_jk >= 0
```

The objective minimizes structure-normalized weighted deviations, a small
total-fluence term, and anisotropic total variation:

```text
min  sum_i w_under[i] * u_i + sum_i w_over[i] * o_i
     + lambda_MU * sum_j x_j + lambda_TV * sum_(j,k) v_jk
```

Target lower/upper bounds and per-structure voxel maximums are hard rows. Soft
penalties therefore model preference tradeoffs without turning every requested
infeasible instance into a feasible one.

Variable count is exact or within one variable for normal targets:

```text
beamlets + target voxels + all voxels + beamlet adjacency edges
```

## `mean_tail_dose`

This variant replaces voxelwise deviation slacks with linear mean-tail-dose
constraints, the radiotherapy analogue of CVaR. It constrains the mean dose in
the coldest fraction of the PTV and the hottest selected fraction of every OAR
and normal-tissue structure.

For a hot-tail fraction `alpha_s`, free threshold `eta_s`, and nonnegative
excess `z_i`:

```text
z_i >= d_i - eta_s
eta_s + sum_(i in s) volume_i * z_i
        / (alpha_s * sum_(i in s) volume_i) <= hot_tail_bound_s
```

The PTV cold tail reverses the deviation:

```text
z_i >= eta_PTV - d_i
eta_PTV - sum_(i in PTV) volume_i * z_i
          / (alpha_PTV * sum_(i in PTV) volume_i) >= cold_tail_bound_PTV
```

The fractions mirror common DVH goals: 90–95% cold target tails, 5–10% hot
tails for serial structures, and 30–50% hot tails for parallel organs. This is
a convex surrogate for exact dose-volume constraints, whose feasible regions
are nonconvex. The objective minimizes structure-weighted mean non-target dose,
total fluence, and the same beamlet total variation.

Variable count is exact or within one variable for normal targets:

```text
beamlets + all voxels + structures + beamlet adjacency edges
```

## `minmax_deviation`

This LP retains the default variant's hinge variables but introduces one
epigraph variable `rho`. Every target underdose and structure-weighted
overdose is bounded by `rho`; the objective minimizes `rho` plus small fluence
and total-variation terms. It is useful when a sum objective would let a large
structure statistically hide a small, badly served region.

```text
rho >= importance_under[i] * u_i       target voxels
rho >= importance_over[i] * o_i        all voxels
min rho + lambda_MU * sum_j x_j + lambda_TV * sum_(j,k) v_jk
```

## `robust_fluence`

This deterministic-equivalent LP uses one fluence map for three dose matrices:
the nominal setup and two coherent rigid patient shifts of approximately
0.25–0.45 cm laterally and up to 0.2 cm longitudinally. A scenario is produced
by moving every anatomical sample together relative to the fixed beamlet grid,
then recomputing the sparse pencil-beam matrix. It is therefore spatially
coherent—not independent noise on matrix entries. Hard safety rows and
underdose/overdose hinges hold in every scenario, while the objective averages
deviation costs across scenarios.

```text
d_i^s = sum_j D^s[i,j] x_j
target_floor <= d_i^s <= target_ceiling             every scenario
d_i^s <= structure_max                              OARs, every scenario
```

The feasible witness is audited against all three matrices by
`_rt_robust_witness_is_valid`.

## `beam_angle_selection`

This variant is a MILP before the package's optional integrality relaxation.
Twelve candidate coplanar fields have binary open variables `y_b`; every
beamlet is linked to its field by a calibrated upper bound.

```text
x_j <= M_j * y_field(j)
minimum_open <= sum_b y_b <= maximum_open
```

The field count is a range rather than an equality, so the positive
`sum_b y_b` objective term changes the optimizer's choice—it is not a constant
masquerading as a selection cost. The stored feasible witness includes both
fluence and the planted open-field set. Use `relax_integer=false` to retain the
natural binary formulation.

## Feasibility controls

- `feasible`: hard limits and mean-tail bounds are placed outside metrics of
  `case_data.reference_fluence`. `feasible_witness` stores that complete
  fluence vector. `_rt_witness_is_valid` checks hard rows and
  `_rt_mean_tail_witness_is_valid` additionally checks every tail goal.
  Robust witnesses are checked across all scenarios; beam-selection witnesses
  additionally record and validate the open fields and linking bounds.
- `infeasible`: one sampled target voxel and one OAR voxel are coincident,
  representing an overlapping contour sample. Their influence rows obey
  `D[oar,:] = lambda * D[target,:]`, while the OAR upper bound is strictly
  below `lambda * target_floor`. The stored
  `RadiotherapyDoseConflictCertificate` is an exact algebraic contradiction,
  and `_rt_certificate_is_valid` verifies it without solving.
- `unknown`: clinical target and OAR tightness is sampled independently of the
  reference fluence. No witness or certificate is exposed and
  `resolved_status` remains `unknown`. Representative deterministic samples
  contain both feasible and overconstrained plans.

All randomness is local to the constructor. Reusing `(variant, target, status,
seed)` reproduces the complete anatomy, beam grid, sparse matrix, limits, and
JuMP model without changing Julia's global RNG state.

## Research basis

- Craft et al., [Shared data for intensity modulated radiation therapy (IMRT)
  optimization research: the CORT dataset](https://doi.org/10.1186/2047-217X-3-37),
  GigaScience 2014. Source for public case sizes, beamlet/voxel metadata,
  sparse dose-influence matrices, and the basic linear FMO example.
- Ezzell et al., [IMRT commissioning: Multiple institution planning and
  dosimetry comparisons, a report from AAPM Task Group
  119](https://doi.org/10.1118/1.3238104), Medical Physics 2009. Source for the
  prostate, head-and-neck, and C-shape structures, dose goals, and field
  arrangements.
- Romeijn et al., [A novel linear programming approach to fluence map
  optimization for intensity modulated radiation therapy treatment
  planning](https://doi.org/10.1118/1.1625449), Medical Physics 2003. Basis for
  piecewise-linear convex FMO.
- Romeijn et al., [A new linear programming approach to radiation therapy
  treatment planning problems](https://doi.org/10.1287/opre.1050.0261),
  Operations Research 2006. Basis for linear hot/cold mean-tail-dose
  constraints.
- Zhu et al., [Using total-variation regularization for intensity modulated
  radiation therapy inverse planning](https://doi.org/10.1088/0031-9155/53/23/002),
  Physics in Medicine and Biology 2008. Basis for adjacent-beamlet L1
  deliverability regularization.
- Chan, Bortfeld, and Tsitsiklis, [A robust approach to IMRT
  optimization](https://doi.org/10.1088/0031-9155/51/10/014), Physics in
  Medicine and Biology 2006. Basis for optimizing one plan over explicitly
  modeled motion/uncertainty realizations.
- Huang et al., [Applying mixed-integer linear programming to the non-coplanar
  beam angle optimization of intensity-modulated radiotherapy for liver
  cancer](https://doi.org/10.21037/qims-24-296), Quantitative Imaging in
  Medicine and Surgery 2024. Contemporary example of binary beam-angle
  selection coupled to IMRT planning.

## Practical notes

These problems exercise sparse but correlated dose blocks, many overlapping
structure rows, free CVaR thresholds, degenerate hinge variables, repeated
scenario blocks, grid-incidence regularization, and binary field activation.
The first four variants are pure LPs. `beam_angle_selection` is naturally a
MILP and follows the package-wide `relax_integer` setting. The anatomy and
pencil-beam engine are research-grade synthetic data—not commissioned clinical
dosimetry—and the generated plans must never be used for patient care.
