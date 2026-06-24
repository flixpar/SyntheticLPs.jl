# Benchmark Pathologies

The `benchmark_pathologies` category contains solver-stress LPs that complement
application-style generators. These variants deliberately exercise numerical
scaling, degeneracy, redundant structure, and infeasibility controls.

## Variants

- `scaling_stress`: bounded sparse LP with coefficients, objective values, and
  variable upper bounds spanning many orders of magnitude.
- `degenerate_network`: layered network-flow LP with many parallel nearly tied
  arcs, creating degenerate bases and large optimal faces.

These generators are useful when synthetic datasets need algorithmic edge cases,
not just domain coverage.
