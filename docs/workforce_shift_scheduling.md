# Workforce Shift Scheduling

The `workforce_shift_scheduling` category models generic contact-center, retail,
and service-workforce planning LPs. It complements domain-specific nurse and
airline crew generators with a reusable shift-pattern covering formulation.

## Variants

- `covering`: labor-pool by shift-pattern staffing variables cover time-bucket
  and skill demands. Undercoverage variables carry high penalties, pattern data
  embeds breaks, and labor-pool upper bounds create realistic capacity limits.

The formulation creates set-covering-style matrices with many interchangeable
columns and soft service-level constraints.
