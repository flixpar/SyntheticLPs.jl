# Service Network Design

The `service_network_design` category models time-expanded logistics and service
networks, a common structure in parcel, trucking, airline, rail, rental-fleet,
and mobility-repositioning planning.

## Variants

- `time_expanded`: commodity flows move through location-time nodes over holding
  and scheduled service arcs. Arcs share capacities across commodities, demands
  have release and due periods, and unmet-demand variables carry high penalties.

This formulation creates large sparse conservation systems with repeated temporal
structure and capacity bottlenecks.
