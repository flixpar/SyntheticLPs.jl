# Crop Planning

`crop_planning/standard` generates continuous farm-planning LPs that allocate
hectares among crop and management-system options. The objective maximizes net
margin while land, irrigation water, seasonal labor, saleable production,
minimum-acreage commitments, and optional crop-group diversity are enforced.

## Data model

The requested variable count maps directly to crop options:

```text
n_crops = max(2, target_variables)
```

Every option combines one of 25 recognizable crops with one of four agronomic
systems: `rainfed`, `irrigated`, `low_input`, or `intensive`. Catalog blocks are
independently shuffled, and blocks after the first receive cultivar identifiers.
This keeps large instances interpretable instead of filling them with anonymous
`Crop_i` columns. Management systems change yield, cost, irrigation, and labor
together: intensive and irrigated options raise potential yield and inputs,
while rainfed and low-input options trade output for lower resource use.

The five crop groups and their base agronomic ranges are:

| Group | Example crops | Yield (t/ha) | Price ($/t) | Water (mm/season) | Labor (h/ha) |
| --- | --- | ---: | ---: | ---: | ---: |
| cereal | wheat, maize, rice | 3-10 | 150-350 | 400-650 normally | 30-80 |
| vegetable | tomatoes, peppers | 15-40 | 400-900 | 350-600 | 120-250 |
| legume | soybeans, lentils | 2-4 | 300-600 | 300-500 | 25-60 |
| industrial | cotton, sugarcane | 3-8 | 500-1200 | 500-800 normally | 80-180 |
| oilseed | sunflower, canola | 1.5-4 | 400-700 | 350-550 | 35-75 |

Rice and sugarcane use separate, higher water ranges. The table describes base
draws; the selected management system subsequently applies correlated
multipliers. Net margin is computed as

```math
m_i = price_i \; yield_i - production\_cost_i.
```

Farm scale controls total land and the frequency of diversity requirements:

| Requested size | Total land | Diversity-generation probability |
| --- | ---: | ---: |
| up to 250 | 50-500 ha | 0.50-0.80 |
| 251-1000 | 500-5,000 ha | 0.60-0.90 |
| above 1000 | 5,000-50,000 ha | 0.70-0.95 |

Market limits are stored in **tonnes**, not hectares. The constructor first
samples an acreage-equivalent market scale, then multiplies it by the option's
yield. This makes the formulation's market row dimensionally meaningful and
ensures that yield participates in both revenue and market consumption.

With probability 0.85, a subset of options receives minimum-acreage
commitments. Cereals and legumes are preferred, modeling staple or rotation
commitments. These minima are capped by market limits and jointly scaled when
necessary so they fit available land.

All randomness uses a constructor-local `MersenneTwister`. Constructing an
instance is deterministic for `(target_variables, feasibility_status, seed)`
and does not reset or advance Julia's global RNG.

## Formulation

For crop options `i in I`, the variable is

```text
x_i >= 0    hectares planted under option i
```

The LP maximizes total net margin:

```math
\max \sum_{i \in I} m_i x_i.
```

Resource capacities are

```math
\sum_i x_i &\le L, \\
\sum_i water_i x_i &\le W, \\
\sum_i labor_i x_i &\le H.
```

Saleable production and mandatory acreage are option-specific:

```math
yield_i x_i &\le demand_i && \forall i, \\
x_i &\ge minimum_i && \forall i \text{ with a positive commitment}.
```

An optional crop-group requirement `g` is

```math
\sum_{i \in I_g} x_i \ge diversity_g.
```

The implementation names the land, water, labor, market, minimum-area, and
diversity row families, which makes exported models and formulation tests easier
to inspect.

## Feasibility contracts and metadata

For a `feasible` request, the constructor stores `feasible_witness`, a complete
acreage vector. It begins at the mandatory minima, distributes optional land by
positive margin without exceeding market limits, and sizes water and labor with
10-30% slack. Every diversity floor is derived from the acreage already present
in its crop group and is at most 95% of the witness value. Consequently the
witness satisfies every implemented row directly; feasibility does not rely on
a solver retry or a tolerance that is looser than the model.

For an `infeasible` request, the constructor ensures that mandatory acreage has
positive resource use, then makes either water or labor capacity 5-25% smaller
than the amount forced by those lower bounds. The stored
`CropResourceCertificate` contains:

- `resource`: `:water` or `:labor`;
- `forced_usage`: resource use implied by all mandatory crop minima;
- `capacity`: the strictly smaller model capacity.

This is a solver-independent proof because leaving land fallow cannot reduce
resource use below the mandatory minima. No diversity rows are needed for the
contradiction.

For an `unknown` request, capacities and diversity floors are sampled without a
witness or certificate. Such instances intentionally retain natural uncertainty
and may be feasible or infeasible.

Exactly one status-specific metadata field is populated for requested feasible
or infeasible instances; both are `nothing` for unknown instances.

## Model characteristics

- Model class: continuous LP.
- Variables: exactly `max(2, target_variables)`.
- Dense rows: land, water, and labor.
- Sparse/singleton rows: crop-group diversity, market limits, and mandatory
  acreage.
- Fallow land is allowed because the land constraint is an upper bound. The
  infeasibility certificate therefore uses only true mandatory lower bounds and
  never assumes that all land must be planted.
