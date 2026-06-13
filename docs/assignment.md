# Assignment

The assignment generator creates worker-task matching models with binary assignment variables, compatibility restrictions, and randomized cost structure.

## Overview

This generator represents assigning workers to tasks at minimum cost. Each task must receive exactly one worker, each worker can perform at most one task, and some worker-task pairs may be forbidden by a compatibility matrix. Costs can reflect specialization and worker/task group affinity.

## Generator Data and Sizing

`target_variables` maps to the worker-task matrix size:

```text
target_variables ~= n_workers * n_tasks
```

The constructor seeds Julia's global RNG with `Random.seed!(seed)`, so all dimension choices, compatibility choices, and costs are reproducible for the same inputs.

Size-dependent parameters:

| Target variables | Balanced probability | Base cost range | Specialization probability | Cost variation weights |
| --- | --- | --- | --- | --- |
| `<= 250` | `0.8` | `5` to `30` | `0.2` | low `0.6`, medium `0.3`, high `0.1` |
| `<= 1000` | `0.6` | `10` to `100` | `0.4` | low `0.3`, medium `0.5`, high `0.2` |
| `> 1000` | `0.4` | `50` to `500` | `0.6` | low `0.2`, medium `0.3`, high `0.5` |

Balanced instances use a square matrix with `n_workers = n_tasks` near `sqrt(target_variables)`, both at least 5. Unbalanced instances use a random ratio in `[0.5, 2.0)` and then make up to three adjustment passes to move the product closer to the target.

After feasibility adjustments, the generator assigns workers and tasks to random skill groups. The number of groups is sampled from `2:gmax`, where `gmax = min(6, max(2, round(sqrt(min(n_workers, n_tasks)))))`. Compatibility density is based on final matrix size:

- `0.85` for `<= 250` possible assignments
- `0.70` for `<= 1000`
- `0.50` for larger matrices

Within-group compatibility probability is `min(0.98, base_density)`. Cross-group compatibility probability is `max(0.02, 0.3 * base_density)`.

The `allowed` matrix is only randomized when `feasibility_status` is `feasible` or `infeasible`. For `unknown`, it remains all `true`.

Costs are integer values. The upper end of the base cost range is multiplied by a random factor in `[0.8, 1.2)`, with a minimum spread of 5 above the low cost. Specialized workers have a few low-cost tasks and higher costs elsewhere. Non-specialized costs use low, medium, or high variation, with lower expected costs for matching worker/task groups.

The struct stores:

- `n_workers::Int`
- `n_tasks::Int`
- `costs::Matrix{Int}`
- `allowed::Matrix{Bool}`

## LP Formulation

Sets:

- `W = {1, ..., n_workers}` for workers
- `T = {1, ..., n_tasks}` for tasks

Decision variable:

```text
x[i,j] in {0,1} = 1 if worker i is assigned to task j
```

Objective:

```text
minimize sum_{i in W, j in T} costs[i,j] * x[i,j]
```

Worker capacity:

```text
sum_{j in T} x[i,j] <= 1    for each worker i
```

Task coverage:

```text
sum_{i in W} x[i,j] = 1     for each task j
```

Compatibility restrictions:

```text
x[i,j] = 0                  for each forbidden pair where allowed[i,j] == false
```

Bounds are binary bounds from the JuMP `Bin` declaration.

## Feasibility Controls

The constructor maps statuses to internal symbols: `feasible` becomes `:feasible`, `infeasible` becomes `:infeasible`, and `unknown` becomes `:all`.

- `feasible`: if there are fewer workers than tasks, it adds enough workers to cover all tasks and may add 1 to 3 extra workers with probability 0.3. If randomized compatibility is active, it then constructs a task order and ensures each task has at least one allowed, previously unused worker, editing `allowed` when necessary. This creates a matching witness at the compatibility level.
- `infeasible`: forces an unbalanced shortage when needed by setting `n_tasks` above `n_workers` by a gap of about 5% to 25% of workers. Since every task must be assigned and every worker can take at most one task, this capacity shortfall is infeasible. With probability 0.4, it instead or additionally creates a Hall violation by selecting a subset of tasks and allowing them to be served only by a smaller subset of workers.
- `unknown`: does not adjust dimensions for feasibility and does not randomize compatibility; `allowed` remains all true.

For `infeasible`, the worker/task count shortfall is a direct infeasibility certificate when applied. Hall violations are also designed to be infeasible because more tasks are restricted to fewer workers.

## Model Characteristics

The model has `n_workers * n_tasks` binary variables. It has `n_workers` worker-capacity constraints, `n_tasks` task-coverage constraints, and one equality-fixing constraint for each forbidden assignment.

This is a mixed-integer model as built by `build_model`, not a pure continuous LP. If the package-level generation path is called with integer relaxation enabled, the binary variables may be relaxed outside this file; the generator itself declares them as `Bin`.

The dense variable matrix is structurally sparse in constraints: each assignment variable appears in one worker row and one task row, plus one compatibility-fixing row if forbidden.

## Practical Notes

These instances are useful for testing assignment structure, binary relaxation behavior, and compatibility-driven infeasibility. The `unknown` mode is unusual because it keeps all assignments allowed, so infeasibility in that mode is mainly from a natural worker/task count imbalance rather than compatibility. The generator stores costs for forbidden pairs too, but the model fixes those variables to zero.
