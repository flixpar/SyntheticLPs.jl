# Job Shop Scheduling Generator Research & Plan

## Research on Realistic Job Shop Scheduling Data
- **Industries & scale**: Job shops appear in aerospace machining, semiconductor fabs, custom metal works, and pharmaceutical batch rooms. Case studies (e.g., Lawrence benchmark suite, Taillard instances) show machine counts ranging from 5 to 20 with 2–5× as many jobs as machines.
- **Operation structure**: Each job is a sequence of operations with machine requirements following process routes. Real shops often have 3–10 steps per job, rarely using the same machine twice consecutively. Machine assignments are usually sampled from dominant flows (e.g., milling→grinding→polishing) but still include occasional repeats.
- **Processing times**: Empirical datasets report skewed distributions (lognormal/gamma) with coefficients of variation between 0.3 and 1.0. Durations often depend on machine class; heavy machining has longer times than finishing.
- **Release times and due dates**: Release dates come from batching policies and follow uniform/exponential spreads within a horizon. Feasible due dates are generally 15–50% larger than the pure processing-time sum, while infeasible test sets intentionally compress this slack below 80%.
- **Weights/costs**: Dispatching uses priority weights tied to revenue or lateness penalties, typically between 1 and 5, sometimes higher for rush orders.
- **Planning horizon**: Practical horizons cover the cumulative processing time plus queue buffers; big-M constants should exceed this to avoid artificial tightness while keeping numerics stable.

## Detailed Plan for the Generator
1. **Dimension selection heuristic**
   - Categorize targets into *small*, *medium*, *large* scenarios that set ranges for machines (5–8, 6–12, 10–20) and ops-per-job distributions.
   - Approximate the required number of operations using \(N_{ops} \approx \sqrt{2 M \times target}\) because binary machine-order variables dominate. Iterate to keep the realized variable count within ±10% of `target_variables` by scaling the job count with a square-root correction.
2. **Job routing construction**
   - Sample per-job operation counts from truncated normals and ensure at least two steps. Create machine sequences by permuting machine groups; if a job needs more steps than machines, allow rare repeats but avoid identical consecutive machines.
3. **Processing times & calendars**
   - Draw base processing times from Gamma distributions with machine-dependent scale. Add modest heteroscedastic noise so heavy machines (lower indices) are slower than finishing ones.
4. **Temporal data**
   - Release times sampled from a staggered uniform window relative to aggregate processing time. Set due dates using feasibility status: feasible cases get 25–70% slack, while infeasible cases compress slack to 40–70% of the theoretical minimum completion time to guarantee violations.
5. **Model creation**
   - Variables: start time per operation, completion per job, global makespan, and binary precedence for each machine-operation pair.
   - Constraints: job precedences, machine disjunctive constraints with big-M horizon, release/start bounds, completion definitions, and due-date caps. Objective minimizes weighted completion plus a small makespan weight to emulate throughput goals.
6. **Feasibility control**
   - Feasible status uses loose due dates ensuring at least 20% buffer beyond sequential processing. Infeasible status shrinks due dates below the minimum feasible bound, guaranteeing inconsistency. Unknown status randomly picks either mode.
7. **Exposure**
   - Register `:job_shop_scheduling` in `SyntheticLPs`, mention in README, and document this research/plan file for future contributors.
