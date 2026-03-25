# Nurse Scheduling Generator Research and Plan

## Real-world practices surveyed
- **Shift structures**: Acute-care hospitals typically staff a mix of 8- and 12-hour shifts with distinct day, evening, and night blocks. Weekend coverage remains high but fluctuates by service line, and night shifts pay a premium plus require recovery time before the next day shift.
- **Skill mix**: Units rarely run homogenous teams—charge nurses, ICU-certified staff, and float pools must be distributed. Regulatory guidelines often enforce minimum counts of registered nurses per shift plus additional specialty coverage (e.g., at least one ICU-capable nurse and one pediatric specialist when census includes children).
- **Labor contracts**: Nurses face caps on consecutive days (3–5 for 12-hour shifts), limits on monthly night assignments, and fairness rules for weekends. Many agreements enforce a minimum number of shifts per schedule block as well.
- **Availability patterns**: Full-time nurses usually work 0.8–1.0 FTE with predictable weekdays, whereas per-diem/part-time staff prefer evenings or weekends. Night-qualified nurses are fewer, and float pools provide coverage variability. Last-minute sick calls create stochastic availability yet hospitals maintain minimum guaranteed coverage per shift.
- **Cost signals**: Premiums accrue for nights (+20–30%), weekends (+10–20%), and overtime (when exceeding contractual maxima). Hospitals often use blended hourly rates but track these premiums shift-by-shift for budgeting and fairness dashboards.

## Generator goals distilled from research
1. **Coverage realism**: Model three-to-four shift blocks per day, weekly patterns, and noisy surges (flu season, elective surgery peaks).
2. **Workforce heterogeneity**: Explicit nurse types (core staff, float pool, specialists) with different availability, skills, and cost structures.
3. **Contract constraints**: Enforce per-nurse min/max shifts, weekend bounds, max consecutive days, night limits, and mandated rest after night duty.
4. **Skill guarantees**: Capture ICU/ER/pediatric skill requirements per shift using a requirement tensor coupled with a skill matrix.
5. **Feasibility control**: Produce feasible instances by deriving requirements from an internally generated roster, and infeasible ones by introducing deliberate skill shortages or impossible coverage spikes.
6. **Size targeting**: Choose (nurses × days × shifts) within ±10% of the requested variable count through a combinatorial search over planning horizons and staff levels.

## Detailed implementation plan
1. **Dimension selection**
   - Iterate over candidate planning horizons (3–56 days) and shift counts (1–4) and choose nurse counts (6–600) that give a variable count within ±10% of the target.
   - Classify the resulting instance as small/medium/large to control distributions for costs, skills, and demand volatility.

2. **Parameter sampling**
   - Build shift labels (day/evening/night) to attach premiums and rest logic.
   - Draw per-nurse contract parameters: base wage, min/max shifts, max consecutive days, night qualifications, rest requirements, weekend tolerance.
   - Sample skill matrix with universal RN skill plus ICU/ER/Peds columns whose coverage depends on scenario size.

3. **Availability generation**
   - Create a 3-D availability tensor using Beta-distributed base probabilities modulated by shift type, weekend flag, and whether the nurse is float/part-time.
   - Ensure each shift has at least two available nurses by flipping random entries if necessary.

4. **Baseline demand and roster construction**
   - Generate base demand per shift using day-of-week, shift-type, and seasonal sine-wave factors.
   - Greedily assign nurses to meet (and slightly exceed) these demands while respecting availability, max shifts, consecutive-day caps, night cooldowns, and single-shift-per-day rules.
   - Track resulting assignment counts per nurse, per shift, weekend usage, and night totals.

5. **Finalize constraints**
   - Set coverage requirements to the lesser of the noisy demand target and actual filled headcount so feasibility is guaranteed.
   - Derive skill requirements by scaling demand fractions but capping by the number of assigned skilled nurses.
   - Set min/max-shift bounds, weekend bounds, and night limits to values satisfied by the constructed roster.

6. **Infeasibility injection**
   - When `feasibility_status == infeasible`, amplify the requirement on a randomly chosen shift/skill so it exceeds the total number of qualified nurses, or boost raw demand beyond aggregate capacity, making the model provably infeasible.
   - For `unknown`, randomly choose feasible or infeasible generation with equal probability.

7. **Model construction**
   - Decision variables `x[nurse, day, shift]` ∈ [0,1].
   - Objective: minimize staffing cost tensor dotted with `x`.
   - Constraints: coverage per shift, skill coverage, availability upper bounds, single shift per day, min/max total shifts, weekend bounds, night limits, consecutive-day windows, and rest after night shift linking night assignments to next-day early shifts.

8. **Integration**
   - Register `:nurse_scheduling` in `SyntheticLPs.jl`, update README, and ensure constructor exports data for reproducibility.
