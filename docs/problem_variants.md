# Problem Generator Variants

This document provides an extensive list of problem variants for each of the 24 problem generators in SyntheticLPs.jl. Variants are organized into categories:

- **Constraint Variants**: Different or additional constraint types
- **Objective Variants**: Alternative objective functions
- **Data Distribution Variants**: Different ways to generate parameter values
- **Structural Variants**: Changes to problem topology or structure
- **Scenario/Application Variants**: Different real-world application contexts

---

## 1. Transportation Problem

### Current Implementation
- Minimize total transportation cost
- Supply constraints (≤) at sources
- Demand constraints (≥) at destinations

### Constraint Variants
1. **Balanced Transportation**: Force total supply = total demand (equality constraints)
2. **Transshipment Nodes**: Add intermediate nodes that can receive and forward goods
3. **Route Capacity Limits**: Maximum flow on individual source-destination pairs
4. **Minimum Shipment Requirements**: Minimum amounts must be shipped on certain routes
5. **Exclusive Routes**: Some routes are mutually exclusive (if one is used, another cannot be)
6. **Service Level Constraints**: Minimum percentage of demand that must be satisfied
7. **Multi-Period Transportation**: Inventory carryover between periods with time-varying demands
8. **Hub Constraints**: Certain nodes must serve as mandatory transshipment hubs
9. **Vehicle Capacity Constraints**: Shipments must fit within discrete vehicle capacities
10. **Environmental Emission Limits**: Total CO2 emissions from transportation must be below threshold

### Objective Variants
1. **Minimize Total Distance**: Minimize sum of distances (instead of cost)
2. **Minimize Maximum Flow Time**: Minimize the longest route used (minimax)
3. **Minimize Number of Routes Used**: Encourage consolidation
4. **Bi-Objective Cost-Time**: Weighted combination of cost and transit time
5. **Maximize Throughput**: Maximize total goods moved within budget constraint
6. **Risk-Adjusted Cost**: Include risk factors for unreliable routes
7. **Minimize Variance in Delivery**: Minimize deviation from target delivery times

### Data Distribution Variants
1. **Distance-Based Costs**: Euclidean distance between randomly placed nodes
2. **Zone-Based Pricing**: Costs depend on geographic zones
3. **Volume Discounts**: Non-linear cost structure (lower per-unit cost for larger shipments)
4. **Seasonal Demand Patterns**: Demands follow seasonal cycles
5. **Correlated Supply-Demand**: High-demand destinations near high-supply sources
6. **Sparse Networks**: Only subset of source-destination pairs are feasible
7. **Heavy-Tailed Demands**: Log-normal or Pareto distributed demands
8. **Bimodal Supplies**: Mix of small and large suppliers

### Structural Variants
1. **Hub-and-Spoke Network**: All flows go through central hub(s)
2. **Multi-Commodity**: Different product types with separate demands
3. **Asymmetric Network**: Different costs for forward vs. return shipments
4. **Regional Clustering**: Sources and destinations clustered geographically
5. **Star Network**: Single source to many destinations (distribution) or many-to-one (collection)

### Scenario Variants
1. **Disaster Relief Logistics**: Urgent supplies to affected areas with capacity constraints
2. **Agricultural Distribution**: Farm-to-market with perishability time windows
3. **Cross-Border Trade**: Different cost structures for international routes
4. **Last-Mile Delivery**: Many small deliveries from regional warehouses
5. **Reverse Logistics**: Product returns and recycling flows

---

## 2. Diet Problem

### Current Implementation
- Minimize food cost
- Nutrient minimum requirements
- Food supply limits
- Cost budget constraint
- Min/max food consumption constraints

### Constraint Variants
1. **Nutrient Upper Bounds**: Maximum limits on nutrients (e.g., sodium, saturated fat)
2. **Food Group Requirements**: Minimum servings from each food group
3. **Allergen Exclusions**: Certain foods completely excluded
4. **Meal Structure**: Constraints on breakfast/lunch/dinner composition
5. **Variety Constraints**: Minimum number of different foods used
6. **Preparation Time Limits**: Total cooking/preparation time constraint
7. **Glycemic Index Limits**: Average GI of diet below threshold
8. **Calorie Range**: Both minimum and maximum calorie constraints
9. **Macronutrient Ratios**: Specific ratios of protein/carbs/fat
10. **Organic/Local Requirements**: Minimum percentage from organic/local sources

### Objective Variants
1. **Minimize Calories**: For weight loss diet
2. **Maximize Protein**: For athletic performance
3. **Maximize Nutrient Density**: Nutrients per calorie
4. **Minimize Environmental Impact**: Carbon footprint of diet
5. **Minimize Preparation Time**: Quick meal planning
6. **Maximize Palatability Score**: Preference-weighted selection
7. **Minimize Sodium**: For heart health
8. **Bi-Objective Cost-Nutrition**: Pareto frontier of cost vs. nutrition quality

### Data Distribution Variants
1. **USDA Database Values**: Realistic nutrient content from food databases
2. **Regional Food Availability**: Different foods available by region
3. **Seasonal Pricing**: Prices vary by season
4. **Whole Foods vs. Processed**: Different nutrient profiles
5. **Restaurant vs. Home Cooking**: Different cost and nutrient structures
6. **Cultural Diet Patterns**: Mediterranean, Asian, Latin American food options

### Scenario Variants
1. **Hospital Patient Diet**: Medical restrictions (renal diet, diabetic diet)
2. **School Cafeteria Planning**: USDA requirements for school meals
3. **Military Ration Design**: High-calorie, long shelf-life requirements
4. **Athlete Training Diet**: High protein, specific macros by training phase
5. **Elderly Nutrition**: Easy-to-prepare, nutrient-dense for reduced appetite
6. **Emergency Rations**: Minimum cost survival diet
7. **Vegetarian/Vegan Diets**: Plant-based protein requirements

---

## 3. Knapsack Problem

### Current Implementation
- Fractional knapsack maximizing value
- Single weight capacity constraint
- 0 ≤ x_i ≤ 1 bounds

### Constraint Variants
1. **Multi-Dimensional Knapsack**: Multiple resource constraints (weight, volume, etc.)
2. **Bounded Knapsack**: Upper bounds on quantity of each item
3. **Multiple Knapsacks**: Assign items to multiple knapsacks
4. **Nested Knapsack**: Items that must be packed together
5. **Conflict Constraints**: Certain items cannot be in same knapsack
6. **Dependency Constraints**: Item i requires item j to be included
7. **Group Constraints**: At least/at most k items from each group
8. **Minimum Fill**: Knapsack must be at least X% full
9. **Balance Constraints**: Weight distribution requirements

### Objective Variants
1. **Minimize Weight for Target Value**: Reverse objective
2. **Maximize Items Count**: Number of items subject to capacity
3. **Maximize Diversity**: Sum of pairwise item differences
4. **Minimize Wasted Capacity**: Minimize unused space
5. **Risk-Adjusted Value**: Value minus variance
6. **Lexicographic Objectives**: Prioritized multi-objective

### Data Distribution Variants
1. **Correlated Value-Weight**: Higher weight items have higher value
2. **Inversely Correlated**: Light items are more valuable (gems scenario)
3. **Clustered Items**: Groups with similar value-to-weight ratios
4. **Heavy-Tailed Values**: Few very valuable items
5. **Uniform Random**: Standard random generation
6. **Subset Sum Structure**: Values equal to weights

### Scenario Variants
1. **Cargo Loading**: Loading trucks with package constraints
2. **Investment Selection**: Budget allocation to projects
3. **Memory Allocation**: Programs fitting in limited RAM
4. **Cutting Budget**: Selecting expense cuts
5. **Advertisement Selection**: Choosing ads within time slot budget

---

## 4. Portfolio Optimization

### Current Implementation
- Maximize returns
- Risk budget constraint
- Full investment constraint (all money invested)
- Risk-free asset available

### Constraint Variants
1. **Sector Limits**: Maximum allocation to any sector
2. **Asset Count Limits**: Maximum number of assets held
3. **Minimum Position Size**: If invested, must be at least X%
4. **Short Selling Allowed**: Negative positions permitted
5. **Liquidity Constraints**: Maximum allocation to illiquid assets
6. **ESG Requirements**: Minimum ESG score for portfolio
7. **Turnover Limits**: Maximum change from previous portfolio
8. **Currency Exposure Limits**: Hedging requirements
9. **Country/Region Limits**: Geographic diversification
10. **Beta Constraints**: Portfolio beta within range

### Objective Variants
1. **Mean-Variance (Markowitz)**: Maximize return - λ × variance
2. **Minimize Risk for Target Return**: Classic efficient frontier
3. **Maximize Sharpe Ratio**: Risk-adjusted return
4. **Minimize Conditional VaR (CVaR)**: Tail risk measure
5. **Maximize Information Ratio**: Active return per tracking error
6. **Minimize Tracking Error**: Match benchmark index
7. **Maximize Dividend Yield**: Income-focused investing
8. **Risk Parity Objective**: Equal risk contribution from assets

### Data Distribution Variants
1. **Historical Returns**: Based on real market data patterns
2. **Factor Model Returns**: Generated from factor exposures
3. **Regime-Dependent**: Different distributions for bull/bear markets
4. **Correlated Assets**: Realistic correlation structures
5. **Fat-Tailed Returns**: Student-t or stable distributions
6. **Time-Varying Volatility**: GARCH-style variance patterns
7. **Crisis Scenarios**: Stressed correlation and volatility

### Scenario Variants
1. **Pension Fund**: Long-term, liability-matching
2. **Hedge Fund**: Absolute return, leverage allowed
3. **Retail Investor**: Simple constraints, few assets
4. **Endowment**: Very long horizon, spending policy
5. **Target Date Fund**: Glide path to conservative allocation
6. **Factor Investing**: Smart beta tilts

---

## 5. Network Flow Problem

### Current Implementation
- Max flow or min cost flow
- Arc capacity constraints
- Flow conservation at intermediate nodes
- Optional target flow requirement

### Constraint Variants
1. **Node Capacities**: Maximum flow through nodes (not just arcs)
2. **Gain/Loss Arcs**: Flow multiplied by gain factor (leaky pipes, evaporation)
3. **Lower Bounds on Arcs**: Minimum flow requirements
4. **Bundled Arcs**: Sets of arcs that must have equal flow
5. **Time-Expanded Network**: Dynamic flow over time periods
6. **Reliable Flow**: Flow paths must be arc-disjoint for redundancy
7. **Priority Flows**: Certain commodities have priority
8. **Storage at Nodes**: Buffer inventory between periods

### Objective Variants
1. **Minimum Cost Maximum Flow**: Find max flow, then minimize cost
2. **Minimize Maximum Arc Utilization**: Load balancing
3. **Maximize Shortest Path Flow**: Maximize flow using k-shortest paths only
4. **Minimize Total Delay**: Congestion-based delay function
5. **Quickest Flow**: Minimize time to send fixed amount
6. **Earliest Arrival Flow**: Maximize flow at each time point

### Data Distribution Variants
1. **Planar Network**: Geographically embedded with crossing limits
2. **Grid Network**: Regular grid topology
3. **Random Geometric Graph**: Edges based on distance
4. **Scale-Free Network**: Power-law degree distribution
5. **Layered Network**: Bipartite-like structure
6. **Hierarchical Network**: Tree with cross-links

### Structural Variants
1. **Multi-Source Multi-Sink**: Multiple origins and destinations
2. **Symmetric Network**: Same capacity in both directions
3. **Sparse Network**: Low edge density
4. **Dense Network**: Nearly complete graph
5. **Series-Parallel Network**: Special tractable structure

### Scenario Variants
1. **Telecommunication Routing**: Data packet routing
2. **Pipeline Network**: Oil/gas/water distribution
3. **Traffic Network**: Vehicle routing
4. **Evacuation Planning**: Emergency egress
5. **Power Grid**: Electricity transmission

---

## 6. Production Planning

### Current Implementation
- Maximize profit
- Resource constraints

### Constraint Variants
1. **Setup Constraints**: Fixed cost/time to start production
2. **Batch Size Requirements**: Produce in multiples of batch size
3. **Sequence-Dependent Setup**: Changeover costs between products
4. **Work-in-Process Limits**: Maximum inventory in pipeline
5. **Machine Assignment**: Products can only use certain machines
6. **Overtime Limits**: Maximum overtime hours per period
7. **Demand Satisfaction**: Meet minimum demand levels
8. **Quality Constraints**: Minimum yield requirements
9. **Labor Skill Requirements**: Products need specific skills
10. **Maintenance Windows**: Periodic downtime for equipment

### Objective Variants
1. **Minimize Makespan**: Complete all jobs as early as possible
2. **Minimize Total Cost**: Production + inventory + shortage costs
3. **Minimize Tardiness**: Weighted late completion penalties
4. **Maximize Throughput**: Units produced per time
5. **Minimize Work-in-Process**: Lean manufacturing
6. **Level Production**: Minimize production variability

### Data Distribution Variants
1. **Seasonal Demand**: Cyclical demand patterns
2. **Learning Curve Effects**: Productivity improves with experience
3. **Correlated Resource Usage**: Products with similar profiles
4. **Stochastic Processing Times**: Variable completion times
5. **Machine-Dependent Rates**: Different productivity by machine

### Scenario Variants
1. **Job Shop**: Flexible routing through machines
2. **Flow Shop**: Fixed sequence of operations
3. **Batch Process Industry**: Chemical, pharmaceutical
4. **Discrete Manufacturing**: Automotive, electronics
5. **Make-to-Order vs. Make-to-Stock**: Different inventory policies
6. **Multi-Site Production**: Coordinated manufacturing

---

## 7. Assignment Problem

### Current Implementation
- Minimize assignment cost
- Each worker to at most one task
- Each task to exactly one worker
- Compatibility constraints

### Constraint Variants
1. **Multi-Assignment**: Workers can do multiple tasks
2. **Team Formation**: Assign teams to projects
3. **Skill Matching**: Tasks require specific skill levels
4. **Workload Balancing**: Each worker gets similar workload
5. **Preference Constraints**: Workers can veto certain tasks
6. **Shift Coverage**: Tasks across multiple shifts
7. **Break Requirements**: Mandatory breaks between assignments
8. **Sequential Tasks**: Some tasks must follow others
9. **Geographic Constraints**: Travel time between task locations

### Objective Variants
1. **Maximize Productivity**: Total output of assignments
2. **Minimize Maximum Workload**: Fairness objective
3. **Maximize Preference Satisfaction**: Worker happiness
4. **Minimize Training Cost**: Assign to already-skilled workers
5. **Minimize Travel Time**: Geographic assignment
6. **Lexicographic Assignment**: Priority ordering of objectives

### Data Distribution Variants
1. **Skill Clustering**: Workers grouped by expertise
2. **Task Complexity Variation**: Mix of easy and hard tasks
3. **Geographic Clustering**: Workers and tasks in regions
4. **Experience-Based Costs**: Senior workers more expensive
5. **Sparse Compatibility**: Many incompatible pairs

### Scenario Variants
1. **Nurse Scheduling**: Hospital ward coverage
2. **Classroom Assignment**: Teachers to classes
3. **Project Staffing**: Consultants to client projects
4. **Ride Matching**: Drivers to passengers
5. **Referee Assignment**: Sports officiating
6. **Organ Donation Matching**: Medical compatibility

---

## 8. Blending Problem

### Current Implementation
- Minimize cost
- Minimum blend amount (equality)
- Supply limits on ingredients
- Cost budget
- Quality bounds (weighted averages)
- Min/max usage constraints

### Constraint Variants
1. **Non-Linear Quality Interactions**: Synergistic/antagonistic ingredient effects
2. **Stability Constraints**: pH, temperature, shelf-life requirements
3. **Safety Limits**: Maximum toxin/contaminant levels
4. **Appearance Requirements**: Color, texture specifications
5. **Processing Constraints**: Mixability, viscosity limits
6. **Regulatory Compliance**: Legal ingredient limits
7. **Batch Sequencing**: Constraints between consecutive batches
8. **Equipment Capacity**: Mixer size limits
9. **Recipe Confidentiality**: Certain combinations restricted

### Objective Variants
1. **Maximize Quality Score**: Highest quality blend
2. **Minimize Environmental Impact**: Sustainable sourcing
3. **Maximize Shelf Life**: Longest stability
4. **Minimize Allergen Content**: Hypoallergenic products
5. **Target Profile Matching**: Match reference blend profile
6. **Robust Quality**: Minimize quality variance

### Data Distribution Variants
1. **Natural Ingredient Variation**: Batch-to-batch quality differences
2. **Seasonal Availability**: Different ingredients by season
3. **Supplier Quality Tiers**: Premium vs. standard ingredients
4. **Correlated Quality Attributes**: Related nutrient profiles
5. **Regional Ingredient Sources**: Geographic origin effects

### Scenario Variants
1. **Gasoline Blending**: Octane, RVP, sulfur requirements
2. **Wine Blending**: Varietal percentages, taste profile
3. **Concrete Mix Design**: Strength, workability specs
4. **Pharmaceutical Formulation**: Bioavailability, stability
5. **Cosmetics Manufacturing**: Skin compatibility, appearance
6. **Paint Formulation**: Color matching, durability

---

## 9. Facility Location

### Current Implementation
- Minimize fixed + shipping costs
- Customer demand satisfaction
- Facility capacity constraints
- Budget constraint on fixed costs
- Binary facility opening decisions

### Constraint Variants
1. **Coverage Constraints**: All customers within maximum distance
2. **Backup Facility Requirements**: Each customer served by ≥2 facilities
3. **Capacity Tiers**: Choose from discrete capacity options
4. **Zoning Restrictions**: Facilities forbidden in certain areas
5. **Competitive Location**: Account for competitor facilities
6. **Service Level Requirements**: Maximum response/delivery time
7. **Expansion Options**: Existing facilities can be expanded
8. **Phased Opening**: Facilities opened over multiple periods
9. **Environmental Permits**: Limited locations available

### Objective Variants
1. **Maximize Coverage**: Maximum customers within service distance
2. **Minimize Maximum Distance**: Equity objective
3. **Minimize Number of Facilities**: Fixed cost minimization
4. **Maximize Market Share**: Competitive capture model
5. **Minimize Response Time**: Emergency services
6. **Bi-Objective Cost-Coverage**: Trade-off analysis
7. **Minimize Carbon Footprint**: Environmental objective

### Data Distribution Variants
1. **Urban Concentration**: Customers clustered in cities
2. **Uniform Distribution**: Rural service areas
3. **Highway Network Costs**: Travel along road network
4. **Population Density Weighting**: Demand proportional to population
5. **Real Estate Price Variation**: Location-dependent fixed costs
6. **Demographic Segmentation**: Different customer types by area

### Structural Variants
1. **Hub-and-Spoke**: Hierarchical facility network
2. **P-Median**: Locate exactly p facilities
3. **P-Center**: Minimize maximum distance
4. **Capacitated vs. Uncapacitated**: With/without capacity limits
5. **Multi-Level Hierarchy**: Distribution centers + retail

### Scenario Variants
1. **Retail Store Location**: Market area maximization
2. **Warehouse Network Design**: Distribution efficiency
3. **Emergency Services**: Fire stations, ambulances
4. **Utility Infrastructure**: Power plants, substations
5. **Healthcare Access**: Hospital and clinic placement
6. **Banking Network**: ATM and branch optimization

---

## 10. Airline Crew Pairing

### Current Implementation
- Minimize pairing cost (set covering)
- Each flight covered exactly once
- Hub-and-spoke network structure

### Constraint Variants
1. **Maximum Duty Time**: Legal flight time limits
2. **Minimum Rest Requirements**: Rest between duties
3. **Maximum Days Away from Base**: Crew return requirements
4. **Crew Qualification**: Aircraft type ratings
5. **Language Requirements**: International flight requirements
6. **Deadheading Limits**: Maximum positioning flights
7. **Night Flying Restrictions**: Circadian considerations
8. **Connection Time Windows**: Minimum/maximum connection times
9. **Hotel Availability**: Overnight stay constraints

### Objective Variants
1. **Minimize Crew Cost + Dead-Heading**: Include positioning costs
2. **Minimize Number of Pairings**: Simplify scheduling
3. **Maximize Utilization**: Crew productivity
4. **Minimize Disruption Risk**: Robust schedules
5. **Minimize Hotel Costs**: Overnight accommodation
6. **Minimize Fatigue Risk**: Safety objective
7. **Maximize Seniority Preference**: Senior crew priority

### Data Distribution Variants
1. **Hub Dominated Network**: Major hub airports
2. **Point-to-Point Network**: Direct routes
3. **Seasonal Schedules**: Holiday flight patterns
4. **Time Zone Patterns**: Long-haul considerations
5. **Crew Base Distribution**: Multiple crew bases
6. **Aircraft Rotation Patterns**: Fleet assignment effects

### Scenario Variants
1. **Mainline Carrier**: Large airline operation
2. **Regional Carrier**: Short-haul flying
3. **Cargo Airline**: Overnight operations
4. **Charter Operations**: Variable schedules
5. **Recovery Scenarios**: Irregular operations

---

## 11. Cutting Stock Problem

### Current Implementation
- Minimize stock pieces used
- Piece demand satisfaction
- Optional stock limit
- Pattern-based formulation

### Constraint Variants
1. **Multiple Stock Sizes**: Different length stock materials
2. **Setup Costs**: Fixed cost per pattern used
3. **Pattern Limits**: Maximum number of different patterns
4. **Trim Loss Limits**: Maximum acceptable waste
5. **Due Date Constraints**: Time-phased demand
6. **Machine Constraints**: Pattern-machine compatibility
7. **Edge Quality Requirements**: Premium edges for certain pieces
8. **Grain Direction**: Wood grain orientation constraints
9. **Inventory Constraints**: Storage limits for pieces

### Objective Variants
1. **Minimize Trim Loss**: Total waste material
2. **Minimize Pattern Count**: Setup reduction
3. **Minimize Cost**: Material + setup + handling costs
4. **Minimize Makespan**: Complete orders earliest
5. **Maximize Material Utilization**: Efficiency percentage
6. **Minimize Overproduction**: Just-in-time production

### Data Distribution Variants
1. **Standard Sizes**: Common piece lengths
2. **Custom Orders**: Unusual piece lengths
3. **Correlated Demands**: Related pieces ordered together
4. **Seasonal Demand**: Construction season patterns
5. **Order Size Variation**: Mix of small and large orders
6. **Rush vs. Regular Orders**: Priority differentiation

### Structural Variants
1. **1D Cutting**: Standard linear cutting
2. **2D Cutting Stock**: Rectangular pieces from sheets
3. **3D Bin Packing**: Boxes in containers
4. **Multi-Period**: Rolling horizon planning
5. **Open vs. Closed Patterns**: End piece reuse

### Scenario Variants
1. **Paper/Film Rolling**: Web cutting
2. **Steel Coil Slitting**: Metal processing
3. **Lumber Cutting**: Sawmill optimization
4. **Glass Cutting**: Window manufacturing
5. **Textile Cutting**: Fabric optimization

---

## 12. Energy Generation Mix

### Current Implementation
- Minimize generation cost
- Demand satisfaction per period
- Capacity limits per source
- Emission constraints
- Renewable fraction requirement

### Constraint Variants
1. **Ramping Constraints**: Maximum rate of change in generation
2. **Minimum Run Time**: Once started, must run minimum periods
3. **Reserve Requirements**: Spinning/non-spinning reserves
4. **Transmission Constraints**: Power flow limits
5. **Storage Constraints**: Battery charge/discharge limits
6. **Startup/Shutdown Costs**: Unit commitment aspects
7. **Fuel Supply Constraints**: Gas/coal availability
8. **Maintenance Windows**: Scheduled outages
9. **Renewable Curtailment**: Maximum renewable can be discarded
10. **Voltage/Frequency Constraints**: Grid stability

### Objective Variants
1. **Minimize Emissions**: Carbon minimization
2. **Minimize Fuel Consumption**: Energy efficiency
3. **Minimize Startup Costs**: Reduce cycling
4. **Minimize Import Costs**: Grid purchase minimization
5. **Maximize Renewable Usage**: Green energy priority
6. **Minimize Load Shedding Cost**: Reliability focus
7. **Bi-Objective Cost-Emissions**: Trade-off analysis

### Data Distribution Variants
1. **Residential Load Profile**: Morning/evening peaks
2. **Commercial Load Profile**: Business hours peaks
3. **Industrial Load Profile**: Flat baseload
4. **Seasonal Variation**: Heating/cooling seasons
5. **Weather-Dependent Renewables**: Solar/wind intermittency
6. **Real-Time Pricing**: Dynamic cost signals
7. **Extreme Weather Events**: Heat waves, cold snaps

### Scenario Variants
1. **Utility-Scale Planning**: Large power system
2. **Microgrid Operation**: Islanded operation capability
3. **Industrial Cogeneration**: Heat and power
4. **Community Energy**: Local renewable projects
5. **Data Center Power**: High reliability requirements
6. **Electrification Scenarios**: EV charging, heat pumps

---

## 13. Feed Blending (Animal Nutrition)

### Current Implementation
- Minimize cost
- Batch size equality
- Nutrient min/max requirements
- Ingredient availability
- Ratio constraints

### Constraint Variants
1. **Palatability Constraints**: Animals must eat the feed
2. **Particle Size Requirements**: Physical mixing properties
3. **Medication Compatibility**: Drug interaction limits
4. **Mycotoxin Limits**: Contamination safety
5. **Protein Quality Scores**: Amino acid balance
6. **Fiber Digestibility**: Animal-specific requirements
7. **Storage Compatibility**: Ingredient segregation
8. **Processing Requirements**: Pelleting, extrusion effects

### Objective Variants
1. **Minimize Cost per Kg Gain**: Economic efficiency
2. **Maximize Feed Conversion Ratio**: Growth optimization
3. **Minimize Environmental Impact**: Nitrogen/phosphorus excretion
4. **Maximize Profit Margin**: Market price consideration
5. **Minimize Ingredient Count**: Simpler formulation
6. **Minimize Variability**: Consistent nutrition

### Data Distribution Variants
1. **Commodity Price Volatility**: Grain market fluctuations
2. **Seasonal Ingredient Quality**: Harvest year effects
3. **Regional Ingredient Availability**: Local vs. imported
4. **By-Product Variation**: DDGS, soybean meal quality
5. **Animal Stage Requirements**: Starter, grower, finisher

### Scenario Variants
1. **Dairy Cattle**: Milk production focus
2. **Beef Cattle**: Weight gain focus
3. **Swine Production**: Different growth stages
4. **Poultry (Layers)**: Egg production
5. **Poultry (Broilers)**: Meat production
6. **Aquaculture**: Fish feed formulation
7. **Pet Food**: Companion animal nutrition

---

## 14. Inventory Problem

### Current Implementation
- Minimize production + holding + backlog costs
- Inventory balance equations
- Production capacity limits
- Optional backlogging

### Constraint Variants
1. **Safety Stock Requirements**: Minimum inventory levels
2. **Shelf Life/Spoilage**: Maximum time in inventory
3. **Warehouse Capacity**: Maximum storage space
4. **Lot Sizing**: Production in fixed batch sizes
5. **Transportation Batch**: Shipment minimum quantities
6. **Multi-Echelon**: Multiple storage locations
7. **Reorder Point Constraints**: Trigger levels
8. **Service Level Constraints**: Fill rate requirements
9. **Cash Flow Constraints**: Payment timing

### Objective Variants
1. **Minimize Stockout Risk**: Robust inventory
2. **Minimize Working Capital**: Cash tied up
3. **Minimize Total Cost of Ownership**: Include ordering costs
4. **Maximize Service Level**: Customer satisfaction
5. **Minimize Obsolescence Cost**: Fashion/tech goods
6. **Minimize Environmental Holding Cost**: Refrigeration energy

### Data Distribution Variants
1. **Lumpy Demand**: Intermittent, spiky demand
2. **Trending Demand**: Growing or declining
3. **Seasonal + Trend**: Combined patterns
4. **Promotional Spikes**: Sale event demand
5. **Lead Time Variability**: Supplier uncertainty
6. **Correlated Demand**: Related products

### Structural Variants
1. **Single-Item Single-Location**: Basic inventory
2. **Multi-Item**: Shared resources/capacity
3. **Multi-Location**: Distribution network
4. **Serial System**: Supply chain echelons
5. **Assembly System**: Multiple components
6. **Distribution System**: One-to-many

### Scenario Variants
1. **Retail Inventory**: Fashion, seasonal goods
2. **Manufacturing WIP**: Work-in-process inventory
3. **Pharmaceutical Supply**: Expiry and cold chain
4. **Spare Parts**: Intermittent demand
5. **Grocery/Perishables**: Short shelf life
6. **E-Commerce Fulfillment**: High velocity, many SKUs

---

## 15. Land Use Planning

### Current Implementation
- Maximize economic benefit
- Each parcel assigned one zoning type
- Environmental restrictions
- Adjacency constraints
- Resource capacity constraints
- Minimum zoning requirements

### Constraint Variants
1. **Contiguity Requirements**: Same zoning in connected areas
2. **Buffer Zones**: Separation between incompatible uses
3. **View Corridor Protection**: Height restrictions
4. **Historical Preservation**: Protected areas
5. **Flood Zone Restrictions**: Building limitations
6. **Traffic Impact Limits**: Development density limits
7. **Green Space Requirements**: Minimum park coverage
8. **Mixed-Use Requirements**: Diverse neighborhood requirements
9. **Phase Sequencing**: Development order constraints

### Objective Variants
1. **Maximize Tax Revenue**: Fiscal impact
2. **Minimize Infrastructure Cost**: Development cost
3. **Maximize Accessibility**: Distance to services
4. **Minimize Environmental Impact**: Habitat protection
5. **Maximize Housing Units**: Affordable housing goals
6. **Maximize Jobs Created**: Economic development
7. **Bi-Objective Economic-Environmental**: Trade-off analysis

### Data Distribution Variants
1. **Topographic Constraints**: Slope, elevation factors
2. **Existing Infrastructure**: Sunk cost considerations
3. **Market Value Variation**: Location premiums
4. **Population Density Projections**: Growth patterns
5. **Environmental Sensitivity Mapping**: Habitat quality

### Scenario Variants
1. **Urban Infill**: Redevelopment of existing areas
2. **Greenfield Development**: New community planning
3. **Industrial Zone Planning**: Manufacturing districts
4. **Coastal Zone Management**: Erosion and flood risk
5. **Transit-Oriented Development**: Station area planning
6. **Rural Land Preservation**: Agricultural protection

---

## 16. Load Balancing Problem

### Current Implementation
- Minimize maximum link utilization
- Link utilization constraints
- Flow requirements for demands
- Network topology constraints

### Constraint Variants
1. **Delay Constraints**: Maximum latency requirements
2. **Path Length Limits**: Maximum hop count
3. **Bandwidth Guarantees**: Minimum throughput per flow
4. **Failure Protection**: Backup path requirements
5. **QoS Classes**: Priority traffic handling
6. **Packet Loss Limits**: Reliability constraints
7. **Jitter Constraints**: Variation in delay
8. **Multicast Routing**: One-to-many flows
9. **Wavelength Continuity**: Optical network constraints

### Objective Variants
1. **Minimize Total Utilization**: Energy efficiency
2. **Minimize Average Delay**: Performance focus
3. **Maximize Throughput**: Capacity utilization
4. **Minimize Cost**: Price-based routing
5. **Minimize Number of Links Used**: Consolidation
6. **Minimize Blocking Probability**: Call admission

### Data Distribution Variants
1. **Traffic Matrix Patterns**: Peak/off-peak variation
2. **Self-Similar Traffic**: Realistic network traffic
3. **Geographic Traffic Distribution**: City-to-city patterns
4. **Application Mix**: Voice, video, data proportions
5. **Burst Traffic**: Flash crowds

### Structural Variants
1. **IP Network**: Internet-scale routing
2. **Data Center Network**: Fat-tree, Clos topologies
3. **WAN Backbone**: Long-haul networks
4. **Optical Network**: Wavelength routing
5. **Overlay Network**: Virtual network topology

### Scenario Variants
1. **Cloud Provider**: Multi-region infrastructure
2. **Enterprise Network**: Campus and branch offices
3. **Content Delivery**: CDN node placement
4. **5G Backhaul**: Mobile network transport
5. **IoT Network**: Many-to-one sensor traffic

---

## 17. Product Mix Problem

### Current Implementation
- Maximize profit
- Resource constraints
- Market constraints (min/max production)
- Industry-specific parameters

### Constraint Variants
1. **Product Family Constraints**: Related products together
2. **Substitute Products**: Either/or production
3. **Complementary Products**: Bundling requirements
4. **Channel Constraints**: Different products for different channels
5. **Quality Grade Mix**: Minimum premium product percentage
6. **Customer Contract Minimums**: Committed volumes
7. **Storage Constraints**: Warehouse limitations
8. **Transportation Constraints**: Delivery capacity

### Objective Variants
1. **Maximize Revenue**: Volume focus
2. **Maximize Contribution Margin**: Variable cost focus
3. **Maximize Market Share**: Competitive position
4. **Minimize Customer Service Issues**: Order fulfillment focus
5. **Maximize Throughput**: Bottleneck utilization
6. **Weighted Multi-Objective**: Balanced scorecard

### Data Distribution Variants
1. **Margin Variation**: High/low margin products
2. **Demand Elasticity**: Price-sensitive products
3. **Seasonal Mix**: Holiday vs. regular products
4. **Life Cycle Stage**: New vs. mature products
5. **Customer Segment Demands**: B2B vs. B2C

### Scenario Variants
1. **Consumer Goods**: FMCG product mix
2. **Industrial Products**: Capital equipment
3. **Service Mix**: Labor-intensive services
4. **Agricultural Products**: Crop selection
5. **Media Products**: Content production mix

---

## 18. Project Selection

### Current Implementation
- Maximize return
- Budget constraint
- Risk budget constraint
- Project dependencies
- High-risk project limits

### Constraint Variants
1. **Resource Sharing**: Projects compete for staff
2. **Timing Constraints**: Project start/end windows
3. **Strategic Fit**: Alignment with business strategy
4. **Synergy Constraints**: Complementary projects together
5. **Portfolio Balance**: Diversity requirements
6. **Milestone Dependencies**: Phased funding decisions
7. **Technology Platform Constraints**: Infrastructure requirements
8. **Regulatory Constraints**: Mandatory projects
9. **Exit Options**: Ability to cancel mid-project

### Objective Variants
1. **Maximize NPV**: Net present value
2. **Maximize Strategic Value**: Non-financial benefits
3. **Minimize Risk-Adjusted Cost**: Downside protection
4. **Maximize Innovation Score**: R&D focus
5. **Maximize Real Options Value**: Flexibility value
6. **Balanced Scorecard**: Multi-dimensional objectives

### Data Distribution Variants
1. **Risk-Return Correlation**: Higher risk, higher return
2. **Project Size Distribution**: Mix of small and large
3. **Duration Distribution**: Short and long projects
4. **Technology Uncertainty**: R&D vs. implementation
5. **Market Uncertainty**: Demand risk variation

### Scenario Variants
1. **R&D Portfolio**: Pharmaceutical pipeline
2. **Capital Investment**: Equipment and facility projects
3. **IT Project Portfolio**: Digital transformation
4. **Product Development**: New product pipeline
5. **M&A Decisions**: Acquisition targeting
6. **Sustainability Initiatives**: ESG project selection

---

## 19. Resource Allocation

### Current Implementation
- Maximize profit
- Resource constraints
- Minimum activity levels

### Constraint Variants
1. **Dynamic Reallocation**: Period-to-period changes limited
2. **Skill Matching**: Activity-resource compatibility
3. **Time Windows**: Activity scheduling constraints
4. **Preemption Rules**: Which activities can be interrupted
5. **Learning Effects**: Productivity increases with allocation
6. **Fairness Constraints**: Minimum allocation per activity
7. **Capacity Flexibility**: Overtime, outsourcing options
8. **Quality-Quantity Trade-off**: Resource quality affects output

### Objective Variants
1. **Minimize Total Completion Time**: Project scheduling
2. **Maximize Weighted Throughput**: Prioritized activities
3. **Minimize Resource Leveling**: Smooth utilization
4. **Maximize Customer Satisfaction**: SLA-based allocation
5. **Minimize Cost**: Budget optimization
6. **Minimize Idle Time**: Efficiency focus

### Data Distribution Variants
1. **Demand Peaks**: Surge capacity needs
2. **Skill Scarcity**: Limited specialized resources
3. **Cost Variation**: Different resource costs
4. **Productivity Variation**: Efficiency differences
5. **Availability Patterns**: Part-time, shift-based

### Scenario Variants
1. **Call Center Staffing**: Service level optimization
2. **Cloud Computing**: VM allocation
3. **Healthcare Staff**: Hospital resource allocation
4. **Manufacturing Cells**: Flexible manufacturing
5. **Consulting Projects**: Staff utilization

---

## 20. Scheduling Problem

### Current Implementation
- Minimize staffing cost
- Staffing requirements per shift
- Worker availability
- Min/max shift limits per worker
- Maximum consecutive working days
- At most one shift per worker per day
- Optional skill-based constraints

### Constraint Variants
1. **Shift Pattern Constraints**: Fixed rotation patterns
2. **Weekend Constraints**: Fair weekend distribution
3. **Holiday Preferences**: Priority for time off requests
4. **On-Call Requirements**: Backup staffing
5. **Break Scheduling**: Meal and rest breaks
6. **Cross-Training Requirements**: Skill development
7. **Seniority Rules**: Union contract requirements
8. **Fatigue Management**: Cumulative tiredness
9. **Split Shifts**: Discontinuous working hours
10. **Team Scheduling**: Groups work together

### Objective Variants
1. **Minimize Overtime**: Regular hour preference
2. **Maximize Preference Satisfaction**: Employee happiness
3. **Minimize Under/Over Staffing**: Match demand closely
4. **Minimize Training Cost**: Use qualified workers
5. **Minimize Agency/Temp Usage**: Reduce external labor
6. **Maximize Skill Development**: Mentoring opportunities
7. **Minimize Schedule Disruption**: Stability preference

### Data Distribution Variants
1. **Variable Demand Patterns**: Peak and off-peak periods
2. **Skill Distribution**: Specialist vs. generalist mix
3. **Availability Patterns**: Full-time vs. part-time
4. **Preference Strength**: Some workers more flexible
5. **Experience Levels**: Junior vs. senior mix

### Scenario Variants
1. **Nurse Scheduling**: Hospital ward coverage
2. **Retail Scheduling**: Store staffing
3. **Call Center Scheduling**: SLA-driven staffing
4. **Manufacturing Shifts**: Plant coverage
5. **Airline Crew Rostering**: Monthly crew schedules
6. **Restaurant Scheduling**: Kitchen and front-of-house
7. **Security Guard Scheduling**: 24/7 coverage

---

## 21. Supply Chain

### Current Implementation
- Minimize fixed + transportation costs
- Customer demand satisfaction
- Facility capacity constraints
- Transport mode capacity constraints
- Budget constraint
- Multiple transport modes

### Constraint Variants
1. **Lead Time Constraints**: Maximum delivery time
2. **Inventory Positioning**: Safety stock requirements
3. **Carbon Footprint Limits**: Emission constraints
4. **Single Sourcing**: Each customer from one facility
5. **Multi-Echelon Inventory**: Coordinated stock levels
6. **Reverse Logistics**: Return flow handling
7. **Perishability Constraints**: Shelf life limits
8. **Cold Chain Requirements**: Temperature control
9. **Tariff/Trade Constraints**: Cross-border regulations
10. **Risk Diversification**: Maximum sourcing from one region

### Objective Variants
1. **Minimize Total Cost of Ownership**: Including hidden costs
2. **Minimize Lead Time**: Responsiveness focus
3. **Maximize Service Level**: Fill rate optimization
4. **Minimize Carbon Footprint**: Sustainability focus
5. **Minimize Supply Risk**: Resilience objective
6. **Maximize Profit**: Revenue minus costs
7. **Bi-Objective Cost-Service**: Trade-off analysis

### Data Distribution Variants
1. **Global Network**: Multi-continent operations
2. **Regional Network**: Single-country focus
3. **Seasonal Demand**: Holiday peaks
4. **Promotional Demand**: Marketing-driven spikes
5. **E-Commerce Growth**: Channel shift patterns
6. **Commodity Price Variation**: Material cost changes

### Structural Variants
1. **Multi-Tier Network**: Suppliers → DC → Retail
2. **Direct Ship Model**: Factory to customer
3. **Cross-Dock Network**: No inventory storage
4. **Omni-Channel**: Store + online fulfillment
5. **Dropship Model**: Third-party fulfillment

### Scenario Variants
1. **FMCG Distribution**: Fast-moving consumer goods
2. **Automotive Supply Chain**: Just-in-time manufacturing
3. **Pharmaceutical Distribution**: Cold chain, compliance
4. **E-Commerce Fulfillment**: High velocity, small orders
5. **Fashion/Apparel**: Seasonal, short lifecycle
6. **Electronics**: High value, obsolescence risk
7. **Fresh Food Supply Chain**: Perishable logistics

---

## 22. Crop Planning

### Current Implementation
- Maximize profit
- Total land constraint (equality)
- Water requirement constraint
- Labor requirement constraint
- Market demand limits
- Minimum area per crop
- Crop diversity constraints

### Constraint Variants
1. **Crop Rotation Requirements**: Soil health constraints
2. **Fallow Land**: Minimum unplanted area
3. **Irrigation System Constraints**: Zone-based water delivery
4. **Machinery Sharing**: Equipment availability
5. **Harvest Window Constraints**: Labor scheduling
6. **Storage Capacity**: Post-harvest handling
7. **Contract Farming Commitments**: Pre-sold production
8. **Organic Certification**: Buffer zones, transition rules
9. **Pest/Disease Risk**: Monoculture limits
10. **Climate Risk**: Drought-tolerant crop requirements

### Objective Variants
1. **Minimize Risk**: Variance of returns
2. **Maximize Yield Per Hectare**: Productivity focus
3. **Minimize Water Usage**: Conservation focus
4. **Minimize Input Costs**: Fertilizer, pesticide reduction
5. **Maximize Nutrition Production**: Food security focus
6. **Maximize Employment**: Rural job creation
7. **Minimize Carbon Footprint**: Sustainable farming

### Data Distribution Variants
1. **Price Volatility**: Commodity price uncertainty
2. **Yield Variability**: Weather-dependent outcomes
3. **Soil Quality Variation**: Field-level differences
4. **Water Availability**: Seasonal irrigation limits
5. **Input Cost Trends**: Fertilizer, fuel prices
6. **Climate Change Scenarios**: Long-term shifts

### Scenario Variants
1. **Commercial Farming**: Large-scale commodity production
2. **Smallholder Agriculture**: Subsistence + cash crops
3. **Organic Farming**: Premium market focus
4. **Irrigated vs. Rainfed**: Water access difference
5. **Perennial Crops**: Orchards, vineyards
6. **Mixed Farming**: Crops + livestock integration
7. **Urban Agriculture**: Space-constrained growing

---

## 23. Multi-Commodity Flow

### Current Implementation
- Minimize total flow cost
- Arc capacity constraints (shared by commodities)
- Flow conservation per commodity
- Optional minimum flow requirements

### Constraint Variants
1. **Commodity-Specific Capacities**: Different limits per commodity
2. **Bundled Capacity**: Commodities compete for shared capacity
3. **Priority Routing**: Certain commodities have precedence
4. **Reliability Requirements**: Backup paths for critical flows
5. **Time-Dependent Flow**: Schedules and time windows
6. **Splitting Limits**: Maximum number of paths per commodity
7. **Node Capacity**: Processing limits at nodes
8. **Compatibility Constraints**: Some commodities cannot share arcs

### Objective Variants
1. **Minimize Maximum Utilization**: Load balancing
2. **Minimize Total Delay**: Congestion-aware routing
3. **Maximize Throughput**: Total flow volume
4. **Minimize Number of Paths**: Simplicity
5. **Fairness Objective**: Max-min fair allocation
6. **Minimize Energy Consumption**: Green routing

### Data Distribution Variants
1. **Hub Traffic Matrix**: Concentrated at hubs
2. **Uniform Random Demands**: Evenly distributed
3. **Gravity Model**: Population-based demands
4. **Time-Varying Traffic**: Peak hour patterns
5. **Commodity Type Mix**: Bulk vs. small parcels

### Structural Variants
1. **Sparse Network**: Few high-capacity links
2. **Dense Network**: Many alternative paths
3. **Hierarchical Network**: Backbone + access
4. **Mesh Network**: Fully connected regions
5. **Ring Network**: Simple topology

### Scenario Variants
1. **IP/MPLS Routing**: Internet traffic engineering
2. **Freight Rail Network**: Multiple cargo types
3. **Pipeline Network**: Different fluids
4. **Air Cargo Network**: Express vs. deferred
5. **Container Shipping**: Different container types

---

## 24. Telecom Network Design

### Current Implementation
- Minimize installation + routing costs
- Binary link installation decisions
- Flow conservation per commodity
- Capacity constraints (depend on installed links)
- Budget constraint on installation

### Constraint Variants
1. **Technology Choice**: Different link technologies (fiber, microwave)
2. **Redundancy Requirements**: K-connected design
3. **Ring Protection**: SONET/SDH ring constraints
4. **Hop Count Limits**: Maximum path length
5. **Delay Constraints**: Latency requirements
6. **Growth Provisions**: Future capacity headroom
7. **Right-of-Way Constraints**: Physical routing restrictions
8. **Co-Location Requirements**: Shared infrastructure
9. **Upgrade Paths**: Existing infrastructure integration

### Objective Variants
1. **Minimize Total Cost**: Installation + operating
2. **Minimize Maximum Latency**: Performance focus
3. **Maximize Reliability**: Network availability
4. **Minimize Number of Links**: Simpler network
5. **Maximize Future Flexibility**: Option value
6. **Minimize Energy Consumption**: Green design

### Data Distribution Variants
1. **Traffic Forecast Uncertainty**: Scenario-based planning
2. **Technology Cost Evolution**: Declining equipment costs
3. **Geographic Cost Variation**: Terrain effects
4. **Demand Growth Patterns**: Urban vs. rural
5. **Existing Infrastructure**: Brownfield constraints

### Structural Variants
1. **Core Network Design**: National backbone
2. **Metro Network**: City-level design
3. **Access Network**: Last-mile connectivity
4. **Data Center Interconnect**: High-capacity links
5. **Submarine Cable**: Transoceanic networks

### Scenario Variants
1. **5G Deployment**: Cell site connectivity
2. **Enterprise WAN**: Corporate network design
3. **ISP Network**: Internet service provider
4. **Utility Network**: Smart grid communication
5. **Campus Network**: University or corporate campus

---

## Cross-Problem Variant Categories

### Stochastic/Robust Variants
For all problem types, consider:
1. **Stochastic Programming**: Expected value optimization with scenarios
2. **Robust Optimization**: Worst-case protection
3. **Chance Constraints**: Probability-based feasibility
4. **Distributionally Robust**: Ambiguous probability distributions
5. **Adaptive/Recourse**: Two-stage decisions

### Multi-Period Extensions
1. **Rolling Horizon**: Sequential decision making
2. **Multi-Stage**: Linked period decisions
3. **Horizon Effects**: Terminal conditions
4. **Discounting**: Time value of money

### Multi-Objective Variants
1. **Weighted Sum**: Scalar combination
2. **ε-Constraint**: One objective constrained
3. **Goal Programming**: Target achievement
4. **Lexicographic**: Priority ordering
5. **Pareto Frontier**: Non-dominated solutions

### Hierarchical/Decomposition Variants
1. **Benders Decomposition Structure**: Complicating variables
2. **Dantzig-Wolfe Structure**: Column generation
3. **Lagrangian Relaxation**: Complicating constraints
4. **Bilevel Programming**: Leader-follower structure

---

## Implementation Priority Recommendations

### High Priority (High value, moderate effort)
1. Multi-dimensional knapsack
2. Mean-variance portfolio optimization
3. Multi-commodity transportation
4. Hub-and-spoke network design
5. Stochastic inventory with backlogging options
6. Capacitated facility location with service levels

### Medium Priority (Good value, requires careful design)
1. Multi-period production planning with setup costs
2. Time-expanded network flow
3. Robust energy generation with ramping
4. Crew scheduling with fatigue management
5. Multi-echelon supply chain

### Lower Priority (Specialized applications)
1. Bilevel problems
2. Chance-constrained variants
3. Game-theoretic extensions
4. Adaptive optimization
